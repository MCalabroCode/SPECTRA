'''
training + testing routines
'''

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad
import wandb
import os
import uuid
from collections import defaultdict


# Absolute imports from the spectra package
from spectra.evaluation.metrics import calc_auprc
from spectra.training.losses import _get_beta_schedule, compute_mmd_withcosine, compute_mmd, compute_weighted_energy_distance
#from losses import _get_beta_schedule, compute_mmd_withcosine, compute_mmd

def train_step_perturb_model(model, data, device, alpha=1., beta=1., gamma=1., eta=1.):
    x, y, pert, pert_idx = data  # x,y: [B,N,1], pert: [B,N]
    x, y, pert = x.to(device), y.to(device), pert.to(device)
    B, N, _ = x.shape

    y_hat, x_hat = model((x,pert)) #[BxN,1]
    x_flat = x.reshape(B * N, 1)
    y_flat = y.reshape(B * N, 1)

    # control cells (x_hat): ELBO loss
    squared_error = (x_hat - x_flat) ** 2 # [B*N,1]
    control_loss_feat = squared_error.view(B, N).mean(dim=1).mean()

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)

    # Cosine similarity (Direction)
    x_true = x_flat.view(B, N)
    y_true = y_flat.view(B, N)
    x_pred = x_hat.view(B, N)
    y_pred = y_hat.view(B, N)

    # Compute pseudobulk
    x_true_mean = x_true.mean(dim=0) # [N]
    y_true_mean = y_true.mean(dim=0) # [N]
    x_pred_mean = x_pred.mean(dim=0) # [N]
    y_pred_mean = y_pred.mean(dim=0) # [N]

    # Extract perturbation weights for this batch
    batch_pert_idx = pert_idx[0].item() # Guaranteed identical across the batch!
    batch_weights = model.weight_lookup[batch_pert_idx] #[N]

    delta_true_mean = y_true_mean - x_true_mean
    delta_pred_mean = y_pred_mean - x_true_mean #x_true_mean

    true_norm = torch.norm(delta_true_mean)
    if true_norm > 1e-6:
        cos_sim = F.cosine_similarity(delta_pred_mean, delta_true_mean, dim=0)
        loss_cosine = 1.0 - cos_sim
    else:
        loss_cosine = torch.tensor(0.0, device=device)
    
    #loss_mmd_y = compute_mmd_withcosine(y_true, y_pred, weights=batch_weights)

    # loss_wmse_y = torch.sum(batch_weights * (y_true_mean - y_pred_mean)**2)

    loss_mmd_y = compute_weighted_energy_distance(y_true, y_pred, batch_weights)

    # if mmd_gamma != 0.0:
    #     loss_mmd_x = compute_mmd(x_true, x_pred)
    # else:
    #     loss_mmd_x = 0.0

    total_loss = (alpha * control_loss_feat 
                + beta * kl_div 
                + gamma * loss_mmd_y 
                # + eta * loss_wmse_y
                + loss_cosine
    )

    return total_loss, kl_div, loss_mmd_y, loss_cosine, control_loss_feat

@torch.no_grad()
def test_perturb_model(model, loader, device):

    model.eval()
    pert_mmd = []
    pert_wmse = []
    for i, data in enumerate(tqdm(loader, desc='testing...')):
        x, y, pert, pert_idx = data
        x, y, pert = x.to(device), y.to(device), pert.to(device)

        B = pert.shape[0]
        N = pert.shape[1]

        #y_hat, _ = model((x,pert)) #[B*N,1]
        y_true = y.view(B, N)
        y_pred = model((x,pert))[0].view(B, N)

        batch_pert_idx = pert_idx[0].item()
        specific_weights = model.weight_lookup[batch_pert_idx].to(device).view(N)

        mu_true = y_true.mean(dim=0)  # [N]
        mu_pred = y_pred.mean(dim=0)  # [N]

        error = torch.sum(specific_weights * (mu_pred - mu_true).pow(2))
        pert_wmse.append(error.item())

        # mmd
        mmd_error = compute_weighted_energy_distance(y_true, y_pred, model.weight_lookup[batch_pert_idx])
        pert_mmd.append(mmd_error.item())

    avg_pert_mmd = sum(pert_mmd)/len(pert_mmd)
    avg_pert_wmse = sum(pert_wmse)/len(pert_wmse)
    return avg_pert_mmd, avg_pert_wmse

@torch.no_grad()
def test_perturb_model_pseudobulk(model, loader, device, compute_mmd_fn=compute_weighted_energy_distance):
    '''
    NOTE: this is the correct code, fully compatible with what written in "Adressing mode collapse"; here, pseudobulk
    is calculated over all perturbation samples, not just minibatches
    '''
    model.eval()

    sum_true = {}
    sum_pred = {}
    count = defaultdict(int)

    # Optional: store cells for per-perturbation MMD
    true_cells = defaultdict(list)
    pred_cells = defaultdict(list)

    for data in tqdm(loader, desc="testing..."):
        x, y, pert, pert_idx = data
        x = x.to(device)
        y = y.to(device)
        pert = pert.to(device)

        B = pert.shape[0]
        N = pert.shape[1]

        y_hat, _ = model((x, pert))

        y_true = y.view(B, N)
        y_pred = y_hat.view(B, N)

        # Assumes your sampler guarantees one perturbation per batch
        if not torch.all(pert_idx == pert_idx[0]):
            raise ValueError(
                "Evaluation batch contains multiple perturbations."
            )

        p = int(pert_idx[0].item())

        if p not in sum_true:
            sum_true[p] = torch.zeros(N, device=device)
            sum_pred[p] = torch.zeros(N, device=device)

        sum_true[p] += y_true.sum(dim=0)
        sum_pred[p] += y_pred.sum(dim=0)
        count[p] += B

        if compute_mmd_fn is not None:
            true_cells[p].append(y_true.detach().cpu())
            pred_cells[p].append(y_pred.detach().cpu())

    wmse_by_pert = {}

    for p in sum_true:
        mu_true = sum_true[p] / count[p]
        mu_pred = sum_pred[p] / count[p]

        w = model.weight_lookup[p].to(device).view(-1)
        w = w / (w.sum() + 1e-8)

        wmse_p = torch.sum(w * (mu_pred - mu_true).pow(2))
        wmse_by_pert[p] = wmse_p.item()
    
    # How well does the model perform on the average perturbation?
    avg_wmse_macro = float(np.mean(list(wmse_by_pert.values())))

    # How well does the model perform on the average cell in this dataset?
    avg_wmse_micro = float(
        np.average([wmse_by_pert[p] for p in wmse_by_pert], weights=[count[p] for p in wmse_by_pert],)
    )

    if compute_mmd_fn is None:
        return _, avg_wmse_micro, avg_wmse_macro

    mmd_by_pert = {}

    for p in true_cells:
        yt = torch.cat(true_cells[p], dim=0).to(device)
        yp = torch.cat(pred_cells[p], dim=0).to(device)

        mmd_by_pert[p] = compute_mmd_fn(yt, yp, model.weight_lookup[p]).item()

    avg_mmd_macro = float(np.mean(list(mmd_by_pert.values())))

    return avg_mmd_macro, avg_wmse_macro, _

@torch.no_grad()
def final_val_AUPRC(model, loader, device, var_names, idx_to_gene):
    model.eval()
    
    pred_expr_list = []
    true_expr_list = []
    control_expr_list = []
    obs_gene_list = []
    
    # 1. Accumulate all predictions and ground truths
    for i, data in enumerate(tqdm(loader, desc='testing with AUPRC...')):
        x, y, pert, pert_idx = data
        x, y, pert = x.to(device), y.to(device), pert.to(device)

        B = pert.shape[0]
        N = pert.shape[1]

        y_hat, _ = model((x, pert)) 

        # Store arrays (moving them off GPU immediately to save VRAM)
        pred_expr_list.append(y_hat.view(B, N).cpu().numpy())
        true_expr_list.append(y.view(B, N).cpu().numpy())
        
        # We must also save the input 'x' cells. calc_auprc needs control 
        control_expr_list.append(x.view(B, N).cpu().numpy())

        # Decode the boolean perturbation tensor back to strings
        for b in range(B):
            pert_indices = torch.where(pert[b])[0]
            if len(pert_indices) == 0:
                obs_gene_list.append('non-targeting')
            else:
                # Reconstruct combinatorial names if needed (e.g., 'A+B')
                pert_name = "+".join([idx_to_gene[idx.item()] for idx in pert_indices])
                obs_gene_list.append(pert_name)

    # 2. Stack everything into dense numpy matrices
    X_pred = np.vstack(pred_expr_list)
    X_true = np.vstack(true_expr_list)
    X_ctrl = np.vstack(control_expr_list)

    # 3. Build DataFrames for AnnData construction
    obs_pert = pd.DataFrame({"target_gene": obs_gene_list})
    obs_ctrl = pd.DataFrame({"target_gene": ['non-targeting'] * X_ctrl.shape[0]})
    var = pd.DataFrame(index=var_names)

    # 4. Construct the AnnData objects
    adata_pred = ad.AnnData(X=X_pred, obs=obs_pert.copy(), var=var)
    adata_true = ad.AnnData(X=X_true, obs=obs_pert.copy(), var=var)
    adata_ctrl = ad.AnnData(X=X_ctrl, obs=obs_ctrl, var=var)

    # Concat the control cells into both datasets so the metric can compute DEGs
    adata_pred_full = ad.concat([adata_pred, adata_ctrl])
    adata_true_full = ad.concat([adata_true, adata_ctrl])

    # 5. Calculate the final dataset-wide metric!
    # (Assuming calc_auprc returns a single float score)
    auprc_score = calc_auprc(adata_true_full, adata_pred_full, pert_col='target_gene', control_name='non-targeting')
    average = sum(auprc_score.values()) / len(auprc_score)

    return average

def train(model, 
    train_loader, 
    test_loader, 
    device, 
    wandb_support, 
    var_names, 
    idx_to_gene, 
    patience=7, 
    metric_mode='min'
):
    """
    metric_mode: Set to 'max' if your validation metric is AUPRC/Accuracy (higher is better). 
                 Set to 'min' if your validation metric is MMD/MSE/Loss (lower is better).
    """

    torch.autograd.set_detect_anomaly(True)

    # estract hyperparameters from model.config
    lr = model.config['lr']
    n_epochs = model.config['n_epochs'] 
    alpha_weight = model.config['alpha'] 
    beta_weight = model.config['beta'] 
    gamma_weight = model.config['gamma'] 

    global_loss = []
    mmd_pert = []
    mmd_ctrl = []
    test_wmse = []
    test_weighted_wmse = []

    first_batch = next(iter(train_loader))
    batch_size = first_batch[0].shape[0]

    accumulation_steps = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True, weight_decay=0.0)

    # for FAGCN convolution, pre-calculated in-degree, out-degree and reverse edge index
    if model.conv_type == 'FAGCN':
        _precompute_n = batch_size * model.num_nodes
        _precompute_edge = model._get_batched_edge_index(batch_size)
        for _module in model.modules():
            if hasattr(_module, 'precompute_degrees'):
                _module.precompute_degrees(_precompute_edge, _precompute_n)


    weights_dir = model.config['weights_folder_path']
    os.makedirs(weights_dir, exist_ok=True)

    # Early Stopping Setup
    best_val_metric = float('-inf') if metric_mode == 'max' else float('inf')
    patience_counter = 0
    if wandb_support and wandb.run is not None:
        best_filepath = os.path.join(weights_dir,f"best_model_{model.architecture_name}_{wandb.run.id}.pth")
    else:
        best_filepath = os.path.join(weights_dir, f"best_model_{model.architecture_name}_{uuid.uuid4().hex}.pth")

    # wandb watch
    if wandb_support:
        wandb.watch(model, log="all", log_freq=10)

    for epoch in range(1, n_epochs + 1):
        model.train()

        total_loss = 0
        total_mmd_pert = 0
        total_mmd_ctrl = 0
        kl_accum = 0
        mse_accum = 0
        cos_accum = 0

        optimizer.zero_grad(set_to_none=True)
        for (i,batch) in enumerate(tqdm(train_loader, desc=f'training at epoch {epoch}')):
            loss, kl_div, loss_mmd_y, loss_cosine, loss_feat = train_step_perturb_model(
                model, 
                batch, 
                model.device, 
                alpha=alpha_weight, 
                beta=beta_weight,
                gamma=gamma_weight)

            loss.backward()

            # sum, we will calculate the mean over all the epoch
            total_loss += loss.item()
            total_mmd_pert += loss_mmd_y.item()
            #total_mmd_ctrl += loss_mmd_x#.item()
            kl_accum += kl_div.item()
            mse_accum += loss_feat.item()
            cos_accum += loss_cosine.item()
            
            # gradient accumulation
            if (i+1)%accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # calculate averages for the epoch
        epoch_loss = total_loss / len(train_loader)
        epoch_mmd_pert = total_mmd_pert / len(train_loader)
        #epoch_mmd_ctrl = total_mmd_ctrl / len(train_loader)
        epoch_kl = kl_accum / len(train_loader)
        epoch_mse = mse_accum / len(train_loader)
        epoch_cos = cos_accum / len(train_loader)

        # Append to lists
        global_loss.append(epoch_loss)
        mmd_pert.append(epoch_mmd_pert)
        #mmd_ctrl.append(epoch_mmd_ctrl)

        # print training methods
        print(f'training KL = {epoch_kl:.5f} | mse = {epoch_mse:.3f} | cos = {epoch_cos:.3f}')

        # save current epoch weights
        # filepath = os.path.join(weights_dir, f"weights__model-{model.architecture_name}_epoch-{epoch}__nodes-{model.num_nodes}_b-{batch_size}_h-{model.n_channels}_p-{model.dropout_p}.pth")
        # torch.save(model.state_dict(), filepath)
        
        # Validation & Early Stopping
        if epoch!=0:

            # We assume this returns your primary metric (MMD)
            _, avg_weighted_pert_wmse, avg_pert_wmse = test_perturb_model_pseudobulk(model, test_loader, device, compute_mmd_fn=None)
            test_wmse.append(avg_pert_wmse)
            test_weighted_wmse.append(avg_weighted_pert_wmse)

            # Check if this is the best model so far
            is_best = (avg_pert_wmse > best_val_metric) if metric_mode == 'max' else (avg_pert_wmse < best_val_metric)

            if is_best:
                print(f"Validation metric improved to {avg_pert_wmse:.4f}. Saving best model...")
                best_val_metric = avg_pert_wmse
                patience_counter = 0
                torch.save(model.state_dict(), best_filepath)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            # W&B Logging
            if wandb_support:
                wandb.log({
                    "epoch": epoch,
                    "train/global_loss": epoch_loss,
                    "train/mmd_pert": epoch_mmd_pert,
                    # "train/mmd_ctrl": epoch_mmd_ctrl,
                    "train/kl_divergence": epoch_kl,
                    "train/mse": epoch_mse,
                    "train/cosine_loss": epoch_cos,
                    "val/test_WMSE": avg_pert_wmse,
                    "val/test_weighted_WMSE": avg_weighted_pert_wmse
                })

            # Trigger Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break
    
    # ==========================================
    # FINAL EVALUATION PHASE
    # ==========================================
    torch.autograd.set_detect_anomaly(False)
    print("\n--- Training concluded. Initiating Final Evaluation ---")
    
    # Reload the absolute best weights before running the final metric
    if os.path.exists(best_filepath):
        print("Reloading best epoch weights for final evaluation...")
        model.load_state_dict(torch.load(best_filepath))
    
    # Calculate the metric strictly ONCE
    final_metric_val = final_val_AUPRC(model, test_loader, model.device, var_names, idx_to_gene)
    print(f"FINAL TEST METRIC: {final_metric_val:.4f}")
    
    if wandb_support:
        wandb.log({"val/AUPRC": final_metric_val})
    
    return mmd_pert, test_weighted_wmse, test_wmse
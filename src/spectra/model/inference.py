import torch
from tqdm import tqdm
from scipy import sparse
import numpy as np
import pandas as pd
import anndata as ad

def generate_from_control(gene_counts_dict, adata_control, model, gene_to_idx, batch_size=32, add_control=True):

    prediction_list = []
    obs_gene_list = []

    var_names = adata_control.var_names

    # ctrl
    #adata_control = adata[adata.obs['target_gene'] == 'non-targeting']
    ctrl_data = adata_control.X.toarray() if hasattr(adata_control.X, "toarray") else np.asarray(adata_control.X)
    ctrl_data = torch.tensor(ctrl_data, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        for pert, n_samples in tqdm(gene_counts_dict.items()):

            # Perturbation one-hot embedding
            pert_embedding = torch.zeros(adata_control.shape[1], dtype=torch.bool)
            if pert != 'non-targeting':
                perturbs = [gene_to_idx[g] for g in pert.split('+') if g in gene_to_idx]
                for single_pert in perturbs:
                    pert_embedding[single_pert] = True
            pert_embedding = pert_embedding.unsqueeze(0)
            
            # Mini-batch generation to prevent CUDA OOM
            for offset in range(0, n_samples, batch_size):

                # Calculate how many cells to generate in this specific chunk
                chunk_size = min(batch_size, n_samples - offset)
                pert_batch = pert_embedding.expand(chunk_size, -1).to(model.device)
                
                # Sample control cells for this chunk
                random_indices = np.random.choice(ctrl_data.shape[0], size=chunk_size, replace=True)
                ctrl_batch = ctrl_data[random_indices, :] # shape [B,N]
                ctrl_batch = ctrl_batch.unsqueeze(-1).to(model.device) # shape [B,N,1]
                
                # Predict
                data = ctrl_batch, pert_batch
                predicted_full_gex = model.predict_full_expression(data)
                
                predicted_full_gex = predicted_full_gex.reshape(chunk_size, -1).detach().cpu().numpy()
                prediction_list.append(predicted_full_gex) 
                obs_gene_list.extend([pert] * chunk_size)

    # Compile the final AnnData
    X = np.vstack(prediction_list)
    obs = pd.DataFrame({"target_gene": obs_gene_list})
    var = pd.DataFrame(index=var_names)
    pred_adata = ad.AnnData(X=X, obs=obs, var=var)

    if add_control:
        # Append the non-targeting controls to the example anndata if they're missing
        if "non-targeting" not in pred_adata.obs["target_gene"].unique():
            assert np.all(pred_adata.var_names.values == adata_control.var_names.values), (
                "Gene-Names are out of order or unequal"
            )
            pred_adata = ad.concat(
                [
                    pred_adata,
                    adata_control,
                ]
            )

    return pred_adata
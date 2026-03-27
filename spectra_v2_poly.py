import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, DirGNNConv, MixHopConv, GATv2Conv, SAGEConv
from torch_geometric.utils import dropout_edge
from torch.nn import ReLU, LeakyReLU, GELU, LayerNorm
from magnet import MagNetConv, precompute_magnet_attributes_sparse

import wandb



class PolyLayer(torch.nn.Module):
    """
    Modular PolyLayer to wrap our directed convolution.
    """
    def __init__(self, in_channels, out_channels, conv):
        super().__init__()
        self.conv = conv
        self.w_h = torch.nn.Linear(in_channels, out_channels)
        self.w_l = torch.nn.Linear(in_channels, out_channels)
        self.beta = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index):
        # 1. Non-linear transformation
        h_i = F.relu(self.w_h(x))
        
        # 2. Directed Convolution
        conv_out = self.conv(x, edge_index)
        
        # 3. Intermediate representation with linear skip connection
        x_prime = conv_out + self.w_l(x)
        
        # 4. Polynomial gating 
        out = (1 - self.beta) * (h_i * x_prime) + self.beta * x_prime
        return out


class VariationalGraphEncoder(torch.nn.Module):
    ''' Adapted Dir-Poly encoder class
    '''
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        # Helper function to generate the Dir-GNN + GATv2 combo cleanly
        def make_dir_poly_conv(in_dim, out_dim):
            base_conv = GATv2Conv(in_dim, out_dim, heads=1, concat=False)
            dir_conv = DirGNNConv(base_conv, alpha=0.5)
            return PolyLayer(in_dim, out_dim, dir_conv)

        # Layer 1: in_channels -> out_channels
        self.conv1 = make_dir_poly_conv(in_channels, out_channels)
        self.ln1 = LayerNorm(out_channels)
        
        # Layer 2: out_channels -> 2*out_channels
        self.conv2 = make_dir_poly_conv(out_channels, 2 * out_channels)
        self.ln2 = LayerNorm(2 * out_channels)
        
        # Mu and LogStd: 2*out_channels -> out_channels
        self.conv_mu = make_dir_poly_conv(2 * out_channels, out_channels)
        self.conv_logstd = make_dir_poly_conv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.gelu(x)

        # Layer 2
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.gelu(x) 
        
        # Latent Space (Mu and LogStd)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        mu = self.conv_mu(x, edge_index)
        logst = self.conv_logstd(x, edge_index)
        
        return mu, logst

class MLP(torch.nn.Module):
    '''
    MLP auxiliary class
    '''
    def __init__(self, sizes, batch_norm=True, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FeatureDecoder(torch.nn.Module):
    ''' 
    Adapted Feature decoder using Dir-Poly
    '''
    def __init__(self, n_channels, num_node_features, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # Helper to build the Dir-Poly SAGE block
        def make_dir_poly_sage(channels):
            base_conv = GATv2Conv(channels, channels, heads=1, concat=False)
            dir_conv = DirGNNConv(base_conv, alpha=0.5)
            return PolyLayer(channels, channels, dir_conv)

        # Layer 1
        self.conv1 = make_dir_poly_sage(n_channels)
        self.ln1 = LayerNorm(n_channels)
        
        # Layer 2
        self.conv2 = make_dir_poly_sage(n_channels)
        self.ln2 = LayerNorm(n_channels)
        
        self.last_layer = torch.nn.Linear(n_channels, num_node_features) 

    def forward(self, z, edge_index):
        # Layer 1 (Notice: No explicit + z skip connection needed)
        h1 = self.conv1(z, edge_index)
        h1 = self.ln1(h1)
        h1_out = F.gelu(h1) 

        # Layer 2 (Notice: No explicit + h1_out skip connection needed)
        h1_drop = F.dropout(h1_out, p=self.dropout_rate, training=self.training)
        h2 = self.conv2(h1_drop, edge_index)
        h2 = self.ln2(h2)
        h2_out = F.gelu(h2) 

        # Final projection to reconstruct node features
        out = self.last_layer(h2_out) 
        
        # LeakyReLU during training to prevent dead gradients, hard ReLU for biological realism at eval
        if self.training:
            return F.leaky_relu(out, negative_slope=0.05) 
        else:
            return F.relu(out)


class PerturbModel(torch.nn.Module):
    '''
    SPECTRA model class
    '''
    def __init__(self, edge_index, num_nodes, device, gene_weights=None, num_node_features=1, n_channels=32, edge_dropout_p=0.1):
        super().__init__()
        self.device = device 
        self.num_nodes = num_nodes 
        self.n_channels = n_channels
        
        self.register_buffer('edge_index', edge_index)
        self.edge_dropout_p = edge_dropout_p

        # Weight Lookup Construction for WMSE
        default_weights = (1/num_nodes)*torch.ones(num_nodes)
        weight_lookup = default_weights.unsqueeze(0).repeat(num_nodes + 1, 1).to(device)    
        if gene_weights is not None:
            for pert_idx, weight_array in gene_weights.items():
                w_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)
                if 0 <= pert_idx < num_nodes:
                    weight_lookup[pert_idx] = w_tensor
        self.register_buffer('weight_lookup', weight_lookup)

        self.encoder_in_channels = 1

        # Learnable KO perturbation Token
        # #self.ko_token = torch.nn.Parameter(torch.randn(1, n_channels) - 2.0)
        # self.ko_mu = torch.nn.Embedding(num_nodes, n_channels)
        # self.ko_sigma = torch.nn.Embedding(num_nodes, n_channels)
        # torch.nn.init.normal_(self.ko_mu.weight, mean=-2.0, std=0.5) # Match your original prior for the mean shift (-2.0)
        # torch.nn.init.zeros_(self.ko_sigma.weight) # Initialize variance scale weights to 0 so that exp(0) = 1.0 (identity scale)

        # self.ko_token = torch.nn.Embedding(1,n_channels)
        # torch.nn.init.xavier_uniform_(self.ko_token.weight)

        self.ko_mu = torch.nn.Embedding(num_nodes, 64)
        self.ko_mlp = MLP([64, n_channels, n_channels])

        self.encoder = VariationalGraphEncoder(self.encoder_in_channels, n_channels)
        self.gex_decoder = FeatureDecoder(n_channels, num_node_features)
        
        self._cached_batch_size = 0
        self._cached_edge_index = None
        self._cached_gene_ids = None

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'number of trainable parameters: {num_trainable_params}')
    
    def _get_batched_edge_index(self, batch_size):
        '''
        batched edge lists - edges of the DAG are always the same for every sample
        '''
        if batch_size == self._cached_batch_size and self._cached_edge_index is not None:
            return self._cached_edge_index
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes
        edge_index_batch = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
        edge_index_batch = edge_index_batch.reshape(2, -1)
        self._cached_batch_size = batch_size
        self._cached_edge_index = edge_index_batch
        return edge_index_batch

    
    def _get_batched_gene_ids(self, batch_size):
        '''
        repeats the genes ids exactly batch_size times (e.g. [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        '''
        if self._cached_gene_ids is not None and len(self._cached_gene_ids) == batch_size * self.num_nodes:
            return self._cached_gene_ids
        ids = torch.arange(self.num_nodes, device=self.device)
        ids = ids.repeat(batch_size) 
        self._cached_gene_ids = ids
        return ids

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu, logstd, threshold=1e-2, verbose=True, free_bits=0.05):

        kl_raw = -0.5 * (1 + 2 * logstd - mu**2 - logstd.exp()**2)
        kl_per_dim = torch.mean(kl_raw, dim=0)

        # print("kl_raw.mean(), kl_raw.std():", kl_raw.mean().item(), kl_raw.std().item())
        # print("kl_per_dim (before clamp):", kl_per_dim.detach().cpu().numpy())
        # print("free_bits:", free_bits)

        #kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        return torch.mean(kl_per_dim)

    def forward(self, data, return_latent=True):

        x, pert = data
        x = x.to(self.device) #[B,N,1]
        pert = pert.to(self.device) #[B,N]

        batch_size, num_nodes, num_features = x.shape

        edge_index_batch = self._get_batched_edge_index(batch_size)
        x = x.reshape(batch_size * num_nodes, num_features) #[BxN,1]
        pert = pert.reshape(batch_size * num_nodes) #[BxN]

        # Edge Dropout
        if self.training and self.edge_dropout_p > 0:
            edge_index_batch, _ = dropout_edge(
                edge_index_batch, 
                p=self.edge_dropout_p, 
                force_undirected=False,
                training=self.training
            )

        mu, logstd = self.encoder(x, edge_index_batch) # both [BxN, C] (C = hidden channels dimension)
        logstd = torch.clamp(logstd, min=-20, max=10) # this is to avoid inf values

        self.last_mu = mu          
        self.last_logstd = logstd  

        z_ctrl = self.reparametrize(mu, logstd)     
        self.last_z = z_ctrl
        x_hat = self.gex_decoder(z_ctrl, edge_index_batch)
        
        # additive logic for perturbation encoding
        mask = pert.unsqueeze(1).to(z_ctrl.dtype) #[BxN,1]

        #z = z + (mask * self.ko_token)

        #NOTE:learnable generic gene-specific function with variance scaling
        # pert_mask = pert.bool()
        # delta_mu = torch.zeros_like(mu)
        # delta_logstd = torch.zeros_like(logstd)
        
        # if pert_mask.any():
        #     gene_ids = self._get_batched_gene_ids(batch_size)
        #     perturbed_gene_ids = gene_ids[pert_mask]
            
        #     # Lookup shifts (No torch.exp needed for logstd addition!)
        #     mu_shift = self.ko_mu(perturbed_gene_ids)
        #     logstd_shift = self.ko_sigma(perturbed_gene_ids) 
            
        #     delta_mu[pert_mask] = mu_shift
        #     delta_logstd[pert_mask] = logstd_shift

        pert_mask = pert.bool()
        delta_mu = torch.zeros_like(mu)
        if pert_mask.any():
            gene_ids = self._get_batched_gene_ids(batch_size)
            perturbed_gene_ids = gene_ids[pert_mask]
            ko_embeddings = self.ko_mu(perturbed_gene_ids)
            mu_shift = self.ko_mlp(ko_embeddings)
            delta_mu[pert_mask] = mu_shift
        mu_pert = mu + delta_mu
        logstd_pert = logstd #+ delta_logstd
        z_pert = self.reparametrize(mu_pert, logstd_pert)
        y_hat = self.gex_decoder(z_pert, edge_index_batch)

        # print(self.ko_token.weight)
        # print('======= mean mu before pert =========')
        # print(mu[mask.bool().squeeze(-1)].mean(dim=0)) # mean of mu value BEFORE perturbation in node that is about to be perturbed
        # print('======= mean mu after pert ==========')
        # print(mu_pert[mask.bool().squeeze(-1)].mean(dim=0)) # mean of mu value AFTER perturbation in node that is about to be perturbed
        # print('=================')
        # print('=================')
        # # OK I HAVE CHEKCED - PERTURBATION SIGNAL IS ACTUALLY QUITE STRONG

        # mu_pert = mu + (mask * self.ko_token.weight)  #[1,C]
        # logstd_pert = logstd # + delta_logstd
        # z_pert = self.reparametrize(mu_pert, logstd_pert)
        # y_hat = self.gex_decoder(z_pert, edge_index_batch)
        
        return y_hat, x_hat


    def predict_full_expression(self, data):
        self.eval()
        return self.forward(data)[0]

######### training + testing routines ##########

def _get_beta_schedule(epoch, n_epochs, warmup_epochs=5, n_cycles=1, ratio=0.5):
    '''
    beta schefuler for the VAE (beta-Vae)

    Args:
        epoch: Current epoch (1-indexed based on your train loop)
        n_epochs: Total number of epochs
        warmup_epochs: Number of initial epochs where beta remains exactly 0.0
        n_cycles: Number of annealing cycles after warmup
        ratio: Fraction of the cycle spent increasing beta
    '''
    # Warmup Phase
    if epoch <= warmup_epochs:
        return 0.0
    
    # Adjust remaining epochs for the cyclical schedule
    adjusted_epoch = epoch - warmup_epochs - 1 # 0-indexed for the math
    adjusted_n_epochs = n_epochs - warmup_epochs
    
    if adjusted_n_epochs <= 0: # Prevent division by zero if warmup is exactly n_epochs
        return 1.0
        
    period = max(1, adjusted_n_epochs // n_cycles)
    step = adjusted_epoch % period
    
    # linear Annealing Phase within the cycle
    if step < period * ratio:
        return step / (period * ratio)
    else:
        return 1.0

def compute_mmd_withcosine(x, y, weights=None, kernel_mul=2.0, kernel_num=5, fix_sigma=None, lambda_cos=0.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches.
    Uses a composite kernel: (1 - lambda_cos) * Multi-RBF + lambda_cos * Cosine
    """
    assert x.shape == y.shape, "real and predicted batches do not match in size."

    batch_size = x.size(0)
    n_samples = int(x.size(0)) + int(y.size(0))
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=x.device)

    total = torch.cat([x, y], dim=0)
    
    # ---------------------------------------------------------
    # 1. RBF Kernel Calculation (Spatial)
    # ---------------------------------------------------------
    if weights is not None:
        # Scale features so that squared L2 distance naturally becomes WMSE
        total_scaled = total * torch.sqrt(weights.unsqueeze(0))
    else:
        total_scaled = total
    L2_distance = torch.cdist(total_scaled, total_scaled, p=2)**2
    #L2_distance = torch.cdist(total, total, p=2)**2
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Add epsilon to prevent bandwidth collapse if samples are identical
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples)# + 1e-5
        
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_rbf = sum([torch.exp(-L2_distance / bw) for bw in bandwidth_list])
    
    # ---------------------------------------------------------
    # 2. Cosine Kernel Calculation (Angular)
    # ---------------------------------------------------------
    if lambda_cos > 0.0:
        # Normalize each sample vector to length 1
        total_norm = F.normalize(total, p=2, dim=1, eps=1e-8)
        # Pairwise cosine similarity is just the dot product of normalized vectors
        kernel_cos = torch.mm(total_norm, total_norm.t())
        
        # Scale Cosine to [0, 1] to match RBF scale (optional but stabilizes lambda)
        kernel_cos = (kernel_cos + 1.0) / 2.0
        
        # Blend the kernels
        kernel_val = (1.0 - lambda_cos) * kernel_rbf + (lambda_cos * kernel_cos)
    else:
        kernel_val = kernel_rbf

    # ---------------------------------------------------------
    # 3. MMD Calculation
    # ---------------------------------------------------------
    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]
    YX = kernel_val[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def compute_mmd(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches of samples x and y.
    Uses a multi-scale RBF kernel by averaging multiple bandwiths.
    """

    assert x.shape == y.shape, "real and predicted batches do not match in size."

    batch_size = x.size(0)
    n_samples = int(x.size(0)) + int(y.size(0))
    
    # If not enough samples to compute distribution statistics, return 0 or simple distance
    if batch_size <= 1:
        return torch.tensor(0.0, device=x.device)

    total = torch.cat([x, y], dim=0)
    
    # L2 Distance Matrix
    L2_distance = torch.cdist(total, total, p=2)**2
    
    # Bandwidth selection
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples) + 1e-5
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # multiple kernels calculation + averaging
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernel_val = sum(kernel_val)

    # MMD Calculation
    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]
    YX = kernel_val[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def train_step_perturb_model(model, data, device, alpha=1., beta=1., mmd_gamma=0.0):
    x, y, pert = data  # x,y: [B,N,1], pert: [B,N]
    x, y, pert = x.to(device), y.to(device), pert.to(device)

    B, N, _ = x.shape

    y_hat, x_hat = model((x,pert)) #[BxN,1]

    # flatten ground truth to match forward outputs for recon
    x_flat = x.reshape(B * N, 1)
    y_flat = y.reshape(B * N, 1)

    # control cells (x_hat): ELBO loss
    squared_error = (x_hat - x_flat) ** 2 # [B*N,1]
    loss_feat = squared_error.view(B, N).mean(dim=1).mean()  # sum over genes, mean over cells

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)

    # Cosine similarity (Direction)
    x_true = x_flat.view(B, N)
    y_true = y_flat.view(B, N)
    x_pred = x_hat.view(B, N)
    y_pred = y_hat.view(B, N)

    # ---- NEW: Extract perturbation weights for this batch ----
    pert_idx = pert[0].int().argmax().item()
    batch_weights = model.weight_lookup[pert_idx].to(device).reshape(-1)

    # Compute pseudobulk - used for proper cosine similarity
    x_true_mean = x_true.mean(dim=0) # [N]
    y_true_mean = y_true.mean(dim=0) # [N]
    x_pred_mean = x_pred.mean(dim=0) # [N]
    y_pred_mean = y_pred.mean(dim=0) # [N]

    delta_true_mean = y_true_mean - x_true_mean
    delta_pred_mean = y_pred_mean - x_pred_mean

    true_norm = torch.norm(delta_true_mean)
    if true_norm > 1e-6:
        cos_sim = F.cosine_similarity(delta_pred_mean, delta_true_mean, dim=0)
        loss_cosine = 1.0 - cos_sim
    else:
        loss_cosine = torch.tensor(0.0, device=device)

    # control loss: ELBO + cosine
    control_loss = alpha * loss_feat + beta * kl_div

    # mmd for perturbed cells ancd control cells
    loss_mmd_x = torch.tensor(0.0, device=device)
    loss_mmd_y = compute_mmd_withcosine(y_true, y_pred, weights=batch_weights)

    if mmd_gamma != 0.0:
        loss_mmd_x = compute_mmd(x_true, x_pred)
    else:
        loss_mmd_x = 0.0

    total_loss = control_loss + loss_mmd_y + mmd_gamma * loss_mmd_x

    return total_loss, loss_mmd_y, loss_mmd_x, kl_div, loss_cosine, loss_feat

@torch.no_grad()
def test_perturb_model(model, loader, device):
    model.eval()
    pert_mmd = []
    for i, data in enumerate(tqdm(loader, desc='testing with MMD...')):
        x, y, pert = data
        x, y, pert = x.to(device), y.to(device), pert.to(device)

        B = pert.shape[0]
        N = pert.shape[1]

        y_hat, _ = model((x,pert)) #[B*N,1]
        y_flat = y.reshape(-1, 1) #[B*N,1]

        mmd_error = compute_mmd(y_flat.view(B,N), y_hat.view(B,N))
        pert_mmd.append(mmd_error.item())

    avg_pert_mmd = sum(pert_mmd)/len(pert_mmd)
    return avg_pert_mmd


def train(model, train_loader, test_loader, lr, n_epochs, device, wandb_support, live_plot):

    global_loss = []
    mmd_pert = []
    mmd_ctrl = []
    test_wmse = []
    
    accumulation_steps = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True, weight_decay=0.0001)

    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)

    # wandb watch
    if wandb_support:
        wandb.watch(model, log="all", log_freq=10) # log="all" tracks both gradients and parameters

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
            loss, loss_mmd_y, loss_mmd_x, kl_div, loss_cosine, loss_feat = train_step_perturb_model(
                model, 
                batch, 
                model.device, 
                alpha=2.0, 
                beta=0.01)
            
            loss.backward()

            # sum, we will calculate the mean over all the epoch
            total_loss += loss.item()
            total_mmd_pert += loss_mmd_y.item()
            total_mmd_ctrl += loss_mmd_x#.item()
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
        epoch_mmd_ctrl = total_mmd_ctrl / len(train_loader)
        epoch_kl = kl_accum / len(train_loader)
        epoch_mse = mse_accum / len(train_loader)
        epoch_cos = cos_accum / len(train_loader)

        # Append to lists
        global_loss.append(epoch_loss)
        mmd_pert.append(epoch_mmd_pert)
        mmd_ctrl.append(epoch_mmd_ctrl)

        # print training methods
        print(f'training KL = {epoch_kl:.5f} | mse = {epoch_mse:.3f} | cos = {epoch_cos:.3f}')

        # save current epoch weights
        filepath = os.path.join(weights_dir, f"cheb_weights_epoch_{epoch}.pth")
        torch.save(model.state_dict(), filepath)
        
        # validation
        if epoch!=0:
            avg_feat_err = test_perturb_model(model, test_loader, model.device)
            test_wmse.append(avg_feat_err)

            # routine for plotting traning/testing metrics during the training
            if live_plot:
                epoch_axis = list(range(1, epoch + 1))
                clear_output(wait=True) 
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))       
                ax1.plot(epoch_axis, global_loss, linestyle= '-', label='Loss')
                ax1.plot(epoch_axis, mmd_pert, label='Perturb cells MMD')
                ax1.plot(epoch_axis, mmd_ctrl, label='Control cells MMD')
                ax1.set_title('Training')
                ax1.set_xlabel('Epoch')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax1.set_xlim(1, n_epochs)
                ax1.legend(frameon=False)
                ax1.grid(True)

                ax2.plot(epoch_axis, test_wmse, linewidth=1.2, label='validation WMSE')
                ax2.set_title('Validation')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MMD')
                ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax2.set_xlim(1, n_epochs)
                ax2.legend(frameon=False)
                ax2.grid(True)

                plt.tight_layout()
                plt.savefig('plot_v2.png', dpi=300, bbox_inches='tight')
                display(fig)   
                plt.close(fig) 
        
        # wandb support
        if wandb_support:
            log_metrics = {
                "epoch": epoch,
                "train/global_loss": epoch_loss,
                "train/mmd_pert": epoch_mmd_pert,
                "train/mmd_ctrl": epoch_mmd_ctrl,
                "train/kl_divergence": epoch_kl,
                "train/mse": epoch_mse,
                "train/cosine_loss": epoch_cos,
            }

            # Only log validation metric if it was calculated
            if avg_feat_err is not None:
                log_metrics["val/test_MMD"] = avg_feat_err
                
            wandb.log(log_metrics)


    return mmd_pert, mmd_ctrl, test_wmse
#
# 
# gene embeddings for identity + void token for the perturbation
#

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
from torch_geometric.utils import dropout_edge
from torch.nn import ReLU, LeakyReLU, GELU, LayerNorm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator


class VariationalGraphEncoder(torch.nn.Module):
    ''' encoder class
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ChebConv(in_channels, out_channels, 3, normalization='rw') 
        self.ln1 = LayerNorm(out_channels)
        self.conv2 = ChebConv(out_channels, 2*out_channels, 3, normalization='rw')
        self.ln2 = LayerNorm(2*out_channels)
        self.conv_mu = ChebConv(2*out_channels, out_channels, 2, normalization='rw')  
        self.conv_logstd = ChebConv(2*out_channels, out_channels, 2, normalization='rw') 
        self.dropout_rate = 0.2

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.ln1(x)
        x = F.gelu(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.ln2(x)
        x = F.gelu(x) 
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return self.conv_mu(x, edge_index),  self.conv_logstd(x, edge_index)


class FeatureDecoder(torch.nn.Module):
    ''' 
    Feature decoder
    '''
    def __init__(self, n_channels, num_node_features, edge_index):
        super().__init__()
        self.conv1 = ChebConv(n_channels, n_channels, 2, normalization='rw')
        self.ln1 = LayerNorm(n_channels)
        self.conv2 = ChebConv(n_channels, n_channels, 2, normalization='rw')
        self.ln2 = LayerNorm(n_channels)
        self.dropout_rate = 0.2
        self.edge_index = edge_index
        self.last_layer = torch.nn.Linear(n_channels, num_node_features)

    def forward(self, z, edge_index):
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.conv1(z, edge_index)
        z = self.ln1(z)
        z = F.gelu(z)

        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.conv2(z, edge_index)
        z = self.ln2(z)
        z = F.gelu(z)
        
        return self.last_layer(z)

# MASTER CLASS
class PerturbModel(torch.nn.Module):
    def __init__(self, num_node_features, n_channels, edge_index, ctrl_mean_tensor, num_nodes, device, gene_weights=None):
        super().__init__()
        self.device = device 
        self.num_nodes = num_nodes 
        self.n_channels = n_channels
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('ctrl_mean', ctrl_mean_tensor)

        self.edge_dropout_p = 0.2

        # Weight Lookup Construction - create tensor of shape [num_nodes + 1, num_nodes]
        
        #default_weights = gene_weights[-1] if gene_weights is not None else torch.ones(num_nodes)
        default_weights = (1/num_nodes)*torch.ones(num_nodes)
        weight_lookup = default_weights.unsqueeze(0).repeat(num_nodes + 1, 1).to(device)
        if isinstance(gene_weights, dict):
            for pert_idx, weight_array in gene_weights.items():
                w_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)
                if pert_idx == -1:
                    weight_lookup[num_nodes] = w_tensor
                elif 0 <= pert_idx < num_nodes:
                    weight_lookup[pert_idx] = w_tensor
                #weight_lookup[pert_idx] = w_tensor
        self.register_buffer('weight_lookup', weight_lookup)

        # 1. Gene Embeddings (Solves the Identity Problem)
        self.embedding_dim = 64 
        self.gene_embedding = torch.nn.Embedding(num_nodes, self.embedding_dim)
        self.encoder_in_channels = num_node_features + self.embedding_dim

        # Instead of 0.0, we learn a vector that means "VOID"
        self.ko_token = torch.nn.Parameter(torch.randn(1, n_channels))

        self.encoder = VariationalGraphEncoder(self.encoder_in_channels, n_channels)
        self.gex_decoder = FeatureDecoder(n_channels, num_node_features, edge_index)

        self._cached_batch_size = 0
        self._cached_edge_index = None
        self._cached_gene_ids = None

    def _get_batched_edge_index(self, batch_size):
        if batch_size == self._cached_batch_size and self._cached_edge_index is not None:
            return self._cached_edge_index
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes
        edge_index_batch = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
        edge_index_batch = edge_index_batch.reshape(2, -1)
        self._cached_batch_size = batch_size
        self._cached_edge_index = edge_index_batch
        return edge_index_batch
    
    def _get_batched_gene_ids(self, batch_size):
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

    def kl_loss_base(self, mu, logstd):
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def kl_loss(self, mu, logstd, threshold=1e-2, verbose=True, free_bits=0.05):
        """
        Computes KL loss and checks for inactive stochastic units (Posterior Collapse).
        
        Args:
            mu (Tensor): Mean of the posterior [batch_size, latent_dim]
            logstd (Tensor): Log standard deviation [batch_size, latent_dim]
            threshold (float): Value below which a dimension is considered 'dead'.
                            Standard Gaussian KL is 0 if mu=0 and std=1.
            verbose (bool): Whether to print warnings when dead units are found.
        
        Returns:
            Tensor: The scalar KL loss (averaged over batch).
        """
        
        # Calculate the KL term for every single element [batch_size, latent_dim]
        kl_raw = -0.5 * (1 + 2 * logstd - mu**2 - logstd.exp()**2)
        
        # We take the mean over the batch (dim=0) to see the behavior of the dimension generally
        kl_per_dim = torch.mean(kl_raw, dim=0)
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        
        # Find indices where the average KL is below the threshold (close to 0)
        inactive_dims = torch.where(kl_per_dim_clamped < threshold)[0]
        
        if verbose and len(inactive_dims) > 0:
            print(f"⚠️ Warning: Found {len(inactive_dims)} inactive stochastic units (KL < {threshold}).")
            print(f"   Indices: {inactive_dims.tolist()}")
            print(f"   Avg KL for these: {kl_per_dim[inactive_dims].detach().cpu().numpy()}")

        # Return the standard scalar loss Logic: Sum over dimensions (dim=1), then Mean over batch
        return torch.sum(kl_per_dim_clamped)

    def forward(self, data):
        x, pert = data
        x = x.to(self.device)
        pert = pert.to(self.device)
        batch_size, num_nodes, num_features = x.shape

        edge_index_batch = self._get_batched_edge_index(batch_size)

        x = x.reshape(batch_size * num_nodes, num_features)
        gene_ids = self._get_batched_gene_ids(batch_size)
        emb = self.gene_embedding(gene_ids)
        x_input = torch.cat([x, emb], dim=1) 

        pert = pert.reshape(batch_size * num_nodes)

        # --- Edge Dropout ---
        if self.training and self.edge_dropout_p > 0:
            edge_index_batch, _ = dropout_edge(
                edge_index_batch, 
                p=self.edge_dropout_p, 
                force_undirected=False,
                training=self.training
            )

        mu, logstd = self.encoder(x_input, edge_index_batch)
        self.last_mu = mu          
        self.last_logstd = logstd  

        z = self.reparametrize(mu, logstd)
        
        # void token for the perturbation
        z = torch.where(pert.unsqueeze(1), self.ko_token, z)

        z = self.gex_decoder(z, edge_index_batch)
        return z

    def predict_full_expression(self, data):
        predicted_delta = self.forward(data)
        return F.relu(predicted_delta + self.ctrl_mean)
    
    def generative_prediction(self, ko_gene_idx):
        self.eval()
        with torch.no_grad():
            z_basal = torch.randn(self.num_nodes, self.n_channels).to(self.device)
            
            z_perturbed = z_basal.clone()
            z_perturbed[ko_gene_idx, :] = self.ko_token
            
            edge_index = self.edge_index.to(self.device)

            predicted_delta = self.gex_decoder(z_perturbed, edge_index)
            predicted_full_gex = F.relu(predicted_delta + self.ctrl_mean)
            
            return predicted_delta, predicted_full_gex

######### training + testing routines ##########

def _get_beta_schedule(epoch, n_epochs, n_cycles=1, ratio=0.5):
    """
    Create a cyclic schedule for beta sclaer of the KL ELBO term. Default n_cycles=1 sets beta to start from 0 and slowly increase to 1.
    This is to solve kl annealing problem
    
    Args:
        epoch (int): Current epoch (0-indexed).
        n_epochs (int): Total training epochs.
        n_cycles (int): Number of cycles.
        ratio (float): Percentage of cycle spent increasing beta (0-1).
    """
    period = n_epochs // n_cycles 
    step = epoch % period
    if step < period * ratio:
        return step / (period * ratio)
    else:
        return 1.0

def train_step_perturb_model(model, data, device, alpha=1., beta=1.):
    x_ = model(data) #[BxN,1]
    x = data[0].reshape((-1,1)).to(device) #[BxN,1]

    # DE weights
    perturbation = data[1] # [B,N]
    batch_size = perturbation.shape[0]
    N = data[1].shape[1]

    # TODO: this is wrong, because perturbation may be null; in this case, for now it returns 0 which is fine, 
    # but we have to implemnt the absence of perturbation in a more reasonable and elegant way. Also this 
    # does not work with combination of perturbations
    pert_indices = perturbation.int().argmax(dim=1)
    pert_indices = pert_indices.to(device)
    is_control = perturbation.sum(dim=1) == 0
    is_control = is_control.to(device)
    weight_lookup_indices = torch.where(is_control, 
                                        torch.tensor(model.num_nodes, device=device), 
                                        pert_indices)
    specific_weights = model.weight_lookup[weight_lookup_indices]

    weights = specific_weights.reshape(-1, 1) # Shape [B*N, 1]
    normalized_weights = weights / torch.sum(weights)
    squared_error = (x_ - x)**2
    loss_feat = torch.sum(normalized_weights * squared_error) #torch.mean(squared_error * weights)

    # cosine similarity (direction)
    x_true = x.reshape(batch_size, model.num_nodes)
    x_pred = x_.reshape(batch_size, model.num_nodes)
    true_norm = torch.norm(x_true, dim=1)
    mask_has_signal = true_norm > 1e-6
    if mask_has_signal.sum() > 0:
        # Cosine Similarity: 1 - cos(x, y). We want to minimize this.
        cos_sim = F.cosine_similarity(x_pred[mask_has_signal], x_true[mask_has_signal], dim=1)
        # Loss is 1 - avg_cosine_similarity
        loss_cosine = 1.0 - torch.mean(cos_sim)
    else:
        loss_cosine = torch.tensor(0.0, device=device)

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    total_loss = alpha * loss_feat + alpha * loss_cosine + beta * (1 / model.num_nodes) * kl_div

    return total_loss, loss_feat, kl_div

def train_step_perturb_model__(model, data, device, alpha=1., beta=1.):
    x_ = model(data)
    x = data[0].reshape((-1,1)).to(device)

    # --- Weighted MSE Logic ---
    # model.gene_weights is [N]. We need to repeat it for the batch size.
    batch_size = x.shape[0] // model.num_nodes
    weights = model.gene_weights.repeat(batch_size).reshape(-1, 1) # [B * N, 1]
    
    # If a gene is perturbed, set its weight to 5.0 (High Priority)
    pert_mask = data[1].reshape(-1, 1).to(device)
    weights = torch.where(pert_mask, 5.0, weights)

    #loss_feat = F.mse_loss(x_, x, reduction='mean', weight=weights)#torch.mean(torch.sum((x_ - x)**2, dim=1)) 
    squared_error = (x_ - x)**2
    loss_feat = torch.mean(squared_error * weights)

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    total_loss = alpha * loss_feat + beta * (1 / model.num_nodes) * kl_div

    return total_loss, loss_feat, kl_div

def train_step_perturb_model_hvg(model, data, device, alpha=1., beta=1.):
    '''
    this contains wmse as well
    '''

    x_ = model(data) # shape [BxN, 1] (N=num_nodes)
    x = data[0].reshape((-1,1)).to(device) # shape [BxN,1]
    # data[1] (pert embeddings) [B,N]

    # --- Weighted MSE Logic ---
    # model.gene_weights is [N]. We need to repeat it for the batch size.
    batch_size = x.shape[0] // model.num_nodes
    weights = model.gene_weights.repeat(batch_size).reshape(-1, 1) # [B * N, 1]
    
    # If a gene is perturbed, set its weight to 5.0 (High Priority)
    pert_mask = data[1].reshape(-1, 1).to(device)
    weights = torch.where(pert_mask, 5.0, weights)

    squared_error = (x_ - x)**2
    loss_feat = torch.mean(squared_error * weights)

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    total_loss = alpha * loss_feat + beta * (1 / model.num_nodes) * kl_div

    return total_loss, loss_feat, kl_div

def train_step_perturb_model_(model, data, device, alpha=1., beta=1.):

    x_pred_flat = model(data) # [B*N, 1]
    x_true_flat = data[0].reshape((-1,1)).to(device) # [B*N, 1]

    # Reshape to [Batch, Nodes] to calculate per-sample metrics
    batch_size = x_true_flat.shape[0] // model.num_nodes
    x_pred = x_pred_flat.reshape(batch_size, model.num_nodes)
    x_true = x_true_flat.reshape(batch_size, model.num_nodes)
    
    # Weighted MSE Logic (Base)
    weights = model.gene_weights.repeat(batch_size).reshape(-1, 1) # [B*N, 1]
    pert_mask = data[1].reshape(-1, 1).to(device)
    weights = torch.where(pert_mask, 5.0, weights)
    
    squared_error = (x_pred_flat - x_true_flat)**2
    weighted_mse = torch.mean(squared_error * weights)

    # --- B. Cosine Similarity Loss (Direction) ---
    # We want the vector of the cell to point in the right direction.
    # CosineEmbeddingLoss takes inputs [B, N] and target {1, -1}. We want target 1 (similar).
    # This prevents the "Lazy Zero" prediction because 0 vector has undefined direction, 
    # but as soon as it predicts *something*, we force that something to align with truth.
    
    # Only compute on samples that have SOME signal (norm > 0) to avoid NaNs
    true_norm = torch.norm(x_true, dim=1)
    mask_has_signal = true_norm > 1e-6
    
    if mask_has_signal.sum() > 0:
        # Cosine Similarity: 1 - cos(x, y). We want to minimize this.
        cos_sim = F.cosine_similarity(x_pred[mask_has_signal], x_true[mask_has_signal], dim=1)
        # Loss is 1 - avg_cosine_similarity
        loss_cosine = 1.0 - torch.mean(cos_sim)
    else:
        loss_cosine = torch.tensor(0.0, device=device)


    # --- C. Top-K MSE Loss (Focus on Peaks) ---
    # We identify the top 50 genes with largest absolute CHANGE in the ground truth
    # and force the model to get those right.
    k = 50
    # Get indices of top k absolute values in true deltas
    # val, indices: [B, K]
    topk_vals, topk_indices = torch.topk(torch.abs(x_true), k, dim=1)
    
    # Gather the predicted values at those specific indices
    # We use gather to pick the corresponding predictions
    x_pred_topk = torch.gather(x_pred, 1, topk_indices)
    x_true_topk = torch.gather(x_true, 1, topk_indices)
    
    loss_topk = F.mse_loss(x_pred_topk, x_true_topk)

    # --------------------------

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    
    # TOTAL LOSS COMPOSITION
    # We combine these terms. 
    # Weighted MSE: General fit
    # Cosine: Fixes direction/pattern (Crucial for DES)
    # TopK: Fixes magnitude of top genes (Crucial for DES)
    
    total_loss = (
        (alpha * 2.0) * weighted_mse + 
        (alpha * 1.0) * loss_cosine +
        (alpha * 1.0) * loss_topk +
        beta * (1 / model.num_nodes) * kl_div
    )

    return total_loss, weighted_mse, kl_div


@torch.no_grad()
def test_perturb_model(model, loader, device):
    model.eval()
    feat_err = []
    for i, data in enumerate(tqdm(loader, desc='testing...')):
        x = data[0].reshape((-1,1)).to(device)
        x_ = model(data) 
        mse = ((x - x_)**2).mean() 
        feat_err.append(mse.item())
    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err

def train(model, train_loader, test_loader, lr, n_epochs, device, live_plot):
    feat_train_loss = []
    kl_train_loss = []
    feat_test_values = []
    
    accumulation_steps = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True, weight_decay=0.0001)

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_feat = 0
        total_kl = 0

        optimizer.zero_grad(set_to_none=True)
        for (i,batch) in enumerate(tqdm(train_loader, desc=f'training at epoch {epoch}')):
            
            beta = _get_beta_schedule(epoch, n_epochs, n_cycles=1, ratio=0.4)

            # Use small beta scaling
            loss, feat_loss, kl_loss = train_step_perturb_model(model, batch, model.device, alpha=1., beta=beta)
            
            loss.backward()
            total_feat += feat_loss.item()
            total_kl += kl_loss.item()

            if (i+1)%accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
        avg_feat = total_feat / len(train_loader)
        avg_kl = total_kl / len(train_loader)
        feat_train_loss.append(avg_feat)
        kl_train_loss.append(avg_kl)

        print(f'finishing the training on epoch {epoch} in {time.time()-epoch_start:.2f}s')
        
        if epoch!=0:
            avg_feat_err = test_perturb_model(model,test_loader, model.device)
            feat_test_values.append(avg_feat_err)

            if live_plot:
                epoch_axis = list(range(1, epoch + 1))
                clear_output(wait=True) 
                fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))       
                ax1.plot(epoch_axis, feat_train_loss, label='Train Feature Loss')
                ax1.plot(epoch_axis, feat_test_values, label='Test Feature MSE')
                ax1.set_title('Reconstruction Error')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax1.set_xlim(1, n_epochs)
                ax1.legend(frameon=False)
                ax1.grid(True)
                plt.tight_layout()
                plt.savefig('plot.png', dpi=300, bbox_inches='tight')
                display(fig)   
                plt.close(fig) 
            
            print(f'test performances at epoch: {epoch:03d} | MAE (features): {avg_feat_err:.6f} | train KL: {avg_kl:.6f}')

    return feat_train_loss, kl_train_loss, feat_test_values
#
# Model v5: Fixed Control Indexing, Correct Loss Scaling, Project & Add Architecture
#

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, DirGNNConv
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
        self.conv1 = DirGNNConv(ChebConv(in_channels, out_channels, 3)) 
        self.ln1 = LayerNorm(out_channels)
        self.conv2 = DirGNNConv(ChebConv(out_channels, 2*out_channels, 3))
        self.ln2 = LayerNorm(2*out_channels)
        self.conv_mu = DirGNNConv(ChebConv(2*out_channels, out_channels, 2))  
        self.conv_logstd = DirGNNConv(ChebConv(2*out_channels, out_channels, 2))
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
        #x = F.dropout(x, p=self.dropout_rate, training=self.training)

        mu = self.conv_mu(x, edge_index)
        logst = self.conv_logstd(x, edge_index)
        return mu, logst


class FeatureDecoder(torch.nn.Module):
    ''' 
    Feature decoder
    '''
    def __init__(self, n_channels, num_node_features):
        super().__init__()
        self.conv1 = DirGNNConv(ChebConv(n_channels, n_channels, 3))
        self.ln1 = LayerNorm(n_channels)
        self.conv2 = DirGNNConv(ChebConv(n_channels, n_channels, 3))
        self.ln2 = LayerNorm(n_channels)
        self.dropout_rate = 0.2
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

        # Weight Lookup Construction
        # We need N+1 rows. The last row (index N) is for CONTROL (gene_weights[-1]).
        default_weights = gene_weights[-1] if gene_weights is not None else torch.ones(num_nodes)
        weight_lookup = default_weights.unsqueeze(0).repeat(num_nodes + 1, 1).to(device)    
        if isinstance(gene_weights, dict):
            for pert_idx, weight_array in gene_weights.items():
                w_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)
                if pert_idx == -1:
                    weight_lookup[num_nodes] = w_tensor # Explicitly set the CONTROL row (index N)
                elif 0 <= pert_idx < num_nodes:
                    weight_lookup[pert_idx] = w_tensor
        self.register_buffer('weight_lookup', weight_lookup)

        # Project & Add Architecture
        self.embedding_dim = 64 
        self.gene_embedding = torch.nn.Embedding(num_nodes, self.embedding_dim)
        
        # Project scalar GEX to matching dimension
        self.gex_projection = torch.nn.Linear(num_node_features, self.embedding_dim)
        
        # Input to encoder is embedding_dim (because we ADD them, not concat)
        self.encoder_in_channels = self.embedding_dim

        # Learnable KO perturbation Token
        self.ko_token = torch.nn.Parameter(torch.randn(1, n_channels) - 2.0)

        self.encoder = VariationalGraphEncoder(self.encoder_in_channels, n_channels)
        self.gex_decoder = FeatureDecoder(n_channels, num_node_features)

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

    def kl_loss(self, mu, logstd, threshold=1e-2, verbose=True, free_bits=0.05):

        kl_raw = -0.5 * (1 + 2 * logstd - mu**2 - logstd.exp()**2)
        kl_per_dim = torch.mean(kl_raw, dim=0)
        kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        
        return torch.sum(kl_per_dim_clamped)

    def forward(self, data):
        x, pert = data
        x = x.to(self.device) #[B,N,1]

        pert = pert.to(self.device) #[B,N]

        batch_size, num_nodes, num_features = x.shape

        edge_index_batch = self._get_batched_edge_index(batch_size)

        x = x.reshape(batch_size * num_nodes, num_features)

        # Project GEX
        gex_vec = self.gex_projection(x) # [B*N, 64]

        # Get Embedding
        gene_ids = self._get_batched_gene_ids(batch_size)
        emb = self.gene_embedding(gene_ids) # [B*N, 64]

        # Add
        x_input = gex_vec + emb 

        pert = pert.reshape(batch_size * num_nodes)

        # Edge Dropout
        if self.training and self.edge_dropout_p > 0:
            edge_index_batch, _ = dropout_edge(
                edge_index_batch, 
                p=self.edge_dropout_p, 
                force_undirected=False,
                training=self.training
            )

        mu, logstd = self.encoder(x_input, edge_index_batch)
        logstd = torch.clamp(logstd, min=-20, max=10) # this is to avoid problems in the exponential

        self.last_mu = mu          
        self.last_logstd = logstd  

        z = self.reparametrize(mu, logstd)
        z = torch.where(pert.unsqueeze(1), self.ko_token, z)
        z = self.gex_decoder(z, edge_index_batch)
        return z

    def predict_full_expression(self, data):
        predicted_delta = self.forward(data)
        return F.relu(predicted_delta + self.ctrl_mean)
    
    def generative_prediction(self, ko_gene_idx, n_samples=1):
        self.eval()
        with torch.no_grad():
            # Support batched generation
            z_basal = torch.randn(self.num_nodes * n_samples, self.n_channels).to(self.device)
            
            z_perturbed = z_basal.clone()
            
            # Apply KO token to specific indices
            offsets = torch.arange(n_samples, device=self.device) * self.num_nodes
            flat_ko_indices = ko_gene_idx + offsets
            z_perturbed[flat_ko_indices, :] = self.ko_token
            
            edge_index_batch = self._get_batched_edge_index(n_samples)

            predicted_delta = self.gex_decoder(z_perturbed, edge_index_batch)
            
            # Reshape
            predicted_delta = predicted_delta.reshape(n_samples, self.num_nodes)
            
            # Add mean
            predicted_full_gex = F.relu(predicted_delta + self.ctrl_mean.reshape(1, -1))
            
            return predicted_delta, predicted_full_gex

######### training + testing routines ##########

def _get_beta_schedule(epoch, n_epochs, n_cycles=1, ratio=0.5):
    period = n_epochs // n_cycles 
    step = epoch % period
    if step < period * ratio:
        return step / (period * ratio)
    else:
        return 1.0

def train_step_perturb_model(model, data, device, alpha=1., beta=1.):
    x_ = model(data) #[BxN,1]
    x = data[0].reshape((-1,1)).to(device) #[BxN,1]

    perturbation = data[1] # [B,N]
    batch_size = perturbation.shape[0]

    # find indices of perturbations
    # TODO: this is wrong, because perturbation may be null; in this case, for now it returns 0 which is fine, 
    # but we have to implemnt the absence of perturbation in a more reasonable and elegant way. Also this 
    # does not work with combination of perturbations
    pert_indices = perturbation.int().argmax(dim=1) 
    pert_indices = pert_indices.to(device)
    
    # identify control cells
    is_control = perturbation.sum(dim=1) == 0
    is_control = is_control.to(device)

    # For Control, set index to num_nodes (special last row), for others, keep the argmax index
    weight_lookup_indices = torch.where(is_control, 
                                        torch.tensor(model.num_nodes, device=device), 
                                        pert_indices)

    # [B, N] weights
    specific_weights = model.weight_lookup[weight_lookup_indices]
    weights = specific_weights.reshape(-1, 1) # Shape [B*N, 1]

    # correct loss scaling
    squared_error = (x_ - x)**2
    squared_error_batch = squared_error.reshape(batch_size, -1)
    weights_batch = weights.reshape(batch_size, -1)
    
    # Weighted Sum per cell
    loss_per_cell = torch.sum(squared_error_batch * weights_batch, dim=1)
    #loss_per_cell = torch.mean(squared_error_batch * weights_batch, dim=1) # Use Mean
    
    # Mean over batch
    loss_feat = torch.mean(loss_per_cell)

    # Cosine similarity (Direction)
    x_true = x.reshape(batch_size, model.num_nodes)
    x_pred = x_.reshape(batch_size, model.num_nodes)
    true_norm = torch.norm(x_true, dim=1)
    mask_has_signal = true_norm > 1e-6
    
    if mask_has_signal.sum() > 0:
        cos_sim = F.cosine_similarity(x_pred[mask_has_signal], x_true[mask_has_signal], dim=1)
        loss_cosine = 1.0 - torch.mean(cos_sim)
    else:
        loss_cosine = torch.tensor(0.0, device=device)

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    
    # Scaled total loss
    total_loss = alpha * loss_feat + alpha * loss_cosine + beta * kl_div

    return total_loss, loss_feat, kl_div

@torch.no_grad()
def test_perturb_model(model, loader, device):
    model.eval()
    feat_err = []
    for i, data in enumerate(tqdm(loader, desc='testing...')):
        # Fix logic to handle batch inputs correctly
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
            
            loss, feat_loss, kl_loss = train_step_perturb_model(model, batch, model.device, alpha=1.0, beta=beta)
            
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
            
            print(f'test performances at epoch: {epoch:03d} | WMSE (Train): {avg_feat:.6f} | MSE (Test): {avg_feat_err:.6f} | KL: {avg_kl:.6f}')

    return feat_train_loss, kl_train_loss, feat_test_values
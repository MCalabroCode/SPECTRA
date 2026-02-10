#
# same as in model_v3 but with standard pytorch batching in order to make this mroe efficient
#

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, VGAE, GAE
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
        self.conv1 = ChebConv(in_channels, out_channels, 3, normalization='rw') #5
        self.ln1 = LayerNorm(out_channels)
        self.conv2 = ChebConv(out_channels, 2*out_channels, 3, normalization='rw') #5
        self.ln2 = LayerNorm(2*out_channels)
        self.conv_mu = ChebConv(2*out_channels, out_channels, 2, normalization='rw')  #5
        self.conv_logstd = ChebConv(2*out_channels, out_channels, 2, normalization='rw') #5 
        self.dropout_rate = 0.25

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

# # auxiliary class for MLP
# class MLP(torch.nn.Module):

#     def __init__(self, sizes, batch_norm=True, dropout=0.2):
#         super(MLP, self).__init__()
#         layers = []
#         for s in range(len(sizes) - 1):
#             layers = layers + [
#                 torch.nn.Dropout(p=dropout),
#                 torch.nn.Linear(sizes[s], sizes[s + 1]),
#                 torch.nn.BatchNorm1d(sizes[s + 1])
#                 if batch_norm and s < len(sizes) - 1 else None,
#                 torch.nn.GELU()
#             ]

#         layers = [l for l in layers if l is not None][:-1]
#         self.network = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         return self.network(x)

# class FeatureDecoder(torch.nn.Module):
#     ''' feature decoder class
#     '''
#     def __init__(self, n_channels, num_node_features):
#         super().__init__()
#         self.mlp = MLP([n_channels, 2*n_channels, n_channels], dropout=0.1)
#         self.last_layer = torch.nn.Sequential(
#             torch.nn.Linear(n_channels, num_node_features),
#         )

#     def forward(self, z):
#         z = self.mlp(z)
#         #return F.softplus(self.last_layer(z)) #NOTE:before, it was relu
#         return self.last_layer(z)

class FeatureDecoder(torch.nn.Module):
    ''' 
    Feature decoder is now a GNN to allow perturbation propagation.
    FIX 2: Final activation is removed to predict residuals (deltas).
    '''
    def __init__(self, n_channels, num_node_features, edge_index):
        super().__init__()
        # GNN layers to propagate the signal
        self.conv1 = ChebConv(n_channels, n_channels, 2, normalization='rw')
        self.ln1 = LayerNorm(n_channels)
        self.conv2 = ChebConv(n_channels, n_channels, 2, normalization='rw')
        self.ln2 = LayerNorm(n_channels)

        self.dropout_rate = 0.25
        self.edge_index = edge_index
        # Final layer to map back to feature space
        # NO F.softplus or F.relu, as we are predicting a DELTA
        # which can be positive or negative.
        self.last_layer = torch.nn.Linear(n_channels, num_node_features)

    def forward(self, z, edge_index):
        # z is the perturbed latent space
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.conv1(z, edge_index)
        z = self.ln1(z)
        z = F.gelu(z)
        z = F.dropout(z, p=self.dropout_rate, training=self.training)
        z = self.conv2(z, edge_index)
        z = self.ln2(z)
        z = F.gelu(z)
        
        # Return the predicted DELTA
        return self.last_layer(z)

# MASTER CLASS
class PerturbModel(torch.nn.Module):
    def __init__(self, num_node_features, n_channels, edge_index, ctrl_mean_tensor, num_nodes, device):
        super().__init__()
        self.device = device 
        self.num_nodes = num_nodes 
        self.n_channels = n_channels
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('ctrl_mean', ctrl_mean_tensor) # ctrl_mean for residuals

        # Hyperparameter for edge dropout probability
        self.edge_dropout_p = 0.25

        # DE weights - TODO: this sucks
        weight_lookup = gene_weights[-1].unsqueeze(0).repeat(num_nodes, 1)
        print(weight_lookup.shape)
        if isinstance(gene_weights, dict):
            for pert_idx, weight_array in gene_weights.items():
                if 0 <= pert_idx < num_nodes:
                    weight_lookup[pert_idx] = torch.tensor(weight_array, dtype=torch.float32, device=device)
        self.register_buffer('weight_lookup', weight_lookup)

        self.encoder = VariationalGraphEncoder(num_node_features, n_channels)
        self.gex_decoder = FeatureDecoder(n_channels, num_node_features, edge_index)
        
        # NOTE: void token foir pertubration encoding
        self.ko_token = torch.nn.Parameter(torch.randn(1, n_channels))

        # Pre-allocate buffers for batched edge indices (will be resized as needed)
        self._cached_batch_size = 0
        self._cached_edge_index = None

    def _get_batched_edge_index(self, batch_size):
        """Efficiently create batched edge index with caching"""
        if batch_size == self._cached_batch_size and self._cached_edge_index is not None:
            return self._cached_edge_index
        
        # Create offsets: [0, num_nodes, 2*num_nodes, ...]
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes
        
        # Broadcast and add: shape [2, num_edges] -> [2, batch_size, num_edges]
        edge_index_batch = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
        
        # Reshape to [2, batch_size * num_edges]
        edge_index_batch = edge_index_batch.reshape(2, -1)
        
        # Cache the result
        self._cached_batch_size = batch_size
        self._cached_edge_index = edge_index_batch
        
        return edge_index_batch

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def _kl_loss(self, mu, logstd):
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
        # x should have shape [B,N,C]
        # pert should have shape [B,N]

        x, pert = data
        x = x.to(self.device)
        pert = pert.to(self.device)
        batch_size, num_nodes, num_features = x.shape

        # # Build batched edge_index manually (adds offsets for each batch sample)
        # edge_index_batch = []
        # for i in range(batch_size):
        #     edge_index_batch.append(self.edge_index + i * num_nodes) # each edge (u, v) becomes (u + inum_nodes, v + inum_nodes)
        # edge_index_batch = torch.cat(edge_index_batch, dim=1)
        edge_index_batch = self._get_batched_edge_index(batch_size)

        x = x.reshape(batch_size * num_nodes, num_features)
        pert = pert.reshape(batch_size * num_nodes)

        # --- EDGE DROPOUT ---
        # only apply during training
        if self.training and self.edge_dropout_p > 0:
            # This function automatically masks both indices and edge weights (attributes)
            edge_index_batch, _ = dropout_edge(
                edge_index_batch, 
                p=self.edge_dropout_p, 
                force_undirected=False, # Set True if your graph is undirected
                training=self.training
            )
        # --------------------

        mu, logstd = self.encoder(x, edge_index_batch)
        self.last_mu = mu          # Store for loss calculation
        self.last_logstd = logstd  # Store for loss calculation
        z = self.reparametrize(mu, logstd) # Use the sampled z

        #z[pert] = 0.
        z = torch.where(pert.unsqueeze(1), self.ko_token, z)
        z = self.gex_decoder(z, edge_index_batch)
        return z

    def predict_full_expression(self, data):
        """
        Helper function to get the full, absolute GEX profile.
        """
        predicted_delta = self.forward(data)
        # Add the control mean back
        return F.relu(predicted_delta + self.ctrl_mean)
    
    def generative_prediction(self, ko_gene_idx):
        """
        Generates one sample of post-perturbation DELTA.
        """
        self.eval()
        with torch.no_grad():

            # 1. Sample basal state from prior
            z_basal = torch.randn(self.num_nodes, self.n_channels)

            # edge_index = edge_index.to(self.device)
            z_basal = z_basal.to(self.device)
            
            # 2. Apply intervention
            z_perturbed = z_basal.clone()
            #z_perturbed[ko_gene_idx, :] = 0.
            z_perturbed[ko_gene_idx, :] = self.ko_token

            #edge_index = edge_index.to(self.device)

            # 3. Decode the predicted delta
            predicted_delta = self.gex_decoder(z_perturbed, self.edge_index)
            
            # 4. (Optional) Add mean back to get full expression
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
    period = n_epochs // n_cycles # period is n_epochs//n_cycles epochs long
    step = epoch % period
    
    # Linear warmup from 0 to 1; becomes 1 after ratio of period
    if step < period * ratio:
        return step / (period * ratio)
    else:
        return 1.0

def train_step_perturb_model(model, data, device, alpha=1., beta=1.):
    x_ = model(data)
    x = data[0].reshape((-1,1)).to(device)

    loss_feat = F.mse_loss(x_, x, reduction='mean') #mse

    # Get mu and logstd stored during the forward pass
    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    beta = 1.
    total_loss = alpha * loss_feat + beta * (1 / model.num_nodes) * kl_div

    return total_loss, loss_feat, kl_div

def train_step(model, data, alpha=1., beta=1., lambda_sparsity=0.005):

    
    z = model.encode(data.x, data.edge_index, data.edge_attr)
    loss_adj = model.recon_loss(z, data.edge_index, data.neg_edge_index) #TODO: this must be modified

    x_ = model.decode_features(z)
    x = data.x
    loss_feat = F.mse_loss(x_, x) + lambda_sparsity * torch.mean(torch.abs(x_))
    
    kl = model.kl_loss()
    
    loss = alpha * loss_adj + beta * loss_feat + (1 / data.num_nodes) * kl

    return loss, loss_feat

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

@torch.no_grad()
def test_perturb_model(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    feat_err = []
    plt.figure(figsize=(15,10))
    z_latent_total = []
    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        #data = data.to(device)
        x = data[0].reshape((-1,1)).to(device)
        x_ = model(data) 
        #mae = ((x - x_).abs()).mean() # this is actually MAE on all the genes
        mae = ((x - x_)**2).mean() # MSE
        feat_err.append(mae.item())

    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err


@torch.no_grad()
def test(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    feat_err = []

    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        data = data.to(device)
        Z = model.encode(data.x, data.edge_index, data.edge_attr) 
        x_ = model.decode_features(z)
        mae = ((data.x - x_)**2).mean() # MSE
        feat_err.append(mae.item())

    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err



def train(model, train_loader, test_loader, lr, n_epochs, device, live_plot):
    ''' Routine to train and evaluate the model
    '''

    feat_train_loss = []
    kl_train_loss = []
    feat_test_values = []

    #model = torch.compile(model)
    accumulation_steps = 2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True, weight_decay=0.001)

    #  training/testing Loop 
    for epoch in range(1, n_epochs + 1):

        epoch_start = time.time()
        model.train()

        total_feat = 0
        total_kl = 0

        # Clear any leftover gradients from previous epoch
        optimizer.zero_grad(set_to_none=True)
        for (i,batch) in enumerate(tqdm(train_loader, desc=f'training at epoch {epoch}')):
            #optimizer.zero_grad(set_to_none=True)

            beta = _get_beta_schedule(epoch, n_epochs, n_cycles=1, ratio=0.5)
            loss, feat_loss, kl_div = train_step_perturb_model(model, batch, model.device, alpha=1., beta=beta)
            loss.backward()
            total_feat += feat_loss.item()
            total_kl += kl_div.item()

            if (i+1)%accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
        avg_feat = total_feat / len(train_loader)
        avg_kl = total_kl / len(train_loader)

        feat_train_loss.append(avg_feat)
        kl_train_loss.append(avg_kl)

        print(f'finishing the training on epoch {epoch} in {time.time()-epoch_start}s')
        
        if epoch!=0:
            avg_feat_err = test_perturb_model(model,test_loader, model.device)
            feat_test_values.append(avg_feat_err)

            # --- Live Plotting ---
            if live_plot:

                epoch_axis = list(range(1, epoch + 1))

                # Clear the previous plot
                clear_output(wait=True) 
                
                fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))       

                # Plot 1: Node features loss
                ax1.plot(epoch_axis, feat_train_loss, label='Train Feature MSE')
                ax1.plot(epoch_axis, feat_test_values, label='Test Feature MSE')
                ax1.set_title('MSE on node features')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('MSE')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax1.set_xlim(1, n_epochs)
                ax1.legend(frameon=False)
                ax1.grid(True)

                plt.tight_layout()
                plt.savefig('plot.png', dpi=300, bbox_inches='tight')
                display(fig)   # Use display to show the plot in the notebook
                plt.close(fig) # Close the figure to prevent it from displaying twice
            
            # Print the text output after the plot
            print(f'test performances at epoch: {epoch:03d} | MAE (features): {avg_feat_err:.6f}, | kl (training): {avg_kl:.6f}')

    return feat_train_loss, feat_test_values

@torch.no_grad()
def test_perturb_model(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    feat_err = []
    z_latent_total = []
    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        #data = data.to(device)
        x = data[0].reshape((-1,1)).to(device)
        x_ = model(data) 
        mae = ((x - x_).abs()).mean() # this is actually MAE on all the genes
        #mae = ((x - x_)**2).mean() # MSE
        feat_err.append(mae.item())

        # latent_z = model.reparametrize(model.last_mu, model.last_logstd).detach().cpu().numpy().flatten()
        # z_latent_total.append(latent_z)

    # z_latent_total = np.vstack(z_latent_total)
    # pca = PCA(n_components=2)
    # z_pca = pca.fit_transform(z_latent_total)

    # # Option B: t-SNE (Better for local structure)
    # tsne = TSNE(n_components=2, perplexity=30)
    # z_tsne = tsne.fit_transform(latent_z)

    # # Plot
    # plt.scatter(z_pca[:, 0], z_pca[:, 1], alpha=0.6, s=0.5)
    # plt.title("Node Embeddings Projected to 2D")
    # plt.show()
    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err

from torch.utils.data import DataLoader

@torch.no_grad()
def test_and_plot_latent_space(model, dataset, device, modality='pca'):
    ''' custom testing function
    '''
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    model.eval()

    feat_err = []

    z_latent_total = []
    labels = []

    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        x = data[0].reshape((-1,1)).to(device)
        pert_indices = data[1].flatten().to(device)
        x_ = model(data) 
        mae = ((x - x_).abs()).mean() # this is actually MAE on all the genes
        #mae = ((x - x_)**2).mean() # MSE
        feat_err.append(mae.item())

        #latent_z = model.reparametrize(model.last_mu, model.last_logstd).detach().cpu().numpy().flatten()
        latent_z = model.last_mu
        #latent_z[pert_indices] = 0.
        latent_z = latent_z.detach().cpu().numpy().flatten()

        z_latent_total.append(latent_z)
        pert_indices = pert_indices.cpu().numpy()
        labels.append(int(pert_indices.sum())) 
        
    z_latent_total = np.vstack(z_latent_total)
    if modality=='pca':
        pca = PCA(n_components=5)
        z_reduced = pca.fit_transform(z_latent_total)

        # explained = pca.explained_variance_ratio_
        # cumulative = np.cumsum(explained)
        # components = np.arange(1, len(explained) + 1)
        # plt.figure(figsize=(15, 10))
        # plt.bar(components, explained, label='Individual Explained Variance', alpha=0.7)
        # plt.step(components, cumulative, where='mid', label='Cumulative Explained Variance')
        # plt.xlabel('Principal Component')
        # plt.ylabel('Explained Variance Ratio')
        # plt.title('PCA Explained Variance')
        # plt.xticks(components)
        # plt.legend(loc='best')
        # plt.tight_layout()
        # plt.show()
    else:
        # Option B: t-SNE (Better for local structure)
        tsne = TSNE(n_components=2, perplexity=30)
        z_reduced = tsne.fit_transform(z_latent_total)

    # Plot
    plt.figure(figsize=(15,10))
    scatter = plt.scatter(z_reduced[:, 0], z_reduced[:, 1], c=labels, alpha=0.6, s=0.5)
    plt.colorbar(scatter, label='Perturbation Intensity')
    plt.title("Node Embeddings Projected to 2D")
    plt.show()
    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err
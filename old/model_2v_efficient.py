# in this code, we revome the GAE part, mantaining this variational but without the graph reconstruction. ALSO, we do not pass all the edge_index!!

import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv, ARMAConv
from torch.nn import ReLU, LeakyReLU, GELU, LayerNorm
import numpy as np

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

        # n_stacks = 1 
        # n_layers = 2 # This is the K replacement. Much faster than 5!

        # # Replaced ChebConv with ARMAConv 
        # self.conv1 = ARMAConv(in_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv2 = ARMAConv(out_channels, 2*out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv_mu = ARMAConv(2*out_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv_logstd = ARMAConv(2*out_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        self.dropout_rate = 0.1

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.ln1(x)
        x = F.gelu(x)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.ln2(x)
        x = F.gelu(x) 

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        mu = self.conv_mu(x, edge_index, edge_weight=edge_attr) 
        logstd = self.conv_logstd(x, edge_index, edge_weight=edge_attr)
        logstd = torch.clamp(logstd, min=-10, max=2)
        
        return mu, logstd

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.GELU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Perturbation Embedding
class PerturbEmbed(torch.nn.Module):
    """
    Maps one-hot perturbation ids to a dense embedding.
    """
    def __init__(self, n_perturb, d_p):
        super().__init__()
        self.embedding = torch.nn.Embedding(n_perturb, d_p)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, pert_ids):
        """
        pert_ids: [B] (long) tensor of perturbation indices
        returns: [B, d_p]
        """
        return self.embedding(pert_ids)

class PerturbModel(torch.nn.Module):
    def __init__(self, num_node_features, n_channels, edge_index, ctrl_gex, device):
        super().__init__()

        #embed_dim = 256
        self.num_genes = ctrl_gex.shape[0]
        self.device = device

        # Register the edge_index as a 'buffer'; this makes it part of the model
        self.register_buffer('edge_index', edge_index)
        self.ctrl_gex = ctrl_gex

        # encoder layers
        self.encoder = VariationalGraphEncoder(num_node_features, n_channels) # gex
        self.pert_embedding = PerturbEmbed(self.num_genes, n_channels)

        # decoder layers
        self.mlp = MLP([2*n_channels, n_channels], dropout=0.3)
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(n_channels, self.num_genes), # before this was (n_channels, 1)
        )

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu, logstd):
        # same as in GVAE
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, edge_attr, pert):
        '''
        edge_attrs = edge weights (0 for edges affected by the pèerturbation, 1 otherwise)
        pert = perturbations list, indices (from 0 to self.num_perturbs-1)
        '''
        
        #batch_size, num_nodes, num_features = x.shape
        batch_size, num_edges, num_edge_features = edge_attr.shape 

        pert = pert.reshape(-1)

        # pert will have shape [B,num_perturb]. TODO: in VCC data num_perturb is always 1, but in general can change!

        # Flatten x and edge_attr from [batch_size, num_nodes, num_features] to [batch_size*num_nodes, num_features]
        # x = x.reshape(batch_size * num_nodes, num_features)
        edge_attr = edge_attr.reshape(batch_size * num_edges, num_edge_features)

        # concat the perturb with the control gex
        self.ctrl_gex = self.ctrl_gex.to(self.device)
        ctrl_batch = self.ctrl_gex.repeat(batch_size,1)
        #x = torch.cat([ctrl_batch, x], dim=1)
        x = ctrl_batch

        # Build batched edge_index manually (adds offsets for each batch sample)
        edge_index_batch = []
        for i in range(batch_size):
            edge_index_batch.append(self.edge_index + i * self.num_genes) # each edge (u, v) becomes (u + inum_nodes, v + inum_nodes)
        edge_index_batch = torch.cat(edge_index_batch, dim=1)

        mu, logstd = self.encoder(x, edge_index_batch, edge_attr)
        self.last_mu = mu        # Store for loss calculation
        self.last_logstd = logstd  # Store for loss calculation

        z = self.reparametrize(mu, logstd) # here z has shape [num_nodes x batch_size, n_channels]
        z = z.view(batch_size, self.num_genes, -1) #(batch_size, num_nodes, n_channels)

        p = self.pert_embedding(pert) #shape: [batch_size, n_channels] (n_channels should be the same dimension of z)

        z = z.mean(dim=1) # after this, z has shape [batch_size, n_channels]

        z = torch.cat([z, p], dim=1) # [batch-size, 2*n_channels] - for each sample (graph) in the batch, we have a state vector of shape  [1,2*n_channels]
        z = self.mlp(z) 

        # z now has shape [batch_size, n_channels]
        z = self.last_layer(z) # shape [batch-size,num_genes]

        #ctrl = x[:,0].view(-1,1)
        return x + z.reshape(-1,1)


######### training + testing routines ##########

def train_step_perturb_model(model, edge_attr_batch, y_batch, pert_batch, batched_hvg_mask, alpha=1., beta=1., lambda_sparsity=0.000):

    #x_ = model(x_batch, edge_attr_batch, embedding_batch)
    x_ = model(edge_attr_batch, pert_batch)

    # mu_mean = model.last_mu.mean().item()
    # mu_std  = model.last_mu.std().item()
    # logvar_mean = model.last_logstd.mean().item()
    # logvar_std  = model.last_logstd.std().item()
    # print(f'mean mu = {mu_mean} | std mu {mu_std} || mean logvar {logvar_mean} | std logvar {logvar_std} |')

    loss_feat = F.mse_loss(x_[batched_hvg_mask], y_batch[batched_hvg_mask]) #mse

    # Get mu and logstd stored during the forward pass
    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    total_loss = alpha*loss_feat + beta*kl_div

    return total_loss, loss_feat, kl_div


@torch.no_grad()
def test_perturb_model(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    feat_err = []

    for i, (edge_attr_batch, pert_batch, y_batch) in enumerate(tqdm(loader, desc='testing...')):
        #x_batch = x_batch.to(device)
        edge_attr_batch = edge_attr_batch.to(device)
        pert_batch = pert_batch.to(device)
        y_batch = y_batch.to(device).reshape(-1, 1)

        x_ = model(edge_attr_batch, pert_batch)

        #mae = ((data.y - x_).abs()).mean() # this is actually MAE on all the genes
        mae = ((y_batch - x_)**2).mean() # MSE
        feat_err.append(mae.item())

    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err


import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

def train(model, train_loader, test_loader, lr, n_epochs, device, hvg_mask, live_plot):
    ''' Routine to train and evaluate the model
    '''

    features_loss = []
    kl_total_loss = []
    feat_test_values = []

    #kl_warmup_epochs = n_epochs
    #model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # accumulation_steps = 2
    warmup_epochs = 5

    hvg_mask = hvg_mask.to(device)

    #  training/testing Loop 
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()

        total_loss, total_feat, total_kl = 0, 0, 0
        if epoch < warmup_epochs:
            current_beta = 0.0
        else:
            # Calculate how far into the ramp-up we are
            ramp_progress = (epoch - warmup_epochs) / n_epochs
            current_beta = 1 * min(ramp_progress, 1.0)

        for edge_attr_batch, pert_batch, y_batch in train_loader:
            
            batched_hvg_mask = hvg_mask.repeat(edge_attr_batch.shape[0])

            #x_batch = x_batch.to(device)
            edge_attr_batch = edge_attr_batch.to(device)
            pert_batch = pert_batch.to(device)
            #embedding_batch = embedding_batch.to(device).reshape(-1,1) # size: [B,5120]

            # y_batch starts as [B, N, 1], must flatten it to [B*N, 1] to match the model's output
            y_batch = y_batch.to(device).reshape(-1, 1)

            loss, feat_loss, kl_loss = train_step_perturb_model(model, edge_attr_batch, y_batch, pert_batch, batched_hvg_mask, alpha=1., beta=current_beta, lambda_sparsity=0.000)

            loss.backward()
            total_feat += feat_loss.item() # before this was loss.item() 
            total_kl += kl_loss.item()
            optimizer.step()
            optimizer.zero_grad()

        avg_kl = total_kl/len(train_loader)
        avg_feat = total_feat / len(train_loader)
        features_loss.append(avg_feat)
        kl_total_loss.append(avg_kl)

        print(f'finishing the training on epoch {epoch} in {time.time()-epoch_start}s')

        if epoch!=-1:
            avg_feat_err = test_perturb_model(model, test_loader, device)
            feat_test_values.append(avg_feat_err)

            # --- Live Plotting ---
            if live_plot:

                epoch_axis = list(range(1, epoch + 1))

                # Clear the previous plot
                clear_output(wait=True) 
                
                fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))       

                # Plot 1: Node features loss
                ax1.plot(epoch_axis, features_loss, label='Train Feature MSE')
                ax1.plot(epoch_axis, kl_total_loss, label='kl')
                ax1.plot(epoch_axis, feat_test_values, label='Test Feature MSE')
                ax1.set_title('MSE on node features')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('MSE')
                #ax1.tick_params(axis='y', labelcolor='royalblue')
                ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax1.set_xlim(1, n_epochs)
                ax1.legend(frameon=False)
                ax1.grid(True)

                # ax2 = ax1.twinx()
                # ax2.plot(epoch_axis, feat_test_values, label='Test Feature MSE', color='darkorange')
                # ax2.tick_params(axis='y', labelcolor='darkorange')

                # plt.legend(frameon=False)
                plt.tight_layout()
                display(fig)   # Use display to show the plot in the notebook
                plt.close(fig) # Close the figure to prevent it from displaying twice
            
            # Print the text output after the plot
            print(f'test performances at epoch: {epoch:03d} | MAE (features): {avg_feat_err:.6f}')

    return feat_test_values
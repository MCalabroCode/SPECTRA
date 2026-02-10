# in this code, we revome the GAE part, mantaining this variational but without the graph reconstruction

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

        # # --- Replaced ChebConv with ARMAConv ---
        # self.conv1 = ARMAConv(in_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv2 = ARMAConv(out_channels, 2*out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv_mu = ARMAConv(2*out_channels, out_channels,
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
        
        return self.conv_mu(x, edge_index, edge_weight=edge_attr),  self.conv_logstd(x, edge_index, edge_weight=edge_attr)

# auxiliary class for MLP
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

class PerturbModel(torch.nn.Module):
    def __init__(self, num_node_features, n_channels):
        super().__init__()
        self.encoder = VariationalGraphEncoder(num_node_features, n_channels)
        self.mlp = MLP([n_channels, n_channels], dropout=0.3)
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(n_channels, 1),
        )

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu, logstd):
        # same as in GVAE
        return -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

    def forward(self, x, edge_index, edge_attr):
        # z = self.encoder(x, edge_index)
        # z = self.mlp(z)
        # ctrl = x[:,0].view(-1,1)
        # return ctrl + F.relu(self.last_layer(z))

        mu, logstd = self.encoder(x, edge_index, edge_attr)
        self.last_mu = mu        # Store for loss calculation
        self.last_logstd = logstd  # Store for loss calculation

        z = self.reparametrize(mu, logstd) # Use the sampled z

        z = self.mlp(z)
        ctrl = x[:,0].view(-1,1)
        return ctrl + self.last_layer(z)


######### training + testing routines ##########

def train_step_perturb_model(model, data, optimizer, alpha=1., beta=1., lambda_sparsity=0.005):

    x_ = model(data.x, data.edge_index, data.edge_attr)
    x = data.y # gex to predict
    loss_feat = F.mse_loss(x_, x) #mse

    # Get mu and logstd stored during the forward pass
    kl_div = model.kl_loss(model.last_mu, model.last_logstd)
    total_loss = alpha*loss_feat + beta*kl_div

    return total_loss, loss_feat


@torch.no_grad()
def test_perturb_model(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    feat_err = []

    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        data = data.to(device)
        x_ = model(data.x, data.edge_index, data.edge_attr) 
        #mae = ((data.y - x_).abs()).mean() # this is actually MAE on all the genes
        mae = ((data.y - x_)**2).mean() # MSE
        feat_err.append(mae.item())

    avg_feat_err = sum(feat_err)/len(feat_err)
    return avg_feat_err


import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from tqdm import tqdm

def train(model, train_loader, test_loader, lr, n_epochs, device, live_plot):
    ''' Routine to train and evaluate the model
    '''

    features_loss = []
    feat_test_values = []

    #model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    accumulation_steps = 2

    #  training/testing Loop 
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()

        total_loss, total_adj, total_feat = 0, 0, 0

        for (i,batch) in enumerate(train_loader):
            batch = batch.to(device)
            loss, feat_loss = train_step_perturb_model(model, batch, optimizer, alpha=1., beta=1., lambda_sparsity=0.000)

            loss.backward()
            total_feat += feat_loss.item() # before this was loss.item() 

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

        avg_feat = total_feat / len(train_loader)
        features_loss.append(avg_feat)

        print(f'finishing the training on epoch {epoch} in {time.time()-epoch_start}s')
        #if epoch==1 or epoch%5==0:
        if epoch!=0:
            avg_feat_err = test_perturb_model(model,test_loader, device)
            feat_test_values.append(avg_feat_err)

            # --- Live Plotting ---
            if live_plot:

                epoch_axis = list(range(1, epoch + 1))

                # Clear the previous plot
                clear_output(wait=True) 
                
                fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))       

                # Plot 1: Node features loss
                ax1.plot(epoch_axis, features_loss, label='Train Feature MSE')
                ax1.plot(epoch_axis, feat_test_values, label='Test Feature MSE')
                ax1.set_title('MSE on node features')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('MSE')
                #ax1.tick_params(axis='y', labelcolor='royalblue')

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
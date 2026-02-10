
#############################################################################################################
#######inspired by: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/autoencoder.py #######
#############################################################################################################

import torch
import torch.nn.functional as F
from torch_geometric.nn import GAE, VGAE, ChebConv, ARMAConv
from torch.nn import ReLU, LeakyReLU, GELU
import numpy as np

class VariationalGraphEncoder(torch.nn.Module):
    ''' encoder class
    '''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.conv1 = ChebConv(in_channels, out_channels, 3) #5
        self.conv2 = ChebConv(out_channels, 2*out_channels, 3) #5
        self.conv_mu = ChebConv(2*out_channels, out_channels, 2)  #5
        self.conv_logstd = ChebConv(2*out_channels, out_channels, 2) #5 
        
        # n_stacks = 1 
        # n_layers = 2 # This is the K replacement. Much faster than 5!

        # # --- Replaced ChebConv with ARMAConv ---
        # self.conv1 = ARMAConv(in_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv2 = ARMAConv(out_channels, 2*out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv_mu = ARMAConv(2*out_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers) 
        # self.conv_logstd = ARMAConv(2*out_channels, out_channels,
        #                     num_stacks=n_stacks, num_layers=n_layers)
        self.dropout_rate = 0.2

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.gelu(x) 
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

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

class FeatureDecoder(torch.nn.Module):
    ''' feature decoder class
    '''
    def __init__(self, n_channels, num_node_features):
        super().__init__()

        self.mlp = MLP([n_channels, n_channels], dropout=0.1)
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(n_channels, num_node_features),
        )

    def forward(self, z):
        #z = F.dropout(z, p=0.3, training=self.training) # this must be removed if considering just GraphConv
        z = self.mlp(z)
        return F.relu(self.last_layer(z))

# this is the master class for the VGAE
class DualDecoderVGAE(VGAE):
    ''' VGAE class containing an additional decoder for feature reconstruction

    '''
    def __init__(self, num_node_features, n_channels): #TODO: temp thing with ctrl_adata
        super().__init__(VariationalGraphEncoder(num_node_features, n_channels))
        self.feature_decoder = FeatureDecoder(n_channels, num_node_features)

    def decode_features(self, z):
        ''' runs the decoder for node features
        '''
        return self.feature_decoder(z)


class PerturbModel(torch.nn.Module):
    def __init__(self, num_node_features, n_channels):
        super().__init__()
        self.encoder = VariationalGraphEncoder(num_node_features, n_channels)
        self.mlp = MLP([n_channels, n_channels], dropout=0.1)
        self.last_layer = torch.nn.Sequential(
            torch.nn.Linear(n_channels, num_node_features),
        )

    def forward(self, x, edge_index):
        z,_ = self.encoder(x, edge_index)
        z = self.mlp(z)
        return x + F.relu(self.last_layer(z))

############ training + tesing rountines ###########

import time
def train_step(model, data, optimizer, alpha=1., beta=1., lambda_sparsity=0.005):

    # optimizer.zero_grad()
    z = model.encode(data.x, data.edge_index)

    loss_adj = model.recon_loss(z, data.edge_index, data.neg_edge_index)

    x_ = model.decode_features(z)
    x = data.y # gex to predict
    loss_feat = F.mse_loss(x_, x) #+ lambda_sparsity * torch.mean(torch.abs(x_))

    kl = model.kl_loss()
    
    loss = alpha * loss_adj + beta * loss_feat + (1 / data.num_nodes) * kl
    #loss.backward()

    # optimizer.step()

    return loss#, loss_adj.item(), loss_feat.item(), kl.item() # before was loss.itm()

def train_step_perturb_model(model, data, optimizer, alpha=1., beta=1., lambda_sparsity=0.005):

    # optimizer.zero_grad()
    x_ = model(data.x, data.edge_index)

    #loss_adj = model.recon_loss(z, data.edge_index, data.neg_edge_index)

    #x_ = model.decode_features(z)
    x = data.y # gex to predict
    loss_feat = F.mse_loss(x_, x) #+ lambda_sparsity * torch.mean(torch.abs(x_))

    #kl = model.kl_loss()
    
    #loss = alpha * loss_adj + beta * loss_feat + (1 / data.num_nodes) * kl
    #loss.backward()

    # optimizer.step()

    return loss_feat#, loss_adj.item(), loss_feat.item(), kl.item() # before was loss.itm()

def train_step_freeze(model, data, optimizer, alpha=1., beta=1., lambda_sparsity=0.005, freeze_state=None, 
patience=5, 
graph_loss_threshold=0.01,
):
    """
    Performs one training step. Automatically freezes encoder and graph decoder
    once graph reconstruction stabilizes.

    Args:
        model: VGAE_FeatDec instance
        data: PyG Data object
        optimizer: current optimizer
        alpha, beta: loss weights
        lambda_sparsity: L1 regularization for sparsity
        freeze_state: dict tracking freezing state between epochs
        patience: epochs to check stabilization
        graph_loss_threshold: min graph loss to trigger freezing
    """

    # Initialize tracking dictionary
    if freeze_state is None:
        freeze_state = {"history": [], "frozen": False}

    model.train()
    optimizer.zero_grad()

    # --- Forward pass ---
    z = model.encode(data.x, data.edge_index)
    loss_adj = model.recon_loss(z, data.pos_edge_label_index)
    x_ = model.decode_features(z)
    loss_feat = F.mse_loss(x_, data.x) + lambda_sparsity * torch.mean(torch.abs(x_))
    kl = model.kl_loss()

    loss = alpha * loss_adj + beta * loss_feat + (1 / data.num_nodes) * kl
    loss.backward()
    optimizer.step()

    # Update graph loss history
    freeze_state["history"].append(loss_adj.item())
    if len(freeze_state["history"]) > patience:
        recent = freeze_state["history"][-patience:]
        if (
            np.std(recent) < 5e-3
            and np.mean(recent) < graph_loss_threshold
            and not freeze_state["frozen"]
        ):
            print("\n⚙️  Freezing encoder + graph decoder (graph loss stabilized)\n")
            for param in model.encoder.parameters():
                param.requires_grad = False
            # VGAE’s internal graph decoder is model.decoder
            for param in model.decoder.parameters():
                param.requires_grad = False

            # Recreate optimizer for only the feature decoder
            optimizer = torch.optim.Adam(model.mlp.parameters(), lr=5e-3)
            optimizer.add_param_group({"params": model.feat_decoder.parameters()})
            freeze_state["frozen"] = True

    return (
        loss.item(),
        loss_adj.item(),
        loss_feat.item(),
        kl.item(),
        optimizer,
        freeze_state,
    )


@torch.no_grad()
def test(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    aucs, aps, feat_err = [], [], []

    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        data = data.to(device)
        z = model.encode(data.x, data.edge_index) 
        auc, ap = model.test(z, data.edge_index, data.neg_edge_index)

        x_ = model.decode_features(z)
        #mse = ((data.y - x_)**2).mean()
        mse = ((data.y - x_).abs()).mean() # this is actually MAE on all the genes
        feat_err.append(mse.item())
        aucs.append(auc)
        aps.append(ap)

    avg_feat_err = sum(feat_err)/len(feat_err)
    avg_auc = sum(aucs) / len(aucs)
    avg_ap  = sum(aps) / len(aps)
    return avg_auc, avg_ap, avg_feat_err

@torch.no_grad()
def test_perturb_model(model, loader, device):
    ''' custom testing function
    '''
    model.eval()
    feat_err = []

    for i, data in enumerate(tqdm(loader, desc='testing...')):  # enumerate so you also know which sample
        data = data.to(device)
        x_ = model(data.x, data.edge_index) 
        mae = ((data.y - x_).abs()).mean() # this is actually MAE on all the genes
        feat_err.append(mae.item())

    avg_feat_err = sum(feat_err)/len(feat_err)

    return avg_feat_err


import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import time
from tqdm import tqdm

def train(model, train_loader, test_loader, lr, n_epochs, device, live_plot):
    ''' Routine to train and evaluate the model

    Args:
        model (torch.nn.Model)
        train_loader, test_loader: pytorch geometric dataloaders for training and testing
        lr (float): learning rate
        n_epochs (int): number of epochs
        device: cuda or cpu
        live_plot (Bool): if true, activates live plotting of feature reconstruction loss and graph reconstruction prformances during training runtime
    Returns:
        feat_test_values, auc_test_values, ap_test_values (lists): list of feature MSE, AUC and AP testing performances
    '''

    loss_values = []
    features_loss = []
    recon_loss = []

    auc_test_values = []
    ap_test_values = []
    feat_test_values = []

    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    accumulation_steps = 2

    #  training/testing Loop 
    for epoch in range(1, n_epochs + 1):
        epoch_start = time.time()
        model.train()

        total_loss, total_adj, total_feat = 0, 0, 0
        #optimizer.zero_grad()

        #for (i,batch) in enumerate(tqdm(train_loader, desc=f'progress epoch {epoch}')):
        for (i,batch) in enumerate(train_loader):
            batch_start = time.time()
            # train_start_time = time.time()
            batch = batch.to(device)
            #print(f'time to pass to device:{time.time()-batch_start}s')

            loss_start = time.time()
            #loss, loss_adj, loss_feat, _ = train_step(model, batch, optimizer, alpha=1., beta=1., lambda_sparsity=0.000)
            loss = train_step_perturb_model(model, batch, optimizer, alpha=1., beta=1., lambda_sparsity=0.000)
            #print(f'time to calculate the loss {time.time()-loss_start}')
            # Scale loss for gradient accumulation
            #loss = loss / accumulation_steps

            backward_start = time.time()
            loss.backward()
            #print(f'time to do backward calculation {time.time()-backward_start}s')

            # total_loss += loss.item() * accumulation_steps # Un-scale for logging
            # total_adj += loss_adj
            #total_feat += loss_feat
            total_feat += loss.item()

            #if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer_time = time.time()
            optimizer.step()
            optimizer.zero_grad()
            #print(f'optimizer time {time.time()-optimizer_time}s')
            #print('============')

            # train_time = time.time() - train_start_time
            # print(f'\t\t batch {i+1} required {train_time}s')

        # avg_loss = total_loss / len(train_loader)
        # avg_adj  = total_adj / len(train_loader)
        avg_feat = total_feat / len(train_loader)

        # training losses
        # loss_values.append(avg_loss)
        features_loss.append(avg_feat)
        # recon_loss.append(avg_adj)

        #print(f'epoch {i} finisced in {time.time()-epoch_start}s')
        print(f'finishing the training on epoch {epoch} in {time.time()-epoch_start}s')
        #if epoch==1 or epoch%2==0:
        #auc, ap, avg_feat_err = test(model,test_loader, device)
        avg_feat_err = test_perturb_model(model,test_loader, device)
        # testing metrics
        feat_test_values.append(avg_feat_err)
        # auc_test_values.append(auc)
        # ap_test_values.append(ap)

        # --- Live Plotting ---
        if live_plot:

            epoch_axis = list(range(1, epoch + 1))

            # Clear the previous plot
            clear_output(wait=True) 
            
            #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))       
            fig, ax1 = plt.subplots(1, 1, figsize=(15, 10))       

            # Plot 1: Node features loss
            ax1.plot(epoch_axis, features_loss, label='Train Feature MSE')
            ax1.plot(epoch_axis, feat_test_values, label='Test Feature MSE')
            ax1.set_title('MSE on node features')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('MSE')
            ax1.set_xlim(1, n_epochs)
            ax1.legend(frameon=False)
            ax1.grid(True)
            
            # Plot 2: Test Metrics (AUC/AP)
            # ax2.plot(epoch_axis, auc_test_values, label='Test AUC')
            # ax2.plot(epoch_axis, ap_test_values, label='Test AP')
            # ax2.set_title('Test Metrics (AUC/AP)')
            # ax2.set_xlabel('Epoch')
            # ax2.set_ylabel('Score')
            # ax2.set_ylim(0, 1.05) # AUC/AP are between 0 and 1
            # ax2.set_xlim(1, n_epochs)
            # ax2.legend(frameon=False)
            # ax2.grid(True)

            plt.tight_layout()
            display(fig)   # Use display to show the plot in the notebook
            plt.close(fig) # Close the figure to prevent it from displaying twice
        
        # Print the text output after the plot
        #print(f'test performances at epoch: {epoch:03d} | AUC: {auc:.4f} | AP: {ap:.4f} | MAE (features): {avg_feat_err:.4f}')
        print(f'test performances at epoch: {epoch:03d} | MAE (features): {avg_feat_err:.4f}')

    return feat_test_values#, auc_test_values, ap_test_values

def train_freeze(model, train_loader, test_loader, lr, n_epochs, device, live_plot):
    ''' Routine to train and evaluate the model

    Compared with the function train, this version freezes the encoder and graph decoder weights once they converge, 
    enabling separate fine-tuning of the feature decoder module alone.

    Args:
        model (torch.nn.Model)
        train_loader, test_loader: pytorch geometric dataloaders for training and testing
        lr (float): learning rate
        n_epochs (int): number of epochs
        device: cuda or cpu
        live_plot (Bool): if true, activates live plotting of feature reconstruction loss and graph reconstruction prformances during training runtime
    Returns:
        feat_test_values, auc_test_values, ap_test_values (lists): list of feature MSE, AUC and AP testing performances
    '''
    
    freeze_state=None

    loss_values = []
    features_loss = []
    recon_loss = []

    auc_test_values = []
    ap_test_values = []
    feat_test_values = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #  training/testing Loop 
    for epoch in range(1, n_epochs + 1):

        total_loss, total_adj, total_feat = 0, 0, 0

        for batch in train_loader:
            batch = batch.to(device)
            loss, loss_adj, loss_feat, _ = train_step_freeze(model, batch, optimizer, alpha=1., beta=1., lambda_sparsity=0.000,
                freeze_state=freeze_state, patience = 5, graph_loss_threshold=1.007)
            total_loss += loss
            total_adj += loss_adj
            total_feat += loss_feat

        avg_loss = total_loss / len(train_loader)
        avg_adj  = total_adj / len(train_loader)
        avg_feat = total_feat / len(train_loader)

        # training losses
        loss_values.append(avg_loss)
        features_loss.append(avg_feat)
        recon_loss.append(avg_adj)

        auc, ap, avg_feat_err = test(model,test_loader, device)
        
        # testing metrics
        feat_test_values.append(avg_feat_err)
        auc_test_values.append(auc)
        ap_test_values.append(ap)

        # --- Live Plotting ---
        if live_plot:

            epoch_axis = list(range(1, epoch + 1))

            # Clear the previous plot
            clear_output(wait=True) 
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            # Plot 1: Node features loss
            ax1.plot(epoch_axis, features_loss, label='Train Feature MSE')
            ax1.plot(epoch_axis, feat_test_values, label='Test Feature MSE')
            ax1.set_title('MSE on node features')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('MSE')
            ax1.set_xlim(1, n_epochs)
            ax1.legend(frameon=False)
            ax1.grid(True)
            
            # Plot 2: Test Metrics (AUC/AP)
            ax2.plot(epoch_axis, auc_test_values, label='Test AUC')
            ax2.plot(epoch_axis, ap_test_values, label='Test AP')
            ax2.set_title('Test Metrics (AUC/AP)')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Score')
            ax2.set_ylim(0, 1.05) # AUC/AP are between 0 and 1
            ax2.set_xlim(1, n_epochs)
            ax2.legend(frameon=False)
            ax2.grid(True)
            
            plt.tight_layout()
            display(fig)   # Use display to show the plot in the notebook
            plt.close(fig) # Close the figure to prevent it from displaying twice
        
        # Print the text output after the plot
        print(f'test performances at epoch: {epoch:03d} | AUC: {auc:.4f} | AP: {ap:.4f} | MAE (features): {avg_feat_err:.4f}')

    return feat_test_values, auc_test_values, ap_test_values
import os
import torch
import json
import networkx as nx
from matplotlib import pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from torch_geometric.utils import from_networkx
from torch.utils.data.sampler import Sampler
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
import os
import wandb

from WMSE import compute_weights
from spectra_v2 import PerturbModel
from spectra_v2 import train

wandb.login()

# torch device setting
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.version.cuda)
    print(torch.cuda.get_device_name())
else:
    device = torch.device('cpu')
print(device)

# hyperparameters, metadata and stuff
config = dict(
    #dataset_size=170000,
    test_ratio=0.2,
    val_ratio=0.1,
    batch_size=48,
    n_channels=24,
    edge_dropout_p=0.0,
    lr=0.001,
    n_epochs=20,
    dataset="VCC",
    architecture="ChebConv_no_edge_dropout")


# Initialize wandb run
wandb.init(
    project="spectra-v2",
    config=config
)


####### data and network loading and filtering

adata = sc.read_h5ad('../data/vcc_data/adata_Training.h5ad') #VCC
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.raw = adata.copy() 
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)

# perturbation filtering
counts = adata.obs['target_gene'].value_counts() 
valid_pert = counts[counts >= 100].index # select only perturbations that are present in at least N cells
adata = adata[adata.obs['target_gene'].isin(valid_pert)]
del counts
del valid_pert

# patrick guanlab (stock)
A = np.load('Patrick_networks/guanlab_stock_3.5k_directed.npz')
adjacency = A['adjacency']
gene_names = A['gene_names']
G = nx.from_numpy_array(
    adjacency,
    parallel_edges=False,
    create_using=nx.DiGraph(),
    edge_attr='weight'
)
mapping = {i: gene_names[i] for i in range(len(gene_names))}
G = nx.relabel_nodes(G, mapping)
G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if np.log(d['weight']+1) < 8])
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
num_features = 1
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Number of node features per node:", num_features)

# filter adata genes
gene_list = gene_names.tolist()
mask = adata.var_names.isin(gene_names)
adata = adata[:, mask].copy()

# check if the node names of the networkx graph aorresponds to adata.var_names
assert set(G.nodes) == set(adata.var_names), "Nodes in G and adata.var_names differ!"

# reordering adnr relabeling 
gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
G = nx.relabel_nodes(G, gene_to_idx)
edge_index = from_networkx(G, group_edge_attrs='all').edge_index 


######## dataloaders preparation

# a special batch sampler that groups only cells from the same interventional distribution into a batch
class SCDATA_sampler(Sampler):
    def __init__(self, data, batchsize, ptb_name=None):
        self.intervindices = []
        self.len = 0
        if ptb_name is None:
            ptb_name = data.ptb_names

        for ptb in set(ptb_name):
            idx = np.where(ptb_name == ptb)[0] # indices of cells with the same pert ptb
            self.intervindices.append(idx) # list of indices of cells with the same pert ptb
            self.len += len(idx) // batchsize # number of batches with pert ptb
        self.batchsize = batchsize
    
    def __iter__(self):
        comb = []
        # loop over each intervention
        for i in range(len(self.intervindices)):
            random.shuffle(self.intervindices[i]) # intra-Perturbation Shuffle
            interv_batches = chunk(self.intervindices[i], self.batchsize)
            if interv_batches:
                comb += interv_batches

        combined = [batch.tolist() for batch in comb]
        random.shuffle(combined) # shuffle the order of the batches
        return iter(combined)

    def __len__(self):
        return self.len


def chunk(indices, chunk_size):
    split = torch.split(torch.tensor(indices), chunk_size) # this divides the torch indices into subsets of equal length chunk_size
    
    if len(indices) % chunk_size == 0:
        return split
    elif len(split) > 0:
        return split[:-1]
    else:
        return None


class PerturbationDataset(Dataset):
    def __init__(self, adata, gene_to_idx, start_idx=0, end_idx=None, seed=42):
        super().__init__()

        self.edge_index = edge_index.share_memory_()
        self.gene_to_idx = gene_to_idx
        self.start_idx = start_idx
        self.seed = seed
        self.end_idx = end_idx if end_idx is not None else len(adata)

        adata = adata[self.start_idx:self.end_idx].copy()
        self.adata = adata

        # perturbed samples
        ptb_adata = adata[adata.obs['target_gene']!='non-targeting'].copy()
        ptb_samples = ptb_adata.X.toarray() if hasattr(ptb_adata.X, 'toarray') else np.asarray(ptb_adata.X)
        self.ptb_samples = torch.tensor(ptb_samples, dtype=torch.float) #NOTE: local to this split
        self.ptb_names = ptb_adata.obs['target_gene'].values #NOTE: local to this split

        # control samples
        self.ctrl_samples = adata[adata.obs['target_gene']=='non-targeting'].X.copy()
        self.ctrl_samples = self.ctrl_samples.toarray() if hasattr(self.ctrl_samples, 'toarray') else np.asarray(self.ctrl_samples)
        self.ctrl_samples = torch.tensor(self.ctrl_samples, dtype=torch.float)

    def _pert_embedding(self, pert_name):
        embedding = torch.zeros(self.adata.shape[1], dtype=torch.bool)
        perturbs = [self.gene_to_idx[g] for g in pert_name.split('+') if g in self.gene_to_idx]
        for pert in perturbs:
            embedding[pert]=True
        return embedding

    def __getitem__(self, idx):
        
        # Map to actual index in adata
        actual_idx = idx

        #x = self.rand_ctrl_samples[actual_idx].view(-1,1)
        j = np.random.randint(0, self.ctrl_samples.shape[0]) #random control cell
        x = self.ctrl_samples[j].view(-1,1)
        y = self.ptb_samples[actual_idx].view(-1,1)
        pert = self._pert_embedding(self.ptb_names[actual_idx])

        return x,y,pert

    def __len__(self):
        return self.ptb_samples.shape[0]


#TODO: add sanity check for dataset_size (must be inferior than the number of the perturbed cells, see how the sampling works)

# Setup
dataset_size = (adata.obs['target_gene'] != 'non-targeting').sum() 
test_ratio = config['test_ratio']
val_ratio = config['val_ratio']
test_size = int(dataset_size * test_ratio)
val_size = int(dataset_size * val_ratio)
train_size = dataset_size - test_size - val_size


# Create datasets
train_dataset = PerturbationDataset(
    adata, gene_to_idx,
    start_idx=0, 
    end_idx=train_size
)

val_dataset = PerturbationDataset(
    adata, gene_to_idx,
    start_idx=train_size, 
    end_idx=train_size+val_size
)

test_dataset = PerturbationDataset(
    adata, gene_to_idx,
    start_idx=train_size+val_size, 
    end_idx=dataset_size
)

N_WORKERS = 4 #TODO this will be passed as an argument to training load dataset
batch_size = config['batch_size']

# create loaders
train_loader = DataLoader(train_dataset, batch_sampler=SCDATA_sampler(train_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) #pin memory optimize transfer to CUDA
val_loader = DataLoader(val_dataset, batch_sampler=SCDATA_sampler(val_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) 
test_loader = DataLoader(test_dataset, batch_sampler=SCDATA_sampler(test_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) 

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


######### DEG weights

gene_weights = compute_weights(adata[0:train_size,:], gene_to_idx, cells_per_pert=256)

######### model setup

model = PerturbModel(
    edge_index, 
    num_nodes, 
    device, 
    gene_weights=gene_weights, 
    num_node_features=1, 
    n_channels=config['n_channels'], 
    edge_dropout_p=config['edge_dropout_p']
)
model = model.to(device)
print(model)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'number of parameters: {num_trainable_params}')


########## training
_, _, test_wmse = train(model=model, 
    train_loader=train_loader, 
    test_loader=val_loader,
    lr=config['lr'], 
    n_epochs=config['n_epochs'],  
    device=device,
    wandb_support=True,
    live_plot=False)


######### save and close
torch.save(model.state_dict(), "test_4_mar.pth")
wandb.finish()

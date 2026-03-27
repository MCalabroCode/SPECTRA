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

# hyperparameters, metadata and stuff
config = dict(
    dataset_size=adata.shape[0],
    test_ratio=0.2,
    val_ratio=0.1,
    batch_size=24,
    n_channels=24,
    edge_dropout_p=0.0,
    lr=0.001,
    n_epochs=20,
    dataset="VCC",
    architecture="DirGCNConv(ChebConv)")

# Initialize wandb run
wandb.init(
    project="spectra-v2",       # The name of your project in wandb
    name="DirGNN(Cheb)",   # (Optional) Name of this specific run
    config=config               # Pass your dictionary here!
)

######## dataloaders preparation

from utils import build_model_dataloaders

train_loader, val_loader, test_loader, train_size, test_size, _ = build_model_dataloaders(adata, edge_index, config)

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
    edge_dropout_p = config['edge_dropout_p']
)

model = model.to(device)
print(model)

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
torch.save(model.state_dict(), "test_DirGNN_Cheb.pth")
wandb.finish()

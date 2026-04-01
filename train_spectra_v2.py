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
from spectra_v2_poly import PerturbModel
from spectra_v2_poly import train
from utils import data_preprocessing


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
adata = data_preprocessing(adata, 'target_gene', 'non-targeting')

import pickle

with open("scGPT_embeddings_all_genes.pkl", "rb") as f:
    scgpt_dict = pickle.load(f)

# patrick guanlab (stock)
A = np.load('Patrick_networks/guanlab_stock_3.5k_directed.npz') #guanlab_stock_3.5k_directed.npz
adjacency = A['adjacency']
gene_names = A['gene_names']

# Create weighted DiGraph (parallel_edges=False treats values as weights)
G = nx.from_numpy_array(
    adjacency,
    parallel_edges=False,
    create_using=nx.DiGraph(),
    edge_attr='weight',  # Attribute name for weights
)

# Relabel nodes with gene names (must be same length as matrix dims)
mapping = {i: gene_names[i] for i in range(len(gene_names))}
G = nx.relabel_nodes(G, mapping)

G.remove_nodes_from([n for n in G.nodes if n not in scgpt_dict])
G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if np.log(d['weight']+1) < 10])


num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
num_features = 1
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Number of node features per node:", num_features)

# filter adata
gene_list = list(G.nodes())
mask = adata.var_names.isin(gene_list)
adata = adata[:, mask].copy()

# check if the node names of the networkx graph aorresponds to adata.var_names
assert set(G.nodes) == set(adata.var_names), "Nodes in G and adata.var_names differ!"

# reordering adnr relabeling 
gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
edges = [(gene_to_idx[u], gene_to_idx[v]) for u, v in G.edges()] #this forces edge_index to match the order of adata.var_names
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# hyperparameters, metadata and stuff
config = dict(
    dataset_size=adata.shape[0],#(adata.obs['target_gene'] != 'non-targeting').sum(),
    test_ratio=0.2,
    val_ratio=0.1,
    batch_size=24,
    n_channels=32,
    dropout_p=0.1,
    num_node_features=1,
    lr=0.001,
    n_epochs=20,
    dataset="VCC",
    architecture="Poly-Dir_scGPT_dumb_correct_cellwise")

# map genes to gene names for scGPT
scgpt_dict = {gene_to_idx[k]: v for k, v in scgpt_dict.items() if k in gene_to_idx}

# embedding lookup table
scgpt_dim = len(next(iter(scgpt_dict.values())))
embedding_matrix = torch.zeros((num_nodes, scgpt_dim))
for gene_id, emb in scgpt_dict.items():
    embedding_matrix[gene_id] = torch.tensor(emb, dtype=torch.float32)

# Initialize wandb run
wandb.init(
    project="spectra-v2",       # The name of your project in wandb
    name=f"{config['architecture']}",   # (Optional) Name of this specific run
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
    config = config,
    gene_embeddings = embedding_matrix,
    gene_weights=gene_weights
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
wandb.finish()

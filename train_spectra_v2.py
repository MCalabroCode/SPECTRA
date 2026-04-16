import os
import torch
import json
import networkx as nx
import scanpy as sc
import numpy as np
import pandas as pd
import random
import os
import wandb
import pickle

import warnings
warnings.filterwarnings("ignore")

from WMSE import compute_weights
from spectra_v2_fagcn import PerturbModel
from spectra_v2_fagcn import train
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

#adata = sc.read_h5ad('../data/vcc_data/adata_Training.h5ad') #VCC
adata = sc.read_h5ad('data/K562_gwps_raw_singlecell_01.h5ad') # replogle
adata.var['gene_name'] = adata.var['gene_name'].astype('string')
adata.var_names = adata.var['gene_name']
adata.var_names_make_unique()
adata = data_preprocessing(adata, 'gene', 'non-targeting', logtransform=True, min_cells_per_pert=100)

# filtering for replogle
gene_list = pd.read_csv('/scratch/michele.calabro/gears/VCC/SPECTRA/data/replogle_2022_5k_genes_mapped.csv')
gene_list = gene_list['gene_name']
gene_list = gene_list.tolist()
mask = adata.var_names.isin(gene_list)
adata = adata[:, mask]

with open("scGPT_embeddings_all_genes.pkl", "rb") as f:
    scgpt_dict = pickle.load(f)

# network load
network_data = pd.read_csv('../SCENIC_GRNs/SCENIC_GRNs/temp_results/replogle_adjacencies_vcc.tsv', sep='\t')
G = nx.DiGraph()
for i in range(network_data.shape[0]):
    edge = network_data.iloc[i,:]
    if np.abs(edge['importance'])>0.1: #TODO: dumb pruning, make somehting better please
        G.add_edge(edge['TF'], edge['target'], weight=edge['importance'])

G.remove_nodes_from([n for n in G.nodes if n not in scgpt_dict])
gene_list = list(G.nodes)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
num_features = 1
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Number of node features per node:", num_features)

# filter adata
adata = adata[:, adata.var_names.isin(gene_list)]

# perturbation filtering
perturbations = list(adata.obs['target_gene'].unique())
perturbations.remove('non-targeting')
perts_not_included = [pert for pert in perturbations if not pert in gene_list]
adata = adata[~adata.obs['target_gene'].isin(perts_not_included)].copy()

# adata rows shuffle (otherwise perturbations are all ordered)
adata = adata[np.random.permutation(adata.n_obs), :]

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
    alpha = 4.,
    beta = 0.01,
    dataset="Replogle",
    architecture="FAGCN")

# map genes to gene names for scGPT
scgpt_dict = {gene_to_idx[k]: v for k, v in scgpt_dict.items() if k in gene_to_idx}

# embedding lookup table
scgpt_dim = len(next(iter(scgpt_dict.values())))
embedding_matrix = torch.zeros((num_nodes, scgpt_dim))
for gene_id, emb in scgpt_dict.items():
    embedding_matrix[gene_id] = torch.tensor(emb, dtype=torch.float32)


######## dataloaders preparation

from utils import build_model_dataloaders_split_perturbs

train_loader, val_loader, test_loader, train_size, test_size, _, train_adata, _, test_adata = build_model_dataloaders_split_perturbs(adata, edge_index, config)

######### DEG weights

# load gene weights
with open('gene_weights_replogle.pkl', 'rb') as f:
    gene_weights = pickle.load(f)

######### model setup

wandb.init(
    project="SPECTRA_replogle",       # The name of your project in wandb
    name=config['architecture'],   # (Optional) Name of this specific run
    config=config               # Pass your dictionary here!
)

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
idx_to_gene = {v: k for k, v in gene_to_idx.items()}
_, _, test_wmse = train(model=model, 
    train_loader=train_loader, 
    test_loader=val_loader,
    lr=config['lr'], 
    n_epochs=config['n_epochs'],  
    device=device,
    wandb_support=True,
    var_names = adata.var_names.tolist(),
    idx_to_gene = idx_to_gene,
    alpha_weight=config['alpha'],   # Injected from Sweep
    beta_weight=config['beta'],     # Injected from Sweep
)

wandb.finish()


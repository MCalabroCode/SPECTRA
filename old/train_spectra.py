import os
import torch
import json
import networkx as nx
from matplotlib import pyplot as plt
import scanpy as sc
import numpy as np
import pandas as pd
from torch_geometric.utils import from_networkx
import os
import pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader



########## torch device setting
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.version.cuda)
    print(torch.cuda.get_device_name())
else:
    device = torch.device('cpu')
print(device)


########## data
adata = sc.read_h5ad('../data/vcc_data/adata_Training.h5ad')
sc.pp.filter_cells(adata, min_genes=200)
#sc.pp.filter_genes(adata, min_cells=3)
adata.raw = adata.copy() 
sc.pp.normalize_total(adata, target_sum = 1e4)
sc.pp.log1p(adata)
counts = adata.obs['target_gene'].value_counts()
valid_pert = counts[counts >= 400].index
adata = adata[adata.obs['target_gene'].isin(valid_pert)]
gene_list = adata.var_names.tolist()

############ grn
# with open("grn_coexpression_vcc_data.pkl", "rb") as file:
#     G = pickle.load(file)
# genes_not_included = [gene for gene in gene_list if gene not in list(G.nodes)] # some genes of gene_list are not included!
# G.add_nodes_from(genes_not_included)
# num_nodes = G.number_of_nodes()
# num_edges = G.number_of_edges()
# num_features = 1

# nodes_to_remove = [node for node in G.nodes() if node not in gene_list]
# G.remove_nodes_from(nodes_to_remove)
# num_nodes = G.number_of_nodes()
# num_edges = G.number_of_edges()
# print("Number of nodes:", num_nodes)
# print("Number of edges:", num_edges)
# print("Number of node features per node:", num_features)

# patrick guanlab (stock)
A = np.load('Patrick_networks/guanlab_stock_3.5k_directed.npz')
adjacency = A['adjacency']
gene_names = A['gene_names']
G = nx.from_numpy_array(
    adjacency,
    parallel_edges=False,
    create_using=nx.DiGraph(),
    edge_attr='weight',  # Attribute name for weights
)

# Relabel nodes with gene names (must be same length as matrix dims)
mapping = {i: gene_names[i] for i in range(len(gene_names))}
G = nx.relabel_nodes(G, mapping)
G.remove_edges_from([(u, v) for u, v, d in G.edges(data=True) if np.log(d['weight']+1) < 8]) # naive pruning
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
num_features = 1
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Number of node features per node:", num_features)

# data filtering
gene_list = gene_names.tolist()
mask = adata.var_names.isin(gene_names)
adata = adata[:, mask].copy()

assert set(G.nodes) == set(adata.var_names), "Nodes in G and adata.var_names differ!"

# relabel networkx node indices to match the adata
gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
G = nx.relabel_nodes(G, gene_to_idx)
edge_index = from_networkx(G, group_edge_attrs='all').edge_index # this way, edge_index follows the gene order of adata.var_names


########## dataloaders 
class PerturbationDataset(Dataset):
    def __init__(self, adata, gene_to_idx, start_idx=0, end_idx=None):
        super().__init__()
        
        self.adata = adata
        self.edge_index = edge_index.share_memory_()  # Share to save memory - this is the "base" GRN
        self.gene_to_idx = gene_to_idx
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(adata)
        self.X_data = self.adata.X.toarray() if hasattr(self.adata.X, "toarray") else np.asarray(self.adata.X)
        self.X_data = torch.tensor(self.X_data, dtype=torch.float)
        self.condition = self.adata.obs['target_gene']
        self.perturbations = self.condition[self.start_idx:self.end_idx]
        
        # Precompute perturbation map and edge cache
        self.perturb_map = {} # map each pertrub to gene indices

        for gene_str in self.condition.unique():
            temp = torch.zeros(self.adata.shape[1], dtype=torch.bool)
            if gene_str == 'non-targeting': 
                self.perturb_map[gene_str] = temp
            else:
                perturbs = [gene_to_idx[g] for g in gene_str.split('+') if g in gene_to_idx]
                for pert in perturbs:
                    temp[pert]=True
                self.perturb_map[gene_str] = temp# torch.tensor(perturbs, dtype=torch.long)
    
    # Returns the number of examples in your dataset.
    def __len__(self):
        return self.end_idx - self.start_idx
    
    # logic to load a single graph.
    def __getitem__(self, idx):
        
        # Map to actual index in adata
        actual_idx = self.start_idx + idx
        
        cond = self.condition.iloc[actual_idx]
        features = self.X_data[actual_idx].view(-1, 1)
        pert = self.perturb_map[cond]

        return features, pert

# Setup
dataset_size = adata.X.shape[0] 
test_ratio = 0.20
val_ratio = 0.03
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

# Create loaders - graphs are created on-the-fly!
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=N_WORKERS, pin_memory=True) #pin memory optimize transfer to CUDA
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=N_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")


##### WMSSE weights
from WMSE import compute_weights
gene_weights = compute_weights(adata[0:train_size,:], gene_to_idx, cells_per_pert=256)

###### model def
from spectra import PerturbModel

model = PerturbModel(
    edge_index, 
    num_nodes, 
    device, 
    gene_weights=gene_weights, 
    num_node_features=1, 
    n_channels=24, 
    edge_dropout_p=0.2
)
model = model.to(device)
print(model)

num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'number of parameters: {num_trainable_params}')

######## model training 
#from model_v3_efficient import train
from spectra import train
feat_train_loss, _, feat_test_values = train(model=model, 
    train_loader=train_loader, 
    test_loader=val_loader,
    lr=0.003, 
    n_epochs=20, 
    device=device, 
    live_plot=True)

torch.save(model.state_dict(), "complete_17feb_guanlab.pth")

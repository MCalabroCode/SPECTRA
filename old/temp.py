import os
import torch
import json
import networkx as nx
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
import math

import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import from_networkx
from torch_geometric.utils import negative_sampling


# ====== setting device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
print('======')

# ====== import data
adata = ad.read_h5ad('../data/vcc_data/adata_Training.h5ad')
adata.raw = adata 
sc.pp.log1p(adata)
gene_list = pd.read_csv('../data/vcc_data/gene_names.csv', header=None)
gene_list = gene_list[0].tolist()

# ====== graph construction
def build_spearman_networks(data_matrix: ad.AnnData, corr_threshold: float):

    # array of pvalues for the correlation of each pair of genes
    gene_names = data_matrix.var_names
    corr, p = spearmanr(data_matrix.X.toarray())
    
    # bonferroni correction
    alpha = 0.05/math.comb(data_matrix.shape[1],2)

    # only significant correlated genes
    coexpression_matrix = ((p<alpha)&(np.absolute(corr)>corr_threshold)).astype(int)
    cexpression_graph = pd.DataFrame(coexpression_matrix, index=gene_names, columns=gene_names)
    return cexpression_graph

adata_ctrl = adata[adata.obs['target_gene']=='non-targeting']
coexpression_graph = build_spearman_networks(adata_ctrl, 0.1)

network_data = pd.read_csv('../SCENIC_GRNs/SCENIC_GRNs/temp_results/adjacencies.tsv', sep='\t')
G = nx.DiGraph()
for i in range(network_data.shape[0]):
    edge = network_data.iloc[i,:]
    if edge['TF'] in coexpression_graph.columns and edge['target'] in coexpression_graph.columns and coexpression_graph.loc[edge['TF'], edge['target']] == 1:
        if edge['importance']>1.:
            G.add_edge(edge['TF'], edge['target'], weight=edge['importance'])
            
genes_not_included = [gene for gene in gene_list if gene not in list(G.nodes)] # some genes of gene_list are not included!
G.add_nodes_from(genes_not_included)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
num_features = 1
print("Number of nodes:", num_nodes)
print("Number of edges:", num_edges)
print("Number of node features per node:", num_features)

print('======')

# ====== relabel and set correct order
gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
G = nx.relabel_nodes(G, gene_to_idx)
edge_index = from_networkx(G, group_edge_attrs='all').edge_index

# ====== dataloaders
class PerturbationDataset(Dataset):
    def __init__(self, adata, edge_index, gene_to_idx, 
                    start_idx=0, end_idx=None):
        super().__init__()  
        self.adata = adata
        self.edge_index = edge_index.share_memory_()  # Share to save memory
        self.gene_to_idx = gene_to_idx
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx else len(adata)
        
        # Pre-compute once
        self.X_data = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        self.X_data = torch.tensor(self.X_data, dtype=torch.float)
        self.condition = adata.obs['target_gene']
        
        # Control mean profile
        ctrl = self.X_data[self.condition == 'non-targeting']
        self.ctrl_gex = ctrl.mean(dim=0, keepdim=True).T.share_memory_()
        
        # Precompute perturbation map and edge cache
        self.perturb_map = {}
        self.edge_cache = {}
        
        for gene_str in self.condition.unique():
            if gene_str == 'non-targeting': 
                continue
            perturbs = [gene_to_idx[g] for g in gene_str.split('+') if g in gene_to_idx]
            self.perturb_map[gene_str] = torch.tensor(perturbs, dtype=torch.long)
            
            # Cache filtered edges
            mask = ~torch.isin(edge_index[1], self.perturb_map[gene_str])
            self.edge_cache[gene_str] = edge_index[:, mask].share_memory_()
    
    # Returns the number of examples in your dataset.
    def len(self):
        return self.end_idx - self.start_idx
    
    # Implements the logic to load a single graph.
    def get(self, idx):
        
        # Map to actual index in adata
        actual_idx = self.start_idx + idx
        
        cond = self.condition.iloc[actual_idx]
        features = self.X_data[actual_idx].view(-1, 1)
        
        if cond != 'non-targeting':
            filtered_edge_index = self.edge_cache[cond]
            data = Data(x=self.ctrl_gex, edge_index=filtered_edge_index, y=features)
        else:
            data = Data(x=features, edge_index=self.edge_index, y=features)
        
        # adding negative edges - this will reduce bottleneck in the train, 
        # otherwise recon_loss does the sampling at training time, which is slow
        num_nodes = data.num_nodes
        pos_edge_index = data.edge_index
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1), # 1-to-1 ratio
            method='sparse' # 'sparse' is generally fastest
        )
        # Attach them to the data object
        data.neg_edge_index = neg_edge_index

        return data


# Setup
dataset_size = 221273
test_ratio = 0.30
val_ratio = 0.001
test_size = int(dataset_size * test_ratio)
val_size = int(dataset_size * val_ratio)
train_size = dataset_size - test_size - val_size


# Create datasets
train_dataset = PerturbationDataset(
    adata, edge_index, gene_to_idx, 
    start_idx=0, 
    end_idx=train_size
)

val_dataset = PerturbationDataset(
    adata, edge_index, gene_to_idx,
    start_idx=train_size, 
    end_idx=train_size+val_size
)

test_dataset = PerturbationDataset(
    adata, edge_index, gene_to_idx,
    start_idx=train_size+val_size, 
    end_idx=dataset_size
)

N_WORKERS = 4
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=N_WORKERS, pin_memory=True) 
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=N_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=N_WORKERS, pin_memory=True)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print(f"Test dataset size: {len(val_dataset)}")

print('======')

# ====== model definition
from model import DualDecoderVGAE, train, train_freeze

in_channels, hidden_channels = num_features, 128

model = DualDecoderVGAE(in_channels, hidden_channels)
model = model.to(device)
print(model)


print('======')

# ====== train
feat_test_values, auc_test_values, ap_test_values = train(model=model, 
    train_loader=train_loader, 
    test_loader=val_loader,
    lr=0.002, 
    n_epochs=20, 
    device=device, 
    live_plot=False)

torch.save(model.state_dict(), "complete_20_epochs_dim128_PerturbGraphAE.pth")
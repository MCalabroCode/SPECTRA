
# functions to configure the model and the data
# TODO: must be adapted for SPECTRA

import os
import json
import argparse
import networkx as nx
import anndata as ad
from tqdm import tqdm
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import torch_geometric.transforms as T
# from torch_geometric.loader import DataLoader
# from torch_geometric.utils import from_networkx
# from torch_geometric.data import Data

from torch.utils.data.sampler import Sampler
import random

#TODO: must be adapted for SPECTRA
def load_model_config(config_file_path: str):
    ''' import json configuration file for the model

    Args:
        config_file_path (str): path to the configuration file
    Retruns:
        args: List of arguments and hyperparameters for the model
    '''
    
    # First, check if the configuration file exists.
    if not os.path.exists(config_file_path):
        raise ValueError(f"Error: The configuration file '{config_file_path}' was not found.")
    else:
        with open(config_file_path, 'r') as f:
            config = json.load(f)

    # Convert the dictionary into a namespace object for easy access (e.g., args.epochs)
    args = argparse.Namespace(**config)

    print("Configuration successfully loaded from JSON:")
    print(f"Epochs: {args.epochs}")
    print(f"Number of channels: {args.n_channels}")
    print(f"Learning rate: {args.lr}")
    print(f"Test ratio: {args.test_ratio}")

    return args

#TODO: must be adapted for SPECTRA
def load_graph(network_path: str):
    ''' import networkx graph from json

    Args:
        network_path (str): path to the network json file (edge lists)
    Retruns:
        networkx DiGraph: the loaded graph
    '''

    #with open('../network_hvg.json', 'r') as json_file:
        # First, check if the configuration file exists.
    if not os.path.exists(network_path):
        raise ValueError(f"Error: The network file '{network_path}' was not found.")
    with open(network_path, 'r') as json_file:
        network = json.load(json_file)
        
    # Create the networkx graph
    G = nx.DiGraph()
    for tf, targets in network.items():
        for target, weight in targets:
            G.add_edge(tf, target, weight=weight)
    
    return G 



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
        # If there are some indices left over (the last chunk is smaller than chunk_size),
        # this line discards the last chunk. 
        # It returns all the chunks except the last one.
        return split[:-1]
    else:
        return None



class PerturbationDataset(Dataset):
    def __init__(self, adata, gene_to_idx, edge_index, start_idx=0, end_idx=None, seed=42):
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

def build_model_dataloaders(adata, edge_index, config):
    
    #TODO: add sanity check for dataset_size (must be inferior than the number of the perturbed cells, see how the sampling works)

    # Setup
    dataset_size = (adata.obs['target_gene'] != 'non-targeting').sum() #config['dataset_size']
    test_ratio = config['test_ratio']
    val_ratio = config['val_ratio']
    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - test_size - val_size

    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}

    # Create datasets
    train_dataset = PerturbationDataset(
        adata, gene_to_idx, edge_index,
        start_idx=0, 
        end_idx=train_size
    )

    val_dataset = PerturbationDataset(
        adata, gene_to_idx, edge_index,
        start_idx=train_size, 
        end_idx=train_size+val_size
    )

    test_dataset = PerturbationDataset(
        adata, gene_to_idx, edge_index,
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

    return train_loader, val_loader, test_loader




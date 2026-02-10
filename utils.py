
# functions to configure the model and the data
# TODO: must be adapted for SPECTRA

import os
import json
import argparse
import networkx as nx
import anndata as ad
from tqdm import tqdm

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

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

#TODO: must be adapted for SPECTRA
def build_model_dataloaders(adata: ad.AnnData, grn: nx.DiGraph, training_size: int, batch_size: int, test_ratio: float):
    ''' build training and testing Pytorch Geometric dataloaders

    Args:
        adata (ad.AnnData): input dataset to be divided into training/testing datasets
        grn (nx.DiGraph): grn baseline graph
        training_size (int): how many samples to take from adata to build the training dataset
        batch_size (int): batch size for Pytorch dataloaders
        test_ratio (float): ratio of edges and node feature to take for testing
    Retruns:
        torch_geometric.loader.Dataloader: training dataloader
        torch_geometric.loader.Dataloader: testing dataloader
    '''

    transform = T.Compose([
    #    T.ToDevice(device),
        T.RandomLinkSplit(num_val=0.05, 
            num_test=test_ratio, 
            is_undirected=False,
            split_labels=True, 
            add_negative_train_samples=True
        ),
    ])

    # load node features
    #adata = ad.read_h5ad("../filtered_intestine_organoid_cell_atlas.h5ad")
    node_list = list(grn.nodes)
    adata = adata[:, adata.var_names.isin(node_list)] # filtering only the genes in the network

    graph_node_order = list(grn.nodes()) # ordered list of the nodes in G
    data = from_networkx(grn, group_edge_attrs='all')
    adata_var_map = {name: i for i, name in enumerate(adata.var.index)} # order of the genes in adata

    # match gene order in grn to the adata
    adata_indices = [adata_var_map[gene] for gene in graph_node_order]
    
    train_graphs, test_graphs = [], []
    X_data = adata.X

    for i in tqdm(range(training_size), "preparing the data..."):
        features = X_data[i,:]

        if hasattr(features, "toarray"): # ensure dense array
            features = features.toarray().flatten()
        else:
            features = np.asarray(features).flatten()

        ordered_features = features[adata_indices]
        data.x = torch.tensor(ordered_features, dtype=torch.float).view(-1, 1)#.to(device)

        train_g, _, test_g = transform(data) # if you want, add val_g
        train_graphs.append(train_g)
        test_graphs.append(test_g)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    #val_loader   = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



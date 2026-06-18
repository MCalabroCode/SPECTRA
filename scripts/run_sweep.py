#
# HOW TO RUN
# 1) python scripts/run_sweep.py --config configs/sweep_replogle.yaml
# 2) python scripts/run_sweep.py --config configs/sweep_replogle.yaml --sweep_id abcde123 --count 10
#

import os
import argparse
import yaml
import torch
import networkx as nx
import scanpy as sc
import numpy as np
import pandas as pd
import wandb
import pickle
import warnings

warnings.filterwarnings("ignore")

# Clean package imports
from spectra.utils import set_seed
from spectra.data import data_preprocessing, build_model_dataloaders_perts_split, compute_weights
from spectra.data import build_model_dataloaders_from_perts_list
from spectra.model import SPECTRA
from spectra.training import train

# Global caching variables for the W&B Worker
GLOBAL_ADATA = None
GLOBAL_EDGE_INDEX = None
GLOBAL_EMBEDDINGS = None
GLOBAL_GENE_TO_IDX = None
GLOBAL_NUM_NODES = None
GLOBAL_PERT_TO_IDX = None
GLOBAL_BASE_CONFIG = None

def load_all_data(config):
    """
    Loads data once per worker. Uses the YAML config paths.
    """
    print("Loading datasets and graphs into memory...")
    
    # Load Data
    adata = sc.read_h5ad(config['data']['adata_path'])
    adata = data_preprocessing(adata,
        logtransform=True, 
        min_cells_per_pert=100)

    with open(config['data']['scgpt_embeddings_path'], "rb") as f:
        scgpt_dict = pickle.load(f)

    # # load gene list (optional)
    # print("Loading gene list...")
    # with open(config['data']['gene_list_path'], 'r') as file:
    #     user_genes = set([line.strip() for line in file.readlines()])
    # if user_genes != user_genes & set(adata.var_names):
    #     print('WARNING: some genes in the provided gene list are not included in the dataset; filtering them out...')
    # adata = adata[:, adata.var_names.isin(user_genes)] # first gene filtering
    
    # grn data
    network_data = pd.read_csv(config['data']['network_path'], sep='\t')
    G = nx.DiGraph()
    for _, edge in network_data.iterrows():
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    G.remove_nodes_from([n for n in G.nodes if n not in scgpt_dict])
    grn_genes = set(G.nodes)
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    # Filter adata to match final network genes
    gene_list = grn_genes & set(adata.var_names)
    if grn_genes != gene_list:
        print('WARNING: some genes in the provided gene list are not included in the grn, or the gene embeddings are missing; filtering them out...')
    adata = adata[:, adata.var_names.isin(gene_list)]
    
    # Filter out perturbations that aren't in the gene list
    perturbations = list(adata.obs['target_gene'].unique())
    perturbations.remove('non-targeting')
    perts_not_included = list({
        pert for pert in perturbations
        if any(single_pert not in gene_list for single_pert in pert.split('+'))
    })
    if len(perts_not_included)>0:
        print(print('WARNING: some perturbed genes are not included in the GRN. Filtering these perturbation samples out...'))
    adata = adata[~adata.obs['target_gene'].isin(perts_not_included)].copy()

    # Shuffle
    adata = adata[np.random.permutation(adata.n_obs), :]
    assert set(G.nodes) == set(adata.var_names), "Nodes in G and adata.var_names differ!"

    # Map Edge Index
    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
    edges = [(gene_to_idx[u], gene_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # map gene emebddings 
    scgpt_dict = {gene_to_idx[k]: v for k, v in scgpt_dict.items() if k in gene_to_idx}
    scgpt_dim = len(next(iter(scgpt_dict.values())))
    embedding_matrix = torch.zeros((G.number_of_nodes(), scgpt_dim))
    for gene_id, emb in scgpt_dict.items():
        embedding_matrix[gene_id] = torch.tensor(emb, dtype=torch.float32)
    
    perturbations = list(adata.obs['target_gene'].unique())
    perturbations.remove('non-targeting')
    pert_to_idx = {pert: i for i, pert in enumerate(perturbations)}

    return adata, edge_index, embedding_matrix, gene_to_idx, pert_to_idx, G.number_of_nodes()


def sweep_train():
    """
    The function executed by the W&B Agent for each hyperparameter combination.
    """
    global GLOBAL_ADATA, GLOBAL_EDGE_INDEX, GLOBAL_EMBEDDINGS, GLOBAL_GENE_TO_IDX, GLOBAL_PERT_TO_IDX, GLOBAL_NUM_NODES, GLOBAL_BASE_CONFIG
    
    # inside the loop so every model starts identically, but W&B's Bayesian sampler outside the loop is unaffected.
    set_seed(seed=GLOBAL_BASE_CONFIG['project']['seed'], deterministic=GLOBAL_BASE_CONFIG['project']['deterministic'])

    with wandb.init() as run:
        w_config = wandb.config
        
        # Merge dynamic W&B config with necessary static parameters
        combined_config = dict(w_config)
        combined_config['dataset_size'] = GLOBAL_ADATA.shape[0]
        combined_config['pert_to_idx'] = GLOBAL_PERT_TO_IDX
        combined_config['num_node_features'] = 1 # Static parameter

        # Dataloaders
        train_loader, val_loader, test_loader, _, _, _, _, _, _ = build_model_dataloaders_from_perts_list(GLOBAL_ADATA, combined_config, '/scratch/michele.calabro/gears/VCC/SPECTRA/data/VCC_h1_hESC_split_indices.json')
        # train_loader, val_loader, test_loader, _, _, _, _, _, _ = build_model_dataloaders_perts_split(GLOBAL_ADATA, combined_config)
        
        # WMSE Weights (Using the caching logic!)
        weights_path = GLOBAL_BASE_CONFIG['data']['gene_weights_path']
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                gene_weights = pickle.load(f)
        else:
            gene_weights = compute_weights(GLOBAL_ADATA, GLOBAL_GENE_TO_IDX, cells_per_pert=256)
            with open(weights_path, 'wb') as f:
                pickle.dump(gene_weights, f)
        
        # model init
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SPECTRA(
            edge_index=GLOBAL_EDGE_INDEX, 
            num_nodes=GLOBAL_NUM_NODES, 
            device=device, 
            config=combined_config,
            gene_embeddings=GLOBAL_EMBEDDINGS,
            gene_weights=gene_weights
        ).to(device)
        
        idx_to_gene = {v: k for k, v in GLOBAL_GENE_TO_IDX.items()}
        
        # Train
        train(
            model=model, 
            train_loader=train_loader, 
            test_loader=val_loader,
            lr=combined_config['lr'], 
            n_epochs=combined_config['n_epochs'],  
            device=device,
            wandb_support=True,
            var_names=GLOBAL_ADATA.var_names.tolist(),
            idx_to_gene=idx_to_gene,
            alpha_weight=combined_config['alpha'],
            beta_weight=combined_config['beta'],
            gamma_weight=combined_config['gamma'],
            eta_weight=combined_config['eta']   
        )

if __name__ == '__main__':
    wandb.login()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to YAML sweep config file")
    parser.add_argument('--sweep_id', type=str, default=None, help="If provided, acts as worker. If None, creates sweep.")
    parser.add_argument('--count', type=int, default=1, help="Number of runs for the worker")
    args = parser.parse_args()

    # Load the YAML configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.sweep_id is None:
        # CREATOR MODE
        sweep_id = wandb.sweep(config['sweep'], project=config['project']['name'])
        print("\n" + "="*50)
        print(f"🎉 SWEEP INITIALIZED! Your Sweep ID is: {sweep_id}")
        print("="*50 + "\n")
    else:
        # WORKER MODE
        print(f"Starting Agent for Sweep ID: {args.sweep_id}")
        
        # Load the data ONCE into globals
        GLOBAL_BASE_CONFIG = config
        GLOBAL_ADATA, GLOBAL_EDGE_INDEX, GLOBAL_EMBEDDINGS, GLOBAL_GENE_TO_IDX, GLOBAL_PERT_TO_IDX, GLOBAL_NUM_NODES = load_all_data(config)
        
        wandb.agent(args.sweep_id, project=config['project']['name'], function=sweep_train, count=args.count)
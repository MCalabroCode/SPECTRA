import os
import argparse
import torch
import networkx as nx
import scanpy as sc
import numpy as np
import pandas as pd
import wandb
import pickle

from WMSE import compute_weights
from spectra_v2_fagcn import PerturbModel
from spectra_v2_fagcn import train
from utils import data_preprocessing
from utils import build_model_dataloaders_split_perturbs

# ---------------------------------------------------------
# GLOBAL VARIABLES: This allows us to load data ONCE per job
# and reuse it across all 10 W&B runs in this array task.
# ---------------------------------------------------------
GLOBAL_ADATA = None
GLOBAL_EDGE_INDEX = None
GLOBAL_EMBEDDINGS = None
GLOBAL_GENE_TO_IDX = None
GLOBAL_NUM_NODES = None

######### W&B Sweep Configuration
# Notice we removed 'dataset_size' from here because adata isn't loaded yet.
# We will inject it dynamically later, or you can just hardcode the integer if it never changes.
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val/test_MMD',
        'goal': 'minimize'   
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3,       
        'eta': 3             
    },
    'parameters': {
        'test_ratio': {'value':0.2},
        'val_ratio': {'value':0.1},
        'batch_size': {'values': [16, 24, 32, 48, 64]},
        'n_channels': {'values': [8, 16, 24, 32, 48, 64]},
        'dropout_p': {'min': 0.0, 'max': 0.3},
        'alpha': {'min': 0.1, 'max': 5.0},      
        'beta': {'distribution': 'log_uniform_values', 'min': 0.0001, 'max': 1.0},    
        'lr': {'value':0.001},
        'n_epochs': {'value':20},
        'architecture': {'value':'FAGCN_FiLM'}
    }
}

def load_all_data():
    """Helper function to load data ONLY when an agent actually needs to train."""
    print("Loading datasets and graphs into memory...")
    adata = sc.read_h5ad('../data/vcc_data/adata_Training.h5ad')
    adata = data_preprocessing(adata, 'target_gene', 'non-targeting')
    
    with open("scGPT_embeddings_all_genes.pkl", "rb") as f:
        scgpt_dict = pickle.load(f)

    A = np.load('Patrick_networks/Hierarchist_14_adjacency.npz')
    adjacency = A['adjacency']
    gene_names = A['gene_names']
    G = nx.from_numpy_array(
        adjacency, parallel_edges=False, create_using=nx.DiGraph(), edge_attr='weight'
    )
    mapping = {i: gene_names[i] for i in range(len(gene_names))}
    G = nx.relabel_nodes(G, mapping)
    G.remove_nodes_from([n for n in G.nodes if n not in scgpt_dict])
    
    gene_list = list(G.nodes())
    mask = adata.var_names.isin(gene_list)
    adata = adata[:, mask]
    
    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
    edges = [(gene_to_idx[u], gene_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    scgpt_dict = {gene_to_idx[k]: v for k, v in scgpt_dict.items() if k in gene_to_idx}
    scgpt_dim = len(next(iter(scgpt_dict.values())))
    embedding_matrix = torch.zeros((G.number_of_nodes(), scgpt_dim))
    for gene_id, emb in scgpt_dict.items():
        embedding_matrix[gene_id] = torch.tensor(emb, dtype=torch.float32)

    return adata, edge_index, embedding_matrix, gene_to_idx, G.number_of_nodes()


######### Sweep Training Wrapper
def sweep_train():
    # Pull in the globals so we don't reload data!
    global GLOBAL_ADATA, GLOBAL_EDGE_INDEX, GLOBAL_EMBEDDINGS, GLOBAL_GENE_TO_IDX, GLOBAL_NUM_NODES
    
    # LOCK SEEDS FIRST!
    import random
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    with wandb.init() as run:
        w_config = wandb.config
        wandb.config.update({"dataset_size": GLOBAL_ADATA.shape[0]}, allow_val_change=True)

        # Use the globals!
        train_loader, val_loader, test_loader, train_size, test_size, _, train_adata, _, _ = build_model_dataloaders_split_perturbs(GLOBAL_ADATA, GLOBAL_EDGE_INDEX, w_config)
        gene_weights = compute_weights(train_adata, GLOBAL_GENE_TO_IDX, cells_per_pert=256)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PerturbModel(
            GLOBAL_EDGE_INDEX, 
            GLOBAL_NUM_NODES, 
            device, 
            config=w_config,
            gene_embeddings=GLOBAL_EMBEDDINGS,
            gene_weights=gene_weights
        ).to(device)
        
        idx_to_gene = {v: k for k, v in GLOBAL_GENE_TO_IDX.items()}
        
        _, _, test_wmse = train(
            model=model, 
            train_loader=train_loader, 
            test_loader=val_loader,
            lr=w_config['lr'], 
            n_epochs=w_config['n_epochs'],  
            device=device,
            wandb_support=True,
            var_names=GLOBAL_ADATA.var_names.tolist(),
            idx_to_gene=idx_to_gene,
            alpha_weight=w_config.alpha,   
            beta_weight=w_config.beta,     
        )

######### Launch Logic
if __name__ == '__main__':
    wandb.login()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('--count', type=int, default=1)
    args = parser.parse_args()

    if args.sweep_id is None:
        # CREATOR MODE: Instantly creates sweep ID and exits. ZERO data loaded.
        sweep_id = wandb.sweep(sweep_config, project="SPECTRA-Sweep")
        print("\n" + "="*50)
        print(f"🎉 SWEEP INITIALIZED! Your Sweep ID is: {sweep_id}")
        print("="*50 + "\n")
    else:
        # WORKER MODE
        print(f"Starting Agent for Sweep ID: {args.sweep_id}")
        
        # Load the data ONCE before starting the agent
        GLOBAL_ADATA, GLOBAL_EDGE_INDEX, GLOBAL_EMBEDDINGS, GLOBAL_GENE_TO_IDX, GLOBAL_NUM_NODES = load_all_data()
        
        # This agent will now run 'sweep_train' args.count times, reusing the memory!
        wandb.agent(args.sweep_id, project="SPECTRA-Sweep", function=sweep_train, count=args.count)
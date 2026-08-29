import argparse
import yaml
import pickle
import torch
import networkx as nx
import scanpy as sc
import numpy as np
import pandas as pd
import warnings
import uuid
import os

# Suppress annoying warnings for a clean console
warnings.filterwarnings("ignore")

from spectra.utils import set_seed
from spectra.data import data_preprocessing, build_model_dataloaders_perts_split, compute_weights, compute_weights_enhanced
from spectra.model import SPECTRA
from spectra.training import train
from spectra.data import build_model_dataloaders_from_perts_list

def main(config):

    # Initialization & Seeding - set it to deterministic for reproducibility
    set_seed(seed=config['project']['seed'], deterministic=config['project']['deterministic'])

    # torch device setting
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(torch.version.cuda)
        print(torch.cuda.get_device_name())
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print('=======================')

    # Load Data using paths from config
    print("Loading datasets and embeddings...")
    adata = sc.read_h5ad(config['data']['adata_path'])
    adata = data_preprocessing(adata,
        logtransform=True, 
        min_cells_per_pert=50)

    with open(config['data']['scgpt_embeddings_path'], "rb") as f:
        scgpt_dict = pickle.load(f)

    # # Filter to user genes - TODO: maybe not needed
    # print("Loading gene list...")
    # with open(config['data']['gene_list_path'], 'r') as file:
    #     user_genes = set([line.strip() for line in file.readlines()])
    # if user_genes != user_genes & set(adata.var_names):
    #     print('WARNING: some genes in the provided gene list are not included in the dataset; filtering them out...')
    # adata = adata[:, adata.var_names.isin(user_genes)] # first gene filtering

    # Build Network Graph
    print("Building Gene Regulatory Network...")
    network_data = pd.read_csv(config['data']['network_path'], sep='\t')
    G = nx.DiGraph()
    for _, edge in network_data.iterrows():
        G.add_edge(edge['source'], edge['target'], weight=edge['weight'])
    G.remove_nodes_from([n for n in G.nodes if n not in scgpt_dict])
    grn_genes = set(G.nodes)

    # Filter adata to match final network genes
    gene_list = grn_genes & set(adata.var_names)
    if grn_genes != gene_list:
        print('WARNING: some genes in the provided gene list are not included in the grn, or the gene embeddings are missing; filtering them out...')
    adata = adata[:, adata.var_names.isin(gene_list)]
    G = G.subgraph(gene_list).copy()
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    # Filter out perturbations that aren't in the gene list
    perturbations = list(adata.obs['target_gene'].unique())
    perturbations.remove('non-targeting')
    perts_not_included = list({
        pert for pert in perturbations
        if any(single_pert not in gene_list for single_pert in pert.split('+'))
    })
    if len(perts_not_included)>0:
        print('WARNING: some perturbed genes are not included in the GRN. Filtering these perturbation samples out...')
    adata = adata[~adata.obs['target_gene'].isin(perts_not_included)].copy()

    # sanity check
    assert set(G.nodes) == set(adata.var_names), "Nodes in G and adata.var_names differ!"
    print('=================')

    # Map Encodings & Edge Index
    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}
    edges = [(gene_to_idx[u], gene_to_idx[v]) for u, v in G.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Build embedding matrix
    scgpt_dict = {gene_to_idx[k]: v for k, v in scgpt_dict.items() if k in gene_to_idx}
    scgpt_dim = len(next(iter(scgpt_dict.values())))
    embedding_matrix = torch.zeros((num_nodes, scgpt_dim))
    for gene_id, emb in scgpt_dict.items():
        embedding_matrix[gene_id] = torch.tensor(emb, dtype=torch.float32)

    # Build Dataloaders
    print("Building dataloaders...")
    perturbations = sorted(pert for pert in adata.obs["target_gene"].unique() if pert != "non-targeting")

    # Merge config dictionaries for the model/dataloaders
    model_config = {**config['model'], **config['training']}
    model_config['dataset_size'] = adata.shape[0]
    model_config['pert_to_idx'] = {pert: i for i, pert in enumerate(perturbations)}
    
    train_loader, val_loader, test_loader, _, _, _, _, _, _ = build_model_dataloaders_from_perts_list(adata, model_config, config['data']['split_path'])

    # Load WMSE Weights
    print('Building DEGs weights...')
    weights_path = config['data']['gene_weights_path']
    if os.path.exists(weights_path):
        print('found already existing gene weights dictionary! Loading...')
        with open(weights_path, 'rb') as f:
            gene_weights = pickle.load(f)
    else:
        print('No gene weights dictionary found. Calculating...')
        gene_weights = compute_weights_enhanced(adata, gene_to_idx, cells_per_pert=128)
        with open(weights_path, 'wb') as f:
            pickle.dump(gene_weights, f)

    # Initialize W&B
    wandb_support = config['project']['use_wandb']
    if wandb_support:
        import wandb
        wandb.login()
        wandb.init(project=config['project']['name'], config=model_config)

    # Build Model
    print("Initializing SPECTRA Model...")
    model = SPECTRA(
        edge_index=edge_index, 
        num_nodes=num_nodes, 
        device=device, 
        config=model_config,
        gene_embeddings=embedding_matrix,
        gene_weights=gene_weights
    ).to(device)

    # Train
    print("Starting training loop...")
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}
    train(
        model=model, 
        train_loader=train_loader, 
        test_loader=val_loader,
        device=device,
        wandb_support=wandb_support,
        var_names=adata.var_names.tolist(),
        idx_to_gene=idx_to_gene,
        patience=7,
    )

    weights_dir = config['model']['weights_folder_path']
    if wandb_support and wandb.run is not None:
        final_filepath = os.path.join(weights_dir,f"final_model_{model.architecture_name}_{wandb.run.id}.pth")
    else:
        final_filepath = os.path.join(weights_dir, f"final_model_{model.architecture_name}_{uuid.uuid4().hex}.pth")
    torch.save(model.state_dict(), final_filepath)

    # exit
    if wandb_support:
        wandb.finish()
    print('model trained and ready to go! Enjoy!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SPECTRA Model")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()

    # Load the YAML configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        
    main(config)
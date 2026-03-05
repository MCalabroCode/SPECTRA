#import os
import networkx as nx
import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
import torch
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from matplotlib import pyplot as plt
from scipy import sparse


# TODO: this must become a function of model
def anndata_predict(model, adata: ad.AnnData, G: nx.DiGraph, device='cpu'):

    graph_node_order = list(G.nodes()) # order of the nodes in the graph

    adatas = []
    for cell in tqdm(range(adata.shape[0]), desc='samples prediction inference'):

        features = adata.X[cell,:]
        if hasattr(features, "toarray"): # ensure dense array
            features = features.toarray().flatten()
        else:
            features = np.asarray(features).flatten()

        # Mapping from gene names to feature values for the current cell
        gene_to_value = dict(zip(adata.var.index, features))
        nx.set_node_attributes(G, gene_to_value, name="feature")
        data = from_networkx(G, group_node_attrs=['feature'], group_edge_attrs='all') # Specify the attribute for x
        data.to(device)

        # Run the model
        with torch.no_grad():
            z = model.encode(data.x, data.edge_index).to(device)
        x_ = model.recon_features(z).cpu().detach().numpy().T

        # Create the temporary AnnData object.
        temp = ad.AnnData(x_)
        temp.obs_names = [str(adata.obs_names[cell])]
        temp.var_names = graph_node_order # Use order provided by networkx - NOT the one of real_data.vars

        # 3. Reindex the new AnnData object to match the anndata order to the netrokx/PyG order
        temp = temp[:, adata.var_names].copy()

        # Add this properly-ordered result to the list
        adatas.append(temp)

    # Concatenate all AnnData objects.
    pred_adata = ad.concat(adatas, join="outer")

    # The resulting pred_adata should now have the correct dimensions and gene order
    print("\nReconstruction Complete!")
    return pred_adata


def anndata_predict_optimized(model, adata: ad.AnnData, G: nx.DiGraph, device='cpu'):
    """
    Optimized prediction function that computes graph topology once, outside the main loop.

    Args:
        model: The trained PyTorch Geometric model with encode and recon_features methods.
        adata (ad.AnnData): AnnData object with cells in .obs and genes in .var.
        G (nx.DiGraph): The networkx graph structure.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        ad.AnnData: An AnnData object containing the reconstructed features.
    """

    # Get the fixed node order from the graph. This is our canonical gene order.
    graph_node_order = list(G.nodes())

    # Compute the edge_index ONCE from the graph topology.
    # We create a temporary Data object just for its structure. Node features are ignored.
    graph_data = from_networkx(G, group_edge_attrs='all')
    edge_index = graph_data.edge_index.to(device)

    # Create an index mapping to efficiently reorder AnnData features to match the graph's node order.
    adata_var_map = {name: i for i, name in enumerate(adata.var.index)}
    
    # These are the indices we need to select from adata.X to match the graph_node_order
    adata_indices = [adata_var_map[gene] for gene in graph_node_order]
    
    all_recon_features = []
    X_data = adata.X     # We directly iterate over the data matrix for efficiency
    
    with torch.no_grad():
        for i in tqdm(range(X_data.shape[0]), desc="Sample prediction inference"):

            # a. Get features for the current cell
            cell_features = X_data[i, :]
            if hasattr(cell_features, "toarray"): # Handle sparse matrices
                cell_features = cell_features.toarray().flatten()
            else:
                cell_features = np.asarray(cell_features).flatten()

            # Reorder the cell's features to match the graph's node order using our pre-computed map
            ordered_features = cell_features[adata_indices]

            x = torch.tensor(ordered_features, dtype=torch.float).view(-1, 1).to(device)
            z = model.encode(x, edge_index)
            x_recon = model.recon_features(z).cpu().numpy().flatten()
            
            # e. Store the flattened result
            all_recon_features.append(x_recon)

    pred_matrix = sparse.csr_matrix(np.vstack(all_recon_features))
    pred_adata = ad.AnnData(
        pred_matrix,
        obs=adata.obs.copy(),
        var=pd.DataFrame(index=graph_node_order)
    )

    # Reindex the final object to match the original AnnData's gene order.
    pred_adata = pred_adata[:, adata.var_names].copy()

    print("\nReconstruction Complete!")
    return pred_adata

def plot_violins_redictions_selected_genes(gene_list: list, real_adata: ad.AnnData, pred_adata: ad.AnnData, save: bool, title: str):

    # 1. Extract and convert expression values
    real_vals = real_adata[:, gene_list].X
    pred_vals = pred_adata[:, gene_list].X

    if hasattr(real_vals, "toarray"):
        real_vals = real_vals.toarray()
    if hasattr(pred_vals, "toarray"):
        pred_vals = pred_vals.toarray()

    # 2. Create the figure
    fig, ax = plt.subplots(figsize=(20, 6))

    # Interleave positions
    positions_real = np.arange(len(gene_list)) * 2
    positions_pred = positions_real + 0.8 

    # Generate Violin plots
    # showmeans=True is often helpful for prediction accuracy checks
    vp1 = ax.violinplot(real_vals, positions=positions_real, widths=0.7, showmedians=True)
    vp2 = ax.violinplot(pred_vals, positions=positions_pred, widths=0.7, showmedians=True)

    # Styling function to color the violins
    def style_violin(vp, color):
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.6)
        vp['cbars'].set_edgecolor('black')
        vp['cmins'].set_edgecolor('black')
        vp['cmaxes'].set_edgecolor('black')
        vp['cmedians'].set_edgecolor('black')

    style_violin(vp1, 'orange')
    style_violin(vp2, 'blue')
    
    # 3. Labels and formatting
    ax.set_xticks(positions_real + 0.4)
    ax.set_xticklabels(gene_list, rotation=270)
    ax.set_xlabel('Gene')
    ax.set_ylabel('Expression')
    ax.set_title(title)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='orange', lw=4, label='Real'),
                       Line2D([0], [0], color='blue', lw=4, label='Predicted')]
    ax.legend(handles=legend_elements, frameon=False)

    plt.tight_layout()
    if save:
        print(f'Saving to {title.replace(" ", "_")}.pdf...')
        fig.savefig(f'{title.replace(" ", "_")}.pdf')

def plot_predictions_selected_genes(gene_list: list, real_adata: ad.AnnData, pred_adata: ad.AnnData, save: bool, title: str):

    # Extract expression values for top genes from AnnData objects
    n_genes = len(gene_list)
    real_vals = real_adata[:, gene_list].X  # (cells × selected genes)
    pred_vals = pred_adata[:, gene_list].X  # same shape

    # Convert sparse matrices (if needed)
    if not isinstance(real_vals, np.ndarray):
        real_vals = real_vals.toarray()
    if not isinstance(pred_vals, np.ndarray):
        pred_vals = pred_vals.toarray()

    # 2. Create box plots — one per gene
    fig, ax = plt.subplots(figsize=(20, 6))

    # interleave real/predicted for visual pairing
    positions_real = np.arange(len(gene_list)) * 2
    positions_pred = positions_real + 0.8  # shift slightly for side-by-side boxes

    # Boxplots
    bp1 = ax.boxplot(real_vals, positions=positions_real, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='orange', alpha=0.6), medianprops=dict(color='black'))
    bp2 = ax.boxplot(pred_vals, positions=positions_pred, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='blue', alpha=0.6), medianprops=dict(color='black'))
    
    # 3. Labels and formatting
    # Use gene names from .var_names
    ax.set_xticks(positions_real + 0.4)
    ax.set_xticklabels(gene_list, rotation=270)
    ax.set_xlabel('Gene')
    ax.set_ylabel('Expression')
    ax.set_title(f'{title}')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Real', 'Predicted'], frameon=False)

    plt.tight_layout()
    if save:
        print('saving...')
        fig.savefig(f'{title.replace(" ", "_")}.pdf')
    #plt.show()

def plot_top_highly_expressed_genes(n_genes: int, real_adata: ad.AnnData, pred_adata: ad.AnnData, save: bool):

    real_adata.X_norm = sc.pp.normalize_total(real_adata, target_sum=1, inplace=False)['X']
    real_adata.var['mean_expression'] = np.ravel(real_adata.X_norm.mean(axis=0))
    top_genes = real_adata.var.nlargest(n_genes, 'mean_expression').index.tolist()
    title = f"Distribution of Top {n_genes} Highly Expressed Genes"

    plot_violins_redictions_selected_genes(top_genes, real_adata, pred_adata, save, title=title)

def plot_top_highly_variable_genes(n_genes: int, real_adata: ad.AnnData, pred_adata: ad.AnnData, save: bool):

    sc.pp.highly_variable_genes(real_adata)
    hvg = real_adata.var[real_adata.var['highly_variable']]
    top_hvg = hvg.sort_values('highly_variable').head(n_genes)
    top_genes = top_hvg.index.tolist()
    title = f"Distribution of Top {n_genes} Highly Variable Genes"

    plot_violins_redictions_selected_genes(top_genes, real_adata, pred_adata, save, title=title)
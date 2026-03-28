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


def eval_barplot(results, metric_name):
    model_average = sum(results.values())/len(results)
    plt.figure(figsize=(15, 7))
    bars = plt.bar(list(results.keys()), list(results.values()))
    plt.xticks(rotation=90, ha='right')  # Vertical labels like your plot
    plt.axhline(y=model_average, color='green', linestyle='--', label=f'Model mean DES: {model_average:.2f}')
    plt.title(f'{metric_name}')
    plt.legend(frameon=False)
    plt.tight_layout()


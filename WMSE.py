# taken from "Diversity by Design: Addressing Mode Collapse Improves scRNA-seq 
# Perturbation Modeling on Well-Calibrated Metric"

# TODO: adapt for multiple perturbations!!

import scanpy as sc
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

# Set maximum number of jobs for Scanpy.
sc.settings.njobs = 4

# ref code for WMSE as implemented by Mejia et al.
def wmse(x1, x2, weights):
    weights_arr = np.array(weights)
    x1_arr = np.array(x1)
    x2_arr = np.array(x2)
    normalized_weights = weights_arr / np.sum(weights_arr)
    return np.sum(normalized_weights * ((x1_arr - x2_arr) ** 2))

# degs calculation
def compute_degs(adata, mode='vsrest', pval_threshold=0.05):
    """
    Compute differentially expressed genes (DEGs) for each perturbation.
    
    Args:
        adata: AnnData object with processed data
        mode: 'vsrest' or 'vscontrol'
            - 'vsrest': Compare each perturbation vs all other perturbations (excluding control)
            - 'vscontrol': Compare each perturbation vs control only
        pval_threshold: P-value threshold for significance (default: 0.05)
    
    Returns:
        dict: rank_genes_groups results dictionary
        
    Adds to adata.uns:
        - deg_dict_{mode}: Dictionary with perturbation as key and dict with 'up'/'down' DEGs as values
        - rank_genes_groups_{mode}: Full rank_genes_groups results
    """
    if mode == 'vsrest':
        # Remove control cells for vsrest analysis
        adata_subset = adata[adata.obs['target_gene'] != 'non-targeting'].copy()
        reference = 'rest'
    elif mode == 'vscontrol':
        # Use full dataset for vscontrol analysis
        adata_subset = adata.copy()
        reference = 'non-targeting'
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")
    
    # Compute DEGs
    sc.tl.rank_genes_groups(
        adata_subset, 
        'target_gene', 
        method='t-test_overestim_var', # t-test_overestim_var
        reference=reference, 
        use_raw=False   
    )

    # Extract results
    names_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["names"])
    pvals_adj_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["pvals_adj"])
    logfc_df = pd.DataFrame(adata_subset.uns["rank_genes_groups"]["logfoldchanges"])
    
    # For each perturbation, get the significant DEGs up and down regulated
    deg_dict = {}
    for pert in tqdm(adata_subset.obs['target_gene'].unique(), desc=f"Computing DEGs {mode}"):
        if mode == 'vscontrol' and pert == 'non-targeting':
            continue  # Skip control when comparing vs control
            
        pert_degs = names_df[pert]
        pert_pvals = pvals_adj_df[pert]
        pert_logfc = logfc_df[pert]
        
        # Get significant DEGs
        significant_mask = pert_pvals < pval_threshold
        pert_degs_sig = pert_degs[significant_mask]
        pert_logfc_sig = pert_logfc[significant_mask]
        
        # Split into up and down regulated
        pert_degs_sig_up = pert_degs_sig[pert_logfc_sig > 0].tolist()
        pert_degs_sig_down = pert_degs_sig[pert_logfc_sig < 0].tolist()
        
        deg_dict[pert] = {'up': pert_degs_sig_up, 'down': pert_degs_sig_down}
    
    # Save results to adata.uns
    adata.uns[f'deg_dict_{mode}'] = deg_dict
    adata.uns[f'rank_genes_groups_{mode}'] = adata_subset.uns['rank_genes_groups'].copy()
    
    return adata_subset.uns['rank_genes_groups']

# weights
def compute_weights(adata, gene_to_idx, cells_per_pert=256, score_type = 'scores'):
    '''
    For each perturbation, downsample to the number of cells in DATASET_CELL_COUNTS
    Then calculate the DEGs vs rest
    Save the selected cells inside the dictionary

    Args:
        adata: Anndata file
        gene_to_idx: dictionary that maps each gene to the corresponding index
        cells_per_pert: max number of cells to use per each perturbation, to better balance the dataset
        score_type: 'scores' or 'logfoldchanges'

    Returns:
        a dictionary of weights for each perturbation
    '''

    # sampling cells_per_pert cells for each perturbation (rebalacing the dataset)
    adata_n_cells = []
    unique_perts = adata.obs['target_gene'].unique()
    for pert in unique_perts:
        # if pert == 'non-targeting': 
        #     continue # Skip control for the training weight calculation (usually)

        # cell indices    
        pert_idx = adata.obs_names[adata.obs['target_gene'] == pert]
        
        # If a pert has fewer cells than target, take all of them; otherwise downsample
        n_available = len(pert_idx)
        n_select = min(n_available, cells_per_pert)
        if n_select < n_available:
            selected_cells = np.random.choice(pert_idx, size=n_select, replace=False)
        else:
            selected_cells = pert_idx
        adata_n_cells.append(adata[selected_cells])

    adata_n_cells = sc.concat(adata_n_cells) # balanced dataset (NOTE: does not contain control)  

    # Get DEGs vs rest
    curr_deg_results = compute_degs(adata_n_cells, mode='vsrest') #vscontrol
    names_df_vsrest = pd.DataFrame(curr_deg_results["names"])
    scores_df_vsrest = pd.DataFrame(curr_deg_results[score_type])
        
    final_weight_dict = {}

    for pert in tqdm(scores_df_vsrest.columns, desc="Calculating WMSE Weights"):
        if pert == 'non-targeting': # Typically no scores for control in vsrest, but good to check
            continue

        abs_scores = np.abs(scores_df_vsrest[pert].values) # Ensure it's a numpy array
        min_val = np.min(abs_scores)
        max_val = np.max(abs_scores)
        
        if max_val == min_val:
            if max_val == 0: # All scores are 0
                normalized_weights = np.zeros_like(abs_scores)
            else: # All scores are the same non-zero value
                normalized_weights = np.ones_like(abs_scores) 
        else:
            normalized_weights = (abs_scores - min_val) / (max_val - min_val)
        
        # Ensure no NaNs in weights, replace with 0 if any (e.g. if a gene had NaN score originally)
        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
        
        # Make weighting stronger by squaring the normalized weights
        stronger_normalized_weights = np.square(normalized_weights)

        # normalization to 1
        sum_weights = np.sum(stronger_normalized_weights)
        stronger_normalized_weights = stronger_normalized_weights / sum_weights
        
        weights = pd.Series(stronger_normalized_weights, index=names_df_vsrest[pert].values, name=pert)

        # Order by the var_names
        weights = weights.reindex(adata.var_names, fill_value=0.0)
        final_weight_dict[gene_to_idx[pert]] = weights.values # TODO: adapt for multiple perturbations!! (I think it is enough to just build the dictionary with perturbations as keys, and not the genes!)

    return final_weight_dict


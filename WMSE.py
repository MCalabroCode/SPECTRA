# taken from "Diversity by Design: Addressing Mode Collapse Improves scRNA-seq 
# Perturbation Modeling on Well-Calibrated Metric". TODO: integrate this with main code

# TODO: adapt for multiple perturbations!!

import scanpy as sc
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd

def precompute_wmse_weights(adata, gene_to_idx, pert_col='condition'):
    """
    Calculates weights for every perturbation as described in Section 3.3.3 of the paper.
    
    Args:
        adata: AnnData object containing the training data.
        pert_col: Column name in adata.obs identifying perturbation labels.
        
    Returns:
        weight_dict: Dictionary {perturbation_name: torch.Tensor(weights)}
    """

    weight_dict = {}
    unique_perts = adata.obs[pert_col].unique()
    
    print("Pre-computing WMSE weights (DEGs vs Rest)...")
    
    # Calculate t-scores for all groups vs 'rest'
    sc.tl.rank_genes_groups(
        adata, 
        groupby=pert_col, 
        reference='rest', 
        method='t-test_overestim_var',
        use_raw=False
    )
    
    # Get the list of all genes to ensure order matches the model output
    all_genes = adata.var_names.tolist()
    for pert in unique_perts:

        # Extract t-scores (scores) for this specific perturbation "pert"
        df = sc.get.rank_genes_groups_df(adata, group=pert)
        
        # Create a series to map gene_name -> score
        score_map = df.set_index('names')['scores']
        
        # Reorder scores to match the model's gene order (adata.var_names)
        raw_t_scores = score_map.loc[all_genes].values
        
        # Absolute value of t-scores
        abs_scores = np.abs(raw_t_scores)

        # Min-Max Normalization to [0, 1]
        min_val = np.min(abs_scores)
        max_val = np.max(abs_scores)

        # Avoid division by zero if max == min
        if max_val - min_val == 0:
            norm_scores = abs_scores # Should basically be zeros
        else:
            norm_scores = (abs_scores - min_val) / (max_val - min_val)
            
        # quare the weights
        squared_weights = norm_scores ** 2
        
        # ormalize sum to 1
        final_weights = squared_weights / np.sum(squared_weights)
        
        if pert != 'non-targeting':
            pert_idx = gene_to_idx[pert]
        else:
            pert_idx = -1

        # Convert to tensor for the loss function
        weight_dict[pert_idx] = torch.tensor(final_weights, dtype=torch.float32)

    return weight_dict

def precompute_wmse_weights_mod(adata, gene_to_idx, pert_col='condition', use_control_ref=False, weight_bias=0.1, power=2.0):
    """
    Calculates weights for WMSE.
    
    Args:
        use_control_ref (bool): 
            If True, calculates DEGs vs Control (Standard, captures stress).
            If False, calculates DEGs vs Rest (Paper method, captures specificity).
        weight_bias (float): 
            Base weight added to all genes before normalization. 
            CRITICAL: Prevents common stress genes (which have low scores vs Rest) from having 0 weight.
            A value of 0.05 ensures the model still learns the "shared" biology while prioritizing unique hits.
        power (float): 
            Exponent to sharpen the weights. Higher values focus more on top genes.
            Paper uses 2.0.
    """
    weight_dict = {}
    unique_perts = adata.obs[pert_col].unique()
    
    reference = 'non-targeting' if use_control_ref else 'rest'
    print(f"Pre-computing WMSE weights (DEGs vs {reference})...")
    
    # Calculate t-scores
    sc.tl.rank_genes_groups(
        adata, 
        groupby=pert_col, 
        reference=reference, 
        method='t-test_overestim_var',
        use_raw=False
    )
    
    all_genes = adata.var_names.tolist()
    
    for pert in unique_perts:
        if pert == 'non-targeting':
            # For control, uniform weights are usually best to ensure general reconstruction quality
            final_weights = np.ones(len(all_genes), dtype=np.float32)
            final_weights = final_weights / np.sum(final_weights)
            pert_idx = -1
        else:
            # Extract scores
            df = sc.get.rank_genes_groups_df(adata, group=pert)
            score_map = df.set_index('names')['scores']
            raw_t_scores = score_map.loc[all_genes].values
            
            # Normalization
            abs_scores = np.abs(raw_t_scores)
            min_val = np.min(abs_scores)
            max_val = np.max(abs_scores)
            
            if max_val - min_val == 0:
                norm_scores = abs_scores 
            else:
                norm_scores = (abs_scores - min_val) / (max_val - min_val)
                
            # Apply Power and Bias
            # The bias is the fix for your problem.
            # Even if t-score is 0 (common gene), weight becomes 0.05, not 0.0.
            squared_weights = (norm_scores ** power) + weight_bias
            
            final_weights = squared_weights / np.sum(squared_weights)
            
            pert_idx = gene_to_idx[pert]

        weight_dict[pert_idx] = torch.tensor(final_weights, dtype=torch.float32)

    return weight_dict



def wmse_loss(pred, y, perts, weight_dict, device='cuda'):
    """
    Weighted MSE Loss.
    
    Args:
        pred (torch.tensor): Predicted expression (Batch x Genes)
        y (torch.tensor): True expression (Batch x Genes)
        perts (list/array): List of perturbation labels for this batch
        weight_dict (dict): The dictionary of perturbation weights
        device (str): 'cuda' or 'cpu'
    
    Returns:
        loss (torch.tensor): Scalar loss value
    """
    loss = 0.0
    unique_batch_perts = set(perts)
    
    # each perturbation has different weights
    for p in unique_batch_perts:

        pert_indices = [i for i, x in enumerate(perts) if x == p]
        pred_p = pred[pert_indices]
        y_p = y[pert_indices]
        
        # Shape: (n_genes,) -> unsqueeze to (1, n_genes) for broadcasting
        if p in weight_dict:
            w = weight_dict[p].to(device).unsqueeze(0)
        else:
            # Fallback if perturbation not in dict (e.g. control), use uniform weights
            n_genes = pred_p.shape[1]
            w = torch.ones((1, n_genes)).to(device) / n_genes
            
        # Sum( w * (y - y_hat)^2 )
        # We calculate squared error per gene
        squared_error = (pred_p - y_p) ** 2
        
        # Multiply by weights
        weighted_error = w * squared_error
        
        # Sum over all genes (dim 1) to get WMSE per cell
        wmse_per_cell = torch.sum(weighted_error, dim=1)
        
        # Add to total loss (mean over the cells in this group)
        loss += torch.sum(wmse_per_cell)

    # Return mean loss over the total batch size
    return loss / len(perts)

########################################################## new code ##############################

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
    sc.tl.rank_genes_groups(adata_subset, 
        'target_gene', 
        method='t-test_overestim_var', 
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
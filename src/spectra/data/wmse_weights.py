'''
Wwighted Mean Squared Error logic inspired by: "Diversity by Design: Addressing Mode Collapse 
Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics", Miller et al.
'''

import scanpy as sc
import numpy as np
from tqdm import tqdm
import pandas as pd

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
        adata_subset = adata[adata.obs['target_gene'] != 'non-targeting'].copy()    # Remove control cells for vsrest analysis
        reference = 'rest'
    elif mode == 'vscontrol':
        adata_subset = adata.copy()     # Use full dataset for vscontrol analysis
        reference = 'non-targeting'
    else:
        raise ValueError("mode must be 'vsrest' or 'vscontrol'")
    
    # Compute DEGs
    sc.tl.rank_genes_groups(
        adata_subset, 
        'target_gene', 
        method='wilcoxon',#t-test_overestim_var
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

def compute_weights(adata, cells_per_pert=256, score_type = 'scores', power=2.5):
    '''
    Args:
        adata: Anndata file
        cells_per_pert: max number of cells to use per each perturbation, to better balance the dataset
        score_type: 'scores' or 'logfoldchanges'

    Returns:
        a dictionary of weights for each perturbation
    '''

    # subsampling cells_per_pert cells for each perturbation (rebalacing the dataset)
    adata_n_cells = []
    unique_perts = adata.obs['target_gene'].unique()
    for pert in unique_perts:
        # if pert == 'non-targeting': 
        #     continue 
        
        # If a pert has fewer cells than target, take all of them; otherwise downsample
        pert_idx = adata.obs_names[adata.obs['target_gene'] == pert]    # cells indices 
        n_available = len(pert_idx)
        n_select = min(n_available, cells_per_pert)
        if n_select < n_available:
            selected_cells = np.random.choice(pert_idx, size=n_select, replace=False)
        else:
            selected_cells = pert_idx
        adata_n_cells.append(adata[selected_cells])

    # balanced dataset
    adata_n_cells = sc.concat(adata_n_cells) 

    # Get DEGs vs rest
    curr_deg_results = compute_degs(adata_n_cells, mode='vsrest') #vscontrol
    names_df_vsrest = pd.DataFrame(curr_deg_results["names"])
    scores_df_vsrest = pd.DataFrame(curr_deg_results[score_type])
        
    final_weight_dict = {}

    for pert in tqdm(scores_df_vsrest.columns, desc="Calculating WMSE Weights"):
        if pert == 'non-targeting':
            continue

        abs_scores = np.abs(scores_df_vsrest[pert].values)
        min_val = np.min(abs_scores)
        max_val = np.max(abs_scores)
        
        if max_val == min_val:
            if max_val == 0: 
                normalized_weights = np.zeros_like(abs_scores) # All scores are 0
            else:
                normalized_weights = np.ones_like(abs_scores) # All scores are the same non-zero value
        else:
            normalized_weights = (abs_scores - min_val) / (max_val - min_val)
        
        # Ensure no NaNs in weights, replace with 0 if any
        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0)
        
        # Make weighting stronger by evalting by power the normalized weights
        stronger_normalized_weights = normalized_weights ** power

        # normalization to 1
        sum_weights = np.sum(stronger_normalized_weights)
        stronger_normalized_weights = stronger_normalized_weights / sum_weights
        
        weights = pd.Series(stronger_normalized_weights, index=names_df_vsrest[pert].values, name=pert)

        # Order by the var_names
        weights = weights.reindex(adata.var_names, fill_value=0.0)
        final_weight_dict[pert] = weights.values

    return final_weight_dict


'''
Wwighted Mean Squared Error logic inspired by: "Diversity by Design: Addressing Mode Collapse 
Improves scRNA-seq Perturbation Modeling on Well-Calibrated Metrics", Miller et al.

here, in respect with the original strategy that takes just the test z-scores, we multiply
those for an effect-size gate to better highlight relative effects magnitude
'''

def compute_weights_enhanced(
    adata,
    cells_per_pert=128,
    power=2.0,
    delta_threshold=0.25,
    pval_threshold=0.01
):
    """
    Compute WMSE weights using Scanpy Wilcoxon scores, modulated by
    a soft mean-expression-difference gate.

    For each perturbation and gene:

        importance = abs(Wilcoxon score)   *   min(abs(mean_pert - mean_rest) / delta_threshold, 1)

    Args:
        adata:
            AnnData object. adata.X should contain the same normalized expression representation used by the WMSE loss.

        cells_per_pert:
            Maximum number of cells used per perturbation.

        power:
            Exponent used to strengthen the normalized weights.

        delta_threshold:
            Absolute mean-expression difference at which the gate
            reaches 1. This is expressed in the same units as adata.X.

    Returns:
        Dictionary containing one weight vector per perturbation.
    """

    # Subsample at most cells_per_pert cells for each perturbation, 
    # to create a more balanced dataset
    adata_n_cells = []
    unique_perts = adata.obs['target_gene'].unique()

    for pert in unique_perts:

        pert_idx = adata.obs_names[adata.obs['target_gene'] == pert]
        n_cells_available = len(pert_idx)
        n_selected_cells = min(n_cells_available, cells_per_pert)

        if n_selected_cells < n_cells_available:
            selected_cells = np.random.choice(
                pert_idx,
                size=n_selected_cells,
                replace=False,
            )
        else:
            selected_cells = pert_idx

        adata_n_cells.append(adata[selected_cells])

    adata_n_cells = sc.concat(adata_n_cells)

    # Remove control cells explicitly ( we apply Wilcoxon comparison between perturbation vs all other perturbations.
    adata_vsrest = adata_n_cells[adata_n_cells.obs['target_gene'] != 'non-targeting'].copy()

    # Scanpy Wilcoxon test versus all other perturbations
    curr_deg_results = compute_degs(adata_vsrest, mode='vsrest', pval_threshold=pval_threshold)
    names_df_vsrest = pd.DataFrame(curr_deg_results['names'])
    scores_df_vsrest = pd.DataFrame(curr_deg_results['scores'])

    final_weight_dict = {}

    for pert in tqdm(scores_df_vsrest.columns, desc='Calculating DEGs Weights',):

        # Absolute Scanpy Wilcoxon scores
        abs_scores = np.abs(scores_df_vsrest[pert].values.astype(float))
        abs_scores = np.nan_to_num(abs_scores, nan=0.0, posinf=0.0, neginf=0.0) # nan handling

        # Mean expression for perturbation and rest
        pert_mask = (adata_vsrest.obs['target_gene'] == pert)
        rest_mask = (adata_vsrest.obs['target_gene'] != pert)
        pert_mean = np.asarray(adata_vsrest[pert_mask].X.mean(axis=0)).ravel()
        rest_mean = np.asarray(adata_vsrest[rest_mask].X.mean(axis=0)).ravel()

        # Absolute mean difference in the same space as adata.X
        abs_mean_difference = np.abs(pert_mean - rest_mean)

        # The mean differences are currently ordered as adata.var_names.
        # Scanpy results are ordered by Wilcoxon ranking, so align them.
        mean_difference_series = pd.Series(abs_mean_difference, index=adata_vsrest.var_names,)
        ranked_genes = names_df_vsrest[pert].values
        ranked_mean_difference = (mean_difference_series.reindex(ranked_genes).fillna(0.0).values)

        # Soft mean-difference gate
        mean_difference_gate = np.minimum(ranked_mean_difference / delta_threshold, 1.0)

        # Combine statistical reliability and effect magnitude
        combined_scores = (abs_scores * mean_difference_gate)

        # Original min-max normalization
        min_val = np.min(combined_scores)
        max_val = np.max(combined_scores)
        if max_val == min_val:
            if max_val == 0:
                normalized_weights = np.zeros_like(combined_scores)
            else:
                normalized_weights = np.ones_like(combined_scores)
        else:
            normalized_weights = (combined_scores - min_val) / (max_val - min_val)

        normalized_weights = np.nan_to_num(normalized_weights, nan=0.0, posinf=0.0, neginf=0.0,)

        # Strengthen the normalized weights
        stronger_normalized_weights = (normalized_weights ** power)

        # Normalize weights to sum to one
        sum_weights = np.sum(stronger_normalized_weights)

        if sum_weights > 0:
            stronger_normalized_weights = (stronger_normalized_weights / sum_weights)
        else:
            # Safe fallback for a completely degenerate perturbation
            stronger_normalized_weights = np.ones_like(
                stronger_normalized_weights
            ) / len(stronger_normalized_weights)

        weights = pd.Series(stronger_normalized_weights, index=ranked_genes, name=pert)

        # Restore original adata gene ordering
        weights = weights.reindex(adata.var_names, fill_value=0.0)
        final_weight_dict[pert] = weights.values

    return final_weight_dict
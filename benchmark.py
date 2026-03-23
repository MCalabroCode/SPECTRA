import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# aus function for data merging
def prepare_merged_adata(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene'):
    """
    Merges adata_true and adata_pred into a single AnnData object with an 'Expcategory' 
    column, preserving the specific perturbation labels.
    """
    adata_control = adata_true[adata_true.obs[condition_col] == control_tag].copy()
    adata_control.obs['Expcategory'] = 'control'
    
    adata_stimulated = adata_true[adata_true.obs[condition_col] != control_tag].copy()
    adata_stimulated.obs['Expcategory'] = 'stimulated'
    
    adata_imputed = adata_pred[adata_pred.obs[condition_col] != control_tag].copy()
    adata_imputed.obs['Expcategory'] = 'imputed'
    
    adata_merged = ad.concat([adata_control, adata_stimulated, adata_imputed])
    adata_merged.layers['X'] = adata_merged.X.copy()
    
    return adata_merged

def subsample_groups(adata, n_samples=1000):
    """
    Subsamples each category to a maximum of `n_samples` to save memory and 
    computation time for heavy distribution metrics, exactly as done in the benchmark.
    """
    def sample_subset(sub_adata):
        if sub_adata.n_obs <= n_samples:
            return sub_adata
        else:
            sampled_indices = np.random.choice(sub_adata.n_obs, n_samples, replace=False)
            return sub_adata[sampled_indices, :]
            
    np.random.seed(42) 
    adata_control = sample_subset(adata[adata.obs['Expcategory'] == 'control'])
    adata_stimulated = sample_subset(adata[adata.obs['Expcategory'] == 'stimulated'])
    adata_imputed = sample_subset(adata[adata.obs['Expcategory'] == 'imputed'])
    
    return ad.concat([adata_control, adata_stimulated, adata_imputed])

# Metric Evaluator Wrapper (Iterates over Perturbations)
def evaluate_metric_per_perturbation(adata_true, adata_pred, metric_func, control_tag='non-targeting', condition_col='target_gene', n_top_degs=None, **kwargs):
    """
    General wrapper that runs a given metric function for each individual perturbation.
    """
    adata_merged = prepare_merged_adata(adata_true, adata_pred, control_tag, condition_col)
    
    # Get all unique perturbations (excluding the control tag)
    perturbations = [p for p in adata_merged.obs[condition_col].unique() if p != control_tag]
    
    results = {}
    
    for pert in tqdm(perturbations):

        # Subset to the control cells and the specific current perturbation cells
        adata_sub = adata_merged[adata_merged.obs[condition_col].isin([control_tag, pert])].copy()
        
        # Check if we have both predictions and ground truth for this perturbation
        if 'imputed' not in adata_sub.obs['Expcategory'].values or 'stimulated' not in adata_sub.obs['Expcategory'].values:
            results[pert] = np.nan
            continue

        if n_top_degs is not None:

            # Need categorical dtype for scanpy rank_genes_groups
            adata_sub.obs['Expcategory'] = adata_sub.obs['Expcategory'].astype('category')
            
            # Isolate the true ground truth data (stimulated vs control)
            adata_true_sub = adata_sub[adata_sub.obs['Expcategory'].isin(['stimulated', 'control'])].copy()
            
            # Calculate DEGs using standard t-test as per benchmark
            sc.tl.rank_genes_groups(adata_true_sub, groupby='Expcategory', reference='control', method='wilcoxon')
            
            # Extract top N genes by absolute score
            res_true = adata_true_sub.uns['rank_genes_groups']
            df_true = pd.DataFrame({'names': res_true['names']['stimulated'], 'scores': res_true['scores']['stimulated']})
            top_true_degs = list(df_true.assign(abs_scores=df_true['scores'].abs()).sort_values('abs_scores', ascending=False).head(n_top_degs)['names'])
            
            # Subset the working object to only these top N genes
            adata_sub = adata_sub[:, top_true_degs].copy()
            
        # Calculate metric
        score = metric_func(adata_sub, **kwargs)
        results[pert] = score
        
    return results

# Core Metric Functions
def _core_mse(adata_sub):
    Distance = pt.tools.Distance(metric='mse', layer_key='X')
    pairwise_df = Distance.onesided_distances(adata_sub, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])
    return round(pairwise_df['stimulated'], 4)

def _core_pcc_delta(adata_sub):
    # Calculate the delta (shift from control) just for this specific perturbation subset
    adata_control = adata_sub[adata_sub.obs['Expcategory'] == 'control'].copy()
    adata_imputed = adata_sub[adata_sub.obs['Expcategory'] == 'imputed'].copy()
    adata_stimulated = adata_sub[adata_sub.obs['Expcategory'] == 'stimulated'].copy()
    
    control_mean = adata_control.X.mean(axis=0)
    adata_imputed.X = adata_imputed.X - control_mean
    adata_stimulated.X = adata_stimulated.X - control_mean
    
    adata_delta = ad.concat([adata_control, adata_imputed, adata_stimulated])
    adata_delta.layers['X'] = adata_delta.X.copy()
    
    Distance = pt.tools.Distance(metric='pearson_distance', layer_key='X')
    pairwise_df = Distance.onesided_distances(adata_delta, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])
    return round(pairwise_df['stimulated'], 4)

def _core_edistance(adata_sub, do_subsample=True):
    if do_subsample:
        adata_sub = subsample_groups(adata_sub, n_samples=1000)
    Distance = pt.tools.Distance(metric='edistance', layer_key='X')
    pairwise_df = Distance.onesided_distances(adata_sub, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])
    return round(pairwise_df['stimulated'], 4)

def _core_wasserstein(adata_sub, do_subsample=True):
    if do_subsample:
        adata_sub = subsample_groups(adata_sub, n_samples=1000)
    Distance = pt.tools.Distance(metric='wasserstein', layer_key='X')
    pairwise_df = Distance.onesided_distances(adata_sub, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])
    return round(pairwise_df['stimulated'], 4)

def _core_kldiv(adata_sub, do_subsample=True):
    if do_subsample:
        adata_sub = subsample_groups(adata_sub, n_samples=1000)
    Distance = pt.tools.Distance(metric='sym_kldiv', layer_key='X')
    pairwise_df = Distance.onesided_distances(adata_sub, groupby="Expcategory", selected_group='imputed', groups=["stimulated"])
    return round(np.log2(pairwise_df['stimulated'] + 1), 4)

def _core_common_degs(adata_sub, top_n=100):
    adata_sub.obs['Expcategory'] = adata_sub.obs['Expcategory'].astype('category')
    
    # 1. Identify True DEGs for this specific perturbation
    adata_true_sub = adata_sub[adata_sub.obs['Expcategory'].isin(['stimulated', 'control'])].copy()
    sc.tl.rank_genes_groups(adata_true_sub, groupby='Expcategory', reference='control', method='t-test')
    res_true = adata_true_sub.uns['rank_genes_groups']
    df_true = pd.DataFrame({'names': res_true['names']['stimulated'], 'scores': res_true['scores']['stimulated']})
    top_true_degs = set(df_true.assign(abs_scores=df_true['scores'].abs()).sort_values('abs_scores', ascending=False).head(top_n)['names'])
    
    # 2. Identify Predicted DEGs for this specific perturbation
    adata_pred_sub = adata_sub[adata_sub.obs['Expcategory'].isin(['imputed', 'control'])].copy()
    sc.tl.rank_genes_groups(adata_pred_sub, groupby='Expcategory', reference='control', method='t-test')
    res_pred = adata_pred_sub.uns['rank_genes_groups']
    df_pred = pd.DataFrame({'names': res_pred['names']['imputed'], 'scores': res_pred['scores']['imputed']})
    top_pred_degs = set(df_pred.assign(abs_scores=df_pred['scores'].abs()).sort_values('abs_scores', ascending=False).head(top_n)['names'])
    
    # 3. Calculate Overlap
    if len(top_true_degs) == 0:
        return 0.0
    return round(len(top_true_degs.intersection(top_pred_degs)) / len(top_true_degs), 4)

# End-User Functions (Call these directly!!!!)
def calc_mse(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', n_top_degs=None):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_mse, control_tag, condition_col, n_top_degs=n_top_degs)

def calc_pcc_delta(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', n_top_degs=None):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_pcc_delta, control_tag, condition_col, n_top_degs=n_top_degs)

def calc_edistance(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', n_top_degs=None, do_subsample=True):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_edistance, control_tag, condition_col, n_top_degs=n_top_degs, do_subsample=do_subsample)

def calc_wasserstein(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', n_top_degs=None , do_subsample=True):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_wasserstein, control_tag, condition_col, n_top_degs=n_top_degs, do_subsample=do_subsample)

def calc_kldiv(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', n_top_degs=None, do_subsample=True):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_kldiv, control_tag, condition_col, n_top_degs=n_top_degs, do_subsample=do_subsample)

def calc_common_degs(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_n=100):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_common_degs, control_tag, condition_col, n_top_degs=None, top_n=top_n)
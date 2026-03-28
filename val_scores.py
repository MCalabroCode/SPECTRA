import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt
import anndata as ad
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
from statsmodels.stats.multitest import multipletests
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

sc.settings.verbosity = 0

import warnings
warnings.filterwarnings('ignore')

def calc_mae(real_adata, pred_adata):
    assert set(real_adata.obs['target_gene'].values) == set(pred_adata.obs['target_gene'].values), "perturbations not matching"

    mae = {}
    for pert in real_adata.obs['target_gene'].unique():
        selected_real = real_adata[real_adata.obs['target_gene']==pert].X
        selected_pred = pred_adata[pred_adata.obs['target_gene']==pert].X
        
        # Convert sparse matrices (if needed)
        if not isinstance(selected_real, np.ndarray):
            selected_real = selected_real.toarray()
        if not isinstance(selected_pred, np.ndarray):
            selected_pred = selected_pred.toarray()

        # pseudobulk
        mean_real = np.mean(selected_real, axis=0)
        mean_pred = np.mean(selected_pred, axis=0)

        mae[pert] = np.mean(np.abs(mean_real - mean_pred))

    return mae

def calc_corr(real_adata, pred_adata, correlation='pearson'):
    assert set(real_adata.obs['target_gene'].values) == set(pred_adata.obs['target_gene'].values), "perturbations not matching"

    corr = {}
    for pert in real_adata.obs['target_gene'].unique():
        selected_real = real_adata[real_adata.obs['target_gene']==pert].X
        selected_pred = pred_adata[pred_adata.obs['target_gene']==pert].X
        
        # Convert sparse matrices (if needed)
        if not isinstance(selected_real, np.ndarray):
            selected_real = selected_real.toarray()
        if not isinstance(selected_pred, np.ndarray):
            selected_pred = selected_pred.toarray()
        
        # CORRECTION: pseudobulk - otherise we would assume aligment of cells - absurd
        real_mean = selected_real.mean(axis=0)
        pred_mean = selected_pred.mean(axis=0)
        if correlation == "spearman":
            r = stats.spearmanr(real_mean, pred_mean)[0]
        else:
            r = stats.pearsonr(real_mean, pred_mean)[0]
        
        corr[pert] = r

    return corr


########################################################
##################### DES SCORES #######################
########################################################

class RobustDES:
    def __init__(self, 
                 lfc_threshold=0.5, 
                 alpha=0.05,
                 min_cell_fraction=0.1):
        """
        lfc_threshold: Minimum abs(Log2 Fold Change) to consider a gene DE.
        alpha: Adjusted p-value threshold (FDR).
        min_cell_fraction: Gene must be expressed in at least this fraction of cells in the perturbation group. (Solves sparsity).
        """
        self.lfc_threshold = lfc_threshold
        self.alpha = alpha
        self.min_cell_fraction = min_cell_fraction

    def get_de_genes(self, adata, perturbation_key, perturb_label, control_label):
        """
        Runs scanpy.tl.rank_genes_groups with robust filtering.
        """
        # Subset to specific perturbation vs control
        subset = adata[adata.obs[perturbation_key].isin([perturb_label, control_label])].copy()
        # Run Wilcoxon (Standard)
        sc.tl.rank_genes_groups(
            subset, 
            groupby=perturbation_key, 
            groups=[perturb_label],
            reference=control_label,
            method='wilcoxon', 
            corr_method="benjamini-hochberg",
            use_raw=False
        )
        # Extract results
        result_df = sc.get.rank_genes_groups_df(subset, group=perturb_label) # columns: [scores, logfoldchanges, pvals, pvals_adj], shape [num_genes, 4]
        
        # Calculate fraction of cells expressing the gene in the perturbation group
        perturb_cells = subset[subset.obs[perturbation_key] == perturb_label]
        X = perturb_cells.X
        if hasattr(X, "toarray") or hasattr(X, "tocsr"):
            detected_vals = np.array((X > 0).mean(axis=0)).flatten()
        else:
            detected_vals = np.array((X > 0).mean(axis=0)).flatten()
        
        # rank_genes_groups_df uses gene names as values, not index
        gene_map = dict(zip(subset.var_names, detected_vals))
        result_df['pct_cells'] = result_df['names'].map(gene_map)

        mask_sig = (result_df['pvals_adj'] < self.alpha)
        mask_lfc = (result_df['logfoldchanges'].abs() > self.lfc_threshold)
        mask_sparsity = (result_df['pct_cells'] > self.min_cell_fraction)
        
        # The robust set of DE genes
        de_genes_df = result_df[mask_sig & mask_lfc & mask_sparsity].copy()
        
        return de_genes_df[['names', 'logfoldchanges']].set_index('names'), result_df

    def _core_precision(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):
        """
        Calculates the Modified Differential Expression Score (DES) for perturb_label
        here we don't have a recall but another binary classification score
        """
        # identify True DE Genes (G_true)
        df_true, result_df_true = self.get_de_genes(adata_true, perturbation_key, perturb_label, control_label)
        G_true = set(df_true.index)
        n_true = len(G_true)

        if n_true == 0:
            return 0.0 # Avoid division by zero, or handle as specific case

        # identify Predicted DE Genes (G_pred)
        df_pred, result_df_pred = self.get_de_genes(adata_pred, perturbation_key, perturb_label, control_label)
        G_pred = set(df_pred.index)
        n_pred = len(G_pred)
        # print(f'pert {perturb_label} - n_true = {n_true}, n_pred = {n_pred}')
        # print(perturb_label in G_pred)

        # true positives
        TP = G_pred.intersection(G_true)
        score = len(TP)/(n_pred+1e-8)

        return score

    def _core_f1(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):

        # identify True DE Genes (G_true)
        df_true, result_df_true = self.get_de_genes(adata_true, perturbation_key, perturb_label, control_label)
        G_true = set(df_true.index)
        n_true = len(G_true)

        if n_true == 0:
            return 0.0 # Avoid division by zero, or handle as specific case

        # identify Predicted DE Genes (G_pred)
        df_pred, result_df_pred = self.get_de_genes(adata_pred, perturbation_key, perturb_label, control_label)
        G_pred = set(df_pred.index)
        n_pred = len(G_pred)
        # print(f'pert {perturb_label} - n_true = {n_true}, n_pred = {n_pred}')

        # true positives
        TP = G_pred.intersection(G_true)

        precision = len(TP)/(n_pred + 1e-8)
        recall = len(TP)/(n_true + 1e-8)

        f1_score = 2*(precision * recall)/(precision + recall + 1e-8)
        return f1_score

    def DES_confusion_matrix(self, perts, adata_true, adata_pred, perturbation_key, control_label):
        all_genes = adata_true.var_names
        total_cm = np.zeros((2, 2))
        for pert in tqdm(perts):
            df_true, _ = self.get_de_genes(adata_true, perturbation_key, pert, control_label)
            G_true = set(df_true.index)
            df_pred, _ = self.get_de_genes(adata_pred, perturbation_key, pert, control_label)
            G_pred = set(df_pred.index)

            # Converts a set of DE genes into a binary vector (1=DE, 0=Not DE) aligned to 'all_genes'
            y_true = all_genes.isin(G_true).astype(int)
            y_pred = all_genes.isin(G_pred).astype(int)

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            total_cm += cm

        mean_cm = total_cm / len(perts)
        mean_cm = mean_cm / mean_cm.sum()   # normalize="all"

        disp = ConfusionMatrixDisplay(
            confusion_matrix=mean_cm,
            display_labels=["non DE", "DE"]
        )

        disp.plot(values_format=".3%")
        ax = disp.ax_
        ax.grid(False)
        plt.title("Confusion Matrix")
        #plt.savefig('cm.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        return mean_cm

def calc_f1(real_adata, pred_adata, lfc_threshold=0.3, alpha=0.01):

    des_calculator = RobustDES(lfc_threshold=lfc_threshold, alpha=alpha)

    assert set(real_adata.obs['target_gene'].unique()) == set(pred_adata.obs['target_gene'].unique()), "perturbations do not match"

    perts = [pert for pert in real_adata.obs['target_gene'].unique() if pert != 'non-targeting']
    scores = {}
    for pert in tqdm(perts):
        scores[pert] = des_calculator._core_f1(real_adata, pred_adata, 'target_gene', pert, 'non-targeting')
        # print(f'score: {scores[pert]}')
        # print('---------------------')

    return scores

def calc_precision(real_adata, pred_adata, lfc_threshold=0.3, alpha=0.01):

    des_calculator = RobustDES(lfc_threshold=lfc_threshold, alpha=alpha)

    assert set(real_adata.obs['target_gene'].unique()) == set(pred_adata.obs['target_gene'].unique()), "perturbations do not match"

    perts = [pert for pert in real_adata.obs['target_gene'].unique() if pert != 'non-targeting']
    scores = {}
    for pert in tqdm(perts):
        scores[pert] = des_calculator._core_precision(real_adata, pred_adata, 'target_gene', pert, 'non-targeting')
        # print(f'score: {scores[pert]}')
        # print('---------------------')

    return scores

########################################################
##################### AUPRC SCORE ######################
########################################################

def calc_auprc(adata_true, adata_pred, pert_col='target_gene', control_name='non-targeting', fdr_thresh=0.01, logfc_thresh=0.3):
    """
    Calculates AUPRC for predicted scRNA-seq perturbation responses.
    Assumes adata.X contains log-normalized counts (e.g., log1p).
    """
    common_genes = adata_true.var_names.intersection(adata_pred.var_names)
    adata_true = adata_true[:, common_genes].copy()
    adata_pred = adata_pred[:, common_genes].copy()

    # isolate the in vitro control cells (used for both GT and Pred comparisons)
    control_mask = adata_true.obs[pert_col] == control_name
    X_control_true = adata_true[control_mask].X.toarray() if hasattr(adata_true.X, 'toarray') else adata_true.X
    #mean_control_true = np.mean(X_control_true, axis=0)
    mean_control_true = np.mean(np.expm1(X_control_true), axis=0)    

    perturbations = [p for p in adata_true.obs[pert_col].unique() if p != control_name]
    
    model_results = {}
    baseline_results = {}
    
    for pert in tqdm(perturbations):

        # GT DEGs (In Vitro vs In Vitro)
        mask_pert_true = adata_true.obs[pert_col] == pert
        X_pert_true = adata_true[mask_pert_true].X.toarray() if hasattr(adata_true.X, 'toarray') else adata_true[mask_pert_true].X
        #mean_pert_true = np.mean(X_pert_true, axis=0)
        mean_pert_true = np.mean(np.expm1(X_pert_true), axis=0)

        # Calculate true log2 Fold Change
        # Add a tiny epsilon to avoid log(0) if necessary, though log1p data handles this well.
        epsilon = 1e-9
        true_logfc = np.log2((mean_pert_true + 1e-9) / (mean_control_true + 1e-9))
        
        # Calculate true p-values using Wilcoxon (Mann-Whitney U)
        _, true_pvals = stats.mannwhitneyu(X_pert_true, X_control_true, axis=0, alternative='two-sided')
        
        # Apply FDR Correction
        _, true_pvals_adj, _, _ = multipletests(true_pvals, alpha=fdr_thresh, method='fdr_bh')

        # Define Ground Truth Indicator Z (1 if DE, 0 if not) using paper's thresholds: p < p_thresh and |logFC| > logfc_thresh
        Z_true = ((true_pvals_adj < fdr_thresh) & (np.abs(true_logfc) > logfc_thresh)).astype(int)
        
        # predicted DEGs (In Silico vs In Vitro Control) ---
        mask_pert_pred = adata_pred.obs[pert_col] == pert
        X_pert_pred = adata_pred[mask_pert_pred].X.toarray() if hasattr(adata_pred.X, 'toarray') else adata_pred[mask_pert_pred].X
        #mean_pert_pred = np.mean(X_pert_pred, axis=0)
        mean_pert_pred = np.mean(np.expm1(X_pert_pred), axis=0)

        # Calculate predicted log2 Fold Change
        pred_logfc = np.log2((mean_pert_pred + 1e-9) / (mean_control_true + 1e-9))
        
        # Calculate predicted p-values using Wilcoxon
        _, pred_pvals = stats.mannwhitneyu(X_pert_pred, X_control_true, axis=0, alternative='two-sided')

        # Apply FDR Correction
        _, pred_pvals_adj, _, _ = multipletests(pred_pvals, alpha=fdr_thresh, method='fdr_bh')
        
        # Calculate Ranking Score R_g = |predicted_logFC| * Indicator(predicted_pval < p_thresh)
        indicator_pred = (pred_pvals_adj < fdr_thresh).astype(int)
        R_score = np.abs(pred_logfc) * indicator_pred

        # Handle edge cases where there are no true DEGs for a perturbation
        if np.sum(Z_true) == 0:
            print(f"Skipping {pert}: 0 Ground Truth DEGs found.")
            continue

        model_results[pert] = average_precision_score(Z_true, R_score)
        
        # Calculate Baseline AUPRC (Number of DEGs / Total Genes)
        baseline_results[pert] = np.sum(Z_true) / len(Z_true)
        
        # results.append({
        #     'perturbation': pert,
        #     'num_true_degs': np.sum(Z_true),
        #     'baseline_auprc': baseline_auprc,
        #     'model_auprc': model_auprc
        # })
        
    # # --- D. SUMMARIZE RESULTS ---
    # df_results = pd.DataFrame(results)
    
    print("\n--- Summary ---")
    print(f"Average Baseline AUPRC: {sum(baseline_results.values())/len(baseline_results):.4f}")
    print(f"Average Model AUPRC:    {sum(model_results.values())/len(model_results):.4f}")
    
    return model_results, baseline_results

########################################################
################## BENCHMARK METRICS ###################
########################################################

# from https://github.com/bm2-lab/scPerturBench/tree/main

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._stdout
        sys.stderr = self._stderr

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
    
    for pert in perturbations:

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
        with SuppressOutput():
            score = metric_func(adata_sub, **kwargs)
        results[pert] = score
        
    return results

# Core Metric Functions
def _core_mse(adata_sub):
    with SuppressOutput():
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
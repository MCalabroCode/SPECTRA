import numpy as np
import pandas as pd
import scanpy as sc

def MAE_error(real_adata, pred_adata):
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

    print(mae)
    print('=====')
    avg = np.mean(list(mae.values()))
    print(f'average over all the perturbations: {avg}')

import scipy.stats as stats
def corr_error(real_adata, pred_adata, correlation='pearson'):
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
        # if correlation=='spearman':
        #     corrs = [stats.spearmanr(selected_real[:, i], selected_pred[:, i])[0] for i in range(selected_pred.shape[1])]
        # else:
        #     corrs = [stats.pearsonr(selected_real[:, i], selected_pred[:, i])[0] for i in range(selected_pred.shape[1])]
        
        corr[pert] = r#np.nanmean(corrs)

    print(corr)
    print('=====')
    avg = np.nanmean(list(corr.values()))
    print(f'average over all the perturbations: {avg}')


########################################################
################## CODE FOR DES SCORE ##################
########################################################


import scanpy as sc
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    def calculate_score(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):
        """
        Calculates the Modified Differential Expression Score (DES), VCC
        """
        # identify True DE Genes (G_true)
        df_true, result_df_true = self.get_de_genes(adata_true, perturbation_key, perturb_label, control_label)
        G_true = set(df_true.index)
        n_true = len(G_true)

        if n_true == 0:
            return 0.0 # Avoid division by zero, or handle as specific case

        # identify Predicted DE Genes (G_pred)
        df_pred, result_df_pred = self.get_de_genes(adata_pred, perturbation_key, perturb_label, control_label)
        G_pred_full = set(df_pred.index)
        n_pred = len(G_pred_full)

        # calculate Intersection based on set sizes (VCC logic, TODO: probably must be changed
        if n_pred <= n_true:
            intersection = G_pred_full.intersection(G_true)
            score = len(intersection) / n_true
        else:
            # "Select n_true genes with the largest absolute values of log fold changes"
            df_pred['abs_lfc'] = df_pred['logfoldchanges'].abs()
            top_genes_df = df_pred.sort_values('abs_lfc', ascending=False).head(n_true)
            
            G_pred_top = set(top_genes_df.index)
            
            # Intersection with the top filtered set
            intersection = G_pred_top.intersection(G_true)
            score = len(intersection) / n_true

        return score

    def calculate_precision_score(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):
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
        print(f'pert {perturb_label} - n_true = {n_true}, n_pred = {n_pred}')
        print(perturb_label in G_pred)

        # true positives
        TP = G_pred.intersection(G_true)
        score = len(TP)/(n_pred+1e-8)

        return score

    def calculate_f1_score(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):

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
        print(f'pert {perturb_label} - n_true = {n_true}, n_pred = {n_pred}')

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

    def calculate_AUPRC_score(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):
        """
        AUPRC paper
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
        print(f'pert {perturb_label} - n_true = {n_true}, n_pred = {n_pred}')
        print(perturb_label in G_pred)

        # true positives
        TP = G_pred.intersection(G_true)
        score = len(TP)/(n_pred+1e-8)

        return score

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from statsmodels.stats.multitest import multipletests

def calculate_auprc_score(adata_true, adata_pred, pert_col='target_gene', control_name='non-targeting', fdr_thresh=0.01, logfc_thresh=0.3):
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
    
    results = []
    
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

        # print('ok è questo:')
        # print(np.isnan(indicator_pred).any())
        # print(np.isinf(indicator_pred).any())
        # print('---------')

        # print('ok è questo 2:')
        # print(np.isnan(pred_logfc).any())
        # print(np.isinf(pred_logfc).any())
        # print('---------')
        # calculate AUPRC

        # Handle edge cases where there are no true DEGs for a perturbation
        if np.sum(Z_true) == 0:
            print(f"Skipping {pert}: 0 Ground Truth DEGs found.")
            continue
            
        # precision, recall, _ = precision_recall_curve(Z_true, R_score)
        # model_auprc = auc(recall, precision)
        # print(np.isnan(Z_true).any())
        # print(np.isnan(R_score).any())
        model_auprc = average_precision_score(Z_true, R_score)
        
        # Calculate Baseline AUPRC (Number of DEGs / Total Genes)
        baseline_auprc = np.sum(Z_true) / len(Z_true)
        
        results.append({
            'perturbation': pert,
            'num_true_degs': np.sum(Z_true),
            'baseline_auprc': baseline_auprc,
            'model_auprc': model_auprc
        })
        
    # --- D. SUMMARIZE RESULTS ---
    df_results = pd.DataFrame(results)
    
    print("\n--- Summary ---")
    print(f"Average Baseline AUPRC: {df_results['baseline_auprc'].mean():.4f}")
    print(f"Average Model AUPRC:    {df_results['model_auprc'].mean():.4f}")
    
    return df_results

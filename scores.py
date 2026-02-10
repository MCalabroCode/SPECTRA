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
        
        if correlation=='spearman':
            corrs = [stats.spearmanr(selected_real[:, i], selected_pred[:, i])[0] for i in range(selected_pred.shape[1])]
        else:
            corrs = [stats.pearsonr(selected_real[:, i], selected_pred[:, i])[0] for i in range(selected_pred.shape[1])]

        corr[pert] = np.nanmean(corrs)

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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tqdm import tqdm

class RobustDES:
    def __init__(self, 
                 lfc_threshold=0.5, 
                 alpha=0.05):
        """
        lfc_threshold: Minimum abs(Log2 Fold Change) to consider a gene DE.
        alpha: Adjusted p-value threshold (FDR).
        min_cell_fraction: Gene must be expressed in at least this fraction of cells in the perturbation group. (Solves sparsity).
        """
        self.lfc_threshold = lfc_threshold
        self.alpha = alpha

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
        # Apply "Sparsity Filter" (Heuristic for pseudobulk robustness)
        # Calculate fraction of cells expressing the gene in the perturbation group
        # perturb_cells = subset[subset.obs[perturbation_key] == perturb_label]
        # X = perturb_cells.X
        # if hasattr(X, "toarray") or hasattr(X, "tocsr"):
        #     detected_vals = np.array((X > 0).mean(axis=0)).flatten()
        # else:
        #     detected_vals = np.array((X > 0).mean(axis=0)).flatten()
        
        # rank_genes_groups_df uses gene names as values, not index
        #gene_map = dict(zip(subset.var_names, detected_vals))
        #result_df['pct_cells'] = result_df['names'].map(gene_map)

        mask_sig = (result_df['pvals_adj'] < self.alpha)
        mask_lfc = (result_df['logfoldchanges'].abs() > self.lfc_threshold)
        #mask_sparsity = (result_df['pct_cells'] > self.min_cell_fraction)
        
        # The robust set of DE genes
        de_genes_df = result_df[mask_sig & mask_lfc].copy()
        
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
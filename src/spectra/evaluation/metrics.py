import numpy as np
import pandas as pd
import scanpy as sc
import pertpy as pt
import anndata as ad
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from numba import njit, prange
from scipy import sparse
from scipy.stats import false_discovery_control  # scipy >= 1.11
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from tqdm import tqdm
import numba
import json
import math
import sys
import os

from sklearn.metrics import (
    precision_recall_curve, 
    auc, 
    average_precision_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

sc.settings.verbosity = 0

import warnings
warnings.filterwarnings('ignore')

# Single source of truth for "what is a DEG": used by the 
# DES metrics (f1/precision/jaccard), by AUPRC and by the
# top-DEG selection for the DEG-restricted metrics.
DEG_LFC_THRESHOLD = 0.3
DEG_ALPHA = 0.01 

# DEG-restricted metrics (mae, corr, mse, ...) are NaN for perturbations with fewer DEGs than this
# MIN_DEG_GENES = 3

# A gene is testable only if detected (x > 0) in more than this fraction of the cells of the
# perturbed OR the control group
DEG_MIN_CELL_FRACTION = 0.0

# ok so this just extract the pre-calculated gt DEGs for pert
def _degs_for(top_degs, pert):
    """Gene list of `pert` from a {pert: [genes]} dict (raises if missing)."""
    key = pert if pert in top_degs else str(pert)
    if key not in top_degs:
        raise KeyError(f"top_degs has no entry for perturbation '{pert}'")
    return top_degs[key]

# this just assures gene var columns are the same, and subset to genes is genes is not None
def _aligned_cols(real_adata, pred_adata, genes=None):
    """Column indices (in real, in pred) of the same genes, in the same order.
    genes=None -> all genes (positional if var_names are identical, else the intersection)."""
    if genes is None:
        if real_adata.var_names.equals(pred_adata.var_names):
            return slice(None), slice(None)
        genes = real_adata.var_names.intersection(pred_adata.var_names)
    else:
        genes = [g for g in genes if g in real_adata.var_names and g in pred_adata.var_names]
    return real_adata.var_names.get_indexer(genes), pred_adata.var_names.get_indexer(genes)

# pseudobulk in log space
def _pseudobulk(adata, rows, cols):
    """Mean expression profile of cells `rows`, restricted to gene columns `cols`."""
    X = adata.X[rows][:, cols]
    return np.asarray(X.mean(axis=0, dtype=np.float64)).ravel()

# Returns (pert, real_mean, pred_mean) for every non-control perturbation (eventually restricted to top_degs).
def _per_pert_profiles(real_adata, pred_adata, top_degs, control_tag, condition_col):
    """
    Returns (pert, real_mean, pred_mean) for every non-control perturbation.
    If top_degs ({pert: [genes]}) is given, profiles are restricted to that 
    perturbation's DEGs.
    """
    assert set(real_adata.obs[condition_col].values) == set(pred_adata.obs[condition_col].values), "perturbations not matching"
    real_idx = real_adata.obs.groupby(condition_col, observed=True).indices
    pred_idx = pred_adata.obs.groupby(condition_col, observed=True).indices
    for pert in real_idx:
        if pert == control_tag:
            continue
        genes = None if top_degs is None else _degs_for(top_degs, pert)
        rc, pc = _aligned_cols(real_adata, pred_adata, genes)
        # if genes is not None and len(rc) < MIN_DEG_GENES:  # too few DEGs for a meaningful score
        #     yield pert, None, None
        #     continue
        yield pert, _pseudobulk(real_adata, real_idx[pert], rc), _pseudobulk(pred_adata, pred_idx[pert], pc)


def calc_mae(real_adata, pred_adata, top_degs=None, control_tag='non-targeting', condition_col='target_gene'):
    """MAE between real and predicted pseudobulk profiles, per perturbation.
    top_degs: optional {pert: [genes]} -> compute on those genes only."""
    return {pert: (np.nan if r is None else float(np.mean(np.abs(r - p))))
            for pert, r, p in _per_pert_profiles(real_adata, pred_adata, top_degs, control_tag, condition_col)}


def calc_corr(real_adata, pred_adata, correlation='pearson', top_degs=None, control_tag='non-targeting', condition_col='target_gene'):
    """Correlation between real and predicted pseudobulk profiles (pseudobulk: otherwise we would
    assume alignment of cells - absurd). top_degs: optional {pert: [genes]} -> those genes only."""
    corr_fn = stats.spearmanr if correlation == "spearman" else stats.pearsonr
    return {pert: (np.nan if r is None else float(corr_fn(r, p)[0]))
            for pert, r, p in _per_pert_profiles(real_adata, pred_adata, top_degs, control_tag, condition_col)}


########################################################
##################### DES SCORES #######################
########################################################

class RobustDES:
    def __init__(self,
                 lfc_threshold=DEG_LFC_THRESHOLD,
                 alpha=DEG_ALPHA,
                 min_cell_fraction=DEG_MIN_CELL_FRACTION):
        """
        DEG calling shared with AUPRC: same Wilcoxon kernel (tie + continuity corrected), same logFC,
        same gene-eligibility filter.

        lfc_threshold: Minimum abs(Log2 Fold Change) to consider a gene DE.
            LFC = log2 ratio of the arithmetic means in linear space, mean(expm1(x)).
        alpha: Adjusted p-value threshold (BH-FDR over all genes).
        min_cell_fraction: a gene is eligible only if it is detected (x > 0) in more than this fraction of
            the cells of EITHER group (perturbed or control), as Seurat's min.pct. Requiring it in the
            perturbed group only would drop strongly down-regulated genes (e.g. the knocked-out target).
            None / 0 disables the filter.
        """
        self.lfc_threshold = lfc_threshold
        self.alpha = alpha
        self.min_cell_fraction = min_cell_fraction
        self._ctx_cache = {}

    def _get_context(self, adata, perturbation_key, control_label):
        """
        Row indices per group + control context, built once per AnnData object.
        """
        key = (id(adata), perturbation_key, control_label)
        hit = self._ctx_cache.get(key)
        if hit is None or hit[0] is not adata:
            groups = adata.obs.groupby(perturbation_key, observed=True).indices
            if control_label not in groups:
                raise ValueError(f"Control '{control_label}' not found in adata.")
            ctrl = _ControlContext(_dense(adata.X[groups[control_label]]))
            hit = (adata, groups, ctrl)
            self._ctx_cache[key] = hit
        return hit[1], hit[2]

    def get_de_genes(self, adata, perturbation_key, perturb_label, control_label, eligible=None):
        """
        Perturbation vs control DE test (fast Wilcoxon) with the shared DEG definition.
        eligible: optional pd.Series (index = gene name, bool). Pass the ground-truth eligibility when
            calling on predictions, so that the filter is decided on real data only. If None, it is
            computed from `adata` itself.
        Returns (DEG table [names -> logfoldchanges], full per-gene table).
        """
        groups, ctrl = self._get_context(adata, perturbation_key, control_label)
        st = _de_stats(_dense(adata.X[groups[perturb_label]]), ctrl)
        result_df = pd.DataFrame({"names": adata.var_names.values, **st})
        if eligible is None:
            elig = _eligible(st["pct_cells"], ctrl.det, self.min_cell_fraction)
        else:
            elig = result_df["names"].map(eligible).fillna(False).values.astype(bool)
        result_df["eligible"] = elig

        mask = _deg_mask(st, elig, self.lfc_threshold, self.alpha)
        return result_df.loc[mask, ["names", "logfoldchanges"]].set_index("names"), result_df

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
        df_pred, result_df_pred = self.get_de_genes(adata_pred, perturbation_key, perturb_label, control_label,
                                                    eligible=result_df_true.set_index('names')['eligible'])
        G_pred = set(df_pred.index)
        n_pred = len(G_pred)

        # true positives
        TP = G_pred.intersection(G_true)
        score = len(TP)/(n_pred+1e-8)

        return score
    
    def _core_jaccard(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):
        """
        Jaccard index |G_true & G_pred| / |G_true | G_pred| between the true and predicted DE gene sets.
        """
        # identify True DE Genes (G_true)
        df_true, result_df_true = self.get_de_genes(adata_true, perturbation_key, perturb_label, control_label)
        G_true = set(df_true.index)
        n_true = len(G_true)

        if n_true == 0:
            return 0.0 # Avoid division by zero, or handle as specific case

        # identify Predicted DE Genes (G_pred)
        df_pred, result_df_pred = self.get_de_genes(adata_pred, perturbation_key, perturb_label, control_label,
                                                    eligible=result_df_true.set_index('names')['eligible'])
        G_pred = set(df_pred.index)
        n_pred = len(G_pred)

        # jaccard index
        intersection = G_pred.intersection(G_true)
        union = G_pred.union(G_true)
        score = len(intersection) / len(union)

        return score

    def _core_f1(self, adata_true, adata_pred, perturbation_key, perturb_label, control_label):

        # identify True DE Genes (G_true)
        df_true, result_df_true = self.get_de_genes(adata_true, perturbation_key, perturb_label, control_label)
        G_true = set(df_true.index)
        n_true = len(G_true)

        if n_true == 0:
            return 0.0 # Avoid division by zero, or handle as specific case

        # identify Predicted DE Genes (G_pred)
        df_pred, result_df_pred = self.get_de_genes(adata_pred, perturbation_key, perturb_label, control_label,
                                                    eligible=result_df_true.set_index('names')['eligible'])
        G_pred = set(df_pred.index)
        n_pred = len(G_pred)

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
            df_true, res_true = self.get_de_genes(adata_true, perturbation_key, pert, control_label)
            G_true = set(df_true.index)
            df_pred, _ = self.get_de_genes(adata_pred, perturbation_key, pert, control_label,
                                           eligible=res_true.set_index('names')['eligible'])
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

def calc_f1(real_adata, pred_adata, lfc_threshold=DEG_LFC_THRESHOLD, alpha=DEG_ALPHA,
            min_cell_fraction=DEG_MIN_CELL_FRACTION):

    des_calculator = RobustDES(lfc_threshold=lfc_threshold, alpha=alpha, min_cell_fraction=min_cell_fraction)

    assert set(real_adata.obs['target_gene'].unique()) == set(pred_adata.obs['target_gene'].unique()), "perturbations do not match"

    perts = [pert for pert in real_adata.obs['target_gene'].unique() if pert != 'non-targeting']
    scores = {}
    for pert in tqdm(perts):
        scores[pert] = des_calculator._core_f1(real_adata, pred_adata, 'target_gene', pert, 'non-targeting')

    return scores

def calc_jaccard(real_adata, pred_adata, lfc_threshold=DEG_LFC_THRESHOLD, alpha=DEG_ALPHA,
                 min_cell_fraction=DEG_MIN_CELL_FRACTION):

    des_calculator = RobustDES(lfc_threshold=lfc_threshold, alpha=alpha, min_cell_fraction=min_cell_fraction)

    assert set(real_adata.obs['target_gene'].unique()) == set(pred_adata.obs['target_gene'].unique()), "perturbations do not match"

    perts = [pert for pert in real_adata.obs['target_gene'].unique() if pert != 'non-targeting']
    scores = {}
    for pert in tqdm(perts):
        scores[pert] = des_calculator._core_jaccard(real_adata, pred_adata, 'target_gene', pert, 'non-targeting')

    return scores

def calc_precision(real_adata, pred_adata, lfc_threshold=DEG_LFC_THRESHOLD, alpha=DEG_ALPHA,
                   min_cell_fraction=DEG_MIN_CELL_FRACTION):

    des_calculator = RobustDES(lfc_threshold=lfc_threshold, alpha=alpha, min_cell_fraction=min_cell_fraction)

    assert set(real_adata.obs['target_gene'].unique()) == set(pred_adata.obs['target_gene'].unique()), "perturbations do not match"

    perts = [pert for pert in real_adata.obs['target_gene'].unique() if pert != 'non-targeting']
    scores = {}
    for pert in tqdm(perts):
        scores[pert] = des_calculator._core_precision(real_adata, pred_adata, 'target_gene', pert, 'non-targeting')

    return scores

########################################################
##################### AUPRC SCORE ######################
########################################################

"""
Fast DEG-recovery AUPRC (Zhu et al. 2025) with a Wilcoxon rank-sum test.

Main speed-up: the control cells are sorted ONCE, and each perturbation's
Mann-Whitney U is computed by binary search against the sorted control
(numba, multi-threaded). Same asymptotic, tie- and continuity-corrected
p-values as scipy.stats.mannwhitneyu(..., method="asymptotic"), without
re-ranking thousands of control cells for every perturbation.
"""

# ----------------------------------------------------------------------------
# Wilcoxon kernel
# ----------------------------------------------------------------------------
@njit(parallel=True, cache=True)
def _ctrl_tie_term(ctrl_sorted):
    G, n2 = ctrl_sorted.shape
    out = np.zeros(G)
    for g in prange(G):
        c = ctrl_sorted[g]
        i = 0
        s = 0.0
        while i < n2:
            j = i + 1
            while j < n2 and c[j] == c[i]:
                j += 1
            t = float(j - i)
            s += t * t * t - t
            i = j
        out[g] = s
    return out


def prepare_control(X_ctrl):
    """
    Sort control once. X_ctrl: dense (n_ctrl, G) -> ((G, n_ctrl), (G,)).
    """
    ctrl_sorted = np.ascontiguousarray(np.sort(X_ctrl, axis=0).T)
    return ctrl_sorted, _ctrl_tie_term(ctrl_sorted)


@njit(parallel=True, cache=True)
def mwu_vs_sorted_control(X_pert, ctrl_sorted, ctrl_tie):
    """Two-sided Mann-Whitney U per gene (asymptotic, tie + continuity corrected).
    X_pert: dense (n1, G). Returns (pvals, z): z is signed (> 0: higher in X_pert) and is
    used to rank DEGs (it does not underflow like the p-value)."""
    n1, G = X_pert.shape
    n2 = ctrl_sorted.shape[1]
    n = n1 + n2
    pvals = np.ones(G)
    zs = np.zeros(G)
    for g in prange(G):
        col = np.sort(X_pert[:, g])
        c = ctrl_sorted[g]
        U = 0.0
        tie = ctrl_tie[g]
        i = 0
        while i < n1:
            v = col[i]
            j = i + 1
            while j < n1 and col[j] == v:
                j += 1
            p = float(j - i)
            lo = np.searchsorted(c, v, side="left")
            hi = np.searchsorted(c, v, side="right")
            cv = float(hi - lo)
            U += p * (lo + 0.5 * cv)
            t = cv + p
            tie += (t * t * t - t) - (cv * cv * cv - cv)
            i = j
        mu = n1 * n2 / 2.0
        U1 = U
        U = max(U, n1 * n2 - U)
        var = n1 * n2 / 12.0 * ((n + 1.0) - tie / (n * (n - 1.0)))
        if var <= 0.0:
            pvals[g] = 1.0
        else:
            z = max((U - mu - 0.5) / math.sqrt(var), 0.0)
            pvals[g] = min(math.erfc(z / math.sqrt(2.0)), 1.0)
            zs[g] = z if U1 >= mu else -z
    return pvals, zs


# ----------------------------------------------------------------------------
# AUPRC
# ----------------------------------------------------------------------------

# this is just to densify the count matrix
def _dense(X):
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    return np.ascontiguousarray(X, dtype=np.float32)


def _mean_expm1(X):
    """
    Per-gene arithmetic mean in linear space of log1p data (dense or sparse).
    """
    if sparse.issparse(X):
        return np.asarray(X.expm1().mean(axis=0, dtype=np.float64)).ravel()
    return np.expm1(X).mean(axis=0, dtype=np.float64)


def _logfc(X_pert, mean_ctrl, eps=1e-9):
    """log2 fold change of mean(expm1(x)) between perturbed cells and a precomputed control mean.
    Single logFC definition shared by AUPRC and DES (F1 / precision)."""
    return np.log2((_mean_expm1(X_pert) + eps) / (mean_ctrl + eps))


class _ControlContext:
    """
    Everything about the control cells that does not depend on the perturbation (computed once).

    self.mean = pseudobulk in linear space
    self.det = pseudobulk in log space
    self.sorted, self.tie = sorting for wilcoxon (to be done only once)
    """
    def __init__(self, X_ctrl):  # dense float32 (n_ctrl, G)
        self.mean = _mean_expm1(X_ctrl)
        self.det = (X_ctrl > 0).mean(axis=0)
        self.sorted, self.tie = prepare_control(X_ctrl)


def _de_stats(X_pert, ctrl, eps=1e-9):
    """
    One perturbation vs. control (Wilcoxon + BH-FDR + logFC). X_pert: dense float32 (n1, G).
    """
    pvals, z = mwu_vs_sorted_control(X_pert, ctrl.sorted, ctrl.tie)
    return {
        "scores": z,
        "pvals": pvals,
        "pvals_adj": false_discovery_control(pvals, method="bh"),
        "logfoldchanges": _logfc(X_pert, ctrl.mean, eps),
        "pct_cells": (X_pert > 0).mean(axis=0),
    }


def _eligible(pct_pert, pct_ctrl, min_cell_fraction):
    """
    Genes detected in more than `min_cell_fraction` of the perturbed OR the control cells.
    """
    if not min_cell_fraction:
        return np.ones(len(pct_pert), dtype=bool)
    return np.maximum(pct_pert, pct_ctrl) > min_cell_fraction


def _deg_mask(st, eligible, lfc_threshold, alpha):
    """
    THE definition of a DEG, shared by the DES metrics, AUPRC and the top-DEG selection.
    """
    return ((np.asarray(st["pvals_adj"]) < alpha)
            & (np.abs(np.asarray(st["logfoldchanges"])) > lfc_threshold)
            & np.asarray(eligible, dtype=bool))


def calc_auprc(
    adata_true,
    adata_pred,
    pert_col="target_gene",
    control_name="non-targeting",
    fdr_thresh=DEG_ALPHA,
    logfc_thresh=DEG_LFC_THRESHOLD,
    min_cell_fraction=DEG_MIN_CELL_FRACTION,
    n_jobs=None,
    eps=1e-9,
    true_labels=None,
    return_labels=False,
):
    """
    DEG-recovery AUPRC (average precision) per perturbation. Assumes log1p X.

    DEGs (ground truth) are defined exactly as in the DES metrics (same kernel, logFC and filter).
    Gene eligibility (detected in > min_cell_fraction of the perturbed or control cells) is decided on
    the REAL data only and applied to both truth and predictions: ineligible genes are removed from
    the evaluation (labels, scores and baseline), so the filter does not depend on how the model
    parametrises its output (continuous predictions would otherwise always be "detected").

    n_jobs        number of numba threads (default: $SLURM_CPUS_PER_TASK, else all cores).
    true_labels   dict filled in place with {pert: (DEG labels, eligible mask)} over the common
                  genes. Ground truth does not depend on the model: pass the same dict for
                  every model to compute it once.
    """
    if n_jobs is None:
        n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    numba.set_num_threads(max(1, min(n_jobs, numba.config.NUMBA_NUM_THREADS)))

    common = adata_true.var_names.intersection(adata_pred.var_names)
    ti = adata_true.var_names.get_indexer(common)
    pi = adata_pred.var_names.get_indexer(common)
    true_groups = adata_true.obs.groupby(pert_col, observed=True).indices
    pred_groups = adata_pred.obs.groupby(pert_col, observed=True).indices
    if control_name not in true_groups:
        raise ValueError(f"Control '{control_name}' not found in adata_true.")

    X_true, X_pred = adata_true.X, adata_pred.X

    # Observed control (predictions are also compared against it): prepared ONCE
    ctrl = _ControlContext(_dense(X_true[true_groups[control_name]][:, ti]))

    perts = [p for p in true_groups if p != control_name]

    # Ground truth (model-independent -> cacheable)
    if true_labels is None:
        true_labels = {}
    todo = [p for p in perts if p not in true_labels]
    for pert in tqdm(todo, desc="Ground-truth DEGs", disable=not todo):
        st = _de_stats(_dense(X_true[true_groups[pert]][:, ti]), ctrl, eps)
        elig = _eligible(st["pct_cells"], ctrl.det, min_cell_fraction)
        true_labels[pert] = (_deg_mask(st, elig, logfc_thresh, fdr_thresh).astype(np.uint8), elig)

    # Predictions
    model_results, baseline_results = {}, {}
    for pert in tqdm(perts, desc="Calculating AUPRC"):
        Z, elig = true_labels[pert]
        if len(Z) != len(common):
            raise ValueError("cached true_labels were computed on a different gene set")
        n_degs = int(Z.sum())  # Z already implies eligible
        if n_degs == 0 or pert not in pred_groups:
            continue
        st = _de_stats(_dense(X_pred[pred_groups[pert]][:, pi]), ctrl, eps)
        R = np.abs(st["logfoldchanges"]) * (st["pvals_adj"] < fdr_thresh)
        model_results[pert] = average_precision_score(Z[elig], R[elig])
        baseline_results[pert] = n_degs / int(elig.sum())

    print(f"Perturbations scored: {len(model_results)}/{len(perts)}")
    print(f"Average baseline AUPRC: {np.mean(list(baseline_results.values())):.4f}")
    print(f"Average model AUPRC:    {np.mean(list(model_results.values())):.4f}")
    return (model_results, true_labels) if return_labels else model_results


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

# Ground-truth top DEGs: depend ONLY on the real data -> compute once, cache on disk
def compute_top_degs(
    adata_true,
    n_top_degs,
    control_tag='non-targeting',
    condition_col='target_gene',
    lfc_threshold=DEG_LFC_THRESHOLD,
    alpha=DEG_ALPHA,
    min_cell_fraction=DEG_MIN_CELL_FRACTION
):
    """
    Ground-truth DEGs of every perturbation, defined exactly as in the DES metrics (RobustDES:
    Wilcoxon, BH-FDR < alpha, |log2FC| > lfc_threshold, logFC = log2 ratio of mean(expm1),
    gene detected in > min_cell_fraction of perturbed or control cells).
    Among the genes passing BOTH thresholds, the `n_top_degs` with the largest |Wilcoxon score| are kept.
    A perturbation can therefore have fewer than n_top_degs genes (or none).
    Returns {pert: [gene names]}.
    """
    des_calculator = RobustDES(lfc_threshold=lfc_threshold, alpha=alpha, min_cell_fraction=min_cell_fraction)
    perts = [p for p in adata_true.obs[condition_col].unique() if p != control_tag]
    top = {}
    for pert in tqdm(perts, desc='Calculating ground-truth DE genes...'):
        # get_de_genes returns (thresholded DEGs, full table); [1] alone would ignore the thresholds
        de_df, full_df = des_calculator.get_de_genes(adata_true, condition_col, pert, control_tag)

        # sel = full_df[full_df['names'].isin(de_df.index)]
        order = full_df['scores'].abs().sort_values(ascending=False, kind='stable').index[:n_top_degs]
        top[str(pert)] = full_df.loc[order, 'names'].tolist()
    return top


def load_or_compute_top_degs(adata_true, n_top_degs, cache_path, fingerprint=None,
                             control_tag='non-targeting', condition_col='target_gene',
                             lfc_threshold=DEG_LFC_THRESHOLD, alpha=DEG_ALPHA,
                             min_cell_fraction=DEG_MIN_CELL_FRACTION):
    """
    Loads {pert: [genes]} from `cache_path` if it was made with the same settings/data
    (checked through a small metadata block), otherwise computes it and writes the cache.
    """

    meta = {
        "n_top_degs": int(n_top_degs), 
        "control_tag": control_tag, 
        "condition_col": condition_col,
        "n_obs": int(adata_true.n_obs), 
        "n_vars": int(adata_true.n_vars),
        "fingerprint": fingerprint, 
        "lfc_threshold": lfc_threshold,
        "alpha": alpha,
        "min_cell_fraction": min_cell_fraction,
        "criterion": "des_fast_wilcoxon_either_group_min_pct_top_abs_z"
    }

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        if cached.get("meta") == meta:
            print(f"[*] Loaded top-{n_top_degs} DEGs from cache: {cache_path}")
            return cached["top_degs"]
        print(f"[!] DEG cache {cache_path} does not match current data/settings: recomputing")
    top = compute_top_degs(adata_true, n_top_degs, control_tag, condition_col, lfc_threshold, alpha, min_cell_fraction)
    try:
        tmp = cache_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"meta": meta, "top_degs": top}, f)
        os.replace(tmp, cache_path)  # atomic: safe with parallel jobs
        print(f"[*] Saved top-{n_top_degs} DEGs to: {cache_path}")
    except OSError as e:
        print(f"[!] Could not write DEG cache ({e}); continuing without it")
    return top


# Metric Evaluator Wrapper (Iterates over Perturbations)
def evaluate_metric_per_perturbation(adata_true, adata_pred, metric_func, control_tag='non-targeting', condition_col='target_gene', top_degs=None, **kwargs):
    """
    General wrapper that runs a given metric function for each individual perturbation.
    top_degs: optional {pert: [genes]} (see compute_top_degs); if given, each perturbation is evaluated on its genes only.
    """

    # new column "Expcategory" in .obs: "control" for control cells, 'stimulated' for gt, 'imputed' for predictions  
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

        if top_degs is not None:
            # restrict to this perturbation's (precomputed) ground-truth DEGs
            genes = [g for g in _degs_for(top_degs, pert) if g in adata_sub.var_names]
            # if len(genes) < MIN_DEG_GENES:  # too few DEGs for a meaningful score
            #     results[pert] = np.nan
            #     continue
            adata_sub = adata_sub[:, genes].copy()

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

def _core_common_degs(adata_sub, top_n=50):
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

    # NOTE: this is recall score; to have precision: return round(len(top_true_degs.intersection(top_pred_degs)) / len(top_pred_degs), 4)
    return round(len(top_true_degs.intersection(top_pred_degs)) / len(top_true_degs), 4)

# End-User Functions - to be called directly
# top_degs: optional {pert: [genes]}
def calc_mse(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_degs=None):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_mse, control_tag, condition_col, top_degs=top_degs)

def calc_pcc_delta(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_degs=None):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_pcc_delta, control_tag, condition_col, top_degs=top_degs)

def calc_edistance(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_degs=None, do_subsample=True):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_edistance, control_tag, condition_col, top_degs=top_degs, do_subsample=do_subsample)

def calc_wasserstein(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_degs=None, do_subsample=True):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_wasserstein, control_tag, condition_col, top_degs=top_degs, do_subsample=do_subsample)

def calc_kldiv(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_degs=None, do_subsample=True):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_kldiv, control_tag, condition_col, top_degs=top_degs, do_subsample=do_subsample)

def calc_common_degs(adata_true, adata_pred, control_tag='non-targeting', condition_col='target_gene', top_n=100):
    return evaluate_metric_per_perturbation(adata_true, adata_pred, _core_common_degs, control_tag, condition_col, top_degs=None, top_n=top_n)
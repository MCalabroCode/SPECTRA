import numpy as np
import pandas as pd
import anndata as ad

def compute_perturbation_mean_baseline(adata_train, adata_test, control_label="DMSO"):
    """
    Implements the Perturbation Mean Baseline

    Parameters
    ----------
    adata_train : AnnData
        Training data with expression matrix and metadata.
    adata_test : AnnData
        Test data (must contain same genes and same 'cell_type' + 'target_gene' columns).
    control_label : str
        The value in adata.obs['target_gene'] that represents the control (e.g. "DMSO").

    Returns
    -------
    adata_pred : AnnData
        AnnData object with predicted expression (Perturbation Mean Baseline).
    """

    # Make sure gene names align
    assert all(adata_train.var_names == adata_test.var_names), "Gene order mismatch"

    # Convert X to dense if needed
    X_train = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X

    df_obs = adata_train.obs.copy()
    cell_types = df_obs['cell_type'].unique()
    perturbations = df_obs['target_gene'].unique()

    # Step 1: compute cell-type–specific control means
    mu_ctrl = {}
    for c in cell_types:
        ctrl_idx = (df_obs['cell_type'] == c) & (df_obs['target_gene'] == control_label)
        if np.sum(ctrl_idx) == 0:
            continue
        mu_ctrl[c] = X_train[ctrl_idx].mean(axis=0)

    # Step 2: compute cell-type–specific perturbation means and offsets
    omega_cp = {}
    for c in cell_types:
        if c not in mu_ctrl:
            continue
        for p in perturbations:
            pert_idx = (df_obs['cell_type'] == c) & (df_obs['target_gene'] == p)
            if np.sum(pert_idx) == 0 or p == control_label:
                continue
            mu_pert = X_train[pert_idx].mean(axis=0)
            omega_cp[(c, p)] = mu_pert - mu_ctrl[c]

    # Step 3: compute global perturbation offsets
    omega_p = {}
    for p in perturbations:
        if p == control_label:
            continue
        offsets = [omega_cp[(c, p)] for (c, p2) in omega_cp if p2 == p]
        if len(offsets) > 0:
            omega_p[p] = np.mean(offsets, axis=0)
        else:
            # If perturbation not seen in training
            omega_p[p] = np.zeros(X_train.shape[1])

    # Step 4: predict test expression
    X_pred = np.zeros((adata_test.n_obs, X_train.shape[1]))
    df_test = adata_test.obs.copy()

    for i, row in df_test.iterrows():
        c = row['cell_type']
        p = row['target_gene']
        # If cell type or perturbation missing, fallback gracefully
        mu_c = mu_ctrl.get(c, np.zeros(X_train.shape[1]))
        offset_p = omega_p.get(p, np.zeros(X_train.shape[1]))
        X_pred[i, :] = mu_c + offset_p

    # Step 5: wrap predictions in a new AnnData object
    adata_pred = ad.AnnData(X_pred, var=adata_test.var.copy(), obs=adata_test.obs.copy())
    adata_pred.obs['prediction_source'] = "perturbation_mean_baseline"

    return adata_pred


def compute_perturbation_mean_baseline_relaxed(adata_train, adata_test, control_label="DMSO"):
    """
    Relaxed version of the Perturbation Mean Baseline (no cell type distinction).
    Predicts perturbed expression as global control mean + global perturbation offset.

    Parameters
    ----------
    adata_train : AnnData
        Training data with expression matrix and 'target_gene' column.
    adata_test : AnnData
        Test data (must contain same genes and 'target_gene' column).
    control_label : str
        The name of the control perturbation (e.g., "DMSO").

    Returns
    -------
    adata_pred : AnnData
        AnnData object with predicted expression (global perturbation mean baseline).
    """

    # Ensure gene order matches
    assert all(adata_train.var_names == adata_test.var_names), "Gene mismatch between train and test"

    X_train = adata_train.X.toarray() if hasattr(adata_train.X, "toarray") else adata_train.X
    df_obs = adata_train.obs

    # compute global control mean
    ctrl_idx = df_obs['target_gene'] == control_label
    mu_ctrl = X_train[ctrl_idx].mean(axis=0)

    # compute perturbation-specific global offsets
    perturbations = df_obs['target_gene'].unique()
    omega_p = {}
    for p in perturbations:
        if p == control_label:
            continue
        pert_idx = df_obs['target_gene'] == p
        if np.sum(pert_idx) == 0:
            continue
        mu_pert = X_train[pert_idx].mean(axis=0)
        omega_p[p] = mu_pert - mu_ctrl

    # predict test expressions
    X_pred = np.zeros((adata_test.n_obs, X_train.shape[1]))
    df_test = adata_test.obs.copy()

    # for i, row in df_test.iterrows():
    #     p = row['target_gene']
    #     offset = omega_p.get(p, np.zeros_like(mu_ctrl))
    #     X_pred[i, :] = mu_ctrl + offset
    for idx, (i, row) in enumerate(df_test.iterrows()):
        p = row['target_gene']
        offset = omega_p.get(p, np.zeros_like(mu_ctrl))
        X_pred[idx, :] = mu_ctrl + offset

    # Step 4: wrap predictions in AnnData
    adata_pred = ad.AnnData(X_pred, var=adata_test.var.copy(), obs=adata_test.obs.copy())
    adata_pred.obs['prediction_source'] = "global_perturbation_mean_baseline"

    return adata_pred


from sklearn.metrics import mean_absolute_error
import pandas as pd

def evaluate_baseline_mae(adata_pred, adata_true, groupby="target_gene"):
    """
    Compute Mean Absolute Error (MAE) between predicted and true gene expression.

    Parameters
    ----------
    adata_pred : AnnData
        Predicted expression profiles (e.g., from the baseline model).
    adata_true : AnnData
        Ground truth test data with real post-perturbation expression.
    groupby : str, optional
        Column in adata.obs to group results by (default: 'target_gene').

    Returns
    -------
    mae_df : pandas.DataFrame
        DataFrame with overall MAE and optionally per-group MAE.
    """

    # Ensure same shape and gene order
    assert all(adata_pred.var_names == adata_true.var_names), "Gene mismatch"
    assert adata_pred.n_obs == adata_true.n_obs, "Different number of cells"

    X_true = adata_true.X.toarray() if hasattr(adata_true.X, "toarray") else adata_true.X
    X_pred = adata_pred.X.toarray() if hasattr(adata_pred.X, "toarray") else adata_pred.X

    df_obs = adata_true.obs.copy()

    # Compute global MAE
    #mae_global = mean_absolute_error(X_true.flatten(), X_pred.flatten())

    mae_per_cell = np.mean(np.abs(X_true - X_pred), axis=1)  # average across genes for each cell
    mae_global = mae_per_cell.mean()  # average across all cells
    
    return mae_global

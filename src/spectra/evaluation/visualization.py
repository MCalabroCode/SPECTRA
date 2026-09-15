import numpy as np
import anndata as ad
import scanpy as sc
from matplotlib import pyplot as plt

def plot_violins_predictions_selected_genes(gene_list: list, real_adata: ad.AnnData, pred_adata: ad.AnnData, save: bool, title: str):

    # Extract and convert expression values
    real_vals = real_adata[:, gene_list].X
    pred_vals = pred_adata[:, gene_list].X

    if hasattr(real_vals, "toarray"):
        real_vals = real_vals.toarray()
    if hasattr(pred_vals, "toarray"):
        pred_vals = pred_vals.toarray()

    # Create the figure
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

    plot_violins_predictions_selected_genes(top_genes, real_adata, pred_adata, save, title=title)

def plot_top_highly_variable_genes(n_genes: int, real_adata: ad.AnnData, pred_adata: ad.AnnData, save: bool):

    sc.pp.highly_variable_genes(real_adata)
    hvg = real_adata.var[real_adata.var['highly_variable']]
    top_hvg = hvg.sort_values('highly_variable').head(n_genes)
    top_genes = top_hvg.index.tolist()
    title = f"Distribution of Top {n_genes} Highly Variable Genes"

    plot_violins_predictions_selected_genes(top_genes, real_adata, pred_adata, save, title=title)

def eval_barplot(results, metric_name):
    model_average = sum(results.values())/len(results)
    plt.figure(figsize=(15, 7))
    bars = plt.bar(list(results.keys()), list(results.values()))
    plt.xticks(rotation=90, ha='right')  # Vertical labels like your plot
    plt.axhline(y=model_average, color='green', linestyle='--', label=f'Model mean DES: {model_average:.2f}')
    plt.title(f'{metric_name}')
    plt.legend(frameon=False)
    plt.tight_layout()

def plot_top_de_gene_expression(real_adata, pred_adata, perturbation, n_top_DEGs=100, pert_key="target_gene",
                                control_label="control", point_size=35, figsize=(8, 8)):
    """
    Compares average ground truth vs predicted expression for top n_top_DEGs DE genes.
    """
    common_genes = real_adata.var_names.intersection(pred_adata.var_names)
    gt = real_adata[:, common_genes]
    pred = pred_adata[:, common_genes]
    genes = np.array(common_genes)

    def mean_expr(adata_sub, mask):
        X = adata_sub[mask].X
        if sparse.issparse(X):
            return np.asarray(X.mean(axis=0)).ravel()
        return np.asarray(X.mean(axis=0))

    gt_labels = gt.obs[pert_key].astype(str)
    pred_labels = pred.obs[pert_key].astype(str)

    gt_pert_mask = gt_labels == perturbation
    gt_ctrl_mask = gt_labels == control_label
    pred_pert_mask = pred_labels == perturbation

    if gt_pert_mask.sum() == 0:
        raise ValueError(f"No ground-truth cells found for perturbation '{perturbation}'")
    if pred_pert_mask.sum() == 0:
        raise ValueError(f"No predicted cells found for perturbation '{perturbation}'")
    if gt_ctrl_mask.sum() == 0:
        raise ValueError(f"No control cells found with label '{control_label}'")

    gt_pert_mean = mean_expr(gt, gt_pert_mask)
    gt_ctrl_mean = mean_expr(gt, gt_ctrl_mask)
    pred_pert_mean = mean_expr(pred, pred_pert_mask)

    de_score = np.abs(gt_pert_mean - gt_ctrl_mean)
    top_idx = np.argsort(de_score)[::-1][:n_top_DEGs]

    x_val = gt_pert_mean[top_idx]
    y_val = pred_pert_mean[top_idx]
    top_genes = genes[top_idx]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x_val, y_val, s=point_size, alpha=0.6, color='#D81B60')
    lim_min = min(x_val.min(), y_val.min())
    lim_max = max(x_val.max(), y_val.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linewidth=2, linestyle='--', color='gray')
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel("Ground-truth average expression")
    ax.set_ylabel("Predicted average expression")
    ax.set_title(f"{perturbation}: top {n_top_DEGs} ground-truth DE genes")
    ax.grid(True, linewidth=0.5)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({
        "gene": top_genes,
        "gt_mean": x_val,
        "pred_mean": y_val,
        "gt_control_mean": gt_ctrl_mean[top_idx],
        "abs_gt_de": de_score[top_idx],
    })
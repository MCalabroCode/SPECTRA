import anndata as ad
import argparse
import os
import gc  # Garbage collection to free up memory
from matplotlib import pyplot as plt
from val_scores import *

METRICS_REGISTRY = {
    "mae": calc_mae,
    "mse": calc_mse,
    "correlation": calc_corr,
    "kldiv": calc_kldiv,
    "common_degs": calc_common_degs,
    "pcc_delta": calc_pcc_delta,
    "wasserstein": calc_wasserstein,
    "edistance":calc_edistance,
    "f1":calc_f1,
    "precision":calc_precision,
    "auprc":calc_auprc
}

def process_datasets(real_path, pred_paths, metrics=None):
    """
    Loads the real dataset once, then iterates through prediction files
    one by one to compare, save, and free memory.
    """
    if metrics is None:
        metrics = METRICS_REGISTRY.keys()
    if not all(metric in METRICS_REGISTRY for metric in metrics):
        raise ValueError("invalid metric")
    results = {metric_name: {} for metric_name in metrics}

    print(f"[*] Loading real dataset from: {real_path}")
    real_adata = ad.read_h5ad(real_path)

    # for each model's prediction
    for path in pred_paths:
        if not os.path.exists(path):
            print(f"[!] Warning: File not found -> {path}")
            continue

        filename = os.path.basename(path)
        name_without_ext = os.path.splitext(filename)[0]

        # 1. Enforce strict naming convention
        if not name_without_ext.startswith("pred_adata-"):
            raise ValueError(
                f"Naming Error: '{filename}' does not follow the required "
                "format 'pred_adata-<model name>.h5ad'"
            )
        
        # Extract the model name by slicing off the "pred_adata-" prefix
        model_name = name_without_ext.replace("pred_adata-", "", 1)
        
        # Catch edge cases where the file is named just "pred_adata-.h5ad"
        if not model_name:
            raise ValueError(f"Naming Error: '{filename}' is missing the model name.")

        # 2. Load the single predicted dataset
        print(f"[*] Loading prediction for model '{model_name}': {filename}")
        p_adata = ad.read_h5ad(path)

        # 3. Calculate all metrics and store them
        for metric_name in metrics:
            print(f"    -> Calculating {metric_name}...")
            metric_func = METRICS_REGISTRY[metric_name]
            score = metric_func(real_adata, p_adata)
            results[metric_name][model_name] = score

        # 4. Free up memory
        del p_adata
        gc.collect()  # Force Python to immediately reclaim the memory
    
    # plot
    for metric in metrics:
        plot_results(results[metric], metric)

    return results


def plot_results(results: dict, metric: str):
    """
    Plots and saves the results for a specific metric.
    """
    figure_dir = 'benchmark_figures'
    os.makedirs(figure_dir, exist_ok=True)
    
    fig, ax = plt.subplots()
    
    models = list(results.keys())
    values = [list(results[key].values()) for key in models]
    
    bp = ax.boxplot(values, patch_artist=True)
    cmap = plt.cm.viridis  # you can choose others: plasma, coolwarm, etc.
    colors = cmap(np.linspace(0, 1, len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xticklabels(models, rotation=45)
    ax.set_title(metric)
    plt.tight_layout()
    plot_name = os.path.join(figure_dir, f"{metric}.pdf")
    plt.savefig(plot_name)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate scRNAseq models across multiple metrics.")
    
    parser.add_argument(
        "--real", 
        type=str, 
        required=True, 
        help="Path to the real AnnData .h5ad file."
    )
    
    parser.add_argument(
        "--preds", 
        type=str, 
        nargs='+', 
        required=True, 
        help="Paths to predicted .h5ad files (Must be named pred_adata-<model name>.h5ad)."
    )

    # NEW: Allow users to specify which metrics to run
    parser.add_argument(
        "--metrics",
        type=str,
        nargs='+',
        choices=list(METRICS_REGISTRY.keys()), # Automatically restricts inputs to valid keys
        help="Specify which metrics to run (e.g., --metrics mae kldiv). If omitted, runs all."
    )

    args = parser.parse_args()

    # Pass args.metrics into the function
    final_results = process_datasets(args.real, args.preds, metrics=args.metrics)
    print("[*] All evaluations complete.")



# SPECTRA: predicting cellular perturbation responses with Graph Learning over Gene Regulatory Networks🧬

**SPectral CRISPR Transcriptome Regulatory Autoencoder**

SPECTRA is a graph-based generative model designed to simulate single-cell transcriptional responses to out-of-distribution (OOD) CRISPR gene knockouts. By leveraging prior Gene Regulatory Networks (GRNs) and foundational gene embeddings (e.g., scGPT), SPECTRA explicitly propagates perturbation signals through directed network topologies to capture sparse, localized regulatory cascades.

---

## 📂 Repository Structure

```text
SPECTRA/
│
├── configs/                # YAML configuration files for training and sweeps
│   ├── replogle_config.yaml
│   ├── vcc_config.yaml
│   └── sweep_config.yaml
│
├── data/                   # Directory for h5ad datasets, GRNs, and gene embeddings
│   └── (See Data Setup section below)
│
├── src/
│   └── spectra/            # Core Python package
│       ├── data/           # Dataloaders, graph preprocessing, and weights
│       ├── models/         # SPECTRA architecture and Directed GNN layers
│       ├── training/       # Training routines and custom loss functions
│       └── evaluation/     # Metrics (AUPRC, MAE, DEG Overlap)
│
├── scripts/                # Executable scripts
│   ├── train.py            # Standard training loop
│   ├── run_sweep.py        # Weights & Biases hyperparameter tuning agent
│   └── benchmark.py        # Multi-model evaluation script
│
├── notebooks/              # Interactive tutorials
│   ├── spectra_testing.ipynb 
│   └── spectra_training.ipynb 
│
├── setup.py
└── requirements.txt
```

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/MCalabroCode/SPECTRA.git
cd SPECTRA
```

2. **Create and activate a virtual environment**:

```bash
python3.11 -m venv spectra_venv
source spectra_venv/bin/activate
```

3. **Install the package locally:**

```bash
python -m pip install -r requirements.txt
pip install -e .
```

## 🚀 Training

To train a SPECTRA model on a specific dataset, use the [`scripts/train.py`](scripts/train.py) script. The YAML config file specifies the paths to your dataset, network file, and hyperparameters.

```bash
python scripts/train.py --config configs/config.yaml
```

Trained model weights will be saved automatically inside the `weights/` directory.

### Hyperparameter Sweeps (Weights & Biases)

SPECTRA supports automated Bayesian hyperparameter optimization via Weights & Biases.
To initialize a sweep and start an agent:

```bash
python scripts/run_sweep.py --config configs/sweep_config.yaml --count 10
```

## 📊 Inference & Evaluation

### 1. Interactive Inference & Interpretability

To load a pre-trained model, generate data and test predictions interactively, refer to [`notebooks/spectra_testing.ipynb`](notebooks/spectra_testing.ipynb). 

This tutorial walks through:
- Generating single-cell and pseudobulk expression predictions for out-of-distribution (OOD) knockouts.
- Extracting directed perturbation-induced cascades and gene regulatory flow.
- Generating Sankey path flow diagrams and Gene Ontology (GO) enrichment plots.

### 2. Single-Run Evaluation CLI

To evaluate a single run against ground truth using standard perturbation metrics, use [`scripts/evaluate.py`](scripts/evaluate.py). 

Ensure your ground-truth AnnData and model predictions follow the naming convention `pred_adata-<model_name>.h5ad`:

```bash
python scripts/evaluate.py \
    --real results/test_cells.h5ad \
    --preds results/pred_adata-SPECTRA.h5ad \
            results/pred_adata-GEARS.h5ad \
            results/pred_adata-scLambda.h5ad \
    --top_DEGs 50 \
    --metrics mse pcc_delta edistance f1 common_degs auprc
```

**Key Arguments:**

* `--real`: Path to the ground-truth test AnnData file (`.h5ad`).
* `--preds`: One or more paths to predicted AnnData files (`pred_adata-<model>.h5ad`).
* `--metrics`: Subset of metrics to compute (`mae`, `mse`, `correlation`, `pcc_delta`, `kldiv`, `wasserstein`, `edistance`, `f1`, `precision`, `auprc`, `common_degs`). If omitted, all available metrics are evaluated.
* `--top_DEGs`: Restricts evaluation to the top $N$ differentially expressed genes (DEGs). If omitted, metrics are computed across all genes.

### 3. Multi-Run Benchmarking Suites

For comprehensive, multi-seed statistical evaluation across baselines:

* **Custom Benchmark Suite (`notebooks/benchmark.ipynb`):**
Evaluates multiple repeated runs across models (`SPECTRA`, `GEARS`, `scLambda`, and mean baselines), computing distribution-level and DEG-focused metrics (WMSE, MSE, Wasserstein distance, AUPRC, and differential expression F1/Precision) with automated disk caching and summary reporting.
* **Standardized Arc Cell-Eval (`notebooks/benchmark_cell_eval.ipynb`):**
Benchmarks models using the standardized [`cell-eval`](https://github.com/ArcInstitute/cell-eval) framework. Computes:
* **DEG Overlap & Precision:** `overlap_at_N`, `overlap_at_50/100/200/500`, and `precision_at_k`.
* **Effect Directionality & Correlation:** `de_direction_match`, `de_spearman_sig`, and `pearson_delta`.
* **Distance & Discrimination:** L1/L2/cosine discrimination scores, `pearson_edistance`, MSE/MAE, and clustering agreement.



<!-- ## 📄 Citation

If you use SPECTRA or find this codebase helpful in your research, please cite our preprint:

```bibtex
@article{calabro2026spectra,
  title={SPECTRA: predicting cellular perturbation responses with Graph Learning over Gene Regulatory Networks},
  author={Calabrò, Michele},
  year={2026}
}

``` -->

## 📜 License

This project was created by Michele Calabrò and is licensed under the terms of the [MIT License](LICENSE).
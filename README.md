# SPECTRA: predicting cellular perturbation responses with Graph Learning over Gene Regulatory Networks🧬

**SPectral CRISPR Transcriptome Regulatory Autoencoder**

SPECTRA is a graph-based generative model designed to simulate single-cell transcriptional responses to out-of-distribution (OOD) CRISPR gene knockouts. By leveraging prior Gene Regulatory Networks (GRNs) SPECTRA explicitly propagates perturbation signals through directed network topologies to capture sparse, localized regulatory cascades.

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

2. **Create and activate a virtual environment**
  
    SPECTRA was developed in python 3.11:

```bash
python3.11 -m venv spectra_venv
source spectra_venv/bin/activate
```

3. **Install hardware-compatible versions of PyTorch and PyTorch Geometric**

   SPECTRA was developed using **PyTorch 2.4.1** and **CUDA 12.1**.
   If your GPU cluster uses a different CUDA version or you are running on CPU, adjust the PyTorch and PyG installation links accordingly.

```bash
pip install torch==2.4.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

4. **Install the package locally**

   To install SPECTRA with only the core dependencies required for training, data preprocessing, and evaluation, run:

```bash
pip install -e .
```

**OPTIONAL DEPENDENCIES**

SPECTRA uses modular optional dependencies so you can install specialized toolkits only when needed:

* run `pip install -e ".[sankey]"` to install SPECTRA together with the requirements to reproduce the **interpretability & Sankey diagrams**.
* run `pip install -e ".[eval]"` to install SPECTRA together with the **extended benchmark suite** (Pertpy and [`cell-eval`](https://github.com/ArcInstitute/cell-eval) metrics)
* run `pip install -e ".[wandb]"` to install SPECTRA together with the **support for hyperparameter sweeps** (wandb logging)

**We strongly suggest to install SPECTRA with every optional tooklit provided** (plotting, benchmarking, tracking, and notebooks, etc) by running:

```bash
pip install -e ".[all]"
```

### installation via `requirements.txt`

For strict reproducibility of the results reported in the paper, we recommend creating the exact environment used in our experiments by installing the dependencies from `requirements.txt`.

After completing steps 1. and 2. above, run:

```bash
python -m pip install -r requirements.txt
```

## 🚀 Training

To train a SPECTRA model on a specific dataset, use the [`scripts/train.py`](scripts/train.py) script. The YAML config file specifies the paths to your dataset, network file, and hyperparameters. For examples of the expected configuration structure see the YAML files in the `config` directory.

```bash
python scripts/train.py --config configs/config.yaml
```

Trained model weights will be saved automatically inside the `weights/` directory.

### Hyperparameter Sweeps (Weights & Biases)

SPECTRA supports automated Bayesian hyperparameter optimization via Weights & Biases. It requires `wandb` to be installed (see the installation guide above).

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
- Generating Sankey path flow diagrams and Gene Ontology (GO) enrichment plots, as well as other figures present in the paper.

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
  Evaluates multiple repeated runs across models (currently `SPECTRA`, `GEARS`, `scLambda`, and mean baselines), computing distribution-level and DEG-focused metrics (WMSE, MSE, Wasserstein distance, AUPRC, and differential expression F1/Precision) with automated disk caching and summary reporting.
* **Standardized Arc Cell-Eval (`notebooks/benchmark_cell_eval.ipynb`):**
  Benchmarks models using the standardized [`cell-eval`](https://github.com/ArcInstitute/cell-eval) framework. Computes:
  * **DEG Overlap & Precision:** `overlap_at_N`, `overlap_at_50/100/200/500`, and `precision_at_k`.
  * **Effect Directionality & Correlation:** `de_direction_match`, `de_spearman_sig`, and `pearson_delta`.
  * **Distance & Discrimination:** L1/L2/cosine discrimination scores, `pearson_edistance`, MSE/MAE, and clustering agreement.

## 📄 Citation

If you use SPECTRA or find this codebase helpful in your research, please cite our preprint:

```bibtex
@article {calabro2026spectra,
	author = {Calabro, Michele and Sheehan, Patrick and Cambuli, Francesco and Sottoriva, Andrea},
	title = {SPECTRA: predicting cellular perturbation responses with Graph Learning over Gene Regulatory Networks},
	year = {2026},
	doi = {10.64898/2026.09.21.752624},
	journal = {bioRxiv}
}
```

## 📜 License

This project was created by Michele Calabrò and is licensed under the terms of the [MIT License](LICENSE).

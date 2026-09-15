# SPECTRA: Graph Signal Propagation over Gene Regulatory Networks for predicting transcriptomic responses to perturbations 🧬

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

1. **Clone the repository (NOTE: not available yet! Just copy the files):**

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

To train a SPECTRA model on a specific dataset, use the `train.py` script. The YAML config file specifies the paths to your dataset, network file, and hyperparameters.

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

## 📊 Evaluation & Inference

### 1. Interactive Inference

To load a pre-trained model and test predictions interactively, check out the provided Jupyter Notebook `notebooks/spectra_testing.ipynb`

### 2. Full Benchmark Pipeline

To evaluate SPECTRA against baselines across multiple metrics, use the `benchmark.py` script. Ensure that your ground-truth data (e.g., `real_adata.h5ad`) and generated predictions (e.g., `pred_adata-<model_name>.h5ad`) are available.

```bash
python scripts/benchmark.py \
    --real results/real_adata.h5ad \
    --preds results/pred_adata-model1.h5ad \
            results/pred_adata-model2.h5ad \
    --top_DEGs 50 \
    --metrics mse pcc_delta edistance f1 common_degs
```

- `--top_DEGs`: Restricts the calculation of metrics to the top $N$ Differentially Expressed Genes. If omitted, metrics are calculated across all genes.
- `--metrics`: List of metrics to compute (e.g., `mae`, `mse`, `edistance`, `common_degs`, `pcc_delta`).

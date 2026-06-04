import scanpy as sc
from spectra.data import data_preprocessing
import pandas as pd
import numpy as np
import anndata as ad

ad.settings.allow_write_nullable_strings = True

adata = sc.read_h5ad('data/K562_gwps_raw_singlecell_01.h5ad')

# this is the column indicating the CRISR perturbed gene (or coontrol condition)
obs = adata.obs[['gene']].copy()

# Separate 'non-targeting' indices
idx_control = obs.index[obs['gene'] == 'non-targeting'] 

# Filter less abundant perturbation conditions
obs_exp = obs[obs['gene'] != 'non-targeting']
counts = obs_exp['gene'].value_counts()
valid_genes = counts[counts >= 100].index

# subset to N cells for most abundand perturbation conditions
idx_experimental = (
    obs_exp[obs_exp['gene'].isin(valid_genes)]
    .groupby('gene', group_keys=False)
    .apply(lambda x: x.sample(n=min(len(x), 500), random_state=42))
    .index
)

# merge with control
keep_indices = idx_control.union(idx_experimental)
adata = adata[adata.obs_names.isin(keep_indices)].copy()

# rename var names - must be HUGO symbols
adata.var['gene_name'] = adata.var['gene_name'].astype('string')
adata.var_names = adata.var['gene_name']
adata.var_names_make_unique()
adata.var = adata.var.drop(columns=['gene_name'])

# Important: avoid conflict between var index name and var column name
adata.var.index.name = None

# Rename perturbation column (must be 'target_gene') and ctrl samples ('non-targeting')
control_tag = 'non-targeting'
adata.obs = adata.obs.rename(columns={'gene': "target_gene"})
adata.obs["target_gene"] = adata.obs["target_gene"].str.replace(control_tag, "non-targeting", regex=False)
adata.obs['target_gene'] = adata.obs['target_gene'].str.replace(r'\+non-targeting$', '', regex=True)

# preprocessing
adata = data_preprocessing(adata, 
    min_genes=200, 
    min_cells=10, 
    min_cells_per_pert=100, 
    logtransform=False
)

# save
adata.write('../data/replogle_k562_filtered.h5ad')
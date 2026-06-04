'''
data preprocessing routines
'''

import scanpy as sc

def data_preprocessing(adata, 
    min_genes=200, 
    min_cells=3, 
    min_cells_per_pert=100, 
    logtransform=True
):

    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    if logtransform:
        sc.pp.normalize_total(adata, target_sum = 1e4)
        sc.pp.log1p(adata)

    # select only perturbations that are present in at least min_cells_per_pert cells
    counts = adata.obs['target_gene'].value_counts()
    valid_pert = counts[counts >= min_cells_per_pert].index
    adata = adata[adata.obs['target_gene'].isin(valid_pert)]
    return adata

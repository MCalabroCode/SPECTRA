import numpy as np 
from tqdm import tqdm
import pandas as pd
import anndata as ad

def generate_adata_baseline(gene_counts_dict, train_adata):
    """
    Generates an AnnData object using a simple pseudobulk average baseline model.
    Duplicates the mean expression profile for n_samples.
    """

    prediction_list = []
    obs_gene_list = []

    # precompute pseudobulk and means
    df = pd.DataFrame(
        train_adata.X.toarray() if hasattr(train_adata.X, "toarray") else np.asarray(train_adata.X),
        index=train_adata.obs["target_gene"],
        columns=train_adata.var_names
    )
    means = df.groupby(level=0, sort=False).mean()
    if 'non-targeting' in means.index:
        means = means.loc[means.index!='non-targeting']
        
    # Precompute global average for zero-shot / unseen perturbations
    global_mean = means.mean(axis=0).values

    # generation
    for pert, n_samples in tqdm(gene_counts_dict.items()):
            
        # Retrieve the specific mean, or fallback to global mean if unseen
        if pert in means.index:
            pert_pred = means.loc[pert].values
        else:
            pert_pred = global_mean
            
        # repeat the 1D prediction array into a 2D array of shape [n_samples, n_genes]
        repeated_preds = np.tile(pert_pred, (n_samples, 1))
        
        prediction_list.append(repeated_preds)
        obs_gene_list.extend([pert] * n_samples)

    # compile final data
    X = np.vstack(prediction_list)
    obs = pd.DataFrame({"target_gene": obs_gene_list})
    var = pd.DataFrame(index=train_adata.var_names)
    pred_adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return pred_adata

def technical_duplicate_baseline(adata):
    '''
    We compute this baseline by randomly dividing the population of cells 
    receiving a perturbation in half and using one half of the cells to
    predict the other half. Works only for pertubrations already seen.
    '''
    indices_1 = []
    indices_2 = []
    for gene, idx in adata.obs.groupby("target_gene").indices.items():
        idx = np.array(idx) # list of indices associated to gene (the target_gene)
        np.random.shuffle(idx)  # Randomize indices
        half = len(idx) // 2
        #indices_1.extend(idx[:half])
        indices_2.extend(idx[half:])
    #real_adata = adata[indices_1].copy()
    pred_adata = adata[indices_2].copy()
    return pred_adata
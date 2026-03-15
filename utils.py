
# functions to configure the model and the data

import os
import json
import argparse
import networkx as nx
import anndata as ad
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from scipy import sparse
import anndata as ad
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


from torch.utils.data.sampler import Sampler
import random

#TODO: must be adapted for SPECTRA
def load_model_config(config_file_path: str):
    ''' import json configuration file for the model

    Args:
        config_file_path (str): path to the configuration file
    Retruns:
        args: List of arguments and hyperparameters for the model
    '''
    
    # First, check if the configuration file exists.
    if not os.path.exists(config_file_path):
        raise ValueError(f"Error: The configuration file '{config_file_path}' was not found.")
    else:
        with open(config_file_path, 'r') as f:
            config = json.load(f)

    # Convert the dictionary into a namespace object for easy access (e.g., args.epochs)
    args = argparse.Namespace(**config)

    print("Configuration successfully loaded from JSON:")
    print(f"Epochs: {args.epochs}")
    print(f"Number of channels: {args.n_channels}")
    print(f"Learning rate: {args.lr}")
    print(f"Test ratio: {args.test_ratio}")

    return args

class SCDATA_sampler(Sampler):
    '''
    a special batch sampler that groups only cells from the same interventional distribution into a batch
    '''
    def __init__(self, data, batchsize, ptb_name=None):
        self.intervindices = []
        self.len = 0
        if ptb_name is None:
            ptb_name = data.ptb_names

        for ptb in set(ptb_name):
            idx = np.where(ptb_name == ptb)[0] # indices of cells with the same pert ptb
            self.intervindices.append(idx) # list of indices of cells with the same pert ptb
            self.len += len(idx) // batchsize # number of batches with pert ptb
        self.batchsize = batchsize
    
    def __iter__(self):
        comb = []
        # loop over each intervention
        for i in range(len(self.intervindices)):
            random.shuffle(self.intervindices[i]) # intra-Perturbation Shuffle
            interv_batches = chunk(self.intervindices[i], self.batchsize)
            if interv_batches:
                comb += interv_batches

        combined = [batch.tolist() for batch in comb]
        random.shuffle(combined) # shuffle the order of the batches
        return iter(combined)

    def __len__(self):
        return self.len

def chunk(indices, chunk_size):
    split = torch.split(torch.tensor(indices), chunk_size) # this divides the torch indices into subsets of equal length chunk_size
    
    if len(indices) % chunk_size == 0:
        return split
    elif len(split) > 0:
        # If there are some indices left over (the last chunk is smaller than chunk_size),
        # this line discards the last chunk. 
        # It returns all the chunks except the last one.
        return split[:-1]
    else:
        return None



class PerturbationDataset(Dataset):
    def __init__(self, adata, gene_to_idx, edge_index, start_idx=0, end_idx=None, seed=42):
        super().__init__()

        self.edge_index = edge_index.share_memory_()
        self.gene_to_idx = gene_to_idx
        self.start_idx = start_idx
        self.seed = seed
        self.end_idx = end_idx if end_idx is not None else len(adata)

        adata = adata[self.start_idx:self.end_idx].copy()
        self.adata = adata

        # perturbed samples
        ptb_adata = adata[adata.obs['target_gene']!='non-targeting'].copy()
        ptb_samples = ptb_adata.X.toarray() if hasattr(ptb_adata.X, 'toarray') else np.asarray(ptb_adata.X)
        self.ptb_samples = torch.tensor(ptb_samples, dtype=torch.float) #NOTE: local to this split
        self.ptb_names = ptb_adata.obs['target_gene'].values #NOTE: local to this split

        # control samples
        self.ctrl_samples = adata[adata.obs['target_gene']=='non-targeting'].X.copy()
        self.ctrl_samples = self.ctrl_samples.toarray() if hasattr(self.ctrl_samples, 'toarray') else np.asarray(self.ctrl_samples)
        self.ctrl_samples = torch.tensor(self.ctrl_samples, dtype=torch.float)

    def _pert_embedding(self, pert_name):
        embedding = torch.zeros(self.adata.shape[1], dtype=torch.bool)
        perturbs = [self.gene_to_idx[g] for g in pert_name.split('+') if g in self.gene_to_idx]
        for pert in perturbs:
            embedding[pert]=True
        return embedding

    def __getitem__(self, idx):
        
        # Map to actual index in adata
        actual_idx = idx

        #x = self.rand_ctrl_samples[actual_idx].view(-1,1)
        j = np.random.randint(0, self.ctrl_samples.shape[0]) #random control cell
        x = self.ctrl_samples[j].view(-1,1)
        y = self.ptb_samples[actual_idx].view(-1,1)
        pert = self._pert_embedding(self.ptb_names[actual_idx])

        return x,y,pert

    def __len__(self):
        return self.ptb_samples.shape[0]

def build_model_dataloaders(adata, edge_index, config):
    
    #TODO: add sanity check for dataset_size (must be inferior than the number of the perturbed cells, see how the sampling works)

    # Setup
    dataset_size = config['dataset_size'] #(adata.obs['target_gene'] != 'non-targeting').sum() 
    test_ratio = config['test_ratio']
    val_ratio = config['val_ratio']
    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - test_size - val_size

    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}

    # Create datasets
    train_dataset = PerturbationDataset(
        adata, gene_to_idx, edge_index,
        start_idx=0, 
        end_idx=train_size
    )

    val_dataset = PerturbationDataset(
        adata, gene_to_idx, edge_index,
        start_idx=train_size, 
        end_idx=train_size+val_size
    )

    test_dataset = PerturbationDataset(
        adata, gene_to_idx, edge_index,
        start_idx=train_size+val_size, 
        end_idx=dataset_size
    )

    N_WORKERS = 4 #TODO this will be passed as an argument to training load dataset
    batch_size = config['batch_size']

    # create loaders
    train_loader = DataLoader(train_dataset, batch_sampler=SCDATA_sampler(train_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) #pin memory optimize transfer to CUDA
    val_loader = DataLoader(val_dataset, batch_sampler=SCDATA_sampler(val_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_sampler=SCDATA_sampler(test_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) 

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    return train_loader, val_loader, test_loader


####### data generation #######


def baseline_model(train_adata, pert):
    '''
    The model should return just one sample. 
    Here, we set up the possibility of return a certain number of samples for each pert, to emulate
    the generation of the predictions in the a<ctual model
    '''

    # pesuedobulk creation
    df = pd.DataFrame(
        train_adata.X.toarray() if hasattr(train_adata.X, "toarray") else train_adata.X,
        index=train_adata.obs["target_gene"],
        columns=train_adata.var_names
    )

    means = df.groupby(level=0, sort=False).mean()
    means = means.loc[means.index!='non-targeting']
    if pert in means.index:
        return means.loc[pert,:] #pandas series
    else:
        return means.mean(axis=0) #if pert is not part of traioning perturbs , return the average on all perturbations

def generate_adata_baseline(gene_counts_dict, train_adata, var_names):
    """
    Generates an AnnData object using a simple pseudobulk average baseline model.
    Duplicates the mean expression profile for n_samples.
    """

    prediction_list = []
    obs_gene_list = []

    # precompute means
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
            
        # Duplicate the 1D prediction array into a 2D array of shape [n_samples, n_genes]
        repeated_preds = np.tile(pert_pred, (n_samples, 1))
        
        prediction_list.append(repeated_preds)
        obs_gene_list.extend([pert] * n_samples)

    # compile final data
    X = np.vstack(prediction_list)
    obs = pd.DataFrame({"target_gene": obs_gene_list})
    var = pd.DataFrame(index=var_names)
    pred_adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return pred_adata

def generate_adata_from_control(gene_counts_dict, real_adata, model, gene_to_idx, var_names, batch_size=32):

    prediction_list = []
    obs_gene_list = []

    # ctrl
    real_adata_control = real_adata[real_adata.obs['target_gene'] == 'non-targeting']
    ctrl_data = real_adata_control.X.toarray() if hasattr(real_adata_control.X, "toarray") else np.asarray(real_adata_control.X)
    ctrl_data = torch.tensor(ctrl_data, dtype=torch.float)

    model.eval()
    with torch.no_grad():
        for pert, n_samples in tqdm(gene_counts_dict.items()):

            # Perturbation one-hot embedding
            pert_embedding = torch.zeros(real_adata.shape[1], dtype=torch.bool)
            if pert != 'non-targeting':
                perturbs = [gene_to_idx[g] for g in pert.split('+') if g in gene_to_idx]
                for single_pert in perturbs:
                    pert_embedding[single_pert] = True
            pert_embedding = pert_embedding.unsqueeze(0)
            
            # Mini-batch generation to prevent CUDA OOM
            for offset in range(0, n_samples, batch_size):

                # Calculate how many cells to generate in this specific chunk
                chunk_size = min(batch_size, n_samples - offset)
                pert_batch = pert_embedding.expand(chunk_size, -1).to(model.device)
                
                # Sample control cells for this chunk
                random_indices = np.random.choice(ctrl_data.shape[0], size=chunk_size, replace=True)
                ctrl_batch = ctrl_data[random_indices, :] # shape [B,N]
                ctrl_batch = ctrl_batch.unsqueeze(-1).to(model.device) # shape [B,N,1]
                
                # Predict
                data = ctrl_batch, pert_batch
                predicted_full_gex = model.predict_full_expression(data)
                

                predicted_full_gex = predicted_full_gex.reshape(chunk_size, -1).detach().cpu().numpy()
                prediction_list.append(predicted_full_gex) 
                obs_gene_list.extend([pert] * chunk_size)

    # Compile the final AnnData
    X = np.vstack(prediction_list)
    obs = pd.DataFrame({"target_gene": obs_gene_list})
    var = pd.DataFrame(index=var_names)
    pred_adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return pred_adata

def generate_adata_from_control_old(gene_counts_dict, real_adata, model, gene_to_idx, var_names):
    '''
    this code is not optimized. Do not use.
    '''

    prediction_list = []
    obs_gene_list = []

    # ctrl
    real_adata_control = real_adata[real_adata.obs['target_gene']=='non-targeting']
    ctrl_data = real_adata_control.X.toarray() if hasattr(real_adata_control.X, "toarray") else np.asarray(real_adata_control.X)
    ctrl_data = torch.tensor(ctrl_data, dtype=torch.float)

    for pert, n_samples in tqdm(gene_counts_dict.items()):

        # perturbation one-hot embedding
        pert_embedding = torch.zeros(len(var_names), dtype=torch.bool)
        if pert!='non-targeting':
            perturbs = [gene_to_idx[g] for g in pert.split('+') if g in gene_to_idx] #NOTE: this takes into account multi-genes pertrurbations
            for single_pert in perturbs:
                pert_embedding[single_pert]=True
            pert_embedding = pert_embedding.unsqueeze(0)
            
        for i in range(n_samples):
            ctrl_sample = ctrl_data[np.random.randint(0, ctrl_data.shape[0], 1), :].reshape(-1,1).unsqueeze(0)
            data = ctrl_sample, pert_embedding
            predicted_full_gex= model.predict_full_expression(data)
            predicted_full_gex = predicted_full_gex.reshape(1,-1).detach().cpu()
            prediction_list.append(predicted_full_gex.numpy()) 
            obs_gene_list.append(pert)

    X = np.vstack(prediction_list)
    obs = pd.DataFrame({"target_gene": obs_gene_list})
    var = pd.DataFrame(index=var_names)
    pred_adata = ad.AnnData(X=X, obs=obs, var=var)
    
    return pred_adata

####### with TSNE
from sklearn.manifold import TSNE

def _generate_adata_from_control_old(gene_counts_dict, real_adata, model, gene_to_idx, var_names):
    '''
    this code is not optimized. Do not use.
    contains TSNE projection
    '''
    prediction_list = []
    obs_gene_list = []
    latent_representations = []

    real_adata_control = real_adata[real_adata.obs['target_gene']=='non-targeting']
    ctrl_data = real_adata_control.X.toarray() if hasattr(real_adata_control.X, "toarray") else np.asarray(real_adata_control.X)
    ctrl_data = torch.tensor(ctrl_data, dtype=torch.float)

    for pert, n_samples in tqdm(gene_counts_dict.items()):

        # perturbation one-hot embedding
        pert_embedding = torch.zeros(len(var_names), dtype=torch.bool)
        if pert!='non-targeting':
            perturbs = [gene_to_idx[g] for g in pert.split('+') if g in gene_to_idx] #NOTE: this takes into account multi-genes pertrurbations
            for single_pert in perturbs:
                pert_embedding[single_pert]=True
            pert_embedding = pert_embedding.unsqueeze(0)
            
        # sampling random control cell
        for i in range(n_samples):
            ctrl_sample = ctrl_data[np.random.randint(0, ctrl_data.shape[0], 1), :].reshape(-1,1).unsqueeze(0)
            data = ctrl_sample, pert_embedding
            predicted_full_gex, z_mean = model.predict_full_expression(data)
            predicted_full_gex = predicted_full_gex.reshape(1,-1).detach().cpu()
            z_mean = z_mean.reshape(1,-1)
            latent_representations.append(z_mean)
            prediction_list.append(predicted_full_gex.numpy()) 
            obs_gene_list.append(pert)

        
    X = np.vstack(prediction_list)
    obs = pd.DataFrame({"target_gene": obs_gene_list})
    var = pd.DataFrame(index=var_names)
    pred_adata = ad.AnnData(X=X, obs=obs, var=var)

    # print('calculating embeddings for the latent space...')
    # latent_complete = np.vstack(latent_representations)
    # embedding_2d = TSNE(n_components=2, init='pca').fit_transform(latent_complete)
    # pred_adata.obsm['X_tsne'] = embedding_2d
    # print('tsne embedding: done')
    
    print('calculating UMAP for the latent space...')
    latent_complete = np.vstack(latent_representations)
    pred_adata.obsm['X_latent'] = latent_complete
    sc.pp.neighbors(pred_adata, use_rep='X_latent', n_neighbors=15, metric='euclidean')
    sc.tl.umap(pred_adata)

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

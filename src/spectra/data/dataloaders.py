'''
dataloaders creation routines
'''

import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
import anndata as ad
from torch.utils.data.sampler import Sampler

import json
import numpy as np
import anndata as ad


class SCDATA_sampler(Sampler):
    '''
    a special batch sampler that groups only cells from the same interventional distribution into a batch.
    From MORPH model: https://github.com/uhlerlab/MORPH/blob/main/morph/dataset.py
    '''
    def __init__(self, data, batchsize, ptb_name=None, shuffle=True):
        self.intervindices = []
        self.len = 0
        self.shuffle = shuffle
        self.batchsize = batchsize
        
        if ptb_name is None:
            ptb_name = data.ptb_names

        for ptb in sorted(set(ptb_name)):
            idx = np.where(ptb_name == ptb)[0] # indices of cells with the same pert ptb
            self.intervindices.append(idx) # list of indices of cells with the same pert ptb
            self.len += len(idx) // batchsize # number of batches with pert ptb
    
    def __iter__(self):
        combined = []

        for original_indices in self.intervindices:
            
            # Important: do not modify the stored indices
            indices = original_indices.copy()
            if self.shuffle:
                np.random.shuffle(indices)

            batches = chunk(indices, self.batchsize)
            if batches:
                combined.extend(batch.tolist() for batch in batches)

        if self.shuffle:
            random.shuffle(combined)

        return iter(combined)

    def __len__(self):
        return self.len

def chunk(indices, chunk_size):
    split = torch.split(torch.tensor(indices), chunk_size) # this divides the torch indices into subsets of equal length chunk_size
    #NOTE: split will be a tuple of tensors
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
    def __init__(self, adata, gene_to_idx, pert_to_idx, start_idx=0, end_idx=None, control_sampling="random", control_seed=42,):
        super().__init__()

        self.gene_to_idx = gene_to_idx
        self.pert_to_idx = pert_to_idx
        self.start_idx = start_idx
        self.end_idx = end_idx if end_idx is not None else len(adata)

        adata = adata[self.start_idx:self.end_idx]
        self.adata = adata

        ptb_adata = adata[adata.obs['target_gene'] != 'non-targeting']
        ctrl_adata = adata[adata.obs['target_gene'] == 'non-targeting']
        self.ptb_samples = ptb_adata.X.tocsr() if sparse.issparse(ptb_adata.X) else sparse.csr_matrix(ptb_adata.X)
        self.ptb_names = ptb_adata.obs['target_gene'].values 
        self.ctrl_samples = ctrl_adata.X.tocsr() if sparse.issparse(ctrl_adata.X) else sparse.csr_matrix(ctrl_adata.X)
        if self.ctrl_samples.shape[0] == 0:
            raise ValueError("No control cells found in this dataset split!")

        self.control_sampling = control_sampling
        if control_sampling == "fixed":
            rng = np.random.default_rng(control_seed)
            permutation = rng.permutation(self.ctrl_samples.shape[0])
            repeats = int(np.ceil(self.ptb_samples.shape[0] / len(permutation)))
            self.fixed_control_indices = np.tile(permutation, repeats,)[:self.ptb_samples.shape[0]]

    def _pert_embedding(self, pert_name):
        embedding = torch.zeros(self.adata.shape[1], dtype=torch.bool)
        perturbs = [self.gene_to_idx[g] for g in pert_name.split('+') if g in self.gene_to_idx]
        for pert in perturbs:
            embedding[pert]=True
        return embedding

    def __getitem__(self, idx):

        if self.control_sampling == "random":
            j = np.random.randint(0, self.ctrl_samples.shape[0])
        else:
            j = self.fixed_control_indices[idx]

        # Convert ONLY these two specific cells to dense arrays on the fly
        x_dense = self.ctrl_samples[j].toarray().squeeze()
        y_dense = self.ptb_samples[idx].toarray().squeeze()
        x = torch.tensor(x_dense, dtype=torch.float).view(-1, 1)
        y = torch.tensor(y_dense, dtype=torch.float).view(-1, 1)

        pert_name = self.ptb_names[idx]
        pert = self._pert_embedding(pert_name)

        pert_condition_idx = torch.tensor(self.pert_to_idx[pert_name], dtype=torch.long)

        return x,y,pert, pert_condition_idx

    def __len__(self):
        return self.ptb_samples.shape[0]

# TODO: adapt with shuffle option
def build_model_dataloaders_cell_split(adata, config):
    
    # Setup
    dataset_size = config['dataset_size'] #(adata.obs['target_gene'] != 'non-targeting').sum() 
    test_ratio = config['test_ratio']
    val_ratio = config['val_ratio']
    test_size = int(dataset_size * test_ratio)
    val_size = int(dataset_size * val_ratio)
    train_size = dataset_size - test_size - val_size

    pert_to_idx = config.get('pert_to_idx')

    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}

    # Create datasets
    train_dataset = PerturbationDataset(
        adata, gene_to_idx, pert_to_idx,
        start_idx=0, 
        end_idx=train_size
    )

    val_dataset = PerturbationDataset(
        adata, gene_to_idx, pert_to_idx,
        start_idx=train_size, 
        end_idx=train_size+val_size
    )

    test_dataset = PerturbationDataset(
        adata, gene_to_idx, pert_to_idx,
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

    return train_loader, val_loader, test_loader, len(train_dataset), len(test_dataset), len(val_dataset)

def build_model_dataloaders_perts_split(adata, config):
    
    test_ratio = config.get('test_ratio', 0.1)
    val_ratio = config.get('val_ratio', 0.1)
    batch_size = config.get('batch_size', 32)
    N_WORKERS = 4 

    pert_to_idx = config.get('pert_to_idx')
    
    # Isolate controls and unique perturbations
    ctrl_adata = adata[adata.obs['target_gene'] == 'non-targeting'].copy()
    unique_perts = adata.obs['target_gene'].unique().tolist()
    unique_perts.remove('non-targeting')
        
    # Shuffle perturbations to ensure random splits
    np.random.seed(42) # Optional: for reproducibility
    np.random.shuffle(unique_perts)
    
    # Calculate split sizes based on the NUMBER OF PERTURBATIONS (not cells)
    num_perts = len(unique_perts)
    test_size = int(num_perts * test_ratio)
    val_size = int(num_perts * val_ratio)
    train_size = num_perts - test_size - val_size
    
    # Slice the perturbation lists
    train_perts = unique_perts[:train_size]
    val_perts = unique_perts[train_size:train_size + val_size]
    test_perts = unique_perts[train_size + val_size:]

    #fix of control data leakage: we split control as well
    num_ctrls = ctrl_adata.shape[0]
    ctrl_indices = np.random.permutation(num_ctrls)

    test_size_c = int(num_ctrls * test_ratio)
    val_size_c = int(num_ctrls * val_ratio)
    train_size_c = num_ctrls - test_size_c - val_size_c
    
    train_ctrl = ctrl_adata[ctrl_indices[:train_size_c]]
    val_ctrl = ctrl_adata[ctrl_indices[train_size_c:train_size_c + val_size_c]]
    test_ctrl = ctrl_adata[ctrl_indices[train_size_c + val_size_c:]]
    
    # Build individual adatas (Specific Perturbations + Split Control Cells)
    train_adata = ad.concat([adata[adata.obs['target_gene'].isin(train_perts)], train_ctrl])
    val_adata = ad.concat([adata[adata.obs['target_gene'].isin(val_perts)], val_ctrl])
    test_adata = ad.concat([adata[adata.obs['target_gene'].isin(test_perts)], test_ctrl])

    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}

    # 6. Create datasets 
    # (Since we pass pre-split adatas, we don't need start_idx/end_idx anymore)
    train_dataset = PerturbationDataset(train_adata, gene_to_idx, pert_to_idx)
    val_dataset = PerturbationDataset(val_adata, gene_to_idx, pert_to_idx)
    test_dataset = PerturbationDataset(test_adata, gene_to_idx, pert_to_idx)

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_sampler=SCDATA_sampler(train_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=SCDATA_sampler(val_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_sampler=SCDATA_sampler(test_dataset, batch_size), num_workers=N_WORKERS, pin_memory=True) 

    print(f"Total Unique Perturbations: {num_perts}")
    print(f"Train set: {len(train_perts)} perts | {len(train_dataset)} cells")
    print(f"Val set: {len(val_perts)} perts | {len(val_dataset)} cells")
    print(f"Test set: {len(test_perts)} perts | {len(test_dataset)} cells")

    return train_loader, val_loader, test_loader, len(train_dataset), len(test_dataset), len(val_dataset), train_adata, val_adata, test_adata

def build_model_dataloaders_from_perts_list(adata, config, json_path):
    """
    Builds dataloaders using predefined perturbation splits from a JSON file.
    """

    N_WORKERS = 4 
    batch_size = config.get('batch_size', 32)
    pert_to_idx = config.get('pert_to_idx')
    
    # 1. Load the splits from the JSON file
    with open(json_path, 'r') as f:
        split_data = json.load(f)
        
    train_perts = split_data.get('train_labels', [])
    val_perts = split_data.get('val_labels', [])
    test_perts = split_data.get('test_labels', [])
    seed = split_data.get('seed', 42)

    num_train_perts = len(train_perts)
    num_val_perts = len(val_perts)
    num_test_perts = len(test_perts)
    total_perts = num_train_perts + num_val_perts + num_test_perts

    # 2. Isolate control cells
    ctrl_adata = adata[adata.obs['target_gene'] == 'non-targeting'].copy()
    num_ctrls = ctrl_adata.shape[0]
    
    # 3. Calculate dynamic split sizes for control cells based on JSON proportions
    train_ratio = num_train_perts / total_perts
    val_ratio = num_val_perts / total_perts
    
    train_size_c = int(num_ctrls * train_ratio)
    val_size_c = int(num_ctrls * val_ratio)
    
    # 4. Shuffle and split the control cells (preventing data leakage)
    np.random.seed(seed)
    ctrl_indices = np.random.permutation(num_ctrls)
    
    train_ctrl = ctrl_adata[ctrl_indices[:train_size_c]]
    val_ctrl = ctrl_adata[ctrl_indices[train_size_c:train_size_c + val_size_c]]
    test_ctrl = ctrl_adata[ctrl_indices[train_size_c + val_size_c:]]
    
    # 5. Build individual adatas by filtering for specific perturbations + controls
    train_adata = ad.concat([adata[adata.obs['target_gene'].isin(train_perts)], train_ctrl])
    val_adata = ad.concat([adata[adata.obs['target_gene'].isin(val_perts)], val_ctrl])
    test_adata = ad.concat([adata[adata.obs['target_gene'].isin(test_perts)], test_ctrl])

    gene_to_idx = {node: i for i, node in enumerate(adata.var_names)}

    # 6. Create datasets 
    train_dataset = PerturbationDataset(train_adata, gene_to_idx, pert_to_idx, control_sampling="random")
    val_dataset = PerturbationDataset(val_adata, gene_to_idx, pert_to_idx, control_sampling="fixed")
    test_dataset = PerturbationDataset(test_adata, gene_to_idx, pert_to_idx, control_sampling="fixed")

    # 7. Create loaders
    train_loader = DataLoader(train_dataset, batch_sampler=SCDATA_sampler(train_dataset, batch_size, shuffle=True), num_workers=N_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_sampler=SCDATA_sampler(val_dataset, batch_size, shuffle=False), num_workers=N_WORKERS, pin_memory=True) 
    test_loader = DataLoader(test_dataset, batch_sampler=SCDATA_sampler(test_dataset, batch_size, shuffle=False), num_workers=N_WORKERS, pin_memory=True) 

    print(f"Total Unique Perturbations: {total_perts}")
    print(f"Train set: {num_train_perts} perts | {len(train_dataset)} cells")
    print(f"Val set: {num_val_perts} perts | {len(val_dataset)} cells")
    print(f"Test set: {num_test_perts} perts | {len(test_dataset)} cells")

    return train_loader, val_loader, test_loader, len(train_dataset), len(test_dataset), len(val_dataset), train_adata, val_adata, test_adata
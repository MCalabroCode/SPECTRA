import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad
import wandb
import os



def _get_beta_schedule(epoch, n_epochs, warmup_epochs=5, n_cycles=1, ratio=0.5):
    '''
    beta schefuler for the VAE (beta-Vae)

    Args:
        epoch: Current epoch (1-indexed based on your train loop)
        n_epochs: Total number of epochs
        warmup_epochs: Number of initial epochs where beta remains exactly 0.0
        n_cycles: Number of annealing cycles after warmup
        ratio: Fraction of the cycle spent increasing beta
    '''
    # Warmup Phase
    if epoch <= warmup_epochs:
        return 0.0
    
    # Adjust remaining epochs for the cyclical schedule
    adjusted_epoch = epoch - warmup_epochs - 1 # 0-indexed for the math
    adjusted_n_epochs = n_epochs - warmup_epochs
    
    # Prevent division by zero if warmup is exactly n_epochs
    if adjusted_n_epochs <= 0: 
        return 1.0
        
    period = max(1, adjusted_n_epochs // n_cycles)
    step = adjusted_epoch % period
    
    # linear Annealing Phase within the cycle
    if step < period * ratio:
        return step / (period * ratio)
    else:
        return 1.0

def compute_mmd_gaussian(x, y, weights=None, kernel_mul=2.0, kernel_num=10, fix_sigma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches,
    using a multi-scale RBF kernel by averaging multiple bandwiths.
    """

    assert x.shape == y.shape, "real and predicted batches do not match in size."

    batch_size = x.size(0)
    n_samples = int(x.size(0)) + int(y.size(0))
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=x.device)

    total = torch.cat([x, y], dim=0)
    
    # RBF Kernel Calculation (Spatial)
    if weights is not None:
        # Scale features so that squared L2 distance naturally becomes WMSE
        total_scaled = total * torch.sqrt(weights.unsqueeze(0))
    else:
        total_scaled = total
    L2_distance = torch.cdist(total_scaled, total_scaled, p=2)**2
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Add epsilon to prevent bandwidth collapse if samples are identical
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples) + 1e-5
        
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_rbf = sum([torch.exp(-L2_distance / bw) for bw in bandwidth_list])
    
    # MMD Calculation
    XX = kernel_rbf[:batch_size, :batch_size]
    YY = kernel_rbf[batch_size:, batch_size:]
    XY = kernel_rbf[:batch_size, batch_size:]
    YX = kernel_rbf[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def compute_mmd_energy_distance(x, y, weights):

    # If not enough samples to compute distribution statistics
    if x.size(0) <= 1:
        return torch.tensor(0.0, device=x.device)
        
    sqrt_w = torch.sqrt(weights)
    x_w = x * sqrt_w
    y_w = y * sqrt_w
    
    # Safe pairwise Euclidean distance (eps prevents NaN gradients at d=0)
    def safe_pairwise_dist(a, b):
        diff = a.unsqueeze(1) - b.unsqueeze(0)
        return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)
        
    d_xy = safe_pairwise_dist(x_w, y_w).mean()
    d_xx = safe_pairwise_dist(x_w, x_w).mean()
    d_yy = safe_pairwise_dist(y_w, y_w).mean()
    
    # Energy distance formula
    return 2 * d_xy - d_xx - d_yy

def compute_mmd_energy_genewise(x, y, weights):
    """
    Weighted gene-wise Energy Distance.
    """
    B, G = x.shape

    assert y.shape == x.shape
    assert weights.shape == (G,)
    chunk_size = 256

    # scalar - inherits x’s dtype and device 
    loss = x.new_zeros(())

    for start in range(0,G, chunk_size):

        end = min(start + chunk_size, G)
        x_chunk = x[:, start:end]          # [B, C]
        y_chunk = y[:, start:end]          # [B, C]
        w_chunk = weights[start:end]       # [C]

        # NOTE: None in position h is equivalent to unsqueeze(1) 
        # "mean(dim=(0, 1))" returns average for every gene in C
        d_xy = torch.abs(
            x_chunk[:, None, :] - y_chunk[None, :, :]
        ).mean(dim=(0, 1))

        d_xx = torch.abs(
            x_chunk[:, None, :] - x_chunk[None, :, :]
        ).mean(dim=(0, 1))

        d_yy = torch.abs(
            y_chunk[:, None, :] - y_chunk[None, :, :]
        ).mean(dim=(0, 1))

        gene_ed = 2 * d_xy - d_xx - d_yy
        loss = loss + torch.sum(w_chunk * gene_ed)
        
    return loss
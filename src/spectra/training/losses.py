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

def compute_mmd_withcosine(x, y, weights=None, kernel_mul=2.0, kernel_num=5, fix_sigma=None, lambda_cos=0.0):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches.
    Uses a composite kernel: (1 - lambda_cos) * Multi-RBF + lambda_cos * Cosine
    """
    assert x.shape == y.shape, "real and predicted batches do not match in size."

    batch_size = x.size(0)
    n_samples = int(x.size(0)) + int(y.size(0))
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=x.device)

    total = torch.cat([x, y], dim=0)
    
    # ---------------------------------------------------------
    # 1. RBF Kernel Calculation (Spatial)
    # ---------------------------------------------------------
    if weights is not None:
        # Scale features so that squared L2 distance naturally becomes WMSE
        total_scaled = total * torch.sqrt(weights.unsqueeze(0))
    else:
        total_scaled = total
    L2_distance = torch.cdist(total_scaled, total_scaled, p=2)**2
    #L2_distance = torch.cdist(total, total, p=2)**2
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Add epsilon to prevent bandwidth collapse if samples are identical
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples)# + 1e-5
        
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_rbf = sum([torch.exp(-L2_distance / bw) for bw in bandwidth_list])
    
    # ---------------------------------------------------------
    # 2. Cosine Kernel Calculation (Angular)
    # ---------------------------------------------------------
    if lambda_cos > 0.0:
        # Normalize each sample vector to length 1
        total_norm = F.normalize(total, p=2, dim=1, eps=1e-8)
        # Pairwise cosine similarity is just the dot product of normalized vectors
        kernel_cos = torch.mm(total_norm, total_norm.t())
        
        # Scale Cosine to [0, 1] to match RBF scale (optional but stabilizes lambda)
        kernel_cos = (kernel_cos + 1.0) / 2.0
        
        # Blend the kernels
        kernel_val = (1.0 - lambda_cos) * kernel_rbf + (lambda_cos * kernel_cos)
    else:
        kernel_val = kernel_rbf

    # ---------------------------------------------------------
    # 3. MMD Calculation
    # ---------------------------------------------------------
    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]
    YX = kernel_val[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def compute_mmd(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two batches of samples x and y.
    Uses a multi-scale RBF kernel by averaging multiple bandwiths.
    """

    assert x.shape == y.shape, "real and predicted batches do not match in size."

    batch_size = x.size(0)
    n_samples = int(x.size(0)) + int(y.size(0))
    
    # If not enough samples to compute distribution statistics, return 0 or simple distance
    if batch_size <= 1:
        return torch.tensor(0.0, device=x.device)

    total = torch.cat([x, y], dim=0)
    
    # L2 Distance Matrix
    L2_distance = torch.cdist(total, total, p=2)**2
    
    # Bandwidth selection
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples) + 1e-5
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # multiple kernels calculation + averaging
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    kernel_val = sum(kernel_val)

    # MMD Calculation
    XX = kernel_val[:batch_size, :batch_size]
    YY = kernel_val[batch_size:, batch_size:]
    XY = kernel_val[:batch_size, batch_size:]
    YX = kernel_val[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def compute_weighted_energy_distance(x, y, weights):

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
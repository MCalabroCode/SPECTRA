import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import os
import uuid
import torch
import torch.nn.functional as F
from torch_geometric.nn import FAConv, DirGNNConv, GATv2Conv
from torch_geometric.utils import dropout_edge
from torch.nn import ReLU, LeakyReLU, GELU, LayerNorm
#from magnet import MagNetConv, precompute_magnet_attributes_sparse

import wandb

import anndata as ad
from val_scores import calc_auprc
import pandas as pd

from torch import Tensor

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# import random
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)


class GeneExpressionFiLM(nn.Module):
    def __init__(self, emb_dim, shift_scale=0.05):
        super().__init__()
        self.scale = nn.Linear(1, emb_dim)
        self.shift = nn.Linear(1, emb_dim)
        self.shift_scale = shift_scale
        
        # Zero-initialize so the layer starts as a pure Identity mapping
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, gene_emb, expr):

        # gene_emb: [B*N, D]
        # expr: [B*N, 1]
        
        gamma = self.scale(expr) 
        gamma = 1.0 + F.elu(gamma) # Starts at 1.0, can't go below 0, can grow infinitely
        beta = self.shift_scale * self.shift(expr)
        return gamma * gene_emb + beta

class DirectedFAGCNConv(MessagePassing):
    """
    Directed FAGCN-style convolution:

        h_i' = eps * x_res_i + 0.5 * [sum_{j in N_in(i)}  alpha_in(i,j) / d_in(i)  * x_j
                + sum_{k in N_out(i)} alpha_out(i,k) / d_out(i) * x_k]

    where:
        alpha_in(i,j)  = tanh(g_in([x_i || x_j]))
        alpha_out(i,k) = tanh(g_out([x_i || x_k]))

    Notes
    -----
    - edge_index follows PyG convention:
        edge_index[0] = source
        edge_index[1] = target
    - Incoming neighbors of i are predecessors j -> i.
    - Outgoing neighbors of i are successors i -> k.
    - The layer supports optional return of per-edge alpha values.
    """

    def __init__(self, channels: int, eps: float = 0.1, share_gates: bool = False, alpha=0.5):
        super().__init__(aggr="add", node_dim=0)

        self.channels = channels
        self.eps = eps
        self.alpha = alpha

        if share_gates:
            gate = nn.Linear(2 * channels, 1, bias=False)
            self.gate_in = gate
            self.gate_out = gate
        else:
            self.gate_in = nn.Linear(2 * channels, 1, bias=False)
            self.gate_out = nn.Linear(2 * channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate_in.weight)
        if self.gate_out is not self.gate_in:
            nn.init.xavier_uniform_(self.gate_out.weight)

    def forward(self, x: Tensor, edge_index: Tensor, x_res: Optional[Tensor] = None, 
        edge_weight: Optional[Tensor] = None, return_alpha: bool = False,) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor
            Node features of shape [N, C].
        edge_index : Tensor
            Graph connectivity in COO format with shape [2, E].
        x_res : Optional[Tensor]
            Residual/self feature term. If None, uses x.
            This is useful if you want FAGCN-style repeated injection of h^(0).
        edge_weight : Optional[Tensor]
            Optional scalar weight per edge, shape [E].
        return_alpha : bool
            If True, also returns a dict with learned alpha values.

        Returns
        -------
        out : Tensor
            Updated node features [N, C].
        alpha_dict : Optional[Dict[str, Tensor]]
            Returned only if return_alpha=True. Contains:
              - alpha_in: raw alpha for original edges, shape [E]
              - alpha_out: raw alpha for original edges, shape [E]
        """
        if x_res is None:
            x_res = x

        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Directed degrees on the ORIGINAL graph:
        deg_in = degree(dst, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)
        deg_out = degree(src, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)

        # Incoming aggregation: j -> i over original edges
        out_in = self.propagate(
            edge_index=edge_index,
            x=x,
            gate_nn=self.gate_in,
            deg_target=deg_in,
            edge_weight=edge_weight,
            size=(num_nodes, num_nodes),
        )

        # Outgoing aggregation: this is equivalent to propagating on reversed edges k -> i.
        rev_edge_index = torch.stack([dst, src], dim=0)
        rev_edge_weight = edge_weight  # same edges, only direction reversed
        out_out = self.propagate(
            edge_index=rev_edge_index,
            x=x,
            gate_nn=self.gate_out,
            deg_target=deg_out,  # because reversed target = original source
            edge_weight=rev_edge_weight,
            size=(num_nodes, num_nodes),
        )

        #out = self.eps * x_res + 0.5 * (out_in + out_out)
        out = self.eps * x_res + (1-self.alpha) * out_in + self.alpha * out_out


        if not return_alpha:
            return out

        # Compute raw learned alpha values explicitly for inspection
        alpha_in = self._compute_alpha(x=x, edge_index=edge_index, gate_nn=self.gate_in)

        # alpha_out is reported on ORIGINAL edges (i -> k), not reversed edges
        alpha_out = self._compute_alpha(x=x, edge_index=edge_index, gate_nn=self.gate_out)

        alpha_dict = {
            "alpha_in": alpha_in,    # for edge j -> i, gate uses [x_i || x_j]
            "alpha_out": alpha_out,  # for edge i -> k, gate uses [x_i || x_k]
        }

        return out, alpha_dict

    def message(self, x_i: Tensor, x_j: Tensor, index: Tensor, gate_nn: nn.Linear, deg_target: Tensor, edge_weight: Optional[Tensor],
        ) -> Tensor:
        """
        On each propagated edge j -> i:
            alpha_ij = tanh(g([x_i || x_j]))
            m_ij = alpha_ij / deg_target[i] * x_j
        """
        alpha = torch.tanh(gate_nn(torch.cat([x_i, x_j], dim=-1))).view(-1)
        norm = 1.0 / deg_target[index]
        if edge_weight is not None:
            norm = norm * edge_weight
        return (alpha * norm).unsqueeze(-1) * x_j

    @torch.no_grad()
    def _compute_alpha(self, x: Tensor, edge_index: Tensor, gate_nn: nn.Linear) -> Tensor:
        """
        Computes raw alpha for ORIGINAL directed edges.

        For an edge u -> v:
        - if gate_nn == gate_in, alpha corresponds conceptually to alpha_in(v,u)
          using [x_v || x_u]
        - if gate_nn == gate_out, alpha corresponds conceptually to alpha_out(u,v)
          using [x_u || x_v]
        """
        src, dst = edge_index[0], edge_index[1]

        if gate_nn is self.gate_in:
            x_center = x[dst]   # target i
            x_neigh = x[src]    # source j
        else:
            x_center = x[src]   # source i
            x_neigh = x[dst]    # successor k

        alpha = torch.tanh(gate_nn(torch.cat([x_center, x_neigh], dim=-1))).view(-1)
        return alpha

##########

class VariationalGraphEncoder(torch.nn.Module):
    ''' encoder class adapted for DirGNN + FAGCN '''
    def __init__(self, in_channels, out_channels, dropout_rate = 0.2):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        
        self.proj_mu = torch.nn.Linear(out_channels, out_channels)
        self.proj_logstd = torch.nn.Linear(out_channels, out_channels)

        # FAGCN Convolutions (dimensions remain constant during message passing)
        self.conv1 = DirectedFAGCNConv(out_channels)
        self.ln1 = LayerNorm(out_channels)
        
        self.conv2 = DirectedFAGCNConv(out_channels)
        self.ln2 = LayerNorm(out_channels)
        
        self.conv_mu = DirectedFAGCNConv(out_channels)
        self.conv_logstd = DirectedFAGCNConv(out_channels)

    def forward(self, x, edge_index):

        h_0 = x

        x_1 = self.conv1(h_0, edge_index, x_res=h_0) 
        x_1 = self.ln1(x_1)
        x_1 = F.gelu(x_1)

        x_2 = F.dropout(x_1, p=self.dropout_rate, training=self.training)
        x_2 = self.conv2(x_2, edge_index, x_res=h_0)
        x_2 = self.ln2(x_2)
        x_2 = F.gelu(x_2) 
        
        x_drop = F.dropout(x_2, p=self.dropout_rate, training=self.training)
        
        # Calculate mu and logstd
        mu = self.conv_mu(x_drop, edge_index, x_res=h_0)
        mu = self.proj_mu(mu)  # Final projection to out_channels
        
        logst = self.conv_logstd(x_drop, edge_index, x_res=h_0)
        logst = self.proj_logstd(logst) # Final projection to out_channels
        
        return mu, logst

class MLP(torch.nn.Module):
    '''
    MLP auxiliary class
    '''
    def __init__(self, sizes, batch_norm=True, dropout=0.2):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Dropout(p=dropout),
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class FeatureDecoder(torch.nn.Module):
    ''' Feature decoder adapted for DirGNN + FAGCN '''
    def __init__(self, n_channels, num_node_features, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        # FAGCN convolutions
        self.conv1 = DirectedFAGCNConv(n_channels, alpha=0.7)
        self.ln1 = LayerNorm(n_channels)
        
        self.conv2 = DirectedFAGCNConv(n_channels, alpha=0.7)
        self.ln2 = LayerNorm(n_channels)
        
        # self.conv3 = DirectedFAGCNConv(n_channels)
        # self.ln3 = LayerNorm(n_channels)

        self.last_layer = torch.nn.Linear(n_channels, num_node_features) 

    def forward(self, z, edge_index, return_alpha=False):
        # In the decoder, the latent representation 'z' acts as the raw feature (x_0)
        x_0 = z 

        if return_alpha:
            h1, alpha_dict_1 = self.conv1(z, edge_index, x_res=x_0, return_alpha=True)
        else:
            h1 = self.conv1(z, edge_index, x_res=x_0)
        h1 = self.ln1(h1)
        h1_out = F.gelu(h1)

        h1_drop = F.dropout(h1_out, p=self.dropout_rate, training=self.training)

        if return_alpha:
            h2, alpha_dict_2 = self.conv2(h1_drop, edge_index, x_res=x_0, return_alpha=True)
        else:
            h2 = self.conv2(h1_drop, edge_index, x_res=x_0)
        h2 = self.conv2(h1_drop, edge_index, x_res=x_0)
        h2 = self.ln2(h2)
        h2_out = F.gelu(h2)

        # h2_drop = F.dropout(h2_out, p=self.dropout_rate, training=self.training)
        # h3 = self.conv3(h2_drop, edge_index, x_res=x_0)
        # h3 = self.ln3(h3)
        # h3_out = F.gelu(h3)

        # Output Head
        out = self.last_layer(h2_out) 
        
        # LeakyReLU during training to prevent dead gradients, hard ReLU for biological realism at eval
        if self.training:
            out = F.leaky_relu(out, negative_slope=0.05) 
        else:
            out = F.relu(out)
        
        if return_alpha:
            return out, (alpha_dict_1, alpha_dict_2)
        
        return out


class PerturbModel(torch.nn.Module):
    '''
    SPECTRA model class
    '''
    def __init__(self, edge_index, num_nodes, device, config, gene_embeddings, gene_weights=None):
        super().__init__()
        self.device = device 

        self.num_nodes = num_nodes 
        self.config = config
        self.n_channels = config['n_channels']
        self.dropout_p = config['dropout_p']
        self.architecture_name = config['architecture']
        #self.num_node_features = config['num_node_features'] 
        
        self.register_buffer('edge_index', edge_index)

        # # Weight Lookup Construction for WMSE
        # default_weights = (1/num_nodes)*torch.ones(num_nodes)
        # weight_lookup = default_weights.unsqueeze(0).repeat(num_nodes + 1, 1).to(device)    
        # if gene_weights is not None:
        #     for pert_idx, weight_array in gene_weights.items():
        #         w_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)
        #         if 0 <= pert_idx < num_nodes:
        #             weight_lookup[pert_idx] = w_tensor
        # self.register_buffer('weight_lookup', weight_lookup)

        num_conditions = len(config['pert_to_idx'])
        default_weights = (1.0 / num_nodes) * torch.ones(num_nodes) # Default flat weights if no DEGs provided
        weight_lookup = default_weights.unsqueeze(0).repeat(num_conditions, 1).to(device)    
        if gene_weights is not None:
            for pert_name, weight_array in gene_weights.items(): # gene_weights has string keys like 'GeneA+GeneB'
                if pert_name in config['pert_to_idx']:
                    idx = config['pert_to_idx'][pert_name]
                    w_tensor = torch.tensor(weight_array, dtype=torch.float32, device=device)
                    weight_lookup[idx] = w_tensor
        self.register_buffer('weight_lookup', weight_lookup)
        
        self.gene_embeddings = torch.nn.Embedding.from_pretrained(gene_embeddings, freeze=True)
        scgpt_dim = gene_embeddings.shape[1]

        self.project_gene = torch.nn.Linear(scgpt_dim, self.n_channels)
        self.project_expr = torch.nn.Linear(1, self.n_channels)
        self.film_layer = GeneExpressionFiLM(self.n_channels, shift_scale=0.05)

        # Optional: A LayerNorm to stabilize the addition
        self.add_norm = LayerNorm(self.n_channels)

        #self.encoder_in_channels = 1

        # Learnable KO perturbation Token
        # #self.ko_token = torch.nn.Parameter(torch.randn(1, n_channels) - 2.0)
        # self.ko_mu = torch.nn.Embedding(num_nodes, n_channels)
        # self.ko_sigma = torch.nn.Embedding(num_nodes, n_channels)
        # torch.nn.init.normal_(self.ko_mu.weight, mean=-2.0, std=0.5) # Match your original prior for the mean shift (-2.0)
        # torch.nn.init.zeros_(self.ko_sigma.weight) # Initialize variance scale weights to 0 so that exp(0) = 1.0 (identity scale)

        # self.ko_token = torch.nn.Embedding(1,n_channels)
        # torch.nn.init.xavier_uniform_(self.ko_token.weight)

        self.ko_mu = torch.nn.Embedding(num_nodes, 64)
        self.ko_mlp = MLP([64, self.n_channels, self.n_channels])

        self.encoder = VariationalGraphEncoder(self.n_channels, self.n_channels, self.dropout_p)
        self.gex_decoder = FeatureDecoder(self.n_channels, 1, self.dropout_p)
        
        self._cached_batch_size = 0
        self._cached_edge_index = None
        self._cached_gene_ids = None

        num_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print('========================================')
        print(f'number of trainable parameters: {num_trainable_params}')
        print('========================================')
    
    def _get_batched_edge_index(self, batch_size):
        '''
        batched edge lists - edges of the DAG are always the same for every sample
        '''
        if batch_size == self._cached_batch_size and self._cached_edge_index is not None:
            return self._cached_edge_index
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes
        edge_index_batch = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
        edge_index_batch = edge_index_batch.reshape(2, -1)
        self._cached_batch_size = batch_size
        self._cached_edge_index = edge_index_batch
        return edge_index_batch

    def _get_batched_gene_ids(self, batch_size):
        '''
        repeats the genes ids exactly batch_size times (e.g. [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])
        '''
        if self._cached_gene_ids is not None and len(self._cached_gene_ids) == batch_size * self.num_nodes:
            return self._cached_gene_ids
        ids = torch.arange(self.num_nodes, device=self.device)
        ids = ids.repeat(batch_size) 
        self._cached_gene_ids = ids
        return ids

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu, logstd, threshold=1e-2, verbose=True, free_bits=0.05):

        kl_raw = -0.5 * (1 + 2 * logstd - mu**2 - logstd.exp()**2)
        kl_per_dim = torch.mean(kl_raw, dim=0)

        # print("kl_raw.mean(), kl_raw.std():", kl_raw.mean().item(), kl_raw.std().item())
        # print("kl_per_dim (before clamp):", kl_per_dim.detach().cpu().numpy())
        # print("free_bits:", free_bits)

        #kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        return torch.mean(kl_per_dim)

    def forward(self, data, return_attention_weights=False):

        x, pert = data
        x = x.to(self.device) #[B,N,1]
        pert = pert.to(self.device) #[B,N]

        batch_size, num_nodes, num_features = x.shape

        edge_index_batch = self._get_batched_edge_index(batch_size)
        #x = x.reshape(batch_size * num_nodes, num_features) #[BxN,1] 
        pert = pert.reshape(batch_size * num_nodes) #[BxN]

        # --- NEW PROJECT & ADD LOGIC ---
        gene_ids = self._get_batched_gene_ids(batch_size) 
        scgpt_base = self.gene_embeddings(gene_ids) 
        x_flat = x.reshape(batch_size * num_nodes, 1) 

        #expr_proj = self.project_expr(x_flat)       # [B * N, n_channels]
        gene_proj = self.project_gene(scgpt_base)   # [B * N, n_channels]
        x_node_features = self.film_layer(gene_proj, x_flat)
        #h_node = expr_proj + gene_proj
        #h_node = self.add_norm(h_node)              # [B * N, n_channels]
        #h_node = F.gelu(h_node)                     # Give it a non-linearity

        mu, logstd = self.encoder(x_node_features, edge_index_batch) # both [BxN, C] (C = hidden channels dimension)
        logstd = torch.clamp(logstd, min=-20, max=10) # this is to avoid inf values

        self.last_mu = mu          
        self.last_logstd = logstd  

        z_ctrl = self.reparametrize(mu, logstd)     
        self.last_z = z_ctrl

        x_hat = self.gex_decoder(z_ctrl, edge_index_batch)
        
        # additive logic for perturbation encoding
        mask = pert.unsqueeze(1).to(z_ctrl.dtype) #[BxN,1]

        #z = z + (mask * self.ko_token)

        #NOTE:learnable generic gene-specific function with variance scaling
        # pert_mask = pert.bool()
        # delta_mu = torch.zeros_like(mu)
        # delta_logstd = torch.zeros_like(logstd)
        
        # if pert_mask.any():
        #     gene_ids = self._get_batched_gene_ids(batch_size)
        #     perturbed_gene_ids = gene_ids[pert_mask]
            
        #     # Lookup shifts (No torch.exp needed for logstd addition!)
        #     mu_shift = self.ko_mu(perturbed_gene_ids)
        #     logstd_shift = self.ko_sigma(perturbed_gene_ids) 
            
        #     delta_mu[pert_mask] = mu_shift
        #     delta_logstd[pert_mask] = logstd_shift

        pert_mask = pert.bool()
        delta_mu = torch.zeros_like(mu)
        if pert_mask.any():
            gene_ids = self._get_batched_gene_ids(batch_size)
            perturbed_gene_ids = gene_ids[pert_mask]
            ko_embeddings = self.ko_mu(perturbed_gene_ids)
            mu_shift = self.ko_mlp(ko_embeddings)
            delta_mu[pert_mask] = mu_shift

            # # --- MAGNITUDE CHECK START ---
            # with torch.no_grad(): # Disable gradients for logging to save memory
            #     # 1. Extract the specific embeddings for the perturbed genes
            #     mu_before = mu[pert_mask]
            #     mu_after = mu_before + mu_shift

            #     # 2. Calculate L2 norm (magnitude) along the feature dimension (dim=1)
            #     mag_before = torch.norm(mu_before, p=2, dim=1)
            #     mag_after = torch.norm(mu_after, p=2, dim=1)
            #     shift_mag = torch.norm(mu_shift, p=2, dim=1)

            #     # 3. Print the average and median magnitudes
            #     print("\n--- Perturbation Magnitudes ---")
            #     print(f"Mean Magnitude BEFORE: {mag_before.mean().item():.4f}")
            #     print(f"Mean Magnitude AFTER:  {mag_after.mean().item():.4f}")
            #     print(f"Mean Magnitude SHIFT:  {shift_mag.mean().item():.4f} (Median: {shift_mag.median().item():.4f})")
            #     print("-------------------------------\n")
            # # --- MAGNITUDE CHECK END ---

        mu_pert = mu + delta_mu
        logstd_pert = logstd #+ delta_logstd
        z_pert = self.reparametrize(mu_pert, logstd_pert)

        if return_attention_weights:
            y_hat, (alpha_dict_1, alpha_dict_2) = self.gex_decoder(z_pert, edge_index_batch, return_alpha=True)

            # Grab the incoming gating weights for Layer 1 and 2
            alpha_L1_batched = alpha_dict_1['alpha_in'] # Shape: [B * E]
            alpha_L2_batched = alpha_dict_2['alpha_in'] # Shape: [B * E]

            E = self.edge_index.shape[1]

            # 4. MEMORY-EFFICIENT BATCH AVERAGING
            alpha_L1_average = alpha_L1_batched.reshape(batch_size, E).mean(dim=0)
            alpha_L2_average = alpha_L2_batched.reshape(batch_size, E).mean(dim=0)

            # 5. BUILD MATRICES AND MULTIPLY
            N = self.num_nodes
            
            # Using self.edge_index (which is safely on self.device)
            A_L1 = torch.sparse_coo_tensor(self.edge_index, alpha_L1_average, (N, N)).to_dense()
            A_L2 = torch.sparse_coo_tensor(self.edge_index, alpha_L2_average, (N, N)).to_dense()

            # The Aggregate Attention Graph
            A_Agg = torch.matmul(A_L2, A_L1)
            
            return A_Agg, y_hat, x_hat

        else:
            y_hat = self.gex_decoder(z_pert, edge_index_batch)
            return y_hat, x_hat

    @torch.no_grad()
    def get_attention_graph(self, data):
        """
        Computes the global, multi-hop Attention Graph for a given batch.
        Assumes the batch contains cells of a SINGLE perturbation type.
        """
        self.eval() # Ensure we are in eval mode (no dropout)
        
        x, pert = data
        x = x.to(self.device) 
        pert = pert.to(self.device) 

        batch_size, num_nodes, num_features = x.shape
        edge_index_batch = self._get_batched_edge_index(batch_size)
        pert = pert.reshape(batch_size * num_nodes) 

        # 1. ENCODER PASS
        gene_ids = self._get_batched_gene_ids(batch_size) 
        scgpt_base = self.gene_embeddings(gene_ids) 
        x_flat = x.reshape(batch_size * num_nodes, 1) 

        gene_proj = self.project_gene(scgpt_base)   
        x_node_features = self.film_layer(gene_proj, x_flat)

        mu, logstd = self.encoder(x_node_features, edge_index_batch) 
        logstd = torch.clamp(logstd, min=-20, max=10) 

        # 2. PERTURBATION INJECTION
        pert_mask = pert.bool()
        delta_mu = torch.zeros_like(mu)
        if pert_mask.any():
            perturbed_gene_ids = gene_ids[pert_mask]
            ko_embeddings = self.ko_mu(perturbed_gene_ids)
            mu_shift = self.ko_mlp(ko_embeddings)
            delta_mu[pert_mask] = mu_shift

        mu_pert = mu + delta_mu
        z_pert = self.reparametrize(mu_pert, logstd)
        
        # 3. DECODER PASS (Extracting Alphas)
        _, (alpha_dict_1, alpha_dict_2) = self.gex_decoder(z_pert, edge_index_batch, return_alpha=True)
        
        # Grab the incoming gating weights for Layer 1 and 2
        alpha_L1_batched = alpha_dict_1['alpha_in'] # Shape: [B * E]
        alpha_L2_batched = alpha_dict_2['alpha_in'] # Shape: [B * E]

        E = self.edge_index.shape[1]

        # 4. MEMORY-EFFICIENT BATCH AVERAGING
        alpha_L1_average = alpha_L1_batched.reshape(batch_size, E).mean(dim=0)
        alpha_L2_average = alpha_L2_batched.reshape(batch_size, E).mean(dim=0)

        # 5. BUILD MATRICES AND MULTIPLY
        N = self.num_nodes
        
        # Using self.edge_index (which is safely on self.device)
        A_L1 = torch.sparse_coo_tensor(self.edge_index, alpha_L1_average, (N, N)).to_dense()
        A_L2 = torch.sparse_coo_tensor(self.edge_index, alpha_L2_average, (N, N)).to_dense()

        # The Aggregate Attention Graph
        A_Agg = torch.matmul(A_L2, A_L1)
        
        return A_Agg, A_L1, A_L2
    
    def predict_full_expression(self, data):
        self.eval()
        return self.forward(data)[0]

######### training + testing routines ##########

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
    
    if adjusted_n_epochs <= 0: # Prevent division by zero if warmup is exactly n_epochs
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

def train_step_perturb_model(model, data, device, alpha=1., beta=1., mmd_gamma=0.0):
    x, y, pert, pert_idx = data  # x,y: [B,N,1], pert: [B,N]
    x, y, pert = x.to(device), y.to(device), pert.to(device)

    B, N, _ = x.shape

    y_hat, x_hat = model((x,pert)) #[BxN,1]

    # flatten ground truth to match forward outputs for recon
    x_flat = x.reshape(B * N, 1)
    y_flat = y.reshape(B * N, 1)

    # control cells (x_hat): ELBO loss
    squared_error = (x_hat - x_flat) ** 2 # [B*N,1]
    loss_feat = squared_error.view(B, N).mean(dim=1).mean()  # sum over genes, mean over cells

    kl_div = model.kl_loss(model.last_mu, model.last_logstd)

    # Cosine similarity (Direction)
    x_true = x_flat.view(B, N)
    y_true = y_flat.view(B, N)
    x_pred = x_hat.view(B, N)
    y_pred = y_hat.view(B, N)

    # ---- NEW: Extract perturbation weights for this batch ----
    # pert_idx = pert[0].int().argmax().item()
    # batch_weights = model.weight_lookup[pert_idx].to(device).reshape(-1)
    batch_pert_idx = pert_idx[0].item() # Guaranteed identical across the batch!
    batch_weights = model.weight_lookup[batch_pert_idx]

    # Compute pseudobulk - used for proper cosine similarity
    x_true_mean = x_true.mean(dim=0) # [N]
    y_true_mean = y_true.mean(dim=0) # [N]
    x_pred_mean = x_pred.mean(dim=0) # [N]
    y_pred_mean = y_pred.mean(dim=0) # [N]

    delta_true_mean = y_true_mean - x_true_mean
    delta_pred_mean = y_pred_mean - x_pred_mean

    true_norm = torch.norm(delta_true_mean)
    if true_norm > 1e-6:
        cos_sim = F.cosine_similarity(delta_pred_mean, delta_true_mean, dim=0)
        loss_cosine = 1.0 - cos_sim
    else:
        loss_cosine = torch.tensor(0.0, device=device)

    # control loss: ELBO + cosine
    control_loss = alpha * loss_feat + beta * kl_div

    # mmd for perturbed cells ancd control cells
    loss_mmd_x = torch.tensor(0.0, device=device)
    loss_mmd_y = compute_mmd_withcosine(y_true, y_pred, weights=batch_weights)

    if mmd_gamma != 0.0:
        loss_mmd_x = compute_mmd(x_true, x_pred)
    else:
        loss_mmd_x = 0.0

    total_loss = control_loss + loss_mmd_y + mmd_gamma * loss_mmd_x + loss_cosine

    return total_loss, loss_mmd_y, loss_mmd_x, kl_div, loss_cosine, loss_feat

@torch.no_grad()
def test_perturb_model(model, loader, device):
    model.eval()
    pert_mmd = []
    pert_wmse = []
    for i, data in enumerate(tqdm(loader, desc='testing...')):
        x, y, pert, pert_idx = data
        x, y, pert = x.to(device), y.to(device), pert.to(device)

        B = pert.shape[0]
        N = pert.shape[1]

        y_hat, _ = model((x,pert)) #[B*N,1]
        y_flat = y.reshape(-1, 1) #[B*N,1]

        # # weights
        # is_control = pert.sum(dim=1) == 0
        # pert_indices = pert.int().argmax(dim=1)
        # weight_lookup_indices = torch.where(
        #     is_control, 
        #     torch.tensor(model.num_nodes, device=device), 
        #     pert_indices
        # )
        # specific_weights = model.weight_lookup[weight_lookup_indices] #shape: [B, N]
        batch_pert_idx = pert_idx[0].item()
        specific_weights = model.weight_lookup[batch_pert_idx].view(1, N)

        squared_error = (y_hat - y_flat)**2 # Shape: [B*N, 1]
        
        # Reshape to [B, N] to match weights
        squared_error_batch = squared_error.reshape(B, N)
        #weights_batch = specific_weights.reshape(B, N)
        
        # Calculate Weighted MSE
        # PyTorch automatically broadcasts the [1, N] weights across the [B, N] error batch
        loss_per_cell = torch.sum(squared_error_batch * specific_weights, dim=1)
        error = torch.mean(loss_per_cell)
        pert_wmse.append(error.item())

        # mmd
        mmd_error = compute_mmd(y_flat.view(B,N), y_hat.view(B,N))
        pert_mmd.append(mmd_error.item())

    avg_pert_mmd = sum(pert_mmd)/len(pert_mmd)
    avg_pert_wmse = sum(pert_wmse)/len(pert_wmse)
    return avg_pert_mmd, avg_pert_wmse

@torch.no_grad()
def test_perturb_model_new(model, loader, device, var_names, idx_to_gene):
    model.eval()
    
    pred_expr_list = []
    true_expr_list = []
    control_expr_list = []
    obs_gene_list = []
    
    # 1. Accumulate all predictions and ground truths
    for i, data in enumerate(tqdm(loader, desc='testing with AUPRC...')):
        x, y, pert, pert_idx = data
        x, y, pert = x.to(device), y.to(device), pert.to(device)

        B = pert.shape[0]
        N = pert.shape[1]

        y_hat, _ = model((x, pert)) 

        # Store arrays (moving them off GPU immediately to save VRAM)
        pred_expr_list.append(y_hat.view(B, N).cpu().numpy())
        true_expr_list.append(y.view(B, N).cpu().numpy())
        
        # We must also save the input 'x' cells. calc_auprc needs control 
        control_expr_list.append(x.view(B, N).cpu().numpy())

        # Decode the boolean perturbation tensor back to strings
        for b in range(B):
            pert_indices = torch.where(pert[b])[0]
            if len(pert_indices) == 0:
                obs_gene_list.append('non-targeting')
            else:
                # Reconstruct combinatorial names if needed (e.g., 'A+B')
                pert_name = "+".join([idx_to_gene[idx.item()] for idx in pert_indices])
                obs_gene_list.append(pert_name)

    # 2. Stack everything into dense numpy matrices
    X_pred = np.vstack(pred_expr_list)
    X_true = np.vstack(true_expr_list)
    X_ctrl = np.vstack(control_expr_list)

    # 3. Build DataFrames for AnnData construction
    obs_pert = pd.DataFrame({"target_gene": obs_gene_list})
    obs_ctrl = pd.DataFrame({"target_gene": ['non-targeting'] * X_ctrl.shape[0]})
    var = pd.DataFrame(index=var_names)

    # 4. Construct the AnnData objects
    adata_pred = ad.AnnData(X=X_pred, obs=obs_pert.copy(), var=var)
    adata_true = ad.AnnData(X=X_true, obs=obs_pert.copy(), var=var)
    adata_ctrl = ad.AnnData(X=X_ctrl, obs=obs_ctrl, var=var)

    # Concat the control cells into both datasets so the metric can compute DEGs
    adata_pred_full = ad.concat([adata_pred, adata_ctrl])
    adata_true_full = ad.concat([adata_true, adata_ctrl])

    # 5. Calculate the final dataset-wide metric!
    # (Assuming calc_auprc returns a single float score)
    auprc_score = calc_auprc(adata_true_full, adata_pred_full, pert_col='target_gene', control_name='non-targeting')
    average = sum(auprc_score.values()) / len(auprc_score)

    return average

def train(model, 
    train_loader, 
    test_loader, 
    lr, 
    n_epochs, 
    device, 
    wandb_support, 
    var_names, 
    idx_to_gene, 
    alpha_weight, 
    beta_weight, 
    patience=5, 
    metric_mode='min'
):
    """
    metric_mode: Set to 'max' if your validation metric is AUPRC/Accuracy (higher is better). 
                 Set to 'min' if your validation metric is MMD/MSE/Loss (lower is better).
    """

    torch.autograd.set_detect_anomaly(True)

    global_loss = []
    mmd_pert = []
    mmd_ctrl = []

    test_wmse = []
    test_mmd = []

    first_batch = next(iter(train_loader))
    batch_size = first_batch[0].shape[0]

    accumulation_steps = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, fused=True, weight_decay=0.001)

    weights_dir = 'weights'
    os.makedirs(weights_dir, exist_ok=True)

    # Early Stopping Setup
    best_val_metric = float('-inf') if metric_mode == 'max' else float('inf')
    patience_counter = 0
    if wandb_support and wandb.run is not None:
        best_filepath = os.path.join(weights_dir,f"best_model_{model.architecture_name}_{wandb.run.id}.pth")
    else:
        best_filepath = os.path.join(weights_dir, f"best_model_{model.architecture_name}_{uuid.uuid4().hex}.pth")

    # wandb watch
    if wandb_support:
        # Initialize wandb run
        # wandb.init(
        #     project="SPECTRA",       # The name of your project in wandb
        #     name=f"{model.config['architecture']}",   # (Optional) Name of this specific run
        #     config=model.config               # Pass your dictionary here!
        # )
        wandb.watch(model, log="all", log_freq=10) # log="all" tracks both gradients and parameters

    for epoch in range(1, n_epochs + 1):
        model.train()

        total_loss = 0
        total_mmd_pert = 0
        total_mmd_ctrl = 0
        kl_accum = 0
        mse_accum = 0
        cos_accum = 0

        optimizer.zero_grad(set_to_none=True)
        for (i,batch) in enumerate(tqdm(train_loader, desc=f'training at epoch {epoch}')):
            loss, loss_mmd_y, loss_mmd_x, kl_div, loss_cosine, loss_feat = train_step_perturb_model(
                model, 
                batch, 
                model.device, 
                alpha=alpha_weight, 
                beta=beta_weight)
            
            loss.backward()

            # sum, we will calculate the mean over all the epoch
            total_loss += loss.item()
            total_mmd_pert += loss_mmd_y.item()
            total_mmd_ctrl += loss_mmd_x#.item()
            kl_accum += kl_div.item()
            mse_accum += loss_feat.item()
            cos_accum += loss_cosine.item()
            
            # gradient accumulation
            if (i+1)%accumulation_steps==0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        # calculate averages for the epoch
        epoch_loss = total_loss / len(train_loader)
        epoch_mmd_pert = total_mmd_pert / len(train_loader)
        epoch_mmd_ctrl = total_mmd_ctrl / len(train_loader)
        epoch_kl = kl_accum / len(train_loader)
        epoch_mse = mse_accum / len(train_loader)
        epoch_cos = cos_accum / len(train_loader)

        # Append to lists
        global_loss.append(epoch_loss)
        mmd_pert.append(epoch_mmd_pert)
        mmd_ctrl.append(epoch_mmd_ctrl)

        # print training methods
        print(f'training KL = {epoch_kl:.5f} | mse = {epoch_mse:.3f} | cos = {epoch_cos:.3f}')

        # save current epoch weights
        # filepath = os.path.join(weights_dir, f"weights__model-{model.architecture_name}_epoch-{epoch}__nodes-{model.num_nodes}_b-{batch_size}_h-{model.n_channels}_p-{model.dropout_p}.pth")
        # torch.save(model.state_dict(), filepath)
        
        # Validation & Early Stopping
        if epoch!=0:

            # We assume this returns your primary metric (MMD)
            avg_pert_mmd, avg_pert_wmse = test_perturb_model(model, test_loader, device)
            test_wmse.append(avg_pert_wmse)
            test_mmd.append(avg_pert_mmd)

            # Check if this is the best model so far
            is_best = (avg_pert_wmse > best_val_metric) if metric_mode == 'max' else (avg_pert_wmse < best_val_metric)

            if is_best:
                print(f"Validation metric improved to {avg_pert_wmse:.4f}. Saving best model...")
                best_val_metric = avg_pert_wmse
                patience_counter = 0
                torch.save(model.state_dict(), best_filepath)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            # W&B Logging
            if wandb_support:
                wandb.log({
                    "epoch": epoch,
                    "train/global_loss": epoch_loss,
                    "train/mmd_pert": epoch_mmd_pert,
                    "train/mmd_ctrl": epoch_mmd_ctrl,
                    "train/kl_divergence": epoch_kl,
                    "train/mse": epoch_mse,
                    "train/cosine_loss": epoch_cos,
                    "val/test_WMSE": avg_pert_wmse, # Renamed slightly for clarity
                    "val/test_MMD": avg_pert_mmd
                })

            # Trigger Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} epochs.")
                break
    
    # ==========================================
    # FINAL EVALUATION PHASE
    # ==========================================
    torch.autograd.set_detect_anomaly(False)
    print("\n--- Training concluded. Initiating Final Evaluation ---")
    
    # Reload the absolute best weights before running the final metric
    if os.path.exists(best_filepath):
        print("Reloading best epoch weights for final evaluation...")
        model.load_state_dict(torch.load(best_filepath))
    
    # Calculate the metric strictly ONCE
    final_metric_val = test_perturb_model_new(model, test_loader, model.device, var_names, idx_to_gene)
    print(f"FINAL TEST METRIC: {final_metric_val:.4f}")
    
    if wandb_support:
        wandb.log({"val/AUPRC": final_metric_val})
    
    return mmd_pert, test_mmd, test_wmse
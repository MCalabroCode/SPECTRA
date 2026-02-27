import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch.nn import LayerNorm
import math

class MagNetConv(torch.nn.Module):
    """
    MagNet Convolutional Layer.
    Processes directed graphs by encoding the edge directionality into the imaginary 
    phase of a complex-valued Magnetic Laplacian.
    """
    def __init__(self, in_channels, out_channels, K=2, q=0.25):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K  # Order of the Chebyshev polynomial
        self.q = q  # Magnetic charge parameter (controls sensitivity to directionality)
        
        # Learnable weights for both Real and Imaginary transformations
        self.weight_real = torch.nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.weight_imag = torch.nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.bias_real = torch.nn.Parameter(torch.Tensor(out_channels))
        self.bias_imag = torch.nn.Parameter(torch.Tensor(out_channels))
            
        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights using uniform distribution scaled by layer size."""
        stdv = 1. / math.sqrt(self.in_channels * (self.K + 1))
        self.weight_real.data.uniform_(-stdv, stdv)
        self.weight_imag.data.uniform_(-stdv, stdv)
        self.bias_real.data.uniform_(-stdv, stdv)
        self.bias_imag.data.uniform_(-stdv, stdv)

    def get_magnetic_laplacian_sparse(self, edge_index, num_nodes):
        """
        Dynamically computes the sparse Magnetic Laplacian matrix (L = L_real + i * L_imag).
        Uses PyG's coalesce to safely align symmetrized edges (A + A^T) with directional differences (A - A^T).
        """
        device = edge_index.device
        row, col = edge_index[0], edge_index[1] # both of size [E]
        edge_weight = torch.ones(row.size(0), device=device) # adjacency weights (here all ones; in general you could accept edge_weight)

        # Symmetrized A_sym = 0.5 * (A + A^T)
        edge_index_sym, edge_weight_sym = pyg_utils.to_undirected(
            edge_index, edge_weight, num_nodes=num_nodes, reduce='add'
        )
        edge_weight_sym = edge_weight_sym * 0.5
        
        # Degree Matrix D
        deg = pyg_utils.degree(edge_index_sym[0], num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0) # replaces any inf value with 0 (happens with nodes having degree 0)
        
        # Normalized A_sym, i.e. D^{-1/2} * A_sym * D^{-1/2} - see formula 1 of the paper
        edge_weight_sym_norm = deg_inv_sqrt[edge_index_sym[0]] * edge_weight_sym * deg_inv_sqrt[edge_index_sym[1]]
        
        # Phase Theta = 2 * pi * q * (A - A^T)
        row_diff = torch.cat([row, col]) #[2E] - cat joins along the existing dim
        col_diff = torch.cat([col, row]) #[2E]
        val_diff = torch.cat([torch.ones(row.size(0), device=device), -torch.ones(row.size(0), device=device)])
        
        # Compute antisymmetric s_ij = (A - A^T)
        # for each directed edge i->j: +1, and for reverse j->i: -1
        edge_index_diff, val_diff = pyg_utils.coalesce(
            torch.stack([row_diff, col_diff]), #torch stacks along a new dim=0
            edge_attr=val_diff, 
            num_nodes=num_nodes, 
            reduce='add'
        )
        
        theta = 2 * math.pi * self.q * val_diff
        
        # 5. Scaled Laplacian L = -exp(i * Theta) * A_sym
        L_real_val = -torch.cos(theta) * edge_weight_sym_norm
        L_imag_val = -torch.sin(theta) * edge_weight_sym_norm
        
        L_real = torch.sparse_coo_tensor(edge_index_sym, L_real_val, (num_nodes, num_nodes)).coalesce()
        L_imag = torch.sparse_coo_tensor(edge_index_sym, L_imag_val, (num_nodes, num_nodes)).coalesce()
        
        return L_real, L_imag

    def forward(self, x_real, x_imag, edge_index):
        """
        Applies the complex-valued Chebyshev polynomial filters to the node features.
        """
        num_nodes = x_real.size(0)
        L_real, L_imag = self.get_magnetic_laplacian_sparse(edge_index, num_nodes)
        
        out_real = torch.zeros(num_nodes, self.out_channels, device=x_real.device)
        out_imag = torch.zeros(num_nodes, self.out_channels, device=x_real.device)
        
        # T_0(L) X = X
        Tx_0_real, Tx_0_imag = x_real, x_imag
        out_real += torch.matmul(Tx_0_real, self.weight_real[0]) - torch.matmul(Tx_0_imag, self.weight_imag[0])
        out_imag += torch.matmul(Tx_0_real, self.weight_imag[0]) + torch.matmul(Tx_0_imag, self.weight_real[0])
        
        if self.K > 0:
            # T_1(L) X = L X
            Tx_1_real = torch.sparse.mm(L_real, Tx_0_real) - torch.sparse.mm(L_imag, Tx_0_imag)
            Tx_1_imag = torch.sparse.mm(L_real, Tx_0_imag) + torch.sparse.mm(L_imag, Tx_0_real)
            
            out_real += torch.matmul(Tx_1_real, self.weight_real[1]) - torch.matmul(Tx_1_imag, self.weight_imag[1])
            out_imag += torch.matmul(Tx_1_real, self.weight_imag[1]) + torch.matmul(Tx_1_imag, self.weight_real[1])
            
        for k in range(2, self.K + 1):
            # T_k(L) X = 2 L T_{k-1}(L) X - T_{k-2}(L) X
            L_Tx_1_real = torch.sparse.mm(L_real, Tx_1_real) - torch.sparse.mm(L_imag, Tx_1_imag)
            L_Tx_1_imag = torch.sparse.mm(L_real, Tx_1_imag) + torch.sparse.mm(L_imag, Tx_1_real)
            
            Tx_2_real = 2 * L_Tx_1_real - Tx_0_real
            Tx_2_imag = 2 * L_Tx_1_imag - Tx_0_imag
            
            out_real += torch.matmul(Tx_2_real, self.weight_real[k]) - torch.matmul(Tx_2_imag, self.weight_imag[k])
            out_imag += torch.matmul(Tx_2_real, self.weight_imag[k]) + torch.matmul(Tx_2_imag, self.weight_real[k])
            
            Tx_0_real, Tx_0_imag = Tx_1_real, Tx_1_imag
            Tx_1_real, Tx_1_imag = Tx_2_real, Tx_2_imag
            
        return out_real + self.bias_real, out_imag + self.bias_imag


class VariationalGraphEncoder(torch.nn.Module):
    """
    Encodes basal cell gene expression profiles into a low-dimensional variational latent space.
    Uses MagNet to rigorously respect the directed regulatory flow (A -> B != B -> A) of the GRN.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.2, q=0.25, K=2):
        super().__init__()
        self.conv1 = MagNetConv(in_channels, out_channels, K=K, q=q)
        self.ln1_real = LayerNorm(out_channels)
        self.ln1_imag = LayerNorm(out_channels)
        
        self.conv2 = MagNetConv(out_channels, 2*out_channels, K=K, q=q)
        self.ln2_real = LayerNorm(2*out_channels)
        self.ln2_imag = LayerNorm(2*out_channels)
        
        # Maps the concatenated complex embeddings down to Real-valued VAE parameters
        self.conv_mu = torch.nn.Linear(4*out_channels, out_channels)
        self.conv_logstd = torch.nn.Linear(4*out_channels, out_channels)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        """
        Passes real-valued gene expressions into the complex-valued MagNet.
        Returns the means (mu) and log standard deviations (logstd) for the VAE latent distributions.
        """
        x_real = F.dropout(x, p=self.dropout_rate, training=self.training)
        x_imag = torch.zeros_like(x_real) # Raw gene expression has no imaginary component
        
        x_real, x_imag = self.conv1(x_real, x_imag, edge_index)
        x_real = F.gelu(self.ln1_real(x_real))
        x_imag = F.gelu(self.ln1_imag(x_imag))

        x_real = F.dropout(x_real, p=self.dropout_rate, training=self.training)
        x_imag = F.dropout(x_imag, p=self.dropout_rate, training=self.training)
        
        x_real, x_imag = self.conv2(x_real, x_imag, edge_index)
        x_real = F.gelu(self.ln2_real(x_real))
        x_imag = F.gelu(self.ln2_imag(x_imag))
        
        # Concat the 2-channel Real and 2-channel Imag arrays (total 4 channels)
        x_cat = torch.cat([x_real, x_imag], dim=-1)
        
        return self.conv_mu(x_cat), self.conv_logstd(x_cat)


class FeatureDecoder(torch.nn.Module):
    """
    Reconstructs the full gene expression profile from the VAE latent space.
    Propagates the latent representations (and targeted CRISPR perturbations) across the directed GRN.
    """
    def __init__(self, n_channels, num_node_features, dropout_rate=0.1, q=0.25, K=2):
        super().__init__()
        self.conv1 = MagNetConv(n_channels, n_channels, K=K, q=q)
        self.ln1_real = LayerNorm(n_channels)
        self.ln1_imag = LayerNorm(n_channels)
        
        self.conv2 = MagNetConv(n_channels, n_channels, K=K, q=q)
        self.ln2_real = LayerNorm(n_channels)
        self.ln2_imag = LayerNorm(n_channels)
        
        self.dropout_rate = dropout_rate
        # Maps the complex structural embeddings back to biological read counts
        self.last_layer = torch.nn.Linear(2*n_channels, num_node_features) 

    def forward(self, z, edge_index):
        """
        Processes real-valued sampled latents (z) through complex convolutions, 
        returning predicted positive expression counts via Softplus.
        """
        z_real = F.dropout(z, p=self.dropout_rate, training=self.training)
        z_imag = torch.zeros_like(z_real) # Sampled latent 'z' is real
        
        z_real, z_imag = self.conv1(z_real, z_imag, edge_index)
        z_real = F.gelu(self.ln1_real(z_real))
        z_imag = F.gelu(self.ln1_imag(z_imag))

        z_real = F.dropout(z_real, p=self.dropout_rate, training=self.training)
        z_imag = F.dropout(z_imag, p=self.dropout_rate, training=self.training)
        
        z_real, z_imag = self.conv2(z_real, z_imag, edge_index)
        z_real = F.gelu(self.ln2_real(z_real))
        z_imag = F.gelu(self.ln2_imag(z_imag))
        
        z_cat = torch.cat([z_real, z_imag], dim=-1)
        
        # Softplus strictly outputs > 0 without causing dead gradients for low-expression genes
        return F.softplus(self.last_layer(z_cat))
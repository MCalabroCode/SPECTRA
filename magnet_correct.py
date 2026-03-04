import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import scipy.sparse as sp
import numpy as np

class MagNetConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, bias=True, **kwargs):
        """
        PyG implementation of MagNet Convolution for general K > 1.
        :param in_channels: Size of input features.
        :param out_channels: Size of output features.
        :param K: Order of Chebyshev polynomial.
        """
        # aggr='add' is required to sum the messages from neighbors
        super(MagNetConv, self).__init__(aggr='add', **kwargs)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        # Weight tensor of shape [K+1, in_channels, out_channels]
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x_real, x_imag, edge_index, norm, phase):
        """
        :param x_real: Real part of node features
        :param x_imag: Imaginary part of node features
        :param edge_index: Symmetrized edge indices (where A_s > 0)
        :param norm: The real magnitude values: - D_s^{-1/2} A_s D_s^{-1/2}
        :param phase: The edge phases: 2 * pi * q * (A_uv - A_vu)
        """
        # T_0(L_tilde) * x = x
        Tx_0_real = x_real
        Tx_0_imag = x_imag
        
        # Multiply by weight for k=0
        out_real = torch.matmul(Tx_0_real, self.weight[0])
        out_imag = torch.matmul(Tx_0_imag, self.weight[0])
        
        if self.K > 0:
            # T_1(L_tilde) * x = L_tilde * x
            Tx_1_stacked = self.propagate(edge_index, x_real=Tx_0_real, x_imag=Tx_0_imag, norm=norm, phase=phase)
            Tx_1_real = Tx_1_stacked[..., 0]
            Tx_1_imag = Tx_1_stacked[..., 1]
            
            # Add to output for k=1
            out_real = out_real + torch.matmul(Tx_1_real, self.weight[1])
            out_imag = out_imag + torch.matmul(Tx_1_imag, self.weight[1])
            
        # Chebyshev recurrence for k >= 2: T_k(x) = 2x*T_{k-1}(x) - T_{k-2}(x)
        for k in range(2, self.K + 1):
            
            # Compute L_tilde * T_{k-1}(L_tilde) * x
            Tx_2_stacked = self.propagate(edge_index, x_real=Tx_1_real, x_imag=Tx_1_imag, norm=norm, phase=phase)
            
            # Apply recurrence: 2 * (L_tilde * Tx_1) - Tx_0
            Tx_2_real = 2.0 * Tx_2_stacked[..., 0] - Tx_0_real
            Tx_2_imag = 2.0 * Tx_2_stacked[..., 1] - Tx_0_imag
            
            # Add to output for current k
            out_real = out_real + torch.matmul(Tx_2_real, self.weight[k])
            out_imag = out_imag + torch.matmul(Tx_2_imag, self.weight[k])
            
            # Shift polynomials for the next iteration
            Tx_0_real, Tx_0_imag = Tx_1_real, Tx_1_imag
            Tx_1_real, Tx_1_imag = Tx_2_real, Tx_2_imag

        if self.bias is not None:
            out_real += self.bias
            out_imag += self.bias

        return out_real, out_imag

    def message(self, x_real_j, x_imag_j, norm, phase):

        # Apply the phase matrix exp(i * Theta_q) via Euler's formula
        cos_theta = torch.cos(phase).view(-1, 1)
        sin_theta = torch.sin(phase).view(-1, 1)
        
        # Complex multiplication: (x_real + i*x_imag) * (cos + i*sin)
        msg_real = x_real_j * cos_theta - x_imag_j * sin_theta
        msg_imag = x_real_j * sin_theta + x_imag_j * cos_theta
        
        # Apply the symmetric transition magnitude norm (edge weight)
        norm = norm.view(-1, 1)
        #return norm * msg_real, norm * msg_imag
        msg_real = norm * msg_real
        msg_imag = norm * msg_imag
        
        # STACK into a single tensor of shape [num_edges, channels, 2] to avoid troubles with "aggregation" using tuples
        return torch.stack([msg_real, msg_imag], dim=-1)
    



def precompute_magnet_attributes_sparse(edge_index, num_nodes, q):
    """
    Scalable preprocessing using scipy.sparse for large graphs.
    """
    # 1. Create Scipy sparse COO matrix from edge_index
    row, col = edge_index.cpu().numpy()
    data = np.ones_like(row, dtype=np.float32)
    adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    
    # 2. Compute symmetric adjacency A_s = (A + A^T) / 2
    adj_t = adj.transpose()
    A_s = 0.5 * (adj + adj_t)
    
    # 3. Compute asymmetry for phase: A_diff = A - A^T
    A_diff = adj - adj_t
    
    # 4. Degree matrix for normalization D_s
    # Sum along rows for the symmetric matrix
    deg = np.array(A_s.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    
    # Create sparse diagonal matrix for normalization: D_s^{-1/2}
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    
    # 5. Compute normalized symmetric adjacency: norm = - D_s^{-1/2} A_s D_s^{-1/2}
    norm_matrix = - D_inv_sqrt.dot(A_s).dot(D_inv_sqrt)
    
    # Convert to COO to easily extract edges and values
    norm_matrix = norm_matrix.tocoo()
    
    # The symmetric edge_index corresponds to the non-zero entries of A_s / norm_matrix
    edge_index_sym = torch.tensor(np.vstack((norm_matrix.row, norm_matrix.col)), dtype=torch.long)
    
    # Extract the real magnitude values
    norm = torch.tensor(norm_matrix.data, dtype=torch.float32)
    
    # 6. Extract phase values for the exact same edges
    # We query the values of A_diff at (norm_matrix.row, norm_matrix.col)
    A_diff_csr = A_diff.tocsr() # CSR allows fast coordinate querying
    phase_data = A_diff_csr[norm_matrix.row, norm_matrix.col].A1  # .A1 flattens to 1D array
    phase = 2 * torch.pi * q * torch.tensor(phase_data, dtype=torch.float32)
    
    # Move back to original device (e.g., GPU)
    device = edge_index.device
    return edge_index_sym.to(device), norm.to(device), phase.to(device)
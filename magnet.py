import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import scipy.sparse as sp
import numpy as np

class MagNetConv(MessagePassing):
    def __init__(self, in_channels, out_channels, K, bias=True, **kwargs):
        super(MagNetConv, self).__init__(aggr='add', **kwargs) # node_dim=0 is no longer needed!
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        
        self.weight = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        
        # Separate biases for real and imaginary parts
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias_real is not None:
            nn.init.zeros_(self.bias_real)
            nn.init.zeros_(self.bias_imag)

    def forward(self, x_real, x_imag, edge_index, edge_weight_complex):
        # 1. Cast inputs to native complex tensor for blazing-fast propagation
        x = torch.complex(x_real, x_imag)
        
        Tx_0 = x
        out_real = torch.matmul(Tx_0.real, self.weight[0])
        out_imag = torch.matmul(Tx_0.imag, self.weight[0])
        
        if self.K > 0:
            Tx_1 = self.propagate(edge_index, x=Tx_0, edge_weight=edge_weight_complex)
            out_real = out_real + torch.matmul(Tx_1.real, self.weight[1])
            out_imag = out_imag + torch.matmul(Tx_1.imag, self.weight[1])
            
        for k in range(2, self.K + 1):
            Tx_2 = 2.0 * self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight_complex) - Tx_0
            out_real = out_real + torch.matmul(Tx_2.real, self.weight[k])
            out_imag = out_imag + torch.matmul(Tx_2.imag, self.weight[k])
            
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias_real is not None:
            out_real += self.bias_real
            out_imag += self.bias_imag

        return out_real, out_imag

    def message(self, x_j, edge_weight):
        # Native complex multiplication handles the rotation (sin/cos) automatically and efficiently
        return x_j * edge_weight.view(-1, 1)
    


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
    
    # Move back to original device
    device = edge_index.device
    
    # Pre-calculate the polar complex rotation once (norm * e^(i * phase))
    edge_weight_complex = norm.to(device) * torch.exp(1j * phase.to(device))
    
    return edge_index_sym.to(device), edge_weight_complex
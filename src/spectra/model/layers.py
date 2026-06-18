import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing, GATv2Conv, DirGNNConv
from torch_geometric.utils import degree
from typing import Optional, Tuple, Dict, Any

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
        
        gamma = self.scale(expr) 
        gamma = 1.0 + F.elu(gamma) # Starts at 1.0, can't go below 0, can grow infinitely
        beta = self.shift_scale * self.shift(expr)
        return gamma * gene_emb + beta

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

class DirFAGCNConv(MessagePassing):
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

    def __init__(self, channels: int, share_gates: bool = False, alpha=0.5):
        super().__init__(aggr="add", node_dim=0)

        self.channels = channels
        self.alpha = alpha

        self._cached_deg_in = None
        self._cached_deg_out = None
        self._cached_reverse_edge = None

        # shared convolutional kernel g (see paper)
        if share_gates:
            gate = nn.Linear(2 * channels, 1, bias=False)
            self.gate_in = gate
            self.gate_out = gate
        else:
            self.gate_in = nn.Linear(2 * channels, 1, bias=False)
            self.gate_out = nn.Linear(2 * channels, 1, bias=False)

        self.reset_parameters()

    def precompute_degrees(self, edge_index, num_nodes):
        src, dst = edge_index[0], edge_index[1]
        self._cached_reverse_edge = edge_index.flip(0)
        self._cached_deg_in  = degree(dst, num_nodes=num_nodes, dtype=torch.float).clamp(min=1.0)
        self._cached_deg_out = degree(src, num_nodes=num_nodes, dtype=torch.float).clamp(min=1.0)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate_in.weight)
        if self.gate_out is not self.gate_in:
            nn.init.xavier_uniform_(self.gate_out.weight)

    def forward(self, x: Tensor, edge_index: Tensor,
        edge_weight: Optional[Tensor] = None, return_alpha: bool = False,) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Parameters
        ----------
        x : Tensor
            Node features of shape [N, C].
        edge_index : Tensor
            Graph connectivity in COO format with shape [2, E].
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

        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # # Directed degrees on the graph:
        # deg_in = degree(dst, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)
        # deg_out = degree(src, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)

        # cached data
        if self._cached_deg_in is not None:
            deg_in, deg_out = self._cached_deg_in, self._cached_deg_out
            rev_edge_index = self._cached_reverse_edge
        else:
            deg_in  = degree(dst, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)
            deg_out = degree(src, num_nodes=num_nodes, dtype=x.dtype).clamp(min=1.0)
            rev_edge_index = edge_index.flip(0)

        # Pre-calculate gates to avoid massive torch.cat: W[h_i || h_j] = W_ih_i + W_jh_j
        w_in_i = self.gate_in.weight[:, :self.channels]
        w_in_j = self.gate_in.weight[:, self.channels:]
        alpha_in_i = F.linear(x, w_in_i) # this is W_i · h_n - Shape: [N, 1]
        alpha_in_j = F.linear(x, w_in_j) # this is W_j · h_n - Shape: [N, 1]

        # Incoming aggregation: src -> dst
        # tanh(W_ih_i + W_jh_j)
        alpha_in_edge = torch.tanh(
            alpha_in_i[dst] + alpha_in_j[src]
        ).view(-1)

        # Incoming aggregation: j -> i over original edges
        out_in = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_alpha=alpha_in_edge,
            deg_target=deg_in,
            edge_weight=edge_weight,
            size=(num_nodes, num_nodes),
        )

        # Outgoing aggregation: this is equivalent to propagating on reversed edges k -> i.
        #rev_edge_index = torch.stack([dst, src], dim=0)
        #rev_edge_index = edge_index.flip(0)

        w_out_i = self.gate_out.weight[:, :self.channels]
        w_out_j = self.gate_out.weight[:, self.channels:]
        alpha_out_i = F.linear(x, w_out_i)
        alpha_out_j = F.linear(x, w_out_j)

        # Outgoing message for node src uses neighbor dst.
        alpha_out_edge = torch.tanh(
            alpha_out_i[src] + alpha_out_j[dst]
        ).view(-1)

        out_out = self.propagate(
            edge_index=rev_edge_index,
            x=x,
            edge_alpha=alpha_out_edge,
            deg_target=deg_out,
            edge_weight=edge_weight,
            size=(num_nodes, num_nodes),
        )

        #out = self.eps * x_res + 0.5 * (out_in + out_out)
        out = (1-self.alpha) * out_in + self.alpha * out_out

        # if not return_alpha:
        #     return out

        # # Compute raw learned alpha values explicitly for inspection
        # alpha_in = self._compute_alpha(x=x, edge_index=edge_index, gate_nn=self.gate_in)
        # alpha_out = self._compute_alpha(x=x, edge_index=edge_index, gate_nn=self.gate_out)

        # alpha_dict = {
        #     "alpha_in": alpha_in,    # for edge j -> i, gate uses [x_i || x_j]
        #     "alpha_out": alpha_out,  # for edge i -> k, gate uses [x_i || x_k]
        # }

        # return out, alpha_dict
        if return_alpha:
            alpha_dict = {
                "alpha_in": alpha_in_edge,
                "alpha_out": alpha_out_edge,
            }
            return out, alpha_dict

        return out

    def message(self, x_j: Tensor, edge_alpha: Tensor, index: Tensor, deg_target: Tensor, edge_weight: Optional[Tensor]
    ) -> Tensor:
        """
        On each propagated edge j -> i:
            alpha_ij = tanh(g([x_i || x_j]))
            m_ij = alpha_ij / deg_target[i] * x_j
        """
        norm = 1.0 / deg_target[index]
        if edge_weight is not None:
            norm = norm * edge_weight
        return (edge_alpha * norm).unsqueeze(-1) * x_j

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

        w_center = gate_nn.weight[:, :self.channels]
        w_neigh = gate_nn.weight[:, self.channels:]

        alpha_center = F.linear(x_center, w_center)
        alpha_neigh = F.linear(x_neigh, w_neigh)

        return torch.tanh(alpha_center + alpha_neigh).view(-1)

class PolyLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, conv):
        super().__init__()
        self.conv = conv
        self.w_h = torch.nn.Linear(in_channels, out_channels)
        self.w_l = torch.nn.Linear(in_channels, out_channels)
        self.beta = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, x, edge_index):
        # Non-linear transformation
        h_i = F.relu(self.w_h(x))
        
        # Directed Convolution
        conv_out = self.conv(x, edge_index)
        
        # Intermediate representation with linear skip connection
        x_prime = conv_out + self.w_l(x)
        
        # Polynomial gating 
        out = (1 - self.beta) * (h_i * x_prime) + self.beta * x_prime
        return out

def dir_poly_conv(in_dim, out_dim):
    base_conv = GATv2Conv(in_dim, out_dim, heads=1, concat=False)
    dir_conv = DirGNNConv(base_conv, alpha=0.5)
    return PolyLayer(in_dim, out_dim, dir_conv)
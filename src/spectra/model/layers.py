"""Neural network layers and message passing modules for graph representations."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import DirGNNConv, GATv2Conv, MessagePassing
from torch_geometric.utils import degree


class GeneExpressionFiLM(nn.Module):
    """Feature-wise linear modulation for conditioning gene representations."""

    def __init__(self, emb_dim: int, shift_scale: float = 0.05) -> None:
        super().__init__()
        self.scale = nn.Linear(1, emb_dim)
        self.shift = nn.Linear(1, emb_dim)
        self.shift_scale = shift_scale

        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, gene_emb: Tensor, expr: Tensor) -> Tensor:
        gamma = 1.0 + F.elu(self.scale(expr))
        beta = self.shift_scale * self.shift(expr)
        return gamma * gene_emb + beta


class MLP(nn.Module):
    """Multi-layer perceptron with optional batch normalization and dropout."""

    def __init__(
        self,
        sizes: list[int],
        batch_norm: bool = True,
        dropout: float = 0.2,
    ) -> None:

        super().__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers.extend(
                [
                    nn.Dropout(p=dropout),
                    nn.Linear(sizes[s], sizes[s + 1]),
                    (
                        nn.BatchNorm1d(sizes[s + 1])
                        if batch_norm and s < len(sizes) - 2
                        else None
                    ),
                    nn.ReLU(),
                ]
            )

        layers = [layer for layer in layers if layer is not None][:-1]
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)


class DirFAGCNConv(MessagePassing):
    """
    Directed Frequency Adaptation Graph Convolution layer.

    From "Beyond Low-frequency Information in Graph Convolutional Networks",
    Deyu Bo et al.
    """

    def __init__(
        self,
        channels: int,
        share_gates: bool = False,
        alpha: float = 0.5,
    ) -> None:
        """Initializes directional projection gates and message aggregators.

        Args:
            channels: Dimensionality of input and output features.
            share_gates: Whether to share gate parameters between in and out edges.
            alpha: Weighting factor balancing incoming vs outgoing messages.
        """
        super().__init__(aggr="add", node_dim=0)
        self.channels = channels
        self.alpha = alpha

        self._cached_deg_in: Optional[Tensor] = None
        self._cached_deg_out: Optional[Tensor] = None
        self._cached_reverse_edge: Optional[Tensor] = None

        if share_gates:
            gate = nn.Linear(2 * channels, 1, bias=False)
            self.gate_in = gate
            self.gate_out = gate
        else:
            self.gate_in = nn.Linear(2 * channels, 1, bias=False)
            self.gate_out = nn.Linear(2 * channels, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initializes linear projection weights with Xavier uniform distribution.
        """
        nn.init.xavier_uniform_(self.gate_in.weight)
        if self.gate_out is not self.gate_in:
            nn.init.xavier_uniform_(self.gate_out.weight)

    def precompute_degrees(self, edge_index: Tensor, num_nodes: int) -> None:
        """
        Precomputes and caches directional node degrees and reversed edges.
        """
        src, dst = edge_index[0], edge_index[1]
        self._cached_reverse_edge = edge_index.flip(0)
        self._cached_deg_in = degree(
            dst, num_nodes=num_nodes, dtype=torch.float
        ).clamp(min=1.0)
        self._cached_deg_out = degree(
            src, num_nodes=num_nodes, dtype=torch.float
        ).clamp(min=1.0)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        return_alpha: bool = False,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        """
        Performs directional gating and message aggregation across incoming/outgoing edges.
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        if self._cached_deg_in is not None:
            deg_in, deg_out = self._cached_deg_in, self._cached_deg_out
            rev_edge_index = self._cached_reverse_edge
        else:
            deg_in = degree(dst, num_nodes=num_nodes, dtype=x.dtype).clamp(
                min=1.0
            )
            deg_out = degree(src, num_nodes=num_nodes, dtype=x.dtype).clamp(
                min=1.0
            )
            rev_edge_index = edge_index.flip(0)

        w_in_i = self.gate_in.weight[:, : self.channels]
        w_in_j = self.gate_in.weight[:, self.channels :]
        alpha_in_i = F.linear(x, w_in_i)
        alpha_in_j = F.linear(x, w_in_j)
        alpha_in_edge = torch.tanh(
            alpha_in_i[dst] + alpha_in_j[src]
        ).view(-1)

        out_in = self.propagate(
            edge_index=edge_index,
            x=x,
            edge_alpha=alpha_in_edge,
            deg_target=deg_in,
            edge_weight=edge_weight,
            size=(num_nodes, num_nodes),
        )

        w_out_i = self.gate_out.weight[:, : self.channels]
        w_out_j = self.gate_out.weight[:, self.channels :]
        alpha_out_i = F.linear(x, w_out_i)
        alpha_out_j = F.linear(x, w_out_j)
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

        out = (1.0 - self.alpha) * out_in + self.alpha * out_out

        if return_alpha:
            alpha_dict = {
                "alpha_in": alpha_in_edge,
                "alpha_out": alpha_out_edge,
            }
            return out, alpha_dict

        return out

    def message(
        self,
        x_j: Tensor,
        edge_alpha: Tensor,
        index: Tensor,
        deg_target: Tensor,
        edge_weight: Optional[Tensor],
    ) -> Tensor:
        """
        Constructs aggregated messages scaled by gating coefficients and node degree.
        """
        norm = 1.0 / deg_target[index]
        if edge_weight is not None:
            norm = norm * edge_weight
        return (edge_alpha * norm).unsqueeze(-1) * x_j

    @torch.no_grad()
    def _compute_alpha(
        self,
        x: Tensor,
        edge_index: Tensor,
        gate_nn: nn.Linear,
    ) -> Tensor:
        """
        Computes unnormalized directional attention coefficients for input edges.
        """
        src, dst = edge_index[0], edge_index[1]

        if gate_nn is self.gate_in:
            x_center = x[dst]
            x_neigh = x[src]
        else:
            x_center = x[src]
            x_neigh = x[dst]

        w_center = gate_nn.weight[:, : self.channels]
        w_neigh = gate_nn.weight[:, self.channels :]

        alpha_center = F.linear(x_center, w_center)
        alpha_neigh = F.linear(x_neigh, w_neigh)

        return torch.tanh(alpha_center + alpha_neigh).view(-1)


class PolyLayer(nn.Module):
    """
    Polynomial gating layer with a residual skip connection for directional GNNs.
    
    from "Flow Matters: Directional and Expressive GNNs for Heterophilic
    Graphs", A. Gupta et al
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv: nn.Module,
    ) -> None:
        """
        Initializes linear projections and learnable blending parameter beta.
        """
        super().__init__()
        self.conv = conv
        self.w_h = nn.Linear(in_channels, out_channels)
        self.w_l = nn.Linear(in_channels, out_channels)
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Evaluates non-linear and directional convolutions via polynomial combination.
        """
        h_i = F.relu(self.w_h(x))
        conv_out = self.conv(x, edge_index)
        x_prime = conv_out + self.w_l(x)

        return (1.0 - self.beta) * (h_i * x_prime) + self.beta * x_prime


def dir_poly_conv(in_dim: int, out_dim: int) -> PolyLayer:
    """
    Instantiates a directional polynomial layer wrapping a DirGNN-GATv2 convolution.
    """
    base_conv = GATv2Conv(in_dim, out_dim, heads=1, concat=False)
    dir_conv = DirGNNConv(base_conv, alpha=0.5)
    return PolyLayer(in_dim, out_dim, dir_conv)
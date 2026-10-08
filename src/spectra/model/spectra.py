"""
SPECTRA model architecture and variational graph autoencoder layers.
"""

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import ChebConv, DirGNNConv, GATv2Conv

from .layers import DirFAGCNConv, GeneExpressionFiLM, MLP, dir_poly_conv


class VariationalGraphEncoder(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout_rate=0.2,
        eps=0.2,
        conv_type="FAGCN",
    ):
        """Initializes the encoder layers based on the convolution type.

        Args:
            in_channels: Dimensionality of input node features.
            out_channels: Dimensionality of output latent features.
            dropout_rate: Dropout probability applied between layers.
            eps: Residual scaling coefficient.
            conv_type: Type of GNN convolution layer to construct.
        """
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.eps = eps

        if conv_type == "FAGCN":
            self.conv1 = DirFAGCNConv(out_channels)
            self.conv2 = DirFAGCNConv(out_channels)
        elif conv_type == "ChebConv":
            self.conv1 = ChebConv(in_channels, out_channels, K=2)
            self.conv2 = ChebConv(out_channels, out_channels, K=2)
        elif conv_type == "DirGATv2":
            self.conv1 = DirGNNConv(GATv2Conv(in_channels, out_channels, heads=2))
            self.conv2 = DirGNNConv(GATv2Conv(out_channels, out_channels, heads=2))
        elif conv_type == "DirPoly":
            self.conv1 = dir_poly_conv(in_channels, out_channels)
            self.conv2 = dir_poly_conv(out_channels, out_channels)
        else:
            raise ValueError(f"Convolution type {conv_type} is not supported!")

        self.ln1 = LayerNorm(out_channels)
        self.ln2 = LayerNorm(out_channels)

        self.proj_mu = torch.nn.Linear(out_channels, out_channels)
        self.proj_logstd = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        """Encodes graph-structured node features into latent parameters."""
        h_0 = x

        x_1 = self.conv1(h_0, edge_index)
        x_1 = x_1 + self.eps * h_0
        x_1 = self.ln1(x_1)
        x_1 = F.gelu(x_1)

        x_2 = F.dropout(x_1, p=self.dropout_rate, training=self.training)
        x_2 = self.conv2(x_2, edge_index)
        x_2 = x_2 + self.eps * h_0
        x_2 = self.ln2(x_2)
        x_2 = F.gelu(x_2)

        x_drop = F.dropout(x_2, p=self.dropout_rate, training=self.training)

        mu = self.proj_mu(x_drop)
        logst = self.proj_logstd(x_drop)

        return mu, logst


class FeatureDecoder(torch.nn.Module):
    """Decodes latent graph embeddings back into gene feature space."""

    def __init__(
        self,
        n_channels,
        num_node_features,
        dropout_rate=0.1,
        eps=0.2,
        conv_type="FAGCN",
    ):
        """Initializes the decoder layers based on the convolution type.

        Args:
            n_channels: Dimensionality of hidden channels.
            num_node_features: Dimensionality of reconstructed node outputs.
            dropout_rate: Dropout probability applied between layers.
            eps: Residual scaling coefficient.
            conv_type: Type of GNN convolution layer to construct.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.eps = eps

        if conv_type == "FAGCN":
            self.conv1 = DirFAGCNConv(n_channels)
            self.conv2 = DirFAGCNConv(n_channels)
            self.conv3 = DirFAGCNConv(n_channels)
        elif conv_type == "ChebConv":
            self.conv1 = ChebConv(n_channels, n_channels, K=1)
            self.conv2 = ChebConv(n_channels, n_channels, K=1)
            self.conv3 = ChebConv(n_channels, n_channels, K=1)
        elif conv_type == "DirGATv2":
            self.conv1 = DirGNNConv(GATv2Conv(n_channels, n_channels, heads=2))
            self.conv2 = DirGNNConv(GATv2Conv(n_channels, n_channels, heads=2))
            self.conv3 = DirGNNConv(GATv2Conv(n_channels, n_channels, heads=2))
        elif conv_type == "DirPoly":
            self.conv1 = dir_poly_conv(n_channels, n_channels)
            self.conv2 = dir_poly_conv(n_channels, n_channels)
            self.conv3 = dir_poly_conv(n_channels, n_channels)

        self.ln1 = LayerNorm(n_channels)
        self.ln2 = LayerNorm(n_channels)
        self.ln3 = LayerNorm(n_channels)
        self.last_layer = torch.nn.Linear(n_channels, num_node_features)

    def forward(self, z, edge_index, return_alpha=False):
        """Decodes latent node representations into feature reconstructions."""

        x_0 = z

        # layer 1
        if return_alpha:
            h1, alpha_dict_1 = self.conv1(
                z, edge_index, return_alpha=True
            )
        else:
            h1 = self.conv1(z, edge_index)
        h1 = h1 + self.eps * x_0
        h1 = self.ln1(h1)
        h1_out = F.gelu(h1)

        # layer 2
        h1_drop = F.dropout(h1_out, p=self.dropout_rate, training=self.training)
        if return_alpha:
            h2, alpha_dict_2 = self.conv2(
                h1_drop, edge_index, return_alpha=True
            )
        else:
            h2 = self.conv2(h1_drop, edge_index)
        h2 = h2 + self.eps * x_0
        h2 = self.ln2(h2)
        h2_out = F.gelu(h2)

        # layer 3
        h2_drop = F.dropout(h2_out, p=self.dropout_rate, training=self.training)
        if return_alpha:
            h3, alpha_dict_3 = self.conv3(
                h2_drop, edge_index, return_alpha=True
            )
        else:
            h3 = self.conv3(h2_drop, edge_index)
        h3 = h3 + self.eps * x_0
        h3 = self.ln3(h3)
        h3_out = F.gelu(h3)

        # output
        out = self.last_layer(h3_out)
        out = F.softplus(out)

        if return_alpha:
            return out, (alpha_dict_1, alpha_dict_2, alpha_dict_3)

        return out


class SPECTRA(torch.nn.Module):
    """Main SPECTRA model class"""

    def __init__(
        self,
        edge_index,
        num_nodes,
        device,
        config,
        gene_embeddings,
        gene_names,
        gene_weights=None,
    ):
        """Initializes the SPECTRA architecture and buffers.

        Args:
            edge_index: Graph connectivity tensor of shape [2, num_edges].
            num_nodes: Total number of genes in the regulatory network.
            device: Computing device for tensors.
            config: Configuration dictionary containing model hyperparameters.
            gene_embeddings: Pretrained embedding matrix for network nodes.
            gene_names: sequence of gene identifiers.
            gene_weights: Optional dictionary of precomputed DEG weights.
        """
        super().__init__()

        self.device = device
        self.num_nodes = num_nodes
        self.config = config
        self.res = config["residual_weight"]
        self.conv_type = config["conv_type"]
        self.n_channels = config["n_channels"]
        self.dropout_p = config["dropout_p"]
        self.architecture_name = config["architecture"]

        self.register_buffer("edge_index", edge_index)

        num_conditions = len(config["pert_to_idx"])
        default_weights = (1.0 / num_nodes) * torch.ones(num_nodes)
        weight_lookup = (
            default_weights.unsqueeze(0).repeat(num_conditions, 1).to(device)
        )
        if gene_weights is not None:
            for pert_name, weight_array in gene_weights.items():
                if pert_name in config["pert_to_idx"]:
                    idx = config["pert_to_idx"][pert_name]
                    w_tensor = torch.tensor(
                        weight_array, dtype=torch.float32, device=device
                    )
                    weight_lookup[idx] = w_tensor
        self.register_buffer("weight_lookup", weight_lookup)

        self.gene_embeddings = torch.nn.Embedding.from_pretrained(
            gene_embeddings, freeze=True
        )
        scgpt_dim = gene_embeddings.shape[1]

        self.project_gene = torch.nn.Linear(scgpt_dim, self.n_channels)
        self.film_layer = GeneExpressionFiLM(self.n_channels, shift_scale=0.05)
        self.ko_mlp = MLP(
            [scgpt_dim, self.n_channels], batch_norm=False, dropout=0.0
        )
        self.encoder = VariationalGraphEncoder(
            self.n_channels,
            self.n_channels,
            self.dropout_p,
            self.res,
            self.conv_type,
        )
        self.gex_decoder = FeatureDecoder(
            self.n_channels, 1, self.dropout_p, self.res, self.conv_type
        )

        self._cached_batch_size = 0
        self._cached_edge_index = None
        self._cached_gene_ids = None

        self.gene_names = list(gene_names) if gene_names is not None else None
        if self.gene_names is not None:
            assert len(self.gene_names) == num_nodes, (
                f"Length of gene_names ({len(self.gene_names)}) "
                f"must match num_nodes ({num_nodes})"
            )
            self.gene_to_idx = {g: i for i, g in enumerate(self.gene_names)}
            self.idx_to_gene = {i: g for i, g in enumerate(self.gene_names)}
        else:
            self.gene_to_idx = None
            self.idx_to_gene = None

    def _get_batched_edge_index(self, batch_size):
        """Constructs and caches batch-replicated edge index tensors."""
        if (
            batch_size == self._cached_batch_size
            and self._cached_edge_index is not None
        ):
            return self._cached_edge_index
        offsets = torch.arange(batch_size, device=self.device) * self.num_nodes
        edge_index_batch = self.edge_index.unsqueeze(1) + offsets.view(1, -1, 1)
        edge_index_batch = edge_index_batch.reshape(2, -1)
        self._cached_batch_size = batch_size
        self._cached_edge_index = edge_index_batch
        return edge_index_batch

    def _get_batched_gene_ids(self, batch_size):
        """Generates and caches batch-replicated gene node indices."""
        if (
            self._cached_gene_ids is not None
            and len(self._cached_gene_ids) == batch_size * self.num_nodes
        ):
            return self._cached_gene_ids
        ids = torch.arange(self.num_nodes, device=self.device)
        ids = ids.repeat(batch_size)
        self._cached_gene_ids = ids
        return ids

    def reparametrize(self, mu, logstd, eps):
        """Applies the reparameterization trick during training."""
        if self.training:
            return mu + eps * torch.exp(logstd)
        return mu

    def kl_loss(self, mu, logstd, threshold=1e-2, verbose=True, free_bits=0.05):
        """Computes the Kullback-Leibler divergence against unit normal prior."""
        kl_raw = -0.5 * (1 + 2 * logstd - mu**2 - logstd.exp() ** 2)
        kl_per_dim = torch.mean(kl_raw, dim=0)
        return torch.mean(kl_per_dim)

    def forward(self, data, return_attention_weights=False):
        """Executes the forward autoencoding and perturbation pass."""

        x, pert = data
        x = x.to(self.device)
        pert = pert.to(self.device)

        batch_size, num_nodes, num_features = x.shape

        edge_index_batch = self._get_batched_edge_index(batch_size)
        pert = pert.reshape(batch_size * num_nodes)

        # project & add logic
        single_gene_ids = torch.arange(self.num_nodes, device=self.device)
        single_scgpt_base = self.gene_embeddings(single_gene_ids)
        single_gene_proj = self.project_gene(single_scgpt_base)
        gene_proj = single_gene_proj.repeat(batch_size, 1)
        x_flat = x.reshape(batch_size * num_nodes, 1)
        x_node_features = self.film_layer(gene_proj, x_flat)

        mu, logstd = self.encoder(x_node_features, edge_index_batch)
        logstd = torch.clamp(logstd, min=-20, max=10)

        self.last_mu = mu
        self.last_logstd = logstd

        eps = torch.randn_like(logstd) if self.training else None

        z_ctrl = self.reparametrize(mu, logstd, eps)
        self.last_z = z_ctrl

        # control reconstruction
        x_hat = self.gex_decoder(z_ctrl, edge_index_batch)

        # perturbation
        pert_mask = pert.bool()
        delta_mu = torch.zeros_like(mu)
        if pert_mask.any():
            gene_ids = self._get_batched_gene_ids(batch_size)
            perturbed_gene_ids = gene_ids[pert_mask]
            scgpt_base = self.gene_embeddings(perturbed_gene_ids)
            mu_shift = self.ko_mlp(scgpt_base)
            delta_mu[pert_mask] = mu_shift
            
        mu_pert = mu + delta_mu
        logstd_pert = logstd
        z_pert = self.reparametrize(mu_pert, logstd_pert, eps)

        if return_attention_weights:
            y_hat, (alpha_dict_1, alpha_dict_2, alpha_dict_3) = (
                self.gex_decoder(z_pert, edge_index_batch, return_alpha=True)
            )
            return y_hat, x_hat, (alpha_dict_1, alpha_dict_2, alpha_dict_3)

        y_hat = self.gex_decoder(z_pert, edge_index_batch)
        return y_hat, x_hat

    def predict_full_expression(self, data):
        """Predicts full post-perturbation gene expression in evaluation mode."""
        self.eval()
        return self.forward(data)[0]

    @staticmethod
    def _downstream_A_from_alpha(edge_index, alpha, n_nodes, edge_weight=None):
        """Constructs an in-degree normalized adjacency matrix from attention weights."""
        src, dst = edge_index
        deg_in = (
            torch.bincount(dst, minlength=n_nodes)
            .float()
            .clamp_min(1.0)
            .to(alpha.device)
        )
        vals = alpha / deg_in[dst]
        if edge_weight is not None:
            vals = vals * edge_weight.to(alpha.device)

        return torch.sparse_coo_tensor(
            torch.stack([src, dst]), vals, (n_nodes, n_nodes), device=alpha.device
        ).to_dense()

    @torch.no_grad()
    def compute_downstream_path_matrices(self, x, pert):
        """Averages layer-wise downstream attention matrices across a batch."""
        self.eval()
        y_hat, x_hat, (a1, a2, a3) = self.forward(
            (x, pert), return_attention_weights=True
        )

        b_size = pert.shape[0]
        n_nodes = self.num_nodes
        n_edges = self.edge_index.shape[1]

        alpha1 = a1["alpha_in"].view(b_size, n_edges)
        alpha2 = a2["alpha_in"].view(b_size, n_edges)
        alpha3 = a3["alpha_in"].view(b_size, n_edges)

        a1_sum = torch.zeros(n_nodes, n_nodes, device=self.device)
        a2_sum = torch.zeros(n_nodes, n_nodes, device=self.device)
        a3_sum = torch.zeros(n_nodes, n_nodes, device=self.device)

        for b in range(b_size):
            a1_sum += self._downstream_A_from_alpha(
                self.edge_index, alpha1[b], n_nodes
            )
            a2_sum += self._downstream_A_from_alpha(
                self.edge_index, alpha2[b], n_nodes
            )
            a3_sum += self._downstream_A_from_alpha(
                self.edge_index, alpha3[b], n_nodes
            )

        return (
            a1_sum / b_size,
            a2_sum / b_size,
            a3_sum / b_size,
            y_hat,
            x_hat,
        )
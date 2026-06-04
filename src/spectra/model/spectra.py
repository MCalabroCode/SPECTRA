import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from .layers import GeneExpressionFiLM, MLP, DirFAGCNConv, dir_poly_conv
from torch_geometric.nn import ChebConv


class VariationalGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate = 0.2, eps = 0.2, conv_type='FAGCN'):
        super().__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.eps = eps

        # Route the layer creation based on the config string!
        if conv_type == 'FAGCN':
            self.conv1 = DirFAGCNConv(out_channels)
            self.conv2 = DirFAGCNConv(out_channels)
            self.conv_mu = DirFAGCNConv(out_channels)
            self.conv_logstd = DirFAGCNConv(out_channels)
        elif conv_type == 'ChebConv':
            self.conv1 = ChebConv(in_channels, out_channels, K=2)
            self.conv2 = ChebConv(out_channels, out_channels, K=2)
            self.conv_mu = ChebConv(out_channels, out_channels, K=1)
            self.conv_logstd = ChebConv(out_channels, out_channels, K=1)
        elif conv_type == 'DirGATv2':
            self.conv1 = DirGNN_GATv2(in_channels, out_channels)
            self.conv2 = DirGNN_GATv2(out_channels, out_channels)
            self.conv_mu = DirGNN_GATv2(in_channels, out_channels)
            self.conv_logstd = DirGNN_GATv2(out_channels, out_channels)
        elif conv_type == 'DirPoly':
            self.conv1 = dir_poly_conv(in_channels, out_channels)
            self.conv2 = dir_poly_conv(out_channels, out_channels)
            self.conv_mu = dir_poly_conv(in_channels, out_channels)
            self.conv_logstd = dir_poly_conv(out_channels, out_channels)
        else:
            raise ValueError(f"Convolution type {conv_type} is not supported!")

        self.ln1 = LayerNorm(out_channels)
        self.ln2 = LayerNorm(out_channels)

        self.proj_mu = torch.nn.Linear(out_channels, out_channels)
        self.proj_logstd = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):

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
        
        # Calculate mu and logstd
        mu = self.conv_mu(x_drop, edge_index)
        mu = mu + self.eps * h_0
        mu = self.proj_mu(mu)  # Final projection to out_channels
        
        logst = self.conv_logstd(x_drop, edge_index)
        logst = logst + self.eps * h_0
        logst = self.proj_logstd(logst) # Final projection to out_channels
        
        return mu, logst

class FeatureDecoder(torch.nn.Module):
    def __init__(self, n_channels, num_node_features, dropout_rate=0.1, eps = 0.2, conv_type='FAGCN'):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.eps = eps
        
        # Route the layer creation based on the config string!
        if conv_type == 'FAGCN':
            self.conv1 = DirFAGCNConv(n_channels)
            self.conv2 = DirFAGCNConv(n_channels)
            # self.conv3 = DirFAGCNConv(n_channels)
        elif conv_type == 'ChebConv':
            self.conv1 = ChebConv(n_channels, n_channels, K=1)
            self.conv2 = ChebConv(n_channels, n_channels, K=1)
            self.conv3 = ChebConv(n_channels, n_channels, K=1)
        elif conv_type == 'DirGATv2':
            self.conv1 = DirGNN_GATv2(n_channels, n_channels)
            self.conv2 = DirGNN_GATv2(n_channels, n_channels)
            self.conv3 = DirGNN_GATv2(n_channels, n_channels)
        elif conv_type == 'DirPoly':
            self.conv1 = dir_poly_conv(n_channels, n_channels)
            self.conv2 = dir_poly_conv(n_channels, n_channels)
            self.conv3 = dir_poly_conv(n_channels, n_channels)

        self.ln1 = LayerNorm(n_channels)
        self.ln2 = LayerNorm(n_channels)
        # self.ln3 = LayerNorm(n_channels)
        self.last_layer = torch.nn.Linear(n_channels, num_node_features) 

    def forward(self, z, edge_index, return_alpha=False):

        # In the decoder, the latent representation 'z' acts as the raw feature (x_0)
        x_0 = z 
        if return_alpha:
            h1, alpha_dict_1 = self.conv1(z, edge_index, return_alpha=True)
        else:
            h1 = self.conv1(z, edge_index)
        h1 = h1 + self.eps * x_0
        h1 = self.ln1(h1)
        h1_out = F.gelu(h1)

        h1_drop = F.dropout(h1_out, p=self.dropout_rate, training=self.training)
        if return_alpha:
            h2, alpha_dict_2 = self.conv2(h1_drop, edge_index, return_alpha=True)
        else:
            h2 = self.conv2(h1_drop, edge_index)
        h2 = h2 + self.eps * x_0
        h2 = self.ln2(h2)
        h2_out = F.gelu(h2)

        # h2_drop = F.dropout(h2_out, p=self.dropout_rate, training=self.training)
        # if return_alpha:
        #     h3, alpha_dict_3 = self.conv3(h2_drop, edge_index, return_alpha=True)
        # else:
        #     h3 = self.conv3(h2_drop, edge_index)
        # h3 = h3 + self.eps * x_0
        # h3 = self.ln3(h3)
        # h3_out = F.gelu(h3)

        # Output Head
        out = self.last_layer(h2_out) #NOTE: before was h3_out! OCIO if you want 3 layers instead of 2!!!!!
        
        # LeakyReLU during training to prevent dead gradients, hard ReLU for biological realism at eval
        if self.training:
            out = F.leaky_relu(out, negative_slope=0.05) 
        else:
            out = F.relu(out)
        
        if return_alpha:
            return out, (alpha_dict_1, alpha_dict_2, alpha_dict_3)
        
        return out


class SPECTRA(torch.nn.Module):
    '''
    SPECTRA model class
    '''
    def __init__(self, edge_index, num_nodes, device, config, gene_embeddings, gene_weights=None):
        super().__init__()

        self.device = device 
        self.num_nodes = num_nodes 
        self.config = config
        self.res = config['residual_weight']
        self.conv_type = config['conv_type']
        self.n_channels = config['n_channels']
        self.dropout_p = config['dropout_p']
        self.architecture_name = config['architecture']
        #self.num_node_features = config['num_node_features'] 
        
        self.register_buffer('edge_index', edge_index)

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
        #self.project_expr = torch.nn.Linear(1, self.n_channels)
        self.film_layer = GeneExpressionFiLM(self.n_channels, shift_scale=0.05)

        # Optional: A LayerNorm to stabilize the addition
        self.add_norm = LayerNorm(self.n_channels)

        #self.ko_mu = torch.nn.Embedding(num_nodes, 64)
        #self.ko_mlp = MLP([64, self.n_channels, self.n_channels])
        #self.ko_mlp = MLP([scgpt_dim, 2*self.n_channels, self.n_channels], dropout=0.1) 
        self.ko_mlp = MLP([scgpt_dim, self.n_channels], dropout=0.0) 

        self.encoder = VariationalGraphEncoder(self.n_channels, self.n_channels, self.dropout_p, self.res, self.conv_type)
        self.gex_decoder = FeatureDecoder(self.n_channels, 1, self.dropout_p, self.res, self.conv_type)
        
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

    def reparametrize(self, mu, logstd, eps):
        if self.training:
            return mu + eps * torch.exp(logstd)
        else:
            return mu

    def kl_loss(self, mu, logstd, threshold=1e-2, verbose=True, free_bits=0.05):

        kl_raw = -0.5 * (1 + 2 * logstd - mu**2 - logstd.exp()**2)
        kl_per_dim = torch.mean(kl_raw, dim=0)

        #kl_per_dim_clamped = torch.clamp(kl_per_dim, min=free_bits)
        return torch.mean(kl_per_dim)

    def forward(self, data, return_attention_weights=False):

        x, pert = data
        x = x.to(self.device) #[B,N,1]
        pert = pert.to(self.device) #[B,N]

        batch_size, num_nodes, num_features = x.shape

        edge_index_batch = self._get_batched_edge_index(batch_size)
        pert = pert.reshape(batch_size * num_nodes) #[BxN]

        # project & add logic
        # gene_ids = self._get_batched_gene_ids(batch_size) 
        # scgpt_base = self.gene_embeddings(gene_ids) 
        # gene_proj = self.project_gene(scgpt_base)   # [B * N, n_channels]
        single_gene_ids = torch.arange(self.num_nodes, device=self.device)
        single_scgpt_base = self.gene_embeddings(single_gene_ids) # [N, 512]
        single_gene_proj = self.project_gene(single_scgpt_base)   # [N, C] (C = hidden channels dimension)
        gene_proj = single_gene_proj.repeat(batch_size, 1) # scGPT embeddings are always the same for each batch
        x_flat = x.reshape(batch_size * num_nodes, 1) 
        x_node_features = self.film_layer(gene_proj, x_flat)

        mu, logstd = self.encoder(x_node_features, edge_index_batch) # both [BxN, C] 
        logstd = torch.clamp(logstd, min=-20, max=10) # this is to avoid inf values

        self.last_mu = mu          
        self.last_logstd = logstd  

        if self.training:
            eps = torch.randn_like(logstd)
        else:
            eps = None

        z_ctrl = self.reparametrize(mu, logstd, eps)     
        self.last_z = z_ctrl

        x_hat = self.gex_decoder(z_ctrl, edge_index_batch)
        
        # additive logic for perturbation encoding
        mask = pert.unsqueeze(1).to(z_ctrl.dtype) #[BxN,1]

        pert_mask = pert.bool()
        delta_mu = torch.zeros_like(mu)
        if pert_mask.any():
            gene_ids = self._get_batched_gene_ids(batch_size)
            perturbed_gene_ids = gene_ids[pert_mask]

            scgpt_base = self.gene_embeddings(perturbed_gene_ids) 
            #ko_embeddings = self.ko_mu(perturbed_gene_ids)

            mu_shift = self.ko_mlp(scgpt_base) 
            #mu_shift = self.ko_mlp(ko_embeddings)

            delta_mu[pert_mask] = mu_shift

        mu_pert = mu + delta_mu
        logstd_pert = logstd #+ delta_logstd
        z_pert = self.reparametrize(mu_pert, logstd_pert, eps)

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

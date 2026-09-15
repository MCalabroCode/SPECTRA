"""
spectra.interpretability
========================
Analysis and visualization toolkit for SPECTRA learned FAGCN weights graphs,
cascade path discovery, and predicted differential expression profiles.
"""

import os
import textwrap
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import wilcoxon, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import plotly.graph_objects as go
import plotly.io as pio
import gseapy as gp


# =====================================================================
# Batch & Tensor Preparation Helper
# =====================================================================

def prepare_perturbation_batch(target_gene, control_adata, model, n_cells=256):
    """
    Constructs matched control inputs `x` and perturbation indicator tensor `pert`
    directly using `model.gene_to_idx`.
    """
    if model.gene_to_idx is None:
        raise ValueError("Model does not have `gene_to_idx` initialized. Pass `gene_names` at model creation.")
    
    pert_genes = [g for g in target_gene.split('+') if g in model.gene_to_idx]
    if not pert_genes:
        raise ValueError(f"Target gene(s) '{target_gene}' not found in model's gene vocabulary.")

    pert_single = torch.zeros(model.num_nodes, dtype=torch.bool)
    for g in pert_genes:
        pert_single[model.gene_to_idx[g]] = True

    ctrl_mat = control_adata.X.tocsr() if sparse.issparse(control_adata.X) else sparse.csr_matrix(control_adata.X)
    ctrl_array = ctrl_mat.toarray()
    
    n_take = min(n_cells, ctrl_array.shape[0])
    x = torch.tensor(ctrl_array[:n_take], dtype=torch.float32, device=model.device).unsqueeze(-1)
    pert = pert_single.repeat(n_take, 1).to(model.device)

    return x, pert


# =====================================================================
# complete Path & Cascade Network Extraction
# =====================================================================

def get_cascade_network(A_matrices, target_gene, model, minimum_signal=0.1, K_edges=20):
    """
    Extracts complete downstream paths of lengths 1, 2, and 3 originating from target_gene.
    Returns:
        hop_to_genes: dict mapping hop level (0-3) to sets of gene symbols
        cascade_edges: list of tuples (src, src_hop, tgt, tgt_hop, w, abs_w)
        all_signed_weights: list of gate weights
        network_genes: unique list of gene symbols across the cascade
    """
    idx_to_gene = model.idx_to_gene
    target_idx = model.gene_to_idx[target_gene]

    A = [matrix.detach().cpu().numpy().copy() for matrix in A_matrices]
    for matrix in A:
        matrix[np.abs(matrix) < minimum_signal] = 0.0

    selected_paths = []
    current_paths = [([target_idx], 1.0)]

    for depth in range(1, 4):
        layer_A = A[depth - 1]
        new_paths = []

        for nodes, score in current_paths:
            source = nodes[-1]
            targets = np.flatnonzero(layer_A[source])

            for target in targets:
                if target in nodes:
                    continue  # prevent self-cycles
                edge_weight = layer_A[source, target]
                new_paths.append((nodes + [target], score * edge_weight))

        new_paths.sort(key=lambda item: abs(item[1]), reverse=True)
        selected_paths.extend(new_paths[:K_edges])
        current_paths = new_paths

    hop_to_genes = {0: {target_gene}, 1: set(), 2: set(), 3: set()}
    edge_dict = {}
    network_genes = {target_gene}

    for nodes, _ in selected_paths:
        path_length = len(nodes) - 1
        for hop in range(path_length):
            src_idx = nodes[hop]
            tgt_idx = nodes[hop + 1]
            src_gene = idx_to_gene[src_idx]
            tgt_gene = idx_to_gene[tgt_idx]

            w = A[hop][src_idx, tgt_idx]
            key = (src_gene, hop, tgt_gene, hop + 1)
            edge_dict[key] = (w, abs(w))

            hop_to_genes[hop].add(src_gene)
            hop_to_genes[hop + 1].add(tgt_gene)
            network_genes.add(src_gene)
            network_genes.add(tgt_gene)

    cascade_edges = [
        (src, s_hop, tgt, t_hop, w, abs_w)
        for (src, s_hop, tgt, t_hop), (w, abs_w) in edge_dict.items()
    ]
    all_signed_weights = [edge[4] for edge in cascade_edges]

    return hop_to_genes, cascade_edges, all_signed_weights, list(network_genes)

def _get_cascade_network_old(A_matrices, target_gene, idx_to_gene, minimum_signal=0.1, K_edges=20):
    """
    Builds the network from attention matrices and extracts the top edges per hop.
    old code, do not use.
    """
    hop_to_genes = {0: {target_gene}, 1: set(), 2: set(), 3: set()}
    cascade_edges = []
    all_signed_weights = []
    labels = []
    
    current_sources = {target_gene}
    
    for hop_idx, A in enumerate(A_matrices):
        current_hop_num = hop_idx + 1
        
        # Densify and threshold
        A_dense = A.cpu().detach().numpy()
        A_dense[np.abs(A_dense) < minimum_signal] = 0.0 
        
        # Build Graph
        G = nx.from_numpy_array(A_dense, create_using=nx.DiGraph)
        G = nx.relabel_nodes(G, idx_to_gene)
        G.remove_edges_from(nx.selfloop_edges(G))
        
        next_sources = set()
        layer_edges = []
        
        # Gather edges for current sources
        for source_gene in current_sources:
            if source_gene in G:
                for target_gene_dest in G.successors(source_gene):
                    w = G[source_gene][target_gene_dest]['weight']
                    layer_edges.append((source_gene, target_gene_dest, w, abs(w)))
                    
        # Sort by absolute weight and keep top K
        layer_edges.sort(key=lambda x: x[3], reverse=True)
        top_layer_edges = layer_edges[:K_edges]
        
        # Record cascade path
        for src, tgt, w, abs_w in top_layer_edges:
            cascade_edges.append((src, current_hop_num - 1, tgt, current_hop_num, w, abs_w))
            labels.extend([src, tgt])
            next_sources.add(tgt)
            hop_to_genes[current_hop_num].add(tgt)
            all_signed_weights.append(w)
            
        current_sources = next_sources
        
    network_genes = list(set([label.split(' (')[0] for label in labels]))
    return hop_to_genes, cascade_edges, all_signed_weights, list(network_genes)

def get_top_complete_paths(A_matrices, target_gene, model, K_paths=20, minimum_signal=1e-3):
    """
    Finds top K complete paths of lengths 1, 2, and 3 scored by the product of FAGCN gate weights.
    """
    idx_to_gene = model.idx_to_gene
    target_idx = model.gene_to_idx[target_gene]
    A = [matrix.detach().cpu().numpy() for matrix in A_matrices]

    current_paths = [([target_idx], [])]
    top_paths = {}

    for layer in range(3):
        new_paths = []
        for nodes, weights in current_paths:
            source = nodes[-1]
            targets = np.flatnonzero(np.abs(A[layer][source]) >= minimum_signal)

            for target in targets:
                if target in nodes:
                    continue
                weight = A[layer][source, target]
                new_paths.append((nodes + [target], weights + [weight]))

        scored_paths = []
        for nodes, weights in new_paths:
            scored_paths.append({
                "nodes": nodes,
                "genes": [idx_to_gene[idx] for idx in nodes],
                "weights": weights,
                "score": float(np.prod(weights))
            })

        scored_paths.sort(key=lambda p: abs(p["score"]), reverse=True)
        top_paths[layer + 1] = scored_paths[:K_paths]
        current_paths = new_paths

    return top_paths


# =====================================================================
# Gene Ontology Functional Annotation
# =====================================================================

def get_gene_ontology(network_genes, target_gene, background_genes, top_n=10):
    """Queries Enrichr (via GSEAPY) to assign primary biological functions to cascade genes."""
    print("Querying Gene Ontology...")
    go_dict = gp.get_library(name='GO_Biological_Process_2026', organism='human')
    hallmark_dict = gp.get_library(name='MSigDB_Hallmark_2020', organism='human')
    combined_sets = {**go_dict, **hallmark_dict}

    enr = gp.enrichr(
        gene_list=network_genes,
        gene_sets=combined_sets,
        organism='human',
        background=background_genes,
        outdir=None
    )

    top_pathways = enr.results.head(top_n)
    gene_to_function = {}

    for gene in network_genes:
        assigned = False
        for _, row in top_pathways.iterrows():
            pathway_genes = row['Genes'].split(';')
            if gene in pathway_genes:
                clean_name = row['Term'].split(' (GO:')[0].capitalize()
                gene_to_function[gene] = clean_name
                assigned = True
                break
        if not assigned:
            gene_to_function[gene] = 'Other / Unassigned'

    gene_to_function[target_gene] = 'Perturbed Target'
    return gene_to_function


# =====================================================================
# Pseudobulk & Wilcoxon DEG Statistics
# =====================================================================

def get_pseudobulk(expr, B, N):
    """Averages cell expressions across the batch in linear space (expm1)."""
    expr_np = expr.detach().cpu().reshape(B, N).numpy()
    return np.expm1(expr_np).mean(axis=0)


def get_predicted_de_stats(y_hat, x_hat, model):
    """Runs a paired Wilcoxon signed-rank test comparing predicted perturbed vs control cells."""
    B = y_hat.shape[0] // model.num_nodes
    N = model.num_nodes

    y = y_hat.detach().cpu().reshape(B, N).numpy()
    x = x_hat.detach().cpu().reshape(B, N).numpy()

    pvals = np.ones(N)
    for g in range(N):
        diff = y[:, g] - x[:, g]
        if np.allclose(diff, 0):
            pvals[g] = 1.0
        else:
            pvals[g] = wilcoxon(y[:, g], x[:, g], alternative="two-sided").pvalue

    _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
    return pvals, qvals


# =====================================================================
# Plotting Functions
# =====================================================================

def plot_sankey(target_gene, hop_to_genes, cascade_edges, all_signed_weights, gene_to_function, out_dir='cascade_plots'):
    """
    Renders and saves a multi-hop Plotly Sankey diagram of the information cascade.
    """
    os.makedirs(out_dir, exist_ok=True)
    unique_groups = sorted(list(set(gene_to_function.values())))
    if 'Perturbed Target' in unique_groups:
        unique_groups.remove('Perturbed Target')
        unique_groups.insert(0, 'Perturbed Target')

    palette = plt.cm.tab20.colors
    group_colors = {}
    for i, group in enumerate(unique_groups):
        if group == 'Perturbed Target':
            group_colors[group] = 'rgba(50, 50, 50, 0.9)'
        elif group == 'Other / Unassigned':
            group_colors[group] = 'rgba(200, 200, 200, 0.6)'
        else:
            c = palette[i % len(palette)]
            group_colors[group] = f'rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.9)'

    node_id_to_idx = {}
    display_labels, node_colors, node_x, node_y = [], [], [], []
    hop_x_map = {0: 0.01, 1: 0.33, 2: 0.66, 3: 0.99}

    for hop_num in range(4):
        genes_in_hop = list(hop_to_genes[hop_num])
        genes_in_hop.sort(key=lambda g: (gene_to_function.get(g, 'Other / Unassigned'), g))
        y_vals = [0.5] if len(genes_in_hop) == 1 else np.linspace(0.05, 0.95, len(genes_in_hop)) if genes_in_hop else []

        for gene, y_val in zip(genes_in_hop, y_vals):
            unique_id = f"{gene}_hop{hop_num}"
            node_id_to_idx[unique_id] = len(display_labels)
            display_labels.append(gene)
            node_colors.append(group_colors[gene_to_function.get(gene, 'Other / Unassigned')])
            node_x.append(hop_x_map[hop_num])
            node_y.append(y_val)

    sources, targets, values, link_colors = [], [], [], []
    max_abs_w = max([abs(w) for w in all_signed_weights]) if all_signed_weights else 1.0
    vmin, vmax = -max_abs_w, max_abs_w
    min_visual_weight = max_abs_w * 0.05

    for src, src_hop, tgt, tgt_hop, w, abs_w in cascade_edges:
        sources.append(node_id_to_idx[f"{src}_hop{src_hop}"])
        targets.append(node_id_to_idx[f"{tgt}_hop{tgt_hop}"])
        values.append(max(abs_w, min_visual_weight))
        norm_w = (w - vmin) / (vmax - vmin)
        rgba = plt.cm.coolwarm(norm_w)
        link_colors.append(f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.8)')

    fig = go.Figure()
    fig.add_trace(go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=20,
            thickness=30,
            line=dict(color="black", width=0.25),
            label=display_labels,
            color=node_colors,
            x=node_x,
            y=node_y
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors)
    ))

    colorscale = [[i / 10, f"rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})"]
                  for i, c in enumerate([plt.cm.coolwarm(v) for v in np.linspace(0, 1, 11)])]

    fig.add_trace(go.Scatter(
        x=[None, None], y=[None, None], mode="markers",
        marker=dict(
            size=0.1, color=[vmin, vmax], cmin=vmin, cmax=vmax,
            colorscale=colorscale, showscale=True,
            colorbar=dict(title="Edge weight", x=1.02, y=0.55, len=0.95, thickness=18)
        ),
        hoverinfo="skip", showlegend=False
    ))

    for group in unique_groups:
        legend_name = "<br>".join(textwrap.wrap(group, width=30))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=12, color=group_colors[group], symbol='square'),
            legendgroup=group, showlegend=True, name=legend_name
        ))

    fig.update_layout(
        title_text=f"SPECTRA Functional Cascade: {target_gene}",
        title_x=0.5, font_size=12, height=800, plot_bgcolor='white',
        xaxis=dict(visible=False, showgrid=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5,
                    entrywidth=0.18, entrywidthmode="fraction")
    )
    fig.show()
    pio.write_image(fig, file=f'{out_dir}/{target_gene}_cascade_sankey.pdf', width=1500, height=800, scale=1)


def plot_cascade_logfc(network_genes, model, pseudobulk_pert, pseudobulk_ctrl, target_gene,
                       qvals=None, fdr_threshold=0.05, logfc_threshold=0.25, pseudocount=1e-3, out_dir='cascade_plots'):
    """Plots predicted pseudobulk log2FC for genes discovered in the cascade."""
    os.makedirs(out_dir, exist_ok=True)
    genes = [g for g in network_genes if g in model.gene_to_idx and g != target_gene]

    logfc = []
    significant = []
    for gene in genes:
        idx = model.gene_to_idx[gene]
        fc = np.log2((pseudobulk_pert[idx] + pseudocount) / (pseudobulk_ctrl[idx] + pseudocount))
        logfc.append(fc)
        if qvals is not None:
            significant.append((qvals[idx] < fdr_threshold) and (abs(fc) >= logfc_threshold))

    logfc = np.asarray(logfc)
    order = np.argsort(logfc)
    genes = np.asarray(genes)[order]
    logfc = logfc[order]
    significant = np.asarray(significant)[order] if qvals is not None else np.ones(len(logfc), dtype=bool)

    max_abs = max(np.max(np.abs(logfc)), 1e-8)
    colors = plt.cm.coolwarm(plt.Normalize(-max_abs, max_abs)(logfc))

    fig, ax = plt.subplots(figsize=(max(3, len(genes) * 0.22), 6))
    bars = ax.bar(genes, logfc, color=colors, edgecolor="black", linewidth=0.2)

    for bar, is_sig in zip(bars, significant):
        if not is_sig:
            bar.set_hatch("///")

    ax.set_ylabel(r"Predicted pseudobulk $\log_2$FC")
    ax.set_xlabel("Gene")
    ax.set_xticklabels(genes, rotation=90)
    ax.set_title(f"Predicted expression changes in {target_gene} cascade")

    if qvals is not None:
        ax.legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="///",
                                 label=f"Not significant (logFC<{logfc_threshold}, fdr>{fdr_threshold})")],
                  frameon=False)

    plt.tight_layout()
    plt.savefig(f'{out_dir}/{target_gene}_cascade_logfc.pdf')
    plt.show()


def plot_gate_vs_coexpression(target_gene, cascade_edges, y_hat, model, B, plot=True, out_dir='cascade_plots'):
    """
    Validates FAGCN gate weights against empirical co-expression correlation in predicted perturbed cells.
    """
    os.makedirs(out_dir, exist_ok=True)
    y_pert = y_hat.detach().cpu().reshape(B, model.num_nodes).numpy()

    gate_weights = []
    gene_corrs = []

    for src, _, tgt, _, w, _ in cascade_edges:
        i = model.gene_to_idx[src]
        j = model.gene_to_idx[tgt]
        r = np.corrcoef(y_pert[:, i], y_pert[:, j])[0, 1]
        if np.isfinite(r):
            gate_weights.append(w)
            gene_corrs.append(r)

    gate_weights = np.asarray(gate_weights)
    gene_corrs = np.asarray(gene_corrs)

    pearson_r, pearson_p = pearsonr(gate_weights, gene_corrs)
    spearman_r, spearman_p = spearmanr(gate_weights, gene_corrs)

    if plot:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(gate_weights, gene_corrs, alpha=0.75)
        ax.axhline(0, color="black", linewidth=0.5, linestyle='dashed')
        ax.axvline(0, color="black", linewidth=0.5, linestyle='dashed')
        ax.set_xlabel("FAGCN edge weight")
        ax.set_ylabel("Pearson correlation of connected genes\n(predicted perturbed cells)")
        ax.set_title(
            f"Gate weight vs gene-expression correlation\n"
            f"Pearson r = {pearson_r:.2f}, p = {pearson_p:.2e}\n"
            f"Spearman ρ = {spearman_r:.2f}, p = {spearman_p:.2e}"
        )
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{target_gene}_correlation_weights_vs_coexpression.pdf')
        plt.show()

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'n_edges': len(gate_weights)
    }


def plot_complete_path_dot_heatmaps(top_paths, target_gene, dot_scale=700, out_dir='cascade_plots'):
    """
    Generates dot-heatmaps displaying layer-wise complete paths and edge weights.
    """
    os.makedirs(out_dir, exist_ok=True)

    all_weights = []
    for path_length in [1, 2, 3]:
        for path in top_paths.get(path_length, []):
            all_weights.extend(path["weights"])

    max_abs = max(np.max(np.abs(all_weights)), 1e-8) if all_weights else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(18, 10), constrained_layout=True)

    for ax, path_length in zip(axes, [1, 2, 3]):
        paths = top_paths.get(path_length, [])
        if len(paths) == 0:
            ax.set_title(f"Top {path_length}-hop paths")
            ax.axis("off")
            continue

        path_labels = [" → ".join(path["genes"]) for path in paths]
        for row, path in enumerate(paths):
            for col, weight in enumerate(path["weights"]):
                dot_size = (abs(weight) / max_abs) * dot_scale
                ax.scatter(col, row, s=dot_size, c=[weight], cmap="coolwarm",
                           vmin=-max_abs, vmax=max_abs, edgecolor="black", linewidth=0.4)

        ax.set_xticks(np.arange(path_length))
        ax.set_xticklabels([f"Edge {i + 1}" for i in range(path_length)])
        ax.set_yticks(np.arange(len(paths)))
        ax.set_yticklabels(path_labels, fontsize=8)
        ax.set_title(f"Top {path_length}-hop paths")
        ax.invert_yaxis()
        ax.set_xticks(np.arange(-0.5, path_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(paths), 1), minor=True)
        ax.grid(which="minor", linewidth=0.4, alpha=0.25)
        ax.tick_params(which="minor", bottom=False, left=False)

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-max_abs, max_abs), cmap="coolwarm")
    cbar = fig.colorbar(sm, ax=axes, shrink=0.75, pad=0.02)
    cbar.set_label("Signed FAGCN edge weight")
    plt.savefig(f'{out_dir}/{target_gene}_cascade_paths.pdf')
    plt.show()



import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx

from schemas import FeatureConfig


def feature_dim(config: FeatureConfig) -> int:
    """Return total node feature dimension for a feature configuration."""
    dim = 0
    if config.use_degree:
        dim += 1
    if config.use_log_degree:
        dim += 1
    if config.use_normalised_degree:
        dim += 1
    if config.use_degree_bucket:
        dim += config.degree_bucket_bins
    if config.use_clustering:
        dim += 1
    if config.use_kcore:
        dim += 1
    if config.use_pagerank:
        dim += 1
    if config.use_laplacian_pe:
        dim += config.laplacian_pe_dim
    if config.use_rwse:
        dim += config.rwse_steps
    return dim


def save_feature_config(config: FeatureConfig, path: str | Path) -> None:
    """Save feature configuration to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config.model_dump(), handle, indent=2)


def _degree_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: FeatureConfig,
) -> list[np.ndarray]:
    """Compute degree-based structural features."""
    deg = degree(edge_index[0], num_nodes=num_nodes).numpy().astype(np.float64)
    max_deg = np.maximum(deg.max(), 1.0)
    features: list[np.ndarray] = []

    if config.use_degree:
        features.append(deg.reshape(-1, 1))

    if config.use_log_degree:
        features.append(np.log1p(deg).reshape(-1, 1))

    if config.use_normalised_degree:
        features.append((deg / max_deg).reshape(-1, 1))

    if config.use_degree_bucket:
        bins = np.linspace(0.0, max_deg, config.degree_bucket_bins + 1)
        bucket_idx = np.digitize(deg, bins[1:-1], right=True)
        one_hot = np.zeros((num_nodes, config.degree_bucket_bins), dtype=np.float64)
        one_hot[np.arange(num_nodes), bucket_idx] = 1.0
        features.append(one_hot)

    return features


def _graph_topology_features(
    graph: nx.Graph,
    num_nodes: int,
    config: FeatureConfig,
) -> list[np.ndarray]:
    """Compute topology-based structural features via NetworkX."""
    features: list[np.ndarray] = []

    if config.use_clustering:
        clustering = nx.clustering(graph)
        coeff = np.array(
            [clustering.get(node, 0.0) for node in range(num_nodes)],
            dtype=np.float64,
        )
        features.append(coeff.reshape(-1, 1))

    if config.use_kcore:
        core_numbers = nx.core_number(graph)
        kcore = np.array(
            [core_numbers.get(node, 0.0) for node in range(num_nodes)],
            dtype=np.float64,
        )
        max_core = np.maximum(kcore.max(), 1.0)
        features.append((kcore / max_core).reshape(-1, 1))

    if config.use_pagerank:
        pagerank = nx.pagerank(
            graph,
            alpha=config.pagerank_alpha,
            max_iter=config.pagerank_max_iter,
        )
        pr = np.array(
            [pagerank.get(node, 0.0) for node in range(num_nodes)],
            dtype=np.float64,
        )
        features.append(pr.reshape(-1, 1))

    return features


def _laplacian_positional_encodings(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: FeatureConfig,
) -> np.ndarray:
    """Compute normalised Laplacian eigenvector positional encodings."""
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    adjacency[row, col] = 1.0
    adjacency[col, row] = 1.0
    degree_vec = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(degree_vec, -0.5, where=degree_vec > 0)
    degree_inv_sqrt[degree_vec == 0] = 0.0
    normalised = (
        degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]
    )
    laplacian = np.eye(num_nodes, dtype=np.float64) - normalised
    k = min(config.laplacian_pe_dim + 1, num_nodes - 1)
    if k < 1:
        return np.zeros((num_nodes, config.laplacian_pe_dim), dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    order = np.argsort(eigenvalues)
    selected = eigenvectors[:, order[1 : k + 1]]
    if selected.shape[1] < config.laplacian_pe_dim:
        padding = np.zeros(
            (num_nodes, config.laplacian_pe_dim - selected.shape[1]),
            dtype=np.float64,
        )
        selected = np.concatenate([selected, padding], axis=1)
    return selected[:, : config.laplacian_pe_dim]


def _random_walk_structural_encodings(
    edge_index: torch.Tensor,
    num_nodes: int,
    config: FeatureConfig,
) -> np.ndarray:
    """Compute random-walk landing probabilities as structural encodings."""
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    adjacency[row, col] = 1.0
    adjacency[col, row] = 1.0
    degree_vec = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    nonzero = degree_vec > 0
    transition[nonzero] = adjacency[nonzero] / degree_vec[nonzero, None]
    landing = np.eye(num_nodes, dtype=np.float64)
    encodings = np.zeros((num_nodes, config.rwse_steps), dtype=np.float64)
    current = landing.copy()
    for step in range(config.rwse_steps):
        current = current @ transition
        encodings[:, step] = np.diag(current)
    return encodings


def compute_node_features(
    data: Data,
    config: FeatureConfig,
) -> torch.Tensor:
    """Build structural node features for a single graph."""
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    feature_blocks: list[np.ndarray] = []

    feature_blocks.extend(_degree_features(edge_index, num_nodes, config))

    needs_nx = (
        config.use_clustering or config.use_kcore or config.use_pagerank
    )
    if needs_nx:
        graph = to_networkx(data, to_undirected=True)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        feature_blocks.extend(
            _graph_topology_features(graph, num_nodes, config)
        )

    if config.use_laplacian_pe:
        feature_blocks.append(
            _laplacian_positional_encodings(edge_index, num_nodes, config)
        )

    if config.use_rwse:
        feature_blocks.append(
            _random_walk_structural_encodings(edge_index, num_nodes, config)
        )

    stacked = np.concatenate(feature_blocks, axis=1).astype(np.float32)
    return torch.from_numpy(stacked)

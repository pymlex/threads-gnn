import json
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from data.splits import build_splits_from_targets
from features.engineering import compute_node_features, save_feature_config
from schemas import DataConfig, ExperimentConfig, FeatureConfig


def _edges_to_undirected(edge_pairs: list[list[int]]) -> torch.Tensor:
    """Convert edge list to bidirectional COO edge_index."""
    source = []
    target = []
    for u, v in edge_pairs:
        source.extend([u, v])
        target.extend([v, u])
    edge_index = torch.tensor([source, target], dtype=torch.long)
    return edge_index


def _build_graph(
    graph_id: str,
    edge_pairs: list[list[int]],
    label: int,
    feature_config: FeatureConfig,
) -> Data:
    """Construct a single PyG Data object with structural node features."""
    edge_index = _edges_to_undirected(edge_pairs)
    num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 1
    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=torch.tensor([label], dtype=torch.long),
        graph_id=int(graph_id),
    )
    data.x = compute_node_features(data, feature_config)
    return data


def preprocess_dataset(config: ExperimentConfig) -> Path:
    """Preprocess raw SNAP data into sharded on-disk PyG graphs."""
    raw_dir = Path(config.data.raw_dir)
    processed_dir = Path(config.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_csv = raw_dir / "reddit_target.csv"
    edges_json = raw_dir / "reddit_edges.json"
    splits_path = processed_dir / "splits.json"
    manifest_path = processed_dir / "manifest.json"
    feature_config_path = processed_dir / "feature_config.json"

    if not splits_path.exists():
        build_splits_from_targets(
            target_csv,
            config.data,
            config.seed,
            splits_path,
        )

    save_feature_config(config.features, feature_config_path)

    if manifest_path.exists():
        return processed_dir

    labels = pd.read_csv(target_csv)
    label_map = dict(zip(labels["id"].astype(str), labels["target"].astype(int)))

    shard_size = config.data.shard_size
    shard_dir = processed_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "shards": [],
        "num_graphs": 0,
        "graph_index": [],
    }

    with edges_json.open("r", encoding="utf-8") as handle:
        edges_payload = json.load(handle)

    graph_ids = sorted(edges_payload.keys(), key=int)
    shard_buffer: list[Data] = []
    shard_index = 0
    global_index = 0

    for graph_id in tqdm(graph_ids, desc="Preprocessing graphs"):
        label = label_map[graph_id]
        graph = _build_graph(
            graph_id,
            edges_payload[graph_id],
            label,
            config.features,
        )
        shard_buffer.append(graph)
        manifest["graph_index"].append([shard_index, len(shard_buffer) - 1])
        global_index += 1

        if len(shard_buffer) >= shard_size:
            shard_name = f"shard_{shard_index:05d}.pt"
            torch.save(shard_buffer, shard_dir / shard_name)
            manifest["shards"].append(shard_name)
            manifest["num_graphs"] += len(shard_buffer)
            shard_buffer = []
            shard_index += 1

    if shard_buffer:
        shard_name = f"shard_{shard_index:05d}.pt"
        torch.save(shard_buffer, shard_dir / shard_name)
        manifest["shards"].append(shard_name)
        manifest["num_graphs"] += len(shard_buffer)

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return processed_dir

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from data.splits import build_splits_from_targets
from features.engineering import compute_node_features, save_feature_config
from schemas import DataConfig, ExperimentConfig, FeatureConfig


def _edges_to_undirected(edge_pairs: list[list[int]]) -> tuple[torch.Tensor, int]:
    """Convert edge list to bidirectional COO edge_index without self-loops."""
    source = []
    target = []
    max_node = 0
    for u, v in edge_pairs:
        max_node = max(max_node, u, v)
        if u == v:
            continue
        source.extend([u, v])
        target.extend([v, u])
    if len(source) == 0:
        num_nodes = max_node + 1
        return torch.zeros((2, 0), dtype=torch.long), num_nodes
    edge_index = torch.tensor([source, target], dtype=torch.long)
    num_nodes = max_node + 1
    return edge_index, num_nodes


def _build_graph(
    graph_id: str,
    edge_pairs: list[list[int]],
    label: int,
    feature_config: FeatureConfig,
) -> Data:
    """Construct a single PyG Data object with structural node features."""
    edge_index, num_nodes = _edges_to_undirected(edge_pairs)
    data = Data(
        edge_index=edge_index,
        num_nodes=num_nodes,
        y=torch.tensor([label], dtype=torch.long),
        graph_id=int(graph_id),
    )
    data.x = compute_node_features(data, feature_config)
    return data


def _build_graph_task(payload: tuple[str, list[list[int]], int, dict]) -> Data:
    """Worker task for parallel graph construction."""
    graph_id, edge_pairs, label, feature_payload = payload
    feature_config = FeatureConfig.model_validate(feature_payload)
    return _build_graph(graph_id, edge_pairs, label, feature_config)


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
    feature_payload = config.features.model_dump()
    tasks = [
        (graph_id, edges_payload[graph_id], label_map[graph_id], feature_payload)
        for graph_id in graph_ids
    ]

    shard_buffer: list[Data] = []
    shard_index = 0
    workers = config.data.preprocess_workers

    with ProcessPoolExecutor(max_workers=workers) as executor:
        graphs = executor.map(
            _build_graph_task,
            tasks,
            chunksize=64,
        )
        for graph in tqdm(graphs, total=len(tasks), desc="Preprocessing graphs"):
            shard_buffer.append(graph)
            manifest["graph_index"].append([shard_index, len(shard_buffer) - 1])

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

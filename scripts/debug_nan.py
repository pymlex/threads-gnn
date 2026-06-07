import argparse
import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm.auto import tqdm

from data.dataset import RedditThreadsDataset, ShardStore, collate_graphs
from features.engineering import (
    _degree_features,
    _graph_topology_features,
    _laplacian_positional_encodings,
    _random_walk_structural_encodings,
    feature_dim,
)
from models import build_model
from schemas import ExperimentConfig, FeatureConfig
from training.metrics import softmax_probabilities
from utils.config import load_config
from utils.pyg_check import require_torch_scatter
from utils.seed import set_seed


def _finite_summary(array: np.ndarray) -> dict[str, object]:
    """Summarise finite-value statistics for a numeric array."""
    finite = np.isfinite(array)
    return {
        "size": int(array.size),
        "finite_count": int(finite.sum()),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(np.nanmin(array)) if finite.any() else None,
        "max": float(np.nanmax(array)) if finite.any() else None,
    }


def _build_train_loader(config: ExperimentConfig) -> DataLoader:
    """Build the training dataloader used by GraphTrainer."""
    processed_dir = Path(config.data.processed_dir)
    splits_path = processed_dir / "splits.json"
    shard_store = ShardStore(processed_dir)
    train_dataset = RedditThreadsDataset(shard_store, "train", splits_path)
    return DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_graphs,
        pin_memory=torch.cuda.is_available(),
    )


def scan_dataset_features(
    config: ExperimentConfig,
    max_graphs: int | None,
) -> dict[str, object]:
    """Scan preprocessed node features for non-finite values."""
    processed_dir = Path(config.data.processed_dir)
    splits_path = processed_dir / "splits.json"
    shard_store = ShardStore(processed_dir)
    train_dataset = RedditThreadsDataset(shard_store, "train", splits_path)

    limit = len(train_dataset) if max_graphs is None else min(max_graphs, len(train_dataset))
    bad_graphs: list[dict[str, object]] = []
    node_counts: list[int] = []
    edge_counts: list[int] = []

    for index in tqdm(range(limit), desc="Scan features"):
        graph = train_dataset[index]
        node_counts.append(int(graph.num_nodes))
        edge_counts.append(int(graph.edge_index.shape[1]))
        features = graph.x.numpy()
        if np.isfinite(features).all():
            continue
        bad_graphs.append({
            "dataset_index": index,
            "graph_id": int(graph.graph_id),
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.edge_index.shape[1]),
            "feature_summary": _finite_summary(features),
            "bad_columns": np.where(~np.isfinite(features).all(axis=0))[0].tolist(),
        })

    return {
        "graphs_scanned": limit,
        "bad_graph_count": len(bad_graphs),
        "bad_graphs": bad_graphs[:20],
        "num_nodes": _finite_summary(np.array(node_counts, dtype=np.float64)),
        "num_edges": _finite_summary(np.array(edge_counts, dtype=np.float64)),
    }


def recompute_feature_blocks(
    config: ExperimentConfig,
    graph_indices: list[int],
) -> list[dict[str, object]]:
    """Recompute feature blocks for graphs with non-finite stored features."""
    processed_dir = Path(config.data.processed_dir)
    splits_path = processed_dir / "splits.json"
    shard_store = ShardStore(processed_dir)
    train_dataset = RedditThreadsDataset(shard_store, "train", splits_path)
    feature_config = config.features
    reports: list[dict[str, object]] = []

    for index in graph_indices:
        graph = train_dataset[index]
        raw = Data(
            edge_index=graph.edge_index,
            num_nodes=graph.num_nodes,
            y=graph.y,
            graph_id=graph.graph_id,
        )
        blocks: dict[str, dict[str, object]] = {}
        edge_index = raw.edge_index
        num_nodes = raw.num_nodes

        for name, block in [
            ("degree", np.concatenate(_degree_features(edge_index, num_nodes, feature_config), axis=1)),
            ("laplacian_pe", _laplacian_positional_encodings(edge_index, num_nodes, feature_config)),
            ("rwse", _random_walk_structural_encodings(edge_index, num_nodes, feature_config)),
        ]:
            blocks[name] = _finite_summary(block.astype(np.float64))

        graph_nx = to_networkx(raw, to_undirected=True)
        graph_nx.remove_edges_from(nx.selfloop_edges(graph_nx))
        topo = np.concatenate(
            _graph_topology_features(graph_nx, num_nodes, feature_config),
            axis=1,
        )
        blocks["topology"] = _finite_summary(topo.astype(np.float64))
        blocks["stored_x"] = _finite_summary(graph.x.numpy())

        reports.append({
            "dataset_index": index,
            "graph_id": int(graph.graph_id),
            "blocks": blocks,
        })

    return reports


def scan_forward_pass(
    config: ExperimentConfig,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
) -> dict[str, object]:
    """Run inference-only forward passes and locate non-finite logits."""
    set_seed(config.seed)
    input_dim = feature_dim(config.features)
    model = build_model(
        config.model.architecture,
        input_dim,
        config.model,
        loader,
    ).to(device)
    model.eval()

    batch_reports: list[dict[str, object]] = []
    first_bad: dict[str, object] | None = None
    elapsed = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Forward amp={use_amp}", total=max_batches)):
        if batch_idx >= max_batches:
            break
        start = time.perf_counter()
        batch = batch.to(device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                logits = model(batch)
        elapsed += time.perf_counter() - start

        logits_np = logits.float().detach().cpu().numpy()
        probabilities = softmax_probabilities(logits)
        x_np = batch.x.detach().cpu().numpy()
        report = {
            "batch_idx": batch_idx,
            "num_graphs": int(batch.num_graphs),
            "num_nodes": int(batch.num_nodes),
            "num_edges": int(batch.edge_index.shape[1]),
            "x": _finite_summary(x_np),
            "logits_fp32": _finite_summary(logits_np),
            "probabilities": _finite_summary(probabilities),
        }
        batch_reports.append(report)

        logits_bad = not np.isfinite(logits_np).all()
        prob_bad = not np.isfinite(probabilities).all()
        x_bad = not np.isfinite(x_np).all()
        if first_bad is None and (logits_bad or prob_bad or x_bad):
            graph_ids = [int(item.graph_id) for item in batch.to_data_list()]
            first_bad = {
                "batch_idx": batch_idx,
                "graph_ids": graph_ids,
                "x_bad": x_bad,
                "logits_bad": logits_bad,
                "probabilities_bad": prob_bad,
                "logits_fp32": _finite_summary(logits_np),
                "probabilities": _finite_summary(probabilities),
            }

    return {
        "architecture": config.model.architecture,
        "use_amp": use_amp,
        "batches_scanned": len(batch_reports),
        "seconds": elapsed,
        "seconds_per_batch": elapsed / max(len(batch_reports), 1),
        "first_bad_batch": first_bad,
        "batch_reports": batch_reports[:5],
    }


def scan_training_steps(
    config: ExperimentConfig,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    max_batches: int,
) -> dict[str, object]:
    """Run training updates and locate the first non-finite loss, logits, or weights."""
    set_seed(config.seed)
    input_dim = feature_dim(config.features)
    model = build_model(
        config.model.architecture,
        input_dim,
        config.model,
        loader,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    step_reports: list[dict[str, object]] = []
    first_bad: dict[str, object] | None = None
    elapsed = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Train amp={use_amp}", total=max_batches)):
        if batch_idx >= max_batches:
            break
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        labels = batch.y.view(-1)

        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()

        logits_np = logits.float().detach().cpu().numpy()
        probabilities = softmax_probabilities(logits)
        weight_finite = all(
            torch.isfinite(parameter).all().item()
            for parameter in model.parameters()
        )
        elapsed += time.perf_counter() - start

        report = {
            "batch_idx": batch_idx,
            "loss": float(loss.item()),
            "grad_norm": float(grad_norm),
            "logits_fp32": _finite_summary(logits_np),
            "probabilities": _finite_summary(probabilities),
            "weights_finite": weight_finite,
        }
        step_reports.append(report)

        if first_bad is None and (
            not np.isfinite(loss.item())
            or not np.isfinite(logits_np).all()
            or not np.isfinite(probabilities).all()
            or not weight_finite
        ):
            graph_ids = [int(item.graph_id) for item in batch.to_data_list()]
            first_bad = {
                "batch_idx": batch_idx,
                "graph_ids": graph_ids,
                "report": report,
            }

    return {
        "architecture": config.model.architecture,
        "use_amp": use_amp,
        "batches_scanned": len(step_reports),
        "seconds": elapsed,
        "seconds_per_batch": elapsed / max(len(step_reports), 1),
        "first_bad_step": first_bad,
        "step_reports": step_reports[:10],
    }


def estimate_epoch_time(
    config: ExperimentConfig,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    probe_batches: int,
) -> dict[str, object]:
    """Estimate full-epoch wall time from a short timed probe."""
    set_seed(config.seed)
    input_dim = feature_dim(config.features)
    model = build_model(
        config.model.architecture,
        input_dim,
        config.model,
        loader,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    start = time.perf_counter()
    batches_done = 0
    for batch_idx, batch in enumerate(loader):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        batch = batch.to(device)
        labels = batch.y.view(-1)
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            logits = model(batch)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        batches_done += 1
        if batches_done >= probe_batches:
            break

    elapsed = time.perf_counter() - start
    total_batches = len(loader)
    estimated_epoch_seconds = elapsed / max(batches_done, 1) * total_batches
    return {
        "train_graphs": len(loader.dataset),
        "batch_size": config.training.batch_size,
        "batches_per_epoch": total_batches,
        "probe_batches": batches_done,
        "probe_seconds": elapsed,
        "seconds_per_batch": elapsed / max(batches_done, 1),
        "estimated_epoch_minutes": estimated_epoch_seconds / 60.0,
        "use_amp": use_amp,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose NaN sources in training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["gin", "pna", "gat"],
        default="gin",
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=50000,
        help="Number of train graphs to scan for feature NaNs, 0 means all",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--probe-batches",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/debug_nan_report.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.model.architecture = args.architecture
    require_torch_scatter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_graphs = None if args.max_graphs == 0 else args.max_graphs
    loader = _build_train_loader(config)

    report: dict[str, object] = {
        "environment": {
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
            "architecture": args.architecture,
            "batch_size": config.training.batch_size,
            "use_amp_default": config.training.use_amp,
            "train_graphs": len(loader.dataset),
            "batches_per_epoch": len(loader),
        },
        "feature_scan": scan_dataset_features(config, max_graphs),
        "forward_amp_true": scan_forward_pass(
            config,
            loader,
            device,
            use_amp=True,
            max_batches=args.max_batches,
        ),
        "forward_amp_false": scan_forward_pass(
            config,
            loader,
            device,
            use_amp=False,
            max_batches=args.max_batches,
        ),
        "training_amp_true": scan_training_steps(
            config,
            loader,
            device,
            use_amp=True,
            max_batches=args.max_batches,
        ),
        "training_amp_false": scan_training_steps(
            config,
            loader,
            device,
            use_amp=False,
            max_batches=args.max_batches,
        ),
        "epoch_time_amp_true": estimate_epoch_time(
            config,
            loader,
            device,
            use_amp=True,
            probe_batches=args.probe_batches,
        ),
        "epoch_time_amp_false": estimate_epoch_time(
            config,
            loader,
            device,
            use_amp=False,
            probe_batches=args.probe_batches,
        ),
    }

    feature_scan = report["feature_scan"]
    bad_graphs = feature_scan["bad_graphs"]
    if bad_graphs:
        indices = [int(item["dataset_index"]) for item in bad_graphs[:5]]
        report["feature_block_recompute"] = recompute_feature_blocks(config, indices)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()

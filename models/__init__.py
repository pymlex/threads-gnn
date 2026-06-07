import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import PNAConv

from models.gat import GATClassifier
from models.gin import GINClassifier
from models.pna import PNAClassifier
from schemas import ModelConfig


def build_model(
    architecture: str,
    input_dim: int,
    config: ModelConfig,
    train_loader: DataLoader | None = None,
) -> torch.nn.Module:
    """Instantiate a graph classifier by architecture name."""
    if architecture == "gin":
        return GINClassifier(input_dim, config)
    if architecture == "gat":
        return GATClassifier(input_dim, config)
    if architecture == "pna":
        deg_histogram = compute_degree_histogram(train_loader)
        return PNAClassifier(input_dim, config, deg_histogram)
    raise ValueError(f"Unknown architecture: {architecture}")


def compute_degree_histogram(loader: DataLoader) -> torch.Tensor:
    """Compute in-degree histogram from a training DataLoader."""
    return PNAConv.get_degree_histogram(loader)

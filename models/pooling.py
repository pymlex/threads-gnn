import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


class AttentionPooling(nn.Module):
    """Graph-level attention pooling over node embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Pool node embeddings with learned attention weights."""
        scores = self.gate(node_embeddings).squeeze(-1)
        weights = softmax(scores, batch)
        weighted = node_embeddings * weights.unsqueeze(-1)
        return global_add_pool(weighted, batch)


def build_pooling(
    pooling: str,
    hidden_dim: int,
) -> nn.Module | str:
    """Return pooling module or identifier for global pooling."""
    if pooling == "mean":
        return "mean"
    if pooling == "sum":
        return "sum"
    if pooling == "attention":
        return AttentionPooling(hidden_dim)
    raise ValueError(f"Unknown pooling method: {pooling}")


def apply_pooling(
    node_embeddings: torch.Tensor,
    batch: torch.Tensor,
    pooling: nn.Module | str,
) -> torch.Tensor:
    """Apply graph-level pooling to node embeddings."""
    if pooling == "mean":
        return global_mean_pool(node_embeddings, batch)
    if pooling == "sum":
        return global_add_pool(node_embeddings, batch)
    return pooling(node_embeddings, batch)

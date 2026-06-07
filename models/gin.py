import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, LayerNorm

from models.base import GraphClassifierBase
from schemas import ModelConfig


class GINBlock(nn.Module):
    """GIN convolution block with MLP, residual connection, and normalisation."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINConv(mlp, eps=eps, train_eps=False)
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply GIN message passing with residual connection."""
        updated = self.conv(node_embeddings, edge_index)
        updated = self.norm(updated)
        updated = torch.relu(updated)
        updated = self.dropout(updated)
        return node_embeddings + updated


class GINClassifier(GraphClassifierBase):
    """Graph Isomorphism Network classifier."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__(input_dim, config)
        self.layers = nn.ModuleList([
            GINBlock(config.hidden_dim, config.dropout, config.gin_eps)
            for _ in range(config.num_layers)
        ])

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        """Encode node embeddings with stacked GIN layers."""
        node_embeddings = self.input_proj(batch.x)
        virtual_state = None
        for layer in self.layers:
            node_embeddings = layer(node_embeddings, batch.edge_index)
            node_embeddings, virtual_state = self.apply_virtual_node(
                node_embeddings,
                batch,
                virtual_state,
            )
        return node_embeddings

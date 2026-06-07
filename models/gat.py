import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, LayerNorm

from models.base import GraphClassifierBase
from schemas import ModelConfig


class GATBlock(nn.Module):
    """Multi-head GAT block with residual connection and normalisation."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        is_last: bool,
    ) -> None:
        super().__init__()
        out_channels = hidden_dim if is_last else hidden_dim // num_heads
        concat = not is_last
        self.conv = GATConv(
            in_channels=hidden_dim,
            out_channels=out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=True,
            negative_slope=0.2,
        )
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.is_last = is_last

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply multi-head graph attention with residual connection."""
        updated = self.conv(node_embeddings, edge_index)
        if not self.is_last:
            updated = updated.view(node_embeddings.shape[0], -1)
        updated = self.norm(updated)
        updated = torch.nn.functional.elu(updated)
        updated = self.dropout(updated)
        return node_embeddings + updated


class GATClassifier(GraphClassifierBase):
    """Graph Attention Network classifier."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__(input_dim, config)
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            is_last = layer_idx == config.num_layers - 1
            self.layers.append(
                GATBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.dropout,
                    is_last,
                )
            )

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        """Encode node embeddings with stacked GAT layers."""
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

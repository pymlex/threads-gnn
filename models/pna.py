import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import LayerNorm, PNAConv

from models.base import GraphClassifierBase
from schemas import ModelConfig


class PNABlock(nn.Module):
    """PNA convolution block with residual connection and normalisation."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        aggregators: list[str],
        scalers: list[str],
        deg_histogram: torch.Tensor,
    ) -> None:
        super().__init__()
        self.conv = PNAConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg_histogram,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
        )
        self.norm = LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Apply PNA message passing with residual connection."""
        updated = self.conv(node_embeddings, edge_index)
        updated = self.norm(updated)
        updated = torch.relu(updated)
        updated = self.dropout(updated)
        return node_embeddings + updated


class PNAClassifier(GraphClassifierBase):
    """Principal Neighbourhood Aggregation classifier."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
        deg_histogram: torch.Tensor,
    ) -> None:
        super().__init__(input_dim, config)
        self.layers = nn.ModuleList([
            PNABlock(
                config.hidden_dim,
                config.dropout,
                config.pna_aggregators,
                config.pna_scalers,
                deg_histogram,
            )
            for _ in range(config.num_layers)
        ])

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        """Encode node embeddings with stacked PNA layers."""
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

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from models.pooling import apply_pooling, build_pooling
from models.virtual_node import VirtualNodeModule
from schemas import ModelConfig


class GraphClassifierBase(nn.Module):
    """Shared graph classifier backbone with projection, pooling, and MLP head."""

    def __init__(
        self,
        input_dim: int,
        config: ModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
        )
        self.pooling = build_pooling(config.pooling, config.hidden_dim)
        self.virtual_node = (
            VirtualNodeModule(config.hidden_dim, config.dropout)
            if config.use_virtual_node
            else None
        )
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, 2),
        )

    def encode_graph(self, batch: Batch) -> torch.Tensor:
        """Encode batched graphs into node embeddings."""
        raise NotImplementedError

    def apply_virtual_node(
        self,
        node_embeddings: torch.Tensor,
        batch: Batch,
        virtual_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply one virtual-node update step inside the encoder."""
        if self.virtual_node is None:
            return node_embeddings, None
        if virtual_state is None:
            virtual_state = self.virtual_node.init_state(
                batch.batch,
                node_embeddings.device,
            )
        virtual_state = self.virtual_node.update(
            node_embeddings,
            virtual_state,
            batch.batch,
        )
        virtual_state = self.virtual_node.transform(virtual_state)
        node_embeddings = self.virtual_node.broadcast(
            node_embeddings,
            virtual_state,
            batch.batch,
        )
        return node_embeddings, virtual_state

    def forward(self, batch: Batch) -> torch.Tensor:
        """Return unnormalised class logits for a graph batch."""
        node_embeddings = self.encode_graph(batch)
        graph_embedding = apply_pooling(
            node_embeddings,
            batch.batch,
            self.pooling,
        )
        return self.classifier(graph_embedding)

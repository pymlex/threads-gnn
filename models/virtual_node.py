import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class VirtualNodeModule(nn.Module):
    """Virtual node mechanism for batched graph encoders."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(1, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def init_state(self, batch: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Initialise per-graph virtual node embeddings."""
        num_graphs = int(batch.max().item()) + 1
        indices = torch.zeros(num_graphs, dtype=torch.long, device=device)
        return self.embedding(indices)

    def update(
        self,
        node_embeddings: torch.Tensor,
        virtual_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Update virtual node states from pooled node embeddings."""
        pooled = global_add_pool(node_embeddings, batch)
        return virtual_embeddings + pooled

    def broadcast(
        self,
        node_embeddings: torch.Tensor,
        virtual_embeddings: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast virtual node embeddings to real nodes."""
        virtual_broadcast = virtual_embeddings[batch]
        return node_embeddings + virtual_broadcast

    def transform(self, virtual_embeddings: torch.Tensor) -> torch.Tensor:
        """Apply MLP transformation to virtual node embeddings."""
        return self.mlp(virtual_embeddings)

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm.auto import tqdm

from data.splits import load_splits


class ShardStore:
    """Shared on-disk shard cache for train, validation, and test splits."""

    def __init__(self, processed_dir: str | Path) -> None:
        processed_dir = Path(processed_dir)
        manifest_path = processed_dir / "manifest.json"

        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        self.shard_dir = processed_dir / "shards"
        self.shard_names = manifest["shards"]
        self.graph_index = manifest["graph_index"]
        self._buffers: dict[int, list[Data]] = {}

    def preload_shards(self, shard_ids: set[int], desc: str) -> None:
        """Load shard files required by a split."""
        missing = sorted(shard_id for shard_id in shard_ids if shard_id not in self._buffers)
        for shard_id in tqdm(missing, desc=desc):
            shard_name = self.shard_names[shard_id]
            self._buffers[shard_id] = torch.load(
                self.shard_dir / shard_name,
                weights_only=False,
            )

    def get_graph(self, global_idx: int) -> Data:
        """Return one preprocessed graph by global index."""
        shard_id, local_idx = self.graph_index[int(global_idx)]
        return self._buffers[shard_id][local_idx]


class RedditThreadsDataset(Dataset):
    """Dataset for Reddit Threads graphs with shared shard preloading."""

    def __init__(
        self,
        shard_store: ShardStore,
        split: str,
        splits_path: str | Path,
    ) -> None:
        self.shard_store = shard_store
        self.split_indices = load_splits(Path(splits_path))[split]
        shard_ids = {
            self.shard_store.graph_index[int(global_idx)][0]
            for global_idx in self.split_indices
        }
        self.shard_store.preload_shards(shard_ids, f"Loading {split} shards")

    def __len__(self) -> int:
        return len(self.split_indices)

    def __getitem__(self, index: int) -> Data:
        global_idx = int(self.split_indices[index])
        return self.shard_store.get_graph(global_idx)


def collate_graphs(batch: list[Data]) -> Batch:
    """Collate a list of graphs into a PyG batch."""
    return Batch.from_data_list(batch)


def load_split_indices(processed_dir: str | Path) -> dict[str, np.ndarray]:
    """Load train, validation, and test split indices."""
    return load_splits(Path(processed_dir) / "splits.json")

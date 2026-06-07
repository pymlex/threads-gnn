---
language: en
license: gpl-3.0
tags:
  - graph-neural-networks
  - pytorch-geometric
  - graph-classification
  - snap
datasets:
  - reddit_threads
metrics:
  - matthews_correlation
  - accuracy
  - f1
  - roc-auc
library_name: pytorch
pipeline_tag: graph-ml
---

# pymlex/threads-gnn

Graph classification model for the [SNAP Reddit Threads](https://snap.stanford.edu/data/reddit_threads.html) dataset. The dataset was collected in May 2018. Nodes are Reddit users and undirected edges are reply relations. The task is binary graph classification.

## Model description

Three encoders are compared under an identical protocol: GIN, PNA, and GAT. Model selection uses validation Matthews correlation coefficient. Structural node features are engineered because the dataset provides no raw node attributes.

## Training data

Source: [SNAP Reddit Threads](https://snap.stanford.edu/data/reddit_threads.html)

- 203,088 graphs
- binary labels
- 11–97 nodes per graph
- no node or edge features

## Usage

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(repo_id="pymlex/threads-gnn", filename="model.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

Full training and evaluation commands are in the [GitHub repository](https://github.com/pymlex/threads-gnn).

## Citation

```bibtex
@misc{threads_gnn,
  author = {Alex Zyukov},
  title = {Graph Classification on SNAP Reddit Threads},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/pymlex/threads-gnn}},
}
```

```bibtex
@inproceedings{karateclub,
  title = {{Karate Club: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs}},
  author = {Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
  year = {2020},
  pages = {3125--3132},
  booktitle = {Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
  organization = {ACM},
}
```

The project is under GPL-3.0 license.

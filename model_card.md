---
license: gpl-3.0
tags:
  - graph-neural-networks
  - pytorch-geometric
  - graph-classification
  - snap
language:
  - en
metrics:
  - matthews_correlation
  - accuracy
  - f1
  - roc-auc
library_name: pytorch
pipeline_tag: graph-ml
datasets:
  - reddit_threads
---

# Graph Classification on SNAP Reddit Threads

Binary graph classification on [SNAP Reddit Threads](https://snap.stanford.edu/data/reddit_threads.html) with PyTorch Geometric. Full training code, preprocessing, metric tables, figures, and reproduction commands:

https://github.com/pymlex/threads-gnn

## Overview

Nodes are Reddit users in a discussion thread. Undirected edges are reply relations. The label marks whether the thread is discussion-based. The dataset has 203,088 graphs, 11--97 nodes per graph, and no raw node features. We engineer 38 structural descriptors per node and compare three encoders under one protocol: GIN, PNA, and GAT. Each model uses four message-passing layers, hidden dimension 128, attention pooling, and a virtual node. Model selection uses validation Matthews correlation coefficient. Experiments used Google Colab with an NVIDIA GPU, batch size 4096, learning rate 0.003, and early stopping with patience 8.

## Results

### Architecture comparison

| Architecture | Best val MCC | Test MCC | Test F1 | Test ROC-AUC |
| --- | --- | --- | --- | --- |
| GIN | 0.5609 | 0.5642 | 0.8017 | 0.8417 |
| PNA | 0.5609 | 0.5635 | 0.8016 | 0.8419 |
| GAT | 0.5592 | 0.5655 | 0.8002 | 0.8418 |

**Selected checkpoint: GIN** (`model.pt`), chosen by best validation MCC. GIN leads validation MCC by a margin of 6e-5 over PNA. On the held-out test split GAT reaches the highest MCC 0.5655, while ROC-AUC stays near 0.842 for all three encoders.

### Training curves

![Training curves for GIN, PNA, and GAT](https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/training_curves.png)

Validation MCC rises in the first five epochs and plateaus near 0.55--0.56 for every encoder. Best checkpoints appear at epoch 31 for GIN, epoch 23 for PNA, and epoch 32 for GAT.

### Test ROC curves

![Combined test ROC curves](https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/test_roc_curves.png)

ROC-AUC on the test split: GIN 0.8417, PNA 0.8419, GAT 0.8418. The three curves overlap across the full false-positive range. Each model reaches true positive rate 0.80 near false positive rate 0.25. Encoder choice affects thresholded metrics more than ranking quality.

Per-architecture plots:

- GIN: https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/gin_seed42/test_roc_curve.png
- PNA: https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/pna_seed42/test_roc_curve.png
- GAT: https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/gat_seed42/test_roc_curve.png

### Logit distributions on the test split

![Combined logit histograms for class 1](https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/test_logit_histograms.png)

Histograms show the class-1 logit density split by ground-truth label.

GIN separates the two classes into negative and positive modes, with medians 0.189 for class 0 and 0.815 for class 1. PNA concentrates true class 1 into a narrow spike near logit 0.3 while spreading class 0 across negative values. GAT keeps the largest overlap near logit 0, which lowers the false-positive rate on class 0 to 30.0% but raises the false-negative rate on class 1 to 14.2%.

- GIN: https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/gin_seed42/test_logit_histogram.png
- PNA: https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/pna_seed42/test_logit_histogram.png
- GAT: https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/gat_seed42/test_logit_histogram.png

### Confusion matrices on the test split

#### GIN

![GIN test confusion matrix](https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/gin_seed42/test_confusion_matrix.png)

#### PNA

![PNA test confusion matrix](https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/pna_seed42/test_confusion_matrix.png)

#### GAT

![GAT test confusion matrix](https://raw.githubusercontent.com/pymlex/threads-gnn/main/runs/gat_seed42/test_confusion_matrix.png)

All models favour recall on the positive class. Class 0 recall stays near 0.67--0.70 while class 1 recall exceeds 0.85. GAT yields the highest class-0 recall 0.700 and test accuracy 0.781.

### Selected model test metrics (GIN)

| Metric | Value |
| --- | --- |
| MCC | 0.5642 |
| Accuracy | 0.7783 |
| Balanced accuracy | 0.7758 |
| Precision | 0.7400 |
| Recall | 0.8745 |
| F1 | 0.8017 |
| ROC-AUC | 0.8417 |
| PR-AUC | 0.8087 |

## Checkpoint

| File | Description |
| --- | --- |
| `model.pt` | Best GIN checkpoint selected by validation MCC |
| `config.json` | Experiment configuration for the selected run |
| `final_metrics.json` | Validation and test metrics for the selected run |
| `selected_model.json` | Architecture comparison and selection record |

## Inference

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(repo_id="pymlex/threads-gnn", filename="model.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model_state_dict"]
```

Clone https://github.com/pymlex/threads-gnn for the full `GINClassifier` definition, structural feature pipeline, and batched inference over PyG `Data` objects.

## Citation

```bibtex
@misc{threads_gnn,
  author = {Alex Zyukov},
  title = {Graph Classification on SNAP Reddit Threads},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/pymlex/threads-gnn}},
  note = {Hugging Face model pymlex/threads-gnn}
}
```

## References

```bibtex
@inproceedings{karateclub,
  title = {{Karate Club: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs}},
  author = {Benedek Rozemberczki and Oliver Kiss and Rik Sarkar},
  year = {2020},
  pages = {3125--3132},
  booktitle = {Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20)},
  organization = {ACM},
}
@inproceedings{xu2019gin,
  title = {How Powerful are Graph Neural Networks?},
  author = {Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
}
@inproceedings{corso2020pna,
  title = {Principal Neighbourhood Aggregation for Graph Nets},
  author = {Gabriele Corso and Luca Cavalleri and Dominique Beaini and Pietro Li and Petar Velickovic},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
}
@inproceedings{velickovic2018gat,
  title = {Graph Attention Networks},
  author = {Petar Velickovic and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Li and Yoshua Bengio},
  booktitle = {International Conference on Learning Representations},
  year = {2018},
}
```

The project is under GPL-3.0 license.

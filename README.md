# Graph Classification on SNAP Reddit Threads

End-to-end graph classification on the [SNAP Reddit Threads](https://snap.stanford.edu/data/reddit_threads.html) dataset with PyTorch Geometric. Three encoders are compared under an identical protocol: GIN, PNA, and GAT. Model selection uses validation Matthews correlation coefficient. Structural node features are engineered because the dataset provides no raw node attributes.

## Dataset

The source is [SNAP Reddit Threads](https://snap.stanford.edu/data/reddit_threads.html). The graphs were collected in **May 2018**. Nodes are Reddit users who participate in a thread. Undirected edges are reply relations between users. The task is binary graph classification: predict whether a thread is discussion-based.

| Property | Value |
|---|---|
| Number of graphs | 203,088 |
| Directed | No |
| Node features | No |
| Edge features | No |
| Graph labels | Binary |
| Temporal | No |
| Nodes per graph | 11–97 |
| Density | 0.021–0.382 |
| Diameter | 2–27 |

Raw files:

- `reddit_target.csv` with columns `id`, `target`
- `reddit_edges.json` mapping graph id to edge lists

## Repository structure

```
threads-gnn/
├── configs/default.yaml
├── .env.example
├── scripts/install.sh
├── data/
├── features/
├── models/
├── training/
├── scripts/
├── schemas.py
└── main.py
```

## Google Colab

Open a terminal in Colab and run the commands below.

### Clone

```bash
git clone https://github.com/pymlex/threads-gnn.git
cd threads-gnn
```

### Install

Creates `.env` from `.env.example`, reinstalls PyG wheels matched to the Colab PyTorch build, and verifies `torch-scatter`. GitHub authentication runs only when `gh` is not already logged in.

```bash
bash scripts/install.sh
```

Edit `.env` and set `HF_TOKEN`. Optional fields: `GITHUB_NAME`, `GITHUB_EMAIL`.

### Preprocess

Downloads SNAP data and builds sharded processed graphs with structural features.

```bash
python scripts/preprocess.py --config configs/default.yaml
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Experiment configuration path |

### Train

Trains GIN, PNA, and GAT by default with identical splits and hyperparameters.

```bash
python scripts/train.py --config configs/default.yaml
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Experiment configuration path |
| `--architecture` | `all` | `all`, `gin`, `pna`, or `gat` |
| `--pooling` | from config | `mean`, `sum`, or `attention` |

### Compare architectures

Ranks models by validation MCC and writes `runs/selected_model.json`.

```bash
python scripts/compare.py
```

| Argument | Default | Description |
|---|---|---|
| `--runs-dir` | `runs` | Directory with run outputs |
| `--seed` | `42` | Random seed in run folder names |

### Evaluate

Evaluates all best checkpoints on the test split by default.

```bash
python scripts/eval.py --config configs/default.yaml
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Experiment configuration path |
| `--architecture` | `all` | `all`, `gin`, `pna`, or `gat` |
| `--checkpoint` | auto | Path for a single-architecture run |
| `--split` | `test` | `train`, `val`, or `test` |
| `--seed` | `42` | Seed used in checkpoint filenames |

### Plot training curves

```bash
python scripts/plot_curves.py
```

| Argument | Default | Description |
|---|---|---|
| `--runs-dir` | `runs` | Directory with epoch metrics |
| `--seed` | `42` | Random seed in run folder names |

### Full pipeline

```bash
python scripts/run_all.py --config configs/default.yaml
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Experiment configuration path |

### Push to Hugging Face

Reads `HF_TOKEN` from `.env`.

```bash
python scripts/push_hf.py --repo-id pymlex/threads-gnn
```

| Argument | Default | Description |
|---|---|---|
| `--repo-id` | `pymlex/threads-gnn` | Hugging Face model repository |
| `--runs-dir` | `runs` | Directory with experiment outputs |
| `--checkpoints-dir` | `checkpoints` | Checkpoint directory |
| `--seed` | `42` | Random seed in checkpoint filenames |

## Structural node features

Because the dataset has no node features, each graph receives engineered structural descriptors controlled by `FeatureConfig` in `schemas.py`. All features are enabled by default. The exact configuration is saved to `data/processed/feature_config.json` during preprocessing.

For a graph $G = (V, E)$ with $n = |V|$ nodes and adjacency matrix $A$, node $i$ receives the following descriptors when enabled.

**Degree**

$$d_i = \sum_{j=1}^{n} A_{ij}$$

**Log degree**

$$\log(1 + d_i)$$

**Normalised degree**

$$\tilde{d}_i = \frac{d_i}{\max_{k \in V} d_k}$$

**Degree bucket embedding**

One-hot encoding of $d_i$ over $B$ uniform bins on $[0, \max_k d_k]$.

**Clustering coefficient**

$$c_i = \frac{2|T_i|}{d_i(d_i - 1)}$$

where $T_i$ is the set of triangles containing node $i$.

**K-core number**

The core number $\kappa_i$ from the $k$-core decomposition, normalised by $\max_k \kappa_k$.

**PageRank**

$$\mathbf{pr} = \alpha \mathbf{P}^{\top} \mathbf{pr} + (1 - \alpha)\frac{\mathbf{1}}{n}$$

with $\alpha = 0.85$.

**Laplacian positional encodings**

Let $L = I - D^{-1/2} A D^{-1/2}$ be the normalised Laplacian. The smallest $k$ non-trivial eigenvectors of $L$ form an $n \times k$ matrix used as positional encodings.

**Random-walk structural encodings**

Let $P = D^{-1}A$ be the random-walk transition matrix and $R^{(t)} = P^t$. The diagonal entries $R^{(t)}_{ii}$ for $t = 1, \ldots, T$ form RWSE features.

With the default configuration, the input dimension is $38$.

## Graph encoders

All models share the same backbone:

1. input projection $\mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d}$
2. graph encoder with $L$ message-passing layers
3. graph-level pooling
4. MLP classifier head

Optional virtual node updates are applied after every encoder layer. For batch $B$ with graph indices $b(i)$:

$$\mathbf{v}_g \leftarrow \mathrm{MLP}\left(\mathbf{v}_g + \sum_{i:\, b(i)=g} \mathbf{h}_i\right), \qquad \mathbf{h}_i \leftarrow \mathbf{h}_i + \mathbf{v}_{b(i)}$$

### GIN

Graph Isomorphism Network convolution with MLP $\phi$ and neighbourhood aggregation $\mathcal{N}(i)$:

$$\mathbf{h}_i^{(\ell+1)} = \mathrm{ReLU}\left(\mathrm{BN}\left((1+\varepsilon)\mathbf{h}_i^{(\ell)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(\ell)}\right)\right)$$

with residual connection $\mathbf{h}_i^{(\ell+1)} \leftarrow \mathbf{h}_i^{(\ell)} + \mathbf{h}_i^{(\ell+1)}$.

### PNA

Principal Neighbourhood Aggregation combines aggregators $\mu$, $\max$, $\min$, $\sigma$ with degree scalers $S(\mathbf{D}, \alpha)$:

$$\mathbf{h}_i^{(\ell+1)} = \gamma_{\Theta}\left(\mathbf{h}_i^{(\ell)}, \bigoplus_{j \in \mathcal{N}(i)} h_{\Theta}(\mathbf{h}_i^{(\ell)}, \mathbf{h}_j^{(\ell)})\right)$$

The operator $\bigoplus$ applies identity, amplification, and attenuation scalers to mean, max, min, and standard deviation aggregators. The in-degree histogram is computed on the training split only.

### GAT

Multi-head graph attention with leaky ReLU scoring:

$$e_{ij} = \mathrm{LeakyReLU}\left(\mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)$$

Intermediate layers concatenate heads. The final layer averages head outputs to keep the hidden dimension fixed.

## Graph-level pooling

Three pooling operators are implemented.

**Global mean pooling**

$$\mathbf{g} = \frac{1}{|V|}\sum_{i \in V} \mathbf{h}_i$$

**Global sum pooling**

$$\mathbf{g} = \sum_{i \in V} \mathbf{h}_i$$

**Attention pooling**

$$s_i = \mathbf{w}^{\top}\tanh(\mathbf{W}\mathbf{h}_i), \qquad \alpha_i = \frac{\exp(s_i)}{\sum_{j \in V}\exp(s_j)}, \qquad \mathbf{g} = \sum_{i \in V} \alpha_i \mathbf{h}_i$$

All three architectures use the same pooling method from `configs/default.yaml`.

## Training protocol

- stratified train, validation, and test split with ratios $0.8 / 0.1 / 0.1$
- random seed $42$
- AdamW optimiser with learning rate $10^{-3}$ and weight decay $10^{-4}$
- cosine learning-rate schedule
- mixed-precision training on GPU
- gradient clipping with max norm $1.0$
- early stopping on validation MCC with patience $20$
- batch size $512$ for Colab T4

The test split is never used for model selection. Architectures are ranked by best validation MCC. Test metrics for the selected architecture are reported once after training.

Per-epoch metrics are logged with `tqdm` and saved to `runs/<architecture>_seed42/epoch_metrics.csv`. Final metrics are saved to `runs/<architecture>_seed42/final_metrics.json`.

### Primary metrics

Matthews correlation coefficient:

$$\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Additional metrics: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, classification report.

## Results

Values below are filled after the Colab run.

### Architecture comparison

| Architecture | Best val MCC | Val F1 | Val ROC-AUC | Test MCC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| GIN | — | — | — | — | — | — |
| PNA | — | — | — | — | — | — |
| GAT | — | — | — | — | — | — |

### Training curves

![Training curves](runs/training_curves.png)

### Confusion matrices

GIN test confusion matrix: `runs/gin_seed42/test_confusion_matrix.png`

PNA test confusion matrix: `runs/pna_seed42/test_confusion_matrix.png`

GAT test confusion matrix: `runs/gat_seed42/test_confusion_matrix.png`

### Selected model test metrics

Filled from `runs/selected_model.json` and the corresponding `final_metrics.json` after model selection.

## Model weights

Best checkpoint: [pymlex/threads-gnn](https://huggingface.co/pymlex/threads-gnn)

```python
from huggingface_hub import hf_hub_download
import torch

checkpoint_path = hf_hub_download(repo_id="pymlex/threads-gnn", filename="model.pt")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

## Default hyperparameters

| Parameter | Value |
|---|---|
| hidden_dim | 128 |
| num_layers | 4 |
| dropout | 0.2 |
| num_heads | 4 |
| batch_size | 512 |
| learning_rate | $10^{-3}$ |
| weight_decay | $10^{-4}$ |
| early_stopping_patience | 20 |
| virtual_node | enabled |
| pooling | attention |

## References

```bibtex
@misc{threads_gnn,
  author = {Alex Zyukov},
  title = {Graph Classification on SNAP Reddit Threads},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/pymlex/threads-gnn}},
}
```

The project is under GPL-3.0 license.

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

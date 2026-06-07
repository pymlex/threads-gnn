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

### Dataset citation

B. Rozemberczki, O. Kiss, R. Sarkar: An API Oriented Open-source Python Framework for Unsupervised Learning on Graphs 2019.

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

## Repository structure

```
threads-gnn/
├── configs/default.yaml
├── data/
│   ├── download.py
│   ├── preprocess.py
│   ├── dataset.py
│   └── splits.py
├── features/engineering.py
├── models/
│   ├── base.py
│   ├── gin.py
│   ├── pna.py
│   ├── gat.py
│   ├── pooling.py
│   └── virtual_node.py
├── training/
│   ├── trainer.py
│   └── metrics.py
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   ├── eval.py
│   ├── compare.py
│   ├── compare_pooling.py
│   ├── ablation.py
│   ├── plot_curves.py
│   ├── push_hf.py
│   └── run_all.py
├── schemas.py
├── main.py
└── requirements.txt
```

## Google Colab setup

```bash
!git clone https://github.com/pymlex/threads-gnn.git
%cd threads-gnn
```

```bash
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
!pip install -q torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
!pip install -q -r requirements.txt
!pip install -q -e .
```

```bash
!python scripts/preprocess.py --config configs/default.yaml
```

```bash
!python scripts/compare_pooling.py --config configs/default.yaml --architecture gin
```

Update `configs/default.yaml` with the pooling method selected by validation MCC, then train all three architectures:

```bash
!python scripts/train.py --config configs/default.yaml --architecture gin
!python scripts/train.py --config configs/default.yaml --architecture pna
!python scripts/train.py --config configs/default.yaml --architecture gat
```

```bash
!python scripts/compare.py
!python scripts/plot_curves.py
```

Evaluate the selected checkpoint on the test split:

```bash
!python scripts/eval.py --config configs/default.yaml --checkpoint checkpoints/<architecture>_seed42_best.pt --split test
```

Upload the selected model to Hugging Face:

```bash
!export HF_TOKEN=<your_token>
!python scripts/push_hf.py --repo-id pymlex/threads-gnn
```

Run the full pipeline in one command:

```bash
!python scripts/run_all.py --config configs/default.yaml
```

Feature ablation:

```bash
!python scripts/ablation.py --config configs/default.yaml --architecture gin
```

## Local setup

```bash
git clone https://github.com/pymlex/threads-gnn.git
cd threads-gnn
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio
pip install torch-geometric torch-scatter torch-sparse
pip install -r requirements.txt
pip install -e .
python scripts/run_all.py --config configs/default.yaml
```

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

$$\mathbf{pr} = \alpha \, \mathbf{P}^{\top} \mathbf{pr} + (1 - \alpha)\,\frac{\mathbf{1}}{n}$$

with $\alpha = 0.85$.

**Laplacian positional encodings**

Let $L = I - D^{-1/2} A D^{-1/2}$ be the normalised Laplacian. The smallest $k$ non-trivial eigenvectors of $L$ form an $n \times k$ matrix used as positional encodings.

**Random-walk structural encodings**

Let $P = D^{-1}A$ be the random-walk transition matrix and $R^{(t)} = P^t$. The diagonal entries $R^{(t)}_{ii}$ for $t = 1, \ldots, T$ form RWSE features.

With the default configuration, the input dimension is $38$.

### Feature ablation table

Run `scripts/ablation.py` to regenerate this table from validation MCC. The table below is produced by `runs/feature_ablation/feature_ablation.csv`.

| Variant | Feature dim | Val MCC | Test MCC |
|---|---:|---:|---:|
| full | 38 | — | — |
| no_laplacian_pe | 30 | — | — |
| no_rwse | 30 | — | — |
| no_pagerank | 37 | — | — |
| no_clustering | 37 | — | — |
| no_kcore | 37 | — | — |
| no_degree_bucket | 22 | — | — |
| degree_only | 1 | — | — |

## Graph encoders

All models share the same backbone:

1. input projection $\mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d}$
2. graph encoder with $L$ message-passing layers
3. graph-level pooling
4. MLP classifier head

Optional virtual node updates are applied after every encoder layer. For batch $B$ with graph indices $b(i)$:

$$\mathbf{v}_g \leftarrow \mathrm{MLP}\!\left(\mathbf{v}_g + \sum_{i:\, b(i)=g} \mathbf{h}_i\right), \qquad \mathbf{h}_i \leftarrow \mathbf{h}_i + \mathbf{v}_{b(i)}$$

### GIN

Graph Isomorphism Network convolution with MLP $\phi$ and neighbourhood aggregation $\mathcal{N}(i)$:

$$\mathbf{h}_i^{(\ell+1)} = \mathrm{ReLU}\!\left(\mathrm{BN}\!\left((1+\varepsilon)\,\mathbf{h}_i^{(\ell)} + \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(\ell)}\right)\right)$$

with residual connection $\mathbf{h}_i^{(\ell+1)} \leftarrow \mathbf{h}_i^{(\ell)} + \mathbf{h}_i^{(\ell+1)}$.

### PNA

Principal Neighbourhood Aggregation combines aggregators $\bigoplus \in \{\mu, \max, \min, \sigma\}$ and degree scalers $S(\mathbf{D}, \alpha)$:

$$\mathbf{h}_i^{(\ell+1)} = \gamma_{\Theta}\!\left(\mathbf{h}_i^{(\ell)}, \bigoplus_{j \in \mathcal{N}(i)} h_{\Theta}(\mathbf{h}_i^{(\ell)}, \mathbf{h}_j^{(\ell)})\right)$$

$$\bigoplus = \begin{bmatrix} 1 \\ S(\mathbf{D}, \alpha{=}1) \\ S(\mathbf{D}, \alpha{=}{-}1) \end{bmatrix} \otimes \begin{bmatrix} \mu \\ \max \\ \min \\ \sigma \end{bmatrix}$$

The in-degree histogram is computed on the training split only.

### GAT

Multi-head graph attention with leaky ReLU scoring:

$$e_{ij} = \mathrm{LeakyReLU}\!\left(\mathbf{a}^{\top} [\mathbf{W}\mathbf{h}_i \,\|\, \mathbf{W}\mathbf{h}_j]\right)$$

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

$$\mathbf{h}_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, \mathbf{W}\mathbf{h}_j\right)$$

Intermediate layers concatenate heads. The final layer averages head outputs to keep the hidden dimension fixed.

## Graph-level pooling

Three pooling operators are implemented.

**Global mean pooling**

$$\mathbf{g} = \frac{1}{|V|}\sum_{i \in V} \mathbf{h}_i$$

**Global sum pooling**

$$\mathbf{g} = \sum_{i \in V} \mathbf{h}_i$$

**Attention pooling**

$$s_i = \mathbf{w}^{\top}\tanh(\mathbf{W}\mathbf{h}_i), \qquad \alpha_i = \frac{\exp(s_i)}{\sum_{j \in V}\exp(s_j)}, \qquad \mathbf{g} = \sum_{i \in V} \alpha_i \mathbf{h}_i$$

`scripts/compare_pooling.py` selects the pooling method by validation MCC. The same pooling is then used for GIN, PNA, and GAT.

## Training protocol

- stratified train, validation, and test split with ratios $0.8 / 0.1 / 0.1$
- random seed $42$
- AdamW optimiser with learning rate $10^{-3}$ and weight decay $10^{-4}$
- cosine learning-rate schedule
- mixed-precision training on GPU
- gradient clipping with max norm $1.0$
- early stopping on validation MCC with patience $20$
- batch size $128$ for Colab T4

The test split is never used for model selection. Architectures are ranked by best validation MCC. Test metrics for the selected architecture are reported once after training.

### Primary metrics

Matthews correlation coefficient:

$$\mathrm{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

Additional metrics: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, classification report.

Per-epoch metrics are saved to `runs/<architecture>_seed42/epoch_metrics.csv`. Final metrics are saved to `runs/<architecture>_seed42/final_metrics.json`.

## Architecture comparison

After training all three models, `scripts/compare.py` writes `runs/architecture_comparison.csv` and `runs/selected_model.json`.

| Architecture | Best val MCC | Val F1 | Val ROC-AUC | Test MCC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| GIN | — | — | — | — | — | — |
| PNA | — | — | — | — | — | — |
| GAT | — | — | — | — | — | — |

Training curves are saved to `runs/training_curves.png`. Confusion matrices and test predictions are stored under each run directory.

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
| batch_size | 128 |
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

```bibtex
@inproceedings{xu2019gin,
  title = {How Powerful are Graph Neural Networks?},
  author = {Keyulu Xu and Weihua Hu and Jure Leskovec and Stefanie Jegelka},
  booktitle = {International Conference on Learning Representations},
  year = {2019},
}
```

```bibtex
@inproceedings{corso2020pna,
  title = {Principal Neighbourhood Aggregation for Graph Nets},
  author = {Gabriele Corso and Luca Cavalleri and Dominique Beaini and Pietro Li{\`o} and Petar Veli{\v{c}}kovi{\'c}},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2020},
}
```

```bibtex
@inproceedings{velickovic2018gat,
  title = {Graph Attention Networks},
  author = {Petar Veli{\v{c}}kovi{\'c} and Guillem Cucurull and Arantxa Casanova and Adriana Romero and Pietro Li{\`o} and Yoshua Bengio},
  booktitle = {International Conference on Learning Representations},
  year = {2018},
}
```

The project is under GPL-3.0 license.

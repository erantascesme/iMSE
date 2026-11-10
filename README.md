# iMSE

Inverse Mean-Shift Extrapolation with Covariance-Inflation Scoring for Post-Hoc Out-of-Distribution Detection.

## Use iMSE on your own features

Extract the inputs to your frozen classifier head. Fit on raw ID training features and their true class labels, with labels indexed from 0 to C−1 in the same order as the classifier outputs. At inference, provide raw query features and their original logits. Higher scores indicate more ID-like samples.

```python
from imse_benchmark.detectors import IMSE

detector = IMSE(k=16, tau=1.0, device="cuda")
detector.fit(train_features, train_labels)  # (N, D), (N,)
scores = detector.score(test_features, test_logits)  # (B, D), (B, C) -> (B,)
```

Set `k="all"` for the full-class leave-one-out setting, or `device="cpu"` to run without a GPU. For the paper's CUDA precision settings, disable TF32 and autocast. Install the package with `pip install -e .`, or copy the standalone PyTorch code below.

<details>
<summary>Standalone iMSE implementation</summary>

The benchmark and the example use this same implementation. PyTorch is its only external dependency.

```python
"""Standalone iMSE detector. Requires only PyTorch."""

import math
import torch
import torch.nn.functional as F

EPS = 1e-8


def symmetric(a):
    return (a + a.T) * 0.5


def scatter(x):
    residual = x - x.mean(0)
    return residual.T @ residual


def spectrum(matrix):
    values, vectors = torch.linalg.eigh(symmetric(matrix))
    limit = 128 * torch.finfo(values.dtype).eps * len(values) * max(1., float(values.abs().max()))
    if not torch.isfinite(values).all() or values.min() < -limit:
        raise RuntimeError("Invalid shift covariance spectrum")
    return values.clamp_min(0), vectors


def _oas(cov, n):
    d = len(cov)
    a, t = float(cov.square().mean()), float(cov.trace() / d)
    b = (n + 1) * (a - t * t / d)
    alpha = 1. if b <= 0 else min((a + t * t) / b, 1.)
    return (1 - alpha) * cov + alpha * t * torch.eye(d, dtype=cov.dtype, device=cov.device)


def _neighbors(q, k):
    unit = F.normalize(q, dim=1)
    distance = 1. - unit @ unit.T
    distance.fill_diagonal_(float("inf"))
    return distance.topk(k, largest=False, sorted=True).indices


def local_neighbors(q):
    return _neighbors(q, min(128, len(q) - 1))


def _paper_order(q, neighbors):
    # The original finite-k fits permuted every parent before accumulating scatter.
    ranking = q.new_zeros(len(q))
    for k in (128, 64, 32, 4):
        residual = torch.empty_like(ranking)
        for start in range(0, len(q), 128):
            sl = slice(start, start + 128)
            residual[sl] = torch.linalg.vector_norm(q[sl] - q[neighbors[sl, :k]].mean(1), dim=1)
        lo, hi = residual.min(), residual.max()
        ranking += (residual - lo) / (hi - lo + EPS)
    return ranking.topk(len(q)).indices


def _historical(k):
    return k != "all" and k <= 64


@torch.inference_mode()
def class_shifts(raw, k, tau):
    q = F.normalize(raw.float(), dim=1)
    if _historical(k):
        neighbors = local_neighbors(q)
        order = _paper_order(q, neighbors)
        center = q[neighbors[order, :k]].mean(1)
        q = q[order]
    elif k == "all" or k >= len(q) - 1:
        q64 = q.double()
        center = ((q64.sum(0, keepdim=True) - q64) / (len(q) - 1)).float()
    else:
        neighbors = _neighbors(q, k)
        center = torch.empty_like(q)
        for start in range(0, len(q), 32):
            center[start:start + 32] = q[neighbors[start:start + 32]].mean(1)
    delta = q - center
    direction = delta / (torch.linalg.vector_norm(delta, dim=1, keepdim=True) + EPS)
    return F.normalize(q + tau * direction, dim=1).double() - q.double()


@torch.inference_mode()
def fit_geometry(features, labels, device, classes=None):
    x = torch.as_tensor(features)
    y = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    if x.ndim != 2 or y.shape != (len(x),) or len(x) == 0 or y.min() < 0:
        raise ValueError("Expected a nonempty (N, D) feature matrix and (N,) class labels")
    n, d = x.shape
    counts = y.bincount(minlength=classes or 0)
    if (classes is not None and len(counts) != classes) or counts.min() < 2:
        raise ValueError("Class indices must be consecutive, with at least two samples per class")
    sums = torch.zeros((len(counts), d), dtype=torch.float64, device=device)
    xtx = torch.zeros((d, d), dtype=torch.float64, device=device)
    for start in range(0, n, 4096):
        z = x[start:start + 4096].to(device=device, dtype=torch.float32)
        if not torch.isfinite(z).all() or (z.norm(dim=1) <= 1e-12).any():
            raise ValueError("Training features must be finite and nonzero")
        q = F.normalize(z, dim=1).double()
        sums.index_add_(0, y[start:start + len(q)].to(device), q)
        xtx += q.T @ q
    counts64 = counts.to(device=device, dtype=torch.float64)
    means = sums / counts64[:, None]
    cov = symmetric(xtx / n - (means.T * (counts64 / n)) @ means)
    values, vectors = torch.linalg.eigh(symmetric(_oas(cov, n)))
    if not torch.isfinite(values).all() or values.min() <= 0:
        raise RuntimeError("ID covariance is not positive definite")
    order = torch.argsort(y, stable=True).to(x.device)
    offsets = [0, *counts.cumsum(0).tolist()]
    return x, order, offsets, vectors * values.rsqrt(), means


@torch.inference_mode()
def fit_shift(x, order, offsets, whitener, k, tau):
    if not (k == "all" or isinstance(k, int) and k > 0) or not math.isfinite(tau) or tau <= 0:
        raise ValueError("k must be a positive integer or 'all', and tau must be positive")
    total = torch.zeros_like(whitener)
    for lo, hi in zip(offsets, offsets[1:]):
        raw = x[order[lo:hi]].to(device=whitener.device, dtype=torch.float32)
        total += scatter(class_shifts(raw, k, tau))
    return total / len(x)


def make_state(whitener, means, shift, k):
    matrix = symmetric(whitener.T @ shift @ whitener)
    values, rotation = spectrum(matrix if _historical(k) else matrix.cpu())
    values, rotation = values.to(whitener.device), rotation.to(whitener.device)
    transform = whitener @ rotation
    logdet = torch.log1p(values).sum()
    return dict(transform=transform, means=means @ transform, weights=values / (1 + values),
                logdet=float(logdet) if _historical(k) else logdet)


@torch.inference_mode()
def score(q, predicted, state):
    residual = q @ state["transform"] - state["means"][predicted]
    return 0.5 * (state["logdet"] - (residual.square() * state["weights"]).sum(1))


class IMSE:
    """Fit on labeled ID features. Score query features using frozen classifier logits."""

    def __init__(self, k=16, tau=1., device="cpu"):
        self.k, self.tau, self.device = k, tau, torch.device(device)
        self.state = None

    @torch.inference_mode()
    def fit(self, features, labels):
        x, order, offsets, w, means = fit_geometry(features, labels, self.device)
        shift = fit_shift(x, order, offsets, w, self.k, self.tau)
        self.state = make_state(w, means, shift, self.k)
        return self

    @torch.inference_mode()
    def score(self, features, logits):
        if self.state is None:
            raise RuntimeError("Call fit before scoring")
        z = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        logits = torch.as_tensor(logits, device=self.device)
        if logits.shape != (len(z), len(self.state["means"])):
            raise ValueError("Expected one logit per known class for each query")
        return score(F.normalize(z, dim=1).double(), logits.argmax(1), self.state)
```

</details>

## Reproduce the benchmark

Install in a Python 3.11–3.13 environment with PyTorch and torchvision builds that match your CUDA runtime.

```bash
pip install -e .
python -m imse_benchmark all --dataset CIFAR10
```

Use [Reproduce.ipynb](Reproduce.ipynb) from a local checkout or [open the Colab notebook](https://colab.research.google.com/github/erantascesme/iMSE/blob/main/Reproduce_Colab.ipynb). Both install and run the same code, with separate download, extraction, and evaluation cells. Choose `CIFAR10`, `CIFAR100`, or `ImageNet1K`. Use `--backbone` to run a single paper backbone, or `--work-dir` to store data, checkpoints, feature caches, and results elsewhere. Command output, including errors, is displayed and saved in timestamped `work/logs/` files. Colab updates a clean checkout by default and prints its commit ID.

Run `download`, `extract`, or `evaluate` to execute individual stages. The evaluator reuses completed feature caches and prints AUROC / FPR95 for All, Near, and Far. Results include checkpoint metrics, backbone averages, and dataset averages in `work/results/`. After a code update, use a fresh `--output` directory for evaluation and point `--features` at your existing feature cache.

**Data and checkpoint access.** CIFAR data and OpenOOD ResNet18 checkpoints are downloaded from the OpenOOD releases. To download ImageNet-1K, accept the [dataset access terms](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and run `hf auth login` or set a read token through `HF_TOKEN`. In Colab, add `HF_TOKEN` to Secrets. For local ImageNet data, use `work/data/imagenet/{train,val,NINCO,SSB_Hard,iNaturalist,Textures,OpenImage_O}` with the canonical archives or extracted images. Full ImageNet-1K requires substantial disk and compute. Use `pip install -e ".[vim]"` for Vim-Base, which builds the pinned extensions with a CUDA toolkit compatible with PyTorch.

No personal Google Drive files or previously saved experiment outputs are required. OpenOOD's **public** Drive releases are the same sources used by the original CIFAR notebooks. Optional Colab Drive mounting only chooses where to store your new run.

If a public Drive link exceeds its download quota, immediate retries cannot resolve it. Wait for the provider to reset the quota, or use the notebook's optional `MANUAL_ARCHIVES` cell to import an already downloaded original CIFAR benchmark ZIP. The cell verifies its pinned SHA-256 before copying it into `work/data/cifar/`. Upload the ZIP to `/content` using Colab's Files sidebar; do not unzip or repack it. Then rerun the import and download cells.

| Data | Download source |
|---|---|
| CIFAR-10/100 and their OOD sets | [OpenOOD release archives](https://github.com/Jingkang50/OpenOOD/blob/3c35632ee91b54b09d1f085d04f94744cece7d0b/scripts/download/download.py), with the original archive IDs and SHA-256 hashes |
| ImageNet-1K | `ILSVRC/imagenet-1k`, revision `script`, five training archives and one validation archive |
| SSB-Hard and OpenImage-O | `torch-uncertainty/SSB_hard` and `torch-uncertainty/Openimage-O`, as in the canonical download notebook |
| NINCO, iNaturalist, and Textures | OpenOOD's public original-image archives; only the pinned benchmark manifest rows are evaluated |

The ordered manifests, checkpoint hashes, preprocessing, and detector settings preserve the reviewed experiment. Numerical checks against the earlier implementation are not a fresh end-to-end GPU rerun. Small floating-point differences can remain across GPU and library versions; bitwise equality with the paper run is not guaranteed.

**Protocol.** Each detector uses the paper's frozen checkpoints and ordered data splits. KNN, ViM, ReAct, and SCALE select settings on validation AUROC. iMSE fits on ID training data with `k=16` or `k=all` and `tau=1`. The CIFAR NNGuide run uses the original ordered bank. AUROC and FPR95 are macro-averaged over OOD datasets, then averaged over checkpoints within each backbone. FPR95 measures ID rejection at 95% OOD recall.

## Repository layout

```
README.md
Reproduce.ipynb
Reproduce_Colab.ipynb
pyproject.toml
imse_benchmark/
├── config.py          # Detector settings and validation grids
├── datasets/          # Downloads, image loading, and benchmark splits
├── backbones/         # Checkpoints, architectures, preprocessing, and feature caches
├── detectors/         # One module per detector family
├── benchmark/         # Evaluation and metric tables
└── assets/            # Split manifests, checkpoint locks, and license notices
```

The [checkpoint manifest](imse_benchmark/assets/checkpoints.json) records the architecture, weight source, revision or hash, and preprocessing for each backbone. Architectures use pinned [timm](https://github.com/huggingface/pytorch-image-models), the [official Vim source](https://github.com/hustvl/Vim), or bundled [OpenOOD](https://github.com/Jingkang50/OpenOOD) and [ODIN](https://github.com/facebookresearch/odin) implementations.

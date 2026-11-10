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

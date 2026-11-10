import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from ..common import require
import faiss
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance

def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    require(np.isfinite(x).all() and (norm > 0).all(), 'Normalization requires nonzero finite features')
    return np.ascontiguousarray(x / norm, dtype=np.float32)

def logit_scores(logits):
    l = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    require(l.ndim == 2 and l.shape[1] >= 2, 'Need at least two logits')
    top = l.max(1).values
    sorted_l = l.sort(dim=1, descending=True).values
    return {'MSP': l.softmax(1).amax(1).numpy(), 'Energy': torch.logsumexp(l, dim=1).numpy(), 'LogitGap': (top - sorted_l[:, 1:].mean(1)).numpy()}

class GaussianFamily:

    def __init__(self, normalized=False, device='cpu'):
        self.normalized = normalized
        self.device = torch.device(device)

    def fit(self, x, y):
        x = normalize(x) if self.normalized else np.asarray(x, dtype=np.float32)
        t = torch.from_numpy(x)
        classes = sorted(set(y.tolist()))
        require(classes == list(range(len(classes))), 'Class labels must be contiguous')
        if self.normalized:
            t = t.double()
        self.means = torch.stack([t[y == c].mean(0) for c in classes])
        residual = torch.cat([t[y == c] - self.means[c] for c in classes]).numpy()
        est = EmpiricalCovariance(assume_centered=self.normalized).fit(residual)
        dtype = torch.float64 if self.normalized else torch.float32
        self.precision = torch.as_tensor(est.precision_, dtype=dtype)
        self.global_mean = t.mean(0)
        est0 = EmpiricalCovariance(assume_centered=self.normalized).fit((t - self.global_mean).numpy())
        self.global_precision = torch.as_tensor(est0.precision_, dtype=dtype)
        require(torch.isfinite(self.precision).all() and torch.isfinite(self.global_precision).all(), 'Invalid Gaussian precision')
        self.diagnostics = {'normalized': self.normalized, 'd': x.shape[1], 'n': len(x), 'within_rank': int(np.linalg.matrix_rank(est.covariance_)), 'global_rank': int(np.linalg.matrix_rank(est0.covariance_)), 'within_eigen_min': float(np.linalg.eigvalsh(est.covariance_).min()), 'within_eigen_max': float(np.linalg.eigvalsh(est.covariance_).max()), 'ridge': 0.0}
        return self

    @torch.inference_mode()
    def scores(self, x, batch=256):
        if not self.normalized:
            return self._openood_raw_scores(x)
        x = normalize(x) if self.normalized else np.asarray(x, dtype=np.float32)
        mu = self.means.to(self.device)
        P = self.precision.to(self.device)
        gm = self.global_mean.to(self.device)
        GP = self.global_precision.to(self.device)
        out = []
        rel = []
        for start in range(0, len(x), batch):
            q = torch.as_tensor(x[start:start + batch], device=self.device, dtype=P.dtype)
            distances = torch.stack([((q - m) @ P * (q - m)).sum(1) for m in mu], dim=1)
            md = -distances.min(1).values
            bg = ((q - gm) @ GP * (q - gm)).sum(1)
            out.append(md.cpu().numpy())
            rel.append((bg + md).cpu().numpy())
        return (np.concatenate(out), np.concatenate(rel))

    @torch.inference_mode()
    def _openood_raw_scores(self, x):
        """Preserve pinned OpenOOD CPU FP32 matmul/diag/subtraction order."""
        x = np.asarray(x, dtype=np.float32)
        mu = self.means.cpu()
        P = self.precision.cpu()
        gm = self.global_mean.cpu()
        GP = self.global_precision.cpu()
        out = []
        relative = []
        for start in range(0, len(x), 200):
            q = torch.from_numpy(x[start:start + 200])
            residual = q - gm.view(1, -1)
            background = -torch.matmul(torch.matmul(residual, GP), residual.t()).diag()
            class_scores = torch.zeros((len(q), len(mu)), dtype=torch.float32)
            relative_scores = torch.zeros_like(class_scores)
            for c in range(len(mu)):
                residual = q - mu[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(torch.matmul(residual, P), residual.t()).diag()
                relative_scores[:, c] = class_scores[:, c] - background
            out.append(class_scores.max(1).values.numpy())
            relative.append(relative_scores.max(1).values.numpy())
        return (np.concatenate(out), np.concatenate(relative))

class ExactKNN:

    def __init__(self, x, device='cpu'):
        self.bank = torch.from_numpy(normalize(x) + np.float32(1e-10)).to(device)
        self.device = torch.device(device)

    @torch.inference_mode()
    def score(self, x, ks, batch=128):
        require(min(ks) > 0 and max(ks) <= len(self.bank), 'Invalid K')
        parts = {k: [] for k in ks}
        bank_norm = (self.bank * self.bank).sum(1)[None, :]
        for start in tqdm(range(0, len(x), batch), desc='Exact KNN', leave=False):
            q = torch.from_numpy(normalize(x[start:start + batch]) + np.float32(1e-10)).to(self.device)
            d = ((q * q).sum(1, keepdim=True) + bank_norm - 2 * q @ self.bank.T).clamp_min(0)
            best = d.topk(max(ks), dim=1, largest=False, sorted=True).values
            for k in ks:
                parts[k].append((-best[:, k - 1]).cpu().numpy())
        return {k: np.concatenate(v) for k, v in parts.items()}

    def reference_score(self, x, ks):
        """Use OpenOOD's exact FAISS backend for validation hyperparameter selection."""
        index = faiss.IndexFlatL2(self.bank.shape[1])
        index.add(self.bank.cpu().numpy())
        distances, _ = index.search(normalize(x) + np.float32(1e-10), max(ks))
        return {k: -distances[:, k - 1].copy() for k in ks}

class NNGuide:

    def __init__(self, x, logits, device='cpu', batch_size=200, alpha=0.01, random_bank=False):
        count = min(len(x), (int(len(x) * alpha) // batch_size + 1) * batch_size)
        self.indices = np.random.default_rng(0).choice(len(x), size=count, replace=False) if random_bank else np.arange(count, dtype=np.int64)
        self.device = torch.device(device)
        guided = (normalize(x[self.indices]) + np.float32(1e-10)) * logsumexp(logits[self.indices], axis=1)[:, None]
        self.bank = torch.from_numpy(guided).to(device)

    @torch.inference_mode()
    def score(self, x, logits, ks=(10, 100), batch=256):
        require(max(ks) <= len(self.bank), 'NNGuide K exceeds bank')
        values = {k: [] for k in ks}
        for start in range(0, len(x), batch):
            q = torch.from_numpy(normalize(x[start:start + batch]) + np.float32(1e-10)).to(self.device)
            v = (q @ self.bank.T).topk(max(ks), dim=1, sorted=True).values.cpu().numpy()
            energy = logsumexp(logits[start:start + batch], axis=1)
            for k in ks:
                values[k].append(energy * v[:, :k].mean(1))
        return {k: np.concatenate(v) for k, v in values.items()}

class FDBD:

    def __init__(self, x, W):
        self.mean = np.asarray(x, dtype=np.float64).mean(0)
        w = np.asarray(W, dtype=np.float64)
        self.den = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=2)
        np.fill_diagonal(self.den, 1.0)
        require((self.den > 0).all(), 'Coincident classifier rows: fDBD undefined')

    def score(self, x, logits):
        l = np.asarray(logits, dtype=np.float64)
        pred = l.argmax(1)
        top = l.max(1)
        distances = np.linalg.norm(np.asarray(x, dtype=np.float64) - self.mean, axis=1)
        require((distances > 0).all(), 'fDBD undefined at training mean')
        return (np.abs(l - top[:, None]) / self.den[pred]).sum(1) / ((l.shape[1] - 1) * distances)

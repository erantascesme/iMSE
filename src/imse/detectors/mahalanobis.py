"""Mahalanobis and Relative Mahalanobis, with raw and L2-normalized variants."""

import gc

import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg
from sklearn.covariance import EmpiricalCovariance
from tqdm.auto import tqdm

from ..common import require
from ._normalization import normalize


DEVICE = torch.device("cpu")
MAHA_SCORE_BATCH_SIZE = 1024


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


class MahalanobisFamilyDetector:

    def __init__(self, num_classes=1000, normalize_features=False, device=DEVICE):
        self.num_classes = int(num_classes)
        self.normalize_features = bool(normalize_features)
        self.device = device
        self.means = None
        self.precision = None
        self.global_mean = None
        self.global_precision = None
        self.class_counts = None
        self.feature_dim = None

    def _prep(self, x):
        if self.normalize_features:
            return F.normalize(x, p=2, dim=1)
        return x

    @torch.no_grad()
    def fit(self, features, labels):
        X = features.detach().cpu().float()
        y = labels.detach().cpu().long()
        n, d = X.shape
        self.feature_dim = int(d)
        class_indices = [torch.where(y == c)[0] for c in range(self.num_classes)]
        class_counts = torch.tensor([len(idx) for idx in class_indices], dtype=torch.float64)
        assert (class_counts > 0).all()
        means = torch.empty(self.num_classes, d, dtype=torch.float64, device=self.device)
        print('Class means...')
        for c in tqdm(range(self.num_classes), leave=False):
            X_c = X[class_indices[c]].to(self.device, non_blocking=True)
            X_c = self._prep(X_c).double()
            means[c] = X_c.mean(dim=0)
        covariance = torch.zeros(d, d, dtype=torch.float64, device=self.device)
        total = 0
        print('Shared within-class covariance...')
        for c in tqdm(range(self.num_classes), leave=False):
            X_c = X[class_indices[c]].to(self.device, non_blocking=True)
            X_c = self._prep(X_c).double()
            centered = X_c - means[c]
            covariance += centered.T @ centered
            total += len(X_c)
        assert total == n
        covariance /= float(n)
        counts_gpu = class_counts.to(self.device)
        global_mean = (means * counts_gpu[:, None]).sum(dim=0) / float(n)
        mean_offsets = means - global_mean[None, :]
        weighted_offsets = mean_offsets * torch.sqrt(counts_gpu)[:, None]
        between_covariance = weighted_offsets.T @ weighted_offsets / float(n)
        global_covariance = covariance + between_covariance
        print('Precision matrices...')
        covariance_np = covariance.cpu().numpy()
        global_covariance_np = global_covariance.cpu().numpy()
        precision_np = linalg.pinvh(covariance_np, check_finite=False)
        global_precision_np = linalg.pinvh(global_covariance_np, check_finite=False)
        self.means = means.cpu()
        self.precision = torch.from_numpy(precision_np).double().cpu()
        self.global_mean = global_mean.cpu()
        self.global_precision = torch.from_numpy(global_precision_np).double().cpu()
        self.class_counts = class_counts.cpu()
        del covariance
        del global_covariance
        del between_covariance
        del covariance_np
        del global_covariance_np
        del precision_np
        del global_precision_np
        del means
        del global_mean
        del class_indices
        gc.collect()
        torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def score_all(self, features, batch_size=MAHA_SCORE_BATCH_SIZE):
        X = features.detach().cpu().float()
        means = self.means.to(self.device)
        precision = self.precision.to(self.device)
        global_mean = self.global_mean.to(self.device)
        global_precision = self.global_precision.to(self.device)
        mean_precision = means @ precision
        mean_quadratic = (mean_precision * means).sum(dim=1)
        dmin_parts = []
        relative_parts = []
        for start in tqdm(range(0, len(X), batch_size), desc='Mahalanobis++' if self.normalize_features else 'Mahalanobis', leave=False):
            x = X[start:start + batch_size].to(self.device, non_blocking=True)
            x = self._prep(x).double()
            x_precision = x @ precision
            x_quadratic = (x_precision * x).sum(dim=1, keepdim=True)
            distances = x_quadratic - 2.0 * (x_precision @ means.T) + mean_quadratic[None, :]
            d_min = distances.min(dim=1).values
            global_diff = x - global_mean[None, :]
            d_global = (global_diff @ global_precision * global_diff).sum(dim=1)
            relative = d_global - d_min
            dmin_parts.append(d_min.cpu())
            relative_parts.append(relative.cpu())
        d_min = torch.cat(dmin_parts).numpy()
        relative = torch.cat(relative_parts).numpy()
        return (d_min, -d_min, relative)

"""Exact k-nearest-neighbor OOD scoring."""

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..common import require
from ._normalization import normalize


DEVICE = torch.device("cpu")


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


class ExactNeighbors:
    """Exhaustive search with bank blocks; no approximate index or class filter."""

    def __init__(self, features, device=DEVICE, query_batch=64, bank_batch=32768):
        self.features = features
        self.device = device
        self.query_batch = query_batch
        self.bank_batch = bank_batch

    @torch.inference_mode()
    def score_many(self, features, ks=(50, 100, 200, 500, 1000)):
        ks = tuple(sorted(set(ks)))
        max_k = max(ks)
        require(max_k <= len(self.features), 'K exceeds training-bank size')
        queries = F.normalize(features.to(self.device).float(), dim=1)
        best = torch.full((len(queries), max_k), float('inf'), device=self.device)
        for offset in tqdm(range(0, len(self.features), self.bank_batch), desc='Exact KNN bank blocks', leave=False):
            bank = F.normalize(self.features[offset:offset + self.bank_batch].to(self.device).float(), dim=1)
            for start in range(0, len(queries), self.query_batch):
                q = queries[start:start + self.query_batch]
                distances = ((q * q).sum(1, keepdim=True) + (bank * bank).sum(1)[None, :] - 2 * (q @ bank.T)).clamp_min(0)
                local = distances.topk(min(max_k, len(bank)), dim=1, largest=False, sorted=False).values
                best[start:start + len(q)] = torch.cat([best[start:start + len(q)], local], dim=1).topk(max_k, dim=1, largest=False, sorted=False).values
        best = best.sort(dim=1).values
        return {k: (-best[:, k - 1]).cpu().numpy() for k in ks}

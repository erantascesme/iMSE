"""Nearest Neighbor Guidance with the benchmark bank policies."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import logsumexp

from ..common import require
from ._normalization import normalize


DEVICE = torch.device("cpu")


class NNGuide:

    def __init__(self, x, logits, device='cpu', batch_size=200, alpha=0.01):
        count = min(len(x), (int(len(x) * alpha) // batch_size + 1) * batch_size)
        self.indices = np.arange(count, dtype=np.int64)
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


def openood_nnguide_indices(official_to_cached, alpha=0.01, batch_size=200, seed=0):
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(range(len(official_to_cached)), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) * batch_size > int(len(official_to_cached) * alpha):
            break
    return np.asarray(official_to_cached, dtype=np.int64)[torch.cat(batches).numpy()]


class NearestNeighborGuide:

    def __init__(self, train_features, train_logits, indices, device=DEVICE):
        self.device = device
        self.indices = np.asarray(indices, dtype=np.int64)
        require(len(set(self.indices)) == len(self.indices), 'Duplicate NNGuide bank indices')
        selected = torch.as_tensor(self.indices, dtype=torch.long)
        raw = train_features[selected].float()
        require(bool((raw.norm(dim=1) > 0).all()), 'Zero NNGuide training norm')
        x = F.normalize(raw, dim=1) + 1e-10
        confidence = torch.logsumexp(train_logits[selected].float(), dim=1)
        self.bank = (x * confidence[:, None]).to(device)

    @torch.inference_mode()
    def score_many(self, features, logits, ks=(10, 100), batch=512):
        require(max(ks) <= len(self.bank), 'NNGuide K exceeds bank size')
        out = {k: [] for k in ks}
        for start in range(0, len(features), batch):
            raw = features[start:start + batch].to(self.device).float()
            require(bool((raw.norm(dim=1) > 0).all()), 'Zero NNGuide query norm')
            q = F.normalize(raw, dim=1) + 1e-10
            energy = torch.logsumexp(logits[start:start + batch].to(self.device).float(), dim=1)
            best = (q @ self.bank.T).topk(max(ks), dim=1, sorted=True).values
            for k in ks:
                out[k].append((energy * best[:, :k].mean(1)).cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}

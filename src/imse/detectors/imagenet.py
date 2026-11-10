import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from ..common import require
import gc
from scipy import linalg
DEVICE = torch.device("cpu")
MAHA_SCORE_BATCH_SIZE = 1024

def score_logits(logits, method, n=200, batch=4096):
    out = []
    for start in range(0, len(logits), batch):
        l = logits[start:start + batch].float()
        if method == 'MSP':
            s = l.softmax(1).amax(1)
        elif method == 'Energy':
            s = torch.logsumexp(l, dim=1)
        elif method == 'LogitGap':
            require(2 <= n <= l.shape[1], 'Invalid LogitGap N')
            top = l.topk(n, dim=1, sorted=True).values
            s = top[:, 0] - top[:, 1:].mean(1)
        else:
            raise ValueError(method)
        out.append(s.cpu().numpy())
    return np.concatenate(out)

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

class FeatureDecisionBoundary:

    def __init__(self, weight, train_features, device=DEVICE):
        self.device = device
        total = torch.zeros(train_features.shape[1], dtype=torch.float64)
        for start in range(0, len(train_features), 4096):
            total += train_features[start:start + 4096].double().sum(0)
        self.mean = (total / len(train_features)).float().to(device)
        w = weight.to(device).float()
        self.denominator = torch.cdist(w, w)
        self.denominator.fill_diagonal_(1.0)
        require(bool((self.denominator > 0).all()), 'Coincident classifier rows make fDBD undefined; inspect checkpoint')

    @torch.inference_mode()
    def score(self, features, logits, batch=1024):
        parts = []
        for start in range(0, len(features), batch):
            x = features[start:start + batch].to(self.device).float()
            l = logits[start:start + batch].to(self.device).float()
            top, pred = l.max(1)
            gaps = (l - top[:, None]).abs()
            numerator = (gaps / self.denominator[pred]).sum(1) / (l.shape[1] - 1)
            distance = (x - self.mean).norm(dim=1)
            require(bool((distance > 0).all()), 'fDBD query exactly equals training mean; score undefined')
            parts.append((numerator / distance).cpu().numpy())
        return np.concatenate(parts)

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

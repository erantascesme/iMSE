import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from ..common import require
import scipy.linalg
from .. import common as core
GRIDS = {"ViM": [256, 1000], "ReAct": [85, 90, 95, 99], "SCALE": [65, 70, 75, 80, 85, 90, 95]}

class UndefinedScore(RuntimeError):
    """A reference equation is undefined; never drop rows or repair the score silently."""

def candidates(d):
    return {name: [p for p in vals if name != 'ViM' or p < d] for name, vals in GRIDS.items()}

def tensor(x, device, dtype=torch.float64):
    return torch.as_tensor(np.array(x, copy=True), dtype=dtype, device=device)

def eigh(cov):
    c = np.asarray(cov, dtype=np.float64)
    core.require(np.isfinite(c).all(), 'Nonfinite covariance')
    vals, vecs = scipy.linalg.eigh((c + c.T) * 0.5, check_finite=True, driver='evd')
    return (torch.from_numpy(vals), torch.from_numpy(vecs))

def build_vim(cov, u, device='cpu'):
    vals, vecs = eigh(cov)
    bound = 128 * np.finfo(float).eps * len(vals) * max(1.0, float(vals.abs().max()))
    core.require(float(vals.min()) >= -bound, 'ViM second moment is not PSD')
    return {'values': vals.to(device), 'vectors': vecs.flip(1).to(device), 'u': torch.as_tensor(u, dtype=torch.float32, device=device), 'alpha': {}}

@torch.inference_mode()
def vim_residual(x, state, dim):
    return torch.linalg.vector_norm((x - state['u']).double() @ state['vectors'][:, int(dim):], dim=1)

@torch.inference_mode()
def vim(x, logits, state, dim):
    return torch.logsumexp(logits.double(), dim=1) - state['alpha'][str(dim)] * vim_residual(x, state, dim)

@torch.inference_mode()
def react(x, weight, bias, threshold):
    return torch.logsumexp(F.linear(x.clamp(max=float(threshold)), weight, bias).double(), dim=1)

@torch.inference_mode()
def scale_features(x, percentile):
    k = x.shape[1] - int(np.round(x.shape[1] * float(percentile) / 100.0))
    core.require(0 < k <= x.shape[1], 'Invalid SCALE retained coordinate count')
    values, indices = torch.topk(x, k, dim=1)
    sparse = torch.zeros_like(x).scatter_(1, indices, values)
    s1, s2 = (x.sum(1), sparse.sum(1))
    if not bool((s2 != 0).all()):
        raise UndefinedScore('SCALE top-activation sum is zero; the author ratio is undefined')
    factor = torch.exp(s1 / s2)
    out = x * factor[:, None]
    if not bool(torch.isfinite(out).all()):
        raise UndefinedScore('SCALE overflow/nonfinite activations; no clipping, abs, or ReLU substitute applied')
    return (out, {'negative_coordinate_count': int((x < 0).sum()), 'coordinates': x.numel(), 'nonpositive_sum_rows': int((s1 <= 0).sum()), 'rows': len(x), 'factor_min': float(factor.min()), 'factor_max': float(factor.max())})

@torch.inference_mode()
def scale(x, weight, bias, percentile):
    features, diagnostic = scale_features(x, percentile)
    scores = torch.logsumexp(F.linear(features, weight, bias).double(), dim=1)
    if not bool(torch.isfinite(scores).all()):
        raise UndefinedScore('SCALE classifier logits overflow')
    return (scores, diagnostic)

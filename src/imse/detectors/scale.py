"""Energy scoring after SCALE activation scaling."""

import numpy as np
import torch
import torch.nn.functional as F

from ..common import require


PERCENTILES = (65, 70, 75, 80, 85, 90, 95)


class UndefinedScore(RuntimeError):
    """A reference equation is undefined; never drop rows or repair the score silently."""


@torch.inference_mode()
def scale_features(x, percentile):
    k = x.shape[1] - int(np.round(x.shape[1] * float(percentile) / 100.0))
    require(0 < k <= x.shape[1], 'Invalid SCALE retained coordinate count')
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

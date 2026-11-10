"""Top logit minus the mean of the competing logits."""

import torch

from ..common import require


def score(logits: torch.Tensor, n: int | None = None) -> torch.Tensor:
    if n is None:
        require(logits.ndim == 2 and logits.shape[1] >= 2, 'Need at least two logits')
        top = logits.max(1).values
        competitors = logits.sort(dim=1, descending=True).values[:, 1:]
    else:
        require(2 <= n <= logits.shape[1], 'Invalid LogitGap N')
        ranked = logits.topk(n, dim=1, sorted=True).values
        top, competitors = ranked[:, 0], ranked[:, 1:]
    return top - competitors.mean(1)

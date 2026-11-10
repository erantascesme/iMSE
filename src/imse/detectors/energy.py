"""Energy confidence at temperature one."""

import torch


def score(logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(logits, dim=1)

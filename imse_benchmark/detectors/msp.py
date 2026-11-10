"""Maximum softmax probability."""

import torch


def score(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(1).amax(1)

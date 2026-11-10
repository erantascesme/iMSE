"""Energy scoring with an upper activation threshold."""

import torch
import torch.nn.functional as F


@torch.inference_mode()
def react(x, weight, bias, threshold):
    return torch.logsumexp(F.linear(x.clamp(max=float(threshold)), weight, bias).double(), dim=1)

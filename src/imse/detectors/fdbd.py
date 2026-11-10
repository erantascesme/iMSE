"""Fast decision boundary distance."""

import numpy as np
import torch

from ..common import require


DEVICE = torch.device("cpu")


class FDBD:

    def __init__(self, x, W):
        self.mean = np.asarray(x, dtype=np.float64).mean(0)
        w = np.asarray(W, dtype=np.float64)
        self.den = np.linalg.norm(w[:, None, :] - w[None, :, :], axis=2)
        np.fill_diagonal(self.den, 1.0)
        require((self.den > 0).all(), 'Coincident classifier rows: fDBD undefined')

    def score(self, x, logits):
        l = np.asarray(logits, dtype=np.float64)
        pred = l.argmax(1)
        top = l.max(1)
        distances = np.linalg.norm(np.asarray(x, dtype=np.float64) - self.mean, axis=1)
        require((distances > 0).all(), 'fDBD undefined at training mean')
        return (np.abs(l - top[:, None]) / self.den[pred]).sum(1) / ((l.shape[1] - 1) * distances)


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

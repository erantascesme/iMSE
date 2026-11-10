"""Float32 feature normalization used by the CIFAR protocol."""

import numpy as np

from ..common import require


def normalize(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    require(np.isfinite(x).all() and (norm > 0).all(), 'Normalization requires nonzero finite features')
    return np.ascontiguousarray(x / norm, dtype=np.float32)

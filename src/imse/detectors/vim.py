"""Virtual-logit matching."""

import numpy as np
import scipy.linalg
import torch

from ..common import require


DIMENSIONS = (256, 1000)


def eigh(cov):
    c = np.asarray(cov, dtype=np.float64)
    require(np.isfinite(c).all(), 'Nonfinite covariance')
    vals, vecs = scipy.linalg.eigh((c + c.T) * 0.5, check_finite=True, driver='evd')
    return (torch.from_numpy(vals), torch.from_numpy(vecs))


def build_vim(cov, u, device='cpu'):
    vals, vecs = eigh(cov)
    bound = 128 * np.finfo(float).eps * len(vals) * max(1.0, float(vals.abs().max()))
    require(float(vals.min()) >= -bound, 'ViM second moment is not PSD')
    return {'values': vals.to(device), 'vectors': vecs.flip(1).to(device), 'u': torch.as_tensor(u, dtype=torch.float32, device=device), 'alpha': {}}


@torch.inference_mode()
def vim_residual(x, state, dim):
    return torch.linalg.vector_norm((x - state['u']).double() @ state['vectors'][:, int(dim):], dim=1)


@torch.inference_mode()
def vim(x, logits, state, dim):
    return torch.logsumexp(logits.double(), dim=1) - state['alpha'][str(dim)] * vim_residual(x, state, dim)

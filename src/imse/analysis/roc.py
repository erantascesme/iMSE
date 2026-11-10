"""Exact weighted ROC metrics for an ordinary paired, fixed-size bootstrap.

Scores are sorted once. Integer multinomial multiplicities are equivalent to
explicitly repeating sampled rows, including ties and sklearn's ROC pruning.
No Gaussian, Poisson, Bayesian, or subsampled bootstrap approximation is used.
"""
import hashlib
import json
import numpy as np


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def multiplicities(n, start, stop, seed, cohort):
    """Exactly n draws with replacement per replicate, independent of batching.

    A cohort key is its dataset, split and authenticated ordered-row digest.
    Thus all detectors and checkpoints share the same sampled image indices.
    """
    require(n > 0 and 0 <= start < stop, 'Invalid bootstrap population/range')
    key = json.dumps([int(seed), cohort], sort_keys=True, separators=(',', ':'))
    entropy = np.frombuffer(hashlib.sha256(key.encode()).digest(), dtype='<u4').tolist()
    result = np.empty((stop-start, n), dtype=np.int32)
    for j, replicate in enumerate(range(start, stop)):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy + [replicate])))
        # Counts encode a FULL size-n resample; zero counts and duplicates are intentional.
        draws = rng.integers(0, n, size=n, dtype=np.int64)
        result[j] = np.bincount(draws, minlength=n)
    require(np.all(result.sum(1) == n), 'A bootstrap sample changed population size')
    return result


class ROCPlan:
    def __init__(self, id_scores, ood_scores):
        ins, outs = np.asarray(id_scores, dtype=np.float64), np.asarray(ood_scores, dtype=np.float64)
        require(ins.ndim == outs.ndim == 1 and min(len(ins), len(outs)) > 0,
                'Both score populations must be nonempty vectors')
        require(np.isfinite(ins).all() and np.isfinite(outs).all(), 'Nonfinite saved scores')
        self.ni, self.no = len(ins), len(outs)
        # Ascending higher-ID score = descending OOD confidence (-score).
        score = np.concatenate([ins, outs])
        self.order = np.argsort(score, kind='stable')
        sorted_score = score[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(sorted_score) != 0), len(score)-1].astype(np.int64)
        self.id_mask = self.order < self.ni
        self._gpu = None

    def numpy(self, wi, wo):
        wi, wo = np.asarray(wi), np.asarray(wo)
        require(wi.shape[1] == self.ni and wo.shape == (wi.shape[0], self.no), 'Weight shape mismatch')
        require((wi >= 0).all() and (wo >= 0).all(), 'Negative bootstrap weights')
        weight = np.concatenate([wi, wo], axis=1)[:, self.order]
        fp = np.cumsum(weight * self.id_mask, axis=1, dtype=np.float64)[:, self.ends]
        tp = np.cumsum(weight * ~self.id_mask, axis=1, dtype=np.float64)[:, self.ends]
        return _numpy_roc(fp, tp)

    def torch(self, wi, wo):
        import torch
        if self._gpu is None or self._gpu[0].device != wi.device:
            self._gpu = (torch.as_tensor(self.order, device=wi.device),
                         torch.as_tensor(self.ends, device=wi.device),
                         torch.as_tensor(self.id_mask, device=wi.device))
        order, ends, mask = self._gpu
        weight = torch.cat([wi, wo], dim=1).index_select(1, order)
        fp = torch.cumsum(weight * mask, dim=1, dtype=torch.float64).index_select(1, ends)
        tp = torch.cumsum(weight * ~mask, dim=1, dtype=torch.float64).index_select(1, ends)
        return _torch_roc(fp, tp)


def _numpy_roc(fp, tp):
    prev_tp = np.concatenate([np.zeros((len(tp), 1)), tp[:, :-1]], axis=1)
    prev_fp = np.concatenate([np.zeros((len(fp), 1)), fp[:, :-1]], axis=1)
    df, dt = fp-prev_fp, tp-prev_tp
    require((fp[:, -1] > 0).all() and (tp[:, -1] > 0).all(), 'Missing bootstrap ID/OOD population')
    auc = ((tp + prev_tp) * df).sum(1) / (2*fp[:, -1]*tp[:, -1])

    # sklearn first removes zero-weight observations, then collinear interior
    # ROC points. Their removal matters for FPR95 in certain tied-score cases.
    active = (df + dt) > 0
    g = fp.shape[1]
    idx = np.arange(g)[None, :]
    before = np.maximum.accumulate(np.where(active, idx, -1), axis=1)
    before = np.concatenate([np.full((len(fp), 1), -1), before[:, :-1]], axis=1)
    after = np.minimum.accumulate(np.where(active, idx, g)[:, ::-1], axis=1)[:, ::-1]
    after = np.concatenate([after[:, 1:], np.full((len(fp), 1), g)], axis=1)
    f0 = np.take_along_axis(fp, np.maximum(before, 0), axis=1)
    t0 = np.take_along_axis(tp, np.maximum(before, 0), axis=1)
    f1 = np.take_along_axis(fp, np.minimum(after, g-1), axis=1)
    t1 = np.take_along_axis(tp, np.minimum(after, g-1), axis=1)
    keep = active & ((before < 0) | (after == g) | (f1-2*fp+f0 != 0) | (t1-2*tp+t0 != 0))
    hit = keep & (tp / tp[:, -1:] >= .95)
    require(hit.any(1).all(), '95% OOD recall is unreachable')
    at = hit.argmax(1)
    fpr = fp[np.arange(len(fp)), at] / fp[:, -1]
    return 100*np.stack([auc, fpr], axis=1)


def _torch_roc(fp, tp):
    import torch
    zeros = torch.zeros((len(fp), 1), dtype=fp.dtype, device=fp.device)
    prev_tp = torch.cat([zeros, tp[:, :-1]], 1)
    prev_fp = torch.cat([zeros, fp[:, :-1]], 1)
    df, dt = fp-prev_fp, tp-prev_tp
    auc = ((tp+prev_tp)*df).sum(1)/(2*fp[:, -1]*tp[:, -1])
    active = (df+dt) > 0
    g = fp.shape[1]
    idx = torch.arange(g, device=fp.device)[None, :].expand(len(fp), -1)
    before = torch.cummax(torch.where(active, idx, -1), 1).values
    before = torch.cat([torch.full_like(idx[:, :1], -1), before[:, :-1]], 1)
    after = torch.cummin(torch.where(active, idx, g).flip([1]), 1).values.flip([1])
    after = torch.cat([after[:, 1:], torch.full_like(idx[:, :1], g)], 1)
    f0, t0 = fp.gather(1, before.clamp_min(0)), tp.gather(1, before.clamp_min(0))
    f1, t1 = fp.gather(1, after.clamp_max(g-1)), tp.gather(1, after.clamp_max(g-1))
    keep = active & ((before < 0) | (after == g) | (f1-2*fp+f0 != 0) | (t1-2*tp+t0 != 0))
    hit = keep & (tp/tp[:, -1:] >= .95)
    at = hit.to(torch.int32).argmax(1)
    fpr = fp.gather(1, at[:, None])[:, 0]/fp[:, -1]
    return 100*torch.stack([auc, fpr], 1)


def reference(ins, outs, wi=None, wo=None):
    """Independent sklearn call with the exact saved OpenOOD primary convention."""
    from sklearn.metrics import roc_curve, auc
    weight = None if wi is None else np.r_[wi, wo]
    y = np.r_[np.zeros(len(ins)), np.ones(len(outs))]
    fpr, tpr, _ = roc_curve(y, -np.r_[ins, outs], sample_weight=weight, drop_intermediate=True)
    return np.array([100*auc(fpr, tpr), 100*fpr[np.flatnonzero(tpr >= .95)[0]]])


def runtime_checks(device='cpu'):
    """Compare optimized metrics with explicit duplicated-row sklearn evaluation."""
    rng = np.random.default_rng(6031)
    cases = [
        (rng.normal(size=57), rng.normal(size=73)),
        (rng.integers(-3, 4, 47), rng.integers(-3, 4, 63)),
        (np.ones(31), np.ones(29)),
        (np.ones(31), np.zeros(29)),
        (np.zeros(31), np.ones(29)),
        (np.arange(100), np.arange(100)),  # diagonal ROC pruning matters at .95
    ]
    records = []
    use_torch = device != 'cpu'
    if use_torch:
        import torch
    for case, (ins, outs) in enumerate(cases):
        plan = ROCPlan(ins, outs)
        wi = multiplicities(len(ins), 0, 11, 1729, ['test', case, 'ID'])
        wo = multiplicities(len(outs), 0, 11, 1729, ['test', case, 'OOD'])
        wi = np.vstack([np.ones(len(ins), dtype=int), wi])
        wo = np.vstack([np.ones(len(outs), dtype=int), wo])
        expected = np.array([reference(np.repeat(ins, a), np.repeat(outs, b)) for a, b in zip(wi, wo)])
        actual = plan.numpy(wi, wo)
        error = float(np.max(abs(actual-expected)))
        require(np.allclose(actual, expected, atol=1e-10, rtol=0), 'CPU ROC check failed')
        row = dict(case=case, n=len(expected), CPU_max_abs_error_pp=error, passed=True)
        if use_torch:
            got = plan.torch(torch.as_tensor(wi, device=device), torch.as_tensor(wo, device=device)).cpu().numpy()
            row['device_max_abs_error_pp'] = float(np.max(abs(got-expected)))
            require(np.allclose(got, expected, atol=1e-10, rtol=0), 'Device ROC check failed')
        records.append(row)
    x = multiplicities(53, 0, 17, 1729, 'resume')
    y = np.vstack([multiplicities(53, 0, 5, 1729, 'resume'), multiplicities(53, 5, 17, 1729, 'resume')])
    require(np.array_equal(x, y), 'Sampling changed across resume/batch boundaries')
    return dict(passed=True, checks=records, batching_and_resume_invariant=True,
                reference='sklearn ROC, OOD-positive, first retained TPR >= 0.95',
                bootstrap='exact fixed-size sampling with replacement', device=device)

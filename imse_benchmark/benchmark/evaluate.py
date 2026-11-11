import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

from .. import config
from ..config import DETECTORS
from ..detectors import energy, fdbd, knn, logit_gap, mahalanobis, msp, react, scale, vim
from ..detectors import imse as covariance
from ..common import atomic_json, hash_file, load_json, require
from ..datasets import DATASETS, ood_groups
from ..backbones.features import configure_precision, load_split

METRICS = ["AUROC", "FPR95"]


def metrics(id_scores, ood_scores):
    a, b = np.asarray(id_scores), np.asarray(ood_scores)
    require(a.ndim == b.ndim == 1 and len(a) and len(b) and np.isfinite(a).all()
            and np.isfinite(b).all(), "Invalid detector scores")
    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(a)), np.ones(len(b))], -np.r_[a, b])
    return dict(AUROC=100 * auc(fpr, tpr), FPR95=100 * fpr[np.flatnonzero(tpr >= config.OOD_RECALL)[0]])


def score_table(dataset, name, run, detector, scores):
    near, far = ood_groups(dataset)
    rows = {split: metrics(scores["id_test"], scores[split]) for split in near + far}
    for scope, splits in (("Near", near), ("Far", far), ("All", near + far)):
        rows[scope] = {m: float(np.mean([rows[s][m] for s in splits])) for m in METRICS}
    return [dict(Dataset=dataset, Backbone=name, Run=run, Detector=detector, Scope=s, **v)
            for s, v in rows.items()]


def _batches(x, device, batch):
    for start in range(0, len(x), batch):
        yield torch.from_numpy(x[start:start + batch]).to(device)


def logit_scores(logits, gap_n=None):
    values = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    parts = {"MSP": [], "Energy": [], "LogitGap": []}
    batch_size = len(values) if gap_n is None else 4096
    for batch in values.split(batch_size):
        parts["MSP"].append(msp.score(batch).numpy())
        parts["Energy"].append(energy.score(batch).numpy())
        parts["LogitGap"].append(logit_gap.score(batch, gap_n).numpy())
    return {name: np.concatenate(arrays) for name, arrays in parts.items()}


@torch.inference_mode()
def fit_imse(x, labels, classes, device):
    x, order, offsets, whitener, means = covariance.fit_geometry(x, labels, device, classes)
    results = {}
    for name, (k, tau) in config.IMSE_SETTINGS.items():
        print(f"Fitting iMSE (k={k}, tau={tau:g})", flush=True)
        shift = covariance.fit_shift(x, order, offsets, whitener, k, tau)
        results[name] = covariance.make_state(whitener, means, shift, k)
    return results


@torch.inference_mode()
def fit_activations(train, id_val, ood_val, device):
    w, b = train["weight"], train["bias"]
    n, d = train["features"].shape
    grids = {
        "ViM": [dim for dim in config.VALIDATION_GRIDS["ViM"] if dim < d],
        "ReAct": config.VALIDATION_GRIDS["ReAct"],
        "SCALE": config.VALIDATION_GRIDS["SCALE"],
    }
    u = (-np.linalg.pinv(w) @ b).astype(np.float32)
    origin = torch.from_numpy(u).to(device)
    cov = torch.zeros(d, d, dtype=torch.float64, device=device)
    for x in _batches(train["features"], device, 4096):
        shift = (x - origin).double()
        cov.add_(shift.T @ shift)
    state = vim.build_vim((cov / n).cpu().numpy(), u, device)
    maxima = train["logits"].max(axis=1).astype(np.float64).mean()
    for dim in grids["ViM"]:
        total = sum(float(vim.vim_residual(x, state, dim).sum())
                    for x in _batches(train["features"], device, 4096))
        require(total > 0, "Invalid ViM residual mean")
        state["alpha"][str(dim)] = float(maxima) / (total / n)
    head = (torch.from_numpy(w).to(device), torch.from_numpy(b).to(device))
    thresholds = {p: float(np.percentile(id_val["features"].flatten(), p)) for p in grids["ReAct"]}

    def predict(method, bank, p, batch):
        parts = []
        for x in _batches(bank["features"], device, batch):
            if method == "ViM":
                values = vim.vim(x, F.linear(x, *head), state, p)
            elif method == "ReAct":
                values = react.react(x, *head, thresholds[p])
            else:
                values, _ = scale.scale(x, *head, p)
            parts.append(values.cpu().numpy())
        return np.concatenate(parts)

    selected = {}
    for method, candidates in grids.items():
        trials = []
        for p in candidates:
            try:
                a, b = (predict(method, bank, p, 1024) for bank in (id_val, ood_val))
                trials.append((p, metrics(a, b)["AUROC"]))
            except scale.UndefinedScore:
                continue
        require(trials, f"No valid validation candidate for {method}")
        selected[method] = max(trials, key=lambda item: item[1])[0]
    return selected, predict


@torch.inference_mode()
def evaluate(dataset, backbone, run, features, output, device, *, verify_cohort=True):
    configure_precision()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = Path(features)
    task = (dataset, backbone, run) if verify_cohort else None
    package = Path(__file__).resolve().parents[1]
    binding = dict(dataset=dataset, model=backbone, run=run, device=str(device),
                   torch=torch.__version__,
                   code={p.relative_to(package).as_posix(): hash_file(p) for p in package.rglob("*.py")},
                   features={s: hash_file(features / s / "complete.json")
                             for s in DATASETS[dataset]["splits"]})
    receipt = output / "evaluation.json"
    if receipt.exists():
        require(load_json(receipt) == binding, f"Use a new evaluation output directory: {output}")
        return summarize_scores(dataset, backbone, run, output, load_json(output / "scores.json"))
    train, id_val, ood_val = (load_split(features, s, task) for s in ("id_train", "id_val", "ood_val"))
    x, y, l, w = (train[k] for k in ("features", "labels", "logits", "weight"))
    classes = l.shape[1]
    knn_grid = config.VALIDATION_GRIDS["KNN"]
    companion_k = config.KNN_TEST_COMPANION[dataset]
    if dataset == "ImageNet1K":
        tx, ty = map(torch.from_numpy, (x, y))
        neighbors = knn.ExactNeighbors(tx, device)
        v_id, v_ood = (neighbors.score_many(torch.from_numpy(v["features"]), knn_grid) for v in (id_val, ood_val))
        boundary = fdbd.FeatureDecisionBoundary(torch.from_numpy(w), tx, device)
        gaussians = {suffix: mahalanobis.MahalanobisFamilyDetector(classes, norm, device).fit(tx, ty)
                     for suffix, norm in config.MAHALANOBIS_VARIANTS}

        def primary(bank, k):
            features, logits = map(torch.from_numpy, (bank["features"], bank["logits"]))
            out = logit_scores(bank["logits"], gap_n=config.LOGIT_GAP_TOP[dataset])
            out["KNN"] = neighbors.score_many(features, (k, companion_k))[k]
            out["fDBD"] = boundary.score(features, logits)
            for suffix, g in gaussians.items():
                _, out["Mahalanobis" + suffix], out["Relative-Mahalanobis" + suffix] = g.score_all(features)
            return out
    else:
        neighbors = knn.ExactKNN(x, device)
        v_id, v_ood = (neighbors.reference_score(v["features"], knn_grid) for v in (id_val, ood_val))
        boundary = fdbd.FDBD(x, w)
        gaussians = {suffix: mahalanobis.GaussianFamily(norm, device).fit(x, y)
                     for suffix, norm in config.MAHALANOBIS_VARIANTS}

        def primary(bank, k):
            features, logits = bank["features"], bank["logits"]
            out = logit_scores(logits, gap_n=config.LOGIT_GAP_TOP[dataset])
            out["KNN"] = neighbors.score(features, sorted({k, companion_k}))[k]
            out["fDBD"] = boundary.score(features, logits)
            for suffix, g in gaussians.items():
                out["Mahalanobis" + suffix], out["Relative-Mahalanobis" + suffix] = g.scores(features)
            return out

    selected_k = max(knn_grid, key=lambda k: metrics(v_id[k], v_ood[k])["AUROC"])
    selected, act_predict = fit_activations(train, id_val, ood_val, device)
    states = fit_imse(x, y, classes, device)
    atomic_json(output / "selection.json", {
        "KNN": selected_k, **selected,
        **{name: {"k": k, "tau": tau} for name, (k, tau) in config.IMSE_SETTINGS.items()},
    })
    del id_val, ood_val
    gc.collect()
    score_files = {}
    near, far = ood_groups(dataset)
    for split in ["id_test", *near, *far]:
        bank = load_split(features, split, task)
        scores = primary(bank, selected_k)
        for method, p in selected.items():
            scores[method] = act_predict(method, bank, p, 2048)
        parts = {name: [] for name in states}
        predicted = bank["logits"].argmax(1)
        for i, batch in enumerate(_batches(bank["features"], device, 2048)):
            labels = torch.from_numpy(predicted[i * 2048:i * 2048 + len(batch)]).to(device)
            q = F.normalize(batch, dim=1).double()
            for name, state in states.items():
                parts[name].append(covariance.score(q, labels, state).cpu().numpy())
        scores.update({name: np.concatenate(values) for name, values in parts.items()})
        path = output / f"{split}.npz"
        np.savez_compressed(path, **scores)
        score_files[split] = hash_file(path)
    table = summarize_scores(dataset, backbone, run, output, score_files)
    atomic_json(receipt, binding)
    return table


def summarize_scores(dataset, backbone, run, output, score_files):
    output = Path(output)
    scores = {}
    for split, digest in score_files.items():
        path = output / f"{split}.npz"
        require(hash_file(path) == digest, f"Changed score file: {path}")
        with np.load(path, allow_pickle=False) as bank:
            for detector in bank.files:
                scores.setdefault(detector, {})[split] = bank[detector]
    require(set(DETECTORS) == scores.keys(), "Unexpected primary detector scores")
    rows = []
    for detector, values in scores.items():
        rows.extend(score_table(dataset, backbone, run, detector, values))
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    atomic_json(output / "scores.json", score_files)
    return table

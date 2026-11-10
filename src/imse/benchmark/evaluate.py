import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from tqdm.auto import tqdm

from ..detectors import activations as activation, imse as covariance, imagenet as large
from ..detectors import cifar as small
from ..common import atomic_json, hash_file, load_json, require
from ..datasets import DATASETS, ood_groups
from ..backbones.features import configure_precision, load_split

BASELINES = (
    "MSP", "Energy", "LogitGap", "KNN", "NNGuide", "fDBD", "Mahalanobis",
    "Mahalanobis++", "Relative-Mahalanobis", "Relative-Mahalanobis++",
    "ViM", "ReAct", "SCALE",
)
DETECTORS = BASELINES + ("iMSE16", "iMSEall")
METRICS = ["AUROC", "FPR95"]
K_GRID = (50, 100, 200, 500, 1000)


def metrics(id_scores, ood_scores):
    a, b = np.asarray(id_scores), np.asarray(ood_scores)
    require(a.ndim == b.ndim == 1 and len(a) and len(b) and np.isfinite(a).all()
            and np.isfinite(b).all(), "Invalid detector scores")
    fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(a)), np.ones(len(b))], -np.r_[a, b])
    return dict(AUROC=100 * auc(fpr, tpr), FPR95=100 * fpr[np.flatnonzero(tpr >= .95)[0]])


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


@torch.inference_mode()
def fit_imse(x, labels, classes, device, settings=((16, 1.), ("all", 1.)), ablation=False):
    x, order, offsets, whitener, means = covariance.fit_geometry(x, labels, device, classes)
    results, controls = {}, {}
    for k, tau in settings:
        print(f"Fitting iMSE (k={k}, tau={tau:g})", flush=True)
        shift = covariance.fit_shift(x, order, offsets, whitener, k, tau)
        state = covariance.make_state(whitener, means, shift, k)
        results[k, tau] = state
        if (k, tau) == (16, 1.):
            controls["Matched-ID-only"] = dict(transform=state["transform"], means=state["means"],
                weights=torch.full_like(state["weights"], 2.), logdet=0.)
            if ablation:
                controls.update(ablation_states(x, order, offsets, whitener, means, shift, device))
    return results, controls


@torch.inference_mode()
def ablation_states(x, order, offsets, w, means, anchor, device):
    names = ("Local-residual", "Tangent-limit")
    moments = {name: torch.zeros_like(anchor) for name in names}
    for first, last in tqdm(zip(offsets, offsets[1:]), total=len(means), desc="Ablation moments"):
        raw = x[order[first:last]].to(device=device, dtype=torch.float32)
        q = F.normalize(raw, dim=1)
        neighbors = covariance.local_neighbors(q)
        local = torch.empty_like(q)
        for start in range(0, len(q), 128):
            local[start:start + 128] = q[neighbors[start:start + 128, :16]].mean(1)
        residual = q - local
        u = (residual / (residual.norm(dim=1, keepdim=True) + covariance.EPS)).double()
        q = q.double()
        norm = q.norm(dim=1, keepdim=True)
        tangent = u / norm - q * (q * u).sum(1, keepdim=True) / norm.pow(3)
        for name, shifts in zip(names, (residual.double(), tangent)):
            moments[name] += covariance.scatter(shifts)
    b0 = covariance.symmetric(w.T @ anchor @ w)
    kappa, _ = covariance.spectrum(b0)
    states = {}

    def add(name, eigenvalues, basis):
        transform = w @ basis
        states[name] = dict(transform=transform, means=means @ transform,
                            weights=eigenvalues / (1 + eigenvalues),
                            logdet=float(torch.log1p(eigenvalues).sum()))

    for name, matrix in moments.items():
        b = covariance.symmetric(w.T @ (matrix / len(x)) @ w)
        require(torch.trace(b) > 0 and torch.trace(b0) > 0, "Undefined trace matching")
        k, q = covariance.spectrum(b)
        add(f"{name} / natural", k, q)
        add(f"{name} / trace-matched", k * (float(torch.trace(b0)) / float(torch.trace(b))), q)
    for seed in (0, 1, 2):
        q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal(anchor.shape))
        q *= np.where(np.diag(r) < 0., -1., 1.)[None, :]
        add(f"Random-orientation / seed={seed}", kappa, torch.from_numpy(q).to(device))
    return states


@torch.inference_mode()
def fit_activations(train, id_val, ood_val, device):
    w, b = train["weight"], train["bias"]
    n, d = train["features"].shape
    grids = activation.candidates(d)
    u = (-np.linalg.pinv(w) @ b).astype(np.float32)
    origin = torch.from_numpy(u).to(device)
    cov = torch.zeros(d, d, dtype=torch.float64, device=device)
    for x in _batches(train["features"], device, 4096):
        shift = (x - origin).double()
        cov.add_(shift.T @ shift)
    state = activation.build_vim((cov / n).cpu().numpy(), u, device)
    maxima = train["logits"].max(axis=1).astype(np.float64).mean()
    for dim in grids["ViM"]:
        total = sum(float(activation.vim_residual(x, state, dim).sum())
                    for x in _batches(train["features"], device, 4096))
        require(total > 0, "Invalid ViM residual mean")
        state["alpha"][str(dim)] = float(maxima) / (total / n)
    head = (torch.from_numpy(w).to(device), torch.from_numpy(b).to(device))
    thresholds = {p: float(np.percentile(id_val["features"].flatten(), p)) for p in grids["ReAct"]}

    def predict(method, bank, p, batch):
        parts = []
        for x in _batches(bank["features"], device, batch):
            if method == "ViM":
                values = activation.vim(x, F.linear(x, *head), state, p)
            elif method == "ReAct":
                values = activation.react(x, *head, thresholds[p])
            else:
                values, _ = activation.scale(x, *head, p)
            parts.append(values.cpu().numpy())
        return np.concatenate(parts)

    selected = {}
    for method, candidates in grids.items():
        trials = []
        for p in candidates:
            try:
                a, b = (predict(method, bank, p, 1024) for bank in (id_val, ood_val))
                trials.append((p, metrics(a, b)["AUROC"]))
            except activation.UndefinedScore:
                continue
        require(trials, f"No valid validation candidate for {method}")
        selected[method] = max(trials, key=lambda item: item[1])[0]
    return selected, predict


@torch.inference_mode()
def evaluate(dataset, backbone, run, features, output, device, sensitivity=False, ablation=False,
             verify_cohort=True):
    configure_precision()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    features = Path(features)
    task = (dataset, backbone, run) if verify_cohort else None
    package = Path(__file__).resolve().parents[1]
    binding = dict(dataset=dataset, model=backbone, run=run, device=str(device),
                   torch=torch.__version__, sensitivity=sensitivity, ablation=ablation,
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
    if dataset == "ImageNet1K":
        tx, ty, tl = map(torch.from_numpy, (x, y, l))
        knn = large.ExactNeighbors(tx, device)
        v_id, v_ood = (knn.score_many(torch.from_numpy(v["features"]), K_GRID) for v in (id_val, ood_val))
        order = train.get("official_order", np.arange(len(tx)))
        guide = large.NearestNeighborGuide(tx, tl, large.openood_nnguide_indices(order), device)
        fdbd = large.FeatureDecisionBoundary(torch.from_numpy(w), tx, device)
        gaussians = [large.MahalanobisFamilyDetector(classes, norm, device).fit(tx, ty)
                     for norm in (False, True)]

        def primary(bank, k):
            features, logits = map(torch.from_numpy, (bank["features"], bank["logits"]))
            out = {m: large.score_logits(logits, m) for m in ("MSP", "Energy", "LogitGap")}
            out["KNN"] = knn.score_many(features, (k, 1000))[k]
            out["NNGuide"] = guide.score_many(features, logits, (100,))[100]
            out["fDBD"] = fdbd.score(features, logits)
            for g, suffix in zip(gaussians, ("", "++")):
                _, out["Mahalanobis" + suffix], out["Relative-Mahalanobis" + suffix] = g.score_all(features)
            return out
    else:
        knn = small.ExactKNN(x, device)
        v_id, v_ood = (knn.reference_score(v["features"], K_GRID) for v in (id_val, ood_val))
        guide = small.NNGuide(x, l, device)
        fdbd = small.FDBD(x, w)
        gaussians = [small.GaussianFamily(norm, device).fit(x, y) for norm in (False, True)]

        def primary(bank, k):
            features, logits = bank["features"], bank["logits"]
            out = small.logit_scores(logits)
            companion_k = 50 if dataset == "CIFAR10" else 200
            out["KNN"] = knn.score(features, sorted({k, companion_k}))[k]
            out["NNGuide"] = guide.score(features, logits, (10, 100))[100]
            out["fDBD"] = fdbd.score(features, logits)
            for g, suffix in zip(gaussians, ("", "++")):
                out["Mahalanobis" + suffix], out["Relative-Mahalanobis" + suffix] = g.scores(features)
            return out

    selected_k = max(K_GRID, key=lambda k: metrics(v_id[k], v_ood[k])["AUROC"])
    selected, act_predict = fit_activations(train, id_val, ood_val, device)
    settings = [(16, 1.), ("all", 1.)]
    if sensitivity:
        settings += [(k, 1.) for k in (4, 8, 32, 64, 128, 256, 512)]
        settings += [(16, tau) for tau in (.1, .25, .5, 2., 4.)]
    imse, controls = fit_imse(x, y, classes, device, settings, ablation)
    states = {f"iMSE{k}" if tau == 1 else f"iMSE{k}-tau{tau:g}": state
              for (k, tau), state in imse.items()}
    states.update(controls)
    atomic_json(output / "selection.json", {"KNN": selected_k, **selected,
                                            "NNGuide_bank_size": len(guide.indices)})
    random_control = None
    if dataset == "CIFAR10" and backbone == "ResNet18":
        random_control = small.NNGuide(x, l, device, random_bank=True)
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
        if random_control is not None:
            scores["NNGuide-IDRandom-K100"] = random_control.score(
                bank["features"], bank["logits"], (10, 100))[100]
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
    require(set(DETECTORS) <= scores.keys(), "Missing primary detector scores")
    rows = []
    for detector, values in scores.items():
        rows.extend(score_table(dataset, backbone, run, detector, values))
    table = pd.DataFrame(rows)
    table[table.Detector.isin(DETECTORS)].to_csv(output / "metrics.csv", index=False)
    table[~table.Detector.isin(DETECTORS)].to_csv(output / "supplementary_metrics.csv", index=False)
    atomic_json(output / "scores.json", score_files)
    return table[table.Detector.isin(DETECTORS)]

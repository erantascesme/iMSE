import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..backbones import runs
from .roc import ROCPlan, multiplicities, reference
from ..common import ASSETS, checked, load_json, require
from ..datasets import DATASETS, ood_groups

LABELS = ("iMSE (k=16)", "Matched ID-only", "MDS++")
METRICS = ("AUROC", "FPR95_OOD_positive")
METHODS = ("iMSE16", "Matched-ID-only", "Mahalanobis++")


def plans_for(pair):
    return [[[ROCPlan(cp['values']['id_test'][m], cp['values'][s][m])
              for s in pair['near']+pair['far']] for m in range(3)]
            for cp in pair['checkpoints']]

def bootstrap_batch(pair, plans, start, stop, seed, device):
    ds = pair['dataset']
    cp0 = pair['checkpoints'][0]
    ni = cp0['values']['id_test'].shape[1]
    wi = multiplicities(ni, start, stop, seed, [ds, 'id_test', pair['streams']['id_test']])
    gpu = device != 'cpu'
    if gpu:
        import torch
        wi = torch.as_tensor(wi, device=device)
    parts = pair['near']+pair['far']
    result = np.zeros((stop-start, 3, len(parts), 2), dtype=np.float64)
    for j, split in enumerate(parts):
        no = cp0['values'][split].shape[1]
        wo = multiplicities(no, start, stop, seed, [ds, split, pair['streams'][split]])
        if gpu:
            wo = torch.as_tensor(wo, device=device)
        for r in range(len(pair['checkpoints'])):
            for m in range(3):
                values = plans[r][m][j].torch(wi, wo).cpu().numpy() if gpu else plans[r][m][j].numpy(wi, wo)
                result[:, m, j] += values / len(pair['checkpoints'])
    require(np.isfinite(result).all() and ((result >= -1e-10) & (result <= 100+1e-10)).all(), 'Invalid bootstrap metrics')
    return result

def scopes(values, near_count, names):
    # values [..., method, OOD dataset, metric]; axes stay intact for point/replicates.
    result = {name: values[..., j, :] for j, name in enumerate(names)}
    result.update(Near=values[..., :near_count, :].mean(-2),
                  Far=values[..., near_count:, :].mean(-2), All=values.mean(-2))
    return result


def effect_vectors(points, draws, near_count, names, models, dataset):
    """points [K,3,J,2], draws [K,B,3,J,2]; checkpoints already averaged."""
    points, draws = np.asarray(points), np.asarray(draws)
    require(points.shape == (len(models), 3, len(names), 2), 'Point estimate axes changed')
    require(draws.ndim == 5 and draws.shape[0] == len(models)
                 and draws.shape[2:] == points.shape[1:], 'Replicate axes changed')
    require(np.isfinite(points).all() and np.isfinite(draws).all(), 'Nonfinite metrics')
    all_points = np.concatenate([points, points.mean(0, keepdims=True)], axis=0)
    all_draws = np.concatenate([draws, draws.mean(0, keepdims=True)], axis=0)
    ps = scopes(all_points, near_count, names)
    bs = scopes(all_draws, near_count, names)
    rows, observed, samples = [], [], []
    for scope in ['All', 'Near', 'Far']:
        for k, model in enumerate([*models, 'Backbone average']):
            for comp in [1, 2]:
                for m, metric in enumerate(METRICS):
                    sign = 1 if m == 0 else -1
                    observed.append(sign*(ps[scope][k, 0, m]-ps[scope][k, comp, m]))
                    samples.append(sign*(bs[scope][k, :, 0, m]-bs[scope][k, :, comp, m]))
                    rows.append(dict(Dataset=dataset, Backbone=model, Scope=scope, Metric=metric,
                        Comparator=LABELS[comp], LMSE_pct=ps[scope][k, 0, m],
                        Comparator_pct=ps[scope][k, comp, m],
                        Backbone_count=len(models) if model == 'Backbone average' else 1))
    return pd.DataFrame(rows), np.asarray(observed), np.stack(samples, axis=1)

def simultaneous(observed, draws, confidence=.95):
    observed, draws = np.asarray(observed, dtype=float), np.asarray(draws, dtype=float)
    require(draws.ndim == 2 and draws.shape[1:] == observed.shape and len(draws) > 1,
                 'Wrong simultaneous interval dimensions')
    require(np.isfinite(draws).all() and np.isfinite(observed).all(), 'Nonfinite effects')
    sd = draws.std(axis=0, ddof=1)
    active = sd > 1e-12
    critical = None
    low, high = np.full(len(observed), -100.), np.full(len(observed), 100.)
    if active.any():
        maximum = np.max(np.abs((draws[:, active]-observed[active])/sd[active]), axis=1)
        # Conservative empirical order statistic; no extrapolation into sparse tails.
        critical = float(np.quantile(maximum, confidence, method='higher'))
        low[active] = observed[active]-critical*sd[active]
        high[active] = observed[active]+critical*sd[active]
    return low, high, sd, active, critical

def family_reports(meta, observed, draws, seed):
    result = meta.copy()
    lo, hi = np.quantile(draws, [.025, .975], axis=0, method='linear')
    result['Improvement_pp'] = observed
    result['CI95_low_pp'], result['CI95_high_pp'] = lo, hi
    result['Repetitions'], result['Seed'] = len(draws), seed
    result['Independent_confirmation'] = False
    records, convergence = [], []
    for family, mask in [('Primary_All', meta.Scope.eq('All').to_numpy()),
                         ('Secondary_Near_Far', meta.Scope.isin(['Near', 'Far']).to_numpy())]:
        indices = np.flatnonzero(mask)
        low, high, sd, active, critical = simultaneous(observed[mask], draws[:, mask])
        result.loc[mask, 'Family'] = family
        result.loc[mask, 'Family_size'] = len(indices)
        result.loc[mask, 'Sim95_low_pp'] = low
        result.loc[mask, 'Sim95_high_pp'] = high
        result.loc[mask, 'Bootstrap_SD_pp'] = sd
        result.loc[mask, 'Degenerate'] = ~active
        result.loc[mask, 'Critical_value'] = critical
        for i, a, l, h in zip(indices, active, low, high):
            result.loc[i, 'Simultaneous_interpretation'] = (
                'unresolved: zero bootstrap variance' if not a else
                'positive interval' if l > 0 else 'negative interval' if h < 0 else 'includes zero')
        records.append(dict(dataset=str(meta.Dataset.iloc[0]), family=family, size=len(indices),
            active_effects=int(active.sum()), degenerate_effects=int((~active).sum()),
            critical_value=critical, confidence=.95, finite_sample_exact=False,
            members=meta.loc[mask, ['Backbone','Scope','Comparator','Metric']].to_dict('records')))
        for b in sorted(set([min(500, len(draws)), min(1000, len(draws)), len(draws)])):
            sl, sh, _, valid, crit = simultaneous(observed[mask], draws[:b, mask])
            pl, ph = np.quantile(draws[:b, mask], [.025,.975], axis=0)
            for j, i in enumerate(indices):
                convergence.append({**meta.iloc[i].to_dict(), 'Family':family, 'Prefix_repetitions':b,
                    'CI95_low_pp':pl[j], 'CI95_high_pp':ph[j],
                    'Sim95_low_pp':sl[j], 'Sim95_high_pp':sh[j], 'Critical_value':crit,
                    'Pointwise_endpoint_change_vs_full_pp':float(max(abs(pl[j]-lo[i]),abs(ph[j]-hi[i]))),
                    'Simultaneous_endpoint_change_vs_full_pp':float(max(abs(sl[j]-low[j]),abs(sh[j]-high[j]))),
                    'Degenerate':not bool(valid[j])})
    return result, records, pd.DataFrame(convergence)


def run_dataset(dataset, root, output, device, batch_size=8):
    near, far = ood_groups(dataset)
    names = near + far
    streams = load_json(ASSETS / "bootstrap_cohorts.json")[dataset]
    groups = {}
    for model, run in runs(dataset):
        groups.setdefault(model, []).append(run)
    points, replicates = [], []
    for model, model_runs in groups.items():
        checkpoints, checkpoint_points = [], []
        for run in model_runs:
            folder = root / dataset / model / run
            receipt = load_json(folder / "scores.json")
            values = {}
            for split in ("id_test", *names):
                with np.load(checked(folder / f"{split}.npz", receipt[split]), allow_pickle=False) as bank:
                    values[split] = np.stack([bank[name] for name in METHODS]).astype(np.float64)
                require(values[split].shape == (3, DATASETS[dataset]["splits"][split]["count"]),
                        f"Unexpected bootstrap cohort: {dataset}/{model}/{run}/{split}")
            checkpoints.append({"values": values})
            checkpoint_points.append(np.array([[reference(values["id_test"][m], values[s][m])
                                                for s in names] for m in range(len(METHODS))]))
        pair = dict(dataset=dataset, checkpoints=checkpoints, near=near, far=far, streams=streams)
        plans = plans_for(pair)
        draws = [bootstrap_batch(pair, plans, start, min(start + batch_size, 2000), 42, device)
                 for start in tqdm(range(0, 2000, batch_size), desc=f"Paired bootstrap {model}")]
        points.append(np.mean(checkpoint_points, axis=0))
        replicates.append(np.concatenate(draws))
    meta, observed, draws = effect_vectors(points, replicates, len(near), names, list(groups), dataset)
    result, _, convergence = family_reports(meta, observed, draws, 42)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / f"{dataset}_paired_intervals.csv", index=False)
    convergence.to_csv(output / f"{dataset}_convergence.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Seed-42 paired bootstrap on the reproduced scores")
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--scores", type=Path, default=Path("work/results"))
    parser.add_argument("--output", type=Path, default=Path("work/bootstrap"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    for dataset in args.dataset:
        run_dataset(dataset, args.scores, args.output, args.device)


if __name__ == "__main__":
    main()

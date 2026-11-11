from pathlib import Path

import numpy as np
import pandas as pd

from ..common import require
from ..config import DETECTORS
from ..datasets import ood_groups

METRICS = ["AUROC", "FPR95"]
KEY = ["Dataset", "Backbone", "Run", "Detector", "Scope"]


def aggregate(rows):
    rows = rows[rows.Detector.isin(DETECTORS)].copy()
    require(not rows.duplicated(KEY).any(), "Duplicate metric rows")
    require(np.isfinite(rows[METRICS].to_numpy()).all(), "Nonfinite metrics")
    groups = KEY[:-1]
    individual = rows[~rows.Scope.isin(["Near", "Far", "All"])]
    macro = []
    for group, frame in individual.groupby(groups, sort=False):
        near, far = ood_groups(group[0])
        require(set(frame.Scope) == set(near + far), f"Incomplete OOD populations: {group}")
        for name, splits in (("Near", near), ("Far", far), ("All", near + far)):
            values = frame.set_index("Scope").loc[splits, METRICS].mean()
            macro.append(dict(zip(groups, group), Scope=name, **values.to_dict()))
    runs = pd.concat([individual, pd.DataFrame(macro)], ignore_index=True)
    require(len(runs) > 0, "No scores to aggregate")
    pair_keys = ["Dataset", "Backbone", "Detector", "Scope"]
    pairs = runs.groupby(pair_keys, sort=False)[METRICS].mean().reset_index()
    avg = pairs.groupby(["Dataset", "Detector", "Scope"], sort=False)[METRICS].mean().reset_index()
    return runs, pairs, avg


def _display_name(name):
    return {"iMSE16": "iMSE (k=16)", "iMSEall": "iMSE (k=all)",
            "Mahalanobis": "MDS", "Mahalanobis++": "MDS++",
            "Relative-Mahalanobis": "RMDS", "Relative-Mahalanobis++": "RMDS++"}.get(name, name)


def display_table(pairs, averages, dataset, scope="All"):
    selected = pairs[(pairs.Dataset == dataset) & (pairs.Scope == scope)].copy()
    if averages is not None:
        avg = averages[(averages.Dataset == dataset) & (averages.Scope == scope)].copy()
        selected = pd.concat([selected, avg.assign(Backbone="Average")], ignore_index=True)
    selected["Score"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(selected.AUROC, selected.FPR95)]
    table = selected.pivot(index="Detector", columns="Backbone", values="Score")
    return table.reindex([d for d in DETECTORS if d in table.index]).rename(index=_display_name)


def write_reports(rows, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    results = aggregate(rows)
    for name, frame in zip(("checkpoints", "backbones", "dataset_means"), results):
        frame.to_csv(directory / f"{name}.csv", index=False)
    return results

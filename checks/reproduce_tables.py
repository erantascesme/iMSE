from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from imse.backbones import runs
from imse.datasets import DATASETS, image_key, manifest
from imse.benchmark.evaluate import DETECTORS
from imse.backbones.features import extraction_layout
from imse.benchmark.report import KEY, METRICS, aggregate

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference"


def run():
    assets = json.loads((REF / "asset_sha256.json").read_text())
    for name, digest in assets.items():
        assert hashlib.sha256((ROOT / "src/imse/assets" / name).read_bytes()).hexdigest() == digest
    checkpoints = pd.read_csv(REF / "checkpoints.csv")
    expected = {(ds, model, run) for ds in DATASETS for model, run in runs(ds)}
    assert set(checkpoints.Detector) == set(DETECTORS)
    assert set(zip(checkpoints.Dataset, checkpoints.Backbone, checkpoints.Run)) == expected
    for dataset, record in DATASETS.items():
        membership = {}
        for split, spec in record["splits"].items():
            rows = manifest(dataset, split)
            if dataset == "ImageNet1K":
                names = {image_key(spec["asset"], name) for name, _ in rows}
            else:
                names = {name for name, _ in rows}
            assert len(names) == spec["count"]
            membership[split] = names if split != "id_train" else set()
            for model, run in runs(dataset):
                layout = extraction_layout(dataset, model, run, split)
                assert layout[0][0] == 0 and layout[-1][1] == spec["count"]
                assert all(a < b and size > 0 for a, b, size in layout)
                assert all(a[1] == b[0] for a, b in zip(layout, layout[1:]))
        assert membership["id_val"].isdisjoint(membership["id_test"])
        for split in record["splits"]:
            if split not in {"id_train", "id_val", "id_test", "ood_val"}:
                assert membership[split].isdisjoint(membership["ood_val"])
    ck, pairs, averages = aggregate(checkpoints)
    for actual, filename, keys in [(ck, "checkpoints", KEY),
                                   (pairs, "pairs", [k for k in KEY if k != "Run"]),
                                   (averages, "dataset_means", ["Dataset", "Detector", "Scope"])]:
        expected_table = pd.read_csv(REF / f"{filename}.csv").set_index(keys).sort_index()
        actual = actual.set_index(keys).sort_index()
        assert actual.index.equals(expected_table.index)
        np.testing.assert_allclose(actual[METRICS], expected_table[METRICS], rtol=0, atol=1e-10)
    run_idx = ck.set_index(KEY)
    pair_idx = pairs.set_index([k for k in KEY if k != "Run"])
    avg_idx = averages.set_index(["Dataset", "Detector", "Scope"])
    cells = pd.read_csv(REF / "table_cell_audit.csv", dtype={"latex": str})
    for row in cells.itertuples(index=False):
        suffix = row.Detector, row.Scope
        if pd.isna(row.Run):
            if row.Backbone == "Average":
                value = avg_idx.loc[(row.Dataset, *suffix), row.Metric]
            else:
                value = pair_idx.loc[(row.Dataset, row.Backbone, *suffix), row.Metric]
        elif row.Run == "Mean":
            value = pair_idx.loc[(row.Dataset, row.Backbone, *suffix), row.Metric]
        else:
            value = run_idx.loc[(row.Dataset, row.Backbone, row.Run, *suffix), row.Metric]
        assert abs(value - row.value) < 1e-10, row
        if pd.notna(row.SD):
            checkpoints_here = ck[(ck.Dataset == row.Dataset) & (ck.Backbone == row.Backbone)
                                  & (ck.Detector == row.Detector) & (ck.Scope == row.Scope)]
            assert abs(checkpoints_here[row.Metric].std(ddof=1) - row.SD) < 1e-10
    print(f"PASS: {len(expected)} checkpoints, {len(ck)} result rows, {len(cells)} audited table cells")


if __name__ == "__main__":
    run()

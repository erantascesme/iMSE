"""Recompute published benchmark metrics from the authors' original score exports."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from imse.datasets import ood_groups
from imse.benchmark.evaluate import BASELINES, score_table
from imse.benchmark.report import compare

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_METHODS = BASELINES[:10]


def arrays(archive, name):
    with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as bank:
        return {k: bank[k] for k in bank.files}


def imagenet(path):
    rows = []
    hashes = json.loads((ROOT / "reference/imagenet_score_sha256.json").read_text())
    with ZipFile(path) as z:
        for name in z.namelist():
            if "/scores/" not in name or Path(name).stem not in ORIGINAL_METHODS:
                continue
            model = name.split("/")[0]
            detector = Path(name).stem
            assert hashlib.sha256(z.read(name)).hexdigest() == hashes[model][detector]
            raw = arrays(z, name)
            raw["id_test"] = raw.pop("test")
            rows.extend(score_table("ImageNet1K", model, "fixed", detector, raw))
    assert len(rows) == 8 * 10 * 8
    return rows


def cifar(path):
    rows = []
    with ZipFile(path) as z:
        for model in ("DenseNet-BC-100-ODIN", "ViT-B16-CIFAR10"):
            base = f"evaluation/{model}/published"
            receipt = json.loads(z.read(f"{base}/complete.json"))
            near, far = ood_groups("CIFAR10")
            scores = {}
            for split in ("id_test", *near, *far):
                relative = f"scores/{split}.npz"
                name = f"{base}/{relative}"
                assert hashlib.sha256(z.read(name)).hexdigest() == receipt["files"][relative]
                scores[split] = arrays(z, name)
            for detector in ORIGINAL_METHODS:
                rows.extend(score_table("CIFAR10", model, "published", detector,
                                        {s: v[detector] for s, v in scores.items()}))
    assert len(rows) == 2 * 10 * 9
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet", required=True)
    parser.add_argument("--cifar10", required=True)
    args = parser.parse_args()
    actual = pd.DataFrame(imagenet(args.imagenet) + cifar(args.cifar10))
    ref = pd.read_csv(ROOT / "reference/checkpoints.csv")
    keys = ["Dataset", "Backbone", "Run", "Detector", "Scope"]
    expected = ref.merge(actual[keys], on=keys, validate="one_to_one")
    compared = compare(actual, expected)
    delta = compared[["AUROC_delta_pp", "FPR95_delta_pp"]].abs().max()
    assert delta.max() < 1e-9, delta
    print(f"PASS: {len(actual)} checkpoint metric rows recomputed from original scores")
    print(delta.to_string())


if __name__ == "__main__":
    main()

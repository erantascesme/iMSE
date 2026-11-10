import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .checkpoints import forward, load
from ..common import ASSETS, atomic_json, checked, hash_file, load_json, require
from ..datasets import DATASETS, ImageSource

ARRAYS = ("features", "logits", "labels", "weight", "bias")


def configure_precision():
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def load_split(root, split, task=None):
    folder = Path(root) / split
    receipt = load_json(folder / "complete.json")
    arrays = {name: np.load(checked(folder / f"{name}.npy", digest),
                            mmap_mode="c", allow_pickle=False)
              for name, digest in receipt["files"].items()}
    require(len(arrays["features"]) == receipt["rows"] == len(arrays["labels"]) ==
            len(arrays["logits"]), f"Feature cache length mismatch: {folder}")
    if task is not None:
        ds, model, run = task
        binding = receipt["binding"]
        spec = DATASETS[ds]["splits"][split]
        require((binding["dataset"], binding["model"], binding["run"]) == task
                and binding["manifest"] == spec["sha256"] and receipt["rows"] == spec["count"],
                f"Wrong benchmark cache: {folder}")
    return arrays


def _finish(folder, binding, extra=()):
    files = {name: hash_file(folder / f"{name}.npy") for name in (*ARRAYS, *extra)}
    atomic_json(folder / "complete.json", {"binding": binding, "files": files, "rows": binding["rows"]})
    (folder / "progress.json").unlink(missing_ok=True)


def extraction_layout(dataset, name, run, split):
    if dataset == "ImageNet1K":
        n = DATASETS[dataset]["splits"][split]["count"]
        return [[0, n, 16 if split == "ood_val" else 24]]
    layout = load_json(ASSETS / "cifar_feature_layout.json")
    return layout[f"{dataset}/{name}"]["runs"][run][split]


@torch.inference_mode()
def extract(dataset, name, run, data_root, weights, root, device):
    root = Path(root)
    splits = DATASETS[dataset]["splits"]
    missing = []
    for split, spec in splits.items():
        done = root / split / "complete.json"
        if done.exists():
            binding = load_json(done)["binding"]
            require(binding["dataset"] == dataset and binding["model"] == name
                    and binding["run"] == run and binding["manifest"] == spec["sha256"],
                    f"Cached split belongs to another experiment: {done}")
        else:
            missing.append(split)
    if not missing:
        return
    configure_precision()
    model, head, transform, digest = load(dataset, name, run, weights, device)
    with ImageSource(dataset, data_root) as source:
        for split in missing:
            spec = splits[split]
            folder = root / split
            binding = dict(dataset=dataset, model=name, run=run, rows=spec["count"],
                           manifest=spec["sha256"], checkpoint=digest,
                           layout=extraction_layout(dataset, name, run, split),
                           device=str(device), torch=torch.__version__)
            records = source.records(split)
            _extract_split(source, records, model, head, transform, folder, binding, device)


@torch.inference_mode()
def _extract_split(source, records, model, head, transform, folder, binding, device):
    folder.mkdir(parents=True, exist_ok=True)
    progress = folder / "progress.json"
    n = binding["rows"]
    shapes = {"features": (n, head.in_features), "logits": (n, head.out_features), "labels": (n,)}
    start = 0
    if progress.exists():
        record = load_json(progress)
        require(record["binding"] == binding, f"Incompatible interrupted cache: {folder}")
        start = record["next_row"]
    arrays = {key: np.lib.format.open_memmap(folder / f"{key}.npy", "r+" if start else "w+",
                                           np.int64 if key == "labels" else np.float32, shape)
              for key, shape in shapes.items()}
    if not start:
        for key in ("weight", "bias"):
            np.save(folder / f"{key}.npy", getattr(head, key).detach().cpu().numpy())
        atomic_json(progress, {"binding": binding, "next_row": 0})
    batches = [(a, min(a + size, end)) for first, end, size in binding["layout"]
               for a in range(first, end, size) if a >= start]
    for index, (offset, end) in enumerate(tqdm(batches, desc=str(folder.name))):
        rows = records[offset:end]
        images = torch.stack([transform(source.open(asset, member))
                              for _, _, asset, member in rows]).to(device)
        x, logits = forward(binding["dataset"], binding["model"], model, head, images)
        require(torch.isfinite(x).all() and torch.isfinite(logits).all(), "Nonfinite model output")
        require(torch.allclose(F.linear(x, head.weight, head.bias), logits, atol=2e-3, rtol=2e-3),
                "Cached features do not reconstruct classifier logits")
        arrays["features"][offset:end] = x.cpu().numpy()
        arrays["logits"][offset:end] = logits.cpu().numpy()
        arrays["labels"][offset:end] = [y for _, y, _, _ in rows]
        if (index + 1) % 100 == 0 or end == n:
            for array in arrays.values():
                array.flush()
            atomic_json(progress, {"binding": binding, "next_row": end})
    _finish(folder, binding)

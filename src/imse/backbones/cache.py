import json
from pathlib import Path

import numpy as np
import torch

from ..common import ASSETS, checked, hash_file, load_json, require
from ..datasets import DATASETS, manifest
from .features import _finish
from ..datasets.manifests import imagenet_key

IMAGE_PATH = Path("Imagenet/audited_experiments/openood_aligned_v2_2026_09_09")


def _pt(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _cifar(legacy, dataset, model, run):
    if dataset == "CIFAR100":
        source = legacy / "CIFAR100_OpenOOD/four_backbones_v1/features" / model
    elif model == "ResNet18":
        source = legacy / "CIFAR10_OpenOOD/features_resnet18_v1"
    else:
        source = legacy / "CIFAR10_OpenOOD/published_backbones_v1/features" / model
    pin = load_json(ASSETS / "cifar_feature_layout.json")[f"{dataset}/{model}"]["sha256"]
    registry = load_json(checked(source / "feature_registry.json", pin))["runs"][run]
    spec = registry["head"]
    with np.load(checked(source / spec["path"], spec["sha256"]), allow_pickle=False) as head:
        weight, bias = head["weight"], head["bias"]

    def parts(split, rows):
        for chunk in registry["splits"][split]["chunks"]:
            p = checked(source / chunk["path"], chunk["sha256"])
            expected = rows[chunk["start"]:chunk["stop"]]
            with np.load(p, allow_pickle=False) as data:
                require(np.array_equal(data["sample_ids"], [name for name, _ in expected])
                        and np.array_equal(data["labels"], [y for _, y in expected]),
                        "Saved CIFAR rows differ from manifest")
                yield data["features"], data["logits"], data["labels"], data["sample_ids"].tolist()
    return weight, bias, parts


def _imagenet(legacy, model):
    from . import CHECKPOINTS
    import hashlib

    plan = load_json(ASSETS / "imagenet_feature_layout.json")[model]
    head = _pt(legacy / IMAGE_PATH / model / "checkpoint_head.pt")
    spec = CHECKPOINTS[f"ImageNet1K/{model}"]
    for name in ("weight", "bias"):
        tensor = head[name].contiguous()
        prefix = str((str(tensor.dtype), tuple(tensor.shape))).encode()
        digest = hashlib.sha256(prefix + tensor.numpy().tobytes()).hexdigest()
        require(digest == spec[f"head_{name}_sha256"], f"Changed ImageNet head: {name}")

    def parts(split, rows):
        if split == "ood_val":
            path = legacy / IMAGE_PATH / model / "new_features/ood_validation/features.pt"
            bank = _pt(checked(path, plan["ood_sha256"]))
            require(np.array_equal(bank["labels"], [y for _, y in rows]), "Wrong validation labels")
            yield *(bank[k].numpy() for k in ("features", "logits", "labels")), [p for p, _ in rows]
            return
        record_key = "full_id" if split in ("id_val", "id_test") else split
        records = plan["train"] if split == "id_train" else [plan["queries"][record_key]]
        key = imagenet_key if split in ("id_train", "id_val", "id_test") else lambda s: s.replace("\\", "/")
        expected = {key(name): y for name, y in rows}
        require(len(expected) == len(rows), "Nonunique ImageNet manifest rows")
        for entry in records:
            path = checked(legacy / entry["path"], entry["bank_sha256"])
            bank = _pt(path)
            ids = checked(path.with_suffix(".sample_ids.txt"), entry["ids_sha256"]).read_text(
                encoding="utf-8").splitlines()
            if split == "id_train":
                ids = [key(name) for name in ids]
                wanted = [expected.pop(name) for name in ids]
                indices = np.arange(len(ids))
            else:
                lookup = {key(name): i for i, name in enumerate(ids)}
                ids = [key(name) for name, _ in rows]
                wanted = [y for _, y in rows]
                indices = np.array([lookup[name] for name in ids])
            for start in range(0, len(indices), 8192):
                ix = indices[start:start + 8192]
                x, l, y = [bank[k][ix].numpy() for k in ("features", "logits", "labels")]
                require(np.array_equal(y, wanted[start:start + len(ix)]), "Wrong ImageNet labels")
                yield x, l, y, ids[start:start + len(ix)]
        if split == "id_train":
            require(not expected, "Missing ImageNet training rows")
    return head["weight"].numpy(), head["bias"].numpy(), parts


def import_cache(dataset, model, run, legacy, root):
    legacy, root = Path(legacy).resolve(), Path(root).resolve()
    require(not root.is_relative_to(legacy), "Write imported caches outside the original experiment root")
    if dataset == "ImageNet1K":
        weight, bias, parts = _imagenet(legacy, model)
    else:
        weight, bias, parts = _cifar(legacy, dataset, model, run)
    for split, spec in DATASETS[dataset]["splits"].items():
        print(f"Import {dataset}/{model}/{run}/{split}", flush=True)
        folder = root / split
        n, classes, d = spec["count"], *weight.shape
        binding = dict(dataset=dataset, model=model, run=run, rows=n, manifest=spec["sha256"],
                       origin="authenticated experiment cache")
        done = folder / "complete.json"
        if done.exists():
            require(load_json(done)["binding"] == binding, f"Use a new cache directory: {folder}")
            continue
        folder.mkdir(parents=True, exist_ok=True)
        rows = manifest(dataset, split)
        shapes = {"features": (n, d), "logits": (n, classes), "labels": (n,)}
        arrays = {k: np.lib.format.open_memmap(folder / f"{k}.npy", "w+",
                                             np.int64 if k == "labels" else np.float32, shape)
                  for k, shape in shapes.items()}
        pos, ids = 0, []
        for x, l, y, names in parts(split, rows):
            require(x.dtype == l.dtype == np.float32 and y.dtype == np.int64, "Unexpected cache dtype")
            end = pos + len(y)
            for k, values in zip(arrays, (x, l, y)):
                arrays[k][pos:end] = values
            if dataset == "ImageNet1K" and split == "id_train":
                ids.extend(names)
            pos = end
        require(pos == n, f"Incomplete split: {split}")
        extra = []
        if ids:
            lookup = {name: i for i, name in enumerate(ids)}
            np.save(folder / "official_order.npy", [lookup[imagenet_key(name)] for name, _ in rows])
            extra.append("official_order")
        for array in arrays.values():
            array.flush()
        np.save(folder / "weight.npy", weight)
        np.save(folder / "bias.npy", bias)
        _finish(folder, binding, extra)

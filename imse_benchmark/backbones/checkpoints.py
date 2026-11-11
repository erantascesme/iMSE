import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import torch
from torch import nn

from ..common import ASSETS, checked, load_json, require
from ..datasets import DATASETS
from ..datasets.download import download_drive

CHECKPOINTS = load_json(ASSETS / "checkpoints.json")


def runs(dataset, backbone=None):
    for key, lock in CHECKPOINTS.items():
        ds, model = key.split("/")
        if ds == dataset and backbone in (None, model):
            for run in lock.get("runs", ["fixed" if ds == "ImageNet1K" else "published"]):
                yield model, run


def checkpoint(dataset, model, run, root):
    from huggingface_hub import hf_hub_download

    spec = CHECKPOINTS[f"{dataset}/{model}"]
    root = Path(root) / dataset / model
    root.mkdir(parents=True, exist_ok=True)
    if "runs" in spec:
        child = spec["runs"][run]
        bundled = ASSETS.parent.parent / "checkpoints" / dataset / model / run / "best.ckpt"
        if bundled.is_file():
            return checked(bundled, child["checkpoint_sha256"]), child["checkpoint_sha256"]
        path = root / run / "best.ckpt"
        if not path.exists():
            archive = download_drive(root / spec["filename"], spec["google_drive_id"],
                                     spec["archive_sha256"])
            with zipfile.ZipFile(archive) as handle:
                data = handle.read(child["member"])
            require(hashlib.sha256(data).hexdigest() == child["checkpoint_sha256"],
                    "Unexpected OpenOOD checkpoint member")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".part")
            temp.write_bytes(data)
            os.replace(temp, path)
        return checked(path, child["checkpoint_sha256"]), child["checkpoint_sha256"]
    spec = spec.get("source", spec)
    path = root / spec["filename"]
    if not path.exists():
        if "repo" in spec:
            name = spec["filename"] if dataset == "ImageNet1K" else "pytorch_model.bin"
            path = hf_hub_download(spec["repo"], name, revision=spec["revision"])
        else:
            import requests

            temp = path.with_suffix(".part")
            with requests.get(spec["url"], stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                with temp.open("wb") as f:
                    shutil.copyfileobj(response.raw, f)
            checked(temp, spec["sha256"])
            os.replace(temp, path)
    return checked(path, spec["sha256"]), spec["sha256"]


def load(dataset, name, run, root, device):
    import timm
    from timm.data import create_transform, resolve_model_data_config
    from torchvision import transforms as T
    from safetensors.torch import load_file

    require(timm.__version__ == "1.0.11", "Install timm==1.0.11")
    path, digest = checkpoint(dataset, name, run, root)
    spec = CHECKPOINTS[f"{dataset}/{name}"]
    classes = DATASETS[dataset]["classes"]
    architecture = spec["architecture"]
    transform_spec = spec["transform"]
    if dataset == "ImageNet1K":
        source = spec["source"]
        if name == "Vim-Base":
            from ._vim_runtime import prepare_vim_source

            vim_root = Path(root) / "Vim-source"
            model = prepare_vim_source(vim_root)(pretrained=False, num_classes=classes,
                                                **spec["architecture_kwargs"])
            for filename, checksum in source.get("source_patch_sha256", {}).items():
                checked(vim_root / filename, checksum)
            state = torch.load(path, map_location="cpu", weights_only=False)[source["state_key"]]
            model.load_state_dict(state, strict=True)
            head = model.head
            transform = T.Compose([
                T.Resize(transform_spec["resize"], interpolation=T.InterpolationMode(transform_spec["interpolation"])),
                T.CenterCrop(transform_spec["crop"]), T.ToTensor(),
                T.Normalize(transform_spec["mean"], transform_spec["std"]),
            ])
        else:
            model = timm.create_model(architecture, pretrained=False, num_classes=classes)
            state = load_file(str(path)) if str(path).endswith(".safetensors") else torch.load(
                path, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            if name == "SwinV2-B":
                from ._swin import _load_swin_state_strict

                _load_swin_state_strict(model, state, Path(root) / "swin_buffer_check.json")
            else:
                model.load_state_dict(state, strict=True)
            head = model.get_classifier()
            config = resolve_model_data_config(model, use_test_size=True)
            for key in ("input_size", "crop_pct", "interpolation", "mean", "std"):
                actual = list(config[key]) if isinstance(config[key], tuple) else config[key]
                require(actual == transform_spec[key], f"Preprocessing drift: {name}/{key}")
            transform = create_transform(**config, is_training=False)
    else:
        is_vit = name.startswith("ViT")
        if "runs" in spec:
            from ._resnet18 import ResNet18_32x32

            model = ResNet18_32x32(num_classes=classes)
        elif is_vit:
            model = timm.create_model(architecture, pretrained=False, num_classes=classes)
            for block in model.blocks:
                block.attn.fused_attn = False
        elif name == "ResNet34-CIFAR100":
            model = timm.create_model(architecture, pretrained=False, num_classes=classes)
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        else:
            from ._legacy_checkpoint import dense_module, legacy_dense_state

            model = dense_module().DenseNet3(num_classes=classes, **spec["architecture_kwargs"])
        if name == "DenseNet-BC-100-ODIN":
            state = legacy_dense_state(path, spec)
            for key in set(model.state_dict()) - set(state):
                require(key.endswith(".num_batches_tracked"), f"Missing learned tensor: {key}")
                state[key] = torch.zeros_like(model.state_dict()[key])
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        head = model.head if is_vit else model.fc
        transform = T.Compose([
            T.Resize(transform_spec["resize"], interpolation=T.InterpolationMode(transform_spec["interpolation"]),
                     antialias=True),
            T.CenterCrop(transform_spec["crop"]), T.ToTensor(),
            T.Normalize(transform_spec["mean"], transform_spec["std"]),
        ])
    del state
    require(head.out_features == classes, "Checkpoint has the wrong classifier")
    if dataset == "ImageNet1K":
        for key, param in (("head_weight_sha256", head.weight), ("head_bias_sha256", head.bias)):
            arr = param.detach().cpu().float().contiguous().numpy()
            prefix = str((str(torch.float32), tuple(arr.shape))).encode()
            require(hashlib.sha256(prefix + arr.tobytes()).hexdigest() == spec[key],
                    f"Classifier changed: {key}")
    return model.eval().requires_grad_(False).to(device), head, transform, digest


def forward(dataset, name, model, head, images):
    if dataset == "ImageNet1K":
        hidden = model.forward_features(images)
        if name == "Vim-Base":
            return hidden, head(hidden)
        return model.forward_head(hidden, pre_logits=True), model.forward_head(hidden, pre_logits=False)
    captured = []
    handle = head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        logits = model(images)
    finally:
        handle.remove()
    require(len(captured) == 1, "Classifier must run exactly once")
    return captured[0], logits

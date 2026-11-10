import hashlib
import os
import shutil
import zipfile
from pathlib import Path

import torch
from torch import nn

from ..common import ASSETS, checked, load_json, require
from ..datasets import DATASETS, download_drive

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
    if dataset == "ImageNet1K":
        source = spec["source"]
        if name == "Vim-Base":
            from ._vim_runtime import prepare_vim_source

            vim_root = Path(root) / "Vim-source"
            model = prepare_vim_source(vim_root)(pretrained=False, num_classes=1000,
                                                img_size=224, drop_path_rate=.1)
            for filename, checksum in source.get("source_patch_sha256", {}).items():
                checked(vim_root / filename, checksum)
            state = torch.load(path, map_location="cpu", weights_only=False)["model_ema"]
            model.load_state_dict(state, strict=True)
            head = model.head
            transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                                   T.CenterCrop(224), T.ToTensor(),
                                   T.Normalize(spec["transform"]["mean"], spec["transform"]["std"])])
        else:
            model = timm.create_model(source["repo"].removeprefix("timm/"), pretrained=False,
                                      num_classes=1000)
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
                require(actual == spec["transform"][key], f"Preprocessing drift: {name}/{key}")
            transform = create_transform(**config, is_training=False)
    else:
        is_vit = name.startswith("ViT")
        if "runs" in spec:
            from ..assets.upstream.resnet18_32x32 import ResNet18_32x32

            model = ResNet18_32x32(num_classes=classes)
        elif is_vit:
            model = timm.create_model("vit_base_patch16_224.orig_in21k_ft_in1k",
                                      pretrained=False, num_classes=classes)
            for block in model.blocks:
                block.attn.fused_attn = False
        elif name == "ResNet34-CIFAR100":
            model = timm.create_model("resnet34", pretrained=False, num_classes=classes)
            model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
        else:
            from ._legacy_checkpoint import dense_module, legacy_dense_state

            model = dense_module().DenseNet3(100, 10, growth_rate=12, reduction=.5,
                                            bottleneck=True, dropRate=0.)
        if name == "DenseNet-BC-100-ODIN":
            state = legacy_dense_state(path, spec)
            for key in set(model.state_dict()) - set(state):
                require(key.endswith(".num_batches_tracked"), f"Missing learned tensor: {key}")
                state[key] = torch.zeros_like(model.state_dict()[key])
            mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
            std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
        else:
            state = torch.load(path, map_location="cpu", weights_only=True)
            mean, std = (([.4914, .4822, .4465], [.247, .2435, .2616]) if classes == 10
                         else ([.5071, .4867, .4408], [.2675, .2565, .2761]))
        model.load_state_dict(state, strict=True)
        head = model.head if is_vit else model.fc
        resize, crop = (248, 224) if is_vit else (32, 32)
        interpolation = T.InterpolationMode.BICUBIC if is_vit else T.InterpolationMode.BILINEAR
        if is_vit:
            mean = std = [.5] * 3
        transform = T.Compose([T.Resize(resize, interpolation=interpolation, antialias=True),
                               T.CenterCrop(crop), T.ToTensor(), T.Normalize(mean, std)])
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

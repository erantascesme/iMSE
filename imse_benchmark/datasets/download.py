"""Download benchmark archives from their original sources."""

import os
import zipfile
from pathlib import Path

from ..common import checked, require
from .manifests import DATASETS, IMAGE_EXTS

IMAGENET_ASSETS = {
    "SSB_Hard": ("SSB_hard", "ssb_hard.zip"),
    "OpenImage_O": ("Openimage-O", "openimage_o.zip"),
}
OPENOOD_IMAGENET_ASSETS = {
    "NINCO": "1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6",
    "iNaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "Textures": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
}


def download_drive(path, file_id, digest):
    import gdown

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(".download")
        require(gdown.download(id=file_id, output=str(temp), resume=True) is not None,
                f"Download failed. Place the original archive at {path}.")
        if digest is not None:
            checked(temp, digest)
        else:
            require(zipfile.is_zipfile(temp), f"Invalid OpenOOD ZIP: {temp}")
        os.replace(temp, path)
    return checked(path, digest) if digest is not None else path


def download(dataset, root):
    root = Path(root)
    if dataset != "ImageNet1K":
        for asset, spec in DATASETS[dataset]["archives"].items():
            download_drive(root / "cifar" / f"{asset}.zip", spec["google_drive_id"], spec["sha256"])
        return

    from huggingface_hub import hf_hub_download

    base = root / "imagenet"
    for asset, names in (("train", [f"train_images_{i}.tar.gz" for i in range(5)]),
                         ("val", ["val_images.tar.gz"])):
        target = base / asset
        if not target.is_dir():
            for name in names:
                hf_hub_download("ILSVRC/imagenet-1k", f"data/{name}", repo_type="dataset",
                                revision="script", local_dir=target)
        elif not any(p.suffix.lower() in IMAGE_EXTS for p in target.iterdir()):
            for name in names:
                if not (target / name).is_file() and not (target / "data" / name).is_file():
                    if not (target / ".prepared").is_file():
                        hf_hub_download("ILSVRC/imagenet-1k", f"data/{name}", repo_type="dataset",
                                        revision="script", local_dir=target)
    for asset, (repo, filename) in IMAGENET_ASSETS.items():
        target = base / asset
        if not target.exists() or not any(target.iterdir()):
            hf_hub_download(f"torch-uncertainty/{repo}", filename, repo_type="dataset", local_dir=target)
    for asset, file_id in OPENOOD_IMAGENET_ASSETS.items():
        target = base / asset
        if not target.exists() or not any(target.iterdir()):
            download_drive(target / f"{asset}.zip", file_id, None)

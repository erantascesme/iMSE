"""Download benchmark archives from their original sources."""

import os
import time
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        import gdown

        temp = path.with_suffix(".download")
        for attempt in range(3):
            print(f"Downloading official archive: {path.name} (attempt {attempt + 1}/3)", flush=True)
            try:
                result = gdown.download(id=file_id, output=str(temp), quiet=False, resume=True)
                require(result is not None, "The download did not return a file")
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(
                        f"Cannot download {path.name} from OpenOOD's public release. "
                        f"Source: https://drive.google.com/file/d/{file_id}/view. "
                        f"Retry later or place that exact archive at {path}. "
                        "No personal Google Drive access is required."
                    ) from exc
                print(f"Download interrupted: {exc}", flush=True)
                time.sleep(3 * (attempt + 1))
        if digest is not None:
            checked(temp, digest)
        else:
            require(zipfile.is_zipfile(temp), f"Invalid OpenOOD ZIP: {temp}")
        os.replace(temp, path)
    require(zipfile.is_zipfile(path), f"Invalid OpenOOD ZIP: {path}")
    return checked(path, digest) if digest is not None else path


def _has_images(folder):
    return folder.is_dir() and any(p.is_file() and p.suffix.lower() in IMAGE_EXTS
                                  for p in folder.rglob("*"))


def _download_hf(repo, filename, target, revision="main"):
    from huggingface_hub import hf_hub_download

    target = Path(target)
    # Existing canonical archives can be placed directly in the dataset folder.
    if (target / Path(filename).name).is_file():
        return
    print(f"Downloading {repo}/{filename}", flush=True)
    hf_hub_download(repo, filename, repo_type="dataset", revision=revision, local_dir=target)


def download(dataset, root):
    root = Path(root)
    print(f"Dataset: {dataset} | destination: {root.resolve()}", flush=True)
    if dataset != "ImageNet1K":
        for asset, spec in DATASETS[dataset]["archives"].items():
            download_drive(root / "cifar" / f"{asset}.zip", spec["google_drive_id"], spec["sha256"])
        return

    base = root / "imagenet"
    for asset, names in (("train", [f"train_images_{i}.tar.gz" for i in range(5)]),
                         ("val", ["val_images.tar.gz"])):
        target = base / asset
        if not _has_images(target):
            for name in names:
                _download_hf("ILSVRC/imagenet-1k", f"data/{name}", target, revision="script")
    for asset, (repo, filename) in IMAGENET_ASSETS.items():
        target = base / asset
        if not _has_images(target):
            _download_hf(f"torch-uncertainty/{repo}", filename, target)
    for asset, file_id in OPENOOD_IMAGENET_ASSETS.items():
        target = base / asset
        if not _has_images(target):
            download_drive(target / f"{asset}.zip", file_id, None)

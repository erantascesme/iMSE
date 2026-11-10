"""Download benchmark archives from their original sources."""

import os
import time
import urllib.request
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
PUBLIC_MIRRORS = {
    "1Co32RiiWe16lTaiOU6JMMnyUYS41IlO1": (
        "5d3c480cd13e8791af7429fde3884f6ac45a7191c17fc1414eec550ed2e6582c",
        "https://github.com/erantascesme/iMSE/releases/download/benchmark-data-v1/cifar10.zip",
    ),
}


def _download_mirror(path, url, digest):
    from tqdm.auto import tqdm

    temporary = path.with_suffix(".mirror.download")
    print(f"Using verified public mirror: {path.name}", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "iMSE-benchmark"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as progress:
            while block := response.read(1024 * 1024):
                output.write(block)
                progress.update(len(block))
    checked(temporary, digest)
    require(zipfile.is_zipfile(temporary), f"Invalid mirrored ZIP: {temporary}")
    os.replace(temporary, path)
    return path


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
                quota = any(message in str(exc).lower() for message in
                            ("too many users", "download quota", "quota exceeded"))
                if quota or attempt == 2:
                    mirror = PUBLIC_MIRRORS.get(file_id)
                    if mirror is not None:
                        require(digest == mirror[0], "Mirror does not match the requested archive identity")
                        return _download_mirror(path, mirror[1], digest)
                if quota:
                    raise RuntimeError(
                        f"OpenOOD's public download quota is exhausted for {path.name}. "
                        "Immediate retries will not help. Wait for the provider to reset it, "
                        "then rerun the download cell. No verified public mirror is "
                        "configured for this archive. "
                        f"Source: https://drive.google.com/file/d/{file_id}/view"
                    ) from exc
                if attempt == 2:
                    raise RuntimeError(
                        f"Cannot download {path.name} from OpenOOD's public release. "
                        f"Source: https://drive.google.com/file/d/{file_id}/view. "
                        "Rerun the download cell after the public source becomes available. "
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

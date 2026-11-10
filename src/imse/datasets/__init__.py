import gzip
import hashlib
import io
import os
import shutil
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

from ..common import ASSETS, checked, load_json, require
from .manifests import imagenet_key, read_manifest_bytes, resolve_members, suffix_rank

DATASETS = load_json(ASSETS / "datasets.json")
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
IMAGENET_ASSETS = {
    "SSB_Hard": ("SSB_hard", "ssb_hard.zip"),
    "OpenImage_O": ("Openimage-O", "openimage_o.zip"),
}
OPENOOD_IMAGENET_ASSETS = {
    "NINCO": "1Z82cmvIB0eghTehxOGP5VTdLt7OD3nk6",
    "iNaturalist": "1zfLfMvoUD0CUlKNnkk7LgxZZBnTBipdj",
    "Textures": "1OSz1m3hHfVWbRdmMwKbUzoU8Hg9UKcam",
}


def manifest(dataset, split):
    spec = DATASETS[dataset]["splits"][split]
    raw = gzip.decompress((ASSETS / "manifests" / spec["file"]).read_bytes())
    require(hashlib.sha256(raw).hexdigest() == spec["sha256"], f"Manifest changed: {split}")
    rows = read_manifest_bytes(raw)
    require(len(rows) == spec["count"], f"Wrong row count: {split}")
    return rows


def ood_groups(dataset):
    if dataset == "ImageNet1K":
        return ["NINCO", "SSB_Hard"], ["iNaturalist", "Textures", "OpenImage_O"]
    near = "near_cifar100" if dataset == "CIFAR10" else "near_cifar10"
    return [near, "near_tin"], ["far_mnist", "far_svhn", "far_texture", "far_places365"]


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


def image_key(asset, name):
    if asset in {"train", "val"}:
        return imagenet_key(name)
    return Path(name).name.replace(" ", "_")


def _unpack_images(folder, asset):
    output = folder / "unpacked"
    for archive in sorted(folder.rglob("*")):
        if not archive.is_file() or output in archive.parents:
            continue
        if not archive.name.endswith((".tar", ".tar.gz", ".tgz", ".zip", ".tar.bz2")):
            continue
        source_id = hashlib.sha256(archive.relative_to(folder).as_posix().encode()).hexdigest()[:16]
        destination = output / source_id
        marker = destination / ".complete"
        if marker.exists():
            continue
        destination.mkdir(parents=True, exist_ok=True)
        if archive.suffix == ".zip":
            with zipfile.ZipFile(archive) as handle:
                _copy_images(handle.namelist(), handle.open, destination, asset)
        else:
            with tarfile.open(archive, "r|*") as handle:
                _copy_images((m for m in handle if m.isfile()), handle.extractfile,
                             destination, asset)
        marker.touch()


def _copy_images(names, open_member, destination, asset):
    seen = set()
    for member in names:
        name = member if isinstance(member, str) else member.name
        path = Path(name)
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        from .manifests import safe_name
        safe = Path(safe_name(name.removeprefix("./")))
        require(safe not in seen, f"Duplicate archive member: {safe}")
        seen.add(safe)
        target = destination / safe
        target.parent.mkdir(parents=True, exist_ok=True)
        with open_member(member) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)


class ImageSource:
    def __init__(self, dataset, root):
        self.dataset = dataset
        self.root = Path(root)
        self.archives = {}
        self.members = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.archives.values():
            handle.close()

    def records(self, split):
        rows = manifest(self.dataset, split)
        asset = DATASETS[self.dataset]["splits"][split]["asset"]
        if self.dataset != "ImageNet1K":
            if asset not in self.archives:
                spec = DATASETS[self.dataset]["archives"][asset]
                path = checked(self.root / "cifar" / f"{asset}.zip", spec["sha256"])
                self.archives[asset] = zipfile.ZipFile(path)
                wanted = [name for s, cfg in DATASETS[self.dataset]["splits"].items()
                          if cfg["asset"] == asset for name, _ in manifest(self.dataset, s)]
                self.members[asset] = resolve_members(self.archives[asset], asset, wanted)
            return [(name, label, asset, self.members[asset][name]) for name, label in rows]

        if asset not in self.members:
            folder = self.root / "imagenet" / asset
            require(folder.is_dir(), f"Missing image directory: {folder}")
            _unpack_images(folder, asset)
            indexed = defaultdict(list)
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    indexed[image_key(asset, p.name)].append(p)
            self.members[asset] = indexed
        resolved = []
        for name, label in rows:
            paths = self.members[asset][image_key(asset, name)]
            require(paths, f"Missing ImageNet benchmark image: {name}")
            if len(paths) > 1:
                ranked = sorted(((suffix_rank(name, p.as_posix()), p) for p in paths), reverse=True)
                require(ranked[0][0] > ranked[1][0], f"Ambiguous ImageNet benchmark image: {name}")
                paths = [ranked[0][1]]
            resolved.append((name, label, asset, paths[0]))
        return resolved

    def open(self, asset, member):
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = self.dataset != "ImageNet1K"
        file = io.BytesIO(self.archives[asset].read(member)) if asset in self.archives else member
        with Image.open(file) as image:
            return image.convert("RGB")

"""Pinned split manifests and benchmark image access."""

import gzip
import hashlib
import io
import re
import shutil
import stat
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from ..common import ASSETS, checked, load_json, require

DATASETS = load_json(ASSETS / "datasets.json")
IMAGE_EXTS = {".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


def safe_name(name):
    require(isinstance(name, str) and '\\' not in name and ('\x00' not in name), f'Invalid archive/manifest path: {name!r}')
    p = PurePosixPath(name)
    require(name and (not p.is_absolute()) and (':' not in name) and all((x not in ('', '.', '..') for x in name.rstrip('/').split('/'))), f'Unsafe or noncanonical relative path: {name!r}')
    return str(p)

def read_manifest_bytes(content):
    rows = []
    for line in content.decode('utf-8').splitlines():
        if not line.strip():
            continue
        name, value = line.rsplit(maxsplit=1)
        rows.append((safe_name(name), int(value)))
    require(len(set((p for p, y in rows))) == len(rows), 'Duplicate sample identity within manifest')
    return rows

def resolve_members(z, dataset, wanted):
    """Resolve the official extraction layout, without basename/fuzzy matching."""
    wanted = set(wanted)
    result = {}
    seen_members = set()
    for info in z.infolist():
        name = safe_name(info.filename)
        require(not stat.S_ISLNK(info.external_attr >> 16), 'ZIP symlinks are unsupported')
        if info.is_dir():
            continue
        require(name not in seen_members, f'Duplicate ZIP member: {name}')
        seen_members.add(name)
        for sample in {name, dataset + '/' + name} & wanted:
            require(sample not in result, f'Ambiguous ZIP mapping: {sample}')
            result[sample] = info.filename
    missing = wanted - result.keys()
    require(not missing, f'{dataset}: {len(missing)} manifest images absent from ZIP: {sorted(missing)[:8]}')
    return result

def imagenet_key(name):
    name = Path(name.replace('\\', '/')).name
    val = re.search('ILSVRC2012_val_\\d{8}', name)
    if val:
        return val.group(0)
    train = re.match('(n\\d{8}_\\d+)(?:_n\\d{8})?\\.[^.]+$', name)
    require(train is not None, f'Unrecognized ImageNet identifier: {name}')
    return train.group(1)

def suffix_rank(a, b):
    aa = a.replace(' ', '_').replace('\\', '/').split('/')
    bb = b.replace(' ', '_').replace('\\', '/').split('/')
    count = 0
    for x, y in zip(reversed(aa), reversed(bb)):
        if x != y:
            break
        count += 1
    return count


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

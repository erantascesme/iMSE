from pathlib import Path, PurePosixPath
import re
import stat
from ..common import require

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

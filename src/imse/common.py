import hashlib
import json
import os
from pathlib import Path

ASSETS = Path(__file__).with_name("assets")


def require(ok, message):
    if not bool(ok):
        raise RuntimeError(message)


def read_json(path):
    return Path(path).read_text(encoding="utf-8")


def load_json(path):
    return json.loads(read_json(path))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def hash_file(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checked(path, digest):
    path = Path(path)
    require(path.is_file() and hash_file(path) == digest, f"Missing or changed file: {path}")
    return path


def close_check(a, b, name, atol, rtol):
    import numpy as np

    a, b = np.asarray(a), np.asarray(b)
    require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol), name)
    return {"name": name, "max_abs_error": float(np.max(np.abs(a - b)))}

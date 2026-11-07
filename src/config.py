"""Configuration loading.

``load_config()`` returns the parsed ``config.yaml`` as a plain dict. The
entries in ``cfg["paths"]`` are resolved to absolute ``pathlib.Path`` objects:

* relative paths are resolved against the repository root;
* ``AA_DATA_DIR``, ``AA_CACHE_DIR`` and ``AA_RESULTS_DIR`` override
  ``data_dir``, ``cache_dir`` and ``results_dir`` respectively.

``config_hash(cfg)`` is a short, stable fingerprint of every *design* value
(everything except ``paths``). It goes into every results JSON so that results
produced with different designs are never mixed silently.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

ENV_OVERRIDES: dict[str, str] = {
    "data_dir": "AA_DATA_DIR",
    "cache_dir": "AA_CACHE_DIR",
    "results_dir": "AA_RESULTS_DIR",
}


def repo_root() -> Path:
    """Return the repository root (the directory containing ``config.yaml``)."""
    return Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` and resolve paths.

    Parameters
    ----------
    path
        Path to a YAML config. Defaults to ``<repo_root>/config.yaml``.

    Returns
    -------
    dict
        The parsed config. ``cfg["paths"][k]`` is an absolute ``Path`` for
        k in {data_dir, cache_dir, results_dir}.
    """
    path = Path(path) if path is not None else repo_root() / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = repo_root()
    for key, env in ENV_OVERRIDES.items():
        raw = os.environ.get(env) or cfg["paths"].get(key)
        if raw is None:
            continue
        p = Path(raw)
        cfg["paths"][key] = p if p.is_absolute() else (base / p).resolve()
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Return a 10-hex-char SHA-1 of the config without the ``paths`` section."""
    c = copy.deepcopy(cfg)
    c.pop("paths", None)
    blob = json.dumps(c, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]

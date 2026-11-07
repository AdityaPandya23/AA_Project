"""Results JSON schema, paths and writer (CONTRACTS.md §i).

Path: ``{results_dir}/{method}/sub-XX_ses-YY_seed-Z.json``. ``protocol="random"``
appends ``_random`` to the stem. Cross-session results use ``session="01to02"``.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from . import chromosome

PROTOCOLS = ("block", "cross_session", "random")
REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "method": str,
    "subject": str,
    "session": str,
    "seed": int,
    "protocol": str,
    "per_fold": list,
    "ba_mean": (int, float),
    "ba_std": (int, float),
    "train_time_s": (int, float),
    "infer_time_ms": (int, float, type(None)),
    "config_hash": str,
}
FOLD_FIELDS: dict[str, type | tuple[type, ...]] = {
    "fold": int,
    "ba": (int, float),
    "chosen_bits": (str, type(None)),
    "cost": (int, float, type(None)),
    "n_features": (int, type(None)),
}


class ResultContractError(ValueError):
    """Raised when a results record violates CONTRACTS.md §i."""


def result_path(method: str, subject: str, session: str, seed: int, protocol: str,
                cfg: dict[str, Any], results_dir: str | os.PathLike | None = None) -> Path:
    """Path of a results JSON, following CONTRACTS.md §i."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    base = Path(results_dir) if results_dir is not None else Path(cfg["paths"]["results_dir"])
    suffix = "_random" if protocol == "random" else ""
    return base / method / f"sub-{subject}_ses-{session}_seed-{seed}{suffix}.json"


def make_fold_record(fold: int, ba: float, chosen_bits=None,
                     n_features: int | None = None, **extra: Any) -> dict[str, Any]:
    """Build one ``per_fold`` entry. ``cost`` and ``n_features`` are derived from bits when given."""
    if chosen_bits is not None:
        bits_str = chosen_bits if isinstance(chosen_bits, str) else chromosome.to_str(chosen_bits)
        c = chromosome.cost(chromosome.from_str(bits_str))
        nf = chromosome.n_features(chromosome.from_str(bits_str)) if n_features is None else n_features
    else:
        bits_str, c, nf = None, None, n_features
    return {"fold": int(fold), "ba": float(ba), "chosen_bits": bits_str,
            "cost": c, "n_features": None if nf is None else int(nf), **extra}


def _type_ok(v: Any, t) -> bool:
    if isinstance(v, bool):  # bool is an int subclass; never accept it for numbers
        return False
    return isinstance(v, t)


def validate_result(rec: dict[str, Any]) -> None:
    """Raise :class:`ResultContractError` if ``rec`` violates the schema."""
    for k, t in REQUIRED_FIELDS.items():
        if k not in rec:
            raise ResultContractError(f"missing field {k!r}")
        if not _type_ok(rec[k], t):
            raise ResultContractError(f"field {k!r} has type {type(rec[k]).__name__}")
    if rec["protocol"] not in PROTOCOLS:
        raise ResultContractError(f"protocol {rec['protocol']!r} not in {PROTOCOLS}")
    if rec["protocol"] == "cross_session" and rec["session"] != "01to02":
        raise ResultContractError("cross_session results must have session '01to02'")
    if not rec["per_fold"]:
        raise ResultContractError("per_fold is empty")
    for i, f in enumerate(rec["per_fold"]):
        for k, t in FOLD_FIELDS.items():
            if k not in f:
                raise ResultContractError(f"per_fold[{i}] missing {k!r}")
            if not _type_ok(f[k], t):
                raise ResultContractError(f"per_fold[{i}].{k} has type {type(f[k]).__name__}")
        if f["chosen_bits"] is not None:
            chromosome.from_str(f["chosen_bits"])
    bas = [f["ba"] for f in rec["per_fold"]]
    mean = sum(bas) / len(bas)
    std = math.sqrt(sum((b - mean) ** 2 for b in bas) / len(bas))
    if not (math.isclose(rec["ba_mean"], mean, abs_tol=1e-6)
            and math.isclose(rec["ba_std"], std, abs_tol=1e-6)):
        raise ResultContractError("ba_mean/ba_std inconsistent with per_fold (ddof=0)")


def write_result(rec: dict[str, Any], cfg: dict[str, Any],
                 results_dir: str | os.PathLike | None = None) -> Path:
    """Validate ``rec`` and write it to :func:`result_path`. Returns the path."""
    validate_result(rec)
    path = result_path(rec["method"], rec["subject"], rec["session"], rec["seed"],
                       rec["protocol"], cfg, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return path

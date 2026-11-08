"""GA / EEGNet-specific helpers for ``run_ga.py`` and ``run_eegnet.py`` (R07).

Everything shared with the FBCSP experiments (cache loading, ``git_commit``,
cache-dir resolution, cross-session subject list, FBCSP latency) lives in
R01's :mod:`experiments.common` and is reused from there. This module adds
only what those experiments do not need:

* subject-range parsing for the CLIs (``sub-17..sub-22``);
* the list of (subject, session) units of a protocol, and loading one unit with its outer splits;
* the per-fold prediction keys shared by every method (``y_true``, ``y_pred``,
  ``n_test``, ``test_trial_id``, ...);
* atomic JSON writes for crash-safe checkpoints.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src import cv

from .common import cross_session_subjects, load

Split = tuple[np.ndarray, np.ndarray]


def parse_subjects(tokens: list[str]) -> list[str]:
    """``["sub-17..sub-22", "05", "sub-91"]`` -> ``["05", "17", ..., "22", "91"]``.

    Accepts ``XX``, ``sub-XX``, ranges ``A..B`` and ``A-B`` (inclusive).
    """
    out: set[int] = set()
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(?:sub-)?(\d+)(?:(?:\.\.|-)(?:sub-)?(\d+))?", part)
            if not m:
                raise ValueError(f"cannot parse subject {part!r}")
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            out.update(range(min(a, b), max(a, b) + 1))
    return [f"{s:02d}" for s in sorted(out)]


def recordings(protocol: str, subjects: list[str], sessions: list[str], kind: str,
               cfg: dict[str, Any], cache_dir: Path) -> list[tuple[str, str]]:
    """``(subject, session_label)`` units of a protocol, skipping missing caches.

    ``block``: one per existing ``sub/ses`` file. ``cross_session``:
    ``(sub, "01to02")`` for subjects with both sessions.
    """
    from src import cache
    have = set(cache.list_cached(kind, cfg, cache_dir))
    units = []
    if protocol == "block":
        for s in subjects:
            for ses in sessions:
                if (s, ses) in have:
                    units.append((s, ses))
                else:
                    print(f"[skip] no {kind} cache for sub-{s} ses-{ses}", file=sys.stderr)
    elif protocol == "cross_session":
        both = set(cross_session_subjects(cfg, cache_dir, kind))
        for s in subjects:
            if s in both:
                units.append((s, "01to02"))
            else:
                print(f"[skip] sub-{s}: cross_session needs ses-01 and ses-02", file=sys.stderr)
    else:
        raise ValueError(f"unsupported protocol {protocol!r}")
    return units


def load_unit(kind: str, subject: str, session: str, cfg: dict[str, Any],
              cache_dir: Path) -> tuple[dict[str, Any], list[Split]]:
    """Load data and outer splits for one unit (``session`` may be ``"01to02"``)."""
    if session == "01to02":
        d1 = load(kind, subject, "01", cfg, cache_dir)
        d2 = load(kind, subject, "02", cfg, cache_dir)
        d = cv.concat_sessions(d1, d2, kind)
        return d, cv.cross_session(len(d1["y"]), len(d2["y"]))
    d = load(kind, subject, session, cfg, cache_dir)
    return d, cv.block_splits(d["block"])


def prediction_keys(train_idx: np.ndarray, test_idx: np.ndarray, y_pred: Sequence[int],
                    d: dict[str, Any]) -> dict[str, Any]:
    """Per-fold extra keys, the same as R01's FBCSP experiments write.

    ``y_true`` / ``y_pred`` / ``test_trial_id`` are aligned with ``test_idx``.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.shape != (len(test_idx),):
        raise ValueError("y_pred must have one entry per test trial")
    return {"n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "test_blocks": sorted(int(b) for b in np.unique(d["block"][test_idx])),
            "test_trial_id": d["trial_id"][test_idx].tolist(),
            "y_true": d["y"][test_idx].tolist(), "y_pred": [int(v) for v in y_pred]}


def provenance(d: dict[str, Any]) -> dict[str, Any]:
    """Top-level extra keys shared with R01's records (``synthetic``, ``host``)."""
    out: dict[str, Any] = {"synthetic": bool(d.get("synthetic", False))}
    if os.environ.get("AA_HOST"):
        out["host"] = os.environ["AA_HOST"]
    return out


def atomic_write_json(obj: Any, path: Path) -> Path:
    """Write JSON via a temp file + rename, so a crash never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    return path

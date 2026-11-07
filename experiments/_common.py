"""Helpers shared by ``run_ga.py`` and ``run_eegnet.py``.

* ``cv_*``: call ``src.cv``; while it is still a stub, fall back to a
  local implementation that follows CONTRACTS.md §e literally. Once the real
  ``src.cv`` exists it is used automatically and the fallback is dead code.
* ``resolve_fbcsp``: the real ``src.fbcsp`` if implemented, otherwise the
  stand-in, but only when explicitly allowed.
* subject / session parsing and data loading.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src import cache, cv

Split = tuple[np.ndarray, np.ndarray]


# ----------------------------------------------------------------- cv fallbacks
def _warn_fallback(name: str) -> None:
    warnings.warn(f"src.cv.{name} is not implemented yet; using the contract fallback "
                  f"in experiments/_common.py", RuntimeWarning, stacklevel=3)


def cv_block_splits(block: np.ndarray) -> list[Split]:
    """``src.cv.block_splits`` or its CONTRACTS §e fallback."""
    try:
        return cv.block_splits(block)
    except NotImplementedError:
        _warn_fallback("block_splits")
    block = np.asarray(block)
    idx = np.arange(block.size, dtype=np.int64)
    return [(idx[block != b], idx[block == b]) for b in np.unique(block)]


def cv_cross_session(n_train: int, n_test: int) -> list[Split]:
    """``src.cv.cross_session`` or its CONTRACTS §e fallback."""
    try:
        return cv.cross_session(n_train, n_test)
    except NotImplementedError:
        _warn_fallback("cross_session")
    return [(np.arange(n_train, dtype=np.int64),
             np.arange(n_train, n_train + n_test, dtype=np.int64))]


def cv_concat_sessions(d1: dict[str, Any], d2: dict[str, Any], kind: str) -> dict[str, Any]:
    """``src.cv.concat_sessions`` or its CONTRACTS §e fallback."""
    try:
        return cv.concat_sessions(d1, d2, kind)
    except NotImplementedError:
        _warn_fallback("concat_sessions")
    axis = 0 if kind == "eegnet" else 2
    out: dict[str, Any] = {}
    for k in set(d1) | set(d2):
        if k == "X":
            out[k] = np.concatenate([d1[k], d2[k]], axis=axis)
        elif k in ("y", "trial_id"):
            out[k] = np.concatenate([d1[k], d2[k]])
        elif k == "block":
            out[k] = np.concatenate([d1[k], d2[k] + 4])
        else:
            if k not in d1 or k not in d2 or not np.array_equal(d1[k], d2[k]):
                raise ValueError(f"concat_sessions: key {k!r} differs between sessions")
            out[k] = d1[k]
    return out


# ----------------------------------------------------------------- fbcsp
def resolve_fbcsp(allow_standin: bool) -> tuple[Callable, type, str]:
    """Return ``(evaluate, FBCSPLDA, impl_name)``.

    Uses ``src.fbcsp`` when it is implemented. Otherwise uses the stand-in if
    ``allow_standin`` and raises ``RuntimeError`` if not.
    """
    from src import fbcsp
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((12, 1, 2, 50)).astype(np.float32)
    ys = np.repeat([0, 1, 2], 4)
    try:
        fbcsp.FBCSPLDA().fit(Xs, ys).predict(Xs)
        return fbcsp.evaluate, fbcsp.FBCSPLDA, "src.fbcsp"
    except NotImplementedError:
        if not allow_standin:
            raise RuntimeError("src.fbcsp is still a stub. Pass --allow-standin-fbcsp to use "
                               "experiments/_standin_fbcsp.py (development only).") from None
    from experiments import _standin_fbcsp as sf
    return sf.evaluate, sf.FBCSPLDA, "standin"


# ----------------------------------------------------------------- CLI parsing
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
               cfg: dict[str, Any], cache_dir: Path | None) -> list[tuple[str, str]]:
    """``(subject, session_label)`` units for a protocol, skipping missing caches.

    ``block``: one per existing ``sub/ses`` file. ``cross_session``: ``(sub,
    "01to02")`` when both sessions exist.
    """
    have = set(cache.list_cached(kind, cfg, cache_dir))
    units = []
    for s in subjects:
        if protocol == "block":
            for ses in sessions:
                if (s, ses) in have:
                    units.append((s, ses))
                else:
                    print(f"[skip] no {kind} cache for sub-{s} ses-{ses}", file=sys.stderr)
        elif protocol == "cross_session":
            if (s, "01") in have and (s, "02") in have:
                units.append((s, "01to02"))
            else:
                print(f"[skip] sub-{s}: cross_session needs ses-01 and ses-02", file=sys.stderr)
        else:
            raise ValueError(f"unsupported protocol {protocol!r}")
    return units


def load_unit(kind: str, subject: str, session: str, cfg: dict[str, Any],
              cache_dir: Path | None) -> tuple[dict[str, Any], list[Split]]:
    """Load data and outer splits for one unit (``session`` may be ``"01to02"``)."""
    if session == "01to02":
        d1 = cache.load_cache(kind, subject, "01", cfg, cache_dir)
        d2 = cache.load_cache(kind, subject, "02", cfg, cache_dir)
        d = cv_concat_sessions(d1, d2, kind)
        return d, cv_cross_session(len(d1["y"]), len(d2["y"]))
    d = cache.load_cache(kind, subject, session, cfg, cache_dir)
    return d, cv_block_splits(d["block"])


# ----------------------------------------------------------------- provenance
def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True, cwd=Path(__file__).resolve().parent.parent
                              ).stdout.strip() or None
    except Exception:
        return None


def host_tag() -> str:
    """``AA_HOST`` (``R01``/``R02``/``R07``) if set, else the hostname."""
    return os.environ.get("AA_HOST") or socket.gethostname()


def atomic_write_json(obj: Any, path: Path) -> Path:
    """Write JSON via a temp file + rename, so a crash never leaves a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)
    return path


def per_trial_latency_ms(predict: Callable[[np.ndarray], Any], Xtest: np.ndarray,
                         max_trials: int = 50) -> float:
    """Mean wall time (ms) of ``predict`` on single trials (batch size 1)."""
    import time
    n = min(len(Xtest), max_trials)
    predict(Xtest[:1])  # warm-up
    t0 = time.perf_counter()
    for i in range(n):
        predict(Xtest[i:i + 1])
    return (time.perf_counter() - t0) * 1000.0 / n

"""Shared helpers for the FBCSP experiments (baseline and sanity checks).

Every record built here follows the results JSON contract (CONTRACTS.md §i).
Extra per-fold keys (allowed by §0 "Unknown keys"): ``n_train``, ``n_test``,
``test_blocks``, ``test_trial_id``, ``y_true`` and ``y_pred``. ``src.analysis`` uses
``y_true`` and ``y_pred`` for permutation tests and confusion matrices.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src import cache, chromosome, cv, fbcsp, results
from src.config import config_hash, load_config, repo_root

Split = tuple[np.ndarray, np.ndarray]


def git_commit() -> str | None:
    """Short hash of HEAD, or None outside a git checkout."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_root(),
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def resolve_cache_dir(cache_dir: str | os.PathLike | None, cfg: dict[str, Any]) -> Path:
    """``--cache-dir`` if given, else ``cfg["paths"]["cache_dir"]`` (env ``AA_CACHE_DIR``)."""
    return Path(cache_dir) if cache_dir is not None else Path(cfg["paths"]["cache_dir"])


def load(kind: str, subject: str, session: str, cfg: dict[str, Any], cache_dir: Path) -> dict:
    """Load and validate one cache file."""
    return cache.load_cache(kind, subject, session, cfg, cache_dir)


def cross_session_subjects(cfg: dict[str, Any], cache_dir: Path, kind: str = "fbcsp") -> list[str]:
    """Subjects with both ses-01 and ses-02 in the cache."""
    pairs = set(cache.list_cached(kind, cfg, cache_dir))
    return sorted({s for s, _ in pairs if (s, "01") in pairs and (s, "02") in pairs})


def infer_latency_ms(X: np.ndarray, y: np.ndarray, bits, splits: Sequence[Split],
                     y_pred_ref: Sequence[np.ndarray] | None = None) -> float:
    """Mean per-trial prediction latency (ms) of the public ``FBCSPLDA.predict`` path.

    It times ``predict`` on raw ``Xsel`` test trials (covariances, CSP
    features, scaler and LDA) after fitting on the training trials. If
    ``y_pred_ref`` is given, it also checks that the public path reproduces
    ``evaluate``'s predictions.
    """
    Xsel = chromosome.select(X, bits)
    lat = []
    for k, (tr, te) in enumerate(splits):
        est = fbcsp.FBCSPLDA().fit(Xsel[tr], y[tr])
        est.predict(Xsel[te[:1]])  # warm-up
        t0 = time.perf_counter()
        yp = est.predict(Xsel[te])
        lat.append((time.perf_counter() - t0) / len(te) * 1e3)
        if y_pred_ref is not None and not np.array_equal(yp, y_pred_ref[k]):
            raise RuntimeError("public FBCSPLDA path disagrees with fbcsp.evaluate")
    return float(np.mean(lat))


def fbcsp_record(method: str, subject: str, session: str, protocol: str, seed: int,
                 X: np.ndarray, y: np.ndarray, block: np.ndarray, trial_id: np.ndarray,
                 splits: Sequence[Split], cfg: dict[str, Any], bits=None,
                 measure_latency: bool = True, **extra: Any) -> dict[str, Any]:
    """Evaluate FBCSP+LDA with ``bits`` (default: all on) and build a results record."""
    bits = chromosome.all_on() if bits is None else bits
    r = fbcsp.evaluate(X, y, bits, splits)
    folds = []
    for k, ((tr, te), ba, yp) in enumerate(zip(splits, r["per_fold"], r["y_pred"])):
        folds.append(results.make_fold_record(
            k, ba, bits, n_train=int(len(tr)), n_test=int(len(te)),
            test_blocks=sorted(int(b) for b in np.unique(block[te])),
            test_trial_id=trial_id[te].tolist(), y_true=y[te].tolist(), y_pred=yp.tolist()))
    rec = {
        "method": method, "subject": subject, "session": session, "seed": int(seed),
        "protocol": protocol, "per_fold": folds,
        "ba_mean": r["ba_mean"], "ba_std": r["ba_std"],
        "train_time_s": float(r["fit_time_s"]),
        "infer_time_ms": infer_latency_ms(X, y, bits, splits, r["y_pred"]) if measure_latency else None,
        "config_hash": config_hash(cfg),
        "git_commit": git_commit(),
    }
    if os.environ.get("AA_HOST"):
        rec["host"] = os.environ["AA_HOST"]
    rec.update(extra)
    return rec


def block_record(method: str, kind: str, subject: str, session: str, cfg: dict[str, Any],
                 cache_dir: Path, **kw: Any) -> dict[str, Any]:
    """Leave-one-block-out record for one recording of cache ``kind`` (fbcsp or view)."""
    d = load(kind, subject, session, cfg, cache_dir)
    return fbcsp_record(method, subject, session, "block", 0, d["X"], d["y"], d["block"],
                        d["trial_id"], cv.block_splits(d["block"]), cfg,
                        synthetic=bool(d.get("synthetic", False)), **kw)


def cross_session_record(method: str, kind: str, subject: str, cfg: dict[str, Any],
                         cache_dir: Path, **kw: Any) -> dict[str, Any]:
    """Train ses-01, test ses-02 record (session ``"01to02"``)."""
    d1 = load(kind, subject, "01", cfg, cache_dir)
    d2 = load(kind, subject, "02", cfg, cache_dir)
    d = cv.concat_sessions(d1, d2, kind)
    splits = cv.cross_session(len(d1["y"]), len(d2["y"]))
    return fbcsp_record(method, subject, "01to02", "cross_session", 0, d["X"], d["y"], d["block"],
                        d["trial_id"], splits, cfg, synthetic=bool(d.get("synthetic", False)), **kw)


def default_cfg() -> dict[str, Any]:
    return load_config()

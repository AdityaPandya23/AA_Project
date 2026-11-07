"""Preprocessed ``.npz`` caches: paths, writers, loader and validators.

Formats are fixed in CONTRACTS.md §a (``fbcsp``), §b (``eegnet``) and §c
(``view``, the same format as ``fbcsp``). Every writer validates before saving,
and :func:`load_cache` validates after loading. A file that violates the
contract never gets written and never gets used silently.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

KINDS = ("fbcsp", "eegnet", "view")
_FBCSP_KEYS = ("X", "y", "block", "trial_id", "ch_names", "band_names",
               "band_edges", "window_starts", "sfreq")
_EEGNET_KEYS = ("X", "y", "block", "trial_id", "ch_names", "sfreq")


class CacheContractError(ValueError):
    """Raised when a cache dict / file violates CONTRACTS.md."""


def stem(subject: str, session: str) -> str:
    """``sub-XX_ses-YY``."""
    return f"sub-{subject}_ses-{session}"


def cache_path(kind: str, subject: str, session: str, cfg: dict[str, Any],
               cache_dir: str | os.PathLike | None = None) -> Path:
    """``{cache_dir}/{kind}/sub-XX_ses-YY.npz``. ``kind`` is one of fbcsp, eegnet or view."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    base = Path(cache_dir) if cache_dir is not None else Path(cfg["paths"]["cache_dir"])
    return base / kind / f"{stem(subject, session)}.npz"


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise CacheContractError(msg)


def _check_labels(d: dict[str, Any], n: int) -> None:
    for k in ("y", "block", "trial_id"):
        _check(d[k].dtype == np.int64, f"{k} must be int64, got {d[k].dtype}")
        _check(d[k].shape == (n,), f"{k} shape {d[k].shape} != ({n},)")
    _check(bool(np.isin(d["y"], [0, 1, 2]).all()), "y must be in {0,1,2}")
    _check(bool(((d["block"] >= 0) & (d["block"] <= 3)).all()), "block must be in 0..3")
    _check(bool(np.all(np.diff(d["trial_id"]) > 0)), "trial_id must be strictly increasing")
    _check(np.ndim(d["sfreq"]) == 0 and float(d["sfreq"]) == 250.0, "sfreq must be scalar 250.0")
    _check(d["ch_names"].dtype.kind == "U", "ch_names must be a unicode array")


def validate_fbcsp(d: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    """Check a dict against CONTRACTS.md §a (also used for §c ``view``).

    Raises :class:`CacheContractError` on the first violation.
    """
    missing = [k for k in _FBCSP_KEYS if k not in d]
    _check(not missing, f"missing keys {missing}")
    X = d["X"]
    _check(X.dtype == np.float32, f"X must be float32, got {X.dtype}")
    _check(X.ndim == 5 and X.shape[:2] == (4, 7) and X.shape[3:] == (8, 250),
           f"X shape {X.shape} != (4, 7, n, 8, 250)")
    _check(bool(np.isfinite(X).all()), "X contains non-finite values")
    _check_labels(d, X.shape[2])
    _check(d["ch_names"].shape == (8,), "ch_names must have shape (8,)")
    _check(d["band_names"].dtype.kind == "U" and d["band_names"].shape == (4,),
           "band_names must be unicode (4,)")
    _check(d["band_edges"].shape == (4, 2), "band_edges must be (4, 2)")
    _check(d["window_starts"].shape == (7,), "window_starts must be (7,)")
    if cfg is not None:
        _check(list(d["ch_names"]) == list(cfg["channels"]["candidates"]),
               "ch_names != config.channels.candidates")
        _check(list(d["band_names"]) == list(cfg["bands"]["names"]),
               "band_names != config.bands.names")
        edges = [cfg["bands"]["edges_hz"][b] for b in cfg["bands"]["names"]]
        _check(np.allclose(d["band_edges"], edges), "band_edges != config")
        _check(np.allclose(d["window_starts"], cfg["windows"]["starts_s"]),
               "window_starts != config.windows.starts_s")


def validate_eegnet(d: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    """Check a dict against CONTRACTS.md §b."""
    missing = [k for k in _EEGNET_KEYS if k not in d]
    _check(not missing, f"missing keys {missing}")
    X = d["X"]
    n_ch = cfg["eegnet"]["n_channels"] if cfg is not None else 32
    _check(X.dtype == np.float32, f"X must be float32, got {X.dtype}")
    _check(X.ndim == 3 and X.shape[1:] == (n_ch, 1000),
           f"X shape {X.shape} != (n, {n_ch}, 1000)")
    _check(bool(np.isfinite(X).all()), "X contains non-finite values")
    _check_labels(d, X.shape[0])
    _check(d["ch_names"].shape == (n_ch,), f"ch_names must have shape ({n_ch},)")
    if cfg is not None:
        missing_ch = set(cfg["channels"]["candidates"]) - set(d["ch_names"])
        _check(not missing_ch, f"candidate channels missing from eegnet ch_names: {missing_ch}")


def _validator(kind: str):
    return validate_eegnet if kind == "eegnet" else validate_fbcsp


def _save(kind: str, path: str | os.PathLike, d: dict[str, Any], cfg) -> Path:
    _validator(kind)(d, cfg)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **d)
    return path


def save_fbcsp_cache(path: str | os.PathLike, *, X: np.ndarray, y: np.ndarray,
                     block: np.ndarray, trial_id: np.ndarray, ch_names, band_names,
                     band_edges, window_starts, sfreq: float = 250.0,
                     cfg: dict[str, Any] | None = None, **extra: Any) -> Path:
    """Validate and write an FBCSP (or ``view``) cache. Extra keys are stored too."""
    d = dict(X=np.asarray(X), y=np.asarray(y, dtype=np.int64),
             block=np.asarray(block, dtype=np.int64),
             trial_id=np.asarray(trial_id, dtype=np.int64),
             ch_names=np.asarray(ch_names, dtype=str), band_names=np.asarray(band_names, dtype=str),
             band_edges=np.asarray(band_edges, dtype=np.float64),
             window_starts=np.asarray(window_starts, dtype=np.float64),
             sfreq=np.float64(sfreq), **extra)
    return _save("fbcsp", path, d, cfg)


def save_eegnet_cache(path: str | os.PathLike, *, X: np.ndarray, y: np.ndarray,
                      block: np.ndarray, trial_id: np.ndarray, ch_names,
                      sfreq: float = 250.0, cfg: dict[str, Any] | None = None,
                      **extra: Any) -> Path:
    """Validate and write an EEGNet cache. Extra keys are stored too."""
    d = dict(X=np.asarray(X), y=np.asarray(y, dtype=np.int64),
             block=np.asarray(block, dtype=np.int64),
             trial_id=np.asarray(trial_id, dtype=np.int64),
             ch_names=np.asarray(ch_names, dtype=str), sfreq=np.float64(sfreq), **extra)
    return _save("eegnet", path, d, cfg)


def load_npz(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Load any cache file into a plain dict (``allow_pickle=False``)."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_cache(kind: str, subject: str, session: str, cfg: dict[str, Any],
               cache_dir: str | os.PathLike | None = None,
               validate: bool = True) -> dict[str, np.ndarray]:
    """Load ``{cache_dir}/{kind}/sub-XX_ses-YY.npz`` and validate it against the contract."""
    d = load_npz(cache_path(kind, subject, session, cfg, cache_dir))
    if validate:
        _validator(kind)(d, cfg)
    return d


def list_cached(kind: str, cfg: dict[str, Any],
                cache_dir: str | os.PathLike | None = None) -> list[tuple[str, str]]:
    """All ``(subject, session)`` pairs present for ``kind``, sorted."""
    base = (Path(cache_dir) if cache_dir is not None else Path(cfg["paths"]["cache_dir"])) / kind
    out = []
    for p in sorted(base.glob("sub-*_ses-*.npz")):
        sub, ses = p.stem.split("_")
        out.append((sub[4:], ses[4:]))
    return out


def candidate_channel_index(ch_names, cfg: dict[str, Any],
                            ch_mask: np.ndarray | None = None) -> np.ndarray:
    """Indices into ``ch_names`` (e.g. EEGNet's 32) of the candidate channels.

    With ``ch_mask`` (bool (8,), e.g. from ``chromosome.decode``), only the
    selected candidates are returned. This is the ``ch_idx`` for ``eegnet_gach``.
    """
    names = list(ch_names)
    cands = list(cfg["channels"]["candidates"])
    if ch_mask is not None:
        cands = [c for c, m in zip(cands, np.asarray(ch_mask, dtype=bool)) if m]
    return np.array([names.index(c) for c in cands], dtype=np.int64)

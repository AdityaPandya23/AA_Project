"""Cross-validation splits (CONTRACTS.md §e).

All functions return ``list[tuple[np.ndarray, np.ndarray]]`` of absolute,
sorted, non-overlapping ``int64`` ``(train_idx, test_idx)`` arrays.

Nesting rule (non-negotiable): the GA sees only ``train_idx`` of an outer
fold. It may call :func:`inner_splits` on it and nothing else. The outer test
fold is touched exactly once, by the final fit on ``train_idx`` with
``chosen_bits``.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

Split = tuple[np.ndarray, np.ndarray]

#: Keys concatenated along the trial axis by :func:`concat_sessions`.
_TRIAL_KEYS = ("X", "y", "block", "trial_id")
_X_TRIAL_AXIS = {"fbcsp": 2, "view": 2, "eegnet": 0}
#: Block offset applied to the second session so blocks stay unique.
SESSION_BLOCK_OFFSET = 4
#: Optional per-trial extra keys (one value per trial), concatenated when both have them.
_EXTRA_TRIAL_KEYS = ("ptp_uv",)
#: Per-recording bookkeeping written by preprocessing. These legitimately differ
#: between sessions, so they are kept per session as ``[value_ses1, value_ses2]``.
_PER_RECORDING_KEYS = ("rejected_trial_id", "n_rejected", "n_out_of_range")


def _split(train: np.ndarray, test: np.ndarray) -> Split:
    return (np.sort(np.asarray(train, dtype=np.int64)),
            np.sort(np.asarray(test, dtype=np.int64)))


def block_splits(block: np.ndarray) -> list[Split]:
    """Leave-one-block-out outer CV.

    Parameters
    ----------
    block
        int64 ``(n_trials,)``, values 0..3.

    Returns
    -------
    list
        One ``(train_idx, test_idx)`` per distinct block value, in ascending
        block order. Fold ``k`` tests block ``k``.
    """
    block = np.asarray(block)
    if block.ndim != 1:
        raise ValueError(f"block must be 1-D, got shape {block.shape}")
    values = np.unique(block)
    if values.size < 2:
        raise ValueError("leave-one-block-out needs at least 2 distinct blocks")
    return [_split(np.flatnonzero(block != b), np.flatnonzero(block == b)) for b in values]


def inner_splits(y: np.ndarray, block: np.ndarray, train_idx: np.ndarray,
                 n: int = 3, seed: int = 0) -> list[Split]:
    """Inner CV inside an outer training set, used for GA fitness only.

    Uses ``StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)``
    on ``train_idx`` with ``groups=block[train_idx]``, stratified by
    ``y[train_idx]``. When ``train_idx`` spans exactly ``n`` blocks (the block
    protocol), this is leave-one-block-out over those blocks.

    If ``train_idx`` spans fewer than ``n`` blocks, blocks cannot be kept
    intact. The function then warns and falls back to
    ``StratifiedKFold(n, shuffle=True, random_state=seed)`` on ``train_idx``.
    See docs/CONTRACT_ISSUES.md.

    Returns
    -------
    list
        ``n`` splits of absolute indices, all ⊆ ``train_idx``.
    """
    y = np.asarray(y)
    block = np.asarray(block)
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    if train_idx.size == 0:
        raise ValueError("train_idx is empty")
    if np.unique(train_idx).size != train_idx.size:
        raise ValueError("train_idx contains duplicates")
    y_tr, g_tr = y[train_idx], block[train_idx]
    dummy = np.zeros((train_idx.size, 1))
    if np.unique(g_tr).size >= n:
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
        parts = cv.split(dummy, y_tr, g_tr)
    else:
        warnings.warn(f"inner_splits: train_idx spans {np.unique(g_tr).size} block(s) < n={n}; "
                      "falling back to StratifiedKFold (blocks are not kept intact)",
                      RuntimeWarning, stacklevel=2)
        parts = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed).split(dummy, y_tr)
    return [_split(train_idx[tr], train_idx[te]) for tr, te in parts]


def random_splits(y: np.ndarray, n: int = 5, seed: int = 0) -> list[Split]:
    """``StratifiedKFold(n, shuffle=True, random_state=seed)`` over all trials.

    This is the "random" protocol. It ignores blocks, is optimistic, and
    exists only for comparison with the literature (and to measure how much
    ignoring blocks inflates accuracy).
    """
    y = np.asarray(y)
    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return [_split(tr, te) for tr, te in cv.split(np.zeros((y.size, 1)), y)]


def cross_session(n_train: int, n_test: int) -> list[Split]:
    """One split ``[(arange(n_train), arange(n_train, n_train + n_test))]``.

    The caller concatenates ses-01 then ses-02 along the trial axis (see
    :func:`concat_sessions`).
    """
    n_train, n_test = int(n_train), int(n_test)
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")
    return [(np.arange(n_train, dtype=np.int64),
             np.arange(n_train, n_train + n_test, dtype=np.int64))]


def _equal(a: Any, b: Any) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in "fc" and b.dtype.kind in "fc":
        return bool(np.allclose(a, b, equal_nan=True))
    return bool(np.array_equal(a, b))


def concat_sessions(d1: dict[str, Any], d2: dict[str, Any], kind: str) -> dict[str, Any]:
    """Concatenate two loaded cache dicts (ses-01, ses-02) along the trial axis.

    * ``kind="fbcsp"`` or ``"view"``: ``X`` is concatenated on axis 2. ``kind="eegnet"``: axis 0.
    * ``y``, ``trial_id``: concatenated.
    * ``block``: ``concat(block1, block2 + 4)`` so blocks stay unique.
    * ``ptp_uv`` (per trial): concatenated. ``rejected_trial_id``, ``n_rejected``,
      ``n_out_of_range`` (per recording): kept as ``[ses1_value, ses2_value]``.
    * Other keys (``ch_names``, ``band_names``, ...) are checked for equality
      and taken from ``d1``. A mismatch raises ``ValueError``.

    Extra key added: ``session_idx`` (int64, 0 for ses-01 trials, 1 for ses-02).
    Note that the result has blocks 4..7, so it does not pass
    ``cache.validate_fbcsp`` (which requires 0..3); do not validate it.
    """
    if kind not in _X_TRIAL_AXIS:
        raise ValueError(f"kind must be one of {tuple(_X_TRIAL_AXIS)}, got {kind!r}")
    ax = _X_TRIAL_AXIS[kind]
    for k in _TRIAL_KEYS:
        if k not in d1 or k not in d2:
            raise ValueError(f"both dicts need key {k!r}")
    X1, X2 = np.asarray(d1["X"]), np.asarray(d2["X"])
    if X1.ndim != X2.ndim or np.delete(X1.shape, ax).tolist() != np.delete(X2.shape, ax).tolist():
        raise ValueError(f"X shapes {X1.shape} and {X2.shape} differ outside the trial axis")
    if X1.dtype != X2.dtype:
        raise ValueError(f"X dtypes differ: {X1.dtype} vs {X2.dtype}")
    n1, n2 = X1.shape[ax], X2.shape[ax]
    out: dict[str, Any] = {
        "X": np.concatenate([X1, X2], axis=ax),
        "y": np.concatenate([d1["y"], d2["y"]]).astype(np.int64),
        "block": np.concatenate([np.asarray(d1["block"]),
                                 np.asarray(d2["block"]) + SESSION_BLOCK_OFFSET]).astype(np.int64),
        "trial_id": np.concatenate([d1["trial_id"], d2["trial_id"]]).astype(np.int64),
        "session_idx": np.repeat(np.array([0, 1], dtype=np.int64), [n1, n2]),
    }
    for k in ("y", "block", "trial_id"):
        if out[k].shape != (n1 + n2,):
            raise ValueError(f"{k} length does not match X along the trial axis")
    for k in _EXTRA_TRIAL_KEYS:
        if k in d1 and k in d2:
            out[k] = np.concatenate([np.asarray(d1[k]), np.asarray(d2[k])])
    for k in _PER_RECORDING_KEYS:
        if k in d1 and k in d2:
            out[k] = [d1[k], d2[k]]
    for k, v in d1.items():
        if k in _TRIAL_KEYS or k in out or k == "session_idx":
            continue
        if k in d2 and not _equal(v, d2[k]):
            raise ValueError(f"key {k!r} differs between sessions")
        out[k] = v
    return out

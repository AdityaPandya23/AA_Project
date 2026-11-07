"""Cross-validation splits (CONTRACTS.md §e). STUB.

All functions return ``list[tuple[np.ndarray, np.ndarray]]`` of absolute,
sorted, non-overlapping ``int64`` ``(train_idx, test_idx)`` arrays.

Nesting rule (non-negotiable): the GA sees only ``train_idx`` of an outer
fold. It may call :func:`inner_splits` on it and nothing else. The outer test
fold is touched exactly once, by the final fit on ``train_idx`` with
``chosen_bits``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


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
    raise NotImplementedError


def inner_splits(y: np.ndarray, block: np.ndarray, train_idx: np.ndarray,
                 n: int = 3, seed: int = 0) -> list[Split]:
    """Inner CV inside an outer training set, used for GA fitness only.

    Uses ``StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)``
    on ``train_idx`` with ``groups=block[train_idx]``, stratified by
    ``y[train_idx]``. When ``train_idx`` spans exactly ``n`` blocks (the block
    protocol), this is leave-one-block-out over those blocks.

    Returns
    -------
    list
        ``n`` splits of absolute indices, all ⊆ ``train_idx``.
    """
    raise NotImplementedError


def random_splits(y: np.ndarray, n: int = 5, seed: int = 0) -> list[Split]:
    """``StratifiedKFold(n, shuffle=True, random_state=seed)`` over all trials.

    This is the "random" protocol. It ignores blocks, is optimistic, and
    exists only for comparison with the literature.
    """
    raise NotImplementedError


def cross_session(n_train: int, n_test: int) -> list[Split]:
    """One split ``[(arange(n_train), arange(n_train, n_train + n_test))]``.

    The caller concatenates ses-01 then ses-02 along the trial axis (see
    :func:`concat_sessions`).
    """
    raise NotImplementedError


def concat_sessions(d1: dict[str, Any], d2: dict[str, Any], kind: str) -> dict[str, Any]:
    """Concatenate two loaded cache dicts (ses-01, ses-02) along the trial axis.

    * ``kind="fbcsp"`` or ``"view"``: ``X`` is concatenated on axis 2. ``kind="eegnet"``: axis 0.
    * ``y``, ``trial_id``: concatenated.
    * ``block``: ``concat(block1, block2 + 4)`` so blocks stay unique.
    * Other keys (``ch_names``, ``band_names``, ...) are checked for equality
      and taken from ``d1``. A mismatch raises ``ValueError``.
    """
    raise NotImplementedError

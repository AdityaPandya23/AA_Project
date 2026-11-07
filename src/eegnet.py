"""EEGNet comparison under the same splits (CONTRACTS.md §h). STUB.

Requires ``requirements-gpu.txt`` (torch, braindecode). Import torch and
braindecode **inside** functions so that the rest of the package works without them.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


def evaluate_eegnet(X: np.ndarray, y: np.ndarray, splits: Sequence[Split],
                    ch_idx: np.ndarray | None = None,
                    cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train and test braindecode EEGNetv4 on each split.

    Parameters
    ----------
    X
        EEGNet cache array ``(n_trials, 32, 1000)``, float32, µV.
    y
        int64 ``(n_trials,)``.
    splits
        Exactly the same split objects as used for FBCSP on the same file stem.
    ch_idx
        Optional channel indices into axis 1. ``None`` means all 32. For
        ``eegnet_gach``, use
        ``cache.candidate_channel_index(ch_names, cfg, ch_mask)``.
    cfg
        ``load_config()`` dict. Hyperparameters come from ``cfg["eegnet"]``.
        ``None`` → ``load_config()``.

    Early-stopping validation is carved from the **training fold only**
    (``cfg["eegnet"]["val_fraction"]``, stratified). The test fold is never
    used for model selection.

    Returns
    -------
    dict
        ``{"per_fold": list[float], "ba_mean": float, "ba_std": float,
        "train_time_s": float}``. It may add ``"infer_time_ms"`` (mean
        per-trial latency) and ``"n_params"``.
    """
    raise NotImplementedError

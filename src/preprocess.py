"""Raw BIDS → caches (CONTRACTS.md §a–c). STUB.

Pipeline per (subject, session):

1. ``io.read_raw``, keep EEG channels, convert to µV.
2. ``preprocess_raw``: resample to 250 Hz (open items: reference, artifacts).
3. ``epoch_period`` for imagery (onset + 9 s) and viewing (onset + 3 s), with
   ``cfg["preprocess"]["filter"]["pad_s"]`` padding on both sides.
4. FBCSP and view caches: ``make_fbcsp_array``, i.e. band-pass the padded epoch
   per band (zero-phase), crop to ``[0, 4)`` s, then cut 7 windows of 250 samples.
5. EEGNet cache: band-pass 1–40 Hz on the padded epoch, crop to ``[0, 4)`` s, all channels.
6. Write with ``cache.save_fbcsp_cache`` / ``cache.save_eegnet_cache``. The
   writers validate against the contract.

``src/synth.py`` follows the same steps 3–6 on fake data. Reuse its filtering
code if useful.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import mne
    import pandas as pd


def preprocess_raw(raw: "mne.io.BaseRaw", cfg: dict[str, Any]) -> "mne.io.BaseRaw":
    """Pick EEG, scale to µV, re-reference (open item), resample to ``cfg["preprocess"]["sfreq"]``."""
    raise NotImplementedError


def epoch_period(raw: "mne.io.BaseRaw", trials: "pd.DataFrame", period: str,
                 cfg: dict[str, Any], pad_s: float | None = None) -> np.ndarray:
    """Cut padded epochs for ``period`` in {"imagery", "viewing"}.

    Returns float64 ``(n_trials, n_channels, 1000 + 2*pad_samples)``, where the
    unpadded part starts at the period onset (``trials.onset_s`` +
    ``cfg["dataset"]["trial_phases_s"][period][0]``).
    """
    raise NotImplementedError


def make_fbcsp_array(epochs_padded: np.ndarray, ch_names: list[str], cfg: dict[str, Any],
                     pad_samples: int) -> np.ndarray:
    """Padded epochs (all channels) → FBCSP array ``(4, 7, n_trials, 8, 250)`` float32.

    Picks ``cfg["channels"]["candidates"]`` in order, band-passes per band,
    crops the padding off, and cuts the windows.
    """
    raise NotImplementedError


def build_caches(subject: str, session: str, cfg: dict[str, Any],
                 overwrite: bool = False) -> dict[str, Path]:
    """Write the ``fbcsp``, ``view`` and ``eegnet`` caches for one recording.

    Returns ``{kind: path}``.
    """
    raise NotImplementedError


def build_all(cfg: dict[str, Any], n_jobs: int = 1, overwrite: bool = False) -> list[dict[str, Path]]:
    """:func:`build_caches` for every pair from ``io.list_subject_sessions`` (joblib parallel)."""
    raise NotImplementedError

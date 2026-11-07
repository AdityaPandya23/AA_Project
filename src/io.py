"""BIDS reading for the Gao Visual Imagery dataset. STUB.

Use ``mne`` and ``mne_bids`` imported inside functions, so that importing
``src.io`` does not require them.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import mne


def list_subject_sessions(cfg: dict[str, Any], task: str = "figure") -> list[tuple[str, str]]:
    """All ``(subject, session)`` pairs with a figure-task recording under ``cfg["paths"]["data_dir"]``.

    Labels are 2-digit strings, e.g. ``("01", "02")``. Sorted.
    """
    raise NotImplementedError


def read_raw(subject: str, session: str, cfg: dict[str, Any], task: str = "figure") -> "mne.io.BaseRaw":
    """Read the raw ``.bdf`` (via ``mne_bids.read_raw_bids``), with channel types set from ``channels.tsv``."""
    raise NotImplementedError


def read_channels_tsv(subject: str, session: str, cfg: dict[str, Any],
                      task: str = "figure") -> pd.DataFrame:
    """Return the BIDS ``channels.tsv`` as a DataFrame (``name, type, units, status, ...``)."""
    raise NotImplementedError


def read_trials(subject: str, session: str, cfg: dict[str, Any], task: str = "figure") -> pd.DataFrame:
    """One row per trial of the session, in presentation order.

    Columns: ``trial_id`` (int, 0-based in the original sequence), ``onset_s``
    (float, trial onset in raw time; imagery onset = ``onset_s + 9.0``),
    ``shape`` (str: circle|square|pentagram), ``y`` (int, ``cfg["classes"]``),
    ``block`` (int 0..3).
    """
    raise NotImplementedError

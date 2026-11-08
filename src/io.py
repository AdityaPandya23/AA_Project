"""BIDS reading for the Gao Visual Imagery dataset.

Facts this module relies on (details in ``docs/DATA_NOTES.md``):

* BIDS root ``cfg["paths"]["data_dir"]``; recordings at
  ``sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-<T>_eeg.bdf`` with ``<T>`` = ``FVI``
  for the figure task (``AVI`` animals, ``OVI`` objects).
* The BDF has 32 EEG channels + ``Status``; ``channels.tsv`` lists only the 32.
* There is **one marker per trial**, at **imagery onset** (trial t = 9 s), in
  both the ``Status`` channel (1-sample pulse, value 1/2/3) and ``events.tsv``
  (``onset``, ``duration`` = 4, ``type``, ``value``, ``latency``, ``trial``).
  Value 1 = pentagram, 2 = square, 3 = circle.
* Blocks are separated by pauses of minutes; within a block markers are
  ~17.1 s apart. Blocks are therefore found from inter-marker gaps.

``mne`` and ``mne_bids`` are imported inside functions, so ``import src.io``
does not need them.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    import mne

log = logging.getLogger(__name__)

#: Project task names -> BIDS ``task-`` labels of this dataset.
TASK_LABELS: dict[str, str] = {"figure": "FVI", "animal": "AVI", "object": "OVI"}
#: events.tsv ``value`` -> shape (from ``task-FVI_events.json``).
FVI_VALUE_TO_SHAPE: dict[int, str] = {1: "pentagram", 2: "square", 3: "circle"}
#: A gap between consecutive markers longer than this many median ITIs starts a new block.
BLOCK_GAP_FACTOR = 2.0


class DataError(RuntimeError):
    """The recording on disk does not look like the dataset described in DATA_NOTES.md."""


def bids_task(task: str, cfg: dict[str, Any] | None = None) -> str:
    """Map ``"figure"`` → ``"FVI"`` (BIDS labels pass through unchanged)."""
    if cfg is not None and task == cfg["dataset"].get("task"):
        return cfg["dataset"].get("bids_task", TASK_LABELS.get(task, task))
    return TASK_LABELS.get(task, task)


def _root(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["data_dir"])


def _eeg_file(subject: str, session: str, cfg: dict[str, Any], task: str, suffix: str) -> Path:
    t = bids_task(task, cfg)
    return (_root(cfg) / f"sub-{subject}" / f"ses-{session}" / "eeg"
            / f"sub-{subject}_ses-{session}_task-{t}_{suffix}")


def list_subject_sessions(cfg: dict[str, Any], task: str = "figure") -> list[tuple[str, str]]:
    """All ``(subject, session)`` pairs with a figure-task recording under ``cfg["paths"]["data_dir"]``.

    Labels are 2-digit strings, e.g. ``("01", "02")``. Sorted. A pair is listed
    only if both the ``.bdf`` and the ``events.tsv`` exist.
    """
    t = bids_task(task, cfg)
    out = []
    for bdf in _root(cfg).glob(f"sub-*/ses-*/eeg/sub-*_ses-*_task-{t}_eeg.bdf"):
        sub, ses = bdf.name.split("_")[:2]
        if bdf.with_name(bdf.name.replace("_eeg.bdf", "_events.tsv")).exists():
            out.append((sub[4:], ses[4:]))
    return sorted(out)


def read_channels_tsv(subject: str, session: str, cfg: dict[str, Any],
                      task: str = "figure") -> pd.DataFrame:
    """Return the BIDS ``channels.tsv`` as a DataFrame (``name, type, units, status, ...``)."""
    return pd.read_csv(_eeg_file(subject, session, cfg, task, "channels.tsv"), sep="\t",
                       dtype={"name": str})


def read_raw(subject: str, session: str, cfg: dict[str, Any], task: str = "figure") -> "mne.io.BaseRaw":
    """Read the raw ``.bdf`` (via ``mne_bids.read_raw_bids``), with channel types set from ``channels.tsv``.

    Returns a preloaded Raw (data in volts, as MNE stores it) with 32 EEG
    channels and the ``Status`` stim channel. Channels whose ``channels.tsv``
    status is ``bad`` are added to ``raw.info["bads"]``. The annotations that
    mne_bids derives from events.tsv are dropped; use :func:`read_trials`.
    """
    import mne
    import mne_bids

    bp = mne_bids.BIDSPath(subject=subject, session=session, task=bids_task(task, cfg),
                           datatype="eeg", suffix="eeg", extension=".bdf", root=_root(cfg))
    with warnings.catch_warnings():
        # Known, harmless: channels.tsv has 32 rows vs 33 in the BDF (Status), the
        # coordsystem uses non-BIDS frame/unit names, extra participants.tsv columns.
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne_bids.read_raw_bids(bp, verbose="ERROR")
    raw.load_data(verbose="ERROR")
    raw.set_annotations(None)

    ch = read_channels_tsv(subject, session, cfg, task)
    type_map = {"EEG": "eeg", "EOG": "eog", "ECG": "ecg", "EMG": "emg", "TRIG": "stim",
                "MISC": "misc"}
    types = {n: type_map.get(str(t).upper(), "misc") for n, t in zip(ch["name"], ch["type"])
             if n in raw.ch_names}
    missing = sorted(set(ch["name"]) - set(raw.ch_names))
    if missing:
        raise DataError(f"sub-{subject} ses-{session}: channels.tsv names not in BDF: {missing}")
    raw.set_channel_types(types, verbose="ERROR")
    if "status" in ch.columns:
        bads = ch.loc[ch["status"].astype(str).str.lower() == "bad", "name"].tolist()
        raw.info["bads"] = sorted(set(raw.info["bads"]) | set(bads))
    return raw


def _status_markers(raw: "mne.io.BaseRaw") -> np.ndarray:
    """Sample indices of isolated 1/2/3 pulses in ``Status`` (baseline is 0 or 131071)."""
    st = raw.get_data(picks="Status")[0]
    vals, counts = np.unique(st, return_counts=True)
    base = vals[np.argmax(counts)]
    idx = np.flatnonzero(np.isin(st, [1, 2, 3]) & (st != base))
    return np.column_stack([idx, st[idx].astype(int)]) if idx.size else np.empty((0, 2), int)


def _blocks_from_onsets(onsets: np.ndarray, n_blocks: int) -> np.ndarray:
    gaps = np.diff(onsets)
    if gaps.size == 0:
        return np.zeros(onsets.size, dtype=np.int64)
    new_block = gaps > BLOCK_GAP_FACTOR * np.median(gaps)
    return np.concatenate([[0], np.cumsum(new_block)]).astype(np.int64)


def read_trials(subject: str, session: str, cfg: dict[str, Any], task: str = "figure",
                raw: "mne.io.BaseRaw | None" = None) -> pd.DataFrame:
    """One row per trial of the session, in presentation order.

    Columns: ``trial_id`` (int, 0-based in the original sequence), ``onset_s``
    (float, trial onset in raw time; imagery onset = ``onset_s + 9.0``),
    ``shape`` (str: circle|square|pentagram), ``y`` (int, ``cfg["classes"]``),
    ``block`` (int 0..3).

    Extra columns: ``imagery_onset_s`` (the recorded marker itself),
    ``value`` (raw event code), ``latency`` (marker sample at 1000 Hz).

    The marker in events.tsv sits at imagery onset, so ``onset_s`` is
    ``marker - trial_phases_s.imagery[0]``. Blocks come from the long pauses
    between markers. If ``raw`` is given, the events.tsv latencies are
    cross-checked against the ``Status`` pulses (must agree within 2 samples).
    """
    ev = pd.read_csv(_eeg_file(subject, session, cfg, task, "events.tsv"), sep="\t")
    ev = ev.sort_values("onset", kind="stable").reset_index(drop=True)
    tag = f"sub-{subject} ses-{session}"
    if "type" in ev.columns:
        shape = ev["type"].astype(str).str.strip().str.lower()
    else:
        shape = ev["value"].astype(int).map(FVI_VALUE_TO_SHAPE)
    if "value" in ev.columns:
        by_value = ev["value"].astype(int).map(FVI_VALUE_TO_SHAPE)
        if not (by_value == shape).all():
            raise DataError(f"{tag}: events.tsv type and value disagree")
    unknown = sorted(set(shape) - set(cfg["classes"]))
    if unknown:
        raise DataError(f"{tag}: unknown event types {unknown}")

    marker = ev["onset"].to_numpy(float)
    if raw is not None:
        st = _status_markers(raw)
        lat = np.round(marker * raw.info["sfreq"]).astype(int)
        if len(st) != len(lat) or np.abs(st[:, 0] - lat).max() > 2 or \
                not (st[:, 1] == ev["value"].to_numpy(int)).all():
            raise DataError(f"{tag}: events.tsv does not match Status channel "
                            f"({len(st)} pulses vs {len(lat)} events)")

    n_blocks = int(cfg["dataset"]["n_blocks"])
    block = _blocks_from_onsets(marker, n_blocks)
    if block.max() + 1 != n_blocks:
        raise DataError(f"{tag}: found {block.max() + 1} blocks from marker gaps, "
                        f"expected {n_blocks}")
    t_imag = float(cfg["dataset"]["trial_phases_s"]["imagery"][0])
    return pd.DataFrame({
        "trial_id": np.arange(len(ev), dtype=np.int64),
        "onset_s": marker - t_imag,
        "shape": shape.to_numpy(str),
        "y": shape.map(cfg["classes"]).to_numpy(np.int64),
        "block": block,
        "imagery_onset_s": marker,
        "value": ev["value"].to_numpy(np.int64) if "value" in ev.columns else -1,
        "latency": ev["latency"].to_numpy(np.int64) if "latency" in ev.columns else -1,
    })

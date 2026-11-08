"""Raw BIDS → caches (CONTRACTS.md §a–c).

Pipeline per (subject, session):

1. ``io.read_raw`` + ``io.read_trials`` (events.tsv, cross-checked against the
   ``Status`` channel). The marker is at imagery onset.
2. ``preprocess_raw`` on the continuous data: keep EEG, notch at 50 Hz and its
   harmonics, zero-phase FIR band-pass 1–40 Hz, common average reference,
   resample to 250 Hz. Values are converted to µV when epoching.
3. ``epoch_period`` for imagery (marker + 0 s, i.e. trial t = 9 s) and viewing
   (trial t = 3 s), with ``cfg["preprocess"]["filter"]["pad_s"]`` padding on
   both sides.
4. Trial rejection: peak-to-peak of the unpadded broadband imagery epoch on
   ``cfg["preprocess"]["reject"]`` channels > ``ptp_uv`` → dropped from all
   three caches (so ``y``, ``block``, ``trial_id`` stay identical).
5. FBCSP and view caches: ``make_fbcsp_array``, i.e. band-pass the padded epoch
   per band (zero-phase Butterworth), crop to ``[0, 4)`` s, then cut 7 windows.
6. EEGNet cache: the continuous data are already 1–40 Hz, so the padded
   epoch is only cropped to ``[0, 4)`` s (all 32 channels).
7. Write with ``cache.save_fbcsp_cache`` / ``cache.save_eegnet_cache``. The
   writers validate against the contract. Extra keys: ``rejected_trial_id``,
   ``n_rejected``, ``n_out_of_range``, ``reject_ptp_uv``.
"""
from __future__ import annotations

import logging
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import cache
from . import io as dio

if TYPE_CHECKING:  # pragma: no cover
    import mne
    import pandas as pd

log = logging.getLogger(__name__)


def bandpass(x: np.ndarray, lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis (same as ``src.synth``)."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def preprocess_raw(raw: "mne.io.BaseRaw", cfg: dict[str, Any]) -> "mne.io.BaseRaw":
    """Pick EEG, notch, band-pass, re-reference, resample to ``cfg["preprocess"]["sfreq"]``.

    Works on a copy. Bad channels (``info["bads"]``) are interpolated from a
    standard 10-05 montage before re-referencing. Data stay in volts (MNE
    convention); :func:`epoch_period` converts to µV.
    """
    import mne

    pp = cfg["preprocess"]
    raw = raw.copy().pick("eeg", exclude=[])
    if raw.info["bads"]:
        raw.set_montage("standard_1005", on_missing="raise", verbose="ERROR")
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    notch = float(pp.get("notch_hz", 50))
    freqs = np.arange(notch, raw.info["sfreq"] / 2.0, notch)
    raw.notch_filter(freqs, verbose="ERROR")
    lo, hi = pp["broadband_hz"]
    raw.filter(lo, hi, verbose="ERROR")
    ref = pp.get("reference", "average")
    if ref not in (None, "none"):
        raw.set_eeg_reference("average" if ref == "average" else ref, projection=False,
                              verbose="ERROR")
    raw.resample(pp["sfreq"], verbose="ERROR")
    return raw


def _period_onsets(trials: "pd.DataFrame", period: str, cfg: dict[str, Any]) -> np.ndarray:
    """Period onset in raw seconds. Anchored on the recorded imagery marker when present."""
    phases = cfg["dataset"]["trial_phases_s"]
    if "imagery_onset_s" in trials.columns:
        return trials["imagery_onset_s"].to_numpy(float) + (phases[period][0] - phases["imagery"][0])
    return trials["onset_s"].to_numpy(float) + phases[period][0]


def epoch_period(raw: "mne.io.BaseRaw", trials: "pd.DataFrame", period: str,
                 cfg: dict[str, Any], pad_s: float | None = None) -> np.ndarray:
    """Cut padded epochs for ``period`` in {"imagery", "viewing"}.

    Returns float64 ``(n_trials, n_channels, 1000 + 2*pad_samples)``, where the
    unpadded part starts at the period onset (``trials.onset_s`` +
    ``cfg["dataset"]["trial_phases_s"][period][0]``). Values in µV. Trials
    whose padded epoch runs off the recording are filled with NaN.
    """
    sf = float(raw.info["sfreq"])
    pad_s = cfg["preprocess"]["filter"]["pad_s"] if pad_s is None else pad_s
    pad = int(round(pad_s * sf))
    n = int(cfg["preprocess"]["epoch_samples"])
    data = raw.get_data() * 1e6
    starts = np.round((_period_onsets(trials, period, cfg) - raw.first_time) * sf).astype(int) - pad
    out = np.full((len(starts), data.shape[0], n + 2 * pad), np.nan)
    for i, s in enumerate(starts):
        if s >= 0 and s + n + 2 * pad <= data.shape[1]:
            out[i] = data[:, s:s + n + 2 * pad]
    return out


def make_fbcsp_array(epochs_padded: np.ndarray, ch_names: list[str], cfg: dict[str, Any],
                     pad_samples: int) -> np.ndarray:
    """Padded epochs (all channels) → FBCSP array ``(4, 7, n_trials, 8, 250)`` float32.

    Picks ``cfg["channels"]["candidates"]`` in order, band-passes per band,
    crops the padding off, and cuts the windows.
    """
    sf = float(cfg["preprocess"]["sfreq"])
    order = int(cfg["preprocess"]["filter"]["order"])
    n_epoch = int(cfg["preprocess"]["epoch_samples"])
    wlen = int(cfg["windows"]["length_samples"])
    starts = [int(round(s * sf)) for s in cfg["windows"]["starts_s"]]
    names = list(ch_names)
    sub = epochs_padded[:, [names.index(c) for c in cfg["channels"]["candidates"]], :]
    out = np.empty((len(cfg["bands"]["names"]), len(starts), sub.shape[0], sub.shape[1], wlen),
                   dtype=np.float32)
    for bi, band in enumerate(cfg["bands"]["names"]):
        lo, hi = cfg["bands"]["edges_hz"][band]
        filt = bandpass(sub, lo, hi, sf, order)[..., pad_samples:pad_samples + n_epoch]
        for wi, s in enumerate(starts):
            out[bi, wi] = filt[..., s:s + wlen]
    return out


def reject_mask(imagery_padded: np.ndarray, ch_names: list[str], cfg: dict[str, Any],
                pad_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """``(keep, ptp_uv)``: keep = finite and peak-to-peak ≤ threshold on the configured channels."""
    rj = cfg["preprocess"].get("reject", {})
    n = int(cfg["preprocess"]["epoch_samples"])
    ep = imagery_padded[..., pad_samples:pad_samples + n]
    if rj.get("channels", "candidates") == "candidates":
        names = list(ch_names)
        ep = ep[:, [names.index(c) for c in cfg["channels"]["candidates"]], :]
    finite = np.isfinite(ep).all(axis=(1, 2))
    ptp = np.full(ep.shape[0], np.inf)
    ptp[finite] = np.ptp(ep[finite], axis=-1).max(axis=-1)
    thr = rj.get("ptp_uv")
    keep = finite & (ptp <= thr) if thr is not None else finite
    return keep, ptp


def _fbcsp_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(ch_names=cfg["channels"]["candidates"], band_names=cfg["bands"]["names"],
                band_edges=[cfg["bands"]["edges_hz"][b] for b in cfg["bands"]["names"]],
                window_starts=cfg["windows"]["starts_s"])


def build_caches(subject: str, session: str, cfg: dict[str, Any],
                 overwrite: bool = False) -> dict[str, Path]:
    """Write the ``fbcsp``, ``view`` and ``eegnet`` caches for one recording.

    Returns ``{kind: path}``.
    """
    paths = {k: cache.cache_path(k, subject, session, cfg) for k in cache.KINDS}
    if not overwrite and all(p.exists() for p in paths.values()):
        return paths
    import mne
    mne.set_log_level("ERROR")
    tag = f"sub-{subject} ses-{session}"

    raw = dio.read_raw(subject, session, cfg)
    trials = dio.read_trials(subject, session, cfg, raw=raw)
    raw = preprocess_raw(raw, cfg)
    ch_names = list(raw.ch_names)
    if len(ch_names) != cfg["eegnet"]["n_channels"]:
        raise dio.DataError(f"{tag}: {len(ch_names)} EEG channels, expected "
                            f"{cfg['eegnet']['n_channels']}")
    sf = float(raw.info["sfreq"])
    pad = int(round(cfg["preprocess"]["filter"]["pad_s"] * sf))
    n = int(cfg["preprocess"]["epoch_samples"])

    imag = epoch_period(raw, trials, "imagery", cfg)
    view = epoch_period(raw, trials, "viewing", cfg)
    del raw
    in_range = np.isfinite(imag).all(axis=(1, 2)) & np.isfinite(view).all(axis=(1, 2))
    keep, ptp = reject_mask(imag, ch_names, cfg, pad)
    keep &= in_range
    n_oor = int((~in_range).sum())
    rejected = trials["trial_id"].to_numpy(np.int64)[~keep]
    log.info("%s: %d trials, %d rejected (ptp > %s uV on %s), %d out of range",
             tag, len(trials), (~keep).sum() - n_oor, cfg["preprocess"]["reject"]["ptp_uv"],
             cfg["preprocess"]["reject"]["channels"], n_oor)

    t = trials[keep]
    common = dict(y=t["y"].to_numpy(np.int64), block=t["block"].to_numpy(np.int64),
                  trial_id=t["trial_id"].to_numpy(np.int64), sfreq=sf, cfg=cfg,
                  rejected_trial_id=rejected, n_rejected=np.int64(rejected.size),
                  n_out_of_range=np.int64(n_oor),
                  reject_ptp_uv=np.float64(cfg["preprocess"]["reject"]["ptp_uv"]),
                  ptp_uv=ptp[keep].astype(np.float32))
    imag, view = imag[keep], view[keep]
    meta = _fbcsp_meta(cfg)
    cache.save_fbcsp_cache(paths["fbcsp"], X=make_fbcsp_array(imag, ch_names, cfg, pad),
                           **meta, **common)
    cache.save_fbcsp_cache(paths["view"], X=make_fbcsp_array(view, ch_names, cfg, pad),
                           **meta, **common)
    cache.save_eegnet_cache(paths["eegnet"], X=imag[..., pad:pad + n].astype(np.float32),
                            ch_names=ch_names, **common)
    return paths


def _build_safe(subject: str, session: str, cfg: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    t0 = time.time()
    try:
        paths = build_caches(subject, session, cfg, overwrite)
        return dict(subject=subject, session=session, ok=True, paths=paths,
                    error="", time_s=time.time() - t0)
    except Exception as e:  # one bad recording must not stop the batch
        return dict(subject=subject, session=session, ok=False, paths={},
                    error=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
                    time_s=time.time() - t0)


def build_all_report(cfg: dict[str, Any], n_jobs: int = 1, overwrite: bool = False,
                     pairs: list[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """Like :func:`build_all` but returns one status dict per recording (incl. failures)."""
    from joblib import Parallel, delayed
    pairs = dio.list_subject_sessions(cfg) if pairs is None else pairs
    return Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_build_safe)(s, ses, cfg, overwrite) for s, ses in pairs)


def build_all(cfg: dict[str, Any], n_jobs: int = 1, overwrite: bool = False) -> list[dict[str, Path]]:
    """:func:`build_caches` for every pair from ``io.list_subject_sessions`` (joblib parallel)."""
    rep = build_all_report(cfg, n_jobs, overwrite)
    for r in rep:
        if not r["ok"]:
            log.error("sub-%s ses-%s failed: %s", r["subject"], r["session"], r["error"])
    return [r["paths"] for r in rep if r["ok"]]

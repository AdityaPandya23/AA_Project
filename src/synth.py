"""FAKE caches for development before real data exists.

Writes caches in the exact formats of CONTRACTS.md (a) ``fbcsp``,
(b) ``eegnet`` and (c) ``view`` for fake subjects ``91``, ``92`` and ``93``,
sessions ``01`` and ``02``.

The signal model, at 250 Hz and in µV:

* background: spatially mixed pink (1/f) noise on 32 BioSemi-style channels;
* alpha (8–12 Hz) narrowband activity on all channels;
* class effect: the alpha amplitude on **O1** is ×``effect`` for *square* and
  on **O2** is ×``effect`` for *pentagram*. *Circle* is the baseline. The
  imagery effect is 1.25 and the viewing effect is 1.6, so the viewing sanity
  check is easier than imagery.
* 4 blocks × 30 trials, each block balanced (10 per class) and shuffled.
* sub-93 ses-01 has 3 "rejected" trials (n_trials = 117) to catch code that
  hard-codes 120.

Filtering mirrors the intended real pipeline: zero-phase Butterworth on the
epoch padded by ``preprocess.filter.pad_s``, then crop and window.

Usage::

    python -m src.synth --out cache_synth
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import cache
from .config import load_config

# BioSemi 32-channel layout; contains all placeholder candidates.
SYNTH_CH_NAMES: list[str] = [
    "Fp1", "AF3", "F7", "F3", "FC1", "FC5", "T7", "C3", "CP1", "CP5", "P7", "P3",
    "Pz", "PO3", "O1", "Oz", "O2", "PO4", "P4", "P8", "CP6", "CP2", "C4", "T8",
    "FC6", "FC2", "F4", "F8", "AF4", "Fp2", "Fz", "Cz",
]
SUBJECTS: tuple[str, ...] = ("91", "92", "93")
SESSIONS: tuple[str, ...] = ("01", "02")
REJECTED: dict[tuple[str, str], list[int]] = {("93", "01"): [7, 64, 111]}
EFFECT_CHANNELS: dict[str, int] = {"O1": 1, "O2": 2}  # channel -> class with boosted alpha


def bandpass(x: np.ndarray, lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the last axis."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def _pink(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    n = shape[-1]
    spec = rng.standard_normal(shape[:-1] + (n // 2 + 1,)) + 1j * rng.standard_normal(
        shape[:-1] + (n // 2 + 1,))
    f = np.arange(n // 2 + 1, dtype=float)
    f[0] = np.inf
    x = np.fft.irfft(spec / np.sqrt(f), n=n, axis=-1)
    return x / x.std(axis=-1, keepdims=True)


def _narrowband(rng: np.random.Generator, shape: tuple[int, ...], lo: float, hi: float,
                sfreq: float) -> np.ndarray:
    """Unit-variance noise with a brick-wall spectrum in [lo, hi] Hz (no leakage into other bands)."""
    n = shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / sfreq)
    spec = rng.standard_normal(shape[:-1] + f.shape) + 1j * rng.standard_normal(shape[:-1] + f.shape)
    spec[..., (f < lo) | (f > hi)] = 0.0
    x = np.fft.irfft(spec, n=n, axis=-1)
    return x / x.std()


def _labels(rng: np.random.Generator, n_blocks: int = 4, per_class: int = 10):
    y, block = [], []
    for b in range(n_blocks):
        yb = np.repeat([0, 1, 2], per_class)
        rng.shuffle(yb)
        y.append(yb)
        block.append(np.full(yb.size, b))
    return np.concatenate(y).astype(np.int64), np.concatenate(block).astype(np.int64)


def _simulate(rng: np.random.Generator, y: np.ndarray, n_samples: int, sfreq: float,
              effect: float, mixing: np.ndarray) -> np.ndarray:
    """Padded epochs ``(n_trials, 32, n_samples)`` in µV."""
    n, n_ch = y.size, len(SYNTH_CH_NAMES)
    bg = np.einsum("ij,njt->nit", mixing, _pink(rng, (n, n_ch, n_samples))) * 10.0
    alpha = _narrowband(rng, (n, n_ch, n_samples), 8.5, 11.5, sfreq)
    gain = np.full((n, n_ch), 4.0)
    for ch, cls in EFFECT_CHANNELS.items():
        gain[y == cls, SYNTH_CH_NAMES.index(ch)] *= effect
    return bg + gain[:, :, None] * alpha


def _fbcsp_from_padded(ep: np.ndarray, cfg: dict[str, Any], pad: int) -> np.ndarray:
    sf = cfg["preprocess"]["sfreq"]
    order = cfg["preprocess"]["filter"]["order"]
    n_epoch = cfg["preprocess"]["epoch_samples"]
    wlen = cfg["windows"]["length_samples"]
    starts = [int(round(s * sf)) for s in cfg["windows"]["starts_s"]]
    ch_idx = [SYNTH_CH_NAMES.index(c) for c in cfg["channels"]["candidates"]]
    sub = ep[:, ch_idx, :]
    out = np.empty((len(cfg["bands"]["names"]), len(starts), ep.shape[0], len(ch_idx), wlen),
                   dtype=np.float32)
    for bi, band in enumerate(cfg["bands"]["names"]):
        lo, hi = cfg["bands"]["edges_hz"][band]
        filt = bandpass(sub, lo, hi, sf, order)[..., pad:pad + n_epoch]
        for wi, s in enumerate(starts):
            out[bi, wi] = filt[..., s:s + wlen]
    return out


def write_session(subject: str, session: str, out_dir: str | Path,
                  cfg: dict[str, Any] | None = None, seed: int = 0) -> dict[str, Path]:
    """Write fbcsp, view and eegnet caches for one fake recording. Returns ``{kind: path}``."""
    cfg = cfg or load_config()
    sf = float(cfg["preprocess"]["sfreq"])
    n_epoch = cfg["preprocess"]["epoch_samples"]
    pad = int(round(cfg["preprocess"]["filter"]["pad_s"] * sf))
    order = cfg["preprocess"]["filter"]["order"]
    rng = np.random.default_rng([seed, int(subject), int(session)])

    y, block = _labels(rng)
    trial_id = np.arange(y.size, dtype=np.int64)
    keep = np.setdiff1d(trial_id, REJECTED.get((subject, session), []))
    y, block, trial_id = y[keep], block[keep], trial_id[keep]

    n_ch = len(SYNTH_CH_NAMES)
    mixing = np.eye(n_ch) + 0.3 * rng.standard_normal((n_ch, n_ch)) / np.sqrt(n_ch)
    imag = _simulate(rng, y, n_epoch + 2 * pad, sf, 1.25, mixing)
    view = _simulate(rng, y, n_epoch + 2 * pad, sf, 1.6, mixing)

    common = dict(y=y, block=block, trial_id=trial_id, sfreq=sf, cfg=cfg)
    fb_meta = dict(ch_names=cfg["channels"]["candidates"], band_names=cfg["bands"]["names"],
                   band_edges=[cfg["bands"]["edges_hz"][b] for b in cfg["bands"]["names"]],
                   window_starts=cfg["windows"]["starts_s"], synthetic=np.bool_(True))
    lo, hi = cfg["preprocess"]["broadband_hz"]
    eeg = bandpass(imag, lo, hi, sf, order)[..., pad:pad + n_epoch].astype(np.float32)

    paths = {
        "fbcsp": cache.save_fbcsp_cache(cache.cache_path("fbcsp", subject, session, cfg, out_dir),
                                        X=_fbcsp_from_padded(imag, cfg, pad), **fb_meta, **common),
        "view": cache.save_fbcsp_cache(cache.cache_path("view", subject, session, cfg, out_dir),
                                       X=_fbcsp_from_padded(view, cfg, pad), **fb_meta, **common),
        "eegnet": cache.save_eegnet_cache(cache.cache_path("eegnet", subject, session, cfg, out_dir),
                                          X=eeg, ch_names=SYNTH_CH_NAMES,
                                          synthetic=np.bool_(True), **common),
    }
    return paths


def write_synthetic(out_dir: str | Path, cfg: dict[str, Any] | None = None,
                    subjects=SUBJECTS, sessions=SESSIONS, seed: int = 0) -> list[dict[str, Path]]:
    """Write all fake recordings into ``out_dir/{fbcsp,eegnet,view}/``."""
    cfg = cfg or load_config()
    return [write_session(s, ses, out_dir, cfg, seed) for s in subjects for ses in sessions]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="cache_synth", help="output cache dir (default: cache_synth)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    for d in write_synthetic(a.out, seed=a.seed):
        print(*[str(v) for v in d.values()], sep="\n")


if __name__ == "__main__":
    main()

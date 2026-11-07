"""19-bit chromosome for channel / band / window selection (CONTRACTS.md §d).

Bit layout (order follows ``config.yaml``)::

    bit  0..7   channels  config.channels.candidates  [O1, Oz, O2, Pz, P3, P4, P7, P8]
    bit  8..11  bands     config.bands.names          [theta, alpha, beta, lowgamma]
    bit 12..18  windows   config.windows.starts_s     [0.0, 0.5, ..., 3.0]

A chromosome is valid iff it has >= 2 channels, >= 1 band and >= 1 window.

``bits`` arguments accept any 1-D array-like of 0/1 (list, tuple, bool or int
array) of length 19. Functions that return bits return ``np.uint8`` arrays.
The JSON representation is the 19-character string from :func:`to_str`.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

N_CH: int = 8
N_BAND: int = 4
N_WIN: int = 7
N_BITS: int = N_CH + N_BAND + N_WIN  # 19
MIN_CH: int = 2
MIN_BAND: int = 1
MIN_WIN: int = 1
N_CLASSES: int = 3
FEATURES_PER_BW: int = N_CLASSES * 2  # 1 filter pair per class (one-vs-rest)

_CH = slice(0, N_CH)
_BAND = slice(N_CH, N_CH + N_BAND)
_WIN = slice(N_CH + N_BAND, N_BITS)
_GROUPS = ((_CH, MIN_CH), (_BAND, MIN_BAND), (_WIN, MIN_WIN))

BitsLike = Sequence[int] | np.ndarray


def _as_bits(bits: BitsLike) -> np.ndarray:
    """Convert to a validated uint8 array of shape (19,). Always returns a copy."""
    a = np.array(bits).ravel()
    if a.shape != (N_BITS,):
        raise ValueError(f"bits must have length {N_BITS}, got shape {a.shape}")
    if a.dtype == bool:
        return a.astype(np.uint8)
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("bits must contain only 0/1")
    return a.astype(np.uint8)


def decode(bits: BitsLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split bits into ``(ch_mask, band_mask, win_mask)``.

    Returns bool arrays of shapes (8,), (4,), (7,).
    """
    b = _as_bits(bits).astype(bool)
    return b[_CH].copy(), b[_BAND].copy(), b[_WIN].copy()


def encode(ch_mask: BitsLike, band_mask: BitsLike, win_mask: BitsLike) -> np.ndarray:
    """Inverse of :func:`decode`. Returns uint8 bits of shape (19,)."""
    parts = [np.asarray(m).ravel() for m in (ch_mask, band_mask, win_mask)]
    if [p.size for p in parts] != [N_CH, N_BAND, N_WIN]:
        raise ValueError(f"mask sizes must be ({N_CH}, {N_BAND}, {N_WIN})")
    return _as_bits(np.concatenate(parts).astype(np.uint8))


def counts(bits: BitsLike) -> tuple[int, int, int]:
    """Return ``(n_ch, n_bands, n_win)`` switched on."""
    b = _as_bits(bits)
    return int(b[_CH].sum()), int(b[_BAND].sum()), int(b[_WIN].sum())


def is_valid(bits: BitsLike) -> bool:
    """True iff channels >= 2, bands >= 1 and windows >= 1."""
    n_ch, n_b, n_w = counts(bits)
    return n_ch >= MIN_CH and n_b >= MIN_BAND and n_w >= MIN_WIN


def repair(bits: BitsLike, rng: np.random.Generator) -> np.ndarray:
    """Return a **new** valid chromosome.

    For each group (channels, bands, windows) that is below its minimum,
    uniformly random *off* genes in that group are switched on until the
    minimum is met. Genes are never switched off. A valid input is returned
    unchanged (as a copy).
    """
    b = _as_bits(bits)
    for sl, minimum in _GROUPS:
        g = b[sl]  # view into b
        deficit = minimum - int(g.sum())
        if deficit > 0:
            off = np.flatnonzero(g == 0)
            g[rng.choice(off, size=deficit, replace=False)] = 1
    return b


def cost(bits: BitsLike) -> float:
    """``(n_ch/8 + n_bands/4 + n_win/7) / 3``, in [0, 1]; all-on = 1.0."""
    n_ch, n_b, n_w = counts(bits)
    return (n_ch / N_CH + n_b / N_BAND + n_w / N_WIN) / 3.0


def random_valid(rng: np.random.Generator, p: float = 0.5) -> np.ndarray:
    """Sample each bit ~ Bernoulli(p), then :func:`repair`."""
    return repair((rng.random(N_BITS) < p).astype(np.uint8), rng)


def all_on() -> np.ndarray:
    """The baseline-1 chromosome: all 19 bits on."""
    return np.ones(N_BITS, dtype=np.uint8)


def n_features(bits: BitsLike) -> int:
    """Number of FBCSP features: ``6 * n_bands_sel * n_win_sel``."""
    _, n_b, n_w = counts(bits)
    return FEATURES_PER_BW * n_b * n_w


def select(X: np.ndarray, bits: BitsLike) -> np.ndarray:
    """Select the components encoded by ``bits`` from an FBCSP cache array.

    Parameters
    ----------
    X
        FBCSP cache array, shape ``(4, 7, n_trials, 8, 250)``
        = ``[band, window, trial, channel, sample]``.
    bits
        Valid chromosome.

    Returns
    -------
    np.ndarray
        ``Xsel`` with shape ``(n_trials, n_bw, n_ch_sel, 250)`` and the same dtype
        as ``X``. ``n_bw = n_bands_sel * n_win_sel``, band-major:
        ``bw = i_band_sel * n_win_sel + i_win_sel``. Always a copy.

    Raises
    ------
    ValueError
        If ``bits`` is invalid or ``X`` has the wrong leading/channel dims.
    """
    if not is_valid(bits):
        raise ValueError(f"invalid chromosome {to_str(bits)}")
    if X.ndim != 5 or X.shape[0] != N_BAND or X.shape[1] != N_WIN or X.shape[3] != N_CH:
        raise ValueError(f"X must have shape ({N_BAND}, {N_WIN}, n, {N_CH}, T), got {X.shape}")
    ch_m, b_m, w_m = decode(bits)
    sub = X[b_m][:, w_m]            # (nb, nw, n, 8, T)
    sub = sub[:, :, :, ch_m, :]     # (nb, nw, n, nc, T)
    nb, nw, n, nc, t = sub.shape
    return np.ascontiguousarray(sub.reshape(nb * nw, n, nc, t).transpose(1, 0, 2, 3))


def to_str(bits: BitsLike) -> str:
    """19-character '0'/'1' string (JSON representation)."""
    return "".join(str(int(v)) for v in _as_bits(bits))


def from_str(s: str) -> np.ndarray:
    """Inverse of :func:`to_str`."""
    s = s.strip()
    if len(s) != N_BITS or set(s) - {"0", "1"}:
        raise ValueError(f"expected {N_BITS}-char 0/1 string, got {s!r}")
    return np.array([int(c) for c in s], dtype=np.uint8)


def describe(bits: BitsLike, cfg: dict[str, Any] | None = None) -> dict[str, list]:
    """Human-readable selection: ``{"channels": [...], "bands": [...], "windows": [...]}``."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    ch_m, b_m, w_m = decode(bits)
    chs = cfg["channels"]["candidates"]
    bands = cfg["bands"]["names"]
    wins = cfg["windows"]["starts_s"]
    return {
        "channels": [c for c, m in zip(chs, ch_m) if m],
        "bands": [b for b, m in zip(bands, b_m) if m],
        "windows": [w for w, m in zip(wins, w_m) if m],
    }

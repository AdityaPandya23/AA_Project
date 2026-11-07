"""One-vs-rest filter-bank CSP + shrinkage LDA (CONTRACTS.md §f).

Implementation notes (open item 6 in CONTRACTS.md, recorded here):

* **Covariance-domain pipeline.** Everything is computed from per-trial
  spatial covariance matrices ``C_i = Xc_i Xc_i^T / T`` (channel means removed,
  float64). CSP needs only class-mean covariances, and the variance of a
  CSP-filtered trial is ``w^T C_i w``, so the log-variance features are exact.
  Per-trial covariances use one trial only, so they involve no fitting and
  cannot leak information across trials. The class means (the only fitted
  quantity) come from training trials only.
* **CSP per band-window and class** (one-vs-rest): ``C_c`` = mean covariance of
  class-``c`` training trials, ``C_r`` = mean covariance of the other training
  trials. Filters solve ``C_c w = λ (C_c + C_r) w``. We keep the
  ``n_filter_pairs`` largest and smallest λ (largest first). Unnormalised trial
  covariances are averaged (the same as MNE's default ``cov_est="concat"``).
* **Regularisation:** ``cov_reg`` (default ``0.0``) adds
  ``cov_reg * trace(C)/n_ch * I`` to ``C_c`` and ``C_r``. If the composite
  covariance is numerically singular, a ridge of ``1e-6 * trace/n_ch`` is added
  automatically, and only for that band-window.
* **Features:** raw ``log(var)`` of each filtered signal (not normalised by the
  sum over filters). Then ``StandardScaler`` and
  ``LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")``.
* **Speed:** :func:`evaluate` computes the per-trial covariances of the
  **full** cache array once per ``X`` object (an LRU cache keyed by object
  identity plus a cheap fingerprint) and then slices channels, bands and
  windows per chromosome. The result is identical to calling
  ``FBCSPLDA().fit(select(X, bits)[tr], y[tr])`` per split; ``tests/test_fbcsp.py``
  checks this. Set ``fbcsp.CACHE_COVARIANCES = False`` to disable the cache, or
  call :func:`clear_cache`.
"""
from __future__ import annotations

import time
import weakref
from collections import OrderedDict
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from . import chromosome
from .chromosome import BitsLike

Split = tuple[np.ndarray, np.ndarray]

#: Use the per-``X`` covariance cache in :func:`evaluate`.
CACHE_COVARIANCES: bool = True
_CACHE_SIZE = 8
_JITTER = 1e-6
_LOG_FLOOR = 1e-30
_TP_CONTROLLER = None


def _blas_single_thread():
    """Context manager limiting BLAS to 1 thread.

    The matrices here are tiny (≤ 8×8 CSP, ≤ 168×168 LDA), and multi-threaded
    BLAS made the all-on evaluation ~5× slower. Parallelise across calls
    (joblib) instead.
    """
    global _TP_CONTROLLER
    if _TP_CONTROLLER is None:
        from threadpoolctl import ThreadpoolController
        _TP_CONTROLLER = ThreadpoolController()
    return _TP_CONTROLLER.limit(limits=1, user_api="blas")


# ---------------------------------------------------------------- covariances
def trial_covariances(Xsel: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Per-trial spatial covariances.

    ``Xsel`` ``(..., n_ch, T)`` → ``(..., n_ch, n_ch)`` float64, channel means
    removed, normalised by ``T``. It works in chunks of ``chunk`` matrices
    through a small float64 buffer. Allocating one large float64 temporary
    was 3–4× slower (page faults dominated).
    """
    X = np.asarray(Xsel)
    c, T = X.shape[-2:]
    Y = X.reshape(-1, c, T)
    N = Y.shape[0]
    out = np.empty((N, c, c), dtype=np.float64)
    buf = np.empty((min(chunk, max(N, 1)), c, T), dtype=np.float64)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        b = buf[: e - s]
        b[...] = Y[s:e]
        b -= b.mean(axis=-1, keepdims=True)
        np.matmul(b, b.transpose(0, 2, 1), out=out[s:e])
    out /= T
    return out.reshape(X.shape[:-1] + (c,))


def _full_covariances_uncached(X: np.ndarray) -> np.ndarray:
    """Covariances of the full cache array ``(4, 7, n, 8, T)``, trial-major: ``(n, 4, 7, 8, 8)``."""
    nb, nw, n, c, _ = X.shape
    out = np.empty((n, nb, nw, c, c), dtype=np.float64)
    for b in range(nb):  # per band, to keep the float64 temporary small
        out[:, b] = trial_covariances(X[b]).transpose(1, 0, 2, 3)
    return out


_COV_CACHE: "OrderedDict[int, tuple[Any, tuple, np.ndarray]]" = OrderedDict()


def _fingerprint(X: np.ndarray) -> tuple:
    flat = X.reshape(-1) if X.flags.c_contiguous else X.ravel(order="K")
    sample = flat[::1009]
    return (X.shape, X.dtype.str, X.__array_interface__["data"][0],
            float(sample.sum(dtype=np.float64)), float(np.abs(sample).sum(dtype=np.float64)))


def full_covariances(X: np.ndarray) -> np.ndarray:
    """Per-trial covariances of the full FBCSP cache array, cached per ``X`` object.

    Returns trial-major ``(n_trials, 4, 7, 8, 8)`` float64 (read-only).
    The cache key is the object identity plus a cheap fingerprint, so do not
    modify ``X`` in place between calls (or call :func:`clear_cache` if you do).
    """
    if not (CACHE_COVARIANCES and isinstance(X, np.ndarray)):
        return _full_covariances_uncached(np.asarray(X))
    key = id(X)
    fp = _fingerprint(X)
    hit = _COV_CACHE.get(key)
    if hit is not None and hit[0]() is X and hit[1] == fp:
        _COV_CACHE.move_to_end(key)
        return hit[2]
    covs = _full_covariances_uncached(X)
    covs.setflags(write=False)
    _COV_CACHE[key] = (weakref.ref(X), fp, covs)
    _COV_CACHE.move_to_end(key)
    while len(_COV_CACHE) > _CACHE_SIZE:
        _COV_CACHE.popitem(last=False)
    return covs


def clear_cache() -> None:
    """Empty the covariance cache used by :func:`evaluate`."""
    _COV_CACHE.clear()


def _check_full_X(X: np.ndarray) -> None:
    if X.ndim != 5 or X.shape[:2] != (chromosome.N_BAND, chromosome.N_WIN) \
            or X.shape[3] != chromosome.N_CH:
        raise ValueError(f"X must have shape (4, 7, n, 8, T), got {X.shape}")


def select_covariances(X: np.ndarray, bits: BitsLike) -> np.ndarray:
    """Covariance-domain equivalent of ``trial_covariances(chromosome.select(X, bits))``.

    Returns ``(n_trials, n_bw, n_ch_sel, n_ch_sel)`` with band-major ``bw`` order.
    Raises ``ValueError`` on invalid bits, like ``chromosome.select``.
    """
    if not chromosome.is_valid(bits):
        raise ValueError(f"invalid chromosome {chromosome.to_str(bits)}")
    X = X if isinstance(X, np.ndarray) else np.asarray(X)
    _check_full_X(X)
    ch_m, b_m, w_m = chromosome.decode(bits)
    covs = full_covariances(X)                                         # (n, 4, 7, 8, 8)
    ch, b, w = np.flatnonzero(ch_m), np.flatnonzero(b_m), np.flatnonzero(w_m)
    n = covs.shape[0]
    sub = covs[np.ix_(np.arange(n), b, w, ch, ch)]                     # (n, nb, nw, c, c)
    return sub.reshape(n, b.size * w.size, ch.size, ch.size)


# ---------------------------------------------------------------- CSP core
def _ridge(C: np.ndarray, frac: float) -> np.ndarray:
    c = C.shape[-1]
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None] / c
    return C + frac * tr * np.eye(c)


def _ovr_csp_filters(C: np.ndarray, y: np.ndarray, classes: np.ndarray, n_pairs: int,
                     cov_reg: float) -> np.ndarray:
    """One-vs-rest CSP filters from trial covariances ``C`` ``(n, B, c, c)``.

    Returns ``(B, K, 2 * n_pairs, c)``: per band-window and class, filters for
    the ``n_pairs`` largest then the ``n_pairs`` smallest generalised eigenvalues.
    """
    n, B, c, _ = C.shape
    if 2 * n_pairs > c:
        raise ValueError(f"n_filter_pairs={n_pairs} needs at least {2 * n_pairs} channels, got {c}")
    sums = np.stack([C[y == k].sum(axis=0) for k in classes], axis=1)   # (B, K, c, c)
    counts = np.array([(y == k).sum() for k in classes], dtype=np.float64)
    total = sums.sum(axis=1, keepdims=True)
    Cc = sums / counts[None, :, None, None]
    Cr = (total - sums) / (n - counts)[None, :, None, None]
    if cov_reg:
        Cc, Cr = _ridge(Cc, cov_reg), _ridge(Cr, cov_reg)
    A = Cc + Cr
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        # Regularise only where needed (positive-definite slices are unchanged).
        ok = np.all(np.linalg.eigvalsh(A) > 1e-12 * np.trace(A, axis1=-2, axis2=-1)[..., None], -1)
        Cc = np.where(ok[..., None, None], Cc, _ridge(Cc, _JITTER))
        Cr = np.where(ok[..., None, None], Cr, _ridge(Cr, _JITTER))
        A = Cc + Cr
        L = np.linalg.cholesky(A)
    Linv = np.linalg.inv(L)
    M = Linv @ Cc @ Linv.swapaxes(-1, -2)
    M = (M + M.swapaxes(-1, -2)) / 2
    _, V = np.linalg.eigh(M)                       # ascending eigenvalues
    W = Linv.swapaxes(-1, -2) @ V                  # columns = filters in channel space
    idx = np.r_[np.arange(c - 1, c - 1 - n_pairs, -1), np.arange(n_pairs)]
    return np.ascontiguousarray(W[..., idx].swapaxes(-1, -2))


def _logvar_features(C: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Log-variance of filtered trials: ``C`` ``(n, B, c, c)``, filters ``(B, K, F, c)`` → ``(n, B*K*F)``."""
    B, K, F, c = filters.shape
    Wf = filters.reshape(B, K * F, c)
    v = np.einsum("bfc,nbfc->nbf", Wf, np.matmul(Wf[None], C), optimize=False)
    return np.log(np.maximum(v, _LOG_FLOOR)).reshape(C.shape[0], B * K * F)


# ---------------------------------------------------------------- fast scaler + LDA
def _fit_scaler(F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``StandardScaler`` statistics ``(mean, var, scale)`` with sklearn's constant-feature rule."""
    n = F.shape[0]
    mean = F.mean(axis=0)
    var = F.var(axis=0)
    eps = np.finfo(np.float64).eps
    constant = var <= n * eps * var + (n * mean * eps) ** 2
    scale = np.where(constant, 1.0, np.sqrt(var))
    return mean, var, scale


def _lw_shrunk_cov(Z: np.ndarray) -> np.ndarray:
    """``sklearn.covariance.ledoit_wolf(Z)[0]`` (not assumed centred), via the n×n Gram matrix."""
    n, p = Z.shape
    Zc = Z - Z.mean(axis=0)
    emp = (Zc.T @ Zc) / n
    X2 = Zc * Zc
    emp_trace = X2.sum(axis=0) / n
    mu = emp_trace.sum() / p
    beta_ = float(np.sum(X2.sum(axis=1) ** 2))          # sum(X2.T @ X2)
    G = Zc @ Zc.T
    delta_ = float(np.sum(G * G)) / n ** 2                # sum((X.T @ X)**2) / n**2
    beta = (beta_ / n - delta_) / (p * n)
    delta = (delta_ - 2.0 * mu * emp_trace.sum() + p * mu ** 2) / p
    beta = min(beta, delta)
    s = 0.0 if beta == 0 else beta / delta
    out = (1.0 - s) * emp
    out.flat[:: p + 1] += s * mu
    return out


def _class_cov_fast(F: np.ndarray, y: np.ndarray, classes: np.ndarray, priors: np.ndarray,
                    shrinkage) -> np.ndarray:
    """``sklearn.discriminant_analysis._class_cov`` for shrinkage in {None, 'empirical', 'auto', float}."""
    p = F.shape[1]
    cov = np.zeros((p, p))
    for pk, k in zip(priors, classes):
        Xg = F[y == k]
        if isinstance(shrinkage, str) and shrinkage == "auto":
            _, _, sc = _fit_scaler(Xg)
            Zg = (Xg - Xg.mean(axis=0)) / sc
            s = _lw_shrunk_cov(Zg) * sc[:, None] * sc[None, :]
        else:
            Xc = Xg - Xg.mean(axis=0)
            s = (Xc.T @ Xc) / Xg.shape[0]
            if shrinkage not in (None, "empirical"):
                a = float(shrinkage)
                mu = np.trace(s) / p
                s = (1.0 - a) * s
                s.flat[:: p + 1] += a * mu
        cov += pk * s
    return cov


def _fast_lda_ok(solver: str, shrinkage) -> bool:
    if solver != "lsqr":
        return False
    if shrinkage is None or (isinstance(shrinkage, str) and shrinkage in ("auto", "empirical")):
        return True
    return isinstance(shrinkage, (int, float)) and not isinstance(shrinkage, bool) \
        and 0.0 <= float(shrinkage) <= 1.0


def _fit_lda_lsqr(Z: np.ndarray, y: np.ndarray, classes: np.ndarray, shrinkage) -> dict:
    """NumPy re-implementation of ``LinearDiscriminantAnalysis(solver="lsqr").fit``.

    Same maths as sklearn 1.4-1.9 (class means, prior-weighted per-class
    (shrunk) covariance, ``coef = cov^-1 means^T``). ``tests/test_fbcsp.py``
    checks it against sklearn. It skips sklearn's input validation, which
    dominated the run time.
    """
    counts = np.array([(y == k).sum() for k in classes], dtype=np.float64)
    priors = counts / counts.sum()
    means = np.stack([Z[y == k].mean(axis=0) for k in classes])
    cov = _class_cov_fast(Z, y, classes, priors, shrinkage)
    shrunk = isinstance(shrinkage, str) and shrinkage == "auto" or (
        not isinstance(shrinkage, str) and shrinkage is not None and float(shrinkage) > 0)
    coef = None
    if shrunk:  # shrunk covariance is positive definite: a direct solve is exact and fast
        try:
            coef = np.linalg.solve(cov, means.T).T
        except np.linalg.LinAlgError:
            coef = None
    if coef is None:  # possibly singular: sklearn's minimum-norm least squares
        from scipy import linalg
        coef = linalg.lstsq(cov, means.T)[0].T
    intercept = -0.5 * np.einsum("kp,kp->k", means, coef) + np.log(priors)
    if classes.size == 2:  # sklearn's binary convention
        coef = (coef[1] - coef[0])[None, :]
        intercept = np.array([intercept[1] - intercept[0]])
    return {"priors_": priors, "means_": means, "covariance_": cov,
            "coef_": coef, "intercept_": intercept}


# ---------------------------------------------------------------- estimator
class FBCSPLDA(BaseEstimator, ClassifierMixin):
    """One-vs-rest FBCSP feature extractor followed by shrinkage LDA.

    Input ``Xsel`` has shape ``(n_trials, n_bw, n_ch_sel, 250)``, the output of
    ``chromosome.select``.

    For each band-window and each class ``c`` (one-vs-rest), CSP is fitted from
    the class-``c`` covariance against the rest covariance, and 1 filter pair
    is kept (largest and smallest generalised eigenvalue). Features are the
    log-variance of the 2 filtered signals. That gives 3 classes × 2 = 6
    features per band-window, so ``(n_trials, 6 * n_bw)`` features in total.
    The features are standardised (``StandardScaler``) and classified with
    ``LinearDiscriminantAnalysis(solver=lda_solver, shrinkage=lda_shrinkage)``
    on all features together. It works with ``n_ch_sel = 2``.

    Parameters
    ----------
    n_filter_pairs
        Filter pairs per class per band-window (contract: 1).
    lda_solver, lda_shrinkage
        Passed to LDA (contract: "lsqr", "auto").
    cov_reg
        Ridge added to the class and rest covariance diagonals, as a fraction
        of ``trace/n_ch``. Default 0.0 (see the module docstring).

    Attributes (after fit)
    ----------------------
    classes_ : ndarray (3,)
    filters_ : ndarray (n_bw, 3, 2 * n_filter_pairs, n_ch_sel)
    scaler_ : fitted StandardScaler
    lda_ : fitted LinearDiscriminantAnalysis
    """

    def __init__(self, n_filter_pairs: int = 1, lda_solver: str = "lsqr",
                 lda_shrinkage: str | float | None = "auto", cov_reg: float = 0.0):
        self.n_filter_pairs = n_filter_pairs
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage
        self.cov_reg = cov_reg

    # -- input handling
    @staticmethod
    def _covs(Xsel: np.ndarray) -> np.ndarray:
        Xsel = np.asarray(Xsel)
        if Xsel.ndim != 4:
            raise ValueError(f"Xsel must be (n_trials, n_bw, n_ch_sel, T), got shape {Xsel.shape}")
        return trial_covariances(Xsel)

    # -- covariance-domain API (used by evaluate(); equivalent to fit/predict on Xsel)
    def _fit_cov(self, C: np.ndarray, y: np.ndarray) -> "FBCSPLDA":
        y = np.asarray(y)
        if C.shape[0] != y.shape[0]:
            raise ValueError(f"{C.shape[0]} trials but {y.shape[0]} labels")
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("need at least 2 classes to fit")
        self.n_bw_, self.n_ch_ = C.shape[1], C.shape[2]
        self.filters_ = _ovr_csp_filters(C, y, self.classes_, int(self.n_filter_pairs),
                                         float(self.cov_reg))
        F = _logvar_features(C, self.filters_)
        mean, var, scale = _fit_scaler(F)
        self.scaler_ = StandardScaler()
        self.scaler_.mean_, self.scaler_.var_, self.scaler_.scale_ = mean, var, scale
        self.scaler_.n_samples_seen_ = np.int64(F.shape[0])
        self.scaler_.n_features_in_ = F.shape[1]
        Z = (F - mean) / scale
        self.lda_ = LinearDiscriminantAnalysis(solver=self.lda_solver,
                                               shrinkage=self.lda_shrinkage)
        if _fast_lda_ok(self.lda_solver, self.lda_shrinkage):
            for k, v in _fit_lda_lsqr(Z, y, self.classes_, self.lda_shrinkage).items():
                setattr(self.lda_, k, v)
            self.lda_.classes_ = self.classes_
            self.lda_.n_features_in_ = Z.shape[1]
        else:  # any other solver: plain sklearn
            self.lda_.fit(Z, y)
        return self

    def _transform_cov(self, C: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "filters_")
        if C.shape[1:3] != (self.n_bw_, self.n_ch_):
            raise ValueError(f"expected (n, {self.n_bw_}, {self.n_ch_}, ...) input, got {C.shape}")
        return _logvar_features(C, self.filters_)

    def _predict_cov(self, C: np.ndarray) -> np.ndarray:
        Z = (self._transform_cov(C) - self.scaler_.mean_) / self.scaler_.scale_
        scores = Z @ self.lda_.coef_.T + self.lda_.intercept_
        if scores.shape[1] == 1:
            return self.classes_[(scores[:, 0] > 0).astype(np.int64)]
        return self.classes_[np.argmax(scores, axis=1)]

    # -- public sklearn API
    def fit(self, Xsel: np.ndarray, y: np.ndarray) -> "FBCSPLDA":
        """Fit CSP filters per band-window and class, then LDA on the log-variance features."""
        with _blas_single_thread():
            return self._fit_cov(self._covs(Xsel), y)

    def transform(self, Xsel: np.ndarray) -> np.ndarray:
        """Return log-variance features, shape ``(n_trials, 6 * n_bw)``."""
        return self._transform_cov(self._covs(Xsel))

    def predict(self, Xsel: np.ndarray) -> np.ndarray:
        """Predict class labels, shape ``(n_trials,)``."""
        with _blas_single_thread():
            return self._predict_cov(self._covs(Xsel))

    def predict_proba(self, Xsel: np.ndarray) -> np.ndarray:
        """LDA class probabilities, shape ``(n_trials, n_classes)``."""
        return self.lda_.predict_proba(self.scaler_.transform(self.transform(Xsel)))

    def decision_function(self, Xsel: np.ndarray) -> np.ndarray:
        """LDA decision values."""
        return self.lda_.decision_function(self.scaler_.transform(self.transform(Xsel)))


# ---------------------------------------------------------------- evaluation
def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast ``sklearn.metrics.balanced_accuracy_score`` (mean recall over classes in ``y_true``)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(y_true)
    return float(np.mean([np.mean(y_pred[y_true == k] == k) for k in classes]))


def evaluate(X: np.ndarray, y: np.ndarray, bits: BitsLike,
             splits: Sequence[Split], **fbcsp_kwargs: Any) -> dict[str, Any]:
    """Cross-validated balanced accuracy of FBCSPLDA on the components in ``bits``.

    Parameters
    ----------
    X
        **Full** FBCSP cache array ``(4, 7, n_trials, 8, 250)``. The selection
        given by ``bits`` (``chromosome.select`` semantics, and its
        ``ValueError`` on invalid bits) is applied here, in the covariance
        domain.
    y
        int64 ``(n_trials,)``.
    bits
        Valid chromosome.
    splits
        Absolute ``(train_idx, test_idx)`` pairs (from ``src.cv``).
    **fbcsp_kwargs
        Passed to ``FBCSPLDA``.

    Returns
    -------
    dict
        ``{"ba_mean": float, "ba_std": float, "per_fold": list[float],
        "fit_time_s": float}``. ``ba_std`` uses ddof=0. ``fit_time_s`` is the
        total wall time for all folds (fit + predict). A fresh FBCSPLDA is
        fitted per split. Extra key: ``"y_pred"``, a list with one int64
        prediction array per split (aligned with that split's ``test_idx``).
    """
    t0 = time.perf_counter()
    y = np.asarray(y)
    per_fold: list[float] = []
    preds: list[np.ndarray] = []
    with _blas_single_thread():
        C = select_covariances(X, bits)
        if C.shape[0] != y.shape[0]:
            raise ValueError(f"X has {C.shape[0]} trials but y has {y.shape[0]}")
        for tr, te in splits:
            tr, te = np.asarray(tr), np.asarray(te)
            if np.intersect1d(tr, te).size:
                raise ValueError("train and test indices overlap")
            est = FBCSPLDA(**fbcsp_kwargs)._fit_cov(C[tr], y[tr])
            yp = est._predict_cov(C[te])
            per_fold.append(balanced_accuracy(y[te], yp))
            preds.append(yp.astype(np.int64))
    if not per_fold:
        raise ValueError("splits is empty")
    return {"ba_mean": float(np.mean(per_fold)), "ba_std": float(np.std(per_fold)),
            "per_fold": per_fold, "fit_time_s": time.perf_counter() - t0, "y_pred": preds}

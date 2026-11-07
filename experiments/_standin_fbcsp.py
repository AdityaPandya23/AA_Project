"""STAND-IN for ``src.fbcsp`` (owned by R01) until the real implementation lands.

It follows CONTRACTS.md §f closely (one-vs-rest CSP, 1 filter pair per class per
band-window, log-variance features, shrinkage LDA), so it is good enough for
developing the GA and for runtime benchmarks. It is **not** the reference
implementation: results produced with it are tagged ``"fbcsp_impl": "standin"``
and must not be used in the paper.

Choices (internal to fbcsp, CONTRACTS open item 6): trace-normalised trial
covariances, ``cov_reg`` ridge as a fraction of trace/n_ch, raw (not
normalised) log-variance.
"""
from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

from src import chromosome


def _trial_covs(Xsel: np.ndarray) -> np.ndarray:
    """``(n, n_bw, c, t)`` -> trace-normalised covariances ``(n, n_bw, c, c)``."""
    Xc = Xsel - Xsel.mean(axis=-1, keepdims=True)
    C = np.einsum("nbct,nbdt->nbcd", Xc, Xc, optimize=True)
    tr = np.trace(C, axis1=-2, axis2=-1)[..., None, None]
    return C / tr


class FBCSPLDA(BaseEstimator, ClassifierMixin):
    """Stand-in one-vs-rest FBCSP + shrinkage LDA (CONTRACTS.md §f)."""

    def __init__(self, n_filter_pairs: int = 1, lda_solver: str = "lsqr",
                 lda_shrinkage: str | float | None = "auto", cov_reg: float = 0.0):
        self.n_filter_pairs = n_filter_pairs
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage
        self.cov_reg = cov_reg

    def fit(self, Xsel: np.ndarray, y: np.ndarray) -> "FBCSPLDA":
        """Fit CSP filters per band-window and class, then LDA."""
        Xsel = np.asarray(Xsel, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        covs = _trial_covs(Xsel)
        n_bw, n_ch = Xsel.shape[1], Xsel.shape[2]
        k = self.n_filter_pairs
        filters = np.empty((n_bw, len(self.classes_), 2 * k, n_ch))
        eye = np.eye(n_ch)
        for ci, c in enumerate(self.classes_):
            Ca = covs[y == c].mean(axis=0)   # (n_bw, c, c)
            Cb = covs[y != c].mean(axis=0)
            for b in range(n_bw):
                A, B = Ca[b], Cb[b]
                if self.cov_reg:
                    A = A + self.cov_reg * np.trace(A) / n_ch * eye
                    B = B + self.cov_reg * np.trace(B) / n_ch * eye
                w, V = eigh(A, A + B)
                order = np.argsort(w)
                pick = np.r_[order[-k:][::-1], order[:k]]
                filters[b, ci] = V[:, pick].T
        self.filters_ = filters
        self.lda_ = LinearDiscriminantAnalysis(solver=self.lda_solver,
                                               shrinkage=self.lda_shrinkage)
        self.lda_.fit(self._features(Xsel), y)
        return self

    def _features(self, Xsel: np.ndarray) -> np.ndarray:
        Xsel = np.asarray(Xsel, dtype=np.float64)
        # (n, bw, c, t) x (bw, cls, f, c) -> (n, bw, cls, f, t)
        Z = np.einsum("bkfc,nbct->nbkft", self.filters_, Xsel, optimize=True)
        return np.log(Z.var(axis=-1)).reshape(Xsel.shape[0], -1)

    def transform(self, Xsel: np.ndarray) -> np.ndarray:
        """Log-variance features ``(n_trials, 6 * n_bw)``."""
        return self._features(Xsel)

    def predict(self, Xsel: np.ndarray) -> np.ndarray:
        """Predicted labels ``(n_trials,)``."""
        return self.lda_.predict(self._features(Xsel))


def evaluate(X: np.ndarray, y: np.ndarray, bits, splits: Sequence[tuple[np.ndarray, np.ndarray]],
             **fbcsp_kwargs: Any) -> dict[str, Any]:
    """Same contract as ``src.fbcsp.evaluate``."""
    t0 = time.perf_counter()
    Xsel = chromosome.select(X, bits)
    scores = []
    for tr, te in splits:
        clf = FBCSPLDA(**fbcsp_kwargs).fit(Xsel[tr], y[tr])
        scores.append(float(balanced_accuracy_score(y[te], clf.predict(Xsel[te]))))
    return {"ba_mean": float(np.mean(scores)), "ba_std": float(np.std(scores)),
            "per_fold": scores, "fit_time_s": time.perf_counter() - t0}

"""One-vs-rest filter-bank CSP + shrinkage LDA (CONTRACTS.md §f). STUB."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from .chromosome import BitsLike

Split = tuple[np.ndarray, np.ndarray]


class FBCSPLDA(BaseEstimator, ClassifierMixin):
    """One-vs-rest FBCSP feature extractor followed by shrinkage LDA.

    Input ``Xsel`` has shape ``(n_trials, n_bw, n_ch_sel, 250)``, the output of
    ``chromosome.select``.

    For each band-window and each class ``c`` (one-vs-rest), CSP is fitted from
    the class-``c`` covariance against the rest covariance, and 1 filter pair
    is kept (largest and smallest generalised eigenvalue). Features are the
    log-variance of the 2 filtered signals. That gives 3 classes × 2 = 6
    features per band-window, so ``(n_trials, 6 * n_bw)`` features in total.
    Classifier: ``LinearDiscriminantAnalysis(solver=lda_solver,
    shrinkage=lda_shrinkage)`` on all features together. It must work with
    ``n_ch_sel = 2``.

    Parameters
    ----------
    n_filter_pairs
        Filter pairs per class per band-window (contract: 1).
    lda_solver, lda_shrinkage
        Passed to LDA (contract: "lsqr", "auto").
    cov_reg
        Ridge added to the covariance diagonal (fraction of trace/n_ch).
        Open item: record the choice made.

    Attributes (after fit)
    ----------------------
    classes_ : ndarray (3,)
    filters_ : ndarray (n_bw, 3, 2 * n_filter_pairs, n_ch_sel)
    lda_ : fitted LinearDiscriminantAnalysis
    """

    def __init__(self, n_filter_pairs: int = 1, lda_solver: str = "lsqr",
                 lda_shrinkage: str | float | None = "auto", cov_reg: float = 0.0):
        self.n_filter_pairs = n_filter_pairs
        self.lda_solver = lda_solver
        self.lda_shrinkage = lda_shrinkage
        self.cov_reg = cov_reg

    def fit(self, Xsel: np.ndarray, y: np.ndarray) -> "FBCSPLDA":
        """Fit CSP filters per band-window and class, then LDA on the log-variance features."""
        raise NotImplementedError

    def transform(self, Xsel: np.ndarray) -> np.ndarray:
        """Return log-variance features, shape ``(n_trials, 6 * n_bw)``."""
        raise NotImplementedError

    def predict(self, Xsel: np.ndarray) -> np.ndarray:
        """Predict class labels, shape ``(n_trials,)``."""
        raise NotImplementedError


def evaluate(X: np.ndarray, y: np.ndarray, bits: BitsLike,
             splits: Sequence[Split], **fbcsp_kwargs: Any) -> dict[str, Any]:
    """Cross-validated balanced accuracy of FBCSPLDA on the components in ``bits``.

    Parameters
    ----------
    X
        **Full** FBCSP cache array ``(4, 7, n_trials, 8, 250)``. This function
        calls ``chromosome.select``.
    y
        int64 ``(n_trials,)``.
    bits
        Valid chromosome.
    splits
        Absolute ``(train_idx, test_idx)`` pairs (from ``src.cv``).

    Returns
    -------
    dict
        ``{"ba_mean": float, "ba_std": float, "per_fold": list[float],
        "fit_time_s": float}``. ``ba_std`` uses ddof=0. ``fit_time_s`` is the
        total wall time for all folds (fit + predict). A fresh FBCSPLDA is
        fitted per split.
    """
    raise NotImplementedError

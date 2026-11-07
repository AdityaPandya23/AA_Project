"""Genetic search over the 19-bit chromosome (CONTRACTS.md §g). STUB.

Fitness of ``bits`` =
``fbcsp.evaluate(X, y, bits, inner_splits(y, block, train_idx, n=cfg["ga"]["inner_folds"], seed=seed))``.
The inner splits are computed once per run, so every individual is scored on
the same folds. Memoise fitness by ``chromosome.to_str(bits)``.

Binary genome, with ``chromosome.repair`` applied after sampling, crossover and
mutation (pymoo ``Repair``). ``pop_size`` and ``n_generations`` come from
``cfg["ga"]``. ``seed`` seeds both pymoo and ``inner_splits``.

Nesting rule: only indices in ``train_idx`` may be used. The outer test fold
must never be seen.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .chromosome import BitsLike

GAResult = dict[str, Any]
"""``{"pareto": [{"bits": ndarray uint8 (19,), "ba_mean": float, "ba_std": float,
"cost": float}, ...], "chosen_bits": ndarray uint8 (19,), "n_evals": int,
"time_s": float}``"""


def run_nsga(X: np.ndarray, y: np.ndarray, block: np.ndarray, train_idx: np.ndarray,
             cfg: dict[str, Any], seed: int) -> GAResult:
    """Stability-aware NSGA-II (pymoo).

    Objectives (all minimised): ``[-ba_mean, ba_std, cost]``.

    Parameters
    ----------
    X
        Full FBCSP cache array ``(4, 7, n_trials, 8, 250)``.
    y, block
        Full-length int64 arrays.
    train_idx
        Absolute indices the GA may use.
    cfg
        From ``load_config()``.
    seed
        Seeds pymoo and the inner splits.

    Returns
    -------
    GAResult
        ``pareto`` = the non-dominated set of the final population (unique bits).
        ``chosen_bits = choose_from_pareto(pareto)``.
    """
    raise NotImplementedError


def run_single_objective(X: np.ndarray, y: np.ndarray, block: np.ndarray,
                         train_idx: np.ndarray, cfg: dict[str, Any], seed: int) -> GAResult:
    """Accuracy-only GA baseline (pymoo ``GA``). Objective: ``-ba_mean``.

    Returns
    -------
    GAResult
        The same shape as :func:`run_nsga`. ``pareto`` = the unique individuals
        of the final population. ``chosen_bits`` = highest ``ba_mean``. Ties go
        to lower ``cost``, then lexicographically smallest ``to_str(bits)``.
        It does **not** use :func:`choose_from_pareto`.
    """
    raise NotImplementedError


def choose_from_pareto(pareto: list[dict[str, Any]]) -> np.ndarray:
    """Lowest-cost solution within 1 SD of the best accuracy.

    Let ``b*`` = the entry with the highest ``ba_mean``. Candidates are the
    entries with ``ba_mean >= b*.ba_mean - b*.ba_std``. Return the candidate
    ``bits`` with the lowest ``cost``. Ties go to higher ``ba_mean``, then lower
    ``ba_std``, then lexicographically smallest ``to_str(bits)``.

    Returns
    -------
    np.ndarray
        uint8 ``(19,)``.
    """
    raise NotImplementedError


def fitness(X: np.ndarray, y: np.ndarray, bits: BitsLike,
            splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """Return ``(ba_mean, ba_std, cost)`` for one chromosome on fixed inner splits."""
    raise NotImplementedError

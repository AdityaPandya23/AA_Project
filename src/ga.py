"""Genetic search over the 19-bit chromosome (CONTRACTS.md §g).

Fitness of ``bits`` =
``fbcsp.evaluate(X, y, bits, inner_splits(y, block, train_idx, n=cfg["ga"]["inner_folds"], seed=seed))``.
The inner splits are computed once per run, so every individual is scored on
the same folds. Fitness is memoised by ``chromosome.to_str(bits)``, so
duplicate chromosomes cost nothing.

Binary genome, with ``chromosome.repair`` applied after sampling, crossover and
mutation (pymoo ``Repair``). ``pop_size`` and ``n_generations`` come from
``cfg["ga"]``. ``seed`` seeds both pymoo and ``inner_splits`` (and the repair RNG).

Nesting rule: only indices in ``train_idx`` may be used. The outer test fold
must never be seen. This is asserted in code (:func:`_check_nesting`).

Implementation hooks (not part of the contract):

* :func:`set_evaluate_fn` replaces ``fbcsp.evaluate`` (tests only). ``None``
  restores the default.

Performance (R01's notes, CONTRACT_ISSUES.md §3): ``fbcsp.evaluate`` caches
per-trial covariances by the identity of ``X``. The GA therefore passes the
**same** ``X`` object to every call (no copy, slice or ``asarray`` that
could copy) and never modifies it. ``evaluate`` pins BLAS to 1 thread, so
parallelism belongs one level up: ``experiments/run_ga.py`` runs folds,
subjects and seeds as separate joblib jobs. Individuals are evaluated
sequentially inside a job, because shipping ``X`` to worker processes would
create a new object per call and miss the cache.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from . import chromosome
from .chromosome import BitsLike

GAResult = dict[str, Any]
"""``{"pareto": [{"bits": ndarray uint8 (19,), "ba_mean": float, "ba_std": float,
"cost": float}, ...], "chosen_bits": ndarray uint8 (19,), "n_evals": int,
"time_s": float}``"""

Split = tuple[np.ndarray, np.ndarray]
EvaluateFn = Callable[..., dict[str, Any]]

_evaluate_override: EvaluateFn | None = None


def set_evaluate_fn(fn: EvaluateFn | None) -> None:
    """Use ``fn`` (same signature as ``fbcsp.evaluate``) for fitness. ``None`` = ``fbcsp.evaluate``."""
    global _evaluate_override
    _evaluate_override = fn


def _evaluate_fn() -> EvaluateFn:
    if _evaluate_override is not None:
        return _evaluate_override
    from . import fbcsp
    return fbcsp.evaluate


# --------------------------------------------------------------------- splits
def _inner_splits(y: np.ndarray, block: np.ndarray, train_idx: np.ndarray,
                  n: int, seed: int) -> list[Split]:
    from . import cv
    return cv.inner_splits(y, block, train_idx, n=n, seed=seed)


def _check_nesting(splits: list[Split], train_idx: np.ndarray, n_total: int) -> None:
    """Assert every inner index lies in ``train_idx`` (never in the outer test fold)."""
    allowed = np.zeros(n_total, dtype=bool)
    allowed[np.asarray(train_idx, dtype=np.int64)] = True
    for tr, te in splits:
        both = np.concatenate([tr, te]).astype(np.int64)
        assert both.size and both.min() >= 0 and both.max() < n_total, "inner index out of range"
        assert allowed[both].all(), "NESTING VIOLATION: GA inner split uses indices outside train_idx"
        assert np.intersect1d(tr, te).size == 0, "inner train/test overlap"


# --------------------------------------------------------------------- fitness
def fitness(X: np.ndarray, y: np.ndarray, bits: BitsLike,
            splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """Return ``(ba_mean, ba_std, cost)`` for one chromosome on fixed inner splits."""
    r = _evaluate_fn()(X, y, chromosome._as_bits(bits), splits)
    return float(r["ba_mean"]), float(r["ba_std"]), float(chromosome.cost(bits))


class _FitnessCache:
    """Memoised fitness keyed by ``to_str(bits)``. ``len()`` = unique evaluations."""

    def __init__(self, X: np.ndarray, y: np.ndarray, splits: list[Split]):
        self.X, self.y, self.splits = X, y, splits
        self.memo: dict[str, tuple[float, float, float]] = {}

    def __call__(self, bits: BitsLike) -> tuple[float, float, float]:
        key = chromosome.to_str(bits)
        if key not in self.memo:
            self.memo[key] = fitness(self.X, self.y, bits, self.splits)
        return self.memo[key]

    def __len__(self) -> int:
        return len(self.memo)


# --------------------------------------------------------------------- pymoo glue
def _make_problem(fit: _FitnessCache, n_obj: int):
    from pymoo.core.problem import Problem

    class _ChromosomeProblem(Problem):
        def __init__(self):
            super().__init__(n_var=chromosome.N_BITS, n_obj=n_obj, xl=0, xu=1, vtype=bool)

        def _evaluate(self, x, out, *args, **kwargs):
            F = np.empty((len(x), n_obj))
            for i, bits in enumerate(x):
                ba_mean, ba_std, c = fit(bits.astype(np.uint8))
                F[i] = [-ba_mean, ba_std, c] if n_obj == 3 else [-ba_mean]
            out["F"] = F

    return _ChromosomeProblem()


def _make_repair(seed: int):
    from pymoo.core.repair import Repair

    class _ChromosomeRepair(Repair):
        """Applies ``chromosome.repair`` to every individual (switches genes on only)."""

        def __init__(self, rng: np.random.Generator):
            super().__init__()
            self.rng = rng

        def _do(self, problem, X, **kwargs):
            X = np.asarray(X)
            return np.array([chromosome.repair(x.astype(np.uint8), self.rng) for x in X],
                            dtype=bool).reshape(X.shape)

    return _ChromosomeRepair(np.random.default_rng([seed, 0x6A]))


def _operators(seed: int) -> dict[str, Any]:
    from pymoo.operators.crossover.pntx import TwoPointCrossover
    from pymoo.operators.mutation.bitflip import BitflipMutation
    from pymoo.operators.sampling.rnd import BinaryRandomSampling
    return dict(sampling=BinaryRandomSampling(), crossover=TwoPointCrossover(),
                mutation=BitflipMutation(prob=1.0, prob_var=1.0 / chromosome.N_BITS),
                repair=_make_repair(seed), eliminate_duplicates=True)


def _run(X, y, block, train_idx, cfg, seed, n_obj: int):
    """Shared driver. Returns ``(final_pop_bits, fitness_cache, time_s)``."""
    from pymoo.optimize import minimize

    t0 = time.perf_counter()
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array (passed through unchanged for the fbcsp cache)")
    y = np.asarray(y)
    block = np.asarray(block)
    train_idx = np.unique(np.asarray(train_idx, dtype=np.int64))
    n_total = y.shape[0]
    if X.shape[2] != n_total or block.shape[0] != n_total:
        raise ValueError("X, y and block must be full-length along the trial axis")
    gcfg = cfg["ga"]
    splits = _inner_splits(y, block, train_idx, n=int(gcfg["inner_folds"]), seed=int(seed))
    _check_nesting(splits, train_idx, n_total)

    fit = _FitnessCache(X, y, splits)
    if n_obj == 3:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        algo = NSGA2(pop_size=int(gcfg["pop_size"]), **_operators(seed))
    else:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        algo = GA(pop_size=int(gcfg["pop_size"]), **_operators(seed))
    res = minimize(_make_problem(fit, n_obj), algo, ("n_gen", int(gcfg["n_generations"])),
                   seed=int(seed), verbose=False, save_history=False)
    pop_bits = np.asarray(res.pop.get("X")).astype(np.uint8)
    return pop_bits, fit, time.perf_counter() - t0


def _unique_entries(pop_bits: np.ndarray, fit: _FitnessCache) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for b in pop_bits:
        key = chromosome.to_str(b)
        if key not in seen:
            m, s, c = fit(b)
            seen[key] = {"bits": chromosome.from_str(key), "ba_mean": m, "ba_std": s, "cost": c}
    return [seen[k] for k in sorted(seen)]


def _non_dominated(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    F = np.array([[-e["ba_mean"], e["ba_std"], e["cost"]] for e in entries])
    keep = []
    for i in range(len(F)):
        dominated = np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
        if not dominated:
            keep.append(entries[i])
    return keep


# --------------------------------------------------------------------- public API
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
    pop_bits, fit, t = _run(X, y, block, train_idx, cfg, seed, n_obj=3)
    pareto = _non_dominated(_unique_entries(pop_bits, fit))
    return {"pareto": pareto, "chosen_bits": choose_from_pareto(pareto),
            "n_evals": len(fit), "time_s": t}


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
    pop_bits, fit, t = _run(X, y, block, train_idx, cfg, seed, n_obj=1)
    entries = _unique_entries(pop_bits, fit)
    best = min(entries, key=lambda e: (-e["ba_mean"], e["cost"], chromosome.to_str(e["bits"])))
    return {"pareto": entries, "chosen_bits": best["bits"].copy(),
            "n_evals": len(fit), "time_s": t}


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
    if not pareto:
        raise ValueError("empty pareto set")
    # b*: highest ba_mean; ties broken like the final choice so b* is deterministic.
    best = min(pareto, key=lambda e: (-e["ba_mean"], e["ba_std"], e["cost"],
                                      chromosome.to_str(e["bits"])))
    thr = best["ba_mean"] - best["ba_std"]
    cands = [e for e in pareto if e["ba_mean"] >= thr - 1e-12]
    pick = min(cands, key=lambda e: (e["cost"], -e["ba_mean"], e["ba_std"],
                                     chromosome.to_str(e["bits"])))
    return chromosome._as_bits(pick["bits"])

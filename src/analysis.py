"""Results aggregation, statistics and plots (CONTRACTS.md §j). STUB.

Subject-level statistics are computed **after** :func:`aggregate_seeds`
(n = subjects), never across seeds as if they were independent samples.
"""
from __future__ import annotations

import os
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- loading
def load_results(results_dir: str | os.PathLike) -> pd.DataFrame:
    """Load every ``{results_dir}/*/*.json`` into one DataFrame.

    One row per JSON file. Columns: ``method, subject, session, seed,
    protocol, ba_mean, ba_std, train_time_s, infer_time_ms, config_hash,
    mean_cost, mean_n_features, per_fold`` (list of dicts), ``path``.
    Each record is validated with ``results.validate_result``. If several
    ``config_hash`` values are present, warn.
    """
    raise NotImplementedError


def explode_folds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (file, fold), with columns ``fold, ba, chosen_bits, cost, n_features`` added."""
    raise NotImplementedError


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per ``(method, subject, session, protocol)``."""
    raise NotImplementedError


# ---------------------------------------------------------------- statistics
def paired_test(df: pd.DataFrame, method_a: str, method_b: str, protocol: str = "block",
                metric: str = "ba_mean", test: str = "wilcoxon") -> dict[str, Any]:
    """Paired comparison over subjects (and sessions). ``test`` is ``"wilcoxon"`` or ``"ttest"``.

    Returns ``{"n", "mean_diff", "statistic", "p", "effect_size"}``.
    """
    raise NotImplementedError


def friedman_test(df: pd.DataFrame, methods: Sequence[str], protocol: str = "block",
                  metric: str = "ba_mean") -> dict[str, Any]:
    """Friedman test across ``methods``. Returns ``{"n", "statistic", "p", "mean_ranks"}``."""
    raise NotImplementedError


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-corrected p-values (``statsmodels...multipletests(method="holm")``)."""
    raise NotImplementedError


def above_chance(df: pd.DataFrame, method: str, protocol: str = "block",
                 chance: float = 1 / 3, n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-subject test of BA > chance. Returns one row per (subject, session) with ``p``."""
    raise NotImplementedError


def selection_frequency(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Frequency with which each of the 19 genes is chosen, over folds × seeds × subjects.

    Columns: ``gene, group (channel|band|window), name, frequency``.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- plots
def plot_method_comparison(df: pd.DataFrame, protocol: str = "block", ax=None):
    """Per-method BA distribution (box/strip + paired lines) with a chance line."""
    raise NotImplementedError


def plot_pareto(pareto: list[dict[str, Any]], chosen_bits=None, ax=None):
    """Cost vs. ba_mean scatter of a Pareto front (colour = ba_std), with the chosen point highlighted."""
    raise NotImplementedError


def plot_selection_frequency(freq_df: pd.DataFrame, ax=None):
    """Bar plot of gene selection frequency grouped by channel, band and window."""
    raise NotImplementedError


def plot_cost_vs_accuracy(df: pd.DataFrame, protocol: str = "block", ax=None):
    """Mean cost vs. ba_mean per method and subject."""
    raise NotImplementedError

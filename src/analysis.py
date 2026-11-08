"""Results aggregation, statistics and plots (CONTRACTS.md §j).

Subject-level statistics are computed **after** :func:`aggregate_seeds`
(n = subjects), never across seeds as if they were independent samples. For the
block protocol, the two sessions of a subject are averaged before a paired
test, so that n = subjects (see docs/CONTRACT_ISSUES.md).

Extra per-fold keys used when present (optional per CONTRACTS.md §0/§i):
``y_true`` / ``y_pred`` (permutation test, confusion matrix), ``n_test``
(binomial fallback) and ``pareto`` (cost-vs-accuracy plot). ``pareto`` may
also come from the side file ``{dir}/pareto/{method}/<same name>.json`` written by
``experiments/run_ga.py`` (``{"folds": [{"fold", "chosen_bits", "pareto"}]}``).
Results tagged ``fbcsp_impl="standin"`` trigger a warning.

Command line::

    python -m src.analysis --results results/baseline_all results/ga --out results/figures
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import chromosome, results

CHANCE = 1.0 / 3.0
GROUP_KEYS = ["method", "subject", "session", "protocol"]
CLASS_NAMES = ["circle", "square", "pentagram"]

# ---------------------------------------------------------------- palette
# Reference data-viz palette (light surface). Colour follows the method, never its rank.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
               "#e34948"]
SEQ_BLUE = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]
METHOD_ORDER = ["ga_nsga", "fbcsp_all", "ga_single", "eegnet", "eegnet_gach", "eegnet_8ch"]
#: Names used by experiments/run_eegnet.py before the contract settles them
#: (CONTRACT_ISSUES "ga" §1). ``eegnet_32ch`` is the contract's ``eegnet`` and
#: shares its colour; ``eegnet_8ch`` gets the next categorical slot.
METHOD_ALIASES = {"eegnet_32ch": "eegnet"}
METHOD_LABELS = {"ga_nsga": "GA-NSGA (proposed)", "fbcsp_all": "FBCSP all", "ga_single":
                 "GA single-obj.", "eegnet": "EEGNet", "eegnet_gach": "EEGNet GA-ch",
                 "eegnet_32ch": "EEGNet (32 ch)", "eegnet_8ch": "EEGNet (8 ch)"}
METHOD_MARKERS = {"ga_nsga": "o", "fbcsp_all": "s", "ga_single": "^", "eegnet": "D",
                  "eegnet_gach": "v", "eegnet_32ch": "D", "eegnet_8ch": "P"}


def _base(method: str) -> str:
    base = method[:-5] if method.endswith("_view") else method
    return METHOD_ALIASES.get(base, base)


def method_color(method: str) -> str:
    """Fixed colour per method (``_view`` variants share their base colour)."""
    base = _base(method)
    if base in METHOD_ORDER:
        return CATEGORICAL[METHOD_ORDER.index(base)]
    return CATEGORICAL[6 + (sum(map(ord, base)) % 2)]  # stable slot for unknown methods


def method_label(method: str) -> str:
    base = method[:-5] if method.endswith("_view") else method
    lab = METHOD_LABELS.get(base, base)
    return lab + (" (viewing)" if method.endswith("_view") else "")


def _sort_methods(methods: Iterable[str]) -> list[str]:
    def key(m):
        base = _base(m)
        return (METHOD_ORDER.index(base) if base in METHOD_ORDER else 99, m)
    return sorted(set(methods), key=key)


def _pareto_side_file(path: Path) -> Path:
    """``{dir}/{method}/X.json`` -> ``{dir}/pareto/{method}/X.json`` (experiments/run_ga.py)."""
    return path.parent.parent / "pareto" / path.parent.name / path.name


def _attach_side_pareto(path: Path, folds: list[dict[str, Any]]) -> None:
    """Attach Pareto fronts from the run_ga side file, matched by fold number."""
    side = _pareto_side_file(path)
    if not side.exists():
        return
    with open(side, encoding="utf-8") as f:
        by_fold = {int(e["fold"]): e.get("pareto") for e in json.load(f).get("folds", [])}
    for f in folds:
        if f.get("pareto") is None and by_fold.get(int(f["fold"])):
            f["pareto"] = by_fold[int(f["fold"])]


# ---------------------------------------------------------------- loading
def load_results(results_dir: str | os.PathLike) -> pd.DataFrame:
    """Load every ``{results_dir}/*/*.json`` into one DataFrame.

    One row per JSON file. Columns: ``method, subject, session, seed,
    protocol, ba_mean, ba_std, train_time_s, infer_time_ms, config_hash,
    mean_cost, mean_n_features, per_fold`` (list of dicts), ``path``.
    Each record is validated with ``results.validate_result``. If several
    ``config_hash`` values are present, warn.

    ``results_dir`` may also be a list of directories, which are concatenated.
    Extra columns: ``n_folds`` and ``synthetic``.
    """
    dirs = [results_dir] if isinstance(results_dir, (str, os.PathLike)) else list(results_dir)
    rows = []
    for d in dirs:
        for p in sorted(Path(d).glob("*/*.json")):
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            try:
                results.validate_result(rec)
            except results.ResultContractError as e:
                raise results.ResultContractError(f"{p}: {e}") from e
            folds = rec["per_fold"]
            costs = [f["cost"] for f in folds if f.get("cost") is not None]
            nfeat = [f["n_features"] for f in folds if f.get("n_features") is not None]
            if "pareto" in rec and all("pareto" not in f for f in folds):
                # top-level per-fold list of Pareto sets -> attach to folds
                for f, par in zip(folds, rec["pareto"]):
                    f["pareto"] = par
            if all("pareto" not in f for f in folds):
                _attach_side_pareto(p, folds)
            impl = rec.get("fbcsp_impl")
            rows.append({
                "method": rec["method"], "subject": rec["subject"], "session": rec["session"],
                "seed": rec["seed"], "protocol": rec["protocol"],
                "ba_mean": float(rec["ba_mean"]), "ba_std": float(rec["ba_std"]),
                "train_time_s": float(rec["train_time_s"]),
                "infer_time_ms": np.nan if rec["infer_time_ms"] is None else float(rec["infer_time_ms"]),
                "config_hash": rec["config_hash"],
                "mean_cost": float(np.mean(costs)) if costs else np.nan,
                "mean_n_features": float(np.mean(nfeat)) if nfeat else np.nan,
                "per_fold": folds, "path": str(p), "n_folds": len(folds),
                "synthetic": bool(rec.get("synthetic", False)),
                "standin": impl == "standin" or (isinstance(impl, list) and "standin" in impl),
            })
    cols = ["method", "subject", "session", "seed", "protocol", "ba_mean", "ba_std",
            "train_time_s", "infer_time_ms", "config_hash", "mean_cost", "mean_n_features",
            "per_fold", "path", "n_folds", "synthetic", "standin"]
    df = pd.DataFrame(rows, columns=cols)
    if df["standin"].any():
        warnings.warn(f"{int(df['standin'].sum())} result file(s) were made with the "
                      "development stand-in FBCSP (fbcsp_impl='standin'); delete them "
                      "before reporting", RuntimeWarning, stacklevel=2)
    if df["config_hash"].nunique() > 1:
        warnings.warn(f"results mix {df['config_hash'].nunique()} config hashes: "
                      f"{sorted(df['config_hash'].unique())}", RuntimeWarning, stacklevel=2)
    if len(df):
        df["seed"] = df["seed"].astype(int)
        if df["synthetic"].any() and not df["synthetic"].all():
            warnings.warn("results mix synthetic and real recordings", RuntimeWarning, stacklevel=2)
    return df


def explode_folds(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (file, fold), with columns ``fold, ba, chosen_bits, cost, n_features`` added.

    The optional fold keys ``y_true``, ``y_pred``, ``pareto`` and ``n_test`` are
    carried along (``None`` when absent).
    """
    rows = []
    keep = [c for c in df.columns if c != "per_fold"]
    for rec in df.to_dict("records"):
        base = {k: rec[k] for k in keep}
        for f in rec["per_fold"]:
            rows.append(base | {"fold": int(f["fold"]), "ba": float(f["ba"]),
                                "chosen_bits": f.get("chosen_bits"), "cost": f.get("cost"),
                                "n_features": f.get("n_features"), "n_test": f.get("n_test"),
                                "y_true": f.get("y_true"), "y_pred": f.get("y_pred"),
                                "pareto": f.get("pareto")})
    return pd.DataFrame(rows, columns=keep + ["fold", "ba", "chosen_bits", "cost", "n_features",
                                              "n_test", "y_true", "y_pred", "pareto"])


_NUMERIC = ["ba_mean", "ba_std", "train_time_s", "infer_time_ms", "mean_cost", "mean_n_features"]


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds per ``(method, subject, session, protocol)``. Adds ``n_seeds``."""
    if not len(df):
        return pd.DataFrame(columns=GROUP_KEYS + _NUMERIC + ["n_seeds"])
    num = [c for c in _NUMERIC if c in df.columns]
    g = df.groupby(GROUP_KEYS, sort=True)
    out = g[num].mean()
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def subject_level(df: pd.DataFrame, protocol: str = "block", metric: str = "ba_mean",
                  sessions: Sequence[str] | None = None) -> pd.DataFrame:
    """Wide table subject × method of ``metric``, after averaging seeds and then sessions.

    ``sessions`` restricts which sessions are averaged (default: all).
    """
    agg = aggregate_seeds(df) if "seed" in df.columns else df
    agg = agg[agg["protocol"] == protocol]
    if sessions is not None:
        agg = agg[agg["session"].isin(list(sessions))]
    return agg.groupby(["subject", "method"])[metric].mean().unstack("method")


# ---------------------------------------------------------------- statistics
def paired_test(df: pd.DataFrame, method_a: str, method_b: str, protocol: str = "block",
                metric: str = "ba_mean", test: str = "wilcoxon") -> dict[str, Any]:
    """Paired comparison over subjects. ``test`` is ``"wilcoxon"`` or ``"ttest"``.

    Seeds are averaged first, then sessions within subject (n = subjects).
    Only subjects that have both methods are used.

    Returns ``{"n", "mean_diff", "statistic", "p", "effect_size"}`` plus
    ``median_diff`` and ``effect_size_name``. The effect size is the
    matched-pairs rank-biserial correlation for Wilcoxon and Cohen's d_z for
    the t-test. ``mean_diff = mean(a - b)``.
    """
    wide = subject_level(df, protocol, metric)
    if method_a not in wide or method_b not in wide:
        raise ValueError(f"methods {method_a!r}/{method_b!r} not both present for {protocol!r}")
    pair = wide[[method_a, method_b]].dropna()
    d = (pair[method_a] - pair[method_b]).to_numpy()
    n = int(d.size)
    out: dict[str, Any] = {"n": n, "mean_diff": float(np.mean(d)) if n else np.nan,
                           "median_diff": float(np.median(d)) if n else np.nan,
                           "statistic": np.nan, "p": np.nan, "effect_size": np.nan,
                           "effect_size_name": "rank_biserial" if test == "wilcoxon" else "cohen_dz"}
    if n < 2:
        return out
    if test == "wilcoxon":
        nz = d[d != 0]
        if nz.size == 0:
            out.update(statistic=0.0, p=1.0, effect_size=0.0)
            return out
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        r = stats.rankdata(np.abs(nz))
        w_plus, w_minus = r[nz > 0].sum(), r[nz < 0].sum()
        out.update(statistic=float(res.statistic), p=float(res.pvalue),
                   effect_size=float((w_plus - w_minus) / (w_plus + w_minus)))
    elif test == "ttest":
        res = stats.ttest_rel(pair[method_a], pair[method_b])
        sd = np.std(d, ddof=1)
        out.update(statistic=float(res.statistic), p=float(res.pvalue),
                   effect_size=float(np.mean(d) / sd) if sd > 0 else np.nan)
    else:
        raise ValueError("test must be 'wilcoxon' or 'ttest'")
    return out


def friedman_test(df: pd.DataFrame, methods: Sequence[str], protocol: str = "block",
                  metric: str = "ba_mean") -> dict[str, Any]:
    """Friedman test across ``methods``. Returns ``{"n", "statistic", "p", "mean_ranks"}``.

    Ranks: 1 = best (highest ``metric``). Subjects missing any method are dropped.
    """
    wide = subject_level(df, protocol, metric)
    missing = [m for m in methods if m not in wide]
    if missing:
        raise ValueError(f"methods missing for protocol {protocol!r}: {missing}")
    w = wide[list(methods)].dropna()
    ranks = pd.DataFrame(stats.rankdata(-w.to_numpy(), axis=1), index=w.index, columns=list(methods))
    out = {"n": int(len(w)), "statistic": np.nan, "p": np.nan,
           "mean_ranks": ranks.mean().to_dict()}
    if len(methods) >= 3 and len(w) >= 2:
        res = stats.friedmanchisquare(*[w[m].to_numpy() for m in methods])
        out.update(statistic=float(res.statistic), p=float(res.pvalue))
    return out


def holm(pvalues: Sequence[float]) -> np.ndarray:
    """Holm-corrected p-values (``statsmodels...multipletests(method="holm")``). NaNs stay NaN."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvalues, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def _ba_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """BA per row: ``y_true`` (P, n) permuted labels, ``y_pred`` (S, n) → (P, S)."""
    classes = np.unique(y_true[0])
    rec = []
    for c in classes:
        m = (y_true == c)                                   # (P, n)
        hit = (y_pred[None, :, :] == c) & m[:, None, :]     # (P, S, n)
        rec.append(hit.sum(-1) / m.sum(-1, keepdims=True))
    return np.mean(rec, axis=0)


def above_chance(df: pd.DataFrame, method: str, protocol: str = "block",
                 chance: float = 1 / 3, n_perm: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-subject test of BA > chance. Returns one row per (subject, session) with ``p``.

    With ``y_true``/``y_pred`` stored per fold, it runs a label-permutation test
    on the fixed predictions. The test labels are permuted within each fold (the
    same permutation for every seed). The statistic is the fold-mean BA
    averaged over seeds (= ``ba_mean`` after :func:`aggregate_seeds`).
    ``p = (1 + #{null >= observed}) / (1 + n_perm)``, one-sided. Without stored
    predictions it falls back to a one-sided binomial test of
    ``round(ba * n_test)`` correct out of ``n_test`` (assumes balanced classes).
    If ``n_test`` is also missing, ``p`` is NaN. Columns: ``subject, session,
    ba, n_seeds, n_trials, null_mean, p, p_holm, test``.
    """
    rng = np.random.default_rng(seed)
    ex = explode_folds(df[(df["method"] == method) & (df["protocol"] == protocol)])
    rows = []
    for (sub, ses), g in ex.groupby(["subject", "session"], sort=True):
        seeds = sorted(g["seed"].unique())
        ba_obs = float(g.groupby("seed")["ba"].mean().mean())
        row = {"subject": sub, "session": ses, "ba": ba_obs, "n_seeds": len(seeds),
               "n_trials": np.nan, "null_mean": np.nan, "p": np.nan, "test": "none"}
        have_pred = g["y_pred"].notna().all() and g["y_true"].notna().all()
        if have_pred:
            null = np.zeros(n_perm)
            n_trials = 0
            folds = sorted(g["fold"].unique())
            for k in folds:
                gk = g[g["fold"] == k].sort_values("seed")
                yt = np.asarray(gk["y_true"].iloc[0])
                if any(not np.array_equal(np.asarray(v), yt) for v in gk["y_true"]):
                    raise ValueError(f"{method} sub-{sub} ses-{ses} fold {k}: y_true differs "
                                     "across seeds (different splits?)")
                yp = np.stack([np.asarray(v) for v in gk["y_pred"]])          # (S, n)
                perm = np.argsort(rng.random((n_perm, yt.size)), axis=1)
                null += _ba_matrix(yt[perm], yp).mean(axis=1)                  # mean over seeds
                n_trials += yt.size
            null /= len(folds)
            row.update(n_trials=n_trials, null_mean=float(null.mean()),
                       p=float((1 + np.sum(null >= ba_obs - 1e-12)) / (1 + n_perm)),
                       test="permutation")
        elif g["n_test"].notna().all():
            n = int(g[g["seed"] == seeds[0]]["n_test"].sum())
            k = int(round(ba_obs * n))
            row.update(n_trials=n, null_mean=chance,
                       p=float(stats.binomtest(k, n, chance, alternative="greater").pvalue),
                       test="binomial_approx")
        rows.append(row)
    out = pd.DataFrame(rows, columns=["subject", "session", "ba", "n_seeds", "n_trials",
                                      "null_mean", "p", "test"])
    out.insert(0, "method", method)
    out.insert(1, "protocol", protocol)
    out["p_holm"] = holm(out["p"].to_numpy()) if len(out) else []
    return out


def confusion(df: pd.DataFrame, method: str, protocol: str = "block",
              normalize: bool = True) -> np.ndarray:
    """Pooled 3×3 confusion matrix (rows = true class) from stored ``y_true``/``y_pred``.

    Each (subject, session) contributes equally: its matrix is averaged over
    seeds and row-normalised before pooling when ``normalize`` is set.
    """
    ex = explode_folds(df[(df["method"] == method) & (df["protocol"] == protocol)])
    ex = ex[ex["y_pred"].notna()]
    if not len(ex):
        raise ValueError(f"no stored predictions for {method!r} / {protocol!r}")
    mats = []
    for _, g in ex.groupby(["subject", "session"]):
        m = np.zeros((3, 3))
        for yt, yp in zip(g["y_true"], g["y_pred"]):
            np.add.at(m, (np.asarray(yt), np.asarray(yp)), 1)
        m /= g["seed"].nunique()
        if normalize:
            m = m / np.maximum(m.sum(1, keepdims=True), 1e-12)
        mats.append(m)
    M = np.mean(mats, axis=0) if normalize else np.sum(mats, axis=0)
    return M


# ---------------------------------------------------------------- gene selection
def gene_table(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """The 19 genes: ``gene, group, name``."""
    if cfg is None:
        from .config import load_config
        cfg = load_config()
    names = (list(cfg["channels"]["candidates"]) + list(cfg["bands"]["names"])
             + [f"{w:.1f}s" for w in cfg["windows"]["starts_s"]])
    groups = (["channel"] * chromosome.N_CH + ["band"] * chromosome.N_BAND
              + ["window"] * chromosome.N_WIN)
    return pd.DataFrame({"gene": np.arange(chromosome.N_BITS), "group": groups, "name": names})


def _bits_matrix(series: pd.Series) -> np.ndarray:
    return np.stack([chromosome.from_str(s) for s in series]) if len(series) else \
        np.zeros((0, chromosome.N_BITS), np.uint8)


def selection_frequency(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Frequency with which each of the 19 genes is chosen, over folds × seeds × subjects.

    Columns: ``gene, group (channel|band|window), name, frequency`` and ``n`` (number of
    chromosomes). Every protocol present in ``df`` is included, so filter first if needed.
    """
    ex = explode_folds(df[df["method"] == method])
    ex = ex[ex["chosen_bits"].notna()]
    B = _bits_matrix(ex["chosen_bits"])
    out = gene_table()
    out["frequency"] = B.mean(axis=0) if len(B) else np.nan
    out["n"] = len(B)
    return out


def selection_matrix(df: pd.DataFrame, method: str, protocol: str = "block") -> pd.DataFrame:
    """Subject × gene selection frequency (over sessions, folds and seeds). Columns = gene names."""
    ex = explode_folds(df[(df["method"] == method) & (df["protocol"] == protocol)])
    ex = ex[ex["chosen_bits"].notna()]
    genes = gene_table()
    rows = {sub: _bits_matrix(g["chosen_bits"]).mean(axis=0) for sub, g in ex.groupby("subject")}
    return pd.DataFrame.from_dict(rows, orient="index", columns=genes["name"].tolist()).sort_index()


def jaccard(a, b) -> float:
    """Jaccard index ``|a ∩ b| / |a ∪ b|`` of two bit vectors (1.0 if both are empty)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    union = np.sum(a | b)
    return 1.0 if union == 0 else float(np.sum(a & b) / union)


def kuncheva(a, b, n: int | None = None) -> float:
    """Kuncheva consistency index of two subsets of ``n`` features (bit vectors).

    For equal sizes ``k`` it is Kuncheva's (2007) ``(r n - k²) / (k (n - k))``,
    where ``r = |a ∩ b|``. For unequal sizes it uses the generalisation of
    Lustgarten et al. (2009), ``(r - k_a k_b / n) / (min(k_a, k_b) - max(0, k_a + k_b - n))``.
    The value is 1 for identical subsets and about 0 for independent random
    subsets. It is NaN when undefined (e.g. both subsets empty or both full).
    """
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = a.size if n is None else n
    ka, kb, r = int(a.sum()), int(b.sum()), int(np.sum(a & b))
    if ka == kb:
        k = ka
        if k in (0, n):
            return np.nan
        return float((r * n - k * k) / (k * (n - k)))
    den = min(ka, kb) - max(0, ka + kb - n)
    return np.nan if den == 0 else float((r - ka * kb / n) / den)


def selection_stability(df: pd.DataFrame, method: str, by: str = "all",
                        groups: str = "all") -> pd.DataFrame:
    """Pairwise stability of ``chosen_bits`` per (subject, session, protocol).

    ``by``: ``"all"`` uses every pair of (fold, seed) chromosomes; ``"seed"``
    uses only pairs from the same fold with different seeds (GA randomness);
    ``"fold"`` uses only pairs from the same seed with different folds (data
    resampling). ``groups``: ``"all"`` (the 19 bits) or one of
    ``"channel"``, ``"band"``, ``"window"`` (that gene group only).
    Columns: ``method, subject, session, protocol, jaccard, kuncheva, n_pairs``.
    """
    ex = explode_folds(df[df["method"] == method])
    ex = ex[ex["chosen_bits"].notna()]
    gt = gene_table()
    mask = np.ones(chromosome.N_BITS, bool) if groups == "all" else (gt["group"] == groups).to_numpy()
    rows = []
    for (sub, ses, prot), g in ex.groupby(["subject", "session", "protocol"], sort=True):
        B = _bits_matrix(g["chosen_bits"])[:, mask]
        folds, seeds = g["fold"].to_numpy(), g["seed"].to_numpy()
        js, ks = [], []
        for i, j in itertools.combinations(range(len(B)), 2):
            if by == "seed" and not (folds[i] == folds[j] and seeds[i] != seeds[j]):
                continue
            if by == "fold" and not (seeds[i] == seeds[j] and folds[i] != folds[j]):
                continue
            js.append(jaccard(B[i], B[j]))
            ks.append(kuncheva(B[i], B[j]))
        rows.append({"method": method, "subject": sub, "session": ses, "protocol": prot,
                     "jaccard": float(np.mean(js)) if js else np.nan,
                     "kuncheva": float(np.nanmean(ks)) if ks and np.isfinite(ks).any() else np.nan,
                     "n_pairs": len(js)})
    return pd.DataFrame(rows, columns=["method", "subject", "session", "protocol", "jaccard",
                                       "kuncheva", "n_pairs"])


def cross_session_drop(df: pd.DataFrame, methods: Sequence[str] | None = None,
                       within_session: str = "02") -> pd.DataFrame:
    """Within-session (block CV) minus cross-session BA, per method and subject.

    ``within_session="02"`` (default) compares against block CV on ses-02,
    the same test data the cross-session model sees. ``"01"`` uses ses-01 and
    ``"mean"`` uses the mean of both sessions. Seeds are averaged first.
    Columns: ``method, subject, within_ba, cross_ba, drop``.
    """
    agg = aggregate_seeds(df)
    if methods is not None:
        agg = agg[agg["method"].isin(list(methods))]
    blk = agg[agg["protocol"] == "block"]
    if within_session != "mean":
        blk = blk[blk["session"] == within_session]
    within = blk.groupby(["method", "subject"])["ba_mean"].mean().rename("within_ba")
    cross = agg[agg["protocol"] == "cross_session"].groupby(["method", "subject"])["ba_mean"] \
        .mean().rename("cross_ba")
    out = pd.concat([within, cross], axis=1, join="inner").reset_index()
    out["drop"] = out["within_ba"] - out["cross_ba"]
    return out[["method", "subject", "within_ba", "cross_ba", "drop"]]


# ---------------------------------------------------------------- plots
def _plt():
    import sys
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules and not os.environ.get("MPLBACKEND"):
        matplotlib.use("Agg")  # files only; set MPLBACKEND for interactive use
    import matplotlib.pyplot as plt
    return plt


def _style_ax(ax, grid_axis: str = "y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_2, labelcolor=TEXT_2, labelsize=9)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)


def _new_ax(ax, figsize):
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    return ax


def _chance_line(ax, chance=CHANCE, horizontal=True):
    kw = dict(color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    if horizontal:
        ax.axhline(chance, **kw)
        ax.annotate("chance", xy=(1.0, chance), xycoords=("axes fraction", "data"),
                    xytext=(4, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=8, color=TEXT_2, annotation_clip=False)
    else:
        ax.axvline(chance, **kw)


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)


def plot_subject_bars(df: pd.DataFrame, protocol: str = "block", methods: Sequence[str] | None = None,
                      ax=None):
    """Per-subject grouped bars of BA (seeds then sessions averaged), with a chance line."""
    wide = subject_level(df, protocol)
    methods = _sort_methods(wide.columns if methods is None else methods)
    methods = [m for m in methods if m in wide]
    ax = _new_ax(ax, (max(6.0, 0.55 * len(wide) * max(1, len(methods)) ** 0.5 + 2), 3.6))
    n = len(methods)
    width = 0.8 / max(n, 1)
    x = np.arange(len(wide))
    for i, m in enumerate(methods):
        ax.bar(x - 0.4 + width * (i + 0.5), wide[m].to_numpy(), width=width * 0.9,
               color=method_color(m), label=method_label(m), zorder=2, edgecolor=SURFACE,
               linewidth=0.5)
    _chance_line(ax)
    ax.set_xticks(x, [f"S{s}" for s in wide.index])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_xlabel("Subject")
    ax.set_title(f"Per-subject balanced accuracy ({protocol.replace('_', '-')})", loc="left",
                 fontsize=11, pad=22 if n > 1 else 6)
    _style_ax(ax)
    if n > 1:
        ax.legend(frameon=False, fontsize=8, ncol=min(n, 5), loc="lower left",
                  bbox_to_anchor=(0, 1.0), labelcolor=TEXT, borderaxespad=0.2)
    return ax


def plot_method_comparison(df: pd.DataFrame, protocol: str = "block", ax=None):
    """Per-method BA distribution (box/strip + paired lines) with a chance line."""
    wide = subject_level(df, protocol)
    methods = _sort_methods(wide.columns)
    wide = wide[methods]
    ax = _new_ax(ax, (1.3 * len(methods) + 2.5, 3.8))
    pos = np.arange(len(methods))
    for _, row in wide.iterrows():  # paired lines per subject
        ax.plot(pos, row.to_numpy(), color=GRID, linewidth=1.0, zorder=1)
    bp = ax.boxplot([wide[m].dropna().to_numpy() for m in methods], positions=pos, widths=0.45,
                    showfliers=False, patch_artist=True, medianprops=dict(color=TEXT, linewidth=1.5),
                    whiskerprops=dict(color=TEXT_2), capprops=dict(color=TEXT_2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(method_color(m))
        patch.set_alpha(0.35)
        patch.set_edgecolor(method_color(m))
    rng = np.random.default_rng(0)
    for i, m in enumerate(methods):
        v = wide[m].dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.08, 0.08, v.size), v, s=22, color=method_color(m),
                   edgecolor=SURFACE, linewidth=1.0, zorder=3, marker=METHOD_MARKERS.get(m, "o"))
    _chance_line(ax)
    ax.set_xticks(pos, [method_label(m) for m in methods], rotation=15, ha="right")
    ax.set_ylabel("Balanced accuracy (subject mean)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Methods compared ({protocol.replace('_', '-')}, n = {len(wide)} subjects)",
                 loc="left", fontsize=11)
    _style_ax(ax)
    return ax


def plot_confusion(df: pd.DataFrame, method: str, protocol: str = "block", ax=None):
    """Row-normalised confusion matrix pooled over subjects (each subject weighted equally)."""
    M = confusion(df, method, protocol, normalize=True)
    ax = _new_ax(ax, (4.0, 3.6))
    im = ax.imshow(M, cmap=_seq_cmap(), vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10,
                    color=SURFACE if M[i, j] > 0.55 else TEXT)
    ax.set_xticks(range(3), CLASS_NAMES)
    ax.set_yticks(range(3), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion: {method_label(method)}", loc="left", fontsize=11)
    _style_ax(ax, grid_axis="")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8, colors=TEXT_2)
    cb.set_label("Row fraction", color=TEXT_2, fontsize=8)
    cb.outline.set_visible(False)
    return ax


def pareto_front(points: np.ndarray) -> np.ndarray:
    """Indices of non-dominated points for (cost ↓, accuracy ↑), sorted by cost."""
    pts = np.asarray(points, float)
    keep = []
    for i, (c, a) in enumerate(pts):
        dominated = np.any((pts[:, 0] <= c) & (pts[:, 1] >= a) & ((pts[:, 0] < c) | (pts[:, 1] > a)))
        if not dominated:
            keep.append(i)
    keep = np.array(keep, dtype=int)
    return keep[np.argsort(pts[keep, 0], kind="stable")] if keep.size else keep


def plot_pareto(pareto: list[dict[str, Any]], chosen_bits=None, ax=None):
    """Cost vs. ba_mean scatter of a Pareto front (colour = ba_std), with the chosen point highlighted."""
    ax = _new_ax(ax, (4.8, 3.6))
    cost = np.array([p["cost"] for p in pareto], float)
    ba = np.array([p["ba_mean"] for p in pareto], float)
    sd = np.array([p["ba_std"] for p in pareto], float)
    sc = ax.scatter(cost, ba, c=sd, cmap=_seq_cmap(), vmin=0, s=40, edgecolor=TEXT_2,
                    linewidth=0.6, zorder=3)
    order = np.argsort(cost)
    ax.plot(cost[order], ba[order], color=GRID, linewidth=1.0, zorder=2)
    if chosen_bits is not None:
        cb = chosen_bits if isinstance(chosen_bits, str) else chromosome.to_str(chosen_bits)
        idx = [i for i, p in enumerate(pareto)
               if (p["bits"] if isinstance(p["bits"], str) else chromosome.to_str(p["bits"])) == cb]
        if idx:
            i = idx[0]
            ax.scatter([cost[i]], [ba[i]], s=140, facecolor="none", edgecolor=CATEGORICAL[1],
                       linewidth=2, zorder=4)
            ax.annotate("chosen", (cost[i], ba[i]), xytext=(6, -12), textcoords="offset points",
                        fontsize=8, color=TEXT)
    _chance_line(ax)
    ax.set_xlabel("Cost (fraction of components)")
    ax.set_ylabel("Inner-CV balanced accuracy")
    ax.set_title("Pareto front", loc="left", fontsize=11)
    _style_ax(ax, grid_axis="both")
    cbar = ax.figure.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Inner-CV SD", color=TEXT_2, fontsize=8)
    cbar.ax.tick_params(labelsize=8, colors=TEXT_2)
    cbar.outline.set_visible(False)
    return ax


def plot_cost_vs_accuracy(df: pd.DataFrame, protocol: str = "block", ax=None):
    """Mean cost vs. ba_mean per method and subject.

    Small markers are subjects (seeds and sessions averaged); large markers are
    method means. Grey points are the GA's inner-CV Pareto sets (pooled), with
    their non-dominated front. Methods without a chromosome cost (EEGNet) are
    drawn in a separate "n/a" column at the right.
    """
    agg = aggregate_seeds(df)
    agg = agg[agg["protocol"] == protocol]
    subj = agg.groupby(["method", "subject"])[["mean_cost", "ba_mean"]].mean().reset_index()
    methods = _sort_methods(subj["method"])
    ax = _new_ax(ax, (8.4, 4.2))
    NA_X = 1.12
    # pooled inner-CV Pareto sets
    ex = explode_folds(df[df["protocol"] == protocol])
    par = [p for lst in ex["pareto"].dropna() for p in lst]
    if par:
        pts = np.array([[p["cost"], p["ba_mean"]] for p in par], float)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color="#c3c2b7", alpha=0.5, zorder=1,
                   label="GA Pareto sets (inner CV)", linewidth=0)
        f = pareto_front(pts)
        ax.plot(pts[f, 0], pts[f, 1], color=TEXT_2, linewidth=1.2, zorder=2,
                drawstyle="steps-post", label="Pooled Pareto front")
    for m in methods:
        g = subj[subj["method"] == m]
        x = g["mean_cost"].to_numpy(float)
        x = np.where(np.isfinite(x), x, NA_X + np.random.default_rng(0).uniform(-0.015, 0.015, x.size))
        mk = METHOD_MARKERS.get(m, "o")
        ax.scatter(x, g["ba_mean"], s=18, color=method_color(m), alpha=0.55, marker=mk,
                   linewidth=0, zorder=3)
        mx = np.nanmean(g["mean_cost"]) if np.isfinite(g["mean_cost"]).any() else NA_X
        ax.scatter([mx], [g["ba_mean"].mean()], s=110, color=method_color(m), marker=mk,
                   edgecolor=SURFACE, linewidth=2, zorder=4, label=method_label(m))
    ax.axvline(1.06, color=GRID, linewidth=1.0)
    _chance_line(ax)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, NA_X], ["0", "0.2", "0.4", "0.6", "0.8", "1", "n/a"])
    ax.set_xlabel("Cost (fraction of channels, bands, windows)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"Accuracy vs. cost ({protocol.replace('_', '-')})", loc="left", fontsize=11)
    _style_ax(ax, grid_axis="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              labelcolor=TEXT)
    return ax


def plot_selection_frequency(freq_df: pd.DataFrame, ax=None):
    """Bar plot of gene selection frequency grouped by channel, band and window."""
    ax = _new_ax(ax, (7.5, 3.2))
    f = freq_df.sort_values("gene")
    x = np.arange(len(f)) + np.searchsorted([chromosome.N_CH, chromosome.N_CH + chromosome.N_BAND],
                                            f["gene"].to_numpy(), side="right") * 0.8
    ax.bar(x, f["frequency"], width=0.75, color=CATEGORICAL[0], zorder=2)
    ax.set_xticks(x, f["name"], rotation=60, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Selection frequency")
    for grp in ("channel", "band", "window"):
        xs = x[(f["group"] == grp).to_numpy()]
        ax.annotate(grp + "s", xy=(xs.mean(), 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_title("Gene selection frequency", loc="left", fontsize=11, pad=16)
    _style_ax(ax)
    return ax


def plot_selection_heatmap(df: pd.DataFrame, method: str, protocol: str = "block", ax=None):
    """Heatmap subject × gene of how often each gene is selected (channels | bands | windows)."""
    M = selection_matrix(df, method, protocol)
    ax = _new_ax(ax, (8.0, 0.32 * max(len(M), 4) + 1.8))
    im = ax.imshow(M.to_numpy(float), cmap=_seq_cmap(), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=60, fontsize=8)
    ax.set_yticks(range(len(M)), [f"S{s}" for s in M.index], fontsize=8)
    for b in (chromosome.N_CH - 0.5, chromosome.N_CH + chromosome.N_BAND - 0.5):
        ax.axvline(b, color=SURFACE, linewidth=3)
    for lab, (a, b) in (("channels", (0, chromosome.N_CH)),
                        ("bands", (chromosome.N_CH, chromosome.N_CH + chromosome.N_BAND)),
                        ("windows", (chromosome.N_CH + chromosome.N_BAND, chromosome.N_BITS))):
        ax.annotate(lab, xy=((a + b - 1) / 2, 1.0), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=9, color=TEXT_2)
    ax.set_title(f"Selection frequency per subject: {method_label(method)}", loc="left",
                 fontsize=11, pad=16)
    _style_ax(ax, grid_axis="")
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Fraction of folds × seeds", color=TEXT_2, fontsize=8)
    cb.ax.tick_params(labelsize=8, colors=TEXT_2)
    cb.outline.set_visible(False)
    return ax


def plot_stability(stab_df: pd.DataFrame, metric: str = "kuncheva", ax=None):
    """Box plot of selection stability (``metric`` from :func:`selection_stability`) per method."""
    methods = _sort_methods(stab_df["method"])
    ax = _new_ax(ax, (1.3 * len(methods) + 2.5, 3.6))
    data = [stab_df.loc[stab_df["method"] == m, metric].dropna().to_numpy() for m in methods]
    bp = ax.boxplot(data, positions=range(len(methods)), widths=0.45, showfliers=False,
                    patch_artist=True, medianprops=dict(color=TEXT, linewidth=1.5),
                    whiskerprops=dict(color=TEXT_2), capprops=dict(color=TEXT_2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(method_color(m))
        patch.set_alpha(0.35)
        patch.set_edgecolor(method_color(m))
    rng = np.random.default_rng(0)
    for i, (m, v) in enumerate(zip(methods, data)):
        ax.scatter(i + rng.uniform(-0.08, 0.08, v.size), v, s=18, color=method_color(m),
                   edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.axhline(0, color=TEXT_2, linewidth=0.8, linestyle=(0, (4, 3)))
    ax.set_xticks(range(len(methods)), [method_label(m) for m in methods], rotation=15, ha="right")
    ax.set_ylabel({"kuncheva": "Kuncheva index", "jaccard": "Jaccard index"}.get(metric, metric))
    ax.set_ylim(-0.2 if metric == "kuncheva" else 0, 1.05)
    ax.set_title("Stability of chosen genes across folds × seeds", loc="left", fontsize=11)
    _style_ax(ax)
    return ax


def make_figures(results_dir, out_dir: str | os.PathLike, protocol: str = "block",
                 dpi: int = 150) -> list[Path]:
    """Build every figure that the available results support, and save PNGs into ``out_dir``."""
    plt = _plt()
    df = load_results(results_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save(ax, name):
        fig = ax.figure
        fig.set_facecolor(SURFACE)
        fig.tight_layout()
        p = out / name
        fig.savefig(p, dpi=dpi, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        saved.append(p)

    if not len(df):
        warnings.warn(f"no results found in {results_dir}", RuntimeWarning, stacklevel=2)
        return saved
    for prot in sorted(df["protocol"].unique()):
        save(plot_subject_bars(df, prot), f"subject_bars_{prot}.png")
    if df[df["protocol"] == protocol]["method"].nunique() >= 1:
        save(plot_method_comparison(df, protocol), f"method_comparison_{protocol}.png")
        save(plot_cost_vs_accuracy(df, protocol), f"cost_vs_accuracy_{protocol}.png")
    ex = explode_folds(df[df["protocol"] == protocol])
    for m in _sort_methods(ex.loc[ex["y_pred"].notna(), "method"]):
        save(plot_confusion(df, m, protocol), f"confusion_{m}_{protocol}.png")
    stab = []
    for m in _sort_methods(ex.loc[ex["chosen_bits"].notna(), "method"]):
        bits = ex.loc[(ex["method"] == m) & ex["chosen_bits"].notna(), "chosen_bits"]
        if bits.nunique() > 1:  # fixed chromosomes (fbcsp_all) carry no selection information
            save(plot_selection_heatmap(df, m, protocol), f"selection_heatmap_{m}.png")
            save(plot_selection_frequency(selection_frequency(df[df["protocol"] == protocol], m)),
                 f"selection_frequency_{m}.png")
            stab.append(selection_stability(df[df["protocol"] == protocol], m))
    if stab:
        save(plot_stability(pd.concat(stab, ignore_index=True)), "stability_boxplot.png")
    return saved


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Build analysis figures from result JSONs.")
    p.add_argument("--results", nargs="+", required=True, help="one or more results dirs")
    p.add_argument("--out", default="results/figures")
    p.add_argument("--protocol", default="block")
    a = p.parse_args(argv)
    for path in make_figures(a.results, a.out, a.protocol):
        print(path)


if __name__ == "__main__":
    main()

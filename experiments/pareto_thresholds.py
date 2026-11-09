"""Offline comparison of NSGA-II selection thresholds (CONTRACT_ISSUES.md R01 #11).

§g chooses the lowest-cost Pareto solution with
``ba_mean >= b*.ba_mean - b*.ba_std``. With 3 small inner folds the SD can be
large, so the rule may pick chromosomes that are too small. This script re-applies
alternative widths to the **saved** per-fold Pareto fronts of ``ga_nsga``
(no new GA search), refits FBCSP+LDA once per (fold, distinct choice) on the
outer training set and scores the outer test fold:

* ``k·SD`` for ``k`` in ``--ks`` (default 0, 0.5, 1, 2; ``k=0`` = best inner BA,
  ``k=1`` is the contract rule and must reproduce the stored ``chosen_bits``);
* ``SE`` = ``SD / sqrt(n_inner)``;
* ``knee``: knee of the 2-D (cost, inner BA) front (:func:`choose_knee`);
* ``mincost``: the cheapest entry of the front.

Exploratory (P5 §3): picking a rule by its outer score would be selection on the
test set. ``compare`` adds selection stability and paired tests against
``fbcsp_all`` and the contract rule.

It also reports the **inner-vs-outer gap** (inner-CV ``ba_mean`` of the chosen
chromosome minus outer BA) for ``ga_nsga`` and ``ga_single``, as evidence for
or against the stability claim.

Outputs (no ``.json``, so ``analysis.load_results`` never reads them):

* ``{out}/pareto_thresholds_folds.csv``: one row per (recording, seed, fold, rule);
* ``{out}/pareto_thresholds_summary.csv`` and ``.md``: subject-level means
  (seeds averaged, then sessions averaged within subject, like ``analysis``).

Usage::

    python -m experiments.pareto_thresholds        # results/ -> results/p5/chooser_rules
    python -m experiments.pareto_thresholds --results-dir R --cache-dir C --protocol block
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import analysis, chromosome, fbcsp
from src.config import load_config

from . import _common as C
from .common import resolve_cache_dir


def choose_with_width(pareto: list[dict[str, Any]], width_of_best) -> np.ndarray:
    """§g rule with a custom width: candidates have ``ba_mean >= b*.ba_mean - width(b*)``.

    Same ``b*`` and tie-breaks as ``ga.choose_from_pareto`` (``width = ba_std``
    reproduces it exactly).
    """
    best = min(pareto, key=lambda e: (-e["ba_mean"], e["ba_std"], e["cost"], e["bits"]))
    thr = best["ba_mean"] - width_of_best(best)
    cands = [e for e in pareto if e["ba_mean"] >= thr - 1e-12]
    pick = min(cands, key=lambda e: (e["cost"], -e["ba_mean"], e["ba_std"], e["bits"]))
    return chromosome.from_str(pick["bits"])


def choose_min_cost(pareto: list[dict[str, Any]]) -> np.ndarray:
    """Cheapest entry of the front (ties: higher ``ba_mean``, lower ``ba_std``, bits)."""
    pick = min(pareto, key=lambda e: (e["cost"], -e["ba_mean"], e["ba_std"], e["bits"]))
    return chromosome.from_str(pick["bits"])


def choose_knee(pareto: list[dict[str, Any]]) -> np.ndarray:
    """Knee of the 2-D (cost, inner ``ba_mean``) front.

    Take the entries that are non-dominated in (lower cost, higher ``ba_mean``),
    scale both axes to [0, 1] over them, and pick the entry with the largest
    distance above the chord from the cheapest entry to the most accurate one.
    With fewer than 3 such entries there is no knee and the most accurate entry
    (``b*``) is returned. Ties: lower cost, then bits.
    """
    pts = sorted(pareto, key=lambda e: (e["cost"], -e["ba_mean"], e["bits"]))
    front, best_ba = [], -np.inf
    for e in pts:                       # sweep by cost: keep strict improvements in ba_mean
        if e["ba_mean"] > best_ba + 1e-12:
            front.append(e)
            best_ba = e["ba_mean"]
    if len(front) < 3:
        return chromosome.from_str(front[-1]["bits"])
    c = np.array([e["cost"] for e in front])
    b = np.array([e["ba_mean"] for e in front])
    cn = (c - c[0]) / (c[-1] - c[0]) if c[-1] > c[0] else np.zeros_like(c)
    bn = (b - b[0]) / (b[-1] - b[0]) if b[-1] > b[0] else np.zeros_like(b)
    dist = bn - cn                      # signed vertical distance above the chord y = x
    i = int(np.flatnonzero(dist >= dist.max() - 1e-12)[0])
    return chromosome.from_str(front[i]["bits"])


def rules(ks: list[float], n_inner: int) -> dict[str, Any]:
    """Chooser rules by name -> function(pareto) -> bits. ``0sd`` = best inner BA."""
    out = {f"{k:g}sd": (lambda k: lambda par: choose_with_width(par, lambda e: k * e["ba_std"]))(k)
           for k in ks}
    out["se"] = lambda par: choose_with_width(par, lambda e: e["ba_std"] / math.sqrt(n_inner))
    out["knee"] = choose_knee
    out["mincost"] = choose_min_cost
    return out


def _inner_of(pareto, bits_str):
    for e in pareto:
        if e["bits"] == bits_str:
            return e["ba_mean"]
    return float("nan")


def run(results_dir: Path, cache_dir: Path, protocol: str, ks: list[float],
        view: bool = False) -> pd.DataFrame:
    """Fold-level table. ``view=True`` uses ``ga_*_view`` results and the viewing cache."""
    cfg = load_config()
    sfx, kind = ("_view", "view") if view else ("", "fbcsp")
    nsga, single = "ga_nsga" + sfx, "ga_single" + sfx
    n_inner = int(cfg["ga"]["inner_folds"])
    rule_fns = rules(ks, n_inner)
    df = analysis.load_results(results_dir)
    df = df[df["method"].isin([nsga, single]) & (df["protocol"] == protocol)]
    if df.empty:
        raise SystemExit(f"no {nsga}/{single} {protocol} results in {results_dir}")
    rows = []
    for (subject, session), g in df.groupby(["subject", "session"]):
        d, splits = C.load_unit(kind, subject, session, cfg, cache_dir)
        X, y = d["X"], d["y"]          # one X object per recording -> fbcsp cache hits
        memo: dict[tuple[int, str], float] = {}

        def outer_ba(fold: int, bits) -> float:
            key = (fold, chromosome.to_str(bits))
            if key not in memo:
                memo[key] = fbcsp.evaluate(X, y, bits, [splits[fold]])["per_fold"][0]
            return memo[key]

        for _, r in g.iterrows():
            for f in r["per_fold"]:
                k, par = f["fold"], f.get("pareto") or []
                base = dict(method=r["method"], subject=subject, session=session,
                            seed=int(r["seed"]), fold=k)
                # stored choice (contract rule for ga_nsga, best mean for ga_single)
                rows.append({**base, "rule": "stored", "bits": f["chosen_bits"],
                             "cost": f["cost"], "inner_ba": _inner_of(par, f["chosen_bits"]),
                             "outer_ba": f["ba"]})
                if r["method"] != nsga or not par:
                    continue
                for name, rule in rule_fns.items():
                    bits = rule(par)
                    s = chromosome.to_str(bits)
                    if name == "1sd" and s != f["chosen_bits"]:
                        raise RuntimeError(f"1·SD rule does not reproduce stored choice for "
                                           f"sub-{subject} ses-{session} seed {r['seed']} fold {k}")
                    rows.append({**base, "rule": name, "bits": s, "cost": chromosome.cost(bits),
                                 "inner_ba": _inner_of(par, s), "outer_ba": outer_ba(k, bits)})
    out = pd.DataFrame(rows)
    out["gap"] = out["inner_ba"] - out["outer_ba"]
    out["n_ch"] = [int(chromosome.counts(chromosome.from_str(b))[0]) for b in out["bits"]]
    return out


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    """Subject-level means: folds -> per seed -> mean over seeds -> mean over sessions."""
    keys = ["method", "rule", "subject", "session", "seed"]
    per_run = folds.groupby(keys)[["outer_ba", "inner_ba", "gap", "cost", "n_ch"]].mean()
    per_rec = per_run.groupby(["method", "rule", "subject", "session"]).mean()
    per_sub = per_rec.groupby(["method", "rule", "subject"]).mean()
    agg = per_sub.groupby(["method", "rule"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_subjects"] = per_sub.groupby(["method", "rule"]).size()
    return agg.reset_index()


def compare(folds: pd.DataFrame, results_dir: Path, protocol: str) -> pd.DataFrame:
    """Per rule: subject-level outer BA, cost, selection stability and paired tests.

    Rows: every ``ga_nsga`` rule plus ``ga_single``'s stored choice. Tests are
    two-sided Wilcoxon over subjects (seeds, then sessions averaged), against
    ``fbcsp_all`` and against the contract rule (``ga_nsga`` / ``1sd``).
    """
    from scipy import stats
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = analysis.load_results(Path(results_dir) / "baseline_all")
    fb = analysis.subject_level(base, protocol)["fbcsp_all"]
    keys = ["method", "rule", "subject", "session", "seed"]
    per_sub = (folds.groupby(keys)[["outer_ba", "inner_ba", "gap", "cost", "n_ch"]].mean()
               .groupby(["method", "rule", "subject", "session"]).mean()
               .groupby(["method", "rule", "subject"]).mean())
    have = set(per_sub.index.droplevel(2))
    ref = per_sub.loc[("ga_nsga", "1sd"), "outer_ba"] if ("ga_nsga", "1sd") in have else None
    rows = []
    for (m, rule), g in per_sub.groupby(level=["method", "rule"]):
        if m == "ga_nsga" and rule == "stored":
            continue                       # identical to 1sd (asserted in run)
        g = g.droplevel([0, 1])
        f = folds[(folds.method == m) & (folds.rule == rule)]
        # selection stability of this rule's chromosomes: all (fold, seed) pairs per recording
        recs = [{"method": "rule", "subject": s_, "session": ses, "seed": sd, "protocol": protocol,
                 "per_fold": [{"fold": int(r.fold), "ba": float(r.outer_ba), "chosen_bits": r.bits}
                              for r in gg.itertuples()]}
                for (s_, ses, sd), gg in f.groupby(["subject", "session", "seed"])]
        stab = analysis.selection_stability(pd.DataFrame(recs), "rule", by="all")
        d_fb = (g.outer_ba - fb.reindex(g.index)).dropna()
        row = {"method": m, "rule": rule, "n_subjects": int(len(g)),
               "outer_ba_mean": g.outer_ba.mean(), "outer_ba_sd": g.outer_ba.std(ddof=1),
               "inner_ba_mean": g.inner_ba.mean(), "gap_mean": g.gap.mean(),
               "cost_mean": g.cost.mean(), "n_ch_mean": g.n_ch.mean(),
               "kuncheva": stab.kuncheva.mean(), "jaccard": stab.jaccard.mean(),
               "vs_fbcsp_diff": d_fb.mean(), "vs_fbcsp_p": float(stats.wilcoxon(d_fb).pvalue),
               "vs_fbcsp_wins": int((d_fb > 0).sum())}
        if ref is not None and not (m == "ga_nsga" and rule == "1sd"):
            d = (g.outer_ba - ref.reindex(g.index)).dropna()
            row["vs_1sd_diff"] = d.mean()
            row["vs_1sd_p"] = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(2, "fbcsp_all_mean", fb.mean())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default=None, help="default: {results_dir}/p5/chooser_rules")
    p.add_argument("--protocol", nargs="+", choices=["block", "cross_session"],
                   default=["block", "cross_session"])
    p.add_argument("--ks", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--view", action="store_true", help="use ga_*_view results + viewing cache")
    p.add_argument("--no-compare", action="store_true",
                   help="skip the fbcsp_all comparison (needs {results_dir}/baseline_all)")
    a = p.parse_args(argv)
    cfg = load_config()
    results_dir = Path(a.results_dir) if a.results_dir else Path(cfg["paths"]["results_dir"])
    out = Path(a.out) if a.out else results_dir / "p5" / "chooser_rules"
    out.mkdir(parents=True, exist_ok=True)
    for protocol in a.protocol:
        folds = run(results_dir, resolve_cache_dir(a.cache_dir, cfg), protocol, a.ks, a.view)
        summ = summarise(folds)
        tag = ("" if protocol == "block" else f"_{protocol}") + ("_view" if a.view else "")
        folds.to_csv(out / f"pareto_thresholds_folds{tag}.csv", index=False)
        summ.to_csv(out / f"pareto_thresholds_summary{tag}.csv", index=False)
        cols = ["method", "rule", "n_subjects", "outer_ba_mean", "outer_ba_std", "inner_ba_mean",
                "gap_mean", "cost_mean", "n_ch_mean"]
        text = summ[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        if not a.no_compare:
            comp = compare(folds, results_dir, protocol)
            comp.to_csv(out / f"chooser_rules{tag}.csv", index=False)
            text = comp.to_string(index=False, float_format=lambda v: f"{v:.3f}")
        (out / f"pareto_thresholds_summary{tag}.md").write_text(
            f"# NSGA-II chooser rules ({protocol}{', viewing period' if a.view else ''})\n\n"
            "Exploratory (CONTRACTS k12, ANALYSIS_PLAN §5). Subject-level means (seeds, then "
            "sessions). `0sd` = best inner BA; `1sd` = contract rule (= stored ga_nsga); "
            "gap = inner-CV BA - outer BA. Choosing a rule by its outer score would be "
            "selection on the test set.\n\n```\n" + text + "\n```\n", encoding="utf-8")
        print(f"== {protocol}\n{text}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

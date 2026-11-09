"""Random-vs-block inflation and viewing vs imagery (P5 §4; exploratory).

Inputs: block results (``results/{method}``, ``results/baseline_all``), random
protocol (``results/p5/drift`` for the GA and EEGNet, ``results/sanity`` for
``fbcsp_all``), and the viewing period (``results/p5/drift/ga_*_view``,
``results/sanity/fbcsp_all_view``). All methods use split / GA seeds 0-2 so that
the comparison is like for like (P2 used seeds 0-4 for ``fbcsp_all``).

Usage::

    python -m reports.p5_drift --results results --out results/p5/drift
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import analysis as A

METHODS = ("fbcsp_all", "ga_nsga", "ga_single", "eegnet_32ch", "eegnet_8ch")
VIEW = ("fbcsp_all", "ga_nsga", "ga_single")


def _paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
    return {"n": int(d.size), "mean_a": float(a.reindex(d.index).mean()),
            "mean_b": float(b.reindex(d.index).mean()), "diff": float(d.mean()),
            "median": float(d.median()), "p": p, "wins": int((d > 0).sum())}


def load(results: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = A.load_results([results, results / "baseline_all", results / "sanity",
                             results / "p5" / "drift"])
    return df[df.seed.isin([0, 1, 2])]


def summarise(df: pd.DataFrame, n_perm: int = 1000) -> dict:
    out: dict = {"inflation": {}, "view_vs_imagery": {}, "view_methods": {}, "above_chance": {}}
    blk, rnd = A.subject_level(df, "block"), A.subject_level(df, "random")
    for m in METHODS:
        if m in blk and m in rnd:
            out["inflation"][m] = _paired(rnd[m], blk[m])
    for m in VIEW:
        v = f"{m}_view"
        if v in blk:
            out["view_vs_imagery"][m] = _paired(blk[v], blk[m])
            ex = A.explode_folds(df[(df.method == v) & (df.protocol == "block")])
            cost = ex.cost.dropna().astype(float)
            out["view_methods"][v] = {"mean": float(blk[v].mean()), "sd": float(blk[v].std(ddof=1)),
                                      "cost": float(cost.mean()) if len(cost) else 1.0}
            ac = A.above_chance(df, v, "block", n_perm=n_perm, seed=0)
            out["above_chance"][v] = {"p05": int((ac.p < 0.05).sum()),
                                      "p05_holm": int((ac.p_holm < 0.05).sum()), "n": int(len(ac))}
    if {"ga_nsga_view", "fbcsp_all_view"} <= set(blk):
        out["view_ga_nsga_vs_fbcsp"] = _paired(blk["ga_nsga_view"], blk["fbcsp_all_view"])
        out["view_ga_single_vs_fbcsp"] = _paired(blk["ga_single_view"], blk["fbcsp_all_view"])
        out["view_nsga_vs_single"] = _paired(blk["ga_nsga_view"], blk["ga_single_view"])
    return out


def to_markdown(s: dict) -> str:
    L = ["# Random-vs-block inflation (subject level, seeds 0-2)", "",
         "| Method | n | Block | Random | Random − block (median) | p | Subjects random > block |",
         "|---|---|---|---|---|---|---|"]
    for m, v in s["inflation"].items():
        L.append(f"| {m} | {v['n']} | {v['mean_b']:.3f} | {v['mean_a']:.3f} | {v['diff']:+.3f} "
                 f"({v['median']:+.3f}) | {v['p']:.3f} | {v['wins']} |")
    L += ["", "# Viewing period (3-7 s) vs imagery, block protocol", "",
          "| Method | n | Imagery | Viewing | Viewing − imagery | p | Viewing recordings p<.05 (uncorr./Holm) | Viewing cost |",
          "|---|---|---|---|---|---|---|---|"]
    for m, v in s["view_vs_imagery"].items():
        ac, vm = s["above_chance"][f"{m}_view"], s["view_methods"][f"{m}_view"]
        L.append(f"| {m} | {v['n']} | {v['mean_b']:.3f} | {v['mean_a']:.3f} | {v['diff']:+.3f} | "
                 f"{v['p']:.3g} | {ac['p05']}/{ac['p05_holm']} of {ac['n']} | {vm['cost']:.2f} |")
    for k in ("view_ga_nsga_vs_fbcsp", "view_ga_single_vs_fbcsp", "view_nsga_vs_single"):
        if k in s:
            v = s[k]
            L.append(f"\n{k}: Δ {v['diff']:+.3f} (median {v['median']:+.3f}), p = {v['p']:.3f}, "
                     f"{v['wins']}/{v['n']} subjects")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/p5/drift")
    p.add_argument("--n-perm", type=int, default=1000)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = summarise(load(Path(a.results)), a.n_perm)
    (out / "summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    md = to_markdown(s)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

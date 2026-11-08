"""Main results table (Markdown + CSV) from result JSONs.

One row per method. Subject-level statistics (seeds averaged, then sessions)::

    n | block BA mean ± SD | median | recordings > chance (Holm) | cross-session BA |
    drop (ses-02 block − cross) | cost | n_features | train s | infer ms |
    Δ vs reference | Wilcoxon p (Holm) | r_rb

Usage::

    python -m reports.make_tables --results results/baseline_all results/ga results/eegnet \
        --out results/tables [--reference ga_nsga]

Writes ``main_results.md``, ``main_results.csv`` and ``per_subject.csv``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src import analysis as A


def _pm(m, s, fmt=".3f"):
    return "–" if not np.isfinite(m) else (f"{m:{fmt}} ± {s:.3f}" if np.isfinite(s) else f"{m:{fmt}}")


def _fmt_p(p):
    if not np.isfinite(p):
        return "–"
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def main_table(df: pd.DataFrame, reference: str = "ga_nsga", protocol: str = "block",
               n_perm: int = 1000, alpha: float = 0.05) -> pd.DataFrame:
    """Numeric main table, one row per method."""
    agg = A.aggregate_seeds(df)
    methods = A._sort_methods(agg.loc[agg["protocol"] == protocol, "method"])
    wide = A.subject_level(df, protocol)
    cross_wide = A.subject_level(df, "cross_session")
    drop = A.cross_session_drop(df)
    rows = []
    for m in methods:
        v = wide[m].dropna()
        a = agg[(agg["method"] == m) & (agg["protocol"] == protocol)]
        ab = A.above_chance(df, m, protocol, n_perm=n_perm)
        cr = cross_wide[m].dropna() if m in cross_wide else pd.Series(dtype=float)
        dr = drop.loc[drop["method"] == m, "drop"]
        row = {
            "method": m, "label": A.method_label(m), "n_subjects": int(v.size),
            "ba_mean": v.mean(), "ba_sd": v.std(ddof=1) if v.size > 1 else np.nan,
            "ba_median": v.median(),
            "n_above_chance": int((ab["p_holm"] < alpha).sum()), "n_recordings": int(len(ab)),
            "cross_ba_mean": cr.mean() if cr.size else np.nan,
            "cross_ba_sd": cr.std(ddof=1) if cr.size > 1 else np.nan,
            "drop_mean": dr.mean() if dr.size else np.nan,
            "drop_sd": dr.std(ddof=1) if dr.size > 1 else np.nan,
            "mean_cost": a["mean_cost"].mean(), "mean_n_features": a["mean_n_features"].mean(),
            "train_time_s": a["train_time_s"].mean(), "infer_time_ms": a["infer_time_ms"].mean(),
            "delta_vs_ref": np.nan, "p_vs_ref": np.nan, "effect_vs_ref": np.nan,
        }
        if m != reference and reference in wide:
            t = A.paired_test(df, reference, m, protocol)
            row.update(delta_vs_ref=t["mean_diff"], p_vs_ref=t["p"], effect_vs_ref=t["effect_size"])
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab["p_vs_ref_holm"] = A.holm(tab["p_vs_ref"].to_numpy()) if len(tab) else []
    return tab


def to_markdown(tab: pd.DataFrame, reference: str, friedman: dict | None = None,
                synthetic: bool = False) -> str:
    head = ["Method", "n", "Block BA", "Median", "> chance", "Cross-session BA", "Drop",
            "Cost", "Features", "Train s", "Infer ms", f"Δ vs {reference}", "p (Holm)", "r_rb"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in tab.to_dict("records"):
        cells = [r["label"], str(r["n_subjects"]), _pm(r["ba_mean"], r["ba_sd"]),
                 f"{r['ba_median']:.3f}", f"{r['n_above_chance']}/{r['n_recordings']}",
                 _pm(r["cross_ba_mean"], r["cross_ba_sd"]), _pm(r["drop_mean"], r["drop_sd"], "+.3f"),
                 "–" if not np.isfinite(r["mean_cost"]) else f"{r['mean_cost']:.2f}",
                 "–" if not np.isfinite(r["mean_n_features"]) else f"{r['mean_n_features']:.0f}",
                 f"{r['train_time_s']:.1f}",
                 "–" if not np.isfinite(r["infer_time_ms"]) else f"{r['infer_time_ms']:.2f}",
                 "–" if not np.isfinite(r["delta_vs_ref"]) else f"{r['delta_vs_ref']:+.3f}",
                 _fmt_p(r["p_vs_ref_holm"]),
                 "–" if not np.isfinite(r["effect_vs_ref"]) else f"{r['effect_vs_ref']:+.2f}"]
        lines.append("| " + " | ".join(cells) + " |")
    notes = ["",
             "Balanced accuracy, chance = 0.333. Subject level: seeds averaged, then sessions "
             "(mean ± SD over subjects).",
             "*> chance*: recordings with permutation p < 0.05 after Holm correction over recordings.",
             "*Drop*: block-CV BA on ses-02 minus cross-session BA (train ses-01, test ses-02).",
             f"*Δ vs {reference}*: mean paired difference ({reference} − method); Wilcoxon signed-rank "
             "over subjects, Holm-corrected over the comparisons; r_rb = rank-biserial correlation.",
             "*Cost*: (n_ch/8 + n_bands/4 + n_win/7)/3, – = not applicable (EEGNet). "
             "*Train s*: per result file, including GA search."]
    if friedman and np.isfinite(friedman.get("p", np.nan)):
        pf = _fmt_p(friedman["p"])
        notes.append(f"Friedman test over methods: χ² = {friedman['statistic']:.2f}, "
                     f"p {pf if pf.startswith('<') else '= ' + pf}, n = {friedman['n']}.")
    warn = ["> **Built from synthetic/fake inputs: not real results.**", ""] if synthetic else []
    return "\n".join(warn + lines + notes) + "\n"


def run(results_dirs: Sequence[str], out_dir: str, reference: str = "ga_nsga",
        protocol: str = "block", n_perm: int = 1000) -> tuple[Path, Path]:
    df = A.load_results(results_dirs)
    if not len(df):
        raise SystemExit(f"no result JSONs found in {results_dirs}")
    tab = main_table(df, reference, protocol, n_perm)
    methods = tab["method"].tolist()
    fr = None
    if len(methods) >= 3:
        try:
            fr = A.friedman_test(df, methods, protocol)
        except ValueError:
            fr = None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    synthetic = bool(df["synthetic"].any()) or any("fake" in str(p) for p in results_dirs)
    md = out / "main_results.md"
    md.write_text(f"# Main results ({protocol} protocol)\n\n"
                  + to_markdown(tab, reference, fr, synthetic), encoding="utf-8")
    csv = out / "main_results.csv"
    tab.to_csv(csv, index=False)
    per_sub = A.aggregate_seeds(df).drop(columns=[], errors="ignore")
    per_sub.to_csv(out / "per_subject.csv", index=False)
    return md, csv


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Main results table as Markdown and CSV.")
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", default="results/tables")
    p.add_argument("--reference", default="ga_nsga")
    p.add_argument("--protocol", default="block")
    p.add_argument("--n-perm", type=int, default=1000)
    a = p.parse_args(argv)
    md, csv = run(a.results, a.out, a.reference, a.protocol, a.n_perm)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {md} and {csv}", file=sys.stderr)


if __name__ == "__main__":
    main()

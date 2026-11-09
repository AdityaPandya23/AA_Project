"""Paper-ready tables and figures (P6): ``results/final/``.

One script regenerates every final table and figure from the committed results:
P3/P4 method folders (``results/{method}``, ``results/baseline_all``) for the
pre-registered analysis (docs/RESULTS.md) and ``results/p5/*`` for the exploratory
checks (docs/P5_REPORT.md). Conventions: sample SD (ddof=1) over subjects;
subject level = seeds averaged, then sessions; cross-session drop = ses-02 block BA
minus cross-session BA; method names and colours from ``src.analysis``.

Outputs::

    results/final/tables/T1..T6 (*.md, *.csv), per_subject_{block,cross_session}.csv
    results/final/tables/TABLES.md          (all tables in one file)
    results/final/figures/F01..F11 (*.png), CAPTIONS.md

Usage::

    python -m reports.make_final --results results --out results/final
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src import analysis as A, chromosome as CH
from src.analysis import SURFACE, TEXT, TEXT_2, GRID, method_color, method_label

from . import make_p5_figures as P5F
from .make_tables import main_table

METHODS = ["fbcsp_all", "ga_nsga", "ga_single", "eegnet_32ch", "eegnet_8ch"]
FAMILY = [("S1", "ga_nsga", "ga_single", "block"), ("S2", "ga_nsga", "fbcsp_all", "cross_session"),
          ("S3", "ga_nsga", "eegnet_32ch", "block"), ("S4", "ga_nsga", "eegnet_8ch", "block")]


def _pm(m, s, fmt=".3f"):
    return "–" if not np.isfinite(m) else f"{m:{fmt}} ± {s:.3f}"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _peq(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def load(results: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = A.load_results([results, results / "baseline_all"])
    return df[df.method.isin(METHODS) & df.protocol.isin(["block", "cross_session"])]


# ------------------------------------------------------------------ tables
def t1_main(df: pd.DataFrame, n_perm: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tab = main_table(df, reference="fbcsp_all", protocol="block", n_perm=n_perm)
    tab = tab.set_index("method").reindex(METHODS).reset_index()
    md = pd.DataFrame({
        "Method": tab["label"], "n": tab["n_subjects"],
        "Block BA (mean ± SD)": [_pm(m, s) for m, s in zip(tab.ba_mean, tab.ba_sd)],
        "Median": tab.ba_median.map("{:.3f}".format),
        "Recordings > chance (Holm)": [f"{a}/{n}" for a, n in zip(tab.n_above_chance, tab.n_recordings)],
        "Cross-session BA (n = 20)": [_pm(m, s) for m, s in zip(tab.cross_ba_mean, tab.cross_ba_sd)],
        "Drop (ses-02 − cross)": [_pm(m, s, "+.3f") for m, s in zip(tab.drop_mean, tab.drop_sd)],
        "Cost": ["–" if not np.isfinite(c) else f"{c:.2f}" for c in tab.mean_cost],
        "Features / params": [f"{f:.0f}" for f in tab.mean_n_features]})
    return tab, md


def t2_tests(df: pd.DataFrame) -> pd.DataFrame:
    wide = A.subject_level(df, "block")
    d = (wide.ga_nsga - wide.fbcsp_all).dropna()
    prim = A.paired_test(df, "ga_nsga", "fbcsp_all", "block")
    one = stats.wilcoxon(d.to_numpy(), zero_method="wilcox", alternative="greater").pvalue
    rows = [{"Test": "Primary", "Comparison": "GA-NSGA > FBCSP all (one-sided)", "Protocol": "block",
             "n": prim["n"], "Mean Δ": f"{prim['mean_diff']:+.3f}", "Median Δ": f"{prim['median_diff']:+.3f}",
             "p": _p(one), "p (Holm)": "–", "r_rb": f"{prim['effect_size']:+.2f}"}]
    res = [A.paired_test(df, a, b, p) for _, a, b, p in FAMILY]
    ph = A.holm([r["p"] for r in res])
    for (k, a, b, p), r, h in zip(FAMILY, res, ph):
        rows.append({"Test": k, "Comparison": f"{method_label(a)} vs {method_label(b)}",
                     "Protocol": p.replace("_", "-"), "n": r["n"], "Mean Δ": f"{r['mean_diff']:+.3f}",
                     "Median Δ": f"{r['median_diff']:+.3f}", "p": _p(r["p"]), "p (Holm)": _p(h),
                     "r_rb": f"{r['effect_size']:+.2f}"})
    r = A.paired_test(df[df.subject != "01"], "ga_nsga", "fbcsp_all", "cross_session")
    rows.append({"Test": "S2 sensitivity", "Comparison": "S2 without sub-01", "Protocol": "cross-session",
                 "n": r["n"], "Mean Δ": f"{r['mean_diff']:+.3f}", "Median Δ": f"{r['median_diff']:+.3f}",
                 "p": _p(r["p"]), "p (Holm)": "outside family", "r_rb": f"{r['effect_size']:+.2f}"})
    return pd.DataFrame(rows)


def t3_selection(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in ("ga_nsga", "ga_single"):
        blk = df[(df.method == m) & (df.protocol == "block")]
        ex = A.explode_folds(blk)
        B = np.stack([CH.from_str(s) for s in ex.chosen_bits])
        fr = A.selection_frequency(blk, m).sort_values("frequency", ascending=False)
        st = {by: A.selection_stability(blk, m, by=by) for by in ("all", "seed", "fold")}
        rows.append({"Method": method_label(m), "Cost": f"{ex.cost.astype(float).mean():.2f}",
                     "Channels / bands / windows": f"{B[:, :8].sum(1).mean():.1f} / {B[:, 8:12].sum(1).mean():.1f} / {B[:, 12:].sum(1).mean():.1f}",
                     "Gene frequency range": f"{100 * fr.frequency.min():.0f}–{100 * fr.frequency.max():.0f} %",
                     "Top 3 genes": ", ".join(f"{r.name} {100 * r.frequency:.0f} %" for r in fr.head(3).itertuples()),
                     "Kuncheva: all / same fold / same seed": " / ".join(f"{st[b].kuncheva.mean():.3f}" for b in ("all", "seed", "fold"))})
    return pd.DataFrame(rows)


def t4_p5(results: Path) -> pd.DataFrame:
    p5 = results / "p5"
    rows = []
    s = json.loads((p5 / "position_control" / "summary.json").read_text(encoding="utf-8"))
    g = s["group"]["position_block30|cross_session|all"]
    g1 = s["group"]["position_block30|cross_session|without_sub01"]
    sm = s["sharing_moderation"]
    rows.append({"§": "1", "Check": "Position-only control (cross-session)",
                 "Result": f"Position predicts class: {g['mean']:.3f} ({_peq(g['p_greater'])}); "
                           f"without sub-01 {g1['mean']:.3f}. Shared-block subjects {sm['position_block30']['mean_shared']:.3f} "
                           f"vs {sm['position_block30']['mean_none']:.3f}; methods unchanged "
                           f"(FBCSP all {sm['fbcsp_all']['mean_shared']:.3f} vs {sm['fbcsp_all']['mean_none']:.3f}, "
                           f"p = {sm['fbcsp_all']['mwu_p']:.2f})"})
    s = json.loads((p5 / "random_control" / "summary.json").read_text(encoding="utf-8"))
    for m in ("ga_nsga", "ga_single"):
        v = s[f"{m}|block"]
        rows.append({"§": "2", "Check": f"Random chromosomes at matched cost: {method_label(m)}",
                     "Result": f"GA {v['ga_mean']:.3f} vs random {v['random_mean']:.3f}: Δ {v['diff_mean']:+.3f} "
                               f"(p = {v['p_two_sided']:.2f}); mean percentile {v['pct_fold_mean']:.2f}"})
    c = pd.read_csv(p5 / "chooser_rules" / "chooser_rules.csv")
    c = c[c.method == "ga_nsga"]
    rows.append({"§": "3", "Check": "Chooser rules (§k12), block",
                 "Result": f"Outer BA {c.outer_ba_mean.min():.3f}–{c.outer_ba_mean.max():.3f} across 7 rules; "
                           f"vs FBCSP all {c.vs_fbcsp_diff.min():+.3f} to {c.vs_fbcsp_diff.max():+.3f} "
                           f"(all p ≥ {c.vs_fbcsp_p.min():.2f})"})
    s = json.loads((p5 / "drift" / "summary.json").read_text(encoding="utf-8"))
    inf = s["inflation"]
    rows.append({"§": "4", "Check": "Random 5-fold vs block",
                 "Result": ", ".join(f"{method_label(m)} {v['diff']:+.3f}" for m, v in inf.items())
                           + f" (all p ≥ {min(v['p'] for v in inf.values()):.2f})"})
    v = s["view_ga_nsga_vs_fbcsp"]
    rows.append({"§": "4", "Check": "Viewing period: GA-NSGA vs FBCSP all",
                 "Result": f"{v['mean_a']:.3f} vs {v['mean_b']:.3f}: Δ {v['diff']:+.3f} (p = {v['p']:.3f})"})
    t = pd.read_csv(p5 / "train_accuracy" / "summary.csv")
    t = t[t.protocol == "block"].set_index("method")
    rows.append({"§": "5", "Check": "Training vs test BA (block)",
                 "Result": "; ".join(f"{method_label(m)} {t.loc[m, 'train_ba_mean']:.3f} vs {t.loc[m, 'test_ba_mean']:.3f}"
                                     for m in ("fbcsp_all", "ga_single", "ga_nsga"))})
    v = pd.read_csv(p5 / "eegnet_diag" / "variants.csv")
    v = v[v.protocol == "block"].set_index("method")
    rows.append({"§": "5", "Check": "EEGNet training diagnostic (block, seed 0)",
                 "Result": f"Training BA {v.loc['eegnet_32ch_logged', 'train_ba']:.2f} (P4 recipe) → "
                           f"{v.loc['eegnet_32ch_fixed300', 'train_ba']:.2f} (300 epochs); test "
                           f"{v.loc['eegnet_32ch_logged', 'test_ba']:.3f} → {v.loc['eegnet_32ch_fixed300', 'test_ba']:.3f} "
                           f"(p = {v.loc['eegnet_32ch_fixed300', 'p_vs_p4']:.3f})"})
    return pd.DataFrame(rows)


def t5_train(results: Path) -> pd.DataFrame:
    t = pd.read_csv(results / "p5" / "train_accuracy" / "summary.csv")
    v = pd.read_csv(results / "p5" / "eegnet_diag" / "variants.csv").set_index(["method", "protocol"])
    rows = []
    for prot in ("block", "cross_session"):
        for m in ("fbcsp_all", "ga_single", "ga_nsga"):
            r = t[(t.method == m) & (t.protocol == prot)].iloc[0]
            rows.append({"Method": method_label(m), "Protocol": prot.replace("_", "-"),
                         "Training BA": f"{r.train_ba_mean:.3f}", "Inner-CV BA": f"{r.inner_ba_mean:.3f}",
                         "Test BA": f"{r.test_ba_mean:.3f}", "Training − test": f"{r.train_minus_test:+.3f}"})
        for m in ("eegnet_32ch", "eegnet_8ch"):
            r = v.loc[(f"{m}_logged", prot)]
            rows.append({"Method": method_label(m) + " (seed 0)", "Protocol": prot.replace("_", "-"),
                         "Training BA": f"{r.train_ba:.3f}", "Inner-CV BA": f"val {r.val_ba:.3f}",
                         "Test BA": f"{r.test_ba:.3f}", "Training − test": f"{r.train_ba - r.test_ba:+.3f}"})
    return pd.DataFrame(rows)


def t6_view(results: Path) -> pd.DataFrame:
    s = json.loads((results / "p5" / "drift" / "summary.json").read_text(encoding="utf-8"))
    rows = []
    for m, v in s["view_vs_imagery"].items():
        ac, vm = s["above_chance"][f"{m}_view"], s["view_methods"][f"{m}_view"]
        rows.append({"Method": method_label(m), "Imagery BA": f"{v['mean_b']:.3f}",
                     "Viewing BA": _pm(vm["mean"], vm["sd"]), "Viewing − imagery": f"{v['diff']:+.3f}",
                     "p": _p(v["p"]), "Viewing recordings > chance (Holm)": f"{ac['p05_holm']}/{ac['n']}",
                     "Cost": f"{vm['cost']:.2f}"})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def _save_ax(ax, path: Path) -> Path:
    fig = ax.figure
    fig.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    A._plt().close(fig)
    return path


def fig_view(results: Path, path: Path) -> Path:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = A.load_results([results, results / "baseline_all", results / "sanity", results / "p5" / "drift"])
    blk = A.subject_level(df[df.seed.isin([0, 1, 2])], "block")
    fig, (ax,) = P5F._fig(7.0, 4.2)
    for i, m in enumerate(["fbcsp_all", "ga_single", "ga_nsga"]):
        a, b = blk[m], blk[f"{m}_view"].reindex(blk[m].index)
        for s in a.index:
            ax.plot([i - 0.15, i + 0.15], [a[s], b[s]], color=GRID, linewidth=0.8, zorder=1)
        ax.scatter(np.full(len(a), i - 0.15), a, s=22, color=method_color(m), zorder=3)
        ax.scatter(np.full(len(b), i + 0.15), b, s=22, marker="D", color=method_color(m),
                   edgecolor=TEXT_2, linewidth=0.6, zorder=3)
        ax.annotate(f"{b.mean():.3f}", (i + 0.15, b.max() + 0.01), ha="center", fontsize=8, color=TEXT)
    A._chance_line(ax)
    ax.set_xticks(range(3))
    ax.set_xticklabels([method_label(m) for m in ["fbcsp_all", "ga_single", "ga_nsga"]], fontsize=9)
    ax.set_ylabel("Subject BA: imagery (left) vs viewing (right)")
    ax.set_title("The viewing period is decodable; GA-NSGA gives up accuracy there", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    A._plt().close(fig)
    return path


def figures(df: pd.DataFrame, results: Path, out: Path, t1: pd.DataFrame, t2: pd.DataFrame) -> list[tuple[str, str]]:
    out.mkdir(parents=True, exist_ok=True)
    caps = []
    blk = df[df.protocol == "block"]
    r = t1.set_index("method")

    def cap(name, text):
        caps.append((name, text))

    _save_ax(A.plot_method_comparison(df, "block"), out / "F01_method_comparison_block.png")
    prim = t2.iloc[0]
    cap("F01_method_comparison_block.png",
        f"**Subject-level balanced accuracy, block protocol (n = 22).** One dot per subject (seeds, then "
        f"sessions averaged); lines connect a subject across methods; dashed line = chance (1/3). Means: "
        + ", ".join(f"{method_label(m)} {r.loc[m, 'ba_mean']:.3f}" for m in METHODS)
        + f". Pre-registered primary test GA-NSGA > FBCSP all: Δ {prim['Mean Δ']}, one-sided p = {prim['p']} (not supported).")
    _save_ax(A.plot_cost_vs_accuracy(df, "block"), out / "F02_cost_vs_accuracy_block.png")
    cap("F02_cost_vs_accuracy_block.png",
        "**Accuracy against cost, block protocol.** Grey: every inner-CV Pareto entry of the GA (inner BA, "
        "optimistic); coloured: outer-test subject means. GA-NSGA reaches FBCSP-all accuracy at about 30 % of "
        f"the cost ({r.loc['ga_nsga', 'mean_cost']:.2f}); EEGNet has no cost (shown at n/a).")
    _save_ax(A.plot_method_comparison(df, "cross_session"), out / "F03_method_comparison_cross_session.png")
    cap("F03_method_comparison_cross_session.png",
        "**Cross-session balanced accuracy** (train ses-01, test ses-02; one dot per subject, n = 20; sub-09/10 "
        "have no ses-02; per-subject values in `tables/per_subject_cross_session.csv`). Means: " + ", ".join(f"{method_label(m)} {r.loc[m, 'cross_ba_mean']:.3f}" for m in METHODS)
        + ". sub-01 is the only subject with the same trial sequence in both sessions (see F08).")
    freq = A.selection_frequency(blk, "ga_nsga")
    _save_ax(A.plot_selection_frequency(freq), out / "F04_selection_frequency_ga_nsga.png")
    top = freq.sort_values("frequency", ascending=False).iloc[0]
    cap("F04_selection_frequency_ga_nsga.png",
        f"**How often GA-NSGA selects each gene** (block, {int(freq.n.iloc[0])} chosen chromosomes). "
        f"Selection is nearly flat ({100 * freq.frequency.min():.0f}–{100 * freq.frequency.max():.0f} %); "
        f"the most frequent gene is {top['name']} ({100 * top.frequency:.0f} %).")
    p5 = results / "p5"
    tmp = out / "_p5tmp"
    tmp.mkdir(exist_ok=True)
    pairs = [("train_test", "train_inner_test.png", "F05_train_inner_test.png",
              "**Training vs inner-CV vs outer-test BA** (block, n = 22; P5 §5.1, exploratory). Every model fits "
              "its training trials (FBCSP all 0.999 with 168 features on about 90 trials) but generalises to "
              "about chance + 0.02. The GA's inner-CV score of its chosen chromosome is inflated by selection "
              "(winner's curse); FBCSP all's unselected inner CV is not."),
             ("random_control", "random_control.png", "F06_random_control.png",
              "**Random-chromosome control at matched cost** (P5 §2, exploratory). Left: the GA's percentile "
              "among 50 random chromosomes with the same channel/band/window counts, per block fold; the flat "
              "histogram at the dashed 10 % line means the GA's pick is a typical random subset. Right: per "
              "subject, GA BA vs the mean BA of the random subsets (diagonal = equal)."),
             ("chooser_rules", "chooser_rules_block.png", "F07_chooser_rules_block.png",
              "**Alternative chooser rules on the stored GA-NSGA fronts** (block, n = 22; P5 §3, exploratory). "
              "No rule differs from FBCSP all or from the contract rule (1sd). These are test-set scores and "
              "must not be used to pick a rule."),
             ("position_control", "position_control_cross_session.png", "F08_position_control_cross_session.png",
              "**Position-only control, cross-session** (P5 §1, exploratory). A classifier that sees only the "
              "trial's position in its block (no EEG) is above chance where 30-trial block orders repeat across "
              "sessions (x axis); the EEG methods do not follow it."),
             ("drift", "random_vs_block.png", "F09_random_vs_block.png",
              "**Random 5-fold vs leave-one-block-out** (subject level, seeds 0–2; P5 §4.1, exploratory). No "
              "method gains from random splits (numbers = mean random − block)."),
             ("eegnet_curves", "eegnet_learning_curves.png", "F10_eegnet_learning_curves.png",
              "**EEGNet learning curves without early stopping** (300 epochs, block folds, seed 0; P5 §5.2, "
              "exploratory). Training BA reaches 1.0 while validation BA stays at chance.")]
    makers = {"train_test": lambda: P5F.train_test(p5, tmp), "random_control": lambda: P5F.random_control(p5, tmp),
              "chooser_rules": lambda: P5F.chooser_rules(p5, tmp), "position_control": lambda: P5F.position_control(p5, tmp),
              "drift": lambda: P5F.drift(results, p5, tmp), "eegnet_curves": lambda: P5F.eegnet_curves(p5, tmp)}
    for key, src, dst, text in pairs:
        if makers[key]() is not None:
            (tmp / src).replace(out / dst)
            cap(dst, text)
    shutil.rmtree(tmp, ignore_errors=True)
    fig_view(results, out / "F11_viewing_vs_imagery.png")
    cap("F11_viewing_vs_imagery.png",
        "**Imagery (3 s) vs viewing period (3–7 s), block protocol** (n = 22; P5 §4.2, exploratory). All "
        "methods decode the viewing period; there GA-NSGA is 0.025 below FBCSP all (p = 0.004).")
    return caps


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/final")
    p.add_argument("--n-perm", type=int, default=1000)
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    tdir, fdir = out / "tables", out / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    df = load(results)
    t1, t1md = t1_main(df, a.n_perm)
    tables = {"T1_main_results": ("Main results (pre-registered methods)", t1md, t1),
              "T2_preregistered_tests": ("Pre-registered tests (ANALYSIS_PLAN §2–3)", t2_tests(df), None),
              "T3_selection": ("GA selection and stability (block)", t3_selection(df), None),
              "T4_p5_summary": ("Exploratory checks (P5)", t4_p5(results), None),
              "T5_training_vs_test": ("Training vs inner-CV / validation vs test BA (P5 §5)", t5_train(results), None),
              "T6_viewing_vs_imagery": ("Viewing period vs imagery (P5 §4.2)", t6_view(results), None)}
    notes = {"T1_main_results": "Balanced accuracy, chance = 0.333. Subject level (seeds, then sessions); mean ± "
             "sample SD over subjects. Recordings > chance: label-permutation p < 0.05 after Holm over the 42 "
             "recordings. Drop = ses-02 block BA − cross-session BA. Cost = (n_ch/8 + n_bands/4 + n_win/7)/3; "
             "– = not applicable. Timings are omitted (measured on three different machines).",
             "T2_preregistered_tests": "Wilcoxon signed-rank over subjects. Primary: one-sided; secondary: "
             "two-sided, Holm over S1–S4. Δ = first − second method.",
             "T5_training_vs_test": "Training BA = resubstitution on the outer training set (FBCSP methods) or the "
             "fit part of it (EEGNet); EEGNet's middle column is its validation BA."}
    all_md = ["# Final tables", ""]
    for name, (title, md, raw) in tables.items():
        (raw if raw is not None else md).to_csv(tdir / f"{name}.csv", index=False)
        text = f"## {name.split('_', 1)[0]}. {title}\n\n{_md(md)}\n" + (f"\n{notes[name]}\n" if name in notes else "")
        (tdir / f"{name}.md").write_text(text, encoding="utf-8")
        all_md.append(text)
    (tdir / "TABLES.md").write_text("\n".join(all_md), encoding="utf-8")
    for prot in ("block", "cross_session"):
        A.subject_level(df, prot)[METHODS].round(4).to_csv(tdir / f"per_subject_{prot}.csv")
    caps = figures(df, results, fdir, t1, tables["T2_preregistered_tests"][1])
    (fdir / "CAPTIONS.md").write_text(
        "# Figure captions\n\n" + "\n\n".join(f"**{n.split('_')[0]}** (`{n}`). {t}" for n, t in caps) + "\n",
        encoding="utf-8")
    print("\n".join(all_md))
    print(f"\n{len(caps)} figures in {fdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

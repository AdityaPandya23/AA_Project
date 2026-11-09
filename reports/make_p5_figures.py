"""Figures for docs/P5_REPORT.md (exploratory), in the style of ``src/analysis.py``.

Reads only committed P5 outputs under ``results/p5/`` (plus P3/P4 results for
reference lines) and writes PNGs to ``results/figures/p5/``. A figure whose inputs
are missing is skipped.

Usage::

    python -m reports.make_p5_figures --results results --out results/figures/p5
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src import analysis as A
from src.analysis import CHANCE, GRID, SURFACE, TEXT, TEXT_2, method_color, method_label

CONTROL = "#8a8985"   # neutral grey for controls / random draws (not a method)


def _fig(w=7.0, h=4.0, ncols=1):
    plt = A._plt()
    fig, axes = plt.subplots(1, ncols, figsize=(w, h), facecolor=SURFACE, squeeze=False)
    for ax in axes[0]:
        A._style_ax(ax)
    return fig, axes[0]


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    A._plt().close(fig)
    return path


def position_control(p5: Path, out: Path) -> Path | None:
    f = p5 / "position_control" / "per_subject.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, dtype={"subject": str})
    cs = df[df.protocol == "cross_session"].pivot(index="subject", columns="method", values="ba")
    shared = df[df.protocol == "cross_session"].groupby("subject").n_identical_cross.first()
    fig, (ax,) = _fig(7.0, 4.2)
    rng = np.random.default_rng(0)
    series = [("position_block30", "Position only (no EEG)", CONTROL, "X"),
              ("fbcsp_all", method_label("fbcsp_all"), method_color("fbcsp_all"), "s"),
              ("ga_nsga", method_label("ga_nsga"), method_color("ga_nsga"), "o")]
    xs = {}
    for i, (m, lab, col, mk) in enumerate(series):
        x = shared.reindex(cs.index).to_numpy() + (i - 1) * 0.18 + rng.uniform(-0.04, 0.04, len(cs))
        xs[m] = pd.Series(x, index=cs.index)
        ax.scatter(x, cs[m], s=46, color=col, marker=mk, edgecolor=SURFACE, linewidth=1.2,
                   label=lab, zorder=3)
    for m in ("position_block30", "fbcsp_all"):
        ax.annotate("sub-01", xy=(xs[m]["01"], cs.loc["01", m]), xytext=(7, -3),
                    textcoords="offset points", fontsize=8, color=TEXT_2)
    A._chance_line(ax)
    ax.set_xticks(range(5))
    ax.set_xlabel("ses-02 blocks with the same 30-trial order as a ses-01 block")
    ax.set_ylabel("Cross-session balanced accuracy")
    ax.set_title("Position predicts class where block orders repeat; the methods do not follow",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, out / "position_control_cross_session.png")


def random_control(p5: Path, out: Path) -> Path | None:
    fl_path, sl_path = p5 / "random_control" / "fold_level.csv", p5 / "random_control" / "subject_level.csv"
    if not fl_path.exists():
        return None
    fl = pd.read_csv(fl_path, dtype={"subject": str})
    sl = pd.read_csv(sl_path, dtype={"subject": str})
    fig, (a1, a2) = _fig(10.0, 4.0, 2)
    bins = np.linspace(0, 1, 11)
    for i, m in enumerate(["ga_nsga", "ga_single"]):
        v = fl[(fl.method == m) & (fl.protocol == "block")].percentile
        h, _ = np.histogram(v, bins=bins)
        a1.bar(bins[:-1] + 0.05 + (i - 0.5) * 0.042, h / len(v), width=0.04,
               color=method_color(m), label=method_label(m), zorder=3)
    a1.axhline(0.1, color=TEXT_2, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1,
               label="Expected if the GA were a random draw")
    a1.set_xlabel("GA's percentile among 50 same-cost random chromosomes (block folds)")
    a1.set_ylabel("Share of folds")
    a1.set_title("The GA's pick is a typical random subset", fontsize=10, loc="left")
    a1.set_ylim(0, 0.19)
    a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=1)
    blk = sl[sl.protocol == "block"]
    lo, hi = 0.27, 0.43
    a2.plot([lo, hi], [lo, hi], color=GRID, linewidth=1.2, zorder=1)
    for m, mk in (("ga_nsga", "o"), ("ga_single", "^")):
        a2.scatter(blk[f"random@{m}"], blk[m], s=40, marker=mk, color=method_color(m),
                   edgecolor=SURFACE, linewidth=1.2, label=method_label(m), zorder=3)
    a2.set_xlim(lo, hi)
    a2.set_ylim(lo, hi)
    a2.set_xlabel("Mean BA of same-cost random chromosomes")
    a2.set_ylabel("GA BA")
    a2.set_title("Per subject (block, n = 22)", fontsize=10, loc="left")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    return _save(fig, out / "random_control.png")


def chooser_rules(p5: Path, out: Path) -> Path | None:
    f = p5 / "chooser_rules" / "chooser_rules.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    fig, (ax,) = _fig(7.0, 4.0)
    nsga = df[df.method == "ga_nsga"].sort_values("cost_mean")
    ax.scatter(nsga.cost_mean, nsga.outer_ba_mean, s=46, color=method_color("ga_nsga"),
               edgecolor=SURFACE, linewidth=1.2, zorder=3, label="GA-NSGA, chooser rules")
    place = {"mincost": ((-7, 5), "right"), "2sd": ((7, -11), "left"), "knee": ((-7, 7), "right"),
             "1sd": ((18, -38), "left"), "se": ((8, -3), "left"), "0.5sd": ((-3, 10), "right"),
             "0sd": ((7, 3), "left")}
    for r in nsga.itertuples():
        off, ha = place.get(r.rule, ((5, 6), "left"))
        lead = dict(arrowstyle="-", color=TEXT_2, linewidth=0.6) if r.rule == "1sd" else None
        ax.annotate(r.rule + (" (contract)" if r.rule == "1sd" else ""), (r.cost_mean, r.outer_ba_mean),
                    xytext=off, textcoords="offset points", fontsize=8, color=TEXT_2, ha=ha,
                    arrowprops=lead)
    s = df[df.method == "ga_single"].iloc[0]
    ax.scatter([s.cost_mean], [s.outer_ba_mean], s=46, marker="^", color=method_color("ga_single"),
               edgecolor=SURFACE, linewidth=1.2, zorder=3, label=method_label("ga_single"))
    ax.scatter([1.0], [df.fbcsp_all_mean.iloc[0]], s=46, marker="s", color=method_color("fbcsp_all"),
               edgecolor=SURFACE, linewidth=1.2, zorder=3, label=method_label("fbcsp_all"))
    A._chance_line(ax)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0.32, 0.37)
    ax.set_xlabel("Mean cost of the chosen chromosome")
    ax.set_ylabel("Outer-test BA (block, n = 22)")
    ax.set_title("No chooser rule moves accuracy (test-set scores; not for picking a rule)",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    return _save(fig, out / "chooser_rules_block.png")


def drift(results: Path, p5: Path, out: Path) -> Path | None:
    d = p5 / "drift"
    if not d.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rnd = A.load_results(d)
        ref = A.load_results([results, results / "baseline_all", results / "sanity"])
    methods = [m for m in ("fbcsp_all", "ga_nsga", "ga_single", "eegnet_32ch", "eegnet_8ch")]
    rnd, ref = rnd[rnd.seed.isin([0, 1, 2])], ref[ref.seed.isin([0, 1, 2])]  # like for like (p5_drift)
    blk = A.subject_level(ref, "block")
    rr = A.subject_level(pd.concat([rnd, ref[ref.protocol == "random"]]), "random")
    fig, (ax,) = _fig(8.0, 4.4)
    top = max(max(blk[m].max(), rr[m].max()) for m in methods if m in rr and m in blk)
    for i, m in enumerate(methods):
        if m not in rr or m not in blk:
            continue
        b, r = blk[m], rr[m].reindex(blk.index)
        for s in blk.index:
            ax.plot([i - 0.15, i + 0.15], [b[s], r[s]], color=GRID, linewidth=0.8, zorder=1)
        ax.scatter(np.full(len(b), i - 0.15), b, s=22, color=method_color(m), zorder=3)
        ax.scatter(np.full(len(r), i + 0.15), r, s=22, color=method_color(m), marker="D",
                   edgecolor=TEXT_2, linewidth=0.6, zorder=3)
        ax.annotate(f"{(r - b).mean():+.3f}", (i, top + 0.012), ha="center", fontsize=8, color=TEXT)
    A._chance_line(ax)
    ax.set_ylim(None, top + 0.022)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([method_label(m) for m in methods], fontsize=8)
    ax.set_ylabel("Subject BA: block (left) vs random 5-fold (right)")
    ax.set_title("Random-split inflation per method (number = mean random − block)",
                 fontsize=10, loc="left")
    return _save(fig, out / "random_vs_block.png")


def train_test(p5: Path, out: Path) -> Path | None:
    f = p5 / "train_accuracy" / "summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    s = s[s.protocol == "block"].set_index("method").reindex(["fbcsp_all", "ga_single", "ga_nsga"])
    fig, (ax,) = _fig(7.0, 4.0)
    shades = {"train_ba_mean": "#184f95", "inner_ba_mean": "#6da7ec", "test_ba_mean": "#cde2fb"}
    labels = {"train_ba_mean": "Training (resubstitution)", "inner_ba_mean": "Inner CV (GA fitness)",
              "test_ba_mean": "Outer test"}
    x = np.arange(len(s))
    for j, (col, c) in enumerate(shades.items()):
        ax.bar(x + (j - 1) * 0.26, s[col], width=0.24, color=c, label=labels[col], zorder=3)
        for xi, v in zip(x, s[col]):
            ax.annotate(f"{v:.2f}", (xi + (j - 1) * 0.26, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=TEXT)
    A._chance_line(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m) for m in s.index], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Balanced accuracy (block, n = 22)")
    ax.set_title("Models fit their training data; held-out accuracy stays near chance",
                 fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    return _save(fig, out / "train_inner_test.png")


def eegnet_curves(p5: Path, out: Path) -> Path | None:
    f = p5 / "eegnet_diag" / "curves.csv"
    if not f.exists():
        return None
    cv = pd.read_csv(f)
    fig, axes = _fig(10.0, 4.0, 2)
    for ax, ch in zip(axes, ("eegnet_32ch", "eegnet_8ch")):
        c = cv[cv.method == f"{ch}_fixed300"]
        if c.empty:
            continue
        col = method_color(ch)
        ax.plot(c.epoch + 1, c.train_ba, color=col, linewidth=2, label="Training BA (fit part)")
        ax.plot(c.epoch + 1, c.val_ba, color=CONTROL, linewidth=2, label="Validation BA")
        A._chance_line(ax)
        ax.set_xlabel("Epoch (no early stopping)")
        ax.set_ylabel("Mean BA over block folds")
        ax.set_ylim(0.2, 1.02)
        ax.set_title(method_label(ch), fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="center right")
    return _save(fig, out / "eegnet_learning_curves.png")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/figures/p5")
    a = p.parse_args(argv)
    results, out = Path(a.results), Path(a.out)
    p5 = results / "p5"
    out.mkdir(parents=True, exist_ok=True)
    made = [position_control(p5, out), random_control(p5, out), chooser_rules(p5, out),
            drift(results, p5, out), train_test(p5, out), eegnet_curves(p5, out)]
    for m in made:
        if m:
            print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())

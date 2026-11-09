"""Random-chromosome control at matched cost (P5; P3_REPORT_R02 §3.2).

Does the GA's *search* add anything beyond the cost reduction? For every outer
fold of ``ga_nsga`` and ``ga_single`` (block and cross-session), draw ``n_draws``
random chromosomes with exactly the same number of channels, bands and windows as
that fold's ``chosen_bits`` (so the same cost and validity), and score each once
on the outer test fold exactly like ``run_ga``'s final step:
``fbcsp.evaluate(X, y, bits, [(train_idx, test_idx)])`` (a fresh FBCSP+LDA fitted
on the full outer training set). The GA's own chromosome is re-scored the same
way and must reproduce the stored fold BA, which shows the pipelines are identical.

Draws are seeded per fold: ``default_rng([method_code, subject, session_code,
seed, fold])``.

Outputs (exploratory, not part of ANALYSIS_PLAN), under ``--out``:

* ``{method}/sub-XX_ses-YY_seed-Z.json``: per fold the GA bits and BA, the random
  bits and BAs, and the GA's percentile in the random distribution.
* ``fold_level.csv``, ``subject_level.csv``, ``summary.json``, ``summary.md``.

Usage::

    AA_HOST=R07 python -m experiments.random_control --n-draws 50 --n-jobs 8 \
        --out results/p5/random_control
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import analysis as A, chromosome as CH, fbcsp
from src.config import config_hash, load_config

from . import _common as C
from .common import git_commit, resolve_cache_dir

METHODS = ("ga_nsga", "ga_single")
METHOD_CODE = {"ga_nsga": 1, "ga_single": 2}
SESSION_CODE = {"01": 1, "02": 2, "01to02": 12}
GROUPS = ((0, CH.N_CH), (CH.N_CH, CH.N_CH + CH.N_BAND), (CH.N_CH + CH.N_BAND, CH.N_BITS))


def draw_matched(bits: CH.BitsLike, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """``n`` random chromosomes with the same per-group counts as ``bits`` (uniform subsets)."""
    bits = CH.from_str(bits) if isinstance(bits, str) else np.asarray(bits, np.uint8)
    out = []
    for _ in range(n):
        b = np.zeros(CH.N_BITS, np.uint8)
        for lo, hi in GROUPS:
            k = int(bits[lo:hi].sum())
            b[lo + rng.choice(hi - lo, size=k, replace=False)] = 1
        assert CH.is_valid(b) and CH.counts(b) == CH.counts(bits)
        out.append(b)
    return out


def percentile(ga_ba: float, rand_bas: np.ndarray) -> float:
    """Mid-rank percentile of the GA's BA among the random BAs (0.5 = typical random draw)."""
    r = np.asarray(rand_bas)
    return float((np.sum(r < ga_ba - 1e-12) + 0.5 * np.sum(np.abs(r - ga_ba) <= 1e-12)) / r.size)


def out_path(out: Path, method: str, subject: str, session: str, seed: int) -> Path:
    return out / method / f"sub-{subject}_ses-{session}_seed-{seed}.json"


def run_file(method: str, result_path: str, cfg: dict[str, Any], cache_dir: str, out: str,
             n_draws: int) -> str:
    rec = json.loads(Path(result_path).read_text(encoding="utf-8"))
    subject, session, seed = rec["subject"], rec["session"], int(rec["seed"])
    dst = out_path(Path(out), method, subject, session, seed)
    if dst.exists():
        return str(dst)
    d, splits = C.load_unit("fbcsp", subject, session, cfg, Path(cache_dir))
    X, y = d["X"], d["y"]
    folds, t0 = [], time.perf_counter()
    for f in sorted(rec["per_fold"], key=lambda f: f["fold"]):
        k = int(f["fold"])
        tr, te = splits[k]
        assert np.intersect1d(tr, te).size == 0
        ga_bits = CH.from_str(f["chosen_bits"])
        ga_ba = fbcsp.evaluate(X, y, ga_bits, [(tr, te)])["per_fold"][0]
        if abs(ga_ba - f["ba"]) > 1e-9:
            raise AssertionError(f"{method} sub-{subject} ses-{session} seed-{seed} fold {k}: "
                                 f"re-scored GA BA {ga_ba} != stored {f['ba']}")
        rng = np.random.default_rng([METHOD_CODE[method], int(subject), SESSION_CODE[session],
                                     seed, k])
        draws = draw_matched(ga_bits, n_draws, rng)
        bas = [fbcsp.evaluate(X, y, b, [(tr, te)])["per_fold"][0] for b in draws]
        folds.append({"fold": k, "counts": list(CH.counts(ga_bits)), "cost": CH.cost(ga_bits),
                      "ga_bits": f["chosen_bits"], "ga_ba": float(ga_ba),
                      "random_bits": [CH.to_str(b) for b in draws],
                      "random_ba": [float(v) for v in bas],
                      "random_mean": float(np.mean(bas)), "random_sd": float(np.std(bas, ddof=1)),
                      "percentile": percentile(ga_ba, np.array(bas))})
    res = {"method": method, "subject": subject, "session": session, "seed": seed,
           "protocol": rec["protocol"], "n_draws": n_draws, "folds": folds,
           "time_s": time.perf_counter() - t0, "config_hash": config_hash(cfg),
           "git_commit": git_commit(), **C.provenance(d)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, dst)
    return str(dst)


def _wilcoxon(d: np.ndarray, alternative: str) -> float:
    d = np.asarray(d)
    return float(stats.wilcoxon(d, alternative=alternative).pvalue) if np.any(d != 0) else 1.0


def summarise(out: Path, results_root: Path) -> dict[str, Any]:
    rows = []
    for m in METHODS:
        for p in sorted((out / m).glob("*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            for f in r["folds"]:
                rows.append({"method": m, "subject": r["subject"], "session": r["session"],
                             "seed": r["seed"], "protocol": r["protocol"], "fold": f["fold"],
                             "cost": f["cost"], "ga_ba": f["ga_ba"], "random_mean": f["random_mean"],
                             "random_sd": f["random_sd"], "random_max": max(f["random_ba"]),
                             "percentile": f["percentile"]})
    fl = pd.DataFrame(rows)
    fl.to_csv(out / "fold_level.csv", index=False)

    # Per result file (fold means), then analysis.subject_level (seeds, then sessions).
    per_file = fl.groupby(["method", "subject", "session", "seed", "protocol"], as_index=False)[
        ["ga_ba", "random_mean", "percentile"]].mean()
    long = pd.concat([per_file.assign(method=per_file.method, ba_mean=per_file.ga_ba),
                      per_file.assign(method="random@" + per_file.method, ba_mean=per_file.random_mean),
                      per_file.assign(method="pct@" + per_file.method, ba_mean=per_file.percentile)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = A.load_results(results_root / "baseline_all")
    long = pd.concat([long[["method", "subject", "session", "seed", "protocol", "ba_mean"]],
                      base[["method", "subject", "session", "seed", "protocol", "ba_mean"]]])

    summary: dict[str, Any] = {"n_draws": int(json.loads(next((out / METHODS[0]).glob("*.json"))
                                                          .read_text(encoding="utf-8"))["n_draws"])}
    subj_tables = []
    for prot in ("block", "cross_session"):
        wide = A.subject_level(long, prot)
        subj_tables.append(wide.assign(protocol=prot))
        for m in METHODS:
            g, r, pct = wide[m], wide["random@" + m], wide["pct@" + m]
            f = fl[(fl.method == m) & (fl.protocol == prot)]
            d = (g - r).to_numpy()
            fb = wide["fbcsp_all"].reindex(g.index)
            summary[f"{m}|{prot}"] = {
                "n_subjects": int(g.size), "n_folds": int(len(f)),
                "ga_mean": float(g.mean()), "random_mean": float(r.mean()),
                "diff_mean": float(d.mean()), "diff_median": float(np.median(d)),
                "diff_sd": float(np.std(d, ddof=1)),
                "p_two_sided": _wilcoxon(d, "two-sided"), "p_greater": _wilcoxon(d, "greater"),
                "subjects_ga_above_random": int((d > 0).sum()),
                "pct_fold_mean": float(f.percentile.mean()), "pct_fold_median": float(f.percentile.median()),
                "pct_fold_ge_95": float((f.percentile >= 0.95).mean()),
                "pct_fold_le_05": float((f.percentile <= 0.05).mean()),
                "pct_subject_mean": float(pct.mean()), "pct_subject_p_vs_half": _wilcoxon(pct - 0.5, "two-sided"),
                "random_sd_within_fold": float(f.random_sd.mean()),
                "fbcsp_all_mean": float(fb.mean()),
                "random_minus_fbcsp_mean": float((r - fb).mean()),
                "random_minus_fbcsp_p": _wilcoxon((r - fb).dropna().to_numpy(), "two-sided"),
                "mean_cost": float(f.cost.mean())}
    pd.concat(subj_tables).to_csv(out / "subject_level.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def to_markdown(s: dict[str, Any]) -> str:
    lines = [f"# Random-chromosome control at matched cost ({s['n_draws']} draws per fold)", "",
             "| Method | Protocol | n | GA BA | Random BA (same cost) | GA − random (p two-sided / greater) | "
             "Subjects GA > random | Fold percentile: mean, ≥ 0.95, ≤ 0.05 | Random − fbcsp_all (p) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, v in s.items():
        if "|" not in k:
            continue
        m, prot = k.split("|")
        lines.append(
            f"| {m} | {prot} | {v['n_subjects']} | {v['ga_mean']:.3f} | {v['random_mean']:.3f} | "
            f"{v['diff_mean']:+.3f} ({v['p_two_sided']:.3f} / {v['p_greater']:.3f}) | "
            f"{v['subjects_ga_above_random']}/{v['n_subjects']} | {v['pct_fold_mean']:.2f}, "
            f"{v['pct_fold_ge_95']:.1%}, {v['pct_fold_le_05']:.1%} | "
            f"{v['random_minus_fbcsp_mean']:+.3f} ({v['random_minus_fbcsp_p']:.3f}) |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--n-draws", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--results-root", default="results", help="holds {method}/ and baseline_all/")
    p.add_argument("--out", default="results/p5/random_control")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-summary", action="store_true")
    a = p.parse_args(argv)
    from joblib import Parallel, delayed

    cfg = load_config()
    cache_dir = resolve_cache_dir(a.cache_dir, cfg)
    root, out = Path(a.results_root), Path(a.out)
    jobs = [(m, str(f)) for m in a.methods for f in sorted((root / m).glob("*.json"))
            if not f.name.endswith("_random.json")]
    print(f"random control: {len(jobs)} result files, {a.n_draws} draws per fold", flush=True)
    t0 = time.perf_counter()
    gen = Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(run_file)(m, f, cfg, str(cache_dir), str(out), a.n_draws) for m, f in jobs)
    for i, path in enumerate(gen, 1):
        if i % 50 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] {path}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    if not a.no_summary:
        summarise(out, root)
        print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

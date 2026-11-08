"""Sanity checks for the FBCSP pipeline (all 19 genes on).

(a) **Shuffled labels.** Labels are permuted *within blocks*, which keeps each
    block's class balance, and the whole leave-one-block-out pipeline is refitted.
    The null mean should be about 0.333. The share of permutations at or above
    the true BA gives a refit-based permutation p-value per recording.
(b) **Block vs random split.** The same data under leave-one-block-out and under
    stratified 5-fold that ignores blocks (``cv.random_splits``). The difference
    measures how much leakage the random protocol adds. Random-protocol results
    are written per contract as ``{out}/fbcsp_all/sub-XX_ses-YY_seed-S_random.json``.
(c) **Viewing vs imagery.** The block protocol on the viewing cache (3–7 s),
    written as ``{out}/fbcsp_all_view/...``. Perception should decode better than imagery.

Outputs: ``{out}/sanity_summary.json`` and ``{out}/sanity_summary.md``.

Usage::

    python -m experiments.sanity --cache-dir cache_synth --n-perm 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from src import cache, chromosome, cv, fbcsp, results
from src.config import config_hash, load_config, repo_root

from .common import block_record, fbcsp_record, load, resolve_cache_dir

CHANCE = 1.0 / 3.0


def permute_within_blocks(y: np.ndarray, block: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Copy of ``y`` with labels shuffled independently inside each block."""
    yp = y.copy()
    for b in np.unique(block):
        idx = np.flatnonzero(block == b)
        yp[idx] = y[rng.permutation(idx)]
    return yp


def shuffled_null(X: np.ndarray, y: np.ndarray, block: np.ndarray, n_perm: int,
                  seed: int) -> np.ndarray:
    """Block-CV BA of the all-on pipeline under ``n_perm`` within-block label permutations."""
    rng = np.random.default_rng(seed)
    splits = cv.block_splits(block)
    bits = chromosome.all_on()
    return np.array([fbcsp.evaluate(X, permute_within_blocks(y, block, rng), bits, splits)["ba_mean"]
                     for _ in range(n_perm)])


def _recording_job(subject: str, session: str, cfg: dict, cache_dir: Path, out_dir: Path,
                   n_perm: int, random_seeds: list[int], has_view: bool) -> dict[str, Any]:
    d = load("fbcsp", subject, session, cfg, cache_dir)
    X, y, block, tid = d["X"], d["y"], d["block"], d["trial_id"]
    synthetic = bool(d.get("synthetic", False))
    row: dict[str, Any] = {"subject": subject, "session": session, "n_trials": int(y.size),
                           "synthetic": synthetic}

    # true block-protocol BA (same as baseline 1)
    blk = fbcsp_record("fbcsp_all", subject, session, "block", 0, X, y, block, tid,
                       cv.block_splits(block), cfg, measure_latency=False, synthetic=synthetic)
    row["block_ba"] = blk["ba_mean"]

    # (a) shuffled labels
    seed = int(subject) * 100 + int(session)
    null = shuffled_null(X, y, block, n_perm, seed) if n_perm > 0 else np.array([])
    row["shuffled"] = {
        "n_perm": int(n_perm),
        "mean": float(null.mean()) if null.size else None,
        "std": float(null.std()) if null.size else None,
        "q95": float(np.quantile(null, 0.95)) if null.size else None,
        "p_refit": float((1 + np.sum(null >= blk["ba_mean"])) / (1 + null.size)) if null.size else None,
        "null": null.round(5).tolist(),
    }

    # (b) block vs random splits
    rand_bas = []
    for s in random_seeds:
        rec = fbcsp_record("fbcsp_all", subject, session, "random", s, X, y, block, tid,
                           cv.random_splits(y, cfg["evaluation"]["random_n_splits"], s), cfg,
                           measure_latency=False, synthetic=synthetic)
        results.write_result(rec, cfg, out_dir)
        rand_bas.append(rec["ba_mean"])
    row["random_ba"] = float(np.mean(rand_bas)) if rand_bas else None
    row["random_ba_per_seed"] = rand_bas
    row["inflation"] = (row["random_ba"] - row["block_ba"]) if rand_bas else None

    # (c) viewing vs imagery
    if has_view:
        vrec = block_record("fbcsp_all_view", "view", subject, session, cfg, cache_dir,
                            measure_latency=False)
        results.write_result(vrec, cfg, out_dir)
        row["view_ba"] = vrec["ba_mean"]
        row["view_minus_imagery"] = vrec["ba_mean"] - row["block_ba"]
    else:
        row["view_ba"] = row["view_minus_imagery"] = None
    return row


def _mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)


def summarise(rows: list[dict], cfg: dict) -> dict[str, Any]:
    """Aggregate per-recording rows into the summary dict."""
    null_means = [r["shuffled"]["mean"] for r in rows if r["shuffled"]["mean"] is not None]
    all_null = np.concatenate([r["shuffled"]["null"] for r in rows]) if rows else np.array([])
    out = {
        "n_recordings": len(rows),
        "synthetic": any(r["synthetic"] for r in rows),
        "config_hash": config_hash(cfg),
        "chance": CHANCE,
        "a_shuffled": {
            "null_mean": float(np.mean(all_null)) if all_null.size else None,
            "null_std": float(np.std(all_null)) if all_null.size else None,
            "mean_of_recording_means": float(np.mean(null_means)) if null_means else None,
            "n_recordings_p_lt_0.05": int(sum(r["shuffled"]["p_refit"] is not None
                                              and r["shuffled"]["p_refit"] < 0.05 for r in rows)),
        },
        "b_block_vs_random": dict(zip(("block_mean", "block_std", "n"), _mean(rows, "block_ba")))
        | dict(zip(("random_mean", "random_std", "n_random"), _mean(rows, "random_ba")))
        | {"inflation_mean": _mean(rows, "inflation")[0],
           "n_random_gt_block": int(sum((r["inflation"] or 0) > 0 for r in rows))},
        "c_view_vs_imagery": dict(zip(("view_mean", "view_std", "n"), _mean(rows, "view_ba")))
        | {"view_minus_imagery_mean": _mean(rows, "view_minus_imagery")[0],
           "n_view_gt_imagery": int(sum((r["view_minus_imagery"] or 0) > 0 for r in rows))},
        "recordings": rows,
    }
    return out


def _f(v, fmt=".3f"):
    return "n/a" if v is None else format(v, fmt)


def to_markdown(s: dict) -> str:
    """Markdown report of the summary."""
    a, b, c = s["a_shuffled"], s["b_block_vs_random"], s["c_view_vs_imagery"]
    lines = ["# FBCSP sanity checks (all 19 genes on)", ""]
    if s["synthetic"]:
        lines += ["> **Synthetic cache** (fake subjects 91-93). These numbers are not results.", ""]
    lines += [f"Recordings: {s['n_recordings']}, config `{s['config_hash']}`, chance = 0.333.", "",
              "| Check | Result |", "|---|---|",
              f"| (a) shuffled labels, null BA | {_f(a['null_mean'])} ± {_f(a['null_std'])} "
              f"(recordings with refit p < 0.05: {a['n_recordings_p_lt_0.05']}) |",
              f"| (b) block-CV BA | {_f(b['block_mean'])} ± {_f(b['block_std'])} |",
              f"| (b) random 5-fold BA | {_f(b['random_mean'])} ± {_f(b['random_std'])} |",
              f"| (b) inflation (random − block) | {_f(b['inflation_mean'], '+.3f')} "
              f"({b['n_random_gt_block']}/{s['n_recordings']} recordings higher) |",
              f"| (c) viewing BA (block CV) | {_f(c['view_mean'])} ± {_f(c['view_std'])} |",
              f"| (c) viewing − imagery | {_f(c['view_minus_imagery_mean'], '+.3f')} "
              f"({c['n_view_gt_imagery']}/{s['n_recordings']} recordings higher) |", "",
              "## Per recording", "",
              "| sub | ses | n | block BA | null mean | null q95 | p (refit) | random BA | view BA |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in s["recordings"]:
        sh = r["shuffled"]
        lines.append(f"| {r['subject']} | {r['session']} | {r['n_trials']} | {_f(r['block_ba'])} | "
                     f"{_f(sh['mean'])} | {_f(sh['q95'])} | {_f(sh['p_refit'])} | "
                     f"{_f(r['random_ba'])} | {_f(r['view_ba'])} |")
    return "\n".join(lines) + "\n"


def run(cache_dir=None, out_dir=None, n_perm: int = 200, random_seeds=(0, 1, 2, 3, 4),
        n_jobs: int = -1, subjects=None) -> dict[str, Any]:
    """Run all sanity checks and write the summary files. Returns the summary dict."""
    cfg = load_config()
    cache_dir = resolve_cache_dir(cache_dir, cfg)
    out_dir = Path(out_dir) if out_dir is not None else repo_root() / "results" / "sanity"
    pairs = cache.list_cached("fbcsp", cfg, cache_dir)
    if subjects:
        pairs = [p for p in pairs if p[0] in subjects]
    if not pairs:
        raise SystemExit(f"no FBCSP cache files in {cache_dir / 'fbcsp'}")
    views = set(cache.list_cached("view", cfg, cache_dir))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_recording_job)(s, ses, cfg, cache_dir, out_dir, n_perm, list(random_seeds),
                                (s, ses) in views) for s, ses in pairs)
    summary = summarise(rows, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sanity_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (out_dir / "sanity_summary.md").write_text(to_markdown(summary), encoding="utf-8")
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="FBCSP sanity checks: shuffled labels, block vs "
                                            "random splits, viewing vs imagery.")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", default=None, help="output dir (default: results/sanity)")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--random-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--subjects", nargs="*", default=None)
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    s = run(a.cache_dir, a.out, a.n_perm, a.random_seeds, a.n_jobs, a.subjects)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(to_markdown(s))
    print(f"done in {time.perf_counter() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

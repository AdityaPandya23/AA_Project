"""Baseline 1 (``fbcsp_all``): FBCSP + shrinkage LDA with all 19 genes on.

For every (subject, session) in the FBCSP cache it runs:

* the **block** protocol (leave-one-block-out, 4 folds), written as
  ``{out}/fbcsp_all/sub-XX_ses-YY_seed-0.json``;
* the **cross-session** protocol (train ses-01, test ses-02) for every subject
  with both sessions, written as ``{out}/fbcsp_all/sub-XX_ses-01to02_seed-0.json``.

Usage::

    python -m experiments.baseline                     # cache from config / AA_CACHE_DIR
    python -m experiments.baseline --cache-dir cache_synth --n-jobs 6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from src import cache, results
from src.config import load_config, repo_root

from .common import block_record, cross_session_record, cross_session_subjects, resolve_cache_dir

METHOD = "fbcsp_all"


def _block_job(subject, session, cfg, cache_dir, out_dir):
    rec = block_record(METHOD, "fbcsp", subject, session, cfg, cache_dir)
    return results.write_result(rec, cfg, out_dir), rec


def _cross_job(subject, cfg, cache_dir, out_dir):
    rec = cross_session_record(METHOD, "fbcsp", subject, cfg, cache_dir)
    return results.write_result(rec, cfg, out_dir), rec


def run(cache_dir=None, out_dir=None, n_jobs: int = -1, subjects=None,
        protocols=("block", "cross_session"), verbose: int = 0) -> list[dict]:
    """Run Baseline 1 and write result JSONs. Returns the records."""
    cfg = load_config()
    cache_dir = resolve_cache_dir(cache_dir, cfg)
    out_dir = Path(out_dir) if out_dir is not None else repo_root() / "results" / "baseline_all"
    pairs = cache.list_cached("fbcsp", cfg, cache_dir)
    if subjects:
        pairs = [p for p in pairs if p[0] in subjects]
    if not pairs:
        raise SystemExit(f"no FBCSP cache files in {cache_dir / 'fbcsp'}")
    jobs = []
    if "block" in protocols:
        jobs += [delayed(_block_job)(s, ses, cfg, cache_dir, out_dir) for s, ses in pairs]
    if "cross_session" in protocols:
        subs = [s for s in cross_session_subjects(cfg, cache_dir) if not subjects or s in subjects]
        jobs += [delayed(_cross_job)(s, cfg, cache_dir, out_dir) for s in subs]
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(jobs)
    return [rec for _, rec in out]


def summarise(recs: list[dict]) -> str:
    """Plain-text summary table."""
    lines = [f"{'subject':>7} {'session':>7} {'protocol':>13} {'BA':>6} {'SD':>6} "
             f"{'train_s':>8} {'infer_ms':>8}"]
    for r in sorted(recs, key=lambda r: (r["protocol"], r["subject"], r["session"])):
        lines.append(f"{r['subject']:>7} {r['session']:>7} {r['protocol']:>13} {r['ba_mean']:6.3f} "
                     f"{r['ba_std']:6.3f} {r['train_time_s']:8.3f} {r['infer_time_ms']:8.3f}")
    for prot in ("block", "cross_session"):
        bas = [r["ba_mean"] for r in recs if r["protocol"] == prot]
        if bas:
            sd = np.std(bas, ddof=1) if len(bas) > 1 else float("nan")  # sample SD, as make_tables
            lines.append(f"{prot}: mean BA {np.mean(bas):.3f} +/- {sd:.3f} over {len(bas)} "
                         f"recordings (chance 0.333)")
    if any(r.get("synthetic") for r in recs):
        lines.append("NOTE: synthetic cache (fake subjects), not real data.")
    return "\n".join(lines)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Baseline 1: FBCSP+LDA on all 19 genes.")
    p.add_argument("--cache-dir", default=None, help="cache root (default: config / AA_CACHE_DIR)")
    p.add_argument("--out", default=None, help="results dir (default: results/baseline_all)")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--protocols", nargs="*", default=["block", "cross_session"],
                   choices=["block", "cross_session"])
    a = p.parse_args(argv)
    t0 = time.perf_counter()
    recs = run(a.cache_dir, a.out, a.n_jobs, a.subjects, tuple(a.protocols))
    print(summarise(recs))
    print(f"wrote {len(recs)} result files in {time.perf_counter() - t0:.1f} s", file=sys.stderr)


if __name__ == "__main__":
    main()

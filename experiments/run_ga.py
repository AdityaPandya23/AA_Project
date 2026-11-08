"""Run the GA methods (``ga_nsga`` = stability, ``ga_single`` = accuracy only).

For each outer fold: run the GA on the outer training indices only, then
score ``chosen_bits`` once on the outer test fold with ``fbcsp.evaluate``
(a fresh FBCSPLDA fitted on the full outer training set). One results JSON per
(method, subject, session, seed) (CONTRACTS.md §i), with per-fold
``y_true``, ``y_pred``, ``n_test`` and ``test_trial_id`` like R01's FBCSP
experiments, and the fold's final-population front in ``per_fold[k]["pareto"]``.
The fronts are also written to a side file
``{results_dir}/pareto/{method}/sub-XX_ses-YY_seed-Z.json``.

Work is split into (subject, session, seed, fold) jobs run with joblib
(``fbcsp.evaluate`` pins BLAS to 1 thread, so one process per job). Each
finished fold is checkpointed to ``{results_dir}/raw/partial/...`` at once,
and the final JSON is written as soon as all folds of a run are done.
Re-running the same command resumes and skips finished runs and folds.

Examples::

    python -m experiments.run_ga --method stability --subjects sub-17..sub-22 \
        --protocol block --seeds 0 1 2
    python -m experiments.run_ga --method ga_single --subjects 91 --protocol cross_session \
        --cache-dir cache_synth --pop-size 10 --n-generations 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src import chromosome, fbcsp, ga, results
from src.config import config_hash, load_config

from . import _common as C
from .common import git_commit, infer_latency_ms, resolve_cache_dir

METHODS = {"stability": "ga_nsga", "accuracy_only": "ga_single",
           "ga_nsga": "ga_nsga", "ga_single": "ga_single"}


def _suffix(protocol: str) -> str:
    return "_random" if protocol == "random" else ""


def partial_path(results_dir: Path, method: str, subject: str, session: str, seed: int,
                 protocol: str, fold: int) -> Path:
    """Crash-resume checkpoint of one fold (``results/raw`` is git-ignored)."""
    return (results_dir / "raw" / "partial" / method
            / f"sub-{subject}_ses-{session}_seed-{seed}{_suffix(protocol)}_fold-{fold}.json")


def pareto_path(results_dir: Path, method: str, subject: str, session: str, seed: int,
                protocol: str) -> Path:
    """Side file with the per-fold Pareto fronts (not read by ``analysis.load_results``)."""
    return (results_dir / "pareto" / method
            / f"sub-{subject}_ses-{session}_seed-{seed}{_suffix(protocol)}.json")


def run_fold(method: str, subject: str, session: str, seed: int, protocol: str, fold: int,
             cfg: dict[str, Any], cache_dir: str, results_dir: str) -> dict[str, Any]:
    """One outer fold: GA on train, final fit, score test. Checkpoints and returns the fold."""
    out = partial_path(Path(results_dir), method, subject, session, seed, protocol, fold)
    if out.exists():
        return json.loads(out.read_text())

    d, splits = C.load_unit("fbcsp", subject, session, cfg, Path(cache_dir))
    X, y, block = d["X"], d["y"], d["block"]   # X: one object for the whole job (fbcsp cache)
    train_idx, test_idx = splits[fold]
    assert np.intersect1d(train_idx, test_idx).size == 0

    t0 = time.perf_counter()
    runner = ga.run_nsga if method == "ga_nsga" else ga.run_single_objective
    # The GA receives train_idx only and asserts its inner splits stay inside it.
    res = runner(X, y, block, train_idx, cfg, seed)
    bits = res["chosen_bits"]
    # The outer test fold is touched exactly once, here.
    final = fbcsp.evaluate(X, y, bits, [(train_idx, test_idx)])
    t_train = time.perf_counter() - t0
    infer_ms = infer_latency_ms(X, y, bits, [(train_idx, test_idx)], final["y_pred"])

    pareto = [{"bits": chromosome.to_str(e["bits"]), "ba_mean": float(e["ba_mean"]),
               "ba_std": float(e["ba_std"]), "cost": float(e["cost"])} for e in res["pareto"]]
    rec = {
        "fold_record": results.make_fold_record(
            fold, final["per_fold"][0], bits, pareto=pareto, n_evals=int(res["n_evals"]),
            ga_time_s=float(res["time_s"]),
            **C.prediction_keys(train_idx, test_idx, final["y_pred"][0], d)),
        "train_time_s": float(t_train), "infer_time_ms": float(infer_ms),
        **C.provenance(d),
    }
    C.atomic_write_json(rec, out)
    return rec


def assemble(method: str, subject: str, session: str, seed: int, protocol: str,
             folds: list[dict[str, Any]], cfg: dict[str, Any], results_dir: Path) -> Path:
    """Write the contract JSON and the Pareto side file of one finished run."""
    folds = sorted(folds, key=lambda f: f["fold_record"]["fold"])
    per_fold = [f["fold_record"] for f in folds]
    bas = [f["ba"] for f in per_fold]
    rec = {
        "method": method, "subject": subject, "session": session, "seed": int(seed),
        "protocol": protocol, "per_fold": per_fold,
        "ba_mean": float(np.mean(bas)), "ba_std": float(np.std(bas)),
        "train_time_s": float(sum(f["train_time_s"] for f in folds)),
        "infer_time_ms": float(np.mean([f["infer_time_ms"] for f in folds])),
        "config_hash": config_hash(cfg),
        "n_evals": int(sum(f["n_evals"] for f in per_fold)),
        "ga_time_s": float(sum(f["ga_time_s"] for f in per_fold)),
        "git_commit": git_commit(),
        "synthetic": bool(folds[0].get("synthetic", False)),
    }
    if folds[0].get("host"):
        rec["host"] = folds[0]["host"]
    path = results.write_result(rec, cfg, results_dir)
    C.atomic_write_json({"method": method, "subject": subject, "session": session,
                         "seed": int(seed), "protocol": protocol,
                         "folds": [{"fold": f["fold"], "chosen_bits": f["chosen_bits"],
                                    "pareto": f["pareto"]} for f in per_fold]},
                        pareto_path(results_dir, method, subject, session, seed, protocol))
    return path


def _job(method, subject, session, seed, protocol, fold, cfg, cache_dir, results_dir):
    rec = run_fold(method, subject, session, seed, protocol, fold, cfg, cache_dir, results_dir)
    return (subject, session, seed, fold), rec


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", choices=sorted(METHODS), required=True,
                   help="stability|ga_nsga -> ga_nsga, accuracy_only|ga_single -> ga_single")
    p.add_argument("--subjects", nargs="+", required=True, help="e.g. sub-17..sub-22 03 05")
    p.add_argument("--protocol", choices=["block", "cross_session"], default="block")
    p.add_argument("--sessions", nargs="+", default=None,
                   help="block protocol only (default: config.dataset.sessions)")
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: config.ga.seeds")
    p.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                   help="parallel fold jobs (default: half the logical CPUs; throughput "
                   "saturated at ~8 on a 10-core i5-13450HX)")
    p.add_argument("--cache-dir", default=None, help="overrides AA_CACHE_DIR / config")
    p.add_argument("--results-dir", default=None, help="overrides AA_RESULTS_DIR / config")
    p.add_argument("--pop-size", type=int, default=None,
                   help="override config.ga.pop_size (changes config_hash)")
    p.add_argument("--n-generations", type=int, default=None,
                   help="override config.ga.n_generations (changes config_hash)")
    p.add_argument("--overwrite", action="store_true", help="recompute finished runs")
    a = p.parse_args(argv)

    from joblib import Parallel, delayed

    cfg = load_config()
    if a.pop_size:
        cfg["ga"]["pop_size"] = a.pop_size
    if a.n_generations:
        cfg["ga"]["n_generations"] = a.n_generations
    method = METHODS[a.method]
    cache_dir = resolve_cache_dir(a.cache_dir, cfg)
    results_dir = Path(a.results_dir) if a.results_dir else Path(cfg["paths"]["results_dir"])
    seeds = a.seeds if a.seeds is not None else list(cfg["ga"]["seeds"])
    sessions = a.sessions or list(cfg["dataset"]["sessions"])

    units = C.recordings(a.protocol, C.parse_subjects(a.subjects), sessions, "fbcsp", cfg,
                         cache_dir)
    runs, jobs = {}, []
    for subject, session in units:
        n_folds = len(C.load_unit("fbcsp", subject, session, cfg, cache_dir)[1])
        for seed in seeds:
            final = results.result_path(method, subject, session, seed, a.protocol, cfg,
                                        results_dir)
            if final.exists() and not a.overwrite:
                print(f"[done] {final}")
                continue
            if a.overwrite:
                for k in range(n_folds):
                    partial_path(results_dir, method, subject, session, seed, a.protocol,
                                 k).unlink(missing_ok=True)
            runs[(subject, session, seed)] = {"n": n_folds, "folds": []}
            jobs += [(subject, session, seed, k) for k in range(n_folds)]

    print(f"{method} {a.protocol}: {len(runs)} runs, {len(jobs)} fold jobs, "
          f"pop={cfg['ga']['pop_size']} gens={cfg['ga']['n_generations']}", flush=True)
    if not jobs:
        return 0
    t0 = time.perf_counter()
    gen = Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
        delayed(_job)(method, s, ses, seed, a.protocol, k, cfg, str(cache_dir), str(results_dir))
        for s, ses, seed, k in jobs)
    for (s, ses, seed, k), rec in gen:
        run = runs[(s, ses, seed)]
        run["folds"].append(rec)
        f = rec["fold_record"]
        print(f"[fold] {method} sub-{s} ses-{ses} seed-{seed} fold-{k}: ba={f['ba']:.3f} "
              f"bits={f['chosen_bits']} evals={f['n_evals']} {rec['train_time_s']:.1f}s",
              flush=True)
        if len(run["folds"]) == run["n"]:
            path = assemble(method, s, ses, seed, a.protocol, run["folds"], cfg, results_dir)
            print(f"[run ] wrote {path}", flush=True)
    print(f"total wall time {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the GA methods (``ga_nsga`` = stability, ``ga_single`` = accuracy only).

For each outer fold: run the GA on the outer training indices only, refit
FBCSPLDA on the full outer training set with ``chosen_bits``, score the outer
test fold once. One results JSON per (method, subject, session, seed)
(CONTRACTS.md §i); the per-fold Pareto fronts go to a side file
``{results_dir}/pareto/{method}/sub-XX_ses-YY_seed-Z.json``.

Work is split into (method, subject, session, seed, fold) jobs run with
joblib. Each finished fold is checkpointed to
``{results_dir}/raw/partial/...`` immediately, and the final JSON is written
as soon as all folds of a run are done. Re-running the same command resumes:
finished runs and finished folds are skipped.

Examples::

    python -m experiments.run_ga --method stability --subjects sub-17..sub-22 \
        --protocol block --seeds 0 1 2
    AA_CACHE_DIR=cache_synth python -m experiments.run_ga --method accuracy_only \
        --subjects 91..93 --protocol cross_session --allow-standin-fbcsp
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
from sklearn.metrics import balanced_accuracy_score

from experiments import _common as C
from src import chromosome, ga, results
from src.config import config_hash, load_config

METHODS = {"stability": "ga_nsga", "accuracy_only": "ga_single"}


def _partial_path(results_dir: Path, method: str, subject: str, session: str, seed: int,
                  protocol: str, fold: int) -> Path:
    suffix = "_random" if protocol == "random" else ""
    return (results_dir / "raw" / "partial" / method
            / f"sub-{subject}_ses-{session}_seed-{seed}{suffix}_fold-{fold}.json")


def pareto_path(results_dir: Path, method: str, subject: str, session: str, seed: int,
                protocol: str) -> Path:
    """Side file with the per-fold Pareto fronts."""
    suffix = "_random" if protocol == "random" else ""
    return results_dir / "pareto" / method / f"sub-{subject}_ses-{session}_seed-{seed}{suffix}.json"


def run_fold(method: str, subject: str, session: str, seed: int, protocol: str, fold: int,
             cfg: dict[str, Any], cache_dir: str | None, results_dir: str,
             allow_standin: bool) -> dict[str, Any]:
    """One outer fold: GA on train, refit, score test. Checkpoints and returns the fold dict."""
    results_dir_p = Path(results_dir)
    out = _partial_path(results_dir_p, method, subject, session, seed, protocol, fold)
    if out.exists():
        return json.loads(out.read_text())

    evaluate, FBCSPLDA, impl = C.resolve_fbcsp(allow_standin)
    ga.set_evaluate_fn(evaluate)
    d, splits = C.load_unit("fbcsp", subject, session, cfg,
                            Path(cache_dir) if cache_dir else None)
    X, y, block = d["X"], d["y"], d["block"]
    train_idx, test_idx = splits[fold]
    assert np.intersect1d(train_idx, test_idx).size == 0

    t0 = time.perf_counter()
    runner = ga.run_nsga if method == "ga_nsga" else ga.run_single_objective
    # The GA only receives train_idx; it asserts internally that its inner splits stay inside it.
    res = runner(X, y, block, train_idx, cfg, seed)
    bits = res["chosen_bits"]

    # Final fit on the full outer training set; the test fold is touched exactly once here.
    Xsel = chromosome.select(X, bits)
    clf = FBCSPLDA().fit(Xsel[train_idx], y[train_idx])
    t_train = time.perf_counter() - t0
    ba = float(balanced_accuracy_score(y[test_idx], clf.predict(Xsel[test_idx])))
    infer_ms = C.per_trial_latency_ms(clf.predict, Xsel[test_idx])

    rec = {
        "fold_record": results.make_fold_record(fold, ba, bits),
        "pareto": [{"bits": chromosome.to_str(e["bits"]), "ba_mean": e["ba_mean"],
                    "ba_std": e["ba_std"], "cost": e["cost"]} for e in res["pareto"]],
        "n_evals": int(res["n_evals"]), "ga_time_s": float(res["time_s"]),
        "train_time_s": float(t_train), "infer_time_ms": float(infer_ms),
        "fbcsp_impl": impl,
    }
    C.atomic_write_json(rec, out)
    return rec


def assemble(method: str, subject: str, session: str, seed: int, protocol: str,
             folds: list[dict[str, Any]], cfg: dict[str, Any], results_dir: Path) -> Path:
    """Write the contract JSON and the Pareto side file for one finished run."""
    folds = sorted(folds, key=lambda f: f["fold_record"]["fold"])
    bas = [f["fold_record"]["ba"] for f in folds]
    impls = sorted({f["fbcsp_impl"] for f in folds})
    rec = {
        "method": method, "subject": subject, "session": session, "seed": int(seed),
        "protocol": protocol, "per_fold": [f["fold_record"] for f in folds],
        "ba_mean": float(np.mean(bas)), "ba_std": float(np.std(bas)),
        "train_time_s": float(sum(f["train_time_s"] for f in folds)),
        "infer_time_ms": float(np.mean([f["infer_time_ms"] for f in folds])),
        "config_hash": config_hash(cfg),
        "n_evals": int(sum(f["n_evals"] for f in folds)),
        "ga_time_s": float(sum(f["ga_time_s"] for f in folds)),
        "fbcsp_impl": impls[0] if len(impls) == 1 else impls,
        "git_commit": C.git_commit(), "host": C.host_tag(),
    }
    path = results.write_result(rec, cfg, results_dir)
    C.atomic_write_json({"method": method, "subject": subject, "session": session,
                         "seed": int(seed), "protocol": protocol,
                         "folds": [{"fold": f["fold_record"]["fold"],
                                    "chosen_bits": f["fold_record"]["chosen_bits"],
                                    "pareto": f["pareto"]} for f in folds]},
                        pareto_path(results_dir, method, subject, session, seed, protocol))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--method", choices=sorted(METHODS), required=True,
                   help="stability -> ga_nsga, accuracy_only -> ga_single")
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
    p.add_argument("--pop-size", type=int, default=None, help="override config.ga.pop_size "
                   "(changes config_hash)")
    p.add_argument("--n-generations", type=int, default=None, help="override config.ga."
                   "n_generations (changes config_hash)")
    p.add_argument("--allow-standin-fbcsp", action="store_true",
                   help="use experiments/_standin_fbcsp.py while src.fbcsp is a stub (dev only)")
    p.add_argument("--overwrite", action="store_true", help="recompute finished runs")
    a = p.parse_args(argv)

    from joblib import Parallel, delayed, parallel_config

    cfg = load_config()
    if a.pop_size:
        cfg["ga"]["pop_size"] = a.pop_size
    if a.n_generations:
        cfg["ga"]["n_generations"] = a.n_generations
    method = METHODS[a.method]
    cache_dir = Path(a.cache_dir) if a.cache_dir else Path(cfg["paths"]["cache_dir"])
    results_dir = Path(a.results_dir) if a.results_dir else Path(cfg["paths"]["results_dir"])
    seeds = a.seeds if a.seeds is not None else list(cfg["ga"]["seeds"])
    sessions = a.sessions or list(cfg["dataset"]["sessions"])
    C.resolve_fbcsp(a.allow_standin_fbcsp)  # fail fast if fbcsp is a stub and not allowed

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
                    _partial_path(results_dir, method, subject, session, seed, a.protocol,
                                  k).unlink(missing_ok=True)
            key = (subject, session, seed)
            runs[key] = {"n": n_folds, "folds": []}
            jobs += [(subject, session, seed, k) for k in range(n_folds)]

    print(f"{method} {a.protocol}: {len(runs)} runs, {len(jobs)} fold jobs, "
          f"pop={cfg['ga']['pop_size']} gens={cfg['ga']['n_generations']}", flush=True)
    if not jobs:
        return 0
    t0 = time.perf_counter()
    with parallel_config(backend="loky", inner_max_num_threads=1):
        gen = Parallel(n_jobs=a.n_jobs, return_as="generator_unordered")(
            delayed(_job)(method, s, ses, seed, a.protocol, k, cfg, str(cache_dir),
                          str(results_dir), a.allow_standin_fbcsp)
            for s, ses, seed, k in jobs)
        for (s, ses, seed, k), rec in gen:
            run = runs[(s, ses, seed)]
            run["folds"].append(rec)
            print(f"[fold] {method} sub-{s} ses-{ses} seed-{seed} fold-{k}: "
                  f"ba={rec['fold_record']['ba']:.3f} bits={rec['fold_record']['chosen_bits']} "
                  f"evals={rec['n_evals']} {rec['train_time_s']:.1f}s", flush=True)
            if len(run["folds"]) == run["n"]:
                path = assemble(method, s, ses, seed, a.protocol, run["folds"], cfg, results_dir)
                print(f"[run ] wrote {path}", flush=True)
    print(f"total wall time {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


def _job(method, subject, session, seed, protocol, fold, cfg, cache_dir, results_dir,
         allow_standin):
    rec = run_fold(method, subject, session, seed, protocol, fold, cfg, cache_dir,
                   results_dir, allow_standin)
    return (subject, session, seed, fold), rec


if __name__ == "__main__":
    sys.exit(main())

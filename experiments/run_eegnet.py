"""Run EEGNet on the same outer splits as the FBCSP/GA methods.

``--channels 32`` -> ``results/eegnet_32ch/`` (all EEG channels),
``--channels 8``  -> ``results/eegnet_8ch/``  (the 8 FBCSP candidate channels,
``cache.candidate_channel_index``). Use ``--method-name`` to write under a
different method directory (e.g. the contract name ``eegnet``).

Each (channels, subject, session, seed) run writes one results JSON
(CONTRACTS.md §i) as soon as it finishes; finished runs are skipped on re-run.

Example::

    python -m experiments.run_eegnet --channels 32 8 --subjects sub-17..sub-22 \
        --protocol block --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

from src import cache, results
from src.config import config_hash, load_config

from . import _common as C
from .common import git_commit, resolve_cache_dir

METHOD_NAMES = {"32": "eegnet_32ch", "8": "eegnet_8ch"}


def run_one(method: str, n_ch: str, subject: str, session: str, seed: int, protocol: str,
            cfg: dict[str, Any], cache_dir: Path, results_dir: Path) -> Path:
    from src.eegnet import evaluate_eegnet

    d, splits = C.load_unit("eegnet", subject, session, cfg, cache_dir, protocol, seed)
    ch_idx = None if n_ch == "32" else cache.candidate_channel_index(d["ch_names"], cfg)
    run_cfg = copy.deepcopy(cfg)
    run_cfg["eegnet"]["seed"] = int(seed)
    r = evaluate_eegnet(d["X"], d["y"], splits, ch_idx=ch_idx, cfg=run_cfg)
    folds = [results.make_fold_record(k, ba, None, n_features=r["n_params"],
                                      best_epoch=r["best_epoch"][k], n_epochs=r["n_epochs"][k],
                                      **C.prediction_keys(tr, te, r["y_pred"][k], d))
             for k, (ba, (tr, te)) in enumerate(zip(r["per_fold"], splits))]
    rec = {
        "method": method, "subject": subject, "session": session, "seed": int(seed),
        "protocol": protocol, "per_fold": folds, "ba_mean": r["ba_mean"], "ba_std": r["ba_std"],
        "train_time_s": r["train_time_s"], "infer_time_ms": r["infer_time_ms"],
        "config_hash": config_hash(cfg),   # hash of the design config, not the per-run seed
        "n_channels": int(d["X"].shape[1] if ch_idx is None else len(ch_idx)),
        "channels": [str(c) for c in (d["ch_names"] if ch_idx is None else d["ch_names"][ch_idx])],
        "infer_time_ms_batch1": r["infer_time_ms_batch1"],
        "device": r["device"], "git_commit": git_commit(), **C.provenance(d),
    }
    return results.write_result(rec, cfg, results_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--channels", nargs="+", choices=sorted(METHOD_NAMES), default=["32", "8"])
    p.add_argument("--subjects", nargs="+", required=True, help="e.g. sub-17..sub-22 03 05")
    p.add_argument("--protocol", choices=["block", "cross_session", "random"], default="block",
                   help="random = stratified 5-fold ignoring blocks, seed = split seed")
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None, help="default: config.ga.seeds")
    p.add_argument("--method-name", default=None,
                   help="override the method/directory name (only with one --channels value)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel runs (they share the GPU; 1-2 is sensible on 6 GB)")
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override config.eegnet.max_epochs (smoke tests; changes config_hash)")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (default: cuda if available; not part of config_hash)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.method_name and len(a.channels) != 1:
        p.error("--method-name needs exactly one --channels value")

    cfg = load_config()
    if a.max_epochs:
        cfg["eegnet"]["max_epochs"] = a.max_epochs
    if a.device:
        cfg["eegnet"]["device"] = a.device
    cache_dir = resolve_cache_dir(a.cache_dir, cfg)
    results_dir = Path(a.results_dir) if a.results_dir else Path(cfg["paths"]["results_dir"])
    seeds = a.seeds if a.seeds is not None else list(cfg["ga"]["seeds"])
    sessions = a.sessions or list(cfg["dataset"]["sessions"])
    units = C.recordings(a.protocol, C.parse_subjects(a.subjects), sessions, "eegnet", cfg,
                         cache_dir)

    jobs = []
    for n_ch in a.channels:
        method = a.method_name or METHOD_NAMES[n_ch]
        for subject, session in units:
            for seed in seeds:
                path = results.result_path(method, subject, session, seed, a.protocol, cfg,
                                           results_dir)
                if path.exists() and not a.overwrite:
                    print(f"[done] {path}")
                    continue
                jobs.append((method, n_ch, subject, session, seed))
    print(f"eegnet {a.protocol}: {len(jobs)} runs", flush=True)

    t0 = time.perf_counter()
    if a.n_jobs == 1:
        it = (run_one(*j, a.protocol, cfg, cache_dir, results_dir) for j in jobs)
    else:
        from joblib import Parallel, delayed
        it = Parallel(n_jobs=a.n_jobs, return_as="generator_unordered")(
            delayed(run_one)(*j, a.protocol, cfg, cache_dir, results_dir) for j in jobs)
    for path in it:
        r = json.loads(Path(path).read_text())
        print(f"[run ] {path}  ba={r['ba_mean']:.3f}±{r['ba_std']:.3f} "
              f"train={r['train_time_s']:.1f}s infer={r['infer_time_ms']:.2f}ms/trial "
              f"device={r['device']}", flush=True)
    print(f"total wall time {time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the fbcsp / eegnet / view caches (CONTRACTS.md §a–c) for every recording.

Recordings are processed in parallel with joblib. Afterwards
``<cache_dir>/manifest.csv`` is (re)written from the cache files on disk:
one row per (subject, session) with the per-class trial counts and the number
of rejected trials; recordings that failed are listed with ``status=failed``.

Usage (from the repo root)::

    python -m scripts.build_cache                    # all recordings, 2 workers
    python -m scripts.build_cache --subjects 01 --n-jobs 1
    python -m scripts.build_cache --overwrite --n-jobs 3

Each worker holds one 1000 Hz recording (~0.7 GB float64) plus filter
buffers; on an 8 GB machine keep ``--n-jobs`` at 2-3.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import cache, io as dio, preprocess  # noqa: E402
from src.config import config_hash, load_config  # noqa: E402

MANIFEST_COLS = ["subject", "session", "status", "n_trials", "n_circle", "n_square",
                 "n_pentagram", "n_rejected", "n_out_of_range", "n_block0", "n_block1",
                 "n_block2", "n_block3", "rejected_trial_ids", "config_hash", "error"]


def manifest_row(subject: str, session: str, cfg) -> dict:
    d = cache.load_cache("fbcsp", subject, session, cfg)
    inv = {v: k for k, v in cfg["classes"].items()}
    counts = np.bincount(d["y"], minlength=3)
    blocks = np.bincount(d["block"], minlength=4)
    row = dict(subject=subject, session=session, status="ok", n_trials=int(d["y"].size),
               n_rejected=int(d.get("n_rejected", -1)),
               n_out_of_range=int(d.get("n_out_of_range", -1)),
               rejected_trial_ids=" ".join(map(str, d.get("rejected_trial_id", []))),
               config_hash=config_hash(cfg), error="")
    for c in range(3):
        row[f"n_{inv[c]}"] = int(counts[c])
    for b in range(4):
        row[f"n_block{b}"] = int(blocks[b])
    return row


def write_manifest(cfg, failures: dict[tuple[str, str], str]) -> Path:
    rows = []
    for sub, ses in cache.list_cached("fbcsp", cfg):
        if all(cache.cache_path(k, sub, ses, cfg).exists() for k in cache.KINDS) \
                and (sub, ses) not in failures:
            rows.append(manifest_row(sub, ses, cfg))
    for (sub, ses), err in failures.items():
        rows.append(dict(subject=sub, session=ses, status="failed",
                         error=err.splitlines()[0] if err else ""))
    df = pd.DataFrame(rows, columns=MANIFEST_COLS).sort_values(["subject", "session"])
    out = Path(cfg["paths"]["cache_dir"]) / "manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-jobs", type=int, default=2)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--subjects", nargs="*", default=None, help="e.g. 01 02 (default: all)")
    p.add_argument("--cache-dir", default=None, help="default: $AA_CACHE_DIR or config")
    a = p.parse_args(argv)
    if a.cache_dir:
        os.environ["AA_CACHE_DIR"] = str(Path(a.cache_dir).resolve())
    cfg = load_config()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    pairs = dio.list_subject_sessions(cfg)
    if a.subjects is not None:
        want = {s.zfill(2) for s in a.subjects}
        pairs = [ps for ps in pairs if ps[0] in want]
    logging.info("data %s -> cache %s; %d recordings, n_jobs=%d", cfg["paths"]["data_dir"],
                 cfg["paths"]["cache_dir"], len(pairs), a.n_jobs)
    rep = preprocess.build_all_report(cfg, n_jobs=a.n_jobs, overwrite=a.overwrite, pairs=pairs)
    failures = {(r["subject"], r["session"]): r["error"] for r in rep if not r["ok"]}
    for r in rep:
        if r["ok"]:
            logging.info("sub-%s ses-%s ok (%.0f s)", r["subject"], r["session"], r["time_s"])
        else:
            logging.error("sub-%s ses-%s FAILED: %s", r["subject"], r["session"], r["error"])
    out = write_manifest(cfg, failures)
    df = pd.read_csv(out, dtype={"subject": str, "session": str})
    print(df.drop(columns=["error", "config_hash"]).to_string(index=False))
    print(f"manifest: {out}; failed: {sorted(failures) or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

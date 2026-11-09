"""EEGNet training diagnostic (P5 §5b; exploratory, does not replace the P4 files).

Re-runs EEGNet with per-epoch logging and three recipe variants, on the same
outer splits as P4, seed 0 only:

* ``logged``: the P4 recipe (§k2) plus logging. ``log_history`` is a runtime key,
  so ``config_hash`` stays ``08e124167b``, and the predictions must equal the P4
  files (checked in ``summarise``).
* ``val03``: ``val_fraction`` 0.3 instead of 0.2 (early stopping as in P4).
* ``fixed100`` / ``fixed300``: no early stopping. Train exactly 100 / 300 epochs and
  keep the final weights (``early_stopping_patience = max_epochs``,
  ``restore_best = False``).

Variants other than ``logged`` change ``config_hash`` and are labelled
``"exploratory": true``. Method names: ``eegnet_{32,8}ch_{variant}``.

Outputs:

* ``{out}/{method}/sub-XX_ses-YY_seed-0.json``: contract result records, plus
  per-fold ``train_ba``, ``val_ba``, ``best_epoch`` and ``n_epochs``.
* ``{history_dir}/{method}/sub-XX_ses-YY_seed-0.json``: full per-epoch curves
  (train/val loss and BA). Large, so they go to the git-ignored ``results/raw/``.
* ``summarise``: ``{out}/curves.csv`` (mean curves per variant), ``summary.json``
  and ``summary.md``.

Usage::

    AA_HOST=R07 python -m experiments.eegnet_diag --out results/p5/eegnet_diag
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import analysis as A, cache, results
from src.config import config_hash, load_config

from . import _common as C
from .common import git_commit, resolve_cache_dir

VARIANTS: dict[str, dict[str, Any]] = {
    "logged": {},
    "val03": {"val_fraction": 0.3},
    "fixed100": {"max_epochs": 100, "early_stopping_patience": 100, "restore_best": False},
    "fixed300": {"max_epochs": 300, "early_stopping_patience": 300, "restore_best": False},
}
CHANNELS = {"32": "eegnet_32ch", "8": "eegnet_8ch"}


def variant_cfg(cfg: dict[str, Any], variant: str, seed: int,
                max_epochs_cap: int | None = None) -> dict[str, Any]:
    c = copy.deepcopy(cfg)
    c["eegnet"].update(VARIANTS[variant])
    if max_epochs_cap:  # smoke tests only
        c["eegnet"]["max_epochs"] = min(int(c["eegnet"]["max_epochs"]), max_epochs_cap)
        c["eegnet"]["early_stopping_patience"] = min(int(c["eegnet"]["early_stopping_patience"]),
                                                     max_epochs_cap)
    c["eegnet"]["seed"] = int(seed)
    c["eegnet"]["log_history"] = True
    return c


def run_one(variant: str, n_ch: str, subject: str, session: str, protocol: str, seed: int,
            cfg: dict[str, Any], cache_dir: Path, out: Path, history_dir: Path,
            max_epochs_cap: int | None = None) -> Path:
    from src.eegnet import evaluate_eegnet
    method = f"{CHANNELS[n_ch]}_{variant}"
    dst = results.result_path(method, subject, session, seed, protocol, cfg, out)
    if dst.exists():
        return dst
    d, splits = C.load_unit("eegnet", subject, session, cfg, cache_dir)
    ch_idx = None if n_ch == "32" else cache.candidate_channel_index(d["ch_names"], cfg)
    vcfg = variant_cfg(cfg, variant, seed, max_epochs_cap)
    r = evaluate_eegnet(d["X"], d["y"], splits, ch_idx=ch_idx, cfg=vcfg)
    folds = [results.make_fold_record(
        k, ba, None, n_features=r["n_params"], best_epoch=r["best_epoch"][k],
        n_epochs=r["n_epochs"][k], train_ba=r["train_ba"][k], val_ba=r["val_ba"][k],
        **C.prediction_keys(tr, te, r["y_pred"][k], d))
        for k, (ba, (tr, te)) in enumerate(zip(r["per_fold"], splits))]
    rec = {"method": method, "subject": subject, "session": session, "seed": int(seed),
           "protocol": protocol, "per_fold": folds, "ba_mean": r["ba_mean"], "ba_std": r["ba_std"],
           "train_time_s": r["train_time_s"], "infer_time_ms": r["infer_time_ms"],
           "config_hash": config_hash(vcfg), "variant": variant, "recipe": VARIANTS[variant],
           "exploratory": variant != "logged", "device": r["device"], "git_commit": git_commit(),
           **C.provenance(d)}
    path = results.write_result(rec, cfg, out)
    h = history_dir / method / dst.name
    h.parent.mkdir(parents=True, exist_ok=True)
    h.write_text(json.dumps({"method": method, "subject": subject, "session": session,
                             "protocol": protocol, "folds": r["history"]}), encoding="utf-8")
    return path


def summarise(out: Path, history_dir: Path, p4_root: Path) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # variants have their own config hashes
        df = A.load_results(out)
        p4 = A.load_results(p4_root)
    p4 = p4[p4.method.isin(CHANNELS.values()) & (p4.seed == 0)]
    summary: dict[str, Any] = {"reproduces_p4": {}, "variants": {}}

    # 1. logged == P4 (same predictions, seed 0)
    for ch in CHANNELS.values():
        lg = df[df.method == f"{ch}_logged"].set_index(["subject", "session", "protocol"])
        ref = p4[p4.method == ch].set_index(["subject", "session", "protocol"])
        same = sum([f["y_pred"] for f in lg.loc[i, "per_fold"]] ==
                   [f["y_pred"] for f in ref.loc[i, "per_fold"]] for i in lg.index)
        summary["reproduces_p4"][ch] = {"identical": int(same), "n": int(len(lg))}

    # 2. per variant: test / train / val BA, best_epoch == 0, epochs, paired test vs P4 seed 0
    rows = []
    for m in sorted(df.method.unique()):
        ch, variant = m.rsplit("_", 1)
        for prot in ("block", "cross_session"):
            sel = df[(df.method == m) & (df.protocol == prot)]
            if sel.empty:
                continue
            ex = [f for folds in sel.per_fold for f in folds]
            wide_v = A.subject_level(sel, prot)[m]
            wide_p = A.subject_level(p4[p4.method == ch], prot)[ch].reindex(wide_v.index)
            d = (wide_v - wide_p).dropna()
            p = float(stats.wilcoxon(d).pvalue) if (d != 0).any() else 1.0
            row = {"method": m, "channels": ch, "variant": variant, "protocol": prot,
                   "n_subjects": int(len(wide_v)), "n_folds": len(ex),
                   "test_ba": float(wide_v.mean()), "test_ba_p4_seed0": float(wide_p.mean()),
                   "diff_vs_p4": float(d.mean()), "p_vs_p4": p,
                   "train_ba": float(np.mean([f["train_ba"] for f in ex])),
                   "val_ba": float(np.mean([f["val_ba"] for f in ex])),
                   "fold_test_ba": float(np.mean([f["ba"] for f in ex])),
                   "best_epoch0_frac": float(np.mean([f["best_epoch"] == 0 for f in ex])),
                   "best_epoch_median": float(np.median([f["best_epoch"] for f in ex])),
                   "n_epochs_median": float(np.median([f["n_epochs"] for f in ex])),
                   "train_time_s_mean": float(sel.train_time_s.mean())}
            rows.append(row)
            summary["variants"][f"{m}|{prot}"] = row
    tab = pd.DataFrame(rows)
    tab.to_csv(out / "variants.csv", index=False)

    # 3. mean learning curves (block protocol), per method and epoch
    curves = []
    for hfile in sorted(history_dir.glob("*/*.json")):
        h = json.loads(hfile.read_text(encoding="utf-8"))
        if h["protocol"] != "block":
            continue
        for k, fold in enumerate(h["folds"] or []):
            for e in fold:
                curves.append({"method": h["method"], **e})
    if curves:
        cv = pd.DataFrame(curves).groupby(["method", "epoch"]).agg(
            train_loss=("train_loss", "mean"), val_loss=("val_loss", "mean"),
            train_ba=("train_ba", "mean"), val_ba=("val_ba", "mean"),
            n_folds=("train_ba", "size")).reset_index()
        cv.to_csv(out / "curves.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = ["method", "protocol", "n_subjects", "test_ba", "test_ba_p4_seed0", "diff_vs_p4",
            "p_vs_p4", "train_ba", "val_ba", "best_epoch0_frac", "best_epoch_median",
            "n_epochs_median"]
    (out / "summary.md").write_text(
        "# EEGNet diagnostic (exploratory; seed 0)\n\nReproduces P4: "
        + json.dumps(summary["reproduces_p4"]) + "\n\n```\n"
        + tab[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n```\n",
        encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--channels", nargs="+", choices=list(CHANNELS), default=list(CHANNELS))
    p.add_argument("--subjects", nargs="+", default=["01..22"])
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--protocols", nargs="+", choices=["block", "cross_session"],
                   default=["block", "cross_session"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/p5/eegnet_diag")
    p.add_argument("--history-dir", default="results/raw/p5_eegnet_history")
    p.add_argument("--p4-root", default="results", help="holds the P4 eegnet_32ch/ eegnet_8ch/")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--max-epochs-cap", type=int, default=None, help="smoke tests only")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel runs sharing the GPU (about 300 MB each)")
    p.add_argument("--no-summary", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config()
    cache_dir = resolve_cache_dir(a.cache_dir, cfg)
    out, hdir = Path(a.out), Path(a.history_dir)
    subjects = C.parse_subjects(a.subjects)
    sessions = a.sessions or list(cfg["dataset"]["sessions"])
    # Longest jobs first (fixed300 > fixed100 > early-stopped; 32ch > 8ch) so parallel
    # workers finish together. Every run is seeded and independent, so n_jobs does not
    # change any result (only the timing fields, which share the GPU).
    cost = {"fixed300": 3, "fixed100": 2}
    jobs = sorted(((v, ch, s, ses, prot) for v in a.variants for prot in a.protocols
                   for s, ses in C.recordings(prot, subjects, sessions, "eegnet", cfg, cache_dir)
                   for ch in a.channels),
                  key=lambda j: (-cost.get(j[0], 1), j[1] != "32"))
    t0, n = time.perf_counter(), 0
    if a.n_jobs == 1:
        it = (run_one(v, ch, s, ses, prot, a.seed, cfg, cache_dir, out, hdir, a.max_epochs_cap)
              for v, ch, s, ses, prot in jobs)
    else:
        from joblib import Parallel, delayed
        it = Parallel(n_jobs=a.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(run_one)(v, ch, s, ses, prot, a.seed, cfg, cache_dir, out, hdir,
                             a.max_epochs_cap) for v, ch, s, ses, prot in jobs)
    for path in it:
        n += 1
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"[run ] {n}/{len(jobs)} {r['method']} {r['protocol']} sub-{r['subject']} "
              f"ses-{r['session']}: ba={r['ba_mean']:.3f} ({time.perf_counter() - t0:.0f}s)",
              flush=True)
    print(f"{n} runs", flush=True)
    if not a.no_summary:
        summarise(out, hdir, Path(a.p4_root))
        print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

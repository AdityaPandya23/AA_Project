"""Generator of FAKE result JSONs (CONTRACTS.md §i) for developing and testing src/analysis.py.

Each file is written through ``results.write_result``, so it passes the contract
validator. Built-in structure, so the tests can check that analysis recovers it:

* every subject has a latent skill; methods add fixed offsets
  (``ga_nsga`` > ``ga_single`` > ``fbcsp_all`` > ``eegnet``);
* subject ``"01"`` is strong (well above chance) and subject ``"02"`` is at chance;
* cross-session BA is about 0.05 below within-session BA;
* GA methods choose O1/O2/alpha most often, and ``ga_nsga`` is more stable
  (less gene noise) than ``ga_single``;
* ``ga_nsga`` folds carry a ``pareto`` list;
* the last subject has only ses-01 (so it is skipped for cross-session).

Usage::

    python -m tests.fake_results --out results_fake
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src import chromosome as C, results
from src.config import config_hash, load_config

METHODS = ("fbcsp_all", "ga_single", "ga_nsga", "eegnet", "eegnet_gach")
OFFSET = {"fbcsp_all": 0.0, "ga_single": 0.03, "ga_nsga": 0.05, "eegnet": -0.02,
          "eegnet_gach": -0.03}
GENE_NOISE = {"ga_single": 0.30, "ga_nsga": 0.10}
SEEDED = {"ga_single", "ga_nsga", "eegnet", "eegnet_gach"}
PREFERRED = np.zeros(19, np.uint8)
PREFERRED[[0, 2, 9, 13, 14, 15]] = 1  # O1, O2, alpha, windows 0.5-1.5 s


def _labels(rng, n_blocks, per_class=10):
    yb = [rng.permutation(np.repeat([0, 1, 2], per_class)) for _ in range(n_blocks)]
    return yb


def _predict(rng, y, p_correct):
    yp = y.copy()
    wrong = rng.random(y.size) > p_correct
    yp[wrong] = (y[wrong] + rng.integers(1, 3, wrong.sum())) % 3
    return yp


def _ba(y, yp):
    return float(np.mean([np.mean(yp[y == k] == k) for k in np.unique(y)]))


def _bits(rng, method):
    if method == "fbcsp_all":
        return C.all_on()
    if method not in GENE_NOISE:
        return None
    flip = rng.random(19) < GENE_NOISE[method]
    return C.repair(np.where(flip, 1 - PREFERRED, PREFERRED).astype(np.uint8), rng)


def _pareto(rng, chosen):
    out = [{"bits": C.to_str(chosen), "ba_mean": float(rng.uniform(0.45, 0.6)),
            "ba_std": float(rng.uniform(0.02, 0.08)), "cost": C.cost(chosen)}]
    for _ in range(int(rng.integers(4, 9))):
        b = C.random_valid(rng, p=float(rng.uniform(0.2, 0.9)))
        c = C.cost(b)
        out.append({"bits": C.to_str(b), "ba_mean": float(np.clip(0.35 + 0.2 * c + rng.normal(0, 0.03), 0, 1)),
                    "ba_std": float(rng.uniform(0.02, 0.12)), "cost": c})
    return out


def _record(rng, cfg, method, sub, ses, seed, protocol, p_correct):
    n_blocks = 1 if protocol == "cross_session" else 4
    folds = []
    # The same trials (labels) for every method and seed of a recording, as with real splits.
    lab_rng = np.random.default_rng([int(sub), int(ses[:2]), len(ses), n_blocks])
    for k, y in enumerate(_labels(lab_rng, n_blocks, 30 if protocol == "cross_session" else 10)):
        yp = _predict(rng, y, p_correct)
        bits = _bits(rng, method)
        extra = {"n_test": int(y.size), "y_true": y.tolist(), "y_pred": yp.tolist()}
        if method == "ga_nsga":
            extra["pareto"] = _pareto(rng, bits)
        nf = None if bits is not None else (2000 if method == "eegnet" else 900)
        folds.append(results.make_fold_record(k, _ba(y, yp), bits, n_features=nf, **extra))
    bas = [f["ba"] for f in folds]
    return {"method": method, "subject": sub, "session": ses, "seed": seed, "protocol": protocol,
            "per_fold": folds, "ba_mean": float(np.mean(bas)), "ba_std": float(np.std(bas)),
            "train_time_s": float(rng.uniform(1, 5) * (60 if method.startswith("ga") else 1)),
            "infer_time_ms": float(rng.uniform(0.2, 2.0)), "config_hash": config_hash(cfg),
            "fake": True}


def write_fake_results(out_dir, subjects=("01", "02", "03", "04", "05", "06", "07", "08"),
                       methods=METHODS, seeds=(0, 1, 2), seed: int = 0) -> list[Path]:
    """Write fake result JSONs into ``out_dir/{method}/``. Returns the written paths."""
    cfg = load_config()
    rng = np.random.default_rng(seed)
    skill = {s: float(rng.uniform(0.42, 0.62)) for s in subjects}
    skill[subjects[0]] = 0.80
    if len(subjects) > 1:
        skill[subjects[1]] = 1 / 3
    paths = []
    for sub in subjects:
        sessions = ("01",) if sub == subjects[-1] else ("01", "02")
        for method in methods:
            for sd in (seeds if method in SEEDED else (0,)):
                p = float(np.clip(skill[sub] + OFFSET[method], 0.2, 0.98))
                if sub == subjects[1] and len(subjects) > 1:
                    p = 1 / 3  # chance subject for every method
                for ses in sessions:
                    paths.append(results.write_result(
                        _record(rng, cfg, method, sub, ses, sd, "block", p), cfg, out_dir))
                if len(sessions) == 2:
                    pc = max(p - 0.05, 0.2) if p > 1 / 3 else p
                    paths.append(results.write_result(
                        _record(rng, cfg, method, sub, "01to02", sd, "cross_session", pc), cfg, out_dir))
    return paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results_fake")
    a = ap.parse_args(argv)
    print(f"wrote {len(write_fake_results(a.out))} fake result files to {a.out}")


if __name__ == "__main__":
    main()

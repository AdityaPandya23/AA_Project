"""Time ``fbcsp.evaluate`` as the GA calls it (inner 3-fold CV on one outer training set).

Usage::

    python -m experiments.time_evaluate --cache-dir cache_synth [--subject 91 --session 01]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src import chromosome as C, cv, fbcsp
from src.config import load_config

from .common import load, resolve_cache_dir


def _time(fn, n):
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return np.array(t) * 1e3


def run(cache_dir=None, subject=None, session=None, n_random: int = 300, seed: int = 0) -> dict:
    cfg = load_config()
    cache_dir = resolve_cache_dir(cache_dir, cfg)
    if subject is None:
        from src.cache import list_cached
        subject, session = list_cached("fbcsp", cfg, cache_dir)[0]
    d = load("fbcsp", subject, session, cfg, cache_dir)
    X, y, block = d["X"], d["y"], d["block"]
    tr, _ = cv.block_splits(block)[0]
    inner = cv.inner_splits(y, block, tr, cfg["ga"]["inner_folds"], seed)
    rng = np.random.default_rng(seed)
    rand_bits = [C.random_valid(rng) for _ in range(n_random)]

    fbcsp.clear_cache()
    t0 = time.perf_counter()
    fbcsp.evaluate(X, y, rand_bits[0], inner)
    cold_ms = (time.perf_counter() - t0) * 1e3
    it = iter(rand_bits)
    rand_ms = _time(lambda: fbcsp.evaluate(X, y, next(it), inner), n_random)
    minimal = C.repair(np.zeros(19, np.uint8), rng)
    out = {
        "recording": f"sub-{subject}_ses-{session}",
        "n_trials_outer_train": int(tr.size),
        "inner_folds": len(inner),
        "cold_first_call_ms": cold_ms,
        "random_p0.5": {"median_ms": float(np.median(rand_ms)), "mean_ms": float(rand_ms.mean()),
                        "p95_ms": float(np.percentile(rand_ms, 95)),
                        "mean_n_features": float(np.mean([C.n_features(b) for b in rand_bits]))},
        "all_on_median_ms": float(np.median(_time(lambda: fbcsp.evaluate(X, y, C.all_on(), inner), 30))),
        "minimal_median_ms": float(np.median(_time(lambda: fbcsp.evaluate(X, y, minimal, inner), 50))),
    }
    fbcsp.CACHE_COVARIANCES = False
    try:
        out["random_no_cache_median_ms"] = float(np.median(
            _time(lambda: fbcsp.evaluate(X, y, rand_bits[1], inner), 20)))
    finally:
        fbcsp.CACHE_COVARIANCES = True
    # Per-trial inference latency on the public path (raw Xsel -> label).
    Xs = C.select(X, C.all_on())
    est = fbcsp.FBCSPLDA().fit(Xs[tr], y[tr])
    one = Xs[:1]
    est.predict(one)
    out["predict_one_trial_all_on_ms"] = float(np.median(_time(lambda: est.predict(one), 200)))
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--subject", default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--n-random", type=int, default=300)
    a = p.parse_args(argv)
    print(json.dumps(run(a.cache_dir, a.subject, a.session, a.n_random), indent=2))


if __name__ == "__main__":
    main()

"""Training vs inner-CV vs test balanced accuracy (P5 §5a; exploratory).

For ``fbcsp_all``, ``ga_nsga`` and ``ga_single``, block and cross-session, every
outer fold of the committed results:

* ``train_ba``: FBCSP+LDA fitted on the outer training set with the committed
  ``chosen_bits`` (all 19 for ``fbcsp_all``) and scored on the **same** training
  trials (resubstitution);
* ``inner_ba``: the inner-CV BA the GA saw for its chosen chromosome (from the
  stored Pareto front). For ``fbcsp_all``, the inner-CV BA of the all-on
  chromosome on the same 3-fold ``cv.inner_splits`` (seed 0);
* ``test_ba``: the outer test BA, re-scored and checked against the stored value.

Usage::

    python -m reports.train_accuracy --results results --out results/p5/train_accuracy
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from experiments import _common as C
from experiments.common import resolve_cache_dir
from src import analysis as A, chromosome as CH, cv, fbcsp
from src.config import load_config

METHODS = ("fbcsp_all", "ga_nsga", "ga_single")


def _unit(subject: str, session: str, protocol: str, recs: list[dict], cfg, cache_dir) -> list[dict]:
    d, splits = C.load_unit("fbcsp", subject, session, cfg, Path(cache_dir))
    X, y, block = d["X"], d["y"], d["block"]
    n_inner = int(cfg["ga"]["inner_folds"])
    rows = []
    with fbcsp._blas_single_thread():
        for rec in recs:
            for f in rec["per_fold"]:
                k = int(f["fold"])
                tr, te = splits[k]
                bits = CH.all_on() if rec["method"] == "fbcsp_all" else CH.from_str(f["chosen_bits"])
                Cv = fbcsp.select_covariances(X, bits)
                est = fbcsp.FBCSPLDA()._fit_cov(Cv[tr], y[tr])
                test_ba = fbcsp.balanced_accuracy(y[te], est._predict_cov(Cv[te]))
                if abs(test_ba - f["ba"]) > 1e-9:
                    raise AssertionError(f"{rec['method']} sub-{subject} ses-{session} fold {k}: "
                                         f"re-scored test BA {test_ba} != stored {f['ba']}")
                if rec["method"] == "fbcsp_all":
                    inner = fbcsp.evaluate(X, y, bits, cv.inner_splits(y, block, tr, n=n_inner,
                                                                         seed=0))["ba_mean"]
                else:
                    inner = next(e["ba_mean"] for e in f["pareto"] if e["bits"] == f["chosen_bits"])
                rows.append({"method": rec["method"], "subject": subject, "session": session,
                             "seed": int(rec["seed"]), "protocol": protocol, "fold": k,
                             "train_ba": fbcsp.balanced_accuracy(y[tr], est._predict_cov(Cv[tr])),
                             "inner_ba": float(inner), "test_ba": float(test_ba),
                             "n_features": CH.n_features(bits), "n_train": int(len(tr))})
    return rows


def run(results_root: Path, cache_dir: Path, n_jobs: int = 8) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = A.load_results([results_root, results_root / "baseline_all"])
    df = df[df.method.isin(METHODS) & df.protocol.isin(["block", "cross_session"])]
    cfg = load_config()
    groups = [(s, ses, prot, g[["method", "seed", "per_fold"]].to_dict("records"))
              for (s, ses, prot), g in df.groupby(["subject", "session", "protocol"])]
    out = Parallel(n_jobs=n_jobs)(delayed(_unit)(s, ses, prot, recs, cfg, str(cache_dir))
                                  for s, ses, prot, recs in groups)
    return pd.DataFrame([r for rows in out for r in rows])


def summarise(fl: pd.DataFrame) -> pd.DataFrame:
    """Subject level (folds -> seeds -> sessions), then mean ± SD over subjects."""
    num = ["train_ba", "inner_ba", "test_ba", "n_features"]
    sub = (fl.groupby(["method", "protocol", "subject", "session", "seed"])[num].mean()
           .groupby(["method", "protocol", "subject", "session"]).mean()
           .groupby(["method", "protocol", "subject"]).mean())
    agg = sub.groupby(["method", "protocol"]).agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg["n_subjects"] = sub.groupby(["method", "protocol"]).size()
    agg["train_minus_test"] = agg["train_ba_mean"] - agg["test_ba_mean"]
    agg["inner_minus_test"] = agg["inner_ba_mean"] - agg["test_ba_mean"]
    return agg.reset_index()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="results/p5/train_accuracy")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--n-jobs", type=int, default=8)
    a = p.parse_args(argv)
    cfg = load_config()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fl = run(Path(a.results), resolve_cache_dir(a.cache_dir, cfg), a.n_jobs)
    fl.to_csv(out / "fold_level.csv", index=False)
    s = summarise(fl)
    s.to_csv(out / "summary.csv", index=False)
    cols = ["method", "protocol", "n_subjects", "train_ba_mean", "train_ba_std", "inner_ba_mean",
            "test_ba_mean", "test_ba_std", "train_minus_test", "inner_minus_test", "n_features_mean"]
    text = s[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")
    (out / "summary.md").write_text(
        "# Training vs inner-CV vs test BA (exploratory)\n\nSubject level (folds, seeds, then "
        "sessions averaged). train = resubstitution on the outer training set; inner = GA "
        "fitness of the chosen chromosome (fbcsp_all: all-on, 3-fold inner CV, seed 0).\n\n```\n"
        + text + "\n```\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

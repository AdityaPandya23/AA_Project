"""Position-only control (P5; P2_REPORT §3, P2_REVIEW §7, DATA_NOTES §4).

Can the class be predicted from *where* a trial sits in the presentation
sequence, without any EEG? A classifier that only sees position is fitted on the
training trials and scored on the test trials of exactly the same splits as the
real methods (``fbcsp`` cache, so the same trials as ``fbcsp_all`` and the GA).

Two keys, two methods (seed 0 only, deterministic):

* ``position_block30``: position within the 30-trial block, ``trial_id % 30``
  (the control specified in DATA_NOTES §4). Block and cross-session protocols.
* ``position_seq120``: absolute position in the 120-trial session sequence,
  ``trial_id``. Cross-session only (in the block protocol a held-out block's
  positions never occur in training). It is what a pure time-in-session signal
  (slow drift, fatigue) could exploit when the label sequence repeats across
  sessions, which happens only for sub-01 (DATA_NOTES §4).

Classifier: a lookup table key -> majority class of the training trials with
that key. Ties and keys absent from training (rejected trials) are resolved
by a seeded RNG. Test labels are never used for fitting.

Outputs (exploratory, not part of ANALYSIS_PLAN):

* ``{out}/{method}/sub-XX_ses-YY_seed-0.json``: contract result records
  (CONTRACTS §i), so ``analysis.load_results`` / ``above_chance`` apply.
* ``{out}/per_subject.csv``, ``{out}/summary.json``, ``{out}/summary.md``:
  position controls next to the cross-session BA of ``fbcsp_all``, ``ga_nsga``,
  ``eegnet_32ch`` and ``eegnet_8ch``.

Usage::

    AA_HOST=R07 python -m experiments.position_control --subjects 01..22 \
        --out results/p5/position_control
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src import analysis as A, results
from src.config import config_hash, load_config
from src.fbcsp import balanced_accuracy

from . import _common as C
from .common import git_commit, resolve_cache_dir

N_CLASSES = 3
CHANCE = 1 / 3
KEYS = {"position_block30": lambda tid: np.asarray(tid) % 30,
        "position_seq120": lambda tid: np.asarray(tid)}
PROTOCOLS = {"position_block30": ("block", "cross_session"),
             "position_seq120": ("cross_session",)}
COMPARE = ("fbcsp_all", "ga_nsga", "ga_single", "eegnet_32ch", "eegnet_8ch")


def fit_lookup(keys: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[int, int]:
    """Key -> majority class among training trials with that key (ties: random)."""
    table: dict[int, int] = {}
    keys, y = np.asarray(keys), np.asarray(y)
    for k in np.unique(keys):
        counts = np.bincount(y[keys == k], minlength=N_CLASSES)
        best = np.flatnonzero(counts == counts.max())
        table[int(k)] = int(best[0] if best.size == 1 else rng.choice(best))
    return table


def predict_lookup(table: dict[int, int], keys: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    """Look up each key; keys unseen in training get a random class."""
    return np.array([table[int(k)] if int(k) in table else int(rng.integers(N_CLASSES))
                     for k in np.asarray(keys)], dtype=np.int64)


def run_unit(method: str, subject: str, session: str, protocol: str, cfg: dict[str, Any],
             cache_dir: Path, out: Path, seed: int = 0) -> Path:
    """Fit/score the position control on every outer split of one unit; write the JSON."""
    d, splits = C.load_unit("fbcsp", subject, session, cfg, cache_dir)  # "01to02" -> cross-session
    key = KEYS[method](d["trial_id"])
    folds, t_fit, t_pred = [], 0.0, 0.0
    for k, (tr, te) in enumerate(splits):
        assert np.intersect1d(tr, te).size == 0
        rng = np.random.default_rng([seed, k])
        t0 = time.perf_counter()
        table = fit_lookup(key[tr], d["y"][tr], rng)
        t1 = time.perf_counter()
        yp = predict_lookup(table, key[te], rng)
        t2 = time.perf_counter()
        t_fit, t_pred = t_fit + t1 - t0, t_pred + (t2 - t1) / len(te)
        folds.append(results.make_fold_record(
            k, balanced_accuracy(d["y"][te], yp), None, n_features=1,
            n_seen_keys=int(sum(int(v) in table for v in key[te])),
            **C.prediction_keys(tr, te, yp, d)))
    bas = [f["ba"] for f in folds]
    rec = {"method": method, "subject": subject, "session": session, "seed": int(seed),
           "protocol": protocol, "per_fold": folds,
           "ba_mean": float(np.mean(bas)), "ba_std": float(np.std(bas)),
           "train_time_s": float(t_fit), "infer_time_ms": float(1000 * t_pred / len(folds)),
           "config_hash": config_hash(cfg), "git_commit": git_commit(), **C.provenance(d)}
    return results.write_result(rec, cfg, out)


def block_matrix(d: dict[str, Any]) -> np.ndarray:
    """``(4, 30)`` labels by (block, position); ``-1`` where the trial was rejected."""
    B = np.full((4, 30), -1, dtype=np.int64)
    B[np.asarray(d["trial_id"]) // 30, np.asarray(d["trial_id"]) % 30] = d["y"]
    return B


def block_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Label agreement of two block sequences over positions present in both."""
    m = (a >= 0) & (b >= 0)
    return float(np.mean(a[m] == b[m])) if m.any() else np.nan


def n_identical_blocks(B_test: np.ndarray, B_train: np.ndarray, thr: float = 0.95) -> int:
    """How many blocks of ``B_test`` have an identical label order in ``B_train`` (any block)."""
    return int(sum(max(block_agreement(bt, br) for br in B_train) > thr for bt in B_test))


def block_sharing(subjects, cfg: dict[str, Any], cache_dir: Path) -> pd.DataFrame:
    """Per subject: ses-02 blocks whose order also occurs in ses-01, and repeats within a session."""
    rows = []
    for s in subjects:
        B = {}
        for ses in ("01", "02"):
            try:
                B[ses] = block_matrix(C.load("fbcsp", s, ses, cfg, cache_dir))
            except FileNotFoundError:
                pass
        within = {ses: sum(block_agreement(b[i], b[j]) > 0.95 for i in range(4)
                           for j in range(i + 1, 4)) for ses, b in B.items()}
        rows.append({"subject": s,
                     "n_identical_cross": n_identical_blocks(B["02"], B["01"]) if len(B) == 2 else np.nan,
                     "within_repeats_ses01": within.get("01", np.nan),
                     "within_repeats_ses02": within.get("02", np.nan)})
    return pd.DataFrame(rows).set_index("subject")


def _wilcoxon_vs_chance(v: pd.Series) -> dict[str, float]:
    v = v.dropna()
    d = v.to_numpy() - CHANCE
    p = float(stats.wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "p_greater": p}


def _agreement(df: pd.DataFrame, method: str, control: str, subject: str) -> dict[str, float]:
    """Do a method's cross-session predictions follow the position control's predictions?

    Pooled over seeds and test trials (matched by ``test_trial_id``):
    ``kappa`` = Cohen's kappa between the two prediction vectors, and
    ``follow_wrong`` = P(method predicts the control's class | control is wrong).
    Under independence ``follow_wrong`` ≈ the method's rate of predicting that wrong class.
    """
    def runs(m):
        """One {test_trial_id: (y_true, y_pred)} dict per result file (seed)."""
        sel = df[(df.method == m) & (df.subject == subject) & (df.protocol == "cross_session")]
        return [{int(t): (int(yt), int(yp)) for f in folds
                 for t, yt, yp in zip(f["test_trial_id"], f["y_true"], f["y_pred"])}
                for folds in sel.per_fold]
    ctrl = runs(control)[0]
    a, b, yt = [], [], []
    for run in runs(method):
        for tid, (y_true, y_pred) in run.items():
            if tid in ctrl:
                assert ctrl[tid][0] == y_true, "test trials disagree on the true label"
                a.append(y_pred)
                b.append(ctrl[tid][1])
                yt.append(y_true)
    a, b, yt = np.array(a), np.array(b), np.array(yt)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in range(N_CLASSES))
    wrong = b != yt
    rate = {c: np.mean(a == c) for c in range(N_CLASSES)}
    return {"kappa": float((po - pe) / (1 - pe)) if pe < 1 else np.nan,
            "follow_wrong": float(np.mean(a[wrong] == b[wrong])) if wrong.any() else np.nan,
            "follow_wrong_null": float(np.mean([rate[c] for c in b[wrong]])) if wrong.any() else np.nan,
            "n_trials": int(a.size)}


def summarise(out: Path, results_root: Path, cfg: dict[str, Any], cache_dir: Path,
              n_perm: int = 1000) -> dict[str, Any]:
    """Tables for the P5 report. Returns the summary dict (also written to ``out``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # position controls carry no cost; hashes are equal
        ctrl = A.load_results(out)
        meth = A.load_results([results_root, results_root / "baseline_all"])
    meth = meth[meth.method.isin(COMPARE)]
    df = pd.concat([ctrl, meth], ignore_index=True)

    tables, perm = {}, {}
    for prot in ("cross_session", "block"):
        wide = A.subject_level(df, prot)
        cols = [m for m in ("position_block30", "position_seq120", *COMPARE) if m in wide]
        tables[prot] = wide[cols]
        for m in cols:
            ac = A.above_chance(df, m, prot, n_perm=n_perm, seed=0)
            perm[(m, prot)] = ac
    cs = tables["cross_session"]

    summary: dict[str, Any] = {"n_perm": n_perm, "group": {}, "sub01": {}, "spearman": {},
                               "agreement_sub01": {}, "agreement_others": {}}
    for prot, wide in tables.items():
        for m in wide:
            for label, w in (("all", wide[m]), ("without_sub01", wide[m].drop("01", errors="ignore"))):
                ac = perm[(m, prot)]
                ac = ac if label == "all" else ac[ac.subject != "01"]
                summary["group"][f"{m}|{prot}|{label}"] = {
                    **_wilcoxon_vs_chance(w), "n_rec_p05": int((ac.p < 0.05).sum()),
                    "n_rec_p05_holm": int((A.holm(ac.p.to_numpy()) < 0.05).sum()),
                    "n_rec": int(len(ac))}
    for m in cs:
        ac = perm[(m, "cross_session")]
        r = ac[ac.subject == "01"].iloc[0]
        summary["sub01"][m] = {"ba": float(r.ba), "p": float(r.p)}
    for c in ("position_block30", "position_seq120"):
        for m in COMPARE:
            if m in cs:
                rho, p = stats.spearmanr(cs[c], cs[m])
                summary["spearman"][f"{c}~{m}"] = {"rho": float(rho), "p": float(p), "n": int(len(cs))}
    for c in ("position_block30", "position_seq120"):
        for m in COMPARE:
            if m not in cs:
                continue
            summary["agreement_sub01"][f"{m}~{c}"] = _agreement(df, m, c, "01")
            ag = [_agreement(df, m, c, s) for s in cs.index if s != "01"]
            summary["agreement_others"][f"{m}~{c}"] = {
                k: float(np.nanmean([g[k] for g in ag]))
                for k in ("kappa", "follow_wrong", "follow_wrong_null")}

    # Block-order reuse (extends DATA_NOTES §4) and whether it moves the methods.
    share = block_sharing(sorted(set(tables["block"].index)), cfg, cache_dir)
    summary["block_sharing"] = share.reset_index().to_dict("records")
    n_id = share.loc[cs.index, "n_identical_cross"]
    shared = n_id >= 1
    summary["sharing_moderation"] = {}
    for m in cs:
        a, b = cs[m][shared], cs[m][~shared]
        summary["sharing_moderation"][m] = {
            "mean_shared": float(a.mean()), "n_shared": int(a.size),
            "mean_shared_without_sub01": float(a.drop("01", errors="ignore").mean()),
            "mean_none": float(b.mean()), "n_none": int(b.size),
            "mwu_p": float(stats.mannwhitneyu(a, b).pvalue),
            "spearman_n_identical": float(stats.spearmanr(n_id, cs[m])[0])}
    ac = perm[("position_block30", "block")]
    summary["block_protocol_hits"] = ac[ac.p < 0.05][["subject", "session", "ba", "p", "p_holm"]] \
        .to_dict("records")

    rows = []
    for prot, wide in tables.items():
        for s, row in wide.iterrows():
            for m, v in row.items():
                ac = perm[(m, prot)]
                p = ac[ac.subject == s].p
                rows.append({"protocol": prot, "subject": s, "method": m, "ba": v,
                             "p_perm_rec_mean": float(p.mean()) if len(p) else np.nan,
                             "n_identical_cross": share.loc[s, "n_identical_cross"]})
    pd.DataFrame(rows).to_csv(out / "per_subject.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(to_markdown(tables, perm, summary), encoding="utf-8")
    return summary


def to_markdown(tables, perm, summary) -> str:
    cs = tables["cross_session"]
    p = {m: perm[(m, "cross_session")].set_index("subject").p for m in cs}
    lines = ["# Position-only control: cross-session per subject (BA, permutation p)", "",
             "| Subject | " + " | ".join(cs.columns) + " |", "|---" * (len(cs.columns) + 1) + "|"]
    for s, row in cs.iterrows():
        lines.append(f"| {s} | " + " | ".join(f"{row[m]:.3f} ({p[m][s]:.3f})" for m in cs) + " |")
    lines += ["", "# Group (subject level, one-sided Wilcoxon vs 1/3)", "",
              "| Method | Protocol | Subjects | Mean ± SD | p | Recordings p<.05 (uncorr./Holm) |",
              "|---|---|---|---|---|---|"]
    for k, g in summary["group"].items():
        m, prot, label = k.split("|")
        lines.append(f"| {m} | {prot} | {label} (n={g['n']}) | {g['mean']:.3f} ± {g['sd']:.3f} | "
                     f"{g['p_greater']:.3f} | {g['n_rec_p05']}/{g['n_rec_p05_holm']} of {g['n_rec']} |")
    lines += ["", "# Block-order reuse across sessions and the methods' cross-session BA", "",
              "| Method | ≥1 shared block (n) | w/o sub-01 | No shared block (n) | MWU p | ρ with n shared |",
              "|---|---|---|---|---|---|"]
    for m, g in summary["sharing_moderation"].items():
        lines.append(f"| {m} | {g['mean_shared']:.3f} ({g['n_shared']}) | "
                     f"{g['mean_shared_without_sub01']:.3f} | {g['mean_none']:.3f} ({g['n_none']}) | "
                     f"{g['mwu_p']:.3f} | {g['spearman_n_identical']:+.2f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--subjects", nargs="+", default=["01..22"])
    p.add_argument("--sessions", nargs="+", default=None)
    p.add_argument("--out", default="results/p5/position_control")
    p.add_argument("--results-root", default="results",
                   help="where the P3/P4 method folders are (for the comparison tables)")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--no-summary", action="store_true")
    a = p.parse_args(argv)
    cfg = load_config()
    cache_dir = resolve_cache_dir(a.cache_dir, cfg)
    out = Path(a.out)
    subjects = C.parse_subjects(a.subjects)
    sessions = a.sessions or list(cfg["dataset"]["sessions"])
    n = 0
    for method, prots in PROTOCOLS.items():
        for prot in prots:
            for subject, session in C.recordings(prot, subjects, sessions, "fbcsp", cfg, cache_dir):
                path = run_unit(method, subject, session, prot, cfg, cache_dir, out)
                n += 1
                r = json.loads(path.read_text())
                print(f"[run ] {method} {prot} sub-{subject} ses-{session}: ba={r['ba_mean']:.3f}",
                      flush=True)
    print(f"{n} result files in {out}", flush=True)
    if not a.no_summary:
        summarise(out, Path(a.results_root), cfg, cache_dir, a.n_perm)
        print((out / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

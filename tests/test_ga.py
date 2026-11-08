"""Tests for src/ga.py and experiments/run_ga.py.

Machinery tests use a cheap toy fitness with the ``fbcsp.evaluate`` signature
(``ga.set_evaluate_fn``). Synthetic-data tests use R01's real ``fbcsp.evaluate``.
"""
import json

import numpy as np
import pytest

from experiments import _common as C
from src import chromosome, fbcsp, ga, results, synth
from src.config import load_config


CFG = load_config()
O1, O2 = (CFG["channels"]["candidates"].index(c) for c in ("O1", "O2"))


def small_cfg(pop=12, gens=4):
    cfg = load_config()
    cfg["ga"]["pop_size"], cfg["ga"]["n_generations"] = pop, gens
    return cfg


@pytest.fixture(autouse=True)
def _reset_evaluate():
    yield
    ga.set_evaluate_fn(None)


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache_synth")
    synth.write_synthetic(d, CFG, subjects=("91", "92"))
    return d


def toy_data(n=120, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1, 2], n // 3).astype(np.int64)
    block = np.repeat(np.arange(4), n // 4).astype(np.int64)
    X = rng.standard_normal((4, 7, n, 8, 4)).astype(np.float32)
    return X, y, block


class RecordingEval:
    """Toy fitness with the ``fbcsp.evaluate`` signature. It rewards O1/O2 and
    records every index and chromosome it is given."""

    def __init__(self):
        self.seen_idx: set[int] = set()
        self.calls: list[str] = []

    def __call__(self, X, y, bits, splits, **kw):
        assert chromosome.is_valid(bits)
        self.calls.append(chromosome.to_str(bits))
        for tr, te in splits:
            self.seen_idx.update(map(int, tr)), self.seen_idx.update(map(int, te))
        b = np.asarray(bits)
        base = 0.34 + 0.15 * b[O1] + 0.15 * b[O2] - 0.01 * (b.sum() - b[O1] - b[O2])
        per = [base + 0.02 * k * (b[8:].sum() % 3) for k in range(len(splits))]
        return {"ba_mean": float(np.mean(per)), "ba_std": float(np.std(per)),
                "per_fold": per, "fit_time_s": 0.0}


# ----------------------------------------------------------------- choose_from_pareto
def _e(bits_str, m, s, c=None):
    b = chromosome.from_str(bits_str)
    return {"bits": b, "ba_mean": m, "ba_std": s, "cost": chromosome.cost(b) if c is None else c}


def test_choose_lowest_cost_within_1sd():
    p = [_e("1" * 19, 0.60, 0.05, 1.0), _e("1" * 8 + "1000" + "1000000", 0.56, 0.01, 0.4),
         _e("11000000" + "1000" + "1000000", 0.54, 0.0, 0.2)]
    # threshold = 0.60 - 0.05 = 0.55 -> candidates are entries 0 and 1; entry 1 is cheaper
    assert chromosome.to_str(ga.choose_from_pareto(p)) == chromosome.to_str(p[1]["bits"])


def test_choose_boundary_inclusive_and_ties():
    a = "11000000" + "1000" + "1000000"
    b = "10100000" + "1000" + "1000000"   # same cost as a
    p = [_e("1" * 19, 0.60, 0.10, 1.0), _e(b, 0.50, 0.02), _e(a, 0.50, 0.02)]
    # 0.50 >= 0.60 - 0.10 (boundary inclusive); equal cost/mean/std -> smallest string
    assert chromosome.to_str(ga.choose_from_pareto(p)) == min(a, b)
    p.append(_e("01100000" + "1000" + "1000000", 0.52, 0.03))
    assert chromosome.to_str(ga.choose_from_pareto(p)) == "01100000" + "1000" + "1000000"


def test_choose_returns_uint8_and_rejects_empty():
    out = ga.choose_from_pareto([_e("1" * 19, 0.5, 0.0)])
    assert out.dtype == np.uint8 and out.shape == (19,)
    with pytest.raises(ValueError):
        ga.choose_from_pareto([])


# ----------------------------------------------------------------- nesting
@pytest.mark.parametrize("runner", [ga.run_nsga, ga.run_single_objective])
def test_ga_never_touches_test_indices(runner):
    X, y, block = toy_data()
    test_idx = np.flatnonzero(block == 2)
    train_idx = np.flatnonzero(block != 2)
    rec = RecordingEval()
    ga.set_evaluate_fn(rec)
    runner(X, y, block, train_idx, small_cfg(), seed=0)
    assert rec.seen_idx and rec.seen_idx <= set(train_idx.tolist())
    assert not rec.seen_idx & set(test_idx.tolist())


def test_nesting_assertion_fires(monkeypatch):
    X, y, block = toy_data()
    train_idx = np.flatnonzero(block != 3)
    leak = [(train_idx[:40], np.flatnonzero(block == 3))]  # inner "test" = outer test fold
    monkeypatch.setattr(ga, "_inner_splits", lambda *a, **k: leak)
    ga.set_evaluate_fn(RecordingEval())
    with pytest.raises(AssertionError, match="NESTING"):
        ga.run_nsga(X, y, block, train_idx, small_cfg(), seed=0)


def test_inner_splits_follow_contract():
    X, y, block = toy_data()
    train_idx = np.flatnonzero(block != 0)
    splits = ga._inner_splits(y, block, train_idx, n=3, seed=0)
    assert len(splits) == 3
    # 3 training blocks + n=3 -> leave-one-block-out
    assert sorted(int(np.unique(block[te])[0]) for _, te in splits) == [1, 2, 3]
    for tr, te in splits:
        assert tr.dtype == np.int64 and np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0)


# ----------------------------------------------------------------- machinery
@pytest.mark.parametrize("runner", [ga.run_nsga, ga.run_single_objective])
def test_result_shape_and_memoisation(runner):
    X, y, block = toy_data()
    rec = RecordingEval()
    ga.set_evaluate_fn(rec)
    res = runner(X, y, block, np.flatnonzero(block != 0), small_cfg(), seed=1)
    assert set(res) >= {"pareto", "chosen_bits", "n_evals", "time_s"}
    assert res["chosen_bits"].dtype == np.uint8 and res["chosen_bits"].shape == (19,)
    assert len(rec.calls) == len(set(rec.calls)) == res["n_evals"]   # duplicates are free
    keys = [chromosome.to_str(e["bits"]) for e in res["pareto"]]
    assert len(keys) == len(set(keys))
    for e in res["pareto"]:
        assert chromosome.is_valid(e["bits"]) and e["bits"].dtype == np.uint8
        assert e["cost"] == pytest.approx(chromosome.cost(e["bits"]))


def test_nsga_pareto_is_non_dominated_and_choice_consistent():
    X, y, block = toy_data()
    ga.set_evaluate_fn(RecordingEval())
    res = ga.run_nsga(X, y, block, np.flatnonzero(block != 0), small_cfg(), seed=2)
    F = np.array([[-e["ba_mean"], e["ba_std"], e["cost"]] for e in res["pareto"]])
    for i in range(len(F)):
        assert not np.any(np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1))
    np.testing.assert_array_equal(res["chosen_bits"], ga.choose_from_pareto(res["pareto"]))


def test_single_objective_picks_best_mean():
    X, y, block = toy_data()
    ga.set_evaluate_fn(RecordingEval())
    res = ga.run_single_objective(X, y, block, np.flatnonzero(block != 0), small_cfg(), seed=0)
    best = max(e["ba_mean"] for e in res["pareto"])
    chosen = [e for e in res["pareto"] if np.array_equal(e["bits"], res["chosen_bits"])][0]
    assert chosen["ba_mean"] == best


@pytest.mark.parametrize("runner", [ga.run_nsga, ga.run_single_objective])
def test_reproducible_with_fixed_seed(runner):
    X, y, block = toy_data()
    train_idx = np.flatnonzero(block != 1)
    out = []
    for _ in range(2):
        ga.set_evaluate_fn(RecordingEval())
        r = runner(X, y, block, train_idx, small_cfg(), seed=7)
        out.append((chromosome.to_str(r["chosen_bits"]), r["n_evals"],
                    [(chromosome.to_str(e["bits"]), e["ba_mean"], e["ba_std"]) for e in r["pareto"]]))
    assert out[0] == out[1]


def test_reproducible_on_synthetic_with_fbcsp(synth_dir):
    d, splits = C.load_unit("fbcsp", "91", "01", CFG, synth_dir)
    a = ga.run_nsga(d["X"], d["y"], d["block"], splits[0][0], small_cfg(10, 3), seed=3)
    b = ga.run_nsga(d["X"], d["y"], d["block"], splits[0][0], small_cfg(10, 3), seed=3)
    assert [chromosome.to_str(e["bits"]) for e in a["pareto"]] == \
           [chromosome.to_str(e["bits"]) for e in b["pareto"]]
    np.testing.assert_array_equal(a["chosen_bits"], b["chosen_bits"])


# ----------------------------------------------------------------- synthetic recovery
@pytest.mark.slow
def test_selects_informative_channels_on_synthetic(synth_dir):
    """O1 and O2 carry the class effect; they must be chosen more often than chance."""
    cfg = small_cfg(16, 6)
    chosen = []
    for sub in ("91", "92"):
        d, splits = C.load_unit("fbcsp", sub, "01", cfg, synth_dir)
        for k in (0, 1, 2):
            r = ga.run_nsga(d["X"], d["y"], d["block"], splits[k][0], cfg, seed=k)
            chosen.append(chromosome.decode(r["chosen_bits"])[0])
    freq = np.mean(chosen, axis=0)
    # Chance for one channel when 2..8 of 8 are picked at random is at most 0.5.
    others = np.delete(freq, [O1, O2])
    assert freq[O1] > 0.5 and freq[O2] > 0.5, freq
    assert min(freq[O1], freq[O2]) > others.mean(), freq


# ----------------------------------------------------------------- run_ga end to end
def test_ga_passes_the_same_X_object_to_evaluate():
    """fbcsp caches covariances by id(X): every call must get the caller's array itself."""
    X, y, block = toy_data()
    seen = set()
    rec = RecordingEval()

    def spy(Xa, *a, **k):
        seen.add(id(Xa))
        return rec(Xa, *a, **k)

    ga.set_evaluate_fn(spy)
    ga.run_nsga(X, y, block, np.flatnonzero(block != 0), small_cfg(), seed=0)
    assert seen == {id(X)}


def test_real_fbcsp_cache_is_hit(synth_dir):
    d, splits = C.load_unit("fbcsp", "91", "01", CFG, synth_dir)
    fbcsp.clear_cache()
    ga.run_nsga(d["X"], d["y"], d["block"], splits[0][0], small_cfg(6, 2), seed=0)
    assert len(fbcsp._COV_CACHE) == 1   # one covariance computation for the whole GA run


def test_run_ga_cli_end_to_end_and_resume(synth_dir, tmp_path, capsys):
    from experiments import run_ga
    args = ["--method", "ga_nsga", "--subjects", "sub-91", "--protocol", "block",
            "--sessions", "01", "--seeds", "0", "--n-jobs", "1", "--pop-size", "6",
            "--n-generations", "2", "--cache-dir", str(synth_dir), "--results-dir",
            str(tmp_path)]
    assert run_ga.main(args) == 0
    p = tmp_path / "ga_nsga" / "sub-91_ses-01_seed-0.json"
    rec = json.loads(p.read_text())
    results.validate_result(rec)
    assert rec["protocol"] == "block" and len(rec["per_fold"]) == 4
    assert [f["fold"] for f in rec["per_fold"]] == [0, 1, 2, 3]
    assert rec["infer_time_ms"] > 0 and rec["synthetic"] is True
    d, splits = C.load_unit("fbcsp", "91", "01", CFG, synth_dir)
    for f, (tr, te) in zip(rec["per_fold"], splits):
        assert f["n_test"] == len(te) == len(f["y_true"]) == len(f["y_pred"])
        assert f["test_trial_id"] == d["trial_id"][te].tolist()
        assert f["y_true"] == d["y"][te].tolist()
        bal = np.mean([np.mean(np.array(f["y_pred"])[np.array(f["y_true"]) == c] == c)
                       for c in range(3)])
        assert f["ba"] == pytest.approx(bal)
        assert f["pareto"] and f["n_evals"] > 0
    side = json.loads((tmp_path / "pareto" / "ga_nsga" / "sub-91_ses-01_seed-0.json").read_text())
    assert len(side["folds"]) == 4 and all(f["pareto"] for f in side["folds"])

    # Finished run is skipped.
    capsys.readouterr()
    run_ga.main(args)
    assert "0 fold jobs" in capsys.readouterr().out

    # Crash after all folds but before assembly: rebuilt from checkpoints, nothing recomputed.
    p.unlink()
    run_ga.main(args)
    assert json.loads(p.read_text())["per_fold"] == rec["per_fold"]


def test_run_ga_cross_session(synth_dir, tmp_path):
    from experiments import run_ga
    run_ga.main(["--method", "accuracy_only", "--subjects", "91", "--protocol",
                 "cross_session", "--seeds", "1", "--n-jobs", "1", "--pop-size", "6",
                 "--n-generations", "2", "--cache-dir", str(synth_dir), "--results-dir",
                 str(tmp_path)])
    rec = json.loads((tmp_path / "ga_single" / "sub-91_ses-01to02_seed-1.json").read_text())
    results.validate_result(rec)
    assert rec["session"] == "01to02" and len(rec["per_fold"]) == 1
    f = rec["per_fold"][0]
    assert f["n_train"] == 120 and f["n_test"] == 120 and len(f["y_pred"]) == 120


def test_parse_subjects():
    assert C.parse_subjects(["sub-17..sub-22"]) == ["17", "18", "19", "20", "21", "22"]
    assert C.parse_subjects(["3", "sub-05", "91-92"]) == ["03", "05", "91", "92"]

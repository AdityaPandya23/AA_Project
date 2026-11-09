"""Tests for experiments/position_control.py (P5 position-only control)."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments import position_control as PC
from src import results, synth
from src.config import load_config
from src.fbcsp import balanced_accuracy

CFG = load_config()


def test_lookup_majority_and_seeded_ties():
    keys = np.array([0, 0, 0, 1, 1, 2])
    y = np.array([2, 2, 1, 0, 1, 1])
    t1 = PC.fit_lookup(keys, y, np.random.default_rng(0))
    t2 = PC.fit_lookup(keys, y, np.random.default_rng(0))
    assert t1[0] == 2 and t1[2] == 1 and t1[1] in (0, 1)
    assert t1 == t2  # ties are resolved deterministically by the seed


def test_repeated_sequence_is_fully_predictable():
    # sub-01 case: the same 120-trial label sequence in train (ses-01) and test (ses-02).
    rng = np.random.default_rng(1)
    seq = rng.permutation(np.repeat([0, 1, 2], 40))
    tid = np.arange(120)
    table = PC.fit_lookup(PC.KEYS["position_seq120"](tid), seq, rng)
    yp = PC.predict_lookup(table, PC.KEYS["position_seq120"](tid), rng)
    assert balanced_accuracy(seq, yp) == 1.0


def test_independent_sequences_are_at_chance():
    rng = np.random.default_rng(2)
    bas = []
    for _ in range(200):
        ytr, yte = (rng.permutation(np.repeat([0, 1, 2], 40)) for _ in range(2))
        tid = np.arange(120)
        table = PC.fit_lookup(PC.KEYS["position_block30"](tid), ytr, rng)
        bas.append(balanced_accuracy(yte, PC.predict_lookup(table, PC.KEYS["position_block30"](tid), rng)))
    assert abs(np.mean(bas) - 1 / 3) < 0.02


def test_unseen_keys_get_valid_random_class():
    table = PC.fit_lookup(np.array([0, 1]), np.array([2, 2]), np.random.default_rng(0))
    yp = PC.predict_lookup(table, np.array([0, 5, 6, 7]), np.random.default_rng(0))
    assert yp[0] == 2 and set(yp[1:]) <= {0, 1, 2}


def test_identical_blocks_are_found_despite_rejections_and_reordering():
    rng = np.random.default_rng(4)
    B1 = np.stack([rng.permutation(np.repeat([0, 1, 2], 10)) for _ in range(4)])
    B2 = np.stack([rng.permutation(np.repeat([0, 1, 2], 10)), B1[0], B1[1], B1[2]])  # shifted
    B2[2, [3, 17]] = -1  # rejected trials
    assert PC.n_identical_blocks(B2, B1) == 3
    assert PC.n_identical_blocks(B1, B1) == 4
    tid = np.array([0, 1, 31, 119])
    B = PC.block_matrix({"trial_id": tid, "y": np.array([2, 1, 0, 1])})
    assert B[0, 0] == 2 and B[1, 1] == 0 and B[3, 29] == 1 and (B == -1).sum() == 116


def test_cli_writes_contract_results(tmp_path):
    cache = tmp_path / "cache"
    for ses in ("01", "02"):
        synth.write_session("93", ses, cache, CFG)
    out = tmp_path / "out"
    PC.main(["--subjects", "93", "--cache-dir", str(cache), "--out", str(out), "--no-summary"])
    files = {p.relative_to(out).as_posix() for p in out.rglob("*.json")}
    assert files == {"position_block30/sub-93_ses-01_seed-0.json",
                     "position_block30/sub-93_ses-02_seed-0.json",
                     "position_block30/sub-93_ses-01to02_seed-0.json",
                     "position_seq120/sub-93_ses-01to02_seed-0.json"}
    for f in files:
        rec = json.loads((out / f).read_text())
        results.validate_result(rec)
        assert rec["synthetic"] is True
        for fold in rec["per_fold"]:
            assert fold["n_test"] == len(fold["y_pred"]) == len(fold["test_trial_id"])
    blk = json.loads((out / "position_block30/sub-93_ses-01_seed-0.json").read_text())
    assert len(blk["per_fold"]) == 4  # leave-one-block-out, same splits as the methods


def test_agreement_detects_a_method_that_copies_the_control():
    rng = np.random.default_rng(3)
    tid = list(range(90))
    yt = rng.integers(0, 3, 90)
    ctrl = rng.integers(0, 3, 90)
    noise = rng.integers(0, 3, 90)

    def rec(method, yp):
        return {"method": method, "subject": "01", "protocol": "cross_session", "per_fold": [
            {"test_trial_id": tid, "y_true": yt.tolist(), "y_pred": list(map(int, yp))}]}
    df = pd.DataFrame([rec("position_block30", ctrl), rec("copier", ctrl), rec("indep", noise)])
    cop = PC._agreement(df, "copier", "position_block30", "01")
    ind = PC._agreement(df, "indep", "position_block30", "01")
    assert cop["kappa"] == pytest.approx(1.0) and cop["follow_wrong"] == 1.0
    assert abs(ind["kappa"]) < 0.2 and abs(ind["follow_wrong"] - ind["follow_wrong_null"]) < 0.15

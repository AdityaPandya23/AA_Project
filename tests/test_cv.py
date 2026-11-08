"""Tests for src/cv.py (CONTRACTS.md §e)."""
import warnings

import numpy as np
import pytest

from src import cache, cv, synth
from src.config import load_config

CFG = load_config()


@pytest.fixture(scope="module")
def recs(tmp_path_factory):
    d = tmp_path_factory.mktemp("cv_synth")
    for sub, ses in [("91", "01"), ("91", "02"), ("93", "01")]:
        synth.write_session(sub, ses, d, CFG)
    return {k: cache.load_cache("fbcsp", *k, CFG, d) for k in [("91", "01"), ("91", "02"), ("93", "01")]} \
        | {("eeg", "91", "01"): cache.load_cache("eegnet", "91", "01", CFG, d),
           ("eeg", "91", "02"): cache.load_cache("eegnet", "91", "02", CFG, d)}


def _check_split_list(splits, n_total=None):
    for tr, te in splits:
        assert tr.dtype == np.int64 and te.dtype == np.int64
        assert np.all(np.diff(tr) > 0) and np.all(np.diff(te) > 0), "sorted, unique"
        assert np.intersect1d(tr, te).size == 0, "train/test overlap"
        if n_total is not None:
            assert tr.min() >= 0 and max(tr.max(), te.max()) < n_total


def test_block_splits_leave_one_block_out():
    block = np.repeat(np.arange(4), 30).astype(np.int64)
    rng = np.random.default_rng(0)
    block = block[rng.permutation(block.size)]
    splits = cv.block_splits(block)
    assert len(splits) == 4
    _check_split_list(splits, block.size)
    for k, (tr, te) in enumerate(splits):
        assert np.all(block[te] == k), "fold k tests block k"
        assert not np.any(block[tr] == k)
        assert tr.size + te.size == block.size
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in splits])),
                                  np.arange(block.size))


def test_block_splits_with_rejected_trials(recs):
    d = recs[("93", "01")]
    splits = cv.block_splits(d["block"])
    _check_split_list(splits, 117)
    assert sum(te.size for _, te in splits) == 117


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_inner_splits_within_outer_train_and_blocks_intact(recs, seed):
    d = recs[("93", "01")]
    y, block = d["y"], d["block"]
    for k, (tr, te) in enumerate(cv.block_splits(block)):
        inner = cv.inner_splits(y, block, tr, n=3, seed=seed)
        assert len(inner) == 3
        _check_split_list(inner)
        outer_test = set(te.tolist())
        tested_blocks = []
        for itr, ite in inner:
            assert set(itr) <= set(tr) and set(ite) <= set(tr), "inner indices must be ⊆ train_idx"
            assert not (set(itr) | set(ite)) & outer_test, "outer test fold leaked into inner CV"
            assert np.union1d(itr, ite).size == tr.size
            # 3 training blocks, 3 folds -> leave-one-block-out (blocks intact)
            assert np.unique(block[ite]).size == 1
            assert not set(np.unique(block[itr])) & set(np.unique(block[ite]))
            tested_blocks.append(int(block[ite][0]))
        assert sorted(tested_blocks) == sorted(set(range(4)) - {k})


def test_inner_splits_deterministic_and_seed_dependent(recs):
    d = recs[("91", "01")]
    # cross-session-like case: 4 blocks, n=3 -> group assignment depends on seed
    tr = np.arange(d["y"].size, dtype=np.int64)
    a = cv.inner_splits(d["y"], d["block"], tr, 3, seed=5)
    b = cv.inner_splits(d["y"], d["block"], tr, 3, seed=5)
    for (a1, a2), (b1, b2) in zip(a, b):
        np.testing.assert_array_equal(a1, b1)
        np.testing.assert_array_equal(a2, b2)
    for _, te in a:  # blocks still intact; one fold holds 2 blocks (open item 10)
        for blk in np.unique(d["block"][te]):
            assert set(np.flatnonzero(d["block"] == blk)) <= set(te)
    assert sorted(np.unique(d["block"][te]).size for _, te in a) == [1, 1, 2]


def test_inner_splits_on_subset_uses_absolute_indices(recs):
    d = recs[("91", "01")]
    tr = np.flatnonzero(d["block"] != 0)[::2]  # arbitrary subset, absolute positions
    inner = cv.inner_splits(d["y"], d["block"], tr, 3, 0)
    for itr, ite in inner:
        assert set(itr) | set(ite) <= set(tr)


def test_inner_splits_fallback_when_too_few_blocks(recs):
    d = recs[("91", "01")]
    tr = np.flatnonzero(d["block"] <= 1)  # only 2 blocks
    with pytest.warns(RuntimeWarning, match="StratifiedKFold"):
        inner = cv.inner_splits(d["y"], d["block"], tr, 3, 0)
    assert len(inner) == 3
    for itr, ite in inner:
        assert set(itr) | set(ite) <= set(tr)
        assert np.bincount(d["y"][ite]).min() >= 6  # stratified


def test_random_splits_stratified_and_ignore_blocks(recs):
    d = recs[("91", "01")]
    sp = cv.random_splits(d["y"], n=5, seed=0)
    assert len(sp) == 5
    _check_split_list(sp, 120)
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in sp])), np.arange(120))
    for _, te in sp:
        assert np.bincount(d["y"][te]).tolist() == [8, 8, 8]
        assert np.unique(d["block"][te]).size > 1, "random split should mix blocks"
    sp2 = cv.random_splits(d["y"], n=5, seed=0)
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(sp, sp2))
    sp3 = cv.random_splits(d["y"], n=5, seed=1)
    assert not all(np.array_equal(a[1], b[1]) for a, b in zip(sp, sp3))


def test_cross_session_split():
    (tr, te), = cv.cross_session(117, 120)
    np.testing.assert_array_equal(tr, np.arange(117))
    np.testing.assert_array_equal(te, np.arange(117, 237))
    assert tr.dtype == te.dtype == np.int64
    with pytest.raises(ValueError):
        cv.cross_session(0, 5)


def test_concat_sessions_fbcsp(recs):
    d1, d2 = recs[("91", "01")], recs[("91", "02")]
    d = cv.concat_sessions(d1, d2, "fbcsp")
    assert d["X"].shape == (4, 7, 240, 8, 250)
    np.testing.assert_array_equal(d["X"][:, :, :120], d1["X"])
    np.testing.assert_array_equal(d["X"][:, :, 120:], d2["X"])
    np.testing.assert_array_equal(d["block"], np.r_[d1["block"], d2["block"] + 4])
    np.testing.assert_array_equal(d["y"], np.r_[d1["y"], d2["y"]])
    assert d["session_idx"].tolist() == [0] * 120 + [1] * 120
    assert d["ch_names"].tolist() == CFG["channels"]["candidates"]
    (tr, te), = cv.cross_session(120, 120)
    assert set(d["session_idx"][tr]) == {0} and set(d["session_idx"][te]) == {1}


def test_concat_sessions_eegnet_axis0(recs):
    d = cv.concat_sessions(recs[("eeg", "91", "01")], recs[("eeg", "91", "02")], "eegnet")
    assert d["X"].shape == (240, 32, 1000)


def test_concat_sessions_mismatch_raises(recs):
    d1, d2 = recs[("91", "01")], dict(recs[("91", "02")])
    d2["ch_names"] = d2["ch_names"][::-1].copy()
    with pytest.raises(ValueError, match="ch_names"):
        cv.concat_sessions(d1, d2, "fbcsp")
    with pytest.raises(ValueError):
        cv.concat_sessions(d1, recs[("eeg", "91", "02")], "fbcsp")
    with pytest.raises(ValueError):
        cv.concat_sessions(d1, d1, "bogus")

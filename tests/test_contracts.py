"""Contract tests: the synthetic cache matches CONTRACTS.md (a)-(c), and the
public API signatures match (d)-(j). If this file needs changing, CONTRACTS.md
changes first."""
import inspect
import json

import numpy as np
import pytest

from src import analysis, cache, chromosome, cv, eegnet, fbcsp, ga, results, synth
from src.config import config_hash, load_config

CFG = load_config()
PAIRS = [(s, ses) for s in synth.SUBJECTS for ses in synth.SESSIONS]


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache_synth")
    synth.write_synthetic(d, CFG)
    return d


def _load(kind, sub, ses, synth_dir):
    # Load raw (no validation) so dtype/shape assertions below are independent of cache.py.
    return cache.load_npz(cache.cache_path(kind, sub, ses, CFG, synth_dir))


# ------------------------------------------------------------------ caches
def test_all_files_exist(synth_dir):
    for kind in cache.KINDS:
        assert cache.list_cached(kind, CFG, synth_dir) == sorted(PAIRS)


@pytest.mark.parametrize("kind", ["fbcsp", "view"])
@pytest.mark.parametrize("sub,ses", PAIRS)
def test_fbcsp_like_contract(kind, sub, ses, synth_dir):
    d = _load(kind, sub, ses, synth_dir)
    n = d["y"].shape[0]
    assert d["X"].dtype == np.float32
    assert d["X"].shape == (4, 7, n, 8, 250)
    for k in ("y", "block", "trial_id"):
        assert d[k].dtype == np.int64 and d[k].shape == (n,)
    assert set(np.unique(d["y"])) <= {0, 1, 2}
    assert set(np.unique(d["block"])) == {0, 1, 2, 3}
    assert np.all(np.diff(d["trial_id"]) > 0)
    assert d["ch_names"].dtype.kind == "U" and d["ch_names"].tolist() == CFG["channels"]["candidates"]
    assert d["band_names"].dtype.kind == "U" and d["band_names"].tolist() == CFG["bands"]["names"]
    assert d["band_edges"].shape == (4, 2)
    np.testing.assert_allclose(d["window_starts"], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    assert float(d["sfreq"]) == 250.0
    cache.validate_fbcsp(d, CFG)


@pytest.mark.parametrize("sub,ses", PAIRS)
def test_eegnet_contract(sub, ses, synth_dir):
    d = _load("eegnet", sub, ses, synth_dir)
    n = d["y"].shape[0]
    assert d["X"].dtype == np.float32
    assert d["X"].shape == (n, 32, 1000)
    for k in ("y", "block", "trial_id"):
        assert d[k].dtype == np.int64 and d[k].shape == (n,)
    assert d["ch_names"].dtype.kind == "U" and d["ch_names"].shape == (32,)
    assert set(CFG["channels"]["candidates"]) <= set(d["ch_names"].tolist())
    cache.validate_eegnet(d, CFG)


@pytest.mark.parametrize("sub,ses", PAIRS)
def test_trials_aligned_across_kinds(sub, ses, synth_dir):
    ds = {k: _load(k, sub, ses, synth_dir) for k in cache.KINDS}
    for k in ("y", "block", "trial_id"):
        np.testing.assert_array_equal(ds["fbcsp"][k], ds["eegnet"][k])
        np.testing.assert_array_equal(ds["fbcsp"][k], ds["view"][k])


def test_load_cache_validates_and_loads_without_pickle(synth_dir):
    d = cache.load_cache("fbcsp", "91", "01", CFG, synth_dir)
    assert d["X"].shape[2] == 120


def test_rejected_trials_case(synth_dir):
    d = _load("fbcsp", "93", "01", synth_dir)
    assert d["y"].shape == (117,)
    assert not np.isin([7, 64, 111], d["trial_id"]).any()


def test_balanced_blocks(synth_dir):
    d = _load("fbcsp", "91", "01", synth_dir)
    assert np.bincount(d["y"]).tolist() == [40, 40, 40]
    for b in range(4):
        assert np.bincount(d["y"][d["block"] == b]).tolist() == [10, 10, 10]


def test_synthetic_alpha_effect(synth_dir):
    """Square has more alpha on O1 and pentagram more on O2 than circle; theta has no effect."""
    alpha = CFG["bands"]["names"].index("alpha")
    theta = CFG["bands"]["names"].index("theta")
    o1, o2 = (CFG["channels"]["candidates"].index(c) for c in ("O1", "O2"))
    ds = [_load("fbcsp", s, ses, synth_dir) for s, ses in PAIRS]  # pool to beat trial noise

    def diff(band, ch, cls):
        """Mean log-variance of class ``cls`` minus circle, pooled over recordings."""
        out = []
        for d in ds:
            lv = np.log(d["X"][band, :, :, ch, :].var(axis=-1)).mean(axis=0)
            out.append(lv[d["y"] == cls].mean() - lv[d["y"] == 0].mean())
        return float(np.mean(out))

    a1, a2 = diff(alpha, o1, 1), diff(alpha, o2, 2)
    assert a1 > 0.25 and a2 > 0.25
    assert abs(diff(theta, o1, 1)) < 0.5 * a1
    assert abs(diff(alpha, o1, 2)) < 0.5 * a1  # pentagram effect is on O2, not O1


def test_view_effect_stronger(synth_dir):
    alpha = CFG["bands"]["names"].index("alpha")
    o1 = CFG["channels"]["candidates"].index("O1")
    diffs = {}
    for kind in ("fbcsp", "view"):
        d = _load(kind, "91", "01", synth_dir)
        lv = np.log(d["X"][alpha, :, :, o1, :].var(axis=-1)).mean(axis=0)
        diffs[kind] = lv[d["y"] == 1].mean() - lv[d["y"] == 0].mean()
    assert diffs["view"] > diffs["fbcsp"]


def test_select_on_cache(synth_dir):
    d = _load("fbcsp", "92", "02", synth_dir)
    Xs = chromosome.select(d["X"], chromosome.all_on())
    assert Xs.shape == (120, 28, 8, 250) and Xs.dtype == np.float32


def test_validator_rejects_wrong_dtype(synth_dir):
    d = _load("fbcsp", "91", "01", synth_dir)
    d["X"] = d["X"].astype(np.float64)
    with pytest.raises(cache.CacheContractError):
        cache.validate_fbcsp(d, CFG)
    d = _load("eegnet", "91", "01", synth_dir)
    d["y"] = d["y"].astype(np.int32)
    with pytest.raises(cache.CacheContractError):
        cache.validate_eegnet(d, CFG)


def test_candidate_channel_index(synth_dir):
    d = _load("eegnet", "91", "01", synth_dir)
    idx = cache.candidate_channel_index(d["ch_names"], CFG)
    assert d["ch_names"][idx].tolist() == CFG["channels"]["candidates"]
    mask = np.zeros(8, bool); mask[[0, 2]] = True
    idx = cache.candidate_channel_index(d["ch_names"], CFG, mask)
    assert d["ch_names"][idx].tolist() == ["O1", "O2"]


# ------------------------------------------------------------------ API signatures
SIGNATURES = {
    cv.block_splits: ["block"],
    cv.inner_splits: ["y", "block", "train_idx", "n", "seed"],
    cv.random_splits: ["y", "n", "seed"],
    cv.cross_session: ["n_train", "n_test"],
    cv.concat_sessions: ["d1", "d2", "kind"],
    fbcsp.evaluate: ["X", "y", "bits", "splits", "fbcsp_kwargs"],
    fbcsp.FBCSPLDA.fit: ["self", "Xsel", "y"],
    fbcsp.FBCSPLDA.predict: ["self", "Xsel"],
    ga.run_nsga: ["X", "y", "block", "train_idx", "cfg", "seed"],
    ga.run_single_objective: ["X", "y", "block", "train_idx", "cfg", "seed"],
    ga.choose_from_pareto: ["pareto"],
    eegnet.evaluate_eegnet: ["X", "y", "splits", "ch_idx", "cfg"],
    analysis.load_results: ["results_dir"],
    chromosome.decode: ["bits"],
    chromosome.repair: ["bits", "rng"],
    chromosome.cost: ["bits"],
    chromosome.random_valid: ["rng", "p"],
    chromosome.select: ["X", "bits"],
}


@pytest.mark.parametrize("fn", list(SIGNATURES), ids=lambda f: f.__qualname__)
def test_signatures(fn):
    assert list(inspect.signature(fn).parameters) == SIGNATURES[fn]
    assert fn.__doc__, "contract functions must have docstrings"


def test_cv_defaults():
    assert inspect.signature(cv.inner_splits).parameters["n"].default == 3
    assert inspect.signature(cv.random_splits).parameters["n"].default == 5
    assert inspect.signature(eegnet.evaluate_eegnet).parameters["ch_idx"].default is None


def test_fbcsp_is_sklearn_estimator():
    from sklearn.base import clone
    est = fbcsp.FBCSPLDA()
    assert est.get_params() == {"n_filter_pairs": 1, "lda_solver": "lsqr",
                                "lda_shrinkage": "auto", "cov_reg": 0.0}
    clone(est)


# ------------------------------------------------------------------ results JSON
def _record(protocol="block", session="01"):
    rng = np.random.default_rng(0)
    folds = [results.make_fold_record(k, 0.4 + 0.05 * k, chromosome.random_valid(rng))
             for k in range(4)]
    bas = [f["ba"] for f in folds]
    return {"method": "ga_nsga", "subject": "91", "session": session, "seed": 0,
            "protocol": protocol, "per_fold": folds, "ba_mean": float(np.mean(bas)),
            "ba_std": float(np.std(bas)), "train_time_s": 12.3, "infer_time_ms": 0.5,
            "config_hash": config_hash(CFG)}


def test_result_roundtrip(tmp_path):
    rec = _record()
    p = results.write_result(rec, CFG, tmp_path)
    assert p == tmp_path / "ga_nsga" / "sub-91_ses-01_seed-0.json"
    back = json.loads(p.read_text())
    results.validate_result(back)
    f0 = back["per_fold"][0]
    assert f0["cost"] == pytest.approx(chromosome.cost(chromosome.from_str(f0["chosen_bits"])))
    assert f0["n_features"] == chromosome.n_features(chromosome.from_str(f0["chosen_bits"]))


def test_result_paths_do_not_collide(tmp_path):
    a = results.result_path("ga_nsga", "91", "02", 0, "block", CFG, tmp_path)
    b = results.result_path("ga_nsga", "91", "01to02", 0, "cross_session", CFG, tmp_path)
    c = results.result_path("ga_nsga", "91", "02", 0, "random", CFG, tmp_path)
    assert len({a, b, c}) == 3


def test_result_validation_errors():
    rec = _record(); rec["ba_mean"] = 0.9
    with pytest.raises(results.ResultContractError):
        results.validate_result(rec)
    rec = _record(); del rec["config_hash"]
    with pytest.raises(results.ResultContractError):
        results.validate_result(rec)
    with pytest.raises(results.ResultContractError):
        results.validate_result(_record("cross_session", "02"))
    results.validate_result(_record("cross_session", "01to02"))


def test_eegnet_fold_record_nulls():
    f = results.make_fold_record(0, 0.4, None, n_features=2000)
    assert f["chosen_bits"] is None and f["cost"] is None and f["n_features"] == 2000

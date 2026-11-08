"""Tests for src/fbcsp.py (CONTRACTS.md §f): estimator API, equivalence, no leakage, above chance."""
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src import cache, chromosome as C, cv, fbcsp, synth
from src.config import load_config

CFG = load_config()
PAIRS = [(s, ses) for s in synth.SUBJECTS for ses in synth.SESSIONS]


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fbcsp_synth")
    synth.write_synthetic(d, CFG)
    return d


@pytest.fixture(scope="module")
def rec(synth_dir):
    return cache.load_cache("fbcsp", "93", "01", CFG, synth_dir)  # 117 trials


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _bits(*on):
    b = np.zeros(19, np.uint8)
    b[list(on)] = 1
    return b


# ---------------------------------------------------------------- sklearn API
def test_cloneable_and_params():
    est = fbcsp.FBCSPLDA(cov_reg=0.05, lda_shrinkage=0.2)
    c = clone(est)
    assert c is not est and c.get_params() == est.get_params()
    assert set(fbcsp.FBCSPLDA().get_params()) == {"n_filter_pairs", "lda_solver",
                                                  "lda_shrinkage", "cov_reg"}


def test_clone_of_fitted_is_unfitted(rec):
    Xs = C.select(rec["X"], C.all_on())
    est = fbcsp.FBCSPLDA().fit(Xs, rec["y"])
    c = clone(est)
    assert not hasattr(c, "filters_") and not hasattr(c, "lda_")
    from sklearn.exceptions import NotFittedError
    with pytest.raises(NotFittedError):
        c.transform(Xs)


def test_attributes_and_shapes(rec):
    bits = _bits(0, 2, 5, 9, 10, 12, 15, 18)  # 3 ch, 2 bands, 3 windows
    Xs = C.select(rec["X"], bits)
    est = fbcsp.FBCSPLDA().fit(Xs, rec["y"])
    n_bw = 6
    assert est.classes_.tolist() == [0, 1, 2]
    assert est.filters_.shape == (n_bw, 3, 2, 3)
    assert isinstance(est.lda_, LinearDiscriminantAnalysis)
    F = est.transform(Xs)
    assert F.shape == (rec["y"].size, 6 * n_bw) == (rec["y"].size, C.n_features(bits))
    assert np.isfinite(F).all()
    assert est.predict(Xs).shape == rec["y"].shape
    P = est.predict_proba(Xs)
    np.testing.assert_allclose(P.sum(1), 1.0)
    assert est.score(Xs, rec["y"]) > 1 / 3  # ClassifierMixin.score works


def test_two_channels(rec):
    bits = _bits(0, 2, 9, 12)  # O1, O2 / alpha / window 0
    Xs = C.select(rec["X"], bits)
    est = fbcsp.FBCSPLDA().fit(Xs, rec["y"])
    assert est.filters_.shape == (1, 3, 2, 2)
    assert est.transform(Xs).shape == (rec["y"].size, 6)


def test_too_many_filter_pairs_raises(rec):
    Xs = C.select(rec["X"], _bits(0, 2, 9, 12))
    with pytest.raises(ValueError):
        fbcsp.FBCSPLDA(n_filter_pairs=2).fit(Xs, rec["y"])


def test_logvar_features_match_explicit_filtering(rec):
    """Covariance-domain features == log(var(w^T x)) computed on the time series."""
    Xs = C.select(rec["X"], _bits(0, 1, 2, 3, 8, 9, 12, 13))
    est = fbcsp.FBCSPLDA().fit(Xs, rec["y"])
    F = est.transform(Xs[:5])
    W = est.filters_                                        # (bw, K, 2, c)
    expl = np.log(np.einsum("bkfc,nbct->nbkft", W, Xs[:5].astype(np.float64)).var(axis=-1))
    np.testing.assert_allclose(F, expl.reshape(5, -1), rtol=1e-8, atol=1e-8)


def test_csp_filters_solve_generalised_eigenproblem(rec):
    Xs = C.select(rec["X"], _bits(0, 1, 2, 3, 4, 9, 12))
    y = rec["y"]
    est = fbcsp.FBCSPLDA().fit(Xs, y)
    covs = fbcsp.trial_covariances(Xs)[:, 0]
    for k in range(3):
        Cc, Cr = covs[y == k].mean(0), covs[y != k].mean(0)
        from scipy.linalg import eigh
        lam = eigh(Cc, Cc + Cr, eigvals_only=True)
        w_max, w_min = est.filters_[0, k]
        r = lambda w: (w @ Cc @ w) / (w @ (Cc + Cr) @ w)
        assert r(w_max) == pytest.approx(lam[-1], rel=1e-8)
        assert r(w_min) == pytest.approx(lam[0], rel=1e-8)


@pytest.mark.parametrize("shrinkage", ["auto", 0.3, None])
def test_fast_lda_matches_sklearn(rec, rng, shrinkage):
    splits = cv.block_splits(rec["block"])
    for _ in range(5):
        # Without shrinkage, p > n is singular and ill-posed (coefficients ~1e9), so compare
        # only well-posed problems there: 2 bands x 2 windows = 24 features < 90 trials.
        bits = C.random_valid(rng) if shrinkage is not None else _bits(
            *rng.choice(8, 4, replace=False), *(8 + rng.choice(4, 2, replace=False)),
            *(12 + rng.choice(7, 2, replace=False)))
        Xs = C.select(rec["X"], bits)
        tr, te = splits[int(rng.integers(4))]
        est = fbcsp.FBCSPLDA(lda_shrinkage=shrinkage).fit(Xs[tr], rec["y"][tr])
        F = est.transform(Xs[tr])
        sc = StandardScaler().fit(F)
        ref = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage).fit(sc.transform(F),
                                                                                 rec["y"][tr])
        np.testing.assert_allclose(est.scaler_.mean_, sc.mean_)
        np.testing.assert_allclose(est.scaler_.scale_, sc.scale_)
        np.testing.assert_allclose(est.lda_.coef_, ref.coef_, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(est.lda_.intercept_, ref.intercept_, rtol=1e-6, atol=1e-8)
        Ft = sc.transform(est.transform(Xs[te]))
        np.testing.assert_array_equal(est.predict(Xs[te]), ref.predict(Ft))
        np.testing.assert_array_equal(est.lda_.predict(Ft), ref.predict(Ft))  # sklearn object usable


def test_other_lda_solver_falls_back_to_sklearn(rec):
    Xs = C.select(rec["X"], _bits(0, 2, 9, 12, 13))
    est = fbcsp.FBCSPLDA(lda_solver="svd", lda_shrinkage=None).fit(Xs, rec["y"])
    assert est.predict(Xs).shape == rec["y"].shape


def test_balanced_accuracy_matches_sklearn(rng):
    for _ in range(50):
        yt = rng.integers(0, 3, 40)
        yp = rng.integers(0, 3, 40)
        assert fbcsp.balanced_accuracy(yt, yp) == pytest.approx(balanced_accuracy_score(yt, yp))


# ---------------------------------------------------------------- evaluate()
def test_evaluate_contract_dict(rec, rng):
    splits = cv.block_splits(rec["block"])
    r = fbcsp.evaluate(rec["X"], rec["y"], C.random_valid(rng), splits)
    assert set(r) >= {"ba_mean", "ba_std", "per_fold", "fit_time_s"}
    assert isinstance(r["ba_mean"], float) and isinstance(r["ba_std"], float)
    assert len(r["per_fold"]) == 4 and all(isinstance(v, float) for v in r["per_fold"])
    assert r["ba_mean"] == pytest.approx(np.mean(r["per_fold"]))
    assert r["ba_std"] == pytest.approx(np.std(r["per_fold"], ddof=0))
    assert r["fit_time_s"] > 0
    assert [p.shape for p in r["y_pred"]] == [te.shape for _, te in splits]


def test_evaluate_rejects_invalid_bits_and_overlap(rec):
    splits = cv.block_splits(rec["block"])
    with pytest.raises(ValueError):
        fbcsp.evaluate(rec["X"], rec["y"], np.zeros(19), splits)
    tr, te = splits[0]
    with pytest.raises(ValueError, match="overlap"):
        fbcsp.evaluate(rec["X"], rec["y"], C.all_on(), [(tr, np.r_[te, tr[:1]])])


@pytest.mark.parametrize("use_cache", [True, False])
def test_evaluate_equals_public_fit_predict(rec, rng, use_cache, monkeypatch):
    """evaluate() == fresh FBCSPLDA().fit(select(X)[tr]).predict(select(X)[te]) per split."""
    monkeypatch.setattr(fbcsp, "CACHE_COVARIANCES", use_cache)
    fbcsp.clear_cache()
    y = rec["y"]
    splits = cv.block_splits(rec["block"])
    for bits in [C.all_on(), _bits(1, 6, 8, 18)] + [C.random_valid(rng) for _ in range(3)]:
        r = fbcsp.evaluate(rec["X"], y, bits, splits)
        Xs = C.select(rec["X"], bits)
        for k, (tr, te) in enumerate(splits):
            yp = fbcsp.FBCSPLDA().fit(Xs[tr], y[tr]).predict(Xs[te])
            np.testing.assert_array_equal(r["y_pred"][k], yp)
            assert r["per_fold"][k] == pytest.approx(balanced_accuracy_score(y[te], yp))


def test_covariance_cache_hit_and_invalidation(rec):
    fbcsp.clear_cache()
    X = rec["X"].copy()
    a = fbcsp.full_covariances(X)
    assert fbcsp.full_covariances(X) is a, "second call should hit the cache"
    X[0, 0, 0, 0, :] += 1000.0          # in-place change of the sampled first element
    assert fbcsp.full_covariances(X) is not a
    fbcsp.clear_cache()
    assert len(fbcsp._COV_CACHE) == 0


# ---------------------------------------------------------------- leakage
def _poison(X, idx):
    Xp = X.copy()
    Xp[:, :, idx] = np.nan
    return Xp


def test_no_leakage_estimator_fit_uses_only_train_trials(rec, monkeypatch):
    """Every fit inside evaluate() receives exactly the train trials, and poisoned test trials
    (NaN) never reach a fit."""
    y, block = rec["y"], rec["block"]
    calls = []
    orig = fbcsp.FBCSPLDA._fit_cov

    def spy(self, Cov, yy):
        calls.append((Cov.copy(), np.asarray(yy).copy()))
        return orig(self, Cov, yy)

    monkeypatch.setattr(fbcsp.FBCSPLDA, "_fit_cov", spy)
    for tr, te in cv.block_splits(block):
        calls.clear()
        Xp = _poison(rec["X"], te)
        fbcsp.evaluate(Xp, y, C.all_on(), [(tr, te)])
        (Cov, yy), = calls
        assert Cov.shape[0] == tr.size
        np.testing.assert_array_equal(yy, y[tr])
        assert np.isfinite(Cov).all(), "a test trial reached the fit"


def test_no_leakage_inner_cv_ignores_outer_test_fold(rec, rng):
    """GA fitness (evaluate on inner splits of an outer train set) must not depend on the
    outer test trials at all: replacing them by NaN or noise gives identical results."""
    y, block = rec["y"], rec["block"]
    for tr, te in cv.block_splits(block):
        inner = cv.inner_splits(y, block, tr, 3, seed=0)
        Xn = rec["X"].copy()
        Xn[:, :, te] = rng.standard_normal(Xn[:, :, te].shape).astype(np.float32) * 50
        y_shuf = y.copy()
        y_shuf[te] = rng.permutation(y[te])
        for bits in (C.all_on(), C.random_valid(rng)):
            ref = fbcsp.evaluate(rec["X"], y, bits, inner)
            for Xv, yv in ((_poison(rec["X"], te), y), (Xn, y_shuf)):
                r = fbcsp.evaluate(Xv, yv, bits, inner)
                assert r["per_fold"] == ref["per_fold"]
                for a, b in zip(r["y_pred"], ref["y_pred"]):
                    np.testing.assert_array_equal(a, b)


def test_no_leakage_fitted_model_independent_of_test_data(rec, rng):
    y, block = rec["y"], rec["block"]
    tr, te = cv.block_splits(block)[2]
    Xs = C.select(rec["X"], C.all_on())
    Xs2 = Xs.copy()
    Xs2[te] = rng.standard_normal(Xs2[te].shape).astype(np.float32)
    a = fbcsp.FBCSPLDA().fit(Xs[tr], y[tr])
    b = fbcsp.FBCSPLDA().fit(Xs2[tr], y[tr])
    np.testing.assert_array_equal(a.filters_, b.filters_)
    np.testing.assert_array_equal(a.lda_.coef_, b.lda_.coef_)


# ---------------------------------------------------------------- above chance
def test_evaluate_beats_chance_on_synthetic(synth_dir):
    """All-on block CV beats chance on average over the synthetic recordings, and the
    O1/O2 alpha selection (where the synthetic effect lives) is clearly better."""
    bas, bas_alpha = [], []
    alpha_o1o2 = _bits(0, 2, 9, *range(12, 19))
    for s, ses in PAIRS:
        d = cache.load_cache("fbcsp", s, ses, CFG, synth_dir)
        sp = cv.block_splits(d["block"])
        bas.append(fbcsp.evaluate(d["X"], d["y"], C.all_on(), sp)["ba_mean"])
        bas_alpha.append(fbcsp.evaluate(d["X"], d["y"], alpha_o1o2, sp)["ba_mean"])
    assert np.mean(bas) > 0.37
    assert np.mean(bas_alpha) > 0.45
    assert np.mean(bas_alpha) > np.mean(bas)


def test_viewing_cache_decodes_well(synth_dir):
    d = cache.load_cache("view", "91", "01", CFG, synth_dir)
    r = fbcsp.evaluate(d["X"], d["y"], C.all_on(), cv.block_splits(d["block"]))
    assert r["ba_mean"] > 0.55


def test_shuffled_labels_near_chance(synth_dir):
    d = cache.load_cache("view", "92", "02", CFG, synth_dir)  # strong effect -> shuffling must kill it
    rng = np.random.default_rng(0)
    sp = cv.block_splits(d["block"])
    null = [fbcsp.evaluate(d["X"], rng.permutation(d["y"]), C.all_on(), sp)["ba_mean"]
            for _ in range(20)]
    assert abs(np.mean(null) - 1 / 3) < 0.05

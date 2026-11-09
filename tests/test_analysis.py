"""Tests for src/analysis.py and reports/make_tables.py on FAKE result JSONs (tests/fake_results.py)."""
import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src import analysis as A, chromosome as C
from tests.fake_results import METHODS, write_fake_results

SUBJECTS = ("01", "02", "03", "04", "05", "06", "07", "08")


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("fake_results")
    write_fake_results(d, SUBJECTS)
    return d


@pytest.fixture(scope="module")
def df(fake_dir):
    return A.load_results(fake_dir)


# ---------------------------------------------------------------- loading
def test_load_results_columns_and_counts(df):
    for c in ["method", "subject", "session", "seed", "protocol", "ba_mean", "ba_std",
              "train_time_s", "infer_time_ms", "config_hash", "mean_cost", "mean_n_features",
              "per_fold", "path"]:
        assert c in df.columns
    assert set(df["method"]) == set(METHODS)
    assert set(df["protocol"]) == {"block", "cross_session"}
    # last subject has only ses-01 -> no cross-session file
    assert not ((df["subject"] == SUBJECTS[-1]) & (df["protocol"] == "cross_session")).any()
    assert (df.loc[df["method"] == "fbcsp_all", "mean_cost"] == 1.0).all()
    assert df.loc[df["method"] == "eegnet", "mean_cost"].isna().all()
    assert (df.loc[df["method"] == "fbcsp_all", "seed"] == 0).all()


def test_load_results_validates(tmp_path, fake_dir):
    bad = tmp_path / "ga_nsga"
    bad.mkdir()
    rec = json.loads(next((fake_dir / "ga_nsga").glob("*.json")).read_text())
    rec["ba_mean"] = 0.99
    (bad / "sub-01_ses-01_seed-0.json").write_text(json.dumps(rec))
    from src.results import ResultContractError
    with pytest.raises(ResultContractError):
        A.load_results(tmp_path)


def test_load_results_warns_on_mixed_config_hash(tmp_path, fake_dir):
    for i, p in enumerate(sorted((fake_dir / "fbcsp_all").glob("*.json"))[:2]):
        rec = json.loads(p.read_text())
        rec["config_hash"] = f"{i:010d}"
        (tmp_path / "fbcsp_all").mkdir(exist_ok=True)
        (tmp_path / "fbcsp_all" / p.name).write_text(json.dumps(rec))
    with pytest.warns(RuntimeWarning, match="config hashes"):
        A.load_results(tmp_path)


def test_load_results_skips_non_result_files(fake_dir, df, tmp_path):
    # results/ root layout: sanity summary, run_ga side files and checkpoints next to methods.
    import shutil
    shutil.copytree(fake_dir, tmp_path, dirs_exist_ok=True)
    (tmp_path / "sanity").mkdir()
    (tmp_path / "sanity" / "sanity_summary.json").write_text(json.dumps({"n": 1}))
    for d in ("pareto", "raw"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "sub-01_ses-01_seed-0.json").write_text(json.dumps({"folds": []}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = A.load_results(tmp_path)
    assert len(d) == len(df)
    assert not d["path"].str.contains(r"[\\/](?:pareto|raw|sanity)[\\/]").any()


def test_load_results_list_of_dirs(fake_dir, df):
    assert len(A.load_results([fake_dir, fake_dir])) == 2 * len(df)


def test_explode_and_aggregate(df):
    ex = A.explode_folds(df)
    assert len(ex) == df["n_folds"].sum()
    assert {"fold", "ba", "chosen_bits", "cost", "n_features"} <= set(ex.columns)
    agg = A.aggregate_seeds(df)
    assert not agg.duplicated(["method", "subject", "session", "protocol"]).any()
    row = agg[(agg.method == "ga_nsga") & (agg.subject == "03") & (agg.session == "01")
              & (agg.protocol == "block")].iloc[0]
    raw = df[(df.method == "ga_nsga") & (df.subject == "03") & (df.session == "01")
             & (df.protocol == "block")]
    assert row["n_seeds"] == 3
    assert row["ba_mean"] == pytest.approx(raw["ba_mean"].mean())


# ---------------------------------------------------------------- statistics
def test_paired_test_matches_scipy_on_subject_means(df):
    r = A.paired_test(df, "ga_nsga", "fbcsp_all")
    wide = A.subject_level(df, "block")
    d = (wide["ga_nsga"] - wide["fbcsp_all"]).dropna()
    assert r["n"] == len(SUBJECTS)  # n = subjects, not subjects x sessions x seeds
    ref = stats.wilcoxon(d)
    assert r["p"] == pytest.approx(ref.pvalue)
    assert r["mean_diff"] == pytest.approx(d.mean())
    assert -1 <= r["effect_size"] <= 1
    rt = A.paired_test(df, "ga_nsga", "fbcsp_all", test="ttest")
    assert rt["p"] == pytest.approx(stats.ttest_rel(wide["ga_nsga"], wide["fbcsp_all"]).pvalue)


def test_paired_test_detects_built_in_ordering(df):
    r = A.paired_test(df, "ga_nsga", "eegnet")
    assert r["mean_diff"] > 0 and r["effect_size"] > 0.5
    with pytest.raises(ValueError):
        A.paired_test(df, "ga_nsga", "nope")
    with pytest.raises(ValueError):
        A.paired_test(df, "ga_nsga", "fbcsp_all", test="anova")


def test_paired_test_identical_methods():
    rows = [{"method": m, "subject": s, "session": "01", "seed": 0, "protocol": "block",
             "ba_mean": 0.5} for m in ("a", "b") for s in ("01", "02", "03")]
    r = A.paired_test(pd.DataFrame(rows), "a", "b")
    assert r["p"] == 1.0 and r["mean_diff"] == 0.0


def test_friedman_and_holm(df):
    r = A.friedman_test(df, ["ga_nsga", "fbcsp_all", "eegnet"])
    assert r["n"] == len(SUBJECTS) and 0 <= r["p"] <= 1
    assert min(r["mean_ranks"], key=r["mean_ranks"].get) == "ga_nsga"  # rank 1 = best
    from statsmodels.stats.multitest import multipletests
    p = [0.01, 0.04, 0.03, 0.5]
    np.testing.assert_allclose(A.holm(p), multipletests(p, method="holm")[1])
    h = A.holm([0.01, np.nan, 0.02])
    assert np.isnan(h[1]) and h[0] == pytest.approx(0.02)


def test_above_chance_permutation(df):
    ab = A.above_chance(df, "ga_nsga", n_perm=500)
    assert {"subject", "session", "ba", "p", "p_holm", "test"} <= set(ab.columns)
    assert len(ab) == len(df[(df.method == "ga_nsga") & (df.protocol == "block")]
                          .groupby(["subject", "session"]))
    assert (ab["test"] == "permutation").all()
    strong = ab[ab.subject == SUBJECTS[0]]
    chance = ab[ab.subject == SUBJECTS[1]]
    assert (strong["p"] < 0.01).all()
    assert (chance["p"] > 0.01).all()
    assert ab["null_mean"].between(0.30, 0.37).all()
    assert (ab["p_holm"] >= ab["p"] - 1e-12).all()
    # observed BA = mean over seeds of ba_mean
    raw = df[(df.method == "ga_nsga") & (df.protocol == "block") & (df.subject == "03")
             & (df.session == "01")]
    assert ab[(ab.subject == "03") & (ab.session == "01")]["ba"].iloc[0] == pytest.approx(
        raw["ba_mean"].mean())


def test_above_chance_binomial_fallback(df):
    d = df[(df.method == "fbcsp_all") & (df.protocol == "block")].copy()
    d["per_fold"] = [[{k: v for k, v in f.items() if k not in ("y_true", "y_pred")} for f in pf]
                     for pf in d["per_fold"]]
    ab = A.above_chance(d, "fbcsp_all")
    assert (ab["test"] == "binomial_approx").all() and ab["p"].between(0, 1).all()


def test_confusion_rows_sum_to_one(df):
    M = A.confusion(df, "ga_nsga")
    assert M.shape == (3, 3)
    np.testing.assert_allclose(M.sum(1), 1.0)
    assert np.all(np.diag(M) > 0.4)


# ---------------------------------------------------------------- selection & stability
def test_jaccard_kuncheva_known_values():
    a = C.from_str("1100000000000000000")
    b = C.from_str("1010000000000000000")
    assert A.jaccard(a, a) == 1.0 and A.jaccard(a, b) == pytest.approx(1 / 3)
    assert A.kuncheva(a, a) == pytest.approx(1.0)
    # equal sizes k=2, n=19, r=1 -> (1*19 - 4) / (2 * 17)
    assert A.kuncheva(a, b) == pytest.approx((19 - 4) / (2 * 17))
    assert np.isnan(A.kuncheva(C.all_on(), C.all_on()))
    c = C.from_str("1110000000000000000")  # unequal sizes -> Lustgarten generalisation
    assert A.kuncheva(a, c) == pytest.approx((2 - 2 * 3 / 19) / (2 - 0))
    rng = np.random.default_rng(0)
    rand = [A.kuncheva(rng.random(19) < 0.5, rng.random(19) < 0.5) for _ in range(3000)]
    assert abs(np.nanmean(rand)) < 0.03  # ~0 for independent random subsets


def test_selection_frequency(df):
    f = A.selection_frequency(df[df.protocol == "block"], "ga_nsga")
    assert list(f.columns[:4]) == ["gene", "group", "name", "frequency"]
    assert len(f) == 19 and f["frequency"].between(0, 1).all()
    assert {"O1", "O2", "alpha"} <= set(f.nlargest(6, "frequency")["name"])
    fa = A.selection_frequency(df, "fbcsp_all")
    assert (fa["frequency"] == 1.0).all()
    assert A.selection_frequency(df, "eegnet")["n"].iloc[0] == 0


def test_selection_matrix_shape(df):
    M = A.selection_matrix(df, "ga_nsga")
    assert M.shape == (len(SUBJECTS), 19)
    assert M.columns[0] == "O1" and M.columns[-1] == "3.0s"


def test_stability_nsga_more_stable_than_single(df):
    s_n = A.selection_stability(df[df.protocol == "block"], "ga_nsga")
    s_s = A.selection_stability(df[df.protocol == "block"], "ga_single")
    assert s_n["kuncheva"].mean() > s_s["kuncheva"].mean()
    assert s_n["jaccard"].between(0, 1).all()
    assert (s_n["n_pairs"] == 66).all()  # 4 folds x 3 seeds = 12 chromosomes -> 66 pairs
    by_seed = A.selection_stability(df[df.protocol == "block"], "ga_nsga", by="seed")
    assert (by_seed["n_pairs"] == 4 * 3).all()  # per fold 3 pairs, 4 folds
    by_fold = A.selection_stability(df[df.protocol == "block"], "ga_nsga", by="fold")
    assert (by_fold["n_pairs"] == 3 * 6).all()
    ch = A.selection_stability(df[df.protocol == "block"], "ga_nsga", groups="channel")
    assert ch["kuncheva"].notna().all()


def test_cross_session_drop(df):
    d = A.cross_session_drop(df)
    assert set(d.columns) == {"method", "subject", "within_ba", "cross_ba", "drop"}
    assert SUBJECTS[-1] not in set(d["subject"])  # only ses-01
    assert d.loc[d.subject != SUBJECTS[1], "drop"].mean() > 0  # built-in drop ≈ 0.05
    np.testing.assert_allclose(d["drop"], d["within_ba"] - d["cross_ba"])
    dm = A.cross_session_drop(df, ["ga_nsga"], within_session="mean")
    assert set(dm["method"]) == {"ga_nsga"}


def test_pareto_front():
    pts = np.array([[0.2, 0.5], [0.3, 0.6], [0.3, 0.4], [0.5, 0.55], [0.8, 0.7]])
    assert A.pareto_front(pts).tolist() == [0, 1, 4]


# ---------------------------------------------------------------- plots and tables
def test_make_figures_writes_all(fake_dir, tmp_path):
    paths = A.make_figures(fake_dir, tmp_path)
    names = {p.name for p in paths}
    for n in ["subject_bars_block.png", "subject_bars_cross_session.png",
              "method_comparison_block.png", "cost_vs_accuracy_block.png",
              "confusion_ga_nsga_block.png", "selection_heatmap_ga_nsga.png",
              "selection_frequency_ga_nsga.png", "stability_boxplot.png"]:
        assert n in names
    assert "selection_heatmap_fbcsp_all.png" not in names  # fixed chromosome: no selection info
    assert all(p.stat().st_size > 5000 for p in paths)


def test_plot_pareto_single_front(df):
    import matplotlib.pyplot as plt
    ex = A.explode_folds(df[df.method == "ga_nsga"])
    r = ex.iloc[0]
    ax = A.plot_pareto(r["pareto"], r["chosen_bits"])
    assert len(ax.collections) >= 2
    plt.close(ax.figure)


def test_method_colors_fixed_by_entity():
    assert A.method_color("ga_nsga") == A.CATEGORICAL[0]
    assert A.method_color("fbcsp_all_view") == A.method_color("fbcsp_all")
    assert A.method_color("something_new") == A.method_color("something_new")


def test_make_tables(fake_dir, tmp_path):
    from reports import make_tables
    md, csv = make_tables.run([str(fake_dir)], str(tmp_path), n_perm=200)
    tab = pd.read_csv(csv)
    assert tab["method"].iloc[0] == "ga_nsga"
    assert set(tab["method"]) == set(METHODS)
    assert tab.loc[tab.method == "ga_nsga", "ba_mean"].iloc[0] > tab.loc[tab.method == "eegnet",
                                                                          "ba_mean"].iloc[0]
    text = md.read_text(encoding="utf-8")
    assert text.count("\n| ") >= 1 + len(METHODS) and "Friedman" in text
    assert (tmp_path / "per_subject.csv").exists()


# ------------------------------------------------- integration with experiments/run_ga.py
def _run_ga_layout(fake_dir, out, impl="src.fbcsp"):
    """Rewrite one fake ga_nsga result in run_ga's layout: Pareto fronts in a side file."""
    src = next((fake_dir / "ga_nsga").glob("*.json"))
    rec = json.loads(src.read_text())
    side = {"method": rec["method"], "subject": rec["subject"], "session": rec["session"],
            "seed": rec["seed"], "protocol": rec["protocol"],
            "folds": [{"fold": f["fold"], "chosen_bits": f["chosen_bits"],
                       "pareto": f.pop("pareto")} for f in rec["per_fold"]]}
    rec["fbcsp_impl"] = impl
    (out / "ga_nsga").mkdir(parents=True)
    (out / "pareto" / "ga_nsga").mkdir(parents=True)
    (out / "ga_nsga" / src.name).write_text(json.dumps(rec))
    (out / "pareto" / "ga_nsga" / src.name).write_text(json.dumps(side))
    return side


def test_load_results_reads_run_ga_pareto_side_file(fake_dir, tmp_path):
    side = _run_ga_layout(fake_dir, tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # real FBCSP -> no stand-in warning
        d = A.load_results(tmp_path)
    assert len(d) == 1  # the side file itself is not read as a result
    folds = d.iloc[0]["per_fold"]
    assert [f["pareto"] for f in folds] == [e["pareto"] for e in side["folds"]]
    ex = A.explode_folds(d)
    assert ex["pareto"].notna().all()


def test_load_results_warns_on_standin_fbcsp(fake_dir, tmp_path):
    _run_ga_layout(fake_dir, tmp_path, impl="standin")
    with pytest.warns(RuntimeWarning, match="stand-in"):
        d = A.load_results(tmp_path)
    assert d["standin"].all()


def test_eegnet_run_names():
    assert A.method_color("eegnet_32ch") == A.method_color("eegnet")
    assert A.method_color("eegnet_8ch") not in {A.method_color(m) for m in A.METHOD_ORDER[:5]}
    assert A.method_label("eegnet_8ch") == "EEGNet (8 ch)"
    assert A._sort_methods(["eegnet_8ch", "eegnet_32ch", "ga_nsga"]) == \
        ["ga_nsga", "eegnet_32ch", "eegnet_8ch"]
    # unknown methods never take a slot of a known one
    known = {A.method_color(m) for m in A.METHOD_ORDER}
    assert A.method_color("something_new") not in known

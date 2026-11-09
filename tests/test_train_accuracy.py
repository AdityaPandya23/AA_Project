"""Tests for reports/train_accuracy.py (P5 training vs inner-CV vs test BA)."""
from src import synth
from src.config import load_config

CFG = load_config()


def test_train_accuracy_on_synthetic(tmp_path):
    from experiments import baseline, run_ga
    from reports import train_accuracy as TA
    cache = tmp_path / "cache"
    for ses in ("01", "02"):
        synth.write_session("93", ses, cache, CFG)
    res = tmp_path / "results"
    baseline.run(cache, res / "baseline_all", n_jobs=1)
    run_ga.main(["--method", "ga_nsga", "--subjects", "93", "--seeds", "0", "--pop-size", "6",
                 "--n-generations", "2", "--n-jobs", "1", "--cache-dir", str(cache),
                 "--results-dir", str(res)])
    fl = TA.run(res, cache, n_jobs=1)  # raises if a re-scored test BA differs from the stored one
    assert set(fl.method) == {"fbcsp_all", "ga_nsga"}
    assert set(fl.protocol) == {"block"} | ({"cross_session"} & set(fl.protocol))
    assert ((fl.train_ba >= 0) & (fl.train_ba <= 1) & (fl.inner_ba >= 0)).all()
    fb = fl[fl.method == "fbcsp_all"]
    assert (fb.n_features == 168).all() and fb.train_ba.mean() > fb.test_ba.mean()
    s = TA.summarise(fl)
    assert {"train_minus_test", "inner_minus_test"} <= set(s.columns)

"""Tests for experiments/pareto_thresholds.py (offline NSGA-II threshold comparison)."""
import numpy as np
import pytest

from experiments import pareto_thresholds as PT
from src import chromosome, ga, synth
from src.config import load_config

CFG = load_config()


def test_one_sd_width_reproduces_contract_rule():
    rng = np.random.default_rng(0)
    for _ in range(200):
        par = []
        for _ in range(rng.integers(1, 12)):
            b = chromosome.random_valid(rng)
            par.append({"bits": b, "ba_mean": float(rng.choice([0.4, 0.45, 0.5, 0.55])),
                        "ba_std": float(rng.choice([0.0, 0.05, 0.1])), "cost": chromosome.cost(b)})
        as_str = [{**e, "bits": chromosome.to_str(e["bits"])} for e in par]
        np.testing.assert_array_equal(PT.choose_with_width(as_str, lambda e: e["ba_std"]),
                                      ga.choose_from_pareto(par))


def test_zero_width_picks_cheapest_among_best():
    par = [{"bits": "1" * 19, "ba_mean": 0.6, "ba_std": 0.1, "cost": 1.0},
           {"bits": "11000000" + "1000" + "1000000", "ba_mean": 0.55, "ba_std": 0.0, "cost": 0.2}]
    assert chromosome.to_str(PT.choose_with_width(par, lambda e: 0.0)) == "1" * 19
    assert chromosome.to_str(PT.choose_with_width(par, lambda e: e["ba_std"])) == par[1]["bits"]


def test_end_to_end_on_synthetic(tmp_path):
    from experiments import run_ga
    cache = tmp_path / "cache"
    synth.write_session("91", "01", cache, CFG)
    res = tmp_path / "results"
    for m in ("ga_nsga", "ga_single"):
        run_ga.main(["--method", m, "--subjects", "91", "--sessions", "01", "--seeds", "0",
                     "--n-jobs", "1", "--pop-size", "8", "--n-generations", "2",
                     "--cache-dir", str(cache), "--results-dir", str(res)])
    folds = PT.run(res, cache, "block", [0.0, 0.5, 1.0])
    nsga = folds[folds.method == "ga_nsga"]
    assert set(nsga.rule) == {"stored", "0sd", "0.5sd", "1sd", "se", "knee", "mincost"}
    stored = nsga[nsga.rule == "stored"].set_index("fold")
    one = nsga[nsga.rule == "1sd"].set_index("fold")
    assert (stored.bits == one.bits).all()
    assert np.allclose(stored.outer_ba, one.outer_ba)      # refit reproduces run_ga's score
    assert set(folds[folds.method == "ga_single"].rule) == {"stored"}
    summ = PT.summarise(folds)
    assert {"outer_ba_mean", "gap_mean", "n_subjects"} <= set(summ.columns)
    assert (nsga[nsga.rule == "mincost"].cost.to_numpy() <= stored.cost.to_numpy() + 1e-12).all()
    assert PT.main(["--results-dir", str(res), "--cache-dir", str(cache), "--protocol", "block",
                    "--no-compare"]) == 0
    out = res / "p5" / "chooser_rules"
    assert (out / "pareto_thresholds_summary.md").exists()
    assert not list(out.glob("*.json"))  # never read by analysis.load_results


def _e(bits, ba, cost, sd=0.05):
    return {"bits": bits, "ba_mean": ba, "ba_std": sd, "cost": cost}


def test_min_cost_and_knee():
    par = [_e("1" * 19, 0.60, 1.0), _e("11110000" + "1100" + "1100000", 0.58, 0.45),
           _e("11000000" + "1000" + "1000000", 0.45, 0.21), _e("11100000" + "1000" + "1000000", 0.44, 0.25)]
    assert chromosome.to_str(PT.choose_min_cost(par)) == par[2]["bits"]
    # 2-D front: (0.21, .45), (0.45, .58), (1.0, .60); the middle one is far above the chord
    assert chromosome.to_str(PT.choose_knee(par)) == par[1]["bits"]
    # fewer than 3 front points -> most accurate entry
    assert chromosome.to_str(PT.choose_knee(par[:1] + par[2:3])) == par[0]["bits"]

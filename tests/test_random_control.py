"""Tests for experiments/random_control.py (P5 random-chromosome control)."""
import json

import numpy as np
import pytest

from experiments import random_control as RC
from src import chromosome as CH, synth
from src.config import load_config

CFG = load_config()


def test_draws_match_counts_cost_and_are_seeded():
    bits = CH.from_str("0101000010000100100")
    a = RC.draw_matched(bits, 20, np.random.default_rng([1, 2, 3]))
    b = RC.draw_matched(bits, 20, np.random.default_rng([1, 2, 3]))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    for x in a:
        assert CH.is_valid(x) and CH.counts(x) == CH.counts(bits) and CH.cost(x) == CH.cost(bits)
    assert len({CH.to_str(x) for x in a}) > 10  # actually random


def test_draws_are_uniform_over_genes():
    bits = CH.from_str("1100000010001000000")  # 2 channels, 1 band, 1 window
    B = np.stack(RC.draw_matched(bits, 4000, np.random.default_rng(0)))
    assert np.allclose(B[:, :8].mean(0), 2 / 8, atol=0.03)
    assert np.allclose(B[:, 8:12].mean(0), 1 / 4, atol=0.03)
    assert np.allclose(B[:, 12:].mean(0), 1 / 7, atol=0.03)


def test_percentile_midrank():
    r = np.array([0.30, 0.33, 0.33, 0.40])
    assert RC.percentile(0.50, r) == 1.0
    assert RC.percentile(0.20, r) == 0.0
    assert RC.percentile(0.33, r) == pytest.approx((1 + 0.5 * 2) / 4)


def test_end_to_end_rescores_ga_exactly(tmp_path):
    from experiments import run_ga
    cache = tmp_path / "cache"
    for ses in ("01", "02"):
        synth.write_session("93", ses, cache, CFG)
    res = tmp_path / "results"
    for prot in ("block", "cross_session"):
        run_ga.main(["--method", "ga_nsga", "--subjects", "93", "--protocol", prot, "--seeds", "0",
                     "--pop-size", "6", "--n-generations", "2", "--n-jobs", "1",
                     "--cache-dir", str(cache), "--results-dir", str(res)])
    out = tmp_path / "rc"
    RC.main(["--methods", "ga_nsga", "--n-draws", "4", "--n-jobs", "1", "--results-root", str(res),
             "--out", str(out), "--cache-dir", str(cache), "--no-summary"])
    files = sorted(p.name for p in (out / "ga_nsga").glob("*.json"))
    assert files == ["sub-93_ses-01_seed-0.json", "sub-93_ses-01to02_seed-0.json",
                     "sub-93_ses-02_seed-0.json"]
    r = json.loads((out / "ga_nsga" / "sub-93_ses-01_seed-0.json").read_text())
    ga = json.loads((res / "ga_nsga" / "sub-93_ses-01_seed-0.json").read_text())
    assert [f["ga_ba"] for f in r["folds"]] == [f["ba"] for f in ga["per_fold"]]
    for f in r["folds"]:
        assert len(f["random_ba"]) == 4 and 0 <= f["percentile"] <= 1
        assert all(CH.counts(CH.from_str(b)) == CH.counts(CH.from_str(f["ga_bits"]))
                   for b in f["random_bits"])

"""Smoke tests for experiments/baseline.py and experiments/sanity.py on a small synthetic cache."""
import json

import numpy as np
import pytest

from src import analysis, results, synth
from src.config import load_config

CFG = load_config()


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("exp_synth")
    for ses in ("01", "02"):
        synth.write_session("93", ses, d, CFG)  # ses-01 has 117 trials
    synth.write_session("91", "01", d, CFG)       # only one session -> no cross-session
    return d


def test_baseline_writes_contract_results(synth_dir, tmp_path):
    from experiments import baseline
    recs = baseline.run(synth_dir, tmp_path, n_jobs=1)
    files = sorted(p.name for p in (tmp_path / "fbcsp_all").glob("*.json"))
    assert files == ["sub-91_ses-01_seed-0.json", "sub-93_ses-01_seed-0.json",
                     "sub-93_ses-01to02_seed-0.json", "sub-93_ses-02_seed-0.json"]
    for p in (tmp_path / "fbcsp_all").glob("*.json"):
        rec = json.loads(p.read_text())
        results.validate_result(rec)
        assert rec["method"] == "fbcsp_all" and rec["seed"] == 0
        assert all(f["chosen_bits"] == "1" * 19 and f["n_features"] == 168 for f in rec["per_fold"])
        assert rec["infer_time_ms"] > 0 and rec["synthetic"] is True
        for f in rec["per_fold"]:
            assert len(f["y_true"]) == len(f["y_pred"]) == f["n_test"]
    cross = json.loads((tmp_path / "fbcsp_all" / "sub-93_ses-01to02_seed-0.json").read_text())
    assert cross["protocol"] == "cross_session" and len(cross["per_fold"]) == 1
    assert cross["per_fold"][0]["n_train"] == 117 and cross["per_fold"][0]["n_test"] == 120
    blk = json.loads((tmp_path / "fbcsp_all" / "sub-93_ses-01_seed-0.json").read_text())
    assert [f["test_blocks"] for f in blk["per_fold"]] == [[0], [1], [2], [3]]
    assert sum(f["n_test"] for f in blk["per_fold"]) == 117
    assert "NOTE: synthetic" in baseline.summarise(recs)
    df = analysis.load_results(tmp_path)  # analysis can read what the baseline writes
    assert len(df) == 4


def test_sanity_outputs(synth_dir, tmp_path):
    from experiments import sanity
    s = sanity.run(synth_dir, tmp_path, n_perm=8, random_seeds=(0,), n_jobs=1, subjects=["93"])
    assert s["n_recordings"] == 2
    assert 0.2 < s["a_shuffled"]["null_mean"] < 0.47
    assert s["c_view_vs_imagery"]["view_mean"] > s["b_block_vs_random"]["block_mean"]
    assert (tmp_path / "sanity_summary.md").exists()
    rand = sorted(p.name for p in (tmp_path / "fbcsp_all").glob("*.json"))
    assert rand == ["sub-93_ses-01_seed-0_random.json", "sub-93_ses-02_seed-0_random.json"]
    view = sorted(p.name for p in (tmp_path / "fbcsp_all_view").glob("*.json"))
    assert view == ["sub-93_ses-01_seed-0.json", "sub-93_ses-02_seed-0.json"]
    for p in list((tmp_path / "fbcsp_all").glob("*.json")) + list((tmp_path / "fbcsp_all_view").glob("*.json")):
        results.validate_result(json.loads(p.read_text()))


def test_permute_within_blocks_keeps_block_balance():
    from experiments.sanity import permute_within_blocks
    rng = np.random.default_rng(0)
    y = np.tile(np.repeat([0, 1, 2], 10), 4)
    block = np.repeat(np.arange(4), 30)
    yp = permute_within_blocks(y, block, rng)
    assert not np.array_equal(yp, y)
    for b in range(4):
        assert np.bincount(yp[block == b]).tolist() == [10, 10, 10]


def test_summary_sd_is_sample_sd():
    """Between-recording SDs use ddof=1, the same as reports/make_tables.py."""
    from experiments.sanity import _mean
    rows = [{"x": v} for v in (0.30, 0.35, 0.40, 0.45)]
    m, sd, n = _mean(rows, "x")
    assert n == 4 and np.isclose(m, 0.375) and np.isclose(sd, np.std([0.30, 0.35, 0.40, 0.45], ddof=1))
    assert _mean([{"x": 0.4}], "x")[1] is None

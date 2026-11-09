"""Tests for src/eegnet.py and experiments/run_eegnet.py (need torch + braindecode)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("braindecode")

from experiments import _common as C  # noqa: E402
from src import cache, eegnet, results, synth  # noqa: E402
from src.config import load_config  # noqa: E402

CFG = load_config()


def quick_cfg(epochs=4, seed=0):
    cfg = load_config()
    cfg["eegnet"].update(max_epochs=epochs, early_stopping_patience=2, seed=seed)
    return cfg


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("cache_synth")
    synth.write_synthetic(d, CFG, subjects=("91",))
    return d


def test_validation_carved_from_training_fold_only():
    y = np.tile([0, 1, 2], 40).astype(np.int64)
    tr = np.arange(30, 120, dtype=np.int64)
    fit_idx, val_idx = eegnet.carve_validation(tr, y, 0.2, seed=0)
    assert np.intersect1d(fit_idx, val_idx).size == 0
    assert set(fit_idx) | set(val_idx) == set(tr)        # nothing from outside the train fold
    assert len(val_idx) == 18 and np.bincount(y[val_idx]).tolist() == [6, 6, 6]   # stratified
    np.testing.assert_array_equal(val_idx, eegnet.carve_validation(tr, y, 0.2, seed=0)[1])


def test_overlapping_split_rejected():
    X = np.zeros((12, 32, 1000), np.float32)
    y = np.tile([0, 1, 2], 4).astype(np.int64)
    with pytest.raises(ValueError, match="overlap"):
        eegnet.evaluate_eegnet(X, y, [(np.arange(8), np.arange(6, 12))], cfg=quick_cfg())


def test_evaluate_eegnet_smoke_and_reproducible(synth_dir):
    d, splits = C.load_unit("eegnet", "91", "01", CFG, synth_dir)
    r1 = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:2], cfg=quick_cfg(seed=1))
    assert set(r1) >= {"per_fold", "ba_mean", "ba_std", "train_time_s", "infer_time_ms"}
    assert len(r1["per_fold"]) == 2 and r1["train_time_s"] > 0 and r1["infer_time_ms"] > 0
    assert [len(p) for p in r1["y_pred"]] == [len(te) for _, te in splits[:2]]
    assert r1["device"] == ("cuda" if torch.cuda.is_available() else "cpu")
    r2 = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:2], cfg=quick_cfg(seed=1))
    assert r1["per_fold"] == r2["per_fold"]


def test_eight_channel_option(synth_dir):
    d, splits = C.load_unit("eegnet", "91", "01", CFG, synth_dir)
    idx = cache.candidate_channel_index(d["ch_names"], CFG)
    r8 = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:1], ch_idx=idx, cfg=quick_cfg(2))
    r32 = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:1], cfg=quick_cfg(2))
    assert r8["n_params"] < r32["n_params"]


def test_run_eegnet_cli(synth_dir, tmp_path):
    from experiments import run_eegnet
    run_eegnet.main(["--channels", "32", "8", "--subjects", "sub-91", "--sessions", "01",
                     "--seeds", "0", "--max-epochs", "2", "--cache-dir", str(synth_dir),
                     "--results-dir", str(tmp_path)])
    for m, n in (("eegnet_32ch", 32), ("eegnet_8ch", 8)):
        rec = json.loads((tmp_path / m / "sub-91_ses-01_seed-0.json").read_text())
        results.validate_result(rec)
        assert rec["n_channels"] == n and len(rec["per_fold"]) == 4
        assert all(f["chosen_bits"] is None and f["n_features"] > 0 for f in rec["per_fold"])
        for f in rec["per_fold"]:
            assert f["n_test"] == len(f["y_true"]) == len(f["y_pred"]) == len(f["test_trial_id"])
        assert rec["infer_time_ms"] > 0 and rec["infer_time_ms_batch1"] > 0
    run_eegnet.main(["--channels", "8", "--subjects", "91", "--protocol", "cross_session",
                     "--seeds", "2", "--max-epochs", "2", "--cache-dir", str(synth_dir),
                     "--results-dir", str(tmp_path)])
    rec = json.loads((tmp_path / "eegnet_8ch" / "sub-91_ses-01to02_seed-2.json").read_text())
    assert rec["session"] == "01to02" and len(rec["per_fold"]) == 1


def test_run_eegnet_device_flag_keeps_config_hash(synth_dir, tmp_path):
    # CONTRACT_ISSUES R07-11: --device must not change config_hash, --max-epochs must.
    from experiments import run_eegnet
    from src.config import config_hash
    args = ["--channels", "8", "--subjects", "91", "--sessions", "01", "--seeds", "0",
            "--cache-dir", str(synth_dir)]
    hashes = {}
    for name, extra in (("plain", []), ("device", ["--device", "cpu"])):
        out = tmp_path / name
        run_eegnet.main(args + ["--max-epochs", "2", "--results-dir", str(out)] + extra)
        rec = json.loads((out / "eegnet_8ch" / "sub-91_ses-01_seed-0.json").read_text())
        hashes[name] = rec["config_hash"]
        if extra:
            assert rec["device"] == "cpu"
    assert hashes["plain"] == hashes["device"]
    assert hashes["plain"] != config_hash(CFG)  # --max-epochs is a design override


def test_history_logging_does_not_change_results(synth_dir):
    # P5: log_history only observes (eval-mode passes, no RNG), restore_best=False is fixed-length.
    from src.config import config_hash
    d, splits = C.load_unit("eegnet", "91", "01", CFG, synth_dir)
    plain = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:2], cfg=quick_cfg(4))
    cfg_log = quick_cfg(4)
    cfg_log["eegnet"]["log_history"] = True
    logged = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:2], cfg=cfg_log)
    assert all(np.array_equal(a, b) for a, b in zip(plain["y_pred"], logged["y_pred"]))
    assert plain["best_epoch"] == logged["best_epoch"] and plain["history"] == [None, None]
    for h, n in zip(logged["history"], logged["n_epochs"]):
        assert len(h) == n and {"train_loss", "train_ba", "val_loss", "val_ba"} <= set(h[0])
    assert all(0 <= v <= 1 for v in logged["train_ba"] + logged["val_ba"])
    assert config_hash(cfg_log) == config_hash(quick_cfg(4))
    fixed = quick_cfg(4)
    fixed["eegnet"].update(early_stopping_patience=4, restore_best=False)
    r = eegnet.evaluate_eegnet(d["X"], d["y"], splits[:1], cfg=fixed)
    assert r["n_epochs"] == [4]
    assert config_hash(fixed) != config_hash(quick_cfg(4))


def test_eegnet_diag_reproduces_p4_and_writes_variants(synth_dir, tmp_path):
    from experiments import eegnet_diag, run_eegnet
    p4 = tmp_path / "p4"
    run_eegnet.main(["--channels", "8", "--subjects", "91", "--sessions", "01", "--seeds", "0",
                     "--max-epochs", "3", "--cache-dir", str(synth_dir), "--results-dir", str(p4)])
    out, hist = tmp_path / "diag", tmp_path / "hist"
    eegnet_diag.main(["--variants", "logged", "fixed100", "--channels", "8", "--subjects", "91",
                      "--sessions", "01", "--protocols", "block", "--max-epochs-cap", "3", "--n-jobs", "2",
                      "--cache-dir", str(synth_dir), "--out", str(out), "--history-dir", str(hist),
                      "--p4-root", str(p4)])
    s = json.loads((out / "summary.json").read_text())
    assert s["reproduces_p4"]["eegnet_8ch"] == {"identical": 1, "n": 1}
    rec = json.loads((out / "eegnet_8ch_fixed100" / "sub-91_ses-01_seed-0.json").read_text())
    results.validate_result(rec)
    assert rec["exploratory"] is True and all(f["n_epochs"] == 3 for f in rec["per_fold"])
    assert (hist / "eegnet_8ch_logged" / "sub-91_ses-01_seed-0.json").exists()
    assert (out / "curves.csv").exists() and (out / "variants.csv").exists()

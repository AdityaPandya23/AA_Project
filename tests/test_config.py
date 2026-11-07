from pathlib import Path

from src.config import config_hash, load_config, repo_root


def test_defaults_relative_to_repo():
    cfg = load_config()
    assert cfg["paths"]["data_dir"] == repo_root() / "data"
    assert cfg["paths"]["cache_dir"] == repo_root() / "cache"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("AA_DATA_DIR", str(tmp_path / "bids"))
    monkeypatch.setenv("AA_CACHE_DIR", str(tmp_path / "c"))
    cfg = load_config()
    assert cfg["paths"]["data_dir"] == tmp_path / "bids"
    assert cfg["paths"]["cache_dir"] == tmp_path / "c"


def test_hash_ignores_paths_but_not_design(monkeypatch, tmp_path):
    h0 = config_hash(load_config())
    monkeypatch.setenv("AA_CACHE_DIR", str(tmp_path))
    cfg = load_config()
    assert config_hash(cfg) == h0 and len(h0) == 10
    cfg["ga"]["pop_size"] = 31
    assert config_hash(cfg) != h0


def test_fixed_design_values():
    cfg = load_config()
    assert cfg["classes"] == {"circle": 0, "square": 1, "pentagram": 2}
    assert cfg["preprocess"]["sfreq"] == 250 and cfg["preprocess"]["epoch_samples"] == 1000
    assert cfg["windows"]["length_samples"] == 250
    assert cfg["windows"]["starts_s"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert cfg["bands"]["edges_hz"] == {"theta": [4.0, 7.0], "alpha": [8.0, 12.0],
                                        "beta": [13.0, 30.0], "lowgamma": [30.0, 40.0]}
    assert cfg["fbcsp"]["lda"] == {"solver": "lsqr", "shrinkage": "auto"}
    assert (cfg["ga"]["pop_size"], cfg["ga"]["n_generations"], cfg["ga"]["inner_folds"],
            cfg["ga"]["seeds"]) == (30, 20, 3, [0, 1, 2])

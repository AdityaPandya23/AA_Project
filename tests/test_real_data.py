"""Real-data checks for sub-01 (skipped when the raw data / cache are absent).

* io: trial table, blocks, marker cross-check against the Status channel.
* cache: shapes and dtypes on the real sub-01 caches match CONTRACTS.md §a–c,
  trials are aligned across the three kinds, and class balance is printed.

Build the cache first: ``python -m scripts.build_cache --subjects 01``.
Run with ``pytest -s tests/test_real_data.py`` to see the class balance.
"""
import numpy as np
import pytest

from src import cache
from src import io as dio
from src.config import load_config

CFG = load_config()
SUB = "01"
PAIRS = [p for p in dio.list_subject_sessions(CFG) if p[0] == SUB] \
    if CFG["paths"]["data_dir"].exists() else []
needs_raw = pytest.mark.skipif(not PAIRS, reason="sub-01 raw data not found")


def _cached(sub, ses):
    return all(cache.cache_path(k, sub, ses, CFG).exists() for k in cache.KINDS)


CACHED = [p for p in PAIRS if _cached(*p)]
needs_cache = pytest.mark.skipif(not CACHED, reason="sub-01 cache not built")


@needs_raw
def test_channels_tsv_has_32_eeg_and_candidates():
    ch = dio.read_channels_tsv(SUB, "01", CFG)
    assert len(ch) == 32 and set(ch["type"]) == {"EEG"}
    assert set(CFG["channels"]["candidates"]) <= set(ch["name"])


@needs_raw
@pytest.mark.parametrize("ses", ["01", "02"])
def test_read_trials_structure(ses):
    t = dio.read_trials(SUB, ses, CFG)
    assert len(t) == 120
    assert t["trial_id"].tolist() == list(range(120))
    assert np.bincount(t["y"]).tolist() == [40, 40, 40]
    for b in range(4):  # 4 blocks of 30, each 10/10/10
        assert np.bincount(t.loc[t["block"] == b, "y"], minlength=3).tolist() == [10, 10, 10]
    assert t["block"].tolist() == sorted(t["block"].tolist())
    np.testing.assert_allclose(t["imagery_onset_s"] - t["onset_s"], 9.0)
    np.testing.assert_allclose(np.median(np.diff(t["onset_s"])), 17.1, atol=0.1)


@needs_raw
def test_events_match_status_channel():
    raw = dio.read_raw(SUB, "01", CFG)
    assert raw.ch_names.count("Status") == 1 and len(raw.copy().pick("eeg").ch_names) == 32
    dio.read_trials(SUB, "01", CFG, raw=raw)  # raises DataError on mismatch


@needs_cache
@pytest.mark.parametrize("sub,ses", CACHED)
def test_real_cache_contract(sub, ses):
    ds = {k: cache.load_npz(cache.cache_path(k, sub, ses, CFG)) for k in cache.KINDS}
    n = ds["fbcsp"]["y"].shape[0]
    assert 0 < n <= 120
    for kind in ("fbcsp", "view"):
        d = ds[kind]
        assert d["X"].dtype == np.float32 and d["X"].shape == (4, 7, n, 8, 250)
        assert d["ch_names"].dtype.kind == "U"
        assert d["ch_names"].tolist() == CFG["channels"]["candidates"]
        assert d["band_names"].tolist() == CFG["bands"]["names"]
        np.testing.assert_allclose(d["window_starts"], CFG["windows"]["starts_s"])
        assert float(d["sfreq"]) == 250.0
        cache.validate_fbcsp(d, CFG)
    e = ds["eegnet"]
    assert e["X"].dtype == np.float32 and e["X"].shape == (n, 32, 1000)
    assert e["ch_names"].dtype.kind == "U" and e["ch_names"].shape == (32,)
    cache.validate_eegnet(e, CFG)
    for k in ("y", "block", "trial_id"):
        assert ds["fbcsp"][k].dtype == np.int64
        np.testing.assert_array_equal(ds["fbcsp"][k], ds["eegnet"][k])
        np.testing.assert_array_equal(ds["fbcsp"][k], ds["view"][k])
    assert set(ds["fbcsp"]["block"].tolist()) == {0, 1, 2, 3}
    assert int(ds["fbcsp"]["n_rejected"]) + n == 120
    assert not np.isin(ds["fbcsp"]["rejected_trial_id"], ds["fbcsp"]["trial_id"]).any()
    # loads without pickle through the public loader
    for k in cache.KINDS:
        cache.load_cache(k, sub, ses, CFG)


@needs_cache
@pytest.mark.parametrize("sub,ses", CACHED)
def test_real_cache_signal_sanity(sub, ses):
    """µV scale, broadband EEGNet window == 1-40 Hz content, candidate channels consistent."""
    e = cache.load_npz(cache.cache_path("eegnet", sub, ses, CFG))
    sd = e["X"].std(axis=(0, 2))
    assert 1.0 < np.median(sd) < 100.0, f"median channel SD {np.median(sd):.2f} not in µV range"
    f = cache.load_npz(cache.cache_path("fbcsp", sub, ses, CFG))
    # alpha-band window variance should be a sizeable share of broadband variance on O1
    o1 = list(e["ch_names"]).index("O1")
    a = CFG["bands"]["names"].index("alpha")
    ratio = f["X"][a, 0, :, 0, :].var() / e["X"][:, o1, :250].var()
    assert 0.02 < ratio < 1.0


@needs_cache
def test_print_class_balance():
    inv = {v: k for k, v in CFG["classes"].items()}
    print()
    for sub, ses in CACHED:
        d = cache.load_npz(cache.cache_path("fbcsp", sub, ses, CFG))
        cnt = np.bincount(d["y"], minlength=3)
        per_block = [np.bincount(d["y"][d["block"] == b], minlength=3).tolist() for b in range(4)]
        print(f"sub-{sub} ses-{ses}: n={d['y'].size} "
              + " ".join(f"{inv[c]}={cnt[c]}" for c in range(3))
              + f" rejected={int(d['n_rejected'])} per-block={per_block}")
        assert cnt.min() >= 30, "class lost more than 25% of its trials"

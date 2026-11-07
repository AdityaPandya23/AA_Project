import numpy as np
import pytest

from src import chromosome as C
from src.config import load_config


@pytest.fixture
def rng():
    return np.random.default_rng(123)


def test_constants_match_config():
    cfg = load_config()
    assert C.N_CH == len(cfg["channels"]["candidates"])
    assert C.N_BAND == len(cfg["bands"]["names"])
    assert C.N_WIN == len(cfg["windows"]["starts_s"])
    assert C.N_BITS == cfg["chromosome"]["n_bits"] == 19
    assert (C.MIN_CH, C.MIN_BAND, C.MIN_WIN) == (
        cfg["chromosome"]["min_channels"], cfg["chromosome"]["min_bands"],
        cfg["chromosome"]["min_windows"])


def test_decode_encode_roundtrip(rng):
    for _ in range(50):
        b = (rng.random(19) < 0.5).astype(np.uint8)
        ch, band, win = C.decode(b)
        assert ch.shape == (8,) and band.shape == (4,) and win.shape == (7,)
        assert ch.dtype == bool
        np.testing.assert_array_equal(C.encode(ch, band, win), b)


def test_decode_layout():
    b = np.zeros(19, dtype=int)
    b[0] = 1    # O1
    b[8] = 1    # theta
    b[18] = 1   # window start 3.0
    ch, band, win = C.decode(b)
    assert ch.tolist() == [True] + [False] * 7
    assert band.tolist() == [True, False, False, False]
    assert win.tolist() == [False] * 6 + [True]


@pytest.mark.parametrize("bad", [[0, 1] * 9, [0] * 20, [2] + [0] * 18])
def test_bad_input_raises(bad):
    with pytest.raises(ValueError):
        C.decode(bad)


def test_accepts_list_bool_and_int():
    ones = [1] * 19
    for b in (ones, tuple(ones), np.ones(19, bool), np.ones(19, np.int64)):
        assert C.is_valid(b)
        assert C.cost(b) == 1.0


def test_is_valid_boundaries():
    b = np.zeros(19, np.uint8)
    assert not C.is_valid(b)
    b[[0, 1, 8, 12]] = 1
    assert C.is_valid(b)
    b[1] = 0  # only 1 channel
    assert not C.is_valid(b)


def test_repair_makes_valid_only_adds_and_minimal(rng):
    for _ in range(500):
        b = (rng.random(19) < rng.random()).astype(np.uint8)
        r = C.repair(b, rng)
        assert C.is_valid(r)
        assert np.all(r >= b), "repair must never switch genes off"
        n_ch, n_b, n_w = C.counts(r)
        ob = C.counts(b)
        assert n_ch == max(ob[0], 2) and n_b == max(ob[1], 1) and n_w == max(ob[2], 1)


def test_repair_valid_unchanged_and_not_in_place(rng):
    b = C.random_valid(rng)
    before = b.copy()
    r = C.repair(b, rng)
    np.testing.assert_array_equal(r, before)
    assert r is not b
    z = np.zeros(19, np.uint8)
    C.repair(z, rng)
    assert z.sum() == 0, "repair must not mutate its input"


def test_repair_all_zero():
    r = C.repair(np.zeros(19), np.random.default_rng(0))
    assert C.counts(r) == (2, 1, 1)


def test_cost_values():
    assert C.cost(C.all_on()) == pytest.approx(1.0)
    minimal = C.repair(np.zeros(19), np.random.default_rng(0))
    assert C.cost(minimal) == pytest.approx((2 / 8 + 1 / 4 + 1 / 7) / 3)
    b = np.zeros(19)
    b[:4] = 1; b[8:10] = 1; b[12:19] = 1
    assert C.cost(b) == pytest.approx((4 / 8 + 2 / 4 + 7 / 7) / 3)


def test_random_valid_deterministic_and_valid():
    a = [C.random_valid(np.random.default_rng(7)) for _ in range(2)]
    np.testing.assert_array_equal(a[0], a[1])
    rng = np.random.default_rng(0)
    for p in (0.0, 0.1, 0.5, 1.0):
        for _ in range(100):
            b = C.random_valid(rng, p)
            assert C.is_valid(b) and b.dtype == np.uint8 and b.shape == (19,)


def test_n_features():
    assert C.n_features(C.all_on()) == 6 * 4 * 7
    b = np.zeros(19); b[[0, 1, 9, 12, 13, 14]] = 1
    assert C.n_features(b) == 6 * 1 * 3


def test_str_roundtrip(rng):
    for _ in range(20):
        b = C.random_valid(rng)
        s = C.to_str(b)
        assert len(s) == 19 and set(s) <= {"0", "1"}
        np.testing.assert_array_equal(C.from_str(s), b)
    with pytest.raises(ValueError):
        C.from_str("0101")


def _labelled_X(n=5):
    """X where each element encodes its (band, window, channel) for checking select order."""
    X = np.zeros((4, 7, n, 8, 250), np.float32)
    for b in range(4):
        for w in range(7):
            for c in range(8):
                X[b, w, :, c, :] = 100 * b + 10 * w + c
    X += np.arange(n, dtype=np.float32)[None, None, :, None, None] * 1000
    return X


def test_select_shape_and_order():
    X = _labelled_X()
    b = np.zeros(19, np.uint8)
    b[[1, 4, 6]] = 1         # channels Oz, P3, P7
    b[[8 + 1, 8 + 3]] = 1    # bands alpha, lowgamma
    b[[12 + 0, 12 + 5]] = 1  # windows 0.0, 2.5
    Xs = C.select(X, b)
    assert Xs.shape == (5, 4, 3, 250)
    assert Xs.dtype == np.float32
    expected_bw = [(1, 0), (1, 5), (3, 0), (3, 5)]  # band-major
    for i_bw, (band, win) in enumerate(expected_bw):
        for i_c, ch in enumerate([1, 4, 6]):
            for t in range(5):
                assert Xs[t, i_bw, i_c, 0] == 1000 * t + 100 * band + 10 * win + ch


def test_select_all_on_and_copy():
    X = _labelled_X(3)
    Xs = C.select(X, C.all_on())
    assert Xs.shape == (3, 28, 8, 250)
    Xs[...] = -1
    assert X.min() >= 0, "select must return a copy"


def test_select_rejects_invalid():
    X = _labelled_X(2)
    with pytest.raises(ValueError):
        C.select(X, np.zeros(19))
    with pytest.raises(ValueError):
        C.select(X[:, :, :, :4], C.all_on())


def test_describe():
    cfg = load_config()
    #                channels    bands   windows
    d = C.describe(C.from_str("10000100" + "0100" + "1000001"), cfg)
    assert d == {"channels": ["O1", "P4"], "bands": ["alpha"], "windows": [0.0, 3.0]}

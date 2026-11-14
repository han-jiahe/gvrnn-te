"""Tests for latent_pair_te.py (10 Hz player-to-player latent TE)."""
import numpy as np

from latent_pair_te import pair_mask, te_pair


def test_pair_mask_requires_both_players_live():
    frame = np.arange(10, 18)
    periods = np.ones(8, dtype=int)
    live_a = np.ones(8, dtype=bool)
    live_b = np.ones(8, dtype=bool)
    live_a[2] = False      # A off at t=2 -> transitions t1->2 and t2->3 masked
    live_b[6] = False      # B off at t=6 -> transitions t5->6 and t6->7 masked
    mask = pair_mask(frame, periods, live_a, live_b)
    assert mask.tolist() == [True, False, False, True, True, False, False]


def test_pair_mask_frame_gap_and_period():
    frame = np.array([10, 11, 20, 21, 22])
    periods = np.array([1, 1, 1, 2, 2])
    live = np.ones(5, dtype=bool)
    mask = pair_mask(frame, periods, live, live)
    assert mask.tolist() == [True, False, False, True]   # gap + period break


def test_te_pair_matches_manual_and_surrogates():
    rng = np.random.default_rng(3)
    T = 600
    sym_b = rng.integers(0, 3, T)
    sym_a = sym_b.copy()                      # copy -> TE should be ~0
    mask = np.ones(T - 1, dtype=bool)
    rng2 = np.random.default_rng(42)
    st = te_pair(sym_a, sym_b, mask, k=3, rng=rng2, n_surr=20, block=25)
    assert st["te"] < 1e-12                   # x=y -> p(y'|y,x)=p(y'|y)
    assert 1 / 21 <= st["p"] <= 1.0
    assert st["n"] == T - 1
    # strongly coupled pair: A determines B's change -> large positive TE
    y = np.zeros(T, dtype=int)
    x = rng.integers(0, 3, T)
    y[1:] = (y[:-1] + x[:-1]) % 3
    st2 = te_pair(x, y, mask, k=3, rng=np.random.default_rng(0), n_surr=20,
                  block=25)
    assert st2["te"] > 0.5                   # near H(y'|y) = log2(3) ~ 1.585
    assert st2["eff"] > 0.3

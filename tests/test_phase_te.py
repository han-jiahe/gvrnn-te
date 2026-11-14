"""Tests for phase_te.py (latent chain -> phase chain TE)."""
import numpy as np

from phase_te import (block_shuffle, p_value, te_discrete, transition_mask)


def test_te_discrete_matches_manual():
    rng = np.random.default_rng(0)
    cnt = rng.integers(0, 20, size=(3, 3, 2))
    p = cnt / cnt.sum()
    py = p.sum(axis=(0, 2))
    pyx = p.sum(axis=0)
    pyy = p.sum(axis=2)
    manual = 0.0
    for a in range(3):
        for b in range(3):
            for c in range(2):
                pabc = p[a, b, c]
                if pabc > 0:
                    manual += pabc * np.log2(pabc * py[b] / (pyy[a, b] * pyx[b, c]))
    assert abs(te_discrete(cnt)["te"] - manual) < 1e-12
    assert te_discrete(cnt)["n"] == cnt.sum()


def test_te_zero_when_source_determined_by_target_history():
    # x_t = y_t -> p(y'|y,x) = p(y'|y) exactly, plug-in TE == 0
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 4000)
    x = y[:-1]                                   # x_t = y_t
    codes = (y[1:] * 3 + y[:-1]) * 3 + x
    cnt = np.bincount(codes, minlength=27).reshape(3, 3, 3)
    assert te_discrete(cnt)["te"] < 1e-12


def test_transition_mask():
    frame = np.array([10, 11, 12, 20, 21, 22, 23, 30])
    periods = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    live = np.array([True, True, True, True, True, True, False, False])
    mask = transition_mask(frame, periods, live)
    # t0->1 ok; t1->2 ok; t2->3 frame gap; t3->4 period break; t4->5 ok;
    # t5->6 not live at t+1; t6->7 frame gap (23 -> 30)
    assert mask.tolist() == [True, True, False, False, True, False, False]
    covered = np.array([True, False, True, True, True, True, True, True])
    mask2 = transition_mask(frame, periods, live, covered)
    # covered[1] is False -> transitions t0->1 and t1->2 are both masked
    assert mask2.tolist() == [False, False, False, False, True, False, False]


def test_block_shuffle_preserves_multiset_and_block_units():
    rng = np.random.default_rng(7)
    x = np.arange(30)
    xs = block_shuffle(x, 5, rng)
    assert sorted(xs.tolist()) == list(range(30))          # multiset
    orig = {tuple(range(i, i + 5)) for i in range(0, 30, 5)}
    got = {tuple(sorted(xs[i:i + 5].tolist())) for i in range(0, 30, 5)}
    assert got == orig                                      # blocks kept whole


def test_p_value_bounds():
    surrs = [0.1, 0.2, 0.3]
    assert p_value(surrs, 10.0) == 0.25          # above all -> 1/(S+1)
    assert p_value(surrs, 0.2) == 0.75           # ties count >= raw
    assert 0.0 < p_value(surrs, -1.0) <= 1.0

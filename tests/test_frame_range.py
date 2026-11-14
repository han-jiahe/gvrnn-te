"""Unit tests for run_player_mte's windowing extensions (--frame-range,
--pairs, --mask-gaps) and the mask_and_stride / run_pairs refactor."""
import numpy as np
import pytest

import config
from run_player_te import frame_numbers, mask_and_stride, run_pairs


def _synthetic(T=1000, K=3, L=8, seed=0):
    """Synthetic latents dict: frames 10..10000 step 10, a dead-ball gap
    injected at row 400 (frame +50), period break at row 500."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(T, K, L)).astype(np.float32)
    frames = np.arange(10, 10 * T + 10, 10, dtype=np.int64)
    frames[400:] += 50                              # gap: 4010 -> 4060
    periods = np.concatenate([np.ones(500, dtype=int), np.full(500, 2, dtype=int)])
    ft = frames.astype(np.float64) / 10.0
    return z, periods, ft, frames


@pytest.fixture(scope="module")
def syn():
    return _synthetic()


def test_range_filter_exact(syn):
    z, periods, ft, frames = syn
    # frame r (r<400) = 10 + 10r; range [110, 4010] -> rows 10..399 (390 rows)
    zs, ps, fts, fs, pm = mask_and_stride(z, periods, ft, frames,
                                          frame_range=[110, 4010], stride=10)
    assert fs[0] == 110 and fs[-1] == 3910   # rows 10..390 step 10
    assert len(zs) == 39


def test_stride_anchors_after_filter(syn):
    z, periods, ft, frames = syn
    zs, ps, fts, fs, pm = mask_and_stride(z, periods, ft, frames,
                                          frame_range=[110, 4010], stride=10)
    assert fs[1] == 210                     # stride grid anchored at range start


def test_period_break_masked(syn):
    z, periods, ft, frames = syn
    # frame r (r>=400) = 60 + 10r; row 499 -> 5050 is period 1's last row
    # (pm[i] is the transition FROM selected row i, so the masked entry is
    # the index of row 499, i.e. the transition 499 -> 500)
    zs, ps, fts, fs, pm = mask_and_stride(z, periods, ft, frames,
                                          frame_range=[4500, 5500], stride=1)
    idx = np.flatnonzero(fs == 5050)[0]
    assert not pm[idx]
    assert bool(pm[idx - 1]) and bool(pm[idx + 1])


def test_gap_masking(syn):
    z, periods, ft, frames = syn
    # rows 399 (4000) and 400..403 (4060..4090): the injected gap sits
    # BETWEEN row 399 and 400; stride 4 keeps rows 399 and 403, so the
    # single transition's stride block contains the gap
    zs, ps, fts, fs, pm_on = mask_and_stride(z, periods, ft, frames,
                                             frame_range=[4000, 4090],
                                             stride=4, mask_gaps=True)
    zs, ps, fts, fs, pm_off = mask_and_stride(z, periods, ft, frames,
                                              frame_range=[4000, 4090],
                                              stride=4, mask_gaps=False)
    assert len(zs) == 2
    assert fs.tolist() == [4000, 4090]
    assert not pm_on[0] and bool(pm_off[0])


def test_empty_range_raises(syn):
    z, periods, ft, frames = syn
    with pytest.raises(ValueError):
        mask_and_stride(z, periods, ft, frames, frame_range=[99999, 100000])


def test_pairs_option():
    rng = np.random.default_rng(7)
    z = rng.normal(size=(300, 3, 4)).astype(np.float32)
    pm = np.ones(len(z) - 1, dtype=bool)
    cfg = config.DEFAULT_TE_CONFIG
    opts = dict(pca_dims=[1], k=[3], n_clusters=[3], block_len=5,
                n_surrogates=5, seed=0)
    results, skipped, frozen, n = run_pairs(z, [0, 1, 2], [(0, 1)], cfg, opts,
                                            pm)
    assert set(results.keys()) == {"0->1"}
    assert "gaussian" in results["0->1"] and "ksg" in results["0->1"]
    assert n == len(z) - 1


def test_frame_numbers_fallback():
    lat = {"frame_times": np.array([0.0, 0.1, 4285.9], dtype=np.float32)}
    assert frame_numbers(lat).tolist() == [0, 1, 42859]
    lat2 = {"frame": np.array([10, 20, 30]), "frame_times": np.array([1.0, 2.0, 3.0])}
    assert frame_numbers(lat2).tolist() == [10, 20, 30]

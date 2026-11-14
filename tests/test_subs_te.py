"""Tests for run_subs_te.py (dotTE aggregation per GVRNN protocol)."""
import numpy as np
import pandas as pd

from run_subs_te import parse_swap_pairs, dot_te_rows


def _rows():
    return pd.DataFrame([
        {"src": 100, "dst": 200, "te": 0.12, "eff": 0.10, "p": 0.01,
         "skipped": False},
        {"src": 100, "dst": 201, "te": 0.06, "eff": 0.05, "p": 0.02,
         "skipped": False},
        # swap partner of 100: protocol-excluded even though not skipped
        {"src": 100, "dst": 999, "te": 0.99, "eff": 0.99, "p": 0.001,
         "skipped": False},
        # no co-presence window
        {"src": 100, "dst": 202, "te": np.nan, "eff": np.nan, "p": np.nan,
         "skipped": True},
    ])


def test_parse_swap_pairs():
    assert parse_swap_pairs("43829,133501;163972,50951") == \
        [(43829, 133501), (163972, 50951)]
    assert parse_swap_pairs("1,2") == [(1, 2)]


def test_dot_te_aggregation_and_partner_exclusion():
    swap = {100: 999, 999: 100}
    minutes = {100: 10.0, 999: 10.0}
    dots = dot_te_rows(_rows(), [100], swap, minutes)
    d = dots[100]
    assert d["n_targets"] == 2                     # partner + skipped excluded
    assert abs(d["sum_eff"] - 0.15) < 1e-12        # 0.10 + 0.05
    assert abs(d["sum_raw"] - 0.18) < 1e-12        # 0.12 + 0.06
    assert abs(d["dot_te"] - 0.15 / 10.0) < 1e-12  # sum(eff) / t
    assert abs(d["dot_te_raw"] - 0.18 / 10.0) < 1e-12
    assert abs(d["mean_eff_per_target"] - 0.075 / 10.0) < 1e-12


def test_dot_te_all_skipped_gives_none():
    rows = _rows()
    rows["skipped"] = True
    dots = dot_te_rows(rows, [100], {100: 999, 999: 100}, {100: 10.0})
    d = dots[100]
    assert d["n_targets"] == 0
    assert d["dot_te"] is None


def test_dot_te_source_not_in_df():
    dots = dot_te_rows(_rows(), [777], {777: 888, 888: 777}, {777: 5.0})
    assert dots[777]["n_targets"] == 0
    assert dots[777]["dot_te"] is None

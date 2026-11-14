"""Tests for latent_phase.py (discrete-count joint probability)."""
import numpy as np
import pandas as pd
import pytest

from latent_phase import (DEAD, assign_phases, count_joint, fit_symbols,
                          mi_of)


@pytest.fixture
def phases_df():
    # rows: [0,5) team1 build_up; [5,10) team2 direct/high_block;
    #       gap [10,12); [12,15) team1 create
    return pd.DataFrame({
        "frame_start": [0, 5, 12],
        "frame_end": [5, 10, 15],
        "team_in_possession_id": [1, 2, 1],
        "team_in_possession_phase_type": ["build_up", "direct", "create"],
        "team_out_of_possession_phase_type": ["medium_block", "high_block",
                                              "medium_block"],
    })


def test_boundary_half_open(phases_df):
    frames = np.arange(0, 17)
    phi, idx, covered = assign_phases(phases_df, frames, team_id=1)
    # [0,5) -> row0; boundary frame 5 -> row1 (half-open)
    assert list(phi[0:5]) == ["build_up"] * 5
    assert phi[5] == "high_block"          # team2 in possession -> out type
    assert list(phi[5:10]) == ["high_block"] * 5
    # gap [10,12) -> dead; [12,15) -> create; tail -> dead
    assert list(phi[10:12]) == [DEAD, DEAD]
    assert list(phi[12:15]) == ["create"] * 3
    assert phi[15] == DEAD and phi[16] == DEAD
    assert covered.sum() == 13
    # team-relative label: team2's view of row0 is the out-of-possession type
    phi2, _, _ = assign_phases(phases_df, frames, team_id=2)
    assert phi2[0] == "medium_block" and phi2[12] == "medium_block"
    assert phi2[5] == "direct"


def test_count_joint_and_mi():
    phi = np.array(["a", "a", "a", "b", "b", "b", "b", "c", "c", "c"])
    lab = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1])
    cnt, phis = count_joint(phi, lab, k=2)
    assert list(phis) == ["a", "b", "c"]
    assert cnt.tolist() == [[2, 1], [2, 2], [1, 2]]
    st = mi_of(cnt)
    assert st["N"] == 10
    # manual: p(a)=0.3 p(b)=0.4 p(c)=0.3, p(s0)=0.5 p(s1)=0.5
    p = cnt / 10
    manual = 0.0
    for i in range(3):
        for j in range(2):
            pij = p[i, j]
            if pij > 0:
                manual += pij * np.log2(pij / (p[i].sum() * p[:, j].sum()))
    assert abs(st["I_phi_s"] - manual) < 1e-12
    assert st["H_phi"] > 0 and st["H_s"] > 0


def test_counts_sum_and_masks():
    rng = np.random.default_rng(0)
    T, D = 60, 4
    z = rng.normal(size=(T, D))
    live = np.ones(T, dtype=bool)
    live[::7] = False                      # off-pitch frames
    z[5] = np.nan                          # non-finite frame
    labels, fin = fit_symbols(z, live, k=3, seed=42)
    assert fin.sum() == (live & np.isfinite(z).all(axis=1)).sum()
    assert (labels[~fin] == -1).all()
    phi = np.array([f"p{t % 4}" for t in range(T)])
    mask = fin & (labels >= 0)
    cnt, phis = count_joint(phi[mask], labels[mask], k=3)
    assert cnt.sum() == mask.sum()         # each (frame, player) counts once
    assert cnt.sum() == int((live.sum()) - 1)   # only the NaN frame excluded


def test_deterministic_seed():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(80, 8))
    live = np.ones(80, dtype=bool)
    l1, _ = fit_symbols(z, live, k=4, seed=7)
    l2, _ = fit_symbols(z, live, k=4, seed=7)
    assert np.array_equal(l1, l2)
    l3, _ = fit_symbols(z, live, k=4, seed=8)
    assert not np.array_equal(l1, l3)      # seeds differ -> fits differ


def test_not_enough_frames_raises():
    z = np.ones((5, 2))
    live = np.ones(5, dtype=bool)
    with pytest.raises(ValueError):
        fit_symbols(z, live, k=6, seed=42)


def test_archetype_frequency_order():
    # cluster labels are renamed by descending frequency: label 0 is largest
    rng = np.random.default_rng(2)
    z = rng.normal(size=(200, 3))
    live = np.ones(200, dtype=bool)
    labels, fin = fit_symbols(z, live, k=3, seed=42)
    counts = np.bincount(labels[fin], minlength=3)
    assert counts[0] >= counts[1] >= counts[2]
    assert set(labels[fin].tolist()) == {0, 1, 2}

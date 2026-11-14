"""Validation of the three MTE estimators on synthetic 3-channel systems
with known ground truth (PROTOCOL.md sec. 13).

Ground truth: the Gaussian closed-form conditional MI evaluated on a
200k-step chain (the plug-in has converged there), via mte.synthetic.gt_mte.

The KEY test is test_common_driver_conditioned: C drives X and Y in the
SAME frame (the honest model of the GVRNN shared-input confound).  The
pairwise TE(X->Y) is then spuriously positive while MTE(X->Y|C) ~ 0 -
conditioning on the third channel must remove the spurious edge.

Run from the project root:  pytest tests/ -x
"""
import numpy as np
import pytest

from mte import (conditional_discrete_mte, conditional_gaussian_mte,
                 conditional_ksg_mte, effective_te, standardize,
                 surrogate_mte)
from mte.discrete import quantize_channels
from mte.synthetic import scenarios, gt_mte
from mte.utils import build_triples

T = 4000            # analysis chain length (per-subsystem order of magnitude)
GT_T = 200_000      # ground-truth chain length


def _prep(chain, with_C=True):
    """X, Y, Z triples from a simulated chain, standardised per column
    (Chebyshev radii mix dimensions, so KSG needs standardised inputs)."""
    X, Y, Z = build_triples(chain["X"], chain["Y"],
                            [chain["C"]] if with_C else [])
    X, _, _ = standardize(X)
    Y, _, _ = standardize(Y)
    Z = np.column_stack([standardize(Z[:, j])[0] for j in range(Z.shape[1])])
    return X, Y, Z


@pytest.fixture(scope="module")
def scen():
    return scenarios(T, GT_T, seed=0)


def test_null_close_to_zero(scen):
    ch = scen["null"]["chain"]
    X, Y, Z = _prep(ch, with_C=True)
    assert abs(conditional_gaussian_mte(X, Y, Z)) < 0.03
    assert abs(conditional_ksg_mte(X, Y, Z, k=5)) < 0.06


def test_direct_matches_gt(scen):
    """X -> Y with an independent C: MTE(X->Y|C) equals the pairwise TE and
    both match the long-chain Gaussian ground truth."""
    ch, gt_ch = scen["direct"]["chain"], scen["direct"]["gt"]
    gt = gt_mte(gt_ch, "X->Y", with_C=True)
    assert gt > 0.10, f"direct-edge GT too weak: {gt:.4f}"

    X, Y, Z = _prep(ch, with_C=True)
    est_g = conditional_gaussian_mte(X, Y, Z)
    est_k = conditional_ksg_mte(X, Y, Z, k=5)
    assert abs(est_g - gt) < 0.05, f"gaussian {est_g:.4f} vs gt {gt:.4f}"
    assert abs(est_k - gt) < 0.10, f"ksg {est_k:.4f} vs gt {gt:.4f}"


def test_common_driver_conditioned(scen):
    """THE conditioning test: C drives X and Y in the same frame.  Pairwise
    TE(X->Y) is spurious; MTE(X->Y|C) must remove it."""
    ch, gt_ch = scen["common_driver"]["chain"], scen["common_driver"]["gt"]
    gt_pair = gt_mte(gt_ch, "X->Y", with_C=False)   # spurious pairwise signal
    assert gt_pair > 0.02, f"spurious pairwise GT too weak: {gt_pair:.4f}"

    Xp, Yp, Zp = _prep(ch, with_C=False)
    est_pair = conditional_gaussian_mte(Xp, Yp, Zp)
    assert est_pair > 0.5 * gt_pair, \
        f"pairwise est {est_pair:.4f} lost the spurious signal ({gt_pair:.4f})"

    X, Y, Z = _prep(ch, with_C=True)
    assert abs(conditional_gaussian_mte(X, Y, Z)) < 0.03
    assert abs(conditional_ksg_mte(X, Y, Z, k=5)) < 0.06


def test_discrete_small_system(scen):
    """Discrete conditional plug-in on a 3-symbol quantisation: direct edge
    survives, the common-driver spurious edge is removed."""
    for name, want in (("direct", "positive"), ("common_driver", "zero")):
        ch = scen[name]["chain"]
        syms, _ = quantize_channels([ch["X"], ch["Y"], ch["C"]], 3, seed=9)
        sX, sY, sC = syms
        X = sX[:-1]
        Y = sY[1:]
        Z = np.column_stack([sY[:-1], sC[:-1]])
        te = conditional_discrete_mte(Y, X, Z)
        if want == "positive":
            assert te > 0.08, f"{name}: discrete MTE {te:.4f} too small"
        else:
            assert te < 0.05, f"{name}: spurious edge not removed, {te:.4f}"


def test_surrogate_null_not_rejected(scen):
    """Under the null, the block-shuffle surrogate test must not reject and
    the surrogate-corrected MTE must be ~0 (frozen PCA/standardisation are
    exercised implicitly: the closure refits nothing)."""
    ch = scen["null"]["chain"]
    A, B, C = ch["X"], ch["Y"], ch["C"]

    def fn(A_, B_, R_):
        X, Y, Z = _prep({"X": A_, "Y": B_, "C": R_[0]}, with_C=True)
        return conditional_ksg_mte(X, Y, Z, k=5)

    raw = fn(A, B, [C])
    surr = surrogate_mte(fn, A, B, [C], block_len=10, n_surrogates=200, seed=9)
    eff, p = effective_te(raw, surr)
    assert p > 0.01, f"null rejected: p={p:.3f}"
    assert abs(eff) < 0.06, f"surrogate-corrected MTE {eff:.4f} not ~0"

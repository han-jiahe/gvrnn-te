"""TE-specific validation: the mte.te wrappers reduce the conditional-MI
estimators to order-1 pairwise TE (Z = target history only), and the
common-driver scenario is a DOCUMENTED limitation of pairwise TE (without
third-party conditioning the spurious edge cannot be removed - the reason the
MTE variant existed; the project now accepts this and reads TE as
model-internal predictive information, not causation)."""
import numpy as np
import pytest

from mte import conditional_gaussian_mte, standardize
from mte.synthetic import scenarios, gt_mte
from mte.te import (transfer_entropy_discrete, transfer_entropy_gaussian,
                    transfer_entropy_ksg)
from mte.utils import build_triples


def _prep2(chain):
    """Standardised A, B channels (pairwise TE does not use the C channel)."""
    A, _, _ = standardize(chain["X"])
    B, _, _ = standardize(chain["Y"])
    return A, B


def test_te_wrappers_match_conditional_gaussian():
    ch = scenarios(T=4000, seed=0)["direct"]["chain"]
    A, B = _prep2(ch)
    te = transfer_entropy_gaussian(A, B)
    X, Y, Z = build_triples(A, B, [])
    ref = conditional_gaussian_mte(X, Y, Z)
    assert abs(te - ref) < 1e-12


def test_te_direct_positive_and_matches_gt():
    sc = scenarios(T=4000, seed=0)
    ch, gt_ch = sc["direct"]["chain"], sc["direct"]["gt"]
    gt = gt_mte(gt_ch, "X->Y", with_C=False)   # pairwise TE ground truth
    assert gt > 0.10
    A, B = _prep2(ch)
    assert abs(transfer_entropy_gaussian(A, B) - gt) < 0.05
    assert transfer_entropy_ksg(A, B, k=5) > 0.0
    assert transfer_entropy_discrete(A, B, n_clusters=3) > 0.02


def test_te_null_close_to_zero():
    ch = scenarios(T=4000, seed=0)["null"]["chain"]
    A, B = _prep2(ch)
    assert abs(transfer_entropy_gaussian(A, B)) < 0.03
    assert abs(transfer_entropy_ksg(A, B, k=5)) < 0.06


def test_te_common_driver_is_spurious():
    """DOCUMENTED limitation: a shared same-frame driver makes pairwise TE
    spuriously positive; only conditioning on the driver (the terminated MTE
    variant) removes it.  Pairwise TE is the chosen scope, so this must be
    read as 'model-internal predictive information', not causation."""
    sc = scenarios(T=4000, seed=0)
    ch, gt_ch = sc["common_driver"]["chain"], sc["common_driver"]["gt"]
    gt_pair = gt_mte(gt_ch, "X->Y", with_C=False)
    assert gt_pair > 0.02, f"spurious pairwise GT too weak: {gt_pair:.4f}"
    A, B = _prep2(ch)
    assert transfer_entropy_gaussian(A, B) > 0.5 * gt_pair

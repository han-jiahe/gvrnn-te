"""
DEPRECATED — the rest-conditioned MTE (multivariate TE) pipeline has been
terminated (2026-11-04) in favour of pairwise TE.

The project now computes pairwise transfer entropy between two latent chains:

    TE(A -> B) = I( z^A_{t-1} ; z^B_t | z^B_{t-1} )

conditioned ONLY on the target's own history (order-1, k = l = 1; no
third-party channels).  Use run_player_te.py:

    python -m run_player_te --latents ... --players 0,1 ...

This module is kept only so old import paths keep working; `python -m
run_player_mte` now runs the TE pipeline.
"""
from run_player_te import frame_numbers, mask_and_stride, run_pairs, main  # noqa: F401

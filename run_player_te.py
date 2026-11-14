"""
Player-level transfer entropy (TE) pipeline (PROTOCOL.md sec. 11-12).

Computes the directed pairwise TE between the per-player GVRNN latent chains:

    TE(A -> B) = I( z^A_{t-1} ; z^B_t | z^B_{t-1} )

for every ordered pair of the selected player set (the ball is a node too),
with three estimators (Gaussian / KSG / discrete plug-in), a parameter grid,
and the block-shuffle surrogate significance protocol.

The conditioning set is ONLY the target's own history (order-1, k = l = 1):
this is the standard Schreiber TE between two chains.  No third-party
channels are conditioned on.

Example:

    python -m run_player_te --latents artifacts/m1/latents.npz \\
        --players 0,1,2 --pca-dims 1 2 --k 3 5 8 --n-clusters 3 5 \\
        --stride 10 --n-surrogates 200 --block-len 25 \\
        --out results/te.json

Two-player TE (a single ordered pair, both directions):

    python -m run_player_te --latents artifacts/m1/latents.npz \\
        --players 0,1 --pca-dims 1 --k 3 5 --n-clusters 3 --stride 10 \\
        --n-surrogates 200 --block-len 25 --out results/te_0_1.json

Window-restricted runs (substitute-window analysis):

    --frame-range START END   restrict to ORIGINAL frame numbers (inclusive)
                              BEFORE striding
    --pairs "8,0;9,0"         compute only the listed ordered pairs
    --mask-gaps               mask transitions whose stride block contains a
                              dead-ball gap (diff(frame_times) > 0.11 s)

All channel transforms (PCA, standardisation, K-means fits) are FROZEN on
the real data; surrogates only re-transform, never refit.
"""
import argparse
import json

import numpy as np

import config
from mte import (conditional_discrete_mte, conditional_gaussian_mte,
                 conditional_ksg_mte, effective_te, gaussian_mte_pvalue,
                 pca_fit, pca_transform, standardize, surrogate_mte)
from mte.discrete import quantize_channels, quantize_with


def _triples(A, B, pm):
    """Order-1 TE triples: pm[t] = keep the transition t -> t+1 (period
    breaks excluded).  X = A[:-1][pm] (source history z^A_{t-1}),
    Y = B[1:][pm] (target future z^B_t), Z = B[:-1][pm] (target history
    z^B_{t-1}) as an (n, 1) column (1-d symbol chains are reshaped, not
    transposed)."""
    X = np.asarray(A)[:-1][pm]
    Y = np.asarray(B)[1:][pm]
    Z = np.asarray(B)[:-1][pm]
    if Z.ndim == 1:
        Z = Z[:, None]
    return X, Y, Z


def frame_numbers(latent: dict) -> np.ndarray:
    """Exact int frame per row: latent['frame'] if present, else
    rint(frame_times * 10) (frame_times is float32 seconds - exact enough
    for frame numbers < 2^24)."""
    if "frame" in latent and latent["frame"] is not None:
        return np.asarray(latent["frame"], dtype=np.int64)
    return np.rint(np.asarray(latent["frame_times"], dtype=float) * 10.0).astype(np.int64)


def mask_and_stride(z, periods, frame_times, frame, frame_range=None,
                    stride=1, mask_gaps=False):
    """Filter rows to frame_range (inclusive, ORIGINAL frame units) BEFORE
    striding, then downsample by stride.  Returns (z, periods, frame_times,
    frame, pm).  pm masks period-break transitions and - when mask_gaps -
    transitions whose stride block contains a dead-ball gap
    (diff(frame_times) > 0.11 s, the same rule as gvrnn/extract.py's h
    reset)."""
    z = np.asarray(z)
    periods = np.asarray(periods)
    frame_times = np.asarray(frame_times, dtype=float)
    frame = np.asarray(frame, dtype=np.int64)
    if frame_range is not None:
        lo, hi = frame_range
        sel = (frame >= lo) & (frame <= hi)
        if not sel.any():
            raise ValueError(f"frame range {lo}-{hi} contains no rows")
        z = z[sel]
        periods = periods[sel]
        frame_times = frame_times[sel]
        frame = frame[sel]
    gap_full = np.diff(frame_times) > 0.11 if mask_gaps else None
    z = z[::stride]
    periods = periods[::stride]
    frame_times = frame_times[::stride]
    frame = frame[::stride]
    pm = periods[:-1] == periods[1:]
    if gap_full is not None:
        n = len(z)
        gap_ds = np.zeros(n - 1, dtype=bool)
        for j in range(n - 1):
            r0, r1 = j * stride, min((j + 1) * stride, len(gap_full))
            gap_ds[j] = gap_full[r0:r1].any()
        pm &= ~gap_ds
    return z, periods, frame_times, frame, pm


def _run_pair(A_ch, B_ch, A, B, cfg, pm, opts):
    """Estimate TE(A->B) for all methods x parameter grid; returns
    (results, skipped, frozen).  All fits are frozen on the real data and
    only the SOURCE channel A is shuffled in the surrogate protocol.
    frozen[method][param_key] = {"fn_windowed": callable(A_w, B_w)}
    recomputes the SAME estimate (frozen transforms) on arbitrary-length
    channel windows (matched-n window studies); the window is assumed
    gap/period-free (pm = all True).

    NOTE: loop variables (d / k / K) and the per-iteration fits are captured
    BY VALUE via default arguments - a plain closure would late-bind and
    every frozen fn_windowed would use the LAST loop value when the bootstrap
    calls it later."""
    results, skipped, frozen = {}, [], {}
    n = int(pm.sum())

    for d in opts["pca_dims"]:
        pcaA = pca_fit(A_ch, d)
        pcaB = pca_fit(B_ch, d)
        tA = standardize(pca_transform(pcaA, A_ch))
        tB = standardize(pca_transform(pcaB, B_ch))
        trA = (pcaA, tA[1], tA[2])
        trB = (pcaB, tB[1], tB[2])

        def prep(A_, B_, _trA=trA, _trB=trB):
            return (standardize(pca_transform(_trA[0], A_), _trA[1], _trA[2])[0],
                    standardize(pca_transform(_trB[0], B_), _trB[1], _trB[2])[0])

        # ---- conditional Gaussian (closed form) --------------------------
        def gauss_fn(A_, B_, *_, _prep=prep):
            A_t, B_t = _prep(A_, B_)
            X, Y, Z = _triples(A_t, B_t, pm)
            return conditional_gaussian_mte(X, Y, Z, ridge=cfg.gaussian_ridge)

        def gauss_windowed(A_w, B_w, _prep=prep):
            A_t, B_t = _prep(A_w, B_w)
            X, Y, Z = _triples(A_t, B_t, np.ones(len(A_t) - 1, dtype=bool))
            return conditional_gaussian_mte(X, Y, Z, ridge=cfg.gaussian_ridge)

        raw = gauss_fn(A_ch, B_ch)
        surr = surrogate_mte(gauss_fn, A_ch, B_ch, [],
                             block_len=opts["block_len"],
                             n_surrogates=opts["n_surrogates"], seed=opts["seed"])
        eff, p = effective_te(raw, surr)
        key = f"pca={d}"
        results.setdefault("gaussian", {})[key] = {
            "raw": raw, "eff": eff, "p": p,
            "chi2_p": gaussian_mte_pvalue(raw, n, d, d),
            "surrogates": surr.tolist(),
        }
        frozen.setdefault("gaussian", {})[key] = {"fn_windowed": gauss_windowed}

        # ---- conditional KSG --------------------------------------------
        for k in opts["k"]:
            def ksg_fn(A_, B_, *_, _prep=prep, _k=k):
                A_t, B_t = _prep(A_, B_)
                X, Y, Z = _triples(A_t, B_t, pm)
                return conditional_ksg_mte(X, Y, Z, k=_k)

            def ksg_windowed(A_w, B_w, _prep=prep, _k=k):
                A_t, B_t = _prep(A_w, B_w)
                X, Y, Z = _triples(A_t, B_t, np.ones(len(A_t) - 1, dtype=bool))
                return conditional_ksg_mte(X, Y, Z, k=_k)

            raw = ksg_fn(A_ch, B_ch)
            surr = surrogate_mte(ksg_fn, A_ch, B_ch, [],
                                 block_len=opts["block_len"],
                                 n_surrogates=opts["n_surrogates"], seed=opts["seed"])
            eff, p = effective_te(raw, surr)
            results.setdefault("ksg", {})[f"pca={d},k={k}"] = {
                "raw": raw, "eff": eff, "p": p, "surrogates": surr.tolist(),
            }
            frozen.setdefault("ksg", {})[f"pca={d},k={k}"] = {"fn_windowed": ksg_windowed}

        # ---- discrete conditional (small systems only) -------------------
        # quantise the PCA-reduced standardised chains, NOT the raw 32-dim:
        # the latent is roughly unimodal and raw-space K-means collapses
        # almost everything into one cluster -> degenerate zero TE
        for K in opts["n_clusters"]:
            syms, fits = quantize_channels([tA[0], tB[0]], K, seed=opts["seed"])
            sA, sB = syms[0], syms[1]
            X, Y, Z = _triples(sA, sB, pm)
            n_cells = len(np.unique(np.column_stack([Y, X, Z]), axis=0))
            if n_cells > n / cfg.min_samples_per_cell:
                skipped.append({"pair": f"{A}->{B}", "method": "discrete",
                                "param": f"pca={d},K={K}",
                                "reason": f"cells {n_cells} > n/"
                                          f"min_samples_per_cell "
                                          f"({n // cfg.min_samples_per_cell})"})
                continue

            def disc_fn(A_, B_, *_, _prep=prep, _fits=fits):
                A_t, B_t = _prep(A_, B_)
                sA_ = quantize_with(_fits[0], A_t)
                sB_ = quantize_with(_fits[1], B_t)
                X_, Y_, Z_ = _triples(sA_, sB_, pm)
                return conditional_discrete_mte(Y_, X_, Z_)

            def disc_windowed(A_w, B_w, _prep=prep, _fits=fits):
                A_t, B_t = _prep(A_w, B_w)
                sA_ = quantize_with(_fits[0], A_t)
                sB_ = quantize_with(_fits[1], B_t)
                X_, Y_, Z_ = _triples(sA_, sB_, np.ones(len(sA_) - 1, dtype=bool))
                return conditional_discrete_mte(Y_, X_, Z_)

            raw = disc_fn(A_ch, B_ch)
            surr = surrogate_mte(disc_fn, A_ch, B_ch, [],
                                 block_len=opts["block_len"],
                                 n_surrogates=opts["n_surrogates"], seed=opts["seed"])
            eff, p = effective_te(raw, surr)
            results.setdefault("discrete", {})[f"pca={d},K={K}"] = {
                "raw": raw, "eff": eff, "p": p, "surrogates": surr.tolist(),
            }
            frozen.setdefault("discrete", {})[f"pca={d},K={K}"] = {"fn_windowed": disc_windowed}

    return results, skipped, frozen


def run_pairs(z, players, pairs, cfg, opts, pm):
    """All ordered pairs among `players` (or only `pairs`, a list of (A, B)
    node-index tuples).  TE(A->B) conditions ONLY on B's own history.
    Returns (results, skipped, frozen, n_transitions)."""
    channels = [z[:, i] for i in players]
    results, skipped, frozen = {}, [], {}
    n = int(pm.sum())
    for ai, A in enumerate(players):
        for bi, B in enumerate(players):
            if A == B:
                continue
            if pairs is not None and (A, B) not in pairs:
                continue
            pair_results, pair_skipped, pair_frozen = _run_pair(
                channels[ai], channels[bi], A, B, cfg, pm, opts)
            results[f"{A}->{B}"] = pair_results
            skipped.extend(pair_skipped)
            frozen[f"{A}->{B}"] = pair_frozen
            print(f"[run_player_te] pair {A}->{B} done", flush=True)
    return results, skipped, frozen, n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", required=True,
                    help="per-player latent npz from gvrnn.extract "
                         "(z_post (T, K, L), periods (T,))")
    ap.add_argument("--players", required=True,
                    help="comma-separated node indices (0-based; the ball is "
                         "node K-1).  TE is computed for every ordered pair "
                         "among them.")
    ap.add_argument("--pairs", default=None,
                    help="semicolon-separated ordered pairs 'A,B;A,C' (node "
                         "indices); omit for all ordered pairs")
    ap.add_argument("--frame-range", nargs=2, type=int, default=None,
                    metavar=("START", "END"),
                    help="restrict analysis to ORIGINAL frame numbers "
                         "(inclusive), applied before striding")
    ap.add_argument("--mask-gaps", action="store_true",
                    help="mask transitions whose stride block contains a "
                         "dead-ball gap")
    ap.add_argument("--pca-dims", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 8])
    ap.add_argument("--n-clusters", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--stride", type=int, default=10,
                    help="downsample the 10 Hz chain (10 -> 1 Hz)")
    ap.add_argument("--n-surrogates", type=int, default=200)
    ap.add_argument("--block-len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    latent = dict(np.load(config.abs_path(args.latents), allow_pickle=False))
    z = np.asarray(latent["z_post"], dtype=np.float32)      # (T, K, L)
    periods = np.asarray(latent["periods"], dtype=int)
    T, K, L = z.shape
    players = [int(x) for x in args.players.split(",")]
    assert all(0 <= i < K for i in players), f"players must be in 0..{K - 1}"
    frame = frame_numbers(latent)
    ft = np.asarray(latent["frame_times"], dtype=float)

    z, periods, ft, frame, pm = mask_and_stride(
        z, periods, ft, frame, frame_range=args.frame_range,
        stride=args.stride, mask_gaps=args.mask_gaps)
    n = int(pm.sum())
    if n < 100:
        print(f"[run_player_te WARN] only {n} valid transitions; TE "
              f"estimates will be very noisy", flush=True)

    pairs = None
    if args.pairs:
        pairs = [tuple(int(x) for x in p.split(",")) for p in args.pairs.split(";")]

    cfg = config.DEFAULT_TE_CONFIG
    opts = dict(pca_dims=args.pca_dims, k=args.k, n_clusters=args.n_clusters,
                block_len=args.block_len, n_surrogates=args.n_surrogates,
                seed=args.seed)

    results, skipped, _, n = run_pairs(z, players, pairs, cfg, opts, pm)

    # directed matrix summaries (raw / eff / p) per method x param
    Np = len(players)
    matrices = {}
    for pair_key, methods in results.items():
        A, B = (int(x) for x in pair_key.split("->"))
        i, j = players.index(A), players.index(B)
        for method, params in methods.items():
            matrices.setdefault(method, {})
            for pkey, vals in params.items():
                matrices[method].setdefault(pkey, {})
                for field in ("raw", "eff", "p"):
                    mat = matrices[method][pkey].setdefault(
                        field, [[None] * Np for _ in range(Np)])
                    mat[i][j] = vals[field]

    out = {
        "config": {
            "pca_dims": args.pca_dims, "k": args.k,
            "n_clusters": args.n_clusters, "stride": args.stride,
            "n_surrogates": args.n_surrogates, "block_len": args.block_len,
            "seed": args.seed,
            "frame_range": args.frame_range, "mask_gaps": args.mask_gaps,
            "pairs": pairs, "gaussian_ridge": cfg.gaussian_ridge,
            "base": cfg.base,
        },
        "input": {"latents": args.latents, "players": players,
                  "T_frames": T, "latent_dim": L,
                  "n_transitions": n, "stride": args.stride},
        "results": results,
        "skipped": skipped,
        "matrices": matrices,
    }
    out_path = config.abs_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[run_player_te] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Player-to-player GVRNN latent TE at native 10 Hz -- same protocol as
phase_te.py, for head-to-head magnitude comparison with the phase-chain TE.

    TE(A->B) = I(z^B_{t+1}; z^A_t | z^B_t)      (Schreiber k = l = 1)

Same pipeline as phase_te: 32-dim z symbolized per player by frozen K-means
(z-score per dim, n_init=10, seed; clusters renamed by descending frequency);
transition mask requires BOTH players live at both ends, contiguous frames,
same period; discrete plug-in estimator; block permutation (block 25) of the
source symbols with frozen symbolization; p = (#{surr >= raw}+1)/(S+1).

Outputs (--out-dir): latent_pair_te.csv (one row per ordered pair x K),
latent_pair_te_report.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from latent_phase import fit_symbols
from phase_te import block_shuffle, p_value, te_discrete


def pair_mask(frame: np.ndarray, periods, live_a: np.ndarray,
              live_b: np.ndarray) -> np.ndarray:
    ok = live_a[:-1] & live_a[1:] & live_b[:-1] & live_b[1:]
    ok &= (frame[1:] == frame[:-1] + 1)
    if periods is not None:
        ok &= (periods[:-1] == periods[1:])
    return ok


def te_pair(sym_a: np.ndarray, sym_b: np.ndarray, mask: np.ndarray, k: int,
            rng, n_surr: int, block: int):
    """TE(A->B) with surrogates.  sym_*: symbol arrays (full length)."""
    ynext = sym_b[1:][mask]
    y = sym_b[:-1][mask]
    x = sym_a[:-1][mask]
    base = (ynext * k + y) * k
    cnt = np.bincount(base + x, minlength=k ** 3).reshape(k, k, k)
    st = te_discrete(cnt)
    surrs = [te_discrete(np.bincount(base + block_shuffle(x, block, rng),
                                     minlength=k ** 3).reshape(k, k, k))["te"]
             for _ in range(n_surr)]
    st["eff"] = st["te"] - float(np.mean(surrs))
    st["surr_mean"] = float(np.mean(surrs))
    st["p"] = p_value(surrs, st["te"])
    return st


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", required=True, type=Path)
    ap.add_argument("--live", required=True, type=Path)
    ap.add_argument("--players", default="0,1,2,3,4,5,6")
    ap.add_argument("--k", default="3,8")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--n-surrogates", default=200, type=int)
    ap.add_argument("--block-len", default=25, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    lat = np.load(args.latents)
    trk = np.load(args.live)
    frames = lat["frame"]
    z_post = lat["z_post"]
    periods = lat["periods"]
    live = trk["live_mask"]
    pids = [int(p) for p in trk["player_ids"]]
    ks = [int(x) for x in args.k.split(",")]
    nodes = [int(x) for x in args.players.split(",")]

    syms, lives = {}, {}
    for node in nodes:
        for k in ks:
            lab, _ = fit_symbols(z_post[:, node, :], live[:, node].astype(bool),
                                 k, args.seed)
            syms[(node, k)] = lab
        lives[node] = live[:, node].astype(bool)

    rows = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            mask = pair_mask(frames, periods, lives[a], lives[b])
            for k in ks:
                rng = np.random.default_rng(args.seed + a * 100 + b * 10 + k)
                st = te_pair(syms[(a, k)], syms[(b, k)], mask, k, rng,
                             args.n_surrogates, args.block_len)
                rows.append({"src": pids[a], "dst": pids[b], "src_node": a,
                             "dst_node": b, "k": k, **st})
                print(f"{pids[a]}->{pids[b]} K={k}: n={st['n']} "
                      f"H={st['h']:.4f} TE={st['te']:.5f} eff={st['eff']:.5f} "
                      f"p={st['p']:.3f}")

    dfm = pd.DataFrame(rows)
    order = np.argsort(dfm["p"].to_numpy())
    m = len(dfm)
    ps = dfm["p"].to_numpy()[order]
    kstar = 0
    for i in range(m - 1, -1, -1):
        if ps[i] <= 0.05 * (i + 1) / m:
            kstar = i + 1
            break
    thr = 0.05 * kstar / m if kstar else 0.0
    dfm["bh_sig"] = dfm["p"] <= thr
    dfm.to_csv(out / "latent_pair_te.csv", index=False, encoding="utf-8")

    g = dfm.groupby("k")
    with open(out / "latent_pair_te_report.md", "w", encoding="utf-8") as f:
        f.write("# Player-to-player latent TE (10 Hz, phase_te protocol)\n\n")
        f.write(f"- latents: `{args.latents}`, pairs: {m // len(ks)} ordered, "
                f"K = {ks}, surrogates = {args.n_surrogates}, block = "
                f"{args.block_len}, seed = {args.seed}\n\n")
        f.write("| K | median TE | median eff | median H(y'\\|y) | median "
                "eff/H | p<0.05 | bh_sig |\n|---|---|---|---|---|---|---|\n")
        for k, grp in g:
            f.write(f"| {k} | {grp['te'].median():.5f} | {grp['eff'].median():.5f} "
                    f"| {grp['h'].median():.4f} | "
                    f"{(grp['eff'] / grp['h']).median():.4f} | "
                    f"{int((grp['p'] < 0.05).sum())}/{len(grp)} | "
                    f"{int(grp['bh_sig'].sum())} |\n")
        f.write("\n`eff` = raw - surrogate mean; `eff/H` = fraction of the "
                "target chain's one-step unpredictability explained.\n")
    print(f"-> {out / 'latent_pair_te_report.md'}")


if __name__ == "__main__":
    main()

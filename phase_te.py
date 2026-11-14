"""TE between a player's GVRNN latent chain and the match phase chain.

Both directions, first-order Schreiber TE (k = l = 1), discrete plug-in:

    TE_{X->Y} = I(y_{t+1}; x_t | y_t)
              = sum_{y',y,x} p(y',y,x) * log2[ p(y'|y,x) / p(y'|y) ]

  z->phi : x_t = z^A_t symbolized (frozen per-player K-means, K symbols),
           y_t  = phase phi_t of frame t,  y' = phi_{t+1}
  phi->z : x_t = phi_t,  y_t = z^A_t symbolized,  y' = z^A_{t+1}

Counting rule -- one occurrence = one transition t -> t+1 satisfying ALL of:
  1. live_mask[t] and live_mask[t+1] (player on pitch at both ends);
  2. frame[t+1] == frame[t] + 1 (no dead-ball / halftime gaps);
  3. periods[t] == periods[t+1] (no cross-half transitions).
Scope `full` keeps the explicit ``dead`` phase class; scope `live_only`
additionally requires both frames covered by a phases CSV row.

Significance: block permutation of the source chain (block length 25),
frozen symbolization, S surrogates; p = (#{surr >= raw} + 1)/(S + 1),
TE_eff = raw - mean(surr); BH-FDR (q = 0.05) across the reported family.

Outputs (--out-dir): phase_te_summary.csv, phase_te_report.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from latent_phase import assign_phases, fit_symbols, load_phases


# ------------------------------------------------------------ transitions
def transition_mask(frame: np.ndarray, periods, live: np.ndarray,
                    covered=None) -> np.ndarray:
    """Bool mask over transitions t -> t+1 (length T-1)."""
    ok = live[:-1] & live[1:]
    ok &= (frame[1:] == frame[:-1] + 1)
    if periods is not None:
        ok &= (periods[:-1] == periods[1:])
    if covered is not None:
        ok &= covered[:-1] & covered[1:]
    return ok


def block_shuffle(x: np.ndarray, block: int, rng) -> np.ndarray:
    """Permute blocks of consecutive values (block length `block`)."""
    n = len(x)
    idx = np.arange(n)
    blocks = [idx[i:i + block] for i in range(0, n, block)]
    rng.shuffle(blocks)
    return x[np.concatenate(blocks)]


def p_value(surrs, raw: float) -> float:
    return (float(np.sum(np.asarray(surrs) >= raw)) + 1.0) / (len(surrs) + 1.0)


# ------------------------------------------------------------- estimator
def te_discrete(cnt: np.ndarray):
    """Plug-in TE and H(y'|y) (bits) from counts cnt[y', y, x]."""
    n = int(cnt.sum())
    p = cnt / n
    p_y = p.sum(axis=(0, 2))          # p(y)
    p_yx = p.sum(axis=0)              # p(y, x)
    p_ynext_y = p.sum(axis=2)         # p(y', y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p * p_y[None, :, None] / (p_ynext_y[:, :, None] * p_yx[None, :, :])
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    ratio = np.where(ratio > 0, ratio, 1.0)
    te = float((p * np.log2(ratio)).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        cond = p_ynext_y / p_y[None, :]
    mask = (p_ynext_y > 0) & np.isfinite(cond) & (cond > 0)
    h = float(-(p_ynext_y[mask] * np.log2(cond[mask])).sum())
    persist = float(cnt[np.arange(cnt.shape[0]), np.arange(cnt.shape[1])].sum() / n)
    return {"te": te, "h": h, "n": n, "persist": persist,
            "te_norm": te / h if h > 0 else np.nan}


def counts_z_to_phi(pinv, labels, mask, k, n_phi):
    ynext = pinv[1:][mask]
    y = pinv[:-1][mask]
    x = labels[:-1][mask]
    base = ynext * n_phi + y
    return np.bincount(base * k + x, minlength=n_phi * n_phi * k).reshape(
        n_phi, n_phi, k)


def counts_phi_to_z(pinv, labels, mask, k, n_phi):
    ynext = labels[1:][mask]
    y = labels[:-1][mask]
    x = pinv[:-1][mask]
    base = (ynext * k + y) * n_phi
    return np.bincount(base + x, minlength=k * k * n_phi).reshape(k, k, n_phi)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", required=True, type=Path)
    ap.add_argument("--live", required=True, type=Path,
                    help="tracking npz with live_mask/frame (aligned with latents)")
    ap.add_argument("--phases", required=True, type=Path)
    ap.add_argument("--team-id", required=True, type=int)
    ap.add_argument("--players", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--k", default="3,8", help="K-means symbol counts")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--n-surrogates", default=200, type=int)
    ap.add_argument("--block-len", default=25, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    lat = np.load(args.latents)
    trk = np.load(args.live)
    if not np.array_equal(lat["frame"], trk["frame"]):
        raise ValueError("latents.frames != tracking.frames: npz not aligned")
    frames = lat["frame"]
    z_post = lat["z_post"]
    periods = lat["periods"]
    live = trk["live_mask"]
    pids = [int(p) for p in trk["player_ids"]]
    ks = [int(x) for x in args.k.split(",")]
    nodes = [int(x) for x in args.players.split(",")]

    df = load_phases(args.phases)
    phi, _, covered = assign_phases(df, frames, args.team_id)
    phis, pinv = np.unique(phi, return_inverse=True)   # global vocabulary
    n_phi = len(phis)
    print(f"phases: {len(df)} rows; covered {int(covered.sum())}/{len(frames)} "
          f"({100 * covered.mean():.1f}%) -> dead {int((~covered).sum())}; "
          f"vocabulary {n_phi}")

    rows = []
    for node in nodes:
        pid = pids[node]
        live_k = live[:, node].astype(bool)
        for k in ks:
            labels, fin = fit_symbols(z_post[:, node, :], live_k, k, args.seed)
            for scope, extra in (("full", None), ("live_only", covered)):
                mask = transition_mask(frames, periods, live_k, extra)
                cnts = {"z->phi": counts_z_to_phi(pinv, labels, mask, k, n_phi),
                        "phi->z": counts_phi_to_z(pinv, labels, mask, k, n_phi)}
                for direction, cnt in cnts.items():
                    st = te_discrete(cnt)
                    if direction == "z->phi":
                        src = labels[:-1][mask]
                        base = pinv[1:][mask] * n_phi + pinv[:-1][mask]
                        rebase = lambda xs: np.bincount(
                            base * k + xs, minlength=n_phi * n_phi * k
                        ).reshape(n_phi, n_phi, k)
                    else:
                        src = pinv[:-1][mask]
                        base = (labels[1:][mask] * k + labels[:-1][mask]) * n_phi
                        rebase = lambda xs: np.bincount(
                            base + xs, minlength=k * k * n_phi
                        ).reshape(k, k, n_phi)
                    rng = np.random.default_rng(
                        args.seed + node * 1000 + k * 100
                        + (0 if scope == "full" else 1) * 10
                        + (0 if direction == "z->phi" else 1))
                    surrs = [te_discrete(rebase(block_shuffle(src, args.block_len, rng)))["te"]
                             for _ in range(args.n_surrogates)]
                    row = {"pid": pid, "node": node, "k": k, "scope": scope,
                           "direction": direction, **st,
                           "eff": st["te"] - float(np.mean(surrs)),
                           "surr_mean": float(np.mean(surrs)),
                           "p": p_value(surrs, st["te"])}
                    rows.append(row)
                    print(f"pid {pid} K={k} {scope:9s} {direction}: "
                          f"n={st['n']} H(y'|y)={st['h']:.4f} TE={st['te']:.5f} "
                          f"eff={row['eff']:.5f} p={row['p']:.3f}")

    dfm = pd.DataFrame(rows)
    # BH-FDR across the reported family
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
    dfm.to_csv(out / "phase_te_summary.csv", index=False, encoding="utf-8")

    def _md_table(d):
        cols = list(d.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |",
                 "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in d.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    show = dfm[["pid", "node", "k", "scope", "direction", "n", "h", "te",
                "eff", "p", "bh_sig"]].round(
        {"h": 4, "te": 5, "eff": 5, "p": 4})
    with open(out / "phase_te_report.md", "w", encoding="utf-8") as f:
        f.write("# Player latent chain x phase chain -- transfer entropy report\n\n")
        f.write(f"- latents: `{args.latents}`\n- phases: `{args.phases}` "
                f"({len(df)} rows, {100 * covered.mean():.1f}% frames covered)\n"
                f"- team: `{args.team_id}`, players: {nodes}, K = {ks}, "
                f"surrogates = {args.n_surrogates}, block = {args.block_len}, "
                f"seed = {args.seed}\n\n")
        f.write("## definition\n\n")
        f.write("$$\nTE^{(1,1)}_{X\\to Y} = I(y_{t+1}; x_t \\mid y_t)"
                "= \\sum_{y',y,x} p(y',y,x) \\log_2 "
                "\\frac{p(y'\\mid y,x)}{p(y'\\mid y)}\n$$\n\n")
        f.write("`z->phi`: $x_t = z^A_t$ (frozen K-means symbol), "
                "$y_t = \\varphi_t$, $y' = \\varphi_{t+1}$.  "
                "`phi->z`: $x_t = \\varphi_t$, $y_t = z^A_t$, $y' = z^A_{t+1}$ "
                "(same symbolization).\n\n")
        f.write("## counting rule\n\none occurrence = one transition t -> t+1 with "
                "live_mask at both ends, frame[t+1] == frame[t]+1 (no gaps), same "
                "period, and (scope `live_only`) both frames covered by a phases row "
                "(`full` keeps the `dead` class).\n\n")
        f.write("## significance\n\n"
                "$$\np = \\frac{\\#\\{\\mathrm{surr} \\ge \\mathrm{raw}\\}+1}{S+1},"
                "\\qquad \\mathrm{TE}_{\\mathrm{eff}} = \\mathrm{raw} - "
                "\\overline{\\mathrm{surr}}\n$$\n\n"
                f"source chain block-permuted (block {args.block_len}), frozen "
                f"symbolization, S = {args.n_surrogates}; BH-FDR q = 0.05 across all "
                f"{m} reported rows ({kstar} below threshold).\n\n")
        f.write("## results\n\n")
        f.write(_md_table(show) + "\n\n")
        f.write("`h` = H(y'|y) (bits), `te` = raw TE (bits), `eff` = "
                "surrogate-corrected, `p` = block-permutation p-value, "
                "`bh_sig` = BH-FDR significant.\n")
    print(f"-> {out / 'phase_te_report.md'}")


if __name__ == "__main__":
    main()

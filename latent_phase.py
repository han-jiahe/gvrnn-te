"""GVRNN latent state x phase of play: discrete-count joint probability.

Locked counting rule -- one occurrence = one (frame t, player k) pair that
satisfies ALL of:
  1. live_mask[t, k] is True  (off-pitch frames carry frozen latents);
  2. z_post[t, k] is finite;
  3. frame t has a phase label: half-open interval [frame_start, frame_end)
     of the phases CSV; frames covered by NO row (dead ball / stoppage /
     half-time gaps) fall into the explicit ``dead`` class.
Each pair contributes exactly 1: n[phi_t, s_t] += 1 and N += 1.

Phase label (modelled-team view): team_in_possession == our team -> the
in-possession type, otherwise -> the out-of-possession type.

Symbolization (frozen fit): per player, per-dim z-score on the live &
finite frames, one K-means fit (n_init=10, random_state=seed), labels for
all frames.  Clusters are renamed by descending frequency (0 = largest).

Outputs (--out-dir): phase_label.npz, mi_summary.csv, joint_<pid>_K<k>.csv,
profile_<pid>_K<k>.csv, fig_*_K<k>.png, latent_phase_report.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

DEAD = "dead"


# ---------------------------------------------------------------- phases
def load_phases(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("frame_start").reset_index(drop=True)
    for col in ("team_in_possession_phase_type", "team_out_of_possession_phase_type"):
        df[col] = df[col].fillna("unknown").astype(str)
    return df


def assign_phases(df: pd.DataFrame, frames: np.ndarray, team_id: int):
    """Half-open [frame_start, frame_end) assignment; gaps -> DEAD.

    Returns (phi, row_idx, covered): phi is an object array of phase labels
    for every frame; row_idx the candidate table row; covered the bool
    mask of frames inside some row.
    """
    starts = df["frame_start"].to_numpy(np.int64)
    ends = df["frame_end"].to_numpy(np.int64)
    pos_team = df["team_in_possession_id"].to_numpy()
    in_type = df["team_in_possession_phase_type"].to_numpy()
    out_type = df["team_out_of_possession_phase_type"].to_numpy()
    row_labels = np.where(pos_team == int(team_id), in_type, out_type)

    frames = np.asarray(frames, dtype=np.int64)
    idx = np.searchsorted(starts, frames, side="right") - 1
    covered = (idx >= 0) & (frames < ends[np.clip(idx, 0, None)])
    phi = np.full(frames.shape, DEAD, dtype=object)
    phi[covered] = row_labels[idx[covered]]
    return phi, idx, covered


# ------------------------------------------------------------- symbols
def fit_symbols(z: np.ndarray, live: np.ndarray, k: int, seed: int):
    """Frozen per-player K-means symbolization.

    z: (T, D); live: (T,) bool.  Returns (labels, fin): labels in
    {0..k-1} on finite+live frames, -1 elsewhere; fin is the fit mask.
    Clusters renamed by descending frequency (0 = most frequent).
    """
    fin = live & np.isfinite(z).all(axis=1)
    n_fin = int(fin.sum())
    if n_fin < k:
        raise ValueError(f"finite frames {n_fin} < k={k}")
    mu = z[fin].mean(axis=0)
    sd = z[fin].std(axis=0)
    zs = (z - mu) / (sd + 1e-9)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(zs[fin])
    labels = np.full(len(z), -1, dtype=np.int64)
    labels[fin] = km.labels_
    order = np.argsort(-np.bincount(km.labels_, minlength=k))
    remap = {old: new for new, old in enumerate(order)}
    labels[fin] = np.array([remap[l] for l in labels[fin]])
    return labels, fin


# ------------------------------------------------------------- counting
def count_joint(phi: np.ndarray, labels: np.ndarray, k: int):
    """Contingency counts over (phi, symbol) pairs."""
    phis, pc = np.unique(phi, return_inverse=True)
    codes = pc.astype(np.int64) * k + labels.astype(np.int64)
    cnt = np.bincount(codes, minlength=len(phis) * k).reshape(len(phis), k)
    return cnt, phis


def mi_of(cnt: np.ndarray):
    """Plug-in MI and entropies (bits) from a contingency count table."""
    N = cnt.sum()
    p = cnt / N
    p_phi = p.sum(axis=1)
    p_s = p.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        logr = np.log2(p / (p_phi[:, None] * p_s[None, :]))
    logr = np.where(np.isfinite(logr), logr, 0.0)   # 0-count cells -> 0
    mi = float((p * logr).sum())
    h_phi = float(-np.sum(p_phi[p_phi > 0] * np.log2(p_phi[p_phi > 0])))
    h_s = float(-np.sum(p_s[p_s > 0] * np.log2(p_s[p_s > 0])))
    return {"N": int(N), "H_phi": h_phi, "H_s": h_s,
            "I_phi_s": mi, "I_norm": mi / min(h_phi, h_s) if min(h_phi, h_s) > 0 else np.nan}


# -------------------------------------------------------------- figures
def heatmap(mat: np.ndarray, row_names, col_names, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.42 * len(row_names) + 1.5)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=float(np.nanmax(mat)), aspect="auto")
    ax.set_xticks(range(len(col_names)))
    ax.set_xticklabels(col_names, rotation=45, ha="right")
    ax.set_yticks(range(len(row_names)))
    ax.set_yticklabels(row_names)
    fs = 8 if mat.shape[1] <= 5 else 6
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=fs, color="0.15")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", required=True, type=Path)
    ap.add_argument("--live", required=True, type=Path,
                    help="tracking npz with live_mask/frame (aligned with latents)")
    ap.add_argument("--phases", required=True, type=Path)
    ap.add_argument("--team-id", required=True, type=int)
    ap.add_argument("--players", default="0,1,2,3,4,5,6,7",
                    help="node indices to analyze (comma separated)")
    ap.add_argument("--k", default="3,8", help="K-means cluster counts")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    lat = np.load(args.latents)
    trk = np.load(args.live)
    if not np.array_equal(lat["frame"], trk["frame"]):
        raise ValueError("latents.frames != tracking.frames: npz not aligned")
    frames = lat["frame"]
    z_post = lat["z_post"]                       # (T, K_nodes, D)
    live = trk["live_mask"]                      # (T, N_players)
    pids = [int(p) for p in trk["player_ids"]]
    ks = [int(x) for x in args.k.split(",")]
    nodes = [int(x) for x in args.players.split(",")]
    if any(n >= live.shape[1] for n in nodes):
        raise ValueError(f"--players index must be < {live.shape[1]} (players only; "
                         f"ball node {z_post.shape[1] - 1} has no live mask)")

    df = load_phases(args.phases)
    phi, row_idx, covered = assign_phases(df, frames, args.team_id)
    np.savez(out / "phase_label.npz", frame=frames, phi=phi,
             row_idx=row_idx, covered=covered)

    # phase marginals (union view, per player N differs)
    print(f"phases: {len(df)} rows; covered {int(covered.sum())}/{len(frames)} frames "
          f"({100 * covered.mean():.1f}%) -> dead class {int((~covered).sum())} frames")

    rows_mi = []
    for node in nodes:
        pid = pids[node]
        zk = z_post[:, node, :]
        live_k = live[:, node].astype(bool)
        for k in ks:
            labels, fin = fit_symbols(zk, live_k, k, args.seed)
            base = fin & (labels >= 0)
            # binary live/dead contrast (always reported)
            phi_bin = np.where(covered[base], "live", DEAD)
            cnt_b, phis_b = count_joint(phi_bin, labels[base], k)
            st_b = mi_of(cnt_b)
            st_b.update({"pid": pid, "node": node, "k": k, "scope": "live_dead",
                         "n_finite": int(base.sum())})
            rows_mi.append(st_b)
            print(f"pid {pid} K={k} live/dead: I(s;live)={st_b['I_phi_s']:.4f} bits")
            scopes = [("full", base, ""), ("live_only", base & covered, "_live")]
            for scope, mask, sfx in scopes:
                phi_m = phi[mask]
                cnt, phis = count_joint(phi_m, labels[mask], k)
                st = mi_of(cnt)
                st.update({"pid": pid, "node": node, "k": k, "scope": scope,
                           "n_finite": int(mask.sum())})
                rows_mi.append(st)
                s_names = [f"s{i}" for i in range(k)]
                df_cnt = pd.DataFrame(cnt, index=list(phis), columns=s_names)
                df_cnt.to_csv(out / f"joint_{pid}_K{k}{sfx}.csv", encoding="utf-8")
                with open(out / f"joint_{pid}_K{k}{sfx}.csv", "a", encoding="utf-8") as f:
                    f.write("_marginal P(phi)," + ",".join(
                        f"{v:.5f}" for v in cnt.sum(axis=1) / cnt.sum()) + "\n")
                p_s_given_phi = cnt / cnt.sum(axis=1, keepdims=True)      # P(s|phi)
                p_phi_given_s = (cnt / cnt.sum(axis=0, keepdims=True)).T   # P(phi|s)
                p_phi_given_s[~np.isfinite(p_phi_given_s)] = 0.0
                pd.DataFrame(p_s_given_phi, index=list(phis), columns=s_names).round(4).to_csv(
                    out / f"profile_{pid}_K{k}{sfx}_s_given_phi.csv", encoding="utf-8")
                pd.DataFrame(p_phi_given_s, index=s_names, columns=list(phis)).round(4).to_csv(
                    out / f"profile_{pid}_K{k}{sfx}_phi_given_s.csv", encoding="utf-8")
                heatmap(p_s_given_phi, list(phis), s_names,
                        f"P(s|phi) pid {pid} K={k} {scope}",
                        out / f"fig_ps_given_phi_{pid}_K{k}{sfx}.png")
                heatmap(p_phi_given_s, s_names, list(phis),
                        f"P(phi|s) pid {pid} K={k} {scope}",
                        out / f"fig_phi_given_s_{pid}_K{k}{sfx}.png")
                print(f"pid {pid} K={k} {scope}: N={st['N']} H(phi)={st['H_phi']:.3f} "
                      f"H(s)={st['H_s']:.3f} I(phi;s)={st['I_phi_s']:.4f} bits "
                      f"(norm {st['I_norm']:.3f})")

    mi_df = pd.DataFrame(rows_mi)
    mi_df.to_csv(out / "mi_summary.csv", index=False, encoding="utf-8")

    # archetype semantics for the report
    sem_lines = []
    for scope, sfx in (("full", ""), ("live_only", "_live")):
        sem_lines.append(f"\n## archetype semantics -- scope={scope}\n")
        for node in nodes:
            pid = pids[node]
            for k in ks:
                blk = pd.read_csv(out / f"profile_{pid}_K{k}{sfx}_phi_given_s.csv",
                                  index_col=0)
                sem_lines.append(f"\n### pid {pid} (node {node}), K={k}\n")
                sem_lines.append("| archetype | dominant phase | P(dom) | top-3 (phi: P) | entropy P(phi|s) |\n"
                                 "|---|---|---|---|---|")
                for s, row in blk.iterrows():
                    order = row.sort_values(ascending=False)
                    top3 = ", ".join(f"{c}: {v:.2f}" for c, v in order.head(3).items())
                    ent = -float(np.sum(row[row > 0] * np.log2(row[row > 0])))
                    sem_lines.append(f"| {s} | {order.index[0]} | {order.iloc[0]:.2f} | {top3} | {ent:.2f} |")

    ph_counts = pd.Series(phi).value_counts()

    def _md_table(dframe, index=False):
        cols = list(dframe.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |",
                 "|" + "|".join("---" for _ in cols) + "|"]
        for _, r in dframe.iterrows():
            if index:
                lines.append("| " + " | ".join(str(v) for v in r) + " |")
            else:
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)
    with open(out / "latent_phase_report.md", "w", encoding="utf-8") as f:
        f.write("# GVRNN latent state x phase of play -- joint probability report\n\n")
        f.write(f"- latents: `{args.latents}`\n- live mask: `{args.live}`\n"
                f"- phases: `{args.phases}` ({len(df)} rows, {100 * covered.mean():.1f}% frames covered)\n"
                f"- modelled team: `{args.team_id}`, players (nodes): {nodes}, K = {ks}, seed = {args.seed}\n\n")
        f.write("## counting rule\n\none occurrence = one (frame, player) with "
                "live_mask true, z finite, and a phase label "
                "([frame_start, frame_end) half-open; uncovered frames -> `dead`).\n"
                "scope `live_dead` = binary contrast ball-in-play vs dead; "
                "scope `full` = all frames (dead as a class); "
                "scope `live_only` = covered frames only.\n\n")
        f.write("## phase marginal (frame counts)\n\n")
        f.write(_md_table(ph_counts.reset_index().rename(
            columns={"index": "phase", 0: "frames"}), index=True) + "\n\n")
        f.write("## MI summary\n\n")
        f.write(_md_table(mi_df) + "\n")
        f.write("## archetype semantics (P(phi|s))\n")
        f.write("\n".join(sem_lines) + "\n")
    print(f"-> {out / 'latent_phase_report.md'}")


if __name__ == "__main__":
    main()

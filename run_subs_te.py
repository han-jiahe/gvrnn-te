"""dotTE for substituted-off starters and substitutes, per the GVRNN
computation protocol (GVRNN计算协议.txt):

    dotTE(p) = (1 / t_p) * sum_over_targets TE(p -> target)

t_p = p's on-pitch tracked minutes (live frames / 600).  Targets = all OTHER
outfield nodes in the model except p himself and p's swap partner (protocol
exclusion rule).  Pairs with no co-presence window (n == 0) are marked
skipped and excluded from the sum.  S = {substitute : dotTE(sub) - dotTE(starter) > 0}.

Estimator: same pipeline as latent_pair_te.py / phase_te.py -- per-player
frozen K-means symbolization (K=8, z-score per dim, seed), 10 Hz transitions
with both players live at both ends + contiguous frames + same period,
discrete plug-in TE, 200 block-permutation surrogates (block 25, source
symbols shuffled), eff = raw - surr_mean, p = (#{surr >= raw}+1)/(S+1),
BH-FDR across the pair family.

Outputs (--out-dir): dot_te.csv (one row per ordered pair x k),
subs_te_summary.json (per-source dotTE, pair differences, S),
dot_te_report.md (Chinese, annotated).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_pair_te import pair_mask, te_pair
from latent_phase import fit_symbols
from mte.significance import bh_fdr


def parse_swap_pairs(spec: str) -> list:
    """'sub,starter;sub,starter' -> [(sub_pid, starter_pid), ...]."""
    out = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        out.append((int(a), int(b)))
    return out


def dot_te_rows(df: pd.DataFrame, sources: list, swap: dict, minutes: dict):
    """Per-source dotTE from the per-pair rows (eff as primary, raw as
    sensitivity, mean-per-target as robustness to unequal target counts)."""
    out = {}
    for pid in sources:
        rows = df[(df["src"] == pid) & (df["skipped"] == False)]
        partner = swap[pid]
        rows = rows[rows["dst"] != partner]          # protocol exclusion (safety)
        n_tgt = len(rows)
        if n_tgt == 0:
            out[pid] = {"n_targets": 0, "sum_eff": None, "dot_te": None,
                        "dot_te_raw": None, "mean_eff_per_target": None}
            continue
        sum_eff = float(rows["eff"].sum())
        sum_raw = float(rows["te"].sum())
        t = minutes[pid]
        out[pid] = {
            "n_targets": int(n_tgt),
            "sum_eff": sum_eff,
            "sum_raw": sum_raw,
            "dot_te": sum_eff / t,                   # protocol: sum(TE)/t
            "dot_te_raw": sum_raw / t,
            "mean_eff_per_target": (sum_eff / n_tgt) / t,
            "minutes_on": t,
        }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", required=True, type=Path)
    ap.add_argument("--live", required=True, type=Path)
    ap.add_argument("--swap-pairs", required=True,
                    help="sub,starter pairs, ';'-separated, e.g. "
                         "43829,133501;163972,50951;31147,38673")
    ap.add_argument("--k", default="8")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--n-surrogates", default=200, type=int)
    ap.add_argument("--block-len", default=25, type=int)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    lat = np.load(args.latents)
    trk = np.load(args.live)
    assert np.array_equal(lat["frame"], trk["frame"]), "frame arrays differ"
    frames = lat["frame"]
    z_post = lat["z_post"]
    periods = lat["periods"]
    live = trk["live_mask"]
    pids = [int(p) for p in trk["player_ids"]]
    node_of = {pid: i for i, pid in enumerate(pids)}
    ks = [int(x) for x in args.k.split(",")]
    swap_pairs = parse_swap_pairs(args.swap_pairs)
    swap = {}                                        # pid -> swap partner pid
    for sub, start in swap_pairs:
        swap[sub] = start
        swap[start] = sub
    sources = sorted(swap.keys())

    syms = {}
    for pid in pids:
        node = node_of[pid]
        for k in ks:
            syms[(pid, k)], _ = fit_symbols(z_post[:, node, :],
                                            live[:, node].astype(bool), k,
                                            args.seed)

    rows = []
    for src in sources:
        for dst in pids:
            if dst == src or dst == swap[src]:
                continue
            a, b = node_of[src], node_of[dst]
            mask = pair_mask(frames, periods, live[:, a].astype(bool),
                             live[:, b].astype(bool))
            for k in ks:
                if mask.sum() == 0:
                    rows.append({"src": src, "dst": dst, "src_node": a,
                                 "dst_node": b, "k": k, "te": np.nan,
                                 "h": np.nan, "n": 0, "persist": np.nan,
                                 "te_norm": np.nan, "eff": np.nan,
                                 "surr_mean": np.nan, "p": np.nan,
                                 "skipped": True})
                    continue
                rng = np.random.default_rng(args.seed + a * 100 + b * 10 + k)
                st = te_pair(syms[(src, k)], syms[(dst, k)], mask, k, rng,
                             args.n_surrogates, args.block_len)
                rows.append({"src": src, "dst": dst, "src_node": a,
                             "dst_node": b, "k": k, **st, "skipped": False})
                print(f"{src}->{dst} K={k}: n={st['n']} H={st['h']:.4f} "
                      f"TE={st['te']:.5f} eff={st['eff']:.5f} p={st['p']:.3f}")

    dfm = pd.DataFrame(rows)
    m = len(dfm)
    ps = dfm["p"].to_numpy()
    dfm["bh_sig"] = bh_fdr(ps, q=0.05) & (~dfm["skipped"]).to_numpy()
    minutes = {pid: float(live[:, node_of[pid]].sum()) / 600.0 for pid in pids}
    dfm["minutes_on"] = dfm["src"].map(minutes)
    dfm.to_csv(out / "dot_te.csv", index=False, encoding="utf-8")

    dots = dot_te_rows(dfm, sources, swap, minutes)
    summary = {
        "meta": {
            "match_id": 1886347, "team_id": 4177, "model": "m2",
            "latents": str(args.latents), "k": ks, "seed": args.seed,
            "n_surrogates": args.n_surrogates, "block_len": args.block_len,
            "nodes": {str(pid): node_of[pid] for pid in pids},
            "minutes_on": {str(pid): minutes[pid] for pid in pids},
        },
        "swap_pairs": [{"sub": s, "starter": t} for s, t in swap_pairs],
        "sources": dots,
        "S": {"pairs": [], "winners": []},
    }
    for sub, start in swap_pairs:
        if dots[sub]["dot_te"] is None or dots[start]["dot_te"] is None:
            continue
        diff = dots[sub]["dot_te"] - dots[start]["dot_te"]
        entry = {"sub": sub, "starter": start, "diff": diff,
                 "sub_dot_te": dots[sub]["dot_te"],
                 "starter_dot_te": dots[start]["dot_te"]}
        summary["S"]["pairs"].append(entry)
        if diff > 0:
            summary["S"]["winners"].append({"winner": sub, "diff": diff})
    with open(out / "subs_te_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Chinese report ----
    L = []
    L.append("# dotTE：替补 / 被替补球员对队友的转移熵（GVRNN 计算协议）\n")
    L.append(f"- 模型：`{args.latents}`（m2 = 4 全勤 + 6 轮换 + 球，11 节点）\n"
             f"- 估计器：10 Hz 离散 plug-in，K = {ks}，代理 {args.n_surrogates} 次"
             f"（块长 {args.block_len}），seed {args.seed}\n\n")
    L.append("## 1. 协议口径（GVRNN计算协议.txt）\n\n"
             "- `TE(p→q)`：源 p 与目标 q 共同在场窗口内、两端 live、帧连续、"
             "同半场的转移上估计（符号化 K-means，K=8）。\n"
             "- `dotTE(p) = Σ_q TE(p→q) / t_p`：**t_p = p 的在场跟踪帧数 / 600**"
             "（跟踪口径分钟；比赛时钟分钟见 meta）。\n"
             "- 目标集：模型内其余外场节点，**排除自己与换人伙伴**（协议规则）；"
             "无共同在场时段的对标记 `skipped`，不计入求和。\n"
             "- `S = {替补 : dotTE(替补) − dotTE(首发) > 0}`。\n\n")
    L.append("## 2. 在场窗口（tracking live_mask 口径）\n\n")
    L.append("| 球员 | pid | node | live 帧区间 | 在场跟踪分钟 |\n|---|---|---|---|---|\n")
    for pid in pids:
        w = np.where(live[:, node_of[pid]])[0]
        L.append(f"| {pid} | {node_of[pid]} | {frames[w[0]]}–{frames[w[-1]]} "
                 f"| {len(w) / 600:.1f} |\n")
    L.append("\n## 3. 每对 TE（eff = raw − 代理均值；p 为块置换单侧上尾）\n\n")
    L.append("| src | dst | k | n | te | eff | p | bh_sig | skipped |\n"
             "|---|---|---|---|---|---|---|---|---|\n")
    for r in dfm.itertuples():
        L.append(f"| {r.src} | {r.dst} | {r.k} | {r.n} | "
                 f"{r.te:.5f} | {r.eff:.5f} | {r.p:.3f} | {r.bh_sig} | "
                 f"{r.skipped} |\n")
    L.append("\n## 4. dotTE 汇总与 S 判定\n\n")
    L.append("| 球员 | 类别 | 目标数 | Σeff | t_p (min) | dotTE (Σeff/t) | "
             "dotTE_raw | 均值口径 |\n|---|---|---|---|---|---|---|---|\n")
    for pid in sources:
        d = dots[pid]
        L.append(f"| {pid} | {'替补' if pid in {s for s, _ in swap_pairs} else '被替补'} "
                 f"| {d['n_targets']} | {d['sum_eff']:.5f} | {d['minutes_on']:.2f} "
                 f"| {d['dot_te']:.6f} | {d['dot_te_raw']:.6f} | "
                 f"{d['mean_eff_per_target']:.6f} |\n")
    L.append("\n| 换人对 | dotTE(替补) | dotTE(首发) | 差值 | 判定 |\n"
             "|---|---|---|---|---|\n")
    winners = []
    for sub, start in swap_pairs:
        d = dots[sub]["dot_te"] - dots[start]["dot_te"]
        w = "替补胜 → 入 S" if d > 0 else "首发胜"
        if d > 0:
            winners.append(sub)
        L.append(f"| {sub}↔{start} | {dots[sub]['dot_te']:.6f} | "
                 f"{dots[start]['dot_te']:.6f} | {d:+.6f} | {w} |\n")
    L.append(f"\n**S = {winners if winners else '∅（无替补胜出）'}**\n\n")
    L.append("## 5. 符号注解\n\n"
             "- `te`（TE_raw）：离散 plug-in 转移熵，bits/转移。\n"
             "- `eff`：TE_eff = raw − 代理均值（偏置校正）。\n"
             "- `p`：块置换单侧上尾 p 值，地板 1/(S+1)。\n"
             "- `bh_sig`：跨 54 对 BH-FDR（q=0.05）显著性。\n"
             "- `dotTE`：Σ TE_eff / t_p（协议主口径）；`dotTE_raw` 用 raw 求和；\n"
             "  `均值口径` = (Σeff/目标数)/t_p，消目标数差异的敏感性口径。\n"
             "- `t_p`：跟踪口径在场分钟（live 帧/600）；比赛时钟分钟略长"
             "（含丢帧死球段）。\n"
             "- `skipped`：两球员无共同在场窗口（n=0），不参与 dotTE。\n")
    with open(out / "dot_te_report.md", "w", encoding="utf-8") as f:
        f.write("".join(L))
    print(f"-> {out / 'dot_te.csv'}, {out / 'subs_te_summary.json'}, "
          f"{out / 'dot_te_report.md'}")


if __name__ == "__main__":
    main()

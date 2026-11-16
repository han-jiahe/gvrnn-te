"""dotTE / Ψ_A(V̇I) 管线结果图（sub1–sub4，PNG，随仓库发布）。

与 te_emergence.figures 同一设计系统：SURFACE/INK 墨色 token、发丝 y 网格、
≥8px 标记 + 2px 表面环、2px 线、文本一律墨色 token（不用序列色）、
类别色固定顺序（蓝 #2a78d6 → 橙 #eb6834 → 青 #1baf7a）、单一 y 轴、
≥2 系列必有图例且 ≤4 系列同时直标。类别色分配（全图一致）：
**替补 = 橙，被替补 = 蓝**。

数据源（本模块只读取、不复算）：
- results/subs_te/dot_te.csv           每对一行（42 有效 + 6 skipped）
- results/subs_te/subs_te_summary.json 每源 dotTE / 换人对差值 / S
- results/vi_te_1886347.json           halves.home|{1,2}.psi（nats）

图：
- sub1_emergence_plane   Ψ vs I(V̇I_t;V̇I_{t'})：Ψ_A 主口径 vs Ψ_uncond 全量对照（P1/P2）
- sub2_dot_te_dumbbell   3 换人对哑铃：替补 vs 被替补 dotTE（bits/min）+ Δ
- sub3_pair_scatter      42 有效对 TE_eff vs 源在场分钟（替补/被替补着色 + 短窗区标注）
- sub4_psi_waterfall     Ψ_A 瀑布分解：i_macro − Σ_S I(A^i;V̇I′）（P1/P2 两面板）

用法：
    python -m te_emergence.sub_figures \
        --dot-te results/subs_te/dot_te.csv \
        --summary results/subs_te/subs_te_summary.json \
        --vi-json results/vi_te_1886347.json \
        --out-dir results/subs_te/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# 导入即应用 figures.py 的全局 rcParams（含 Agg 后端、Microsoft YaHei、150 dpi）
from te_emergence.figures import (  # noqa: F401
    SURFACE, INK, INK2, MUTED, GRID, BASE, BLUE, ORANGE, AQUA, _style_ax,
)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_NAME = {43829: "Moreno", 163972: "Mata", 31147: "Smith",
         133501: "Rogerson", 50951: "Brimmer", 38673: "May"}
_SUBS = {43829, 163972, 31147}


def _load(dot_te_csv: Path, summary_json: Path, vi_json: Path):
    d = pd.read_csv(dot_te_csv)
    valid = d[~d["skipped"] & d["eff"].notna()].copy()
    valid["role"] = valid["src"].map(lambda s: "替补" if s in _SUBS else "被替补")
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    psi = {k: v["psi"] for k, v in
           json.loads(Path(vi_json).read_text(encoding="utf-8"))["halves"].items()}
    return valid, summary, psi


def _save(fig, out_dir: Path, name: str) -> Path:
    out = Path(out_dir) / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out}")
    return out


# ------------------------------------------------------------------ sub1 涌现平面
def sub1_emergence_plane(psi: dict, out_dir: Path) -> Path:
    """Ψ vs I(V̇I_t;V̇I_{t'})：每半场一个点，Ψ_A（主口径）与 Ψ_uncond（对照）成对。"""
    fig, ax = plt.subplots(figsize=(6.9, 4.7))
    ax.axhline(0, color=BASE, linewidth=1.0)

    xs, ya, yu = [], [], []
    for key in ("home|1", "home|2"):
        p = psi[key]
        im = p["i_macro_nats"]
        sub_sum = sum(p["per_player_mi_nats"].get(str(s), 0.0) for s in _SUBS)
        xs.append(im)
        ya.append(im - sub_sum)
        yu.append(p["psi_uncond_nats"])

    ax.scatter(xs, ya, s=64, color=BLUE, alpha=0.9, linewidths=2,
               edgecolors=SURFACE, zorder=4, label="Ψ_A（协议主口径）")
    ax.scatter(xs, yu, s=64, facecolor=SURFACE, edgecolors=MUTED, linewidths=1.6,
               zorder=4, label="Ψ_uncond（全量对照）")
    for x, a, u in zip(xs, ya, yu):
        ax.plot([x, x], [a, u], color=MUTED, linewidth=1.0,
                linestyle=(0, (4, 3)), zorder=3)

    ax.text(xs[0] + 0.0003, ya[0] + 0.0016, "P1 · 上半场", color=INK,
            fontweight="bold", fontsize=10)
    ax.text(xs[0] + 0.0003, ya[0] - 0.0046, r"S 无人上场 → Ψ_A = $I(\dot{VI};\,\dot{VI}')$ = +0.010953",
            color=INK2, fontsize=8.5)
    ax.text(xs[0] + 0.0003, yu[0] - 0.0046, "Ψ_uncond −0.011512",
            color=INK2, fontsize=8.5)
    ax.text(xs[1] + 0.0003, ya[1] + 0.0016, "P2 · 下半场", color=INK,
            fontweight="bold", fontsize=10)
    ax.text(xs[1] - 0.0003, ya[1] - 0.0046, "Ψ_A = −0.002459",
            color=INK2, fontsize=8.5, ha="right")
    ax.text(xs[1] - 0.0003, (ya[1] + yu[1]) / 2, r"$\Sigma_S I(A^i;\,\dot{VI}')$ = 0.021854",
            color=INK2, fontsize=8.5, va="center", ha="right")
    ax.text(xs[1] - 0.0003, yu[1] - 0.0046, "Ψ_uncond −0.042334",
            color=INK2, fontsize=8.5, ha="right")

    ax.set_xlim(0.0080, 0.0218)
    ax.set_ylim(-0.0530, 0.0210)
    ax.set_xlabel(r"$I(\dot{VI}_t;\,\dot{VI}_{t'})$（nats）")
    ax.set_ylabel("Ψ（nats）")
    ax.set_title("涌现平面：Ψ_A 与 Ψ_uncond（1886347 · Auckland 主队，两半场）")
    ax.legend(loc="lower right", frameon=False)
    _style_ax(ax)
    fig.text(0.012, 0.02,
             r"Ψ_A = $I(\dot{VI}_t;\,\dot{VI}_{t'}) - \sum_{i\in S} I(A^i_t;\,\dot{VI}_{t'})$，S = dotTE 胜过被替补的"
             "三名替补（43829 · 163972 · 31147）；虚线跨度 = 被减项 Σ_S。MI 直接取自 "
             "vi_te_1886347.json（1 Hz · nats · 不复算）。P2 的 Ψ_A 落在零线以下、"
             "但高于 Ψ_uncond 0.0399 nats。",
             color=MUTED, fontsize=8.5)
    return _save(fig, out_dir, "sub1_emergence_plane.png")


# ------------------------------------------------------------------ sub2 dotTE 哑铃
def sub2_dot_te_dumbbell(summary: dict, out_dir: Path) -> Path:
    """3 个换人对：替补（橙）vs 被替补（蓝）dotTE，箭头指向更大者。"""
    pairs = summary["S"]["pairs"]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ys = list(range(len(pairs)))[::-1]  # 第 1 对在最上
    labels = [f"{_NAME[p['sub']]} {p['sub']} · 替补  —  "
              f"{_NAME[p['starter']]} {p['starter']} · 被替补" for p in pairs]

    for y, p, lab in zip(ys, pairs, labels):
        xs = [p["sub_dot_te"], p["starter_dot_te"]]
        ax.annotate("", xy=(xs[0], y), xytext=(xs[1], y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2,
                                    shrinkA=15, shrinkB=15), zorder=2)
        ax.scatter([xs[0]], [y], s=64, color=ORANGE, alpha=0.9, linewidths=2,
                   edgecolors=SURFACE, zorder=3,
                   label="替补" if y == ys[0] else None)
        ax.scatter([xs[1]], [y], s=64, color=BLUE, alpha=0.9, linewidths=2,
                   edgecolors=SURFACE, zorder=3,
                   label="被替补" if y == ys[0] else None)
        ax.text(xs[0] - 0.00025, y + 0.22, f"{xs[0]:.5f}", color=INK2,
                fontsize=9, ha="right")
        ax.text(xs[1] + 0.00025, y - 0.26, f"{xs[1]:.5f}", color=INK2,
                fontsize=9, ha="left")
        ax.text(0.01125, y, f"Δ = +{p['diff']:.5f}", color=INK,
                fontsize=9, fontweight="bold", va="center")

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-0.0010, 0.0125)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("dotTE = Σ_q TE_eff(p→q) / t_p（bits/min）")
    ax.set_title("dotTE 哑铃：替补 vs 被替补（3 个换人对）")
    ax.legend(loc="upper right", frameon=False)
    _style_ax(ax)
    fig.text(0.012, 0.02,
             "Δ = dotTE(替补) − dotTE(被替补) > 0 → 替补进入 S；三对全胜（+0.00263 / "
             "+0.00094 / +0.00873）。Smith（31147）t_p 仅 6.95 跟踪分钟，短窗代理偏差"
             "致其 0.00930 为全样本最高值——须存疑（见 sub3 与 dot_te_report §4）。",
             color=MUTED, fontsize=8.5)
    return _save(fig, out_dir, "sub2_dot_te_dumbbell.png")


# ------------------------------------------------------------------ sub3 成对散点
def sub3_pair_scatter(valid: pd.DataFrame, out_dir: Path) -> Path:
    """42 有效对：TE_eff vs 源在场分钟；短窗区（t_p < 10 min）着色标注。"""
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ymax = valid["eff"].max() * 1.12
    ax.set_ylim(-0.002, ymax)
    ax.set_xlim(0, 70)

    ax.axvspan(0, 10, color=GRID, alpha=0.55, zorder=0)
    ax.text(9.3, ymax * 0.97, "短窗区（t_p < 10 min）", color=INK2,
            fontsize=8.5, ha="right", va="top")

    starters = valid[valid["role"] == "被替补"]
    subs = valid[valid["role"] == "替补"]
    ax.scatter(starters["minutes_on"], starters["eff"], s=46, color=BLUE, alpha=0.85,
               linewidths=2, edgecolors=SURFACE, zorder=3, label="被替补")
    ax.scatter(subs["minutes_on"], subs["eff"], s=46, color=ORANGE, alpha=0.85,
               linewidths=2, edgecolors=SURFACE, zorder=3, label="替补")

    smith = subs[subs["src"] == 31147]
    ax.annotate("Smith 31147 · t_p = 6.95 min\n代理偏差 → eff 虚高（存疑）",
                xy=(smith["minutes_on"].iloc[0], smith["eff"].max()),
                xytext=(28, ymax * 0.86), color=INK2, fontsize=8.5, ha="left",
                arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1.0))

    ax.set_xlabel("源球员在场跟踪分钟 t_p（live 帧数 / 600）")
    ax.set_ylabel("TE_eff（bits/转移）")
    ax.set_title("42 有效对：TE_eff 与源在场时间（替补 / 被替补源）")
    ax.legend(loc="upper right", frameon=False)
    _style_ax(ax)
    fig.text(0.012, 0.02,
             "每个点 = 一个有序对（源 → 共同在场队友）；42 有效对 + 6 对无共同在场 skipped。"
             "全部 42 对 p = 1/201 地板、BH-FDR 全真——块置换代理（块 25）的短窗偏差伪影，"
             "显著性不作依据（dot_te_report §4）；eff = raw − 代理均值。",
             color=MUTED, fontsize=8.5)
    return _save(fig, out_dir, "sub3_pair_scatter.png")


# ------------------------------------------------------------------ sub4 Ψ_A 瀑布
def sub4_psi_waterfall(psi: dict, out_dir: Path) -> Path:
    """Ψ_A 减法分解（P1/P2 两面板，共享 y）：i_macro − Σ_S I(A^i;V̇I′) = Ψ_A。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    fig.suptitle(r"$\Psi_A(\dot{VI})$ 瀑布分解：$I(\dot{VI}_t;\,\dot{VI}_{t'}) - \sum_{i\in S} I(A^i_t;\,\dot{VI}_{t'})$（nats）",
                 y=1.02)
    subs_order = [43829, 163972, 31147]
    panels = [("P1 · 上半场（S 无人上场）", "home|1"),
              ("P2 · 下半场（3 替补在场）", "home|2")]
    ylim = (-0.052, 0.026)

    for ax, (title, key) in zip(axes, panels):
        p = psi[key]
        im = p["i_macro_nats"]
        contribs = [p["per_player_mi_nats"].get(str(s), 0.0) for s in subs_order]
        psi_a = im - sum(contribs)
        heights = [im] + [-c for c in contribs] + [psi_a]
        colors = [BLUE] + [ORANGE] * len(contribs) + [AQUA]
        labels = [r"$I(\dot{VI};\,\dot{VI}')$"] + [f"{_NAME[s]}\n{s}" for s in subs_order] + ["Ψ_A"]

        ax.set_ylim(*ylim)
        ax.axhline(0, color=BASE, linewidth=1.0)
        ax.axhline(p["psi_uncond_nats"], color=MUTED, linewidth=1.0,
                   linestyle=(0, (5, 4)), zorder=2)
        ax.text(0.012, (p["psi_uncond_nats"] - ylim[0]) / (ylim[1] - ylim[0]),
                "Ψ_uncond", transform=ax.transAxes, color=MUTED, fontsize=8.5,
                va="bottom")

        xs = np.arange(len(heights))
        ax.bar(xs, heights, width=0.45, color=colors, zorder=3)
        for x, h, is_res in zip(xs, heights, [False] * (len(heights) - 1) + [True]):
            va = "bottom" if h >= 0 else "top"
            off = 0.0015 if h >= 0 else -0.0015
            ax.text(x, h + off, f"{h:+.6f}" if is_res else f"{h:+.5f}",
                    ha="center", va=va, color=INK2, fontsize=8.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title)
        _style_ax(ax)

    axes[0].set_ylabel("nats")
    axes[1].legend(handles=[
        Patch(facecolor=BLUE, label=r"$I(\dot{VI}_t;\,\dot{VI}_{t'})$ 总 MI"),
        Patch(facecolor=ORANGE, label=r"S 成员 $I(A^i_t;\,\dot{VI}_{t'})$（被减）"),
        Patch(facecolor=AQUA, label="Ψ_A 净额"),
    ], loc="upper right", frameon=False, fontsize=8.5)
    fig.text(0.012, 0.02,
             "P2：0.019395 − 0.003014 − 0.010100 − 0.008739 = −0.002459 nats；"
             "虚线 = Ψ_uncond 全量对照（P2 −0.042334，Ψ_A 相对其回升 +0.0399）。"
             r"P1 替补未上场，Ψ_A = $I(\dot{VI};\,\dot{VI}')$。MI 直接取自 vi_te_1886347.json，"
             "本模块只做减法。",
             color=MUTED, fontsize=8.5)
    return _save(fig, out_dir, "sub4_psi_waterfall.png")


# ------------------------------------------------------------------ 入口
def make_figures(dot_te_csv: Path, summary_json: Path, vi_json: Path,
                 out_dir: Path) -> list[Path]:
    valid, summary, psi = _load(dot_te_csv, summary_json, vi_json)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return [
        sub1_emergence_plane(psi, out_dir),
        sub2_dot_te_dumbbell(summary, out_dir),
        sub3_pair_scatter(valid, out_dir),
        sub4_psi_waterfall(psi, out_dir),
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="dotTE / Ψ_A(V̇I) 管线结果图（sub1–sub4）")
    ap.add_argument("--dot-te", default="results/subs_te/dot_te.csv")
    ap.add_argument("--summary", default="results/subs_te/subs_te_summary.json")
    ap.add_argument("--vi-json", default="results/vi_te_1886347.json")
    ap.add_argument("--out-dir", default="results/subs_te/figures")
    args = ap.parse_args()
    paths = make_figures(Path(args.dot_te), Path(args.summary),
                         Path(args.vi_json), Path(args.out_dir))
    print(f"完成：{len(paths)} 张图 → {args.out_dir}")

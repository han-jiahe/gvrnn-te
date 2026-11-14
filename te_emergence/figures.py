"""vi_bridge 汇总图（PNG，静态报告用）。

遵循 dataviz 规范：细标记（柱 ≤24px、点 ≥8px + 2px 表面环、2px 线）、
发丝网格、文本一律用墨色 token（不用序列色）、≥2 系列必有图例、
类别色固定顺序（蓝 #2a78d6 → 橙 #eb6834 → 青 #1baf7a）、单一 y 轴。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

# ---------------------------------------------------------------- design tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW = "#eda100"
POS_COLORS = {"Defender": BLUE, "Midfielder": ORANGE, "Forward": AQUA}
POS_ORDER = ["Defender", "Midfielder", "Forward"]
# SkillCorner position_group → 粗粒度位置组
_POS_MAP = {
    "Central Defender": "Defender", "Full Back": "Defender",
    "Midfield": "Midfielder",
    "Wide Attacker": "Forward", "Center Forward": "Forward",
}

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASE,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "axes.linewidth": 1.0,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
})


def _style_ax(ax):
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _pos_color(pg: str):
    return POS_COLORS.get(pg, MUTED)


# ------------------------------------------------------------------ figures
def fig_emergence_plane(ht: pd.DataFrame, out_dir: Path) -> None:
    """图 1：涌现平面——Δ（净下行）vs Γ_TE（TE 因果涌现），40 半场。

    象限判据（Rosas 实用准则的 TE 形式）：右上 = Δ>0 且 Γ_TE>0 →
    「TE 层面既涌现又下行」；右下 = Γ_TE>0 且 Δ≤0 → 解耦式涌现。
    主口径 gaussian；ksg/discrete 作灵敏度子图。
    """
    mk = "gaussian"
    d = ht[[f"{mk}|delta_bits", f"{mk}|gamma_te_bits",
            "match_id", "half", "team"]].dropna()
    if d.empty:
        print("fig1: 无 gaussian 半场数据，跳过")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=False)
    fig.suptitle(f"涌现平面：净下行 Δ 与 TE 因果涌现 Γ_TE"
                 f"（{len(ht)} 半场，1 Hz VI 链）", y=1.02)
    mks = ["gaussian", "ksg|k=3", "discrete|K=3"]
    titles = ["主口径 · 高斯闭式", "灵敏度 · KSG k=3（零分布偏移大，仅参考）",
              "灵敏度 · 离散 K=3"]
    for ax, mk_, title in zip(axes, mks, titles):
        d = ht[[f"{mk_}|delta_bits", f"{mk_}|gamma_te_bits"]].dropna()
        if d.empty:
            ax.set_visible(False)
            continue
        x = d[f"{mk_}|delta_bits"].to_numpy()
        y = d[f"{mk_}|gamma_te_bits"].to_numpy()
        ax.axhline(0, color=BASE, linewidth=1.0)
        ax.axvline(0, color=BASE, linewidth=1.0)
        ax.scatter(x, y, s=52, color=BLUE, alpha=0.85,
                   linewidths=2, edgecolors=SURFACE, zorder=3)
        both = (x > 0) & (y > 0)
        if both.any():
            ax.scatter(x[both], y[both], s=64, facecolor="none",
                       edgecolors="#d03b3b", linewidths=1.6, zorder=4)
        ax.set_xlabel("Δ = ΣD − ΣU（bits，净下行）")
        ax.set_ylabel("Γ_TE = I(V;V′) − ΣU（bits）")
        ax.set_title(title)
        _style_ax(ax)
    fig.text(0.012, 0.02, "每个点 = 一支球队的一个半场；红圈 = Δ>0 且 Γ_TE>0。"
             "注意：gaussian 口径下 Δ 量级与代理蒙特卡洛误差同阶，象限为定性；"
             "KSG 子图受零分布偏移主导（见报告 §4）", color=MUTED, fontsize=8.5)
    fig.savefig(out_dir / "fig1_emergence_plane.png")
    plt.close(fig)


def fig_du_dumbbell(ht: pd.DataFrame, out_dir: Path) -> None:
    """图 2：40 半场 D_sum 与 U_sum 哑铃图（gaussian 主口径）。

    蓝点 = 下行总和 ΣD、橙点 = 上行总和 ΣU，按 Δ 排序；连线长度即净下行。
    """
    mk = "gaussian"
    d = ht[["match_id", "half", "team", f"{mk}|downward_sum_bits",
            f"{mk}|upward_sum_bits"]].dropna().copy()
    if d.empty:
        print("fig2: 无 gaussian 半场数据，跳过")
        return
    d["delta"] = d[f"{mk}|downward_sum_bits"] - d[f"{mk}|upward_sum_bits"]
    d = d.sort_values("delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    y = np.arange(len(d))
    xd = d[f"{mk}|downward_sum_bits"].to_numpy()
    xu = d[f"{mk}|upward_sum_bits"].to_numpy()
    for yi, a, b in zip(y, xd, xu):
        ax.plot([a, b], [yi, yi], color=MUTED, linewidth=1.0, zorder=1)
    ax.scatter(xd, y, s=44, color=BLUE, linewidths=2, edgecolors=SURFACE,
               zorder=3, label="下行 ΣD（TE(V→X_k)）")
    ax.scatter(xu, y, s=44, color=ORANGE, linewidths=2, edgecolors=SURFACE,
               zorder=3, label="上行 ΣU（TE(X_k→V)）")
    ax.axvline(0, color=BASE, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{str(r.match_id)[-4:]} {r.team[:6]} P{r.half[-1]}"
                        for r in d.itertuples()], fontsize=7.5, color=INK2)
    ax.set_xlabel("Σ TE（bits）")
    ax.set_title("每半场出场球员（n_trans≥60，10–15 人）的下行 / 上行 TE 总和"
                 "（按 Δ 排序）")
    ax.legend(frameon=False, loc="lower right")
    _style_ax(ax)
    fig.text(0.012, 0.02, "点间距 = 净下行 Δ = ΣD − ΣU；蓝点在右 → 宏观→微观"
             "主导", color=MUTED, fontsize=8.5)
    fig.savefig(out_dir / "fig2_du_dumbbell.png")
    plt.close(fig)


def fig_player_du_scatter(pt: pd.DataFrame, out_dir: Path) -> None:
    """图 3：球员×半场 D vs U 散点（按位置组着色，gaussian）。

    对角线上方 = 该球员向宏观的贡献超过接收（U>D，发起方）。
    """
    mk = "gaussian"
    d = pt[pt["direction"] == "D"][["match_id", "half", "pid",
                                   "position_group", f"{mk}|eff"]].copy()
    u = pt[pt["direction"] == "U"][["match_id", "half", "pid",
                                   f"{mk}|eff"]].rename(
        columns={f"{mk}|eff": "u_eff"})
    d = d.merge(u, on=["match_id", "half", "pid"])
    d = d.rename(columns={f"{mk}|eff": "d_eff"}).dropna(
        subset=["d_eff", "u_eff"])
    d["position_group"] = d["position_group"].map(_POS_MAP).fillna(
        d["position_group"])
    d = d[d["position_group"].isin(POS_ORDER)]
    if d.empty:
        print("fig3: 无球员数据，跳过")
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    lims = [min(d["d_eff"].min(), d["u_eff"].min()),
            max(d["d_eff"].max(), d["u_eff"].max())]
    ax.plot(lims, lims, color=BASE, linewidth=1.0, zorder=1)
    for pg in POS_ORDER:
        g = d[d["position_group"] == pg]
        ax.scatter(g["d_eff"], g["u_eff"], s=38, color=_pos_color(pg),
                   alpha=0.75, linewidths=2, edgecolors=SURFACE,
                   zorder=3, label=pg)
    ax.set_xlabel("D = TE(V→X_k) eff（bits，宏观→该球员）")
    ax.set_ylabel("U = TE(X_k→V) eff（bits，该球员→宏观）")
    ax.set_title("球员层信息流方向（每点 = 球员 × 半场，gaussian eff）")
    ax.legend(frameon=False, title="位置组")
    _style_ax(ax)
    fig.text(0.012, 0.02, "对角线上方 = 该球员向宏观的贡献超过接收"
             "（发起方）；下方 = 接收方", color=MUTED, fontsize=8.5)
    fig.savefig(out_dir / "fig3_player_du_scatter.png")
    plt.close(fig)


def fig_cross_control(ht: pd.DataFrame, out_dir: Path) -> None:
    """图 4：队内 D_sum 与跨队负对照 N×cross_D（重叠伪影基线）。

    蓝箱 = 队内（真实因果 + 滑窗重叠伪影）；橙箱 = 跨队（仅重叠伪影）。
    两箱之差 = 伪影校正后的下行因果 D*_sum（N = 该半场出场球员数）。
    """
    mks = sorted({c[: -len("|downward_sum_bits")]
                  for c in ht.columns if c.endswith("|downward_sum_bits")})
    if not mks:
        print("fig4: 无 downward_sum_bits 列，跳过")
        return
    fig, axes = plt.subplots(1, len(mks), figsize=(4 * len(mks), 4.6),
                             sharey=False, squeeze=False)
    fig.suptitle("下行因果的跨队负对照（队内 − 跨队 = 重叠伪影校正）", y=1.02)
    for ax, mk in zip(axes.flat, mks):
        d = ht[[f"{mk}|downward_sum_bits",
                f"{mk}|cross_down_mean_bits",
                f"{mk}|n_downward"]].dropna()
        if d.empty:
            ax.set_visible(False)
            continue
        within = d[f"{mk}|downward_sum_bits"].to_numpy()
        cross = (d[f"{mk}|n_downward"].to_numpy()
                 * d[f"{mk}|cross_down_mean_bits"].to_numpy())
        parts = ax.violinplot([within, cross], positions=[0, 1],
                              showextrema=False, widths=0.55)
        for body, c in zip(parts["bodies"], (BLUE, ORANGE)):
            body.set_facecolor(c)
            body.set_alpha(0.28)
        for i, arr in enumerate((within, cross)):
            bp = ax.boxplot(arr, positions=[i], widths=0.18,
                            showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, linewidth=1.4),
                            boxprops=dict(facecolor=SURFACE, edgecolor=INK2,
                                          linewidth=1.0),
                            whiskerprops=dict(color=INK2, linewidth=1.0),
                            capprops=dict(color=INK2, linewidth=1.0))
        ax.axhline(0, color=BASE, linewidth=1.0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["队内 ΣD", "跨队 N×D_ctrl"], fontsize=9)
        ax.set_title(mk if not mk.startswith("ksg") else f"{mk}（退化，仅参考）",
                     fontsize=10)
        _style_ax(ax)
    fig.text(0.012, 0.02, "跨队 TE(V_opp→X_own,k) 与队内 TE(V_own→X_own,k) "
             "共享同一 VI 滑窗重叠结构 → 两者的差是伪影的保守对照（跨队项"
             "可能还含对手真实耦合 → 可能过校正，非精确移除）。"
             "N×D_ctrl 与汇总 JSON 的 downward_star_sum_bits 恒等"
             "（N = 该半场出场球员数）", color=MUTED, fontsize=8.5)
    fig.savefig(out_dir / "fig4_cross_control.png")
    plt.close(fig)


def make_figures(ht: pd.DataFrame, pt: pd.DataFrame, out_dir) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 清旧图：避免本次运行未产出某图时陈旧 PNG 混入报告（只清 fig1-4，
    # fig5-9 属 diagrams.py，由 pool_vi_te 另行刷新）
    for stale in out_dir.glob("fig[1-4]_*.png"):
        stale.unlink()
        print(f"清除陈旧图 {stale.name}")
    fig_emergence_plane(ht, out_dir)
    fig_du_dumbbell(ht, out_dir)
    fig_player_du_scatter(pt, out_dir)
    fig_cross_control(ht, out_dir)
    print(f"figures -> {out_dir}")

"""项目全景流程图：桥接概念 / 数据流向 / 模块组件 / 技术路径 / 成果汇总。

与 figures.py 同一套设计 token（墨色文字、细边、蓝→橙→青固定顺序），
输出 PNG 到 results/figures/fig5..fig9。fig9 的成果数字从
vi_te_pool_stats.json 读取（保持与统计一致）；其余为结构图（静态）。
用法：python -m te_emergence.diagrams [--out-dir DIR] [--stats FILE]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

from te_emergence.figures import (        # noqa: E402
    SURFACE, INK, INK2, MUTED, BASE, BLUE, ORANGE, AQUA, YELLOW)

OUT_DIR = TE_ROOT / "results" / "figures"
STATS = TE_ROOT / "results" / "vi_te_pool_stats.json"

# 淡色填充（与序列色同色相的浅底）
TINT_BLUE = "#eaf1fb"
TINT_ORANGE = "#fdeee6"
TINT_AQUA = "#e7f6f0"
TINT_GRAY = "#f1f0ec"
TINT_RED = "#fbe9e7"
TINT_YELLOW = "#fbf3dd"

FIG_KW = dict(facecolor=SURFACE)


def _new_ax(w=13.0, h=7.0, x=100.0, y=100.0):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(SURFACE)
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, title, sub=None, fc=TINT_GRAY, ec=BASE,
         fs=9.5, tc=INK, subc=INK2):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.7",
        fc=fc, ec=ec, lw=1.1, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold", zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=fs - 2.4, color=subc, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold", zorder=3)
    return (x, y, w, h)


def _arrow(ax, x0, y0, x1, y1, color=INK2, style="-|>", lw=1.3, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0),
                zorder=1)


def _title(ax, txt, y=96.5):
    ax.text(50, y, txt, ha="center", va="center", fontsize=13.5,
            color=INK, fontweight="bold")


def _col_head(ax, xs, txts, y=88.0, color=INK):
    for x, t in zip(xs, txts):
        ax.text(x, y, t, ha="center", va="center", fontsize=10.5,
                color=color, fontweight="bold")


# ---------------------------------------------------------------- 图5 桥接概念
def fig_bridge_concept(out: Path) -> None:
    fig, ax = _new_ax(w=13.5, h=6.0)
    _title(ax, "桥接概念：Rosas 因果涌现 $\leftrightarrow$ 转移熵 TE")
    _col_head(ax, [16, 52, 87], ["Rosas 因果涌现", "桥接映射（成对 TE）",
                                 "本报告指标"], y=88.0)

    rows = [
        ("下行因果 D_t\n= Σ_i I(V_t;X^i_{t+1}|X^i_t)",
         "成对转移熵 TE(V→X_i)\n= I(V_t;X_i(t+1)|X_i(t))，1 阶",
         "D_k（精确 Rosas 下行）", TINT_BLUE, "恒等"),
        ("严格涌现 Ψ\n= 冗余 − 协同（需 PID）",
         "$\hat{\gamma}_{TE}$ = I(V;V′) − Σ_k U_k\n（TE 条件化操作定义）",
         "$\hat{\gamma}_{TE}$ / Ψ_uncond（非 PID 项）", TINT_AQUA, "操作化"),
        ("（无直接对应）",
         "跨队 TE(V_opp→X_own,k)\n共享同一滑窗重叠结构",
         "D*_k = D_k − cross_D_k\n（保守对照，可能过校正）", TINT_ORANGE, "伪影对照"),
    ]
    ytop = 74.0
    h = 16.0
    gap = 18.0
    for i, (left, mid, right, tint, tag) in enumerate(rows):
        y = ytop - i * gap
        _box(ax, 2, y, 28, h, left, fc=TINT_GRAY)
        _box(ax, 37, y, 30, h, mid, fc=tint)
        _box(ax, 73, y, 25, h, right, fc=TINT_GRAY, fs=8.6)
        _arrow(ax, 30, y + h / 2, 37, y + h / 2)
        _arrow(ax, 67, y + h / 2, 73, y + h / 2)
        ax.text(33.5, y + h - 1.6, tag, ha="center", fontsize=7.5,
                color=INK2)
    fig.text(0.01, 0.012,
             "Δ=ΣD−ΣU 为桥接自定的净下行指标（Rosas 未定义此判据）；"
             "$\hat{\gamma}_{TE}$/Ψ_uncond 不声称严格 Rosas Ψ（非 PID 冗余项）。",
             color=MUTED, fontsize=8)
    fig.savefig(out / "fig5_bridge_concept.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图6 数据流向
def fig_dataflow(out: Path) -> None:
    fig, ax = _new_ax(w=13.5, h=8.0)
    _title(ax, "数据流向：两条链汇合到成果报告")

    # 左支（VI 空间主链）
    left = [
        ("SkillCorner tracking", "opendata 20 场", TINT_GRAY),
        ("vi/data/raw/<id>/tracking_cache_v2.npz", "VI 1 Hz 链，V_t 聚类", TINT_GRAY),
        ("vi_te.py", "逐半场 TE：4 方法 × 4 方向", TINT_BLUE),
        ("vi_te_<id>.json ×10 场", "200 块置换代理", TINT_BLUE),
        ("verify_pool.py → pool_vi_te.py", "不变量筛查 → 池化+统计", TINT_AQUA),
        ("vi_te_pool_halves/players.csv + stats.json", "40 半场 / 1928 球员行", TINT_AQUA),
    ]
    # 右支（GVRNN 潜空间）
    right = [
        ("tracking_pair1.npz", "11 节点定制导出", TINT_GRAY),
        ("gvrnn.train / extract", "GVRNN m1p1（1886347）", TINT_GRAY),
        ("latents.npz", "潜链 PCA-1（宏观+节点）", TINT_BLUE),
        ("latent_te.py", "宏观$\leftrightarrow$节点 TE 交叉校验", TINT_BLUE),
        ("latent_te_m1p1.json", "11 节点 × 2 方向", TINT_AQUA),
    ]

    _col_head(ax, [28, 72], ["VI 空间主链", "GVRNN 潜空间（Stage 2）"], y=92.0)
    xw = 30
    xl = 13
    xr = 57
    y0 = 82.0
    h = 10.5
    step = 12.0
    for i, (t, s, c) in enumerate(left):
        y = y0 - i * step
        _box(ax, xl, y, xw, h, t, s, fc=c)
        if i < len(left) - 1:
            _arrow(ax, xl + xw / 2, y, xl + xw / 2, y - (step - h), color=INK2)
    for i, (t, s, c) in enumerate(right):
        y = y0 - i * step
        _box(ax, xr, y, xw, h, t, s, fc=c)
        if i < len(right) - 1:
            _arrow(ax, xr + xw / 2, y, xr + xw / 2, y - (step - h), color=INK2)

    # 汇合
    _box(ax, 30, 2, 40, 12, "vi_te_report.py", "→ vi_te_report.md（+ 8 张图）",
         fc=TINT_ORANGE)
    _arrow(ax, xl + xw / 2, y0 - 5 * step, 50, 14, color=ORANGE)
    _arrow(ax, xr + xw / 2, y0 - 4 * step, 50, 14, color=ORANGE)
    fig.savefig(out / "fig6_dataflow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图7 模块组件
def fig_architecture(out: Path) -> None:
    fig, ax = _new_ax(w=13.5, h=7.5)
    _title(ax, "模块组件：vi_bridge 包分层")

    layers = [
        ("引擎层", "vi_te.py",
         "核心：D_k/U_k/cross/D*/Δ/$\hat{\gamma}_{TE}$/Ψ（4 方法 × 4 方向，逐半场）", TINT_BLUE),
        ("编排层", "run_vi_te.py / run_all.py",
         "单场运行 + 10 场 ProcessPool 并行", TINT_BLUE),
        ("质控层", "verify_pool.py / fix_skipped_pids.py",
         "池化前不变量筛查 + 一次性 skipped-pid 修复", TINT_GRAY),
        ("池化/可视化", "pool_vi_te.py / figures.py / diagrams.py",
         "汇总表 + 符号检验/wilcoxon/BH-FDR + 4 图 + 流程图", TINT_AQUA),
        ("Stage 2", "latent_te.py",
         "GVRNN 潜空间宏观$\leftrightarrow$节点 TE 交叉校验", TINT_AQUA),
        ("报告层", "vi_te_report.py",
         "生成 vi_te_report.md（中文成果报告）", TINT_ORANGE),
    ]
    y0 = 86.0
    h = 10.5
    step = 13.5
    for i, (layer, mod, role, c) in enumerate(layers):
        y = y0 - i * step
        _box(ax, 2, y, 15, h, layer, fc=TINT_GRAY, fs=8.6)
        _box(ax, 19, y, 38, h, mod, fc=c, fs=8.8)
        _box(ax, 59, y, 39, h, role, fc=TINT_GRAY, fs=8.2, tc=INK, subc=INK2)
        _arrow(ax, 17, y + h / 2, 19, y + h / 2)
        _arrow(ax, 57, y + h / 2, 59, y + h / 2)
    fig.text(0.01, 0.02, "估计引擎与代理协议由 vi-te / gvrnn-te 仓库提供"
             "（各自含 pytest 用例）。",
             color=MUTED, fontsize=8.5)
    fig.savefig(out / "fig7_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图8 技术路径
def fig_techpath(out: Path) -> None:
    fig, ax = _new_ax(w=14.0, h=6.5)
    _title(ax, "技术路径：单个半场的 TE 估计流水线")

    # 上排：估计
    y1 = 72.0
    h1 = 16.0
    s1 = _box(ax, 2, y1, 15, h1, "VI 链", "V_t, X_k(t)\n1 Hz", fc=TINT_GRAY)
    s2 = _box(ax, 22, y1, 25, h1, "成对条件 TE",
              "gaussian 闭式 / KSG k∈{3,5} / 离散 K=3\n× downward/upward/cross", fc=TINT_BLUE)
    s3 = _box(ax, 52, y1, 22, h1, "代理检验", "200 次 block=25\n冻结标准化", fc=TINT_BLUE)
    s4 = _box(ax, 79, y1, 19, h1, "eff / p", "eff=raw−mean(surr)\np=(#surr≥raw+1)/201", fc=TINT_AQUA)
    _arrow(ax, 17, y1 + h1 / 2, 22, y1 + h1 / 2)
    _arrow(ax, 47, y1 + h1 / 2, 52, y1 + h1 / 2)
    _arrow(ax, 74, y1 + h1 / 2, 79, y1 + h1 / 2)

    # 中排：聚合
    y2 = 42.0
    h2 = 16.0
    a1 = _box(ax, 2, y2, 30, h2, "聚合（半场级）",
              "D_sum / U_sum / Δ / $\hat{\gamma}_{TE}$ / Ψ", fc=TINT_AQUA)
    a2 = _box(ax, 36, y2, 30, h2, "跨队负对照",
              "Δ* = Σ(D_k − N×cross_D_k)\nN = 该半场出场人数", fc=TINT_ORANGE)
    a3 = _box(ax, 70, y2, 28, h2, "统计判定",
              "符号检验（二项）+ wilcoxon\n+ BH-FDR（描述性）", fc=TINT_GRAY)
    _arrow(ax, 50, y1, 17, y2 + h2)
    _arrow(ax, 55, y1, 51, y2 + h2)
    _arrow(ax, 88, y1, 84, y2 + h2)

    # 下排：结论
    y3 = 12.0
    h3 = 16.0
    _box(ax, 8, y3, 84, h3,
         "头条口径 = gaussian + discrete → 保守口径无净下行涌现",
         "Δ（未校正）35/40 正，但跨队对照后 Δ* 18/40（p=0.64）→ 下行信号被重叠伪影消解；KSG 因退化移出头条。",
         fc=TINT_RED, fs=10.5)
    _arrow(ax, 17, y2, 30, y3 + h3, color=INK2)
    _arrow(ax, 51, y2, 50, y3 + h3, color=INK2)
    _arrow(ax, 84, y2, 70, y3 + h3, color=INK2)
    fig.savefig(out / "fig8_techpath.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 图9 成果汇总
def fig_results(out: Path, st: dict) -> None:
    g = st["halves"]["sign_tests"]
    ctc = st["halves"]["cross_team_control"]["gaussian"]
    degen = st["halves"]["estimator_degeneracy"]["ksg|k=3"]

    def _s(key):
        return g.get(key, {})

    d = _s("gaussian|delta_bits")
    ds = _s("gaussian|delta_star_bits")
    ge = _s("gaussian|gamma_te_nats")
    ps = _s("psi_uncond_nats")

    tiles = [
        ("Δ（gaussian，未校正）", f"{d.get('n_pos')}/{d.get('n')} 半场为正",
         f"p={d.get('binom_p'):.1e}，中位 {d.get('median'):+.3f} bits", TINT_BLUE),
        ("Δ*（跨队校正后）", f"{ds.get('n_pos')}/{ds.get('n')} 半场为正",
         f"p={ds.get('binom_p'):.2f}，中位 {ds.get('median'):+.3f} → 无净下行", TINT_ORANGE),
        ("$\hat{\gamma}_{TE}$（gaussian）", f"{ge.get('n_pos')}/{ge.get('n')} 半场为正",
         f"p={ge.get('binom_p'):.1e}，中位 {ge.get('median'):+.3f} nats", TINT_AQUA),
        ("Ψ_uncond", f"{ps.get('n_pos')}/{ps.get('n')} 半场为正",
         f"中位 {ps.get('median'):+.3f} nats（与 vi 负 Ψ 一致）", TINT_GRAY),
        ("跨队对照（N×缩放）", f"队内 ΣD > 跨队 {ctc.get('n_within_gt')}/{ctc.get('n')}",
         f"wilcoxon p={ctc.get('wilcoxon_p'):.3f}，差 {ctc.get('mean_diff'):+.3f} bits", TINT_ORANGE),
        ("KSG（被排除头条）", "退化：p==1.0 占",
         f"{degen.get('frac_p_eq_1'):.1%}，raw<0 占 {degen.get('frac_raw_lt_0'):.1%}", TINT_RED),
    ]
    fig, ax = _new_ax(w=13.5, h=6.0)
    _title(ax, "成果汇总：TE $\leftrightarrow$ VI 因果涌现桥接（10 场 × 40 半场）")
    xw = 30.0
    xh = 20.0
    gx = 2.0
    gy = 3.5
    for i, (t, big, small, c) in enumerate(tiles):
        col = i % 3
        row = 1 - i // 3
        x = gx + col * (xw + gx)
        y = 38 + row * (xh + gy)
        _box(ax, x, y, xw, xh, t, None, fc=c, fs=9.5)
        ax.text(x + xw / 2, y + xh * 0.46, big, ha="center", va="center",
                fontsize=10.5, color=INK, fontweight="bold")
        ax.text(x + xw / 2, y + xh * 0.18, small, ha="center", va="center",
                fontsize=7.6, color=INK2)
    # 底部核心结论
    _box(ax, 2, 8, 96, 16,
         "核心结论：TE 条件化翻转 Ψ 符号（涌现成立），但下行信号被跨队对照消解",
         "$\hat{\gamma}_{TE}$ 40/40 正、Ψ_uncond 0/40 正 → TE 涌现；Δ* 校正后 18/40（p=0.64）→ 无净下行涌现；KSG 退化排除；BH 零功效。",
         fc=TINT_YELLOW, fs=11)
    fig.savefig(out / "fig9_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_diagrams(out_dir=None, stats=None) -> int:
    out = Path(out_dir) if out_dir else OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    fig_bridge_concept(out)
    fig_dataflow(out)
    fig_architecture(out)
    fig_techpath(out)
    st = stats
    if st is None:
        st = {}
        if STATS.exists():
            st = json.loads(STATS.read_text(encoding="utf-8"))
    if st:
        fig_results(out, st)
    else:
        print("无 stats JSON，跳过 fig9 成果汇总")
    print(f"diagrams -> {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--stats", default=str(STATS),
                    help="vi_te_pool_stats.json（fig9 数字源）")
    args = ap.parse_args()
    st = {}
    if Path(args.stats).exists():
        st = json.loads(Path(args.stats).read_text(encoding="utf-8"))
    sys.exit(make_diagrams(args.out_dir, stats=st))

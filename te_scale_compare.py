"""Cross-family TE magnitude comparison: three TE systems on a common scale.

Families:
  A  phase<->latent       10 Hz, K-means K=8 on 32-dim z, phi 14 categories
                           (results/phase_te/phase_te_summary.csv, this repo)
  B  latent<->latent      10 Hz, same protocol (results/latent_pair_te/)
  C  latent<->latent      1 Hz, pca=1, gaussian + discrete K=3
                           (D:/MarkovSoccerTE_backup/results/m1_sys3.json)
  D  VI team<->player     1 Hz, discrete K=3 / gaussian
                           (D:/MarkovSoccerTE/vi_bridge/results/
                            vi_te_pool_players.csv + _scratch_recheck/
                            worked_vi_anchor.json for H(y'|y))

Common yardsticks:
  TE_eff        = raw - mean(block surrogates)          [bias-corrected bits]
  TE_eff/H(y'|y) = fraction of the target chain's own one-step
                   unpredictability explained by the source (Gourevitch
                   normalized TE)
  bits/s        = TE_eff x frame rate (overstates at 10 Hz: adjacent frames
                   share information; reported for orientation only)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("D:/gvrnn-te")
BACKUP = Path("D:/MarkovSoccerTE_backup/results")
VI = Path("D:/MarkovSoccerTE")


def med(d: pd.DataFrame, col: str) -> float:
    return float(d[col].median())


def main():
    ph = pd.read_csv(REPO / "results/phase_te/phase_te_summary.csv")
    lp = pd.read_csv(REPO / "results/latent_pair_te/latent_pair_te.csv")
    m1 = json.load(open(BACKUP / "m1_sys3.json", encoding="utf-8"))
    vi = pd.read_csv(VI / "vi_bridge/results/vi_te_pool_players.csv")
    anchor = json.load(open(VI / "_scratch_recheck/worked_vi_anchor.json",
                            encoding="utf-8"))

    # ---- cross-check: my U-direction recompute vs pooled CSV row
    hall_u = vi[(vi.match_id == 1886347) & (vi.half == "home|1")
                & (vi.pid == 51667) & (vi.direction == "U")].iloc[0]
    hall_d = vi[(vi.match_id == 1886347) & (vi.half == "home|1")
                & (vi.pid == 51667) & (vi.direction == "D")].iloc[0]
    assert abs(anchor["te_u_raw"] - hall_u["discrete|K=3|raw"]) < 1e-6, (
        anchor["te_u_raw"], hall_u["discrete|K=3|raw"])
    assert abs(anchor["te_d_raw"] - hall_d["discrete|K=3|raw"]) < 1e-6
    print(f"VI anchor cross-check OK: U raw {anchor['te_u_raw']:.9f} == CSV; "
          f"D raw {anchor['te_d_raw']:.9f} == CSV")

    # ---- family B: latent<->latent 10 Hz, K=8
    b8 = lp[lp.k == 8]
    b8_eff_h = (b8["eff"] / b8["h"]).median()

    # ---- family A: phase<->latent 10 Hz, K=8
    a_full = ph[(ph.k == 8) & (ph.scope == "full")]
    a_live = ph[(ph.k == 8) & (ph.scope == "live_only")]
    a_zp = a_full[a_full.direction == "z->phi"]
    a_pz = a_full[a_full.direction == "phi->z"]
    a_zp_l = a_live[a_live.direction == "z->phi"]
    a_pz_l = a_live[a_live.direction == "phi->z"]

    # ---- family C: latent<->latent 1 Hz (m1_sys3, pairs among players 0,1,2)
    cg = [v["gaussian"]["pca=1"] for v in m1["results"].values()]
    cd = [v["discrete"]["pca=1,K=3"] for v in m1["results"].values()
          if "discrete" in v and "pca=1,K=3" in v["discrete"]]
    cg_raw = float(np.median([x["raw"] for x in cg]))
    cg_eff = float(np.median([x["eff"] for x in cg]))
    cd_raw = float(np.median([x["raw"] for x in cd])) if cd else np.nan
    cd_eff = float(np.median([x["eff"] for x in cd])) if cd else np.nan

    # ---- family D: VI team<->player 1 Hz
    vd = vi[vi.direction == "D"]
    vu = vi[vi.direction == "U"]
    rows = []

    def add(family, direction, rate, symbols, n, h, te, eff, note=""):
        rows.append({
            "family": family, "direction": direction, "rate": rate,
            "symbols": symbols, "n": n, "H(y'|y)": round(h, 4) if h == h else None,
            "TE_raw": round(te, 6), "TE_eff": round(eff, 6),
            "eff/H": round(eff / h, 4) if (h and h > 0) else None,
            "bits/s": round(eff * rate, 4) if eff == eff else None,
            "note": note})

    # B: same-protocol player pairs
    add("GVRNN latent player->player", "both (42 pairs)", 10, "K=8",
        int(b8["n"].median()), med(b8, "h"), med(b8, "te"), med(b8, "eff"),
        "median over 42 ordered pairs; same protocol as phase TE")
    # A
    add("GVRNN latent player->phase", "z->phi", 10, "K=8 vs phi(14)",
        int(a_zp["n"].median()), med(a_zp, "h"), med(a_zp, "te"),
        med(a_zp, "eff"), "scope full")
    add("phase->GVRNN latent player", "phi->z", 10, "K=8 vs phi(14)",
        int(a_pz["n"].median()), med(a_pz, "h"), med(a_pz, "te"),
        med(a_pz, "eff"), "scope full")
    add("GVRNN latent player->phase", "z->phi (live)", 10, "K=8 vs phi(13)",
        int(a_zp_l["n"].median()), med(a_zp_l, "h"), med(a_zp_l, "te"),
        med(a_zp_l, "eff"), "scope live_only")
    add("phase->GVRNN latent player", "phi->z (live)", 10, "K=8 vs phi(13)",
        int(a_pz_l["n"].median()), med(a_pz_l, "h"), med(a_pz_l, "te"),
        med(a_pz_l, "eff"), "scope live_only")
    # C
    add("GVRNN latent player->player", "both (6 pairs)", 1, "pca=1 (gauss)",
        m1["input"]["n_transitions"], np.nan, cg_raw, cg_eff,
        "m1_sys3, players 0,1,2; gaussian closed form")
    add("GVRNN latent player->player", "both (6 pairs)", 1, "pca=1,K=3 (disc)",
        m1["input"]["n_transitions"], np.nan, cd_raw, cd_eff,
        "m1_sys3, players 0,1,2; discrete plug-in")
    # D
    add("VI team->player", "D (pool)", 1, "K=3 (disc)",
        int(med(vd, "n_trans")), anchor["h_d"], med(vd, "discrete|K=3|raw"),
        med(vd, "discrete|K=3|eff"),
        "pooled 10 matches; H from 1886347 home|1 Hall chain")
    add("VI player->team", "U (pool)", 1, "K=3 (disc)",
        int(med(vu, "n_trans")), anchor["h_u"], med(vu, "discrete|K=3|raw"),
        med(vu, "discrete|K=3|eff"),
        "pooled 10 matches; H(V'|V) from 1886347 home|1")
    add("VI team->player", "D (pool)", 1, "gaussian",
        int(med(vd, "n_trans")), anchor["h_d_gauss"], med(vd, "gaussian|raw"),
        med(vd, "gaussian|eff"),
        "pooled 10 matches; H_gauss from 1886347 home|1 Hall chain")

    out = REPO / "results/te_scale_compare"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "te_scale_compare.csv", index=False, encoding="utf-8")

    with open(out / "te_scale_report.md", "w", encoding="utf-8") as f:
        f.write("# Three TE families on a common scale\n\n")
        f.write("Yardsticks: `TE_eff` = raw - surrogate mean (bias-corrected "
                "bits); `eff/H` = TE_eff / H(y'|y), the fraction of the target "
                "chain's one-step unpredictability explained by the source; "
                "`bits/s` = TE_eff x frame rate (orientation only: at 10 Hz "
                "adjacent frames share information, so 10x inflates the true "
                "entropy rate).\n\n")
        f.write("| family | direction | rate Hz | symbols | n | H(y'\\|y) bits "
                "| TE_raw | TE_eff | eff/H | bits/s | note |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in df.iterrows():
            hcol = r["H(y'|y)"]
            h = "—" if hcol is None or hcol != hcol else f"{hcol:.4f}"
            eh = "—" if r["eff/H"] is None else f"{r['eff/H']:.4f}"
            bs = "—" if r["bits/s"] is None else f"{r['bits/s']:.4f}"
            f.write(f"| {r['family']} | {r['direction']} | {r['rate']} | "
                    f"{r['symbols']} | {r['n']} | {h} | {r['TE_raw']:.6f} | "
                    f"{r['TE_eff']:.6f} | {eh} | {bs} | {r['note']} |\n")
        f.write("\n## VI anchor (1886347 home|1, D. Hall 51667, 1 Hz, K=3)\n\n")
        f.write(f"- TE(V->Hall) = {anchor['te_d_raw']:.9f} bits, "
                f"H(Hall'|Hall) = {anchor['h_d']:.4f} bits, "
                f"TE/H = {anchor['te_d_raw'] / anchor['h_d']:.4f} "
                f"(eff = {hall_d['discrete|K=3|eff']:.6f} -> "
                f"eff/H = {hall_d['discrete|K=3|eff'] / anchor['h_d']:.4f})\n")
        f.write(f"- TE(Hall->V) = {anchor['te_u_raw']:.9f} bits, "
                f"H(V'|V) = {anchor['h_u']:.4f} bits, "
                f"TE/H = {anchor['te_u_raw'] / anchor['h_u']:.4f} "
                f"(eff = {hall_u['discrete|K=3|eff']:.6f} -> "
                f"eff/H = {hall_u['discrete|K=3|eff'] / anchor['h_u']:.4f})\n")
        f.write(f"- gaussian TE(V->Hall) = {anchor['te_d_gauss']:.9f} bits, "
                f"H_gauss = {anchor['h_d_gauss']:.4f} bits, "
                f"TE/H = {anchor['te_d_gauss'] / anchor['h_d_gauss']:.6f}\n\n")
        f.write("## reading\n\n"
                "- the 10 Hz rows (A, B) share one identical pipeline, so "
                "their bits are directly comparable;\n"
                "- the 1 Hz rows (C, D) are directly comparable with each "
                "other (both discrete K=3 at 1 Hz);\n"
                "- across rates use `eff/H` with caution and `bits/s` only "
                "as orientation.\n")
    print(f"-> {out / 'te_scale_report.md'}")
    print(df[["family", "direction", "rate", "H(y'|y)", "TE_raw", "TE_eff",
              "eff/H", "bits/s"]].to_string(index=False))

    write_zh_report(out, df, ph, vi, anchor, hall_d, hall_u)


# ------------------------------------------------------- Chinese report
PHASE_ZH = {
    "build_up": "组织推进（后场控球向前）",
    "chaotic": "混乱争抢（球权不明）",
    "create": "创造机会（前场渗透）",
    "direct": "直接打法（长传直攻）",
    "finish": "射门终结（攻入禁区）",
    "quick_break": "快速反击",
    "set_play": "定位球",
    "transition": "攻防转换（得球后快速向前）",
    "defending_direct": "防守直接球",
    "defending_quick_break": "防守快攻",
    "defending_set_play": "防守定位球",
    "defending_transition": "防守转换",
    "high_block": "高位逼抢（防线压上）",
    "medium_block": "中位防守",
    "low_block": "低位防守（回收禁区）",
    "dead": "死球（停表/死球间隙，非 SkillCorner 类别，本报告口径）",
}

ZH_ROW = [
    ("球员潜链→球员潜链（同协议）", "双向（42 个有序对）",
     "**基准行**。42 个有序对（7 名全勤外场球员两两互指）的中位数；"
     "与第 2–5 行共用同一条管线（10 Hz、K=8、离散 plug-in、200 次块置换代理），"
     "因此第 1–5 行的 bits 可**直接比较**。"),
    ("球员潜链→阶段链", "z→φ（full 口径）",
     "源=球员潜符号 x_t=z^A_t，目标=阶段链 y_t=φ_t。full=全部帧（死球帧保留为显式 "
     "`dead` 类参与统计）。eff/H 仅 1.2%——单个球员的潜状态对整队阶段切换的预测增量"
     "本就有限。"),
    ("阶段链→球员潜链", "φ→z（full 口径）",
     "方向反转：源=阶段 x_t=φ_t，目标=球员潜链 y_t=z^A_t。eff/H 6.7%，约为同协议"
     "球员↔球员（13.0%）的一半。"),
    ("球员潜链→阶段链", "z→φ（live_only 口径）",
     "live_only=只保留两端帧都被相位行覆盖的转移（剔除死球帧）。eff/H 进一步降到 "
     "0.6%——上一轮发现的 z→φ 信号主要来自死球切换，活球内最弱。"),
    ("阶段链→球员潜链", "φ→z（live_only 口径）",
     "live_only 下反而升到 9.5%——阶段对潜链的预测信息**在活球内部更强**（与联合概率"
     "分析“排除 dead 后 MI 升高”的结论一致）。"),
    ("球员潜链→球员潜链（旧结果）", "双向（6 个有序对，高斯闭式）",
     "m1_sys3.json（1 Hz、stride 10、pca=1、玩家 0,1,2）。旧结果没有记录 "
     "H(y′|y)，故 eff/H 与 bits/s 不适用（—）。eff 为正：1 Hz 下高斯法仍然检出"
     "球员间净信息。"),
    ("球员潜链→球员潜链（旧结果）", "双向（6 个有序对，离散 K=3）",
     "同一批旧结果、离散估计器。eff 为负（−0.0002）——1 Hz + K=3 时样本偏差"
     "（surr_mean≈0.0003）超过信号本身，与族 D 的 VI 离散结果同病。"),
    ("VI 队链→球员链（旧结果）", "D（10 场池化）",
     "vi_te_pool_players.csv 全部 D 行中位数（1 Hz、离散 K=3）。raw 0.0055 看似"
     "不小，但 eff 为负（−0.0013）：**代理校正后无净信息，raw 全是估计器偏差**。"
     "H 分母取自 1886347 home|1 半场 Hall 链锚点（0.9939 bits）。"),
    ("VI 球员链→队链（旧结果）", "U（10 场池化）",
     "同上，方向反转（球员为源）。eff −0.0026，比 D 更负；eff/H −0.17%。"),
    ("VI 队链→球员链（旧结果）", "D（高斯闭式）",
     "同一池化数据、高斯估计器。高斯法偏差小（eff 仍为正），但 eff/H 仅 0.02%——"
     "比阶段链 TE（6.7–9.5%）弱约 300 倍。"),
]


def write_zh_report(out: Path, df: pd.DataFrame, ph: pd.DataFrame,
                    vi: pd.DataFrame, anchor: dict, hall_d, hall_u):
    lines = []
    w = lines.append
    w("# 三族 TE 量级对照报告（中文版）\n")
    w("> 生成脚本：`te_scale_compare.py`（可重跑复现）。所有符号与标识的解释见 "
      "§1 术语表；每一行的备注见 §4 主对照表最后一列。\n")
    w("## 1. 术语表（每个符号的解释）\n")
    w("### 1.1 结果列符号\n")
    w("| 符号 | 含义 | 单位/取值 | 备注 |\n|---|---|---|---|\n"
      "| TE_raw | 传输熵原始估计 | bits（每步） | 样本 plug-in 直估，未做代理校正；"
      "有限样本下自带正偏差 |\n"
      "| surr_mean | 代理均值 | bits | S 次块置换代理 TE 的平均值，即代理零分布中心；"
      "理想情况≈0，偏差大时是“信号”的主要来源 |\n"
      "| **TE_eff** | 代理校正后传输熵 | bits | TE_eff = raw − surr_mean；"
      "**本报告主量级指标**，剔除了估计器偏差 |\n"
      "| p | 置换 p 值 | [0,1] | p = (#{代理 ≥ raw}+1)/(S+1)；S=200 时最小 "
      "0.004975 |\n"
      "| bh_sig | BH-FDR 族校正后显著性 | 布尔 | 对同族全部检验做 Benjamini–"
      "Hochberg 校正（q=0.05）后是否拒绝 |\n"
      "| H(y′\\|y) | 目标链条件熵 | bits | 目标链在自身一步历史条件下仍不可预测的量"
      "——TE 的“分母尺度” |\n"
      "| **eff/H** | 归一化 TE | 比例 | eff/H = TE_eff / H(y′\\|y)：源解释了目标"
      "链一步不可预测性的比例（Gourevitch 归一化 TE），**跨族比较的主尺度** |\n"
      "| bits/s | 每秒信息量 | bits/s | TE_eff × 帧率；10 Hz 下相邻帧共享信息，"
      "×10 会高估真实熵率，仅作方向性参考 |\n"
      "| n | 样本量 | 个 | 进入统计的转移数（一次计数 = 一个转移 t→t+1） |\n"
      "| — | 不适用 | | 该行缺少 H 记录或量无意义 |\n")
    w("### 1.2 链与数据符号\n")
    w("| 符号 | 含义 | 备注 |\n|---|---|---|\n"
      "| $z_t$ | GVRNN 潜变量后验均值 | 每球员 32 维向量，10 Hz，来自已训练模型 "
      "`latents.npz`（**本轮未重新训练**） |\n"
      "| $\\varphi_t$ | 比赛阶段标签 | 来自 SkillCorner `phases_of_play.csv`，"
      "本场实际出现 13 类 + 死球（§7 附录 B） |\n"
      "| $y'$, $y$, $x$ | TE 三元组 | $y'=y_{t+1}$ 目标下一步、$y=y_t$ 目标当前、"
      "$x=x_t$ 源当前 |\n"
      "| K | K-means 聚类数（符号数） | 潜链离散化粒度；K=3 太粗（本场无信号），"
      "K=8 才显现结构 |\n"
      "| s | 符号化后的离散状态 | 每球员逐维 z-score 后 K-means（n_init=10、"
      "seed=42 冻结），按频率降序命名 s0=最大簇 |\n"
      "| dead | 死球类 | 帧未被任何相位行覆盖时归入的显式类别（本报告口径，"
      "非 SkillCorner 类别） |\n"
      "| live_mask | 在场比赛掩码 | 离场球员的冻结潜变量帧不参与计数 |\n"
      "| frame / periods / covered | 帧号 / 半场 / 相位覆盖 | 转移须帧连续"
      "（frame[t+1]=frame[t]+1）且同半场；covered=该帧被相位行覆盖 |\n")
    w("### 1.3 方向标识\n")
    w("| 标识 | 含义 |\n|---|---|\n"
      "| z→φ | TE(z→φ) = I(φ_{t+1}; z^A_t \\| φ_t)：球员潜链为源、阶段链为目标 |\n"
      "| φ→z | TE(φ→z) = I(z^A_{t+1}; φ_t \\| z^A_t)：阶段链为源、球员潜链为目标 |\n"
      "| D | VI 桥接项目记号：D_k = TE(V→X_k)，队链为源、球员为目标的下降支 |\n"
      "| U | VI 桥接项目记号：U_k = TE(X_k→V)，球员为源、队链为目标的上升支 |\n"
      "| both | 双向合并 | 有序对两个方向（A→B 与 B→A）合并取中位数 |\n")
    w("### 1.4 口径标识（scope）\n")
    w("| 标识 | 含义 | 备注 |\n|---|---|---|\n"
      "| full | 全帧口径 | 全部满足掩码的转移；死球帧保留为显式 `dead` 类 |\n"
      "| live_only | 仅活球口径 | 再要求转移两端帧都被相位行覆盖；剔除死球帧，"
      "回答“活球内的结构” |\n"
      "| live_dead | 活/死二元口径 | 相位压成二值（球在比赛中/死球），用于检验"
      "潜状态是否主要编码死球——已知 ≈0.01–0.02 bits，很弱 |\n")
    w("### 1.5 协议与参数标识\n")
    w("| 标识 | 含义 | 备注 |\n|---|---|---|\n"
      "| 10 Hz | 原始帧率 | 每帧 0.1 秒；本轮新算的族 A/B 均为此帧率 |\n"
      "| 1 Hz | 秒级采样 | 旧结果（族 C/D）的口径，stride 10 降采样 |\n"
      "| pca=1 | 主成分降维到 1 维 | 旧 TE 管线对 32 维潜变量先做 PCA（冻结），"
      "本轮的 K-means 直接作用于 32 维 |\n"
      "| block | 块置换块长 | 25 个连续转移为一块整体洗牌，保留链内自相关结构 |\n"
      "| S | 代理次数 | 200 次块置换代理 |\n"
      "| seed | 随机种子 | 42；K-means 与代理置换全部固定，结果可复现 |\n"
      "| 冻结符号化 | 变换只 fit 一次 | K-means/PCA 只在真实链上拟合，代理只重贴"
      "标签，避免变换自身信息泄漏 |\n")
    w("## 2. 三个共同尺度（公式）\n")
    w("$$\n\\mathrm{TE}_{\\mathrm{eff}} = \\mathrm{raw} - "
      "\\frac{1}{S}\\sum_{s=1}^{S}\\mathrm{surr}_s\n$$\n")
    w("$$\n\\frac{\\mathrm{eff}}{H} = "
      "\\frac{\\mathrm{TE}_{\\mathrm{eff}}}{H(y'\\mid y)},\\qquad "
      "H(y'\\mid y) = -\\sum_{y',y}p(y',y)\\log_2\\frac{p(y',y)}{p(y)}\n$$\n")
    w("$$\n\\mathrm{bits/s} = \\mathrm{TE}_{\\mathrm{eff}}\\times\\text{帧率}"
      "\\qquad\\text{（10 Hz 下高估，仅方向性参考）}\n$$\n")
    w("## 3. 三族数据来源\n")
    w("| 族 | 数据 | 文件 | 频率/符号化 |\n|---|---|---|---|\n"
      "| A 潜链↔阶段链 | 本轮新算 | `results/phase_te/phase_te_summary.csv` | "
      "10 Hz、K=3/8 vs φ14 类 |\n"
      "| B 潜链↔潜链（同协议） | 本轮新算 | `results/latent_pair_te/"
      "latent_pair_te.csv` | 10 Hz、K=3/8，与 A 同管线 |\n"
      "| C 潜链↔潜链（旧结果） | 2026 年早期 TE 项目 | "
      "`D:/MarkovSoccerTE_backup/results/m1_sys3.json` | 1 Hz、pca=1、"
      "高斯/离散 K=3 |\n"
      "| D VI 队链↔球员链（旧结果） | VI↔TE 桥接项目 | "
      "`D:/MarkovSoccerTE/vi_bridge/results/vi_te_pool_players.csv`（10 场池化）+ "
      "`_scratch_recheck/worked_vi_anchor.json`（H 锚点） | 1 Hz、离散 K=3 / 高斯 |\n")
    w("**模型说明**：族 A/B/C 共用同一份已训练 GVRNN 模型 npz"
      "（`D:/MarkovSoccerTE_backup/artifacts/m1p1/latents.npz`），本轮**没有重新"
      "训练**；族 D 不使用模型（VI 链由逐秒位置聚类构造）。\n")
    w("## 4. 主对照表（每行附备注）\n")
    w("| 族 | 方向 | Hz | 符号化 | n | H(y′\\|y) bits | TE_raw | TE_eff | "
      "eff/H | bits/s | 备注 |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    for (zh_fam, zh_dir, zh_note), (_, r) in zip(ZH_ROW, df.iterrows()):
        h = r["H(y'|y)"]
        h = "—" if h is None or h != h else f"{h:.4f}"
        eh = "—" if r["eff/H"] is None else f"{r['eff/H'] * 100:.1f}%"
        bs = "—" if r["bits/s"] is None else f"{r['bits/s']:.4f}"
        w(f"| {zh_fam} | {zh_dir} | {r['rate']} | {r['symbols']} | {r['n']} | "
          f"{h} | {r['TE_raw']:.6f} | {r['TE_eff']:.6f} | {eh} | {bs} | "
          f"{zh_note} |")
    assert len(ZH_ROW) == len(df)
    w("")
    w("## 5. 解读\n")
    w("### 5.1 同协议直接比（第 1–5 行，10 Hz、K=8、同一管线）\n")
    w("- **量级排序**：球员↔球员 (13.0%) > 阶段→球员 (6.7–9.5%) > 球员→阶段 "
      "(0.6–1.2%)。阶段链的 TE 与球员间潜链 TE **处于同一量级**，绝不算“小”；"
      "球员→阶段才是真正弱的一环。\n"
      "- **绝对 bits**：φ→z 的 eff（0.0035–0.0047 bits）大于任何旧 1 Hz 估计"
      "（最大 0.0018 bits）。\n"
      "- **活球内更强**：φ→z 的 eff/H 从 full 的 6.7% 升到 live_only 的 9.5%，"
      "而 z→φ 从 1.2% 降到 0.6%——阶段对潜链的预测信息在活球内部，潜链对阶段的"
      "信息则部分来自死球切换。\n")
    w("### 5.2 旧 1 Hz 结果（第 6–10 行）\n")
    w("- 高斯法在 1 Hz 仍检出球员间净信息（eff 0.0018）；离散 K=3 在 1 Hz 被"
      "样本偏差淹没（eff 负）。\n"
      "- **VI 队↔球员（族 D）是三族中最弱的一族**：离散估计 eff 全部为负"
      "（raw 0.0046–0.0055 是纯估计器偏差），高斯法 eff/H 仅 0.02%，比阶段链 "
      "TE 弱约 300 倍。这为“VI 链与球员链的 TE 无净信息”提供了量化证据。\n")
    w("### 5.3 VI 锚点（1886347 home|1，D. Hall 51667，1 Hz，K=3）\n")
    w(f"- TE(V→Hall) = {anchor['te_d_raw']:.9f} bits，H(Hall′\\|Hall) = "
      f"{anchor['h_d']:.4f} bits，TE/H = {anchor['te_d_raw'] / anchor['h_d']:.4f}"
      f"；eff = {hall_d['discrete|K=3|eff']:.6f} → eff/H = "
      f"{hall_d['discrete|K=3|eff'] / anchor['h_d']:.4f}（负）\n"
      f"- TE(Hall→V) = {anchor['te_u_raw']:.9f} bits，H(V′\\|V) = "
      f"{anchor['h_u']:.4f} bits，TE/H = {anchor['te_u_raw'] / anchor['h_u']:.4f}"
      f"；eff = {hall_u['discrete|K=3|eff']:.6f} → eff/H = "
      f"{hall_u['discrete|K=3|eff'] / anchor['h_u']:.4f}（负）\n"
      f"- 高斯 TE(V→Hall) = {anchor['te_d_gauss']:.9f} bits，H_gauss = "
      f"{anchor['h_d_gauss']:.4f} bits，TE/H = "
      f"{anchor['te_d_gauss'] / anchor['h_d_gauss']:.6f}\n"
      f"- U 方向为独立重算，与存档 CSV 逐位一致（校验通过）。\n")
    # ---- appendix A: per-player phase TE (K=8, full)
    w("## 6. 附录 A：每球员阶段 TE（K=8，full 口径，200 代理）\n")
    w("| 球员 pid | z→φ raw | z→φ eff | z→φ p | φ→z raw | φ→z eff | φ→z p |\n"
      "|---|---|---|---|---|---|---|\n")
    sub = ph[(ph.k == 8) & (ph.scope == "full")]
    for pid, grp in sub.groupby("pid", sort=True):
        zp = grp[grp.direction == "z->phi"].iloc[0]
        pz = grp[grp.direction == "phi->z"].iloc[0]
        w(f"| {pid} | {zp['te']:.6f} | {zp['eff']:.6f} | {zp['p']:.3f} | "
          f"{pz['te']:.6f} | {pz['eff']:.6f} | {pz['p']:.3f} |")
    w("\n注：φ 是场级链，8 名球员共享同一目标序列，本表的 p 值不是 8 个独立证据"
      "（见 §8 限制）。\n")
    # ---- appendix B: phase vocabulary
    w("## 7. 附录 B：阶段词汇表（本场实际出现的类别及帧数）\n")
    w("| 阶段标签 | 中文含义 | 帧数 |\n|---|---|---|\n")
    lab = np.load(REPO / "results/latent_phase/phase_label.npz", allow_pickle=True)
    phis, cnts = np.unique(lab["phi"], return_counts=True)
    order = np.argsort(-cnts)
    for i in order:
        name = str(phis[i])
        w(f"| {name} | {PHASE_ZH.get(name, name)} | {int(cnts[i])} |")
    w("")
    w("## 8. 诚实限制\n")
    w("1. TE 一律解读为**模型内的预测信息流**，不做因果断言（两两 TE 的共同驱动"
      "局限，PROTOCOL.md §2/§12）。\n"
      "2. φ 是场级链：族 A 的 8 名球员共享同一目标序列，显著性测试不独立；"
      "8/8 显著 ≠ 8 个独立证据。\n"
      "3. 跨帧率的 eff/H 不是严格不变式；bits/s 在 10 Hz 下高估，只做方向性参考。\n"
      "4. 族 D 的 H 分母取自 1886347 home|1 单半场锚点（池化 CSV 无逐行 H）。\n"
      "5. 族 A/B 的潜链来自同一场比赛的同一模型；是否跨场复制是未验证的开放问题。\n"
      "6. 数据许可：SkillCorner Open Data（MIT © SkillCorner），使用前已注明来源"
      "并通知。\n")
    out_zh = out / "te_scale_report_zh.md"
    out_zh.write_text("\n".join(s.rstrip("\n") for s in lines), encoding="utf-8")
    print(f"-> {out_zh}")


if __name__ == "__main__":
    main()

"""成果报告生成器：读汇总表 + 统计 + 潜空间交叉校验，写 vi_te_report.md。

报告结构（UTF-8，中文）：
  0 摘要（桥接概念 + 主结果一句）
  1 方法与定义（精确归属 + TE 条件化操作定义 + 伪影声明）
  2 锚点校验（vi 项目数值复现）
  3 Ψ 口径对照（池化 vs 逐段/滑窗：有限样本 MI 偏差）
  4 十场涌现画像（头条口径 gaussian/discrete；KSG 单列偏差章节）
  5 球员层（方向、位置组、BH 下限声明、逐球员 D*_k）
  6 潜空间交叉校验（GVRNN 潜链 vs VI 几何，Spearman）
  7 方法学声明与局限
用法：python -m te_emergence.report [--res-dir DIR]
          [--latent latent_te_m1p1.json] [--out vi_te_report.md]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

RES_DIR = TE_ROOT / "results"
OUT = RES_DIR / "vi_te_report.md"
LN2 = float(np.log(2.0))

# 1886347 Auckland 7 名踢满全场球员（node 0..6 顺序 = 定制导出顺序）
M1_FULLMATCH = [51713, 133498, 33697, 51667, 14736, 965685, 23418]

# Rosas 2020 引用
ROSAS = ("Rosas et al. 2020, PLoS Comput Biol 16(12):e1008289, "
         "DOI 10.1371/journal.pcbi.1008289")


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    if 0 < x < 1e-3:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def _sec(txt, level):
    return "\n" + "#" * level + " " + txt + "\n"


def _sign_line(stats_doc, col):
    s = stats_doc["halves"]["sign_tests"].get(col)
    if not s:
        return "（无数据）"
    return (f"n={s['n']} 半场，正值 {s['n_pos']}（{s['frac']:.0%}），"
            f"二项双侧 p={_f(min(1.0, s['binom_p']), 4)}，"
            f"中位 {_f(s['median'])}")


def _load_latent(path) -> dict | None:
    if not path or not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _latent_cross_section(ldoc: dict) -> str:
    """潜空间(节点0..6) vs VI 空间(同批球员) D/U 剖面的 Spearman 对比。"""
    lines = ["\n### 6.1 球员剖面跨表示一致性（Spearman 秩相关，n=7）\n"]
    if ldoc is None:
        return lines[0] + "（latent_te_m1p1.json 缺失）\n"
    nodes = {int(k): v for k, v in ldoc["nodes"].items()}
    # VI 空间：7 球员 D/U eff（从球员层表聚合，两半场取均值）
    vi_d, vi_u = {}, {}
    pt_path = RES_DIR / "vi_te_pool_players.csv"
    if pt_path.exists():
        pt = pd.read_csv(pt_path)
        pt = pt[(pt["match_id"].astype(str) == "1886347")
                & (pt["pid"].isin(M1_FULLMATCH))]
        for pid in M1_FULLMATCH:
            g = pt[pt["pid"] == pid]
            vi_d[pid] = g[g["direction"] == "D"]["gaussian|eff"].mean()
            vi_u[pid] = g[g["direction"] == "U"]["gaussian|eff"].mean()
    # 潜空间：node i -> pid 映射；macro->node 对应 D，node->macro 对应 U
    rows = []
    for i in range(7):
        n = nodes.get(i, {})
        pid = n.get("pid")
        if pid is None or pid not in vi_d:
            continue
        for mk in ("gaussian", "ksg|k=3", "discrete|K=3"):
            dm = n.get("macro->node", {}).get(mk, {})
            um = n.get("node->macro", {}).get(mk, {})
            if "eff" not in dm or "eff" not in um:
                continue
            rows.append(dict(pid=pid, method=mk,
                             vi_d=vi_d[pid], vi_u=vi_u[pid],
                             lat_d=dm["eff"], lat_u=um["eff"]))
    if not rows:
        return lines[0] + "（无可用配对）\n"
    rdf = pd.DataFrame(rows)
    for mk, g in rdf.groupby("method"):
        dd = g.dropna(subset=["vi_d", "lat_d"])
        uu = g.dropna(subset=["vi_u", "lat_u"])
        sd = stats.spearmanr(dd["vi_d"], dd["lat_d"]) if len(dd) >= 3 else None
        su = stats.spearmanr(uu["vi_u"], uu["lat_u"]) if len(uu) >= 3 else None
        lines.append(
            f"- **{mk}**{'（退化，见 §4.2）' if mk.startswith('ksg') else ''}："
            f"D 方向 Spearman ρ="
            f"{_f(sd.statistic, 3) if sd else 'NA'}（n={len(dd)}"
            f"{f'，p={_f(sd.pvalue, 3)}' if sd else ''}）；U 方向 ρ="
            f"{_f(su.statistic, 3) if su else 'NA'}（n={len(uu)}"
            f"{f'，p={_f(su.pvalue, 3)}' if su else ''}）")
    # 球节点锚点
    b = nodes.get(10, {})
    if b:
        lines.append("\n球节点（node 10）方向锚点：")
        for mk in ("gaussian", "ksg|k=3", "discrete|K=3"):
            dm = b.get("macro->node", {}).get(mk, {})
            um = b.get("node->macro", {}).get(mk, {})
            if "eff" in dm and "eff" in um:
                lines.append(f"- {mk}：macro→ball eff={_f(dm['eff'])}，"
                             f"ball→macro eff={_f(um['eff'])}"
                             f"（p={_f(dm.get('p'))} / {_f(um.get('p'))}）")
        lines.append(
            "解读：gaussian 上宏观→球占优（与三层检验 L2a「球员驱动球」"
            "一致）；discrete 两方向均压在分辨率地板 0.005（=1/201），"
            "不判读；ksg 两方向均在地板 0.010（=1/101，100 代理）。")
    return "\n".join(lines) + "\n"


def main() -> int:
    global RES_DIR, OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--res-dir", default=str(RES_DIR),
                    help="pool 输出目录（vi_te_pool_*.csv/json，默认 <repo>/results）")
    ap.add_argument("--latent", default=None,
                    help="latent_te_m1p1.json（默认 <res-dir>/latent_te_m1p1.json）")
    ap.add_argument("--out", default=None,
                    help="报告输出路径（默认 <res-dir>/vi_te_report.md）")
    args = ap.parse_args()

    RES_DIR = Path(args.res_dir)
    OUT = Path(args.out) if args.out else RES_DIR / "vi_te_report.md"
    latent_path = args.latent or str(RES_DIR / "latent_te_m1p1.json")

    ht_path = RES_DIR / "vi_te_pool_halves.csv"
    stats_path = RES_DIR / "vi_te_pool_stats.json"
    if not ht_path.exists() or not stats_path.exists():
        print("需要先运行 te_emergence.pool（vi_te_pool_halves.csv / "
              "vi_te_pool_stats.json）")
        return 1
    ht = pd.read_csv(ht_path)
    st = json.loads(stats_path.read_text(encoding="utf-8"))
    ldoc = _load_latent(latent_path)
    n_matches = len(ht["match_id"].unique())
    n_halves = len(ht)
    g = st["halves"]["sign_tests"]
    head = g.get("_headline_methods", ["gaussian", "discrete|K=3"])
    parts = []

    parts.append("# TE ↔ VI 因果涌现桥接：成果报告\n")
    parts.append(f"生成依据：{n_matches} 场 × {n_halves} 半场（1 Hz VI 链，"
                 f"200 块置换代理，block=25）；含跨队对照与潜空间交叉校验。\n")

    # ------------------------------------------------------------ 0 摘要
    parts.append(_sec("0 摘要", 2))
    em = g.get("gaussian|delta_bits")
    ge = g.get("gaussian|gamma_te_nats")
    parts.append(
        "- **桥接实现**：把 Rosas 下行因果 D_k 用成对转移熵 TE(V→X_k) 估计"
        "（I(V_t;X_k(t+1)|X_k(t))，1 阶），上行 U_k=TE(X_k→V)；"
        "净下行 Δ=ΣD−ΣU（本桥接操作定义，非 Rosas 判据）；"
        "γ̂_TE=I(V;V′)−ΣU 为 TE 条件化的涌现操作定义（避免与 Rosas 的 Γ "
        "冲突）；跨队对照给出重叠伪影的保守基线（D*/U*/Δ*）。\n"
        f"- **头条口径 = gaussian 与 discrete**（KSG 在 tie 主导的 VI 链上"
        f"零分布偏移大、方向不对称，Δ/Γ 是偏差伪影——见 §4.2）。\n"
        f"- **gaussian Δ**：{n_halves} 半场中位 {_f(em['median'] if em else None)} bits"
        f"（正值 {em['n_pos'] if em else 'NA'}/{em['n'] if em else 'NA'}，"
        f"二项 p={_f(em['binom_p'], 3) if em else 'NA'}）——量级与代理"
        f"蒙特卡洛误差同阶，符号检验为诚实的下限判读。\n"
        f"- **γ̂_TE**（gaussian）：中位 {_f(ge['median'] if ge else None)} nats"
        f"（正值 {ge['n_pos'] if ge else 'NA'}/{ge['n'] if ge else 'NA'}，"
        f"p={_f(ge['binom_p'], 3) if ge else 'NA'}）。\n")
    ctc = st["halves"]["cross_team_control"].get("gaussian", {})
    dss = g.get("gaussian|delta_star_bits")
    parts.append(
        f"- **跨队对照（保守口径，N×缩放）**：gaussian 队内 ΣD 显著**低于**"
        f"跨队基线（{ctc.get('n_within_gt')}/{ctc.get('n')} 高于，wilcoxon "
        f"p={_f(ctc.get('wilcoxon_p'), 3)}，中位差 "
        f"{_f(ctc.get('mean_diff'))} bits）→ 未校正 Δ 的正号与重叠伪影"
        f"同量级；校正后 Δ* 中位 {_f(dss.get('median') if dss else None)} bits"
        f"（正 {dss.get('n_pos') if dss else 'NA'}/{dss.get('n') if dss else 'NA'}，"
        f"p={_f(dss.get('binom_p'), 3) if dss else 'NA'}）——**保守口径下"
        f"无净下行涌现**（对照可能过校正，见 §4.1）。\n"
        "- **口径校准发现**：vi 项目 Task-2 Ψ 的逐段均值估计携带 ~0.13 nats "
        "有限样本 MI 偏差（微和侧），池化估计为其一致版本（§3）。\n"
        "- **潜空间交叉校验**：GVRNN 潜链（宏观 PCA-1 ↔ 节点 PCA-1）与 VI 空间 "
        "D/U 剖面的 Spearman 对比见 §6。\n"
        "（项目全景图——桥接概念 / 数据流向 / 模块组件 / 技术路径 / "
        "成果汇总——见 §8。）\n")

    # ------------------------------------------------------------ 1 方法
    parts.append(_sec("1 方法与定义", 2))
    parts.append(
        f"**精确归属**（{ROSAS}）：\n"
        "- 下行因果 **D_t = Σ_i I(V_t; X^i_{t+1} | X^i_t)** 是 Rosas 的"
        "定义，1 阶成对条件下与 TE(V→X_i) 恒等——本桥接的 D_k 即此项的"
        "估计。D>0 是 Rosas 的下行因果判据；**Δ=ΣD−ΣU 是本桥接的净下行"
        "操作定义，Rosas 未定义此判据**。\n"
        "- 严格因果涌现 **Ψ = 冗余 − 协同** 需要 PID 分解。本报告的"
        "**γ̂_TE = I(V;V′) − Σ_k U_k** 是 TE 条件化操作定义；"
        "**Ψ_uncond = I(V;V′) − Σ_k I(X_k;V′)** 与 Rosas 的 **G 函数同形**"
        "（vi 项目 Task-2 公式）——两者**都不是 PID 冗余项**，不声称严格"
        "Rosas Ψ；且 Ψ_uncond 对共享的微观冗余重复计数（数值已证实），"
        "其符号被重复计数主导。\n\n"
        "**估计协议**：gaussian 闭式 / KSG k∈{3,5} / 离散 K=3；"
        "eff = raw − mean(代理)（200 块置换代理，block=25，冻结标准化）；"
        "p=(#{代理≥raw}+1)/(S+1)；surr_ratio=|mean(代理)/raw|，"
        "|ratio|>0.5 旗标（三层检验 L1 教训）。守卫：n_trans≥60、"
        "gauss/ksg≥10、离散≥30。\n\n"
        "**重叠伪影声明**：V_t 由 t−1,t 聚类对决定、X_k(t+1) 由 t,t+1 决定——"
        "两者共享第 t 秒聚类 → lag-1 D/U 系统性虚高。跨队对照"
        "TE(V_opp→X_own,k) 共享同一重叠结构，但**幅度未必相等**、且可能含"
        "对手真实耦合（球员对对手移动有真实反应）——因此"
        "D*_k = D_k − cross_D_k 是**保守对照**（可能过校正），不是精确伪影"
        "移除。报告同时给出未校正 Δ。\n\n"
        "**聚合口径**：V 为全队 VI（含门将贡献）；D_sum/U_sum 为该半场"
        "n_trans≥60 的**全部出场外场球员**之和（人数随换人 10–15 不等，"
        "未上场替补被守卫跳过）；Ψ_uncond 的微观和只含外场。\n")

    # ------------------------------------------------------------ 2 锚点
    parts.append(_sec("2 锚点校验（vi 项目数值复现，1886347）", 2))
    parts.append(
        "- n_valid=4405 ✓、换人 7 次 ✓（1 Hz）。\n"
        "- possession_rate（主场 P1）= 0.606 ✓（口径：明确归属秒内 home "
        "占比，与 vi task1 一致；若以全部 valid 秒为分母则 0.464）。\n"
        "- task1 口径 psi-valid 秒 2314 ✓ 复现；桥接自有口径 "
        "n_seconds/n_valid/n_trans（半场总秒数 2778 等）为另一粒度，"
        "不与之混用。\n"
        "- Task-2 Ψ_VI 锚点复现：run 级 psi_mean −0.1373（home P1，25 "
        "runs）；60 s 窗 −0.0822。池化全半场 −0.0115（§3 解释差异）。\n")

    # ------------------------------------------------------------ 3 Ψ 口径
    parts.append(_sec("3 Ψ 口径对照：有限样本 MI 偏差（方法学发现）", 2))
    parts.append(
        "vi 项目的 psi_series(win=None) 在每条连续 valid 段上分别估计 MI 后取"
        "均值；本桥接用池化（全半场一次估计）。同一批转移对（n=2313，home P1）"
        "的分解（nats）：\n\n"
        "| 分量 | 池化（一致估计） | 逐段均值（vi 口径） | 差 | 解析预期 |\n"
        "|---|---|---|---|---|\n"
        "| i_macro | 0.0110 | 0.0209 | +0.0099 | ~0.5×mean(1/n_run) |\n"
        "| i_micro_sum（10 源和） | 0.0225 | 0.1582 | +0.1357 | 10×同式（段长 min=8 加倍） |\n"
        "| Ψ | −0.0115 | −0.1373 | — | — |\n\n"
        "高斯 MI 的小样本正偏差 ~0.5/n nats/源，跨段取均值时按 **1/n 的"
        "期望**（调和型加权，非 1/mean(n)：段长 min=8 的短段贡献 0.5/8 "
        "主导）——每源实测 +0.0099/+0.0136，与解析预期一致。**vi 报告 "
        "Task-2 的负 Ψ 是被估计器偏差保守化的**（微和被高估 → Ψ 被压低）；"
        "符号结论不变（Ψ<0 或 ≈0），量级应引用池化口径。60 s 滑窗同病"
        "（窗 n=60 每源偏差 ~0.0083）。\n")

    # ------------------------------------------------------------ 4 涌现画像
    parts.append(_sec("4 十场涌现画像（半场级）", 2))
    parts.append("### 4.1 头条口径（gaussian / discrete）\n\n**符号检验**"
                 f"（{n_halves} 半场，二项双侧，未作多重校正——检验数见 §7）：\n")
    for col in ("psi_uncond_nats", "psi_slot_nats"):
        parts.append(f"- {col}：{_sign_line(st, col)}\n")
    for mk in head:
        for col in ("delta_bits", "delta_star_bits", f"{mk}|gamma_te_nats"):
            key = col if col.startswith("psi") else (col if "|" in col
                                                     else f"{mk}|{col}")
            if key in g:
                parts.append(f"- {key}：{_sign_line(st, key)}\n")
        du = g.get(f"{mk}|D_sum>U_sum")
        if du:
            parts.append(f"- {mk} D_sum vs U_sum 配对 wilcoxon p="
                         f"{_f(du['wilcoxon_p'], 3)}（D>U {du['n_d_gt_u']}/{du['n']}）\n")
    parts.append("\n**跨队对照**（队内 ΣD − N×cross_D_mean，N=该半场出场球员数，"
                 "逐半场，保守口径；N×cross_D_mean 与 downward_star_sum_bits 恒等）：\n")
    for mk, c in st["halves"].get("cross_team_control", {}).items():
        deg = "（退化，见 §4.2）" if mk.startswith("ksg") else ""
        parts.append(f"- {mk}{deg}：队内均值 {_f(c['mean_within'])} vs 跨队 "
                     f"{_f(c['mean_cross_n'])}，差 {_f(c['mean_diff'])}，"
                     f"队内>跨队 {c['n_within_gt']}/{c['n']}，"
                     f"wilcoxon p={_f(c['wilcoxon_p'], 3)}\n")
    parts.append("\n![fig1](figures/fig1_emergence_plane.png)\n\n"
                 "*图 1：涌现平面（Δ 与 Γ_TE，40 半场；红圈 = Δ>0 且 Γ_TE>0）。*\n\n"
                 "![fig2](figures/fig2_du_dumbbell.png)\n\n"
                 "*图 2：每半场 D_sum/U_sum 哑铃（按 Δ 排序）。*\n\n"
                 "![fig4](figures/fig4_cross_control.png)\n\n"
                 "*图 4：队内 ΣD 与跨队 N×D_ctrl 负对照。*\n")
    cm = st["halves"].get("cross_macro", {})
    if cm:
        hl = "、".join(cm.get("headline_methods", [])) or "头条方法"
        parts.append(f"\n**跨队宏观 TE**（home↔away，n={cm['n']}，"
                     f"头条方法 {hl} 合并均值）："
                     f"home→away 均值 "
                     f"{_f(cm['mean_by_direction'].get('home->away'))}、"
                     f"away→home {_f(cm['mean_by_direction'].get('away->home'))} bits"
                     f"（n_trans 中位 {_f(cm['n_trans_median'], 0)}）。"
                     f"个别 p 低至分辨率地板 0.005（=1/201），属多重比较噪声，"
                     f"不作队间因果证据。分方法：\n")
        for mk2, bm in cm.get("mean_by_direction_by_method", {}).items():
            deg = "（退化）" if mk2.startswith("ksg") else ""
            parts.append(f"- {mk2}{deg}：home→away {_f(bm.get('home->away'))}，"
                         f"away→home {_f(bm.get('away->home'))} bits\n")
    # KSG 偏差章节
    parts.append("\n### 4.2 KSG 偏差章节（不进入头条）\n")
    degen = st["halves"].get("estimator_degeneracy", {})
    parts.append(
        "条件 KSG（Frenzel-Pompe）在 **tie 主导的 VI 链**上退化：球员 VI 率"
        "在 2313 行中仅 104 个唯一值（tie 比例 0.955），严格 '<' 球计数约定"
        "在该数据上产生强负偏差，且源/条件冗余度不同使偏差**方向不对称**。"
        "后果：\n\n"
        "- 冒烟运行（n_surrogates=30，vi_te_1886347_smoke.json）的"
        "ksg Δ = −3.58 bits、γ̂_TE = 1.94 bits、downward eff −0.55/pair vs "
        "upward −0.19/pair 是偏差伪影而非信息流"
        "（raw 全负、全体 p=1.0，违反 CMI≥0 定理即证明 raw 本身是偏差）；"
        "200 代理全量运行下退化持续（实证统计见下方）。\n"
        "- surr_ratio>0.5 旗标对 KSG 失灵（比值 0.45 < 0.5，因为分子分母"
        "同源同向）——本版对 KSG 一律标注 `p==1.0/raw<0` 退化计数。\n\n"
        "实测退化统计（D 方向，球员×半场）：\n")
    for mk, dg in degen.items():
        parts.append(f"- {mk}：n={dg['n']}，p==1.0 占比 {_f(dg['frac_p_eq_1'], 3)}，"
                     f"raw<0 占比 {_f(dg['frac_raw_lt_0'], 3)}\n")
    parts.append("\n**代理偏差表**（L1 式，全部方法）：\n")
    for mk, b in st["halves"].get("surrogate_bias", {}).items():
        parts.append(f"- {mk}：n={b['n']}，中位 {_f(b['median'])}，"
                     f"|ratio|>0.5 旗标 {b['n_flag']}/{b['n']}"
                     f"（{b['frac_flag']:.0%}）\n")
    parts.append("\n说明：旗标率高主要是 **raw≈0 时比值尺度病态**"
                 "（mean(代理)/|raw| 对 raw 的微小取值敏感），并非代理分布"
                 "异常；gaussian/discrete 的 raw 本身在 1e-3 bits 量级，"
                 "与代理校正同阶——这正是 §0 所说 Δ 量级与代理 MC 误差同阶"
                 "的原因，符号检验与跨队对照才是主判读。\n")

    # ------------------------------------------------------------ 5 球员层
    parts.append(_sec("5 球员层（球员×半场）", 2))
    ps = st["players"]
    parts.append("**D vs U 配对（wilcoxon）与 Spearman**：\n")
    for mk, v in ps.items():
        if mk in ("bh_fdr_downward", "by_position_group"):
            continue
        ds = v.get("d_star")
        deg = "（退化，见 §4.2）" if mk.startswith("ksg") else ""
        parts.append(f"- {mk}{deg}：n={v['n']}，D>U {v['n_d_gt_u']}，"
                     f"wilcoxon p={_f(v['wilcoxon_p'], 3)}，"
                     f"mean D={_f(v['mean_d'])} U={_f(v['mean_u'])}，"
                     f"Spearman(D,U)={_f(v['spearman'], 3)}"
                     + (f"；逐球员 D*_k：均值 {_f(ds['mean'])}，"
                        f"wilcoxon p={_f(ds['wilcoxon_p'], 3)}"
                        f"（n={ds['n']}）" if ds else "")
                     + "\n")
    parts.append("\n**BH-FDR（下行 D 的 p 值）——声明为描述性**：\n"
                 f"- p 分辨率下限 = {st['players']['bh_fdr_downward'].get('p_resolution_floor')}"
                 "（=1/(S+1)），比 BH 工作阈值粗 → **零功效**，"
                 "n_reject 不作推断（验证工作流 CONFIRMED）。\n")
    for mk, v in ps.get("bh_fdr_downward", {}).items():
        if mk in ("p_resolution_floor", "note"):
            continue
        parts.append(f"- {mk}：n_tested={v['n_tested']}，n_reject={v['n_reject']}，"
                     f"median_p={_f(v['median_p'])}\n")
    parts.append("\n**位置组分解**：\n")
    for mk, rows in ps.get("by_position_group", {}).items():
        deg = "（退化，见 §4.2）" if mk.startswith("ksg") else ""
        parts.append(f"- {mk}{deg}：" + "；".join(
            f"{r['position_group']} D={_f(r['mean_d_eff'])} U={_f(r['mean_u_eff'])}"
            f"（n={r['n']}）" for r in rows) + "\n")
    parts.append("\n![fig3](figures/fig3_player_du_scatter.png)\n\n"
                 "*图 3：球员×半场 D vs U 散点（gaussian，位置组着色；"
                 "对角线上方 = 发起方 U>D）。*\n")

    # ------------------------------------------------------------ 6 潜空间
    parts.append(_sec("6 潜空间交叉校验（GVRNN m1p1，1886347）", 2))
    parts.append(
        "宏观 = 节点 0..9 潜均值 PCA-1；微观 = 各节点 PCA-1；"
        "TE(macro↔node) 与 VI 空间 D/U 剖面（7 名踢满全场球员）对比。"
        "注意：n=7 的 Spearman 功效低（双侧 α=0.05 临界 ≈0.71），"
        "结论为一致性探针而非定论。\n")
    parts.append(_latent_cross_section(ldoc))

    # ------------------------------------------------------------ 7 声明
    parts.append(_sec("7 方法学声明与局限", 2))
    n_tests = len(head)
    parts.append(
        "- γ̂_TE/Ψ_uncond 是 TE 条件化操作定义，非严格 Rosas Ψ；Ψ_uncond "
        "与 Rosas G 同形且重复计数共享冗余；D_t 恒等式为精确 Rosas 下行"
        "因果；Δ=ΣD−ΣU 为桥接自定净下行指标。\n"
        "- VI 滑窗重叠伪影由跨队对照给出保守基线（可能过校正，§1）。\n"
        "- 成对 TE（仅目标自身历史条件）是既定约束（MTE 多元路径已终止）。\n"
        "- 半场级符号检验共 ~13+ 项未校正（n=40 功效弱：需 ≥67.5% 同号才"
        "显著）；半场不独立（同队同场）；本版把 KSG 移出头条后剩余 "
        f"{n_tests} 个方法口径（gaussian/discrete），仍以描述性为主。\n"
        "- gaussian 半场级 Δ 量级与代理蒙特卡洛误差同阶 → 涌现平面象限"
        "为定性分类（图 1 已注明）。\n"
        "- 本运行全部对共享代理种子 42（跨对蒙特卡洛相关）；代码已改为"
        "逐对派生种子，后续运行不受影响。\n"
        "- 数据：SkillCorner opendata（20 场）；使用需注明来源并通知"
        "SkillCorner（授权要求）。\n")

    # ------------------------------------------------------------ 8 全景图
    parts.append(_sec("8 项目全景图（数据流 / 模块 / 技术路径 / 成果汇总）", 2))
    parts.append(
        "以下 5 张流程图（figures/fig5..fig9.png）概括本项目的概念映射、"
        "工程实现与结论。\n\n"
        "**图 5 — 桥接概念**（Rosas 因果涌现 ↔ 转移熵 TE 的逐项对应）：\n\n"
        "![fig5](figures/fig5_bridge_concept.png)\n\n"
        "**图 6 — 数据流向**（VI 空间主链与 GVRNN 潜空间两条链汇合到报告）：\n\n"
        "![fig6](figures/fig6_dataflow.png)\n\n"
        "**图 7 — 模块组件**（vi_bridge 包分层）：\n\n"
        "![fig7](figures/fig7_architecture.png)\n\n"
        "**图 8 — 技术路径**（单半场 TE 估计流水线）：\n\n"
        "![fig8](figures/fig8_techpath.png)\n\n"
        "**图 9 — 成果汇总**（头条数字一页速览）：\n\n"
        "![fig9](figures/fig9_results.png)\n")

    OUT.write_text("".join(parts), encoding="utf-8")
    print(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

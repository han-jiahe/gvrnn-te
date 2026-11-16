"""替补信息增益 → Ψ_A(V̇I)（GVRNN计算协议.txt 的换人桥接）。

协议定义：
    S       = { 替补 p : dotTE(p) > dotTE(被替换首发(p)) }（TE 上胜过首发的替补）
    Ψ_A(V̇I) := I(V̇I_t; V̇I_{t'}) − Σ_{i∈S} I(A^i_t; V̇I_{t'})

其中 V̇I 为信息变差率（逐样本 Variation of Information，Pereira et al. 2021，
vi-te 数据文件中记作 VI）；I(V̇I_t; V̇I_{t'}) 与每球员 I(A^i_t; V̇I_{t'}) 直接取自 vi-te 仓库
run_all 产出的 vi_te_<match>.json → halves["home|1"/"home|2"].psi
（i_macro_nats / per_player_mi_nats，1 Hz 全局口径），本模块只做减法，
不复算任何 MI。替补只在比赛末段（下半场）出场 → 上半场 S 交集为空，
Ψ_A(P1) = i_macro 原值；下半场减去 S 中各替补的贡献。

用法：
    python -m te_emergence.sub_gain \
        --vi-json results/vi_te_1886347.json \
        --subs-summary results/subs_te/subs_te_summary.json \
        --out results/subs_te/sub_gain_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_psi(vi_json: Path, team: str = "home"):
    """halves[team|1 / team|2].psi -> dict with i_macro/per_player/psi_uncond."""
    doc = json.loads(Path(vi_json).read_text(encoding="utf-8"))
    out = {}
    for p in ("1", "2"):
        half = doc["halves"][f"{team}|{p}"]
        psi = half["psi"]
        ppm = {int(k): float(v) for k, v in psi["per_player_mi_nats"].items()}
        out[p] = {
            "i_macro_nats": float(psi["i_macro_nats"]),
            "psi_uncond_nats": float(psi["psi_uncond_nats"]),
            "per_player_mi_nats": ppm,
            "n_trans_macro": int(psi.get("n_trans_macro", 0)),
        }
    return out


def psi_a(half: dict, winners: list) -> dict:
    """Ψ_A(V̇I) = i_macro − Σ_{k∈S∩half} I(A^k; V̇I′) for one half."""
    on_pitch = [w for w in winners if w in half["per_player_mi_nats"]]
    absent = [w for w in winners if w not in half["per_player_mi_nats"]]
    sub = {w: half["per_player_mi_nats"][w] for w in on_pitch}
    return {
        "i_macro_nats": half["i_macro_nats"],
        "psi_uncond_nats": half["psi_uncond_nats"],
        "S_on_pitch": on_pitch,
        "S_absent": absent,
        "subtracted": sub,
        "subtracted_sum_nats": sum(sub.values()),
        "psi_a_nats": half["i_macro_nats"] - sum(sub.values()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vi-json", required=True, type=Path)
    ap.add_argument("--subs-summary", required=True, type=Path)
    ap.add_argument("--team", default="home")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    psi = load_psi(args.vi_json, args.team)
    summ = json.loads(Path(args.subs_summary).read_text(encoding="utf-8"))
    winners = [int(e["winner"]) for e in summ["S"]["winners"]]
    pairs = summ["S"]["pairs"]

    res = {p: psi_a(psi[p], winners) for p in ("1", "2")}
    L = []
    L.append("# Ψ_A(V̇I)：替补信息增益与比赛宏观动态的因果涌现（GVRNN 计算协议）\n")
    L.append(f"- VI 数据：`{args.vi_json}`（home|1 / home|2，1 Hz，MI 单位 nats）\n"
             f"- dotTE 数据：`{args.subs_summary}`\n\n")
    L.append("## 1. 协议公式\n\n"
             "$$S = \\left\\{\\ \\text{sub}\\ p : \\dot{TE}(p) > \\dot{TE}(\\text{starter}(p))\\ \\right\\}$$\n\n"
             "$$\\Psi_A(\\dot{VI}) := I(\\dot{VI}_t; \\dot{VI}_{t'}) - \\sum_{i \\in S} I(A^i_t; \\dot{VI}_{t'})$$\n\n"
             "> 术语：宏观变量为**信息变差率** $\\dot{VI}$ （逐样本 Variation of Information，"
             "Pereira et al. 2021）；vi-te 数据文件中记作 `VI`（i_macro / per_player_mi "
             "字段），下文符号注解中的 `I(VI_t; VI_{t'})` 即 $I(\\dot{VI}_t; \\dot{VI}_{t'})$ 。\n\n"
             "- `dotTE(p)`（带权成对转移熵 $\\dot{TE}$ ）：p 对全部共同在场队友的 TE_eff 之和 ÷ 在场跟踪分钟"
             "（见 dot_te_report.md）。\n"
             "- `S`：TE 上胜过所替换首发的**替补**集合。\n"
             "- `I(VI_t; VI_{t'})`：i_macro_nats——全队信息变差率链的一步自预测信息。\n"
             "- `I(A^i_t; VI_{t'})`：per_player_mi_nats——球员 i 对信息变差率下一步的"
             "互信息（vi-te 半场全局口径，不复算）。\n"
             "- 替补只在下半场出场 → 上半场 S 交集为空，Ψ_A(P1) = i_macro 原值。\n\n")
    L.append("## 2. S 判定（dotTE 差值）\n\n")
    L.append("| 替补 | 首发 | dotTE(替补) | dotTE(首发) | 差值 |\n"
             "|---|---|---|---|---|\n")
    for e in pairs:
        L.append(f"| {e['sub']} | {e['starter']} | {e['sub_dot_te']:.6f} | "
                 f"{e['starter_dot_te']:.6f} | {e['diff']:+.6f} |\n")
    L.append(f"\n**S = {winners if winners else '∅（无替补胜出）'}**\n\n")
    L.append("## 3. Ψ_A(V̇I) 计算\n\n")
    L.append("| 半场 | i_macro | S 中在场者 | 减项合计 | **Ψ_A(V̇I)** | "
             "全量 Ψ_uncond（对照） | Δ(Ψ_A − Ψ) |\n"
             "|---|---|---|---|---|---|---|\n")
    for p in ("1", "2"):
        r = res[p]
        L.append(f"| P{p} | {r['i_macro_nats']:.5f} | "
                 f"{r['S_on_pitch'] if r['S_on_pitch'] else '∅'} | "
                 f"{r['subtracted_sum_nats']:.5f} | **{r['psi_a_nats']:.5f}** | "
                 f"{r['psi_uncond_nats']:.5f} | "
                 f"{r['psi_a_nats'] - r['psi_uncond_nats']:+.5f} |\n")
    L.append("\n- 全量 Ψ_uncond = i_macro − Σ_{全体 10 人} I(A^i; VI′)（vi 项目口径）。\n"
             "- Δ = Ψ_A − Ψ_uncond = 被“放行”的非 S 球员贡献之和（恒正）："
             "只减 S 后，涌现值比全量口径高出的部分，即轮换体系内其余球员"
             "（含被替换首发与全勤球员）贡献的规模。\n\n")
    L.append("## 4. 被减项明细\n\n")
    L.append("| 半场 | 替补 pid | I(A^i; VI′) [nats] |\n|---|---|---|\n")
    for p in ("1", "2"):
        for w, v in res[p]["subtracted"].items():
            L.append(f"| P{p} | {w} | {v:.5f} |\n")
    if not any(res[p]["subtracted"] for p in ("1", "2")):
        L.append("| — | — | — |\n")
    L.append("\n## 5. 符号注解\n\n"
             "- `i_macro`（= I(VI_t; VI_{t'})）：全队 VI 自预测互信息，nats/步。\n"
             "- `per_player_mi`（= I(A^i_t; VI_{t'})）：球员 i 与 VI 下一步的"
             "无条件互信息，nats/步，按半场全局计算。\n"
             "- `Ψ_uncond`：全量口径涌现 = i_macro − Σ_{10 人} MI（vi 项目 Task 2）。\n"
             "- `Ψ_A(V̇I)`：协议口径 = i_macro − Σ_{i∈S} MI。\n"
             "- 单位：nats（×1.4427 → bits）。\n")
    L.append("\n## 6. 结果可视化\n\n"
             "（由 `te_emergence/sub_figures.py` 生成，`results/subs_te/figures/`，"
             "与 `te_emergence/figures` 同一设计系统）\n\n"
             "- **图 1 · 涌现平面**：Ψ_A（主口径，蓝）与 Ψ_uncond（全量对照，灰）"
             "对 I(V̇I_t; V̇I_{t'})，P1/P2 两半场，虚线跨度 = Σ_S I(A^i; V̇I′)\n\n"
             "  ![sub1 涌现平面](figures/sub1_emergence_plane.png)\n\n"
             "- **图 4 · Ψ_A 瀑布**：I(V̇I_t; V̇I_{t'}) − Σ_{i∈S} I(A^i_t; V̇I_{t'}) "
             "的减法分解（P1/P2 两面板，nats），虚线 = Ψ_uncond 对照\n\n"
             "  ![sub4 Ψ_A 瀑布](figures/sub4_psi_waterfall.png)\n\n"
             "（四张图的汇总版见 [pipeline_report.md](pipeline_report.md)）\n")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(L), encoding="utf-8")
    print(f"-> {out}")
    print(json.dumps({p: {k: res[p][k] for k in
                          ("i_macro_nats", "subtracted_sum_nats", "psi_a_nats",
                           "psi_uncond_nats", "S_on_pitch")}
                      for p in ("1", "2")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

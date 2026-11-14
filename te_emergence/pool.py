"""te_emergence 10 场汇总：统计表 + 检验 + 图。

读 --res-dir/vi_te_<id>.json（每场全量，由 vi-te 仓库产出），产出：
  <res-dir>/vi_te_pool_halves.csv      40 半场 × 方法 汇总表
  <res-dir>/vi_te_pool_players.csv     球员×半场 逐对表（D/U/cross）
  <res-dir>/vi_te_pool_stats.json      跨半场统计检验（符号/二项/BH-FDR/偏差表）
  <res-dir>/figures/…                  png 图（figures.py / diagrams.py）

统计口径（与三层检验一致）：
  eff = raw − mean(surr)；surr_ratio = mean(surr)/|raw|，|ratio|>0.5 旗标；
  跨 40 半场符号检验（二项，双侧）；BH-FDR 跨球员 p 值；
  跨队负对照 = 重叠伪影基线（队内 − 跨队 = D*/U*）。

依赖：numpy/pandas/scipy/matplotlib（本仓库）；输入 JSON 由
https://github.com/<your-github-username>/vi-te 的 run_all.py 产出。
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
MATCHES = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]
LN2 = float(np.log(2.0))


# ------------------------------------------------------------------ loading
def load_docs(matches=None) -> dict[str, dict]:
    docs = {}
    for mid in (matches or MATCHES):
        p = RES_DIR / f"vi_te_{mid}.json"
        if p.exists():
            docs[mid] = json.loads(p.read_text(encoding="utf-8"))
    return docs


def halves_table(docs: dict) -> pd.DataFrame:
    rows = []
    for mid, d in docs.items():
        for key, h in d["halves"].items():
            base = dict(match_id=mid, half=key, team=h["team"],
                        period=h["period"], side=h["side"],
                        n_valid=h["n_valid"], n_seconds=h["n_seconds"],
                        possession_rate=h["possession_rate"],
                        i_macro_nats=h["psi"]["i_macro_nats"],
                        i_micro_sum_uncond_nats=h["psi"][
                            "i_micro_sum_uncond_nats"],
                        psi_uncond_nats=h["psi"]["psi_uncond_nats"],
                        psi_slot_nats=h["psi"]["psi_slot_nats"],
                        n_trans_macro=h["psi"]["n_trans_macro"])
            for mk, sm in h["methods"].items():
                for col in ("downward_sum_bits", "upward_sum_bits",
                            "delta_bits", "delta_star_bits",
                            "cross_down_mean_bits", "cross_up_mean_bits",
                            "downward_star_sum_bits", "upward_star_sum_bits",
                            "gamma_te_bits", "gamma_te_nats",
                            "n_downward", "n_upward",
                            "n_cross_down", "n_cross_up"):
                    base[f"{mk}|{col}"] = sm.get(col)
                # 从逐对表重算 raw 和（JSON 只存 eff 和；Γ̂_raw 变体需要）
                dr = [r["methods"][mk]["raw"]
                      for r in h["pairs"].get("downward", [])
                      if "raw" in r["methods"].get(mk, {})]
                ur = [r["methods"][mk]["raw"]
                      for r in h["pairs"].get("upward", [])
                      if "raw" in r["methods"].get(mk, {})]
                base[f"{mk}|downward_raw_sum_bits"] = (
                    float(sum(dr)) if dr else np.nan)
                base[f"{mk}|upward_raw_sum_bits"] = (
                    float(sum(ur)) if ur else np.nan)
                # 空 ur 与生成端口径一致（vi_te.py 空列表 sum=0.0）
                base[f"{mk}|gamma_te_raw_bits"] = (
                    base["i_macro_nats"] / LN2 - (float(sum(ur)) if ur else 0.0)
                    if np.isfinite(base["i_macro_nats"]) else np.nan)
            rows.append(base)
    return pd.DataFrame(rows)


def players_table(docs: dict) -> pd.DataFrame:
    """每行 = (match, half, pid, name, position_group, 方法 × D/U/cross 各指标)。

    只保留有估计值的条目（该半场 n_trans≥60 的出场球员）；skipped 行
    （未上场/不足守卫）无估计，不入表。
    """
    rows = []
    for mid, d in docs.items():
        for key, h in d["halves"].items():
            meta = {p["pid"]: p for p in h["players_meta"]}
            for lab, tag in (("downward", "D"), ("upward", "U"),
                             ("cross_down", "CD"), ("cross_up", "CU")):
                for r in h["pairs"][lab]:
                    pid = r["pid"]
                    base = dict(match_id=mid, half=key, team=h["team"],
                                side=h["side"], period=h["period"], pid=pid,
                                name=r["name"], direction=tag,
                                n_trans=r["n_trans"],
                                position_group=meta.get(pid, {}).get(
                                    "position_group", ""))
                    if r.get("skipped"):
                        continue
                    for mk, g in r["methods"].items():
                        if "raw" not in g:
                            continue
                        for col in ("raw", "eff", "p", "surr_mean",
                                    "surr_sd", "surr_ratio"):
                            base[f"{mk}|{col}"] = g[col]
                    rows.append(base)
    return pd.DataFrame(rows)


def bh(pvals, alpha=0.05):
    """BH-FDR（无 NA）。"""
    pv = np.asarray([p for p in pvals if p is not None
                     and np.isfinite(p)], dtype=float)
    if pv.size == 0:
        return 0, np.asarray([])
    order = np.argsort(pv)
    n = pv.size
    ranks = np.arange(1, n + 1)
    thr = alpha * ranks / n
    below = pv[order] <= thr
    k = int(np.max(np.where(below)[0])) + 1 if below.any() else 0
    return k, pv[order][:k]


def binom_two_sided(k, n):
    return float(min(1.0, 2 * min(stats.binom.cdf(k, n, 0.5),
                                  1 - stats.binom.cdf(k - 1, n, 0.5))))


def half_stats(docs: dict) -> dict:
    ht = halves_table(docs)
    out: dict = dict(n_halves=int(len(ht)), matches=sorted(ht["match_id"].unique()))
    if ht.empty:
        return out

    # 1) 涌现指标符号检验（40 半场）
    sign = {}
    for col in ("psi_uncond_nats", "psi_slot_nats"):
        v = ht[col].dropna()
        if v.empty:
            continue
        pos = int((v > 0).sum())
        sign[col] = dict(n=int(len(v)), n_pos=pos, frac=pos / len(v),
                         binom_p=binom_two_sided(pos, len(v)),
                         median=float(v.median()))
    mks = _method_keys(ht)
    for mk in mks:
        # Γ_TE（γ_te，每方法列）
        c = f"{mk}|gamma_te_nats"
        v = ht[c].dropna()
        if not v.empty:
            pos = int((v > 0).sum())
            sign[c] = dict(n=int(len(v)), n_pos=pos, frac=pos / len(v),
                           binom_p=binom_two_sided(pos, len(v)),
                           median=float(v.median()))
        for col in ("delta_bits", "delta_star_bits"):
            c = f"{mk}|{col}"
            v = ht[c].dropna()
            if v.empty:
                continue
            pos = int((v > 0).sum())
            sign[c] = dict(n=int(len(v)), n_pos=pos, frac=pos / len(v),
                           binom_p=binom_two_sided(pos, len(v)),
                           median=float(v.median()))
            # 逐半场 D_sum vs U_sum 配对符号
            d = ht[f"{mk}|downward_sum_bits"]
            u = ht[f"{mk}|upward_sum_bits"]
            both = d.notna() & u.notna()
            if both.sum() >= 6:
                w = float(stats.wilcoxon(d[both] - u[both]).pvalue)
                sign[f"{mk}|D_sum>U_sum"] = dict(
                    n=int(both.sum()), n_d_gt_u=int((d[both] > u[both]).sum()),
                    wilcoxon_p=w)
    # 头条口径 = 非 KSG（KSG 在 tie 主导的 VI 链上零分布偏移大、方向不对称，
    # Δ/Γ 是偏差伪影而非信息流——见验证工作流 CONFIRMED 发现）
    sign["_headline_methods"] = [mk for mk in mks
                                 if not mk.startswith("ksg")]
    sign["_n_method_tests"] = len(mks)
    out["sign_tests"] = sign

    # 2) 代理偏差表（L1 式）
    pt = players_table(docs)
    bias = {}
    for mk in _method_keys(ht):
        r = pt[f"{mk}|surr_ratio"].dropna()
        if r.empty:
            continue
        flagged = int((r.abs() > 0.5).sum())
        bias[mk] = dict(n=int(len(r)), median=float(r.median()),
                        min=float(r.min()), max=float(r.max()),
                        n_flag=flagged, frac_flag=flagged / len(r))
    out["surrogate_bias"] = bias

    # 2b) 估计器退化诊断（验证工作流 CONFIRMED：KSG 在 tie 主导的 VI 链上
    # 全体 p==1.0、raw 全负——Δ/Γ 是偏差伪影；raw<0 违反 CMI≥0 定理）
    degen = {}
    for mk in _method_keys(ht):
        d = pt[pt["direction"] == "D"]
        pv = d[f"{mk}|p"].dropna()
        rv = d[f"{mk}|raw"].dropna()
        degen[mk] = dict(
            n=int(len(pv)),
            frac_p_eq_1=float((pv == 1.0).mean()) if len(pv) else np.nan,
            frac_raw_lt_0=float((rv < 0).mean()) if len(rv) else np.nan)
    out["estimator_degeneracy"] = degen

    # 3) 队内 vs 跨队负对照（重叠伪影）：逐半场 D_sum − N×cross_down_mean
    # N = 该半场 n_trans≥60 的出场球员数（10–15 随换人变化）；
    # N×cross_down_mean 与 JSON 的 downward_star_sum_bits 恒等（≤2e-15）
    ctrl = {}
    for mk in _method_keys(ht):
        d = ht[f"{mk}|downward_sum_bits"]
        cd = ht[f"{mk}|cross_down_mean_bits"]
        nd = ht[f"{mk}|n_downward"]
        both = d.notna() & cd.notna() & nd.notna()
        if both.sum() < 6:
            continue
        within = d[both].to_numpy()
        cross = nd[both].to_numpy() * cd[both].to_numpy()
        diff = within - cross
        w = stats.wilcoxon(diff)
        ctrl[mk] = dict(n=int(both.sum()),
                        n_halves_n_gt_10=int((nd[both] > 10).sum()),
                        mean_within=float(within.mean()),
                        mean_cross_n=float(cross.mean()),
                        mean_diff=float(diff.mean()),
                        n_within_gt=int((diff > 0).sum()),
                        wilcoxon_p=float(w.pvalue))
    out["cross_team_control"] = ctrl

    # 4) 跨队宏观 TE（home->away vs away->home，逐 period 聚合）
    cm_rows = []
    for mid, d in docs.items():
        for per, cm in d.get("cross_macro", {}).items():
            for dr, e in cm.items():
                for mk, g in e["methods"].items():
                    if "raw" in g:
                        g = {k: v for k, v in g.items() if k != "n_trans"}
                        cm_rows.append(dict(match_id=mid, period=per,
                                            direction=dr, method=mk,
                                            n_trans=e["n_trans"], **g))
    cm = pd.DataFrame(cm_rows)
    if not cm.empty:
        # 分方法×方向拆开：混合 4 个量纲悬殊的估计器（含已判退化的 KSG）
        # 的总体均值不可解释，只报头条方法（gaussian/discrete）总体
        headline = [mk for mk in _method_keys(ht)
                    if not mk.startswith("ksg")]
        hl = cm[cm["method"].isin(headline)]
        by_method = {}
        for mk, g in cm.groupby("method"):
            by_method[mk] = {dr: float(g[g["direction"] == dr]["eff"].mean())
                             for dr in ("home->away", "away->home")}
        out["cross_macro"] = dict(
            n=int(len(cm_rows)),
            headline_methods=headline,
            mean_by_direction={
                dr: float(hl[hl["direction"] == dr]["eff"].mean())
                for dr in ("home->away", "away->home")},
            mean_by_direction_by_method=by_method,
            n_trans_median=float(cm["n_trans"].median()))
    return out


def _method_keys(ht: pd.DataFrame):
    """从 halves 表列名提取完整方法键（"ksg|k=3" 等多段键不再被截断）。"""
    seen = []
    for c in ht.columns:
        for suffix in ("|downward_sum_bits", "|upward_sum_bits"):
            if c.endswith(suffix):
                mk = c[: -len(suffix)]
                if mk not in seen:
                    seen.append(mk)
    return seen


def player_stats(docs: dict) -> dict:
    """球员层：D vs U（配对）、方向优势、BH-FDR、位置组分解。"""
    pt = players_table(docs)
    out: dict = {}
    if pt.empty:
        return out
    wide = []
    for (mid, half, pid), g in pt.groupby(["match_id", "half", "pid"]):
        row = dict(match_id=mid, half=half, pid=pid,
                   name=g["name"].iloc[0], team=g["team"].iloc[0],
                   position_group=g["position_group"].iloc[0])
        D = g[g["direction"] == "D"]
        U = g[g["direction"] == "U"]
        CD = g[g["direction"] == "CD"]
        for mk in _method_keys_from_players(pt):
            if f"{mk}|eff" in D.columns:
                dv = D[f"{mk}|eff"].dropna()
                uv = U[f"{mk}|eff"].dropna()
                row[f"{mk}|d_eff"] = dv.iloc[0] if len(dv) else np.nan
                row[f"{mk}|u_eff"] = uv.iloc[0] if len(uv) else np.nan
                cd = CD[f"{mk}|eff"].dropna()
                if np.isfinite(row[f"{mk}|d_eff"]) and len(cd):
                    row[f"{mk}|d_star"] = (row[f"{mk}|d_eff"]
                                           - cd.iloc[0])
                else:
                    row[f"{mk}|d_star"] = np.nan
        wide.append(row)
    wp = pd.DataFrame(wide)
    for mk in _method_keys_from_players(pt):
        d = wp[f"{mk}|d_eff"].dropna()
        u = wp[f"{mk}|u_eff"].dropna()
        both = wp[f"{mk}|d_eff"].notna() & wp[f"{mk}|u_eff"].notna()
        if both.sum() < 6:
            continue
        dv = wp.loc[both, f"{mk}|d_eff"].to_numpy()
        uv = wp.loc[both, f"{mk}|u_eff"].to_numpy()
        w = stats.wilcoxon(dv - uv)
        entry = dict(n=int(both.sum()),
                     n_d_gt_u=int((dv > uv).sum()),
                     wilcoxon_p=float(w.pvalue),
                     mean_d=float(dv.mean()), mean_u=float(uv.mean()),
                     spearman=float(stats.spearmanr(dv, uv).statistic))
        # 逐球员伪影校正 D*_k = D_k − cross_D_k（同球员的跨队对照）
        sv = wp[f"{mk}|d_star"].dropna()
        if len(sv) >= 6:
            entry["d_star"] = dict(
                n=int(len(sv)),
                n_pos=int((sv > 0).sum()),
                mean=float(sv.mean()),
                wilcoxon_p=float(stats.wilcoxon(sv).pvalue))
        out[mk] = entry
    # BH-FDR：下行 D 的 p 值跨全部球员×半场
    # 注意：S=200 → p 分辨率 1/201≈0.005，比 BH 工作阈值粗 → 零功效，
    # 仅作描述性统计（验证工作流 CONFIRMED）
    S = None
    for _mid, doc in docs.items():
        S = doc.get("meta", {}).get("n_surrogates")
        if S:
            break
    out["bh_fdr_downward"] = dict(p_resolution_floor=1.0 / (S + 1)
                                  if S else None,
                                  note="p 分辨率 1/(S+1) 比 BH 阈值粗，"
                                       "n_reject 无推断力（描述性）")
    for mk in _method_keys_from_players(pt):
        pvals = pt.loc[pt["direction"] == "D", f"{mk}|p"]
        k, _ = bh(pvals.dropna().tolist())
        out["bh_fdr_downward"][mk] = dict(
            n_tested=int(pvals.notna().sum()), n_reject=int(k),
            median_p=float(pvals.dropna().median()))
    # 位置组
    out["by_position_group"] = {}
    for mk in _method_keys_from_players(pt):
        rows = []
        for pg, g in wp.groupby("position_group"):
            if pg == "" or g[f"{mk}|d_eff"].notna().sum() < 5:
                continue
            rows.append(dict(position_group=pg, n=int(g[f"{mk}|d_eff"].notna().sum()),
                             mean_d_eff=float(g[f"{mk}|d_eff"].mean()),
                             mean_u_eff=float(g[f"{mk}|u_eff"].mean())))
        if rows:
            out["by_position_group"][mk] = rows
    return out


def _method_keys_from_players(pt: pd.DataFrame):
    seen = []
    for c in pt.columns:
        if "|eff" in c and c.endswith("|eff"):
            mk = c.rsplit("|", 1)[0]
            if mk not in seen:
                seen.append(mk)
    return seen


# ------------------------------------------------------------------ CLI
def main() -> int:
    global RES_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default=None,
                    help="逗号分隔场次覆盖（默认全部已有 JSON）")
    ap.add_argument("--res-dir", default=str(RES_DIR),
                    help="vi_te_<id>.json 所在目录（默认 <repo>/results）")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    RES_DIR = Path(args.res_dir)
    matches = args.matches.split(",") if args.matches else None
    docs = load_docs(matches)
    if not docs:
        print("no vi_te_*.json found")
        return 1
    print(f"loaded {len(docs)} matches")

    ht = halves_table(docs)
    ht.to_csv(RES_DIR / "vi_te_pool_halves.csv", index=False)
    pt = players_table(docs)
    pt.to_csv(RES_DIR / "vi_te_pool_players.csv", index=False)
    stats_doc = dict(halves=half_stats(docs), players=player_stats(docs))
    (RES_DIR / "vi_te_pool_stats.json").write_text(
        json.dumps(stats_doc, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"halves table: {len(ht)} rows -> vi_te_pool_halves.csv")
    print(f"players table: {len(pt)} rows -> vi_te_pool_players.csv")
    print("stats -> vi_te_pool_stats.json")

    if not args.no_figures:
        from te_emergence.figures import make_figures
        make_figures(ht, pt, RES_DIR / "figures")
        from te_emergence.diagrams import make_diagrams
        make_diagrams(RES_DIR / "figures", stats=stats_doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())

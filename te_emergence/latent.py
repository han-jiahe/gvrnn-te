"""第二阶段交叉校验：GVRNN 潜空间的宏观↔微观 TE 与 VI 空间的一致性。

用 pair1 定制的 11 节点模型（--latents，比赛 1886347，Auckland 视角；
节点 0–6 全勤外场、7=May filler、8=Moreno 替补、9=Rogerson 首发、10=球）
构造潜空间"宏观链"：

  macro_t = PCA-1( mean_{i=0..9} z^i_t )      （10 名外场节点潜均值的第 1 主成分，
                                              冻结拟合，1 维粗粒化宏观状态）

再算 TE(macro→z^i) 与 TE(z^i→macro)（各节点 PCA-1 微观链），与 vi 空间的
D_k/U_k 剖面（同一批球员）做 Spearman 秩相关——检验"团队宏观因果签名"是否
跨表示（VI 几何 vs GVRNN 潜动力学）成立。球节点（10）作方向锚点（L2a：
球员/宏观驱动球）。

成本控制（n=43457 帧 @1 Hz，KSG ≈2.8 s/次）：
- ksg 只算 k=3、100 代理（p 分辨率 1/101≈0.01，排序对比足够；代理统计
  照常持久化 + surr_ratio 旗标）；
- gaussian/discrete 保持 200 代理（闭式/符号计数，成本可忽略）；
- 22 个（节点×方向）任务按 --jobs 并行（默认 4），每 worker 自载 npz。

输出 results/latent_te_m1p1.json。

依赖（GitHub 调用）：
- latents.npz 由 https://github.com/<your-github-username>/gvrnn-te 的
  data_loading.py + gvrnn.train/extract 产出（见其 README）；
- TE 估计器与变换来自 https://github.com/<your-github-username>/vi-te
  （pip install git+https://github.com/<your-github-username>/vi-te.git）：
  mte.*（成对条件 TE + 代理协议）与 vi_bridge.vi_te 的 disc_fit/std_fit。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

TE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TE_ROOT))

from mte.significance import effective_te, surrogate_mte       # noqa: E402
from mte.gaussian import conditional_gaussian_mte             # noqa: E402
from mte.ksg import conditional_ksg_mte                       # noqa: E402
from mte.discrete import conditional_discrete_mte             # noqa: E402
from vi_bridge.vi_te import disc_fit, std_fit                 # noqa: E402

# 默认数据位置：本地 artifacts/（由 gvrnn-te 流程产出），可用 --latents 覆盖
LATENTS = TE_ROOT / "artifacts" / "m1p1" / "latents.npz"
OUT_DEFAULT = TE_ROOT / "results" / "latent_te_m1p1.json"

KSEED = 42


def _pca1(ch: np.ndarray):
    """(T, d) -> (T,) 1 维粗粒化（PCA 在输入上拟合，调用方负责冻结/代理）。"""
    pca = PCA(n_components=1, random_state=KSEED)
    return pca.fit_transform(ch)[:, 0], pca


def _closure(method, macro, target, fits_m, fits_t, k):
    """潜空间 TE 闭包：通道无 NaN、无掩码（潜链连续）；变换冻结。"""
    if method == "gaussian":
        mu_m, std_m = fits_m
        mu_t, std_t = fits_t
        Y = (target[1:] - mu_t) / (std_t + 1e-9)
        Z = (target[:-1] - mu_t) / (std_t + 1e-9)
        n = len(Y)

        def fn(m_sh, B, R_list):
            X = (m_sh[:-1] - mu_m) / (std_m + 1e-9)
            return conditional_gaussian_mte(X[:, None], Y, Z)
    elif method == "ksg":
        mu_m, std_m = fits_m
        mu_t, std_t = fits_t
        Y = (target[1:] - mu_t) / (std_t + 1e-9)
        Z = (target[:-1] - mu_t) / (std_t + 1e-9)
        n = len(Y)

        def fn(m_sh, B, R_list):
            X = (m_sh[:-1] - mu_m) / (std_m + 1e-9)
            return conditional_ksg_mte(X[:, None], Y, Z, k=k)
    else:
        km_m, mu_m, std_m = fits_m
        km_t, mu_t, std_t = fits_t
        m_s = (macro - mu_m) / (std_m + 1e-9)
        t_s = (target - mu_t) / (std_t + 1e-9)
        X_sym = km_m.predict(m_s[:, None])[:-1]
        Y_sym = km_t.predict(t_s[:, None])[1:]
        Z_sym = km_t.predict(t_s[:, None])[:-1]
        n = len(X_sym)

        def fn(m_sh, B, R_list):
            m_s = (m_sh - mu_m) / (std_m + 1e-9)
            X = km_m.predict(m_s[:, None])[:-1]
            return conditional_discrete_mte(Y_sym, X, Z_sym[:, None])
    return fn, n


def run_latent_pair(macro, target, method, k, block_len, n_surrogates, seed):
    fits_m = std_fit(macro)
    fits_t = std_fit(target)
    fits_md = disc_fit(macro, 3, seed)
    fits_td = disc_fit(target, 3, seed)
    fn, n = _closure(method, macro, target,
                     fits_md if method == "discrete" else fits_m,
                     fits_td if method == "discrete" else fits_t, k)
    raw = fn(macro, target, [])
    surr = surrogate_mte(fn, macro, target, [], block_len, n_surrogates, seed)
    eff, p = effective_te(raw, surr)
    return dict(n_trans=n, raw=float(raw), eff=float(eff), p=float(p),
                surr_mean=float(surr.mean()), surr_sd=float(surr.std()),
                surr_ratio=(float(surr.mean() / abs(raw))
                            if abs(raw) >= 1e-9 else None))


def _node_task(task):
    """worker：自载 npz，算一个（节点, 方向）的 4 方法 TE。"""
    latents_path, node, direction, block_len, n_surr, ksg_surr, seed = task
    z = np.load(latents_path, allow_pickle=False)
    z_post = z["z_post"]
    micro_mean = z_post[:, :10, :].mean(axis=1)
    macro, _ = _pca1(micro_mean)
    micro, _ = _pca1(z_post[:, node, :])
    src, tgt = (macro, micro) if direction == "macro->node" \
        else (micro, macro)
    entry = {}
    entry["gaussian"] = run_latent_pair(src, tgt, "gaussian", None,
                                        block_len, n_surr, seed)
    entry["ksg|k=3"] = run_latent_pair(src, tgt, "ksg", 3,
                                       block_len, ksg_surr, seed)
    entry["discrete|K=3"] = run_latent_pair(src, tgt, "discrete", None,
                                            block_len, n_surr, seed)
    return node, direction, entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--latents", default=str(LATENTS))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--n-surrogates", type=int, default=200,
                    help="gaussian/discrete 代理数")
    ap.add_argument("--ksg-surrogates", type=int, default=100,
                    help="ksg 代理数（n=43k 时 2.8 s/次，成本控制）")
    ap.add_argument("--block-len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    z = np.load(args.latents, allow_pickle=False)
    z_post = z["z_post"]
    player_ids = z["player_ids"] if "player_ids" in z else None
    T, N, _ = z_post.shape
    print(f"latents {z_post.shape} nodes={N}")

    tasks = [(args.latents, node, direction, args.block_len,
              args.n_surrogates, args.ksg_surrogates, args.seed)
             for node in range(N)
             for direction in ("macro->node", "node->macro")]
    out = dict(meta=dict(source=str(args.latents), n_frames=int(T),
                         n_nodes=int(N),
                         player_ids=player_ids.tolist() if player_ids
                         is not None else None,
                         macro="PCA-1 of mean latent over nodes 0..9",
                         block_len=args.block_len,
                         n_surrogates=args.n_surrogates,
                         ksg_surrogates=args.ksg_surrogates,
                         ksg_k=[3], seed=args.seed, jobs=args.jobs),
               nodes={str(i): dict(
                   node=int(i),
                   pid=int(player_ids[i])
                   if player_ids is not None and i < len(player_ids)
                   else None)
                   for i in range(N)})
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for node, direction, entry in ex.map(_node_task, tasks):
            out["nodes"][str(node)][direction] = entry
            print(f"node {node} {direction} done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    print(f"total {time.time() - t0:.1f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

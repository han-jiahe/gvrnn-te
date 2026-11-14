# TE ↔ VI 因果涌现桥接 —— Mermaid 流程图（文本版）

对应 PNG 版：`D:\MarkovSoccerTE\vi_bridge\results\figures\fig5..fig9.png`。
在支持 Mermaid 的 Markdown 阅读器（VS Code + Mermaid 插件 / GitHub / Typora）中可直接渲染。

## 图 5 · 桥接概念（Rosas 因果涌现 ↔ 转移熵 TE）

```mermaid
flowchart LR
  subgraph ROSAS["Rosas 因果涌现"]
    D1["下行因果 D_t<br/>= Σ_i I(V_t; X^i_{t+1} | X^i_t)"]
    D2["严格涌现 Ψ = 冗余 − 协同<br/>（需 PID 分解）"]
    D3["（无直接对应）"]
  end
  subgraph BRIDGE["桥接映射（成对 TE）"]
    T1["TE(V→X_i) = I(V_t; X_i(t+1) | X_i(t))<br/>1 阶条件 —— 恒等"]
    T2["γ̂_TE = I(V;V′) − Σ_k U_k<br/>（TE 条件化操作定义）"]
    T3["跨队 TE(V_opp→X_own,k)<br/>共享同一滑窗重叠结构"]
  end
  subgraph METRIC["本报告指标"]
    M1["D_k —— 精确 Rosas 下行"]
    M2["γ̂_TE / Ψ_uncond<br/>（非 PID 项，不声称 Ψ）"]
    M3["D*_k = D_k − cross_D_k<br/>（保守对照，可能过校正）"]
  end
  D1 -->|恒等| T1 --> M1
  D2 -->|操作化| T2 --> M2
  D3 -->|伪影对照| T3 --> M3
```

## 图 6 · 数据流向（两条链汇合到成果报告）

```mermaid
flowchart TD
  A["SkillCorner tracking（opendata）"] --> B["vi/data/raw/&lt;id&gt;/tracking_cache_v2.npz<br/>（VI 1 Hz 链，V_t 聚类）"] --> C["vi_te.py<br/>（逐半场 TE：4 方法 × 4 方向）"] --> D["vi_te_&lt;id&gt;.json ×10 场<br/>（200 块置换代理）"] --> E["verify_pool.py → pool_vi_te.py<br/>（不变量筛查 → 池化 + 统计）"] --> F["vi_te_pool_halves/players.csv<br/>+ vi_te_pool_stats.json"]

  G["tracking_pair1.npz（11 节点定制）"] --> H["gvrnn.train / extract<br/>（GVRNN m1p1，1886347）"] --> I["latents.npz<br/>（潜链 PCA-1：宏观 + 节点）"] --> J["latent_te.py<br/>（宏观 ↔ 节点 TE 交叉校验）"] --> K["latent_te_m1p1.json"]

  F --> L["vi_te_report.py"]
  K --> L
  L --> M["vi_te_report.md（+ figures 1–9）"]

  subgraph S2["GVRNN 潜空间（Stage 2）"]
    G
    H
    I
    J
    K
  end
  subgraph S1["VI 空间主链"]
    A
    B
    C
    D
    E
    F
  end
```

## 图 7 · 模块组件（vi_bridge 包分层）

```mermaid
flowchart TD
  subgraph E1["引擎层"]
    M1["vi_te.py —— D_k / U_k / cross / D* / Δ / γ̂_TE / Ψ<br/>（4 方法 × 4 方向，逐半场）"]
  end
  subgraph E2["编排层"]
    M2["run_vi_te.py / run_all.py —— 单场 + 10 场 ProcessPool 并行"]
  end
  subgraph E3["质控层"]
    M3["verify_pool.py / fix_skipped_pids.py<br/>（池化前不变量筛查 + 一次性 pid 修复）"]
  end
  subgraph E4["池化 / 可视化层"]
    M4["pool_vi_te.py / figures.py / diagrams.py<br/>（汇总表 + 符号检验 / wilcoxon / BH-FDR + 9 图）"]
  end
  subgraph E5["Stage 2 层"]
    M5["latent_te.py —— GVRNN 潜空间宏观 ↔ 节点 TE 交叉校验"]
  end
  subgraph E6["报告层"]
    M6["vi_te_report.py —— 生成 vi_te_report.md"]
  end
  M1 --> M2 --> M3 --> M4 --> M6
  M5 --> M6
```

## 图 8 · 技术路径（单半场 TE 估计流水线）

```mermaid
flowchart LR
  S1["VI 链<br/>V_t, X_k(t)（1 Hz）"] --> S2["成对条件 TE<br/>gaussian 闭式 / KSG k∈{3,5} / 离散 K=3<br/>× downward / upward / cross_down / cross_up"] --> S3["代理检验<br/>200 次 block=25<br/>冻结标准化"] --> S4["eff / p<br/>eff = raw − mean(surr)<br/>p = (#surr≥raw + 1) / 201"]

  S4 --> A1["聚合（半场级）<br/>D_sum / U_sum / Δ / γ̂_TE / Ψ"]
  S4 --> A2["跨队负对照<br/>Δ* = Σ(D_k − N×cross_D_k)<br/>N = 该半场出场人数"]
  S4 --> A3["统计判定<br/>符号检验（二项）+ wilcoxon<br/>+ BH-FDR（描述性）"]

  A1 --> C["结论：头条口径 = gaussian + discrete<br/>保守口径无净下行涌现"]
  A2 --> C
  A3 --> C
```

## 图 9 · 成果汇总（10 场 × 40 半场头条数字）

```mermaid
flowchart TD
  T1["Δ（gaussian，未校正）<br/>35/40 半场为正<br/>p = 1.4e-6，中位 +0.021 bits"]
  T2["Δ*（跨队校正后）<br/>18/40 半场为正<br/>p = 0.64，中位 −0.002 → 无净下行"]
  T3["γ̂_TE（gaussian）<br/>40/40 半场为正<br/>p = 1.8e-12，中位 +0.027 nats"]
  T4["Ψ_uncond<br/>0/40 半场为正<br/>中位 −0.029 nats（与 vi 负 Ψ 一致）"]
  T5["跨队对照（N×缩放）<br/>队内 ΣD > 跨队 16/40<br/>wilcoxon p = 0.034"]
  T6["KSG（被排除头条）<br/>退化：p==1.0 占 99.2%<br/>raw<0 占 99.8%"]
  C["核心结论：TE 条件化翻转 Ψ 符号（涌现成立），<br/>但下行信号被跨队对照消解 → 保守口径无净下行涌现"]
  T1 --> C
  T2 --> C
  T3 --> C
  T4 --> C
  T5 --> C
  T6 --> C
```

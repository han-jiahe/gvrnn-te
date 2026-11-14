# gvrnn-te —— GVRNN 潜链提取 · 成对/替补转移熵 · 因果涌现桥接

从 SkillCorner 足球轨迹数据中，用 **GVRNN（Yeh et al., CVPR 2019）** 提取每名
球员的潜动力学链，计算球员间的**成对转移熵**（含替补/被替补球员的 dotTE），
并与 **VI 因果涌现**（Rosas 2020 下行因果框架）桥接为一个完整项目：

```
TE(A → B) = I( z^A_{t-1} ; z^B_t | z^B_{t-1} )        （成对，仅条件于 B 自身历史）
dotTE(p)  = (1 / t_p) · Σ_q TE_eff(p → q)             （对共同在场队友求和 ÷ 在场分钟）
Ψ_A(VI)   = I(VI_t; VI_{t'}) − Σ_{i∈S} I(A^i_t; VI_{t'})   （S = TE 胜过首发的替补）
```

- 方法学协议见 [PROTOCOL.md](PROTOCOL.md)（估计器、代理检验、守卫、§15 dotTE）
- 高斯 TE 闭式推导见 [docs/te_gaussian_explained.md](docs/te_gaussian_explained.md)
- VI 桥接的 10 场成果报告由 `te_emergence/report.py` 生成（`results/vi_te_report.md`）

## 仓库布局

| 路径 | 内容 |
|---|---|
| `gvrnn/` | GVRNN 模型（train / extract / model） |
| `mte/` | 成对 TE 估计器栈（gaussian 闭式 / KSG / 离散 + 代理显著性 + BH-FDR） |
| `te_emergence/` | 因果涌现计算与应用层（pool / figures / diagrams / latent / report / sub_gain） |
| `config.py` | 全部可调参数（潜维度、stride、k、代理数等）的唯一事实源 |
| `data_loading.py` | SkillCorner 轨迹 → GVRNN npz 导出（含 `--player-ids` 定制节点集） |
| `run_player_mte.py` | 球员间成对 TE 全量计算 |
| `run_player_te.py` | 底层 TE 运行器（窗口/掩码/冻结变换/代理） |
| `run_subs_te.py` | 替补/被替补球员 dotTE（GVRNN 计算协议，§15） |
| `latent_phase.py` | 潜状态 × 比赛阶段联合概率（离散计数，含 live/dead 与 live_only 口径） |
| `phase_te.py` | 球员潜链 ↔ 比赛阶段链双向 TE（离散 plug-in + 块置换代理 + BH-FDR） |
| `latent_pair_te.py` | 10 Hz 同协议球员↔球员潜链 TE（量级对照） |
| `te_scale_compare.py` | 三族 TE（潜链↔潜链 / 潜链↔阶段 / VI↔球员）统一尺度对照报告 |
| `tests/` | pytest 用例（全部合成数据，`python -m pytest tests/ -q`） |
| `docs/` | 方法学文档与图 |

## 安装

```bash
git clone https://github.com/han-jiahe/gvrnn-te.git
cd gvrnn-te
pip install -e .
```

依赖：`numpy scipy scikit-learn pandas torch matplotlib`（见
[requirements.txt](requirements.txt)）。计算依赖（VI 桥接）：
[vi-te](https://github.com/han-jiahe/vi-te)（`vi_bridge.vi_te` 提供变换函数与
`mte.*` 估计器，随 `pip install -e .` 自动安装）。

## 快速开始（工作流）

> 数据说明：默认使用 **SkillCorner opendata** 的比赛轨迹
> （`matches/<id>/<id>_match.json` 与 `<id>_tracking_extrapolated.jsonl`）。
> 任何使用须注明来源并通知 SkillCorner，见下方「数据许可」。

```bash
# 0. 数据：下载 SkillCorner opendata（见下）并解压到本地，例如 D:/FootaballDB/SkillCorner/opendata/data

# 1. 导出 GVRNN 训练数据（默认 10 外场球员 + 球；--player-ids 可定制节点集，
#    节点序 = CLI 顺序，球自动末位）
python -m data_loading --data-dir D:/FootaballDB/SkillCorner/opendata/data \
    --match 1886347 --team 4177 --out artifacts/m1/tracking.npz

# 2. 训练 GVRNN（约 6–8 分钟/GPU；val NLL ≈ −10.4 为健康参考值）
python -m gvrnn.train --npz artifacts/m1/tracking.npz \
    --out-dir artifacts/m1/model --device auto

# 3. 提取潜链 latents.npz（z_post (T, K, 32) + player_ids + frame）
python -m gvrnn.extract --npz artifacts/m1/tracking.npz \
    --model artifacts/m1/model/best_model.pt --out artifacts/m1/latents.npz

# 4. 球员间成对 TE（gaussian/ksg/discrete，200 块置换代理）
python -m run_player_mte --latents artifacts/m1/latents.npz \
    --players 0,1,2,3,4,5,6,7,8,9 --out results/m1_te.json

# 5. 替补/被替补 dotTE（需一个含 6 名轮换球员的定制模型，如 m2）
python -m run_subs_te --latents artifacts/m2/latents.npz \
    --live artifacts/m2/tracking_m2.npz \
    --swap-pairs 43829,133501;163972,50951;31147,38673 \
    --k 8 --out-dir results/subs_te

# 6. VI 桥接（输入由 vi-te 的 run_all 产出）→ 汇总/检验/报告/图
python -m te_emergence.pool        # 10 场汇总表 + 统计检验（默认读 results/）
python -m te_emergence.report      # vi_te_report.md 成果报告

# 7. 替补信息增益 → Ψ_A(VI)（GVRNN 计算协议）
python -m te_emergence.sub_gain \
    --vi-json results/vi_te_1886347.json \
    --subs-summary results/subs_te/subs_te_summary.json \
    --out results/subs_te/sub_gain_report.md
```

产出物（`artifacts/`、`results/`）已加入 `.gitignore`，不上传仓库。

## 测试

```bash
python -m pytest tests/ -q     # 全部合成数据，无外部依赖
```

## 数据许可（SkillCorner opendata）

- 数据来源：**[SkillCorner / opendata](https://github.com/SkillCorner/opendata)**（MIT License，© SkillCorner）。
- 本仓库**不包含任何比赛数据**；请自行从 SkillCorner opendata 下载。
- **使用该数据须注明来源**（"Data: SkillCorner Open Data, MIT licensed, © SkillCorner"），
  并在发布时**通知 SkillCorner**（见其仓库说明）。
- 本仓库代码以 MIT 许可发布（见 [LICENSE](LICENSE)），与数据许可相互独立。

## 关联仓库

- [vi-te](https://github.com/han-jiahe/vi-te) —— VI 空间主链提取与 TE 计算（数据口径为
  VI 链；本仓库 `te_emergence` 的计算依赖）

## 引用

- Yeh et al. 2019, "Diverse Generation for Multi-Agent Sports Games", CVPR（GVRNN）
- Schreiber 2000, "Measuring Information Transfer", PRL 85:461（转移熵）
- Kraskov et al. 2004, PRE 69:066138 / Frenzel & Pompe 2007, PRL 99:204101（KSG）
- Rosas et al. 2020, PLoS Comput Biol 16(12):e1008289（因果涌现框架）

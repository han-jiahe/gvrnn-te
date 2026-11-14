"""te_emergence —— 因果涌现的计算与应用层。

把 TE 与 VI 因果涌现桥接起来的分析层（Rosas 2020 下行因果的成对 TE 估计、
跨队负对照、涌现指标聚合与检验、成果报告、全景流程图）。

计算依赖（通过 GitHub 安装，见 README）：
- https://github.com/han-jiahe/vi-te    提供 mte.*（成对条件 TE
  估计器与代理协议）与 vi_bridge.vi_te（变换函数）；
- GVRNN 潜链提取在本仓库（gvrnn/，latent.py 的输入 latents.npz）。

数据不随仓库：使用 SkillCorner opendata（MIT © SkillCorner），
使用需注明来源并通知 SkillCorner（见 README）。

模块：
- pool.py      10 场汇总表 + 统计检验（符号/二项/wilcoxon/BH-FDR/代理偏差）
- figures.py   图 1–4（涌现平面/哑铃/球员散点/跨队对照）
- diagrams.py  图 5–9（桥接概念/数据流向/模块组件/技术路径/成果汇总）
- latent.py    GVRNN 潜空间宏观↔节点 TE 交叉校验（Stage 2）
- report.py    生成 vi_te_report.md 成果报告
- sub_gain.py  替补 dotTE → S 集 → Ψ_A(VI)（GVRNN计算协议 换人桥接）
"""

__version__ = "0.2.0"

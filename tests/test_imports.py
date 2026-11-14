"""te_emergence 导入冒烟（只覆盖不依赖 vi-te 安装的模块）。

figures / diagrams 只依赖 numpy/pandas/matplotlib；pool 的统计函数同样
只依赖 numpy/pandas/scipy——这里做模块导入与纯函数冒烟。latent 需要
pip 安装 vi-te 后才能导入（README 有说明），不在此测试。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np


def test_import_figures():
    import te_emergence.figures as fg
    assert fg.make_figures is not None


def test_import_diagrams():
    import te_emergence.diagrams as dg
    assert dg.fig_bridge_concept is not None
    assert dg.fig_results is not None


def test_import_pool_pure():
    import te_emergence.pool as pl
    # 纯统计函数冒烟（不读数据）
    assert pl.binom_two_sided(40, 40) <= 1.0
    assert pl.bh([0.001, 0.5, 0.9], alpha=0.05)[0] == 1
    assert pl.bh([0.5, 0.9], alpha=0.05)[0] == 0


def test_report_helpers():
    import te_emergence.report as rp
    assert rp._f(0.0001) == "1.00e-04"
    assert rp._f(np.nan) == "NA"
    assert rp._sign_line({"halves": {"sign_tests": {}}}, "missing") == "（无数据）"

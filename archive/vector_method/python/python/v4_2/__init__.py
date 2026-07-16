"""
V4.2 模块化管线 - 拆分 V4.1 单体求解器为 5 个独立模块

模块:
  - StarSelector: 选星器 (Phase 0 不对称选星 + Gaia 密度匹配)
  - VectorMatcher: 向量匹配器 (Phase A PROSAC + Phase B SVD)
  - PairExpander: 匹配对扩增器 (Phase C 线性扫描 NN + 区域均匀化)
  - PairVerifier: 验证器 (Phase D MAD + Phase D' 贝叶斯+三角形)
  - WcsFitter: WCS 拟合器 (Phase E 分层 SIP)
  - V42Pipeline: 管线编排器，串联 5 模块
"""

from .star_selector import StarSelector
from .vector_matcher import VectorMatcher
from .pair_expander import PairExpander
from .pair_verifier import PairVerifier
from .wcs_fitter import WcsFitter
from .pipeline import V42Pipeline

__all__ = ['StarSelector', 'VectorMatcher', 'PairExpander',
           'PairVerifier', 'WcsFitter', 'V42Pipeline']

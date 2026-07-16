"""
V4.4 统一管线 + IRM 闭环迭代精化 + 相对向量法 Phase A

功能:
    V4.3 基础上引入相对向量法 (DMPDV) 完全替代单θ Phase A
    消除平移 t 假设, 在 t≠0 场景 (Type3失败帧) 仍能成功

用途:
    天文底片求解, 输入 FITS 图像 + 中心指向 + 焦距/像元, 输出标准 WCS

模块:
    - vector_match_v4_4_cpp: V44Solver ctypes 封装
"""

from .vector_match_v4_4_cpp import V44Solver, VM44SolveParams

__all__ = ["V44Solver", "VM44SolveParams"]
__version__ = "4.4.0"

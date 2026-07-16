"""
V4.3 统一管线 + IRM 闭环迭代精化

功能:
    将 V4.2 的 5 个独立 DLL 合并为单一 vector_match_v4_3.dll
    引入 IRM 闭环迭代: 扩增 ↔ 验证 ↔ 拟合, 借鉴 ICP 单调收敛

用途:
    天文底片求解, 输入 FITS 图像 + 中心指向 + 焦距/像元, 输出标准 WCS

模块:
    - vector_match_v4_3_cpp: V43Solver ctypes 封装
"""

from .vector_match_v4_3_cpp import V43Solver, VM43SolveParams

__all__ = ["V43Solver", "VM43SolveParams"]
__version__ = "4.3.0"

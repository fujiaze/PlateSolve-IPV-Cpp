"""
4SADQ-KV 盲解析模块 (4-Star Absolute Distance Quad with K-Vector)
功能: 基于k-vector索引+6绝对角距+金字塔选星+Kolomenkin投票的盲解析算法
用途: 仅以像素尺度s0为先验的天文图像盲plate solving
"""

__all__ = [
    "pipeline",
    "io_wrappers",
    "quad_geometry",
    "kvector",
    "quad_selector",
    "matcher",
    "wcs_solver",
    "voting",
    "logging_setup",
]

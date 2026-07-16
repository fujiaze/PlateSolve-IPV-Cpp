"""
ADV-PA 盲解析模块 (Absolute Distance Voting with Position Angle)
功能: 基于二星绝对角距对+k-vector索引+(天区,旋转角)二维投票的盲解析算法
用途: 仅以像素尺度s0为先验的天文图像盲plate solving (Phase 1 区域内验证)
"""

__all__ = [
    "pipeline",
    "io_wrappers",
    "logging_setup",
    "spherical_geom",
    "pair_index",
    "image_features",
    "voting",
    "wcs_verify",
]

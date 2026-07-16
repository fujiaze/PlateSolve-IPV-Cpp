"""
图像星对特征提取模块 (Task 3)
功能: 从检测到的图像星点提取所有二星对的(d_img, θ_img)特征
用途: ADV-PA盲解析的图像侧特征，d_img为旋转不变的绝对角距(SO(3)不变量)
依赖: numpy

特征定义:
    d_img = sqrt((xi-xj)^2 + (yi-yj)^2) * s0  [arcsec, 旋转/平移不变]
    θ_img = atan2(yj-yi, xj-xi)  [度, 图像坐标系方位角, [0,360)]

注意:
    - 饱和星位置仍可用, 包含在配对中(spec: 饱和星不跳过)
    - 图像y轴方向(向下/向上)影响θ_img的绝对值, voting模块同时投票rot和-rot以处理Y翻转
    - 大量星点时C(N,2)爆炸增长, pipeline应限制参与配对的星数(如Top-100最亮)
"""
from __future__ import annotations

import numpy as np

from .logging_setup import get_logger

logger = get_logger(__name__)

_RADTODEG = 180.0 / np.pi


def extract_image_pairs(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    s0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    提取所有C(N,2)图像星对特征 (d_img, θ_img)。

    向量化实现: 用numpy广播计算全差分矩阵, 取上三角(i<j)避免重复。

    Args:
        x_arr: 图像星x坐标(像素), shape (N,)
        y_arr: 图像星y坐标(像素), shape (N,)
        s0: 像素尺度(arcsec/pixel)

    Returns:
        (d_img, theta_img, idx_i, idx_j): 四个数组, shape (C(N,2),)
            d_img: 绝对角距(arcsec)
            theta_img: 图像方位角(度, [0,360))
            idx_i, idx_j: 星对在输入数组中的下标 (i < j)
    """
    x = np.asarray(x_arr, dtype=np.float64)
    y = np.asarray(y_arr, dtype=np.float64)
    n = len(x)
    if n < 2:
        logger.warning("图像星数不足: %d < 2, 无法提取星对", n)
        empty = np.array([], dtype=np.float64)
        empty_i = np.array([], dtype=np.int64)
        return empty, empty, empty_i, empty_i

    # 广播计算全差分矩阵 (N, N)
    dx = x[np.newaxis, :] - x[:, np.newaxis]   # dx[i,j] = x_j - x_i
    dy = y[np.newaxis, :] - y[:, np.newaxis]   # dy[i,j] = y_j - y_i

    # 取上三角 (i < j), 排除对角线
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dx_flat = dx[triu_mask]
    dy_flat = dy[triu_mask]

    # 计算d_img和theta_img
    pixel_dist = np.sqrt(dx_flat ** 2 + dy_flat ** 2)
    d_img = pixel_dist * float(s0)  # arcsec
    theta_img = np.mod(np.arctan2(dy_flat, dx_flat) * _RADTODEG, 360.0)  # [0, 360)

    # 生成i, j下标对
    idx_i, idx_j = np.where(triu_mask)
    idx_i = idx_i.astype(np.int64)
    idx_j = idx_j.astype(np.int64)

    logger.info("图像星对特征提取: %d颗星 → %d星对 (s0=%.4f\"/px)",
                 n, len(d_img), s0)
    return d_img, theta_img, idx_i, idx_j

"""
球面几何计算模块 (Task 2)
功能: 计算球面角距 d_cat (arcsec) 和位置角 PA_cat (度, 从北向东)
用途: ADV-PA盲解析中构建星表星对特征 — (d_cat, PA_cat) 为二星对的核心几何不变量
依赖: numpy

公式:
    角距 (大圆距离):
        cos_c = sin(dec1)*sin(dec2) + cos(dec1)*cos(dec2)*cos(ra2-ra1)
        d_rad = arccos(clip(cos_c, -1, 1))
        d_arcsec = d_rad * 180/pi * 3600

    位置角 (从北向东):
        PA = atan2(sin(dra)*cos(dec2), cos(dec1)*sin(dec2) - sin(dec1)*cos(dec2)*cos(dra))
        PA_deg = (PA * 180/pi) % 360
        其中 dra = (ra2-ra1) * pi/180 (弧度)

向量化: 全部用numpy广播，支持批量星对计算
"""
from __future__ import annotations

import numpy as np

from .logging_setup import get_logger

logger = get_logger(__name__)

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi
_RADTOASEC = (180.0 / np.pi) * 3600.0


def angular_distance_arcsec(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
) -> np.ndarray:
    """
    计算两组(RA,Dec)之间的大圆角距(arcsec)。

    使用球面余弦定理: cos(c) = sin(d1)sin(d2) + cos(d1)cos(d2)cos(dra)
    对小角度和大角度均精确(arccos在0和pi处有数值梯度问题，但clip后可接受)。

    Args:
        ra1, dec1, ra2, dec2: 角度(度), shape可广播

    Returns:
        角距(arcsec), 同输入shape
    """
    ra1r = np.asarray(ra1, dtype=np.float64) * _DEGTORAD
    dec1r = np.asarray(dec1, dtype=np.float64) * _DEGTORAD
    ra2r = np.asarray(ra2, dtype=np.float64) * _DEGTORAD
    dec2r = np.asarray(dec2, dtype=np.float64) * _DEGTORAD

    sin_d1 = np.sin(dec1r)
    cos_d1 = np.cos(dec1r)
    sin_d2 = np.sin(dec2r)
    cos_d2 = np.cos(dec2r)
    dra = ra2r - ra1r
    cos_c = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(dra)
    cos_c = np.clip(cos_c, -1.0, 1.0)
    d_rad = np.arccos(cos_c)
    return d_rad * _RADTOASEC


def position_angle_deg(
    ra1: np.ndarray,
    dec1: np.ndarray,
    ra2: np.ndarray,
    dec2: np.ndarray,
) -> np.ndarray:
    """
    计算从星1到星2的位置角(度, 从北向东, [0, 360))。

    PA = atan2(sin(dra)*cos(dec2), cos(dec1)*sin(dec2) - sin(dec1)*cos(dec2)*cos(dra))

    Args:
        ra1, dec1: 星1坐标(度)
        ra2, dec2: 星2坐标(度)

    Returns:
        PA(度), shape同输入广播结果, 范围[0, 360)
    """
    ra1r = np.asarray(ra1, dtype=np.float64) * _DEGTORAD
    dec1r = np.asarray(dec1, dtype=np.float64) * _DEGTORAD
    ra2r = np.asarray(ra2, dtype=np.float64) * _DEGTORAD
    dec2r = np.asarray(dec2, dtype=np.float64) * _DEGTORAD

    dra = ra2r - ra1r
    sin_dra = np.sin(dra)
    cos_dra = np.cos(dra)
    sin_d1 = np.sin(dec1r)
    cos_d1 = np.cos(dec1r)
    sin_d2 = np.sin(dec2r)
    cos_d2 = np.cos(dec2r)

    y = sin_dra * cos_d2
    x = cos_d1 * sin_d2 - sin_d1 * cos_d2 * cos_dra
    pa_rad = np.arctan2(y, x)
    pa_deg = pa_rad * _RADTODEG
    pa_deg = np.mod(pa_deg, 360.0)  # 映射到 [0, 360)
    return pa_deg

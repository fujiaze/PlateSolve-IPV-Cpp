"""
WCS求解器模块 (Task 6)
功能: 从匹配四边形(4图像像素↔4参考切平面arcsec)用Umeyama SVD求解(s,θ,tx,ty), 导出CD/CRVAL/CRPIX
用途: 4SADQ-KV盲解析的最终WCS输出, 计算投影残差RMS验证匹配正确性
依赖: numpy, lib.plate_solve.python.vector_match_v2 (gnomonic_inverse)

Umeyama SVD:
    H = P_c^T @ Q_c, SVD(H)=U S Vt, R=V@U^T(含反射检查), s=ΣS/var(P), t=Q_mean - s·R·P_mean
    tangent_arcsec = s·R·pixel + t

WCS导出:
    A = s·R (arcsec/pixel), CD = A/3600 (deg/pixel)
    CRPIX = -A^{-1}·t + 1 (FITS 1-indexed)
    CRVAL = (ra0, dec0) 切平面中心
    Y翻转: 图像y向下, Dec向上, Umeyama自然捕获(不手动翻转)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_setup import get_logger
from .quad_geometry import Quad, ReferenceQuad
from .matcher import MatchCandidate

# 复用V3.5的gnomonic投影
from lib.plate_solve.python.vector_match_v2 import gnomonic_inverse

logger = get_logger(__name__)

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi
_RADTOASEC = (180.0 / np.pi) * 3600.0
_ASECTORAD = np.pi / (180.0 * 3600.0)


@dataclass
class WCSResult:
    """
    WCS求解结果。

    Attributes:
        cd: 2x2 CD矩阵 (deg/pixel)
        crpix1: FITS CRPIX1 (1-indexed)
        crpix2: FITS CRPIX2 (1-indexed)
        crval1: CRVAL1 = ra0 (度)
        crval2: CRVAL2 = dec0 (度)
        s: Umeyama尺度
        R: 2x2 旋转矩阵
        t: 平移向量 (arcsec)
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)
        rms_arcsec: 投影残差RMS(arcsec)
        n_points: 用于求解的点数
    """
    cd: np.ndarray
    crpix1: float
    crpix2: float
    crval1: float
    crval2: float
    s: float
    R: np.ndarray
    t: np.ndarray
    ra0: float
    dec0: float
    rms_arcsec: float
    n_points: int


def umeyama_svd(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Umeyama SVD求解: Q ≈ s·R·P + t

    Args:
        P: (N, 2) 源点 (图像像素)
        Q: (N, 2) 目标点 (切平面arcsec)

    Returns:
        (R, s, t): R=(2,2), s=标量, t=(2,) 使 Q ≈ s·R·P + t
    """
    N = P.shape[0]
    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_c = P - P_mean   # (N, 2)
    Q_c = Q - Q_mean   # (N, 2)

    # H = P_c^T @ Q_c (2x2), 不除N (N因子在尺度公式中抵消)
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)

    # R = V @ U^T (不强制det=+1, 允许反射)
    # 原因: 图像y向下, Dec向上, Y翻转本质是反射(det=-1), 必须允许
    # 标准Umeyama的反射检查(D=diag(1,d))会阻止Y翻转捕获, 此处不使用
    R = Vt.T @ U.T   # (2, 2)

    # 尺度: s = ΣS / Σ||P_c||² (H未除N, var_P也不除N, 因子抵消)
    var_P_total = float((P_c ** 2).sum())
    if var_P_total < 1e-30:
        s = 1.0
    else:
        s = float(S.sum() / var_P_total)

    # 平移
    t = Q_mean - s * (R @ P_mean)   # (2,)

    return R, s, t


def solve_wcs(
    image_quad: Quad,
    ref_quad: ReferenceQuad,
    ra0: float,
    dec0: float,
) -> Optional[WCSResult]:
    """
    从匹配四边形求解WCS。

    Args:
        image_quad: 图像四边形 (points=像素坐标)
        ref_quad: 参考四边形 (points=切平面arcsec, ra/dec=度)
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)

    Returns:
        WCSResult 或 None(求解失败)
    """
    P = image_quad.points.astype(np.float64)    # (4, 2) 像素
    Q = ref_quad.points.astype(np.float64)      # (4, 2) 切平面arcsec

    if P.shape[0] < 3 or Q.shape[0] < 3:
        logger.warning("匹配点数不足: P=%d, Q=%d", P.shape[0], Q.shape[0])
        return None

    # Umeyama SVD
    R, s, t = umeyama_svd(P, Q)

    # A = s·R (arcsec/pixel)
    A = s * R   # (2, 2)
    # CD矩阵 (deg/pixel) = A / 3600
    cd = A / 3600.0

    # CRPIX: 求切平面=0处的像素坐标 0 = A·pixel + t → pixel = -A^{-1}·t
    det_A = float(np.linalg.det(A))
    if abs(det_A) < 1e-20:
        logger.warning("A矩阵奇异, 无法求CRPIX")
        return None
    A_inv = np.linalg.inv(A)
    crpix_pixel = -A_inv @ t   # (2,) [x, y]
    crpix1 = float(crpix_pixel[0] + 1.0)   # FITS 1-indexed
    crpix2 = float(crpix_pixel[1] + 1.0)

    # CRVAL = 切平面中心
    crval1 = float(ra0)
    crval2 = float(dec0)

    # RMS: 投影图像像素→切平面→RA/Dec, 与参考星RA/Dec比较
    rms = compute_rms(P, R, s, t, ra0, dec0, ref_quad.ra, ref_quad.dec)

    logger.info("WCS求解: s=%.5f, CRPIX=(%.2f, %.2f), CRVAL=(%.5f, %.5f), RMS=%.3f\"",
                 s, crpix1, crpix2, crval1, crval2, rms)

    return WCSResult(
        cd=cd,
        crpix1=crpix1,
        crpix2=crpix2,
        crval1=crval1,
        crval2=crval2,
        s=s,
        R=R,
        t=t,
        ra0=crval1,
        dec0=crval2,
        rms_arcsec=rms,
        n_points=P.shape[0],
    )


def solve_wcs_from_candidate(
    candidate: MatchCandidate,
    ra0: float,
    dec0: float,
) -> Optional[WCSResult]:
    """
    从MatchCandidate求解WCS的便捷封装。

    Args:
        candidate: 匹配候选
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)

    Returns:
        WCSResult 或 None
    """
    return solve_wcs(candidate.image_quad, candidate.ref_quad, ra0, dec0)


def compute_rms(
    P: np.ndarray,
    R: np.ndarray,
    s: float,
    t: np.ndarray,
    ra0: float,
    dec0: float,
    ref_ra: np.ndarray,
    ref_dec: np.ndarray,
) -> float:
    """
    计算投影残差RMS。

    流程: 图像像素 P → 切平面arcsec (s·R·P + t) → gnomonic_inverse → RA/Dec → 与参考星比较

    Args:
        P: (N, 2) 图像像素
        R: (2, 2) 旋转矩阵
        s: 尺度
        t: (2,) 平移(arcsec)
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)
        ref_ra: (N,) 参考星RA(度)
        ref_dec: (N,) 参考星Dec(度)

    Returns:
        RMS (arcsec)
    """
    # 图像像素 → 切平面arcsec
    tangent = (s * (R @ P.T)).T + t  # (N, 2) [xi, eta] arcsec
    xi = tangent[:, 0]
    eta = tangent[:, 1]

    # 切平面 → RA/Dec
    ra_pred, dec_pred = gnomonic_inverse(xi, eta, ra0, dec0)

    # 角距离(arcsec)
    sep = angular_separation_arcsec(ra_pred, dec_pred, ref_ra, ref_dec)

    rms = float(np.sqrt(np.mean(sep ** 2)))
    return rms


def angular_separation_arcsec(
    ra1: np.ndarray, dec1: np.ndarray,
    ra2: np.ndarray, dec2: np.ndarray,
) -> np.ndarray:
    """
    计算两组(RA,Dec)之间的角距离(arcsec), haversine公式。

    Args:
        ra1, dec1, ra2, dec2: (N,) 度

    Returns:
        (N,) 角距离(arcsec)
    """
    ra1r = np.asarray(ra1) * _DEGTORAD
    dec1r = np.asarray(dec1) * _DEGTORAD
    ra2r = np.asarray(ra2) * _DEGTORAD
    dec2r = np.asarray(dec2) * _DEGTORAD

    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1r) * np.cos(dec2r) * np.sin(dra / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    sep_rad = 2.0 * np.arcsin(np.sqrt(a))
    return sep_rad * _RADTOASEC


def apply_wcs(
    pixel_xy: np.ndarray,
    wcs: WCSResult,
) -> tuple[np.ndarray, np.ndarray]:
    """
    用WCS将图像像素坐标投影到天球RA/Dec。

    Args:
        pixel_xy: (N, 2) 图像像素坐标
        wcs: WCS结果

    Returns:
        (ra, dec): (N,) 度
    """
    P = np.asarray(pixel_xy, dtype=np.float64)
    tangent = (wcs.s * (wcs.R @ P.T)).T + wcs.t  # (N, 2) arcsec
    xi = tangent[:, 0]
    eta = tangent[:, 1]
    ra, dec = gnomonic_inverse(xi, eta, wcs.ra0, wcs.dec0)
    return ra, dec

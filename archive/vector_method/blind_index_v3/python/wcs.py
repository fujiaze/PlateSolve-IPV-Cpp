"""
WCS 构建与内点验证 (Task 6)
功能: 从 (θ, dx, dy, s₀) 构建 WCS (CD/CRVAL/CRPIX), KD-tree 内点验证
用途: DD-SPPS 阶段 4, 频域求解结果→天球坐标
依赖: numpy, scipy.spatial.cKDTree, vector_match_v2.gnomonic_inverse

算法 (Convention 1 = 标准 FITS WCS, CD 无 cos(δ) 因子):
    - CD 矩阵: s0/3600 × 旋转矩阵 (像素→切平面 xi,eta 度)
    - flip_mode: 对 CD 行取负体现翻转 (mode1→CD1_* 负, mode2→CD2_* 负, mode3→全负)
    - CRVAL: 用 gnomonic_inverse 精确反推切点 (TAN 投影, 非简单仿射)
    - 验证: 像素→(xi,eta)→gnomonic_inverse→(ra,dec), cKDTree 最近邻, haversine RMS
    - 容差: max(3×σ_pos×s0, 5×s0) 角秒 (兼顾质心+网格离散误差)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from lib.plate_solve.blind_index_v2.python.logging_setup import get_logger
from lib.plate_solve.python.vector_match_v2 import gnomonic_forward, gnomonic_inverse

logger = get_logger("ddspps.wcs")

_DEGTORAD = math.pi / 180.0
_RADTOASEC = (180.0 / math.pi) * 3600.0
_ASECTORAD = math.pi / (180.0 * 3600.0)


@dataclass
class WCSResult:
    """
    WCS 求解结果 (网格坐标系, 0-indexed)。

    Attributes:
        crval1, crval2: 切点/参考点天球坐标 (度)
        cd11, cd12, cd21, cd22: CD 矩阵元素 (度/原始像素)
        crpix1, crpix2: 参考像素 (网格坐标, 0-indexed, 默认 grid/2)
        grid: 信号网格尺寸
        flip_mode: 翻转模式 0/1/2/3
        theta_deg: 旋转角 (度)
        scale_factor: 网格→原始像素缩放因子 (image_w/grid)
    """
    crval1: float
    crval2: float
    cd11: float
    cd12: float
    cd21: float
    cd22: float
    crpix1: float
    crpix2: float
    grid: int
    flip_mode: int
    theta_deg: float
    scale_factor: float = 1.0


@dataclass
class VerifyResult:
    """
    WCS 验证结果。

    Attributes:
        n_inliers: 内点数 (距离 < 容差的匹配数)
        rms_arcsec: 内点 haversine 球面距离 RMS (角秒)
        matched_pairs: list of (img_idx, gaia_idx, sep_arcsec)
    """
    n_inliers: int
    rms_arcsec: float
    matched_pairs: List[Tuple[int, int, float]] = field(default_factory=list)


def build_wcs(
    theta_deg: float,
    dx_sub: float,
    dy_sub: float,
    s0: float,
    ra_c: float,
    dec_c: float,
    grid: int,
    flip_mode: int,
    scale_factor: float = 1.0,
) -> WCSResult:
    """
    从频域求解结果 (θ, dx, dy) 构建 WCS (Convention 1: 标准 FITS WCS)。

    CD 矩阵 (度/原始像素, 像素→切平面 xi/eta):
        CD1_1 =  s0/3600 × cos(θ)        (无 cos(δ) 因子, TAN 投影自处理球面收缩)
        CD1_2 = -s0/3600 × sin(θ)
        CD2_1 =  s0/3600 × sin(θ)
        CD2_2 =  s0/3600 × cos(θ)
    flip_mode 通过对 CD 行取负体现:
        mode 1 (x翻转): CD1_1, CD1_2 取负
        mode 2 (y翻转): CD2_1, CD2_2 取负
        mode 3 (双翻转): 都取负

    CRVAL 用 gnomonic_inverse 精确反推:
        指向中心 (ra_c, dec_c) 位于网格 (CRPIX+dx, CRPIX+dy)
        → (xi_c, eta_c) = (dx_eff×CD1_1+dy_eff×CD1_2, dx_eff×CD2_1+dy_eff×CD2_2) 度
        → (CRVAL1, CRVAL2) = gnomonic_inverse(-xi_c, -eta_c, ra_c, dec_c)

    Args:
        theta_deg: 旋转角 (度)
        dx_sub: x 方向平移 (网格像素, 来自相位相关)
        dy_sub: y 方向平移 (网格像素)
        s0: 像素尺度 (arcsec/原始像素)
        ra_c: 指向中心 RA (度)
        dec_c: 指向中心 Dec (度)
        grid: 信号网格尺寸
        flip_mode: 翻转模式 0/1/2/3
        scale_factor: 网格→原始像素缩放因子 (image_w/grid), 默认 1.0

    Returns:
        WCSResult
    """
    theta_rad = theta_deg * _DEGTORAD
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    s_deg = s0 / 3600.0  # 度/原始像素
    # 基础 CD (无翻转, 无 cos_dec — Convention 1)
    cd11 = s_deg * cos_t
    cd12 = -s_deg * sin_t
    cd21 = s_deg * sin_t
    cd22 = s_deg * cos_t

    # flip_mode 调整 (行取负)
    if flip_mode & 1:  # x 翻转 → CD 第一行取负
        cd11 = -cd11
        cd12 = -cd12
    if flip_mode & 2:  # y 翻转 → CD 第二行取负
        cd21 = -cd21
        cd22 = -cd22

    # 网格位移 → 原始像素位移
    dx_eff = dx_sub * scale_factor
    dy_eff = dy_sub * scale_factor

    # CRVAL: 指向中心 (ra_c, dec_c) 位于像素 (CRPIX+dx_eff, CRPIX+dy_eff)
    # CD: 像素偏移 → 切平面 (xi, eta) 度
    xi_c_deg = dx_eff * cd11 + dy_eff * cd12
    eta_c_deg = dx_eff * cd21 + dy_eff * cd22
    # TAN 反投影: (CRVAL1, CRVAL2) = gnomonic_inverse(-xi_c, -eta_c, ra_c, dec_c)
    # (负号: 切点相对于指向中心的偏移 = -指向中心相对于切点的偏移)
    xi_c_asec = -xi_c_deg * 3600.0
    eta_c_asec = -eta_c_deg * 3600.0
    crval1, crval2 = gnomonic_inverse(np.array([xi_c_asec]), np.array([eta_c_asec]), ra_c, dec_c)
    crval1 = float(crval1[0])
    crval2 = float(crval2[0])

    crpix1 = grid / 2.0
    crpix2 = grid / 2.0

    logger.info("WCS 构建: θ=%.3f°, dx=%.3f, dy=%.3f (scale=%.3f), s0=%.4f\", flip=%d, "
                "CRVAL=(%.6f, %.6f), CD=[[%.3e,%.3e],[%.3e,%.3e]]",
                theta_deg, dx_sub, dy_sub, scale_factor, s0, flip_mode,
                crval1, crval2, cd11, cd12, cd21, cd22)

    return WCSResult(
        crval1=crval1, crval2=crval2,
        cd11=cd11, cd12=cd12, cd21=cd21, cd22=cd22,
        crpix1=crpix1, crpix2=crpix2,
        grid=grid, flip_mode=flip_mode, theta_deg=theta_deg,
        scale_factor=scale_factor,
    )


def _haversine_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg) -> np.ndarray:
    """
    haversine 球面角距离 (角秒), 向量化。

    Args:
        ra1, dec1, ra2, dec2: 度, 标量或数组

    Returns:
        分离角 (角秒)
    """
    ra1 = np.asarray(ra1_deg) * _DEGTORAD
    dec1 = np.asarray(dec1_deg) * _DEGTORAD
    ra2 = np.asarray(ra2_deg) * _DEGTORAD
    dec2 = np.asarray(dec2_deg) * _DEGTORAD
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(dra / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    sep = 2.0 * np.arcsin(np.sqrt(a)) * _RADTOASEC
    return sep


def verify_wcs(
    wcs: WCSResult,
    stars_x: np.ndarray,
    stars_y: np.ndarray,
    gaia_ra: np.ndarray,
    gaia_dec: np.ndarray,
    s0: float,
    sigma_pos: float = 1.5,
    image_w: Optional[float] = None,
    image_h: Optional[float] = None,
) -> VerifyResult:
    """
    KD-tree 内点验证: 将图像星点投影到天球, 与 Gaia 星表匹配。

    像素→天球 (Convention 1: 标准 FITS WCS, TAN 投影):
        1. 图像像素 → 网格坐标 (各向同性缩放 + 居中, 与 build_image_signal 一致)
        2. 网格坐标 → 原始像素偏移: (dx_pix, dy_pix) = (x_grid-crpix1, y_grid-crpix2) × scale_factor
        3. 原始像素偏移 → 切平面 (xi, eta) 度: xi = dx_pix×CD11 + dy_pix×CD12
        4. 切平面 → 天球 (ra, dec): gnomonic_inverse(xi, eta, CRVAL1, CRVAL2)
    匹配: cKDTree 最近邻
    容差: max(3×σ_pos×s0, 5×s0) 角秒 (σ_pos=1.5 兼顾质心+网格离散)

    Args:
        wcs: WCSResult
        stars_x, stars_y: 图像星点坐标 (原始像素)
        gaia_ra, gaia_dec: Gaia 参考星 (度)
        s0: 像素尺度 (arcsec/pixel)
        sigma_pos: 位置噪声 (像素, 默认 1.5 = 0.5 质心 + 1.0 网格离散)
        image_w, image_h: 原图像宽高 (像素, 用于坐标缩放与居中)

    Returns:
        VerifyResult (n_inliers, rms_arcsec, matched_pairs)
    """
    stars_x = np.asarray(stars_x, dtype=np.float64)
    stars_y = np.asarray(stars_y, dtype=np.float64)
    gaia_ra = np.asarray(gaia_ra, dtype=np.float64)
    gaia_dec = np.asarray(gaia_dec, dtype=np.float64)

    n_img = len(stars_x)
    n_gaia = len(gaia_ra)
    if n_img == 0 or n_gaia == 0:
        logger.warning("WCS 验证: 空输入 (img=%d, gaia=%d)", n_img, n_gaia)
        return VerifyResult(n_inliers=0, rms_arcsec=float("inf"), matched_pairs=[])

    grid = wcs.grid
    # 图像像素 → 网格坐标 (各向同性缩放 + 居中, 与 build_image_signal 一致)
    if image_w is not None and image_w > 0 and image_h is not None and image_h > 0:
        image_size = max(float(image_w), float(image_h))
        scale = float(grid) / image_size
        # 居中偏移: 让图像中心 (w/2, h/2) 落在 (grid/2, grid/2)
        offset_x = (grid - image_w * scale) / 2.0
        offset_y = (grid - image_h * scale) / 2.0
        x_grid = stars_x * scale + offset_x
        y_grid = stars_y * scale + offset_y
    else:
        x_grid = stars_x
        y_grid = stars_y

    # 网格坐标 → 原始像素偏移 (×scale_factor)
    dx_pix = (x_grid - wcs.crpix1) * wcs.scale_factor
    dy_pix = (y_grid - wcs.crpix2) * wcs.scale_factor

    # 原始像素偏移 → 切平面 (xi, eta) 度 (CD 矩阵)
    xi_deg = dx_pix * wcs.cd11 + dy_pix * wcs.cd12
    eta_deg = dx_pix * wcs.cd21 + dy_pix * wcs.cd22

    # 切平面 → 天球 (TAN 反投影, gnomonic_inverse)
    xi_asec = xi_deg * 3600.0
    eta_asec = eta_deg * 3600.0
    ra_pred, dec_pred = gnomonic_inverse(xi_asec, eta_asec, wcs.crval1, wcs.crval2)

    # cKDTree 最近邻匹配 (Gaia 星表)
    gaia_pts = np.column_stack([gaia_ra, gaia_dec])
    tree = cKDTree(gaia_pts)
    pred_pts = np.column_stack([ra_pred, dec_pred])
    dists_deg, idxs = tree.query(pred_pts, k=1)

    # haversine 精确距离 (角秒)
    sep_arcsec = _haversine_arcsec(
        ra_pred, dec_pred, gaia_ra[idxs], gaia_dec[idxs]
    )

    # 容差: max(3×σ_pos×s0, 5×s0) 角秒 (兼顾质心+网格离散)
    tol_arcsec = max(3.0 * sigma_pos * s0, 5.0 * s0)
    tol_deg = tol_arcsec / 3600.0

    inlier_mask = dists_deg < tol_deg
    n_inliers = int(np.sum(inlier_mask))

    matched_pairs: List[Tuple[int, int, float]] = []
    if n_inliers > 0:
        inlier_idx = np.where(inlier_mask)[0]
        for i_local, i_img in enumerate(inlier_idx):
            gaia_idx = int(idxs[i_img])
            sep = float(sep_arcsec[i_img])
            matched_pairs.append((int(i_img), gaia_idx, sep))
        rms = float(np.sqrt(np.mean(sep_arcsec[inlier_mask] ** 2)))
    else:
        rms = float("inf")

    logger.info("WCS 验证: n_img=%d, n_gaia=%d, n_inliers=%d, RMS=%.3f\", 容差=%.3f\" (max(3×%.1f×s0, 5×s0))",
                n_img, n_gaia, n_inliers, rms, tol_arcsec, sigma_pos)
    return VerifyResult(n_inliers=n_inliers, rms_arcsec=rms, matched_pairs=matched_pairs)


def refine_crval(
    wcs: WCSResult,
    matched_pairs: List[Tuple[int, int, float]],
    stars_x: np.ndarray,
    stars_y: np.ndarray,
    gaia_ra: np.ndarray,
    gaia_dec: np.ndarray,
    image_w: float,
    image_h: float,
) -> WCSResult:
    """
    用 KD-tree 匹配对的中位残差精化 CRVAL。

    phase_correlate_2d 的 dx/dy 可能包含信号质心偏移 (图像信号与 Gaia 信号星点分布差异),
    导致 CRVAL 偏离真实指向。但匹配对 (在 KD-tree 容差内) 的残差直接反映此偏移。
    用匹配对在切平面坐标系的中位残差修正 CRVAL, 不改变 CD 矩阵和 CRPIX。

    流程:
        1. 每个匹配对: 图像星 → 切平面 (xi_pred, eta_pred) via CD/CRPIX
                       Gaia 星 → 切平面 (xi_gaia, eta_gaia) via gnomonic_forward(CRVAL)
        2. 残差: d_xi = xi_gaia - xi_pred, d_eta = eta_gaia - eta_pred (arcsec)
        3. 中位残差 → CRVAL 修正: CRVAL' = gnomonic_inverse(median_d_xi, median_d_eta, CRVAL, CRVAL2)
        (用中位数而非均值, 抗异常点)

    Args:
        wcs: 初始 WCSResult
        matched_pairs: verify_wcs 返回的匹配对 [(img_idx, gaia_idx, sep_arcsec), ...]
        stars_x, stars_y: 图像星点坐标 (原始像素)
        gaia_ra, gaia_dec: Gaia 星 (度)
        image_w, image_h: 原图像宽高

    Returns:
        精化后的 WCSResult (CRVAL 修正, CD/CRPIX 不变)
    """
    if len(matched_pairs) < 3:
        logger.warning("CRVAL 精化: 匹配对不足 (%d < 3), 跳过", len(matched_pairs))
        return wcs

    grid = wcs.grid
    image_size = max(float(image_w), float(image_h))
    scale = float(grid) / image_size
    offset_x = (grid - image_w * scale) / 2.0
    offset_y = (grid - image_h * scale) / 2.0

    # 收集所有匹配对的切平面残差
    d_xi_list = []
    d_eta_list = []
    for img_idx, gaia_idx, _ in matched_pairs:
        # 图像星 → 网格坐标 → 切平面 (xi_pred, eta_pred) arcsec
        x_grid = stars_x[img_idx] * scale + offset_x
        y_grid = stars_y[img_idx] * scale + offset_y
        dx_pix = (x_grid - wcs.crpix1) * wcs.scale_factor
        dy_pix = (y_grid - wcs.crpix2) * wcs.scale_factor
        xi_pred_deg = dx_pix * wcs.cd11 + dy_pix * wcs.cd12
        eta_pred_deg = dx_pix * wcs.cd21 + dy_pix * wcs.cd22
        xi_pred_asec = xi_pred_deg * 3600.0
        eta_pred_asec = eta_pred_deg * 3600.0

        # Gaia 星 → 切平面 (xi_gaia, eta_gaia) arcsec (相对 CRVAL)
        xi_gaia, eta_gaia, _ = gnomonic_forward(
            np.array([gaia_ra[gaia_idx]]),
            np.array([gaia_dec[gaia_idx]]),
            wcs.crval1, wcs.crval2,
        )
        xi_gaia_asec = float(xi_gaia[0])
        eta_gaia_asec = float(eta_gaia[0])

        d_xi_list.append(xi_gaia_asec - xi_pred_asec)
        d_eta_list.append(eta_gaia_asec - eta_pred_asec)

    d_xi_list = np.array(d_xi_list)
    d_eta_list = np.array(d_eta_list)
    # 用中位数 (抗异常点), 并用 MAD 估计散布
    med_d_xi = float(np.median(d_xi_list))
    med_d_eta = float(np.median(d_eta_list))
    mad_d_xi = float(np.median(np.abs(d_xi_list - med_d_xi)))
    mad_d_eta = float(np.median(np.abs(d_eta_list - med_d_eta)))

    # CRVAL 修正: 在切平面偏移 (d_xi, d_eta) 处的反投影
    crval1_new, crval2_new = gnomonic_inverse(
        np.array([med_d_xi]), np.array([med_d_eta]),
        wcs.crval1, wcs.crval2,
    )
    crval1_new = float(crval1_new[0])
    crval2_new = float(crval2_new[0])

    logger.info("CRVAL 精化: n_pairs=%d, 中位残差=(%.2f\", %.2f\"), MAD=(%.2f\", %.2f\"), "
                "CRVAL (%.6f,%.6f) → (%.6f,%.6f)",
                len(matched_pairs), med_d_xi, med_d_eta, mad_d_xi, mad_d_eta,
                wcs.crval1, wcs.crval2, crval1_new, crval2_new)

    # 返回新 WCS (CD/CRPIX 不变, 仅 CRVAL 修正)
    return WCSResult(
        crval1=crval1_new, crval2=crval2_new,
        cd11=wcs.cd11, cd12=wcs.cd12, cd21=wcs.cd21, cd22=wcs.cd22,
        crpix1=wcs.crpix1, crpix2=wcs.crpix2,
        grid=wcs.grid, flip_mode=wcs.flip_mode, theta_deg=wcs.theta_deg,
        scale_factor=wcs.scale_factor,
    )


def check_self_consistency(
    wcs: WCSResult,
    s0: float,
    dec_c: float,
) -> Tuple[float, bool]:
    """
    WCS 自洽性检查: 从 CD 行列式反推像素尺度, 与输入 s0 比较。

    Convention 1 (无 cos_dec): det(CD) = (s0/3600)²  →  s0 = 3600 × sqrt(|det(CD)|)
    (flip_mode 对 CD 行取负, |det| 不变; dec_c 参数保留兼容但不再使用)

    Args:
        wcs: WCSResult
        s0: 输入像素尺度 (arcsec/pixel)
        dec_c: 指向中心 Dec (度, 保留兼容, Convention 1 不再使用)

    Returns:
        (s_out, consistent): s_out 为反推像素尺度, consistent 为 |s_out-s0|/s0 < 1%
    """
    det_cd = wcs.cd11 * wcs.cd22 - wcs.cd12 * wcs.cd21
    if det_cd <= 0:
        logger.warning("自洽性检查: det(CD)=%.3e (异常, 应为正)", det_cd)
        return 0.0, False
    s_out = 3600.0 * math.sqrt(abs(det_cd))
    rel_err = abs(s_out - s0) / s0 if s0 > 0 else float("inf")
    consistent = rel_err < 0.01
    logger.info("自洽性检查: s0=%.4f\", s_out=%.4f\", 相对误差=%.4f%%, %s",
                s0, s_out, rel_err * 100.0, "通过" if consistent else "不通过")
    return s_out, consistent


def refine_wcs_2d(
    wcs: WCSResult,
    matched_pairs: List[Tuple[int, int, float]],
    stars_x: np.ndarray,
    stars_y: np.ndarray,
    gaia_ra: np.ndarray,
    gaia_dec: np.ndarray,
    image_w: float,
    image_h: float,
    s0: float,
) -> WCSResult:
    """
    用匹配对做 Umeyama 2D 拟合精化 WCS (θ, s, tx, ty)。

    解决 1D 相位相关角度精度不足 (0.5° 步长) 导致边缘星点超出容差的问题。
    在切平面坐标系做 Umeyama 2D, 同时修正旋转角/尺度/平移。

    流程:
        1. 对每个匹配对:
           - 图像星 → 网格坐标 → 原始像素偏移 → 切平面 (xi_pred, eta_pred) arcsec via CD
           - Gaia 星 → 切平面 (xi_gaia, eta_gaia) arcsec via gnomonic_forward(CRVAL)
        2. Umeyama 2D: 求 (s, R, t) 使 s×R×src + t ≈ dst
           - src = [(xi_pred, eta_pred), ...] (图像切平面)
           - dst = [(xi_gaia, eta_gaia), ...] (Gaia 切平面)
        3. 修正 WCS:
           - CD_new = s × R × CD (角度 + 尺度修正)
           - CRVAL_new = gnomonic_inverse(t_xi, t_eta, CRVAL) (平移修正)
           - CRPIX 不变

    Args:
        wcs: 初始 WCSResult
        matched_pairs: verify_wcs 返回的匹配对
        stars_x, stars_y: 图像星点坐标 (原始像素)
        gaia_ra, gaia_dec: Gaia 星 (度)
        image_w, image_h: 原图像宽高
        s0: 原始像素尺度 (arcsec/pixel, 用于约束 s)

    Returns:
        精化后的 WCSResult (CD/CRVAL 修正, CRPIX 不变)
    """
    if len(matched_pairs) < 5:
        logger.warning("WCS 2D 精化: 匹配对不足 (%d < 5), 跳过", len(matched_pairs))
        return wcs

    grid = wcs.grid
    image_size = max(float(image_w), float(image_h))
    scale = float(grid) / image_size
    offset_x = (grid - image_w * scale) / 2.0
    offset_y = (grid - image_h * scale) / 2.0

    # 收集匹配对的切平面坐标 (arcsec)
    src_pts = []  # 图像切平面 (xi_pred, eta_pred)
    dst_pts = []  # Gaia 切平面 (xi_gaia, eta_gaia)
    for img_idx, gaia_idx, _ in matched_pairs:
        # 图像星 → 网格坐标 → 原始像素偏移 → 切平面 (xi_pred, eta_pred) arcsec
        x_grid = stars_x[img_idx] * scale + offset_x
        y_grid = stars_y[img_idx] * scale + offset_y
        dx_pix = (x_grid - wcs.crpix1) * wcs.scale_factor
        dy_pix = (y_grid - wcs.crpix2) * wcs.scale_factor
        xi_pred_deg = dx_pix * wcs.cd11 + dy_pix * wcs.cd12
        eta_pred_deg = dx_pix * wcs.cd21 + dy_pix * wcs.cd22
        xi_pred_asec = xi_pred_deg * 3600.0
        eta_pred_asec = eta_pred_deg * 3600.0

        # Gaia 星 → 切平面 (xi_gaia, eta_gaia) arcsec via gnomonic_forward(CRVAL)
        xi_gaia, eta_gaia, _ = gnomonic_forward(
            np.array([gaia_ra[gaia_idx]]),
            np.array([gaia_dec[gaia_idx]]),
            wcs.crval1, wcs.crval2,
        )
        xi_gaia_asec = float(xi_gaia[0])
        eta_gaia_asec = float(eta_gaia[0])

        src_pts.append([xi_pred_asec, eta_pred_asec])
        dst_pts.append([xi_gaia_asec, eta_gaia_asec])

    src = np.array(src_pts, dtype=np.float64)
    dst = np.array(dst_pts, dtype=np.float64)
    n = len(src)

    # MAD 迭代 outlier 剔除 + Umeyama 2D 拟合
    # 大容差匹配中错误匹配占主导 (RMS~29"), Umeyama 最小二乘不鲁棒,
    # 需要先剔除 outlier 再拟合。MAD (中位绝对偏差) 阈值对 outlier 鲁棒。
    inlier_mask = np.ones(n, dtype=bool)
    R = np.eye(2)
    s = 1.0
    t = np.zeros(2)
    for iter_idx in range(5):
        src_in = src[inlier_mask]
        dst_in = dst[inlier_mask]
        n_in = len(src_in)
        if n_in < 3:
            break

        # Umeyama 2D 拟合 (仅用 inlier)
        mu_src = np.mean(src_in, axis=0)
        mu_dst = np.mean(dst_in, axis=0)
        src_c = src_in - mu_src
        dst_c = dst_in - mu_dst
        H = src_c.T @ dst_c / n_in
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        var_src = float(np.sum(src_c ** 2) / n_in)
        s = float(np.sum(S) / var_src) if var_src > 0 else 1.0
        t = mu_dst - s * R @ mu_src

        # 计算所有匹配对的残差 (用当前 s, R, t)
        pred = (s * R @ src.T).T + t  # (n, 2)
        resid = dst - pred  # (n, 2)
        resid_norm = np.linalg.norm(resid, axis=1)  # (n,) arcsec

        # MAD 阈值: max(5", 3 × 1.4826 × MAD)
        med = float(np.median(resid_norm[inlier_mask]))
        mad = float(np.median(np.abs(resid_norm[inlier_mask] - med)))
        thresh = max(5.0, 3.0 * 1.4826 * mad)
        new_inlier_mask = resid_norm <= thresh
        n_new = int(np.sum(new_inlier_mask))

        logger.info("WCS 2D 精化 iter%d: n_in=%d, med=%.2f\", MAD=%.2f\", "
                    "thresh=%.2f\", n_new=%d, d_θ=%.4f°, s=%.4f",
                    iter_idx + 1, n_in, med, mad, thresh, n_new,
                    math.degrees(math.atan2(R[1, 0], R[0, 0])), s)

        if n_new < 3:
            break
        # 收敛: inlier 集合不再变化
        if np.array_equal(new_inlier_mask, inlier_mask):
            break
        inlier_mask = new_inlier_mask

    n_inliers_final = int(np.sum(inlier_mask))
    if n_inliers_final < 3:
        logger.warning("WCS 2D 精化: outlier 剔除后 inlier 不足 (%d < 3), 跳过", n_inliers_final)
        return wcs

    # 角度修正 (从 R 提取)
    d_theta_rad = math.atan2(R[1, 0], R[0, 0])
    d_theta_deg = math.degrees(d_theta_rad)

    # 尺度约束: s 应在 ±10% 范围内 (防止过拟合)
    s_clamped = max(0.9, min(1.1, s))

    logger.info("WCS 2D 精化: n_pairs=%d→%d (outlier 剔除), d_θ=%.4f°, s=%.4f (约束后=%.4f), "
                "t=(%.2f\", %.2f\")",
                n, n_inliers_final, d_theta_deg, s, s_clamped, float(t[0]), float(t[1]))

    # 修正 CD: CD_new = s × R × CD
    # R 是 2×2 旋转矩阵, CD 是 2×2
    cd_mat = np.array([[wcs.cd11, wcs.cd12],
                       [wcs.cd21, wcs.cd22]])
    cd_new = s_clamped * R @ cd_mat
    cd11_new = float(cd_new[0, 0])
    cd12_new = float(cd_new[0, 1])
    cd21_new = float(cd_new[1, 0])
    cd22_new = float(cd_new[1, 1])

    # 修正 CRVAL: t 是切平面平移 (arcsec), CRVAL_new = gnomonic_inverse(t_xi, t_eta, CRVAL)
    # t 表示 dst = s×R×src + t, 即 Gaia 切平面 = 变换(图像切平面) + t
    # 所以 CRVAL 应该向 t 方向移动
    crval1_new, crval2_new = gnomonic_inverse(
        np.array([float(t[0])]), np.array([float(t[1])]),
        wcs.crval1, wcs.crval2,
    )
    crval1_new = float(crval1_new[0])
    crval2_new = float(crval2_new[0])

    # 新 theta (从 CD 反推)
    theta_new = math.degrees(math.atan2(cd21_new, cd22_new))

    logger.info("WCS 2D 精化结果: θ=%.4f°→%.4f° (d_θ=%.4f°), "
                "CRVAL=(%.6f,%.6f)→(%.6f,%.6f), s_out=%.4f\"",
                wcs.theta_deg, theta_new, d_theta_deg,
                wcs.crval1, wcs.crval2, crval1_new, crval2_new,
                3600.0 * math.sqrt(abs(cd11_new * cd22_new - cd12_new * cd21_new)))

    return WCSResult(
        crval1=crval1_new, crval2=crval2_new,
        cd11=cd11_new, cd12=cd12_new, cd21=cd21_new, cd22=cd22_new,
        crpix1=wcs.crpix1, crpix2=wcs.crpix2,
        grid=wcs.grid, flip_mode=wcs.flip_mode, theta_deg=theta_new,
        scale_factor=wcs.scale_factor,
    )

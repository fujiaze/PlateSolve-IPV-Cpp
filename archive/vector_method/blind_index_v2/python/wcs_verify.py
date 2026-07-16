"""
向量法验证模块 (Task 6)
功能: 对候选(天区中心, 旋转角)查询Gaia局部星表, 旋转配对 + Umeyama SVD求解WCS + RMS验证
用途: ADV-PA盲解析的最终验证阶段, 复用V3.5的gnomonic投影+Umeyama SVD引擎

算法流程:
    1. 查询Gaia局部星表(锥角1.5×FOV, 极限星等14)
    2. gnomonic投影到切平面(xi, eta arcsec)
    3. 将星表切平面坐标旋转到图像帧(含Y翻转): R_img = rot_angle + 90°
       u_cat = cos(R_img)*xi + sin(R_img)*eta
       v_cat = sin(R_img)*xi - cos(R_img)*eta
    4. 图像点居中+Y翻转: U = [s0*(x-cx), -s0*(y-cy)]
    5. cKDTree最近邻匹配U与W(星表图像帧), 阈值5*s0
    6. Umeyama SVD(允许反射, 捕获Y翻转) on (image_pixel, catalog_tangent)
    7. 导出CD/CRPIX/CRVAL, 计算RMS
    8. 迭代精修: 紧阈值重匹配 + Umeyama

关键推导 (Y翻转处理):
    y-down图像中 theta_img = PA_cat + R - 90° (R=图像旋转角)
    投票峰值 rot = theta_img - PA_cat = R - 90°
    故 R = rot_angle + 90° (将投票角转换为实际图像旋转)
    旋转星表切平面坐标 by R 即可与图像帧对齐(含Y翻转)

依赖: numpy, scipy.spatial.cKDTree, lib.plate_solve.python.vector_match_v2 (gnomonic_forward/inverse, _umeyama)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .logging_setup import get_logger

# 复用V3.5 gnomonic投影 + Umeyama SVD
from lib.plate_solve.python.vector_match_v2 import (
    gnomonic_forward, gnomonic_inverse, _umeyama, GaiaClientPy,
)

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
        success: 是否成功
        cd: 2x2 CD矩阵 (deg/pixel)
        crpix1: FITS CRPIX1 (1-indexed)
        crpix2: FITS CRPIX2 (1-indexed)
        crval1: CRVAL1 (度)
        crval2: CRVAL2 (度)
        s: Umeyama尺度
        theta: Umeyama旋转角(弧度)
        t: 平移向量 (arcsec)
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)
        rms_arcsec: 投影残差RMS(arcsec)
        n_inliers: 内点数(匹配对数)
    """
    success: bool
    cd: np.ndarray
    crpix1: float
    crpix2: float
    crval1: float
    crval2: float
    s: float
    theta: float
    t: np.ndarray
    ra0: float
    dec0: float
    rms_arcsec: float
    n_inliers: int


def _umeyama_reflect(P: np.ndarray, Q: np.ndarray) -> Optional[tuple[np.ndarray, float, np.ndarray]]:
    """
    Umeyama SVD (允许反射, 捕获Y翻转): Q ≈ s·R·P + t

    与vector_match_v2._umeyama的区别: 不强制det(R)=+1, 允许det(R)=-1(反射)
    这对图像y-down → 天球eta-up的Y翻转至关重要

    Args:
        P: (N, 2) 源点 (图像像素)
        Q: (N, 2) 目标点 (切平面arcsec)

    Returns:
        (R, s, t): R=(2,2), s=标量, t=(2,) 使 Q ≈ s·R·P + t, 或None
    """
    n = P.shape[0]
    if n < 2:
        return None

    P_mean = P.mean(axis=0)
    Q_mean = Q.mean(axis=0)
    P_c = P - P_mean
    Q_c = Q - Q_mean

    # H = P_c^T @ Q_c (2x2)
    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)

    # R = V @ U^T (不强制det=+1, 允许反射)
    R = Vt.T @ U.T

    # 尺度
    var_P = float((P_c ** 2).sum())
    if var_P < 1e-30:
        return None
    s = float(S.sum() / var_P)

    # 平移
    t = Q_mean - s * (R @ P_mean)

    return R, s, t


def _match_1to1(
    U: np.ndarray, W: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    cKDTree最近邻1对1互斥匹配。

    Args:
        U: (N, 2) 图像点
        W: (M, 2) 星表点
        threshold: 匹配阈值(同单位)

    Returns:
        (u_idx, w_idx, dists): 匹配的索引和距离
    """
    tree = cKDTree(W)
    dists, idxs = tree.query(U, k=1)
    mask = dists < threshold
    u_idx = np.where(mask)[0]
    w_idx = idxs[mask]
    dists_match = dists[mask]
    return u_idx, w_idx, dists_match


def _compute_rms_arcsec(
    image_xy: np.ndarray,
    R: np.ndarray,
    s: float,
    t: np.ndarray,
    ra0: float,
    dec0: float,
    ref_ra: np.ndarray,
    ref_dec: np.ndarray,
) -> float:
    """
    计算投影残差RMS(arcsec)。

    流程: 图像像素 → 切平面arcsec (s·R·P + t) → gnomonic_inverse → RA/Dec → 与参考星比较

    Args:
        image_xy: (N, 2) 图像像素
        R: (2, 2) 旋转矩阵
        s: 尺度
        t: (2,) 平移(arcsec)
        ra0, dec0: 切平面中心(度)
        ref_ra, ref_dec: (N,) 参考星RA/Dec(度)

    Returns:
        RMS (arcsec)
    """
    tangent = (s * (R @ image_xy.T)).T + t  # (N, 2) [xi, eta] arcsec
    xi = tangent[:, 0]
    eta = tangent[:, 1]
    ra_pred, dec_pred = gnomonic_inverse(xi, eta, ra0, dec0)

    # haversine角距离
    ra1r = ra_pred * _DEGTORAD
    dec1r = dec_pred * _DEGTORAD
    ra2r = ref_ra * _DEGTORAD
    dec2r = ref_dec * _DEGTORAD
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = np.sin(ddec / 2.0) ** 2 + np.cos(dec1r) * np.cos(dec2r) * np.sin(dra / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    sep = 2.0 * np.arcsin(np.sqrt(a)) * _RADTOASEC
    return float(np.sqrt(np.mean(sep ** 2)))


def _verify_with_rot(
    image_xy: np.ndarray,
    s0: float,
    xi: np.ndarray,
    eta: np.ndarray,
    ra_cat: np.ndarray,
    dec_cat: np.ndarray,
    ra_center: float,
    dec_center: float,
    rot_angle: float,
    image_width: int,
    image_height: int,
    sigma_pos: float,
) -> Optional[tuple]:
    """
    用指定 rot_angle 执行一次"旋转+匹配+Umeyama+精修"流程 (verify_candidate 步骤3-7)。

    fix-adv-pa-phase1-bugs: 4-way 投票为每个 cell 产生 4 个 rot bin 变体
    (rot, rot+180, 360-rot, 180-rot)。本函数对单个 rot 变体执行验证,
    由 verify_candidate 对 4 个变体分别调用后选最佳。

    Args:
        image_xy: (N, 2) 图像星点像素坐标
        s0: 像素尺度(arcsec/pixel)
        xi, eta: (M,) 星表切平面坐标(arcsec, 已投影到 ra_center/dec_center)
        ra_cat, dec_cat: (M,) 星表 RA/Dec(度)
        ra_center, dec_center: 切平面中心(度)
        rot_angle: 旋转角变体(度)
        image_width, image_height: 图像宽高(像素)
        sigma_pos: 位置噪声(像素)

    Returns:
        (R, s, t, n_inliers, rms, img_matched, ra_matched, dec_matched) 或 None
    """
    # 3. 将星表切平面旋转到图像帧 (含Y翻转)
    # 推导: y-down图像中 rot_vote = R_img - 90°, 故 R_img = rot_angle + 90°
    R_img = (rot_angle + 90.0) * _DEGTORAD
    cos_R = math.cos(R_img)
    sin_R = math.sin(R_img)
    # M = [[cos_R, sin_R], [sin_R, -cos_R]] (含Y翻转的变换矩阵, M^2=I)
    u_cat = cos_R * xi + sin_R * eta
    v_cat = sin_R * xi - cos_R * eta

    # 4. 图像点居中+Y翻转
    cx = image_width / 2.0
    cy = image_height / 2.0
    u_img = s0 * (image_xy[:, 0] - cx)
    v_img = -s0 * (image_xy[:, 1] - cy)  # Y翻转

    # 5. cKDTree最近邻匹配 (图像帧)
    W = np.column_stack([u_cat, v_cat])  # (M, 2) 星表在图像帧
    U = np.column_stack([u_img, v_img])  # (N, 2) 图像在图像帧

    # 初始匹配: 宽阈值 (5*s0, 容忍rot误差+残差平移)
    threshold_init = 5.0 * s0
    u_idx, w_idx, _ = _match_1to1(U, W, threshold_init)
    n_matched = len(u_idx)
    if n_matched < 3:
        return None

    # 6. Umeyama SVD (允许反射) on (image_pixel, catalog_tangent)
    img_matched = image_xy[u_idx]  # 原始像素坐标
    xi_matched = xi[w_idx]
    eta_matched = eta[w_idx]
    ra_matched = ra_cat[w_idx]
    dec_matched = dec_cat[w_idx]

    result = _umeyama_reflect(img_matched, np.column_stack([xi_matched, eta_matched]))
    if result is None:
        return None
    R, s, t = result

    # 7. 迭代精修: 紧阈值重匹配 + Umeyama
    threshold_fine = max(1.0, 2.0 * sigma_pos * s0)
    cat_tan = np.column_stack([xi, eta])
    for iteration in range(3):
        # 用当前变换重新匹配
        tangent_pred = (s * (R @ image_xy.T)).T + t  # (N, 2) arcsec
        tree_cat = cKDTree(cat_tan)
        dists_ref, idxs_ref = tree_cat.query(tangent_pred, k=1)
        mask_ref = dists_ref < threshold_fine
        if np.sum(mask_ref) < 3:
            break

        img_ref = image_xy[mask_ref]
        cat_ref_idx = idxs_ref[mask_ref]
        xi_ref = xi[cat_ref_idx]
        eta_ref = eta[cat_ref_idx]
        ra_ref = ra_cat[cat_ref_idx]
        dec_ref = dec_cat[cat_ref_idx]

        # 重新Umeyama
        result_ref = _umeyama_reflect(
            img_ref, np.column_stack([xi_ref, eta_ref])
        )
        if result_ref is None:
            break
        R_new, s_new, t_new = result_ref

        # 收敛检查
        s_diff = abs(s_new - s) / max(s, 1e-10)
        R, s, t = R_new, s_new, t_new
        img_matched = img_ref
        ra_matched = ra_ref
        dec_matched = dec_ref
        if s_diff < 0.001:
            break

    n_inliers = len(img_matched)

    # 计算 RMS (用于变体间比较)
    rms = _compute_rms_arcsec(
        img_matched, R, s, t, ra_center, dec_center, ra_matched, dec_matched
    )

    return R, s, t, n_inliers, rms, img_matched, ra_matched, dec_matched


def verify_candidate(
    image_xy: np.ndarray,
    s0: float,
    ra_center: float,
    dec_center: float,
    rot_angle: float,
    gaia_client: GaiaClientPy,
    fov_diag_deg: float,
    image_width: int,
    image_height: int,
    mag_limit: float = 14.0,
    sigma_pos: float = 0.5,
) -> Optional[WCSResult]:
    """
    验证候选(天区中心, 旋转角), 求解WCS。

    fix-adv-pa-phase1-bugs: 4-way 投票为每个 cell 产生 4 个 rot bin 变体
    (rot, rot+180, 360-rot, 180-rot), 对应 Y-flip 与 PA 方向歧义的 4 种组合。
    真匹配只落在其中 1 个变体, 其余 3 个为噪声。本函数对 4 个变体分别验证,
    选 n_inliers 最多 (并列时选 RMS 最低) 的解, 避免错误变体产生低 RMS 假阳性。

    Args:
        image_xy: (N, 2) 图像星点像素坐标
        s0: 像素尺度(arcsec/pixel)
        ra_center, dec_center: 候选天区中心(度)
        rot_angle: 候选旋转角(度, 来自投票rot_bin)
        gaia_client: GaiaClientPy实例
        fov_diag_deg: FOV对角线(度)
        image_width, image_height: 图像宽高(像素)
        mag_limit: Gaia查询极限星等
        sigma_pos: 位置噪声(像素)

    Returns:
        WCSResult 或 None(验证失败)
    """
    image_xy = np.asarray(image_xy, dtype=np.float64)
    n_img = len(image_xy)
    if n_img < 4:
        logger.warning("图像星点不足: %d < 4", n_img)
        return None

    # 1. 查询Gaia局部星表
    query_radius = 1.5 * fov_diag_deg
    ra_cat, dec_cat, mag_cat = gaia_client.cone_search(
        ra_center, dec_center, query_radius, mag_limit
    )
    n_cat = len(ra_cat)
    if n_cat < 4:
        logger.warning("Gaia查询星数不足: %d < 4 (中心=%.4f,%.4f, 半径=%.3f°)",
                        n_cat, ra_center, dec_center, query_radius)
        return None
    logger.info("候选验证: 中心(%.4f,%.4f), rot=%.1f°, Gaia星数=%d, 图像星数=%d",
                 ra_center, dec_center, rot_angle, n_cat, n_img)

    # 2. gnomonic投影到切平面
    xi, eta, valid = gnomonic_forward(ra_cat, dec_cat, ra_center, dec_center)
    xi = xi[valid]
    eta = eta[valid]
    ra_cat = ra_cat[valid]
    dec_cat = dec_cat[valid]
    n_cat = len(ra_cat)

    # fix-adv-pa-phase1-bugs: 4-way 投票变体试错
    # 投票阶段为每个真匹配投票 4 个 rot bin: rot, (rot+180)%360, (360-rot)%360, (180-rot)%360
    # 峰值检测返回的 rot_bin 只是其中之一; 真匹配可能落在任一变体。
    # 对 4 个变体都尝试验证, 选 n_inliers 最多的解。
    rot_variants = sorted(set([
        rot_angle % 360.0,
        (rot_angle + 180.0) % 360.0,
        (360.0 - rot_angle) % 360.0,
        (180.0 - rot_angle) % 360.0,
    ]))

    best_result = None
    best_n_inliers = -1
    best_rms = float("inf")
    best_rot_var = rot_angle

    for rot_var in rot_variants:
        result = _verify_with_rot(
            image_xy, s0, xi, eta, ra_cat, dec_cat,
            ra_center, dec_center, rot_var,
            image_width, image_height, sigma_pos,
        )
        if result is None:
            logger.info("rot=%.1f°: 初始匹配不足, 跳过", rot_var)
            continue
        R_v, s_v, t_v, n_inliers_v, rms_v, _, _, _ = result
        # 选 n_inliers 最多, 并列时选 RMS 最低
        is_better = (n_inliers_v > best_n_inliers or
                     (n_inliers_v == best_n_inliers and rms_v < best_rms))
        tag = "新最佳" if is_better else "跳过"
        logger.info("rot=%.1f°: n_inliers=%d, RMS=%.3f\", s=%.5f (%s)",
                     rot_var, n_inliers_v, rms_v, s_v, tag)
        if is_better:
            best_result = result
            best_n_inliers = n_inliers_v
            best_rms = rms_v
            best_rot_var = rot_var

    if best_result is None:
        logger.warning("所有 %d 种 rot 变体验证失败 (初始匹配均不足)", len(rot_variants))
        return None

    R, s, t, n_inliers, rms, img_matched, ra_matched, dec_matched = best_result

    # fix-adv-pa-phase1-bugs: 切点重居中
    # cell 中心可能偏离真天区 (等距网格 0.84° 分辨率), 导致投影畸变和真匹配丢失。
    # 用 Umeyama 变换计算图像中心的 sky 位置, 作为新切点重新投影+验证。
    # 对错误天区: Umeyama 将图像中心映射到 cell 中心 (tangent≈0), 重居中是 no-op。
    # 对真天区附近: Umeyama 将图像中心映射到真天区, 重居中后投影畸变减小, 可找到更多真匹配。
    cx = image_width / 2.0
    cy = image_height / 2.0
    tangent_center = s * (R @ np.array([cx, cy])) + t
    new_ra0, new_dec0 = gnomonic_inverse(
        float(tangent_center[0]), float(tangent_center[1]),
        ra_center, dec_center
    )
    # 计算切点偏移 (考虑 RA 方向 cos(dec) 收缩)
    offset_ra = (new_ra0 - ra_center) * math.cos(dec_center * _DEGTORAD)
    offset_dec = (new_dec0 - dec_center)
    offset_arcsec = math.sqrt(offset_ra ** 2 + offset_dec ** 2) * 3600.0

    if offset_arcsec > 360.0:  # > 0.1°
        logger.info("切点重居中: (%.4f,%.4f) → (%.4f,%.4f) (偏移=%.1f\")",
                     ra_center, dec_center, new_ra0, new_dec0, offset_arcsec)
        # 重新投影 catalog 星到新切点
        xi_new, eta_new, valid_new = gnomonic_forward(
            ra_cat, dec_cat, new_ra0, new_dec0
        )
        # 用最佳 rot 变体重新验证
        recenter_result = _verify_with_rot(
            image_xy, s0,
            xi_new[valid_new], eta_new[valid_new],
            ra_cat[valid_new], dec_cat[valid_new],
            new_ra0, new_dec0, best_rot_var,
            image_width, image_height, sigma_pos,
        )
        if recenter_result is not None:
            R_r, s_r, t_r, n_inliers_r, rms_r, _, _, _ = recenter_result
            if n_inliers_r > n_inliers:
                logger.info("重居中后内点提升: %d → %d, RMS=%.3f\" → %.3f\"",
                             n_inliers, n_inliers_r, rms, rms_r)
                best_result = recenter_result
                R, s, t, n_inliers, rms, img_matched, ra_matched, dec_matched = best_result
                ra_center = new_ra0
                dec_center = new_dec0
            else:
                logger.info("重居中后内点未提升: %d → %d, 保留原结果",
                             n_inliers, n_inliers_r)
        else:
            logger.info("重居中后验证失败, 保留原结果")

    # 8. 导出WCS
    A = s * R  # arcsec/pixel (image_pixel → tangent_arcsec)
    det_A = float(np.linalg.det(A))
    if abs(det_A) < 1e-20:
        logger.warning("A矩阵奇异, 无法求CRPIX")
        return None

    cd = A / 3600.0  # deg/pixel
    A_inv = np.linalg.inv(A)
    crpix_xy = -A_inv @ t  # (2,) [x, y] at tangent=(0,0)
    crpix1 = float(crpix_xy[0] + 1.0)  # FITS 1-indexed
    crpix2 = float(crpix_xy[1] + 1.0)

    theta = float(math.atan2(R[1, 0], R[0, 0]))

    logger.info("WCS验证成功: s=%.5f, θ=%.2f°, n_inliers=%d, RMS=%.3f\" (best rot 变体)",
                 s, math.degrees(theta), n_inliers, rms)

    return WCSResult(
        success=True,
        cd=cd,
        crpix1=crpix1,
        crpix2=crpix2,
        crval1=float(ra_center),
        crval2=float(dec_center),
        s=float(s),
        theta=theta,
        t=t,
        ra0=float(ra_center),
        dec0=float(dec_center),
        rms_arcsec=rms,
        n_inliers=int(n_inliers),
    )

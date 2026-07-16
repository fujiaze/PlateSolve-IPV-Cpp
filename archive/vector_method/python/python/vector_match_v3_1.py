"""
Vector Match V3.1 - 加权蒙特卡洛1点抽样法

相比V2的核心改动:
    用蒙特卡洛1点抽样+内点数加权替代RANSAC 2点法

V2问题:
    RANSAC 2点法在大旋转角(±90°)时失效:
    - 候选对中正确比例极低(1/295)
    - P(2点都正确) ≈ 10⁻⁹, K=3000远远不够

V3.1方案:
    1点法: 随机抽1个(u_i, w_j)对，直接计算变换参数(s, θ, tx, ty)
    - s = |u_i| / |w_j|, 筛选s∈[0.9,1.1]
    - 应用变换后统计内点数
    - 选内点数最多的变换作为粗匹配结果
    - SVD精修与V2相同

依赖: numpy, scipy(scipy.spatial.cKDTree), ctypes(Gaia DLL)
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger("vector_match_v3_1")

_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi
_RADTOASEC = (180.0 / np.pi) * 3600.0
_ASECTORAD = np.pi / (180.0 * 3600.0)


# ============================================================================
# 复用V2的Gaia客户端、投影、查询函数
# ============================================================================

from vector_match_v2 import (
    GaiaClientPy,
    gnomonic_forward,
    gnomonic_inverse,
    bisection_mag_limit,
    _build_image_vectors,
    _build_catalog_vectors,
    _apply_flip,
    _apply_similarity,
    _count_inliers_1to1,
    _compute_normalized_score,
    _umeyama,
    _iterative_svd_refine,
    _find_fine_correspondences,
    _ransac_rigid_v2,
    VectorMatchResult,
)


# ============================================================================
# V3.1 核心: 加权蒙特卡洛1点抽样
# ============================================================================

def _count_inliers_fast(U, Wt, tau):
    """快速内点计数（粗匹配阶段用，不做1对1互斥）

    只统计U中有多少点在Wt中距离<tau的最近邻，速度比_count_inliers_1to1快5-10x。
    粗匹配阶段不需要1对1互斥，因为内点数本身就是权重信号。
    """
    tree = cKDTree(Wt)
    dists, _ = tree.query(U, k=1)
    n_inliers = int(np.sum(dists < tau))
    if n_inliers == 0:
        return 0, 0.0
    rms = float(np.sqrt(np.mean(dists[dists < tau] ** 2)))
    return n_inliers, rms


def _mc_weighted_sampling(U, Wf, tau, N_samples, rng, s_range=(0.9, 1.1)):
    """V3.1 加权蒙特卡洛1点抽样

    单次抽样收集所有参数(θ, s, tx, ty, n_inliers)，
    直接选内点数最多的变换作为粗匹配结果。

    参数:
        U: 图像向量组 (N, 2)
        Wf: 翻转后星表向量组 (M, 2)
        tau: 内点阈值(角秒)
        N_samples: 抽样次数
        rng: 随机数生成器
        s_range: 比例尺范围
    返回:
        (s, theta, tx, ty, n_inliers, rms, inlier_mask)
    """
    N = len(U)
    M = len(Wf)
    if N < 2 or M < 2:
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    # 预计算cos/sin可以批量做，但Python循环中逐个计算更简单
    # 收集所有s≈1的变换参数
    best_n_inliers = 0
    best_s = 1.0
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0
    best_rms = 0.0

    n_tried = 0
    n_s_in_range = 0

    for _ in range(N_samples):
        i = rng.integers(0, N)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        n_tried += 1

        s = norm_U[i] / norm_Wf[j]
        if s < s_range[0] or s > s_range[1]:
            continue
        n_s_in_range += 1

        theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
        ct, st = math.cos(theta), math.sin(theta)
        tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])

        # 变换全部Wf
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty

        # 快速内点计数（不做1对1互斥）
        n_inliers, rms = _count_inliers_fast(U, Wt, tau)

        if n_inliers > best_n_inliers:
            best_n_inliers = n_inliers
            best_s = s
            best_theta = theta
            best_tx = tx
            best_ty = ty
            best_rms = rms

    logger.debug("  MC抽样: 尝试=%d s≈1=%d(%.1f%%) 最佳n=%d rms=%.3f",
                 n_tried, n_s_in_range,
                 n_s_in_range / max(n_tried, 1) * 100,
                 best_n_inliers, best_rms)

    if best_n_inliers == 0:
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    # 用最佳变换重新统计1对1互斥内点（用于SVD精修）
    Wt = _apply_similarity(Wf, best_s, best_theta, best_tx, best_ty)
    n_inliers, rms, inlier_mask = _count_inliers_1to1(U, Wt, tau)

    logger.debug("  MC最佳(1对1): s=%.4f θ=%.2f° tx=%.1f ty=%.1f n=%d rms=%.3f",
                 best_s, math.degrees(best_theta), best_tx, best_ty, n_inliers, rms)

    return best_s, best_theta, best_tx, best_ty, n_inliers, rms, inlier_mask


# ============================================================================
# V3.1 主类
# ============================================================================

class VectorMatch:
    """V3.1 向量匹配Plate Solving算法

    核心改动: 用加权蒙特卡洛1点抽样替代RANSAC 2点法
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._rng = np.random.default_rng(42)
        logger.info("VectorMatchV3.1初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

    def solve(
        self,
        img_x: np.ndarray,
        img_y: np.ndarray,
        img_flux: np.ndarray,
        img_saturated: np.ndarray,
        center_ra: float,
        center_dec: float,
        focal_length_mm: float,
        pixel_size_um: float,
        width: int,
        height: int,
    ) -> Optional[VectorMatchResult]:
        """V3.1 向量匹配主入口"""
        s0 = 206.265 * pixel_size_um / focal_length_mm
        logger.info("像素尺度 s0=%.4f 角秒/像素 (FOCALLEN=%.1fmm XPIXSZ=%.1fμm)",
                     s0, focal_length_mm, pixel_size_um)

        fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        logger.info("FOV对角线=%.2f度, 查询半径=%.2f度", fov_diag, radius_deg)

        # 构建U + 稀疏度权重
        U, N_img, n_sat, sparsity = _build_image_vectors(
            np.asarray(img_x, dtype=np.float64),
            np.asarray(img_y, dtype=np.float64),
            np.asarray(img_flux, dtype=np.float64),
            np.asarray(img_saturated, dtype=np.int32),
            s0, width, height,
        )
        if N_img < 2:
            logger.error("图像亮星不足: N_img=%d", N_img)
            return None
        logger.info("图像向量组: N_img=%d (饱和星=%d)", N_img, n_sat)

        # Gaia查询
        if n_sat >= 50:
            N_gaia = math.ceil(1.5 * n_sat)
        else:
            N_gaia = 150
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            self._gaia, center_ra, center_dec, radius_deg, N_gaia
        )
        if M < 2:
            logger.error("星表星数不足: M=%d", M)
            return None
        logger.info("星表查询: 极限星等=%.2f, 星数=%d (目标N_gaia=%d)", mag_limit, M, N_gaia)

        # V3.1参数
        tau_coarse = max(1.0, 2.5 * s0)
        min_inliers = max(5, int(N_img * 0.1))

        # 宽视场帧(FOV>5°): gnomonic投影畸变大，1点法失效，回退到V2 RANSAC
        use_ransac = fov_diag > 5.0
        if use_ransac:
            N_samples = 0
            K_ransac = 3000
            candidate_radius = fov_diag * 3600.0 * 0.5
            logger.info("V3.1参数(宽视场回退RANSAC): tau_coarse=%.2f K=%d candidate_radius=%.1f″",
                         tau_coarse, K_ransac, candidate_radius)
        else:
            N_samples = 50000
            logger.info("V3.1参数(1点法): tau_coarse=%.2f N_samples=%d min_inliers=%d",
                         tau_coarse, N_samples, min_inliers)

        best_mode = -1
        best_norm_score = -1.0
        best_result = None

        for mode in range(4):
            W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
            Wf = _apply_flip(W, mode)
            logger.info("翻转模式%d: 星表向量组 %d 颗", mode, len(Wf))

            if use_ransac:
                # ── 宽视场: V2 RANSAC 2点法 ──
                s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
                    U, Wf, tau_coarse, K_ransac, min_inliers, self._rng,
                    candidate_radius, sparsity
                )
            else:
                # ── 窄视场: V3.1 加权蒙特卡洛1点抽样 ──
                s, theta, tx, ty, n_inliers, rms, inlier_mask = _mc_weighted_sampling(
                    U, Wf, tau_coarse, N_samples, self._rng
                )
            if n_inliers < min_inliers:
                logger.info("  模式%d: MC抽样内点不足 n=%d < min=%d", mode, n_inliers, min_inliers)
                continue
            logger.info("  模式%d MC粗匹配: s=%.4f theta=%.2f° tx=%.2f ty=%.2f n=%d rms=%.3f",
                        mode, s, math.degrees(theta), tx, ty, n_inliers, rms)

            # ── SVD精修 (与V2相同) ──
            s_refined, theta_refined, tx_refined, ty_refined, n_refined, rms_refined, mask_refined = \
                _iterative_svd_refine(U, Wf, inlier_mask, s0, s, theta, tx, ty, max_iter=10)

            if n_refined >= min_inliers:
                s, theta, tx, ty = s_refined, theta_refined, tx_refined, ty_refined
                n_inliers, rms, inlier_mask = n_refined, rms_refined, mask_refined
                logger.info("  模式%d SVD精修: s=%.4f theta=%.2f° n=%d rms=%.3f",
                            mode, s, math.degrees(theta), n_inliers, rms)

            norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)
            logger.info("  模式%d 最终: s=%.4f theta=%.2f° n=%d rms=%.3f norm_score=%.4f",
                        mode, s, math.degrees(theta), n_inliers, rms, norm_score)

            if norm_score > best_norm_score:
                best_norm_score = norm_score
                best_mode = mode
                best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

        if best_mode < 0 or best_norm_score < 0.10:
            logger.warning("所有模式匹配失败: best_norm_score=%.4f", best_norm_score)
            return None

        s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result

        if s < 0.9 or s > 1.1:
            logger.warning("s=%.4f超出有效范围[0.9, 1.1]，判定为无效结果", s)
            return None

        logger.info("最佳模式=%d, 归一化得分=%.4f", best_mode, best_norm_score)

        result = self._extract_wcs_and_converge(
            s, theta, tx, ty, best_mode, s0,
            center_ra, center_dec, width, height,
            U, Wf, inlier_mask, N_img, M,
            cat_ra, cat_dec, cat_mag,
            fov_diag, sparsity,
        )
        return result

    def _extract_wcs_and_converge(
        self, s, theta, tx, ty, flip_mode, s0,
        ra0, dec0, width, height,
        U, Wf, inlier_mask, N_img, M,
        cat_ra, cat_dec, cat_mag,
        fov_diag, sparsity,
    ):
        """WCS参数提取 + 中心修正 + SVD精修 (与V2相同)"""
        cur_ra, cur_dec = ra0, dec0
        cur_s, cur_theta = s, theta
        cur_tx, cur_ty = tx, ty
        cur_flip = flip_mode

        # 中心修正
        cos_d0 = math.cos(cur_dec * _DEGTORAD)
        if abs(cos_d0) < 1e-10:
            cos_d0 = 1e-10
        delta_ra = -cur_tx / (cos_d0 * 3600.0)
        delta_dec = -cur_ty / 3600.0
        cur_ra += delta_ra
        cur_dec += delta_dec
        logger.info("中心修正: ΔRA=%.6f° ΔDec=%.6f° → RA=%.6f Dec=%.6f",
                     delta_ra, delta_dec, cur_ra, cur_dec)

        # 重新投影 + 精修
        W_new = _build_catalog_vectors(cat_ra, cat_dec, cur_ra, cur_dec)
        Wf_new = _apply_flip(W_new, cur_flip)

        refine_radius = fov_diag * 3600.0 * 0.5
        min_inliers_refine = max(5, int(N_img * 0.1))
        use_ransac_refine = fov_diag > 5.0

        refine_success = False
        for tau_mult in [1.0, 2.0, 3.0, 5.0]:
            tau_try = max(0.5, tau_mult * s0)
            if use_ransac_refine:
                s2, theta2, tx2, ty2, n2, rms2, mask2 = _ransac_rigid_v2(
                    U, Wf_new, tau_try, 3000, min_inliers_refine, self._rng,
                    refine_radius, sparsity
                )
            else:
                s2, theta2, tx2, ty2, n2, rms2, mask2 = _mc_weighted_sampling(
                    U, Wf_new, tau_try, 20000, self._rng
                )
            if n2 >= min_inliers_refine:
                s3, theta3, tx3, ty3, n3, rms3, mask3 = _iterative_svd_refine(
                    U, Wf_new, mask2, s0, s2, theta2, tx2, ty2, max_iter=10
                )
                if n3 >= min_inliers_refine:
                    cur_s, cur_theta = s3, theta3
                    cur_tx, cur_ty = tx3, ty3
                    inlier_mask = mask3
                    Wf = Wf_new
                    refine_success = True
                    logger.info("  中心修正后MC+SVD(tau=%.1fx): s=%.4f θ=%.2f° n=%d rms=%.3f",
                                tau_mult, s3, math.degrees(theta3), n3, rms3)
                else:
                    cur_s, cur_theta = s2, theta2
                    cur_tx, cur_ty = tx2, ty2
                    inlier_mask = mask2
                    Wf = Wf_new
                    refine_success = True
                break

        # 最终参数
        rotation_deg = math.degrees(cur_theta)
        s_final = s0 * cur_s

        rms_arcsec = 0.0
        rms_px = 0.0
        if np.any(inlier_mask):
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            tree = cKDTree(Wt)
            dists, idxs = tree.query(U, k=1)
            U_in = U[inlier_mask]
            W_in = Wt[idxs[inlier_mask]]
            diffs = U_in - W_in
            rms_arcsec = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
            rms_px = rms_arcsec / s0 if s0 > 0 else 0.0

        cos_t, sin_t = math.cos(cur_theta), math.sin(cur_theta)
        affine = (cur_tx, cur_s * cos_t, -cur_s * sin_t,
                  cur_ty, cur_s * sin_t, cur_s * cos_t)

        return VectorMatchResult(
            center_ra=cur_ra, center_dec=cur_dec,
            original_ra=ra0, original_dec=dec0,
            rotation_deg=rotation_deg, scale_arcsec_px=s_final,
            flip_mode=cur_flip, matched_count=int(np.sum(inlier_mask)),
            rms_px=rms_px, rms_arcsec=rms_arcsec, affine=affine,
        )

    def close(self):
        if not self._closed and self._gaia:
            self._gaia.close()
            self._gaia = None
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

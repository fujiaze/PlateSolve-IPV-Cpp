"""
Vector Match V3.2 - 三阶段1点抽样法

相比V3.1的核心改动:
    引入三阶段(θ,s)精搜索策略，解决宽视场帧1点法失效问题

V3.1问题:
    - 1点法s=|u|/|w|受gnomonic投影畸变影响，偏差0.37%
    - θ直方图1°分辨率不够，宽视场帧θ偏差0.5°导致内点数骤降
    - 固定(s,θ)后tx/ty可由1个正确对自动确定

V3.2方案:
    1. 阶段1(预热): 1点法抽样，建立θ加权直方图，找θ粗峰值
    2. 阶段2(粗搜): 在θ峰值附近遍历(θ,s)组合，抽样搜索(tx,ty)
    3. 阶段3(精搜): 在最佳(θ,s)附近精细搜索

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

logger = logging.getLogger("vector_match_v3_2")

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
    VectorMatchResult,
)


# ============================================================================
# V3.2 核心: 自适应噪声基线蒙特卡洛1点抽样
# ============================================================================

def _count_inliers_fast(U, Wt, tau):
    """快速内点计数（粗匹配阶段用，不做1对1互斥）"""
    tree = cKDTree(Wt)
    dists, _ = tree.query(U, k=1)
    n_inliers = int(np.sum(dists < tau))
    if n_inliers == 0:
        return 0, 0.0
    rms = float(np.sqrt(np.mean(dists[dists < tau] ** 2)))
    return n_inliers, rms


def _mc_adaptive_sampling(U, Wf, tau, N_max, rng,
                          s_range=(0.9, 1.1),
                          N_warmup=5000,
                          snr_threshold=3.0,
                          inlier_ratio=5.0,
                          theta_constraint=5.0):
    """V3.2 三阶段1点抽样法

    阶段1 (预热): 1点法抽样, 建立θ加权直方图, 找θ粗峰值
    阶段2 (θ+s粗搜): 在θ峰值附近遍历(θ,s)组合, 抽样搜索(tx,ty)
    阶段3 (θ+s精搜): 在最佳(θ,s)附近精细搜索

    关键发现:
        - θ精度要求<0.5°(宽视场), 预热直方图1°分辨率不够
        - s精度要求<0.4%(0.004), 1点法s=|u|/|w|受投影畸变影响
        - 固定(θ,s)后, 只需1个正确对即可计算(tx,ty)

    参数:
        U: 图像向量组 (N, 2)
        Wf: 翻转后星表向量组 (M, 2)
        tau: 内点阈值(角秒)
        N_max: 最大抽样次数(粗搜+精搜)
        rng: 随机数生成器
        s_range: 比例尺范围
        N_warmup: 预热抽样次数
        snr_threshold: θ峰值信噪比阈值
        inlier_ratio: 未使用(保留接口兼容)
        theta_constraint: θ粗搜范围(度, 相对峰值)
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

    # ── 阶段1: 预热, 找θ粗峰值 ──
    thetas = []
    weights = []

    for _ in range(N_warmup):
        i = rng.integers(0, N)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        s = norm_U[i] / norm_Wf[j]
        if s < s_range[0] or s > s_range[1]:
            continue
        theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
        ct, st = math.cos(theta), math.sin(theta)
        tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n_inliers, rms = _count_inliers_fast(U, Wt, tau)
        theta_deg = ((math.degrees(theta) + 180) % 360) - 180
        thetas.append(theta_deg)
        weights.append(n_inliers)

    if not thetas:
        logger.debug("  预热: 无有效抽样")
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    thetas = np.array(thetas)
    weights = np.array(weights)

    # 加权θ直方图
    n_bins = 360
    bin_w = 360.0 / n_bins
    weighted_counts = np.zeros(n_bins)
    for k in range(len(thetas)):
        bin_idx = int((thetas[k] + 180) / bin_w)
        if 0 <= bin_idx < n_bins:
            weighted_counts[bin_idx] += weights[k]

    peak_idx = np.argmax(weighted_counts)
    bin_centers = np.arange(-180, 180) + 0.5
    peak_theta_deg = bin_centers[peak_idx]

    # SNR计算
    bg_mask = np.ones(n_bins, dtype=bool)
    bg_mask[max(0, peak_idx - 3):min(n_bins, peak_idx + 4)] = False
    bg_mean = np.mean(weighted_counts[bg_mask]) if np.sum(bg_mask) > 10 else 1.0
    snr = weighted_counts[peak_idx] / max(bg_mean, 1e-10)

    logger.debug("  预热: peak_theta=%.1f° SNR=%.1fx", peak_theta_deg, snr)

    if snr < snr_threshold:
        logger.debug("  SNR=%.1fx < 阈值%.1fx, 该模式无信号", snr, snr_threshold)
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    # ── 阶段2: θ+s粗搜 ──
    # θ范围: 峰值±theta_constraint, 步长0.2°
    # s范围: [0.94, 1.06], 步长0.02
    # 每组抽样K_coarse次搜索(tx,ty)
    K_coarse = min(1000, N_max // 40)

    theta_coarse_range = np.arange(
        peak_theta_deg - theta_constraint,
        peak_theta_deg + theta_constraint + 0.01,
        0.2
    )
    s_coarse_range = np.arange(0.94, 1.07, 0.02)

    best_n = 0
    best_params = None

    for theta_deg in theta_coarse_range:
        theta_rad = theta_deg * _DEGTORAD
        ct, st = math.cos(theta_rad), math.sin(theta_rad)
        Wf_rot_x = ct * Wf[:, 0] - st * Wf[:, 1]
        Wf_rot_y = st * Wf[:, 0] + ct * Wf[:, 1]

        for s_try in s_coarse_range:
            max_n = 0
            best_tx, best_ty = 0.0, 0.0
            for _ in range(K_coarse):
                i = rng.integers(0, N)
                j = rng.integers(0, M)
                tx = U[i, 0] - s_try * Wf_rot_x[j]
                ty = U[i, 1] - s_try * Wf_rot_y[j]
                Wt = np.column_stack([
                    s_try * Wf_rot_x + tx,
                    s_try * Wf_rot_y + ty
                ])
                n, rms = _count_inliers_fast(U, Wt, tau)
                if n > max_n:
                    max_n = n
                    best_tx, best_ty = tx, ty

            if max_n > best_n:
                best_n = max_n
                best_params = (theta_deg, s_try, best_tx, best_ty)

    if not best_params or best_n < 3:
        logger.debug("  粗搜: 无有效结果 (best_n=%d)", best_n)
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    theta_coarse_best, s_coarse_best, tx_coarse_best, ty_coarse_best = best_params
    logger.debug("  粗搜: θ=%.1f° s=%.2f n=%d", theta_coarse_best, s_coarse_best, best_n)

    # ── 阶段3: θ+s精搜 ──
    # θ范围: 粗搜最佳±0.3°, 步长0.1°
    # s范围: 粗搜最佳±0.02, 步长0.002
    # 每组抽样K_fine次搜索(tx,ty)
    K_fine = min(3000, N_max // 20)

    theta_fine_range = np.arange(
        theta_coarse_best - 0.3,
        theta_coarse_best + 0.31,
        0.1
    )
    s_fine_range = np.arange(
        max(s_range[0], s_coarse_best - 0.02),
        min(s_range[1], s_coarse_best + 0.021),
        0.002
    )

    best_n_fine = 0
    best_params_fine = None

    for theta_deg in theta_fine_range:
        theta_rad = theta_deg * _DEGTORAD
        ct, st = math.cos(theta_rad), math.sin(theta_rad)
        Wf_rot_x = ct * Wf[:, 0] - st * Wf[:, 1]
        Wf_rot_y = st * Wf[:, 0] + ct * Wf[:, 1]

        for s_try in s_fine_range:
            max_n = 0
            best_tx, best_ty = 0.0, 0.0
            for _ in range(K_fine):
                i = rng.integers(0, N)
                j = rng.integers(0, M)
                tx = U[i, 0] - s_try * Wf_rot_x[j]
                ty = U[i, 1] - s_try * Wf_rot_y[j]
                Wt = np.column_stack([
                    s_try * Wf_rot_x + tx,
                    s_try * Wf_rot_y + ty
                ])
                n, rms = _count_inliers_fast(U, Wt, tau)
                if n > max_n:
                    max_n = n
                    best_tx, best_ty = tx, ty

            if max_n > best_n_fine:
                best_n_fine = max_n
                best_params_fine = (theta_deg, s_try, best_tx, best_ty)

    if not best_params_fine:
        # 回退到粗搜结果
        best_params_fine = best_params

    theta_best, s_best, tx_best, ty_best = best_params_fine
    best_theta = theta_best * _DEGTORAD

    # 用最佳变换重新统计1对1互斥内点
    Wt = _apply_similarity(Wf, s_best, best_theta, tx_best, ty_best)
    n_inliers, rms, inlier_mask = _count_inliers_1to1(U, Wt, tau)

    logger.debug("  V3.2最佳(1对1): s=%.4f θ=%.2f° tx=%.1f ty=%.1f n=%d rms=%.3f",
                 s_best, theta_best, tx_best, ty_best, n_inliers, rms)

    return s_best, best_theta, tx_best, ty_best, n_inliers, rms, inlier_mask


# ============================================================================
# V3.2 主类
# ============================================================================

class VectorMatch:
    """V3.2 向量匹配Plate Solving算法

    核心改动: 自适应噪声基线蒙特卡洛1点抽样
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._rng = np.random.default_rng(42)
        logger.info("VectorMatchV3.2初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

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
        """V3.2 向量匹配主入口"""
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

        # V3.2参数
        tau_coarse = max(1.0, 2.5 * s0)
        N_max = 100000
        N_warmup = 5000
        min_inliers = max(5, int(N_img * 0.1))
        logger.info("V3.2参数: tau_coarse=%.2f N_warmup=%d N_max=%d min_inliers=%d",
                     tau_coarse, N_warmup, N_max, min_inliers)

        best_mode = -1
        best_norm_score = -1.0
        best_result = None

        for mode in range(4):
            W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
            Wf = _apply_flip(W, mode)
            logger.info("翻转模式%d: 星表向量组 %d 颗", mode, len(Wf))

            # ── V3.2核心: 三阶段1点抽样法 ──
            s, theta, tx, ty, n_inliers, rms, inlier_mask = _mc_adaptive_sampling(
                U, Wf, tau_coarse, N_max, self._rng,
                N_warmup=N_warmup,
                snr_threshold=3.0,
                inlier_ratio=5.0,
                theta_constraint=2.0,
            )
            if n_inliers < min_inliers:
                logger.info("  模式%d: 抽样内点不足 n=%d < min=%d", mode, n_inliers, min_inliers)
                continue
            logger.info("  模式%d 粗匹配: s=%.4f theta=%.2f° tx=%.2f ty=%.2f n=%d rms=%.3f",
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
        """WCS参数提取 + 中心修正 + SVD精修"""
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

        # 重新投影 + V3.2精修
        W_new = _build_catalog_vectors(cat_ra, cat_dec, cur_ra, cur_dec)
        Wf_new = _apply_flip(W_new, cur_flip)

        min_inliers_refine = max(5, int(N_img * 0.1))

        refine_success = False
        for tau_mult in [1.0, 2.0, 3.0, 5.0]:
            tau_try = max(0.5, tau_mult * s0)
            s2, theta2, tx2, ty2, n2, rms2, mask2 = _mc_adaptive_sampling(
                U, Wf_new, tau_try, 20000, self._rng,
                N_warmup=2000,
                snr_threshold=3.0,
                inlier_ratio=3.0,
                theta_constraint=2.0,
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
                    logger.info("  中心修正后V3.2+SVD(tau=%.1fx): s=%.4f θ=%.2f° n=%d rms=%.3f",
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

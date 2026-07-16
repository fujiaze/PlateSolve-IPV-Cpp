# -*- coding: utf-8 -*-
"""
诊断脚本: 验证 ncorr (空间域归一化互相关) 是否能区分 180° 歧义。

核心改动 (用户要求: 角度直接求解, 0~360° 全范围, 不用固定角度列表):
  - 用 search_rotation_full 返回的 top-K 真实候选 (数据驱动, 任意角度)
  - 对每个候选计算 ncorr, 选 ncorr 最高的
  - 不再用 [0, 90, 180, 270] 固定角度列表

对每帧:
  1. 构建 f (top-N 图像星)
  2. 构建 g (top-N Gaia 星, N 与图像星数量匹配)
  3. search_rotation_full 返回 top-K 候选 (任意角度, 数据驱动)
  4. 对每个候选 θ: 旋转 g, 平移对齐, 计算 ncorr
  5. 检查 ncorr 最高的候选是否是正确角度 (vs header θ)

测试不同星点数量 (500, 2000, all) 找到能区分所有帧的配置。

运行:
    cd "f:\\Astro dev\\Astro CS Normalization Database"
    python lib/plate_solve/blind_index_v3/tests/diag_ncorr.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v2.python.io_wrappers import read_image, detect_stars, get_pointing_from_header, get_s0_from_header
from lib.plate_solve.blind_index_v2.python.logging_setup import setup_logging, get_logger
setup_logging()
logger = get_logger("ddspps.diag_ncorr")

from lib.plate_solve.blind_index_v3.python import density as density_mod
from lib.plate_solve.blind_index_v3.python import signal as signal_mod
from lib.plate_solve.blind_index_v3.python import phase_correlation as pc_mod


def _ncorr(f: np.ndarray, g_aligned: np.ndarray) -> float:
    """空间域归一化互相关 <f, g> / (||f|| · ||g||)。"""
    f_norm = float(np.linalg.norm(f))
    g_norm = float(np.linalg.norm(g_aligned))
    if f_norm < 1e-10 or g_norm < 1e-10:
        return 0.0
    return float(np.sum(f * g_aligned)) / (f_norm * g_norm)


def _eval_candidate_ncorr(
    f: np.ndarray,
    F_f: np.ndarray,
    g: np.ndarray,
    theta: float,
) -> tuple:
    """
    对候选角度 θ: 旋转 g, 相位相关求平移, ncorr 评分。

    Returns:
        (ncorr, dx_sub, dy_sub, peak)
    """
    g_rot = pc_mod.rotate_signal(g, theta)
    dx, dy, peak, _, C = pc_mod.phase_correlate_2d(F_f, g_rot)
    dx_sub, dy_sub = pc_mod.subpixel_refine(C, dx, dy)
    g_aligned = np.roll(g_rot, (int(round(dy_sub)), int(round(dx_sub))), axis=(0, 1))
    ncorr_val = _ncorr(f, g_aligned)
    return ncorr_val, dx_sub, dy_sub, peak


def diagnose_frame(image_path: str, name: str, header_theta: float = None) -> None:
    logger.info("=" * 70)
    logger.info("诊断帧: %s (θ_header=%.3f°)", name, header_theta if header_theta else -999)
    logger.info("=" * 70)

    # 1. 读取图像 + 检测星
    uint16_img, metadata = read_image(image_path)
    image_w, image_h = uint16_img.shape[1], uint16_img.shape[0]
    stars = detect_stars(uint16_img)
    s0 = get_s0_from_header(metadata)
    if s0 is None:
        logger.error("无法推导 s0")
        return

    # 2. 指向
    from lib.plate_solve.blind_index_v3.python.pipeline import _compute_image_center_pointing
    center_pt = _compute_image_center_pointing(metadata, image_w, image_h)
    if center_pt is not None:
        query_ra, query_dec = center_pt
    else:
        pt = get_pointing_from_header(metadata)
        if pt is None:
            logger.error("无指向信息")
            return
        query_ra, query_dec = pt

    # 3. FOV + Gaia
    fov_deg = density_mod.get_fov_from_header(metadata)
    if fov_deg <= 0:
        fov_deg = float(np.sqrt((image_w * s0 / 3600.0) ** 2 + (image_h * s0 / 3600.0) ** 2))
    density = density_mod.estimate_density(stars, fov_deg)
    gaia_ra, gaia_dec, gaia_mag = density_mod.load_gaia_subset(query_ra, query_dec, fov_deg, density.g_cutoff)
    logger.info("星点=%d, Gaia=%d, s0=%.4f, FOV=%.4f°", stars.count, len(gaia_ra), s0, fov_deg)

    # 4. 星点排序 (饱和优先, flux 降序)
    grid = 512
    sigma = 4.5
    sat = np.asarray(stars.saturated, dtype=np.int32)
    flux = np.asarray(stars.flux, dtype=np.float64)
    sat_priority = sat.astype(np.float64) * 1e18
    eff_flux = np.where(sat == 1, 0.0, np.maximum(flux, 0.0))
    sort_key = sat_priority + eff_flux
    order = np.argsort(-sort_key, kind="stable")

    # Gaia 星按 magnitude 排序 (亮星优先)
    if gaia_mag is not None and len(gaia_mag) > 0:
        mag_order = np.argsort(gaia_mag, kind="stable")
    else:
        mag_order = np.arange(len(gaia_ra))

    n_gaia_total = len(gaia_ra)

    # 5. 测试不同星点数量: 500, 2000, all
    test_sizes = [500, 2000, min(int(stars.count), 50000)]
    test_sizes = sorted(set([s for s in test_sizes if s > 0]))

    for n_signal in test_sizes:
        n_signal = min(n_signal, int(stars.count))
        signal_indices = order[:n_signal]
        signal_x = stars.x[signal_indices]
        signal_y = stars.y[signal_indices]
        f = signal_mod.build_image_signal(signal_x, signal_y, image_w, image_h, grid=grid, sigma=sigma)
        F_f = np.fft.fft2(f)

        # Gaia 星数量与图像星数量匹配 (top-N by magnitude)
        n_gaia_use = min(n_signal, n_gaia_total)
        gaia_indices = mag_order[:n_gaia_use]
        g_ra = gaia_ra[gaia_indices]
        g_dec = gaia_dec[gaia_indices]
        g = signal_mod.build_gaia_signal(
            g_ra, g_dec, query_ra, query_dec, s0,
            image_w=image_w, image_h=image_h, grid=grid, sigma=sigma, flip_mode=0,
        )

        logger.info("-" * 50)
        logger.info("N_stars=%d (f用%d, g用%d)", n_signal, n_signal, n_gaia_use)

        # 数据驱动搜索: search_rotation_full 返回 top-K 候选 (任意角度, 非固定)
        candidates = pc_mod.search_rotation_full(
            f, F_f, g, coarse_step=5.0, fine_step=0.5, fine_range=5.0, top_k=4,
        )
        if not candidates:
            logger.warning("  无候选")
            continue

        logger.info("  top-%d 候选 (数据驱动):", len(candidates))
        # 对每个候选计算 ncorr
        best_ncorr = -1.0
        best_theta_ncorr = -1.0
        for theta_c, dx_c, dy_c, peak_c in candidates:
            ncorr_val, _, _, _ = _eval_candidate_ncorr(f, F_f, g, theta_c)
            # 判断是否接近 header θ (考虑 180° 对称: mod180)
            if header_theta is not None:
                diff_mod180 = abs(((theta_c - header_theta + 180.0) % 360.0) - 180.0)
                is_correct = "✓正确" if diff_mod180 < 5.0 else "✗错误"
            else:
                is_correct = "?无header"
            logger.info("    θ=%.3f°: peak=%.4f, ncorr=%.4f %s",
                        theta_c, peak_c, ncorr_val, is_correct)
            if ncorr_val > best_ncorr:
                best_ncorr = ncorr_val
                best_theta_ncorr = theta_c

        # 判断 ncorr 是否选对 (mod180, 因 180° 对称)
        if header_theta is not None:
            diff_best = abs(((best_theta_ncorr - header_theta + 180.0) % 360.0) - 180.0)
            ncorr_correct = "✓能区分" if diff_best < 5.0 else "✗不能区分"
            logger.info("  → ncorr 选 θ=%.3f° (header θ=%.3f°, mod180差=%.3f°) %s",
                        best_theta_ncorr, header_theta, diff_best, ncorr_correct)


if __name__ == "__main__":
    frames = [
        ("M20_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights", "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts"), -89.311),
        ("LDN43", os.path.join(_PROJECT_ROOT, "testdata", "lights", "LDN43_LRGBH_flying_dutchman-20250503@032713-1200S-Red.fts"), -89.311),
        ("NGC247_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights", "NGC247_T2_flying_dutchman-20250816@034607-600S-Red.fts"), -89.168),
        ("NGC55_T3", os.path.join(_PROJECT_ROOT, "testdata", "lights", "NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts"), None),
    ]
    for name, path, theta_hdr in frames:
        if os.path.isfile(path):
            try:
                diagnose_frame(path, name, theta_hdr)
            except Exception as e:
                logger.error("%s 失败: %s", name, e)
        else:
            logger.warning("文件不存在: %s", path)

# -*- coding: utf-8 -*-
"""
诊断脚本: 验证 1D 相位相关 (Hann 窗) 是否能求解任意角度 (不受 4 重对称限制)。

对比:
  A. 直接 FFT2 (无窗) → angular_projection → 1D 相位相关
  B. windowed_fft2 (Hann 窗) → angular_projection → 1D 相位相关
  C. search_rotation_full (2D peak 评分, 当前方案)

预期:
  - A 受 4 重对称限制, θ 收敛到 0/90/180/270°
  - B 加 Hann 窗消除 4 重对称, 能求解任意角度
  - C 也受 4 重对称限制 (top-K 都是 90° 整数倍)

运行:
    cd "f:\\Astro dev\\Astro CS Normalization Database"
    python lib/plate_solve/blind_index_v3/tests/diag_1d_rotation.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v2.python.io_wrappers import read_image, detect_stars, get_pointing_from_header, get_s0_from_header
from lib.plate_solve.blind_index_v2.python.logging_setup import setup_logging, get_logger
setup_logging()
logger = get_logger("ddspps.diag_1d")

from lib.plate_solve.blind_index_v3.python import density as density_mod
from lib.plate_solve.blind_index_v3.python import signal as signal_mod
from lib.plate_solve.blind_index_v3.python import phase_correlation as pc_mod


def diagnose_frame(image_path: str, name: str) -> None:
    logger.info("=" * 70)
    logger.info("诊断帧: %s", name)
    logger.info("=" * 70)

    # 1. 读取图像 + 检测星
    uint16_img, metadata = read_image(image_path)
    image_w, image_h = uint16_img.shape[1], uint16_img.shape[0]
    stars = detect_stars(uint16_img)
    s0 = get_s0_from_header(metadata)
    if s0 is None:
        logger.error("无法推导 s0")
        return

    # 2. 指向 (从 WCS 计算图像中心)
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
    gaia_ra, gaia_dec, _ = density_mod.load_gaia_subset(query_ra, query_dec, fov_deg, density.g_cutoff)
    logger.info("星点=%d, Gaia=%d, s0=%.4f, FOV=%.4f°", stars.count, len(gaia_ra), s0, fov_deg)

    # 4. 构建信号
    grid = 512
    sigma = 4.5
    sat = np.asarray(stars.saturated, dtype=np.int32)
    flux = np.asarray(stars.flux, dtype=np.float64)
    sat_priority = sat.astype(np.float64) * 1e18
    eff_flux = np.where(sat == 1, 0.0, np.maximum(flux, 0.0))
    sort_key = sat_priority + eff_flux
    order = np.argsort(-sort_key, kind="stable")
    n_signal = min(500, int(stars.count))
    signal_indices = order[:n_signal]
    signal_x = stars.x[signal_indices]
    signal_y = stars.y[signal_indices]

    f = signal_mod.build_image_signal(signal_x, signal_y, image_w, image_h, grid=grid, sigma=sigma)
    g = signal_mod.build_gaia_signal(gaia_ra, gaia_dec, query_ra, query_dec, s0, image_w=image_w, image_h=image_h, grid=grid, sigma=sigma, flip_mode=0)

    # 5. 对比 3 种方法
    logger.info("-" * 50)
    logger.info("方法 A: 直接 FFT2 (无窗) → 1D 相位相关")
    F_f_raw = np.fft.fft2(f)
    F_g_raw = np.fft.fft2(g)
    phi_f_raw = pc_mod.angular_projection(np.abs(F_f_raw))
    phi_g_raw = pc_mod.angular_projection(np.abs(F_g_raw))
    theta_a, snr_a, _ = pc_mod.phase_correlate_1d(phi_f_raw, phi_g_raw)
    logger.info("  → θ=%.3f°, SNR=%.3f", theta_a, snr_a)

    logger.info("-" * 50)
    logger.info("方法 B: windowed_fft2 (Hann 窗) → 1D 相位相关")
    F_f_win = pc_mod.windowed_fft2(f)
    F_g_win = pc_mod.windowed_fft2(g)
    phi_f_win = pc_mod.angular_projection(np.abs(F_f_win))
    phi_g_win = pc_mod.angular_projection(np.abs(F_g_win))
    theta_b, snr_b, _ = pc_mod.phase_correlate_1d(phi_f_win, phi_g_win)
    logger.info("  → θ=%.3f°, SNR=%.3f", theta_b, snr_b)
    # 也检查 θ+180° (180° 共轭对称)
    theta_b_alt = (theta_b + 180.0) % 360.0
    logger.info("  → θ+180°=%.3f° (180° 共轭对称歧义)", theta_b_alt)

    logger.info("-" * 50)
    logger.info("方法 C: search_rotation_full (2D peak 评分, top-4)")
    candidates = pc_mod.search_rotation_full(f, F_f_raw, g, coarse_step=5.0, fine_step=0.5, fine_range=5.0, top_k=4)
    for i, (theta, dx, dy, peak) in enumerate(candidates):
        logger.info("  候选 %d: θ=%.3f°, dx=%.2f, dy=%.2f, peak=%.4f", i, theta, dx, dy, peak)

    # 6. 总结
    logger.info("-" * 50)
    logger.info("对比总结:")
    logger.info("  方法 A (无窗 1D): θ=%.3f° (SNR=%.2f) %s", theta_a, snr_a,
                "[90°整数倍]" if abs(theta_a % 90.0) < 1.0 else "[任意角度]")
    logger.info("  方法 B (Hann 1D): θ=%.3f° (SNR=%.2f) %s", theta_b, snr_b,
                "[90°整数倍]" if abs(theta_b % 90.0) < 1.0 else "[任意角度]")
    logger.info("  方法 C (2D top-4): θ=%.3f° %s", candidates[0][0],
                "[90°整数倍]" if abs(candidates[0][0] % 90.0) < 1.0 else "[任意角度]")


if __name__ == "__main__":
    frames = [
        ("M20_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights", "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts")),
        ("LDN43", os.path.join(_PROJECT_ROOT, "testdata", "lights", "LDN43_LRGBH_flying_dutchman-20250503@032713-1200S-Red.fts")),
        ("NGC247_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights", "NGC247_T2_flying_dutchman-20250816@034607-600S-Red.fts")),
    ]
    for name, path in frames:
        if os.path.isfile(path):
            try:
                diagnose_frame(path, name)
            except Exception as e:
                logger.error("%s 失败: %s", name, e)
        else:
            logger.warning("文件不存在: %s", path)

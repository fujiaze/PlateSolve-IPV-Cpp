# -*- coding: utf-8 -*-
"""
诊断脚本: 验证 1D 相位相关 (Fourier-Mellin) 能否直接求解任意旋转角。

用户要求: "角度应该能直接求解出来, 而不是几个固定角度。因为实际上角度可以是0-360。"

方法:
  1. f = build_image_signal (top-N 图像星)
  2. g = build_gaia_signal (top-N Gaia 星)
  3. F_f = windowed_fft2(f) (加 Hann 窗消除 4 重对称频谱泄漏)
  4. F_g = windowed_fft2(g)
  5. phi_f = angular_projection(|F_f|) (极坐标角向求和 → 1D 角度签名)
  6. phi_g = angular_projection(|F_g|)
  7. theta_1d, snr_1d = phase_correlate_1d(phi_f, phi_g) (1D 相位相关直接求解)
  8. 候选 = [theta_1d, theta_1d + 180°] (180° 共轭对称歧义)

预期:
  - theta_1d 应接近 header θ (mod 180°)
  - 能求解任意角度, 不固定在 0/90/180/270°

运行:
    cd "f:\\Astro dev\\Astro CS Normalization Database"
    python lib/plate_solve/blind_index_v3/tests/diag_1d_phase.py
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
logger = get_logger("ddspps.diag_1d_phase")

from lib.plate_solve.blind_index_v3.python import density as density_mod
from lib.plate_solve.blind_index_v3.python import signal as signal_mod
from lib.plate_solve.blind_index_v3.python import phase_correlation as pc_mod
from lib.plate_solve.blind_index_v3.python.pipeline import _compute_image_center_pointing


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

    # 4. 星点排序
    grid = 512
    sigma = 4.5
    sat = np.asarray(stars.saturated, dtype=np.int32)
    flux = np.asarray(stars.flux, dtype=np.float64)
    sat_priority = sat.astype(np.float64) * 1e18
    eff_flux = np.where(sat == 1, 0.0, np.maximum(flux, 0.0))
    sort_key = sat_priority + eff_flux
    order = np.argsort(-sort_key, kind="stable")

    if gaia_mag is not None and len(gaia_mag) > 0:
        mag_order = np.argsort(gaia_mag, kind="stable")
    else:
        mag_order = np.arange(len(gaia_ra))

    # 5. 测试不同 N: 500, 2000
    for n_signal in [500, 2000]:
        n_signal = min(n_signal, int(stars.count))
        signal_indices = order[:n_signal]
        signal_x = stars.x[signal_indices]
        signal_y = stars.y[signal_indices]
        f = signal_mod.build_image_signal(signal_x, signal_y, image_w, image_h, grid=grid, sigma=sigma)

        n_gaia_use = min(n_signal, len(gaia_ra))
        gaia_indices = mag_order[:n_gaia_use]
        g = signal_mod.build_gaia_signal(
            gaia_ra[gaia_indices], gaia_dec[gaia_indices], query_ra, query_dec, s0,
            image_w=image_w, image_h=image_h, grid=grid, sigma=sigma, flip_mode=0,
        )

        logger.info("-" * 50)
        logger.info("N_stars=%d (f用%d, g用%d)", n_signal, n_signal, n_gaia_use)

        # === 1D 相位相关直接求解角度 (Fourier-Mellin) ===
        # 加 Hann 窗消除方形网格 4 重对称频谱泄漏
        F_f = pc_mod.windowed_fft2(f)
        F_g = pc_mod.windowed_fft2(g)
        M_f = np.abs(F_f)
        M_g = np.abs(F_g)
        phi_f = pc_mod.angular_projection(M_f, n_bins=720)
        phi_g = pc_mod.angular_projection(M_g, n_bins=720)

        theta_1d, snr_1d, r = pc_mod.phase_correlate_1d(phi_f, phi_g)
        theta_180 = (theta_1d + 180.0) % 360.0

        # 判断正确性 (mod 180°, 因 180° 共轭对称)
        if header_theta is not None:
            header_mod360 = header_theta % 360.0
            diff1 = abs(((theta_1d - header_mod360 + 180.0) % 360.0) - 180.0)
            is_correct_1d = "✓" if diff1 < 5.0 else "✗"
        else:
            is_correct_1d = "?"

        logger.info("  1D相位相关: θ=%.3f°, SNR=%.3f, 180°候选=%.3f° %s",
                    theta_1d, snr_1d, theta_180, is_correct_1d)

        # === 对比: search_rotation_full 暴力搜索 (当前方法) ===
        F_f_plain = np.fft.fft2(f)
        candidates = pc_mod.search_rotation_full(
            f, F_f_plain, g, coarse_step=5.0, fine_step=0.5, fine_range=5.0, top_k=4,
        )
        cand_str = ", ".join("θ=%.2f°/peak=%.3f" % (c[0], c[3]) for c in candidates)
        logger.info("  暴力搜索top4: %s", cand_str)


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

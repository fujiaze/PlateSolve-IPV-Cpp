# -*- coding: utf-8 -*-
"""
DD-SPPS 诊断脚本: 用 header WCS 直接验证星点匹配
功能: 读取图像 + header WCS + Gaia 星表, 用 header WCS 预测星点位置, 检查是否能匹配 Gaia 星
用途: 隔离 verify_wcs / build_wcs 的 bug——如果 header WCS 都不能匹配, 说明坐标转换有 bug

运行:
    cd "f:\\Astro dev\\Astro CS Normalization Database"
    python lib/plate_solve/blind_index_v3/tests/diag_header_wcs.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v2.python.io_wrappers import (
    read_image, detect_stars, get_pointing_from_header, get_s0_from_header,
)
from lib.plate_solve.blind_index_v2.python.logging_setup import setup_logging, get_logger
from lib.plate_solve.blind_index_v3.python import density as density_mod

setup_logging()
logger = get_logger("ddspps.diag_header_wcs")


def haversine_arcsec(ra1, dec1, ra2, dec2):
    """haversine 球面角距离 (角秒), 向量化。"""
    import numpy as _np
    ra1 = _np.asarray(ra1) * math.pi / 180.0
    dec1 = _np.asarray(dec1) * math.pi / 180.0
    ra2 = _np.asarray(ra2) * math.pi / 180.0
    dec2 = _np.asarray(dec2) * math.pi / 180.0
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = _np.sin(ddec / 2.0) ** 2 + _np.cos(dec1) * _np.cos(dec2) * _np.sin(dra / 2.0) ** 2
    a = _np.clip(a, 0.0, 1.0)
    return 2.0 * _np.arcsin(_np.sqrt(a)) * (180.0 / math.pi) * 3600.0


def diag_frame(frame_path: str, name: str):
    """对单帧用 header WCS 直接验证。"""
    print(f"\n{'='*70}")
    print(f"诊断帧: {name}")
    print(f"路径: {frame_path}")
    print(f"{'='*70}")

    # 1. 读取图像
    uint16_img, metadata = read_image(frame_path)
    image_w = uint16_img.shape[1]
    image_h = uint16_img.shape[0]
    print(f"图像尺寸: {image_w}x{image_h}")

    # 2. 检查 header WCS
    wcs = metadata.wcs if metadata is not None else None
    if wcs is None or not wcs.has_wcs:
        print("[SKIP] 无 header WCS, 跳过")
        return
    crval1 = float(wcs.crval1)
    crval2 = float(wcs.crval2)
    crpix1 = float(wcs.crpix1)
    crpix2 = float(wcs.crpix2)
    cd11 = float(wcs.cd1_1)
    cd12 = float(wcs.cd1_2)
    cd21 = float(wcs.cd2_1)
    cd22 = float(wcs.cd2_2)
    s0 = float(wcs.pixel_scale)
    print(f"Header WCS:")
    print(f"  CRVAL=({crval1:.6f}, {crval2:.6f})")
    print(f"  CRPIX=({crpix1:.3f}, {crpix2:.3f})")
    print(f"  CD=[[{cd11:.6e}, {cd12:.6e}], [{cd21:.6e}, {cd22:.6e}]]")
    print(f"  s0={s0:.4f}\"/px")

    # 3. 星点检测
    stars = detect_stars(uint16_img)
    print(f"检测星点: {stars.count} 颗")
    if stars.count < 5:
        print("[SKIP] 星点不足")
        return

    # 4. 加载 Gaia 子集 (用 header CRVAL 作为指向)
    fov_deg = density_mod.get_fov_from_header(metadata)
    density = density_mod.estimate_density(stars, fov_deg)
    g_cutoff = density.g_cutoff
    print(f"密度: ρ={density.rho:.4f}, N_bright={density.n_bright}, G_cutoff={g_cutoff:.2f}")

    gaia_ra, gaia_dec, gaia_mag = density_mod.load_gaia_subset(
        crval1, crval2, fov_deg, g_cutoff,
    )
    print(f"Gaia 星: {len(gaia_ra)} 颗")
    if len(gaia_ra) < 5:
        print("[SKIP] Gaia 星不足")
        return

    # 5. 用 header WCS 直接预测星点位置 (原始像素 → 天球)
    # header WCS: ra = crval1 + (x - crpix1)*cd11 + (y - crpix2)*cd12
    #             dec = crval2 + (x - crpix1)*cd21 + (y - crpix2)*cd22
    sx = np.asarray(stars.x, dtype=np.float64)
    sy = np.asarray(stars.y, dtype=np.float64)
    dx_pix = sx - crpix1  # 原始像素位移
    dy_pix = sy - crpix2
    ra_pred_header = crval1 + dx_pix * cd11 + dy_pix * cd12
    dec_pred_header = crval2 + dx_pix * cd21 + dy_pix * cd22

    # KD-tree 匹配
    gaia_pts = np.column_stack([gaia_ra, gaia_dec])
    tree = cKDTree(gaia_pts)
    pred_pts = np.column_stack([ra_pred_header, dec_pred_header])
    dists_deg, idxs = tree.query(pred_pts, k=1)
    sep_arcsec = haversine_arcsec(ra_pred_header, dec_pred_header, gaia_ra[idxs], gaia_dec[idxs])

    tol_arcsec = 3.0 * 0.5 * s0  # 3×sigma_pos×s0

    # 5a. 全部星匹配统计
    n_inliers_all = int(np.sum(sep_arcsec < tol_arcsec))
    print(f"\n--- header WCS 直接预测 (全部星) ---")
    print(f"  容差: {tol_arcsec:.3f}\" (3×0.5×s0)")
    print(f"  n_inliers (全部): {n_inliers_all}/{stars.count}")
    print(f"  中位分离: {np.median(sep_arcsec):.3f}\"")
    print(f"  最小分离: {sep_arcsec.min():.3f}\"")

    # 5b. 只对 top-N 亮星匹配 (饱和星 + 高 flux 星)
    flux = np.asarray(stars.flux, dtype=np.float64)
    sat = np.asarray(stars.saturated, dtype=np.int32)
    sat_priority = sat.astype(np.float64) * 1e18
    eff_flux = np.where(sat == 1, 0.0, np.maximum(flux, 0.0))
    sort_key = sat_priority + eff_flux
    order = np.argsort(-sort_key, kind="stable")
    for top_n in (50, 100, 200, 500):
        n_top = min(top_n, stars.count)
        top_idx = order[:n_top]
        top_sep = sep_arcsec[top_idx]
        n_inl_top = int(np.sum(top_sep < tol_arcsec))
        print(f"  top-{top_n} 亮星: n_inliers={n_inl_top}/{n_top} ({100*n_inl_top/n_top:.1f}%), "
              f"中位分离={np.median(top_sep):.3f}\", 最小={top_sep.min():.3f}\"")

    # 5c. 放宽容差测试 (5", 10", 30")
    for tol in (1.45, 3.0, 5.0, 10.0, 30.0):
        n_inl = int(np.sum(sep_arcsec < tol))
        n_top100_inl = int(np.sum(sep_arcsec[order[:100]] < tol))
        print(f"  容差 {tol:.2f}\": 全部 {n_inl}/{stars.count}, top-100 {n_top100_inl}/100")

    # 前 5 个亮星样本
    print(f"  样本 (top-5 亮星):")
    for i in order[:5]:
        print(f"    star[{i}]: x={sx[i]:.1f}, y={sy[i]:.1f}, flux={flux[i]:.1f}, sat={sat[i]}, "
              f"ra_pred={ra_pred_header[i]:.6f}, dec_pred={dec_pred_header[i]:.6f}, "
              f"nearest_gaia=({gaia_ra[idxs[i]]:.6f}, {gaia_dec[idxs[i]]:.6f}), "
              f"dist={sep_arcsec[i]:.3f}\"")

    # 6. 模拟 verify_wcs 的坐标转换 (网格坐标 → 天球)
    # verify_wcs: x_grid = x * scale, scale = grid / max(w, h)
    #             dx_pix = (x_grid - crpix_grid) * scale_factor
    #             ra = crval + dx_pix * cd
    grid = 512
    image_size = max(float(image_w), float(image_h))
    scale = float(grid) / image_size
    scale_factor = image_size / float(grid)  # = 1/scale

    # 用 header CD, 但 CRPIX = grid/2 (网格中心)
    # 这测试 build_wcs 的约定是否与 header WCS 一致
    crpix_grid = grid / 2.0
    x_grid = sx * scale
    y_grid = sy * scale
    dx_grid = (x_grid - crpix_grid) * scale_factor  # 转回原始像素
    dy_grid = (y_grid - crpix_grid) * scale_factor

    # 用 header CD 矩阵预测
    ra_pred_v2 = crval1 + dx_grid * cd11 + dy_grid * cd12
    dec_pred_v2 = crval2 + dx_grid * cd21 + dy_grid * cd22

    # 但这里有个问题: header CRPIX 是原始像素坐标 (如 2048.5), 而网格 CRPIX = grid/2 = 256
    # 如果图像中心 (w/2, h/2) 对应 header CRPIX, 则:
    # - 原始像素: dx = x - crpix1 (crpix1 = w/2)
    # - 网格: x_grid = x * scale, dx_grid = (x_grid - grid/2) * scale_factor = (x*scale - grid/2) / scale = x - grid/2/scale = x - image_size/2
    # 如果 image_w = image_h = image_size, 则 dx_grid = x - image_size/2 = x - w/2 = x - crpix1 (当 crpix1 = w/2)
    print(f"\n--- verify_wcs 坐标转换测试 (网格 → 天球, 用 header CD) ---")
    print(f"  scale={scale}, scale_factor={scale_factor}")
    print(f"  crpix1(header)={crpix1}, image_w/2={image_w/2.0}")
    print(f"  dx_pix[0] (header)={sx[0]-crpix1:.3f}, dx_grid[0]={dx_grid[0]:.3f}")
    # 如果 crpix1 != w/2, 会有偏差
    crpix_offset = crpix1 - image_w / 2.0
    print(f"  CRPIX offset (crpix1 - w/2): {crpix_offset:.3f} px (如果非0, verify_wcs 会有系统偏差)")

    sep_v2 = haversine_arcsec(ra_pred_v2, dec_pred_v2, gaia_ra[idxs], gaia_dec[idxs])
    n_inliers_v2 = int(np.sum(sep_v2 < tol_arcsec))
    print(f"  n_inliers (verify_wcs 坐标转换): {n_inliers_v2}/{stars.count}")
    print(f"  中位分离: {np.median(sep_v2):.3f}\"")

    # 7. 测试 build_wcs 的 CD 公式 (含 cos_dec)
    # build_wcs: cd11 = s_deg * cos(θ) / cos(dec)
    #            cd12 = -s_deg * sin(θ) / cos(dec)
    # header θ = atan2(cd21, cd22)
    theta = math.degrees(math.atan2(cd21, cd22))
    cos_dec = math.cos(crval2 * math.pi / 180.0)
    s_deg = s0 / 3600.0
    cd11_b = s_deg * math.cos(theta * math.pi / 180.0) / cos_dec
    cd12_b = -s_deg * math.sin(theta * math.pi / 180.0) / cos_dec
    cd21_b = s_deg * math.sin(theta * math.pi / 180.0)
    cd22_b = s_deg * math.cos(theta * math.pi / 180.0)

    print(f"\n--- build_wcs CD 公式 vs header CD (θ={theta:.3f}°, 含 cos_dec) ---")
    print(f"  build_wcs CD: [[{cd11_b:.6e}, {cd12_b:.6e}], [{cd21_b:.6e}, {cd22_b:.6e}]]")
    print(f"  header    CD: [[{cd11:.6e}, {cd12:.6e}], [{cd21:.6e}, {cd22:.6e}]]")
    print(f"  cd11 ratio: {cd11_b/cd11 if cd11 != 0 else 'inf':.4f}")
    print(f"  cd12 ratio: {cd12_b/cd12 if cd12 != 0 else 'inf':.4f}")
    print(f"  cd21 ratio: {cd21_b/cd21 if cd21 != 0 else 'inf':.4f}")
    print(f"  cd22 ratio: {cd22_b/cd22 if cd22 != 0 else 'inf':.4f}")

    # 测试不含 cos_dec 的 CD
    cd11_nc = s_deg * math.cos(theta * math.pi / 180.0)
    cd12_nc = -s_deg * math.sin(theta * math.pi / 180.0)
    cd21_nc = s_deg * math.sin(theta * math.pi / 180.0)
    cd22_nc = s_deg * math.cos(theta * math.pi / 180.0)
    print(f"\n--- build_wcs CD 公式 (无 cos_dec) ---")
    print(f"  no_cosdec CD: [[{cd11_nc:.6e}, {cd12_nc:.6e}], [{cd21_nc:.6e}, {cd22_nc:.6e}]]")
    print(f"  cd11 ratio: {cd11_nc/cd11 if cd11 != 0 else 'inf':.4f}")
    print(f"  cd12 ratio: {cd12_nc/cd12 if cd12 != 0 else 'inf':.4f}")
    print(f"  cd21 ratio: {cd21_nc/cd21 if cd21 != 0 else 'inf':.4f}")
    print(f"  cd22 ratio: {cd22_nc/cd22 if cd22 != 0 else 'inf':.4f}")


def main():
    frames = [
        ("LDN43", os.path.join(_PROJECT_ROOT, "testdata", "lights",
                               "LDN43_LRGBH_flying_dutchman-20250503@032713-1200S-Red.fts")),
        ("M20_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights",
                                "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts")),
        ("NGC247_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights",
                                   "NGC247_T2_flying_dutchman-20250816@034607-600S-Red.fts")),
    ]
    for name, path in frames:
        if os.path.isfile(path):
            try:
                diag_frame(path, name)
            except Exception as e:
                import traceback
                print(f"[ERROR] {name}: {e}")
                print(traceback.format_exc())
        else:
            print(f"[SKIP] {name}: 文件不存在")


if __name__ == "__main__":
    main()

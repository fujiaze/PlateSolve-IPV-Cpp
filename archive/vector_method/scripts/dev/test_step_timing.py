"""
V2/V3 solve 全流程分步计时

对solve()的每个步骤单独计时, 定位耗时瓶颈:
  1. 星检测
  2. 向量构建 (V2: _build_image_vectors / V3: _build_gold_and_validation_pools)
  3. Gaia查询 (bisection_mag_limit)
  4. RANSAC粗匹配 (4个翻转模式)
  5. SVD精修 (4个翻转模式)
  6. 中心修正+精修 (_extract_wcs_and_converge)
"""

import os
import sys
import time
import logging
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2
from vector_match_v3 import VectorMatch as VectorMatchV3
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# 只测2帧: Red(星多) + Blue(星少)
TEST_FRAMES = [
    r"testdata\lights\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@061821-180S-Blue.fts",
]

GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def detect_stars(img_data):
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    return detector.detect_ex(img_data)


def run_step_by_step(frame_path, version="v2"):
    """对单帧分步计时"""
    full_path = os.path.join(PROJECT_ROOT, frame_path)
    fname = os.path.basename(frame_path)
    print(f"\n{'='*80}")
    print(f"分步计时 [{version.upper()}]: {fname}")
    print(f"{'='*80}")

    # 1. 读取FITS
    reader = ImageReader()
    img = reader.read(full_path)
    width = img.width
    height = img.height

    center_ra = 0.0; center_dec = 0.0
    focal_length = 200.0; pixel_size = 6.0

    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        center_ra = img.metadata.wcs.crval1
        center_dec = img.metadata.wcs.crval2
    if img.metadata.observation:
        if img.metadata.observation.focallen is not None:
            focal_length = img.metadata.observation.focallen
        if img.metadata.observation.xpixsz is not None:
            pixel_size = img.metadata.observation.xpixsz

    if center_ra == 0.0 and center_dec == 0.0:
        for kw in img.keywords:
            name = kw.name.upper()
            if name in ("OBJCTRA", "RA"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                    if len(parts) >= 3:
                        center_ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
            elif name in ("OBJCTDEC", "DEC"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                    if len(parts) >= 3:
                        sign = -1 if parts[0].startswith("-") else 1
                        center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)

    print(f"  图像: {width}x{height}, RA={center_ra:.4f}, Dec={center_dec:.4f}")
    print(f"  焦距={focal_length}mm, 像素={pixel_size}um")

    # 2. 星检测
    t0 = time.perf_counter()
    det_result = detect_stars(img.data)
    t_detect = time.perf_counter() - t0
    n_stars = det_result.count
    n_sat = int(np.sum(det_result.saturated))
    print(f"  [步骤1] 星检测: {t_detect:.2f}s ({n_stars}颗, 饱和{n_sat})")

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)

    s0 = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0

    if version == "v2":
        return _run_v2_steps(img_x, img_y, img_flux, img_saturated,
                            center_ra, center_dec, focal_length, pixel_size, width, height,
                            s0, fov_diag, radius_deg, n_stars, n_sat, t_detect)
    else:
        return _run_v3_steps(img_x, img_y, img_flux, img_saturated,
                            center_ra, center_dec, focal_length, pixel_size, width, height,
                            s0, fov_diag, radius_deg, n_stars, n_sat, t_detect)


def _run_v2_steps(img_x, img_y, img_flux, img_saturated,
                   center_ra, center_dec, focal_length, pixel_size, width, height,
                   s0, fov_diag, radius_deg, n_stars, n_sat, t_detect):
    """V2分步计时"""
    from vector_match_v2 import (
        _build_image_vectors, bisection_mag_limit, _build_catalog_vectors,
        _apply_flip, _ransac_rigid_v2, _find_fine_correspondences,
        _iterative_svd_refine, _compute_normalized_score, _apply_similarity,
        _count_inliers_1to1, GaiaClientPy,
    )
    from scipy.spatial import cKDTree

    vm = VectorMatchV2(GAIA_DATA_DIR, db_type=0)

    # 步骤2: 构建图像向量
    t0 = time.perf_counter()
    U, N_img, n_sat_v2, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_saturated, s0, width, height)
    t_build_u = time.perf_counter() - t0
    print(f"  [步骤2] 构建图像向量: {t_build_u:.3f}s (N_img={N_img}, 饱和={n_sat_v2})")

    # 步骤3: Gaia查询
    if n_sat_v2 >= 50:
        N_gaia = math.ceil(1.5 * n_sat_v2)
    else:
        N_gaia = 150

    t0 = time.perf_counter()
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        vm._gaia, center_ra, center_dec, radius_deg, N_gaia)
    t_gaia = time.perf_counter() - t0
    print(f"  [步骤3] Gaia查询: {t_gaia:.3f}s (M={M}, 极限星等={mag_limit:.2f})")

    # 步骤4-6: 4个翻转模式
    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_img * 0.2))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1

    t_total_ransac = 0
    t_total_svd = 0
    t_total_build_w = 0
    best_mode = -1
    best_norm_score = -1.0
    best_result = None

    for mode in range(4):
        # 构建星表向量
        t0 = time.perf_counter()
        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
        Wf = _apply_flip(W, mode)
        t_bw = time.perf_counter() - t0
        t_total_build_w += t_bw

        # RANSAC
        t0 = time.perf_counter()
        s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
            U, Wf, tau_coarse, K, min_inliers, vm._rng,
            candidate_radius_coarse, sparsity)
        t_r = time.perf_counter() - t0
        t_total_ransac += t_r

        if n_inliers < min_inliers:
            print(f"    模式{mode}: RANSAC {t_r:.3f}s (skip, n={n_inliers})")
            continue

        # 精细候选对
        pairs_fine = _find_fine_correspondences(U, Wf, s, theta, tx, ty, tau_coarse)

        # SVD精修
        t0 = time.perf_counter()
        s_ref, theta_ref, tx_ref, ty_ref, n_ref, rms_ref, mask_ref = \
            _iterative_svd_refine(U, Wf, inlier_mask, s0, s, theta, tx, ty, max_iter=10)
        t_svd = time.perf_counter() - t0
        t_total_svd += t_svd

        if n_ref >= min_inliers:
            s, theta, tx, ty = s_ref, theta_ref, tx_ref, ty_ref
            n_inliers, rms, inlier_mask = n_ref, rms_ref, mask_ref

        norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)
        print(f"    模式{mode}: RANSAC {t_r:.3f}s + SVD {t_svd:.3f}s = {t_r+t_svd:.3f}s "
              f"(n={n_inliers} rms={rms:.3f} score={norm_score:.4f})")

        if norm_score > best_norm_score:
            best_norm_score = norm_score
            best_mode = mode
            best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

    print(f"  [步骤4] 构建星表向量(4模式): {t_total_build_w:.3f}s")
    print(f"  [步骤5] RANSAC粗匹配(4模式): {t_total_ransac:.3f}s")
    print(f"  [步骤6] SVD精修(4模式): {t_total_svd:.3f}s")

    # 步骤7: 中心修正+精修
    t0 = time.perf_counter()
    if best_mode >= 0 and best_norm_score >= 0.10:
        s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result
        result = vm._extract_wcs_and_converge(
            s, theta, tx, ty, best_mode, s0,
            center_ra, center_dec, width, height,
            U, Wf, inlier_mask, N_img, M,
            cat_ra, cat_dec, cat_mag,
            fov_diag, sparsity,
        )
    else:
        result = None
    t_converge = time.perf_counter() - t0
    print(f"  [步骤7] 中心修正+精修: {t_converge:.3f}s")

    t_solve_total = t_build_u + t_gaia + t_total_build_w + t_total_ransac + t_total_svd + t_converge
    print(f"\n  --- 耗时汇总 ---")
    print(f"  星检测:       {t_detect:.2f}s ({t_detect/(t_detect+t_solve_total)*100:.1f}%)")
    print(f"  构建图像向量: {t_build_u:.3f}s ({t_build_u/t_solve_total*100:.1f}%)")
    print(f"  Gaia查询:     {t_gaia:.3f}s ({t_gaia/t_solve_total*100:.1f}%)")
    print(f"  构建星表向量: {t_total_build_w:.3f}s ({t_total_build_w/t_solve_total*100:.1f}%)")
    print(f"  RANSAC粗匹配: {t_total_ransac:.3f}s ({t_total_ransac/t_solve_total*100:.1f}%)")
    print(f"  SVD精修:      {t_total_svd:.3f}s ({t_total_svd/t_solve_total*100:.1f}%)")
    print(f"  中心修正+精修:{t_converge:.3f}s ({t_converge/t_solve_total*100:.1f}%)")
    print(f"  solve总计:    {t_solve_total:.2f}s")
    print(f"  全流程总计:   {t_detect+t_solve_total:.2f}s")

    if result:
        print(f"  结果: RMS={result.rms_px:.3f}px, matched={result.matched_count}")

    return {
        't_detect': t_detect, 't_build_u': t_build_u, 't_gaia': t_gaia,
        't_build_w': t_total_build_w, 't_ransac': t_total_ransac,
        't_svd': t_total_svd, 't_converge': t_converge,
        't_solve': t_solve_total, 't_total': t_detect + t_solve_total,
    }


def _run_v3_steps(img_x, img_y, img_flux, img_saturated,
                   center_ra, center_dec, focal_length, pixel_size, width, height,
                   s0, fov_diag, radius_deg, n_stars, n_sat, t_detect):
    """V3分步计时"""
    from vector_match_v3 import (
        _build_gold_and_validation_pools, bisection_mag_limit,
        _build_catalog_vectors, _apply_flip, _ransac_v3,
        _find_fine_correspondences, _iterative_svd_refine,
        _compute_normalized_score, GaiaClientPy,
    )

    vm = VectorMatchV3(GAIA_DATA_DIR, db_type=0)

    # 步骤2: 构建黄金池+验证池
    t0 = time.perf_counter()
    U_gold, N_gold, n_sat_v3, sparsity, U_val, N_val = _build_gold_and_validation_pools(
        img_x, img_y, img_flux, img_saturated, s0, width, height, 1000)
    t_build_u = time.perf_counter() - t0
    print(f"  [步骤2] 构建黄金池+验证池: {t_build_u:.3f}s (N_gold={N_gold}, N_val={N_val}, 饱和={n_sat_v3})")

    # 步骤3: Gaia查询
    if n_sat_v3 >= 50:
        N_gaia = math.ceil(1.5 * n_sat_v3)
    else:
        N_gaia = 150

    t0 = time.perf_counter()
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        vm._gaia, center_ra, center_dec, radius_deg, N_gaia)
    t_gaia = time.perf_counter() - t0
    print(f"  [步骤3] Gaia查询: {t_gaia:.3f}s (M={M}, 极限星等={mag_limit:.2f})")

    # 步骤4-6: 4个翻转模式
    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_val * 0.03))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1
    fov_diag_arcsec = fov_diag * 3600.0

    t_total_ransac = 0
    t_total_svd = 0
    t_total_build_w = 0
    best_mode = -1
    best_norm_score = -1.0
    best_result = None

    for mode in range(4):
        # 构建星表向量
        t0 = time.perf_counter()
        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
        Wf = _apply_flip(W, mode)
        t_bw = time.perf_counter() - t0
        t_total_build_w += t_bw

        # RANSAC
        t0 = time.perf_counter()
        s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_v3(
            U_gold, U_val, Wf, tau_coarse, K, min_inliers, vm._rng,
            candidate_radius_coarse, sparsity,
            sin_tau=0.2, fov_diag_arcsec=fov_diag_arcsec)
        t_r = time.perf_counter() - t0
        t_total_ransac += t_r

        if n_inliers < min_inliers:
            print(f"    模式{mode}: RANSAC {t_r:.3f}s (skip, n={n_inliers})")
            continue

        # 精细候选对
        pairs_fine = _find_fine_correspondences(U_val, Wf, s, theta, tx, ty, tau_coarse)

        # SVD精修
        t0 = time.perf_counter()
        s_ref, theta_ref, tx_ref, ty_ref, n_ref, rms_ref, mask_ref = \
            _iterative_svd_refine(U_val, Wf, inlier_mask, s0, s, theta, tx, ty,
                                  max_iter=10, sin_tau=0.1)
        t_svd = time.perf_counter() - t0
        t_total_svd += t_svd

        if n_ref >= min_inliers:
            s, theta, tx, ty = s_ref, theta_ref, tx_ref, ty_ref
            n_inliers, rms, inlier_mask = n_ref, rms_ref, mask_ref

        norm_score = _compute_normalized_score(n_inliers, rms, N_val, M, tau_coarse)
        print(f"    模式{mode}: RANSAC {t_r:.3f}s + SVD {t_svd:.3f}s = {t_r+t_svd:.3f}s "
              f"(n={n_inliers} rms={rms:.3f} score={norm_score:.4f})")

        if norm_score > best_norm_score:
            best_norm_score = norm_score
            best_mode = mode
            best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

    print(f"  [步骤4] 构建星表向量(4模式): {t_total_build_w:.3f}s")
    print(f"  [步骤5] RANSAC粗匹配(4模式): {t_total_ransac:.3f}s")
    print(f"  [步骤6] SVD精修(4模式): {t_total_svd:.3f}s")

    # 步骤7: 中心修正+精修
    t0 = time.perf_counter()
    if best_mode >= 0 and best_norm_score >= 0.10:
        s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result
        if s >= 0.9 and s <= 1.1:
            result = vm._extract_wcs_and_converge(
                s, theta, tx, ty, best_mode, s0,
                center_ra, center_dec, width, height,
                U_val, Wf, inlier_mask, N_val, M,
                cat_ra, cat_dec, cat_mag,
                fov_diag, sparsity, U_gold,
            )
        else:
            result = None
    else:
        result = None
    t_converge = time.perf_counter() - t0
    print(f"  [步骤7] 中心修正+精修: {t_converge:.3f}s")

    t_solve_total = t_build_u + t_gaia + t_total_build_w + t_total_ransac + t_total_svd + t_converge
    print(f"\n  --- 耗时汇总 ---")
    print(f"  星检测:       {t_detect:.2f}s ({t_detect/(t_detect+t_solve_total)*100:.1f}%)")
    print(f"  构建向量池:   {t_build_u:.3f}s ({t_build_u/t_solve_total*100:.1f}%)")
    print(f"  Gaia查询:     {t_gaia:.3f}s ({t_gaia/t_solve_total*100:.1f}%)")
    print(f"  构建星表向量: {t_total_build_w:.3f}s ({t_total_build_w/t_solve_total*100:.1f}%)")
    print(f"  RANSAC粗匹配: {t_total_ransac:.3f}s ({t_total_ransac/t_solve_total*100:.1f}%)")
    print(f"  SVD精修:      {t_total_svd:.3f}s ({t_total_svd/t_solve_total*100:.1f}%)")
    print(f"  中心修正+精修:{t_converge:.3f}s ({t_converge/t_solve_total*100:.1f}%)")
    print(f"  solve总计:    {t_solve_total:.2f}s")
    print(f"  全流程总计:   {t_detect+t_solve_total:.2f}s")

    if result:
        print(f"  结果: RMS={result.rms_px:.3f}px, matched={result.matched_count}")

    return {
        't_detect': t_detect, 't_build_u': t_build_u, 't_gaia': t_gaia,
        't_build_w': t_total_build_w, 't_ransac': t_total_ransac,
        't_svd': t_total_svd, 't_converge': t_converge,
        't_solve': t_solve_total, 't_total': t_detect + t_solve_total,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)

    for frame in TEST_FRAMES:
        for ver in ["v2", "v3"]:
            try:
                run_step_by_step(frame, version=ver)
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()

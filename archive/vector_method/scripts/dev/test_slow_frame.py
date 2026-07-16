"""
单帧详细计时 - 用全量测试中的慢帧, 定位真实瓶颈

测试帧: panel3 Red帧 (全量测试中V2耗时251-282秒)
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
from vector_match_v2 import (
    _build_image_vectors, bisection_mag_limit, _build_catalog_vectors,
    _apply_flip, _ransac_rigid_v2, _find_fine_correspondences,
    _iterative_svd_refine, _compute_normalized_score, GaiaClientPy,
)
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# 全量测试中的慢帧
FRAME = r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"
GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    full_path = os.path.join(PROJECT_ROOT, FRAME)
    fname = os.path.basename(FRAME)
    print(f"帧: {fname}")

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

    print(f"图像: {width}x{height}, RA={center_ra:.4f}, Dec={center_dec:.4f}")
    print(f"焦距={focal_length}mm, 像素={pixel_size}um")

    # 2. 星检测
    t0 = time.perf_counter()
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)
    t_detect = time.perf_counter() - t0
    n_stars = det_result.count
    n_sat = int(np.sum(det_result.saturated))
    print(f"[1] 星检测: {t_detect:.2f}s ({n_stars}颗, 饱和{n_sat})")

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)

    s0 = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0
    print(f"s0={s0:.4f}, FOV对角线={fov_diag:.2f}度, 查询半径={radius_deg:.2f}度")

    # 3. 构建图像向量
    t0 = time.perf_counter()
    U, N_img, n_sat_v2, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_saturated, s0, width, height)
    t_build_u = time.perf_counter() - t0
    print(f"[2] 构建图像向量: {t_build_u:.3f}s (N_img={N_img}, 饱和={n_sat_v2})")

    # 4. Gaia查询 - 详细计时每次cone_search
    if n_sat_v2 >= 50:
        N_gaia = math.ceil(1.5 * n_sat_v2)
    else:
        N_gaia = 150

    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)
    mag_low, mag_high = 6.0, 22.0
    target_count = N_gaia
    target_high = int(target_count * 1.1)
    tolerance = 0.1

    print(f"\n[3] Gaia查询 (二分搜索, 最多30次cone_search)")
    print(f"    目标星数: {target_count}, 上限: {target_high}")

    t_gaia_total = 0
    n_iterations = 0
    for _ in range(30):
        t0 = time.perf_counter()
        mid = (mag_low + mag_high) / 2.0
        ra, dec, mag = gaia.cone_search(center_ra, center_dec, radius_deg, mid)
        t_iter = time.perf_counter() - t0
        t_gaia_total += t_iter
        n_iterations += 1
        count = len(ra)
        print(f"    迭代{n_iterations}: mag_limit={mid:.2f}, 星数={count}, 耗时={t_iter:.2f}s (累计{t_gaia_total:.2f}s)")

        if count < target_count:
            mag_low = mid
        elif count > target_high:
            mag_high = mid
        else:
            break
        if (mag_high - mag_low) <= tolerance:
            break

    print(f"    Gaia查询总耗时: {t_gaia_total:.2f}s ({n_iterations}次cone_search)")

    # 5. RANSAC (只测模式0)
    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_img * 0.2))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1

    W = _build_catalog_vectors(ra, dec, center_ra, center_dec)
    Wf = _apply_flip(W, 0)
    print(f"\n[4] 星表向量: {len(Wf)}颗")

    t0 = time.perf_counter()
    s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
        U, Wf, tau_coarse, K, min_inliers, np.random.default_rng(42),
        candidate_radius_coarse, sparsity)
    t_ransac = time.perf_counter() - t0
    print(f"[5] RANSAC(模式0): {t_ransac:.2f}s (n={n_inliers}, rms={rms:.3f})")

    # 汇总
    print(f"\n{'='*60}")
    print(f"耗时汇总:")
    print(f"  星检测:     {t_detect:.2f}s ({t_detect/(t_detect+t_build_u+t_gaia_total+t_ransac)*100:.1f}%)")
    print(f"  构建向量:   {t_build_u:.3f}s")
    print(f"  Gaia查询:   {t_gaia_total:.2f}s ({t_gaia_total/(t_detect+t_build_u+t_gaia_total+t_ransac)*100:.1f}%)")
    print(f"  RANSAC:     {t_ransac:.2f}s ({t_ransac/(t_detect+t_build_u+t_gaia_total+t_ransac)*100:.1f}%)")
    print(f"  总计:       {t_detect+t_build_u+t_gaia_total+t_ransac:.2f}s")


if __name__ == '__main__':
    main()

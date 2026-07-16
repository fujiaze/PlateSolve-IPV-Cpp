"""
db_type=2 vs db_type=0 Gaia查询对比

全量测试用db_type=2, 之前测试用db_type=0
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

from vector_match_v2 import VectorMatch as VectorMatchV2, bisection_mag_limit, GaiaClientPy, _build_image_vectors
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# panel3 Red帧 (全量测试中V2耗时282秒)
FRAME = r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"
GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def main():
    logging.basicConfig(level=logging.WARNING)

    full_path = os.path.join(PROJECT_ROOT, FRAME)
    reader = ImageReader()
    img = reader.read(full_path)

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

    # 星检测
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)
    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)

    s0 = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(img.width ** 2 + img.height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0

    U, N_img, n_sat, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_saturated, s0, img.width, img.height)

    if n_sat >= 50:
        N_gaia = math.ceil(1.5 * n_sat)
    else:
        N_gaia = 150

    print(f"帧: {os.path.basename(FRAME)}")
    print(f"RA={center_ra:.4f}, Dec={center_dec:.4f}, 饱和={n_sat}, N_gaia={N_gaia}")
    print(f"查询半径={radius_deg:.2f}度")

    # db_type=0
    print(f"\n--- db_type=0 ---")
    gaia0 = GaiaClientPy(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    mag0, M0, ra0, dec0, mag0_arr = bisection_mag_limit(gaia0, center_ra, center_dec, radius_deg, N_gaia)
    t_gaia0 = time.perf_counter() - t0
    print(f"  Gaia查询: {t_gaia0:.2f}s (M={M0}, mag={mag0:.2f})")
    gaia0.close()

    # db_type=2
    print(f"\n--- db_type=2 ---")
    gaia2 = GaiaClientPy(GAIA_DATA_DIR, db_type=2)
    t0 = time.perf_counter()
    mag2, M2, ra2, dec2, mag2_arr = bisection_mag_limit(gaia2, center_ra, center_dec, radius_deg, N_gaia)
    t_gaia2 = time.perf_counter() - t0
    print(f"  Gaia查询: {t_gaia2:.2f}s (M={M2}, mag={mag2:.2f})")
    gaia2.close()

    # 完整solve对比
    print(f"\n--- 完整solve对比 ---")

    print(f"  db_type=0:")
    vm0 = VectorMatchV2(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    r0 = vm0.solve(img_x, img_y, img_flux, img_saturated,
                   center_ra, center_dec, focal_length, pixel_size, img.width, img.height)
    t_solve0 = time.perf_counter() - t0
    print(f"    solve: {t_solve0:.2f}s, RMS={r0.rms_px:.3f}px" if r0 else f"    solve: {t_solve0:.2f}s, 失败")
    vm0.close()

    print(f"  db_type=2:")
    vm2 = VectorMatchV2(GAIA_DATA_DIR, db_type=2)
    t0 = time.perf_counter()
    r2 = vm2.solve(img_x, img_y, img_flux, img_saturated,
                   center_ra, center_dec, focal_length, pixel_size, img.width, img.height)
    t_solve2 = time.perf_counter() - t0
    print(f"    solve: {t_solve2:.2f}s, RMS={r2.rms_px:.3f}px" if r2 else f"    solve: {t_solve2:.2f}s, 失败")
    vm2.close()


if __name__ == '__main__':
    main()

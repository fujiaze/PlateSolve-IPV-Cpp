"""
验证Gaia查询缓存效应 - 连续查询不同天区

测试: 先查panel3 Red帧(Dec=-23), 再查panel1 Red帧(Dec=-13), 再查panel3
看Gaia查询耗时是否受缓存影响
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

FRAMES = [
    ("panel3-Red", r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"),
    ("panel1-Red", r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts"),
    ("panel3-Red", r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"),
    ("panel2-Red", r"testdata\lights\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts"),
    ("panel3-Red", r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"),
]

GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def get_frame_info(frame_path):
    full_path = os.path.join(PROJECT_ROOT, frame_path)
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

    return center_ra, center_dec, radius_deg, N_gaia, n_sat


def main():
    logging.basicConfig(level=logging.WARNING)

    # 预加载所有帧信息
    print("预加载帧信息...")
    frame_infos = []
    for name, path in FRAMES:
        info = get_frame_info(path)
        frame_infos.append((name, info))
        print(f"  {name}: RA={info[0]:.4f}, Dec={info[1]:.4f}, 饱和={info[4]}")

    # 创建一个共享的Gaia客户端
    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    print(f"\n{'='*60}")
    print("连续Gaia查询测试 (共享客户端)")
    print(f"{'='*60}")

    for i, (name, info) in enumerate(frame_infos):
        center_ra, center_dec, radius_deg, N_gaia, n_sat = info
        t0 = time.perf_counter()
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            gaia, center_ra, center_dec, radius_deg, N_gaia)
        t_gaia = time.perf_counter() - t0
        print(f"  [{i+1}] {name}: Gaia查询 {t_gaia:.2f}s (M={M}, mag={mag_limit:.2f}, Dec={center_dec:.2f})")

    # 关闭后重新创建客户端, 再测一次
    gaia.close()
    print(f"\n--- 重新创建Gaia客户端 ---")
    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    for i, (name, info) in enumerate(frame_infos):
        center_ra, center_dec, radius_deg, N_gaia, n_sat = info
        t0 = time.perf_counter()
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            gaia, center_ra, center_dec, radius_deg, N_gaia)
        t_gaia = time.perf_counter() - t0
        print(f"  [{i+1}] {name}: Gaia查询 {t_gaia:.2f}s (M={M}, mag={mag_limit:.2f})")

    gaia.close()


if __name__ == '__main__':
    main()

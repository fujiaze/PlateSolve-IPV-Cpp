"""
优化测试: Gaia查询策略对比

1. 当前: bisection_mag_limit (7次cone_search, ~12.6s)
2. 优化A: 一次查询mag_limit=22, 内存过滤
3. 优化B: 同天区缓存复用

同时测量单次cone_search的耗时和返回数据量
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

from vector_match_v2 import GaiaClientPy, bisection_mag_limit, _build_image_vectors
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# 3个panel的Red帧 (同RA不同Dec)
FRAMES = [
    ("panel1-Red", r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts"),
    ("panel2-Red", r"testdata\lights\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts"),
    ("panel3-Red", r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"),
    # panel3第二帧 (同天区)
    ("panel3-Red2", r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@004333-180S-Red.fts"),
]

GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def get_frame_params(frame_path):
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

    # 预加载
    print("预加载帧参数...")
    frame_params = {}
    for name, path in FRAMES:
        ra, dec, radius, N_gaia, n_sat = get_frame_params(path)
        frame_params[name] = (ra, dec, radius, N_gaia, n_sat)
        print(f"  {name}: RA={ra:.4f}, Dec={dec:.4f}, radius={radius:.2f}°, N_gaia={N_gaia}, n_sat={n_sat}")

    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    # ================================================================
    # 测试1: 单次cone_search耗时 vs mag_limit
    # ================================================================
    print(f"\n{'='*60}")
    print("测试1: 单次cone_search耗时 vs mag_limit (panel3, Dec=-23.22)")
    print(f"{'='*60}")
    ra, dec, radius, N_gaia, n_sat = frame_params["panel3-Red"]

    for mag in [8.0, 8.5, 9.0, 10.0, 12.0, 14.0, 18.0, 22.0]:
        t0 = time.perf_counter()
        cat_ra, cat_dec, cat_mag = gaia.cone_search(ra, dec, radius, mag)
        t1 = time.perf_counter() - t0
        print(f"  mag_limit={mag:5.1f}: {t1:.2f}s, 星数={len(cat_ra):>8d}")

    # ================================================================
    # 测试2: 当前bisection_mag_limit vs 一次查询+内存过滤
    # ================================================================
    print(f"\n{'='*60}")
    print("测试2: bisection vs 一次查询+内存过滤")
    print(f"{'='*60}")

    for name in ["panel1-Red", "panel2-Red", "panel3-Red"]:
        ra, dec, radius, N_gaia, n_sat = frame_params[name]

        # 方法A: bisection (当前)
        t0 = time.perf_counter()
        mag_bis, M_bis, ra_bis, dec_bis, mag_bis_arr = bisection_mag_limit(
            gaia, ra, dec, radius, N_gaia)
        t_bis = time.perf_counter() - t0

        # 方法B: 一次查询mag=22, 内存过滤
        t0 = time.perf_counter()
        all_ra, all_dec, all_mag = gaia.cone_search(ra, dec, radius, 22.0)
        t_query = time.perf_counter() - t0

        t0 = time.perf_counter()
        sort_idx = np.argsort(all_mag)
        n_top = min(N_gaia, len(all_mag))
        top_idx = sort_idx[:n_top]
        ra_filt = all_ra[top_idx]
        dec_filt = all_dec[top_idx]
        mag_filt = all_mag[top_idx]
        mag_limit_filt = float(all_mag[top_idx[-1]]) if n_top > 0 else 22.0
        t_filter = time.perf_counter() - t0

        print(f"  {name}: bisection={t_bis:.2f}s (M={M_bis}, mag={mag_bis:.2f}) | "
              f"一次查询={t_query:.2f}s + 过滤={t_filter*1000:.1f}ms (M={len(ra_filt)}, mag={mag_limit_filt:.2f}) | "
              f"加速={t_bis/(t_query+t_filter):.1f}x")

    # ================================================================
    # 测试3: 同天区缓存复用
    # ================================================================
    print(f"\n{'='*60}")
    print("测试3: 同天区缓存复用 (panel3-Red vs panel3-Red2)")
    print(f"{'='*60}")

    ra1, dec1, r1, N1, _ = frame_params["panel3-Red"]
    ra2, dec2, r2, N2, _ = frame_params["panel3-Red2"]
    print(f"  panel3-Red:  RA={ra1:.4f}, Dec={dec1:.4f}, radius={r1:.2f}°, N_gaia={N1}")
    print(f"  panel3-Red2: RA={ra2:.4f}, Dec={dec2:.4f}, radius={r2:.2f}°, N_gaia={N2}")
    print(f"  天区相同? RA差={abs(ra1-ra2):.6f}°, Dec差={abs(dec1-dec2):.6f}°")

    # 分别查询
    t0 = time.perf_counter()
    mag1, M1, _, _, _ = bisection_mag_limit(gaia, ra1, dec1, r1, N1)
    t1 = time.perf_counter() - t0

    t0 = time.perf_counter()
    mag2, M2, _, _, _ = bisection_mag_limit(gaia, ra2, dec2, r2, N2)
    t2 = time.perf_counter() - t0

    print(f"  panel3-Red:  {t1:.2f}s (M={M1}, mag={mag1:.2f})")
    print(f"  panel3-Red2: {t2:.2f}s (M={M2}, mag={mag2:.2f})")
    print(f"  两次查询总耗时: {t1+t2:.2f}s (无缓存)")

    # 缓存方案: 一次查大天区, 两次过滤
    t0 = time.perf_counter()
    # 用更大的radius覆盖两个天区
    max_radius = max(r1, r2)
    all_ra, all_dec, all_mag = gaia.cone_search(ra1, dec1, max_radius, 22.0)
    t_query = time.perf_counter() - t0

    t0 = time.perf_counter()
    # 过滤给N1
    sort_idx = np.argsort(all_mag)
    n1 = min(N1, len(all_mag))
    ra1_f, dec1_f, mag1_f = all_ra[sort_idx[:n1]], all_dec[sort_idx[:n1]], all_mag[sort_idx[:n1]]
    # 过滤给N2
    n2 = min(N2, len(all_mag))
    ra2_f, dec2_f, mag2_f = all_ra[sort_idx[:n2]], all_dec[sort_idx[:n2]], all_mag[sort_idx[:n2]]
    t_filter = time.perf_counter() - t0

    print(f"  缓存方案: 查询={t_query:.2f}s + 过滤={t_filter*1000:.1f}ms = {t_query+t_filter:.2f}s")
    print(f"  加速: {(t1+t2)/(t_query+t_filter):.1f}x")

    gaia.close()


if __name__ == '__main__':
    main()

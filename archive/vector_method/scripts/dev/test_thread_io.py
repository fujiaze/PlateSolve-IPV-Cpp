"""
验证: 多线程Gaia查询I/O竞争

对比:
  1. 单线程顺序: 5帧
  2. 16线程并发: 5帧

看每帧的solve墙上时间差异
"""

import os
import sys
import time
import math
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

FRAMES = [
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
    r"testdata\lights\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",
    r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@055414-180S-Blue.fts",
    r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@025717-600S-Oiii.fts",
]

GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
detect_lock = threading.Lock()


def solve_single(file_path):
    full_path = os.path.join(PROJECT_ROOT, file_path)
    fname = os.path.basename(file_path)

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

    with detect_lock:
        params = SDetParamsPy(fitRadius=0)
        detector = StarDetector(params=params)
        det_result = detector.detect_ex(img.data)

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)

    vm = VectorMatchV2(GAIA_DATA_DIR, db_type=2)
    t0 = time.perf_counter()
    result = vm.solve(img_x, img_y, img_flux, img_saturated,
                      center_ra, center_dec, focal_length, pixel_size, img.width, img.height)
    t_solve = time.perf_counter() - t0
    vm.close()

    rms = result.rms_px if result else -1
    return fname, t_solve, rms


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)

    # 测试1: 单线程顺序
    print("=" * 60)
    print("测试1: 单线程顺序 (5帧)")
    print("=" * 60)
    t_start = time.perf_counter()
    results_seq = []
    for f in FRAMES:
        fname, t_solve, rms = solve_single(f)
        results_seq.append((fname, t_solve, rms))
        print(f"  {fname}: solve={t_solve:.2f}s, RMS={rms:.3f}px")
    t_total_seq = time.perf_counter() - t_start
    print(f"  总耗时: {t_total_seq:.2f}s, 平均: {t_total_seq/len(FRAMES):.2f}s/帧")

    # 测试2: 16线程并发
    print(f"\n{'='*60}")
    print("测试2: 16线程并发 (5帧)")
    print("=" * 60)
    t_start = time.perf_counter()
    results_par = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(solve_single, f): f for f in FRAMES}
        for future in as_completed(futures):
            fname, t_solve, rms = future.result()
            results_par.append((fname, t_solve, rms))
            print(f"  {fname}: solve={t_solve:.2f}s, RMS={rms:.3f}px")
    t_total_par = time.perf_counter() - t_start
    print(f"  总耗时: {t_total_par:.2f}s, 平均: {t_total_par/len(FRAMES):.2f}s/帧")

    # 对比
    print(f"\n{'='*60}")
    print("对比")
    print(f"{'='*60}")
    print(f"{'帧名':<60} {'顺序(s)':>8} {'并发(s)':>8} {'并发/顺序':>10}")
    for fname_seq, t_seq, _ in results_seq:
        for fname_par, t_par, _ in results_par:
            if fname_seq == fname_par:
                print(f"{fname_seq:<60} {t_seq:>8.2f} {t_par:>8.2f} {t_par/t_seq:>10.2f}x")
                break
    print(f"\n总耗时: 顺序={t_total_seq:.2f}s, 并发={t_total_par:.2f}s, 并发/顺序={t_total_par/t_total_seq:.2f}x")
    avg_seq = t_total_seq / len(FRAMES)
    avg_par = t_total_par / len(FRAMES)
    print(f"平均每帧: 顺序={avg_seq:.2f}s, 并发={avg_par:.2f}s, 并发/顺序={avg_par/avg_seq:.2f}x")


if __name__ == '__main__':
    main()

"""
单帧测试 - C++ V2 vs Python V2 对比
"""
import os
import sys
import time
import logging
import math
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2Py
from vector_match_v2_cpp import VectorMatch as VectorMatchV2Cpp
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

GAIA_DATA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")
FRAME = os.path.join(PROJECT_ROOT, "testdata", "lights", "panel3",
                     "Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts")


def main():
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    print(f"帧: {os.path.basename(FRAME)}")

    # 读取图像
    reader = ImageReader()
    img = reader.read(FRAME)
    width = img.width
    height = img.height

    center_ra = 0.0
    center_dec = 0.0
    focal_length = 200.0
    pixel_size = 6.0

    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        center_ra = img.metadata.wcs.crval1
        center_dec = img.metadata.wcs.crval2
    if img.metadata.observation:
        if img.metadata.observation.focallen is not None:
            focal_length = img.metadata.observation.focallen
        if img.metadata.observation.xpixsz is not None:
            pixel_size = img.metadata.observation.xpixsz

    # 从header提取中心坐标
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

    print(f"中心: RA={center_ra:.6f} Dec={center_dec:.6f}")
    print(f"焦距: {focal_length}mm 像元: {pixel_size}μm 图像: {width}x{height}")

    # 星检测
    t0 = time.perf_counter()
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)
    t_detect = time.perf_counter() - t0

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)
    n_sat = int(np.sum(img_saturated))
    print(f"星检测: {len(img_x)}颗 (饱和={n_sat}), 耗时={t_detect:.2f}s")

    # ── Python V2 ──
    print(f"\n{'='*60}")
    print("Python V2 solve")
    print(f"{'='*60}")
    v2py = VectorMatchV2Py(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    result_py = v2py.solve(img_x, img_y, img_flux, img_saturated,
                           center_ra, center_dec, focal_length, pixel_size, width, height)
    t_py = time.perf_counter() - t0
    v2py.close()

    if result_py:
        print(f"  成功: RMS={result_py.rms_px:.3f}px ({result_py.rms_arcsec:.3f}\"), "
              f"matched={result_py.matched_count}, mode={result_py.flip_mode}, "
              f"scale={result_py.scale_arcsec_px:.4f}\"/px, rot={result_py.rotation_deg:.4f}°")
    else:
        print("  失败")
    print(f"  耗时: {t_py:.2f}s")

    # ── C++ V2 ──
    print(f"\n{'='*60}")
    print("C++ V2 solve (OpenMP多线程)")
    print(f"{'='*60}")
    try:
        v2cpp = VectorMatchV2Cpp(GAIA_DATA_DIR, db_type=0)
        t0 = time.perf_counter()
        result_cpp = v2cpp.solve(img_x, img_y, img_flux, img_saturated,
                                  center_ra, center_dec, focal_length, pixel_size, width, height)
        t_cpp = time.perf_counter() - t0
        v2cpp.close()

        if result_cpp:
            print(f"  成功: RMS={result_cpp.rms_px:.3f}px ({result_cpp.rms_arcsec:.3f}\"), "
                  f"matched={result_cpp.matched_count}, mode={result_cpp.flip_mode}, "
                  f"scale={result_cpp.scale_arcsec_px:.4f}\"/px, rot={result_cpp.rotation_deg:.4f}°")
        else:
            print("  失败")
        print(f"  耗时: {t_cpp:.2f}s")

        # 对比
        if result_py and result_cpp:
            print(f"\n{'='*60}")
            print("对比")
            print(f"{'='*60}")
            print(f"  Python V2: RMS={result_py.rms_px:.3f}px mode={result_py.flip_mode} t={t_py:.2f}s")
            print(f"  C++    V2: RMS={result_cpp.rms_px:.3f}px mode={result_cpp.flip_mode} t={t_cpp:.2f}s")
            print(f"  加速比: {t_py/t_cpp:.1f}x")
            ra_diff = abs(result_py.center_ra - result_cpp.center_ra) * 3600
            dec_diff = abs(result_py.center_dec - result_cpp.center_dec) * 3600
            print(f"  中心差: ΔRA={ra_diff:.2f}\" ΔDec={dec_diff:.2f}\"")
    except Exception as e:
        print(f"  C++ V2异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

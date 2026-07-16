"""
单帧完整solve计时 - 用全量测试中的慢帧, 开启INFO日志
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
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

FRAME = r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@003945-180S-Red.fts"
GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def main():
    # 开启INFO日志看详细流程
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    full_path = os.path.join(PROJECT_ROOT, FRAME)
    fname = os.path.basename(FRAME)
    print(f"帧: {fname}")

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

    # V2 solve
    print(f"\n{'='*60}")
    print(f"V2 solve 开始 (全量测试中此帧耗时282秒)")
    print(f"{'='*60}")
    v2 = VectorMatchV2(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    result = v2.solve(img_x, img_y, img_flux, img_saturated,
                      center_ra, center_dec, focal_length, pixel_size, width, height)
    t_solve = time.perf_counter() - t0

    print(f"\n{'='*60}")
    print(f"V2 solve 完成: {t_solve:.2f}s")
    if result:
        print(f"  RMS={result.rms_px:.3f}px, matched={result.matched_count}")
    else:
        print(f"  匹配失败")
    print(f"  星检测: {t_detect:.2f}s")
    print(f"  solve: {t_solve:.2f}s")
    print(f"  总计: {t_detect+t_solve:.2f}s")


if __name__ == '__main__':
    main()

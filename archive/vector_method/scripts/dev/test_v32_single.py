# -*- coding: utf-8 -*-
"""V3.2 单帧测试脚本"""
import sys, os, time, logging
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "gaia_client", "python"))

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

import numpy as np
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v3_2 import VectorMatch


def extract_center(keywords):
    ra, dec = 0.0, 0.0
    for kw in keywords:
        name = kw.name.upper()
        if name in ("OBJCTRA", "RA"):
            val = kw.value
            if isinstance(val, str):
                parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                if len(parts) >= 3:
                    ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
        elif name in ("OBJCTDEC", "DEC"):
            val = kw.value
            if isinstance(val, str):
                parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                if len(parts) >= 3:
                    sign = -1 if parts[0].startswith("-") else 1
                    dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
    return ra, dec


def test_frame(fits_path, gaia_dir):
    r = ImageReader()
    det = StarDetector(params=SDetParamsPy(fitRadius=0))
    vm = VectorMatch(gaia_dir, db_type=0)

    img = r.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz

    # 中心坐标
    ra0, dec0 = 0.0, 0.0
    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        ra0 = img.metadata.wcs.crval1
        dec0 = img.metadata.wcs.crval2
    if ra0 == 0.0 and dec0 == 0.0:
        ra0, dec0 = extract_center(img.keywords)

    print(f"Image: {w}x{h} RA={ra0:.6f} Dec={dec0:.6f} FL={fl:.1f}mm PS={ps:.2f}um")

    d = det.detect_ex(img.data)
    img_x = np.array(d.x, dtype=np.float64)
    img_y = np.array(d.y, dtype=np.float64)
    img_flux = np.array(d.flux, dtype=np.float64)
    img_saturated = np.array(d.saturated, dtype=np.int32)

    t0 = time.time()
    result = vm.solve(img_x, img_y, img_flux, img_saturated, ra0, dec0, fl, ps, w, h)
    dt = time.time() - t0

    if result:
        print(f"SUCCESS: RA={result.center_ra:.6f} Dec={result.center_dec:.6f}")
        print(f"  rotation={result.rotation_deg:.2f} deg, scale={result.scale_arcsec_px:.4f} arcsec/px")
        print(f"  flip_mode={result.flip_mode}, matched={result.matched_count}")
        print(f"  rms_px={result.rms_px:.4f}, rms_arcsec={result.rms_arcsec:.4f}")
        print(f"  time={dt:.2f}s")
    else:
        print(f"FAILED: time={dt:.2f}s")

    vm.close()
    return result


if __name__ == "__main__":
    GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")

    # T2帧
    t2_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
                           "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
    # T4帧
    t4_path = os.path.join(PROJECT_ROOT, "testdata", "lights1", "panel1",
                           "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@053123-180S-Red.fts")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", choices=["t2", "t4", "both"], default="both")
    args = parser.parse_args()

    if args.frame in ("t2", "both"):
        print("=" * 60)
        print("T2 Frame Test (V3.2)")
        print("=" * 60)
        if os.path.exists(t2_path):
            test_frame(t2_path, GAIA_DIR)
        else:
            print(f"T2 frame not found: {t2_path}")

    if args.frame in ("t4", "both"):
        print("=" * 60)
        print("T4 Frame Test (V3.2)")
        print("=" * 60)
        if os.path.exists(t4_path):
            test_frame(t4_path, GAIA_DIR)
        else:
            print(f"T4 frame not found: {t4_path}")

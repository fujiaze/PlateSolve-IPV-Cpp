# -*- coding: utf-8 -*-
"""
初始WCS生成模块 - 端到端测试
功能: 使用testdata中的实际FITS图像测试initial_wcs模块
用途: 验证饱和星优先三角匹配+4种翻转模式+迭代重投影的完整流程
"""
import os
import sys
import time
import numpy as np

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "astro_image_io", "python"))

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from initial_wcs import InitialWCS, InitialWCSResult
import logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

GAIA_DIR = os.path.join(project_root, "GaiaDR3SP")

TEST_FILES = [
    os.path.join(project_root, "testdata", "lights", "panel1",
                 "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts"),
]

FOCAL_LENGTH_MM = 200.0
PIXEL_SIZE_UM = 6.0


def parse_ra(ra_str):
    if isinstance(ra_str, (int, float)):
        return float(ra_str)
    if isinstance(ra_str, str):
        parts = ra_str.strip().split()
        if len(parts) == 3:
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return (h + m / 60.0 + s / 3600.0) * 15.0
        elif len(parts) == 1:
            try:
                return float(parts[0])
            except ValueError:
                return 0.0
    return 0.0


def parse_dec(dec_str):
    if isinstance(dec_str, (int, float)):
        return float(dec_str)
    if isinstance(dec_str, str):
        parts = dec_str.strip().split()
        if len(parts) == 3:
            sign = -1.0 if parts[0].startswith('-') else 1.0
            d = abs(float(parts[0]))
            m = float(parts[1])
            s = float(parts[2])
            return sign * (d + m / 60.0 + s / 3600.0)
        elif len(parts) == 1:
            try:
                return float(parts[0])
            except ValueError:
                return 0.0
    return 0.0


def read_fits_image(path):
    reader = ImageReader()
    img_data = reader.read(path)
    image = img_data.data
    width = img_data.width
    height = img_data.height

    center_ra = 0.0
    center_dec = 0.0

    wcs = img_data.wcs
    if wcs and wcs.has_wcs:
        center_ra = wcs.crval1
        center_dec = wcs.crval2
        print(f"  WCS中心: RA={center_ra:.6f}, Dec={center_dec:.6f}")

    if center_ra == 0.0 and center_dec == 0.0:
        for kw in img_data.keywords:
            if kw.name == "OBJCTRA":
                center_ra = parse_ra(kw.value)
            elif kw.name == "OBJCTDEC":
                center_dec = parse_dec(kw.value)
            elif kw.name == "RA" and center_ra == 0.0:
                center_ra = parse_ra(kw.value)
            elif kw.name == "DEC" and center_dec == 0.0:
                center_dec = parse_dec(kw.value)
        if center_ra != 0.0 or center_dec != 0.0:
            print(f"  FITS头中心: RA={center_ra:.6f}, Dec={center_dec:.6f}")

    if center_ra == 0.0 and center_dec == 0.0:
        center_ra = 266.42
        center_dec = -28.99
        print(f"  使用默认中心: RA={center_ra:.4f}, Dec={center_dec:.4f}")

    if image.dtype != np.uint16:
        image = np.clip(image, 0, 65535).astype(np.uint16)

    return image, center_ra, center_dec, width, height


def detect_stars(image):
    params = SDetParamsPy()
    detector = StarDetector(params=params)
    coords, fluxes, saturated = detector.detect_ex(image)
    img_x = np.array([c[0] for c in coords], dtype=np.float64)
    img_y = np.array([c[1] for c in coords], dtype=np.float64)
    img_flux = np.array(fluxes, dtype=np.float64)
    img_saturated = np.array(saturated, dtype=np.int32)
    return img_x, img_y, img_flux, img_saturated


def center_coords(img_x, img_y, width, height):
    cx = img_x - width / 2.0
    cy = -(img_y - height / 2.0)
    return cx, cy


def run_test(fits_path):
    print("\n" + "=" * 80)
    print(f"测试文件: {os.path.basename(fits_path)}")
    print("=" * 80)

    if not os.path.exists(fits_path):
        print(f"文件不存在: {fits_path}")
        return None

    image, center_ra, center_dec, width, height = read_fits_image(fits_path)
    print(f"  图像尺寸: {width}x{height}")
    print(f"  中心坐标: RA={center_ra:.4f}, Dec={center_dec:.4f}")

    t0 = time.time()
    img_x, img_y, img_flux, img_saturated = detect_stars(image)
    t_detect = time.time() - t0
    n_sat = int(np.sum(img_saturated))
    print(f"  星点检测: {len(img_x)}颗 (饱和{n_sat}颗), 耗时{t_detect:.1f}s")

    cx, cy = center_coords(img_x, img_y, width, height)

    solver = InitialWCS(GAIA_DIR, db_type=0)
    try:
        t0 = time.time()
        result = solver.solve(
            cx, cy, img_flux, img_saturated,
            center_ra, center_dec,
            FOCAL_LENGTH_MM, PIXEL_SIZE_UM,
            width, height
        )
        t_solve = time.time() - t0
    except Exception as e:
        import traceback
        print(f"  solve()异常: {e}")
        traceback.print_exc()
        solver.close()
        return None
    finally:
        solver.close()

    if result is None:
        print(f"  solve()返回None - 匹配失败")
        return None

    print(f"\n  === 结果 ===")
    print(f"  中心: RA={result.center_ra:.6f}, Dec={result.center_dec:.6f}")
    print(f"  旋转: {result.rotation_deg:.4f}度")
    print(f"  比例尺: {result.scale_arcsec_px:.4f} arcsec/px")
    print(f"  翻转模式: {result.flip_mode}")
    print(f"  仿射: a0={result.affine[0]:.4f} a1={result.affine[1]:.6f} a2={result.affine[2]:.6f}")
    print(f"         b0={result.affine[3]:.4f} b1={result.affine[4]:.6f} b2={result.affine[5]:.6f}")
    print(f"  匹配数: {result.matched_count}")
    print(f"  RMS: {result.rms_px:.3f} px ({result.rms_arcsec:.4f} arcsec)")
    print(f"  耗时: {t_solve:.2f}s")

    if result.rms_px < 2.0:
        print(f"  ✅ RMS < 2px 目标达成!")
    else:
        print(f"  ❌ RMS >= 2px, 需要调试")

    if result.matched_count > 100:
        print(f"  ✅ 匹配数 > 100")
    else:
        print(f"  ❌ 匹配数 <= 100, 需要调试")

    return result


if __name__ == "__main__":
    results = []
    for f in TEST_FILES:
        r = run_test(f)
        results.append(r)

    print("\n" + "=" * 80)
    print("汇总:")
    for i, (f, r) in enumerate(zip(TEST_FILES, results)):
        name = os.path.basename(f)
        if r is not None:
            print(f"  [{i+1}] {name[:60]}: RMS={r.rms_px:.3f}px, 匹配={r.matched_count}, 翻转={r.flip_mode}")
        else:
            print(f"  [{i+1}] {name[:60]}: FAILED")
    print("=" * 80)

# -*- coding: utf-8 -*-
"""
Plate Solve 性能分析
分析各阶段耗时，找出瓶颈
"""

import os
import sys
import numpy as np
import time

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "gaia_xpsd_client", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "astro_image_io", "python"))

from plate_solve import PlateSolve, PlateSolveConfig
from star_detector import StarDetector
from astro_image_io import ImageReader

def parse_hms_ra(value):
    if not value:
        return None
    parts = value.split()
    if len(parts) >= 3:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return (h + m/60.0 + s/3600.0) * 15.0
    return None

def parse_dms_dec(value):
    if not value:
        return None
    parts = value.split()
    if len(parts) >= 3:
        sign = -1 if parts[0].startswith('-') else 1
        d = abs(float(parts[0]))
        m = float(parts[1])
        s = float(parts[2])
        return sign * (d + m/60.0 + s/3600.0)
    return None

def get_fits_info(img_data):
    info = {'ra': None, 'dec': None, 'focal_mm': 200.0, 'pixel_um': 6.0}
    for kw in img_data.keywords:
        name = kw.name.upper()
        val = kw.value
        if name == 'RA' or name == 'OBJCTRA':
            ra = parse_hms_ra(val)
            if ra is not None:
                info['ra'] = ra
        elif name == 'DEC' or name == 'OBJCTDEC':
            dec = parse_dms_dec(val)
            if dec is not None:
                info['dec'] = dec
        elif name == 'FOCALLEN':
            try:
                info['focal_mm'] = float(val)
            except:
                pass
        elif name == 'XPIXSZ':
            try:
                info['pixel_um'] = float(val)
            except:
                pass
    return info

def profile_single_frame():
    print("=" * 80)
    print("Plate Solve 性能分析")
    print("=" * 80)
    
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    
    fits_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".fts")])
    fits_path = os.path.join(test_dir, fits_files[0])
    
    print(f"\n测试文件: {fits_files[0]}")
    
    reader = ImageReader()
    
    t_total_start = time.time()
    
    t0 = time.time()
    img_data = reader.read(fits_path)
    img = img_data.data
    h, w = img.shape[:2]
    t_read = time.time() - t0
    print(f"\n[1] 文件读取: {t_read:.3f}s")
    
    info = get_fits_info(img_data)
    print(f"    尺寸: {w}x{h}, RA={info['ra']:.4f}, Dec={info['dec']:.4f}")
    
    t0 = time.time()
    detector = StarDetector()
    coords, fluxes, saturated = detector.detect_ex(img)
    t_detect = time.time() - t0
    print(f"\n[2] 星点检测: {t_detect:.3f}s")
    
    n_stars = len(coords)
    n_saturated = sum(saturated)
    print(f"    检测到: {n_stars}颗星, 饱和: {n_saturated}颗")
    
    t0 = time.time()
    stars = np.zeros((n_stars, 6), dtype=np.float64)
    for j in range(n_stars):
        stars[j, 0] = coords[j][0]
        stars[j, 1] = coords[j][1]
        stars[j, 2] = fluxes[j]
        stars[j, 5] = saturated[j]
    
    img_x = stars[:, 0] - w / 2.0
    img_y = -(stars[:, 1] - h / 2.0)
    img_flux = stars[:, 2]
    saturated_mask = stars[:, 5].astype(np.int32)
    t_convert = time.time() - t0
    print(f"\n[3] 数据转换: {t_convert:.3f}s")
    
    config = PlateSolveConfig()
    config.sip_order = 0
    config.max_iterations = 1
    
    t0 = time.time()
    with PlateSolve(gaia_data_dir=gaia_dir) as solver:
        result = solver.solve(
            img_x=img_x, img_y=img_y, img_flux=img_flux,
            img_saturated=saturated_mask, n_saturated=n_saturated,
            center_ra=info['ra'], center_dec=info['dec'],
            focal_length_mm=info['focal_mm'], pixel_size_um=info['pixel_um'],
            width=w, height=h,
            config=config
        )
    t_solve = time.time() - t0
    print(f"\n[4] Plate Solve: {t_solve:.3f}s")
    print(f"    匹配: {result.matched_count}颗, RMS: {result.rms_px:.3f}px")
    
    t_total = time.time() - t_total_start
    print(f"\n[5] 总耗时: {t_total:.3f}s")
    
    print("\n" + "=" * 80)
    print("性能分析总结")
    print("=" * 80)
    print(f"{'阶段':<25} {'耗时(s)':<12} {'占比':<10}")
    print("-" * 50)
    print(f"{'文件读取':<25} {t_read:<12.3f} {t_read/t_total*100:<10.1f}%")
    print(f"{'星点检测':<25} {t_detect:<12.3f} {t_detect/t_total*100:<10.1f}%")
    print(f"{'数据转换':<25} {t_convert:<12.3f} {t_convert/t_total*100:<10.1f}%")
    print(f"{'Plate Solve':<25} {t_solve:<12.3f} {t_solve/t_total*100:<10.1f}%")
    print("-" * 50)
    print(f"{'总计':<25} {t_total:<12.3f} {100.0:<10.1f}%")
    
    print("\n瓶颈分析:")
    if t_detect > t_solve:
        print("  主要瓶颈: 星点检测")
        print("  建议: 减少Moffat4拟合的星点数量，或使用更快的PSF拟合算法")
    else:
        print("  主要瓶颈: Plate Solve (三角形匹配)")
        print("  建议: 减少三角形数量，或优化三角形匹配算法")
    
    img_data.close()

if __name__ == "__main__":
    profile_single_frame()

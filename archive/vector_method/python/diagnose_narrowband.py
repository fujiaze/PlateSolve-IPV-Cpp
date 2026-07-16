# -*- coding: utf-8 -*-
"""
诊断窄带帧匹配失败的原因
分析饱和星分布、三角形构建、匹配过程
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

def analyze_triangle_potential(x, y, n_select, label=""):
    """分析给定星点构建三角形的潜力"""
    if n_select < 3:
        print(f"  {label}: 星点不足3颗，无法构建三角形")
        return 0
    
    n = min(n_select, len(x))
    sel_x = x[:n]
    sel_y = y[:n]
    
    valid_tris = 0
    ba_list = []
    ca_list = []
    
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x0, y0 = sel_x[i], sel_y[i]
                x1, y1 = sel_x[j], sel_y[j]
                x2, y2 = sel_x[k], sel_y[k]
                
                area = abs((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)) * 0.5
                if area < 50.0:
                    continue
                
                da = np.sqrt((x1-x0)**2 + (y1-y0)**2)
                db = np.sqrt((x2-x0)**2 + (y2-y0)**2)
                dc = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if da < 1.0 or db < 1.0 or dc < 1.0:
                    continue
                
                a, b, c = da, db, dc
                if a < b: a, b = b, a
                if b < c: b, c = c, b
                if a < b: a, b = b, a
                
                ba = b / a
                ca = c / a
                
                if ba > 0.92:
                    continue
                if ba - ca < 0.03:
                    continue
                
                cosA = (b*b + c*c - a*a) / (2*b*c)
                cosB = (a*a + c*c - b*b) / (2*a*c)
                cosC = (a*a + b*b - c*c) / (2*a*b)
                min_cos = max(cosA, cosB, cosC)
                min_angle = np.arccos(np.clip(min_cos, -1, 1))
                
                if min_angle < 10.0 * np.pi / 180.0:
                    continue
                if min_angle > 70.0 * np.pi / 180.0:
                    continue
                
                valid_tris += 1
                ba_list.append(ba)
                ca_list.append(ca)
    
    if valid_tris > 0:
        ba_std = np.std(ba_list) if len(ba_list) > 1 else 0
        ca_std = np.std(ca_list) if len(ca_list) > 1 else 0
        print(f"  {label}: {n}颗星 → {valid_tris}个有效三角形, ba_std={ba_std:.4f}, ca_std={ca_std:.4f}")
    else:
        print(f"  {label}: {n}颗星 → 0个有效三角形（过滤后无剩余）")
    
    return valid_tris

def diagnose_frame(fits_path, gaia_dir):
    """诊断单帧"""
    reader = ImageReader()
    
    try:
        img_data = reader.read(fits_path)
        img = img_data.data
        h, w = img.shape[:2]
        
        print(f"\n{'='*80}")
        print(f"文件: {os.path.basename(fits_path)}")
        print(f"图像尺寸: {w}x{h}")
        
        detector = StarDetector()
        coords, fluxes, saturated = detector.detect_ex(img)
        
        n_stars = len(coords)
        n_saturated = sum(saturated)
        print(f"检测星点: {n_stars}颗, 饱和星: {n_saturated}颗")
        
        if n_stars == 0:
            print("无星点，跳过")
            img_data.close()
            return
        
        img_x = np.array([c[0] - w/2.0 for c in coords])
        img_y = np.array([-(c[1] - h/2.0) for c in coords])
        img_flux = np.array(fluxes)
        
        sorted_idx = np.argsort(-img_flux)
        img_x_sorted = img_x[sorted_idx]
        img_y_sorted = img_y[sorted_idx]
        saturated_sorted = saturated[sorted_idx]
        
        sat_x = img_x_sorted[saturated_sorted.astype(bool)]
        sat_y = img_y_sorted[saturated_sorted.astype(bool)]
        
        print(f"\n饱和星分布:")
        if len(sat_x) > 0:
            print(f"  X范围: [{sat_x.min():.1f}, {sat_x.max():.1f}] px (图像宽{w})")
            print(f"  Y范围: [{sat_y.min():.1f}, {sat_y.max():.1f}] px (图像高{h})")
            sat_cx = sat_x.mean()
            sat_cy = sat_y.mean()
            print(f"  重心: ({sat_cx:.1f}, {sat_cy:.1f})")
            sat_spread = np.sqrt(np.mean((sat_x - sat_cx)**2 + (sat_y - sat_cy)**2))
            print(f"  分布半径: {sat_spread:.1f} px")
        
        print(f"\n三角形构建潜力分析:")
        analyze_triangle_potential(sat_x, sat_y, len(sat_x), "饱和星全部")
        analyze_triangle_potential(sat_x, sat_y, min(15, len(sat_x)), "饱和星TOP15")
        analyze_triangle_potential(img_x_sorted, img_y_sorted, 50, "亮星TOP50")
        analyze_triangle_potential(img_x_sorted, img_y_sorted, 100, "亮星TOP100")
        
        from gaia_xpsd_client import GaiaXPSDClient
        with GaiaXPSDClient(gaia_dir) as gaia:
            center_ra = 272.808333
            center_dec = -13.176944
            focal_mm = 200.0
            pixel_um = 6.0
            scale_arcsec_px = 206265.0 * pixel_um / focal_mm
            radius_deg = np.sqrt((w/2)**2 + (h/2)**2) * scale_arcsec_px / 3600.0
            
            cat_ra, cat_dec, cat_mag = gaia.query_cone(center_ra, center_dec, radius_deg, limit=5000)
            
            if len(cat_ra) > 0:
                print(f"\nGaia星表: {len(cat_ra)}颗")
                
                cat_sorted_idx = np.argsort(cat_mag)
                cat_x_px = (cat_ra - center_ra) * 3600.0 / scale_arcsec_px * np.cos(np.radians(center_dec))
                cat_y_px = (cat_dec - center_dec) * 3600.0 / scale_arcsec_px
                
                cat_x_sorted = cat_x_px[cat_sorted_idx]
                cat_y_sorted = cat_y_px[cat_sorted_idx]
                
                analyze_triangle_potential(cat_x_sorted, cat_y_sorted, 15, "Gaia TOP15")
                analyze_triangle_potential(cat_x_sorted, cat_y_sorted, 50, "Gaia TOP50")
                analyze_triangle_potential(cat_x_sorted, cat_y_sorted, 100, "Gaia TOP100")
                
                print(f"\n匹配潜力评估:")
                n_coarse_img = n_saturated if n_saturated >= 10 else min(100, n_stars)
                n_coarse_cat = int(n_saturated * 1.2 + 0.5) if n_saturated >= 10 else 150
                print(f"  当前策略: 图像侧{n_coarse_img}颗, Gaia侧{n_coarse_cat}颗")
                
                if n_saturated >= 10:
                    img_tris = analyze_triangle_potential(sat_x, sat_y, n_coarse_img, "  图像三角形")
                    cat_tris = analyze_triangle_potential(cat_x_sorted, cat_y_sorted, n_coarse_cat, "  Gaia三角形")
                    
                    if img_tris < 10 or cat_tris < 10:
                        print(f"\n  ⚠️ 三角形数量不足，可能导致匹配失败！")
                        print(f"  建议: 扩大n_coarse_cat到50或100")
        
        img_data.close()
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

def main():
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    
    failed_files = [
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@063631-600S-Oiii.fts",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@061318-300S-H-alpha.fts",
    ]
    
    success_files = [
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@052346-180S-Red.fts",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@053512-180S-Green.fts",
    ]
    
    print("=" * 80)
    print("失败帧诊断")
    print("=" * 80)
    for f in failed_files:
        fits_path = os.path.join(test_dir, f)
        if os.path.exists(fits_path):
            diagnose_frame(fits_path, gaia_dir)
    
    print("\n" + "=" * 80)
    print("成功帧对比")
    print("=" * 80)
    for f in success_files:
        fits_path = os.path.join(test_dir, f)
        if os.path.exists(fits_path):
            diagnose_frame(fits_path, gaia_dir)

if __name__ == "__main__":
    main()

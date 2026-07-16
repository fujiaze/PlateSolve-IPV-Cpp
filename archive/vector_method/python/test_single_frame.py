# -*- coding: utf-8 -*-
"""
单帧测试 - 网格控制点精细拟合测试
"""
import os
import sys

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))

from plate_solve import PlateSolve, PlateSolveConfig

gaia_dir = os.path.join(project_root, "GaiaDR3SP")
test_file = os.path.join(project_root, "testdata", "lights", "panel1", 
                         "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@051551-180S-Red.fts")

print("=" * 80)
print(f"测试文件: {os.path.basename(test_file)}")
print("=" * 80)

config = PlateSolveConfig(
    use_saturated_priority=1,
    n_img_bright=500,
    n_cat_bright=600,
    max_match_dist_px=25.0,
    max_iterations=5,
    match_threshold=10.0,
    sip_order=5,
)

with PlateSolve(gaia_data_dir=gaia_dir) as solver:
    result = solver.solve_with_file(test_file, config)
    
    print(f"\n粗匹配结果 (Step 1):")
    print(f"  中心: RA={result.center_ra:.6f}, Dec={result.center_dec:.6f}")
    print(f"  旋转: {result.rotation_deg:.3f}°")
    print(f"  比例尺: {result.scale_arcsec_px:.3f} arcsec/px")
    print(f"  翻转模式: {result.flip_mode}")
    print(f"  匹配数: {result.matched_count}")
    print(f"  RMS: {result.rms_px:.3f} px")
    print(f"  Step1耗时: {result.step1_time_sec:.2f}s")
    
    print(f"\n精细拟合结果 (Step 2 - 网格控制点):")
    print(f"  控制点数: {result.matched_count}")
    print(f"  RMS: {result.rms_px:.3f} px")
    print(f"  Step2耗时: {result.step2_time_sec:.2f}s")
    print(f"  SIP有效: {result.sip_valid}")
    
    if result.sip_valid and result.sip is not None:
        print(f"\nSIP系数 (order={result.wcs.get('sip_order', 5)}):")
        print(f"  CD矩阵:")
        print(f"    CD1_1 = {result.wcs['cd1_1']:.6e}")
        print(f"    CD1_2 = {result.wcs['cd1_2']:.6e}")
        print(f"    CD2_1 = {result.wcs['cd2_1']:.6e}")
        print(f"    CD2_2 = {result.wcs['cd2_2']:.6e}")
        
        print(f"\n  CRPIX: ({result.wcs['crpix1']:.1f}, {result.wcs['crpix2']:.1f})")
        print(f"  CRVAL: ({result.wcs['crval1']:.6f}, {result.wcs['crval2']:.6f})")
        
        A = result.sip[0]
        B = result.sip[1]
        print(f"\n  SIP正向系数 A (部分):")
        for i in range(3):
            for j in range(3):
                if i + j >= 2 and i + j <= 5:
                    val = A[i][j]
                    if abs(val) > 1e-10:
                        print(f"    A[{i},{j}] = {val:.4e}")
    
    print(f"\n总耗时: {result.step1_time_sec + result.step2_time_sec:.2f}s")

print("\n" + "=" * 80)
print("测试完成")
print("=" * 80)
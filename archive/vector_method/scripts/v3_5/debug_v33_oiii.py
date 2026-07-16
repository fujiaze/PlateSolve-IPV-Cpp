"""
V3.3 C++版向量匹配调试脚本 - GC_P1 Oiii帧

功能: 使用VectorMatchV33Cpp对GC_P1 Oiii帧执行plate solving，输出详细调试信息
用途: 验证V3.3算法在该帧上的匹配效果，包括s, theta, n_inliers, rms, SNR等关键指标
"""

import sys
import os
import logging
import math

# 强制UTF-8编码
if sys.platform == 'win32':
    import locale
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'astro_image_io', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'star_detector', 'python'))

import numpy as np
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v3_3_cpp import VectorMatch as VectorMatchV33Cpp

# 配置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)

def main():
    fits_path = os.path.join(
        PROJECT_ROOT,
        'testdata', 'lights1', 'panel1',
        'Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts'
    )

    print("=" * 70)
    print("V3.3 C++ VectorMatch 调试 - GC_P1 Oiii")
    print("=" * 70)
    print(f"文件: {fits_path}")
    print()

    # 1. 读取图像
    print("[1] 读取图像...")
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    print(f"    图像尺寸: {w} x {h}")

    # 2. 提取元数据
    print("[2] 提取元数据...")
    meta = img.metadata
    obs = meta.observation
    wcs = meta.wcs

    focal_length = obs.focallen if obs and obs.focallen else 0.0
    pixel_size = obs.xpixsz if obs and obs.xpixsz else 0.0

    if wcs and wcs.has_wcs:
        center_ra = wcs.crval1
        center_dec = wcs.crval2
        scale_from_wcs = wcs.pixel_scale
        print(f"    WCS中心: RA={center_ra:.6f}°, Dec={center_dec:.6f}°")
        print(f"    WCS像素尺度: {scale_from_wcs:.4f} arcsec/px")
    else:
        center_ra = 0.0
        center_dec = 0.0
        scale_from_wcs = 0.0
        print("    无WCS信息")

    print(f"    焦距: {focal_length:.1f} mm")
    print(f"    像元尺寸: {pixel_size:.2f} μm")

    if focal_length > 0 and pixel_size > 0:
        s0 = 206.265 * pixel_size / focal_length
        print(f"    计算s0: {s0:.4f} arcsec/px")
    else:
        print("    错误: 无法计算s0")
        return

    # 3. 星点检测
    print("[3] 星点检测 (fitRadius=0)...")
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    img_x = np.array(det.x, dtype=np.float64)
    img_y = np.array(det.y, dtype=np.float64)
    img_flux = np.array(det.flux, dtype=np.float64)
    img_sat = np.array(det.saturated, dtype=np.int32)

    total_stars = len(img_x)
    nsat = int(img_sat.sum())
    n_normal = total_stars - nsat
    print(f"    总检测星点: {total_stars}")
    print(f"    饱和星: {nsat}")
    print(f"    正常星: {n_normal}")

    # 4. 向量匹配 V3.3
    print("[4] V3.3 C++ 向量匹配...")
    gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
    print(f"    Gaia目录: {gaia_dir}")

    with VectorMatchV33Cpp(gaia_dir, db_type=1) as vm:
        result = vm.solve(
            img_x=img_x,
            img_y=img_y,
            img_flux=img_flux,
            img_saturated=img_sat,
            center_ra=center_ra,
            center_dec=center_dec,
            focal_length_mm=focal_length,
            pixel_size_um=pixel_size,
            width=w,
            height=h,
        )

    # 5. 输出结果
    print()
    print("=" * 70)
    print("匹配结果")
    print("=" * 70)

    if result is None:
        print("状态: FAIL (匹配失败)")
        print()
        print("诊断信息:")
        print(f"  N (图像星点总数): {total_stars}")
        print(f"  nsat (饱和星数): {nsat}")
        print(f"  M (Gaia星数): 未知 (匹配未成功)")
        print(f"  s0 (初始像素尺度): {s0:.4f} arcsec/px")
    else:
        success = "SUCCESS" if result.matched_count >= 5 else "FAIL"
        print(f"状态: {success}")
        print()
        print("--- 核心参数 ---")
        print(f"  s (缩放因子): {result.solve_s:.6f}")
        print(f"  theta (旋转角): {result.rotation_deg:.4f}°")
        print(f"  n_inliers (内点数): {result.matched_count}")
        print(f"  rms (像素): {result.rms_px:.4f} px")
        print(f"  rms (角秒): {result.rms_arcsec:.4f} arcsec")
        print(f"  SNR (peak_snr): {result.theta_snr:.2f}x")
        print()
        print("--- 中心坐标 ---")
        print(f"  原始RA:  {result.original_ra:.6f}°")
        print(f"  原始Dec: {result.original_dec:.6f}°")
        print(f"  修正RA:  {result.center_ra:.6f}°")
        print(f"  修正Dec: {result.center_dec:.6f}°")
        print(f"  ΔRA:     {result.center_ra - result.original_ra:.6f}°")
        print(f"  ΔDec:    {result.center_dec - result.original_dec:.6f}°")
        print()
        print("--- 像素尺度 ---")
        print(f"  s0 (初始): {result.s0:.4f} arcsec/px")
        print(f"  s_final:   {result.scale_arcsec_px:.4f} arcsec/px")
        print(f"  缩放比:    {result.solve_s:.6f}")
        print()
        print("--- 翻转模式 ---")
        print(f"  flip_mode: {result.flip_mode}")
        flip_names = {0: "无翻转", 1: "X翻转", 2: "Y翻转", 3: "XY翻转"}
        print(f"  含义: {flip_names.get(result.flip_mode, '未知')}")
        print()
        print("--- 仿射参数 ---")
        a0, a1, a2, b0, b1, b2 = result.affine
        print(f"  a0={a0:.4f} a1={a1:.6f} a2={a2:.6f}")
        print(f"  b0={b0:.4f} b1={b1:.6f} b2={b2:.6f}")
        print()
        print("--- V3.3 Debug信息 ---")
        print(f"  theta_snr: {result.theta_snr:.2f}x")
        print(f"  theta_peak_deg: {result.theta_peak_deg:.4f}°")
        print(f"  best_n_range: {result.best_n_range}")
        print(f"  median_noise: {result.median_noise:.4f}")
        print(f"  n_phaseb_pairs: {result.n_phaseb_pairs}")
        print(f"  n_phaseb_corr: {result.n_phaseb_corr}")
        print(f"  n_phasea_records: {result.n_phasea_records}")
        print()
        print("--- 统计摘要 ---")
        print(f"  N (图像星点总数): {total_stars}")
        print(f"  nsat (饱和星数): {nsat}")
        print(f"  M (Gaia星数): 见日志")
        print(f"  n_inliers / N: {result.matched_count}/{total_stars} = {result.matched_count/total_stars*100:.1f}%")
        print(f"  rms / s0: {result.rms_px:.4f} / {s0:.4f} = {result.rms_px/s0:.4f} px")

    print()
    print("=" * 70)

    # 关闭图像
    img.close()


if __name__ == '__main__':
    main()

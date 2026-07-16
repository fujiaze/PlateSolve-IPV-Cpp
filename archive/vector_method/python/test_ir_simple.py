#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二步迭代精化模块 - 简化测试
直接测试三角形匹配功能
"""

import os
import sys
import time
import ctypes
import numpy as np

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, project_root)

mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(mingw_bin):
    os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(mingw_bin)
    except OSError:
        pass

def load_dll(dll_path):
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    try:
        os.add_dll_directory(dll_dir)
    except OSError:
        pass
    return ctypes.cdll.LoadLibrary(dll_path)

def main():
    print("=" * 70)
    print("第二步迭代精化模块 - 简化测试")
    print("=" * 70)

    ir_dll_path = os.path.join(project_root, "lib", "plate_solve", "modules", "iterative_refine", "psm_iterative_refine.dll")
    ir_dll = load_dll(ir_dll_path)

    class IRInitialTransformC(ctypes.Structure):
        _fields_ = [
            ("center_ra", ctypes.c_double),
            ("center_dec", ctypes.c_double),
            ("rotation_deg", ctypes.c_double),
            ("scale_arcsec_px", ctypes.c_double),
            ("flip_mode", ctypes.c_int),
            ("img_width", ctypes.c_int),
            ("img_height", ctypes.c_int),
        ]

    class IRImageStarsC(ctypes.Structure):
        _fields_ = [
            ("img_x", ctypes.POINTER(ctypes.c_double)),
            ("img_y", ctypes.POINTER(ctypes.c_double)),
            ("img_flux", ctypes.POINTER(ctypes.c_double)),
            ("img_count", ctypes.c_int),
            ("img_saturated", ctypes.POINTER(ctypes.c_int)),
            ("n_saturated", ctypes.c_int),
        ]

    class IRCatalogStarsC(ctypes.Structure):
        _fields_ = [
            ("cat_ra", ctypes.POINTER(ctypes.c_double)),
            ("cat_dec", ctypes.POINTER(ctypes.c_double)),
            ("cat_mag", ctypes.POINTER(ctypes.c_double)),
            ("cat_count", ctypes.c_int),
        ]

    class IRConfigC(ctypes.Structure):
        _fields_ = [
            ("max_stars_triangle", ctypes.c_int),
            ("tri_ratio_radius", ctypes.c_double),
            ("tri_min_area", ctypes.c_double),
            ("tri_max_ba_ratio", ctypes.c_double),
            ("tri_equilateral_thresh", ctypes.c_double),
            ("grid_size", ctypes.c_int),
            ("outlier_angle_thresh", ctypes.c_double),
            ("outlier_mag_ratio", ctypes.c_double),
            ("max_iterations", ctypes.c_int),
            ("converge_thresh", ctypes.c_double),
            ("match_threshold", ctypes.c_double),
        ]

    class IRRefineResultC(ctypes.Structure):
        _fields_ = [
            ("final_ra", ctypes.c_double),
            ("final_dec", ctypes.c_double),
            ("final_rotation", ctypes.c_double),
            ("final_scale", ctypes.c_double),
            ("dist_a0", ctypes.c_double), ("dist_a1", ctypes.c_double),
            ("dist_a2", ctypes.c_double), ("dist_a3", ctypes.c_double),
            ("dist_a4", ctypes.c_double), ("dist_a5", ctypes.c_double),
            ("dist_b0", ctypes.c_double), ("dist_b1", ctypes.c_double),
            ("dist_b2", ctypes.c_double), ("dist_b3", ctypes.c_double),
            ("dist_b4", ctypes.c_double), ("dist_b5", ctypes.c_double),
            ("distortion_valid", ctypes.c_int),
            ("matched_count", ctypes.c_int),
            ("rms_x", ctypes.c_double),
            ("rms_y", ctypes.c_double),
            ("rms_total", ctypes.c_double),
            ("rms_arcsec", ctypes.c_double),
            ("iteration_count", ctypes.c_int),
            ("triangle_matches", ctypes.c_int),
            ("img_indices", ctypes.POINTER(ctypes.c_int)),
            ("cat_indices", ctypes.POINTER(ctypes.c_int)),
            ("residual_x", ctypes.POINTER(ctypes.c_double)),
            ("residual_y", ctypes.POINTER(ctypes.c_double)),
        ]

    ir_dll.psm_iterative_refine.argtypes = [
        ctypes.POINTER(IRImageStarsC),
        ctypes.POINTER(IRCatalogStarsC),
        ctypes.POINTER(IRInitialTransformC),
        ctypes.POINTER(IRConfigC),
        ctypes.POINTER(IRRefineResultC),
    ]
    ir_dll.psm_iterative_refine.restype = ctypes.c_int
    ir_dll.psm_free_refine_result.argtypes = [ctypes.POINTER(IRRefineResultC)]
    ir_dll.psm_free_refine_result.restype = None

    np.random.seed(42)
    n_stars = 1000

    center_ra_rad = np.radians(272.825523)
    center_dec_rad = np.radians(-13.131606)
    scale = 6.294
    deg_to_px = 3600.0 / scale

    img_x = np.random.uniform(-2000, 2000, n_stars).astype(np.float64)
    img_y = np.random.uniform(-1500, 1500, n_stars).astype(np.float64)
    img_flux = np.random.uniform(100, 10000, n_stars).astype(np.float64)
    img_saturated = np.zeros(n_stars, dtype=np.int32)
    img_saturated[:50] = 1

    offset_x = 10.5
    offset_y = -5.3
    rotation = np.radians(0.5)
    cos_r, sin_r = np.cos(rotation), np.sin(rotation)

    cat_x_px = (cos_r * img_x - sin_r * img_y) + offset_x + np.random.normal(0, 0.5, n_stars)
    cat_y_px = (sin_r * img_x + cos_r * img_y) + offset_y + np.random.normal(0, 0.5, n_stars)
    cat_mag = np.random.uniform(5, 15, n_stars).astype(np.float64)

    cat_x_deg = cat_x_px / deg_to_px
    cat_y_deg = cat_y_px / deg_to_px

    r_deg = np.sqrt(cat_x_deg**2 + cat_y_deg**2)
    c_rad = np.arctan(r_deg)
    sin_c = np.sin(c_rad)
    cos_c = np.cos(c_rad)
    sin_dec0 = np.sin(center_dec_rad)
    cos_dec0 = np.cos(center_dec_rad)

    cat_dec_rad = np.arcsin(cos_c * sin_dec0 + sin_c * cos_dec0 * cat_y_deg / np.maximum(r_deg, 1e-10))
    cat_ra_rad = center_ra_rad + np.arctan2(cat_x_deg * sin_c / np.maximum(r_deg, 1e-10), 
                                             cos_c * cos_dec0 - sin_c * sin_dec0 * cat_y_deg / np.maximum(r_deg, 1e-10))

    cat_ra = np.degrees(cat_ra_rad)
    cat_dec = np.degrees(cat_dec_rad)

    center_ra = 272.825523
    center_dec = -13.131606

    print(f"\n=== 测试数据 ===")
    print(f"  星点数: {n_stars}")
    print(f"  饱和星: {np.sum(img_saturated)}")
    print(f"  真实偏移: ({offset_x:.2f}, {offset_y:.2f}) px")
    print(f"  真实旋转: {np.degrees(rotation):.3f}°")

    img_stars = IRImageStarsC()
    img_stars.img_x = img_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_y = img_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_flux = img_flux.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_count = n_stars
    img_stars.img_saturated = img_saturated.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    img_stars.n_saturated = int(np.sum(img_saturated))

    cat_stars = IRCatalogStarsC()
    cat_stars.cat_ra = cat_ra.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_dec = cat_dec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_mag = cat_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_count = n_stars

    init_transform = IRInitialTransformC()
    init_transform.center_ra = center_ra
    init_transform.center_dec = center_dec
    init_transform.rotation_deg = 0.0
    init_transform.scale_arcsec_px = scale
    init_transform.flip_mode = 0
    init_transform.img_width = 4500
    init_transform.img_height = 3600

    config = IRConfigC()
    config.max_stars_triangle = 500
    config.tri_ratio_radius = 0.002
    config.tri_min_area = 100.0
    config.tri_max_ba_ratio = 0.95
    config.tri_equilateral_thresh = 0.1
    config.grid_size = 5
    config.outlier_angle_thresh = 1.57079632679
    config.outlier_mag_ratio = 3.0
    config.max_iterations = 5
    config.converge_thresh = 0.01
    config.match_threshold = 50.0

    print(f"\n=== 调用 iterative_refine.dll ===")
    t0 = time.time()
    result = IRRefineResultC()
    ret = ir_dll.psm_iterative_refine(
        ctypes.byref(img_stars),
        ctypes.byref(cat_stars),
        ctypes.byref(init_transform),
        ctypes.byref(config),
        ctypes.byref(result)
    )
    elapsed = time.time() - t0

    if ret != 0:
        print(f"  失败, 返回码: {ret}")
        return

    print(f"  耗时: {elapsed:.2f}s")
    print(f"\n=== 结果 ===")
    print(f"  最终中心: RA={result.final_ra:.6f}°, Dec={result.final_dec:.6f}°")
    print(f"  三角形匹配数: {result.triangle_matches}")
    print(f"  匹配星对数: {result.matched_count}")

    delta_ra = (result.final_ra - center_ra) * 3600 * np.cos(np.radians(center_dec))
    delta_dec = (result.final_dec - center_dec) * 3600
    print(f"\n=== 中心修正量 ===")
    print(f"  ΔRA = {delta_ra:.2f}\" (期望: {offset_x * scale:.2f}\")")
    print(f"  ΔDec = {delta_dec:.2f}\" (期望: {offset_y * scale:.2f}\")")

    if result.distortion_valid:
        print(f"\n=== 畸变模型 ===")
        print(f"  a0={result.dist_a0:.4f}, b0={result.dist_b0:.4f}")

    ir_dll.psm_free_refine_result(ctypes.byref(result))

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

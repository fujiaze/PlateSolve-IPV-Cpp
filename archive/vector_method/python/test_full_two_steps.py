#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整两步测试脚本
第一步: star_alignment.dll 粗匹配
第二步: psm_iterative_refine.dll 迭代精化
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

from lib.astro_image_io.python.astro_image_io import ImageReader
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
from star_detector import StarDetector, SDetParamsPy

def load_dll(dll_path):
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    try:
        os.add_dll_directory(dll_dir)
    except OSError:
        pass
    return ctypes.cdll.LoadLibrary(dll_path)

def main():
    print("=" * 70)
    print("完整两步测试: 第一步粗匹配 + 第二步迭代精化")
    print("=" * 70)

    test_image_path = os.path.join(project_root, "testdata", "lights", "panel1",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts")
    dr3_dir = os.path.join(project_root, "GaiaDR3")

    print("\n=== 步骤1: 读取FITS图像 ===")
    t0 = time.time()
    reader = ImageReader()
    with reader.read(test_image_path) as img_hdr:
        w, h = img_hdr.width, img_hdr.height
        image_array = img_hdr.data.copy()
        wcs = img_hdr.wcs
        has_wcs = img_hdr.has_wcs and wcs is not None
        if has_wcs:
            init_ra = wcs.crval1
            init_dec = wcs.crval2
            scale = wcs.pixel_scale
            crpix_x = wcs.crpix1 - 1
            crpix_y = wcs.crpix2 - 1
        else:
            init_ra = 266.4167
            init_dec = -29.0078
            scale = 6.188
            crpix_x = w / 2.0
            crpix_y = h / 2.0
            print("  图像无WCS信息，使用银心坐标和估算比例尺")
    exptime = img_hdr.metadata.calibration.exptime if img_hdr.metadata.calibration else 180.0
    focal_mm = 200.0
    print(f"  图像: {w}x{h}, 像元尺度: {scale:.3f}\"/px")
    print(f"  WCS中心: RA={init_ra:.6f}°, Dec={init_dec:.6f}°")
    print(f"  读取耗时: {time.time()-t0:.2f}s")

    print("\n=== 步骤2: 星点检测 ===")
    t0 = time.time()
    sd_params = SDetParamsPy(iterativeClipSigma=9.0, fwhmClipSigma=3.0, maxAxisRatio=2.0, fitRadius=8)
    detector = StarDetector(params=sd_params)
    coords, fluxes, saturated = detector.detect_ex(image_array)
    detector.close()
    img_x = np.array([x - crpix_x for x, y in coords], dtype=np.float64)
    img_y = np.array([crpix_y - y for x, y in coords], dtype=np.float64)
    img_flux = np.array(fluxes, dtype=np.float64)
    saturated_arr = np.array(saturated, dtype=np.int32)
    n_saturated = int(np.sum(saturated_arr))
    print(f"  检测星数: {len(coords)}, 饱和星: {n_saturated}, 耗时: {time.time()-t0:.1f}s")
    print(f"  CRPIX: ({crpix_x:.1f}, {crpix_y:.1f})")
    print(f"  图像坐标范围: x=[{img_x.min():.0f}, {img_x.max():.0f}], y=[{img_y.min():.0f}, {img_y.max():.0f}]")

    print("\n=== 步骤3: Gaia查询 ===")
    t0 = time.time()
    gaia_dll = load_dll(os.path.join(project_root, "lib", "plate_solve", "plate_solve.dll"))
    gaia_dll.gaia_client_create.argtypes = [ctypes.c_char_p]
    gaia_dll.gaia_client_create.restype = ctypes.c_void_p
    gaia_dll.gaia_client_destroy.argtypes = [ctypes.c_void_p]
    gaia_dll.gaia_client_cone_search_for_solver.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
    ]
    gaia_dll.gaia_client_cone_search_for_solver.restype = ctypes.c_int

    gaia_handle = gaia_dll.gaia_client_create(dr3_dir.encode("utf-8"))
    fov_diag = np.sqrt(w * w + h * h) * scale / 3600
    query_radius = fov_diag * 1.2 / 2

    mag_low, mag_high = 6.0, 22.0
    target_count = int(len(coords) * 1.5)
    msvcrt = ctypes.CDLL("msvcrt.dll")
    for i in range(10):
        mag_mid = (mag_low + mag_high) / 2.0
        ra_ptr, dec_ptr, mag_ptr = ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_float)()
        n_stars = ctypes.c_int()
        gaia_dll.gaia_client_cone_search_for_solver(gaia_handle, init_ra, init_dec, query_radius, mag_mid,
            ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars))
        count = n_stars.value
        msvcrt.free(ra_ptr); msvcrt.free(dec_ptr); msvcrt.free(mag_ptr)
        if count > target_count: mag_high = mag_mid
        else: mag_low = mag_mid
        if mag_high - mag_low < 0.1: break

    final_mag = (mag_low + mag_high) / 2.0
    ra_ptr, dec_ptr, mag_ptr = ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_float)()
    n_stars = ctypes.c_int()
    gaia_dll.gaia_client_cone_search_for_solver(gaia_handle, init_ra, init_dec, query_radius, final_mag,
        ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars))
    raw_count = n_stars.value
    cat_ra = np.array([ra_ptr[i] for i in range(raw_count)], dtype=np.float64)
    cat_dec = np.array([dec_ptr[i] for i in range(raw_count)], dtype=np.float64)
    cat_mag = np.array([float(mag_ptr[i]) for i in range(raw_count)], dtype=np.float64)
    msvcrt.free(ra_ptr); msvcrt.free(dec_ptr); msvcrt.free(mag_ptr)
    gaia_dll.gaia_client_destroy(gaia_handle)

    sort_idx = np.argsort(cat_mag)
    cat_ra, cat_dec, cat_mag = cat_ra[sort_idx], cat_dec[sort_idx], cat_mag[sort_idx]
    print(f"  Gaia星数: {len(cat_ra)}, 极限星等: {final_mag:.2f}, 耗时: {time.time()-t0:.1f}s")

    print("\n=== 步骤4: 投影Gaia星到像素坐标 (Gnomonic) ===")
    center_ra_rad = np.radians(init_ra)
    center_dec_rad = np.radians(init_dec)
    cat_ra_rad = np.radians(cat_ra)
    cat_dec_rad = np.radians(cat_dec)
    cos_dec = np.cos(cat_dec_rad)
    sin_dec = np.sin(cat_dec_rad)
    cos_dec0 = np.cos(center_dec_rad)
    sin_dec0 = np.sin(center_dec_rad)
    ra_diff = cat_ra_rad - center_ra_rad
    cos_ra_diff = np.cos(ra_diff)
    sin_ra_diff = np.sin(ra_diff)
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff
    valid = cos_c > 1e-10
    cat_ra, cat_dec, cat_mag = cat_ra[valid], cat_dec[valid], cat_mag[valid]
    cat_ra_rad, cat_dec_rad = cat_ra_rad[valid], cat_dec_rad[valid]
    cos_dec, sin_dec, cos_c = cos_dec[valid], sin_dec[valid], cos_c[valid]
    cos_ra_diff, sin_ra_diff = cos_ra_diff[valid], sin_ra_diff[valid]
    xi = cos_dec * sin_ra_diff / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c
    rad_to_px = 180.0 / np.pi * 3600.0 / scale
    cat_x_px = xi * rad_to_px
    cat_y_px = -eta * rad_to_px
    print(f"  投影后星数: {len(cat_x_px)}")
    print(f"  Gaia像素坐标范围: x=[{cat_x_px.min():.1f}, {cat_x_px.max():.1f}], y=[{cat_y_px.min():.1f}, {cat_y_px.max():.1f}]")

    print("\n" + "=" * 70)
    print("第一步: star_alignment.dll 粗匹配")
    print("=" * 70)

    sa_dll = load_dll(os.path.join(project_root, "lib", "plate_solve", "modules", "star_alignment", "star_alignment.dll"))

    class PSMStarAlignmentInputC(ctypes.Structure):
        _fields_ = [
            ("img_x", ctypes.POINTER(ctypes.c_double)),
            ("img_y", ctypes.POINTER(ctypes.c_double)),
            ("img_count", ctypes.c_int),
            ("cat_x", ctypes.POINTER(ctypes.c_double)),
            ("cat_y", ctypes.POINTER(ctypes.c_double)),
            ("cat_mag", ctypes.POINTER(ctypes.c_double)),
            ("cat_count", ctypes.c_int),
            ("n_img_bright", ctypes.c_int),
            ("n_cat_bright", ctypes.c_int),
            ("max_dist_px", ctypes.c_double),
            ("max_iterations", ctypes.c_int),
            ("match_threshold", ctypes.c_double),
            ("img_saturated", ctypes.POINTER(ctypes.c_int)),
            ("n_saturated", ctypes.c_int),
        ]

    class PSMAffineC(ctypes.Structure):
        _fields_ = [("a0", ctypes.c_double), ("a1", ctypes.c_double), ("a2", ctypes.c_double),
                    ("b0", ctypes.c_double), ("b1", ctypes.c_double), ("b2", ctypes.c_double)]

    class PSMStarAlignmentResultC(ctypes.Structure):
        _fields_ = [
            ("offset_x", ctypes.c_double), ("offset_y", ctypes.c_double),
            ("rotation_deg", ctypes.c_double), ("scale_factor", ctypes.c_double),
            ("flip_mode", ctypes.c_int),
            ("a0", ctypes.c_double), ("a1", ctypes.c_double), ("a2", ctypes.c_double),
            ("a3", ctypes.c_double), ("a4", ctypes.c_double), ("a5", ctypes.c_double),
            ("b0", ctypes.c_double), ("b1", ctypes.c_double), ("b2", ctypes.c_double),
            ("b3", ctypes.c_double), ("b4", ctypes.c_double), ("b5", ctypes.c_double),
            ("matched_count", ctypes.c_int), ("rms_px", ctypes.c_double),
            ("mean_dist_px", ctypes.c_double),
            ("img_indices", ctypes.POINTER(ctypes.c_int)),
            ("cat_indices", ctypes.POINTER(ctypes.c_int)),
            ("distortion_valid", ctypes.c_int), ("rms_affine_px", ctypes.c_double),
        ]

    sa_dll.psm_star_align.argtypes = [ctypes.POINTER(PSMStarAlignmentInputC), ctypes.POINTER(PSMStarAlignmentResultC)]
    sa_dll.psm_star_align.restype = ctypes.c_int
    sa_dll.psm_free_star_alignment_result.argtypes = [ctypes.POINTER(PSMStarAlignmentResultC)]

    sa_input = PSMStarAlignmentInputC()
    sa_input.img_x = img_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.img_y = img_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.img_count = len(img_x)
    sa_input.cat_x = cat_x_px.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_y = cat_y_px.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_mag = cat_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_count = len(cat_ra)
    sa_input.n_img_bright = min(500, len(img_x))
    sa_input.n_cat_bright = min(600, len(cat_ra))
    sa_input.max_dist_px = 25.0
    sa_input.max_iterations = 5
    sa_input.match_threshold = 10.0
    sa_input.img_saturated = saturated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    sa_input.n_saturated = n_saturated

    sa_result = PSMStarAlignmentResultC()
    t0 = time.time()
    ret = sa_dll.psm_star_align(ctypes.byref(sa_input), ctypes.byref(sa_result))
    step1_time = time.time() - t0

    if ret != 0:
        print(f"  第一步失败, 返回码: {ret}")
        return

    flip_names = {0: "无翻转", 1: "X翻转", 2: "Y翻转", 3: "XY翻转"}
    print(f"  耗时: {step1_time:.2f}s")
    print(f"  匹配星对: {sa_result.matched_count}")
    print(f"  RMS: {sa_result.rms_px:.3f}px = {sa_result.rms_px * scale:.2f}\"")
    print(f"  偏移: ({sa_result.offset_x:.2f}, {sa_result.offset_y:.2f}) px")
    print(f"  旋转: {sa_result.rotation_deg:.3f}°")
    print(f"  比例尺因子: {sa_result.scale_factor:.6f}")
    print(f"  翻转: {flip_names.get(sa_result.flip_mode, str(sa_result.flip_mode))}")
    print(f"  仿射: a0={sa_result.a0:.4f} a1={sa_result.a1:.6f} a2={sa_result.a2:.6f}")
    print(f"        b0={sa_result.b0:.4f} b1={sa_result.b1:.6f} b2={sa_result.b2:.6f}")

    step1_ra = init_ra + sa_result.offset_x * scale / 3600.0 / np.cos(np.radians(init_dec))
    step1_dec = init_dec + sa_result.offset_y * scale / 3600.0
    step1_rotation = sa_result.rotation_deg
    step1_flip = sa_result.flip_mode
    step1_scale = scale * sa_result.scale_factor

    print(f"\n  精确中心: RA={step1_ra:.6f}°, Dec={step1_dec:.6f}°")

    sa_dll.psm_free_star_alignment_result(ctypes.byref(sa_result))

    print("\n" + "=" * 70)
    print("第二步: psm_iterative_refine.dll 迭代精化")
    print("=" * 70)

    ir_dll = load_dll(os.path.join(project_root, "lib", "plate_solve", "modules", "iterative_refine", "psm_iterative_refine.dll"))

    class IRInitialTransformC(ctypes.Structure):
        _fields_ = [
            ("center_ra", ctypes.c_double), ("center_dec", ctypes.c_double),
            ("rotation_deg", ctypes.c_double), ("scale_arcsec_px", ctypes.c_double),
            ("flip_mode", ctypes.c_int), ("img_width", ctypes.c_int), ("img_height", ctypes.c_int),
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
            ("cat_x_px", ctypes.POINTER(ctypes.c_double)),
            ("cat_y_px", ctypes.POINTER(ctypes.c_double)),
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
            ("final_ra", ctypes.c_double), ("final_dec", ctypes.c_double),
            ("final_rotation", ctypes.c_double), ("final_scale", ctypes.c_double),
            ("dist_a0", ctypes.c_double), ("dist_a1", ctypes.c_double),
            ("dist_a2", ctypes.c_double), ("dist_a3", ctypes.c_double),
            ("dist_a4", ctypes.c_double), ("dist_a5", ctypes.c_double),
            ("dist_b0", ctypes.c_double), ("dist_b1", ctypes.c_double),
            ("dist_b2", ctypes.c_double), ("dist_b3", ctypes.c_double),
            ("dist_b4", ctypes.c_double), ("dist_b5", ctypes.c_double),
            ("distortion_valid", ctypes.c_int),
            ("matched_count", ctypes.c_int),
            ("rms_x", ctypes.c_double), ("rms_y", ctypes.c_double),
            ("rms_total", ctypes.c_double), ("rms_arcsec", ctypes.c_double),
            ("iteration_count", ctypes.c_int), ("triangle_matches", ctypes.c_int),
            ("img_indices", ctypes.POINTER(ctypes.c_int)),
            ("cat_indices", ctypes.POINTER(ctypes.c_int)),
            ("residual_x", ctypes.POINTER(ctypes.c_double)),
            ("residual_y", ctypes.POINTER(ctypes.c_double)),
        ]

    ir_dll.psm_iterative_refine.argtypes = [
        ctypes.POINTER(IRImageStarsC), ctypes.POINTER(IRCatalogStarsC),
        ctypes.POINTER(IRInitialTransformC), ctypes.POINTER(IRConfigC),
        ctypes.POINTER(IRRefineResultC),
    ]
    ir_dll.psm_iterative_refine.restype = ctypes.c_int
    ir_dll.psm_free_refine_result.argtypes = [ctypes.POINTER(IRRefineResultC)]

    img_stars = IRImageStarsC()
    img_stars.img_x = img_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_y = img_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_flux = img_flux.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_count = len(img_x)
    img_stars.img_saturated = saturated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    img_stars.n_saturated = n_saturated

    cat_stars = IRCatalogStarsC()
    cat_stars.cat_ra = cat_ra.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_dec = cat_dec.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_mag = cat_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_x_px = cat_x_px.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_y_px = cat_y_px.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_count = len(cat_ra)

    init_transform = IRInitialTransformC()
    init_transform.center_ra = step1_ra
    init_transform.center_dec = step1_dec
    init_transform.rotation_deg = step1_rotation
    init_transform.scale_arcsec_px = step1_scale
    init_transform.flip_mode = step1_flip
    init_transform.img_width = w
    init_transform.img_height = h

    print(f"\n  第二步初始化参数:")
    print(f"    center_ra={step1_ra:.6f}°")
    print(f"    center_dec={step1_dec:.6f}°")
    print(f"    rotation={step1_rotation:.3f}°")
    print(f"    scale={step1_scale:.3f}\"/px (修正后)")
    print(f"    flip_mode={step1_flip}")

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

    ir_result = IRRefineResultC()
    t0 = time.time()
    ret = ir_dll.psm_iterative_refine(
        ctypes.byref(img_stars), ctypes.byref(cat_stars),
        ctypes.byref(init_transform), ctypes.byref(config),
        ctypes.byref(ir_result)
    )
    step2_time = time.time() - t0

    if ret != 0:
        print(f"  第二步失败, 返回码: {ret}")
    else:
        print(f"  耗时: {step2_time:.2f}s")
        print(f"  三角形匹配: {ir_result.triangle_matches}")
        print(f"  匹配星对: {ir_result.matched_count}")
        print(f"  最终中心: RA={ir_result.final_ra:.6f}°, Dec={ir_result.final_dec:.6f}°")
        print(f"  最终旋转: {ir_result.final_rotation:.3f}°")
        print(f"  最终尺度: {ir_result.final_scale:.3f}\"/px")
        if ir_result.distortion_valid:
            print(f"  畸变: a0={ir_result.dist_a0:.4f} b0={ir_result.dist_b0:.4f}")

        delta_ra = (ir_result.final_ra - step1_ra) * 3600 * np.cos(np.radians(step1_dec))
        delta_dec = (ir_result.final_dec - step1_dec) * 3600
        print(f"\n  第二步修正: ΔRA={delta_ra:.2f}\" ΔDec={delta_dec:.2f}\"")

        print("\n=== 计算第二步拟合后的RMS ===")
        from scipy.spatial import cKDTree
        cat_valid_mask = (np.abs(cat_x_px) < w/2) & (np.abs(cat_y_px) < h/2)
        cat_x_valid = cat_x_px[cat_valid_mask]
        cat_y_valid = cat_y_px[cat_valid_mask]
        
        tree = cKDTree(np.column_stack([cat_x_valid, cat_y_valid]))
        dists, idxs = tree.query(np.column_stack([img_x, img_y]), k=1)
        
        matched_mask = dists < 10.0
        n_matched = np.sum(matched_mask)
        rms_final = np.sqrt(np.mean(dists[matched_mask]**2)) if n_matched > 0 else 0
        print(f"  匹配星对: {n_matched} (距离<10px)")
        print(f"  最终RMS: {rms_final:.3f}px = {rms_final * scale:.2f}\"")
        print(f"  距离分布: 中位数={np.median(dists):.2f}px, 均值={np.mean(dists):.2f}px")
        print(f"  距离<5px: {np.sum(dists<5)}颗, <3px: {np.sum(dists<3)}颗, <1px: {np.sum(dists<1)}颗")

        print("\n=== 生成调试图像 ===")
        try:
            from PIL import Image as PILImage
            
            img_min, img_max = np.percentile(image_array, [0.5, 99.5])
            stretched = np.clip((image_array - img_min) / (img_max - img_min), 0, 1)
            rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float64)

            n_gaia_show = min(10000, len(cat_ra))
            print(f"  显示Gaia星: {n_gaia_show}颗 (最亮)")
            
            for i in range(n_gaia_show):
                gx = cat_x_px[i] + crpix_x
                gy = crpix_y - cat_y_px[i]
                ix, iy = int(gx), int(gy)
                if 0 <= ix < w and 0 <= iy < h:
                    cross_r = 5
                    for dx in range(-cross_r, cross_r + 1):
                        if 0 <= ix + dx < w:
                            rgb[iy, ix + dx, 0] = 1.0
                    for dy in range(-cross_r, cross_r + 1):
                        if 0 <= iy + dy < h:
                            rgb[iy + dy, ix, 0] = 1.0

            n_img_show = min(5000, len(img_x))
            for i in range(n_img_show):
                px = int(img_x[i] + crpix_x)
                py = int(crpix_y - img_y[i])
                if 0 <= px < w and 0 <= py < h:
                    circle_r = 3
                    for dx in range(-circle_r, circle_r + 1):
                        for dy in range(-circle_r, circle_r + 1):
                            if abs(dx*dx + dy*dy - circle_r*circle_r) < circle_r:
                                nx, ny = px + dx, py + dy
                                if 0 <= nx < w and 0 <= ny < h:
                                    rgb[ny, nx, 1] = 1.0

            rgb = np.clip(rgb, 0, 1)
            pil_img = PILImage.fromarray((rgb * 255).astype(np.uint8))
            output_path = os.path.join(os.path.dirname(test_image_path), 'step2_debug_annotated.png')
            pil_img.save(output_path)
            print(f"  调试图像已保存: {output_path}")
            print(f"  图例: 红色十字=Gaia星(最亮{n_gaia_show}颗), 绿色圆圈=图像星点(前{n_img_show}颗)")
        except Exception as e:
            print(f"  生成调试图像失败: {e}")

        ir_dll.psm_free_refine_result(ctypes.byref(ir_result))

    print("\n" + "=" * 70)
    print("测试完成!")
    print(f"总耗时: 第一步 {step1_time:.2f}s + 第二步 {step2_time:.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

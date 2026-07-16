#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Refine 测试脚本
功能: 测试第二步迭代精化模块
用途: 验证 psm_iterative_refine.dll 的三角形匹配+畸变拟合功能
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

PSM_FLIP_NONE = 0
PSM_FLIP_X = 1
PSM_FLIP_Y = 2
PSM_FLIP_XY = 3

FLIP_NAMES = {PSM_FLIP_NONE: "无翻转", PSM_FLIP_X: "X翻转", PSM_FLIP_Y: "Y翻转", PSM_FLIP_XY: "XY翻转"}


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


class PSMAffineC(ctypes.Structure):
    _fields_ = [
        ("a0", ctypes.c_double), ("a1", ctypes.c_double), ("a2", ctypes.c_double),
        ("b0", ctypes.c_double), ("b1", ctypes.c_double), ("b2", ctypes.c_double),
    ]


class PSMStarAlignmentResultC(ctypes.Structure):
    _fields_ = [
        ("offset_x", ctypes.c_double), ("offset_y", ctypes.c_double),
        ("rotation_deg", ctypes.c_double),
        ("flip_mode", ctypes.c_int),
        ("affine", PSMAffineC),
        ("matched_count", ctypes.c_int),
        ("rms_px", ctypes.c_double),
        ("img_indices", ctypes.POINTER(ctypes.c_int)),
        ("cat_indices", ctypes.POINTER(ctypes.c_int)),
        ("n_img", ctypes.c_int), ("n_cat", ctypes.c_int),
    ]


def load_dll(dll_path):
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    try:
        os.add_dll_directory(dll_dir)
    except OSError:
        pass
    return ctypes.cdll.LoadLibrary(dll_path)


def main():
    print("=" * 70)
    print("Iterative Refine 测试脚本 (psm_iterative_refine.dll)")
    print("=" * 70)

    test_image_path = os.path.join(project_root, "testdata", "lights", "panel1",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts")
    dr3_dir = os.path.join(project_root, "GaiaDR3")

    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像不存在: {test_image_path}")
        return
    if not os.path.exists(dr3_dir):
        print(f"错误: DR3数据库目录不存在: {dr3_dir}")
        return

    print("\n=== 步骤1: 读取FITS图像 + WCS ===")
    t0 = time.time()
    reader = ImageReader()
    with reader.read(test_image_path) as img_hdr:
        w, h = img_hdr.width, img_hdr.height
        image_array = img_hdr.data.copy()
        print(f"  图像: {w}x{h}")

        if not img_hdr.has_wcs:
            print("  错误: 图像没有WCS信息")
            return

        wcs = img_hdr.wcs
        init_ra = wcs.crval1
        init_dec = wcs.crval2
        scale = wcs.pixel_scale
        print(f"  WCS中心: RA={init_ra:.6f}°, Dec={init_dec:.6f}°")
        print(f"  像元尺度: {scale:.3f}\"/px")

    exptime = img_hdr.metadata.calibration.exptime if img_hdr.metadata.calibration else 180.0
    focal_mm = img_hdr.metadata.observation.focallen if img_hdr.metadata.observation and img_hdr.metadata.observation.focallen else None
    if focal_mm is None:
        focal_mm = img_hdr.get_keyword_float("FOCALLEN", 200.0)
    print(f"  曝光时间: {exptime:.1f}s")
    print(f"  焦距: {focal_mm:.1f}mm")
    print(f"  读取耗时: {time.time()-t0:.2f}s")

    print("\n=== 步骤2: 星点检测 ===")
    t0 = time.time()
    sd_params = SDetParamsPy(iterativeClipSigma=9.0, fwhmClipSigma=3.0, maxAxisRatio=2.0, fitRadius=8)
    detector = StarDetector(params=sd_params)
    coords, fluxes, saturated = detector.detect_ex(image_array)
    detector.close()
    det_time = time.time() - t0
    print(f"  检测星数: {len(coords)}, 耗时: {det_time:.1f}s")

    img_x = np.array([x for x, y in coords], dtype=np.float64)
    img_y = np.array([y for x, y in coords], dtype=np.float64)
    img_flux = np.array(fluxes, dtype=np.float64)
    saturated_arr = np.array(saturated, dtype=np.int32)
    n_saturated = int(np.sum(saturated_arr))
    print(f"  饱和星: {n_saturated}, 正常星: {len(coords) - n_saturated}")

    print("\n=== 步骤3: 初始化Gaia DR3客户端 ===")
    t0 = time.time()
    gaia_dll_path = os.path.join(project_root, "lib", "plate_solve", "plate_solve.dll")
    gaia_dll = load_dll(gaia_dll_path)

    gaia_dll.gaia_client_create.argtypes = [ctypes.c_char_p]
    gaia_dll.gaia_client_create.restype = ctypes.c_void_p
    gaia_dll.gaia_client_destroy.argtypes = [ctypes.c_void_p]
    gaia_dll.gaia_client_destroy.restype = None
    gaia_dll.gaia_client_cone_search_for_solver.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
    ]
    gaia_dll.gaia_client_cone_search_for_solver.restype = ctypes.c_int

    gaia_handle = gaia_dll.gaia_client_create(dr3_dir.encode("utf-8"))
    if not gaia_handle:
        print("  错误: 创建Gaia客户端失败")
        return
    print(f"  DR3客户端初始化成功, 耗时: {time.time()-t0:.2f}s")

    print("\n=== 步骤4: 二分法迭代极限星等 ===")
    t0 = time.time()
    fov_diag = np.sqrt(w * w + h * h) * scale / 3600
    query_radius = fov_diag * 1.2 / 2
    target_count = int(len(coords) * 1.5)

    est_mag = 6 + 1.5 * np.log10(focal_mm) + 2 * np.log10(exptime)
    print(f"  FOV对角线: {fov_diag:.2f}°, 查询半径: {query_radius:.2f}°")
    print(f"  检测星数: {len(coords)}, 目标Gaia星数: {target_count}")
    print(f"  估算极限星等: {est_mag:.2f}")

    mag_low = max(6.0, est_mag - 3)
    mag_high = min(22.0, est_mag + 3)
    msvcrt = ctypes.CDLL("msvcrt.dll")

    for i in range(10):
        mag_mid = (mag_low + mag_high) / 2.0
        ra_ptr, dec_ptr, mag_ptr = ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_float)()
        n_stars = ctypes.c_int()
        gaia_dll.gaia_client_cone_search_for_solver(
            gaia_handle, init_ra, init_dec, query_radius, mag_mid,
            ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars)
        )
        count = n_stars.value
        print(f"  二分法迭代 {i+1}: mag={mag_mid:.3f}, count={count}, target={target_count}")
        msvcrt.free(ra_ptr)
        msvcrt.free(dec_ptr)
        msvcrt.free(mag_ptr)
        if count == target_count:
            break
        elif count > target_count:
            mag_high = mag_mid
        else:
            mag_low = mag_mid
        if mag_high - mag_low < 0.1:
            break

    final_mag = (mag_low + mag_high) / 2.0
    print(f"  二分法搜索完成, 耗时: {time.time()-t0:.2f}s")
    print(f"  最终极限星等: {final_mag:.3f}")

    print("\n=== 步骤5: 查询Gaia DR3数据库 ===")
    t0 = time.time()

    ra_ptr, dec_ptr, mag_ptr = ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_double)(), ctypes.POINTER(ctypes.c_float)()
    n_stars = ctypes.c_int()
    gaia_dll.gaia_client_cone_search_for_solver(
        gaia_handle, init_ra, init_dec, query_radius, final_mag,
        ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars)
    )
    raw_count = n_stars.value
    cat_ra = np.array([ra_ptr[i] for i in range(raw_count)], dtype=np.float64)
    cat_dec = np.array([dec_ptr[i] for i in range(raw_count)], dtype=np.float64)
    cat_mag = np.array([float(mag_ptr[i]) for i in range(raw_count)], dtype=np.float64)

    sort_idx = np.argsort(cat_mag)
    cat_ra = cat_ra[sort_idx]
    cat_dec = cat_dec[sort_idx]
    cat_mag = cat_mag[sort_idx]

    msvcrt.free(ra_ptr)
    msvcrt.free(dec_ptr)
    msvcrt.free(mag_ptr)
    gaia_dll.gaia_client_destroy(gaia_handle)

    gaia_time = time.time() - t0
    print(f"  Gaia星数: {len(cat_ra)}, 耗时: {gaia_time:.2f}s")
    if len(cat_ra) > 0:
        print(f"  最亮星等: {cat_mag[0]:.2f}, 最暗星等: {cat_mag[-1]:.2f}")

    print("\n=== 步骤6: 第一步粗匹配 ===")
    t0 = time.time()
    sa_dll_path = os.path.join(project_root, "lib", "plate_solve", "modules", "star_alignment", "star_alignment.dll")
    sa_dll = load_dll(sa_dll_path)

    sa_dll.psm_star_align.argtypes = [
        ctypes.POINTER(PSMStarAlignmentInputC),
        ctypes.POINTER(PSMStarAlignmentResultC),
    ]
    sa_dll.psm_star_align.restype = ctypes.c_int
    sa_dll.psm_free_star_alignment_result.argtypes = [ctypes.POINTER(PSMStarAlignmentResultC)]
    sa_dll.psm_free_star_alignment_result.restype = None

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

    img_x_arr = np.ascontiguousarray(img_x, dtype=np.float64)
    img_y_arr = np.ascontiguousarray(img_y, dtype=np.float64)
    img_flux_arr = np.ascontiguousarray(img_flux, dtype=np.float64)
    cat_ra_arr = np.ascontiguousarray(cat_ra, dtype=np.float64)
    cat_dec_arr = np.ascontiguousarray(cat_dec, dtype=np.float64)
    cat_mag_arr = np.ascontiguousarray(cat_mag, dtype=np.float64)

    sa_input = PSMStarAlignmentInputC()
    sa_input.img_x = img_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.img_y = img_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.img_count = len(img_x_arr)
    sa_input.cat_x = cat_ra_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_y = cat_dec_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_mag = cat_mag_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_count = len(cat_ra_arr)
    sa_input.n_img_bright = min(500, len(img_x_arr))
    sa_input.n_cat_bright = min(500, len(cat_ra_arr))
    sa_input.max_dist_px = 25.0
    sa_input.max_iterations = 5
    sa_input.match_threshold = 10.0
    sa_input.img_saturated = saturated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    sa_input.n_saturated = n_saturated

    sa_result = PSMStarAlignmentResultC()
    ret = sa_dll.psm_star_align(
        ctypes.byref(sa_input),
        ctypes.byref(sa_result),
    )
    step1_time = time.time() - t0

    if ret != 0:
        print(f"  第一步失败, 返回码: {ret}")
        return

    step1_ra = init_ra + sa_result.offset_x * scale / 3600.0 / np.cos(np.radians(init_dec))
    step1_dec = init_dec + sa_result.offset_y * scale / 3600.0
    step1_rotation = sa_result.rotation_deg
    step1_flip = sa_result.flip_mode

    print(f"  第一步耗时: {step1_time:.2f}s")
    print(f"  匹配星对: {sa_result.matched_count}")
    print(f"  RMS: {sa_result.rms_px:.3f}px")
    print(f"  精确中心: RA={step1_ra:.6f}°, Dec={step1_dec:.6f}°")
    print(f"  旋转角: {step1_rotation:.3f}°, 翻转: {FLIP_NAMES.get(step1_flip, str(step1_flip))}")
    sa_dll.psm_star_alignment_free(ctypes.byref(sa_result))

    print("\n=== 步骤7: 第二步迭代精化 ===")
    t0 = time.time()
    ir_dll_path = os.path.join(project_root, "lib", "plate_solve", "modules", "iterative_refine", "psm_iterative_refine.dll")
    ir_dll = load_dll(ir_dll_path)

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

    img_stars = IRImageStarsC()
    img_stars.img_x = img_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_y = img_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_flux = img_flux_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    img_stars.img_count = len(img_x_arr)
    img_stars.img_saturated = saturated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    img_stars.n_saturated = n_saturated

    cat_stars = IRCatalogStarsC()
    cat_stars.cat_ra = cat_ra_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_dec = cat_dec_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_mag = cat_mag_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    cat_stars.cat_count = len(cat_ra_arr)

    init_transform = IRInitialTransformC()
    init_transform.center_ra = step1_ra
    init_transform.center_dec = step1_dec
    init_transform.rotation_deg = step1_rotation
    init_transform.scale_arcsec_px = scale
    init_transform.flip_mode = step1_flip
    init_transform.img_width = w
    init_transform.img_height = h

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

    result = IRRefineResultC()
    ret = ir_dll.psm_iterative_refine(
        ctypes.byref(img_stars),
        ctypes.byref(cat_stars),
        ctypes.byref(init_transform),
        ctypes.byref(config),
        ctypes.byref(result)
    )
    step2_time = time.time() - t0

    if ret != 0:
        print(f"  第二步失败, 返回码: {ret}")
        return

    print(f"  第二步耗时: {step2_time:.2f}s")
    print(f"\n  === 第二步结果 ===")
    print(f"  最终中心: RA={result.final_ra:.6f}°, Dec={result.final_dec:.6f}°")
    print(f"  最终旋转: {result.final_rotation:.3f}°")
    print(f"  最终尺度: {result.final_scale:.3f}\"/px")
    print(f"  三角形匹配数: {result.triangle_matches}")
    print(f"  匹配星对数: {result.matched_count}")
    print(f"  迭代次数: {result.iteration_count}")
    print(f"  RMS: x={result.rms_x:.3f}px, y={result.rms_y:.3f}px, total={result.rms_total:.3f}px")
    print(f"  RMS角秒: {result.rms_arcsec:.3f}\"")

    if result.distortion_valid:
        print(f"\n  === 畸变模型 ===")
        print(f"  a: [{result.dist_a0:.4f}, {result.dist_a1:.6f}, {result.dist_a2:.6f}, {result.dist_a3:.2e}, {result.dist_a4:.2e}, {result.dist_a5:.2e}]")
        print(f"  b: [{result.dist_b0:.4f}, {result.dist_b1:.6f}, {result.dist_b2:.6f}, {result.dist_b3:.2e}, {result.dist_b4:.2e}, {result.dist_b5:.2e}]")

    print("\n=== 中心修正量 ===")
    delta_ra = (result.final_ra - step1_ra) * 3600 * np.cos(np.radians(step1_dec))
    delta_dec = (result.final_dec - step1_dec) * 3600
    print(f"  ΔRA = {delta_ra:.2f}\"")
    print(f"  ΔDec = {delta_dec:.2f}\"")

    ir_dll.psm_free_refine_result(ctypes.byref(result))

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

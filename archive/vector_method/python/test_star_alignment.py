#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Star Alignment 测试脚本
功能: 完整 star alignment 流程 — 读取FITS+WCS → 星点检测 → Gaia查询(二分法极限星等) → Gnomonic投影 → DLL匹配 → WCS修正 → 标注图像
用途: 验证 star_alignment.dll 匹配精度和WCS修正效果
"""

import os
import sys
import time
import ctypes
import numpy as np

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, project_root)

from lib.astro_image_io.python.astro_image_io import ImageReader
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
from star_detector import StarDetector, SDetParamsPy

PSOLVE_DB_DR3 = 1
PSOLVE_DB_DR3SP = 2

PSM_FLIP_NONE = 0
PSM_FLIP_X = 1
PSM_FLIP_Y = 2
PSM_FLIP_XY = 3

FLIP_NAMES = {PSM_FLIP_NONE: "无翻转", PSM_FLIP_X: "X翻转", PSM_FLIP_Y: "Y翻转", PSM_FLIP_XY: "XY翻转"}


class PSMStarAlignmentInput(ctypes.Structure):
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


class PSMStarAlignmentResult(ctypes.Structure):
    _fields_ = [
        ("offset_x", ctypes.c_double),
        ("offset_y", ctypes.c_double),
        ("rotation_deg", ctypes.c_double),
        ("scale_factor", ctypes.c_double),
        ("flip_mode", ctypes.c_int),
        ("a0", ctypes.c_double), ("a1", ctypes.c_double), ("a2", ctypes.c_double),
        ("a3", ctypes.c_double), ("a4", ctypes.c_double), ("a5", ctypes.c_double),
        ("b0", ctypes.c_double), ("b1", ctypes.c_double), ("b2", ctypes.c_double),
        ("b3", ctypes.c_double), ("b4", ctypes.c_double), ("b5", ctypes.c_double),
        ("matched_count", ctypes.c_int),
        ("rms_px", ctypes.c_double),
        ("mean_dist_px", ctypes.c_double),
        ("img_indices", ctypes.POINTER(ctypes.c_int)),
        ("cat_indices", ctypes.POINTER(ctypes.c_int)),
        ("distortion_valid", ctypes.c_int),
        ("rms_affine_px", ctypes.c_double),
    ]


def load_star_alignment_dll():
    dll_path = os.path.join(project_root, "lib", "plate_solve", "modules", "star_alignment", "star_alignment.dll")
    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.isdir(mingw_bin):
        os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(mingw_bin)
        except OSError:
            pass
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    try:
        os.add_dll_directory(dll_dir)
    except OSError:
        pass
    dll = ctypes.cdll.LoadLibrary(dll_path)

    dll.psm_star_align.argtypes = [ctypes.POINTER(PSMStarAlignmentInput), ctypes.POINTER(PSMStarAlignmentResult)]
    dll.psm_star_align.restype = ctypes.c_int
    dll.psm_free_star_alignment_result.argtypes = [ctypes.POINTER(PSMStarAlignmentResult)]
    dll.psm_free_star_alignment_result.restype = None
    return dll


def load_gaia_dll():
    dll_path = os.path.join(project_root, "lib", "plate_solve", "plate_solve.dll")
    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.isdir(mingw_bin):
        os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
    dll = ctypes.CDLL(dll_path)

    dll.gaia_client_create.argtypes = [ctypes.c_char_p]
    dll.gaia_client_create.restype = ctypes.c_void_p
    dll.gaia_client_destroy.argtypes = [ctypes.c_void_p]
    dll.gaia_client_destroy.restype = None
    dll.gaia_client_cone_search_for_solver.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.gaia_client_cone_search_for_solver.restype = ctypes.c_int
    return dll


def project_gaia_to_pixels(stars, center_ra, center_dec, scale_arcsec_px):
    rad_to_px = 180.0 / np.pi * 3600.0 / scale_arcsec_px
    ra = np.array([s[0] for s in stars])
    dec = np.array([s[1] for s in stars])
    mag = np.array([s[2] for s in stars])
    dec_rad = np.radians(dec)
    ra_rad = np.radians(ra)
    center_dec_rad = np.radians(center_dec)
    center_ra_rad = np.radians(center_ra)
    cos_c = np.sin(center_dec_rad) * np.sin(dec_rad) + \
            np.cos(center_dec_rad) * np.cos(dec_rad) * np.cos(ra_rad - center_ra_rad)
    valid = cos_c > 1e-10
    ra, dec, mag, cos_c = ra[valid], dec[valid], mag[valid], cos_c[valid]
    dec_rad, ra_rad = dec_rad[valid], ra_rad[valid]
    px_rad = np.cos(dec_rad) * np.sin(ra_rad - center_ra_rad) / cos_c
    py_rad = (np.cos(center_dec_rad) * np.sin(dec_rad) -
              np.sin(center_dec_rad) * np.cos(dec_rad) * np.cos(ra_rad - center_ra_rad)) / cos_c
    return px_rad * rad_to_px, -py_rad * rad_to_px, mag


def bisection_mag_search(gaia_dll, gaia_handle, ra, dec, radius_deg, target_count,
                         mag_low=6.0, mag_high=22.0, tolerance=0.1, max_iter=10):
    from ctypes import POINTER, c_double, c_float, c_int, byref
    best_mag = 15.0
    best_count = 0
    msvcrt = ctypes.CDLL("msvcrt.dll")

    for i in range(max_iter):
        mag_mid = (mag_low + mag_high) / 2.0
        ra_ptr, dec_ptr, mag_ptr = POINTER(c_double)(), POINTER(c_double)(), POINTER(c_float)()
        n_stars = c_int()
        gaia_dll.gaia_client_cone_search_for_solver(
            gaia_handle, ra, dec, radius_deg, mag_mid,
            byref(ra_ptr), byref(dec_ptr), byref(mag_ptr), byref(n_stars)
        )
        count = n_stars.value
        print(f"  二分法迭代 {i+1}: mag={mag_mid:.3f}, count={count}, target={target_count}")
        best_mag = mag_mid
        best_count = count
        msvcrt.free(ra_ptr)
        msvcrt.free(dec_ptr)
        msvcrt.free(mag_ptr)
        if count == target_count:
            break
        elif count > target_count:
            mag_high = mag_mid
        else:
            mag_low = mag_mid
        if mag_high - mag_low < tolerance:
            break

    return best_mag, best_count


def apply_wcs_correction(init_ra, init_dec, scale, cd1_1, cd1_2, cd2_1, cd2_2,
                         offset_x, offset_y, rotation_deg, flip_mode):
    rad_to_px = 180.0 / np.pi * 3600.0 / scale
    xi_rad = offset_x / rad_to_px
    eta_rad = offset_y / rad_to_px
    delta_ra = np.degrees(xi_rad) / np.cos(np.radians(init_dec))
    delta_dec = np.degrees(eta_rad)
    new_ra = init_ra + delta_ra
    new_dec = init_dec + delta_dec

    CD_old = np.array([[cd1_1, cd1_2], [cd2_1, cd2_2]])
    theta = np.radians(rotation_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    CD_new = R @ CD_old

    if flip_mode == PSM_FLIP_X:
        CD_new[0, :] *= -1
    elif flip_mode == PSM_FLIP_Y:
        CD_new[1, :] *= -1
    elif flip_mode == PSM_FLIP_XY:
        CD_new *= -1

    return new_ra, new_dec, CD_new


def draw_annotated_image(image_array, w, h, img_x, img_y, cat_x, cat_y, center_x, center_y,
                         result, output_path, cat_mag=None):
    try:
        from PIL import Image as PILImage

        img_min, img_max = np.percentile(image_array, [0.5, 99.5])
        stretched = np.clip((image_array - img_min) / (img_max - img_min), 0, 1)
        rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float64)

        a0, a1, a2 = result.a0, result.a1, result.a2
        b0, b1, b2 = result.b0, result.b1, result.b2
        a3, a4, a5 = result.a3, result.a4, result.a5
        b3, b4, b5 = result.b3, result.b4, result.b5
        has_distortion = result.distortion_valid
        flip_mode = result.flip_mode
        flip_x = 1 if flip_mode in [1, 3] else 0
        flip_y = 1 if flip_mode in [2, 3] else 0

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            print("  无法绘制匹配对: 仿射矩阵奇异")
            return

        inv_a1 = b2 / det
        inv_a2 = -a2 / det
        inv_b1 = -b1 / det
        inv_b2 = a1 / det

        n_pairs = min(500, result.matched_count)
        img_indices = result.img_indices
        cat_indices = result.cat_indices

        matched_cat_set = set(cat_indices[:result.matched_count])

        print(f"  绘制匹配对: {n_pairs} 对")

        for k in range(n_pairs):
            i_idx = img_indices[k]
            c_idx = cat_indices[k]

            px_img = int(img_x[i_idx] + center_x)
            py_img = int(img_y[i_idx] + center_y)

            gaia_x = cat_x[c_idx]
            gaia_y = cat_y[c_idx]
            fimg_x = inv_a1 * (gaia_x - a0) + inv_a2 * (gaia_y - b0)
            fimg_y = inv_b1 * (gaia_x - a0) + inv_b2 * (gaia_y - b0)
            if flip_x:
                fimg_x = -fimg_x
            if flip_y:
                fimg_y = -fimg_y
            px_gaia = int(fimg_x + center_x)
            py_gaia = int(fimg_y + center_y)

            if not (0 <= px_img < w and 0 <= py_img < h):
                continue
            if not (0 <= px_gaia < w and 0 <= py_gaia < h):
                continue

            circle_r = 8
            for dx in range(-circle_r, circle_r + 1):
                for dy in range(-circle_r, circle_r + 1):
                    if abs(dx*dx + dy*dy - circle_r*circle_r) < circle_r:
                        nx, ny = px_img + dx, py_img + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            rgb[ny, nx, 1] = 1.0

            cross_r = 6
            for dx in range(-cross_r, cross_r + 1):
                if 0 <= px_gaia + dx < w:
                    rgb[py_gaia, px_gaia + dx, 0] = 1.0
            for dy in range(-cross_r, cross_r + 1):
                if 0 <= py_gaia + dy < h:
                    rgb[py_gaia + dy, px_gaia, 0] = 1.0

            n_steps = max(abs(px_gaia - px_img), abs(py_gaia - py_img), 1)
            for step in range(n_steps + 1):
                t = step / n_steps
                lx = int(px_img + (px_gaia - px_img) * t)
                ly = int(py_img + (py_gaia - py_img) * t)
                if 0 <= lx < w and 0 <= ly < h:
                    rgb[ly, lx, 2] = 1.0

        n_unmatched_show = min(10000, len(cat_x))
        unmatched_count = 0
        for c_idx in range(n_unmatched_show):
            if c_idx in matched_cat_set:
                continue

            gaia_x = cat_x[c_idx]
            gaia_y = cat_y[c_idx]
            fimg_x = inv_a1 * (gaia_x - a0) + inv_a2 * (gaia_y - b0)
            fimg_y = inv_b1 * (gaia_x - a0) + inv_b2 * (gaia_y - b0)
            if flip_x:
                fimg_x = -fimg_x
            if flip_y:
                fimg_y = -fimg_y
            px_gaia = int(fimg_x + center_x)
            py_gaia = int(fimg_y + center_y)

            if not (0 <= px_gaia < w and 0 <= py_gaia < h):
                continue

            cross_r = 3
            for dx in range(-cross_r, cross_r + 1):
                if 0 <= px_gaia + dx < w:
                    rgb[py_gaia, px_gaia + dx, 0] = 1.0
                    rgb[py_gaia, px_gaia + dx, 1] = 1.0
            for dy in range(-cross_r, cross_r + 1):
                if 0 <= py_gaia + dy < h:
                    rgb[py_gaia + dy, px_gaia, 0] = 1.0
                    rgb[py_gaia + dy, px_gaia, 1] = 1.0
            unmatched_count += 1

        rgb = np.clip(rgb, 0, 1)
        pil_img = PILImage.fromarray((rgb * 255).astype(np.uint8))
        pil_img.save(output_path)
        print(f"  标注图像已保存: {output_path}")
        print(f"  图例: 绿色圆圈=图像星点, 红色十字=匹配Gaia星, 黄色十字=未匹配Gaia星, 蓝色连线=匹配对")
        print(f"  未匹配Gaia星(前{n_unmatched_show}颗最亮): {unmatched_count}颗在图像范围内")
    except Exception as e:
        print(f"  生成标注图像失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("=" * 70)
    print("Star Alignment 测试脚本 (star_alignment.dll)")
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
        cd1_1 = wcs.cd1_1
        cd1_2 = wcs.cd1_2
        cd2_1 = wcs.cd2_1
        cd2_2 = wcs.cd2_2
        print(f"  WCS中心: RA={init_ra:.6f}°, Dec={init_dec:.6f}°")
        print(f"  像元尺度: {scale:.3f}\"/px")
        print(f"  CD矩阵: [[{cd1_1:.6e}, {cd1_2:.6e}], [{cd2_1:.6e}, {cd2_2:.6e}]]")
        print(f"  WCS旋转角: {wcs.rotation_deg:.3f}°")

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

    half_w, half_h = w / 2.0, h / 2.0
    crpix_x = wcs.crpix1 - 1
    crpix_y = wcs.crpix2 - 1
    img_x = np.array([x - crpix_x for x, y in coords], dtype=np.float64)
    img_y = np.array([y - crpix_y for x, y in coords], dtype=np.float64)
    saturated_arr = np.array(saturated, dtype=np.int32)
    n_saturated = int(np.sum(saturated_arr))
    print(f"  饱和星: {n_saturated}, 正常星: {len(coords) - n_saturated}")
    print(f"  CRPIX: ({wcs.crpix1:.1f}, {wcs.crpix2:.1f}) -> 0-indexed: ({crpix_x:.1f}, {crpix_y:.1f})")
    print(f"  图像中心: ({half_w:.1f}, {half_h:.1f}), 偏移: ({crpix_x - half_w:.1f}, {crpix_y - half_h:.1f})px")
    print(f"  图像坐标范围: x=[{img_x.min():.0f}, {img_x.max():.0f}], y=[{img_y.min():.0f}, {img_y.max():.0f}]")

    print("\n=== 步骤3: 初始化Gaia DR3客户端 ===")
    t0 = time.time()
    gaia_dll = load_gaia_dll()
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

    final_mag, final_count = bisection_mag_search(
        gaia_dll, gaia_handle, init_ra, init_dec, query_radius, target_count,
        mag_low=max(6.0, est_mag - 3), mag_high=min(22.0, est_mag + 3)
    )
    print(f"  二分法搜索完成, 耗时: {time.time()-t0:.2f}s")
    print(f"  最终极限星等: {final_mag:.3f}, 实际返回星数: {final_count}")

    print("\n=== 步骤5: 查询Gaia DR3数据库 ===")
    t0 = time.time()
    from ctypes import POINTER, c_double, c_float, c_int, byref

    ra_ptr, dec_ptr, mag_ptr = POINTER(c_double)(), POINTER(c_double)(), POINTER(c_float)()
    n_stars = c_int()
    gaia_dll.gaia_client_cone_search_for_solver(
        gaia_handle, init_ra, init_dec, query_radius, final_mag,
        byref(ra_ptr), byref(dec_ptr), byref(mag_ptr), byref(n_stars)
    )
    raw_count = n_stars.value
    raw_stars = [(ra_ptr[i], dec_ptr[i], float(mag_ptr[i])) for i in range(raw_count)]
    raw_stars = sorted(raw_stars, key=lambda s: s[2])

    msvcrt = ctypes.CDLL("msvcrt.dll")
    msvcrt.free(ra_ptr)
    msvcrt.free(dec_ptr)
    msvcrt.free(mag_ptr)
    gaia_dll.gaia_client_destroy(gaia_handle)

    gaia_time = time.time() - t0
    print(f"  Gaia星数: {len(raw_stars)}, 耗时: {gaia_time:.2f}s")
    if len(raw_stars) > 0:
        print(f"  最亮星等: {raw_stars[0][2]:.2f}, 最暗星等: {raw_stars[-1][2]:.2f}")

    print("\n=== 步骤6: Gnomonic投影 + 比例尺映射 ===")
    t0 = time.time()
    cat_x, cat_y, cat_mag = project_gaia_to_pixels(raw_stars, init_ra, init_dec, scale)

    proj_time = time.time() - t0
    print(f"  投影后星数: {len(cat_x)}, 耗时: {proj_time:.2f}s")
    print(f"  Gaia坐标范围: x=[{cat_x.min():.0f}, {cat_x.max():.0f}], y=[{cat_y.min():.0f}, {cat_y.max():.0f}]")
    print(f"  图像坐标范围: x=[{img_x.min():.0f}, {img_x.max():.0f}], y=[{img_y.min():.0f}, {img_y.max():.0f}]")
    print(f"  Gaia坐标中位数: x={np.median(cat_x):.1f}, y={np.median(cat_y):.1f}")
    print(f"  图像坐标中位数: x={np.median(img_x):.1f}, y={np.median(img_y):.1f}")

    print("\n=== 步骤7: 调用 star_alignment.dll 匹配 ===")
    t0 = time.time()
    sa_dll = load_star_alignment_dll()

    img_x_arr = np.ascontiguousarray(img_x, dtype=np.float64)
    img_y_arr = np.ascontiguousarray(img_y, dtype=np.float64)
    cat_x_arr = np.ascontiguousarray(cat_x, dtype=np.float64)
    cat_y_arr = np.ascontiguousarray(cat_y, dtype=np.float64)
    cat_mag_arr = np.ascontiguousarray(cat_mag, dtype=np.float64)

    sa_input = PSMStarAlignmentInput()
    sa_input.img_x = img_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.img_y = img_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.img_count = len(img_x_arr)
    sa_input.cat_x = cat_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_y = cat_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_mag = cat_mag_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    sa_input.cat_count = len(cat_x_arr)
    sa_input.n_img_bright = min(2000, len(img_x_arr))
    sa_input.n_cat_bright = min(5000, len(cat_x_arr))
    sa_input.max_dist_px = 25.0
    sa_input.max_iterations = 5
    sa_input.match_threshold = 3
    sa_input.img_saturated = saturated_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    sa_input.n_saturated = n_saturated

    sa_result = PSMStarAlignmentResult()
    ret = sa_dll.psm_star_align(ctypes.byref(sa_input), ctypes.byref(sa_result))
    match_time = time.time() - t0

    if ret != 0:
        print(f"  匹配失败, 返回码: {ret}")
        return

    print(f"  匹配耗时: {match_time:.2f}s")
    print(f"  匹配星数: {sa_result.matched_count}")
    print(f"  RMS: {sa_result.rms_px:.3f}px = {sa_result.rms_px * scale:.2f}\"")
    print(f"  平均距离: {sa_result.mean_dist_px:.3f}px")
    print(f"  偏移量: offset_x={sa_result.offset_x:.2f}px, offset_y={sa_result.offset_y:.2f}px")
    print(f"  旋转角: {sa_result.rotation_deg:.3f}°")
    print(f"  比例尺因子: {sa_result.scale_factor:.6f}")
    print(f"  翻转模式: {FLIP_NAMES.get(sa_result.flip_mode, str(sa_result.flip_mode))}")
    print(f"  仿射参数: a0={sa_result.a0:.4f} a1={sa_result.a1:.6f} a2={sa_result.a2:.6f}")
    print(f"             b0={sa_result.b0:.4f} b1={sa_result.b1:.6f} b2={sa_result.b2:.6f}")
    print(f"  仿射RMS: {sa_result.rms_affine_px:.3f}px")
    if sa_result.distortion_valid:
        print(f"  畸变参数: a3={sa_result.a3:.2e} a4={sa_result.a4:.2e} a5={sa_result.a5:.2e}")
        print(f"             b3={sa_result.b3:.2e} b4={sa_result.b4:.2e} b5={sa_result.b5:.2e}")
        print(f"  ★ 畸变模型已启用 (RMS改善: {sa_result.rms_affine_px:.3f}→{sa_result.rms_px:.3f}px)")
    else:
        print(f"  畸变模型: 未启用")

    print("\n=== 匹配分析 ===")
    a0, a1, a2 = sa_result.a0, sa_result.a1, sa_result.a2
    b0, b1, b2 = sa_result.b0, sa_result.b1, sa_result.b2
    a3, a4, a5 = sa_result.a3, sa_result.a4, sa_result.a5
    b3, b4, b5 = sa_result.b3, sa_result.b4, sa_result.b5
    flip_mode = sa_result.flip_mode
    flip_x = 1 if flip_mode in [1, 3] else 0
    flip_y = 1 if flip_mode in [2, 3] else 0

    fimg_x = np.where(flip_x, -img_x, img_x)
    fimg_y = np.where(flip_y, -img_y, img_y)

    if sa_result.distortion_valid:
        tx = a0 + a1*fimg_x + a2*fimg_y + a3*fimg_x**2 + a4*fimg_x*fimg_y + a5*fimg_y**2
        ty = b0 + b1*fimg_x + b2*fimg_y + b3*fimg_x**2 + b4*fimg_x*fimg_y + b5*fimg_y**2
    else:
        tx = a0 + a1*fimg_x + a2*fimg_y
        ty = b0 + b1*fimg_x + b2*fimg_y

    from scipy.spatial import cKDTree
    tree = cKDTree(np.column_stack([cat_x, cat_y]))
    dists, idxs = tree.query(np.column_stack([tx, ty]), k=1)

    print(f"  图像星点总数: {len(img_x)}")
    print(f"  Gaia星点总数: {len(cat_x)}")
    print(f"  距离分布统计:")
    thresholds = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0]
    for th in thresholds:
        n_match = np.sum(dists < th)
        print(f"    < {th:4.1f}px: {n_match:5d} ({100*n_match/len(img_x):.1f}%)")
    print(f"  距离中位数: {np.median(dists):.3f}px")
    print(f"  距离均值: {np.mean(dists):.3f}px")
    print(f"  距离90%分位: {np.percentile(dists, 90):.3f}px")
    print(f"  距离95%分位: {np.percentile(dists, 95):.3f}px")

    print("\n=== 步骤8: WCS修正 ===")
    new_ra, new_dec, CD_new = apply_wcs_correction(
        init_ra, init_dec, scale, cd1_1, cd1_2, cd2_1, cd2_2,
        sa_result.offset_x, sa_result.offset_y, sa_result.rotation_deg, sa_result.flip_mode
    )

    print(f"\n  --- WCS修正前后对比 ---")
    print(f"  中心RA:  {init_ra:.6f}° → {new_ra:.6f}°  (ΔRA={abs(new_ra - init_ra) * 3600:.2f}\")")
    print(f"  中心Dec: {init_dec:.6f}° → {new_dec:.6f}°  (ΔDec={abs(new_dec - init_dec) * 3600:.2f}\")")
    print(f"  CD矩阵修正前:")
    print(f"    [{cd1_1:.6e}, {cd1_2:.6e}]")
    print(f"    [{cd2_1:.6e}, {cd2_2:.6e}]")
    print(f"  CD矩阵修正后:")
    print(f"    [{CD_new[0,0]:.6e}, {CD_new[0,1]:.6e}]")
    print(f"    [{CD_new[1,0]:.6e}, {CD_new[1,1]:.6e}]")

    det_old = cd1_1 * cd2_2 - cd1_2 * cd2_1
    det_new = CD_new[0, 0] * CD_new[1, 1] - CD_new[0, 1] * CD_new[1, 0]
    scale_old = np.sqrt(abs(det_old)) * 3600
    scale_new = np.sqrt(abs(det_new)) * 3600
    print(f"  像元尺度: {scale_old:.3f}\"/px → {scale_new:.3f}\"/px")

    rot_old = np.degrees(np.arctan2(cd2_1, cd1_1))
    rot_new = np.degrees(np.arctan2(CD_new[1, 0], CD_new[0, 0]))
    print(f"  旋转角: {rot_old:.3f}° → {rot_new:.3f}°")

    print("\n=== 步骤9: 生成标注图像 ===")
    output_path = os.path.join(os.path.dirname(test_image_path), 'star_alignment_annotated.png')
    draw_annotated_image(image_array, w, h, img_x, img_y, cat_x, cat_y, crpix_x, crpix_y,
                         sa_result, output_path)

    print("\n=== 性能统计 ===")
    total_time = det_time + gaia_time + proj_time + match_time
    print(f"  星点检测: {det_time:.2f}s")
    print(f"  Gaia查询: {gaia_time:.2f}s")
    print(f"  坐标投影: {proj_time:.2f}s")
    print(f"  DLL匹配:  {match_time:.2f}s")
    print(f"  总耗时: {total_time:.2f}s")

    print("\n=== 结果评估 ===")
    mc = sa_result.matched_count
    rms = sa_result.rms_px
    if mc >= 50:
        print(f"  ✓ 匹配对数达标 (>=50): {mc}")
    else:
        print(f"  ✗ 匹配对数不足: {mc} < 50")
    if rms < 10.0:
        print(f"  ✓ RMS精度达标 (<10px): {rms:.3f}px")
    else:
        print(f"  ✗ RMS精度不足: {rms:.3f}px >= 10px")

    sa_dll.psm_free_star_alignment_result(ctypes.byref(sa_result))

    print("\n" + "=" * 70)
    print("Star Alignment 测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()

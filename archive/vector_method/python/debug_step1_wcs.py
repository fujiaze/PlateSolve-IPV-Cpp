#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step1 WCS调试脚本
功能: 读取图像+检测星点+查询Gaia+投影 → 调用C++匹配(Siril算法) → 输出调试
Python只做调度，所有匹配算法在C++中严格遵循Siril
坐标系: 图像星=像素(居中+Y翻转), 星表星=角秒(gnomonic投影)
"""

import os, sys, ctypes
import numpy as np

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, project_root)

mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(mingw_bin):
    os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
    try: os.add_dll_directory(mingw_bin)
    except OSError: pass

from lib.astro_image_io.python.astro_image_io import ImageReader
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
from star_detector import StarDetector, SDetParamsPy

def gnomonic_projection_arcsec(ra_deg, dec_deg, center_ra_deg, center_dec_deg):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    center_ra_rad = np.radians(center_ra_deg)
    center_dec_rad = np.radians(center_dec_deg)
    cos_dec = np.cos(dec_rad); sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(center_dec_rad); sin_dec0 = np.sin(center_dec_rad)
    ra_diff = ra_rad - center_ra_rad
    cos_ra_diff = np.cos(ra_diff); sin_ra_diff = np.sin(ra_diff)
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff
    valid = cos_c > 1e-10
    xi = np.zeros_like(ra_deg); eta = np.zeros_like(dec_deg)
    xi[valid] = cos_dec[valid] * sin_ra_diff[valid] / cos_c[valid]
    eta[valid] = (cos_dec0 * sin_dec[valid] - sin_dec0 * cos_dec[valid] * cos_ra_diff[valid]) / cos_c[valid]
    rad_to_asec = 180.0 / np.pi * 3600.0
    return xi * rad_to_asec, eta * rad_to_asec, valid

def load_dll(path):
    try: os.add_dll_directory(os.path.dirname(os.path.abspath(path)))
    except OSError: pass
    return ctypes.cdll.LoadLibrary(path)

def main():
    print("=" * 70)
    print("Step1 WCS调试脚本 (Siril算法)")
    print("坐标系: 图像星=像素(居中+Y翻转), 星表星=角秒")
    print("=" * 70)

    test_image_path = os.path.join(project_root, "testdata", "lights", "panel1",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts")
    dr3_dir = os.path.join(project_root, "GaiaDR3")

    print("\n=== 读取FITS图像 ===")
    reader = ImageReader()
    with reader.read(test_image_path) as img_hdr:
        w, h = img_hdr.width, img_hdr.height
        image_array = img_hdr.data.copy()
        if img_hdr.has_wcs and img_hdr.wcs:
            wcs = img_hdr.wcs
            init_ra, init_dec, scale = wcs.crval1, wcs.crval2, wcs.pixel_scale
            cd1_1, cd1_2, cd2_1, cd2_2 = wcs.cd1_1, wcs.cd1_2, wcs.cd2_1, wcs.cd2_2
            crpix_x, crpix_y = wcs.crpix1 - 1, wcs.crpix2 - 1
        else:
            from astropy.io import fits as afits
            hdr = afits.getheader(test_image_path)
            objctra = hdr.get('OBJCTRA', hdr.get('RA', None))
            objctdec = hdr.get('OBJCTDEC', hdr.get('DEC', None))
            if objctra and objctdec:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                sc = SkyCoord(objctra, objctdec, unit=(u.hourangle, u.deg))
                init_ra = sc.ra.deg
                init_dec = sc.dec.deg
                print(f"  从OBJCTRA/OBJCTDEC读取: RA={init_ra:.4f}, Dec={init_dec:.4f}")
            else:
                init_ra, init_dec = 266.4167, -29.0078
            scale = 6.188
            cd1_1, cd1_2, cd2_1, cd2_2 = -scale, 0, 0, scale
            crpix_x, crpix_y = w / 2.0, h / 2.0
    print(f"  {w}x{h}, RA={init_ra:.4f}, Dec={init_dec:.4f}, scale={scale:.3f}\"/px")
    print(f"  CD matrix: [[{cd1_1:.6f}, {cd1_2:.6f}], [{cd2_1:.6f}, {cd2_2:.6f}]]")
    rot_deg = np.degrees(np.arctan2(cd2_1, cd1_1))
    print(f"  旋转角: {rot_deg:.2f}°")

    print("\n=== 星点检测 ===")
    sd_params = SDetParamsPy(iterativeClipSigma=9.0, fwhmClipSigma=3.0, maxAxisRatio=2.0, fitRadius=8)
    detector = StarDetector(params=sd_params)
    coords, fluxes, saturated = detector.detect_ex(image_array)
    detector.close()
    n_saturated = int(np.sum(saturated))
    n_img = len(coords)
    print(f"  检测星数: {n_img}, 饱和星数: {n_saturated}")

    x0 = w * 0.5
    y0 = h * 0.5
    img_x = np.array([x - x0 for x, y in coords], dtype=np.float64)
    img_y = np.array([y0 - y for x, y in coords], dtype=np.float64)
    img_flux = np.array(fluxes, dtype=np.float64)
    print(f"  图像星坐标范围: x=[{img_x.min():.0f}, {img_x.max():.0f}], y=[{img_y.min():.0f}, {img_y.max():.0f}]")

    print("\n=== Gaia查询 ===")
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
    fov_diag = np.sqrt(w*w + h*h) * scale / 3600
    query_radius = fov_diag * 1.2 / 2
    target_count = int(n_img * 1.5)
    msvcrt = ctypes.CDLL("msvcrt.dll")

    mag_low, mag_high = 6.0, 22.0
    for _ in range(10):
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
    n_cat = len(cat_ra)
    print(f"  Gaia星数: {n_cat}, 极限星等: {final_mag:.2f}")

    print("\n=== 投影Gaia星到角秒坐标 (Siril方式) ===")
    cat_x_asec, cat_y_asec, valid = gnomonic_projection_arcsec(cat_ra, cat_dec, init_ra, init_dec)
    print(f"  投影后星数: {np.sum(valid)}")
    
    fov_margin = 1.2
    img_x_range = (w * 0.5) * scale * fov_margin
    img_y_range = (h * 0.5) * scale * fov_margin
    fov_mask = (np.abs(cat_x_asec) < img_x_range) & (np.abs(cat_y_asec) < img_y_range) & valid
    cat_x_asec_f = cat_x_asec[fov_mask]
    cat_y_asec_f = cat_y_asec[fov_mask]
    cat_ra_f = cat_ra[fov_mask]
    cat_dec_f = cat_dec[fov_mask]
    cat_mag_f = cat_mag[fov_mask]
    print(f"  FOV过滤: {np.sum(fov_mask)}/{len(cat_ra)} (范围 ±{img_x_range:.0f}\"×±{img_y_range:.0f}\")")
    print(f"  星表投影坐标范围: x=[{cat_x_asec_f.min():.1f}, {cat_x_asec_f.max():.1f}]\", y=[{cat_y_asec_f.min():.1f}, {cat_y_asec_f.max():.1f}]\"")
    print(f"  图像星坐标范围: x=[{img_x.min():.0f}, {img_x.max():.0f}]px, y=[{img_y.min():.0f}, {img_y.max():.0f}]px")
    print(f"  预期比例尺: 1/{scale:.3f} = {1/scale:.6f} px/arcsec")

    print("\n=== 关键诊断: top 60星对应检查 (CD矩阵+偏移校正) ===")
    from scipy.spatial import cKDTree
    img_x_asec = cd1_1 * img_x + cd1_2 * img_y
    img_y_asec = cd2_1 * img_x + cd2_2 * img_y
    tree_cat = cKDTree(np.column_stack([cat_x_asec_f, cat_y_asec_f]))
    dists_nn0, idxs_nn0 = tree_cat.query(np.column_stack([img_x_asec, img_y_asec]), k=1)
    bright_order = np.argsort(-img_flux)[:60]
    top60_dists = dists_nn0[bright_order]
    median_offset = np.median(top60_dists[top60_dists < 500])
    dx = np.median((img_x_asec - cat_x_asec_f[idxs_nn0])[bright_order[top60_dists < 500]])
    dy = np.median((img_y_asec - cat_y_asec_f[idxs_nn0])[bright_order[top60_dists < 500]])
    print(f"  初始偏移: dx={dx:.1f}\" dy={dy:.1f}\" median_dist={median_offset:.1f}\"")
    img_x_asec_corr = img_x_asec - dx
    img_y_asec_corr = img_y_asec - dy
    dists_nn, idxs_nn = tree_cat.query(np.column_stack([img_x_asec_corr, img_y_asec_corr]), k=1)
    
    bright_order = np.argsort(-img_flux)[:60]
    good_count = 0
    for rank, i in enumerate(bright_order):
        j = idxs_nn[i]
        d = dists_nn[i]
        if d < 200:
            good_count += 1
        if rank < 15:
            print(f"  img[{rank}]({img_x[i]:8.1f},{img_y[i]:8.1f})px=({img_x_asec[i]:8.1f},{img_y_asec[i]:8.1f})\" -> cat({cat_x_asec_f[j]:8.1f},{cat_y_asec_f[j]:8.1f})\" d={d:.1f}\" flux={img_flux[i]:.0f} cat_mag={cat_mag_f[j]:.2f}")
    print(f"  top 60中在200\"内找到对应: {good_count}/60")
    
    dists_top60 = dists_nn[bright_order]
    for th in [50, 100, 200, 500]:
        n = np.sum(dists_top60 < th)
        print(f"  top 60中在{th}\"内: {n}/60")

    print("\n=== 三角形ba/ca偏差诊断 (仅用确认对应星) ===")
    bright_order_all = np.argsort(-img_flux)
    confirmed_img = []
    confirmed_cat = []
    for i in bright_order_all[:200]:
        j = idxs_nn[i]
        d = dists_nn[i]
        if d < 30:
            confirmed_img.append(i)
            confirmed_cat.append(j)
    print(f"  确认对应星 (d<30\"): {len(confirmed_img)}")
    if len(confirmed_img) >= 3:
        n_check = min(20, len(confirmed_img))
        img_check = [(img_x[confirmed_img[i]], img_y[confirmed_img[i]]) for i in range(n_check)]
        cat_check = [(cat_x_asec_f[confirmed_cat[i]], cat_y_asec_f[confirmed_cat[i]]) for i in range(n_check)]
        ba_diffs = []
        for i in range(n_check):
            for j in range(i+1, n_check):
                for k in range(j+1, n_check):
                    ix = [img_check[i], img_check[j], img_check[k]]
                    cx = [cat_check[i], cat_check[j], cat_check[k]]
                    img_dists = [np.sqrt((ix[a][0]-ix[b][0])**2 + (ix[a][1]-ix[b][1])**2) for a,b in [(0,1),(1,2),(0,2)]]
                    cat_dists = [np.sqrt((cx[a][0]-cx[b][0])**2 + (cx[a][1]-cx[b][1])**2) for a,b in [(0,1),(1,2),(0,2)]]
                    img_sorted = sorted(img_dists, reverse=True)
                    cat_sorted = sorted(cat_dists, reverse=True)
                    if img_sorted[0] > 0 and cat_sorted[0] > 0:
                        img_ba = img_sorted[1] / img_sorted[0]
                        cat_ba = cat_sorted[1] / cat_sorted[0]
                        ba_diffs.append(abs(img_ba - cat_ba))
        ba_diffs = np.array(ba_diffs)
        print(f"  确认对应三角形的ba偏差 ({len(ba_diffs)}个):")
        print(f"    中位数: {np.median(ba_diffs):.6f}")
        print(f"    < 0.002: {np.sum(ba_diffs < 0.002)}/{len(ba_diffs)} ({100*np.mean(ba_diffs < 0.002):.1f}%)")
        print(f"    < 0.005: {np.sum(ba_diffs < 0.005)}/{len(ba_diffs)} ({100*np.mean(ba_diffs < 0.005):.1f}%)")
    else:
        print("  确认对应星不足3颗，无法计算ba偏差")

    print("\n=== 三角形ba/ca偏差诊断 (top 30 by flux) ===")
    n_check = min(30, len(bright_order))
    check_indices = bright_order[:n_check]
    img_check = [(img_x[i], img_y[i]) for i in check_indices]
    cat_check = [(cat_x_asec_f[idxs_nn[i]], cat_y_asec_f[idxs_nn[i]]) for i in check_indices]
    
    ba_diffs = []
    for i in range(n_check):
        for j in range(i+1, n_check):
            for k in range(j+1, n_check):
                ix = [img_check[i], img_check[j], img_check[k]]
                cx = [cat_check[i], cat_check[j], cat_check[k]]
                img_dists = [np.sqrt((ix[a][0]-ix[b][0])**2 + (ix[a][1]-ix[b][1])**2) for a,b in [(0,1),(1,2),(0,2)]]
                cat_dists = [np.sqrt((cx[a][0]-cx[b][0])**2 + (cx[a][1]-cx[b][1])**2) for a,b in [(0,1),(1,2),(0,2)]]
                img_sorted = sorted(img_dists, reverse=True)
                cat_sorted = sorted(cat_dists, reverse=True)
                if img_sorted[0] > 0 and cat_sorted[0] > 0:
                    img_ba = img_sorted[1] / img_sorted[0]
                    img_ca = img_sorted[2] / img_sorted[0]
                    cat_ba = cat_sorted[1] / cat_sorted[0]
                    cat_ca = cat_sorted[2] / cat_sorted[0]
                    ba_diff = abs(img_ba - cat_ba)
                    ca_diff = abs(img_ca - cat_ca)
                    ba_diffs.append(ba_diff)
    
    ba_diffs = np.array(ba_diffs)
    print(f"  检查了{len(ba_diffs)}个对应三角形的ba偏差:")
    print(f"    中位数: {np.median(ba_diffs):.6f}")
    print(f"    均值: {np.mean(ba_diffs):.6f}")
    print(f"    最大: {np.max(ba_diffs):.6f}")
    print(f"    < 0.002: {np.sum(ba_diffs < 0.002)}/{len(ba_diffs)} ({100*np.mean(ba_diffs < 0.002):.1f}%)")
    print(f"    < 0.005: {np.sum(ba_diffs < 0.005)}/{len(ba_diffs)} ({100*np.mean(ba_diffs < 0.005):.1f}%)")
    print(f"    < 0.01:  {np.sum(ba_diffs < 0.01)}/{len(ba_diffs)} ({100*np.mean(ba_diffs < 0.01):.1f}%)")

    print("\n=== 调用Step1匹配 (C++: 饱和星优先) ===")
    sa_dll = load_dll(os.path.join(project_root, "lib", "plate_solve", "modules", "star_alignment", "star_alignment.dll"))

    class PSMResult(ctypes.Structure):
        _fields_ = [
            ("a0", ctypes.c_double), ("a1", ctypes.c_double), ("a2", ctypes.c_double),
            ("b0", ctypes.c_double), ("b1", ctypes.c_double), ("b2", ctypes.c_double),
            ("matched_count", ctypes.c_int), ("rms_arcsec", ctypes.c_double),
            ("center_ra", ctypes.c_double), ("center_dec", ctypes.c_double),
            ("img_indices", ctypes.POINTER(ctypes.c_int)),
            ("cat_indices", ctypes.POINTER(ctypes.c_int)),
        ]

    sa_dll.psm_star_alignment.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(PSMResult),
    ]
    sa_dll.psm_star_alignment.restype = ctypes.c_int
    sa_dll.psm_free_result.argtypes = [ctypes.POINTER(PSMResult)]

    img_sat = np.array(saturated, dtype=np.int32)

    sa_result = PSMResult()
    percent_scale_range = 10.0

    ret = sa_dll.psm_star_alignment(
        img_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        img_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        img_flux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        img_sat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), n_img,
        cat_x_asec_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cat_y_asec_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cat_mag_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(cat_x_asec_f),
        scale,
        percent_scale_range,
        init_ra,
        init_dec,
        cat_ra_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cat_dec_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cd1_1, cd1_2, cd2_1, cd2_2,
        ctypes.byref(sa_result),
    )

    if ret != 0:
        print(f"  匹配失败, 返回码: {ret}")
        return

    print(f"\n=== 匹配结果 ===")
    print(f"  匹配星对: {sa_result.matched_count}")
    print(f"  RMS: {sa_result.rms_arcsec:.4f} arcsec ({sa_result.rms_arcsec / scale:.3f} px)")
    derived_scale = 1.0 / np.sqrt(sa_result.a1**2 + sa_result.b1**2)
    print(f"  导出比例尺: {derived_scale:.4f} arcsec/px (初始: {scale:.4f})")
    print(f"  投影中心: RA={sa_result.center_ra:.6f} Dec={sa_result.center_dec:.6f} (初始: {init_ra:.6f}, {init_dec:.6f})")
    print(f"  仿射变换: a0={sa_result.a0:.4f} a1={sa_result.a1:.8f} a2={sa_result.a2:.8f}")
    print(f"            b0={sa_result.b0:.4f} b1={sa_result.b1:.8f} b2={sa_result.b2:.8f}")

    img_indices = np.array([sa_result.img_indices[i] for i in range(sa_result.matched_count)], dtype=np.int32)
    cat_indices = np.array([sa_result.cat_indices[i] for i in range(sa_result.matched_count)], dtype=np.int32)

    print(f"\n  前10个匹配对 (像素→角秒):")
    for i in range(min(10, sa_result.matched_count)):
        ii = img_indices[i]
        ci = cat_indices[i]
        sx, sy = img_x[ii], img_y[ii]
        tx = sa_result.a0 + sa_result.a1 * sx + sa_result.a2 * sy
        ty = sa_result.b0 + sa_result.b1 * sx + sa_result.b2 * sy
        d = np.sqrt((tx - cat_x_asec_f[ci])**2 + (ty - cat_y_asec_f[ci])**2)
        print(f"    [{i}] img({sx:8.1f},{sy:8.1f})px -> pred({tx:8.1f},{ty:8.1f})\" vs cat({cat_x_asec_f[ci]:8.1f},{cat_y_asec_f[ci]:8.1f})\" dist={d:.3f}\"")

    sa_dll.psm_free_result(ctypes.byref(sa_result))
    print("\n=== 完成 ===")

if __name__ == "__main__":
    main()

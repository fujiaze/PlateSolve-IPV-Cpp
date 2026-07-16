# -*- coding: utf-8 -*-
"""
单帧测试 - 网格控制点精细拟合测试

测试流程:
1. 粗匹配获取初始变换参数
2. 调用网格控制点匹配模块
3. 打印控制点数量、拟合参数、RMS
"""
import os
import sys
import ctypes
import numpy as np

_mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(_mingw_bin):
    os.environ["PATH"] = _mingw_bin + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_mingw_bin)
    except OSError:
        pass

_project_root = r"F:\Astro dev\Astro CS Normalization Database"
_dll_path = os.path.join(_project_root, "lib", "plate_solve", "plate_solve.dll")

if os.path.exists(_dll_path):
    try:
        os.add_dll_directory(os.path.dirname(_dll_path))
    except OSError:
        pass
    _dll = ctypes.CDLL(_dll_path)
else:
    raise RuntimeError(f"plate_solve.dll not found: {_dll_path}")

class _GMGridCell(ctypes.Structure):
    _fields_ = [
        ("row", ctypes.c_int),
        ("col", ctypes.c_int),
        ("x_start", ctypes.c_int),
        ("y_start", ctypes.c_int),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("center_x", ctypes.c_double),
        ("center_y", ctypes.c_double),
    ]

class _GMControlPoint(ctypes.Structure):
    _fields_ = [
        ("grid_row", ctypes.c_int),
        ("grid_col", ctypes.c_int),
        ("img_star_idx", ctypes.c_int),
        ("cat_star_idx", ctypes.c_int),
        ("img_x", ctypes.c_double),
        ("img_y", ctypes.c_double),
        ("cat_x", ctypes.c_double),
        ("cat_y", ctypes.c_double),
        ("residual_x", ctypes.c_double),
        ("residual_y", ctypes.c_double),
        ("valid", ctypes.c_int),
    ]

class _GMConfig(ctypes.Structure):
    _fields_ = [
        ("grid_size", ctypes.c_int),
        ("max_cat_candidates", ctypes.c_int),
        ("match_tolerance", ctypes.c_double),
        ("max_ransac_iter", ctypes.c_int),
        ("ransac_sigma", ctypes.c_double),
        ("sip_order", ctypes.c_int),
    ]

class _GMResult(ctypes.Structure):
    _fields_ = [
        ("n_control_points", ctypes.c_int),
        ("n_grids_matched", ctypes.c_int),
        ("n_grids_total", ctypes.c_int),
        ("rms_x", ctypes.c_double),
        ("rms_y", ctypes.c_double),
        ("rms_total", ctypes.c_double),
        ("rms_arcsec", ctypes.c_double),
        ("n_ransac_removed", ctypes.c_int),
        ("control_points", ctypes.POINTER(_GMControlPoint)),
        ("sip_A", (ctypes.c_double * 6) * 6),
        ("sip_B", (ctypes.c_double * 6) * 6),
        ("sip_AP", (ctypes.c_double * 6) * 6),
        ("sip_BP", (ctypes.c_double * 6) * 6),
        ("sip_order", ctypes.c_int),
        ("sip_valid", ctypes.c_int),
        ("cd", (ctypes.c_double * 2) * 2),
        ("crpix", ctypes.c_double * 2),
        ("crval", ctypes.c_double * 2),
    ]

class _GMImageStars(ctypes.Structure):
    _fields_ = [
        ("img_x", ctypes.POINTER(ctypes.c_double)),
        ("img_y", ctypes.POINTER(ctypes.c_double)),
        ("img_flux", ctypes.POINTER(ctypes.c_double)),
        ("img_count", ctypes.c_int),
    ]

class _GMCatalogStars(ctypes.Structure):
    _fields_ = [
        ("cat_ra", ctypes.POINTER(ctypes.c_double)),
        ("cat_dec", ctypes.POINTER(ctypes.c_double)),
        ("cat_mag", ctypes.POINTER(ctypes.c_double)),
        ("cat_count", ctypes.c_int),
    ]

class _GMInitialTransform(ctypes.Structure):
    _fields_ = [
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("rotation_deg", ctypes.c_double),
        ("scale_arcsec_px", ctypes.c_double),
        ("flip_mode", ctypes.c_int),
        ("img_width", ctypes.c_int),
        ("img_height", ctypes.c_int),
    ]

_dll.psm_grid_match_perform.argtypes = [
    ctypes.POINTER(_GMImageStars),
    ctypes.POINTER(_GMCatalogStars),
    ctypes.POINTER(_GMInitialTransform),
    ctypes.POINTER(_GMConfig),
    ctypes.POINTER(_GMResult),
]
_dll.psm_grid_match_perform.restype = ctypes.c_int

_dll.psm_grid_match_free_result.argtypes = [ctypes.POINTER(_GMResult)]
_dll.psm_grid_match_free_result.restype = None

_dll.psolve_create.argtypes = [ctypes.c_char_p]
_dll.psolve_create.restype = ctypes.c_void_p
_dll.psolve_destroy.argtypes = [ctypes.c_void_p]
_dll.psolve_destroy.restype = None

_dll.psolve_solve_with_file.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_dll.psolve_solve_with_file.restype = ctypes.c_int

_dll.psolve_get_last_coarse_result.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
]
_dll.psolve_get_last_coarse_result.restype = ctypes.c_int

_dll.psolve_get_last_stars.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
    ctypes.POINTER(ctypes.c_int),
]
_dll.psolve_get_last_stars.restype = ctypes.c_int

_dll.psolve_free_last_stars.argtypes = [
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
]
_dll.psolve_free_last_stars.restype = None

def test_grid_match(file_path: str, gaia_dir: str):
    print("=" * 80)
    print(f"测试文件: {os.path.basename(file_path)}")
    print("=" * 80)
    
    handle = _dll.psolve_create(gaia_dir.encode('utf-8'))
    if not handle:
        print("ERROR: Failed to create solver")
        return
    
    try:
        rc = _dll.psolve_solve_with_file(handle, file_path.encode('utf-8'), None, None)
        if rc != 0:
            print(f"ERROR: psolve_solve_with_file failed with code {rc}")
            return
        
        center_ra = ctypes.c_double()
        center_dec = ctypes.c_double()
        rotation_deg = ctypes.c_double()
        scale_arcsec_px = ctypes.c_double()
        flip_mode = ctypes.c_int()
        matched_count = ctypes.c_int()
        img_width = ctypes.c_int()
        img_height = ctypes.c_int()
        
        rc = _dll.psolve_get_last_coarse_result(
            handle,
            ctypes.byref(center_ra),
            ctypes.byref(center_dec),
            ctypes.byref(rotation_deg),
            ctypes.byref(scale_arcsec_px),
            ctypes.byref(flip_mode),
            ctypes.byref(matched_count),
            ctypes.byref(img_width),
            ctypes.byref(img_height),
        )
        
        if rc != 0:
            print(f"ERROR: psolve_get_last_coarse_result failed with code {rc}")
            return
        
        print(f"\n粗匹配结果:")
        print(f"  中心: RA={center_ra.value:.6f}, Dec={center_dec.value:.6f}")
        print(f"  旋转: {rotation_deg.value:.3f}°")
        print(f"  比例尺: {scale_arcsec_px.value:.3f} arcsec/px")
        print(f"  翻转模式: {flip_mode.value}")
        print(f"  匹配数: {matched_count.value}")
        print(f"  图像尺寸: {img_width.value} x {img_height.value}")
        
        img_x_ptr = ctypes.POINTER(ctypes.c_double)()
        img_y_ptr = ctypes.POINTER(ctypes.c_double)()
        img_flux_ptr = ctypes.POINTER(ctypes.c_double)()
        img_count = ctypes.c_int()
        cat_ra_ptr = ctypes.POINTER(ctypes.c_double)()
        cat_dec_ptr = ctypes.POINTER(ctypes.c_double)()
        cat_mag_ptr = ctypes.POINTER(ctypes.c_double)()
        cat_count = ctypes.c_int()
        
        rc = _dll.psolve_get_last_stars(
            handle,
            ctypes.byref(img_x_ptr),
            ctypes.byref(img_y_ptr),
            ctypes.byref(img_flux_ptr),
            ctypes.byref(img_count),
            ctypes.byref(cat_ra_ptr),
            ctypes.byref(cat_dec_ptr),
            ctypes.byref(cat_mag_ptr),
            ctypes.byref(cat_count),
        )
        
        if rc != 0:
            print(f"ERROR: psolve_get_last_stars failed with code {rc}")
            return
        
        print(f"\n星点数据:")
        print(f"  图像星点: {img_count.value}")
        print(f"  Gaia星点: {cat_count.value}")
        
        gm_config = _GMConfig()
        gm_config.grid_size = 50
        gm_config.max_cat_candidates = 10
        gm_config.match_tolerance = 5.0
        gm_config.max_ransac_iter = 5
        gm_config.ransac_sigma = 3.0
        gm_config.sip_order = 5
        
        gm_img = _GMImageStars()
        gm_img.img_x = img_x_ptr
        gm_img.img_y = img_y_ptr
        gm_img.img_flux = img_flux_ptr
        gm_img.img_count = img_count.value
        
        gm_cat = _GMCatalogStars()
        gm_cat.cat_ra = cat_ra_ptr
        gm_cat.cat_dec = cat_dec_ptr
        gm_cat.cat_mag = cat_mag_ptr
        gm_cat.cat_count = cat_count.value
        
        gm_init = _GMInitialTransform()
        gm_init.center_ra = center_ra.value
        gm_init.center_dec = center_dec.value
        gm_init.rotation_deg = rotation_deg.value
        gm_init.scale_arcsec_px = scale_arcsec_px.value
        gm_init.flip_mode = flip_mode.value
        gm_init.img_width = img_width.value
        gm_init.img_height = img_height.value
        
        gm_result = _GMResult()
        
        print(f"\n开始网格控制点匹配...")
        rc = _dll.psm_grid_match_perform(
            ctypes.byref(gm_img),
            ctypes.byref(gm_cat),
            ctypes.byref(gm_init),
            ctypes.byref(gm_config),
            ctypes.byref(gm_result),
        )
        
        if rc != 0:
            print(f"ERROR: psm_grid_match_perform failed with code {rc}")
        else:
            print(f"\n网格控制点结果:")
            print(f"  总网格数: {gm_result.n_grids_total}")
            print(f"  有匹配的网格数: {gm_result.n_grids_matched}")
            print(f"  有效控制点数: {gm_result.n_control_points}")
            print(f"  RANSAC剔除数: {gm_result.n_ransac_removed}")
            print(f"  RMS X: {gm_result.rms_x:.3f} px")
            print(f"  RMS Y: {gm_result.rms_y:.3f} px")
            print(f"  RMS Total: {gm_result.rms_total:.3f} px")
            print(f"  RMS (arcsec): {gm_result.rms_arcsec:.3f} arcsec")
            
            if gm_result.sip_valid:
                print(f"\nSIP拟合参数 (order={gm_result.sip_order}):")
                print(f"  CD矩阵:")
                print(f"    CD1_1 = {gm_result.cd[0][0]:.6e}")
                print(f"    CD1_2 = {gm_result.cd[0][1]:.6e}")
                print(f"    CD2_1 = {gm_result.cd[1][0]:.6e}")
                print(f"    CD2_2 = {gm_result.cd[1][1]:.6e}")
                
                print(f"\n  SIP正向系数 (A):")
                for i in range(3):
                    row_str = "    "
                    for j in range(3):
                        if i + j <= 5:
                            val = gm_result.sip_A[i][j]
                            if abs(val) > 1e-10:
                                row_str += f"A[{i},{j}]={val:.4e}  "
                    if row_str.strip():
                        print(row_str)
                
                print(f"\n  CRPIX: ({gm_result.crpix[0]:.1f}, {gm_result.crpix[1]:.1f})")
                print(f"  CRVAL: ({gm_result.crval[0]:.6f}, {gm_result.crval[1]:.6f})")
            else:
                print(f"\nSIP拟合无效")
            
            _dll.psm_grid_match_free_result(ctypes.byref(gm_result))
        
        _dll.psolve_free_last_stars(
            img_x_ptr, img_y_ptr, img_flux_ptr,
            cat_ra_ptr, cat_dec_ptr, cat_mag_ptr,
        )
        
    finally:
        _dll.psolve_destroy(handle)

if __name__ == "__main__":
    gaia_dir = r"F:\Astro dev\Astro CS Normalization Database\GaiaDR3SP"
    test_file = r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@051551-180S-Red.fts"
    
    test_grid_match(test_file, gaia_dir)
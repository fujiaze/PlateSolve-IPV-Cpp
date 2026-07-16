# -*- coding: utf-8 -*-
"""
PlateSolve Debug Runner - 运行Plate Solve并生成可视化验证

流程:
1. 运行plate_solve获取WCS参数（包含SIP系数）
2. 根据WCS参数调用gaia_client查询星点
3. 使用WCS参数将Gaia星点投影到像素坐标
4. 生成可视化图片对比预测位置与图像星点
"""
import os
import sys
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import Tuple, List, Optional

_mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(_mingw_bin):
    os.environ["PATH"] = _mingw_bin + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_mingw_bin)
    except OSError:
        pass

_project_root = r"F:\Astro dev\Astro CS Normalization Database"
_gaia_dll_path = os.path.join(_project_root, "lib", "gaia_xpsd_client", "gaia_xpsd_client.dll")
_psolve_dll_path = os.path.join(_project_root, "lib", "plate_solve", "plate_solve.dll")

def load_gaia_dll():
    if not os.path.exists(_gaia_dll_path):
        raise RuntimeError(f"gaia_xpsd_client.dll not found: {_gaia_dll_path}")
    try:
        os.add_dll_directory(os.path.dirname(_gaia_dll_path))
    except OSError:
        pass
    dll = ctypes.CDLL(_gaia_dll_path)
    
    dll.gaia_client_create.argtypes = [ctypes.c_char_p]
    dll.gaia_client_create.restype = ctypes.c_void_p
    dll.gaia_client_destroy.argtypes = [ctypes.c_void_p]
    dll.gaia_client_destroy.restype = None
    
    dll.gaia_client_cone_search_for_solver.argtypes = [
        ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int),
    ]
    dll.gaia_client_cone_search_for_solver.restype = ctypes.c_int
    
    return dll

def wcs_pixel_to_world(x, y, wcs_params, sip_coeffs=None):
    """
    像素坐标到天球坐标（逆变换）
    
    参数:
        x, y: 像素坐标（相对于CRPIX）
        wcs_params: dict包含crval1, crval2, cd1_1, cd1_2, cd2_1, cd2_2
        sip_coeffs: SIP AP/BP系数（逆变换系数）
    
    返回:
        (ra, dec) 天球坐标（度）
    """
    cd11 = wcs_params['cd1_1']
    cd12 = wcs_params['cd1_2']
    cd21 = wcs_params['cd2_1']
    cd22 = wcs_params['cd2_2']
    crval1 = wcs_params['crval1']
    crval2 = wcs_params['crval2']
    
    det = cd11 * cd22 - cd12 * cd21
    if abs(det) < 1e-15:
        return crval1, crval2
    
    cd_inv11 = cd22 / det
    cd_inv12 = -cd12 / det
    cd_inv21 = -cd21 / det
    cd_inv22 = cd11 / det
    
    u = x
    v = y
    
    if sip_coeffs is not None and sip_coeffs['valid']:
        AP = sip_coeffs['AP']
        BP = sip_coeffs['BP']
        order = sip_coeffs['order']
        
        u_corr = u
        v_corr = v
        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i + j >= 1:
                    u_corr += AP[i][j] * (u ** i) * (v ** j)
                    v_corr += BP[i][j] * (u ** i) * (v ** j)
        u = u_corr
        v = v_corr
    
    xi = cd_inv11 * u + cd_inv12 * v
    eta = cd_inv21 * u + cd_inv22 * v
    
    ra0 = crval1 * np.pi / 180.0
    dec0 = crval2 * np.pi / 180.0
    
    r = np.sqrt(xi * xi + eta * eta)
    if r < 1e-10:
        return crval1, crval2
    
    c = np.arctan(r)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)
    
    dec_rad = np.arcsin(cos_c * sin_dec0 + sin_c * cos_dec0 * eta / r)
    ra_rad = ra0 + np.arctan2(xi * sin_c / r, cos_c * cos_dec0 - sin_c * sin_dec0 * eta / r)
    
    ra = ra_rad * 180.0 / np.pi
    dec = dec_rad * 180.0 / np.pi
    
    ra = ra % 360.0
    if ra < 0:
        ra += 360.0
    
    return ra, dec

def wcs_world_to_pixel(ra, dec, wcs_params, sip_coeffs=None):
    """
    天球坐标到像素坐标（正向变换）
    
    参数:
        ra, dec: 天球坐标（度）
        wcs_params: dict包含crval1, crval2, cd1_1, cd1_2, cd2_1, cd2_2, crpix1, crpix2
        sip_coeffs: SIP A/B系数（正向变换系数）
    
    返回:
        (x, y) 像素坐标（绝对坐标）
    """
    cd11 = wcs_params['cd1_1']
    cd12 = wcs_params['cd1_2']
    cd21 = wcs_params['cd2_1']
    cd22 = wcs_params['cd2_2']
    crval1 = wcs_params['crval1']
    crval2 = wcs_params['crval2']
    crpix1 = wcs_params['crpix1']
    crpix2 = wcs_params['crpix2']
    
    ra_rad = ra * np.pi / 180.0
    dec_rad = dec * np.pi / 180.0
    ra0 = crval1 * np.pi / 180.0
    dec0 = crval2 * np.pi / 180.0
    
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(dec0)
    sin_dec0 = np.sin(dec0)
    
    ra_diff = ra_rad - ra0
    cos_ra_diff = np.cos(ra_diff)
    sin_ra_diff = np.sin(ra_diff)
    
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff
    if cos_c < 1e-10:
        return crpix1, crpix2
    
    xi = cos_dec * sin_ra_diff / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c
    
    u = cd11 * xi + cd12 * eta
    v = cd21 * xi + cd22 * eta
    
    if sip_coeffs is not None and sip_coeffs['valid']:
        A = sip_coeffs['A']
        B = sip_coeffs['B']
        order = sip_coeffs['order']
        
        u_corr = u
        v_corr = v
        for i in range(order + 1):
            for j in range(order + 1 - i):
                if i + j >= 1:
                    u_corr += A[i][j] * (u ** i) * (v ** j)
                    v_corr += B[i][j] * (u ** i) * (v ** j)
        u = u_corr
        v = v_corr
    
    x = u + crpix1
    y = v + crpix2
    
    return x, y

def run_plate_solve_with_visualization(
    fits_path: str,
    gaia_data_dir: str,
    output_dir: str,
    focal_length_mm: float = 0.0,
    pixel_size_um: float = 0.0,
    scale_arcsec_px: float = 0.0,
):
    """
    运行plate_solve并生成可视化验证图片
    
    参数:
        fits_path: FITS图像路径
        gaia_data_dir: Gaia数据库目录
        output_dir: 输出目录
        focal_length_mm: 焦距（可选）
        pixel_size_um: 像元尺寸（可选）
        scale_arcsec_px: 比例尺（可选）
    """
    from plate_solve import PlateSolve, PlateSolveConfig
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Plate Solve Debug Runner")
    print(f"{'='*60}")
    print(f"  FITS: {fits_path}")
    print(f"  Gaia: {gaia_data_dir}")
    print(f"  Output: {output_dir}")
    
    with fits.open(fits_path) as hdul:
        img_data = hdul[0].data
        header = hdul[0].header
    
    height, width = img_data.shape
    print(f"  Image: {width}x{height}")
    
    center_ra = header.get('RA', 0.0)
    center_dec = header.get('DEC', 0.0)
    
    if center_ra == 0.0 or center_dec == 0.0:
        center_ra = header.get('OBJCTRA', 0.0)
        center_dec = header.get('OBJCTDEC', 0.0)
    
    if isinstance(center_ra, str):
        parts = center_ra.split()
        if len(parts) == 3:
            center_ra = float(parts[0]) + float(parts[1])/60.0 + float(parts[2])/3600.0
    if isinstance(center_dec, str):
        parts = center_dec.split()
        if len(parts) == 3:
            sign = 1 if parts[0].startswith('+') or float(parts[0]) >= 0 else -1
            center_dec = sign * (abs(float(parts[0])) + float(parts[1])/60.0 + float(parts[2])/3600.0)
    
    if focal_length_mm == 0.0:
        focal_length_mm = header.get('FOCALLEN', 200.0)
    if pixel_size_um == 0.0:
        pixel_size_um = header.get('PIXSIZE', 6.0)
    
    print(f"  Initial center: RA={center_ra:.6f}, Dec={center_dec:.6f}")
    print(f"  Focal length: {focal_length_mm}mm, Pixel size: {pixel_size_um}um")
    
    config = PlateSolveConfig(
        sip_order=5,
        n_img_bright=1000,
        n_cat_bright=1500,
    )
    
    solver = PlateSolve(gaia_data_dir)
    
    print(f"\n{'='*60}")
    print(f"Running Plate Solve...")
    print(f"{'='*60}")
    
    result = solver.solve_with_file(fits_path, config)
    
    print(f"\nResults:")
    print(f"  Center: RA={result.center_ra:.6f}, Dec={result.center_dec:.6f}")
    print(f"  Rotation: {result.rotation_deg:.3f} deg")
    print(f"  Scale: {result.scale_arcsec_px:.3f} arcsec/px")
    print(f"  Flip: {result.flip_mode}")
    print(f"  Matched: {result.matched_count}")
    print(f"  RMS: {result.rms_px:.3f} px")
    print(f"  SIP valid: {result.sip_valid}")
    print(f"  Time: Step1={result.step1_time_sec:.2f}s, Step2={result.step2_time_sec:.2f}s")
    
    wcs_params = result.wcs
    print(f"\nWCS:")
    print(f"  CRPIX: ({wcs_params['crpix1']:.1f}, {wcs_params['crpix2']:.1f})")
    print(f"  CRVAL: ({wcs_params['crval1']:.6f}, {wcs_params['crval2']:.6f})")
    print(f"  CD: [[{wcs_params['cd1_1']:.6e}, {wcs_params['cd1_2']:.6e}]")
    print(f"       [{wcs_params['cd2_1']:.6e}, {wcs_params['cd2_2']:.6e}]]")
    
    sip_coeffs = None
    if result.sip_valid and result.sip is not None:
        sip_coeffs = {
            'A': result.sip[0],
            'B': result.sip[1],
            'AP': result.sip[2],
            'BP': result.sip[3],
            'order': 5,
            'valid': True,
        }
        print(f"  SIP order: 5")
    
    solver.close()
    
    print(f"\n{'='*60}")
    print(f"Generating visualization...")
    print(f"{'='*60}")
    
    img_min, img_max = np.percentile(img_data, [0.5, 99.5])
    stretched = np.clip((img_data - img_min) / (img_max - img_min), 0, 1)
    rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float32)
    
    dpi = 100
    fig = plt.figure(figsize=(width/100, height/100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin='lower', extent=[0, width, 0, height])
    
    step2_pred_csv = os.path.join(_project_root, "debug_step2_predictions.csv")
    step1_csv = os.path.join(_project_root, "debug_step1_predictions.csv")
    step2_cp_csv = os.path.join(_project_root, "debug_control_points.csv")
    
    if os.path.exists(step2_pred_csv):
        step2_pred_data = np.loadtxt(step2_pred_csv, delimiter=',', skiprows=1)
        if len(step2_pred_data) > 0:
            pred_x = step2_pred_data[:, 0]
            pred_y = step2_pred_data[:, 1]
            ax.scatter(pred_x, pred_y, c='red', s=8, marker='+', linewidths=0.5, label='Step2 Gaia predicted')
            print(f"  Step2 predictions: {len(pred_x)}")
    
    if os.path.exists(step1_csv):
        step1_data = np.loadtxt(step1_csv, delimiter=',', skiprows=1)
        if len(step1_data) > 0:
            step1_pred_x = step1_data[:, 4]
            step1_pred_y = step1_data[:, 5]
            ax.scatter(step1_pred_x, step1_pred_y, c='cyan', s=6, marker='+', linewidths=0.3, 
                       alpha=0.5, label='Step1 predicted')
    
    if os.path.exists(step2_cp_csv):
        step2_data = np.loadtxt(step2_cp_csv, delimiter=',', skiprows=1)
        valid_mask = step2_data[:, 6] == 1
        step2_valid = step2_data[valid_mask]
        if len(step2_valid) > 0:
            step2_img_x = step2_valid[:, 0]
            step2_img_y = step2_valid[:, 1]
            step2_cat_x = step2_valid[:, 2]
            step2_cat_y = step2_valid[:, 3]
            
            for i in range(len(step2_valid)):
                ax.plot([step2_img_x[i], step2_cat_x[i]], 
                        [step2_img_y[i], step2_cat_y[i]], 
                        'yellow', linewidth=0.5, alpha=0.8)
            
            ax.scatter(step2_img_x, step2_img_y, c='yellow', s=8, marker='+', linewidths=0.5)
            ax.scatter(step2_cat_x, step2_cat_y, c='yellow', s=8, marker='+', linewidths=0.5)
            print(f"  Control points: {len(step2_valid)}")
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    output_png = os.path.join(output_dir, "debug_wcs_alignment.png")
    plt.savefig(output_png, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"  Saved: {output_png}")
    
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  WCS center offset from initial:")
    print(f"    RA: {result.center_ra - center_ra:.6f} deg")
    print(f"    Dec: {result.center_dec - center_dec:.6f} deg")
    print(f"  RMS: {result.rms_px:.3f} px ({result.rms_px * result.scale_arcsec_px:.3f} arcsec)")
    print(f"  SIP distortion correction: {'Yes' if result.sip_valid else 'No'}")
    
    return result

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plate Solve Debug Runner')
    parser.add_argument('--fits', required=True, help='FITS image path')
    parser.add_argument('--gaia', default=r"F:\Astro dev\Astro CS Normalization Database\GaiaDR3SP",
                        help='Gaia database directory')
    parser.add_argument('--output', default=r"F:\Astro dev\Astro CS Normalization Database\debug_output",
                        help='Output directory')
    parser.add_argument('--focal', type=float, default=0.0, help='Focal length (mm)')
    parser.add_argument('--pixel', type=float, default=0.0, help='Pixel size (um)')
    parser.add_argument('--scale', type=float, default=0.0, help='Scale (arcsec/px)')
    
    args = parser.parse_args()
    
    result = run_plate_solve_with_visualization(
        fits_path=args.fits,
        gaia_data_dir=args.gaia,
        output_dir=args.output,
        focal_length_mm=args.focal,
        pixel_size_um=args.pixel,
        scale_arcsec_px=args.scale,
    )
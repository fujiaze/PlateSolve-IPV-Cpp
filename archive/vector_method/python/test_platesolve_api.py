#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试PlateSolve统一API并绘制Gaia星点
动态从FITS头读取焦距和WCS信息
"""
import os
import sys
import time
import numpy as np
from PIL import Image, ImageDraw

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
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
from plate_solve import PlateSolve, PlateSolveConfig

def stretch_image(data):
    """直方图拉伸（0.1%-99.9%百分位）"""
    data = data.astype(np.float64)
    min_val = np.percentile(data, 0.1)
    max_val = np.percentile(data, 99.9)
    data = (data - min_val) / (max_val - min_val)
    data = np.clip(data, 0, 1)
    return (data * 255).astype(np.uint8)

def solve_image(image_path, gaia_dir, output_path=None):
    """
    对单帧图像执行PlateSolve
    
    参数:
        image_path: FITS图像路径
        gaia_dir: Gaia数据库目录
        output_path: 输出图像路径（可选）
    
    返回:
        result: PlateSolveResult对象
    """
    print(f"\n{'='*70}")
    print(f"处理: {os.path.basename(image_path)}")
    print(f"{'='*70}")
    
    # 读取FITS
    t0 = time.time()
    reader = ImageReader()
    with reader.read(image_path) as img_hdr:
        w, h = img_hdr.width, img_hdr.height
        image_array = img_hdr.data.copy()
        wcs = img_hdr.wcs
        
        # 动态获取焦距和像元尺寸
        focallen = img_hdr.get_keyword("FOCALLEN")
        if focallen is not None:
            focallen = float(focallen)
        pixel_size = img_hdr.get_keyword("XPIXSZ")
        if pixel_size is None:
            pixel_size = img_hdr.get_keyword("PIXSIZE")
        if pixel_size is None:
            pixel_size = 6.0  # 默认值
        else:
            pixel_size = float(pixel_size)
        
        # 动态获取初始坐标
        if wcs is not None:
            init_ra = wcs.crval1
            init_dec = wcs.crval2
            scale = wcs.pixel_scale
            has_wcs = True
        else:
            # 从FITS头读取
            objra = img_hdr.get_keyword("OBJRA")
            objdec = img_hdr.get_keyword("OBJDEC")
            if objra is not None and objdec is not None:
                init_ra = float(objra)
                init_dec = float(objdec)
            else:
                ra_str = img_hdr.get_keyword("RA")
                dec_str = img_hdr.get_keyword("DEC")
                if ra_str is not None and dec_str is not None:
                    # 解析时分秒格式 "18 11 14.00"
                    def parse_hms(s):
                        parts = str(s).split()
                        if len(parts) == 3:
                            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                            return h + m/60.0 + s/3600.0
                        return float(s)
                    init_ra = parse_hms(ra_str) * 15.0  # 时转度
                    init_dec = parse_hms(dec_str)
                else:
                    raise ValueError("无法从FITS头获取初始坐标")
            
            # 从焦距计算比例尺
            if focallen is not None:
                scale = 206.265 * pixel_size / float(focallen)
            else:
                scale = 6.0  # 默认值
            has_wcs = False
        
        crpix_x = wcs.crpix1 - 1 if wcs else w / 2.0
        crpix_y = wcs.crpix2 - 1 if wcs else h / 2.0
    
    print(f"  图像: {w}x{h}")
    print(f"  焦距: {focallen:.1f}mm" if focallen else "  焦距: 未指定")
    print(f"  像元尺寸: {pixel_size:.2f}um")
    print(f"  比例尺: {scale:.3f}\"/px")
    print(f"  初始中心: RA={init_ra:.6f}°, Dec={init_dec:.6f}°")
    print(f"  WCS状态: {'有效' if has_wcs else '无效'}")
    print(f"  读取耗时: {time.time()-t0:.2f}s")
    
    # 星点检测
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
    
    # PlateSolve
    t0 = time.time()
    
    config = PlateSolveConfig(
        use_saturated_priority=1,
        n_img_bright=500,
        n_cat_bright=600,
        max_match_dist_px=25.0,
        max_iterations=5,
        match_threshold=10.0,
        sip_order=5,
        converge_thresh=0.01
    )
    
    # 使用焦距计算比例尺
    if focallen is not None:
        focal_length_mm = float(focallen)
    else:
        focal_length_mm = 206.265 * pixel_size / scale
    
    with PlateSolve(gaia_data_dir=gaia_dir) as solver:
        result = solver.solve(
            img_x=img_x, img_y=img_y, img_flux=img_flux,
            img_saturated=saturated_arr, n_saturated=n_saturated,
            center_ra=init_ra, center_dec=init_dec,
            focal_length_mm=focal_length_mm, pixel_size_um=pixel_size,
            width=w, height=h,
            config=config,
            scale_arcsec_px=scale
        )
    
    solve_time = time.time() - t0
    print(f"\n  解析耗时: {solve_time:.2f}s")
    print(f"  第一步耗时: {result.step1_time_sec:.2f}s")
    print(f"  第二步耗时: {result.step2_time_sec:.2f}s")
    print(f"\n  === 解析结果 ===")
    print(f"  中心: RA={result.center_ra:.6f}°, Dec={result.center_dec:.6f}°")
    print(f"  旋转: {result.rotation_deg:.4f}°")
    print(f"  比例尺: {result.scale_arcsec_px:.4f}\"/px")
    print(f"  翻转模式: {result.flip_mode}")
    print(f"  匹配星对: {result.matched_count}")
    print(f"  RMS: {result.rms_px:.3f} px = {result.rms_px * result.scale_arcsec_px:.3f}\"")
    print(f"  SIP有效: {result.sip_valid}")
    
    # 绘制结果
    if output_path:
        draw_result(image_array, result, gaia_dir, w, h, output_path)
    
    return result

def sip_apply_inverse(u, v, ap, bp, order):
    """SIP逆向变换（Python实现）"""
    du = 0.0
    dv = 0.0
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i == 0 and j == 0: continue
            if i == 1 and j == 0: continue
            if i == 0 and j == 1: continue
            term = u ** i * v ** j
            du += ap[i][j] * term
            dv += bp[i][j] * term
    return du, dv

def sip_inverse_transform(x_px, y_px, ap, bp, order, center_x, center_y, max_radius):
    """对像素坐标应用SIP逆向变换"""
    u = (x_px - center_x) / max_radius
    v = (y_px - center_y) / max_radius
    du, dv = sip_apply_inverse(u, v, ap, bp, order)
    return x_px + du * max_radius, y_px + dv * max_radius

def draw_result(image_array, result, gaia_dir, w, h, output_path):
    """绘制Gaia星点到图像上（使用完整WCS变换）"""
    import ctypes
    
    print(f"\n  === 绘制Gaia星点 ===")
    print(f"  RMS: {result.rms_px:.3f} px = {result.rms_px * result.scale_arcsec_px:.3f}\"")
    
    wcs = result.wcs
    crpix1 = wcs["crpix1"]
    crpix2 = wcs["crpix2"]
    cd1_1 = wcs["cd1_1"]
    cd1_2 = wcs["cd1_2"]
    cd2_1 = wcs["cd2_1"]
    cd2_2 = wcs["cd2_2"]
    
    print(f"  WCS: crpix=({crpix1:.1f}, {crpix2:.1f})")
    print(f"  CD矩阵: [[{cd1_1:.6f}, {cd1_2:.6f}], [{cd2_1:.6f}, {cd2_2:.6f}]]")
    
    det = cd1_1 * cd2_2 - cd1_2 * cd2_1
    if abs(det) < 1e-15:
        print(f"  CD矩阵奇异，使用简化投影")
        use_cd = False
    else:
        cd_inv = [[cd2_2 / det, -cd1_2 / det], [-cd2_1 / det, cd1_1 / det]]
        use_cd = True
        print(f"  CD逆矩阵: [[{cd_inv[0][0]:.6f}, {cd_inv[0][1]:.6f}], [{cd_inv[1][0]:.6f}, {cd_inv[1][1]:.6f}]]")
    
    center_x = crpix1
    center_y = crpix2
    max_radius = np.sqrt((w/2)**2 + (h/2)**2)
    
    gaia_dll_path = os.path.join(project_root, "lib", "plate_solve", "plate_solve.dll")
    gaia_dll = ctypes.CDLL(gaia_dll_path)
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
    
    gaia_handle = gaia_dll.gaia_client_create(gaia_dir.encode("utf-8"))
    fov_radius = np.sqrt(w*w + h*h) * result.scale_arcsec_px / 3600.0 / 2.0 * 1.1
    
    ra_ptr = ctypes.POINTER(ctypes.c_double)()
    dec_ptr = ctypes.POINTER(ctypes.c_double)()
    mag_ptr = ctypes.POINTER(ctypes.c_float)()
    n_gaia = ctypes.c_int()
    
    rc = gaia_dll.gaia_client_cone_search_for_solver(
        gaia_handle, result.center_ra, result.center_dec, fov_radius, 14.0,
        ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_gaia)
    )
    
    if rc != 0 or n_gaia.value <= 0:
        print(f"  Gaia查询失败")
        return
    
    gaia_ra = np.ctypeslib.as_array(ra_ptr, shape=(n_gaia.value,)).copy()
    gaia_dec = np.ctypeslib.as_array(dec_ptr, shape=(n_gaia.value,)).copy()
    gaia_mag = np.ctypeslib.as_array(mag_ptr, shape=(n_gaia.value,)).copy()
    
    print(f"  Gaia星数: {n_gaia.value}")
    
    crval1 = result.center_ra * np.pi / 180.0
    crval2 = result.center_dec * np.pi / 180.0
    
    gaia_ra_rad = gaia_ra * np.pi / 180.0
    gaia_dec_rad = gaia_dec * np.pi / 180.0
    
    cos_dec0 = np.cos(crval2)
    sin_dec0 = np.sin(crval2)
    cos_dec = np.cos(gaia_dec_rad)
    sin_dec = np.sin(gaia_dec_rad)
    ra_diff = gaia_ra_rad - crval1
    cos_ra_diff = np.cos(ra_diff)
    sin_ra_diff = np.sin(ra_diff)
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff
    
    xi = cos_dec * sin_ra_diff / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c
    
    if use_cd:
        gaia_x_init = np.zeros(len(xi))
        gaia_y_init = np.zeros(len(xi))
        for i in range(len(xi)):
            gaia_x_init[i] = cd_inv[0][0] * xi[i] + cd_inv[0][1] * eta[i]
            gaia_y_init[i] = cd_inv[1][0] * xi[i] + cd_inv[1][1] * eta[i]
        gaia_x_init = gaia_x_init * 180.0 / np.pi + crpix1
        gaia_y_init = gaia_y_init * 180.0 / np.pi + crpix2
    else:
        rad_to_px = 180.0 / np.pi * 3600.0 / result.scale_arcsec_px
        gaia_x_init = xi * rad_to_px + w / 2.0
        gaia_y_init = -eta * rad_to_px + h / 2.0
    
    if result.sip is not None and result.sip_valid:
        ap = result.sip[2]
        bp = result.sip[3]
        order = 5
        
        n = len(gaia_x_init)
        gaia_x = np.zeros(n)
        gaia_y = np.zeros(n)
        
        for i in range(n):
            gaia_x[i], gaia_y[i] = sip_inverse_transform(
                gaia_x_init[i], gaia_y_init[i],
                ap, bp, order, center_x, center_y, max_radius
            )
        print(f"  SIP逆向变换: 已应用")
    else:
        gaia_x = gaia_x_init
        gaia_y = gaia_y_init
        print(f"  SIP逆向变换: 未应用")
    
    valid = (gaia_x >= 0) & (gaia_x < w) & (gaia_y >= 0) & (gaia_y < h)
    gaia_x = gaia_x[valid]
    gaia_y = gaia_y[valid]
    gaia_mag = gaia_mag[valid]
    print(f"  有效Gaia星: {len(gaia_x)}")
    print(f"  Gaia坐标范围: x=[{gaia_x.min():.0f}, {gaia_x.max():.0f}], y=[{gaia_y.min():.0f}, {gaia_y.max():.0f}]")
    
    bright_mask = gaia_mag < 12
    bright_x = gaia_x[bright_mask]
    bright_y = gaia_y[bright_mask]
    bright_mag = gaia_mag[bright_mask]
    print(f"  亮星(mag<12): {len(bright_x)}")
    
    stretched = stretch_image(image_array)
    img_pil = Image.fromarray(stretched).convert("RGB")
    draw = ImageDraw.Draw(img_pil)
    
    for i in range(len(bright_x)):
        x, y = int(bright_x[i]), int(bright_y[i])
        size = 10
        draw.line([x - size, y, x + size, y], fill=(255, 0, 0), width=2)
        draw.line([x, y - size, x, y + size], fill=(255, 0, 0), width=2)
    
    img_pil.save(output_path)
    print(f"  输出图像: {output_path}")
    
    gaia_dll.gaia_client_destroy(gaia_handle)

def main():
    print("=" * 70)
    print("PlateSolve统一API测试 - 动态读取FITS头信息")
    print("=" * 70)
    
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    dr3_dir = os.path.join(project_root, "GaiaDR3")
    
    # 获取所有FITS文件
    files = sorted([f for f in os.listdir(test_dir) if f.endswith('.fts')])
    
    # 测试两帧：第一帧有WCS，第二帧没有WCS
    test_files = []
    for f in files:
        path = os.path.join(test_dir, f)
        reader = ImageReader()
        with reader.read(path) as img:
            has_wcs = img.wcs is not None
            focallen = img.get_keyword("FOCALLEN")
        test_files.append((f, has_wcs, focallen))
        if len(test_files) >= 2:
            break
    
    print(f"\n测试文件:")
    for f, has_wcs, focallen in test_files:
        print(f"  {f}: WCS={'有' if has_wcs else '无'}, FOCALLEN={focallen}")
    
    # 处理每帧
    for f, has_wcs, focallen in test_files:
        image_path = os.path.join(test_dir, f)
        output_path = os.path.join(test_dir, f.replace('.fts', '_platesolve.png'))
        try:
            result = solve_image(image_path, dr3_dir, output_path)
        except Exception as e:
            print(f"\n  错误: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)

if __name__ == "__main__":
    main()

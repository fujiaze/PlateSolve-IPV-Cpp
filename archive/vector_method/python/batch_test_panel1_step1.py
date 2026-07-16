# -*- coding: utf-8 -*-
"""
Panel1所有帧第一步解析批量测试
测试动态重试机制（C++端实现）
"""

import os
import sys
import numpy as np
import time

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "gaia_xpsd_client", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "astro_image_io", "python"))

from plate_solve import PlateSolve, PlateSolveConfig
from star_detector import StarDetector
from astro_image_io import ImageReader

def parse_hms_ra(value):
    """解析RA的hms格式: '18 11 14.00' -> 度"""
    if not value:
        return None
    parts = value.split()
    if len(parts) >= 3:
        h = float(parts[0])
        m = float(parts[1])
        s = float(parts[2])
        return (h + m/60.0 + s/3600.0) * 15.0
    return None

def parse_dms_dec(value):
    """解析Dec的dms格式: '-13 10 37.0' -> 度"""
    if not value:
        return None
    parts = value.split()
    if len(parts) >= 3:
        sign = -1 if parts[0].startswith('-') else 1
        d = abs(float(parts[0]))
        m = float(parts[1])
        s = float(parts[2])
        return sign * (d + m/60.0 + s/3600.0)
    return None

def get_fits_info(img_data):
    """从FITS头提取信息"""
    info = {
        'ra': None,
        'dec': None,
        'focal_mm': 200.0,
        'pixel_um': 6.0,
        'filter': 'Unknown',
        'exposure': 0.0,
        'object': 'Unknown'
    }
    
    for kw in img_data.keywords:
        name = kw.name.upper()
        val = kw.value
        
        if name == 'RA' or name == 'OBJCTRA':
            ra = parse_hms_ra(val)
            if ra is not None:
                info['ra'] = ra
        elif name == 'DEC' or name == 'OBJCTDEC':
            dec = parse_dms_dec(val)
            if dec is not None:
                info['dec'] = dec
        elif name == 'FOCALLEN':
            try:
                info['focal_mm'] = float(val)
            except:
                pass
        elif name == 'XPIXSZ':
            try:
                info['pixel_um'] = float(val)
            except:
                pass
        elif name == 'FILTER':
            info['filter'] = val
        elif name == 'EXPTIME' or name == 'EXPOSURE':
            try:
                info['exposure'] = float(val)
            except:
                pass
        elif name == 'OBJECT':
            info['object'] = val
    
    return info

def batch_test_panel1():
    print("=" * 100)
    print("Panel1 所有帧第一步解析批量测试")
    print("策略: 饱和星>=10用1.5xGaia; 饱和星<10用动态重试(150->300->600)")
    print("=" * 100)
    
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    
    fits_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".fts")])
    if not fits_files:
        print("未找到测试文件")
        return
    
    print(f"\n共找到 {len(fits_files)} 个FITS文件")
    
    reader = ImageReader()
    results = []
    
    for i, fits_file in enumerate(fits_files):
        fits_path = os.path.join(test_dir, fits_file)
        print(f"\n[{i+1}/{len(fits_files)}] {fits_file}")
        
        try:
            img_data = reader.read(fits_path)
            img = img_data.data
            h, w = img.shape[:2]
            
            info = get_fits_info(img_data)
            
            if info['ra'] is None or info['dec'] is None:
                print("  跳过: 无法解析坐标")
                img_data.close()
                continue
            
            print(f"  原始: RA={info['ra']:.6f}° Dec={info['dec']:.6f}°")
            print(f"        焦距={info['focal_mm']:.1f}mm 像元={info['pixel_um']:.1f}μm")
            
            t0 = time.time()
            detector = StarDetector()
            coords, fluxes, saturated = detector.detect_ex(img)
            t_detect = time.time() - t0
            
            if len(coords) == 0:
                print("  跳过: 未检测到星点")
                img_data.close()
                continue
            
            n_stars = len(coords)
            n_saturated = sum(saturated)
            
            stars = np.zeros((n_stars, 6), dtype=np.float64)
            for j in range(n_stars):
                stars[j, 0] = coords[j][0]
                stars[j, 1] = coords[j][1]
                stars[j, 2] = fluxes[j]
                stars[j, 5] = saturated[j]
            
            img_x = stars[:, 0] - w / 2.0
            img_y = -(stars[:, 1] - h / 2.0)
            img_flux = stars[:, 2]
            saturated_mask = stars[:, 5].astype(np.int32)
            
            config = PlateSolveConfig()
            config.sip_order = 0
            config.max_iterations = 1
            
            t_solve_start = time.time()
            
            with PlateSolve(gaia_data_dir=gaia_dir) as solver:
                result = solver.solve(
                    img_x=img_x, img_y=img_y, img_flux=img_flux,
                    img_saturated=saturated_mask, n_saturated=n_saturated,
                    center_ra=info['ra'], center_dec=info['dec'],
                    focal_length_mm=info['focal_mm'], pixel_size_um=info['pixel_um'],
                    width=w, height=h,
                    config=config
                )
            
            t_solve = time.time() - t_solve_start
            
            matched_threshold = max(20, min(50, n_stars // 200))
            success = result.matched_count >= matched_threshold and result.rms_px < 5.0
            
            result_info = {
                'file': fits_file,
                'filter': info['filter'],
                'exposure': info['exposure'],
                'orig_ra': info['ra'],
                'orig_dec': info['dec'],
                'orig_focal': info['focal_mm'],
                'orig_scale': 206265.0 * info['pixel_um'] / info['focal_mm'],
                'solved_ra': result.center_ra,
                'solved_dec': result.center_dec,
                'solved_scale': result.scale_arcsec_px,
                'flip': result.flip_mode,
                'rotation': result.rotation_deg,
                'matched': result.matched_count,
                'rms': result.rms_px,
                'n_stars': n_stars,
                'n_saturated': n_saturated,
                't_detect': t_detect,
                't_solve': t_solve,
                'success': success
            }
            results.append(result_info)
            
            status = "✓ 成功" if success else "✗ 失败"
            print(f"  解析: RA={result.center_ra:.6f}° Dec={result.center_dec:.6f}°")
            print(f"        比例尺={result.scale_arcsec_px:.3f}\"/px 翻转={result.flip_mode} 旋转={result.rotation_deg:.3f}°")
            print(f"        匹配={result.matched_count} RMS={result.rms_px:.3f}px {status}")
            print(f"        耗时: 检测={t_detect:.1f}s 解析={t_solve:.1f}s")
            
            img_data.close()
                    
        except Exception as e:
            print(f"  文件错误: {e}")
            results.append({
                'file': fits_file,
                'success': False,
                'error': str(e)
            })
    
    print("\n" + "=" * 100)
    print("测试结果汇总")
    print("=" * 100)
    
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count
    
    print(f"\n总计: {len(results)} 帧, 成功: {success_count}, 失败: {fail_count}")
    
    if success_count > 0:
        print("\n" + "-" * 100)
        print(f"{'文件名':<55} {'原始坐标':>20} {'解析坐标':>20} {'比例尺':>8} {'翻转':>4} {'旋转':>8} {'匹配':>5} {'RMS':>6}")
        print("-" * 100)
        
        for r in results:
            if r.get('success', False):
                fname = r['file'][:52] + "..." if len(r['file']) > 55 else r['file']
                orig_coord = f"({r['orig_ra']:.3f},{r['orig_dec']:.3f})"
                solved_coord = f"({r['solved_ra']:.3f},{r['solved_dec']:.3f})"
                print(f"{fname:<55} {orig_coord:>20} {solved_coord:>20} {r['solved_scale']:>7.3f}\" {r['flip']:>4} {r['rotation']:>7.3f}° {r['matched']:>5} {r['rms']:>5.3f}px")
        
        print("-" * 100)
        
        rms_values = [r['rms'] for r in results if r.get('success', False)]
        matched_values = [r['matched'] for r in results if r.get('success', False)]
        
        print(f"\nRMS统计: min={min(rms_values):.3f}px, max={max(rms_values):.3f}px, avg={np.mean(rms_values):.3f}px")
        print(f"匹配统计: min={min(matched_values)}, max={max(matched_values)}, avg={np.mean(matched_values):.1f}")
    
    if fail_count > 0:
        print("\n失败文件:")
        for r in results:
            if not r.get('success', False):
                print(f"  {r['file']}: {r.get('error', 'matched<20 or rms>5')}")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    batch_test_panel1()
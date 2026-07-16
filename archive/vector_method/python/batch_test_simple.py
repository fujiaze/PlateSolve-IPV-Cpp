# -*- coding: utf-8 -*-
"""
Panel1批量测试 - 简化版
Python只做调度，C++端完成星点检测和匹配
"""

import os
import sys
import numpy as np
import time

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "astro_image_io", "python"))

from plate_solve import PlateSolve, PlateSolveConfig
from astro_image_io import ImageReader

def parse_hms_ra(value):
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
    info = {
        'ra': None, 'dec': None,
        'focal_mm': 200.0, 'pixel_um': 6.0,
        'filter': 'Unknown', 'exposure': 0.0
    }
    for kw in img_data.keywords:
        name = kw.name.upper()
        val = kw.value
        if name in ('RA', 'OBJCTRA'):
            ra = parse_hms_ra(val)
            if ra: info['ra'] = ra
        elif name in ('DEC', 'OBJCTDEC'):
            dec = parse_dms_dec(val)
            if dec: info['dec'] = dec
        elif name == 'FOCALLEN':
            try: info['focal_mm'] = float(val)
            except: pass
        elif name == 'XPIXSZ':
            try: info['pixel_um'] = float(val)
            except: pass
        elif name == 'FILTER':
            info['filter'] = val
        elif name in ('EXPTIME', 'EXPOSURE'):
            try: info['exposure'] = float(val)
            except: pass
    return info

def batch_test():
    print("=" * 80)
    print("Panel1批量测试 - C++端自动星点检测")
    print("策略: 匹配数不变时使用最大匹配对继续")
    print("=" * 80)
    
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    
    fits_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".fts")])
    print(f"\n共 {len(fits_files)} 个FITS文件")
    
    reader = ImageReader()
    results = []
    
    config = PlateSolveConfig()
    config.sip_order = 0
    config.max_iterations = 1
    
    for i, fits_file in enumerate(fits_files):
        fits_path = os.path.join(test_dir, fits_file)
        print(f"\n[{i+1}/{len(fits_files)}] {fits_file}")
        
        try:
            img_data = reader.read(fits_path)
            img = img_data.data
            h, w = img.shape[:2]
            
            info = get_fits_info(img_data)
            if info['ra'] is None or info['dec'] is None:
                print("  跳过: 无坐标")
                img_data.close()
                continue
            
            print(f"  原始: RA={info['ra']:.6f} Dec={info['dec']:.6f}")
            
            t0 = time.time()
            
            with PlateSolve(gaia_data_dir=gaia_dir) as solver:
                result = solver.solve_with_image(
                    image=img,
                    center_ra=info['ra'],
                    center_dec=info['dec'],
                    focal_length_mm=info['focal_mm'],
                    pixel_size_um=info['pixel_um'],
                    config=config
                )
            
            t_total = time.time() - t0
            
            success = result.matched_count > 3 and result.rms_px < 5.0
            status = "OK" if success else "FAIL"
            
            print(f"  结果: 匹配={result.matched_count} RMS={result.rms_px:.3f}px [{status}]")
            print(f"  耗时: {t_total:.1f}s")
            
            results.append({
                'file': fits_file, 'filter': info['filter'],
                'matched': result.matched_count, 'rms': result.rms_px,
                'success': success, 'time': t_total
            })
            
            img_data.close()
            
        except Exception as e:
            print(f"  错误: {e}")
            results.append({'file': fits_file, 'success': False, 'error': str(e)})
    
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count
    print(f"\n总计: {len(results)}帧, 成功: {success_count}, 失败: {fail_count}")
    
    if success_count > 0:
        rms_vals = [r['rms'] for r in results if r.get('success')]
        match_vals = [r['matched'] for r in results if r.get('success')]
        print(f"RMS: min={min(rms_vals):.3f} max={max(rms_vals):.3f} avg={np.mean(rms_vals):.3f}")
        print(f"匹配: min={min(match_vals)} max={max(match_vals)} avg={np.mean(match_vals):.1f}")
    
    if fail_count > 0:
        print("\n失败文件:")
        for r in results:
            if not r.get('success'):
                print(f"  {r['file']}: {r.get('error', 'matched<20')}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    batch_test()
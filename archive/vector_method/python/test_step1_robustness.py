# -*- coding: utf-8 -*-
"""
第一步解析鲁棒性测试
测试对没有WCS的图像，使用粗略的中心坐标、焦距和像元大小进行解析
"""

import os
import sys
import numpy as np

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "gaia_xpsd_client", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "astro_image_io", "python"))

from plate_solve import PlateSolve, PlateSolveConfig
from star_detector import StarDetector
from astro_image_io import ImageReader

def parse_ra_dec(img_data):
    """从img_data解析RA/Dec坐标"""
    ra_str = None
    dec_str = None
    
    for kw in img_data.keywords:
        if kw.name.upper() == "RA":
            ra_str = kw.value
        elif kw.name.upper() == "DEC":
            dec_str = kw.value
    
    if ra_str is None or dec_str is None:
        return 272.8, -13.1
    
    try:
        if " " in ra_str:
            parts = ra_str.split()
            init_ra = float(parts[0]) + float(parts[1])/60 + float(parts[2])/3600
        else:
            init_ra = float(ra_str)
        
        if " " in dec_str:
            parts = dec_str.split()
            sign = -1 if parts[0].startswith("-") else 1
            init_dec = sign * (abs(float(parts[0])) + float(parts[1])/60 + float(parts[2])/3600)
        else:
            init_dec = float(dec_str)
        
        return init_ra, init_dec
    except:
        return 272.8, -13.1

def test_step1_robustness():
    print("=" * 70)
    print("第一步解析鲁棒性测试")
    print("=" * 70)
    
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    
    fits_files = [f for f in os.listdir(test_dir) if f.endswith(".fts")]
    if not fits_files:
        print("未找到测试文件")
        return
    
    test_cases = [
        {"name": "精确坐标", "ra_offset": 0, "dec_offset": 0, "scale_factor": 1.0},
        {"name": "RA偏移+1度", "ra_offset": 1, "dec_offset": 0, "scale_factor": 1.0},
        {"name": "Dec偏移+1度", "ra_offset": 0, "dec_offset": 1, "scale_factor": 1.0},
        {"name": "比例尺偏大20%", "ra_offset": 0, "dec_offset": 0, "scale_factor": 1.2},
        {"name": "比例尺偏小20%", "ra_offset": 0, "dec_offset": 0, "scale_factor": 0.8},
        {"name": "坐标+比例都偏", "ra_offset": 0.5, "dec_offset": 0.5, "scale_factor": 1.1},
        {"name": "RA偏移-2度", "ra_offset": -2, "dec_offset": 0, "scale_factor": 1.0},
        {"name": "Dec偏移-2度", "ra_offset": 0, "dec_offset": -2, "scale_factor": 1.0},
        {"name": "比例尺偏大50%", "ra_offset": 0, "dec_offset": 0, "scale_factor": 1.5},
        {"name": "比例尺偏小50%", "ra_offset": 0, "dec_offset": 0, "scale_factor": 0.5},
    ]
    
    reader = ImageReader()
    
    for fits_file in fits_files[:3]:
        fits_path = os.path.join(test_dir, fits_file)
        print(f"\n测试文件: {fits_file}")
        
        try:
            img_data = reader.read(fits_path)
            img = img_data.data
            h, w = img.shape[:2]
            print(f"  图像尺寸: {w}x{h}")
            
            detector = StarDetector()
            coords, fluxes, saturated = detector.detect_ex(img)
            if len(coords) == 0:
                print("  未检测到星点")
                continue
            
            n_stars = len(coords)
            stars = np.zeros((n_stars, 6), dtype=np.float64)
            for i in range(n_stars):
                stars[i, 0] = coords[i][0]
                stars[i, 1] = coords[i][1]
                stars[i, 2] = fluxes[i]
                stars[i, 5] = saturated[i]
            
            img_x = stars[:, 0] - w / 2.0
            img_y = -(stars[:, 1] - h / 2.0)
            img_flux = stars[:, 2]
            
            saturated_mask = stars[:, 5].astype(np.int32)
            n_saturated = int(np.sum(saturated_mask))
            
            print(f"  检测星数: {len(stars)}, 饱和星: {n_saturated}")
            
            focal_length_mm = 200.0
            pixel_size = 6.0
            
            init_ra, init_dec = parse_ra_dec(img_data)
            
            print(f"  初始坐标: RA={init_ra:.4f}°, Dec={init_dec:.4f}°")
            
            for case in test_cases:
                print(f"\n  --- 测试: {case['name']} ---")
                
                test_ra = init_ra + case["ra_offset"]
                test_dec = init_dec + case["dec_offset"]
                test_focal = focal_length_mm / case["scale_factor"]
                
                config = PlateSolveConfig()
                config.sip_order = 0
                config.max_iterations = 1
                
                try:
                    with PlateSolve(gaia_data_dir=gaia_dir) as solver:
                        result = solver.solve(
                            img_x=img_x, img_y=img_y, img_flux=img_flux,
                            img_saturated=saturated_mask, n_saturated=n_saturated,
                            center_ra=test_ra, center_dec=test_dec,
                            focal_length_mm=test_focal, pixel_size_um=pixel_size,
                            width=w, height=h,
                            config=config
                        )
                        
                        print(f"    匹配数: {result.matched_count}")
                        print(f"    RMS: {result.rms_px:.3f} px")
                        print(f"    翻转: {result.flip_mode}")
                        print(f"    旋转: {result.rotation_deg:.3f}°")
                        print(f"    中心: RA={result.center_ra:.6f}°, Dec={result.center_dec:.6f}°")
                        
                        if result.matched_count > 50 and result.rms_px < 5.0:
                            print(f"    结果: ✓ 成功")
                        elif result.matched_count > 20 and result.rms_px < 10.0:
                            print(f"    结果: ~ 部分成功")
                        else:
                            print(f"    结果: ✗ 失败")
                            
                except Exception as e:
                    print(f"    错误: {e}")
            
            img_data.close()
                    
        except Exception as e:
            import traceback
            print(f"  文件读取错误: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)

if __name__ == "__main__":
    test_step1_robustness()

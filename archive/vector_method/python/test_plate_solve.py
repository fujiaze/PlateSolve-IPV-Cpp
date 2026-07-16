"""
Plate Solve 测试脚本
功能: 验证astap集成和解析流程
用途: 测试文件头WCS检测、astap盲解析、Gaia匹配
"""

import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

import numpy as np

from lib.astro_image_io.python.astro_image_io import ImageReader
from lib.plate_solve.python.plate_solve import PlateSolver, PSolveImageDataPy
from lib.plate_solve.python.config import PlateSolveConfig


def main():
    print("=== Plate Solve 测试脚本 ===")
    print(f"当前目录: {os.getcwd()}")
    
    test_image_path = r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts"
    
    if not os.path.exists(test_image_path):
        print(f"测试图像不存在: {test_image_path}")
        return
    
    print(f"测试图像: {test_image_path}")
    
    config = PlateSolveConfig.from_file()
    print(f"配置文件加载成功")
    print(f"  ASTAP启用: {config.astap.enabled}")
    print(f"  ASTAP路径: {config.astap.path}")
    print(f"  Gaia数据目录: {config.gaia.data_dir}")
    print(f"  使用文件头WCS: {config.solver.use_header_wcs}")
    
    reader = ImageReader()
    
    try:
        print("\n=== 1. 读取图像 ===")
        start_time = time.time()
        with reader.read(test_image_path) as img_data:
            print(f"  图像尺寸: {img_data.width} x {img_data.height}")
            print(f"  源格式: {img_data.source_format}")
            print(f"  有WCS: {img_data.has_wcs}")
            
            if img_data.wcs:
                print(f"  CRVAL1: {img_data.wcs.crval1:.6f}")
                print(f"  CRVAL2: {img_data.wcs.crval2:.6f}")
                print(f"  像元尺度: {img_data.pixel_scale_arcsec:.3f}\"/px")
            
            if img_data.metadata.observation:
                obs = img_data.metadata.observation
                print(f"  焦距: {obs.focallen}mm")
                print(f"  像元尺寸: {obs.xpixsz}um")
            
            if img_data.metadata.calibration:
                cal = img_data.metadata.calibration
                print(f"  曝光时间: {cal.exptime}s")
                print(f"  滤镜: {cal.filter_name}")
            
            image_array = img_data.data
            print(f"  读取耗时: {time.time() - start_time:.3f}s")
        
        print("\n=== 2. 初始化PlateSolver ===")
        start_time = time.time()
        solver = PlateSolver(
            gaia_data_dir=config.gaia.data_dir
        )
        print(f"  初始化耗时: {time.time() - start_time:.3f}s")
        
        print("\n=== 3. 测试ASTAP解析 ===")
        start_time = time.time()
        astap_result = solver.solve_with_astap(test_image_path)
        print(f"  ASTAP耗时: {time.time() - start_time:.3f}s")
        
        if astap_result and astap_result.success:
            print(f"  ASTAP成功:")
            print(f"    RA: {astap_result.ra_deg:.6f}")
            print(f"    DEC: {astap_result.dec_deg:.6f}")
            print(f"    Scale: {astap_result.scale_arcsec_px:.3f}\"/px")
            print(f"    Rotation: {astap_result.rotation_deg:.2f}deg")
            print(f"    匹配星数: {astap_result.matched_stars}")
            print(f"    RMS: {astap_result.rms_arcsec:.3f}\"")
        else:
            print(f"  ASTAP失败: {astap_result.error_message if astap_result else '未初始化'}")
        
        print("\n=== 4. 准备图像数据 ===")
        img_metadata = reader.read_metadata(test_image_path)
        
        img_data = PSolveImageDataPy()
        img_data.width = img_metadata.geometry.width
        img_data.height = img_metadata.geometry.height
        
        if img_metadata.observation:
            if img_metadata.observation.focallen:
                img_data.focal_length_mm = img_metadata.observation.focallen
            else:
                img_data.focal_length_mm = 200.0
            
            if img_metadata.observation.xpixsz:
                img_data.pixel_size_um = img_metadata.observation.xpixsz
            else:
                img_data.pixel_size_um = 6.0
        
        if img_metadata.calibration:
            img_data.exposure_time_s = img_metadata.calibration.exptime
        
        print(f"  焦距: {img_data.focal_length_mm}mm")
        print(f"  像元尺寸: {img_data.pixel_size_um}um")
        print(f"  曝光时间: {img_data.exposure_time_s}s")
        
        print("\n=== 5. 测试粗解析 (使用ASTAP结果) ===")
        start_time = time.time()
        
        det_x = []
        det_y = []
        
        try:
            result = solver.solve_coarse(
                image=image_array,
                img_data=img_data,
                det_x=det_x,
                det_y=det_y,
                image_path=test_image_path
            )
            
            print(f"  粗解析成功!")
            print(f"  耗时: {time.time() - start_time:.3f}s")
            print(f"  像元尺度: {result.scale_arcsec_px:.3f}\"/px")
            print(f"  FOV: {result.fov_w_arcmin:.0f}' x {result.fov_h_arcmin:.0f}'")
            print(f"  Gaia星数: {result.gaia_star_count}")
            print(f"  匹配星数: {result.matched_count}")
            print(f"  RMS: {result.rms_total:.3f}px")
            print(f"  迭代次数: {result.iteration_count}")
            
        except Exception as e:
            print(f"  粗解析失败: {e}")
        
        solver.close()
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
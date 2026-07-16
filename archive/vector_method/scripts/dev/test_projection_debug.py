"""
投影验证debug脚本

测试正常帧和问题帧的投影误差，生成debug图像
"""

import os
import sys
import math
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch, gnomonic_forward, _apply_flip
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "output", "debug_projection")
os.makedirs(DEBUG_DIR, exist_ok=True)

# 正常帧（H-alpha）
NORMAL_FILE = "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@061318-300S-H-alpha.fts"

# 问题帧（Blue）
PROBLEM_FILE = "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@055805-180S-Blue.fts"


def test_and_debug(file_path, label):
    """测试并生成debug图像"""
    file_name = os.path.basename(file_path)
    full_path = os.path.join(PROJECT_ROOT, file_path)
    
    print(f"\n{'='*60}")
    print(f"{label}: {file_name}")
    print(f"{'='*60}")
    
    # 读取图像
    reader = ImageReader()
    img = reader.read(full_path)
    
    # 星点检测
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)
    
    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_sat = np.array(det_result.saturated, dtype=np.int32)
    
    print(f"检测星数: {len(img_x)}, 饱和星: {int(np.sum(img_sat))}")
    
    # 提取初始参数
    center_ra = 0.0
    center_dec = 0.0
    focal_length = 200.0
    pixel_size = 6.0
    
    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        center_ra = img.metadata.wcs.crval1
        center_dec = img.metadata.wcs.crval2
    if img.metadata.observation:
        if img.metadata.observation.focallen is not None:
            focal_length = img.metadata.observation.focallen
        if img.metadata.observation.xpixsz is not None:
            pixel_size = img.metadata.observation.xpixsz
    
    if center_ra == 0.0 and center_dec == 0.0:
        for kw in img.keywords:
            name = kw.name.upper()
            if name in ("OBJCTRA", "RA"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                    if len(parts) >= 3:
                        center_ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
            elif name in ("OBJCTDEC", "DEC"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                    if len(parts) >= 3:
                        sign = -1 if parts[0].startswith("-") else 1
                        center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
    
    s0 = 206.265 * pixel_size / focal_length
    print(f"初始参数: RA={center_ra:.6f}° Dec={center_dec:.6f}° s0={s0:.4f}\"/px")
    
    # 向量匹配
    vm = VectorMatch(GAIA_DIR, db_type=2)
    vm_result = vm.solve(
        img_x=img_x, img_y=img_y, img_flux=img_flux, img_saturated=img_sat,
        center_ra=center_ra, center_dec=center_dec,
        focal_length_mm=focal_length, pixel_size_um=pixel_size,
        width=img.width, height=img.height,
    )
    
    if vm_result is None:
        print("匹配失败!")
        vm.close()
        return
    
    print(f"匹配成功: RA={vm_result.center_ra:.6f}° Dec={vm_result.center_dec:.6f}°")
    print(f"  s={vm_result.scale_arcsec_px:.4f}\"/px RMS={vm_result.rms_px:.3f}px n={vm_result.matched_count}")
    print(f"  affine: tx={vm_result.affine[0]:.2f}\" ty={vm_result.affine[3]:.2f}\"")
    
    # 投影验证
    a0, a1, a2, b0, b1, b2 = vm_result.affine
    fov_diag = math.sqrt(img.width ** 2 + img.height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0
    
    ra_arr, dec_arr, mag_arr = vm._gaia.cone_search(
        vm_result.center_ra, vm_result.center_dec, radius_deg, 22.0
    )
    
    # 只用最亮的1000颗Gaia星验证（和之前报告一致）
    sort_idx = np.argsort(mag_arr)
    n_top = min(1000, len(mag_arr))
    top_idx = sort_idx[:n_top]
    top_ra, top_dec = ra_arr[top_idx], dec_arr[top_idx]
    
    # 用修正后的中心投影
    xi, eta, valid = gnomonic_forward(top_ra, top_dec, vm_result.center_ra, vm_result.center_dec)
    W = np.column_stack([xi[valid], eta[valid]])
    Wf = _apply_flip(W, vm_result.flip_mode)
    
    u = a0 + a1 * Wf[:, 0] + a2 * Wf[:, 1]
    v = b0 + b1 * Wf[:, 0] + b2 * Wf[:, 1]
    px = u / s0 + img.width / 2.0
    py = -v / s0 + img.height / 2.0
    
    in_img = (px >= 0) & (px < img.width) & (py >= 0) & (py < img.height)
    print(f"Gaia星在图内: {np.sum(in_img)}/{len(Wf)}")
    
    if np.any(in_img):
        tree = cKDTree(np.column_stack([img_x, img_y]))
        dists, idxs = tree.query(np.column_stack([px[in_img], py[in_img]]))
        med_err = float(np.median(dists))
        lt2px = float(np.sum(dists < 2) / len(dists) * 100)
        print(f"投影误差: 中位={med_err:.2f}px <2px={lt2px:.1f}%")
        
        # 分析误差分布
        print(f"误差分布: min={dists.min():.2f} p25={np.percentile(dists,25):.2f} "
              f"p50={np.percentile(dists,50):.2f} p75={np.percentile(dists,75):.2f} "
              f"p90={np.percentile(dists,90):.2f} max={dists.max():.2f}px")
        
        # 系统性偏移分析
        matched_img_x = img_x[idxs]
        matched_img_y = img_y[idxs]
        dx = px[in_img] - matched_img_x
        dy = py[in_img] - matched_img_y
        print(f"系统偏移: dx_med={np.median(dx):.2f}px dy_med={np.median(dy):.2f}px")
        
        # 按距离分桶统计
        cx_img, cy_img = img.width / 2.0, img.height / 2.0
        r_from_center = np.sqrt((px[in_img] - cx_img)**2 + (py[in_img] - cy_img)**2)
        r_max = math.sqrt(cx_img**2 + cy_img**2)
        r_norm = r_from_center / r_max
        
        print(f"误差vs距中心距离:")
        for r_bin in [(0,0.25), (0.25,0.5), (0.5,0.75), (0.75,1.0)]:
            mask_bin = (r_norm >= r_bin[0]) & (r_norm < r_bin[1])
            if np.any(mask_bin):
                bin_err = dists[mask_bin]
                bin_dx = dx[mask_bin]
                bin_dy = dy[mask_bin]
                print(f"  r=[{r_bin[0]:.2f},{r_bin[1]:.2f}]: n={np.sum(mask_bin)} "
                      f"err={np.median(bin_err):.2f}px dx={np.median(bin_dx):.2f}px dy={np.median(bin_dy):.2f}px")
    
    # 生成debug图像
    try:
        from PIL import Image as PILImage
        data = img.data.astype(np.float64)
        p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
        data_norm = np.clip((data - p2) / (p98 - p2), 0, 1)
        rgb = np.stack([data_norm, data_norm, data_norm], axis=-1)
        rgb = (rgb * 255).astype(np.uint8)
        
        # 标记Gaia星预测位置（红色十字）
        if np.any(in_img):
            for i in range(min(200, np.sum(in_img))):
                idx = np.where(in_img)[0][i]
                x, y = int(px[idx]), int(py[idx])
                if 0 <= x < img.width and 0 <= y < img.height:
                    # 红色十字
                    for dx in range(-10, 11):
                        if 0 <= x+dx < img.width:
                            rgb[y, x+dx, 0] = 255
                            rgb[y, x+dx, 1] = 0
                            rgb[y, x+dx, 2] = 0
                    for dy in range(-10, 11):
                        if 0 <= y+dy < img.height:
                            rgb[y+dy, x, 0] = 255
                            rgb[y+dy, x, 1] = 0
                            rgb[y+dy, x, 2] = 0
        
        # 标记检测到的星点（绿色圆）
        for i in range(min(100, len(img_x))):
            x, y = int(img_x[i]), int(img_y[i])
            if 0 <= x < img.width and 0 <= y < img.height:
                for dx in range(-3, 4):
                    for dy in range(-3, 4):
                        if abs(dx) <= 2 or abs(dy) <= 2:
                            if 0 <= x+dx < img.width and 0 <= y+dy < img.height:
                                rgb[y+dy, x+dx, 1] = 255
        
        debug_path = os.path.join(DEBUG_DIR, f"{file_name}_debug.png")
        PILImage.fromarray(rgb).save(debug_path)
        print(f"Debug图像已保存: {debug_path}")
        
    except Exception as e:
        print(f"生成debug图像失败: {e}")
    
    vm.close()


if __name__ == "__main__":
    # 测试正常帧
    test_and_debug(NORMAL_FILE, "正常帧(H-alpha)")
    
    # 测试问题帧
    test_and_debug(PROBLEM_FILE, "问题帧(Blue)")
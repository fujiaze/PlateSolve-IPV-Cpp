[Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[System.Text.Encoding]::Default = [System.Text.Encoding]::UTF8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import ctypes
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plate_solve import PlateSolveAPI

def visualize_matching():
    dll_path = os.path.join(os.path.dirname(__file__), '..', 'plate_solve.dll')
    gaia_dir = r'F:\Astro dev\Astro DR3SP'
    
    test_file = r'F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@051551-180S-Red.fts'
    
    ps = PlateSolveAPI(dll_path, gaia_dir)
    
    img_data, metadata = ps.read_image(test_file)
    if img_data is None:
        print("Failed to read image")
        return
    
    width = metadata['width']
    height = metadata['height']
    
    stars = ps.detect_stars(img_data, width, height)
    if stars is None:
        print("Failed to detect stars")
        return
    
    print(f"Detected {stars['count']} stars")
    
    config = {
        'center_ra': metadata.get('center_ra', 272.8),
        'center_dec': metadata.get('center_dec', -13.2),
        'scale_arcsec_px': metadata.get('scale_arcsec_px', 6.2),
        'pixel_size_um': metadata.get('pixel_size_um', 3.76),
        'focal_length_mm': metadata.get('focal_length_mm', 250),
        'sip_order': 5
    }
    
    result = ps.solve(img_data, width, height, stars, config)
    if result is None:
        print("Failed to solve")
        return
    
    print(f"\n粗匹配结果:")
    print(f"  中心: RA={result['center_ra']:.6f}, Dec={result['center_dec']:.6f}")
    print(f"  旋转: {result['rotation_deg']:.3f}°")
    print(f"  比例尺: {result['scale_arcsec_px']:.3f} arcsec/px")
    print(f"  翻转模式: {result['flip_mode']}")
    print(f"  匹配数: {result['matched_count']}")
    print(f"  RMS: {result['rms_px']:.3f} px")
    
    print(f"\n精细拟合结果:")
    print(f"  控制点数: {result['control_points']}")
    print(f"  RMS: {result['rms_px']:.3f} px")
    print(f"  SIP有效: {result['sip_valid']}")
    
    det_x = stars['x']
    det_y = stars['y']
    det_flux = stars['flux']
    det_count = stars['count']
    
    img_x = det_x - width / 2.0
    img_y = -(det_y - height / 2.0)
    
    top1000_idx = np.argsort(det_flux)[-1000:][::-1]
    img_x_top1000 = img_x[top1000_idx]
    img_y_top1000 = img_y[top1000_idx]
    
    center_ra = config['center_ra']
    center_dec = config['center_dec']
    scale = result['scale_arcsec_px']
    rotation = result['rotation_deg']
    flip = result['flip_mode']
    
    aff_a0 = result.get('affine_a0', 0)
    aff_a1 = result.get('affine_a1', 1)
    aff_a2 = result.get('affine_a2', 0)
    aff_b0 = result.get('affine_b0', 0)
    aff_b1 = result.get('affine_b1', 0)
    aff_b2 = result.get('affine_b2', 1)
    
    print(f"\nAffine参数:")
    print(f"  a0={aff_a0:.3f} a1={aff_a1:.6f} a2={aff_a2:.6f}")
    print(f"  b0={aff_b0:.3f} b1={aff_b1:.6f} b2={aff_b2:.6f}")
    
    gaia_stars = ps.query_gaia(center_ra, center_dec, scale, width, height)
    if gaia_stars is None or len(gaia_stars['ra']) == 0:
        print("Failed to query Gaia")
        return
    
    print(f"Gaia星点数: {len(gaia_stars['ra'])}")
    
    ra_rad = np.array(gaia_stars['ra']) * np.pi / 180.0
    dec_rad = np.array(gaia_stars['dec']) * np.pi / 180.0
    center_ra_rad = center_ra * np.pi / 180.0
    center_dec_rad = center_dec * np.pi / 180.0
    
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(center_dec_rad)
    sin_dec0 = np.sin(center_dec_rad)
    ra_diff = ra_rad - center_ra_rad
    cos_ra_diff = np.cos(ra_diff)
    sin_ra_diff = np.sin(ra_diff)
    
    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff
    xi = cos_dec * sin_ra_diff / cos_c
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c
    
    rad_to_px = 180.0 / np.pi * 3600.0 / scale
    px = xi * rad_to_px
    py = -eta * rad_to_px
    
    det = aff_a1 * aff_b2 - aff_a2 * aff_b1
    if abs(det) > 1e-10:
        inv_a1 = aff_b2 / det
        inv_a2 = -aff_a2 / det
        inv_b1 = -aff_b1 / det
        inv_b2 = aff_a1 / det
        inv_a0 = -(inv_a1 * aff_a0 + inv_a2 * aff_b0)
        inv_b0 = -(inv_b1 * aff_a0 + inv_b2 * aff_b0)
    else:
        inv_a0, inv_a1, inv_a2 = 0, 1, 0
        inv_b0, inv_b1, inv_b2 = 0, 0, 1
    
    pred_x = inv_a0 + inv_a1 * px + inv_a2 * py
    pred_y = inv_b0 + inv_b1 * px + inv_b2 * py
    
    if flip == 1:
        pred_x = -pred_x
    elif flip == 2:
        pred_y = -pred_y
    elif flip == 3:
        pred_x = -pred_x
        pred_y = -pred_y
    
    pred_x_abs = pred_x + width / 2.0
    pred_y_abs = -(pred_y - height / 2.0)
    
    gaia_mag = np.array(gaia_stars['mag'])
    top1000_gaia_idx = np.argsort(gaia_mag)[:1000]
    
    pred_x_top1000 = pred_x_abs[top1000_gaia_idx]
    pred_y_top1000 = pred_y_abs[top1000_gaia_idx]
    
    img_display = img_data.astype(np.float32)
    if img_display.max() > 0:
        img_display = img_display / img_display.max()
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.imshow(img_display, cmap='gray', origin='lower')
    
    ax.scatter(pred_x_top1000, pred_y_top1000, c='red', s=30, marker='+', linewidths=1.5, label='Gaia前1000亮星预测位置')
    
    control_points = result.get('control_points_data', None)
    if control_points is not None:
        cp_img_x = control_points['img_x'] + width / 2.0
        cp_img_y = -(control_points['img_y'] - height / 2.0)
        ax.scatter(cp_img_x, cp_img_y, c='green', s=50, marker='o', facecolors='none', linewidths=1.5, label='控制点')
    
    ax.scatter(img_x_top1000 + width/2, -(img_y_top1000 - height/2), c='blue', s=20, marker='x', linewidths=1, alpha=0.5, label='图像前1000亮星')
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_title(f'匹配可视化: 红十字=Gaia预测位置, 绿圈=控制点, 蓝X=图像星点\nRMS={result["rms_px"]:.3f}px, 控制点={result["control_points"]}')
    ax.legend(loc='upper right')
    
    output_path = os.path.join(os.path.dirname(__file__), 'debug_matching_visual.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化保存到: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    visualize_matching()
# -*- coding: utf-8 -*-
"""
Debug visualization - Minimal output with histogram stretching
Red cross: Gaia predictions, Red circle: Control points
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(project_root, "lib", "astro_image_io", "python"))

_mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(_mingw_bin):
    os.environ["PATH"] = _mingw_bin + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_mingw_bin)
    except OSError:
        pass

from plate_solve import PlateSolve, PlateSolveConfig
from star_detector import StarDetector
from astro_image_io import ImageReader

def visualize_debug():
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_file = os.path.join(project_root, "testdata", "lights", "panel1",
                             "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@051551-180S-Red.fts")
    
    print("=" * 80)
    print(f"Debug visualization: {os.path.basename(test_file)}")
    print("=" * 80)
    
    reader = ImageReader()
    img_obj = reader.read_fits(test_file)
    if img_obj is None:
        print("Failed to read image")
        return
    
    img_data = img_obj.data
    width = img_obj.metadata.geometry.width
    height = img_obj.metadata.geometry.height
    print(f"Image size: {width}x{height}")
    
    config = PlateSolveConfig(
        use_saturated_priority=1,
        n_img_bright=500,
        n_cat_bright=600,
        max_match_dist_px=25.0,
        max_iterations=5,
        match_threshold=10.0,
        sip_order=5,
    )
    
    solver = PlateSolve(gaia_data_dir=gaia_dir)
    result = solver.solve_with_file(test_file, config)
    
    print(f"\nCoarse matching result:")
    print(f"  Center: RA={result.center_ra:.6f}, Dec={result.center_dec:.6f}")
    print(f"  Rotation: {result.rotation_deg:.3f} deg")
    print(f"  Scale: {result.scale_arcsec_px:.3f} arcsec/px")
    print(f"  Flip mode: {result.flip_mode}")
    print(f"  Matched: {result.matched_count}")
    print(f"  RMS: {result.rms_px:.3f} px")
    
    solver.close()
    
    gaia_csv = os.path.join(project_root, "debug_gaia_predictions.csv")
    cp_csv = os.path.join(project_root, "debug_control_points.csv")
    
    gaia_data_csv = None
    cp_data_csv = None
    
    if os.path.exists(gaia_csv):
        gaia_data_csv = np.genfromtxt(gaia_csv, delimiter=',', skip_header=1)
        print(f"\nLoaded Gaia predictions: {len(gaia_data_csv)} entries")
    
    if os.path.exists(cp_csv):
        cp_data_csv = np.genfromtxt(cp_csv, delimiter=',', skip_header=1)
        print(f"Loaded control points: {len(cp_data_csv)} entries")
    
    half_w = width / 2.0
    half_h = height / 2.0
    
    img_min, img_max = np.percentile(img_data, [0.5, 99.5])
    stretched = np.clip((img_data - img_min) / (img_max - img_min), 0, 1)
    rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float32)
    
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    ax.imshow(rgb, origin='lower')
    
    if gaia_data_csv is not None:
        gaia_pred_x = gaia_data_csv[:, 0] + half_w
        gaia_pred_y = -(gaia_data_csv[:, 1] - half_h)
        ax.scatter(gaia_pred_x, gaia_pred_y, c='red', s=15, marker='+', linewidths=1)
    
    if cp_data_csv is not None:
        cp_img_x = cp_data_csv[:, 0]
        cp_img_y = cp_data_csv[:, 1]
        valid_mask = cp_data_csv[:, 6] == 1
        ax.scatter(cp_img_x[valid_mask], cp_img_y[valid_mask], c='red', s=25,
                   marker='o', facecolors='none', linewidths=1.5)
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    ax.axis('off')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    output_path = os.path.join(project_root, "lib", "plate_solve", "python", "debug_inverse_map.png")
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    print(f"\nSaved to: {output_path}")
    
    plt.close()

if __name__ == '__main__':
    visualize_debug()
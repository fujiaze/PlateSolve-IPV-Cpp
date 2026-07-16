# -*- coding: utf-8 -*-
"""
PlateSolve Debug Visualization - 调试可视化工具

用于验证Plate Solving结果的准确性:
- Step1粗匹配预测位置与图像星点的对齐
- Step2精细匹配控制点的偏移量
- Gaia预测位置与图像星点的对齐
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from typing import Optional, Tuple

def visualize_alignment(
    fits_path: str,
    gaia_csv: str,
    cp_csv: str,
    output_path: str,
    dpi: int = 100,
) -> Tuple[int, int]:
    """
    生成全尺寸对齐验证图片
    
    参数:
        fits_path: FITS图像文件路径
        gaia_csv: Gaia预测位置CSV文件路径
        cp_csv: 控制点CSV文件路径
        output_path: 输出PNG文件路径
        dpi: 输出DPI
    
    返回:
        (width, height) 图片尺寸
    """
    with fits.open(fits_path) as hdul:
        img_data = hdul[0].data
    
    height, width = img_data.shape
    half_w = width / 2.0
    half_h = height / 2.0
    
    gaia_data = np.loadtxt(gaia_csv, delimiter=',', skiprows=1)
    gaia_pred_x = gaia_data[:, 0] + half_w
    gaia_pred_y = half_h - gaia_data[:, 1]
    
    cp_data = np.loadtxt(cp_csv, delimiter=',', skiprows=1)
    valid_mask = cp_data[:, 6] == 1
    cp_valid = cp_data[valid_mask]
    cp_img_x = cp_valid[:, 0]
    cp_img_y = cp_valid[:, 1]
    cp_cat_x = cp_valid[:, 2]
    cp_cat_y = cp_valid[:, 3]
    
    img_min, img_max = np.percentile(img_data, [0.5, 99.5])
    stretched = np.clip((img_data - img_min) / (img_max - img_min), 0, 1)
    rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float32)
    
    fig = plt.figure(figsize=(width/100, height/100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin='lower', extent=[0, width, 0, height])
    
    ax.scatter(gaia_pred_x, gaia_pred_y, c='red', s=8, marker='+', linewidths=0.5)
    
    for i in range(len(cp_valid)):
        ax.plot([cp_img_x[i], cp_cat_x[i]], [cp_img_y[i], cp_cat_y[i]], 
                'yellow', linewidth=0.5, alpha=0.8)
    
    ax.scatter(cp_img_x, cp_img_y, c='yellow', s=8, marker='+', linewidths=0.5)
    ax.scatter(cp_cat_x, cp_cat_y, c='yellow', s=8, marker='+', linewidths=0.5)
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return width, height

def visualize_step1_step2_comparison(
    fits_path: str,
    step1_csv: str,
    step2_csv: str,
    output_path: str,
    dpi: int = 100,
) -> Tuple[int, int]:
    """
    生成Step1和Step2对比可视化图片
    
    参数:
        fits_path: FITS图像文件路径
        step1_csv: Step1粗匹配预测CSV
        step2_csv: Step2控制点CSV
        output_path: 输出PNG文件路径
        dpi: 输出DPI
    
    返回:
        (width, height) 图片尺寸
    """
    with fits.open(fits_path) as hdul:
        img_data = hdul[0].data
    
    height, width = img_data.shape
    
    step1_data = np.loadtxt(step1_csv, delimiter=',', skiprows=1)
    step1_img_x = step1_data[:, 0]
    step1_img_y = step1_data[:, 1]
    step1_cat_x = step1_data[:, 2]
    step1_cat_y = step1_data[:, 3]
    step1_pred_x = step1_data[:, 4]
    step1_pred_y = step1_data[:, 5]
    
    step2_data = np.loadtxt(step2_csv, delimiter=',', skiprows=1)
    valid_mask = step2_data[:, 6] == 1
    step2_valid = step2_data[valid_mask]
    step2_img_x = step2_valid[:, 0]
    step2_img_y = step2_valid[:, 1]
    step2_cat_x = step2_valid[:, 2]
    step2_cat_y = step2_valid[:, 3]
    
    img_min, img_max = np.percentile(img_data, [0.5, 99.5])
    stretched = np.clip((img_data - img_min) / (img_max - img_min), 0, 1)
    rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float32)
    
    fig = plt.figure(figsize=(width/100, height/100), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, origin='lower', extent=[0, width, 0, height])
    
    for i in range(len(step1_data)):
        ax.plot([step1_img_x[i], step1_pred_x[i]], [step1_img_y[i], step1_pred_y[i]], 
                'cyan', linewidth=0.5, alpha=0.7)
    ax.scatter(step1_img_x, step1_img_y, c='cyan', s=10, marker='+', linewidths=0.5)
    ax.scatter(step1_pred_x, step1_pred_y, c='cyan', s=10, marker='o', facecolors='none', linewidths=0.5)
    
    for i in range(len(step2_valid)):
        ax.plot([step2_img_x[i], step2_cat_x[i]], [step2_img_y[i], step2_cat_y[i]], 
                'yellow', linewidth=0.5, alpha=0.8)
    ax.scatter(step2_img_x, step2_img_y, c='yellow', s=10, marker='+', linewidths=0.5)
    ax.scatter(step2_cat_x, step2_cat_y, c='yellow', s=10, marker='o', facecolors='none', linewidths=0.5)
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    return width, height

def analyze_step1_step2_residuals(
    step1_csv: str,
    step2_csv: str,
) -> dict:
    """
    分析Step1和Step2的残差
    
    参数:
        step1_csv: Step1粗匹配预测CSV
        step2_csv: Step2控制点CSV
    
    返回:
        包含残差统计信息的字典
    """
    step1_data = np.loadtxt(step1_csv, delimiter=',', skiprows=1)
    step1_res_x = step1_data[:, 4] - step1_data[:, 0]
    step1_res_y = step1_data[:, 5] - step1_data[:, 1]
    step1_total_res = np.sqrt(step1_res_x**2 + step1_res_y**2)
    
    step2_data = np.loadtxt(step2_csv, delimiter=',', skiprows=1)
    valid_mask = step2_data[:, 6] == 1
    step2_valid = step2_data[valid_mask]
    step2_res_x = step2_valid[:, 4]
    step2_res_y = step2_valid[:, 5]
    step2_total_res = np.sqrt(step2_res_x**2 + step2_res_y**2)
    
    return {
        'step1_n': len(step1_data),
        'step1_res_x_mean': np.mean(step1_res_x),
        'step1_res_x_std': np.std(step1_res_x),
        'step1_res_y_mean': np.mean(step1_res_y),
        'step1_res_y_std': np.std(step1_res_y),
        'step1_total_mean': np.mean(step1_total_res),
        'step1_total_std': np.std(step1_total_res),
        'step2_n': len(step2_valid),
        'step2_res_x_mean': np.mean(step2_res_x),
        'step2_res_x_std': np.std(step2_res_x),
        'step2_res_y_mean': np.mean(step2_res_y),
        'step2_res_y_std': np.std(step2_res_y),
        'step2_total_mean': np.mean(step2_total_res),
        'step2_total_std': np.std(step2_total_res),
    }

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PlateSolve Debug Visualization')
    parser.add_argument('--fits', required=True, help='FITS image path')
    parser.add_argument('--step1', help='Step1 predictions CSV')
    parser.add_argument('--step2', help='Step2 control points CSV')
    parser.add_argument('--gaia', help='Gaia predictions CSV')
    parser.add_argument('--output', required=True, help='Output PNG path')
    parser.add_argument('--mode', choices=['alignment', 'comparison'], default='comparison',
                        help='Visualization mode')
    
    args = parser.parse_args()
    
    if args.mode == 'comparison' and args.step1 and args.step2:
        w, h = visualize_step1_step2_comparison(args.fits, args.step1, args.step2, args.output)
        print(f"Saved: {args.output}")
        print(f"Image size: {w}x{h} pixels")
        
        stats = analyze_step1_step2_residuals(args.step1, args.step2)
        print(f"\nStep1 residuals:")
        print(f"  N: {stats['step1_n']}")
        print(f"  X: mean={stats['step1_res_x_mean']:.3f}, std={stats['step1_res_x_std']:.3f}")
        print(f"  Y: mean={stats['step1_res_y_mean']:.3f}, std={stats['step1_res_y_std']:.3f}")
        print(f"  Total: mean={stats['step1_total_mean']:.3f}, std={stats['step1_total_std']:.3f}")
        
        print(f"\nStep2 residuals:")
        print(f"  N: {stats['step2_n']}")
        print(f"  X: mean={stats['step2_res_x_mean']:.3f}, std={stats['step2_res_x_std']:.3f}")
        print(f"  Y: mean={stats['step2_res_y_mean']:.3f}, std={stats['step2_res_y_std']:.3f}")
        print(f"  Total: mean={stats['step2_total_mean']:.3f}, std={stats['step2_total_std']:.3f}")
    
    elif args.mode == 'alignment' and args.gaia and args.step2:
        w, h = visualize_alignment(args.fits, args.gaia, args.step2, args.output)
        print(f"Saved: {args.output}")
        print(f"Image size: {w}x{h} pixels")
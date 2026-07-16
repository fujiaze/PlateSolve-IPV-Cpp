"""
从panel1_debug_data.json生成调试报告
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUTPUT_DIR = r"output"

with open(os.path.join(PROJECT_ROOT, OUTPUT_DIR, "panel1_debug_data.json"), 'r', encoding='utf-8') as f:
    results = json.load(f)

report_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, "panel1_debug_report.txt")

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("Panel1 批量解析调试报告\n")
    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"帧数: {len(results)}\n")
    f.write("=" * 100 + "\n\n")

    # ---- 1. 总体统计 ----
    n_success = sum(1 for r in results if r['success'])
    n_fail = len(results) - n_success
    f.write("## 1. 总体统计\n\n")
    f.write(f"  总帧数: {len(results)}\n")
    f.write(f"  成功: {n_success} ({n_success/len(results)*100:.1f}%)\n")
    f.write(f"  失败: {n_fail} ({n_fail/len(results)*100:.1f}%)\n\n")

    # ---- 2. 按滤镜统计 ----
    f.write("## 2. 按滤镜统计\n\n")
    f.write(f"  {'滤镜':<10} {'总数':>5} {'成功':>5} {'成功率':>7} {'RMS中位':>9} {'RMS均值':>9} {'匹配中位':>8} {'饱和中位':>8}\n")
    f.write(f"  {'-'*10} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*9} {'-'*8} {'-'*8}\n")

    for filt in ["Red", "Green", "Blue", "H-alpha", "Oiii"]:
        fr = [r for r in results if r['filter'] == filt]
        if not fr:
            continue
        fs = [r for r in fr if r['success']]
        n_total = len(fr)
        n_ok = len(fs)
        rate = n_ok / n_total * 100 if n_total > 0 else 0
        rms_vals = [r['rms_px'] for r in fs]
        rms_med = np.median(rms_vals) if rms_vals else 0
        rms_avg = np.mean(rms_vals) if rms_vals else 0
        match_vals = [r['matched'] for r in fs]
        match_med = int(np.median(match_vals)) if match_vals else 0
        sat_vals = [r['n_sat'] for r in fr]
        sat_med = int(np.median(sat_vals)) if sat_vals else 0
        f.write(f"  {filt:<10} {n_total:>5} {n_ok:>5} {rate:>6.1f}% {rms_med:>8.3f}px {rms_avg:>8.3f}px {match_med:>8} {sat_med:>8}\n")

    f.write("\n")

    # ---- 3. 耗时分布统计 ----
    f.write("## 3. 耗时分布统计\n\n")
    f.write(f"  {'滤镜':<10} {'星检测':>8} {'Gaia查询':>9} {'RANSAC':>8} {'SVD':>8} {'中心修正':>9} {'solve总计':>9}\n")
    f.write(f"  {'-'*10} {'-'*8} {'-'*9} {'-'*8} {'-'*8} {'-'*9} {'-'*9}\n")

    for filt in ["Red", "Green", "Blue", "H-alpha", "Oiii"]:
        fr = [r for r in results if r['filter'] == filt]
        if not fr:
            continue
        t_det = np.median([r['t_detect'] for r in fr])
        t_gaia = np.median([r['timings'].get('gaia_query', 0) for r in fr])
        t_ransac = np.median([r['timings'].get('ransac', 0) for r in fr])
        t_svd = np.median([r['timings'].get('svd', 0) for r in fr])
        t_conv = np.median([r['timings'].get('converge', 0) for r in fr])
        t_solve = np.median([r['t_solve'] for r in fr])
        f.write(f"  {filt:<10} {t_det:>7.2f}s {t_gaia:>8.2f}s {t_ransac:>7.2f}s {t_svd:>7.3f}s {t_conv:>8.2f}s {t_solve:>8.2f}s\n")

    f.write("\n")

    # ---- 4. 逐帧详细数据 ----
    f.write("## 4. 逐帧详细数据\n\n")
    f.write(f"  {'#':>3} {'滤镜':<7} {'星数':>6} {'饱和':>5} {'Gaia':>5} {'成功':>4} "
            f"{'RMS':>7} {'匹配':>5} {'s':>8} {'θ°':>7} {'flip':>4} "
            f"{'检测':>6} {'Gaia':>6} {'RANSAC':>7} {'SVD':>6} {'修正':>6} {'solve':>7}\n")
    f.write(f"  {'-'*3} {'-'*7} {'-'*6} {'-'*5} {'-'*5} {'-'*4} "
            f"{'-'*7} {'-'*5} {'-'*8} {'-'*7} {'-'*4} "
            f"{'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*7}\n")

    for i, r in enumerate(results):
        success = "OK" if r['success'] else "FAIL"
        rms_str = f"{r['rms_px']:.3f}" if r['success'] else "---"
        match_str = f"{r['matched']}" if r['success'] else "---"
        scale_str = f"{r.get('scale',0):.4f}" if r['success'] else "---"
        rot_str = f"{r.get('rotation',0):.2f}" if r['success'] else "---"
        flip_str = f"{r.get('flip',-1)}" if r['success'] else "---"

        t_det = r['t_detect']
        t_gaia = r['timings'].get('gaia_query', 0)
        t_ransac = r['timings'].get('ransac', 0)
        t_svd = r['timings'].get('svd', 0)
        t_conv = r['timings'].get('converge', 0)
        t_solve = r['t_solve']

        f.write(f"  {i+1:>3} {r['filter']:<7} {r['n_stars']:>6} {r['n_sat']:>5} {r['variables'].get('M',0):>5} {success:>4} "
                f"{rms_str:>7} {match_str:>5} {scale_str:>8} {rot_str:>7} {flip_str:>4} "
                f"{t_det:>5.2f}s {t_gaia:>5.2f}s {t_ransac:>6.2f}s {t_svd:>5.3f}s {t_conv:>5.2f}s {t_solve:>6.2f}s\n")

    f.write("\n")

    # ---- 5. 失败帧分析 ----
    failed = [r for r in results if not r['success']]
    if failed:
        f.write("## 5. 失败帧分析\n\n")
        for r in failed:
            reason = r['variables'].get('fail_reason', 'unknown')
            f.write(f"  {r['filename']} ({r['filter']}): {reason}\n")
            f.write(f"    星数={r['n_stars']}, 饱和={r['n_sat']}, Gaia={r['variables'].get('M',0)}\n")
            for md in r.get('mode_details', []):
                f.write(f"    模式{md['mode']}: {md.get('status','?')} n={md.get('n_inliers',0)}\n")
        f.write("\n")

    # ---- 6. 计算开销分析 ----
    f.write("## 6. 计算开销分析\n\n")

    all_t_detect = [r['t_detect'] for r in results]
    all_t_solve = [r['t_solve'] for r in results]
    all_t_gaia = [r['timings'].get('gaia_query', 0) for r in results]
    all_t_ransac = [r['timings'].get('ransac', 0) for r in results]
    all_t_svd = [r['timings'].get('svd', 0) for r in results]
    all_t_conv = [r['timings'].get('converge', 0) for r in results]

    total_detect = sum(all_t_detect)
    total_gaia = sum(all_t_gaia)
    total_ransac = sum(all_t_ransac)
    total_svd = sum(all_t_svd)
    total_conv = sum(all_t_conv)
    total_solve = sum(all_t_solve)
    grand_total = total_detect + total_solve

    f.write(f"  总耗时分布:\n")
    f.write(f"    星检测:       {total_detect:>8.1f}s ({total_detect/grand_total*100:>5.1f}%)\n")
    f.write(f"    Gaia查询:     {total_gaia:>8.1f}s ({total_gaia/grand_total*100:>5.1f}%)\n")
    f.write(f"    RANSAC:       {total_ransac:>8.1f}s ({total_ransac/grand_total*100:>5.1f}%)\n")
    f.write(f"    SVD精修:      {total_svd:>8.1f}s ({total_svd/grand_total*100:>5.1f}%)\n")
    f.write(f"    中心修正+精修:{total_conv:>8.1f}s ({total_conv/grand_total*100:>5.1f}%)\n")
    f.write(f"    solve其他:    {total_solve-total_gaia-total_ransac-total_svd-total_conv:>8.1f}s\n")
    f.write(f"    ────────────────────────\n")
    f.write(f"    总计:         {grand_total:>8.1f}s\n\n")

    f.write(f"  单帧中位耗时:\n")
    f.write(f"    星检测:       {np.median(all_t_detect):>8.2f}s\n")
    f.write(f"    Gaia查询:     {np.median(all_t_gaia):>8.2f}s\n")
    f.write(f"    RANSAC:       {np.median(all_t_ransac):>8.2f}s\n")
    f.write(f"    SVD精修:      {np.median(all_t_svd):>8.3f}s\n")
    f.write(f"    中心修正+精修:{np.median(all_t_conv):>8.2f}s\n")
    f.write(f"    solve总计:    {np.median(all_t_solve):>8.2f}s\n\n")

    # ---- 7. 关键变量统计 ----
    f.write("## 7. 关键变量统计\n\n")

    success_results = [r for r in results if r['success']]
    if success_results:
        scales = [r['scale'] for r in success_results]
        rotations = [r['rotation'] for r in success_results]
        rms_vals = [r['rms_px'] for r in success_results]
        matched = [r['matched'] for r in success_results]
        delta_ras = [abs(r['delta_ra']) for r in success_results]
        delta_decs = [abs(r['delta_dec']) for r in success_results]

        f.write(f"  像素尺度 (角秒/像素):\n")
        f.write(f"    范围: [{min(scales):.4f}, {max(scales):.4f}]\n")
        f.write(f"    中位: {np.median(scales):.4f}\n")
        f.write(f"    均值: {np.mean(scales):.4f} ± {np.std(scales):.4f}\n\n")

        f.write(f"  旋转角 (度):\n")
        f.write(f"    范围: [{min(rotations):.2f}, {max(rotations):.2f}]\n")
        f.write(f"    中位: {np.median(rotations):.2f}\n\n")

        f.write(f"  RMS (像素):\n")
        f.write(f"    范围: [{min(rms_vals):.3f}, {max(rms_vals):.3f}]\n")
        f.write(f"    中位: {np.median(rms_vals):.3f}\n")
        f.write(f"    均值: {np.mean(rms_vals):.3f}\n\n")

        f.write(f"  匹配星数:\n")
        f.write(f"    范围: [{min(matched)}, {max(matched)}]\n")
        f.write(f"    中位: {int(np.median(matched))}\n\n")

        f.write(f"  中心偏移 (度):\n")
        f.write(f"    ΔRA:  中位={np.median(delta_ras):.6f}° 均值={np.mean(delta_ras):.6f}°\n")
        f.write(f"    ΔDec: 中位={np.median(delta_decs):.6f}° 均值={np.mean(delta_decs):.6f}°\n\n")

    # ---- 8. 翻转模式分布 ----
    f.write("## 8. 翻转模式分布\n\n")
    for mode in range(4):
        cnt = sum(1 for r in success_results if r.get('flip') == mode)
        f.write(f"  模式{mode}: {cnt}帧 ({cnt/len(success_results)*100:.1f}%)\n")
    f.write("\n")

print(f"报告已保存: {report_path}")

"""
Panel1 批量解析 - 详细调试数据

记录每帧的:
  1. 单步计时 (星检测/Gaia查询/4模式RANSAC/4模式SVD/中心修正+精修)
  2. 关键变量 (星数/饱和星/Gaia星数/各模式s/theta/n/rms/score)
  3. 准确率 (RMS/匹配数/成功率)
  4. 计算开销 (各步骤耗时占比)

输出: output/panel1_debug_report.txt
"""

import os
import sys
import time
import logging
import math
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import (
    VectorMatch as VectorMatchV2, VectorMatchResult,
    _build_image_vectors, bisection_mag_limit, _build_catalog_vectors,
    _apply_flip, _ransac_rigid_v2, _find_fine_correspondences,
    _iterative_svd_refine, _compute_normalized_score, _apply_similarity,
    GaiaClientPy,
)
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree

PANEL1_DIR = r"testdata\lights\panel1"
GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUTPUT_DIR = r"output"
LOG_DIR = r"output\logs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def extract_filter(filename):
    """从文件名提取滤镜类型"""
    for f in ["H-alpha", "Oiii", "Red", "Green", "Blue"]:
        if f in filename:
            return f
    return "Unknown"


def get_frame_params(frame_path):
    """读取FITS并提取参数"""
    reader = ImageReader()
    img = reader.read(frame_path)
    width = img.width
    height = img.height

    center_ra = 0.0; center_dec = 0.0
    focal_length = 200.0; pixel_size = 6.0

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

    return img, width, height, center_ra, center_dec, focal_length, pixel_size


def solve_with_debug(vm, img_x, img_y, img_flux, img_saturated,
                     center_ra, center_dec, focal_length, pixel_size, width, height):
    """带详细计时的V2 solve"""
    timings = {}
    variables = {}
    mode_details = []

    # Step 1: s0和FOV
    t0 = time.perf_counter()
    s0 = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0
    timings['fov_calc'] = time.perf_counter() - t0
    variables['s0'] = s0
    variables['fov_diag'] = fov_diag
    variables['radius_deg'] = radius_deg

    # Step 2: 构建图像向量
    t0 = time.perf_counter()
    U, N_img, n_sat, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_saturated, s0, width, height)
    timings['build_vectors'] = time.perf_counter() - t0
    variables['N_img'] = N_img
    variables['n_sat'] = n_sat
    variables['sparsity'] = sparsity

    if N_img < 2:
        return None, timings, variables, mode_details

    # Step 3: Gaia查询
    if n_sat >= 50:
        N_gaia = math.ceil(1.5 * n_sat)
    else:
        N_gaia = 150
    variables['N_gaia_target'] = N_gaia

    t0 = time.perf_counter()
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        vm._gaia, center_ra, center_dec, radius_deg, N_gaia)
    timings['gaia_query'] = time.perf_counter() - t0
    variables['mag_limit'] = mag_limit
    variables['M'] = M

    if M < 2:
        return None, timings, variables, mode_details

    # RANSAC参数
    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_img * 0.2))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1
    variables['tau_coarse'] = tau_coarse
    variables['min_inliers'] = min_inliers

    # Step 4: 4种翻转模式
    best_mode = -1
    best_norm_score = -1.0
    best_result = None

    t_ransac_total = 0
    t_svd_total = 0
    t_build_w_total = 0

    for mode in range(4):
        mode_info = {'mode': mode}

        # 构建星表向量
        t0 = time.perf_counter()
        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
        Wf = _apply_flip(W, mode)
        t_bw = time.perf_counter() - t0
        t_build_w_total += t_bw
        mode_info['build_w_time'] = t_bw

        # RANSAC
        t0 = time.perf_counter()
        s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
            U, Wf, tau_coarse, K, min_inliers, vm._rng,
            candidate_radius_coarse, sparsity)
        t_r = time.perf_counter() - t0
        t_ransac_total += t_r
        mode_info['ransac_time'] = t_r
        mode_info['s'] = s
        mode_info['theta_deg'] = math.degrees(theta)
        mode_info['tx'] = tx
        mode_info['ty'] = ty
        mode_info['n_inliers'] = n_inliers
        mode_info['rms'] = rms

        if n_inliers < min_inliers:
            mode_info['status'] = 'skip_insufficient'
            mode_details.append(mode_info)
            continue

        # 精细候选对
        t0 = time.perf_counter()
        pairs_fine = _find_fine_correspondences(U, Wf, s, theta, tx, ty, tau_coarse)
        t_fine = time.perf_counter() - t0
        mode_info['fine_pairs'] = len(pairs_fine)
        mode_info['fine_time'] = t_fine

        # SVD精修
        t0 = time.perf_counter()
        s_ref, theta_ref, tx_ref, ty_ref, n_ref, rms_ref, mask_ref = \
            _iterative_svd_refine(U, Wf, inlier_mask, s0, s, theta, tx, ty, max_iter=10)
        t_svd = time.perf_counter() - t0
        t_svd_total += t_svd
        mode_info['svd_time'] = t_svd

        if n_ref >= min_inliers:
            s, theta, tx, ty = s_ref, theta_ref, tx_ref, ty_ref
            n_inliers, rms, inlier_mask = n_ref, rms_ref, mask_ref
            mode_info['svd_refined'] = True
        else:
            mode_info['svd_refined'] = False

        mode_info['s_final'] = s
        mode_info['theta_final_deg'] = math.degrees(theta)
        mode_info['n_final'] = n_inliers
        mode_info['rms_final'] = rms

        norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)
        mode_info['norm_score'] = norm_score
        mode_info['status'] = 'ok'

        if norm_score > best_norm_score:
            best_norm_score = norm_score
            best_mode = mode
            best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

        mode_details.append(mode_info)

    timings['build_w'] = t_build_w_total
    timings['ransac'] = t_ransac_total
    timings['svd'] = t_svd_total
    variables['best_mode'] = best_mode
    variables['best_norm_score'] = best_norm_score

    if best_mode < 0 or best_norm_score < 0.10:
        variables['fail_reason'] = f'best_score={best_norm_score:.4f}'
        return None, timings, variables, mode_details

    s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result

    # s范围限制
    if s < 0.9 or s > 1.1:
        variables['fail_reason'] = f's={s:.4f} out of [0.9,1.1]'
        return None, timings, variables, mode_details

    # Step 5: 中心修正+精修
    t0 = time.perf_counter()
    result = vm._extract_wcs_and_converge(
        s, theta, tx, ty, best_mode, s0,
        center_ra, center_dec, width, height,
        U, Wf, inlier_mask, N_img, M,
        cat_ra, cat_dec, cat_mag,
        fov_diag, sparsity,
    )
    timings['converge'] = time.perf_counter() - t0

    return result, timings, variables, mode_details


def main():
    # 设置日志: 文件+控制台
    log_file = os.path.join(LOG_DIR, f"panel1_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    # 扫描panel1所有帧
    panel1_dir = os.path.join(PROJECT_ROOT, PANEL1_DIR)
    files = sorted([f for f in os.listdir(panel1_dir) if f.endswith('.fts')])
    print(f"Panel1: {len(files)}帧")

    # 创建VectorMatch实例
    vm = VectorMatchV2(GAIA_DATA_DIR, db_type=0)

    # 结果收集
    results = []
    t_total_start = time.perf_counter()

    for i, fname in enumerate(files):
        frame_path = os.path.join(panel1_dir, fname)
        filt = extract_filter(fname)
        print(f"\n[{i+1}/{len(files)}] {fname} ({filt})")

        # 读取FITS
        t0 = time.perf_counter()
        img, width, height, center_ra, center_dec, focal_length, pixel_size = get_frame_params(frame_path)
        t_read = time.perf_counter() - t0

        # 星检测
        t0 = time.perf_counter()
        params = SDetParamsPy(fitRadius=0)
        detector = StarDetector(params=params)
        det_result = detector.detect_ex(img.data)
        t_detect = time.perf_counter() - t0

        img_x = np.array(det_result.x, dtype=np.float64)
        img_y = np.array(det_result.y, dtype=np.float64)
        img_flux = np.array(det_result.flux, dtype=np.float64)
        img_saturated = np.array(det_result.saturated, dtype=np.int32)
        n_stars = det_result.count
        n_sat = int(np.sum(det_result.saturated))

        # 带调试的solve
        t0 = time.perf_counter()
        result, timings, variables, mode_details = solve_with_debug(
            vm, img_x, img_y, img_flux, img_saturated,
            center_ra, center_dec, focal_length, pixel_size, width, height)
        t_solve = time.perf_counter() - t0

        # 汇总
        record = {
            'filename': fname,
            'filter': filt,
            'n_stars': n_stars,
            'n_sat': n_sat,
            't_read': t_read,
            't_detect': t_detect,
            't_solve': t_solve,
            'timings': timings,
            'variables': variables,
            'mode_details': mode_details,
            'success': result is not None,
        }
        if result:
            record['rms_px'] = result.rms_px
            record['rms_arcsec'] = result.rms_arcsec
            record['matched'] = result.matched_count
            record['scale'] = result.scale_arcsec_px
            record['rotation'] = result.rotation_deg
            record['flip'] = result.flip_mode
            record['center_ra'] = result.center_ra
            record['center_dec'] = result.center_dec
            record['delta_ra'] = result.center_ra - center_ra
            record['delta_dec'] = result.center_dec - center_dec

        results.append(record)

        # 实时打印
        if result:
            print(f"  OK: RMS={result.rms_px:.3f}px ({result.rms_arcsec:.3f}\"), "
                  f"matched={result.matched_count}, s={result.scale_arcsec_px:.4f}\"/px, "
                  f"θ={result.rotation_deg:.2f}°, flip={result.flip_mode}")
        else:
            reason = variables.get('fail_reason', 'unknown')
            print(f"  FAIL: {reason}")

        print(f"  耗时: read={t_read:.2f}s detect={t_detect:.2f}s solve={t_solve:.2f}s")
        print(f"  星: {n_stars}颗(饱和{n_sat}), Gaia={variables.get('M',0)}颗")

        # 各步骤耗时
        if timings:
            t_gaia = timings.get('gaia_query', 0)
            t_ransac = timings.get('ransac', 0)
            t_svd = timings.get('svd', 0)
            t_converge = timings.get('converge', 0)
            print(f"  solve分解: Gaia={t_gaia:.2f}s RANSAC={t_ransac:.2f}s SVD={t_svd:.3f}s converge={t_converge:.2f}s")

        # 各模式结果
        for md in mode_details:
            status = md.get('status', '?')
            if status == 'ok':
                print(f"    模式{md['mode']}: s={md.get('s_final',0):.4f} θ={md.get('theta_final_deg',0):.2f}° "
                      f"n={md.get('n_final',0)} rms={md.get('rms_final',0):.3f} score={md.get('norm_score',0):.4f} "
                      f"RANSAC={md.get('ransac_time',0):.2f}s SVD={md.get('svd_time',0):.3f}s")
            else:
                print(f"    模式{md['mode']}: {status} (n={md.get('n_inliers',0)})")

    t_total = time.perf_counter() - t_total_start
    vm.close()

    # ============================================================
    # 保存原始数据到JSON
    # ============================================================
    json_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, "panel1_debug_data.json")
    def convert_obj(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_obj(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_obj(i) for i in obj]
        return obj
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(convert_obj(results), jf, ensure_ascii=False, indent=2)
    print(f"原始数据已保存: {json_path}")

    # ============================================================
    # 生成报告
    # ============================================================
    report_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR, "panel1_debug_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("Panel1 批量解析调试报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {t_total:.1f}s ({len(files)}帧)\n")
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
        f.write(f"  {'滤镜':<10} {'总数':>5} {'成功':>5} {'成功率':>7} {'RMS中位':>9} {'RMS均值':>9} {'匹配中位':>8}\n")
        f.write(f"  {'-'*10} {'-'*5} {'-'*5} {'-'*7} {'-'*9} {'-'*9} {'-'*8}\n")

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
            f.write(f"  {filt:<10} {n_total:>5} {n_ok:>5} {rate:>6.1f}% {rms_med:>8.3f}px {rms_avg:>8.3f}px {match_med:>8}\n")

        f.write("\n")

        # ---- 3. 耗时分布统计 ----
        f.write("## 3. 耗时分布统计\n\n")

        # 按滤镜统计各步骤耗时
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
            scale_str = f"{r.get('scale_arcsec_px',0):.4f}" if r['success'] else "---"
            rot_str = f"{r.get('rotation_deg',0):.2f}" if r['success'] else "---"
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
                for md in r['mode_details']:
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
            scales = [r['scale_arcsec_px'] for r in success_results]
            rotations = [r['rotation_deg'] for r in success_results]
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

    print(f"\n报告已保存: {report_path}")
    print(f"日志已保存: {log_file}")
    print(f"总耗时: {t_total:.1f}s")


if __name__ == '__main__':
    main()

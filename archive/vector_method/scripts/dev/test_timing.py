"""
V3 计时测试 - 分析各阶段耗时

选取代表性帧(不同滤镜/星密度), 详细计时每个阶段:
  1. 星检测(sdet)
  2. V2 vs V3 solve总耗时
  3. RANSAC内部耗时分析(采样/剪枝/变换/内点统计)
  4. 内点统计微基准(V2无方向 vs V3有方向)
"""

import os
import sys
import time
import logging
import math
import numpy as np

# 正确的导入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2
from vector_match_v3 import VectorMatch as VectorMatchV3
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# 测试帧选择: 不同滤镜、不同星密度
TEST_FRAMES = [
    # Red帧 - 星多, 应该最快
    r"testdata\lights\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",
    # Blue帧 - 星少, 可能最慢
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@061821-180S-Blue.fts",
    # Green帧 - 中等
    r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@010234-180S-Green.fts",
    # H-alpha帧 - 窄带, 星少
    r"testdata\lights\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@012420-300S-H-alpha.fts",
    # Oiii帧 - 窄带, 星最少
    r"testdata\lights\panel3\Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@025717-600S-Oiii.fts",
]

GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def detect_stars(img_data):
    """星检测"""
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img_data)
    return det_result


def run_solve_timing(frame_path):
    """对单帧运行V2和V3 solve, 计时"""
    full_path = os.path.join(PROJECT_ROOT, frame_path)
    fname = os.path.basename(frame_path)
    print(f"\n{'='*80}")
    print(f"帧: {fname}")
    print(f"{'='*80}")

    # 1. 读取FITS
    reader = ImageReader()
    img = reader.read(full_path)
    width = img.width
    height = img.height

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

    print(f"  图像: {width}x{height}, RA={center_ra:.4f}, Dec={center_dec:.4f}")
    print(f"  焦距={focal_length}mm, 像素={pixel_size}um")

    # 2. 星检测
    t0 = time.perf_counter()
    det_result = detect_stars(img.data)
    t_detect = time.perf_counter() - t0
    n_stars = det_result.count
    n_sat = int(np.sum(det_result.saturated))
    print(f"  星检测: {t_detect:.2f}s, {n_stars}颗(饱和{n_sat})")

    if n_stars < 10:
        print("  星数不足, 跳过")
        return None

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)

    # 3. V2 solve
    print(f"\n  --- V2 ---")
    v2 = VectorMatchV2(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    result_v2 = v2.solve(img_x, img_y, img_flux, img_saturated,
                         center_ra, center_dec, focal_length, pixel_size, width, height)
    t_v2 = time.perf_counter() - t0
    v2_info = f"RMS={result_v2.rms_px:.3f}px, matched={result_v2.matched_count}" if result_v2 else "失败"
    print(f"  V2 耗时: {t_v2:.2f}s, {v2_info}")

    # 4. V3 solve
    print(f"\n  --- V3 ---")
    v3 = VectorMatchV3(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    result_v3 = v3.solve(img_x, img_y, img_flux, img_saturated,
                         center_ra, center_dec, focal_length, pixel_size, width, height)
    t_v3 = time.perf_counter() - t0
    v3_info = f"RMS={result_v3.rms_px:.3f}px, matched={result_v3.matched_count}" if result_v3 else "失败"
    print(f"  V3 耗时: {t_v3:.2f}s, {v3_info}")
    print(f"  V3/V2 耗时比: {t_v3/t_v2:.2f}x")

    return {
        'fname': fname,
        't_detect': t_detect,
        't_v2': t_v2,
        't_v3': t_v3,
        'n_stars': n_stars,
        'n_sat': n_sat,
    }


def run_ransac_breakdown(frame_path):
    """对单帧进行RANSAC内部耗时分解"""
    full_path = os.path.join(PROJECT_ROOT, frame_path)
    fname = os.path.basename(frame_path)
    print(f"\n{'='*80}")
    print(f"RANSAC耗时分解: {fname}")
    print(f"{'='*80}")

    # 读取FITS
    reader = ImageReader()
    img = reader.read(full_path)
    width = img.width
    height = img.height

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

    # 星检测
    det_result = detect_stars(img.data)
    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_saturated = np.array(det_result.saturated, dtype=np.int32)

    s0 = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0

    # 导入V3内部函数
    from vector_match_v3 import (
        _build_gold_and_validation_pools, bisection_mag_limit,
        _build_catalog_vectors, _apply_flip, _ransac_v3,
        _iterative_svd_refine, _compute_normalized_score,
        _count_inliers_1to1, _apply_similarity,
        _find_coarse_correspondences, _weighted_choice,
    )
    from vector_match_v2 import _count_inliers_1to1 as v2_count_inliers

    # 构建池
    U_gold, N_gold, n_sat_v3, sparsity, U_val, N_val = _build_gold_and_validation_pools(
        img_x, img_y, img_flux, img_saturated, s0, width, height, 1000)
    print(f"  黄金池: {N_gold}, 验证池: {N_val}, 饱和星: {n_sat_v3}")

    # Gaia查询
    v3_inst = VectorMatchV3(GAIA_DATA_DIR, db_type=0)
    N_gaia = math.ceil(1.5 * n_sat_v3) if n_sat_v3 >= 50 else 150
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        v3_inst._gaia, center_ra, center_dec, radius_deg, N_gaia)
    print(f"  Gaia星数: {M}, 极限星等: {mag_limit:.2f}")

    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_val * 0.03))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1
    fov_diag_arcsec = fov_diag * 3600.0

    # 对每个翻转模式计时RANSAC
    print(f"\n  --- 各模式RANSAC耗时 ---")
    for mode in range(4):
        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
        Wf = _apply_flip(W, mode)

        t0 = time.perf_counter()
        s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_v3(
            U_gold, U_val, Wf, tau_coarse, K, min_inliers, v3_inst._rng,
            candidate_radius_coarse, sparsity,
            sin_tau=0.2,
            fov_diag_arcsec=fov_diag_arcsec,
        )
        t_ransac = time.perf_counter() - t0
        status = f"n={n_inliers} rms={rms:.3f}" if n_inliers >= min_inliers else "skip"
        print(f"    模式{mode}: {t_ransac:.2f}s ({status})")

    # RANSAC内部耗时分解(模式0)
    print(f"\n  --- RANSAC内部耗时分解(模式0) ---")
    W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
    Wf = _apply_flip(W, 0)

    # 构建候选对
    pairs = _find_coarse_correspondences(U_gold, Wf, candidate_radius_coarse)
    P = len(pairs)
    print(f"  粗候选对: {P}")

    u_to_pair_indices = {}
    for p_idx in range(P):
        u_idx = int(pairs[p_idx, 0])
        if u_idx not in u_to_pair_indices:
            u_to_pair_indices[u_idx] = []
        u_to_pair_indices[u_idx].append(p_idx)

    unique_u = np.array(list(u_to_pair_indices.keys()), dtype=np.int64)
    u_weights = sparsity[unique_u]
    u_weights = np.maximum(u_weights, 1e-10)
    u_prob = u_weights / u_weights.sum()
    actual_K = min(K, len(unique_u) * (len(unique_u) - 1) // 2)
    max_tx_ty = 0.5 * fov_diag_arcsec

    rng = np.random.default_rng(42)
    sin_tau_coarse = 0.2

    # 阶段1: 采样+剪枝 (整体计时)
    t0 = time.perf_counter()
    valid_transforms = []
    n_pruned_norm = 0
    n_pruned_scale = 0
    n_pruned_angle = 0
    n_pruned_trans = 0

    for iter_idx in range(actual_K):
        sel = _weighted_choice(u_prob, rng, n=2)
        u_idx_a = int(unique_u[sel[0]])
        u_idx_b = int(unique_u[sel[1]])
        pairs_a = u_to_pair_indices[u_idx_a]
        pairs_b = u_to_pair_indices[u_idx_b]
        idx_a = int(pairs_a[rng.integers(len(pairs_a))])
        idx_b = int(pairs_b[rng.integers(len(pairs_b))])
        w_idx_a = int(pairs[idx_a, 1])
        w_idx_b = int(pairs[idx_b, 1])
        u_a, u_b = U_gold[u_idx_a], U_gold[u_idx_b]
        w_a, w_b = Wf[w_idx_a], Wf[w_idx_b]

        # 剪枝0: 单向量模长一致性
        norm_u_a = math.sqrt(u_a[0] ** 2 + u_a[1] ** 2)
        norm_w_a = math.sqrt(w_a[0] ** 2 + w_a[1] ** 2)
        norm_u_b = math.sqrt(u_b[0] ** 2 + u_b[1] ** 2)
        norm_w_b = math.sqrt(w_b[0] ** 2 + w_b[1] ** 2)
        if norm_w_a > 1e-10 and norm_w_b > 1e-10:
            if abs(norm_u_a / norm_w_a - 1.0) > 0.10 or abs(norm_u_b / norm_w_b - 1.0) > 0.10:
                n_pruned_norm += 1
                continue

        du = u_a - u_b
        dw = w_a - w_b
        norm_du = math.sqrt(du[0] ** 2 + du[1] ** 2)
        norm_dw = math.sqrt(dw[0] ** 2 + dw[1] ** 2)
        if norm_dw < 1e-12 or norm_du < 1e-12:
            continue

        s = norm_du / norm_dw
        if s < 0.9 or s > 1.1:
            n_pruned_scale += 1
            continue

        angle_du = math.atan2(du[1], du[0])
        angle_dw = math.atan2(dw[1], dw[0])
        angle_diff = abs(angle_du - angle_dw)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        if angle_diff > math.pi * 0.5:
            n_pruned_angle += 1
            continue

        theta = angle_du - angle_dw
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx = u_a[0] - s * (cos_t * w_a[0] - sin_t * w_a[1])
        ty = u_a[1] - s * (sin_t * w_a[0] + cos_t * w_a[1])
        if abs(tx) > max_tx_ty or abs(ty) > max_tx_ty:
            n_pruned_trans += 1
            continue

        valid_transforms.append((s, theta, tx, ty))

    t_sample_prune = time.perf_counter() - t0
    n_total_pruned = n_pruned_norm + n_pruned_scale + n_pruned_angle + n_pruned_trans
    print(f"  采样+剪枝: {t_sample_prune:.4f}s")
    print(f"    总迭代: {actual_K}, 通过: {len(valid_transforms)}, 剪枝: {n_total_pruned} ({n_total_pruned/max(actual_K,1)*100:.1f}%)")
    print(f"    模长剪枝: {n_pruned_norm}, 尺度剪枝: {n_pruned_scale}, 角度剪枝: {n_pruned_angle}, 平移剪枝: {n_pruned_trans}")

    # 阶段2: 变换投影 (批量计时)
    t0 = time.perf_counter()
    for s, theta, tx, ty in valid_transforms:
        Wt = _apply_similarity(Wf, s, theta, tx, ty)
    t_transform = time.perf_counter() - t0
    print(f"  变换投影: {t_transform:.4f}s ({len(valid_transforms)}次, {t_transform/max(len(valid_transforms),1)*1000:.3f}ms/次)")

    # 阶段3: 内点统计V3 (批量计时)
    t0 = time.perf_counter()
    for s, theta, tx, ty in valid_transforms:
        Wt = _apply_similarity(Wf, s, theta, tx, ty)
        n_inliers, rms, mask = _count_inliers_1to1(U_val, Wt, tau_coarse, sin_tau_coarse)
    t_inlier_v3 = time.perf_counter() - t0
    print(f"  内点统计V3(含变换+方向): {t_inlier_v3:.4f}s ({len(valid_transforms)}次, {t_inlier_v3/max(len(valid_transforms),1)*1000:.3f}ms/次)")

    # 阶段4: 内点统计V2 (批量计时, 无方向检验)
    t0 = time.perf_counter()
    for s, theta, tx, ty in valid_transforms:
        Wt = _apply_similarity(Wf, s, theta, tx, ty)
        n_inliers, rms, mask = v2_count_inliers(U_val, Wt, tau_coarse)
    t_inlier_v2 = time.perf_counter() - t0
    print(f"  内点统计V2(含变换+无方向): {t_inlier_v2:.4f}s ({len(valid_transforms)}次, {t_inlier_v2/max(len(valid_transforms),1)*1000:.3f}ms/次)")

    # 纯内点统计微基准(不做变换, 只比较_count_inliers_1to1)
    if len(valid_transforms) > 0:
        s, theta, tx, ty = valid_transforms[0]
        Wt = _apply_similarity(Wf, s, theta, tx, ty)

        N_BENCH = 200
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            n2, r2, m2 = v2_count_inliers(U_val, Wt, tau_coarse)
        t_v2_pure = (time.perf_counter() - t0) / N_BENCH

        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            n3, r3, m3 = _count_inliers_1to1(U_val, Wt, tau_coarse, sin_tau_coarse)
        t_v3_pure = (time.perf_counter() - t0) / N_BENCH

        print(f"\n  --- 纯内点统计微基准 ({N_BENCH}次平均) ---")
        print(f"    V2(无方向): {t_v2_pure*1000:.3f}ms/次, n={n2}")
        print(f"    V3(有方向): {t_v3_pure*1000:.3f}ms/次, n={n3}")
        print(f"    方向检验额外开销: {(t_v3_pure-t_v2_pure)*1000:.3f}ms ({(t_v3_pure/t_v2_pure-1)*100:.1f}%)")

    # 总结
    print(f"\n  --- RANSAC耗时总结(模式0, {actual_K}次迭代) ---")
    print(f"  采样+剪枝: {t_sample_prune:.4f}s")
    print(f"  变换投影:   {t_transform:.4f}s")
    print(f"  内点V3:     {t_inlier_v3:.4f}s")
    print(f"  内点V2:     {t_inlier_v2:.4f}s")
    print(f"  V3额外开销: {t_inlier_v3-t_inlier_v2:.4f}s ({(t_inlier_v3/t_inlier_v2-1)*100:.1f}%)")
    print(f"  剪枝节省: {n_total_pruned}次变换+内点计算 避免了")


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)

    # 第一部分: V2 vs V3 solve总耗时
    print("=" * 80)
    print("第一部分: V2 vs V3 solve总耗时对比")
    print("=" * 80)
    results = []
    for frame in TEST_FRAMES:
        try:
            r = run_solve_timing(frame)
            if r:
                results.append(r)
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    if results:
        print(f"\n{'='*80}")
        print("汇总")
        print(f"{'='*80}")
        print(f"{'帧名':<50} {'V2(s)':>8} {'V3(s)':>8} {'V3/V2':>8} {'星数':>6} {'饱和':>5}")
        for r in results:
            print(f"{r['fname']:<50} {r['t_v2']:>8.2f} {r['t_v3']:>8.2f} {r['t_v3']/r['t_v2']:>8.2f} {r['n_stars']:>6} {r['n_sat']:>5}")

    # 第二部分: RANSAC内部耗时分解
    print(f"\n{'='*80}")
    print("第二部分: RANSAC内部耗时分解")
    print(f"{'='*80}")
    for frame in TEST_FRAMES[:3]:  # 只测前3帧
        try:
            run_ransac_breakdown(frame)
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

"""
从V2正确结果中分析V3参数的最优值

统计:
  1. 单向量模长比 (ratio_a, ratio_b) 的分布 → 剪枝0阈值
  2. 点对距离比 (scale) 的分布 → 剪枝1阈值
  3. 点对角度差 (angle_diff) 的分布 → 剪枝2阈值
  4. 平移 (tx, ty) 的分布 → 剪枝3阈值
  5. 方向偏差 (sin_dtheta) 的分布 → sin_tau阈值
  6. 内点RMS分布 → tau阈值

用V2的匹配结果作为"真值"，反向计算这些参数
"""

import os, sys, time, math, logging
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import (
    VectorMatch as VectorMatchV2, VectorMatchResult,
    _build_image_vectors, bisection_mag_limit, _build_catalog_vectors,
    _apply_flip, _apply_similarity, _ransac_rigid_v2,
    _iterative_svd_refine, _compute_normalized_score, _count_inliers_1to1,
    GaiaClientPy,
)
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

PANEL1_DIR = r"testdata\lights\panel1"
GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

# 只取10帧做详细分析 (每种滤镜2帧)
TEST_FRAMES = [
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@062844-180S-Red.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@054249-180S-Green.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@055414-180S-Blue.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@061821-180S-Blue.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@062946-300S-H-alpha.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@064714-300S-H-alpha.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@065304-600S-Oiii.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@070354-600S-Oiii.fts",
    r"testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@005126-600S-Oiii.fts",
]


def get_frame_params(frame_path):
    # 确保使用绝对路径
    if not os.path.isabs(frame_path):
        frame_path = os.path.join(PROJECT_ROOT, frame_path)
    reader = ImageReader()
    img = reader.read(frame_path)
    width, height = img.width, img.height
    center_ra, center_dec = 0.0, 0.0
    focal_length, pixel_size = 200.0, 6.0

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

    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)

    return (np.array(det_result.x, dtype=np.float64),
            np.array(det_result.y, dtype=np.float64),
            np.array(det_result.flux, dtype=np.float64),
            np.array(det_result.saturated, dtype=np.int32),
            center_ra, center_dec, focal_length, pixel_size, width, height)


def analyze_frame(vm, img_x, img_y, img_flux, img_saturated,
                  center_ra, center_dec, focal_length, pixel_size, width, height):
    """对单帧做完整V2匹配，然后从匹配结果中分析参数分布"""
    s0 = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0

    U, N_img, n_sat, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_saturated, s0, width, height)

    if n_sat >= 50:
        N_gaia = math.ceil(1.5 * n_sat)
    else:
        N_gaia = 150

    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        vm._gaia, center_ra, center_dec, radius_deg, N_gaia)

    if M < 2:
        return None

    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_img * 0.2))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1

    # 4种翻转模式
    best_mode = -1
    best_norm_score = -1.0
    best_result = None

    for mode in range(4):
        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
        Wf = _apply_flip(W, mode)

        s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
            U, Wf, tau_coarse, K, min_inliers, vm._rng,
            candidate_radius_coarse, sparsity)

        if n_inliers < min_inliers:
            continue

        s_ref, theta_ref, tx_ref, ty_ref, n_ref, rms_ref, mask_ref = \
            _iterative_svd_refine(U, Wf, inlier_mask, s0, s, theta, tx, ty, max_iter=10)

        if n_ref >= min_inliers:
            s, theta, tx, ty = s_ref, theta_ref, tx_ref, ty_ref
            n_inliers, rms, inlier_mask = n_ref, rms_ref, mask_ref

        norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)

        if norm_score > best_norm_score:
            best_norm_score = norm_score
            best_mode = mode
            best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

    if best_mode < 0 or best_norm_score < 0.10:
        return None

    s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result

    if s < 0.9 or s > 1.1:
        return None

    # ============================================================
    # 从匹配结果中分析参数分布
    # ============================================================
    Wt = _apply_similarity(Wf, s, theta, tx, ty)
    tree = cKDTree(Wt)
    dists, idxs = tree.query(U, k=1)

    # 只分析内点
    inlier_U = U[inlier_mask]
    inlier_Wt = Wt[idxs[inlier_mask]]
    inlier_Wf = Wf[idxs[inlier_mask]]
    inlier_dists = dists[inlier_mask]

    stats = {}

    # 1. 内点距离分布 (RMS)
    diffs = inlier_U - inlier_Wt
    dist_arr = np.sqrt(np.sum(diffs ** 2, axis=1))
    stats['inlier_dist'] = {
        'mean': float(np.mean(dist_arr)),
        'median': float(np.median(dist_arr)),
        'p90': float(np.percentile(dist_arr, 90)),
        'p95': float(np.percentile(dist_arr, 95)),
        'p99': float(np.percentile(dist_arr, 99)),
        'max': float(np.max(dist_arr)),
    }

    # 2. 方向偏差分布 (sin_dtheta)
    # 对所有星对(U[i], Wt[idxs[i]])计算方向偏差
    all_U = U
    all_Wt = Wt[idxs]
    norm_U = np.sqrt(all_U[:, 0]**2 + all_U[:, 1]**2)
    norm_Wt = np.sqrt(all_Wt[:, 0]**2 + all_Wt[:, 1]**2)
    valid = (norm_U > 1e-10) & (norm_Wt > 1e-10)

    # sin_dtheta = |U x Wt| / (|U| * |Wt|)
    cross = np.abs(all_U[:, 0] * all_Wt[:, 1] - all_U[:, 1] * all_Wt[:, 0])
    sin_dtheta = np.zeros(len(all_U))
    sin_dtheta[valid] = cross[valid] / (norm_U[valid] * norm_Wt[valid])

    # 内点的方向偏差
    inlier_sin = sin_dtheta[inlier_mask]
    stats['sin_dtheta_inlier'] = {
        'mean': float(np.mean(inlier_sin)),
        'median': float(np.median(inlier_sin)),
        'p90': float(np.percentile(inlier_sin, 90)),
        'p95': float(np.percentile(inlier_sin, 95)),
        'p99': float(np.percentile(inlier_sin, 99)),
        'max': float(np.max(inlier_sin)),
        'n_above_0.2': int(np.sum(inlier_sin > 0.2)),
        'n_above_0.3': int(np.sum(inlier_sin > 0.3)),
        'n_above_0.5': int(np.sum(inlier_sin > 0.5)),
        'pct_above_0.2': float(np.mean(inlier_sin > 0.2) * 100),
    }

    # 非内点的方向偏差
    non_inlier_sin = sin_dtheta[~inlier_mask & valid]
    if len(non_inlier_sin) > 0:
        stats['sin_dtheta_non_inlier'] = {
            'mean': float(np.mean(non_inlier_sin)),
            'median': float(np.median(non_inlier_sin)),
            'p90': float(np.percentile(non_inlier_sin, 90)),
        }

    # 3. 从RANSAC采样角度分析剪枝参数
    # 重新构建粗候选对
    from vector_match_v2 import _find_coarse_correspondences
    pairs = _find_coarse_correspondences(U, Wf, candidate_radius_coarse)
    P = len(pairs)

    # 采样分析
    ratio_a_list = []
    ratio_b_list = []
    scale_list = []
    angle_diff_list = []
    tx_list = []
    ty_list = []

    # 随机采样1000对
    rng = np.random.default_rng(42)
    n_sample = min(1000, P * (P - 1) // 2)

    for _ in range(n_sample):
        p_idx_a = rng.integers(P)
        p_idx_b = rng.integers(P)
        if p_idx_a == p_idx_b:
            continue

        u_idx_a = int(pairs[p_idx_a, 0])
        u_idx_b = int(pairs[p_idx_b, 0])
        w_idx_a = int(pairs[p_idx_a, 1])
        w_idx_b = int(pairs[p_idx_b, 1])

        u_a, u_b = U[u_idx_a], U[u_idx_b]
        w_a, w_b = Wf[w_idx_a], Wf[w_idx_b]

        # 剪枝0: 模长比
        norm_u_a = math.sqrt(u_a[0]**2 + u_a[1]**2)
        norm_w_a = math.sqrt(w_a[0]**2 + w_a[1]**2)
        norm_u_b = math.sqrt(u_b[0]**2 + u_b[1]**2)
        norm_w_b = math.sqrt(w_b[0]**2 + w_b[1]**2)

        if norm_w_a > 1e-10:
            ratio_a_list.append(norm_u_a / norm_w_a)
        if norm_w_b > 1e-10:
            ratio_b_list.append(norm_u_b / norm_w_b)

        du = u_a - u_b
        dw = w_a - w_b
        norm_du = math.sqrt(du[0]**2 + du[1]**2)
        norm_dw = math.sqrt(dw[0]**2 + dw[1]**2)

        if norm_dw < 1e-12 or norm_du < 1e-12:
            continue

        # 剪枝1: 尺度
        s_pair = norm_du / norm_dw
        scale_list.append(s_pair)

        # 剪枝2: 角度差
        angle_du = math.atan2(du[1], du[0])
        angle_dw = math.atan2(dw[1], dw[0])
        angle_diff = abs(angle_du - angle_dw)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        angle_diff_list.append(math.degrees(angle_diff))

        # 剪枝3: 平移
        theta_pair = math.atan2(du[1], du[0]) - math.atan2(dw[1], dw[0])
        cos_t = math.cos(theta_pair)
        sin_t = math.sin(theta_pair)
        tx_pair = u_a[0] - s_pair * (cos_t * w_a[0] - sin_t * w_a[1])
        ty_pair = u_a[1] - s_pair * (sin_t * w_a[0] + cos_t * w_a[1])
        tx_list.append(tx_pair)
        ty_list.append(ty_pair)

    # 统计
    ratio_all = np.array(ratio_a_list + ratio_b_list)
    stats['ratio'] = {
        'mean': float(np.mean(ratio_all)),
        'median': float(np.median(ratio_all)),
        'std': float(np.std(ratio_all)),
        'p1': float(np.percentile(ratio_all, 1)),
        'p5': float(np.percentile(ratio_all, 5)),
        'p95': float(np.percentile(ratio_all, 95)),
        'p99': float(np.percentile(ratio_all, 99)),
        'pct_outside_0.10': float(np.mean(np.abs(ratio_all - 1.0) > 0.10) * 100),
        'pct_outside_0.15': float(np.mean(np.abs(ratio_all - 1.0) > 0.15) * 100),
        'pct_outside_0.20': float(np.mean(np.abs(ratio_all - 1.0) > 0.20) * 100),
    }

    scale_arr = np.array(scale_list)
    stats['scale'] = {
        'mean': float(np.mean(scale_arr)),
        'median': float(np.median(scale_arr)),
        'std': float(np.std(scale_arr)),
        'p1': float(np.percentile(scale_arr, 1)),
        'p5': float(np.percentile(scale_arr, 5)),
        'p95': float(np.percentile(scale_arr, 95)),
        'p99': float(np.percentile(scale_arr, 99)),
        'pct_outside_0.10': float(np.mean((scale_arr < 0.9) | (scale_arr > 1.1)) * 100),
        'pct_outside_0.15': float(np.mean((scale_arr < 0.85) | (scale_arr > 1.15)) * 100),
        'pct_outside_0.20': float(np.mean((scale_arr < 0.8) | (scale_arr > 1.2)) * 100),
    }

    angle_arr = np.array(angle_diff_list)
    stats['angle_diff'] = {
        'mean': float(np.mean(angle_arr)),
        'median': float(np.median(angle_arr)),
        'p90': float(np.percentile(angle_arr, 90)),
        'p95': float(np.percentile(angle_arr, 95)),
        'p99': float(np.percentile(angle_arr, 99)),
        'pct_above_90': float(np.mean(angle_arr > 90) * 100),
        'pct_above_60': float(np.mean(angle_arr > 60) * 100),
    }

    tx_arr = np.array(tx_list)
    ty_arr = np.array(ty_list)
    fov_diag_arcsec = fov_diag * 3600.0
    stats['translation'] = {
        'tx_mean': float(np.mean(tx_arr)),
        'tx_std': float(np.std(tx_arr)),
        'tx_p99': float(np.percentile(np.abs(tx_arr), 99)),
        'ty_mean': float(np.mean(ty_arr)),
        'ty_std': float(np.std(ty_arr)),
        'ty_p99': float(np.percentile(np.abs(ty_arr), 99)),
        'fov_diag_arcsec': fov_diag_arcsec,
        'pct_outside_0.5fov': float(np.mean((np.abs(tx_arr) > 0.5 * fov_diag_arcsec) |
                                              (np.abs(ty_arr) > 0.5 * fov_diag_arcsec)) * 100),
    }

    stats['match_info'] = {
        's': s, 'theta_deg': math.degrees(theta), 'tx': tx, 'ty': ty,
        'n_inliers': n_inliers, 'rms': rms, 'mode': best_mode,
        'N_img': N_img, 'M': M, 'n_sat': n_sat,
    }

    return stats


def main():
    logging.basicConfig(level=logging.WARNING)

    vm = VectorMatchV2(GAIA_DATA_DIR, db_type=0)

    all_stats = {}
    for i, frame_path in enumerate(TEST_FRAMES):
        fname = os.path.basename(frame_path)
        filt = "Unknown"
        for f in ["H-alpha", "Oiii", "Red", "Green", "Blue"]:
            if f in fname: filt = f; break

        print(f"\n[{i+1}/{len(TEST_FRAMES)}] {fname} ({filt})")

        img_x, img_y, img_flux, img_saturated, center_ra, center_dec, \
            focal_length, pixel_size, width, height = get_frame_params(frame_path)

        stats = analyze_frame(vm, img_x, img_y, img_flux, img_saturated,
                              center_ra, center_dec, focal_length, pixel_size, width, height)

        if stats:
            all_stats[fname] = stats
            mi = stats['match_info']
            print(f"  匹配: s={mi['s']:.4f} θ={mi['theta_deg']:.2f}° n={mi['n_inliers']} rms={mi['rms']:.3f}")
            print(f"  sin_dtheta内点: 中位={stats['sin_dtheta_inlier']['median']:.4f} "
                  f"P95={stats['sin_dtheta_inlier']['p95']:.4f} "
                  f"P99={stats['sin_dtheta_inlier']['p99']:.4f} "
                  f">0.2: {stats['sin_dtheta_inlier']['pct_above_0.2']:.1f}%")
            print(f"  模长比: 中位={stats['ratio']['median']:.4f} "
                  f"±10%外: {stats['ratio']['pct_outside_0.10']:.1f}% "
                  f"±15%外: {stats['ratio']['pct_outside_0.15']:.1f}%")
            print(f"  尺度: 中位={stats['scale']['median']:.4f} "
                  f"±10%外: {stats['scale']['pct_outside_0.10']:.1f}% "
                  f"±15%外: {stats['scale']['pct_outside_0.15']:.1f}%")
            print(f"  角度差: 中位={stats['angle_diff']['median']:.1f}° "
                  f">90°: {stats['angle_diff']['pct_above_90']:.1f}% "
                  f">60°: {stats['angle_diff']['pct_above_60']:.1f}%")
        else:
            print(f"  匹配失败")

    vm.close()

    # ============================================================
    # 汇总分析
    # ============================================================
    print(f"\n{'='*80}")
    print("参数分布汇总分析 (基于V2正确结果)")
    print(f"{'='*80}")

    # sin_dtheta
    all_sin_inlier = []
    for fname, s in all_stats.items():
        sin_data = s['sin_dtheta_inlier']
        all_sin_inlier.append(sin_data)

    print(f"\n1. 方向偏差 sin_dtheta (内点):")
    print(f"  {'帧':<55} {'中位':>6} {'P95':>6} {'P99':>6} {'>0.2':>6} {'>0.3':>6}")
    for fname, s in all_stats.items():
        d = s['sin_dtheta_inlier']
        print(f"  {fname:<55} {d['median']:>6.4f} {d['p95']:>6.4f} {d['p99']:>6.4f} {d['pct_above_0.2']:>5.1f}% {d.get('pct_above_0.3',0):>5.1f}%")

    # 汇总
    sin_medians = [s['sin_dtheta_inlier']['median'] for s in all_stats.values()]
    sin_p95s = [s['sin_dtheta_inlier']['p95'] for s in all_stats.values()]
    sin_p99s = [s['sin_dtheta_inlier']['p99'] for s in all_stats.values()]
    sin_above_02 = [s['sin_dtheta_inlier']['pct_above_0.2'] for s in all_stats.values()]
    print(f"\n  汇总: 中位范围=[{min(sin_medians):.4f},{max(sin_medians):.4f}] "
          f"P95范围=[{min(sin_p95s):.4f},{max(sin_p95s):.4f}] "
          f"P99范围=[{min(sin_p99s):.4f},{max(sin_p99s):.4f}]")
    print(f"  >0.2比例范围=[{min(sin_above_02):.1f}%,{max(sin_above_02):.1f}%]")

    # 模长比
    print(f"\n2. 单向量模长比 (ratio):")
    print(f"  {'帧':<55} {'中位':>6} {'±10%外':>7} {'±15%外':>7} {'±20%外':>7}")
    for fname, s in all_stats.items():
        d = s['ratio']
        print(f"  {fname:<55} {d['median']:>6.4f} {d['pct_outside_0.10']:>6.1f}% {d['pct_outside_0.15']:>6.1f}% {d['pct_outside_0.20']:>6.1f}%")

    ratio_10 = [s['ratio']['pct_outside_0.10'] for s in all_stats.values()]
    ratio_15 = [s['ratio']['pct_outside_0.15'] for s in all_stats.values()]
    print(f"\n  汇总: ±10%外=[{min(ratio_10):.1f}%,{max(ratio_10):.1f}%] "
          f"±15%外=[{min(ratio_15):.1f}%,{max(ratio_15):.1f}%]")

    # 尺度
    print(f"\n3. 点对距离比 (scale):")
    print(f"  {'帧':<55} {'中位':>6} {'±10%外':>7} {'±15%外':>7}")
    for fname, s in all_stats.items():
        d = s['scale']
        print(f"  {fname:<55} {d['median']:>6.4f} {d['pct_outside_0.10']:>6.1f}% {d['pct_outside_0.15']:>6.1f}%")

    scale_10 = [s['scale']['pct_outside_0.10'] for s in all_stats.values()]
    print(f"\n  汇总: ±10%外=[{min(scale_10):.1f}%,{max(scale_10):.1f}%]")

    # 角度差
    print(f"\n4. 点对角度差 (angle_diff):")
    print(f"  {'帧':<55} {'中位':>6} {'P95':>6} {'>90°':>6} {'>60°':>6}")
    for fname, s in all_stats.items():
        d = s['angle_diff']
        print(f"  {fname:<55} {d['median']:>5.1f}° {d['p95']:>5.1f}° {d['pct_above_90']:>5.1f}% {d['pct_above_60']:>5.1f}%")

    angle_90 = [s['angle_diff']['pct_above_90'] for s in all_stats.values()]
    angle_60 = [s['angle_diff']['pct_above_60'] for s in all_stats.values()]
    print(f"\n  汇总: >90°=[{min(angle_90):.1f}%,{max(angle_90):.1f}%] "
          f">60°=[{min(angle_60):.1f}%,{max(angle_60):.1f}%]")

    # 平移
    print(f"\n5. 平移 (tx, ty):")
    print(f"  {'帧':<55} {'tx_p99':>8} {'ty_p99':>8} {'FOV对角':>8} {'>0.5FOV':>7}")
    for fname, s in all_stats.items():
        d = s['translation']
        print(f"  {fname:<55} {d['tx_p99']:>7.1f}\" {d['ty_p99']:>7.1f}\" {d['fov_diag_arcsec']:>7.1f}\" {d['pct_outside_0.5fov']:>6.1f}%")

    # ============================================================
    # 最优参数推荐
    # ============================================================
    print(f"\n{'='*80}")
    print("V3参数优化建议")
    print(f"{'='*80}")

    print(f"""
基于V2正确结果的统计分布:

1. sin_tau (方向偏差阈值):
   - 当前V3: sin_tau_coarse=0.2 (约12°), sin_tau_refine=0.1 (约6°)
   - 内点P95: {max(sin_p95s):.4f}, P99: {max(sin_p99s):.4f}
   - 内点>0.2比例: {max(sin_above_02):.1f}%
   - 建议: sin_tau_coarse=0.5 (约30°), sin_tau_refine=0.3 (约17°)
     原因: 内点中有{max(sin_above_02):.1f}%超过0.2, 当前阈值过严会误杀正确内点

2. 模长比阈值 (剪枝0):
   - 当前V3: ±10% (|ratio-1| > 0.10 剪枝)
   - ±10%外比例: {max(ratio_10):.1f}%
   - ±15%外比例: {max(ratio_15):.1f}%
   - 建议: 放宽到±20% (|ratio-1| > 0.20 剪枝)
     原因: ±10%会剪掉{max(ratio_10):.1f}%的正确采样对

3. 尺度约束 (剪枝1):
   - 当前V3: s ∈ [0.9, 1.1] (±10%)
   - ±10%外比例: {max(scale_10):.1f}%
   - 建议: 放宽到s ∈ [0.85, 1.15] (±15%)
     原因: 粗匹配阶段s精度有限, ±10%过严

4. 角度单调性 (剪枝2):
   - 当前V3: angle_diff > 90° 剪枝
   - >90°比例: {max(angle_90):.1f}%
   - 建议: 保持90°或放宽到120°
     原因: {max(angle_90):.1f}%的采样对角度差>90°, 90°阈值合理

5. 平移约束 (剪枝3):
   - 当前V3: |tx|, |ty| < 0.5×FOV对角线
   - 建议: 保持不变, 物理约束合理
""")


if __name__ == '__main__':
    main()

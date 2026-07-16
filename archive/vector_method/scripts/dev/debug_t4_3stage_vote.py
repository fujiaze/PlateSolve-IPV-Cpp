# -*- coding: utf-8 -*-
"""T4帧: 三阶段1点法 - 向量化高效版
阶段1: 1点法预热, 找θ峰值
阶段2: 固定θ, s粗搜索(步长0.01), 向量化计算
阶段3: 在最佳s附近精搜索(步长0.002)

关键优化: 固定(s,θ)后, 对每个w_j计算Wt平移到u_i时的内点数
  Wt = s*R(θ)*Wf + [tx, ty]
  当tx = u_i[0] - s*Wf_rot_x[j], ty = u_i[1] - s*Wf_rot_y[j]时
  Wt = s*Wf_rot + [u_i[0]-s*Wf_rot_x[j], u_i[1]-s*Wf_rot_y[j]]
     = s*(Wf_rot - Wf_rot[j]) + U[i]

所以对于固定的s, Wt_k = s*(Wf_rot[k] - Wf_rot[j]) + U[i]
内点数 = |{k: |Wt_k - U[l]| < tau for some l}|
       = |{k: |s*(Wf_rot[k] - Wf_rot[j]) + U[i] - U[l]| < tau for some l}|

这可以用KDTree高效计算: 对每个(i,j), 构建Wt后查询U的KDTree
但N_img*M = 163*259 = 42217, 每次KDTree查询O(M*logN), 总计O(N*M*M*logN)太慢

更好的方法: 对于固定的s, Wt的形状固定(只是s*Wf_rot), 平移(tx,ty)使得某个u_i匹配某个w_j
所以内点数 = |{k: |s*Wf_rot[k] + [tx,ty] - U[l]| < tau for some l}|
           = |{k: U中存在l使得 |s*Wf_rot[k] + [tx,ty] - U[l]| < tau}|

用KDTree: tree_U.query_ball_point(s*Wf_rot + [tx,ty], tau) 的非空结果数

但遍历所有(i,j)对仍然太多。优化: 只遍历(i,)或(j,), 用投票法
  对于固定的s和θ, tx = u_i[0] - s*Wf_rot_x[j], ty = u_i[1] - s*Wf_rot_y[j]
  如果(i,j)是正确对, 则(tx,ty)≈真实平移
  用(tx,ty)的2D直方图投票: 正确的(tx,ty)会聚集, 错误的会分散

这个方法更快: 只需计算N_img*M个(tx,ty), 然后2D直方图找峰值
"""
import sys, os, math, logging, time
import numpy as np
from scipy.spatial import cKDTree
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "gaia_client", "python"))

logging.basicConfig(level=logging.WARNING)

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import (
    GaiaClientPy, bisection_mag_limit,
    _build_image_vectors, _build_catalog_vectors, _apply_flip,
    _apply_similarity, _count_inliers_1to1,
)
from vector_match_v3_2 import _count_inliers_fast

fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights1", "panel1",
    "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@053123-180S-Red.fts")
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3SP")

r = ImageReader()
det = StarDetector(params=SDetParamsPy(fitRadius=0))
gaia = GaiaClientPy(gaia_dir, db_type=0)

img = r.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl

ra0, dec0 = 0.0, 0.0
for kw in img.keywords:
    name = kw.name.upper()
    if name == "OBJCTRA":
        val = kw.value
        if isinstance(val, str):
            parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
            if len(parts) >= 3:
                ra0 = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
    elif name == "OBJCTDEC":
        val = kw.value
        if isinstance(val, str):
            parts = val.replace("d", " ").replace('"', " ").split()
            if len(parts) >= 3:
                sign = -1 if parts[0].startswith("-") else 1
                dec0 = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)

d = det.detect_ex(img.data)
img_x = np.array(d.x, dtype=np.float64)
img_y = np.array(d.y, dtype=np.float64)
img_flux = np.array(d.flux, dtype=np.float64)
img_saturated = np.array(d.saturated, dtype=np.int32)

U, N_img, n_sat, sparsity = _build_image_vectors(img_x, img_y, img_flux, img_saturated, s0, w, h)
fov_diag = math.sqrt(w**2 + h**2) * s0 / 3600.0
radius_deg = fov_diag * 1.2 / 2.0
N_gaia = math.ceil(1.5 * n_sat)
mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, ra0, dec0, radius_deg, N_gaia)

print(f"N_img={N_img} M={M} s0={s0:.4f} FOV={fov_diag:.2f}")

W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)
tau = max(1.0, 2.5 * s0)

V2_S, V2_THETA, V2_TX, V2_TY = 0.983, 0.0, 40.81, 134.54

for mode in [2]:
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)

    rng = np.random.default_rng(42)

    # ── 阶段1: 预热找θ峰值 ──
    t0 = time.time()
    K_WARMUP = 5000
    thetas = []
    weights = []
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    for _ in range(K_WARMUP):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        s = norm_U[i] / norm_Wf[j]
        if s < 0.9 or s > 1.1:
            continue
        theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
        ct, st = math.cos(theta), math.sin(theta)
        tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n, rms = _count_inliers_fast(U, Wt, tau)
        theta_deg = ((math.degrees(theta) + 180) % 360) - 180
        thetas.append(theta_deg)
        weights.append(n)

    thetas = np.array(thetas)
    weights = np.array(weights)

    n_bins = 360
    bin_w = 360.0 / n_bins
    weighted_counts = np.zeros(n_bins)
    for k in range(len(thetas)):
        bin_idx = int((thetas[k] + 180) / bin_w)
        if 0 <= bin_idx < n_bins:
            weighted_counts[bin_idx] += weights[k]
    peak_idx = np.argmax(weighted_counts)
    bg_mask = np.ones(n_bins, dtype=bool)
    bg_mask[max(0, peak_idx-3):min(n_bins, peak_idx+4)] = False
    bg_mean = np.mean(weighted_counts[bg_mask]) if np.sum(bg_mask) > 10 else 1.0
    snr = weighted_counts[peak_idx] / max(bg_mean, 1e-10)
    bin_centers = np.arange(-180, 180) + 0.5
    theta_peak = bin_centers[peak_idx]

    t1 = time.time()
    print(f"阶段1: θ_peak={theta_peak:.1f}° SNR={snr:.1f}x ({t1-t0:.2f}s)")

    # ── 阶段2: s搜索, 用(tx,ty)投票法 ──
    # 固定θ后, 对每个s值, 计算所有(i,j)对的(tx,ty), 用2D直方图找峰值
    theta_fixed = theta_peak * math.pi / 180.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

    # 预计算旋转后的Wf
    Wf_rot = np.empty_like(Wf)
    Wf_rot[:, 0] = ct * Wf[:, 0] - st * Wf[:, 1]
    Wf_rot[:, 1] = st * Wf[:, 0] + ct * Wf[:, 1]

    # 构建U的KDTree
    tree_U = cKDTree(U)

    # 粗搜索: s步长0.01
    s_coarse_values = np.arange(0.92, 1.10, 0.01)
    s_coarse_scores = []

    for s_try in s_coarse_values:
        # 计算所有(i,j)对的(tx,ty)
        # tx[i,j] = U[i,0] - s*Wf_rot[j,0]
        # ty[i,j] = U[i,1] - s*Wf_rot[j,1]
        # 向量化: tx_all shape (N_img, M), ty_all shape (N_img, M)
        tx_all = U[:, 0:1] - s_try * Wf_rot[:, 0]  # (N_img, M) via broadcast
        ty_all = U[:, 1:2] - s_try * Wf_rot[:, 1]  # (N_img, M) via broadcast

        # 用(tx,ty)的2D直方图投票
        # 正确的(tx,ty)会聚集, 错误的会分散
        tx_flat = tx_all.ravel()
        ty_flat = ty_all.ravel()

        # 2D直方图
        tx_range = (tx_flat.min(), tx_flat.max())
        ty_range = (ty_flat.min(), ty_flat.max())

        # 用较粗的bin (10角秒)
        bin_size = 10.0
        tx_bins = max(1, int((tx_range[1] - tx_range[0]) / bin_size))
        ty_bins = max(1, int((ty_range[1] - ty_range[0]) / bin_size))

        H, tx_edges, ty_edges = np.histogram2d(tx_flat, ty_flat, bins=[tx_bins, ty_bins])
        peak_val = H.max()
        peak_idx_flat = np.argmax(H)
        peak_i, peak_j = np.unravel_index(peak_idx_flat, H.shape)
        tx_peak = (tx_edges[peak_i] + tx_edges[peak_i+1]) / 2
        ty_peak = (ty_edges[peak_j] + ty_edges[peak_j+1]) / 2

        # 用峰值(tx,ty)计算内点数
        Wt = np.column_stack([
            s_try * Wf_rot[:, 0] + tx_peak,
            s_try * Wf_rot[:, 1] + ty_peak
        ])
        pairs = tree_U.query_ball_point(Wt, tau)
        n_inliers = sum(1 for p in pairs if len(p) > 0)

        s_coarse_scores.append(n_inliers)
        if n_inliers >= 10:
            print(f"  s={s_try:.2f}: n={n_inliers} tx={tx_peak:.1f} ty={ty_peak:.1f} (投票峰值={peak_val:.0f})")

    best_coarse_idx = np.argmax(s_coarse_scores)
    s_coarse_best = s_coarse_values[best_coarse_idx]
    n_coarse_best = s_coarse_scores[best_coarse_idx]

    t2 = time.time()
    print(f"\n阶段2粗搜: s_best={s_coarse_best:.2f} n={n_coarse_best} ({t2-t1:.2f}s)")

    # ── 阶段3: s精搜索 ──
    s_fine_start = max(0.90, s_coarse_best - 0.015)
    s_fine_end = min(1.10, s_coarse_best + 0.015)
    s_fine_values = np.arange(s_fine_start, s_fine_end + 0.001, 0.002)

    max_n_fine = 0
    best_s_fine = s_coarse_best
    best_tx_fine, best_ty_fine = 0, 0

    for s_try in s_fine_values:
        tx_all = U[:, 0:1] - s_try * Wf_rot[:, 0]
        ty_all = U[:, 1:2] - s_try * Wf_rot[:, 1]
        tx_flat = tx_all.ravel()
        ty_flat = ty_all.ravel()

        # 用更细的bin (5角秒)
        bin_size = 5.0
        tx_range = (tx_flat.min(), tx_flat.max())
        ty_range = (ty_flat.min(), ty_flat.max())
        tx_bins = max(1, int((tx_range[1] - tx_range[0]) / bin_size))
        ty_bins = max(1, int((ty_range[1] - ty_range[0]) / bin_size))

        H, tx_edges, ty_edges = np.histogram2d(tx_flat, ty_flat, bins=[tx_bins, ty_bins])
        peak_val = H.max()
        peak_idx_flat = np.argmax(H)
        peak_i, peak_j = np.unravel_index(peak_idx_flat, H.shape)
        tx_peak = (tx_edges[peak_i] + tx_edges[peak_i+1]) / 2
        ty_peak = (ty_edges[peak_j] + ty_edges[peak_j+1]) / 2

        Wt = np.column_stack([
            s_try * Wf_rot[:, 0] + tx_peak,
            s_try * Wf_rot[:, 1] + ty_peak
        ])
        pairs = tree_U.query_ball_point(Wt, tau)
        n_inliers = sum(1 for p in pairs if len(p) > 0)

        if n_inliers > max_n_fine:
            max_n_fine = n_inliers
            best_s_fine = s_try
            best_tx_fine, best_ty_fine = tx_peak, ty_peak

        print(f"  s={s_try:.3f}: n={n_inliers} tx={tx_peak:.1f} ty={ty_peak:.1f} (投票峰值={peak_val:.0f})")

    t3 = time.time()

    # 用最佳参数做1对1验证
    if max_n_fine > 0:
        Wt_best = np.empty_like(Wf)
        Wt_best[:, 0] = best_s_fine * (ct * Wf[:, 0] - st * Wf[:, 1]) + best_tx_fine
        Wt_best[:, 1] = best_s_fine * (st * Wf[:, 0] + ct * Wf[:, 1]) + best_ty_fine
        n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_best, tau)
        print(f"\n阶段3精搜: s={best_s_fine:.3f} θ={theta_peak:.1f}° tx={best_tx_fine:.1f} ty={best_ty_fine:.1f}")
        print(f"1对1: n={n_1to1} rms={rms_1to1:.3f} ({t3-t2:.2f}s)")
        print(f"V2参考: s={V2_S} θ={V2_THETA}° tx={V2_TX} ty={V2_TY}")

gaia.close()

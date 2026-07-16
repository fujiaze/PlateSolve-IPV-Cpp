# -*- coding: utf-8 -*-
"""T4帧: 多种固定策略对比验证
策略1: 固定s(中位数)+θ(峰值), 只算(tx,ty)
策略2: 固定θ(峰值), 每个对用自己的s算(tx,ty)
策略3: 固定θ(峰值), 用加权s(内点数加权)算(tx,ty)
策略4: 不固定任何参数, 但用内点数加权θ直方图+内点数加权s直方图, 取top-k个(tx,ty)聚类
"""
import sys, os, math, logging
import numpy as np
from collections import defaultdict
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
            parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
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

# V2正确变换参考值
V2_S, V2_THETA, V2_TX, V2_TY = 0.983, 0.0, 40.81, 134.54

for mode in [2]:  # T4帧用Mode 2
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    rng = np.random.default_rng(42)

    # ── 阶段1: 预热抽样, 收集所有参数 ──
    K_WARMUP = 10000
    samples = []  # (theta_deg, s, tx, ty, n_inliers)

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
        samples.append((theta_deg, s, tx, ty, n))

    thetas = np.array([x[0] for x in samples])
    s_vals = np.array([x[1] for x in samples])
    weights = np.array([x[4] for x in samples])

    # 加权θ直方图
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

    # 加权s直方图
    s_bins = 200
    s_range = (0.9, 1.1)
    s_weighted_counts = np.zeros(s_bins)
    for k in range(len(s_vals)):
        bin_idx = int((s_vals[k] - s_range[0]) / (s_range[1] - s_range[0]) * s_bins)
        if 0 <= bin_idx < s_bins:
            s_weighted_counts[bin_idx] += weights[k]
    s_peak_idx = np.argmax(s_weighted_counts)
    s_peak = s_range[0] + (s_peak_idx + 0.5) / s_bins * (s_range[1] - s_range[0])
    s_median = float(np.median(s_vals))
    s_weighted_median = float(np.median(s_vals[weights > np.median(weights)])) if np.sum(weights > np.median(weights)) > 0 else s_median

    print(f"\nMode {mode}: θ_peak={theta_peak:.1f}° SNR={snr:.1f}x")
    print(f"  s_median={s_median:.4f} s_peak={s_peak:.4f} s_weighted_median={s_weighted_median:.4f}")
    print(f"  V2参考: s={V2_S} θ={V2_THETA}° tx={V2_TX} ty={V2_TY}")

    # ── 策略1: 固定s(中位数)+θ(峰值), 只算(tx,ty) ──
    print("\n=== 策略1: 固定s(中位数)+θ(峰值) ===")
    s_fixed = s_median
    theta_fixed = theta_peak * math.pi / 180.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)
    max_n = 0
    best_tx, best_ty = 0, 0
    for _ in range(50000):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        tx = U[i, 0] - s_fixed * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s_fixed * (st * Wf[j, 0] + ct * Wf[j, 1])
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s_fixed * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s_fixed * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n, rms = _count_inliers_fast(U, Wt, tau)
        if n > max_n:
            max_n = n
            best_tx, best_ty = tx, ty
    print(f"  max_n={max_n} tx={best_tx:.1f} ty={best_ty:.1f} (V2: tx={V2_TX} ty={V2_TY})")
    if max_n > 0:
        Wt_best = np.empty_like(Wf)
        Wt_best[:, 0] = s_fixed * (ct * Wf[:, 0] - st * Wf[:, 1]) + best_tx
        Wt_best[:, 1] = s_fixed * (st * Wf[:, 0] + ct * Wf[:, 1]) + best_ty
        n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_best, tau)
        print(f"  1对1: n={n_1to1} rms={rms_1to1:.3f}")

    # ── 策略2: 固定θ(峰值), 每个对用自己的s算(tx,ty) ──
    print("\n=== 策略2: 固定θ, 每个对用自己的s ===")
    max_n = 0
    best_params = (0, 0, 0)
    for _ in range(50000):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        s = norm_U[i] / norm_Wf[j]
        if s < 0.9 or s > 1.1:
            continue
        tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n, rms = _count_inliers_fast(U, Wt, tau)
        if n > max_n:
            max_n = n
            best_params = (s, tx, ty)
    s_best, tx_best, ty_best = best_params
    print(f"  max_n={max_n} s={s_best:.4f} tx={tx_best:.1f} ty={ty_best:.1f}")
    if max_n > 0:
        Wt_best = np.empty_like(Wf)
        Wt_best[:, 0] = s_best * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx_best
        Wt_best[:, 1] = s_best * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty_best
        n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_best, tau)
        print(f"  1对1: n={n_1to1} rms={rms_1to1:.3f}")

    # ── 策略3: 用θ峰值附近(±5°)的样本, 对s和(tx,ty)做内点数加权直方图 ──
    print("\n=== 策略3: θ峰值附近样本的s/tx/ty加权直方图 ===")
    theta_mask = np.abs(thetas - theta_peak) < 5.0
    if np.sum(theta_mask) > 0:
        s_near = s_vals[theta_mask]
        w_near = weights[theta_mask]
        tx_near = np.array([x[2] for x in np.array(samples)[theta_mask]])
        ty_near = np.array([x[3] for x in np.array(samples)[theta_mask]])

        # s加权直方图
        s_hist, s_edges = np.histogram(s_near, bins=100, range=(0.9, 1.1), weights=w_near)
        s_peak_idx = np.argmax(s_hist)
        s_peak_val = (s_edges[s_peak_idx] + s_edges[s_peak_idx+1]) / 2

        # tx加权直方图
        tx_hist, tx_edges = np.histogram(tx_near, bins=200, range=(-200, 200), weights=w_near)
        tx_peak_idx = np.argmax(tx_hist)
        tx_peak_val = (tx_edges[tx_peak_idx] + tx_edges[tx_peak_idx+1]) / 2

        # ty加权直方图
        ty_hist, ty_edges = np.histogram(ty_near, bins=200, range=(-200, 200), weights=w_near)
        ty_peak_idx = np.argmax(ty_hist)
        ty_peak_val = (ty_edges[ty_peak_idx] + ty_edges[ty_peak_idx+1]) / 2

        print(f"  θ附近样本数: {np.sum(theta_mask)}")
        print(f"  s_peak={s_peak_val:.4f} tx_peak={tx_peak_val:.1f} ty_peak={ty_peak_val:.1f}")
        print(f"  V2参考: s={V2_S} tx={V2_TX} ty={V2_TY}")

        # 用直方图峰值验证
        s_v = s_peak_val
        theta_v = theta_peak * math.pi / 180.0
        ct_v, st_v = math.cos(theta_v), math.sin(theta_v)
        Wt_v = np.empty_like(Wf)
        Wt_v[:, 0] = s_v * (ct_v * Wf[:, 0] - st_v * Wf[:, 1]) + tx_peak_val
        Wt_v[:, 1] = s_v * (st_v * Wf[:, 0] + ct_v * Wf[:, 1]) + ty_peak_val
        n_v, rms_v = _count_inliers_fast(U, Wt_v, tau)
        n_1to1_v, rms_1to1_v, _ = _count_inliers_1to1(U, Wt_v, tau)
        print(f"  快速内点: n={n_v} rms={rms_v:.1f}")
        print(f"  1对1内点: n={n_1to1_v} rms={rms_1to1_v:.3f}")

    # ── 策略4: 预热阶段top-10内点数的变换, 直接验证 ──
    print("\n=== 策略4: 预热阶段top-10内点数变换 ===")
    sorted_idx = np.argsort(weights)[::-1]
    for rank, idx in enumerate(sorted_idx[:10]):
        theta_d, s_v, tx_v, ty_v, n_v = samples[idx]
        print(f"  #{rank+1}: n={n_v} θ={theta_d:.1f}° s={s_v:.4f} tx={tx_v:.1f} ty={ty_v:.1f}")

    # ── 策略5: 用正确s(V2参考)固定, 只搜索θ和(tx,ty) ──
    print("\n=== 策略5: 固定s=V2参考值(0.983), 搜索θ和(tx,ty) ===")
    s_fixed = V2_S
    max_n = 0
    best_result = None
    for _ in range(50000):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
        ct_t, st_t = math.cos(theta), math.sin(theta)
        tx = U[i, 0] - s_fixed * (ct_t * Wf[j, 0] - st_t * Wf[j, 1])
        ty = U[i, 1] - s_fixed * (st_t * Wf[j, 0] + ct_t * Wf[j, 1])
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s_fixed * (ct_t * Wf[:, 0] - st_t * Wf[:, 1]) + tx
        Wt[:, 1] = s_fixed * (st_t * Wf[:, 0] + ct_t * Wf[:, 1]) + ty
        n, rms = _count_inliers_fast(U, Wt, tau)
        if n > max_n:
            max_n = n
            theta_deg = ((math.degrees(theta) + 180) % 360) - 180
            best_result = (theta_deg, tx, ty)
    if best_result:
        print(f"  max_n={max_n} θ={best_result[0]:.1f}° tx={best_result[1]:.1f} ty={best_result[2]:.1f}")
        theta_v = best_result[0] * math.pi / 180.0
        ct_v, st_v = math.cos(theta_v), math.sin(theta_v)
        Wt_v = np.empty_like(Wf)
        Wt_v[:, 0] = s_fixed * (ct_v * Wf[:, 0] - st_v * Wf[:, 1]) + best_result[1]
        Wt_v[:, 1] = s_fixed * (st_v * Wf[:, 0] + ct_v * Wf[:, 1]) + best_result[2]
        n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_v, tau)
        print(f"  1对1: n={n_1to1} rms={rms_1to1:.3f}")

gaia.close()

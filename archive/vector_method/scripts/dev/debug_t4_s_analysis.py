# -*- coding: utf-8 -*-
"""T4帧: 分析1点法s的分布, 找出正确对的s特征

核心问题: 1点法的s=|u_i|/|w_j|, 正确对的s应该接近真实s=0.983
但预热阶段s_median=0.9867, s_peak=0.9155, 都不够准

分析:
1. θ在峰值附近的样本, s的分布是什么?
2. 内点数>0的样本, s的分布是什么?
3. 是否存在s的子区间, 使得正确变换的信号更强?
"""
import sys, os, math, logging
import numpy as np
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

V2_S = 0.983

for mode in [2]:
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    rng = np.random.default_rng(42)

    # ── 全量1点法抽样, 收集所有参数 ──
    K = 20000
    samples = []  # (theta_deg, s, n_inliers)

    for _ in range(K):
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
        samples.append((theta_deg, s, n))

    thetas = np.array([x[0] for x in samples])
    s_vals = np.array([x[1] for x in samples])
    n_vals = np.array([x[2] for x in samples])

    print(f"\n总样本数: {len(samples)}")

    # ── 分析1: θ峰值附近(±3°)的s分布 ──
    theta_mask = np.abs(thetas) < 3.0
    s_near_theta = s_vals[theta_mask]
    n_near_theta = n_vals[theta_mask]
    print(f"\nθ∈[-3°,3°]样本数: {np.sum(theta_mask)}")
    if len(s_near_theta) > 0:
        print(f"  s: mean={np.mean(s_near_theta):.4f} median={np.median(s_near_theta):.4f} std={np.std(s_near_theta):.4f}")
        print(f"  s分位数: 10%={np.percentile(s_near_theta,10):.4f} 25%={np.percentile(s_near_theta,25):.4f} 50%={np.percentile(s_near_theta,50):.4f} 75%={np.percentile(s_near_theta,75):.4f} 90%={np.percentile(s_near_theta,90):.4f}")

        # 内点数加权的s统计
        w = n_near_theta.astype(float)
        if np.sum(w) > 0:
            s_wmean = np.average(s_near_theta, weights=w)
            # 加权中位数
            sorted_idx = np.argsort(s_near_theta)
            s_sorted = s_near_theta[sorted_idx]
            w_sorted = w[sorted_idx]
            cumw = np.cumsum(w_sorted)
            s_wmedian = s_sorted[np.searchsorted(cumw, cumw[-1]/2)]
            print(f"  s加权: mean={s_wmean:.4f} median={s_wmedian:.4f}")

    # ── 分析2: 内点数>0的样本, s和θ的联合分布 ──
    pos_mask = n_vals > 0
    print(f"\n内点数>0样本数: {np.sum(pos_mask)}")
    if np.sum(pos_mask) > 0:
        s_pos = s_vals[pos_mask]
        theta_pos = thetas[pos_mask]
        n_pos = n_vals[pos_mask]
        print(f"  θ: mean={np.mean(theta_pos):.1f}° median={np.median(theta_pos):.1f}°")
        print(f"  s: mean={np.mean(s_pos):.4f} median={np.median(s_pos):.4f}")

        # 内点数>=2的样本
        mask2 = n_vals >= 2
        print(f"\n内点数>=2样本数: {np.sum(mask2)}")
        if np.sum(mask2) > 0:
            s_m2 = s_vals[mask2]
            theta_m2 = thetas[mask2]
            n_m2 = n_vals[mask2]
            print(f"  θ: mean={np.mean(theta_m2):.1f}° median={np.median(theta_m2):.1f}°")
            print(f"  s: mean={np.mean(s_m2):.4f} median={np.median(s_m2):.4f}")
            # 加权
            w2 = n_m2.astype(float)
            s_wmean2 = np.average(s_m2, weights=w2)
            sorted_idx = np.argsort(s_m2)
            s_sorted2 = s_m2[sorted_idx]
            w_sorted2 = w2[sorted_idx]
            cumw2 = np.cumsum(w_sorted2)
            s_wmedian2 = s_sorted2[np.searchsorted(cumw2, cumw2[-1]/2)]
            print(f"  s加权: mean={s_wmean2:.4f} median={s_wmedian2:.4f}")

    # ── 分析3: s在[0.97, 0.99]范围内的样本特征 ──
    s_mask = (s_vals >= 0.97) & (s_vals <= 0.99)
    print(f"\ns∈[0.97,0.99]样本数: {np.sum(s_mask)}")
    if np.sum(s_mask) > 0:
        theta_s = thetas[s_mask]
        n_s = n_vals[s_mask]
        print(f"  θ: mean={np.mean(theta_s):.1f}° median={np.median(theta_s):.1f}°")
        print(f"  n: mean={np.mean(n_s):.1f} max={np.max(n_s)}")

    # ── 分析4: 用s的精细直方图(内点数加权), 找s峰值 ──
    print("\n--- s内点数加权直方图 (0.001精度) ---")
    s_bins = np.arange(0.90, 1.10, 0.001)
    s_weighted = np.zeros(len(s_bins)-1)
    s_counts = np.zeros(len(s_bins)-1)
    for k in range(len(s_vals)):
        bin_idx = np.searchsorted(s_bins, s_vals[k]) - 1
        if 0 <= bin_idx < len(s_weighted):
            s_weighted[bin_idx] += n_vals[k]
            s_counts[bin_idx] += 1

    # Top-10 s bins by weighted count
    top_idx = np.argsort(s_weighted)[::-1][:10]
    print("  Top-10 s bins (by weighted inlier count):")
    for rank, idx in enumerate(top_idx):
        s_center = (s_bins[idx] + s_bins[idx+1]) / 2
        print(f"    #{rank+1}: s={s_center:.4f} weight={s_weighted[idx]:.0f} count={s_counts[idx]:.0f}")

    # ── 分析5: 只看θ∈[-3°,3°] AND s∈[0.97,0.99]的样本 ──
    combo_mask = theta_mask & s_mask
    print(f"\nθ∈[-3°,3°] AND s∈[0.97,0.99]样本数: {np.sum(combo_mask)}")
    if np.sum(combo_mask) > 0:
        n_combo = n_vals[combo_mask]
        print(f"  n: mean={np.mean(n_combo):.1f} max={np.max(n_combo)}")

    # ── 分析6: 遍历s候选值, 固定θ=0°搜索(tx,ty), 找最佳s ──
    print("\n--- 遍历s候选值, 固定θ=0°搜索(tx,ty) ---")
    theta_fixed = 0.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

    for s_try in np.arange(0.970, 0.995, 0.002):
        max_n = 0
        best_tx, best_ty = 0, 0
        for _ in range(10000):
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            if not valid_U[i] or not valid_Wf[j]:
                continue
            tx = U[i, 0] - s_try * (ct * Wf[j, 0] - st * Wf[j, 1])
            ty = U[i, 1] - s_try * (st * Wf[j, 0] + ct * Wf[j, 1])
            Wt = np.empty_like(Wf)
            Wt[:, 0] = s_try * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
            Wt[:, 1] = s_try * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
            n, rms = _count_inliers_fast(U, Wt, tau)
            if n > max_n:
                max_n = n
                best_tx, best_ty = tx, ty
        print(f"  s={s_try:.3f}: max_n={max_n} tx={best_tx:.1f} ty={best_ty:.1f}")

gaia.close()

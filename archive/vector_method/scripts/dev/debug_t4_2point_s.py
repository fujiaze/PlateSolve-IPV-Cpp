# -*- coding: utf-8 -*-
"""T4帧: 两阶段1点法
阶段1: 预热找θ峰值
阶段2: 在θ峰值附近(±3°), 用2点法确定s, 然后固定(s,θ)搜索(tx,ty)

核心思路: θ≈0°时tx≈U[i,0]-s*Wf[j,0], s的微小偏差直接映射到tx偏差
所以必须先准确确定s, 才能正确计算tx/ty

2点法确定s: 如果(u_i,w_j)和(u_k,w_l)都是正确对, 则
  s = |u_i - u_k| / |w_j - w_l|
这个s不受gnomonic投影畸变影响(因为减法消去了中心偏移)
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

V2_S, V2_THETA, V2_TX, V2_TY = 0.983, 0.0, 40.81, 134.54

for mode in [2]:
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    rng = np.random.default_rng(42)

    # ── 阶段1: 预热, 找θ峰值 ──
    K_WARMUP = 10000
    thetas = []
    weights = []

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

    print(f"阶段1: θ_peak={theta_peak:.1f}° SNR={snr:.1f}x")

    # ── 阶段2: 用2点法确定s ──
    # 思路: 随机抽2个图像向量(u_i, u_k)和2个星表向量(w_j, w_l)
    # 用向量差计算s = |u_i - u_k| / |w_j - w_l|
    # 如果s在[0.9, 1.1]范围内, 则用θ_peak和这个s计算(tx,ty), 统计内点数
    # 用内点数加权s直方图

    theta_fixed = theta_peak * math.pi / 180.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

    K_S_SEARCH = 50000
    s_weights_map = {}  # s_bin -> weighted_count

    max_n = 0
    best_result = None

    for trial in range(K_S_SEARCH):
        i = rng.integers(0, N_img)
        k = rng.integers(0, N_img)
        if i == k:
            continue
        j = rng.integers(0, M)
        l = rng.integers(0, M)
        if j == l:
            continue
        if not valid_U[i] or not valid_U[k] or not valid_Wf[j] or not valid_Wf[l]:
            continue

        # 用向量差计算s
        du = U[i] - U[k]
        dw = Wf[j] - Wf[l]
        norm_du = math.sqrt(du[0]**2 + du[1]**2)
        norm_dw = math.sqrt(dw[0]**2 + dw[1]**2)
        if norm_dw < 1e-10:
            continue
        s = norm_du / norm_dw
        if s < 0.9 or s > 1.1:
            continue

        # 用固定θ和这个s计算(tx,ty)
        tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])

        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n, rms = _count_inliers_fast(U, Wt, tau)

        # s加权直方图
        s_bin = round(s, 3)  # 0.001精度
        if s_bin not in s_weights_map:
            s_weights_map[s_bin] = 0
        s_weights_map[s_bin] += n

        if n > max_n:
            max_n = n
            best_result = (s, tx, ty, rms)
            if n >= 20:
                print(f"  命中: n={n} s={s:.4f} tx={tx:.1f} ty={ty:.1f} rms={rms:.1f}")

    # s加权直方图峰值
    if s_weights_map:
        s_peak = max(s_weights_map, key=s_weights_map.get)
        s_peak_weight = s_weights_map[s_peak]
        # 背景均值
        all_weights = list(s_weights_map.values())
        bg_s = np.median(all_weights)
        s_snr = s_peak_weight / max(bg_s, 1e-10)
        print(f"\n阶段2: s_peak={s_peak:.4f} (权重={s_peak_weight:.0f}) SNR={s_snr:.1f}x")
        print(f"  V2参考: s={V2_S}")

    if best_result:
        s_best, tx_best, ty_best, rms_best = best_result
        print(f"\n最佳变换: n={max_n} s={s_best:.4f} tx={tx_best:.1f} ty={ty_best:.1f}")
        print(f"V2参考:   s={V2_S} tx={V2_TX} ty={V2_TY}")

        # 用最佳变换做1对1验证
        Wt_best = np.empty_like(Wf)
        Wt_best[:, 0] = s_best * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx_best
        Wt_best[:, 1] = s_best * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty_best
        n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_best, tau)
        print(f"1对1: n={n_1to1} rms={rms_1to1:.3f}")

        # 用s_peak重新搜索(tx,ty)
        print(f"\n--- 用s_peak={s_peak:.4f}重新搜索(tx,ty) ---")
        s_fixed = s_peak
        max_n2 = 0
        best_tx2, best_ty2 = 0, 0
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
            if n > max_n2:
                max_n2 = n
                best_tx2, best_ty2 = tx, ty
        print(f"  max_n={max_n2} tx={best_tx2:.1f} ty={best_ty2:.1f}")
        if max_n2 > 0:
            Wt_v = np.empty_like(Wf)
            Wt_v[:, 0] = s_fixed * (ct * Wf[:, 0] - st * Wf[:, 1]) + best_tx2
            Wt_v[:, 1] = s_fixed * (st * Wf[:, 0] + ct * Wf[:, 1]) + best_ty2
            n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_v, tau)
            print(f"  1对1: n={n_1to1} rms={rms_1to1:.3f}")

gaia.close()

# -*- coding: utf-8 -*-
"""T4帧: 三阶段1点法验证
阶段1: 1点法预热, 找θ峰值
阶段2: 固定θ, s粗搜索(步长0.01), 每个s用1000次抽样找(tx,ty)
阶段3: 在最佳s附近精搜索(步长0.002), 用10000次抽样找(tx,ty)

同时测试T2帧验证算法通用性
"""
import sys, os, math, logging, time
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

def test_frame(fits_path, gaia_dir, gaia, mode_list=[0, 1, 2, 3]):
    r = ImageReader()
    det = StarDetector(params=SDetParamsPy(fitRadius=0))

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

    print(f"N_img={N_img} M={M} s0={s0:.4f} FOV={fov_diag:.2f} n_sat={n_sat}")

    W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)
    tau = max(1.0, 2.5 * s0)

    norm_U = np.linalg.norm(U, axis=1)
    norm_W = np.linalg.norm(W, axis=1)
    valid_U = norm_U > 1e-10
    valid_W = norm_W > 1e-10

    best_overall = None  # (n, s, theta, tx, ty, mode, rms)

    for mode in mode_list:
        t0 = time.time()
        Wf = _apply_flip(W, mode)
        norm_Wf = np.linalg.norm(Wf, axis=1)
        valid_Wf = norm_Wf > 1e-10

        rng = np.random.default_rng(42)

        # ── 阶段1: 预热找θ峰值 ──
        K_WARMUP = 5000
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

        t1 = time.time()
        print(f"  Mode {mode}: θ_peak={theta_peak:.1f}° SNR={snr:.1f}x ({t1-t0:.2f}s)")

        if snr < 3.0:
            print(f"    SNR太低, 跳过")
            continue

        # ── 阶段2: s粗搜索 ──
        theta_fixed = theta_peak * math.pi / 180.0
        ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

        s_coarse_values = np.arange(0.91, 1.10, 0.01)
        s_coarse_scores = []

        for s_try in s_coarse_values:
            max_n = 0
            for _ in range(1000):
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
            s_coarse_scores.append(max_n)

        best_coarse_idx = np.argmax(s_coarse_scores)
        s_coarse_best = s_coarse_values[best_coarse_idx]
        n_coarse_best = s_coarse_scores[best_coarse_idx]

        t2 = time.time()
        print(f"    粗搜: s_best={s_coarse_best:.2f} n={n_coarse_best} ({t2-t1:.2f}s)")

        # ── 阶段3: s精搜索 ──
        s_fine_start = max(0.90, s_coarse_best - 0.015)
        s_fine_end = min(1.10, s_coarse_best + 0.015)
        s_fine_values = np.arange(s_fine_start, s_fine_end + 0.001, 0.002)

        max_n_fine = 0
        best_s_fine = s_coarse_best
        best_tx_fine, best_ty_fine = 0, 0

        for s_try in s_fine_values:
            max_n = 0
            best_tx, best_ty = 0, 0
            for _ in range(5000):
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
            if max_n > max_n_fine:
                max_n_fine = max_n
                best_s_fine = s_try
                best_tx_fine, best_ty_fine = best_tx, best_ty

        t3 = time.time()

        # 用最佳参数做1对1验证
        if max_n_fine > 0:
            Wt_best = np.empty_like(Wf)
            Wt_best[:, 0] = best_s_fine * (ct * Wf[:, 0] - st * Wf[:, 1]) + best_tx_fine
            Wt_best[:, 1] = best_s_fine * (st * Wf[:, 0] + ct * Wf[:, 1]) + best_ty_fine
            n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_best, tau)
            print(f"    精搜: s={best_s_fine:.3f} θ={theta_peak:.1f}° tx={best_tx_fine:.1f} ty={best_ty_fine:.1f}")
            print(f"    1对1: n={n_1to1} rms={rms_1to1:.3f} ({t3-t2:.2f}s)")

            if best_overall is None or n_1to1 > best_overall[0]:
                best_overall = (n_1to1, best_s_fine, theta_peak, best_tx_fine, best_ty_fine, mode, rms_1to1)

    return best_overall

# ── 测试T4帧 ──
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3SP")
gaia = GaiaClientPy(gaia_dir, db_type=0)

print("=" * 60)
print("T4帧 (FOV=9.91°, θ≈0°)")
print("=" * 60)
t4_path = os.path.join(PROJECT_ROOT, "testdata", "lights1", "panel1",
    "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@053123-180S-Red.fts")
result = test_frame(t4_path, gaia_dir, gaia)
if result:
    n, s, theta, tx, ty, mode, rms = result
    print(f"\n>>> T4最佳: Mode {mode} n={n} s={s:.3f} θ={theta:.1f}° tx={tx:.1f} ty={ty:.1f} rms={rms:.3f}")

# ── 测试T2帧 ──
print("\n" + "=" * 60)
print("T2帧 (FOV=3.31°, θ≈-90°)")
print("=" * 60)
t2_path = os.path.join(PROJECT_ROOT, "testdata", "lights1", "panel1",
    "Galaxy_Center_mosaic1_T2_skywatcher-20250703@053828-180S-Ha.fts")
result = test_frame(t2_path, gaia_dir, gaia)
if result:
    n, s, theta, tx, ty, mode, rms = result
    print(f"\n>>> T2最佳: Mode {mode} n={n} s={s:.3f} θ={theta:.1f}° tx={tx:.1f} ty={ty:.1f} rms={rms:.3f}")

gaia.close()

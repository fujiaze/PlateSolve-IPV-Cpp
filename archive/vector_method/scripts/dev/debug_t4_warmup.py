# -*- coding: utf-8 -*-
"""T4帧V3.2预热阶段调试"""
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
W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)

tau = max(1.0, 2.5 * s0)
print(f"N_img={N_img} M={M} s0={s0:.4f} FOV={fov_diag:.2f} tau={tau:.2f}")

# 对4种模式做详细分析
for mode in range(4):
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    rng = np.random.default_rng(42)
    thetas = []
    weights = []
    n_s1 = 0

    for _ in range(5000):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        s = norm_U[i] / norm_Wf[j]
        if s < 0.9 or s > 1.1:
            continue
        n_s1 += 1
        theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
        ct, st = math.cos(theta), math.sin(theta)
        tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n_inliers, rms = _count_inliers_fast(U, Wt, tau)
        theta_deg = ((math.degrees(theta) + 180) % 360) - 180
        thetas.append(theta_deg)
        weights.append(n_inliers)

    thetas = np.array(thetas)
    weights = np.array(weights)

    if len(weights) == 0:
        print(f"Mode {mode}: n_s1=0, 无有效抽样")
        continue

    # 加权直方图
    n_bins = 360
    bin_w = 360.0 / n_bins
    weighted_counts = np.zeros(n_bins)
    for k in range(len(thetas)):
        bin_idx = int((thetas[k] + 180) / bin_w)
        if 0 <= bin_idx < n_bins:
            weighted_counts[bin_idx] += weights[k]

    peak_idx = np.argmax(weighted_counts)
    bg_mask = np.ones(n_bins, dtype=bool)
    bg_mask[max(0, peak_idx - 3):min(n_bins, peak_idx + 4)] = False
    bg_mean = np.mean(weighted_counts[bg_mask]) if np.sum(bg_mask) > 10 else 1.0
    snr = weighted_counts[peak_idx] / max(bg_mean, 1e-10)
    bin_centers = np.arange(-180, 180) + 0.5

    print(f"Mode {mode}: n_s1={n_s1} median_n={np.median(weights):.1f} max_n={np.max(weights)}")
    print(f"  peak_theta={bin_centers[peak_idx]:.1f} SNR={snr:.1f}x bg_mean={bg_mean:.2f}")
    # 内点数分布
    for n_val in range(min(11, int(np.max(weights)) + 1)):
        cnt = np.sum(weights == n_val)
        if cnt > 0:
            print(f"  n_inliers={n_val}: {cnt}次")

gaia.close()

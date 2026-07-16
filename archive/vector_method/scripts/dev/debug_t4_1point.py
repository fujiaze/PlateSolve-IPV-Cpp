# -*- coding: utf-8 -*-
"""T4帧: 验证1点法是否可行 - 用V2正确变换参数"""
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
from scipy.spatial import cKDTree

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

# V2 C++的正确变换参数 (从日志中提取)
# 模式2: s=0.983434 theta=-0.0096° tx=40.8087 ty=134.5429
s_v2 = 0.983434
theta_v2 = -0.0096 * math.pi / 180.0
tx_v2 = 40.8087
ty_v2 = 134.5429
flip_mode = 2

# 用原始中心计算W
W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)
Wf = _apply_flip(W, flip_mode)

# 应用V2变换
Wt = _apply_similarity(Wf, s_v2, theta_v2, tx_v2, ty_v2)

# 计算投影误差
tree = cKDTree(Wt)
dists, idxs = tree.query(U, k=1)
dists_px = dists / s0

print(f"\nV2变换投影误差 (原始中心, Mode 2):")
print(f"  中位数: {np.median(dists_px):.3f} px ({np.median(dists):.3f} arcsec)")
print(f"  均值: {np.mean(dists_px):.3f} px")
print(f"  90%: {np.percentile(dists_px, 90):.3f} px")
print(f"  <1px: {np.sum(dists_px < 1)/len(dists_px)*100:.1f}%")
print(f"  <2px: {np.sum(dists_px < 2)/len(dists_px)*100:.1f}%")
print(f"  <3px: {np.sum(dists_px < 3)/len(dists_px)*100:.1f}%")

for tau_mult in [1.0, 2.5, 5.0]:
    tau = tau_mult * s0
    n = np.sum(dists < tau)
    print(f"  tau={tau_mult:.1f}*s0={tau:.1f}as: {n}个内点 ({n/len(dists)*100:.1f}%)")

# 1对1互斥内点
n_1to1, rms_1to1, mask_1to1 = _count_inliers_1to1(U, Wt, max(1.0, 2.5*s0))
print(f"  1对1互斥(tau=2.5*s0): n={n_1to1} rms={rms_1to1:.3f}")

# ── 关键测试: 1点法能否找到类似变换? ──
print(f"\n--- 1点法详细测试 (原始中心, Mode 2) ---")
norm_U = np.linalg.norm(U, axis=1)
norm_Wf = np.linalg.norm(Wf, axis=1)
valid_U = norm_U > 1e-10
valid_Wf = norm_Wf > 1e-10

tau = max(1.0, 2.5 * s0)
rng = np.random.default_rng(42)

# 收集所有s≈1的变换, 记录内点数和θ
results = []
for trial in range(50000):
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
    Wt_test = np.empty_like(Wf)
    Wt_test[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
    Wt_test[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
    tree_test = cKDTree(Wt_test)
    d_test, _ = tree_test.query(U, k=1)
    n = int(np.sum(d_test < tau))
    theta_deg = math.degrees(theta)
    results.append((n, theta_deg, s, tx, ty))

results.sort(key=lambda x: -x[0])
print(f"Top 10 变换 (按内点数排序):")
for k, (n, th, s, tx, ty) in enumerate(results[:10]):
    print(f"  #{k+1}: n={n} theta={th:.2f}° s={s:.4f} tx={tx:.1f} ty={ty:.1f}")

# θ接近0°的变换
near_zero = [(n, th, s, tx, ty) for n, th, s, tx, ty in results if abs(th) < 5.0]
near_zero.sort(key=lambda x: -x[0])
print(f"\nθ∈[-5°,5°]的变换: {len(near_zero)}个")
for k, (n, th, s, tx, ty) in enumerate(near_zero[:5]):
    print(f"  #{k+1}: n={n} theta={th:.2f}° s={s:.4f} tx={tx:.1f} ty={ty:.1f}")

# 加权θ直方图
thetas = np.array([r[1] for r in results])
weights = np.array([r[0] for r in results])
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
print(f"\n加权θ直方图: peak={bin_centers[peak_idx]:.1f}° SNR={snr:.1f}x")

gaia.close()

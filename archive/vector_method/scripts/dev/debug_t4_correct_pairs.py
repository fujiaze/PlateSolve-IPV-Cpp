# -*- coding: utf-8 -*-
"""T4帧: 用V2内点识别正确对, 验证1点法从正确对计算的变换"""
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

# V2正确变换
s_v2 = 0.983434
theta_v2 = -0.0096 * math.pi / 180.0
tx_v2 = 40.8087
ty_v2 = 134.5429
flip_mode = 2
tau = max(1.0, 2.5 * s0)

W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)
Wf = _apply_flip(W, flip_mode)

# 用V2变换找正确匹配对
Wt_v2 = _apply_similarity(Wf, s_v2, theta_v2, tx_v2, ty_v2)
tree_v2 = cKDTree(Wt_v2)
dists_v2, idxs_v2 = tree_v2.query(U, k=1)
inlier_mask_v2 = dists_v2 < tau
n_v2 = int(np.sum(inlier_mask_v2))
print(f"V2变换内点: {n_v2}")

# 提取正确匹配对
correct_pairs = []
for i in range(N_img):
    if inlier_mask_v2[i]:
        j = idxs_v2[i]
        correct_pairs.append((i, j, dists_v2[i]))

print(f"正确匹配对: {len(correct_pairs)}个")
print(f"距离分布: min={min(d for _,_,d in correct_pairs):.2f} median={np.median([d for _,_,d in correct_pairs]):.2f} max={max(d for _,_,d in correct_pairs):.2f} arcsec")

# 从每个正确对计算1点变换, 统计内点数
print(f"\n--- 从正确对计算1点变换 ---")
norm_U = np.linalg.norm(U, axis=1)
norm_Wf = np.linalg.norm(Wf, axis=1)

one_point_results = []
for i, j, d_ij in correct_pairs[:30]:  # 测试前30个正确对
    s = norm_U[i] / norm_Wf[j]
    if s < 0.9 or s > 1.1:
        one_point_results.append((i, j, s, 0, 0, 0, 0, "s_out"))
        continue
    theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
    ct, st = math.cos(theta), math.sin(theta)
    tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
    ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
    
    Wt_test = np.empty_like(Wf)
    Wt_test[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
    Wt_test[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
    
    n_fast, rms_fast = _count_inliers_fast(U, Wt_test, tau)
    n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_test, tau)
    
    one_point_results.append((i, j, s, math.degrees(theta), tx, ty, n_fast, n_1to1))

print(f"{'i':>4} {'j':>4} {'s':>8} {'theta':>8} {'tx':>8} {'ty':>8} {'n_fast':>7} {'n_1to1':>7}")
for r in one_point_results[:20]:
    if len(r) == 8:
        print(f"{r[0]:4d} {r[1]:4d} {r[2]:8.4f} {r[3]:8.2f}° {r[4]:8.1f} {r[5]:8.1f} {r[6]:7d} {r[7]:7d}")
    else:
        print(f"{r[0]:4d} {r[1]:4d} {r[2]:8.4f}  {r[7]}")

# 统计1点变换的内点数分布
n_fast_values = [r[6] for r in one_point_results if len(r) == 8]
if n_fast_values:
    print(f"\n1点变换内点数统计 (从正确对):")
    print(f"  min={min(n_fast_values)} max={max(n_fast_values)} median={np.median(n_fast_values):.0f} mean={np.mean(n_fast_values):.1f}")
    print(f"  >=20: {sum(1 for n in n_fast_values if n >= 20)}个")
    print(f"  >=50: {sum(1 for n in n_fast_values if n >= 50)}个")
    print(f"  >=80: {sum(1 for n in n_fast_values if n >= 80)}个")

gaia.close()

# -*- coding: utf-8 -*-
"""T4帧: 用V2 C++获取正确变换, 验证1点法是否应该能工作"""
import sys, os, math, logging
import numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "gaia_client", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v2"))

logging.basicConfig(level=logging.WARNING)

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2_cpp import VectorMatch as VectorMatchV2Cpp
from vector_match_v2 import (
    GaiaClientPy, bisection_mag_limit,
    _build_image_vectors, _build_catalog_vectors, _apply_flip,
    _apply_similarity,
)

fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights1", "panel1",
    "Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@053123-180S-Red.fts")
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3SP")

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

# V2 C++求解
vm = VectorMatchV2Cpp(gaia_dir, db_type=0)
result = vm.solve(img_x, img_y, img_flux, img_saturated, ra0, dec0, fl, ps, w, h)
vm.close()

if result is None:
    print("V2 C++ 求解失败!")
    sys.exit(1)

print(f"V2 C++ 结果:")
print(f"  RA={result.center_ra:.6f} Dec={result.center_dec:.6f}")
print(f"  rotation={result.rotation_deg:.2f}° scale={result.scale_arcsec_px:.4f} arcsec/px")
print(f"  flip_mode={result.flip_mode} matched={result.matched_count}")
print(f"  rms_px={result.rms_px:.4f} rms_arcsec={result.rms_arcsec:.4f}")

# 用V2结果构建变换, 检查投影误差分布
U, N_img, n_sat, sparsity = _build_image_vectors(img_x, img_y, img_flux, img_saturated, s0, w, h)
fov_diag = math.sqrt(w**2 + h**2) * s0 / 3600.0
radius_deg = fov_diag * 1.2 / 2.0
N_gaia = math.ceil(1.5 * n_sat)
gaia = GaiaClientPy(gaia_dir, db_type=0)
mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, ra0, dec0, radius_deg, N_gaia)

# 用V2的中心重新投影
W = _build_catalog_vectors(cat_ra, cat_dec, result.center_ra, result.center_dec)
Wf = _apply_flip(W, result.flip_mode)

# 用V2的变换参数
s_v2 = result.scale_arcsec_px / s0
theta_v2 = result.rotation_deg * math.pi / 180.0

# 从affine提取tx, ty
affine = result.affine
tx_v2 = affine[0]
ty_v2 = affine[3]

print(f"\nV2变换参数: s={s_v2:.6f} theta={result.rotation_deg:.2f}° tx={tx_v2:.2f} ty={ty_v2:.2f}")

# 应用变换
Wt = _apply_similarity(Wf, s_v2, theta_v2, tx_v2, ty_v2)

# 计算每个U点到最近Wt点的距离
from scipy.spatial import cKDTree
tree = cKDTree(Wt)
dists, idxs = tree.query(U, k=1)

dists_px = dists / s0
print(f"\n投影误差分布 (V2正确变换):")
print(f"  中位数: {np.median(dists_px):.3f} px ({np.median(dists):.3f} arcsec)")
print(f"  均值: {np.mean(dists_px):.3f} px")
print(f"  90%: {np.percentile(dists_px, 90):.3f} px")
print(f"  95%: {np.percentile(dists_px, 95):.3f} px")
print(f"  最大: {np.max(dists_px):.3f} px")
print(f"  <1px: {np.sum(dists_px < 1)/len(dists_px)*100:.1f}%")
print(f"  <2px: {np.sum(dists_px < 2)/len(dists_px)*100:.1f}%")
print(f"  <3px: {np.sum(dists_px < 3)/len(dists_px)*100:.1f}%")

# 不同tau下的内点数
for tau_mult in [1.0, 2.5, 5.0]:
    tau = tau_mult * s0
    n = np.sum(dists < tau)
    print(f"  tau={tau_mult:.1f}*s0={tau:.1f}as: {n}个内点 ({n/len(dists)*100:.1f}%)")

# 现在测试1点法: 用V2的正确中心, 看能否找到正确变换
print(f"\n--- 1点法测试 (用V2正确中心) ---")
W2 = _build_catalog_vectors(cat_ra, cat_dec, result.center_ra, result.center_dec)
Wf2 = _apply_flip(W2, result.flip_mode)
norm_U = np.linalg.norm(U, axis=1)
norm_Wf2 = np.linalg.norm(Wf2, axis=1)
valid_U = norm_U > 1e-10
valid_Wf2 = norm_Wf2 > 1e-10

rng = np.random.default_rng(42)
tau = max(1.0, 2.5 * s0)
max_n = 0
best_info = ""

for trial in range(50000):
    i = rng.integers(0, N_img)
    j = rng.integers(0, M)
    if not valid_U[i] or not valid_Wf2[j]:
        continue
    s = norm_U[i] / norm_Wf2[j]
    if s < 0.9 or s > 1.1:
        continue
    theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf2[j, 1], Wf2[j, 0])
    ct, st = math.cos(theta), math.sin(theta)
    tx = U[i, 0] - s * (ct * Wf2[j, 0] - st * Wf2[j, 1])
    ty = U[i, 1] - s * (st * Wf2[j, 0] + ct * Wf2[j, 1])
    Wt2 = np.empty_like(Wf2)
    Wt2[:, 0] = s * (ct * Wf2[:, 0] - st * Wf2[:, 1]) + tx
    Wt2[:, 1] = s * (st * Wf2[:, 0] + ct * Wf2[:, 1]) + ty
    tree2 = cKDTree(Wt2)
    d2, _ = tree2.query(U, k=1)
    n = int(np.sum(d2 < tau))
    if n > max_n:
        max_n = n
        best_info = f"theta={math.degrees(theta):.2f}° s={s:.4f} tx={tx:.1f} ty={ty:.1f}"
        if n >= 20:
            print(f"  trial={trial}: n={n} {best_info}")

print(f"\n1点法最佳(V2中心): max_n={max_n} {best_info}")

# 再测试: 用OBJCTRA/OBJCTDEC中心
print(f"\n--- 1点法测试 (用OBJCTRA中心) ---")
W3 = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)
Wf3 = _apply_flip(W3, result.flip_mode)
norm_Wf3 = np.linalg.norm(Wf3, axis=1)
valid_Wf3 = norm_Wf3 > 1e-10
max_n3 = 0
best_info3 = ""

for trial in range(50000):
    i = rng.integers(0, N_img)
    j = rng.integers(0, M)
    if not valid_U[i] or not valid_Wf3[j]:
        continue
    s = norm_U[i] / norm_Wf3[j]
    if s < 0.9 or s > 1.1:
        continue
    theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf3[j, 1], Wf3[j, 0])
    ct, st = math.cos(theta), math.sin(theta)
    tx = U[i, 0] - s * (ct * Wf3[j, 0] - st * Wf3[j, 1])
    ty = U[i, 1] - s * (st * Wf3[j, 0] + ct * Wf3[j, 1])
    Wt3 = np.empty_like(Wf3)
    Wt3[:, 0] = s * (ct * Wf3[:, 0] - st * Wf3[:, 1]) + tx
    Wt3[:, 1] = s * (st * Wf3[:, 0] + ct * Wf3[:, 1]) + ty
    tree3 = cKDTree(Wt3)
    d3, _ = tree3.query(U, k=1)
    n = int(np.sum(d3 < tau))
    if n > max_n3:
        max_n3 = n
        best_info3 = f"theta={math.degrees(theta):.2f}° s={s:.4f} tx={tx:.1f} ty={ty:.1f}"
        if n >= 20:
            print(f"  trial={trial}: n={n} {best_info3}")

print(f"\n1点法最佳(OBJCTRA中心): max_n={max_n3} {best_info3}")

gaia.close()

# -*- coding: utf-8 -*-
"""T4帧V3.2大tau测试 - 验证宽视场帧1点法+中心修正是否可行"""
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
    _apply_similarity, _count_inliers_1to1, _iterative_svd_refine,
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

print(f"N_img={N_img} M={M} s0={s0:.4f} FOV={fov_diag:.2f}")

# 用大tau做1点法粗匹配，然后中心修正+SVD精修
# 宽视场帧需要大tau来容忍投影畸变
tau_coarse = max(1.0, 5.0 * s0)  # 5×s0 = 30.9角秒
print(f"tau_coarse={tau_coarse:.2f}角秒")

Wf = _apply_flip(W, 0)  # T4帧旋转≈0°, Mode 0
norm_U = np.linalg.norm(U, axis=1)
norm_Wf = np.linalg.norm(Wf, axis=1)
valid_U = norm_U > 1e-10
valid_Wf = norm_Wf > 1e-10

rng = np.random.default_rng(42)
best_n = 0
best_params = None

for _ in range(50000):
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
    n_inliers, rms = _count_inliers_fast(U, Wt, tau_coarse)
    if n_inliers > best_n:
        best_n = n_inliers
        best_params = (s, theta, tx, ty, rms)
        print(f"  新最佳: n={n_inliers} theta={math.degrees(theta):.1f} s={s:.4f} rms={rms:.1f}")

if best_params is None:
    print("未找到任何匹配")
    gaia.close()
    sys.exit(0)

s, theta, tx, ty, rms = best_params
print(f"\n粗匹配: s={s:.4f} theta={math.degrees(theta):.2f}° tx={tx:.1f} ty={ty:.1f} n={best_n} rms={rms:.1f}")

# 中心修正
cos_d0 = math.cos(dec0 * math.pi / 180)
delta_ra = -tx / (cos_d0 * 3600.0)
delta_dec = -ty / 3600.0
new_ra = ra0 + delta_ra
new_dec = dec0 + delta_dec
print(f"中心修正: ΔRA={delta_ra:.6f}° ΔDec={delta_dec:.6f}° → RA={new_ra:.6f} Dec={new_dec:.6f}")

# 重新投影
W_new = _build_catalog_vectors(cat_ra, cat_dec, new_ra, new_dec)
Wf_new = _apply_flip(W_new, 0)

# 用小tau做1点法精修
for tau_mult in [2.5, 5.0, 10.0]:
    tau_try = max(1.0, tau_mult * s0)
    best_n2 = 0
    best_params2 = None
    for _ in range(20000):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        if not valid_U[i] or not valid_Wf[j]:
            continue
        s2 = norm_U[i] / norm_Wf[j]
        if s2 < 0.9 or s2 > 1.1:
            continue
        theta2 = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf_new[j, 1], Wf_new[j, 0])
        ct2, st2 = math.cos(theta2), math.sin(theta2)
        tx2 = U[i, 0] - s2 * (ct2 * Wf_new[j, 0] - st2 * Wf_new[j, 1])
        ty2 = U[i, 1] - s2 * (st2 * Wf_new[j, 0] + ct2 * Wf_new[j, 1])
        Wt2 = np.empty_like(Wf_new)
        Wt2[:, 0] = s2 * (ct2 * Wf_new[:, 0] - st2 * Wf_new[:, 1]) + tx2
        Wt2[:, 1] = s2 * (st2 * Wf_new[:, 0] + ct2 * Wf_new[:, 1]) + ty2
        n2, rms2 = _count_inliers_fast(U, Wt2, tau_try)
        if n2 > best_n2:
            best_n2 = n2
            best_params2 = (s2, theta2, tx2, ty2, rms2)

    if best_params2:
        s2, theta2, tx2, ty2, rms2 = best_params2
        print(f"精修(tau={tau_mult:.1f}x): n={best_n2} theta={math.degrees(theta2):.2f}° s={s2:.4f} rms={rms2:.1f}")
    else:
        print(f"精修(tau={tau_mult:.1f}x): 无匹配")

gaia.close()

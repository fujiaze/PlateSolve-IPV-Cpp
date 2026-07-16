# -*- coding: utf-8 -*-
"""T4帧: θ精度对结果的影响
对比θ=0°和θ=0.5°, 固定s=0.982, 抽样搜索(tx,ty)
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

for mode in [2]:
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    rng = np.random.default_rng(42)

    # 对比不同θ值
    for theta_deg in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
        theta_fixed = theta_deg * math.pi / 180.0
        ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

        Wf_rot = np.empty_like(Wf)
        Wf_rot[:, 0] = ct * Wf[:, 0] - st * Wf[:, 1]
        Wf_rot[:, 1] = st * Wf[:, 0] + ct * Wf[:, 1]

        # 固定s=0.982, 抽样搜索(tx,ty)
        s_try = 0.982
        max_n = 0
        best_tx, best_ty = 0, 0
        for _ in range(20000):
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            tx = U[i, 0] - s_try * Wf_rot[j, 0]
            ty = U[i, 1] - s_try * Wf_rot[j, 1]
            Wt = np.column_stack([
                s_try * Wf_rot[:, 0] + tx,
                s_try * Wf_rot[:, 1] + ty
            ])
            n, rms = _count_inliers_fast(U, Wt, tau)
            if n > max_n:
                max_n = n
                best_tx, best_ty = tx, ty

        print(f"θ={theta_deg:+.1f}°: max_n={max_n} tx={best_tx:.1f} ty={best_ty:.1f}")

    # 也测试: θ=0°, 不同s值
    print(f"\n--- θ=0°, 不同s值 ---")
    theta_fixed = 0.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)
    Wf_rot = np.empty_like(Wf)
    Wf_rot[:, 0] = ct * Wf[:, 0] - st * Wf[:, 1]
    Wf_rot[:, 1] = st * Wf[:, 0] + ct * Wf[:, 1]

    for s_try in [0.980, 0.982, 0.984]:
        max_n = 0
        best_tx, best_ty = 0, 0
        for _ in range(20000):
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            tx = U[i, 0] - s_try * Wf_rot[j, 0]
            ty = U[i, 1] - s_try * Wf_rot[j, 1]
            Wt = np.column_stack([
                s_try * Wf_rot[:, 0] + tx,
                s_try * Wf_rot[:, 1] + ty
            ])
            n, rms = _count_inliers_fast(U, Wt, tau)
            if n > max_n:
                max_n = n
                best_tx, best_ty = tx, ty
        print(f"s={s_try:.3f}: max_n={max_n} tx={best_tx:.1f} ty={best_ty:.1f}")

gaia.close()

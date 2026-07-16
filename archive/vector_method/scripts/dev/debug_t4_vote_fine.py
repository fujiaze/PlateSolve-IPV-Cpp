# -*- coding: utf-8 -*-
"""T4帧: (tx,ty)投票法 - 细bin验证
固定θ=0.5°, 对s=0.982和s=0.986分别做(tx,ty)投票
用1角秒bin, 看正确(tx,ty)是否能在投票中胜出
"""
import sys, os, math, logging, time
import numpy as np
from scipy.spatial import cKDTree
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

V2_TX, V2_TY = 40.81, 134.54

for mode in [2]:
    Wf = _apply_flip(W, mode)

    theta_fixed = 0.5 * math.pi / 180.0  # θ峰值
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

    Wf_rot = np.empty_like(Wf)
    Wf_rot[:, 0] = ct * Wf[:, 0] - st * Wf[:, 1]
    Wf_rot[:, 1] = st * Wf[:, 0] + ct * Wf[:, 1]

    tree_U = cKDTree(U)

    for s_try in [0.982, 0.984, 0.986, 0.988]:
        t0 = time.time()

        # 计算所有(i,j)对的(tx,ty)
        # tx[i,j] = U[i,0] - s*Wf_rot[j,0]
        # ty[i,j] = U[i,1] - s*Wf_rot[j,1]
        tx_all = U[:, 0:1] - s_try * Wf_rot[:, 0]  # (N_img, M)
        ty_all = U[:, 1:2] - s_try * Wf_rot[:, 1]  # (N_img, M)

        tx_flat = tx_all.ravel()
        ty_flat = ty_all.ravel()

        # 用细bin做2D直方图
        # 正确的(tx,ty)应该在V2_TX=40.81, V2_TY=134.54附近
        # 限制搜索范围: tx∈[-200, 200], ty∈[-200, 200]
        mask = (np.abs(tx_flat) < 200) & (np.abs(ty_flat) < 200)
        tx_sub = tx_flat[mask]
        ty_sub = ty_flat[mask]

        # 用2角秒bin
        bin_size = 2.0
        H, tx_edges, ty_edges = np.histogram2d(
            tx_sub, ty_sub,
            bins=[int(400/bin_size), int(400/bin_size)],
            range=[[-200, 200], [-200, 200]]
        )

        peak_val = H.max()
        peak_idx_flat = np.argmax(H)
        peak_i, peak_j = np.unravel_index(peak_idx_flat, H.shape)
        tx_peak = (tx_edges[peak_i] + tx_edges[peak_i+1]) / 2
        ty_peak = (ty_edges[peak_j] + ty_edges[peak_j+1]) / 2

        # 用峰值(tx,ty)计算内点数
        Wt = np.column_stack([
            s_try * Wf_rot[:, 0] + tx_peak,
            s_try * Wf_rot[:, 1] + ty_peak
        ])
        pairs = tree_U.query_ball_point(Wt, tau)
        n_inliers = sum(1 for p in pairs if len(p) > 0)

        # 也用随机抽样法对比
        rng = np.random.default_rng(42)
        max_n_sample = 0
        best_tx_s, best_ty_s = 0, 0
        for _ in range(20000):
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            tx = U[i, 0] - s_try * Wf_rot[j, 0]
            ty = U[i, 1] - s_try * Wf_rot[j, 1]
            Wt_s = np.column_stack([
                s_try * Wf_rot[:, 0] + tx,
                s_try * Wf_rot[:, 1] + ty
            ])
            n_s, _ = _count_inliers_fast(U, Wt_s, tau)
            if n_s > max_n_sample:
                max_n_sample = n_s
                best_tx_s, best_ty_s = tx, ty

        t1 = time.time()
        print(f"\ns={s_try:.3f}:")
        print(f"  投票法: n={n_inliers} tx={tx_peak:.1f} ty={ty_peak:.1f} (峰值={peak_val:.0f})")
        print(f"  抽样法: n={max_n_sample} tx={best_tx_s:.1f} ty={best_ty_s:.1f}")
        print(f"  V2参考: tx={V2_TX} ty={V2_TY}")
        print(f"  耗时: {t1-t0:.1f}s")

        # 检查V2参考(tx,ty)附近的投票数
        tx_bin = int((V2_TX + 200) / bin_size)
        ty_bin = int((V2_TY + 200) / bin_size)
        if 0 <= tx_bin < H.shape[0] and 0 <= ty_bin < H.shape[1]:
            v2_votes = H[tx_bin, ty_bin]
            print(f"  V2参考位置投票数: {v2_votes:.0f}")

gaia.close()

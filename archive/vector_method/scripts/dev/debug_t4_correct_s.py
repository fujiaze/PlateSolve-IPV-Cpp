# -*- coding: utf-8 -*-
"""T4帧: 分析正确匹配对的1点法s分布
用V2的正确变换, 找到正确匹配对, 然后看1点法计算的s分布

核心问题: 1点法的s=|u_i|/|w_j|, 正确对的s是否接近真实s=0.983?
如果不接近, 那么从1点法样本中无法恢复真实s
"""
import sys, os, math, logging
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

# V2正确变换参数
V2_S, V2_THETA, V2_TX, V2_TY = 0.983, 0.0, 40.81, 134.54

for mode in [2]:
    Wf = _apply_flip(W, mode)

    # 用V2正确变换计算Wt
    ct, st = math.cos(V2_THETA * math.pi / 180), math.sin(V2_THETA * math.pi / 180)
    Wt = np.empty_like(Wf)
    Wt[:, 0] = V2_S * (ct * Wf[:, 0] - st * Wf[:, 1]) + V2_TX
    Wt[:, 1] = V2_S * (st * Wf[:, 0] + ct * Wf[:, 1]) + V2_TY

    # 找正确匹配对 (1对1, tau=2.5*s0)
    tau = max(1.0, 2.5 * s0)
    tree_U = cKDTree(U)
    tree_Wt = cKDTree(Wt)

    # U -> Wt 匹配
    dist_UW, idx_UW = tree_Wt.query(U, k=1)
    # Wt -> U 匹配
    dist_WU, idx_WU = tree_U.query(Wt, k=1)

    # 1对1互斥匹配
    correct_pairs = []
    for i in range(N_img):
        j = idx_UW[i]
        if dist_UW[i] < tau and idx_WU[j] == i:
            correct_pairs.append((i, j, dist_UW[i]))

    print(f"\n正确匹配对数: {len(correct_pairs)} (tau={tau:.1f})")

    # 分析正确对的1点法s
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)

    s_1point = []
    for i, j, d in correct_pairs:
        s = norm_U[i] / norm_Wf[j]
        s_1point.append((s, d, i, j))

    s_1point.sort(key=lambda x: x[0])
    s_vals = np.array([x[0] for x in s_1point])
    d_vals = np.array([x[1] for x in s_1point])

    print(f"\n正确对1点法s统计:")
    print(f"  n={len(s_vals)}")
    print(f"  mean={np.mean(s_vals):.4f} median={np.median(s_vals):.4f} std={np.std(s_vals):.4f}")
    print(f"  min={np.min(s_vals):.4f} max={np.max(s_vals):.4f}")
    print(f"  真实s={V2_S}")
    print(f"  偏差: mean={np.mean(s_vals)-V2_S:.4f} median={np.median(s_vals)-V2_S:.4f}")

    # s分位数
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}%: {np.percentile(s_vals, p):.4f}")

    # 按距离分组
    print(f"\n按投影误差分组:")
    for d_thresh in [2, 5, 10, 20]:
        mask = d_vals < d_thresh
        if np.sum(mask) > 0:
            s_sub = s_vals[mask]
            print(f"  d<{d_thresh}px: n={np.sum(mask)} s_mean={np.mean(s_sub):.4f} s_median={np.median(s_sub):.4f}")

    # 用正确对的s中位数做变换, 看效果
    s_med = np.median(s_vals)
    theta_fixed = V2_THETA * math.pi / 180.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

    # 用s_med和V2的θ, 随机抽样(tx,ty)
    rng = np.random.default_rng(42)
    max_n = 0
    best_tx, best_ty = 0, 0
    for _ in range(50000):
        i = rng.integers(0, N_img)
        j = rng.integers(0, M)
        tx = U[i, 0] - s_med * (ct * Wf[j, 0] - st * Wf[j, 1])
        ty = U[i, 1] - s_med * (st * Wf[j, 0] + ct * Wf[j, 1])
        Wt_try = np.empty_like(Wf)
        Wt_try[:, 0] = s_med * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt_try[:, 1] = s_med * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        n, rms = 0, 0
        # 快速内点计数
        dists, _ = tree_U.query(Wt_try, k=1)
        n = np.sum(dists < tau)
        if n > max_n:
            max_n = n
            best_tx, best_ty = tx, ty

    print(f"\n用正确对s中位数={s_med:.4f}搜索(tx,ty):")
    print(f"  max_n={max_n} tx={best_tx:.1f} ty={best_ty:.1f}")
    print(f"  V2参考: tx={V2_TX} ty={V2_TY}")

    # 分析: 为什么1点法的s偏离真实s?
    print(f"\n--- gnomonic投影畸变分析 ---")
    # 正确对的u_i和w_j的模长
    for rank, (s, d, i, j) in enumerate(s_1point[:5]):
        u_mod = norm_U[i]
        w_mod = norm_Wf[j]
        print(f"  对{rank}: s={s:.4f} |u|={u_mod:.1f} |w|={w_mod:.1f} d={d:.1f}px")

    # u和w的模长范围
    print(f"\n  |u|范围: [{norm_U.min():.1f}, {norm_U.max():.1f}] median={np.median(norm_U):.1f}")
    print(f"  |w|范围: [{norm_Wf.min():.1f}, {norm_Wf.max():.1f}] median={np.median(norm_Wf):.1f}")

    # 正确对的u和w模长
    correct_u_mod = norm_U[[x[2] for x in s_1point]]
    correct_w_mod = norm_Wf[[x[3] for x in s_1point]]
    print(f"  正确对|u|: [{correct_u_mod.min():.1f}, {correct_u_mod.max():.1f}] median={np.median(correct_u_mod):.1f}")
    print(f"  正确对|w|: [{correct_w_mod.min():.1f}, {correct_w_mod.max():.1f}] median={np.median(correct_w_mod):.1f}")

gaia.close()

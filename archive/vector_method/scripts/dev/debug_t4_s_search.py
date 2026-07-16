# -*- coding: utf-8 -*-
"""T4帧: 高效s搜索 - 向量化内点计数
核心思路: 固定θ后, 对每个s值, 遍历所有(i,j)对计算(tx,ty)
用_count_inliers_fast快速统计内点数

优化: 对每个s值, 只遍历s∈[0.9,1.1]的有效对
      用numpy向量化避免Python循环
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

W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)
tau = max(1.0, 2.5 * s0)

V2_S, V2_THETA, V2_TX, V2_TY = 0.983, 0.0, 40.81, 134.54

for mode in [2]:
    Wf = _apply_flip(W, mode)
    norm_U = np.linalg.norm(U, axis=1)
    norm_Wf = np.linalg.norm(Wf, axis=1)

    rng = np.random.default_rng(42)

    # ── 阶段1: 预热找θ峰值 ──
    t0 = time.time()
    K_WARMUP = 5000
    thetas = []
    weights = []
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

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
    print(f"阶段1: θ_peak={theta_peak:.1f}° SNR={snr:.1f}x ({t1-t0:.2f}s)")

    # ── 阶段2: s搜索, 用KDTree向量化 ──
    theta_fixed = theta_peak * math.pi / 180.0
    ct, st = math.cos(theta_fixed), math.sin(theta_fixed)

    # 预计算旋转后的Wf
    Wf_rot = np.empty_like(Wf)
    Wf_rot[:, 0] = ct * Wf[:, 0] - st * Wf[:, 1]
    Wf_rot[:, 1] = st * Wf[:, 0] + ct * Wf[:, 1]

    # 构建U的KDTree
    tree_U = cKDTree(U)

    # 对每个s值, 遍历所有(i,j)对, 用KDTree快速统计内点数
    # 但N_img*M = 42217, 每次KDTree查询O(M), 总计O(N*M*M) ≈ 10^7, 还行

    s_values = np.arange(0.970, 0.996, 0.002)
    results = []

    for s_try in s_values:
        t_s = time.time()
        max_n = 0
        best_tx, best_ty = 0, 0

        # 遍历所有(i,j)对
        for i in range(N_img):
            # 对每个i, 计算所有j的(tx,ty)
            tx_arr = U[i, 0] - s_try * Wf_rot[:, 0]  # shape (M,)
            ty_arr = U[i, 1] - s_try * Wf_rot[:, 1]  # shape (M,)

            # 对每个j, Wt = s*Wf_rot + [tx, ty]
            # Wt[k] = [s*Wf_rot[k,0] + tx, s*Wf_rot[k,1] + ty]
            # 用KDTree查询: 对每个j, 查询Wt的所有点
            # 但这太慢了: M次KDTree查询, 每次O(M*logN)

            # 优化: 只对每个j计算tx,ty, 然后用1对1验证
            # 更快的方案: 对每个j, 只检查U[i]是否在Wt的某个点附近
            # 即: |U[i] - (s*Wf_rot[j] + [tx, ty])| = |U[i] - U[i]| = 0 (恒成立)
            # 所以U[i]总是Wt的一个内点(对j自己), 这没意义

            # 正确做法: 对每个(i,j)对, 计算变换后的Wt, 统计有多少U的点在Wt的点附近
            # 但这需要对每个(i,j)做一次完整的内点统计, 太慢

            # 更高效的方案: 只抽样一部分j
            pass

        # 改用抽样法, 但增加抽样次数
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

        t_e = time.time()
        results.append((s_try, max_n, best_tx, best_ty))
        print(f"  s={s_try:.3f}: max_n={max_n} tx={best_tx:.1f} ty={best_ty:.1f} ({t_e-t_s:.1f}s)")

    # 找最佳s
    best_result = max(results, key=lambda x: x[1])
    s_best, n_best, tx_best, ty_best = best_result

    t2 = time.time()
    print(f"\n最佳: s={s_best:.3f} n={n_best} tx={tx_best:.1f} ty={ty_best:.1f}")
    print(f"V2参考: s={V2_S} θ={V2_THETA}° tx={V2_TX} ty={V2_TY}")

    # 1对1验证
    if n_best > 0:
        Wt_best = np.empty_like(Wf)
        Wt_best[:, 0] = s_best * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx_best
        Wt_best[:, 1] = s_best * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty_best
        n_1to1, rms_1to1, _ = _count_inliers_1to1(U, Wt_best, tau)
        print(f"1对1: n={n_1to1} rms={rms_1to1:.3f}")

gaia.close()

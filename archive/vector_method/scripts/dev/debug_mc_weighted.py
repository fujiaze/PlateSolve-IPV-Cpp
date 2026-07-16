# -*- coding: utf-8 -*-
"""
蒙特卡洛加权抽样分析
思路：随机抽1个(u_i, w_j)对，若s∈[0.9,1.1]则计算变换，
      应用变换后统计内点数作为权重，加权theta直方图
      正确变换应有大量内点，噪声变换内点极少
"""
import sys, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'astro_image_io', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'star_detector', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'gaia_client', 'python'))

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import (
    GaiaClientPy, gnomonic_forward, bisection_mag_limit,
    _build_image_vectors, _build_catalog_vectors, _apply_flip,
    _count_inliers_1to1,
)

S_RANGE = (0.9, 1.1)
N_MC = 50000       # 蒙特卡洛抽样次数
TAU_INLIER = None   # 内点阈值，None=自动(2.5*s0)


def analyze_frame(fits_path, gaia_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(fits_path))[0]

    r = ImageReader()
    det = StarDetector(params=SDetParamsPy(fitRadius=0))
    gaia = GaiaClientPy(gaia_dir, db_type=0)

    img = r.read(fits_path)
    w, h = img.width, img.height
    ra0 = img.metadata.wcs.crval1
    dec0 = img.metadata.wcs.crval2
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    tau = TAU_INLIER if TAU_INLIER else max(1.0, 2.5 * s0)

    # WCS旋转角
    wcs_rot = None
    cd11 = cd12 = None
    for kw in img.keywords:
        if kw.name == 'CD1_1': cd11 = float(kw.value)
        elif kw.name == 'CD1_2': cd12 = float(kw.value)
    if cd11 is not None and cd12 is not None:
        wcs_rot = math.degrees(math.atan2(cd12, cd11))

    d = det.detect_ex(img.data)
    img_x = np.array(d.x, dtype=np.float64)
    img_y = np.array(d.y, dtype=np.float64)
    img_flux = np.array(d.flux, dtype=np.float64)
    img_saturated = np.array(d.saturated, dtype=np.int32)

    U, N_img, n_sat, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_saturated, s0, w, h)

    fov_diag = math.sqrt(w**2 + h**2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0
    N_gaia = math.ceil(1.5 * n_sat)
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        gaia, ra0, dec0, radius_deg, N_gaia)
    W = _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0)

    # 预建U的KDTree
    tree_U = cKDTree(U)
    norm_U = np.linalg.norm(U, axis=1)

    print(f"=== {basename} ===")
    print(f"s0={s0:.4f}″/px tau={tau:.2f}″ WCS旋转={wcs_rot:.2f}°" if wcs_rot else f"s0={s0:.4f}″/px tau={tau:.2f}″")
    print(f"N_img={N_img} M={M} 抽样={N_MC}")

    rng = np.random.default_rng(42)

    # 对4种翻转模式分别分析
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle(
        f'{basename}\nWeighted MC Sampling: inlier_count as weight '
        f'(N={N_img}, M={M}, s0={s0:.3f}, tau={tau:.2f}″, '
        f'WCS={wcs_rot:.1f}°)' if wcs_rot else
        f'{basename}\nWeighted MC Sampling (N={N_img}, M={M}, s0={s0:.3f}, tau={tau:.2f}″)',
        fontsize=12)

    for mode in range(4):
        Wf = _apply_flip(W, mode)
        norm_Wf = np.linalg.norm(Wf, axis=1)
        valid_Wf = norm_Wf > 1e-10

        # 蒙特卡洛抽样
        thetas = []
        weights = []  # 内点数
        s_values = []
        tx_values = []
        ty_values = []

        n_tried = 0
        n_s_in_range = 0

        for _ in range(N_MC):
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            if norm_U[i] < 1e-10 or not valid_Wf[j]:
                continue
            n_tried += 1

            s = norm_U[i] / norm_Wf[j]
            if s < S_RANGE[0] or s > S_RANGE[1]:
                continue
            n_s_in_range += 1

            theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
            ct, st = math.cos(theta), math.sin(theta)
            tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
            ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])

            # 变换全部Wf
            Wt = np.empty_like(Wf)
            Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
            Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty

            # 统计内点（1对1互斥）
            inl = _count_inliers_1to1(U, Wt, tau)
            n_inliers = inl[0]
            rms = inl[1]

            thetas.append(((math.degrees(theta) + 180) % 360) - 180)
            weights.append(n_inliers)
            s_values.append(s)
            tx_values.append(tx)
            ty_values.append(ty)

        thetas = np.array(thetas)
        weights = np.array(weights)
        s_values = np.array(s_values)

        print(f"\nMode {mode}: 尝试={n_tried} s≈1={n_s_in_range} ({n_s_in_range/max(n_tried,1)*100:.1f}%)")
        if len(thetas) == 0:
            print("  无s≈1的抽样")
            continue

        # 基本统计
        print(f"  内点数: mean={np.mean(weights):.1f} max={np.max(weights)} "
              f"median={np.median(weights):.0f} >10={np.sum(weights>10)} >20={np.sum(weights>20)}")

        # ── 无权重theta直方图 ──
        ax0 = axes[mode, 0]
        counts, bin_edges, _ = ax0.hist(thetas, bins=360, density=False, alpha=0.7, color='steelblue')
        if wcs_rot is not None:
            ref = wcs_rot if mode in (0, 1, 2) else wcs_rot + 180
            ref = ((ref + 180) % 360) - 180
            ax0.axvline(ref, color='red', linestyle='--', linewidth=2, label=f'WCS={ref:.1f}°')
        ax0.legend(fontsize=7)
        ax0.set_title(f'Mode {mode}: θ unweighted (n={len(thetas)})', fontsize=9)
        ax0.set_xlabel('θ (°)')
        ax0.set_ylabel('Count')

        # ── 加权theta直方图（权重=内点数）──
        ax1 = axes[mode, 1]
        # 手动构建加权直方图
        bin_w = 360 / 360  # 1度/bin
        weighted_counts = np.zeros(360)
        for k in range(len(thetas)):
            bin_idx = int((thetas[k] + 180) / bin_w)
            if 0 <= bin_idx < 360:
                weighted_counts[bin_idx] += weights[k]
        bin_centers = np.arange(-180, 180) + 0.5
        ax1.bar(bin_centers, weighted_counts, width=1.0, alpha=0.7, color='crimson')
        if wcs_rot is not None:
            ax1.axvline(ref, color='blue', linestyle='--', linewidth=2, label=f'WCS={ref:.1f}°')
        # 标记峰值
        peak_idx = np.argmax(weighted_counts)
        peak_theta = bin_centers[peak_idx]
        ax1.axvline(peak_theta, color='green', linestyle='-', linewidth=1.5, label=f'peak={peak_theta:.1f}°')
        ax1.legend(fontsize=7)
        # 峰值SNR
        bg = np.median(weighted_counts[weighted_counts > 0]) if np.sum(weighted_counts > 0) > 10 else 1
        peak_snr = weighted_counts[peak_idx] / bg
        ax1.set_title(f'Mode {mode}: θ weighted by inliers (SNR={peak_snr:.1f}x)', fontsize=9)
        ax1.set_xlabel('θ (°)')
        ax1.set_ylabel('Sum of inlier counts')

        # ── 内点数分布 ──
        ax2 = axes[mode, 2]
        ax2.hist(weights, bins=range(0, int(np.max(weights)) + 2), density=False,
                 alpha=0.7, color='forestgreen')
        ax2.set_title(f'Mode {mode}: inlier count distribution', fontsize=9)
        ax2.set_xlabel('n_inliers')
        ax2.set_ylabel('Count')

        # ── theta vs 内点数散点 ──
        ax3 = axes[mode, 3]
        sample_n = min(10000, len(thetas))
        sample_idx = rng.choice(len(thetas), sample_n, replace=False)
        sc = ax3.scatter(thetas[sample_idx], weights[sample_idx],
                         c=s_values[sample_idx], cmap='viridis', s=2, alpha=0.3)
        plt.colorbar(sc, ax=ax3, label='s')
        if wcs_rot is not None:
            ax3.axvline(ref, color='red', linestyle='--', linewidth=1.5)
        ax3.set_title(f'Mode {mode}: θ vs n_inliers', fontsize=9)
        ax3.set_xlabel('θ (°)')
        ax3.set_ylabel('n_inliers')

        # 打印峰值附近统计
        near_peak = np.abs(thetas - peak_theta) < 5
        if np.sum(near_peak) > 0:
            peak_weights = weights[near_peak]
            peak_tx = np.array(tx_values)[near_peak]
            peak_ty = np.array(ty_values)[near_peak]
            best_idx = np.argmax(peak_weights)
            print(f"  峰值θ={peak_theta:.1f}°±5°: n_samples={np.sum(near_peak)} "
                  f"max_inliers={np.max(peak_weights)} "
                  f"mean_inliers={np.mean(peak_weights):.1f}")
            print(f"  最佳变换: θ={thetas[near_peak][best_idx]:.2f}° "
                  f"s={s_values[near_peak][best_idx]:.4f} "
                  f"tx={peak_tx[best_idx]:.1f}″ ty={peak_ty[best_idx]:.1f}″ "
                  f"inliers={peak_weights[best_idx]}")

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{basename}_mc_weighted.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {out_path}")

    gaia.close()


if __name__ == '__main__':
    GAIA_DIR = os.path.join(PROJECT_ROOT, 'GaiaDR3SP')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'debug_output')

    # T2帧（旋转~90度）
    t2_path = os.path.join(PROJECT_ROOT, 'testdata', 'lights',
                           'M20_T2_flying_dutchman-20250701@073331-300S-Red.fts')
    if os.path.exists(t2_path):
        analyze_frame(t2_path, GAIA_DIR, OUTPUT_DIR)

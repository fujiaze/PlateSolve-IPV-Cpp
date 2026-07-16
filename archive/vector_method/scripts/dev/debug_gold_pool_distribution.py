# -*- coding: utf-8 -*-
"""
黄金池变换对分布分析脚本（中心向量法）
功能：用从质心出发的向量，对每个(u_i, w_j)对计算变换参数(s, theta, tx, ty)
      筛选s≈1的对，绘制theta和位移的分布
原理：刚体变换 U = s*R*W + t，中心化后 u_c = s*R*w_c
      正确匹配的(s, theta)必然一致，错误匹配随机分布
"""
import sys, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
)

S0_RANGE = (0.9, 1.1)


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

    print(f"=== {basename} ===")
    print(f"s0={s0:.4f}″/px WCS旋转={wcs_rot:.2f}°" if wcs_rot else f"s0={s0:.4f}″/px")
    print(f"黄金池: n_sat={n_sat} Gaia: M={M}")

    # 黄金池：只用饱和星
    sat_mask = img_saturated.astype(bool)
    gold_indices = np.where(sat_mask)[0]
    n_gold = len(gold_indices)
    U_gold = U[gold_indices]

    # 质心
    U_center = U_gold.mean(axis=0)
    W_center = W.mean(axis=0)

    # 中心化向量
    U_c = U_gold - U_center  # (n_gold, 2)
    W_c = W - W_center       # (M, 2)

    print(f"U质心: ({U_center[0]:.1f}, {U_center[1]:.1f})″")
    print(f"W质心: ({W_center[0]:.1f}, {W_center[1]:.1f})″")

    # 对4种翻转模式分别分析
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle(
        f'{basename}\nCenter-Vector Transform Analysis '
        f'(n_gold={n_gold}, M={M}, s0={s0:.3f}, WCS_rot={wcs_rot:.1f}°)' if wcs_rot else
        f'{basename}\nCenter-Vector Transform Analysis (n_gold={n_gold}, M={M}, s0={s0:.3f})',
        fontsize=13)

    for mode in range(4):
        Wf = _apply_flip(W, mode)
        Wf_c = Wf - Wf.mean(axis=0)  # 翻转后重新中心化

        # 对每个(u_i, w_j)对，用中心化向量计算变换参数
        # u_c_i = s * R(theta) * wf_c_j
        # s = |u_c_i| / |wf_c_j|
        # theta = atan2(u_c_i_y, u_c_i_x) - atan2(wf_c_j_y, wf_c_j_x)

        norm_u = np.linalg.norm(U_c, axis=1)  # (n_gold,)
        norm_wf = np.linalg.norm(Wf_c, axis=1)  # (M,)

        # 过滤零向量
        valid_u = norm_u > 1e-10
        valid_wf = norm_wf > 1e-10

        # 全部 n_gold * M 个对
        all_s = np.zeros(n_gold * M)
        all_theta = np.zeros(n_gold * M)
        all_tx = np.zeros(n_gold * M)
        all_ty = np.zeros(n_gold * M)
        idx = 0

        for i in range(n_gold):
            if not valid_u[i]:
                continue
            for j in range(M):
                if not valid_wf[j]:
                    continue
                s = norm_u[i] / norm_wf[j]
                theta = math.atan2(U_c[i, 1], U_c[i, 0]) - math.atan2(Wf_c[j, 1], Wf_c[j, 0])
                ct, st = math.cos(theta), math.sin(theta)
                tx = U_gold[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
                ty = U_gold[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
                all_s[idx] = s
                all_theta[idx] = math.degrees(theta)
                all_tx[idx] = tx
                all_ty[idx] = ty
                idx += 1

        all_s = all_s[:idx]
        all_theta = all_theta[:idx]
        all_tx = all_tx[:idx]
        all_ty = all_ty[:idx]

        # 归一化theta到[-180, 180]
        all_theta = ((all_theta + 180) % 360) - 180

        # 筛选s∈[0.9, 1.1]
        mask_s = (all_s >= S0_RANGE[0]) & (all_s <= S0_RANGE[1])
        s_filtered = all_s[mask_s]
        theta_filtered = all_theta[mask_s]
        tx_filtered = all_tx[mask_s]
        ty_filtered = all_ty[mask_s]
        disp_filtered = np.sqrt(tx_filtered**2 + ty_filtered**2)

        print(f"\nMode {mode}: 全部对={len(all_s)} s∈[0.9,1.1]={np.sum(mask_s)} ({np.sum(mask_s)/len(all_s)*100:.1f}%)")
        if len(theta_filtered) == 0:
            print("  无s≈1的对")
            continue

        # theta统计
        print(f"  theta(s≈1): mean={np.mean(theta_filtered):.2f}° std={np.std(theta_filtered):.2f}° "
              f"median={np.median(theta_filtered):.2f}°")
        # 找theta的众数（用直方图峰值）
        hist, bin_edges = np.histogram(theta_filtered, bins=360)
        peak_idx = np.argmax(hist)
        peak_theta = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        peak_count = hist[peak_idx]
        print(f"  theta峰值: {peak_theta:.1f}° (count={peak_count}, 占{peak_count/len(theta_filtered)*100:.2f}%)")

        # 在峰值±5度范围内的统计
        near_peak = np.abs(theta_filtered - peak_theta) < 5
        if np.sum(near_peak) > 0:
            print(f"  峰值±5°: n={np.sum(near_peak)} s_mean={np.mean(s_filtered[near_peak]):.4f} "
                  f"tx_mean={np.mean(tx_filtered[near_peak]):.1f}″ ty_mean={np.mean(ty_filtered[near_peak]):.1f}″")

        # 绘制4个子图：theta分布、s分布、位移分布、theta vs tx散点
        ax1 = axes[mode, 0]
        ax1.hist(theta_filtered, bins=360, density=False, alpha=0.7, color='steelblue')
        if wcs_rot is not None:
            ref = wcs_rot if mode in (0, 1) else (wcs_rot + 180 if mode == 3 else wcs_rot)
            ref = ((ref + 180) % 360) - 180
            ax1.axvline(ref, color='red', linestyle='--', linewidth=2, label=f'WCS={ref:.1f}°')
            ax1.legend(fontsize=8)
        ax1.axvline(peak_theta, color='green', linestyle='-', linewidth=1.5, label=f'peak={peak_theta:.1f}°')
        ax1.legend(fontsize=7)
        ax1.set_title(f'Mode {mode}: θ (s≈1, n={len(theta_filtered)})', fontsize=10)
        ax1.set_xlabel('θ (degrees)')
        ax1.set_ylabel('Count')

        ax2 = axes[mode, 1]
        ax2.hist(s_filtered, bins=100, density=True, alpha=0.7, color='forestgreen')
        ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5)
        ax2.set_title(f'Mode {mode}: s distribution', fontsize=10)
        ax2.set_xlabel('s')
        ax2.set_ylabel('Density')

        ax3 = axes[mode, 2]
        ax3.hist(disp_filtered, bins=200, density=False, alpha=0.7, color='darkorange')
        if np.sum(near_peak) > 0:
            disp_peak = disp_filtered[near_peak]
            ax3.axvline(np.median(disp_peak), color='red', linestyle='--',
                        label=f'peak_θ median={np.median(disp_peak):.1f}″')
            ax3.legend(fontsize=7)
        ax3.set_title(f'Mode {mode}: |displacement|', fontsize=10)
        ax3.set_xlabel('displacement (arcsec)')
        ax3.set_ylabel('Count')

        ax4 = axes[mode, 3]
        # theta vs displacement散点图（采样避免太密）
        sample = np.random.default_rng(42).choice(len(theta_filtered),
                                                    min(5000, len(theta_filtered)), replace=False)
        sc = ax4.scatter(theta_filtered[sample], disp_filtered[sample],
                         c=s_filtered[sample], cmap='viridis', s=1, alpha=0.3)
        plt.colorbar(sc, ax=ax4, label='s')
        if wcs_rot is not None:
            ax4.axvline(ref, color='red', linestyle='--', linewidth=1.5)
        ax4.set_title(f'Mode {mode}: θ vs |disp|', fontsize=10)
        ax4.set_xlabel('θ (degrees)')
        ax4.set_ylabel('|displacement| (arcsec)')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{basename}_center_vector_analysis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {out_path}")

    # 额外：对最佳模式绘制theta的精细直方图（±10度范围）
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    fig2.suptitle(f'{basename} - θ Fine Distribution (±30° around peak)', fontsize=13)

    for mode in range(4):
        Wf = _apply_flip(W, mode)
        Wf_c = Wf - Wf.mean(axis=0)
        norm_wf = np.linalg.norm(Wf_c, axis=1)

        thetas_s1 = []
        txs_s1 = []
        tys_s1 = []

        for i in range(n_gold):
            if norm_u[i] < 1e-10:
                continue
            for j in range(M):
                if norm_wf[j] < 1e-10:
                    continue
                s = norm_u[i] / norm_wf[j]
                if s < S0_RANGE[0] or s > S0_RANGE[1]:
                    continue
                theta = math.atan2(U_c[i, 1], U_c[i, 0]) - math.atan2(Wf_c[j, 1], Wf_c[j, 0])
                theta_deg = ((math.degrees(theta) + 180) % 360) - 180
                ct, st = math.cos(theta), math.sin(theta)
                tx = U_gold[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
                ty = U_gold[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
                thetas_s1.append(theta_deg)
                txs_s1.append(tx)
                tys_s1.append(ty)

        thetas_s1 = np.array(thetas_s1)
        txs_s1 = np.array(txs_s1)
        tys_s1 = np.array(tys_s1)

        if len(thetas_s1) == 0:
            continue

        # 找峰值
        hist, edges = np.histogram(thetas_s1, bins=360)
        peak_i = np.argmax(hist)
        peak_t = (edges[peak_i] + edges[peak_i + 1]) / 2

        # ±30度范围
        near = np.abs(thetas_s1 - peak_t) < 30
        if np.sum(near) < 10:
            near = np.abs(thetas_s1 - peak_t) < 90

        axes2[mode].hist(thetas_s1[near], bins=120, density=False, alpha=0.7, color='steelblue')
        if wcs_rot is not None:
            ref = wcs_rot if mode in (0, 1) else (wcs_rot + 180 if mode == 3 else wcs_rot)
            ref = ((ref + 180) % 360) - 180
            axes2[mode].axvline(ref, color='red', linestyle='--', linewidth=2, label=f'WCS={ref:.1f}°')
        axes2[mode].axvline(peak_t, color='green', linestyle='-', linewidth=1.5, label=f'peak={peak_t:.1f}°')
        axes2[mode].legend(fontsize=8)
        axes2[mode].set_title(f'Mode {mode}', fontsize=10)
        axes2[mode].set_xlabel('θ (degrees)')
        axes2[mode].set_ylabel('Count')

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, f'{basename}_theta_fine.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"精细图已保存: {out_path2}")

    gaia.close()


if __name__ == '__main__':
    GAIA_DIR = os.path.join(PROJECT_ROOT, 'GaiaDR3SP')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'debug_output')

    # T2帧（旋转~90度）
    t2_path = os.path.join(PROJECT_ROOT, 'testdata', 'lights',
                           'M20_T2_flying_dutchman-20250701@073331-300S-Red.fts')
    if os.path.exists(t2_path):
        analyze_frame(t2_path, GAIA_DIR, OUTPUT_DIR)

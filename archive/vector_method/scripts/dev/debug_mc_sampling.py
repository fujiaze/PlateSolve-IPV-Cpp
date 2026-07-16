# -*- coding: utf-8 -*-
"""
蒙特卡洛单点变换抽样分析
思路：随机抽1个图像侧向量u_i + 1个星表侧向量w_j，
      计算变换参数(s, theta, tx, ty)，
      大量抽样后观察分布是否有峰值
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

S_RANGE = (0.9, 1.1)
N_SAMPLES = 200000  # 蒙特卡洛抽样次数


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
    print(f"N_img={N_img} n_sat={n_sat} M={M} 抽样次数={N_SAMPLES}")

    rng = np.random.default_rng(42)

    # 预计算U和W的模
    norm_U = np.linalg.norm(U, axis=1)   # (N_img,)
    valid_U = norm_U > 1e-10

    # 对4种翻转模式分别分析
    fig, axes = plt.subplots(4, 5, figsize=(28, 20))
    fig.suptitle(
        f'{basename}\nMonte Carlo 1-Point Sampling '
        f'(N_img={N_img}, M={M}, s0={s0:.3f}, N_samples={N_SAMPLES}, '
        f'WCS_rot={wcs_rot:.1f}°)' if wcs_rot else
        f'{basename}\nMonte Carlo 1-Point Sampling '
        f'(N_img={N_img}, M={M}, s0={s0:.3f}, N_samples={N_SAMPLES})',
        fontsize=12)

    for mode in range(4):
        Wf = _apply_flip(W, mode)
        norm_Wf = np.linalg.norm(Wf, axis=1)  # (M,)
        valid_Wf = norm_Wf > 1e-10

        # 蒙特卡洛抽样
        s_list = []
        theta_list = []
        tx_list = []
        ty_list = []
        disp_list = []

        for _ in range(N_SAMPLES):
            # 随机选1个U和1个Wf
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            if not valid_U[i] or not valid_Wf[j]:
                continue

            # 变换参数：U_i = s * R(theta) * Wf_j + (tx, ty)
            # s = |U_i| / |Wf_j|
            s = norm_U[i] / norm_Wf[j]
            # theta = atan2(U_i.y, U_i.x) - atan2(Wf_j.y, Wf_j.x)
            theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
            ct, st = math.cos(theta), math.sin(theta)
            tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
            ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])

            s_list.append(s)
            theta_list.append(math.degrees(theta))
            tx_list.append(tx)
            ty_list.append(ty)
            disp_list.append(math.sqrt(tx**2 + ty**2))

        s_arr = np.array(s_list)
        theta_arr = np.array(theta_list)
        tx_arr = np.array(tx_list)
        ty_arr = np.array(ty_list)
        disp_arr = np.array(disp_list)

        # 归一化theta到[-180, 180]
        theta_arr = ((theta_arr + 180) % 360) - 180

        # 筛选s∈[0.9, 1.1]
        mask_s = (s_arr >= S_RANGE[0]) & (s_arr <= S_RANGE[1])
        n_in_range = np.sum(mask_s)
        pct = n_in_range / len(s_arr) * 100

        s_f = s_arr[mask_s]
        theta_f = theta_arr[mask_s]
        tx_f = tx_arr[mask_s]
        ty_f = ty_arr[mask_s]
        disp_f = disp_arr[mask_s]

        print(f"\nMode {mode}: 总抽样={len(s_arr)} s∈[0.9,1.1]={n_in_range} ({pct:.1f}%)")

        if len(theta_f) == 0:
            print("  无s≈1的抽样")
            continue

        # theta峰值检测
        hist_t, edges_t = np.histogram(theta_f, bins=360)
        peak_idx = np.argmax(hist_t)
        peak_theta = (edges_t[peak_idx] + edges_t[peak_idx + 1]) / 2
        peak_count = hist_t[peak_idx]
        # 峰值附近±2度的平均计数（排除峰值本身）
        bg_mask = np.ones(360, dtype=bool)
        bg_mask[max(0, peak_idx-3):min(360, peak_idx+4)] = False
        bg_mean = np.mean(hist_t[bg_mask]) if np.sum(bg_mask) > 0 else 0
        snr = peak_count / bg_mean if bg_mean > 0 else 0

        print(f"  theta峰值: {peak_theta:.1f}° count={peak_count} bg_mean={bg_mean:.1f} SNR={snr:.1f}x")

        # 峰值±5°统计
        near_peak = np.abs(theta_f - peak_theta) < 5
        if np.sum(near_peak) > 0:
            print(f"  峰值±5°: n={np.sum(near_peak)} "
                  f"s={np.mean(s_f[near_peak]):.4f} "
                  f"tx={np.mean(tx_f[near_peak]):.1f}″ ty={np.mean(ty_f[near_peak]):.1f}″ "
                  f"disp={np.median(disp_f[near_peak]):.1f}″")

        # ── 绘图 ──
        # Col 0: theta分布（全范围）
        ax0 = axes[mode, 0]
        ax0.hist(theta_f, bins=360, density=False, alpha=0.7, color='steelblue')
        if wcs_rot is not None:
            ref = wcs_rot if mode in (0, 1, 2) else ((wcs_rot + 180) % 360 - 180 if (wcs_rot + 180) > 180 else wcs_rot + 180)
            ref = ((ref + 180) % 360) - 180
            ax0.axvline(ref, color='red', linestyle='--', linewidth=2, label=f'WCS={ref:.1f}°')
        ax0.axvline(peak_theta, color='green', linestyle='-', linewidth=1.5, label=f'peak={peak_theta:.1f}°')
        ax0.legend(fontsize=7)
        ax0.set_title(f'Mode {mode}: θ (s≈1, n={n_in_range}, SNR={snr:.1f}x)', fontsize=9)
        ax0.set_xlabel('θ (°)')
        ax0.set_ylabel('Count')

        # Col 1: theta精细分布（峰值±30°）
        ax1 = axes[mode, 1]
        near30 = np.abs(theta_f - peak_theta) < 30
        if np.sum(near30) > 0:
            ax1.hist(theta_f[near30], bins=120, density=False, alpha=0.7, color='steelblue')
            if wcs_rot is not None:
                ax1.axvline(ref, color='red', linestyle='--', linewidth=2, label=f'WCS={ref:.1f}°')
            ax1.axvline(peak_theta, color='green', linestyle='-', linewidth=1.5, label=f'peak={peak_theta:.1f}°')
            ax1.legend(fontsize=7)
        ax1.set_title(f'Mode {mode}: θ fine (±30°)', fontsize=9)
        ax1.set_xlabel('θ (°)')
        ax1.set_ylabel('Count')

        # Col 2: s分布
        ax2 = axes[mode, 2]
        ax2.hist(s_f, bins=100, density=True, alpha=0.7, color='forestgreen')
        ax2.axvline(1.0, color='red', linestyle='--', linewidth=1.5)
        ax2.set_title(f'Mode {mode}: s distribution', fontsize=9)
        ax2.set_xlabel('s')
        ax2.set_ylabel('Density')

        # Col 3: 位移分布
        ax3 = axes[mode, 3]
        ax3.hist(disp_f, bins=200, density=False, alpha=0.7, color='darkorange')
        if np.sum(near_peak) > 0:
            ax3.axvline(np.median(disp_f[near_peak]), color='red', linestyle='--',
                        label=f'peak_θ median={np.median(disp_f[near_peak]):.1f}″')
            ax3.legend(fontsize=7)
        ax3.set_title(f'Mode {mode}: |displacement|', fontsize=9)
        ax3.set_xlabel('displacement (″)')
        ax3.set_ylabel('Count')

        # Col 4: theta vs displacement散点
        ax4 = axes[mode, 4]
        sample_n = min(10000, len(theta_f))
        sample_idx = rng.choice(len(theta_f), sample_n, replace=False)
        sc = ax4.scatter(theta_f[sample_idx], disp_f[sample_idx],
                         c=s_f[sample_idx], cmap='viridis', s=1, alpha=0.2)
        plt.colorbar(sc, ax=ax4, label='s')
        if wcs_rot is not None:
            ax4.axvline(ref, color='red', linestyle='--', linewidth=1.5)
        ax4.set_title(f'Mode {mode}: θ vs |disp|', fontsize=9)
        ax4.set_xlabel('θ (°)')
        ax4.set_ylabel('|disp| (″)')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'{basename}_mc_sampling.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n图表已保存: {out_path}")

    # ── 额外：2D直方图 theta × displacement ──
    fig2, axes2 = plt.subplots(1, 4, figsize=(24, 5))
    fig2.suptitle(f'{basename} - θ × |displacement| 2D Histogram (s≈1)', fontsize=13)

    for mode in range(4):
        Wf = _apply_flip(W, mode)
        norm_Wf = np.linalg.norm(Wf, axis=1)
        valid_Wf = norm_Wf > 1e-10

        s_l, th_l, dp_l = [], [], []
        for _ in range(N_SAMPLES):
            i = rng.integers(0, N_img)
            j = rng.integers(0, M)
            if not valid_U[i] or not valid_Wf[j]:
                continue
            s = norm_U[i] / norm_Wf[j]
            if s < S_RANGE[0] or s > S_RANGE[1]:
                continue
            theta = math.atan2(U[i, 1], U[i, 0]) - math.atan2(Wf[j, 1], Wf[j, 0])
            ct, st = math.cos(theta), math.sin(theta)
            tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
            ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
            th_l.append(((math.degrees(theta) + 180) % 360) - 180)
            dp_l.append(math.sqrt(tx**2 + ty**2))

        if len(th_l) == 0:
            continue

        ax = axes2[mode]
        h = ax.hist2d(th_l, dp_l, bins=[180, 100], cmap='hot')
        plt.colorbar(h[3], ax=ax, label='Count')
        if wcs_rot is not None:
            ref = wcs_rot if mode in (0, 1, 2) else ((wcs_rot + 180) % 360 - 180 if (wcs_rot + 180) > 180 else wcs_rot + 180)
            ref = ((ref + 180) % 360) - 180
            ax.axvline(ref, color='cyan', linestyle='--', linewidth=1.5)
        ax.set_title(f'Mode {mode}', fontsize=10)
        ax.set_xlabel('θ (°)')
        ax.set_ylabel('|disp| (″)')

    plt.tight_layout()
    out_path2 = os.path.join(output_dir, f'{basename}_mc_2d_histogram.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"2D直方图已保存: {out_path2}")

    gaia.close()


if __name__ == '__main__':
    GAIA_DIR = os.path.join(PROJECT_ROOT, 'GaiaDR3SP')
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'debug_output')

    # T2帧（旋转~90度）
    t2_path = os.path.join(PROJECT_ROOT, 'testdata', 'lights',
                           'M20_T2_flying_dutchman-20250701@073331-300S-Red.fts')
    if os.path.exists(t2_path):
        analyze_frame(t2_path, GAIA_DIR, OUTPUT_DIR)

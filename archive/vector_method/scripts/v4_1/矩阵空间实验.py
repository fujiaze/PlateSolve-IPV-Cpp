"""V4.1 矩阵空间聚类实验 — Phase A变换矩阵(s,θ,tx,ty)的4D空间聚类行为分析

功能:
  Part A: 合成数据实验 — 验证真匹配星对推导的变换矩阵在4D空间中是否形成紧密聚类
    实验1: nr投票数分布直方图（真匹配vs假匹配）
    实验2: 矩阵空间2D投影聚类（s-θ, tx-ty）
    实验3: θ直方图SNR分析
  Part B: 真实数据实验 — 在NGC55/Victory成功/失败帧上验证聚类可区分性

用途: V4.1 Phase A抽样投票机制的理论验证和诊断工具

用法: python 矩阵空间实验.py
      可用于指导Phase A参数整定和失败帧调试
"""
import os, sys, math, json, time, traceback
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import multiprocessing as mp

# ============================================================================
# UTF-8 编码初始化（Windows GBK 兼容）
# ============================================================================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# 路径初始化
# ============================================================================
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

# ============================================================================
# 全局常量
# ============================================================================
LOGS_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "matrix_space")
SYNTHETIC_OUT = os.path.join(LOGS_DIR, "synthetic")
REAL_OUT = os.path.join(LOGS_DIR, "real")
ANALYSIS_TXT = os.path.join(LOGS_DIR, "analysis.txt")

os.makedirs(SYNTHETIC_OUT, exist_ok=True)
os.makedirs(REAL_OUT, exist_ok=True)

USE_PARALLEL = True
N_WORKERS = 16

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ============================================================================
# 核心数学工具函数
# ============================================================================

def derive_transform(ui, wj):
    """从单对(u_i, w_j)推导变换参数 T_ij = (s, θ_deg, tx, ty)
    ui: (2,) 图像星在arcsec空间的(x,y)坐标（从图像中心出发）
    wj: (2,) Gaia星在arcsec空间的(x,y)坐标（从参考点出发）
    Returns: (s, theta_deg, tx, ty)
    """
    ux, uy = ui[0], ui[1]
    wx, wy = wj[0], wj[1]

    norm_ui = math.sqrt(ux * ux + uy * uy)
    norm_wj = math.sqrt(wx * wx + wy * wy)

    if norm_ui < 1e-10 or norm_wj < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)

    s = norm_ui / norm_wj
    theta_ui = math.atan2(uy, ux)
    theta_wj = math.atan2(wy, wx)
    theta = theta_ui - theta_wj

    # 标准化θ到 [-π, π]
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi

    theta_deg = math.degrees(theta)

    ct = math.cos(theta)
    st = math.sin(theta)

    wx_rot = s * (ct * wx - st * wy)
    wy_rot = s * (st * wx + ct * wy)

    tx = ux - wx_rot
    ty = uy - wy_rot

    return (s, theta_deg, tx, ty)


def apply_transform_multiple(W, s, theta_deg, tx, ty):
    """将变换T = (s,θ,tx,ty)应用到多个Gaia星向量
    W: (M,2) Gaia星向量组
    Returns: (M,2) 变换后的向量组
    """
    theta = math.radians(theta_deg)
    ct = math.cos(theta)
    st = math.sin(theta)

    Wx = W[:, 0]
    Wy = W[:, 1]

    Wx_rot = s * (ct * Wx - st * Wy) + tx
    Wy_rot = s * (st * Wx + ct * Wy) + ty

    return np.column_stack([Wx_rot, Wy_rot])


def count_s_in_range_fast(U, W, s, theta_deg, tx, ty, tol_arcsec, s0, s_min=0.5, s_max=2.0):
    """快速投票计数：将T应用到所有W，用cKDTree找最近的U匹配
    U: (N,2) 图像星向量
    W: (M,2) Gaia星向量
    s, theta_deg, tx, ty: 变换参数
    tol_arcsec: 匹配距离阈值(角秒)
    s0: 像素尺度(角秒/像素)
    s_min, s_max: 尺度比合理范围
    Returns: nr (匹配对数)
    """
    from scipy.spatial import cKDTree

    Wt = apply_transform_multiple(W, s, theta_deg, tx, ty)
    tree = cKDTree(U)

    norm_U = np.linalg.norm(U, axis=1)
    norm_W = np.linalg.norm(W, axis=1)

    dists, idxs = tree.query(Wt, distance_upper_bound=tol_arcsec)
    nr = 0
    for j in range(len(Wt)):
        i = idxs[j]
        d = dists[j]
        if i < len(U) and d < tol_arcsec:
            s_ratio = norm_U[i] / max(norm_W[j], 1e-10)
            if s_min <= s_ratio <= s_max:
                nr += 1
    return nr


# ============================================================================
# Part A: 合成数据生成
# ============================================================================

def generate_synthetic(N=250, overlap_ratio=0.2, s0=1.0, fov_diag_deg=1.5, noise_arcsec=0.15, seed=42):
    """生成合成U(图像星)和W(Gaia星)向量组

    N: 图像星数量
    overlap_ratio: 重叠率（有真匹配的图像星比例）
    s0: 角秒/像素
    fov_diag_deg: FOV对角(度)
    noise_arcsec: 质心误差(角秒)

    Returns:
        U: (N,2) 图像星在arcsec空间的(x,y)坐标，从图像中心出发
        W: (M,2) Gaia星在arcsec空间的(x,y)坐标
        true_matches: dict, {img_idx: gaia_idx} 真匹配关系
        true_T: (s, theta_deg, tx, ty) 真变换参数
    """
    rng = np.random.RandomState(seed)
    fov_diag_arcsec = fov_diag_deg * 3600.0
    max_r = fov_diag_arcsec / 2.0

    true_s = 0.95 + 0.1 * rng.random()
    true_theta_deg = rng.uniform(-180, 180)
    true_theta = np.radians(true_theta_deg)
    true_tx = rng.uniform(-max_r * 0.03, max_r * 0.03)
    true_ty = rng.uniform(-max_r * 0.03, max_r * 0.03)

    ct, st = np.cos(true_theta), np.sin(true_theta)

    M = max(int(N / overlap_ratio), 50)
    r_W = max_r * np.sqrt(rng.random(M))
    ang_W = rng.uniform(0, 2 * np.pi, M)
    W = np.column_stack([r_W * np.cos(ang_W), r_W * np.sin(ang_W)])

    n_match = int(N * overlap_ratio)
    match_gaia_idx = rng.choice(M, size=n_match, replace=False)

    U = np.zeros((N, 2))
    true_matches = {}

    for k, g_idx in enumerate(match_gaia_idx):
        w_vec = W[g_idx]
        ux = true_s * (ct * w_vec[0] - st * w_vec[1]) + true_tx
        uy = true_s * (st * w_vec[0] + ct * w_vec[1]) + true_ty
        ux += rng.normal(0, noise_arcsec)
        uy += rng.normal(0, noise_arcsec)
        U[k] = [ux, uy]
        true_matches[k] = g_idx

    for k in range(n_match, N):
        r = max_r * np.sqrt(rng.random())
        ang = rng.uniform(0, 2 * np.pi)
        U[k] = [r * np.cos(ang), r * np.sin(ang)]

    return U, W, true_matches, (true_s, true_theta_deg, true_tx, true_ty)


# ============================================================================
# Part A: 合成数据实验函数
# ============================================================================

def sample_and_compute_transforms(U, W, N_samples, true_matches, s0, rng_seed=12345):
    """采样K对(i,j)，计算每对的变换和nr
    Returns:
        results: [(i, j, s, θ_deg, tx, ty, nr, is_true_match), ...]
    """
    rng = np.random.RandomState(rng_seed)
    N = len(U)
    M = len(W)

    tol_arcsec = 5.0 * s0
    s_min, s_max = 0.5, 2.0

    true_match_set = set(true_matches.items())

    results = []
    sampled_pairs = set()

    # 优先采样所有真匹配对
    for img_idx, gaia_idx in true_matches.items():
        ui = U[img_idx]
        wj = W[gaia_idx]
        s, theta_deg, tx, ty = derive_transform(ui, wj)
        nr = count_s_in_range_fast(U, W, s, theta_deg, tx, ty, tol_arcsec, s0, s_min, s_max)
        results.append((img_idx, gaia_idx, s, theta_deg, tx, ty, nr, True))
        sampled_pairs.add((img_idx, gaia_idx))

    n_true_sampled = len(results)
    n_random_needed = N_samples - n_true_sampled
    actual_samples = min(n_random_needed, N * M - n_true_sampled)
    attempts = 0
    max_attempts = actual_samples * 100

    while len(results) < N_samples and attempts < max_attempts:
        i = rng.randint(0, N)
        j = rng.randint(0, M)
        pair_key = (i, j)
        if pair_key in sampled_pairs:
            attempts += 1
            continue
        sampled_pairs.add(pair_key)
        attempts += 1

        ui = U[i]
        wj = W[j]
        s, theta_deg, tx, ty = derive_transform(ui, wj)

        is_true = (i in true_matches) and (true_matches[i] == j)
        nr = count_s_in_range_fast(U, W, s, theta_deg, tx, ty, tol_arcsec, s0, s_min, s_max)

        results.append((i, j, s, theta_deg, tx, ty, nr, is_true))

    return results


def experiment_1_nr_histogram(results, overlap_ratio, fov_label, out_dir):
    """实验1: nr投票数分布直方图"""
    true_nr = [r[6] for r in results if r[7]]
    false_nr = [r[6] for r in results if not r[7]]

    n_true = len(true_nr)
    n_false = len(false_nr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 分开的直方图
    ax = axes[0]
    if true_nr:
        ax.hist(true_nr, bins=50, alpha=0.6, color='red', label=f'真匹配对 (n={n_true})')
    if false_nr:
        ax.hist(false_nr, bins=50, alpha=0.6, color='gray', label=f'假匹配对 (n={n_false})')
    ax.set_xlabel('nr (投票数)')
    ax.set_ylabel('频数')
    ax.set_title(f'nr分布 — 重叠率={overlap_ratio} FOV={fov_label}')
    ax.legend()
    ax.set_yscale('log')

    # 子图2: 对数箱宽
    ax = axes[1]
    if true_nr:
        ax.hist(true_nr, bins=np.logspace(max(0, np.log10(max(1, min(true_nr)))), np.log10(max(true_nr) + 1), 40),
                alpha=0.6, color='red', label=f'真匹配对 (n={n_true})')
    if false_nr:
        ax.hist(false_nr, bins=np.logspace(max(0, np.log10(max(1, min(false_nr)))), np.log10(max(false_nr) + 1), 40),
                alpha=0.6, color='gray', label=f'假匹配对 (n={n_false})')
    ax.set_xlabel('nr (对数刻度)')
    ax.set_ylabel('频数')
    ax.set_title(f'nr分布(对数箱宽) — 重叠率={overlap_ratio}')
    ax.set_xscale('log')
    ax.legend()

    plt.tight_layout()
    filename = os.path.join(out_dir, f"nr_histogram_{overlap_ratio}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # 统计
    stats = {}
    if true_nr:
        stats['true_count'] = n_true
        stats['true_nr_mean'] = float(np.mean(true_nr))
        stats['true_nr_median'] = float(np.median(true_nr))
        stats['true_nr_p50'] = float(np.percentile(true_nr, 50))
        stats['true_nr_p90'] = float(np.percentile(true_nr, 90))
        stats['true_nr_p95'] = float(np.percentile(true_nr, 95))
        stats['true_nr_p99'] = float(np.percentile(true_nr, 99))
        stats['true_nr_max'] = int(max(true_nr))
    if false_nr:
        stats['false_count'] = n_false
        stats['false_nr_mean'] = float(np.mean(false_nr))
        stats['false_nr_median'] = float(np.median(false_nr))
        stats['false_nr_p95'] = float(np.percentile(false_nr, 95))
        stats['false_nr_p99'] = float(np.percentile(false_nr, 99))
        stats['false_nr_max'] = int(max(false_nr))

    if true_nr and false_nr:
        stats['nr_separation_ratio'] = float(stats['true_nr_median'] / max(stats['false_nr_p99'], 1))

    return stats


def experiment_2_matrix_space(results, overlap_ratio, fov_label, out_dir):
    """实验2: 矩阵空间2D投影聚类"""
    true_results = [r for r in results if r[7]]
    false_results = [r for r in results if not r[7]]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 子图1: s vs θ
    ax = axes[0, 0]
    if false_results:
        fr_nr = np.array([r[6] for r in false_results])
        fr_s = np.array([r[2] for r in false_results])
        fr_theta = np.array([r[3] for r in false_results])
        scatter_f = ax.scatter(fr_theta, fr_s, c=fr_nr, cmap='Blues', alpha=0.4, s=3, vmin=0, vmax=max(fr_nr) if len(fr_nr) > 0 else 1)
    if true_results:
        tr_s = np.array([r[2] for r in true_results])
        tr_theta = np.array([r[3] for r in true_results])
        ax.scatter(tr_theta, tr_s, c='red', alpha=0.8, s=25, marker='o', edgecolors='darkred', linewidths=0.5,
                   label=f'真匹配对 (n={len(true_results)})')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('s (尺度)')
    ax.set_title(f's vs θ — 重叠率={overlap_ratio} FOV={fov_label}')
    if true_results:
        ax.legend(loc='upper right', fontsize=8)

    # 子图2: tx vs ty
    ax = axes[0, 1]
    if false_results:
        fr_tx = np.array([r[4] for r in false_results])
        fr_ty = np.array([r[5] for r in false_results])
        scatter_f2 = ax.scatter(fr_tx, fr_ty, c=fr_nr, cmap='Blues', alpha=0.4, s=3, vmin=0, vmax=max(fr_nr) if len(fr_nr) > 0 else 1)
    if true_results:
        tr_tx = np.array([r[4] for r in true_results])
        tr_ty = np.array([r[5] for r in true_results])
        ax.scatter(tr_tx, tr_ty, c='red', alpha=0.8, s=25, marker='o', edgecolors='darkred', linewidths=0.5,
                   label=f'真匹配对 (n={len(true_results)})')
    ax.set_xlabel('tx (角秒)')
    ax.set_ylabel('ty (角秒)')
    ax.set_title(f'tx vs ty — 重叠率={overlap_ratio} FOV={fov_label}')
    if true_results:
        ax.legend(loc='upper right', fontsize=8)

    # 子图3: nr着色 s vs θ
    ax = axes[1, 0]
    all_s = np.array([r[2] for r in results])
    all_theta = np.array([r[3] for r in results])
    all_nr = np.array([r[6] for r in results])
    all_is_true = np.array([r[7] for r in results])

    false_mask = ~all_is_true
    if np.any(false_mask):
        sc = ax.scatter(all_theta[false_mask], all_s[false_mask], c=all_nr[false_mask],
                        cmap='viridis', alpha=0.5, s=4, vmin=0, vmax=max(all_nr) if len(all_nr) > 0 else 1)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('nr')
    if np.any(all_is_true):
        ax.scatter(all_theta[all_is_true], all_s[all_is_true], c='red', alpha=0.9, s=30,
                   marker='*', edgecolors='darkred', linewidths=0.3, label='真匹配')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('s (尺度)')
    ax.set_title(f's vs θ (nr着色) — 重叠率={overlap_ratio}')
    if np.any(all_is_true):
        ax.legend(loc='upper right', fontsize=8)

    # 子图4: nr着色 tx vs ty
    ax = axes[1, 1]
    all_tx = np.array([r[4] for r in results])
    all_ty = np.array([r[5] for r in results])
    if np.any(false_mask):
        sc = ax.scatter(all_tx[false_mask], all_ty[false_mask], c=all_nr[false_mask],
                        cmap='viridis', alpha=0.5, s=4, vmin=0, vmax=max(all_nr) if len(all_nr) > 0 else 1)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label('nr')
    if np.any(all_is_true):
        ax.scatter(all_tx[all_is_true], all_ty[all_is_true], c='red', alpha=0.9, s=30,
                   marker='*', edgecolors='darkred', linewidths=0.3, label='真匹配')
    ax.set_xlabel('tx (角秒)')
    ax.set_ylabel('ty (角秒)')
    ax.set_title(f'tx vs ty (nr着色) — 重叠率={overlap_ratio}')
    if np.any(all_is_true):
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    filename = os.path.join(out_dir, f"matrix_space_{overlap_ratio}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # 聚类紧密度统计
    stats = {}
    if true_results and len(true_results) >= 3:
        tr_s = np.array([r[2] for r in true_results])
        tr_theta = np.array([r[3] for r in true_results])
        tr_tx = np.array([r[4] for r in true_results])
        tr_ty = np.array([r[5] for r in true_results])
        stats['true_s_mean'] = float(np.mean(tr_s))
        stats['true_s_std'] = float(np.std(tr_s))
        stats['true_theta_mean'] = float(np.mean(tr_theta))
        stats['true_theta_std'] = float(np.std(tr_theta))
        stats['true_tx_mean'] = float(np.mean(tr_tx))
        stats['true_tx_std'] = float(np.std(tr_tx))
        stats['true_ty_mean'] = float(np.mean(tr_ty))
        stats['true_ty_std'] = float(np.std(tr_ty))

    if false_results and len(false_results) >= 3:
        fr_s = np.array([r[2] for r in false_results])
        fr_theta = np.array([r[3] for r in false_results])
        stats['false_s_std'] = float(np.std(fr_s))
        stats['false_theta_std'] = float(np.std(fr_theta))

    return stats


def experiment_3_theta_histogram(results, overlap_ratio, fov_label, out_dir):
    """实验3: θ直方图SNR — 模拟Phase A的θ投票机制"""
    theta_bins = 360
    bin_width = 360.0 / theta_bins

    histogram = np.zeros(theta_bins)
    for r in results:
        nr = r[6]
        theta_deg = r[3]
        bin_idx = int(((theta_deg + 180.0) % 360.0) / bin_width)
        if 0 <= bin_idx < theta_bins:
            histogram[bin_idx] += nr

    # SNR计算
    peak_val = np.max(histogram)
    peak_bin = np.argmax(histogram)
    peak_theta = peak_bin * bin_width - 180.0

    # 中位背景（排除峰值±10°区域）
    exclude_half = int(10.0 / bin_width)
    mask = np.ones(theta_bins, dtype=bool)
    for d in range(-exclude_half, exclude_half + 1):
        idx = (peak_bin + d) % theta_bins
        mask[idx] = False
    background = histogram[mask]
    median_bg = np.median(background) if len(background) > 0 else 1.0
    snr = peak_val / max(median_bg, 1e-10)

    # 绘图
    fig, ax = plt.subplots(figsize=(12, 5))
    theta_centers = np.linspace(-180, 180, theta_bins)
    ax.bar(theta_centers, histogram, width=bin_width, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axhline(y=median_bg, color='orange', linestyle='--', linewidth=1.5, label=f'中位背景={median_bg:.1f}')
    ax.axvline(x=peak_theta, color='red', linestyle='-', linewidth=2, label=f'峰值θ={peak_theta:.1f}° SNR={snr:.1f}')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('累积投票数')
    ax.set_title(f'Phase A θ直方图 — 重叠率={overlap_ratio} FOV={fov_label} SNR={snr:.1f}')
    ax.legend(fontsize=9)
    plt.tight_layout()
    filename = os.path.join(out_dir, f"theta_histogram_{overlap_ratio}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    return {'peak_theta': float(peak_theta), 'snr': float(snr), 'median_bg': float(median_bg),
            'peak_val': float(peak_val), 'theta_bins': theta_bins}


def run_synthetic_summary(all_stats, all_theta_stats, overlap_ratios, fov_configs, out_dir):
    """生成合成数据汇总图"""
    n_configs = len(fov_configs)
    n_ratios = len(overlap_ratios)

    fig, axes = plt.subplots(n_configs, 4, figsize=(20, 5 * n_configs))
    if n_configs == 1:
        axes = axes.reshape(1, -1)

    for ci, (fov_label, fov_deg, _s0) in enumerate(fov_configs):
        ratios = []
        snrs = []
        separation = []
        true_s_std = []
        false_theta_std = []

        for ri, overlap in enumerate(overlap_ratios):
            key = f"{overlap}_{fov_label}"
            if key in all_stats and 'true_nr_median' in all_stats[key]:
                ratios.append(overlap)
                snrs.append(all_theta_stats.get(key, {}).get('snr', 0))
                separation.append(all_stats[key].get('nr_separation_ratio', 0))
                true_s_std.append(all_stats[key].get('true_s_std', 0))
                false_theta_std.append(all_stats[key].get('false_theta_std', 0))

        # SNR vs 重叠率
        ax = axes[ci, 0]
        if ratios:
            ax.plot(ratios, snrs, 'o-', color='darkblue', markersize=8)
        ax.set_xlabel('重叠率')
        ax.set_ylabel('θ SNR')
        ax.set_title(f'{fov_label} θ SNR vs 重叠率')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.55)

        # nr分离度 vs 重叠率
        ax = axes[ci, 1]
        if ratios:
            ax.plot(ratios, separation, 's-', color='darkgreen', markersize=8)
        ax.set_xlabel('重叠率')
        ax.set_ylabel('nr分离比 (真/假)')
        ax.set_title(f'{fov_label} nr分离度 vs 重叠率')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.55)

        # s聚类紧密度
        ax = axes[ci, 2]
        if ratios:
            ax.plot(ratios, true_s_std, 'D-', color='darkred', markersize=8)
        ax.set_xlabel('重叠率')
        ax.set_ylabel('真匹配s标准差')
        ax.set_title(f'{fov_label} 尺度聚类紧密度')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.55)

        # θ分散度
        ax = axes[ci, 3]
        if ratios:
            ax.plot(ratios, false_theta_std, '^-', color='darkorange', markersize=8)
        ax.set_xlabel('重叠率')
        ax.set_ylabel('假匹配θ标准差 (°)')
        ax.set_title(f'{fov_label} 假匹配θ分散度')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.55)

    plt.tight_layout()
    filename = os.path.join(out_dir, "synthetic_summary.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Part B: 真实数据实验
# ============================================================================

def parse_ra_dec_from_keywords(keywords):
    """从FITS关键字解析RA/Dec指向（度）"""
    kw_dict = {}
    for kw in keywords:
        kw_dict[kw.name.upper()] = kw.value.strip() if isinstance(kw.value, str) else str(kw.value)

    ra_str = kw_dict.get('OBJCTRA') or kw_dict.get('RA', '0 0 0')
    dec_str = kw_dict.get('OBJCTDEC') or kw_dict.get('DEC', '0 0 0')

    try:
        ra_parts = ra_str.strip().split()
        ra_h = float(ra_parts[0])
        ra_m = float(ra_parts[1]) if len(ra_parts) > 1 else 0.0
        ra_s = float(ra_parts[2]) if len(ra_parts) > 2 else 0.0
        cra = (ra_h + ra_m / 60.0 + ra_s / 3600.0) * 15.0
    except Exception:
        try:
            cra = float(ra_str)
        except Exception:
            cra = 0.0

    try:
        dec_str_clean = dec_str.strip()
        sign = -1.0 if dec_str_clean.startswith('-') else 1.0
        dec_str_clean = dec_str_clean.lstrip('+-')
        dec_parts = dec_str_clean.split()
        dec_d = float(dec_parts[0])
        dec_m = float(dec_parts[1]) if len(dec_parts) > 1 else 0.0
        dec_s = float(dec_parts[2]) if len(dec_parts) > 2 else 0.0
        cdec = sign * (abs(dec_d) + dec_m / 60.0 + dec_s / 3600.0)
    except Exception:
        try:
            cdec = float(dec_str)
        except Exception:
            cdec = 0.0

    return cra, cdec


def load_real_data(fits_path, verbose=True):
    """读取FITS+星点检测+Gaia查询，返回U,W向量组

    Returns:
        U: (N,2) 图像星在arcsec空间的(x,y)坐标
        W: (M,2) Gaia星在arcsec空间的(x,y)坐标
        s0: 角秒/像素
        fov_diag: FOV对角(度)
        metadata: dict 包含w_img, h_img, fl, ps, cra, cdec等信息
    """
    from astro_image_io import ImageReader
    from star_detector import StarDetector, SDetParamsPy
    from vector_match_v2 import GaiaClientPy

    reader = ImageReader()
    img = reader.read(fits_path)
    w_img, h_img = img.width, img.height
    meta = img.metadata
    fl = meta.observation.focallen or 0.0
    ps = meta.observation.xpixsz or 1.0
    s0 = 206.265 * ps / fl if fl > 0 else 1.0

    if verbose:
        print(f"  图像: {w_img}x{h_img}, 焦距={fl}mm, 像素={ps}um, s0={s0:.4f}\"/px")

    detector = StarDetector(params=SDetParamsPy(fitRadius=0, maxStars=500))
    det = detector.detect_ex(img.data)

    n_img_total = 250
    sat_idx = np.where(np.array(det.saturated) == 1)[0]
    nsat = len(sat_idx)
    non_sat_idx = np.where(np.array(det.saturated) == 0)[0]
    if len(non_sat_idx) > 0:
        non_sat_sorted = non_sat_idx[np.argsort(-np.array(det.flux)[non_sat_idx])]
    else:
        non_sat_sorted = np.array([], dtype=np.int64)
    n_needed = max(0, n_img_total - nsat)
    top_non_sat = non_sat_sorted[:n_needed]
    sel_idx = np.concatenate([sat_idx, top_non_sat]).astype(np.int64)

    cx, cy = w_img / 2.0, h_img / 2.0
    ux = (np.array(det.x)[sel_idx] - cx) * s0
    uy = -(np.array(det.y)[sel_idx] - cy) * s0
    U = np.column_stack([ux, uy])

    if verbose:
        print(f"  星点检测: {det.count}颗 ({nsat}饱和), 选取{len(sel_idx)}颗 (前{nsat}饱和+{len(top_non_sat)}亮星)")

    kw_dict = {}
    for kw in img.keywords:
        kw_dict[kw.name.upper()] = kw.value.strip() if isinstance(kw.value, str) else str(kw.value)

    cra, cdec = parse_ra_dec_from_keywords(img.keywords)

    fov_diag = math.sqrt(w_img * w_img + h_img * h_img) * s0 / 3600.0
    query_radius = max(fov_diag * 0.7, 1.0)

    if verbose:
        print(f"  FOV对角={fov_diag:.2f}°, RA={cra:.4f}°, Dec={cdec:.4f}°, 查询半径={query_radius:.2f}°")

    mag_limit = 16.0 if fov_diag < 4.0 else 13.0
    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    try:
        ra_t, dec_t, mag_t = gaia.cone_search(cra, cdec, query_radius, mag_limit)
    finally:
        gaia.close()

    cos_dec = math.cos(math.radians(cdec))
    wx_all = (np.array(ra_t) - cra) * 3600.0 * cos_dec
    wy_all = (np.array(dec_t) - cdec) * 3600.0
    W = np.column_stack([wx_all, wy_all])

    if verbose:
        print(f"  Gaia查询: {len(W)}颗星 (极限星等{mag_limit})")

    img.close()
    detector.close()

    metadata = {'w_img': w_img, 'h_img': h_img, 'fl': fl, 'ps': ps, 's0': s0,
                'cra': cra, 'cdec': cdec, 'fov_diag': fov_diag, 'nsat': nsat}

    return U, W, s0, fov_diag, metadata


def run_real_data_experiment(fits_path, frame_name, out_dir, K=20000, verbose=True):
    """对单帧运行矩阵空间实验

    Returns:
        results: list of (i, j, s, θ_deg, tx, ty, nr)
        metadata: dict
        stats: dict
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"处理: {frame_name}")
        print(f"  文件: {fits_path}")
        print(f"{'='*60}")

    U, W, s0, fov_diag, metadata = load_real_data(fits_path, verbose=verbose)

    N = len(U)
    M = len(W)
    if N < 5 or M < 5:
        print(f"  警告: 星数不足 U={N} W={M}, 跳过")
        return None, metadata, None

    if verbose:
        print(f"  U={N}颗 (arcsec向量从图像中心), W={M}颗 (arcsec向量从RA/Dec参考点)")

    tol_arcsec = 5.0 * s0
    s_min, s_max = 0.5, 2.0

    # 采样
    rng = np.random.RandomState(42)
    actual_K = min(K, N * M)
    if verbose:
        print(f"  采样K={actual_K}对...")

    results = []
    sampled = set()
    attempts = 0
    max_attempts = actual_K * 50

    t_start = time.time()
    while len(results) < actual_K and attempts < max_attempts:
        i = rng.randint(0, N)
        j = rng.randint(0, M)
        key = (i, j)
        if key in sampled:
            attempts += 1
            continue
        sampled.add(key)
        attempts += 1

        ui = U[i]
        wj = W[j]
        s, theta_deg, tx, ty = derive_transform(ui, wj)
        nr = count_s_in_range_fast(U, W, s, theta_deg, tx, ty, tol_arcsec, s0, s_min, s_max)
        results.append((i, j, s, theta_deg, tx, ty, nr))

    t_elapsed = time.time() - t_start
    if verbose:
        print(f"  采样完成: {len(results)}对, 耗时 {t_elapsed:.1f}s ({len(results)/t_elapsed:.0f} 对/s)")

    # 按nr排序
    results.sort(key=lambda x: -x[6])

    # 统计
    nr_values = [r[6] for r in results]
    stats = {
        'n_samples': len(results),
        'n_U': N, 'n_W': M,
        's0': s0, 'fov_diag': fov_diag,
        'nr_mean': float(np.mean(nr_values)),
        'nr_median': float(np.median(nr_values)),
        'nr_p50': float(np.percentile(nr_values, 50)),
        'nr_p90': float(np.percentile(nr_values, 90)),
        'nr_p95': float(np.percentile(nr_values, 95)),
        'nr_p99': float(np.percentile(nr_values, 99)),
        'nr_max': int(max(nr_values)),
        'nr_gt_10_count': int(sum(1 for x in nr_values if x > 10)),
        'nr_gt_20_count': int(sum(1 for x in nr_values if x > 20)),
        'nr_gt_50_count': int(sum(1 for x in nr_values if x > 50)),
        'time_s': t_elapsed,
    }

    if verbose:
        print(f"  nr统计: mean={stats['nr_mean']:.1f} median={stats['nr_median']:.1f} "
              f"P90={stats['nr_p90']:.1f} P95={stats['nr_p95']:.1f} P99={stats['nr_p99']:.1f} max={stats['nr_max']}")
        print(f"  nr>10:{stats['nr_gt_10_count']} nr>20:{stats['nr_gt_20_count']} nr>50:{stats['nr_gt_50_count']}")

    # Top-10高nr对
    top10 = results[:10]
    if verbose:
        print(f"\n  Top-10最高nr的(s,θ,tx,ty):")
        for rank, r in enumerate(top10):
            print(f"    #{rank+1}: nr={r[6]:4d} s={r[2]:.4f} θ={r[3]:8.2f}° tx={r[4]:8.1f}\" ty={r[5]:8.1f}\"")
        # 检查θ一致性
        top_theta = [r[3] for r in top10]
        if len(top_theta) >= 2:
            theta_spread = max(top_theta) - min(top_theta)
            print(f"    θ范围: [{min(top_theta):.2f}°, {max(top_theta):.2f}°] 跨度={theta_spread:.2f}°")

    # 绘图
    plot_real_data_results(results, frame_name, out_dir, stats, s0, fov_diag)

    return results, metadata, stats


def plot_real_data_results(results, frame_name, out_dir, stats, s0, fov_diag):
    """绘制真实数据的矩阵空间图"""
    nr_arr = np.array([r[6] for r in results])
    s_arr = np.array([r[2] for r in results])
    theta_arr = np.array([r[3] for r in results])
    tx_arr = np.array([r[4] for r in results])
    ty_arr = np.array([r[5] for r in results])

    # nr直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.hist(nr_arr, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(stats['nr_p50'], color='green', linestyle='--', linewidth=1.5, label=f"P50={stats['nr_p50']:.0f}")
    ax.axvline(stats['nr_p95'], color='orange', linestyle='--', linewidth=1.5, label=f"P95={stats['nr_p95']:.0f}")
    ax.axvline(stats['nr_p99'], color='red', linestyle='--', linewidth=1.5, label=f"P99={stats['nr_p99']:.0f}")
    ax.set_xlabel('nr (投票数)')
    ax.set_ylabel('频数')
    ax.set_title(f'{frame_name} nr分布')
    ax.legend(fontsize=8)
    ax.set_yscale('log')

    # s vs θ (nr着色)
    ax = axes[0, 1]
    vmax = min(stats['nr_p99'] * 1.2, max(nr_arr)) if max(nr_arr) > 0 else 1
    sc = ax.scatter(theta_arr, s_arr, c=nr_arr, cmap='plasma', alpha=0.6, s=3, vmin=0, vmax=vmax)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('nr')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('s (尺度)')
    ax.set_title(f'{frame_name} s vs θ (nr着色)')

    # tx vs ty (nr着色)
    ax = axes[1, 0]
    sc = ax.scatter(tx_arr, ty_arr, c=nr_arr, cmap='plasma', alpha=0.6, s=3, vmin=0, vmax=vmax)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('nr')
    ax.set_xlabel('tx (角秒)')
    ax.set_ylabel('ty (角秒)')
    ax.set_title(f'{frame_name} tx vs ty (nr着色)')

    # θ直方图 (加权)
    ax = axes[1, 1]
    theta_bins = 360
    bin_width = 360.0 / theta_bins
    theta_hist = np.zeros(theta_bins)
    for nr, th in zip(nr_arr, theta_arr):
        bin_idx = int(((th + 180.0) % 360.0) / bin_width)
        if 0 <= bin_idx < theta_bins:
            theta_hist[bin_idx] += nr

    peak_val = np.max(theta_hist)
    peak_bin = np.argmax(theta_hist)
    peak_theta = peak_bin * bin_width - 180.0
    exclude_half = int(10.0 / bin_width)
    mask = np.ones(theta_bins, dtype=bool)
    for d in range(-exclude_half, exclude_half + 1):
        mask[(peak_bin + d) % theta_bins] = False
    bg = theta_hist[mask]
    median_bg = np.median(bg) if len(bg) > 0 else 1.0
    snr = peak_val / max(median_bg, 1e-10)

    theta_centers = np.linspace(-180, 180, theta_bins)
    ax.bar(theta_centers, theta_hist, width=bin_width, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axvline(x=peak_theta, color='red', linestyle='-', linewidth=2, label=f'SNR={snr:.1f}')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('加权投票数')
    ax.set_title(f'{frame_name} θ加权直方图 SNR={snr:.1f}')
    ax.legend(fontsize=8)

    stats['theta_snr'] = float(snr)
    stats['theta_peak'] = float(peak_theta)

    plt.tight_layout()
    filename = os.path.join(out_dir, f"{frame_name}_matrix_space.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

    # nr直方图单独
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(nr_arr, bins=100, color='darkblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('nr (投票数)')
    ax.set_ylabel('频数')
    ax.set_title(f'{frame_name} nr分布 (K={len(results)})  s0={s0:.3f}\"/px FOV={fov_diag:.2f}°')
    ax.set_yscale('log')
    ax.axvline(stats['nr_p50'], color='green', linestyle='--', label=f"median={stats['nr_p50']:.0f}")
    ax.axvline(stats['nr_p95'], color='orange', linestyle='--', label=f"P95={stats['nr_p95']:.0f}")
    ax.axvline(stats['nr_p99'], color='red', linestyle='--', label=f"P99={stats['nr_p99']:.0f}")
    ax.legend()
    plt.tight_layout()
    filename = os.path.join(out_dir, f"{frame_name}_nr_hist.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_real_comparison(results_a, stats_a, label_a, results_b, stats_b, label_b, title, out_path):
    """对比两个真实数据帧的矩阵空间图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for idx, (results, stats, label, color) in enumerate([
        (results_a, stats_a, label_a, 'red'),
        (results_b, stats_b, label_b, 'blue')
    ]):
        nr_arr = np.array([r[6] for r in results])
        s_arr = np.array([r[2] for r in results])
        theta_arr = np.array([r[3] for r in results])
        tx_arr = np.array([r[4] for r in results])
        ty_arr = np.array([r[5] for r in results])

        # nr直方图
        ax = axes[0, 0]
        ax.hist(nr_arr, bins=80, alpha=0.5, color=color, label=f'{label} (median={stats["nr_median"]:.0f})')

        # s vs θ
        ax = axes[0, 1]
        ax.scatter(theta_arr, s_arr, c=color, alpha=0.3, s=2, label=label)

        # tx vs ty
        ax = axes[0, 2]
        ax.scatter(tx_arr, ty_arr, c=color, alpha=0.3, s=2, label=label)

        # θ直方图
        ax = axes[1, 0]
        theta_bins = 360
        bin_w = 360.0 / theta_bins
        th_hist = np.zeros(theta_bins)
        for nr_val, th_val in zip(nr_arr, theta_arr):
            bi = int(((th_val + 180.0) % 360.0) / bin_w)
            if 0 <= bi < theta_bins:
                th_hist[bi] += nr_val
        theta_c = np.linspace(-180, 180, theta_bins)
        ax.plot(theta_c, th_hist, color=color, alpha=0.7, linewidth=1,
                label=f'{label} SNR={stats.get("theta_snr",0):.1f}')

        # s分布
        ax = axes[1, 1]
        ax.hist(s_arr, bins=80, alpha=0.4, color=color, label=f'{label} (std={np.std(s_arr):.4f})')

        # nr累积
        ax = axes[1, 2]
        sorted_nr = np.sort(nr_arr)[::-1]
        cumsum = np.cumsum(sorted_nr)
        ax.plot(cumsum, color=color, alpha=0.8, linewidth=1.5, label=label)

    axes[0, 0].set_xlabel('nr')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('nr分布对比')
    axes[0, 0].set_yscale('log')
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_xlabel('θ (°)')
    axes[0, 1].set_ylabel('s (尺度)')
    axes[0, 1].set_title('s vs θ')
    axes[0, 1].legend(fontsize=7)

    axes[0, 2].set_xlabel('tx (角秒)')
    axes[0, 2].set_ylabel('ty (角秒)')
    axes[0, 2].set_title('tx vs ty')
    axes[0, 2].legend(fontsize=7)

    axes[1, 0].set_xlabel('θ (°)')
    axes[1, 0].set_ylabel('加权投票')
    axes[1, 0].set_title('θ加权直方图对比')
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_xlabel('s (尺度)')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('s分布对比')
    axes[1, 1].legend(fontsize=7)

    axes[1, 2].set_xlabel('排名 (按nr降序)')
    axes[1, 2].set_ylabel('累积nr')
    axes[1, 2].set_title('nr累积分布')
    axes[1, 2].legend(fontsize=7)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 分析报告输出
# ============================================================================

def write_analysis_report(sections, out_path):
    """将分析结果写入文本文件"""
    lines = []
    lines.append("=" * 80)
    lines.append("V4.1 矩阵空间聚类实验 — 分析报告")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    for section_title, content in sections:
        lines.append("")
        lines.append("-" * 60)
        lines.append(f"  {section_title}")
        lines.append("-" * 60)
        if isinstance(content, list):
            lines.extend(content)
        elif isinstance(content, dict):
            for k, v in content.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(str(content))

    lines.append("")
    lines.append("=" * 80)
    lines.append("报告结束")
    lines.append("=" * 80)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("V4.1 矩阵空间聚类实验")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    sections = []
    all_synth_stats = {}
    all_theta_stats = {}

    # ========================================================================
    # Part A: 合成数据实验
    # ========================================================================
    print("\n" + "=" * 60)
    print("Part A: 合成数据实验")
    print("=" * 60)

    overlap_ratios = [0.1, 0.2, 0.3, 0.5]
    fov_configs = [
        ('narrow_1.5deg', 1.5, 1.0),    # 1935mm, ~1.0"/px
        ('wide_9.9deg', 9.9, 6.0),       # 200mm, ~6.0"/px
    ]

    for fov_label, fov_deg, s0 in fov_configs:
        print(f"\n--- FOV={fov_label} s0={s0:.1f}\"/px ---")

        for overlap in overlap_ratios:
            print(f"\n  重叠率={overlap}...")
            key = f"{overlap}_{fov_label}"

            # 生成合成数据
            U, W, true_matches, true_T = generate_synthetic(
                N=250, overlap_ratio=overlap, s0=s0,
                fov_diag_deg=fov_deg, noise_arcsec=0.15, seed=42
            )
            print(f"  U={len(U)}颗, W={len(W)}颗, 真匹配={len(true_matches)}对")
            print(f"  真变换: s={true_T[0]:.4f} θ={true_T[1]:.2f}° tx={true_T[2]:.1f}\" ty={true_T[3]:.1f}\"")

            # 采样
            results = sample_and_compute_transforms(U, W, N_samples=5000, true_matches=true_matches, s0=s0)

            true_count = sum(1 for r in results if r[7])
            false_count = len(results) - true_count
            print(f"  采样: {len(results)}对 (真匹配{true_count}, 假匹配{false_count})")

            # 实验1
            nr_stats = experiment_1_nr_histogram(results, overlap, fov_label, SYNTHETIC_OUT)
            all_synth_stats[key] = nr_stats
            if nr_stats:
                print(f"  实验1 nr: 真中位={nr_stats.get('true_nr_median','N/A')} "
                      f"假中位={nr_stats.get('false_nr_median','N/A')} "
                      f"分离比={nr_stats.get('nr_separation_ratio','N/A')}")

            # 实验2
            space_stats = experiment_2_matrix_space(results, overlap, fov_label, SYNTHETIC_OUT)
            for k, v in space_stats.items():
                all_synth_stats[key][k] = v
            if 'true_s_std' in space_stats:
                print(f"  实验2 聚类紧密度: s_std={space_stats['true_s_std']:.4f} "
                      f"θ_std={space_stats['true_theta_std']:.2f}°")

            # 实验3
            theta_s = experiment_3_theta_histogram(results, overlap, fov_label, SYNTHETIC_OUT)
            all_theta_stats[key] = theta_s
            print(f"  实验3 θ SNR={theta_s['snr']:.1f} 峰值θ={theta_s['peak_theta']:.1f}°")

    # 汇总
    run_synthetic_summary(all_synth_stats, all_theta_stats, overlap_ratios, fov_configs, SYNTHETIC_OUT)

    # Part A分析文本
    part_a_lines = []
    part_a_lines.append("合成数据实验分析")
    part_a_lines.append("")
    for fov_label, fov_deg, s0 in fov_configs:
        part_a_lines.append(f"  FOV={fov_label} ({fov_deg}°, s0={s0:.1f}\"/px):")
        for overlap in overlap_ratios:
            key = f"{overlap}_{fov_label}"
            ns = all_synth_stats.get(key, {})
            ts = all_theta_stats.get(key, {})
            part_a_lines.append(f"    重叠率={overlap}: nr分离比={ns.get('nr_separation_ratio','N/A')}, "
                              f"θ_SNR={ts.get('snr','N/A'):.1f}, "
                              f"真s_std={ns.get('true_s_std','N/A')}, "
                              f"真θ_std={ns.get('true_theta_std','N/A')}")

    part_a_lines.append("")
    part_a_lines.append("  结论:")
    # 找出最佳分离的配置
    best_sep = 0
    best_key = ""
    for k, v in all_synth_stats.items():
        if v.get('nr_separation_ratio', 0) > best_sep:
            best_sep = v['nr_separation_ratio']
            best_key = k
    part_a_lines.append(f"    - 最优nr分离度: {best_key} (分离比={best_sep:.1f})")
    best_snr = 0
    best_snr_key = ""
    for k, v in all_theta_stats.items():
        if v.get('snr', 0) > best_snr:
            best_snr = v['snr']
            best_snr_key = k
    part_a_lines.append(f"    - 最优θ SNR: {best_snr_key} (SNR={best_snr:.1f})")

    sections.append(("Part A: 合成数据实验", part_a_lines))

    # ========================================================================
    # Part B: 真实数据实验
    # ========================================================================
    print("\n" + "=" * 60)
    print("Part B: 真实数据实验")
    print("=" * 60)

    REAL_FRAMES = {
        'NGC55_Oiii_fail': (os.path.join(PROJECT_ROOT,
            'testdata', 'lights', 'NGC55_T3_flying_dutchman-20250915@042902-1200S-Oiii.fts'), 20000),
        'NGC55_Oiii_success': (os.path.join(PROJECT_ROOT,
            'testdata', 'lights', 'NGC55_T3_flying_dutchman-20250915@025221-1200S-Oiii.fts'), 20000),
        'Victory_mosaic2_Lum_fail': (os.path.join(PROJECT_ROOT,
            'testdata', 'lights', 'Victory_Nebula_mosaic2_flying_dutchman-20250205@062533-180S-Lum.fts'), 10000),
        'Victory_mosaic2_Lum_success': (os.path.join(PROJECT_ROOT,
            'testdata', 'lights', 'Victory_Nebula_mosaic2_flying_dutchman-20250205@062145-180S-Lum.fts'), 10000),
    }

    real_results = {}
    real_stats = {}

    for frame_key, (fits_path, K_samples) in REAL_FRAMES.items():
        if not os.path.exists(fits_path):
            print(f"\n  警告: 文件不存在 {fits_path}")
            continue

        results, metadata, stats = run_real_data_experiment(
            fits_path, frame_key, REAL_OUT, K=K_samples, verbose=True
        )
        if results is not None:
            real_results[frame_key] = (results, metadata)
            real_stats[frame_key] = stats

    # 对比图
    print(f"\n{'='*60}")
    print("生成对比图...")
    print(f"{'='*60}")

    # NGC55对比
    if 'NGC55_Oiii_fail' in real_results and 'NGC55_Oiii_success' in real_results:
        results_fail, meta_fail = real_results['NGC55_Oiii_fail']
        results_success, meta_success = real_results['NGC55_Oiii_success']
        stats_fail = real_stats['NGC55_Oiii_fail']
        stats_success = real_stats['NGC55_Oiii_success']

        print(f"\nNGC55 Oiii 对比:")
        print(f"  失败帧 nr中位={stats_fail['nr_median']:.1f} θ_SNR={stats_fail.get('theta_snr',0):.1f}")
        print(f"  成功帧 nr中位={stats_success['nr_median']:.1f} θ_SNR={stats_success.get('theta_snr',0):.1f}")

        plot_real_comparison(
            results_fail, stats_fail, '失败帧(042902)',
            results_success, stats_success, '成功帧(025221)',
            'NGC55 T3 Oiii 失败 vs 成功',
            os.path.join(REAL_OUT, 'comparison_ngc55.png')
        )

    # Victory对比
    if 'Victory_mosaic2_Lum_fail' in real_results and 'Victory_mosaic2_Lum_success' in real_results:
        results_fail, meta_fail = real_results['Victory_mosaic2_Lum_fail']
        results_success, meta_success = real_results['Victory_mosaic2_Lum_success']
        stats_fail = real_stats['Victory_mosaic2_Lum_fail']
        stats_success = real_stats['Victory_mosaic2_Lum_success']

        print(f"\nVictory mosaic2 Lum 对比:")
        print(f"  失败帧 nr中位={stats_fail['nr_median']:.1f} θ_SNR={stats_fail.get('theta_snr',0):.1f}")
        print(f"  成功帧 nr中位={stats_success['nr_median']:.1f} θ_SNR={stats_success.get('theta_snr',0):.1f}")

        plot_real_comparison(
            results_fail, stats_fail, '失败帧(062533)',
            results_success, stats_success, '成功帧(062145)',
            'Victory mosaic2 Lum 失败 vs 成功',
            os.path.join(REAL_OUT, 'comparison_victory.png')
        )

    # Part B分析文本
    part_b_lines = []
    part_b_lines.append("真实数据实验分析")
    part_b_lines.append("")

    for frame_key in ['NGC55_Oiii_fail', 'NGC55_Oiii_success',
                       'Victory_mosaic2_Lum_fail', 'Victory_mosaic2_Lum_success']:
        if frame_key not in real_stats:
            continue
        s = real_stats[frame_key]
        part_b_lines.append(f"  {frame_key}:")
        part_b_lines.append(f"    U={s['n_U']}颗, W={s['n_W']}颗, s0={s['s0']:.4f}\"/px, FOV={s['fov_diag']:.2f}°")
        part_b_lines.append(f"    nr: mean={s['nr_mean']:.1f} median={s['nr_median']:.1f} "
                          f"P90={s['nr_p90']:.1f} P95={s['nr_p95']:.1f} P99={s['nr_p99']:.1f} max={s['nr_max']}")
        part_b_lines.append(f"    nr>10:{s['nr_gt_10_count']} nr>20:{s['nr_gt_20_count']} nr>50:{s['nr_gt_50_count']}")
        part_b_lines.append(f"    θ_SNR={s.get('theta_snr','N/A'):.1f} θ_peak={s.get('theta_peak','N/A')}°")
        part_b_lines.append(f"    耗时={s['time_s']:.1f}s")

    part_b_lines.append("")
    part_b_lines.append("  失败vs成功帧对比:")
    if 'NGC55_Oiii_fail' in real_stats and 'NGC55_Oiii_success' in real_stats:
        sf = real_stats['NGC55_Oiii_fail']
        ss = real_stats['NGC55_Oiii_success']
        part_b_lines.append(f"    NGC55 Oiii:")
        part_b_lines.append(f"      失败nr中位={sf['nr_median']:.1f} vs 成功nr中位={ss['nr_median']:.1f}")
        part_b_lines.append(f"      失败θ_SNR={sf.get('theta_snr',0):.1f} vs 成功θ_SNR={ss.get('theta_snr',0):.1f}")
        if sf.get('theta_snr', 0) > 0 and ss.get('theta_snr', 0) > 0:
            part_b_lines.append(f"      SNR比={ss.get('theta_snr',0)/sf.get('theta_snr',1):.2f}x")

    if 'Victory_mosaic2_Lum_fail' in real_stats and 'Victory_mosaic2_Lum_success' in real_stats:
        vf = real_stats['Victory_mosaic2_Lum_fail']
        vs = real_stats['Victory_mosaic2_Lum_success']
        part_b_lines.append(f"    Victory mosaic2 Lum:")
        part_b_lines.append(f"      失败nr中位={vf['nr_median']:.1f} vs 成功nr中位={vs['nr_median']:.1f}")
        part_b_lines.append(f"      失败θ_SNR={vf.get('theta_snr',0):.1f} vs 成功θ_SNR={vs.get('theta_snr',0):.1f}")
        if vf.get('theta_snr', 0) > 0 and vs.get('theta_snr', 0) > 0:
            part_b_lines.append(f"      SNR比={vs.get('theta_snr',0)/vf.get('theta_snr',1):.2f}x")

    part_b_lines.append("")
    part_b_lines.append("  结论:")
    part_b_lines.append("    - 成功帧的nr中位应显著高于失败帧")
    part_b_lines.append("    - 成功帧的θ_SNR应显著高于失败帧")
    part_b_lines.append("    - 高nr对在s-θ和tx-ty空间应形成可见聚类")
    part_b_lines.append("    - 失败帧的高nr对量少且θ值不一致（无聚类）")

    sections.append(("Part B: 真实数据实验", part_b_lines))

    # ========================================================================
    # 写入分析报告
    # ========================================================================
    print(f"\n{'='*60}")
    print(f"写入分析报告: {ANALYSIS_TXT}")
    write_analysis_report(sections, ANALYSIS_TXT)
    print(f"{'='*60}")

    print("\n" + "=" * 80)
    print("实验完成!")
    print(f"合成数据图: {SYNTHETIC_OUT}")
    print(f"真实数据图: {REAL_OUT}")
    print(f"分析报告: {ANALYSIS_TXT}")
    print("=" * 80)


if __name__ == "__main__":
    main()

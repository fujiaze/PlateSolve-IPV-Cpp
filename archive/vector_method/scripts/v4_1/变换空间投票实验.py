"""V4.1 变换矩阵空间理论+实验分析 — Phase A投票机制的4D空间聚类行为

功能:
  Part A (合成数据): 模拟NGC55帧(N=250图像星,M=265 Gaia星,s0=0.96"/px,FOV=1.5°)
    在三个重叠率(10%/20%/50%)下运行5000次随机采样(i,j)对
    分析: nr投票分布 / θ直方图真vs噪声峰值 / 变换矩阵4D空间聚类
  Part B (真实数据): 在NGC55真实帧上运行对照实验

核心问题:
  1. 重叠率10%/20%/50%时，真匹配对和假匹配对各自的投票数分布
  2. 变换矩阵在4D空间(s,θ,tx,ty)中的聚类情况
  3. θ直方图中真峰值vs噪声峰值的形成机制
  4. 为什么NGC55(N=250,M=265)有时成功有时失败

用法: python 变换空间投票实验.py

输出目录: lib/plate_solve/logs/v4/experiments/
"""

import os, sys, math, time, json
import numpy as np

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ============================================================================
# 核心工具函数
# ============================================================================

def norm_and_angle(pts):
    norms = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    return norms, angles


def derive_transform(ui, wj):
    """从单对(ui, wj)推导相似变换 T = (s, θ_deg, tx, ty)"""
    norm_u = math.sqrt(ui[0]**2 + ui[1]**2)
    norm_w = math.sqrt(wj[0]**2 + wj[1]**2)
    if norm_u < 1e-10 or norm_w < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    s = norm_u / norm_w
    ang_u = math.atan2(ui[1], ui[0])
    ang_w = math.atan2(wj[1], wj[0])
    theta = ang_u - ang_w
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta < -math.pi:
        theta += 2 * math.pi
    theta_deg = math.degrees(theta)
    ct, st = math.cos(theta), math.sin(theta)
    tx = ui[0] - s * (ct * wj[0] - st * wj[1])
    ty = ui[1] - s * (st * wj[0] + ct * wj[1])
    return (s, theta_deg, tx, ty)


def apply_transform_to_all(W, s, theta_deg, tx, ty):
    theta = math.radians(theta_deg)
    ct, st = math.cos(theta), math.sin(theta)
    Wx_rot = s * (ct * W[:, 0] - st * W[:, 1]) + tx
    Wy_rot = s * (st * W[:, 0] + ct * W[:, 1]) + ty
    return np.column_stack([Wx_rot, Wy_rot])


def count_votes(T, U, W, s0, tol_arcsec=5.0, s_min=0.85, s_max=1.15):
    """统计变换T下匹配星对数nr"""
    from scipy.spatial import cKDTree
    s, theta_deg, tx, ty = T
    thr = tol_arcsec * s0
    Wt = apply_transform_to_all(W, s, theta_deg, tx, ty)
    tree = cKDTree(U)
    dists, idxs = tree.query(Wt, distance_upper_bound=thr)
    norm_U = np.linalg.norm(U, axis=1)
    norm_W = np.linalg.norm(W, axis=1)
    nr = 0
    s_ratio_vals = []
    for j in range(len(Wt)):
        i = idxs[j]
        d = dists[j]
        if i < len(U) and d < thr:
            s_pair = norm_U[i] / max(norm_W[j], 1e-10)
            if s_min <= s_pair <= s_max:
                nr += 1
                s_ratio_vals.append(s_pair)
    return nr


# ============================================================================
# Part A: 合成数据生成 (模拟NGC55帧)
# ============================================================================

def generate_ngc55_synthetic(overlap_ratio, seed_data=42):
    """生成模拟NGC55帧的合成数据

    NGC55参数: N_img=250, M_gaia=265, s0=0.96"/px, FOV=1.5°
    真变换: s=1.005, θ=-90.5°, tx=ty=0

    Returns:
        U: (N,2) 图像星arcsec向量 (从图像中心出发)
        W: (M,2) Gaia星arcsec向量
        is_match: (N,) bool 哪些图像星有真匹配
        match_gaia_idx: np.array 真匹配对应的Gaia星索引
        true_T: (s, θ_deg, tx, ty) 真变换
    """
    rng_data = np.random.RandomState(seed_data)

    N_img = 250
    M_gaia = 265
    s0 = 0.96
    fov_half_asec = 1.5 * 3600.0 / 2.0  # 2700"
    s_true = 1.005
    theta_true = np.radians(-90.5)
    tx_true = 0.0
    ty_true = 0.0
    sigma_img = 0.3  # 质心噪声(角秒)

    ct, st = np.cos(theta_true), np.sin(theta_true)
    R = np.array([[ct, -st], [st, ct]])

    # 生成Gaia星 (标准坐标，角秒)
    rng_gaia = np.random.RandomState(seed_data + 100)
    r_gaia = fov_half_asec * np.sqrt(rng_gaia.random(M_gaia))
    ang_gaia = rng_gaia.random(M_gaia) * 2 * np.pi
    W = np.column_stack([r_gaia * np.cos(ang_gaia), r_gaia * np.sin(ang_gaia)])

    # 真匹配星
    n_match = int(N_img * overlap_ratio)
    n_match = min(n_match, M_gaia)
    match_gaia_idx = rng_data.choice(M_gaia, n_match, replace=False)
    match_gaia = W[match_gaia_idx]
    match_img = s_true * (match_gaia @ R.T) + np.array([tx_true, ty_true])
    match_img += rng_data.randn(n_match, 2) * sigma_img

    # 噪声星(随机分布在FOV内)
    n_noise = N_img - n_match
    rng_noise = np.random.RandomState(seed_data + 200)
    r_noise = fov_half_asec * np.sqrt(rng_noise.random(n_noise))
    ang_noise = rng_noise.random(n_noise) * 2 * np.pi
    noise_img = np.column_stack([r_noise * np.cos(ang_noise), r_noise * np.sin(ang_noise)])

    U = np.vstack([match_img, noise_img])
    is_match = np.array([True] * n_match + [False] * n_noise)

    true_T = (s_true, np.degrees(theta_true), tx_true, ty_true)

    return U, W, is_match, match_gaia_idx, true_T, s0, fov_half_asec * 2


# ============================================================================
# Part A: 采样实验
# ============================================================================

def run_sampling(U, W, is_match, match_gaia_idx, s0, true_T, n_samples=5000, rng_seed=12345):
    """随机采样(i,j)对，推导变换并统计投票"""
    N = len(U)
    M = len(W)
    rng = np.random.RandomState(rng_seed)
    results = []
    sampled = set()
    attempts = 0
    max_attempts = n_samples * 100

    match_idx_set = set(range(len(match_gaia_idx)))
    match_gaia_set = set(match_gaia_idx)

    while len(results) < n_samples and attempts < max_attempts:
        i = rng.randint(0, N)
        j = rng.randint(0, M)
        key = (i, j)
        if key in sampled:
            attempts += 1
            continue
        sampled.add(key)
        attempts += 1

        T = derive_transform(U[i], W[j])
        s, theta_deg, tx, ty = T
        if s < 0.7 or s > 1.3:
            continue

        nr = count_votes(T, U, W, s0)
        is_true_pair = is_match[i] and (i in match_idx_set) and (match_gaia_idx[i] == j)

        results.append({
            'i': i, 'j': j,
            's': s,
            'theta_deg': theta_deg,
            'tx': tx,
            'ty': ty,
            'nr': nr,
            'is_true_pair': is_true_pair
        })

    return results


# ============================================================================
# Part A: 可视化
# ============================================================================

def plot_results(results, overlap_ratio, s0, true_T, out_dir, rng_seed=None):
    """为单个重叠率绘制4子图分析

    子图1: θ直方图 (蓝色=真匹配对贡献nr, 红色=假匹配对贡献nr)
    子图2: s vs θ 散点图 (绿色=真匹配对, 灰色=假匹配对, 大小∝nr)
    子图3: tx vs ty 散点图 (绿色=真匹配对, 灰色=假匹配对)
    子图4: nr投票数分布 (真/假分别直方图)
    """
    true_results = [r for r in results if r['is_true_pair']]
    false_results = [r for r in results if not r['is_true_pair']]
    n_true = len(true_results)
    n_false = len(false_results)

    print(f"  真匹配对采样: {n_true}, 假匹配对采样: {n_false}")

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # ---- 子图1: θ直方图 (真vs假nr贡献分离) ----
    ax = axes[0, 0]
    theta_bins = 180
    bin_width = 360.0 / theta_bins

    hist_true = np.zeros(theta_bins)
    hist_false = np.zeros(theta_bins)
    for r in results:
        nr = r['nr']
        th = r['theta_deg']
        bi = int(((th + 180.0) % 360.0) / bin_width)
        if 0 <= bi < theta_bins:
            if r['is_true_pair']:
                hist_true[bi] += nr
            else:
                hist_false[bi] += nr

    theta_centers = np.linspace(-180, 180, theta_bins)
    ax.bar(theta_centers, hist_true, width=bin_width, alpha=0.7,
           color='steelblue', edgecolor='none', label=f'真匹配对贡献 (n={n_true}对)')
    ax.bar(theta_centers, hist_false, width=bin_width, alpha=0.5,
           color='salmon', edgecolor='none', label=f'假匹配对贡献 (n={n_false}对)')

    # SNR计算
    peak_val = hist_true.max()
    peak_bin = hist_true.argmax()
    peak_theta = peak_bin * bin_width - 180.0
    exclude_half = int(10.0 / bin_width)
    mask = np.ones(theta_bins, dtype=bool)
    for d in range(-exclude_half, exclude_half + 1):
        mask[(peak_bin + d) % theta_bins] = False
    bg = hist_true[mask]
    median_bg = np.median(bg) if len(bg) > 0 else 1.0
    snr = peak_val / max(median_bg, 1e-10)

    ax.axvline(x=peak_theta, color='darkblue', linestyle='--', linewidth=2,
               label=f'真峰值θ={peak_theta:.1f} SnR={snr:.1f}')
    ax.axvline(x=true_T[1], color='green', linestyle=':', linewidth=2,
               label=f'真值θ={true_T[1]:.1f}')

    ax.set_xlabel('θ ()', fontsize=11)
    ax.set_ylabel('nr (nr)', fontsize=11)
    ax.set_title(f'  nr 重叠率={overlap_ratio:.0%}   s0={s0:.2f}/px', fontsize=12)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-185, 185)

    # ---- 子图2: s vs  散点图 ----
    ax = axes[0, 1]
    if false_results:
        fr_theta = [r['theta_deg'] for r in false_results]
        fr_s = [r['s'] for r in false_results]
        fr_nr = [r['nr'] for r in false_results]
        sizes_f = np.clip(np.array(fr_nr) * 2, 1, 80)
        ax.scatter(fr_theta, fr_s, c='lightgray', alpha=0.35, s=sizes_f,
                   edgecolors='none', label=f'假匹配对 (n={n_false})')
    if true_results:
        tr_theta = [r['theta_deg'] for r in true_results]
        tr_s = [r['s'] for r in true_results]
        tr_nr = [r['nr'] for r in true_results]
        sizes_t = np.clip(np.array(tr_nr) * 4, 10, 200)
        ax.scatter(tr_theta, tr_s, c='limegreen', alpha=0.8, s=sizes_t,
                   edgecolors='darkgreen', linewidths=0.5, label=f'真匹配对 (n={n_true})')

    ax.axhline(y=true_T[0], color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f's真值={true_T[0]:.4f}')
    ax.axvline(x=true_T[1], color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel(' ()', fontsize=11)
    ax.set_ylabel('s ()', fontsize=11)
    ax.set_title(f's vs   重叠率={overlap_ratio:.0%}', fontsize=12)
    ax.legend(loc='upper right', fontsize=7)

    # ---- 子图3: tx vs ty 散点图 ----
    ax = axes[1, 0]
    if false_results:
        fr_tx = [r['tx'] for r in false_results]
        fr_ty = [r['ty'] for r in false_results]
        ax.scatter(fr_tx, fr_ty, c='lightgray', alpha=0.35, s=sizes_f,
                   edgecolors='none', label=f'假匹配对 (n={n_false})')
    if true_results:
        tr_tx = [r['tx'] for r in true_results]
        tr_ty = [r['ty'] for r in true_results]
        ax.scatter(tr_tx, tr_ty, c='limegreen', alpha=0.8, s=sizes_t,
                   edgecolors='darkgreen', linewidths=0.5, label=f'真匹配对 (n={n_true})')

    ax.axhline(y=true_T[3], color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'ty真值={true_T[3]:.1f}"')
    ax.axvline(x=true_T[2], color='red', linestyle=':', linewidth=1.5, alpha=0.7, label=f'tx真值={true_T[2]:.1f}"')
    ax.set_xlabel('tx (")', fontsize=11)
    ax.set_ylabel('ty (")', fontsize=11)
    ax.set_title(f'tx vs ty  重叠率={overlap_ratio:.0%}', fontsize=12)
    ax.legend(loc='upper right', fontsize=7)

    # ---- 子图4: nr投票数分布 ----
    ax = axes[1, 1]
    bins_nr = np.linspace(0, max(r['nr'] for r in results) + 2, 60)
    if true_results:
        tr_nr = [r['nr'] for r in true_results]
        ax.hist(tr_nr, bins=bins_nr, alpha=0.7, color='steelblue', edgecolor='white',
                label=f'真匹配对 (n={n_true}, med={np.median(tr_nr):.0f})')
    if false_results:
        fr_nr = [r['nr'] for r in false_results]
        ax.hist(fr_nr, bins=bins_nr, alpha=0.5, color='salmon', edgecolor='white',
                label=f'假匹配对 (n={n_false}, med={np.median(fr_nr):.0f})')
    ax.set_xlabel('nr ()', fontsize=11)
    ax.set_ylabel('', fontsize=11)
    ax.set_title(f'nr   重叠率={overlap_ratio:.0%}', fontsize=12)
    ax.set_yscale('log')
    ax.legend(loc='upper right', fontsize=8)

    plt.suptitle(f'NGC55  (N=250, M=265, s0={s0:.2f}"/px, FOV=1.5)|   ={overlap_ratio:.0%}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    filename = os.path.join(out_dir, f'transform_space_overlap_{overlap_ratio:.0%}.png'.replace('%', 'pct'))
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    : {filename}")

    # SNR
    theta_stats = {
        'peak_theta': float(peak_theta),
        'snr': float(snr),
        'median_bg': float(median_bg),
        'peak_val': float(peak_val),
        'true_theta': true_T[1]
    }
    return theta_stats


# ============================================================================
# Part A: 统计汇总
# ============================================================================

def plot_summary_figure(all_results_list, all_theta_stats, overlap_ratios, s0, out_dir):
    """汇总图: 3重叠率对比"""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', 'D']

    # (a) SNR vs 重叠率
    ax = axes[0]
    snrs = [all_theta_stats[ri]['snr'] for ri in range(len(overlap_ratios))]
    ax.plot(overlap_ratios, snrs, 'o-', color='darkblue', markersize=10, linewidth=2)
    for i, (ov, s_val) in enumerate(zip(overlap_ratios, snrs)):
        ax.annotate(f'{s_val:.0f}', (ov, s_val), textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('重叠率', fontsize=12)
    ax.set_ylabel('θ SNR', fontsize=12)
    ax.set_title('θ峰值SNR vs 重叠率', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.55)

    # (b) nr中位数 vs 重叠率
    ax = axes[1]
    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        tr_nr = [r['nr'] for r in results if r['is_true_pair']]
        fr_nr = [r['nr'] for r in results if not r['is_true_pair']]
        if tr_nr:
            ax.scatter(ov, np.median(tr_nr), c=colors[ri], marker='o', s=150,
                       edgecolors='black', linewidths=1, label=f'真匹配' if ri == 0 else '')
        if fr_nr:
            ax.scatter(ov, np.median(fr_nr), c=colors[ri], marker='^', s=80,
                       edgecolors='black', linewidths=0.5, label=f'假匹配' if ri == 0 else '')
    ax.set_xlabel('重叠率', fontsize=12)
    ax.set_ylabel('nr中位数', fontsize=12)
    ax.set_title('真/假匹配对nr中位数', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.55)

    # (c) 真匹配s聚类紧密度
    ax = axes[2]
    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        tr_s = [r['s'] for r in results if r['is_true_pair']]
        if tr_s:
            s_std = np.std(tr_s)
            ax.bar(ri, s_std, color=colors[ri], alpha=0.7, edgecolor='black', linewidth=0.5,
                   label=f'重叠率={ov:.0%}')
            ax.text(ri, s_std + max(0.0001, s_std * 0.02), f'{s_std:.5f}',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(overlap_ratios)))
    ax.set_xticklabels([f'{ov:.0%}' for ov in overlap_ratios])
    ax.set_ylabel('sσ', fontsize=12)
    ax.set_title('真匹配对s聚类紧密度', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # (d) 假匹配θ分散度
    ax = axes[3]
    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        fr_theta = [r['theta_deg'] for r in results if not r['is_true_pair']]
        if fr_theta:
            theta_std = np.std(fr_theta)
            ax.bar(ri, theta_std, color=colors[ri], alpha=0.7, edgecolor='black', linewidth=0.5,
                   label=f'重叠率={ov:.0%}')
            ax.text(ri, theta_std + 0.5, f'{theta_std:.1f}',
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(overlap_ratios)))
    ax.set_xticklabels([f'{ov:.0%}' for ov in overlap_ratios])
    ax.set_ylabel('θσ (°)', fontsize=12)
    ax.set_title('假匹配对θ分散度', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'NGC55合成数据实验汇总 (N=250, M=265, s0={s0:.2f}"/px, FOV=1.5°)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    filename = os.path.join(out_dir, 'transform_space_summary.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  汇总图: {filename}")


# ============================================================================
# Part A: nr分布深度分析图
# ============================================================================

def plot_nr_distribution_detail(all_results_list, overlap_ratios, out_dir):
    """nr分布的深度分析: 直方图+累积分布"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['steelblue', 'darkorange', 'forestgreen']

    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        tr_nr = [r['nr'] for r in results if r['is_true_pair']]
        fr_nr = [r['nr'] for r in results if not r['is_true_pair']]

        if not tr_nr:
            continue

        max_nr = max(max(tr_nr), max(fr_nr) if fr_nr else 0)

        # 重叠直方图
        ax = axes[ri]
        bins = np.arange(0, max_nr + 3, 1)
        if tr_nr:
            ax.hist(tr_nr, bins=bins, alpha=0.7, color='steelblue', edgecolor='white',
                    label=f'真匹配 (n={len(tr_nr)}, med={np.median(tr_nr):.1f})')
        if fr_nr:
            ax.hist(fr_nr, bins=bins, alpha=0.5, color='salmon', edgecolor='white',
                    label=f'假匹配 (n={len(fr_nr)}, med={np.median(fr_nr):.1f})')

        # KS统计
        if tr_nr and fr_nr:
            from scipy import stats
            ks_stat, ks_p = stats.ks_2samp(tr_nr, fr_nr)
            ax.set_title(f'重叠率={ov:.0%}  真中位={np.median(tr_nr):.0f}  假中位={np.median(fr_nr):.0f}  '
                         f'KS={ks_stat:.3f} p={ks_p:.1e}', fontsize=10)

        ax.set_xlabel('nr (投票数)', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_yscale('log')
        ax.legend(fontsize=8)

    plt.suptitle('nr投票数分布 — 真匹配对(蓝) vs 假匹配对(红)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    filename = os.path.join(out_dir, 'transform_space_nr_detail.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  nr详细图: {filename}")


# ============================================================================
# Part B: 真实数据实验 (NGC55 Oiii帧)
# ============================================================================

def load_real_ngc55_data(fits_path, n_img_total=250, mag_limit=16.0):
    """读取NGC55真实FITS帧，提取图像星和Gaia星"""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))

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

    print(f"   图像: {w_img}x{h_img}, fl={fl}mm, ps={ps}um, s0={s0:.4f}\"/px")

    detector = StarDetector(params=SDetParamsPy(fitRadius=0, maxStars=500))
    det = detector.detect_ex(img.data)

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

    print(f"   星点: {det.count}颗 ({nsat}饱和), 选取{len(sel_idx)}颗")

    # 解析RA/Dec
    kw_dict = {}
    for kw in img.keywords:
        kw_dict[kw.name.upper()] = kw.value.strip() if isinstance(kw.value, str) else str(kw.value)

    ra_str = kw_dict.get('OBJCTRA', '0 0 0')
    dec_str = kw_dict.get('OBJCTDEC', '0 0 0')
    try:
        ra_parts = ra_str.strip().split()
        cra = (float(ra_parts[0]) + float(ra_parts[1]) / 60.0 + float(ra_parts[2]) / 3600.0) * 15.0
    except Exception:
        cra = 0.0
    try:
        dec_clean = dec_str.strip()
        sign = -1.0 if dec_clean.startswith('-') else 1.0
        dec_clean = dec_clean.lstrip('+-')
        dec_parts = dec_clean.split()
        cdec = sign * (abs(float(dec_parts[0])) + float(dec_parts[1]) / 60.0 + float(dec_parts[2]) / 3600.0)
    except Exception:
        cdec = 0.0

    fov_diag = math.sqrt(w_img * w_img + h_img * h_img) * s0 / 3600.0
    query_radius = max(fov_diag * 0.7, 1.0)

    print(f"   FOV={fov_diag:.2f}°, RA={cra:.4f}°, Dec={cdec:.4f}°, 查询半径={query_radius:.2f}°")

    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    try:
        ra_t, dec_t, mag_t = gaia.cone_search(cra, cdec, query_radius, mag_limit)
    finally:
        gaia.close()

    cos_dec = math.cos(math.radians(cdec))
    wx_all = (np.array(ra_t) - cra) * 3600.0 * cos_dec
    wy_all = (np.array(dec_t) - cdec) * 3600.0
    W = np.column_stack([wx_all, wy_all])

    print(f"   Gaia查询: {len(W)}颗星 (极限星等{mag_limit})")

    img.close()
    detector.close()

    return U, W, s0, fov_diag


def run_real_experiment(fits_path, frame_label, out_dir, K=20000, s0_override=None):
    """对真实帧运行采样实验"""
    print(f"\n{'='*60}")
    print(f"真实数据实验: {frame_label}")
    print(f"  文件: {fits_path}")
    print(f"{'='*60}")

    U, W, s0, fov_diag = load_real_ngc55_data(fits_path)
    N, M = len(U), len(W)
    if N < 5 or M < 5:
        print(f"  警告: 星数不足 U={N} W={M}, 跳过")
        return None, None

    if s0_override is not None:
        s0 = s0_override

    print(f"  U={N}颗, W={M}颗, s0={s0:.4f}\"/px, FOV={fov_diag:.2f}°")

    actual_K = min(K, N * M)
    rng = np.random.RandomState(42)
    results = []
    sampled = set()
    attempts, max_attempts = 0, actual_K * 50

    t0 = time.time()
    while len(results) < actual_K and attempts < max_attempts:
        i = rng.randint(0, N)
        j = rng.randint(0, M)
        key = (i, j)
        if key in sampled:
            attempts += 1
            continue
        sampled.add(key)
        attempts += 1
        T = derive_transform(U[i], W[j])
        s, theta_deg, tx, ty = T
        if s < 0.7 or s > 1.3:
            continue
        nr = count_votes(T, U, W, s0)
        results.append({
            'i': i, 'j': j, 's': s, 'theta_deg': theta_deg, 'tx': tx, 'ty': ty, 'nr': nr
        })

    dt = time.time() - t0
    print(f"  采样: {len(results)}对, 耗时{dt:.1f}s ({len(results)/max(dt,0.01):.0f}对/s)")

    nr_arr = np.array([r['nr'] for r in results])
    stats = {
        'n_U': N, 'n_W': M, 's0': s0, 'fov_diag': fov_diag,
        'n_samples': len(results), 'time_s': dt,
        'nr_mean': float(np.mean(nr_arr)), 'nr_median': float(np.median(nr_arr)),
        'nr_p95': float(np.percentile(nr_arr, 95)), 'nr_p99': float(np.percentile(nr_arr, 99)),
        'nr_max': int(nr_arr.max()),
        'nr_gt_10': int(np.sum(nr_arr > 10)), 'nr_gt_20': int(np.sum(nr_arr > 20))
    }
    print(f"  nr: 中位={stats['nr_median']:.1f} P95={stats['nr_p95']:.1f} P99={stats['nr_p99']:.1f} "
          f"max={stats['nr_max']} >10={stats['nr_gt_10']} >20={stats['nr_gt_20']}")

    # θ SNR (在排序前计算)
    theta_bins = 360
    bw = 360.0 / theta_bins
    th_hist = np.zeros(theta_bins)
    for r in results:
        bi = int(((r['theta_deg'] + 180.0) % 360.0) / bw)
        if 0 <= bi < theta_bins:
            th_hist[bi] += r['nr']
    peak_val = th_hist.max()
    peak_bin = th_hist.argmax()
    peak_theta = peak_bin * bw - 180.0
    ex = int(10.0 / bw)
    mask = np.ones(theta_bins, dtype=bool)
    for d in range(-ex, ex + 1):
        mask[(peak_bin + d) % theta_bins] = False
    bg = th_hist[mask]
    median_bg = np.median(bg) if len(bg) > 0 else 1.0
    snr = peak_val / max(median_bg, 1e-10)
    stats['theta_snr'] = float(snr)
    stats['theta_peak'] = float(peak_theta)

    print(f"  θ SNR: {snr:.1f}, 峰值θ: {peak_theta:.1f}°")

    # 转换为dict格式再绘图
    results_dict = {k: [r[k] for r in results] for k in results[0].keys()}
    plot_real_frame(results_dict, frame_label, stats, out_dir)

    # Top-10
    results.sort(key=lambda x: -x['nr'])
    top10 = results[:10]
    print(f"  Top-10高nr对:")
    for rank, r in enumerate(top10):
        print(f"    #{rank+1}: nr={r['nr']:4d} s={r['s']:.4f} θ={r['theta_deg']:8.2f}° "
              f"tx={r['tx']:8.1f}\" ty={r['ty']:8.1f}\"")
    if len(top10) >= 2:
        tops = [r['theta_deg'] for r in top10]
        print(f"    θ范围: [{min(tops):.2f}°, {max(tops):.2f}°] 跨度={max(tops)-min(tops):.2f}°")

    return results_dict, stats


def plot_real_frame(results, frame_label, stats, out_dir):
    """绘制真实帧的矩阵空间分析图"""
    nr_arr = np.array(results['nr'])
    s_arr = np.array(results['s'])
    theta_arr = np.array(results['theta_deg'])
    tx_arr = np.array(results['tx'])
    ty_arr = np.array(results['ty'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # nr直方图
    ax = axes[0, 0]
    ax.hist(nr_arr, bins=80, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(stats['nr_median'], color='green', linestyle='--', linewidth=1.5, label=f"中位={stats['nr_median']:.0f}")
    ax.axvline(stats['nr_p95'], color='orange', linestyle='--', linewidth=1.5, label=f"P95={stats['nr_p95']:.0f}")
    ax.axvline(stats['nr_p99'], color='red', linestyle='--', linewidth=1.5, label=f"P99={stats['nr_p99']:.0f}")
    ax.set_xlabel('nr')
    ax.set_ylabel('频数')
    ax.set_title(f'{frame_label} nr分布')
    ax.set_yscale('log')
    ax.legend(fontsize=8)

    # s vs θ
    ax = axes[0, 1]
    vmax = min(stats['nr_p99'] * 1.2, stats['nr_max'])
    sc = ax.scatter(theta_arr, s_arr, c=nr_arr, cmap='plasma', alpha=0.5, s=3, vmin=0, vmax=max(vmax, 1))
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('nr')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('s')
    ax.set_title(f'{frame_label} s vs θ (nr着色)')

    # tx vs ty
    ax = axes[1, 0]
    sc = ax.scatter(tx_arr, ty_arr, c=nr_arr, cmap='plasma', alpha=0.5, s=3, vmin=0, vmax=max(vmax, 1))
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('nr')
    ax.set_xlabel('tx (")')
    ax.set_ylabel('ty (")')
    ax.set_title(f'{frame_label} tx vs ty (nr着色)')

    # θ加权直方图
    ax = axes[1, 1]
    theta_bins = 360
    bw = 360.0 / theta_bins
    th_hist = np.zeros(theta_bins)
    for nr_v, th_v in zip(nr_arr, theta_arr):
        bi = int(((th_v + 180.0) % 360.0) / bw)
        if 0 <= bi < theta_bins:
            th_hist[bi] += nr_v
    peak_val = th_hist.max()
    peak_bin = th_hist.argmax()
    peak_theta = peak_bin * bw - 180.0
    ex = int(10.0 / bw)
    mask = np.ones(theta_bins, dtype=bool)
    for d in range(-ex, ex + 1):
        mask[(peak_bin + d) % theta_bins] = False
    bg = th_hist[mask]
    median_bg = np.median(bg) if len(bg) > 0 else 1.0
    snr = peak_val / max(median_bg, 1e-10)

    theta_c = np.linspace(-180, 180, theta_bins)
    ax.bar(theta_c, th_hist, width=bw, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axvline(x=peak_theta, color='red', linestyle='-', linewidth=2, label=f'SNR={snr:.1f}')
    ax.set_xlabel('θ (°)')
    ax.set_ylabel('加权nr')
    ax.set_title(f'{frame_label} θ加权直方图 SNR={snr:.1f}')
    ax.legend(fontsize=8)

    plt.suptitle(f'{frame_label} — U={stats["n_U"]} W={stats["n_W"]} s0={stats["s0"]:.4f}"/px FOV={stats["fov_diag"]:.2f}°',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    filename = os.path.join(out_dir, f'{frame_label}_matrix_space.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    stats['theta_snr'] = float(snr)


def plot_real_comparison(results_a, stats_a, label_a, results_b, stats_b, label_b, title_suffix, out_dir):
    """对比两帧真实数据"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'a': 'red', 'b': 'blue'}

    for idx, (res, st, label, color) in enumerate([
        (results_a, stats_a, label_a, 'red'),
        (results_b, stats_b, label_b, 'blue')
    ]):
        nr_arr = np.array(res['nr'])
        s_arr = np.array(res['s'])
        theta_arr = np.array(res['theta_deg'])
        tx_arr = np.array(res['tx'])
        ty_arr = np.array(res['ty'])

        axes[0, 0].hist(nr_arr, bins=80, alpha=0.5, color=color, label=f'{label}')
        axes[0, 1].scatter(theta_arr, s_arr, c=color, alpha=0.3, s=2, label=label)
        axes[0, 2].scatter(tx_arr, ty_arr, c=color, alpha=0.3, s=2, label=label)

        theta_bins = 360
        bw = 360.0 / theta_bins
        th_hist = np.zeros(theta_bins)
        for nr_v, th_v in zip(nr_arr, theta_arr):
            bi = int(((th_v + 180.0) % 360.0) / bw)
            if 0 <= bi < theta_bins:
                th_hist[bi] += nr_v
        theta_c = np.linspace(-180, 180, theta_bins)
        axes[1, 0].plot(theta_c, th_hist, color=color, alpha=0.7, linewidth=1,
                        label=f'{label} SNR={st.get("theta_snr", 0):.1f}')

        axes[1, 1].hist(s_arr, bins=80, alpha=0.4, color=color, label=f'{label}')

        sorted_nr = np.sort(nr_arr)[::-1]
        axes[1, 2].plot(np.cumsum(sorted_nr), color=color, alpha=0.8, linewidth=1.5, label=label)

    axes[0, 0].set_xlabel('nr'); axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('nr分布对比'); axes[0, 0].set_yscale('log'); axes[0, 0].legend(fontsize=7)

    axes[0, 1].set_xlabel('θ (°)'); axes[0, 1].set_ylabel('s')
    axes[0, 1].set_title('s vs θ'); axes[0, 1].legend(fontsize=7)

    axes[0, 2].set_xlabel('tx (")'); axes[0, 2].set_ylabel('ty (")')
    axes[0, 2].set_title('tx vs ty'); axes[0, 2].legend(fontsize=7)

    axes[1, 0].set_xlabel('θ (°)'); axes[1, 0].set_ylabel('加权nr')
    axes[1, 0].set_title('θ加权直方图对比'); axes[1, 0].legend(fontsize=7)

    axes[1, 1].set_xlabel('s'); axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('s分布对比'); axes[1, 1].legend(fontsize=7)

    axes[1, 2].set_xlabel('排名 (nr降序)'); axes[1, 2].set_ylabel('累积nr')
    axes[1, 2].set_title('nr累积分布'); axes[1, 2].legend(fontsize=7)

    plt.suptitle(title_suffix, fontsize=13, fontweight='bold')
    plt.tight_layout()
    filename = os.path.join(out_dir, f'comparison_{title_suffix.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  对比图: {filename}")


# ============================================================================
# 分析报告
# ============================================================================

def write_report(all_results_list, all_theta_stats, overlap_ratios, true_T, s0,
                 ngc55_fail_stats, ngc55_success_stats, out_dir):
    """生成Markdown分析报告"""
    lines = []
    lines.append("# V4.1 变换矩阵空间理论+实验分析报告")
    lines.append(f"")
    lines.append(f"**生成时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append("---")
    lines.append("")
    lines.append("## Part A: 合成数据实验 (模拟NGC55)")
    lines.append("")
    lines.append(f"**参数**: N=250图像星, M=265 Gaia星, s0={s0:.2f}\"/px, FOV=1.5°")
    lines.append(f"**真变换**: s={true_T[0]:.4f}, θ={true_T[1]:.1f}°, tx={true_T[2]:.1f}\", ty={true_T[3]:.1f}\"")
    lines.append("**采样**: 每配置5000对随机(i,j)")
    lines.append("**噪声**: 质心σ=0.3\"")
    lines.append("")
    lines.append("### 实验结果汇总")
    lines.append("")
    lines.append("| 重叠率 | 真匹配对n | 假匹配对n | 真nr中位 | 假nr中位 | nr分离比 | θ SNR | θ峰值° | 真sσ | 假θσ° |")
    lines.append("|--------|----------|----------|---------|---------|---------|-------|--------|------|-------|")

    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        ts = all_theta_stats[ri]
        tr_nr = [r['nr'] for r in results if r['is_true_pair']]
        fr_nr = [r['nr'] for r in results if not r['is_true_pair']]
        n_tr = len(tr_nr)
        n_fr = len(fr_nr)
        med_tr = np.median(tr_nr) if tr_nr else 0
        med_fr = np.median(fr_nr) if fr_nr else 0
        sep = med_tr / max(med_fr, 1)
        tr_s = [r['s'] for r in results if r['is_true_pair']]
        fr_th = [r['theta_deg'] for r in results if not r['is_true_pair']]
        s_std = np.std(tr_s) if tr_s else 0
        th_std = np.std(fr_th) if fr_th else 0
        lines.append(f"| {ov:.0%} | {n_tr} | {n_fr} | {med_tr:.1f} | {med_fr:.1f} | "
                     f"{sep:.1f}x | {ts['snr']:.1f} | {ts['peak_theta']:.1f} | "
                     f"{s_std:.5f} | {th_std:.1f} |")

    lines.append("")
    lines.append("### 关键发现")
    lines.append("")

    # 分析1: nr分离度
    lines.append("#### 1. nr投票数分布分析")
    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        tr_nr = [r['nr'] for r in results if r['is_true_pair']]
        fr_nr = [r['nr'] for r in results if not r['is_true_pair']]
        if tr_nr and fr_nr:
            from scipy import stats
            ks_stat, ks_p = stats.ks_2samp(tr_nr, fr_nr)
            p95_false = np.percentile(fr_nr, 95)
            tr_above_p95 = sum(1 for x in tr_nr if x > p95_false) / len(tr_nr)
            lines.append(f"- **重叠率{ov:.0%}**: 真中位={np.median(tr_nr):.1f}, 假中位={np.median(fr_nr):.1f}, "
                         f"KS={ks_stat:.3f}(p={ks_p:.1e}), 真匹配{tr_above_p95*100:.1f}%高于假P95")
    lines.append("")

    # 分析2: θ峰值形成机制
    lines.append("#### 2. θ直方图峰值形成机制")
    lines.append("")
    lines.append("θ直方图中**真峰值**(蓝色)和**噪声背景**(红色)的来源: ")
    lines.append("- **真峰值贡献**: 真匹配对(i,j)推导出的θ高度一致, 集中在真值θ_true附近(±几度内), nr也高")
    lines.append("- **噪声背景贡献**: 假匹配对(i,j)推导出的θ随机分布在[-180°,180°], nr也低")
    lines.append("- SNR = 真峰高度 / 中位背景 = 衡量可分性的核心指标")
    lines.append("")
    for ri, ov in enumerate(overlap_ratios):
        ts = all_theta_stats[ri]
        lines.append(f"- 重叠率{ov:.0%}: θ_SNR={ts['snr']:.1f}, "
                     f"真峰值θ={ts['peak_theta']:.1f}° (真值θ={true_T[1]:.1f}°, 偏差{abs(ts['peak_theta']-true_T[1]):.1f}°)")
    lines.append("")
    lines.append("**结论**: 重叠率越高 → 真匹配对越多 → 真峰越高 → SNR越大 → θ可被可靠检测")
    lines.append("")

    # 分析3: 变换矩阵4D空间聚类
    lines.append("#### 3. 变换矩阵4D空间聚类分析")
    lines.append("")
    lines.append("在(s,θ,tx,ty)空间中: ")
    lines.append("- 真匹配对推导的T**紧密聚类**在真值附近 (s≈1.005, θ≈-90.5°, tx≈ty≈0)")
    lines.append("- 假匹配对推导的T**散乱分布**在整个空间")
    lines.append("")
    for ri, ov in enumerate(overlap_ratios):
        results = all_results_list[ri]
        tr_s = [r['s'] for r in results if r['is_true_pair']]
        tr_tx = [r['tx'] for r in results if r['is_true_pair']]
        tr_ty = [r['ty'] for r in results if r['is_true_pair']]
        if tr_s:
            lines.append(f"- 重叠率{ov:.0%}: s聚类σ={np.std(tr_s):.5f}, "
                         f"tx聚类σ={np.std(tr_tx):.1f}\", ty聚类σ={np.std(tr_ty):.1f}\"")
    lines.append("")

    # 分析4: NGC55成功/失败机制
    lines.append("#### 4. NGC55成功/失败机制分析")
    lines.append("")
    lines.append("NGC55(N=250, M=265, FOV≈1.5°)有时成功有时失败的原因: ")
    lines.append("")
    lines.append("1. **重叠率波动**: 不同滤镜(Oiii/Hα/Lum)的星点检测结果不同")
    lines.append("   - 窄带滤镜(Oiii/Hα)星点少 → 重叠率低 → SNR不足")
    lines.append("   - 宽带滤镜(Lum/Red/Green/Blue)星点多 → 重叠率高 → SNR足够")
    lines.append("2. **质心噪声**: 暗星质心误差大 → 推导的T有偏差 → nr降低")
    lines.append("3. **Phase 0密度匹配**: gaia查询星数远超n_target时, Phase C扩充失败")
    lines.append("   - n_gaia=421 vs n_target=375 → 偏差+12%(超标)")
    lines.append("4. **PROSAC随机性**: 初始抽样顺序影响Phase A收敛")
    lines.append("   - 同一天区、同一滤镜的相邻帧(042902 vs 025221), 一个成功一个失败")
    lines.append("")

    # Part B结果
    lines.append("---")
    lines.append("")
    lines.append("## Part B: 真实数据实验 (NGC55 Oiii)")
    lines.append("")
    lines.append("| 帧 | U | W | s0 | nr中位 | θ SNR | nr>10 | nr>20 | 耗时 |")
    lines.append("|-----|---|----|-----|--------|-------|-------|-------|------|")

    for label, stats in [('NGC55_Oiii_失败(042902)', ngc55_fail_stats),
                          ('NGC55_Oiii_成功(025221)', ngc55_success_stats)]:
        if stats is None:
            continue
        lines.append(f"| {label} | {stats['n_U']} | {stats['n_W']} | {stats['s0']:.4f} | "
                     f"{stats['nr_median']:.1f} | {stats.get('theta_snr', 0):.1f} | "
                     f"{stats['nr_gt_10']} | {stats['nr_gt_20']} | {stats['time_s']:.1f}s |")

    lines.append("")
    if ngc55_fail_stats and ngc55_success_stats:
        lines.append("### 失败vs成功帧对比")
        lines.append("")
        sf = ngc55_fail_stats
        ss = ngc55_success_stats
        lines.append(f"- nr中位: 失败{sf['nr_median']:.1f} vs 成功{ss['nr_median']:.1f} "
                     f"(差距{ss['nr_median']-sf['nr_median']:.1f})")
        lines.append(f"- θ SNR: 失败{sf.get('theta_snr',0):.1f} vs 成功{ss.get('theta_snr',0):.1f} "
                     f"(比值{ss.get('theta_snr',0)/max(sf.get('theta_snr',1),1e-10):.1f}x)")
        lines.append(f"- nr>10: 失败{sf['nr_gt_10']}对 vs 成功{ss['nr_gt_10']}对")
        nr_diff = ss['nr_median'] - sf['nr_median']
        if nr_diff > 2:
            lines.append(f"- **结论**: 成功帧的nr中位显著高于失败帧, θ SNR更强, 说明成功帧的重叠率更高或图像质量更好")
        else:
            lines.append(f"- **结论**: 两帧nr相近, 失败主要原因可能是Phase C/D′阶段而非Phase A抽样阶段")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 总体结论")
    lines.append("")
    lines.append("1. **真匹配对推导向量在4D空间中形成紧密聚类**, 假匹配对散乱分布")
    lines.append("2. **θ SNR是判定Phase A成功的可靠指标**: SNR>10时可可靠检测正确旋转角")
    lines.append("3. **重叠率是关键因素**: 重叠率从10%→50%, θ SNR可从不足提升到非常可靠")
    lines.append("4. **NGC55失败根因**: 窄带滤镜星点少 → 重叠率低 → 真匹配对少 → θ SNR不足 → Phase C/D′填充失败")
    lines.append("5. **优化方向**: 改进密度匹配使n_gaia趋近n_target(±10%), 或增大n_img_total提高重叠率")

    report_path = os.path.join(out_dir, 'transform_space_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\n分析报告: {report_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 80)
    print("V4.1 变换矩阵空间理论+实验分析")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 80)

    overlap_ratios = [0.10, 0.20, 0.50]
    n_samples = 5000

    # ========================================================================
    # Part A: 合成数据实验
    # ========================================================================
    print("\n" + "=" * 60)
    print("Part A: 合成数据实验 (模拟NGC55)")
    print(f"  参数: N=250图像星, M=265Gaia星, s0=0.96\"/px, FOV=1.5°")
    print(f"  真变换: s=1.005, θ=-90.5°, tx=ty=0")
    print(f"  采样: {n_samples}对/配置")
    print("=" * 60)

    all_results_list = []
    all_theta_stats = []
    true_T = None
    s0_val = None

    for ri, overlap in enumerate(overlap_ratios):
        print(f"\n--- 重叠率={overlap:.0%} ---")

        U, W, is_match, match_gaia_idx, true_T_val, s0_val, _ = generate_ngc55_synthetic(overlap)
        if true_T is None:
            true_T = true_T_val

        n_true_match = int(np.sum(is_match))
        print(f"U={len(U)}颗, W={len(W)}颗, 真匹配={n_true_match}颗 ({overlap*250:.0f}颗)")
        print(f"真变换: s={true_T_val[0]:.4f}, θ={true_T_val[1]:.1f}°, "
              f"tx={true_T_val[2]:.1f}\", ty={true_T_val[3]:.1f}\"")

        results = run_sampling(U, W, is_match, match_gaia_idx, s0_val, true_T_val, n_samples=n_samples)

        n_true_pairs = sum(1 for r in results if r['is_true_pair'])
        n_false_pairs = sum(1 for r in results if not r['is_true_pair'])
        print(f"有效采样: {len(results)}对 (真匹配对{n_true_pairs}, 假匹配对{n_false_pairs})")

        tr_nr = [r['nr'] for r in results if r['is_true_pair']]
        fr_nr = [r['nr'] for r in results if not r['is_true_pair']]
        if tr_nr:
            print(f"真匹配nr: 中位={np.median(tr_nr):.1f} 均值={np.mean(tr_nr):.1f} "
                  f"P95={np.percentile(tr_nr,95):.0f} max={max(tr_nr)}")
        if fr_nr:
            print(f"假匹配nr: 中位={np.median(fr_nr):.1f} 均值={np.mean(fr_nr):.1f} "
                  f"P95={np.percentile(fr_nr,95):.0f} max={max(fr_nr)}")

        theta_stats = plot_results(results, overlap, s0_val, true_T_val, OUT_DIR)
        all_results_list.append(results)
        all_theta_stats.append(theta_stats)

    # 汇总图
    print(f"\n--- 生成汇总图 ---")
    plot_summary_figure(all_results_list, all_theta_stats, overlap_ratios, s0_val, OUT_DIR)
    plot_nr_distribution_detail(all_results_list, overlap_ratios, OUT_DIR)

    # ========================================================================
    # Part B: 真实数据实验 (NGC55 Oiii帧)
    # ========================================================================
    print("\n" + "=" * 60)
    print("Part B: 真实数据实验 (NGC55 Oiii)")
    print("=" * 60)

    ngc55_files = {
        'NGC55_Oiii_fail': r'f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@042902-1200S-Oiii.fts',
        'NGC55_Oiii_success': r'f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@025221-1200S-Oiii.fts',
    }

    ngc55_fail_stats = None
    ngc55_success_stats = None
    ngc55_fail_results = None
    ngc55_success_results = None

    for label, fits_path in ngc55_files.items():
        if not os.path.exists(fits_path):
            print(f"\n  警告: 文件不存在 {fits_path}, 跳过")
            continue

        results, stats = run_real_experiment(fits_path, label, OUT_DIR, K=10000)
        if label == 'NGC55_Oiii_fail':
            ngc55_fail_results = results
            ngc55_fail_stats = stats
        else:
            ngc55_success_results = results
            ngc55_success_stats = stats

    # 对比图
    if ngc55_fail_results and ngc55_success_results:
        print(f"\n--- 生成对比图 ---")
        plot_real_comparison(
            ngc55_fail_results, ngc55_fail_stats, '失败帧(042902)',
            ngc55_success_results, ngc55_success_stats, '成功帧(025221)',
            'NGC55_Oiii_失败vs成功',
            OUT_DIR
        )

    # ========================================================================
    # 分析报告
    # ========================================================================
    print(f"\n{'='*60}")
    print("生成分析报告...")
    write_report(all_results_list, all_theta_stats, overlap_ratios, true_T, s0_val,
                 ngc55_fail_stats, ngc55_success_stats, OUT_DIR)

    print("\n" + "=" * 80)
    print("实验完成!")
    print(f"所有输出: {OUT_DIR}")
    print(f"  - 合成数据图: transform_space_overlap_*.png")
    print(f"  - 汇总图: transform_space_summary.png")
    print(f"  - nr详细图: transform_space_nr_detail.png")
    print(f"  - 真实数据图: NGC55_Oiii_*_matrix_space.png")
    print(f"  - 对比图: comparison_*.png")
    print(f"  - 分析报告: transform_space_report.md")
    print("=" * 80)


if __name__ == "__main__":
    main()

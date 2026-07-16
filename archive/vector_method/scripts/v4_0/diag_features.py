"""
V4.0 向量特征探索脚本
功能: 对每个候选配对计算多种向量组特征, 找出有区分度的特征
用途: 替代SCM, 找到能区分正确/错误匹配的特征
"""
import sys, os, math, numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import GaiaClientPy, gnomonic_forward, _apply_flip
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# ── 读图+检测 ──
FITS = os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
MODE = 1
reader = ImageReader()
img = reader.read(FITS)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2
s0 = 206.265 * ps / fl
cx, cy = w / 2.0, h / 2.0
halfW = w / 2.0 * s0
halfH = h / 2.0 * s0

detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)

sat_mask = np.array(det.saturated, dtype=bool)
sat_x = np.array(det.x, np.float64)[sat_mask]
sat_y = np.array(det.y, np.float64)[sat_mask]
nsat = len(sat_x)
U_all = np.column_stack([(sat_x - cx) * s0, -(sat_y - cy) * s0])

gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
gaia = GaiaClientPy(gaia_dir, 1)
fov_diag = math.sqrt(w*w + h*h) * s0
qr = fov_diag * 0.5 / 3600.0
m_cut = 6.0 + 1.5 * math.log10(fl) + 2.0 * math.log10(300.0)
ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, qr, min(m_cut, 22.0))
gaia.close()

xi_all, eta_all, valid = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
in_fov = valid & (np.abs(xi_all) < halfW * 2) & (np.abs(eta_all) < halfH * 2)
fov_idx = np.where(in_fov)[0]
sorted_mag = np.argsort(mag_t[fov_idx])
W_all = np.column_stack([xi_all[fov_idx[sorted_mag]], eta_all[fov_idx[sorted_mag]]])
Wf_all = _apply_flip(W_all, MODE)

N_init = min(nsat, 50)
M_init = min(100, len(Wf_all))
U = U_all[:N_init]
Wf = Wf_all[:M_init]

print(f"U={N_init} W={M_init}")

# ── 预计算: U的KDTree ──
U_tree = cKDTree(U)

# ── 特征计算函数 ──

def feat_nn_overlap(s, theta, tx, ty):
    """特征1: 最近邻重叠率 — Wt中有多少在U的近邻距离内"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    # 查询每个Wt到最近U的距离
    dists, _ = U_tree.query(Wt)
    # 阈值: U的3-NN中位数距离
    nn3_dists, _ = U_tree.query(U, k=4)
    med_nn = np.median(nn3_dists[:, 1:])
    # 重叠率
    n_overlap = np.sum(dists < med_nn)
    ratio = n_overlap / max(len(U), 1)
    # 平均最近邻距离
    mean_dist = np.mean(dists)
    # 中位最近邻距离
    med_dist = np.median(dists)
    return ratio, mean_dist, med_dist, n_overlap


def feat_bidirectional_match(s, theta, tx, ty, threshold_asec=None):
    """特征2: 双向匹配数 — U→Wt和Wt→U最近邻一致性"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    Wt_tree = cKDTree(Wt)

    if threshold_asec is None:
        nn3_dists, _ = U_tree.query(U, k=4)
        threshold_asec = np.median(nn3_dists[:, 1:]) * 1.5

    # 正向: U → Wt
    dists_fwd, nns_fwd = Wt_tree.query(U)
    # 反向: Wt → U
    dists_bwd, nns_bwd = U_tree.query(Wt)

    # 双向一致
    n_bidir = 0
    for i in range(len(U)):
        j = nns_fwd[i]
        if dists_fwd[i] < threshold_asec and nns_bwd[j] == i and dists_bwd[j] < threshold_asec:
            n_bidir += 1
    return n_bidir, threshold_asec


def feat_angle_hist_corr(s, theta, tx, ty, n_bins=18):
    """特征3: 角度直方图相关性 — U和Wt的向量角度分布是否一致"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    # 角度直方图 (0~2π)
    u_angles = np.arctan2(U[:, 1], U[:, 0])
    w_angles = np.arctan2(Wt[:, 1], Wt[:, 0])
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    h_u, _ = np.histogram(u_angles, bins=bins)
    h_w, _ = np.histogram(w_angles, bins=bins)
    # 归一化
    h_u = h_u / max(h_u.sum(), 1)
    h_w = h_w / max(h_w.sum(), 1)
    # 相关系数
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr


def feat_radial_hist_corr(s, theta, tx, ty, n_bins=10):
    """特征4: 径向直方图相关性 — U和Wt的向量长度分布是否一致"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    u_radii = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    w_radii = np.sqrt(Wt[:, 0]**2 + Wt[:, 1]**2)
    max_r = max(u_radii.max(), w_radii.max())
    bins = np.linspace(0, max_r, n_bins + 1)
    h_u, _ = np.histogram(u_radii, bins=bins)
    h_w, _ = np.histogram(w_radii, bins=bins)
    h_u = h_u / max(h_u.sum(), 1)
    h_w = h_w / max(h_w.sum(), 1)
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr


def feat_pairwise_dist_corr(s, theta, tx, ty, n_sample=30):
    """特征5: 两两距离矩阵相关性 — U内部距离 vs Wt内部距离"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    # 取前n_sample颗星
    n_u = min(n_sample, len(U))
    n_w = min(n_sample, len(Wt))
    n = min(n_u, n_w)

    # U的两两距离
    from scipy.spatial.distance import pdist
    d_u = pdist(U[:n])
    d_w = pdist(Wt[:n])
    corr = np.corrcoef(d_u, d_w)[0, 1]
    return corr


def feat_nn_distance_ratio_var(s, theta, tx, ty):
    """特征6: 最近邻距离比的方差 — 正确变换时每个U到最近Wt的距离应该一致"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    Wt_tree = cKDTree(Wt)
    dists, _ = Wt_tree.query(U)
    # 变异系数
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    cv = std_d / max(mean_d, 1e-10)
    return cv, mean_d, std_d


def feat_coverage(s, theta, tx, ty, grid_n=8):
    """特征7: 空间覆盖度 — U和Wt在网格中的分布一致性"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt = np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])
    # 网格
    bins_x = np.linspace(-halfW, halfW, grid_n + 1)
    bins_y = np.linspace(-halfH, halfH, grid_n + 1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bins_x, bins_y])
    h_w, _, _ = np.histogram2d(Wt[:, 0], Wt[:, 1], bins=[bins_x, bins_y])
    h_u = (h_u > 0).astype(float)
    h_w = (h_w > 0).astype(float)
    # IoU
    intersection = np.sum(h_u * h_w)
    union = np.sum(np.maximum(h_u, h_w))
    iou = intersection / max(union, 1)
    return iou


# ── 遍历所有配对, 计算特征 ──
print("\n计算所有配对的特征...")
results = []
for ui in range(N_init):
    for wi in range(M_init):
        u_vec = U[ui]
        w_vec = Wf[wi]
        u_norm = math.hypot(u_vec[0], u_vec[1])
        w_norm = math.hypot(w_vec[0], w_vec[1])
        if w_norm < 1e-10:
            continue
        s = u_norm / w_norm
        if s < 0.85 or s > 1.15:
            continue
        theta = math.atan2(u_vec[1], u_vec[0]) - math.atan2(w_vec[1], w_vec[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx = u_vec[0] - s * (cos_t * w_vec[0] - sin_t * w_vec[1])
        ty = u_vec[1] - s * (sin_t * w_vec[0] + cos_t * w_vec[1])
        if abs(tx) > 0.6 * halfW * 2 or abs(ty) > 0.6 * halfH * 2:
            continue

        # 计算所有特征
        nn_ratio, nn_mean, nn_med, nn_count = feat_nn_overlap(s, theta, tx, ty)
        bidir, bidir_thr = feat_bidirectional_match(s, theta, tx, ty)
        angle_corr = feat_angle_hist_corr(s, theta, tx, ty)
        radial_corr = feat_radial_hist_corr(s, theta, tx, ty)
        pw_corr = feat_pairwise_dist_corr(s, theta, tx, ty)
        cv, mean_d, std_d = feat_nn_distance_ratio_var(s, theta, tx, ty)
        coverage_iou = feat_coverage(s, theta, tx, ty)

        results.append({
            'ui': ui, 'wi': wi,
            's': s, 'theta': theta,
            'nn_ratio': nn_ratio, 'nn_mean': nn_mean, 'nn_med': nn_med, 'nn_count': nn_count,
            'bidir': bidir,
            'angle_corr': angle_corr,
            'radial_corr': radial_corr,
            'pw_corr': pw_corr,
            'cv': cv, 'mean_d': mean_d, 'std_d': std_d,
            'coverage_iou': coverage_iou,
        })

print(f"有效配对: {len(results)}")

# ── 已知正确θ范围: -89° ~ -91° (或等价 269° ~ 271°) ──
for r in results:
    th = math.degrees(r['theta'])
    # 归一化到-180~180
    while th > 180: th -= 360
    while th < -180: th += 360
    r['theta_deg'] = th
    r['is_correct'] = abs(th - (-89.0)) < 2.0  # θ在-89°±2°内视为正确

n_correct = sum(1 for r in results if r['is_correct'])
n_wrong = len(results) - n_correct
print(f"正确配对(θ≈-89°±2°): {n_correct}, 错误配对: {n_wrong}")

# ── 每个特征的区分度分析 ──
features = ['nn_ratio', 'nn_mean', 'nn_med', 'nn_count',
            'bidir', 'angle_corr', 'radial_corr', 'pw_corr',
            'cv', 'mean_d', 'std_d', 'coverage_iou']

print(f"\n{'='*90}")
print(f"  特征区分度分析 (正确={n_correct}, 错误={n_wrong})")
print(f"{'='*90}")
print(f"  {'特征':<18} {'正确-中位':>10} {'错误-中位':>10} {'正确-均值':>10} {'错误-均值':>10} {'区分度':>8}")
print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

for feat in features:
    correct_vals = [r[feat] for r in results if r['is_correct']]
    wrong_vals = [r[feat] for r in results if not r['is_correct']]
    if not correct_vals or not wrong_vals:
        continue
    c_med = np.median(correct_vals)
    w_med = np.median(wrong_vals)
    c_mean = np.mean(correct_vals)
    w_mean = np.mean(wrong_vals)
    # 区分度: |均值差| / max(标准差, 1e-10)
    c_std = max(np.std(correct_vals), 1e-10)
    w_std = max(np.std(wrong_vals), 1e-10)
    pooled_std = math.sqrt((c_std**2 + w_std**2) / 2)
    discrimination = abs(c_mean - w_mean) / pooled_std
    print(f"  {feat:<18} {c_med:>10.4f} {w_med:>10.4f} {c_mean:>10.4f} {w_mean:>10.4f} {discrimination:>8.2f}")

# ── Top 10 by each feature ──
print(f"\n{'='*90}")
print(f"  Top 10 by bidir (双向匹配数)")
print(f"{'='*90}")
sorted_by_bidir = sorted(results, key=lambda r: -r['bidir'])
for rank, r in enumerate(sorted_by_bidir[:10]):
    tag = "✓" if r['is_correct'] else "✗"
    print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
          f"bidir={r['bidir']:2d} nn_ratio={r['nn_ratio']:.3f} "
          f"angle_corr={r['angle_corr']:.3f} pw_corr={r['pw_corr']:.3f} "
          f"cv={r['cv']:.3f} cov_iou={r['coverage_iou']:.3f} "
          f"θ={r['theta_deg']:+.1f}°")

print(f"\n{'='*90}")
print(f"  Top 10 by nn_ratio (最近邻重叠率)")
print(f"{'='*90}")
sorted_by_nn = sorted(results, key=lambda r: -r['nn_ratio'])
for rank, r in enumerate(sorted_by_nn[:10]):
    tag = "✓" if r['is_correct'] else "✗"
    print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
          f"nn_ratio={r['nn_ratio']:.3f} bidir={r['bidir']:2d} "
          f"angle_corr={r['angle_corr']:.3f} pw_corr={r['pw_corr']:.3f} "
          f"cv={r['cv']:.3f} cov_iou={r['coverage_iou']:.3f} "
          f"θ={r['theta_deg']:+.1f}°")

print(f"\n{'='*90}")
print(f"  Top 10 by pw_corr (两两距离相关)")
print(f"{'='*90}")
sorted_by_pw = sorted(results, key=lambda r: -r['pw_corr'])
for rank, r in enumerate(sorted_by_pw[:10]):
    tag = "✓" if r['is_correct'] else "✗"
    print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
          f"pw_corr={r['pw_corr']:.3f} bidir={r['bidir']:2d} "
          f"nn_ratio={r['nn_ratio']:.3f} angle_corr={r['angle_corr']:.3f} "
          f"cv={r['cv']:.3f} cov_iou={r['coverage_iou']:.3f} "
          f"θ={r['theta_deg']:+.1f}°")

print(f"\n{'='*90}")
print(f"  Top 10 by angle_corr (角度直方图相关)")
print(f"{'='*90}")
sorted_by_angle = sorted(results, key=lambda r: -r['angle_corr'])
for rank, r in enumerate(sorted_by_angle[:10]):
    tag = "✓" if r['is_correct'] else "✗"
    print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
          f"angle_corr={r['angle_corr']:.3f} bidir={r['bidir']:2d} "
          f"nn_ratio={r['nn_ratio']:.3f} pw_corr={r['pw_corr']:.3f} "
          f"cv={r['cv']:.3f} cov_iou={r['coverage_iou']:.3f} "
          f"θ={r['theta_deg']:+.1f}°")

print(f"\n{'='*90}")
print(f"  Top 10 by coverage_iou (空间覆盖IoU)")
print(f"{'='*90}")
sorted_by_cov = sorted(results, key=lambda r: -r['coverage_iou'])
for rank, r in enumerate(sorted_by_cov[:10]):
    tag = "✓" if r['is_correct'] else "✗"
    print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
          f"cov_iou={r['coverage_iou']:.3f} bidir={r['bidir']:2d} "
          f"nn_ratio={r['nn_ratio']:.3f} angle_corr={r['angle_corr']:.3f} "
          f"pw_corr={r['pw_corr']:.3f} cv={r['cv']:.3f} "
          f"θ={r['theta_deg']:+.1f}°")

# ── 组合特征: 多特征投票 ──
print(f"\n{'='*90}")
print(f"  组合评分: bidir*3 + nn_ratio*10 + pw_corr*5 + angle_corr*3 + coverage_iou*5 - cv*3")
print(f"{'='*90}")
for r in results:
    r['combo'] = (r['bidir'] * 3 +
                  r['nn_ratio'] * 10 +
                  r['pw_corr'] * 5 +
                  r['angle_corr'] * 3 +
                  r['coverage_iou'] * 5 -
                  r['cv'] * 3)

sorted_by_combo = sorted(results, key=lambda r: -r['combo'])
n_correct_in_top20 = sum(1 for r in sorted_by_combo[:20] if r['is_correct'])
n_correct_in_top10 = sum(1 for r in sorted_by_combo[:10] if r['is_correct'])
print(f"  Top10中正确: {n_correct_in_top10}/10, Top20中正确: {n_correct_in_top20}/20")
for rank, r in enumerate(sorted_by_combo[:20]):
    tag = "✓" if r['is_correct'] else "✗"
    print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
          f"combo={r['combo']:.2f} bidir={r['bidir']:2d} nn_ratio={r['nn_ratio']:.3f} "
          f"pw_corr={r['pw_corr']:.3f} angle_corr={r['angle_corr']:.3f} "
          f"cov_iou={r['coverage_iou']:.3f} cv={r['cv']:.3f} "
          f"θ={r['theta_deg']:+.1f}°")

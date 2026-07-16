"""
V4.0 向量组一致性特征探索 — 扩展版
功能: 计算20+种向量组一致性特征, 找出区分度最高的
核心问题: 给定U(图像侧点集)和Wt(变换后星表侧点集), 如何衡量一致性?
"""
import sys, os, math, numpy as np
from scipy.spatial import cKDTree, distance
from scipy.stats import wasserstein_distance

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import GaiaClientPy, gnomonic_forward, _apply_flip
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# ── 数据准备 (与diag_features.py相同) ──
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
U_tree = cKDTree(U)

# 预计算: U的统计量
nn3_dists_U, _ = U_tree.query(U, k=4)
U_med_nn = np.median(nn3_dists_U[:, 1:])
U_radii = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
U_angles = np.arctan2(U[:, 1], U[:, 0])

print(f"U={N_init} W={M_init}, U_med_nn={U_med_nn:.2f}\"")

# ── 所有特征函数 ──

def transform_Wf(s, theta, tx, ty):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]) + tx,
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]) + ty
    ])

def feat_chamfer(Wt):
    """F1: Chamfer Distance — 双向平均最近邻距离 (越小越一致)"""
    Wt_tree = cKDTree(Wt)
    d_u2w, _ = Wt_tree.query(U)
    d_w2u, _ = U_tree.query(Wt)
    cd = np.mean(d_u2w) + np.mean(d_w2u)
    cd_sym = 0.5 * (np.mean(d_u2w) + np.mean(d_w2u))
    return cd, cd_sym, np.mean(d_u2w), np.mean(d_w2u)

def feat_partial_hausdorff(Wt, k_ratio=0.7):
    """F2: Partial Hausdorff Distance — 第k%小的最近邻距离"""
    Wt_tree = cKDTree(Wt)
    d_u2w, _ = Wt_tree.query(U)
    d_w2u, _ = U_tree.query(Wt)
    k_u = max(int(len(d_u2w) * k_ratio), 1)
    k_w = max(int(len(d_w2u) * k_ratio), 1)
    phd_u2w = np.sort(d_u2w)[k_u]
    phd_w2u = np.sort(d_w2u)[k_w]
    phd = max(phd_u2w, phd_w2u)
    return phd, phd_u2w, phd_w2u

def feat_fscore(Wt, threshold=None):
    """F3: F-score — 精确率和召回率的调和平均"""
    if threshold is None:
        threshold = U_med_nn
    Wt_tree = cKDTree(Wt)
    d_u2w, _ = Wt_tree.query(U)
    d_w2u, _ = U_tree.query(Wt)
    precision = np.mean(d_w2u < threshold)  # Wt中有多少在U附近
    recall = np.mean(d_u2w < threshold)     # U中有多少在Wt附近
    if precision + recall < 1e-10:
        return 0.0, precision, recall
    fscore = 2 * precision * recall / (precision + recall)
    return fscore, precision, recall

def feat_bidirectional_match(Wt, threshold=None):
    """F4: 双向匹配数 — U→Wt和Wt→U最近邻一致性"""
    if threshold is None:
        threshold = U_med_nn * 1.5
    Wt_tree = cKDTree(Wt)
    dists_fwd, nns_fwd = Wt_tree.query(U)
    dists_bwd, nns_bwd = U_tree.query(Wt)
    n_bidir = 0
    for i in range(len(U)):
        j = nns_fwd[i]
        if dists_fwd[i] < threshold and nns_bwd[j] == i and dists_bwd[j] < threshold:
            n_bidir += 1
    return n_bidir

def feat_mutual_nn_ratio(Wt, threshold=None):
    """F5: 互最近邻比例 — 双向最近邻一致的对数 / min(|U|, |Wt|)"""
    if threshold is None:
        threshold = U_med_nn * 2.0
    Wt_tree = cKDTree(Wt)
    dists_fwd, nns_fwd = Wt_tree.query(U)
    dists_bwd, nns_bwd = U_tree.query(Wt)
    n_mutual = 0
    for i in range(len(U)):
        j = nns_fwd[i]
        if dists_fwd[i] < threshold and nns_bwd[j] == i and dists_bwd[j] < threshold:
            n_mutual += 1
    for j in range(len(Wt)):
        i = nns_bwd[j]
        if dists_bwd[j] < threshold and nns_fwd[i] == j and dists_fwd[i] < threshold:
            pass  # 已在上面计数
    return n_mutual / max(min(len(U), len(Wt)), 1)

def feat_angle_hist_corr(Wt, n_bins=18):
    """F6: 角度直方图相关 — 向量角度分布一致性"""
    w_angles = np.arctan2(Wt[:, 1], Wt[:, 0])
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    h_u, _ = np.histogram(U_angles, bins=bins)
    h_w, _ = np.histogram(w_angles, bins=bins)
    h_u = h_u / max(h_u.sum(), 1)
    h_w = h_w / max(h_w.sum(), 1)
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def feat_radial_hist_corr(Wt, n_bins=10):
    """F7: 径向直方图相关 — 向量长度分布一致性"""
    w_radii = np.sqrt(Wt[:, 0]**2 + Wt[:, 1]**2)
    max_r = max(U_radii.max(), w_radii.max())
    bins = np.linspace(0, max_r, n_bins + 1)
    h_u, _ = np.histogram(U_radii, bins=bins)
    h_w, _ = np.histogram(w_radii, bins=bins)
    h_u = h_u / max(h_u.sum(), 1)
    h_w = h_w / max(h_w.sum(), 1)
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def feat_annular_hist_corr(Wt, n_rings=5, n_sectors=8):
    """F8: 环形扇区直方图相关 — 极坐标网格分布一致性"""
    def _polar_hist(pts, max_r, n_rings, n_sectors):
        radii = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
        angles = np.arctan2(pts[:, 1], pts[:, 0])
        r_bins = np.linspace(0, max_r, n_rings + 1)
        a_bins = np.linspace(-np.pi, np.pi, n_sectors + 1)
        h = np.zeros(n_rings * n_sectors)
        for i in range(n_rings):
            for j in range(n_sectors):
                mask = (radii >= r_bins[i]) & (radii < r_bins[i+1]) & \
                       (angles >= a_bins[j]) & (angles < a_bins[j+1])
                h[i * n_sectors + j] = np.sum(mask)
        return h / max(h.sum(), 1)

    w_radii = np.sqrt(Wt[:, 0]**2 + Wt[:, 1]**2)
    max_r = max(U_radii.max(), w_radii.max())
    h_u = _polar_hist(U, max_r, n_rings, n_sectors)
    h_w = _polar_hist(Wt, max_r, n_rings, n_sectors)
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def feat_coverage_iou(Wt, grid_n=8):
    """F9: 空间覆盖IoU — 网格占用交并比"""
    bins_x = np.linspace(-halfW, halfW, grid_n + 1)
    bins_y = np.linspace(-halfH, halfH, grid_n + 1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bins_x, bins_y])
    h_w, _, _ = np.histogram2d(Wt[:, 0], Wt[:, 1], bins=[bins_x, bins_y])
    h_u = (h_u > 0).astype(float)
    h_w = (h_w > 0).astype(float)
    intersection = np.sum(h_u * h_w)
    union = np.sum(np.maximum(h_u, h_w))
    return intersection / max(union, 1)

def feat_nn_dist_wasserstein(Wt):
    """F10: 最近邻距离分布的Wasserstein距离 — U内部NN距离 vs Wt内部NN距离"""
    Wt_tree = cKDTree(Wt)
    nn_U = U_tree.query(U, k=2)[0][:, 1]
    nn_Wt = Wt_tree.query(Wt, k=2)[0][:, 1]
    return wasserstein_distance(nn_U, nn_Wt)

def feat_density_map_corr(Wt, grid_n=16):
    """F11: 密度图相关 — 核密度估计的网格相关性"""
    bins_x = np.linspace(-halfW, halfW, grid_n + 1)
    bins_y = np.linspace(-halfH, halfH, grid_n + 1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bins_x, bins_y])
    h_w, _, _ = np.histogram2d(Wt[:, 0], Wt[:, 1], bins=[bins_x, bins_y])
    h_u = h_u.flatten().astype(float)
    h_w = h_w.flatten().astype(float)
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def feat_sorted_dist_mat_corr(Wt, n_sample=20):
    """F12: 排序距离矩阵相关 — U内部距离矩阵排序后与Wt内部距离矩阵排序后的相关"""
    n = min(n_sample, len(U), len(Wt))
    from scipy.spatial.distance import pdist
    d_u = pdist(U[:n])
    d_w = pdist(Wt[:n])
    # 排序后比较
    d_u_sorted = np.sort(d_u)
    d_w_sorted = np.sort(d_w)
    corr = np.corrcoef(d_u_sorted, d_w_sorted)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def feat_nn_dist_ratio_corr(Wt, n_sample=30):
    """F13: 最近邻距离比相关 — 每个点的NN距离/中位NN距离, 比较分布"""
    nn_U = U_tree.query(U, k=2)[0][:, 1]
    Wt_tree = cKDTree(Wt)
    nn_Wt = Wt_tree.query(Wt, k=2)[0][:, 1]
    # 归一化
    ratio_U = nn_U / max(np.median(nn_U), 1e-10)
    ratio_Wt = nn_Wt / max(np.median(nn_Wt), 1e-10)
    # 直方图相关
    bins = np.linspace(0, 3, 15)
    h_u, _ = np.histogram(ratio_U, bins=bins)
    h_w, _ = np.histogram(ratio_Wt, bins=bins)
    h_u = h_u / max(h_u.sum(), 1)
    h_w = h_w / max(h_w.sum(), 1)
    corr = np.corrcoef(h_u, h_w)[0, 1]
    return corr if not np.isnan(corr) else 0.0

def feat_centroid_offset(Wt):
    """F14: 质心偏移 — U和Wt质心的距离 (正确变换应≈0)"""
    c_u = np.mean(U, axis=0)
    c_w = np.mean(Wt, axis=0)
    return np.linalg.norm(c_u - c_w)

def feat_inertia_ratio(Wt):
    """F15: 惯量比 — U和Wt的主惯量方向和大小比"""
    def _inertia(pts):
        c = np.mean(pts, axis=0)
        pts_c = pts - c
        cov = pts_c.T @ pts_c / len(pts)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        return eigvals
    eig_U = _inertia(U)
    eig_Wt = _inertia(Wt)
    # 特征值比
    ratio1 = eig_U[0] / max(eig_Wt[0], 1e-10)
    ratio2 = eig_U[1] / max(eig_Wt[1], 1e-10)
    # 椭圆率比
    ecc_U = eig_U[1] / max(eig_U[0], 1e-10)
    ecc_Wt = eig_Wt[1] / max(eig_Wt[0], 1e-10)
    ecc_diff = abs(ecc_U - ecc_Wt)
    return min(ratio1, 1.0/ratio1), min(ratio2, 1.0/ratio2), ecc_diff

def feat_nn_count_asymmetry(Wt, threshold=None):
    """F16: 最近邻计数不对称性 — U→Wt匹配数 vs Wt→U匹配数"""
    if threshold is None:
        threshold = U_med_nn * 1.5
    Wt_tree = cKDTree(Wt)
    d_u2w, _ = Wt_tree.query(U)
    d_w2u, _ = U_tree.query(Wt)
    n_u2w = np.sum(d_u2w < threshold)
    n_w2u = np.sum(d_w2u < threshold)
    asym = abs(n_u2w - n_w2u) / max(n_u2w + n_w2u, 1)
    return asym, n_u2w, n_w2u

def feat_convex_hull_overlap(Wt):
    """F17: 凸包重叠率 — U和Wt凸包的IoU"""
    from scipy.spatial import ConvexHull
    try:
        hull_u = ConvexHull(U)
        hull_w = ConvexHull(Wt)
        # 面积比 (近似)
        area_u = hull_u.volume
        area_w = hull_w.volume
        # 重叠: 用点在对方凸包内的比例近似
        from scipy.spatial import Delaunay
        tri_u = Delaunay(U[hull_u.vertices])
        tri_w = Delaunay(Wt[hull_w.vertices])
        n_u_in_w = np.sum(tri_w.find_simplex(U) >= 0)
        n_w_in_u = np.sum(tri_u.find_simplex(Wt) >= 0)
        overlap = (n_u_in_w / len(U) + n_w_in_u / len(Wt)) / 2
        return overlap
    except:
        return 0.0

def feat_rank_corr_nn(Wt):
    """F18: 最近邻距离排序相关 — U中每个点的NN距离排序 vs Wt中对应点的NN距离排序"""
    Wt_tree = cKDTree(Wt)
    d_u2w, nns = Wt_tree.query(U)
    # U中每个点的NN距离
    nn_U = U_tree.query(U, k=2)[0][:, 1]
    # Wt中对应点的NN距离
    nn_Wt_at_u = Wt_tree.query(Wt[nns], k=2)[0][:, 1]
    # Spearman相关
    from scipy.stats import spearmanr
    corr, _ = spearmanr(nn_U, nn_Wt_at_u)
    return corr if not np.isnan(corr) else 0.0

def feat_kde_overlap(Wt, bandwidth=None):
    """F19: 核密度估计重叠 — U和Wt的KDE在网格点上的重叠积分"""
    if bandwidth is None:
        bandwidth = U_med_nn
    grid_n = 16
    gx = np.linspace(-halfW, halfW, grid_n)
    gy = np.linspace(-halfH, halfH, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    def _kde(pts, grid, bw):
        tree = cKDTree(pts)
        counts = tree.query_ball_point(grid, bw)
        density = np.array([len(c) for c in counts], dtype=float)
        density /= max(density.sum(), 1)
        return density

    d_u = _kde(U, grid_pts, bandwidth)
    d_w = _kde(Wt, grid_pts, bandwidth)
    overlap = np.sum(np.minimum(d_u, d_w))
    return overlap

def feat_local_structure_corr(Wt, k=5):
    """F20: 局部结构相关 — 每个点的k-NN距离向量排序后的相关性"""
    nn_dists_U, _ = U_tree.query(U, k=min(k+1, len(U)))
    nn_dists_U = nn_dists_U[:, 1:]  # 去掉自身
    Wt_tree = cKDTree(Wt)
    nn_dists_Wt, _ = Wt_tree.query(Wt, k=min(k+1, len(Wt)))
    nn_dists_Wt = nn_dists_Wt[:, 1:]

    # 对每个U点, 找Wt中最近的点, 比较它们的k-NN距离向量
    dists_fwd, nns_fwd = Wt_tree.query(U)
    corrs = []
    for i in range(len(U)):
        j = nns_fwd[i]
        if dists_fwd[i] > U_med_nn * 2:
            continue
        # 排序后相关
        v_u = np.sort(nn_dists_U[i])
        v_w = np.sort(nn_dists_Wt[j])
        min_len = min(len(v_u), len(v_w))
        if min_len < 2:
            continue
        c = np.corrcoef(v_u[:min_len], v_w[:min_len])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    return np.mean(corrs) if corrs else 0.0


# ── 遍历配对, 计算所有特征 ──
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

        Wt = transform_Wf(s, theta, tx, ty)

        # 计算所有特征
        r = {'ui': ui, 'wi': wi, 's': s, 'theta': theta}

        cd, cd_sym, d_u2w, d_w2u = feat_chamfer(Wt)
        r['chamfer'] = cd_sym
        r['d_u2w_mean'] = d_u2w
        r['d_w2u_mean'] = d_w2u

        phd, phd_u, phd_w = feat_partial_hausdorff(Wt)
        r['phd_70'] = phd

        fscore, prec, rec = feat_fscore(Wt)
        r['fscore'] = fscore
        r['precision'] = prec
        r['recall'] = rec

        r['bidir'] = feat_bidirectional_match(Wt)
        r['mutual_nn'] = feat_mutual_nn_ratio(Wt)
        r['angle_corr'] = feat_angle_hist_corr(Wt)
        r['radial_corr'] = feat_radial_hist_corr(Wt)
        r['annular_corr'] = feat_annular_hist_corr(Wt)
        r['coverage_iou'] = feat_coverage_iou(Wt)
        r['nn_wasserstein'] = feat_nn_dist_wasserstein(Wt)
        r['density_corr'] = feat_density_map_corr(Wt)
        r['sorted_dist_corr'] = feat_sorted_dist_mat_corr(Wt)
        r['nn_ratio_corr'] = feat_nn_dist_ratio_corr(Wt)
        r['centroid_offset'] = feat_centroid_offset(Wt)

        r1, r2, ecc_diff = feat_inertia_ratio(Wt)
        r['inertia_r1'] = r1
        r['inertia_r2'] = r2
        r['ecc_diff'] = ecc_diff

        asym, n_u2w, n_w2u = feat_nn_count_asymmetry(Wt)
        r['nn_asymmetry'] = asym
        r['n_u2w'] = n_u2w
        r['n_w2u'] = n_w2u

        r['hull_overlap'] = feat_convex_hull_overlap(Wt)
        r['rank_corr_nn'] = feat_rank_corr_nn(Wt)
        r['kde_overlap'] = feat_kde_overlap(Wt)
        r['local_struct_corr'] = feat_local_structure_corr(Wt)

        # 标记正确/错误
        th = math.degrees(theta)
        while th > 180: th -= 360
        while th < -180: th += 360
        r['theta_deg'] = th
        r['is_correct'] = abs(th - (-89.0)) < 2.0

        results.append(r)

n_correct = sum(1 for r in results if r['is_correct'])
n_wrong = len(results) - n_correct
print(f"有效配对: {len(results)}, 正确: {n_correct}, 错误: {n_wrong}")

# ── 区分度分析 ──
features = ['chamfer', 'd_u2w_mean', 'd_w2u_mean', 'phd_70',
            'fscore', 'precision', 'recall', 'bidir', 'mutual_nn',
            'angle_corr', 'radial_corr', 'annular_corr', 'coverage_iou',
            'nn_wasserstein', 'density_corr', 'sorted_dist_corr',
            'nn_ratio_corr', 'centroid_offset', 'inertia_r1', 'inertia_r2',
            'ecc_diff', 'nn_asymmetry', 'n_u2w', 'n_w2u',
            'hull_overlap', 'rank_corr_nn', 'kde_overlap', 'local_struct_corr']

# 判断方向: 特征值越大越好的为True, 越小越好的为False
higher_is_better = {
    'chamfer': False, 'd_u2w_mean': False, 'd_w2u_mean': False, 'phd_70': False,
    'fscore': True, 'precision': True, 'recall': True, 'bidir': True, 'mutual_nn': True,
    'angle_corr': True, 'radial_corr': True, 'annular_corr': True, 'coverage_iou': True,
    'nn_wasserstein': False, 'density_corr': True, 'sorted_dist_corr': True,
    'nn_ratio_corr': True, 'centroid_offset': False, 'inertia_r1': True, 'inertia_r2': True,
    'ecc_diff': False, 'nn_asymmetry': False, 'n_u2w': True, 'n_w2u': True,
    'hull_overlap': True, 'rank_corr_nn': True, 'kde_overlap': True, 'local_struct_corr': True,
}

print(f"\n{'='*100}")
print(f"  特征区分度分析 (正确={n_correct}, 错误={n_wrong})")
print(f"{'='*100}")
print(f"  {'特征':<22} {'方向':>4} {'正确-中位':>10} {'错误-中位':>10} {'区分度':>8} {'Top10正确率':>10}")
print(f"  {'-'*22} {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

discriminations = []
for feat in features:
    correct_vals = [r[feat] for r in results if r['is_correct']]
    wrong_vals = [r[feat] for r in results if not r['is_correct']]
    if not correct_vals or not wrong_vals:
        continue
    c_med = np.median(correct_vals)
    w_med = np.median(wrong_vals)
    c_mean = np.mean(correct_vals)
    w_mean = np.mean(wrong_vals)
    c_std = max(np.std(correct_vals), 1e-10)
    w_std = max(np.std(wrong_vals), 1e-10)
    pooled_std = math.sqrt((c_std**2 + w_std**2) / 2)
    disc = abs(c_mean - w_mean) / pooled_std

    # Top10正确率
    hib = higher_is_better.get(feat, True)
    sorted_r = sorted(results, key=lambda r: -r[feat] if hib else r[feat])
    top10_correct = sum(1 for r in sorted_r[:10] if r['is_correct'])

    direction = "↑" if hib else "↓"
    discriminations.append((disc, feat, direction, c_med, w_med, top10_correct))

# 按区分度排序
discriminations.sort(key=lambda x: -x[0])
for disc, feat, direction, c_med, w_med, top10_correct in discriminations:
    print(f"  {feat:<22} {direction:>4} {c_med:>10.4f} {w_med:>10.4f} {disc:>8.2f} {top10_correct:>8}/10")

# ── Top 5特征的Top 10详情 ──
top5_feats = [d[1] for d in discriminations[:5]]
for feat in top5_feats:
    hib = higher_is_better.get(feat, True)
    sorted_r = sorted(results, key=lambda r: -r[feat] if hib else r[feat])
    print(f"\n{'='*100}")
    print(f"  Top 10 by {feat} ({'higher=better' if hib else 'lower=better'})")
    print(f"{'='*100}")
    for rank, r in enumerate(sorted_r[:10]):
        tag = "OK" if r['is_correct'] else "XX"
        print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
              f"{feat}={r[feat]:.4f} bidir={r['bidir']:2d} fscore={r['fscore']:.3f} "
              f"angle_corr={r['angle_corr']:.3f} coverage_iou={r['coverage_iou']:.3f} "
              f"annular_corr={r['annular_corr']:.3f} kde_overlap={r['kde_overlap']:.3f} "
              f"local_struct={r['local_struct_corr']:.3f} "
              f"theta={r['theta_deg']:+.1f}deg")

# ── 最优组合特征搜索 ──
print(f"\n{'='*100}")
print(f"  最优2特征组合搜索 (Top 10正确率)")
print(f"{'='*100}")

# 归一化所有特征到[0,1]
feat_min = {}
feat_max = {}
for feat in features:
    vals = [r[feat] for r in results]
    feat_min[feat] = min(vals)
    feat_max[feat] = max(vals)

def _norm(val, feat):
    rng = feat_max[feat] - feat_min[feat]
    if rng < 1e-10:
        return 0.5
    v = (val - feat_min[feat]) / rng
    if not higher_is_better.get(feat, True):
        v = 1.0 - v
    return v

# 选区分度>1.0的特征
good_feats = [d[1] for d in discriminations if d[0] > 1.0]
print(f"  区分度>1.0的特征: {len(good_feats)}个")

best_combo_score = 0
best_combo = None
for i in range(len(good_feats)):
    for j in range(i+1, len(good_feats)):
        f1, f2 = good_feats[i], good_feats[j]
        # 组合分数 = norm(f1) + norm(f2)
        for r in results:
            r['_combo2'] = _norm(r[f1], f1) + _norm(r[f2], f2)
        sorted_r = sorted(results, key=lambda r: -r['_combo2'])
        top10_correct = sum(1 for r in sorted_r[:10] if r['is_correct'])
        top20_correct = sum(1 for r in sorted_r[:20] if r['is_correct'])
        combo_score = top10_correct * 10 + top20_correct
        if combo_score > best_combo_score:
            best_combo_score = combo_score
            best_combo = (f1, f2, top10_correct, top20_correct)

if best_combo:
    f1, f2, t10, t20 = best_combo
    print(f"  最优2特征: {f1} + {f2} → Top10={t10}/10, Top20={t20}/20")

# 3特征组合 (从top5中选)
top5 = [d[1] for d in discriminations[:5]]
best3_score = 0
best3 = None
for i in range(len(top5)):
    for j in range(i+1, len(top5)):
        for k in range(j+1, len(top5)):
            f1, f2, f3 = top5[i], top5[j], top5[k]
            for r in results:
                r['_combo3'] = _norm(r[f1], f1) + _norm(r[f2], f2) + _norm(r[f3], f3)
            sorted_r = sorted(results, key=lambda r: -r['_combo3'])
            top10_correct = sum(1 for r in sorted_r[:10] if r['is_correct'])
            top20_correct = sum(1 for r in sorted_r[:20] if r['is_correct'])
            combo_score = top10_correct * 10 + top20_correct
            if combo_score > best3_score:
                best3_score = combo_score
                best3 = (f1, f2, f3, top10_correct, top20_correct)

if best3:
    f1, f2, f3, t10, t20 = best3
    print(f"  最优3特征: {f1} + {f2} + {f3} → Top10={t10}/10, Top20={t20}/20")
    # 打印Top20
    for r in results:
        r['_combo3'] = _norm(r[f1], f1) + _norm(r[f2], f2) + _norm(r[f3], f3)
    sorted_r = sorted(results, key=lambda r: -r['_combo3'])
    for rank, r in enumerate(sorted_r[:20]):
        tag = "OK" if r['is_correct'] else "XX"
        print(f"  #{rank:2d} {tag} u={r['ui']:2d} w={r['wi']:3d} | "
              f"combo3={r['_combo3']:.3f} {f1}={r[f1]:.4f} {f2}={r[f2]:.4f} {f3}={r[f3]:.4f} "
              f"theta={r['theta_deg']:+.1f}deg")

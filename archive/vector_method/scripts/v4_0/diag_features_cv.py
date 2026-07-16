"""
V4.0 向量组一致性特征 — 盲抽样交叉验证 (v2)
不依赖WCS, 模拟盲求解: 随机1点抽样→变换→特征评分→自洽质量验证

核心修正:
  - 1点法: s=|u|/|w|, θ=angle(u)-angle(w), t≈0 (中心化向量无平移)
  - 质量标签: θ直方图峰值聚类 (修正环绕bug) + s收敛
  - 验证: 好变换的chamfer距离应<<坏变换

区分度: Cohen's d = |mean_good - mean_bad| / pooled_std
伪真值: Cohen_d高但Spearman_ρ弱 + Top10好率低
"""
import sys, os, math, time, numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import GaiaClientPy, gnomonic_forward, _apply_flip
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_0"))
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# ── 数据准备 ──
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
U = np.column_stack([(sat_x - cx) * s0, -(sat_y - cy) * s0])

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
W = np.column_stack([xi_all[fov_idx[sorted_mag]], eta_all[fov_idx[sorted_mag]]])
Wf = _apply_flip(W, MODE)

N = min(nsat, 50)
M = min(100, len(Wf))
U = U[:N]
Wf = Wf[:M]
U_tree = cKDTree(U)
nn3_U = U_tree.query(U, k=min(4, N))[0][:, 1:]
U_med_nn = np.median(nn3_U) if nn3_U.size > 0 else 100.0
U_angles = np.arctan2(U[:, 1], U[:, 0])

print(f"U={N} W={M}, U_med_nn={U_med_nn:.2f}\", s0={s0:.4f}\"/px")

# ── 角度距离 (正确处理环绕) ──
def angle_dist(a, b):
    """两个角度的最短距离, 处理环绕"""
    d = (a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

# ── 特征函数 ──

def _transform(s, theta):
    """1点变换: Wt = s * R(θ) * Wf (无平移, 因为向量已中心化)"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]),
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1])
    ])

def f_angle_corr(Wt):
    """成本:1 — 1D角度直方图相关"""
    w_a = np.arctan2(Wt[:, 1], Wt[:, 0])
    bins = np.linspace(-np.pi, np.pi, 19)
    h_u, _ = np.histogram(U_angles, bins=bins)
    h_w, _ = np.histogram(w_a, bins=bins)
    c = np.corrcoef(h_u, h_w)[0, 1]
    return c if not np.isnan(c) else 0.0

def f_coverage_iou(Wt, gn=8):
    """成本:1 — 2D覆盖IoU"""
    bx = np.linspace(-halfW, halfW, gn+1)
    by = np.linspace(-halfH, halfH, gn+1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bx, by])
    h_w, _, _ = np.histogram2d(Wt[:, 0], Wt[:, 1], bins=[bx, by])
    h_u = (h_u > 0).astype(float)
    h_w = (h_w > 0).astype(float)
    inter = np.sum(h_u * h_w)
    union = np.sum(np.maximum(h_u, h_w))
    return inter / max(union, 1)

def f_density_corr(Wt, gn=16):
    """成本:1 — 2D密度相关"""
    bx = np.linspace(-halfW, halfW, gn+1)
    by = np.linspace(-halfH, halfH, gn+1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bx, by])
    h_w, _, _ = np.histogram2d(Wt[:, 0], Wt[:, 1], bins=[bx, by])
    c = np.corrcoef(h_u.ravel(), h_w.ravel())[0, 1]
    return c if not np.isnan(c) else 0.0

def f_d_u2w_mean(Wt):
    """成本:2 — U→Wt平均最近邻距离"""
    Wt_tree = cKDTree(Wt)
    d, _ = Wt_tree.query(U)
    return float(np.mean(d))

def f_chamfer(Wt):
    """成本:2 — 双向平均最近邻距离"""
    Wt_tree = cKDTree(Wt)
    d1, _ = Wt_tree.query(U)
    d2, _ = U_tree.query(Wt)
    return float(0.5 * (np.mean(d1) + np.mean(d2)))

def f_bidir(Wt, thr=None):
    """成本:3 — 双向匹配对数"""
    if thr is None: thr = U_med_nn * 1.5
    Wt_tree = cKDTree(Wt)
    d_fwd, nns_fwd = Wt_tree.query(U)
    d_bwd, nns_bwd = U_tree.query(Wt)
    n = 0
    for i in range(len(U)):
        j = nns_fwd[i]
        if d_fwd[i] < thr and nns_bwd[j] == i and d_bwd[j] < thr:
            n += 1
    return n

def f_mutual_nn(Wt, thr=None):
    """成本:3 — 归一化双向匹配率"""
    if thr is None: thr = U_med_nn * 2.0
    Wt_tree = cKDTree(Wt)
    d_fwd, nns_fwd = Wt_tree.query(U)
    d_bwd, nns_bwd = U_tree.query(Wt)
    n = 0
    for i in range(len(U)):
        j = nns_fwd[i]
        if d_fwd[i] < thr and nns_bwd[j] == i and d_bwd[j] < thr:
            n += 1
    return n / max(min(len(U), len(Wt)), 1)

def f_kde_overlap(Wt, bw=None):
    """成本:4 — KDE重叠度"""
    if bw is None: bw = U_med_nn
    gn = 16
    gx = np.linspace(-halfW, halfW, gn)
    gy = np.linspace(-halfH, halfH, gn)
    GX, GY = np.meshgrid(gx, gy)
    grid = np.column_stack([GX.ravel(), GY.ravel()])
    def _kde(pts):
        t = cKDTree(pts)
        c = t.query_ball_point(grid, bw)
        d = np.array([len(x) for x in c], dtype=float)
        return d / max(d.sum(), 1)
    d_u = _kde(U)
    d_w = _kde(Wt)
    return float(np.sum(np.minimum(d_u, d_w)))

# ── 全量抽样 ──
print("\n全量1点抽样 + 特征计算...")
t0 = time.perf_counter()
results = []
for ui in range(N):
    for wi in range(M):
        u_vec = U[ui]
        w_vec = Wf[wi]
        u_norm = math.hypot(u_vec[0], u_vec[1])
        w_norm = math.hypot(w_vec[0], w_vec[1])
        if w_norm < 1e-10: continue
        s = u_norm / w_norm
        if s < 0.85 or s > 1.15: continue
        theta = math.atan2(u_vec[1], u_vec[0]) - math.atan2(w_vec[1], w_vec[0])

        # 1点法: t = u - s*R(θ)*w, 但因为s和θ从同一对推导, t≈0
        # 验证:
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx = u_vec[0] - s * (cos_t * w_vec[0] - sin_t * w_vec[1])
        ty = u_vec[1] - s * (sin_t * w_vec[0] + cos_t * w_vec[1])

        Wt = _transform(s, theta)

        r = {'ui': ui, 'wi': wi, 's': s, 'theta': theta, 'tx': tx, 'ty': ty}

        # 低成本特征
        r['angle_corr'] = f_angle_corr(Wt)
        r['coverage_iou'] = f_coverage_iou(Wt)
        r['density_corr'] = f_density_corr(Wt)
        r['d_u2w'] = f_d_u2w_mean(Wt)
        r['chamfer'] = f_chamfer(Wt)
        r['bidir'] = f_bidir(Wt)
        r['mutual_nn'] = f_mutual_nn(Wt)
        r['kde_overlap'] = f_kde_overlap(Wt)

        results.append(r)

elapsed = time.perf_counter() - t0
print(f"有效配对: {len(results)}, 耗时: {elapsed:.1f}s")

# ── 验证1点法t≈0 ──
tx_vals = [abs(r['tx']) for r in results]
ty_vals = [abs(r['ty']) for r in results]
print(f"\n1点法t验证: |tx|_max={max(tx_vals):.6f}\" |ty|_max={max(ty_vals):.6f}\" "
      f"(应≈0, 因为s和θ从同一对推导)")

# ── θ直方图聚类 (修正环绕) ──
thetas = np.array([r['theta'] for r in results])
ss = np.array([r['s'] for r in results])

# 用环绕直方图找峰值
n_bins = 72  # 5°/bin
theta_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
theta_hist, _ = np.histogram(thetas, bins=theta_bins)
peak_bin = np.argmax(theta_hist)
peak_theta = 0.5 * (theta_bins[peak_bin] + theta_bins[peak_bin + 1])
peak_count = theta_hist[peak_bin]

# 用正确环绕计算每个样本到峰值的角距离
theta_dists = angle_dist(thetas, peak_theta)

# 在峰值附近(5°内)的样本中找s峰值
near_peak = theta_dists < np.radians(5)
if np.sum(near_peak) > 0:
    peak_s = np.median(ss[near_peak])
else:
    peak_s = 1.0

print(f"\nθ直方图: peak={math.degrees(peak_theta):.1f}° count={peak_count}/{len(results)}")
print(f"峰值附近(5°内)样本: {np.sum(near_peak)}, s_median={peak_s:.4f}")

# ── 质量标签 (修正环绕) ──
# 好: θ在峰值5°内 + s在峰值3%内
# 坏: θ距峰值>30° (明确不在峰值)
# 不确定: 中间区域, 不参与分析
for i, r in enumerate(results):
    d_th = theta_dists[i]
    ds = abs(r['s'] - peak_s) / peak_s
    if d_th < np.radians(5) and ds < 0.03:
        r['label'] = 'good'
    elif d_th > np.radians(30):
        r['label'] = 'bad'
    else:
        r['label'] = 'uncertain'

n_good = sum(1 for r in results if r['label'] == 'good')
n_bad = sum(1 for r in results if r['label'] == 'bad')
n_unc = sum(1 for r in results if r['label'] == 'uncertain')
print(f"\n质量标签: 好={n_good}, 坏={n_bad}, 不确定={n_unc}")

# ── 验证: 好变换的匹配点RMS应很小 ──
# chamfer被不匹配点拉高, 需要用紧阈值只看匹配点
good_chamfer = [r['chamfer'] for r in results if r['label'] == 'good']
bad_chamfer = [r['chamfer'] for r in results if r['label'] == 'bad']
if good_chamfer and bad_chamfer:
    print(f"\n标签验证 (chamfer距离, 包含不匹配点):")
    print(f"  好变换: med={np.median(good_chamfer):.1f}\" "
          f"({np.median(good_chamfer)/s0:.1f}px)")
    print(f"  坏变换: med={np.median(bad_chamfer):.1f}\" "
          f"({np.median(bad_chamfer)/s0:.1f}px)")
    ratio = np.median(bad_chamfer) / max(np.median(good_chamfer), 0.01)
    print(f"  坏/好比: {ratio:.1f}x")

# 紧阈值匹配RMS: 只看U→Wt距离<10"的匹配对
tight_thr = 10.0  # arcsec
print(f"\n紧阈值({tight_thr}\")匹配验证:")
for label_name, label_val in [('好', 'good'), ('坏', 'bad')]:
    subset = [r for r in results if r['label'] == label_val][:20]
    matched_counts = []
    matched_rms_list = []
    for r in subset:
        Wt = _transform(r['s'], r['theta'])
        Wt_tree = cKDTree(Wt)
        d_fwd, _ = Wt_tree.query(U)
        close = d_fwd < tight_thr
        n_matched = int(np.sum(close))
        matched_counts.append(n_matched)
        if n_matched >= 3:
            matched_rms_list.append(float(np.sqrt(np.mean(d_fwd[close]**2))))
    if matched_counts:
        print(f"  {label_name}变换: 匹配数中位={np.median(matched_counts):.0f}/{N}", end="")
        if matched_rms_list:
            print(f", 匹配点RMS={np.median(matched_rms_list):.2f}\" "
                  f"({np.median(matched_rms_list)/s0:.2f}px)")
        else:
            print(", 匹配点<3, 无法算RMS")

# 打印好变换的(s,θ)分布
good_thetas = [math.degrees(r['theta']) for r in results if r['label'] == 'good']
good_ss = [r['s'] for r in results if r['label'] == 'good']
if good_thetas:
    print(f"\n好变换θ: [{min(good_thetas):.1f}°, {max(good_thetas):.1f}°] "
          f"std={np.std(good_thetas):.2f}°")
    print(f"好变换s: [{min(good_ss):.4f}, {max(good_ss):.4f}] "
          f"std={np.std(good_ss):.4f}")

# ── 特征区分度分析 ──
features = ['angle_corr', 'coverage_iou', 'density_corr', 'd_u2w',
            'chamfer', 'bidir', 'mutual_nn', 'kde_overlap']

hib = {'angle_corr': True, 'coverage_iou': True, 'density_corr': True,
       'd_u2w': False, 'chamfer': False, 'bidir': True, 'mutual_nn': True,
       'kde_overlap': True}

cost = {'angle_corr': 1, 'coverage_iou': 1, 'density_corr': 1,
        'd_u2w': 2, 'chamfer': 2, 'bidir': 3, 'mutual_nn': 3,
        'kde_overlap': 4}

good_results = [r for r in results if r['label'] == 'good']
bad_results = [r for r in results if r['label'] == 'bad']

if n_good < 3 or n_bad < 3:
    print(f"\n好/坏样本不足 (好={n_good}, 坏={n_bad}), 无法分析")
    sys.exit(1)

print(f"\n{'='*120}")
print(f"  特征区分度 (好={n_good}, 坏={n_bad})")
print(f"{'='*120}")
print(f"  {'特征':<18} {'成本':>4} {'方向':>4} {'好-中位':>10} {'坏-中位':>10} "
      f"{'Cohen_d':>8} {'Spearman_ρ':>11} {'Top10好率':>9} {'Top20好率':>9}")
print(f"  {'-'*18} {'-'*4} {'-'*4} {'-'*10} {'-'*10} {'-'*8} {'-'*11} {'-'*9} {'-'*9}")

feat_stats = []
for feat in features:
    good_vals = np.array([r[feat] for r in good_results])
    bad_vals = np.array([r[feat] for r in bad_results])
    if len(good_vals) < 2 or len(bad_vals) < 2:
        continue
    g_med = np.median(good_vals)
    b_med = np.median(bad_vals)
    g_mean = np.mean(good_vals)
    b_mean = np.mean(bad_vals)
    g_std = max(np.std(good_vals), 1e-10)
    b_std = max(np.std(bad_vals), 1e-10)
    pooled = math.sqrt((g_std**2 + b_std**2) / 2)
    cohen_d = abs(g_mean - b_mean) / pooled

    # Spearman: 特征 vs θ距离 (连续质量指标, 越小越好)
    feat_all = [r[feat] for r in results]
    sp_corr, _ = spearmanr(feat_all, theta_dists.tolist())
    # 统一方向: 期望负 (特征好→θ距离小)
    sp_signed = sp_corr * (1 if hib[feat] else -1)

    # Top10/20好率
    is_hib = hib[feat]
    sorted_r = sorted(results, key=lambda r: -r[feat] if is_hib else r[feat])
    top10_good = sum(1 for r in sorted_r[:10] if r['label'] == 'good')
    top20_good = sum(1 for r in sorted_r[:20] if r['label'] == 'good')

    direction = "↑" if is_hib else "↓"
    print(f"  {feat:<18} {cost[feat]:>4} {direction:>4} {g_med:>10.4f} {b_med:>10.4f} "
          f"{cohen_d:>8.2f} {sp_signed:>+11.3f} {top10_good:>7}/10 {top20_good:>7}/20")

    feat_stats.append((cohen_d, feat, sp_signed, top10_good, top20_good, cost[feat]))

# ── 综合排名 ──
print(f"\n{'='*120}")
print(f"  综合排名 (ρ<0 且 Top10≥7 = 可靠)")
print(f"{'='*120}")
feat_stats_sorted = sorted(feat_stats, key=lambda x: -x[0])
for rank, (cd, feat, sp, t10, t20, c) in enumerate(feat_stats_sorted):
    direction = "↑好" if hib[feat] else "↓好"
    if sp < -0.3 and t10 >= 7:
        tag = "可靠"
    elif cd > 1.5 and (sp > 0 or t10 < 5):
        tag = "**伪真值**"
    else:
        tag = "待定"
    print(f"  #{rank+1} {feat:<18} Cohen_d={cd:.2f} ρ={sp:+.3f} "
          f"Top10={t10}/10 Top20={t20}/20 成本={c} {direction} → {tag}")

# ── 单特征阈值分类 ──
print(f"\n{'='*120}")
print(f"  单特征阈值分类 (最优F1)")
print(f"{'='*120}")
labels_arr = np.array([1 if r['label'] == 'good' else 0 for r in results])
for feat in features:
    is_hib_feat = hib[feat]
    vals = np.array([r[feat] for r in results])

    best_f1 = 0
    best_thr = None
    best_tp = best_fp = best_fn = 0
    percentiles = np.percentile(vals, np.arange(5, 96, 1))
    for thr in percentiles:
        pred = vals >= thr if is_hib_feat else vals <= thr
        tp = np.sum(pred & labels_arr)
        fp = np.sum(pred & ~labels_arr)
        fn = np.sum(~pred & labels_arr)
        if tp + fp > 0 and tp + fn > 0:
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
                best_tp, best_fp, best_fn = tp, fp, fn

    if best_thr is not None:
        prec = best_tp / (best_tp + best_fp) if (best_tp + best_fp) > 0 else 0
        rec = best_tp / (best_tp + best_fn) if (best_tp + best_fn) > 0 else 0
        print(f"  {feat:<18} F1={best_f1:.3f} P={prec:.3f} R={rec:.3f} "
              f"thr={best_thr:.4f} TP={best_tp} FP={best_fp} FN={best_fn} 成本={cost[feat]}")

# ── 性价比排名 ──
print(f"\n{'='*120}")
print(f"  性价比排名: Cohen_d / 成本")
print(f"{'='*120}")
perf_ratio = [(cd / c, feat, cd, sp, t10, t20, c) for cd, feat, sp, t10, t20, c in feat_stats]
perf_ratio.sort(key=lambda x: -x[0])
for rank, (ratio, feat, cd, sp, t10, t20, c) in enumerate(perf_ratio):
    print(f"  #{rank+1} {feat:<18} 性价比={ratio:.2f} Cohen_d={cd:.2f} "
          f"ρ={sp:+.3f} Top10={t10}/10 成本={c}")

# ── 最优低成本组合 ──
print(f"\n{'='*120}")
print(f"  最优低成本组合 (成本<=2)")
print(f"{'='*120}")
low_cost_feats = [f for f in features if cost[f] <= 2]
print(f"  低成本特征: {low_cost_feats}")

feat_min = {f: min(r[f] for r in results) for f in low_cost_feats}
feat_max = {f: max(r[f] for r in results) for f in low_cost_feats}

def _norm(val, feat):
    rng = feat_max[feat] - feat_min[feat]
    if rng < 1e-10: return 0.5
    v = (val - feat_min[feat]) / rng
    if not hib[feat]: v = 1.0 - v
    return v

# 2特征组合
best2 = (0, None)
for i in range(len(low_cost_feats)):
    for j in range(i+1, len(low_cost_feats)):
        f1, f2 = low_cost_feats[i], low_cost_feats[j]
        for r in results:
            r['_c2'] = _norm(r[f1], f1) + _norm(r[f2], f2)
        sr = sorted(results, key=lambda r: -r['_c2'])
        t10 = sum(1 for r in sr[:10] if r['label'] == 'good')
        t20 = sum(1 for r in sr[:20] if r['label'] == 'good')
        score = t10 * 10 + t20
        if score > best2[0]:
            best2 = (score, (f1, f2, t10, t20))

if best2[1]:
    f1, f2, t10, t20 = best2[1]
    print(f"  最优2特征: {f1} + {f2} → Top10={t10}/10 Top20={t20}/20")
    for r in results:
        r['_c2'] = _norm(r[f1], f1) + _norm(r[f2], f2)
    sr = sorted(results, key=lambda r: -r['_c2'])
    for rank, r in enumerate(sr[:10]):
        tag = "OK" if r['label'] == 'good' else "XX"
        print(f"    #{rank} {tag} u={r['ui']:2d} w={r['wi']:3d} "
              f"combo={r['_c2']:.3f} {f1}={r[f1]:.4f} {f2}={r[f2]:.4f} "
              f"θ={math.degrees(r['theta']):.1f}° s={r['s']:.4f} chamfer={r['chamfer']:.1f}\"")

# 3特征组合
best3 = (0, None)
for i in range(len(low_cost_feats)):
    for j in range(i+1, len(low_cost_feats)):
        for k in range(j+1, len(low_cost_feats)):
            f1, f2, f3 = low_cost_feats[i], low_cost_feats[j], low_cost_feats[k]
            for r in results:
                r['_c3'] = _norm(r[f1], f1) + _norm(r[f2], f2) + _norm(r[f3], f3)
            sr = sorted(results, key=lambda r: -r['_c3'])
            t10 = sum(1 for r in sr[:10] if r['label'] == 'good')
            t20 = sum(1 for r in sr[:20] if r['label'] == 'good')
            score = t10 * 10 + t20
            if score > best3[0]:
                best3 = (score, (f1, f2, f3, t10, t20))

if best3[1]:
    f1, f2, f3, t10, t20 = best3[1]
    print(f"\n  最优3特征: {f1} + {f2} + {f3} → Top10={t10}/10 Top20={t20}/20")
    for r in results:
        r['_c3'] = _norm(r[f1], f1) + _norm(r[f2], f2) + _norm(r[f3], f3)
    sr = sorted(results, key=lambda r: -r['_c3'])
    for rank, r in enumerate(sr[:10]):
        tag = "OK" if r['label'] == 'good' else "XX"
        print(f"    #{rank} {tag} u={r['ui']:2d} w={r['wi']:3d} "
              f"combo={r['_c3']:.3f} {f1}={r[f1]:.4f} {f2}={r[f2]:.4f} {f3}={r[f3]:.4f} "
              f"θ={math.degrees(r['theta']):.1f}° s={r['s']:.4f} chamfer={r['chamfer']:.1f}\"")

# ── Top10详细对比 ──
print(f"\n{'='*120}")
print(f"  Top10详细: 各特征选出的Top10的chamfer和θ")
print(f"{'='*120}")
for feat in features:
    is_hib_feat = hib[feat]
    sr = sorted(results, key=lambda r: -r[feat] if is_hib_feat else r[feat])
    top10 = sr[:10]
    n_good_top10 = sum(1 for r in top10 if r['label'] == 'good')
    top10_chamfer = [r['chamfer'] for r in top10]
    top10_thetas = [math.degrees(r['theta']) for r in top10]
    print(f"  {feat:<18} Top10好率={n_good_top10}/10 "
          f"chamfer_med={np.median(top10_chamfer):.1f}\" "
          f"θ=[{min(top10_thetas):.1f}°,{max(top10_thetas):.1f}°]")

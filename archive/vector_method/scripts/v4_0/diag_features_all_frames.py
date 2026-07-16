"""
V4.0 特征鲁棒性实验 — 4模式平行, 全帧验证 (v3)
实验文档: experiment_feature_robustness.md

核心改动:
  - 饱和星≥50: 全部饱和; <50: 全部饱和+flux高补足50
  - 星表侧: FOV对角线为搜索直径, 按图像密度×面积取星
  - 变换后裁剪Wt到FOV内再算特征 (解决M>>N噪声问题)
  - 抽样数随N×M缩放
  - 4模式平行, 打印所有模式结果
  - 打印正确模式的向量组供目视验证
"""
import sys, os, math, time, random, numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import GaiaClientPy, gnomonic_forward, _apply_flip
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

LIGHTS = os.path.join(PROJECT_ROOT, "testdata", "lights")
GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3")

# ── 选帧: 每个target+filter抽1帧 ──
all_files = [f for f in os.listdir(LIGHTS) if f.endswith('.fts')]
by_tf = defaultdict(list)
for f in all_files:
    parts = f.split('_')
    target = parts[0]
    filt = parts[-1].replace('.fts', '').split('-')[-1]
    by_tf[(target, filt)].append(f)

selected = []
for key, files in sorted(by_tf.items()):
    selected.append((key, os.path.join(LIGHTS, files[0])))

print(f"共{len(selected)}个目标+滤镜组合")

# ── 角度距离 ──
def angle_dist(a, b):
    d = (a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

# ── 特征函数 (裁剪Wt到FOV内) ──
def f_density_corr(U, Wt, halfW, halfH, gn=16):
    # 裁剪Wt到FOV内
    mask = (np.abs(Wt[:, 0]) < halfW) & (np.abs(Wt[:, 1]) < halfH)
    Wt_clip = Wt[mask]
    if len(Wt_clip) < 3: return 0.0
    bx = np.linspace(-halfW, halfW, gn+1)
    by = np.linspace(-halfH, halfH, gn+1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bx, by])
    h_w, _, _ = np.histogram2d(Wt_clip[:, 0], Wt_clip[:, 1], bins=[bx, by])
    c = np.corrcoef(h_u.ravel(), h_w.ravel())[0, 1]
    return c if not np.isnan(c) else 0.0

def f_coverage_iou(U, Wt, halfW, halfH, gn=8):
    mask = (np.abs(Wt[:, 0]) < halfW) & (np.abs(Wt[:, 1]) < halfH)
    Wt_clip = Wt[mask]
    if len(Wt_clip) < 3: return 0.0
    bx = np.linspace(-halfW, halfW, gn+1)
    by = np.linspace(-halfH, halfH, gn+1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bx, by])
    h_w, _, _ = np.histogram2d(Wt_clip[:, 0], Wt_clip[:, 1], bins=[bx, by])
    h_u = (h_u > 0).astype(float)
    h_w = (h_w > 0).astype(float)
    inter = np.sum(h_u * h_w)
    union = np.sum(np.maximum(h_u, h_w))
    return inter / max(union, 1)

# ── 处理单帧 (4模式全打印) ──
def process_frame(fits_path):
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    wcs = img.metadata.wcs
    if wcs and wcs.crval1 is not None:
        cra0, cdec0 = wcs.crval1, wcs.crval2
    else:
        from astropy.io import fits as pf
        hdr = pf.getheader(fits_path)
        ra_raw = hdr.get('RA', hdr.get('OBJRA', None))
        dec_raw = hdr.get('DEC', hdr.get('OBJDEC', None))
        if ra_raw is None or dec_raw is None:
            return None
        def _ps(s, is_ra=False):
            s = str(s).strip().replace(' ', ':')
            if ':' not in s: return float(s)
            p = s.split(':')
            v = abs(float(p[0])) + float(p[1])/60.0 + float(p[2])/3600.0
            if is_ra: v *= 15.0
            if str(p[0]).startswith('-'): v = -v
            return v
        cra0, cdec0 = _ps(ra_raw, True), _ps(dec_raw, False)

    s0 = 206.265 * ps / fl
    cx, cy = w / 2.0, h / 2.0
    halfW = w / 2.0 * s0
    halfH = h / 2.0 * s0

    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)

    # ── 星点选取: 饱和≥50全用, <50补足50 ──
    sat_mask = np.array(det.saturated, dtype=bool)
    sat_x = np.array(det.x, np.float64)
    sat_y = np.array(det.y, np.float64)
    sat_flux = np.array(det.flux, np.float64)
    nsat = int(np.sum(sat_mask))

    if nsat >= 50:
        sel_x = sat_x[sat_mask]
        sel_y = sat_y[sat_mask]
    else:
        sel_x = sat_x[sat_mask]
        sel_y = sat_y[sat_mask]
        non_sat = ~sat_mask
        if np.any(non_sat):
            non_sat_idx = np.where(non_sat)[0]
            non_sat_flux = sat_flux[non_sat_idx]
            top_k = min(50 - nsat, len(non_sat_idx))
            top_idx = non_sat_idx[np.argsort(non_sat_flux)[-top_k:]]
            sel_x = np.concatenate([sel_x, sat_x[top_idx]])
            sel_y = np.concatenate([sel_y, sat_y[top_idx]])

    N = len(sel_x)
    if N < 5:
        return None

    U = np.column_stack([(sel_x - cx) * s0, -(sel_y - cy) * s0])

    # ── 星表侧: FOV对角线搜索, 按密度取星 ──
    fov_diag = math.sqrt(w*w + h*h) * s0
    fov_area = (w * s0) * (h * s0)
    star_density = N / fov_area

    gaia = GaiaClientPy(GAIA_DIR, 1)
    qr = fov_diag / 3600.0  # 搜索半径(度)
    m_cut = 6.0 + 1.5 * math.log10(fl) + 2.0 * math.log10(300.0)
    ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, qr, min(m_cut, 22.0))
    gaia.close()

    xi_all, eta_all, valid = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
    in_fov = valid & (np.abs(xi_all) < halfW * 2) & (np.abs(eta_all) < halfH * 2)
    fov_idx = np.where(in_fov)[0]
    if len(fov_idx) == 0:
        return None

    search_area = math.pi * (fov_diag/2)**2
    M_target = max(int(star_density * search_area * 1.5), N)
    M_target = min(M_target, len(fov_idx))

    sorted_mag = np.argsort(mag_t[fov_idx])
    W = np.column_stack([xi_all[fov_idx[sorted_mag[:M_target]]],
                         eta_all[fov_idx[sorted_mag[:M_target]]]])

    # 抽样数: 保证正确配对命中率, 正确对约N个(每颗U星对应1颗W星)
    # 命中率 ≈ N / (N*M) = 1/M, 需要 ~50次命中 → n_samples = 50*M
    n_samples = min(max(50 * M_target, 2000), 100000)

    # 4种模式
    mode_results = {}
    for mode in range(4):
        Wf = _apply_flip(W, mode)
        M = len(Wf)

        results = []
        for _ in range(n_samples):
            ui = random.randint(0, N-1)
            wi = random.randint(0, M-1)
            u_vec = U[ui]
            w_vec = Wf[wi]
            u_norm = math.hypot(u_vec[0], u_vec[1])
            w_norm = math.hypot(w_vec[0], w_vec[1])
            if w_norm < 1e-10: continue
            s = u_norm / w_norm
            if s < 0.85 or s > 1.15: continue
            theta = math.atan2(u_vec[1], u_vec[0]) - math.atan2(w_vec[1], w_vec[0])

            cos_t, sin_t = math.cos(theta), math.sin(theta)
            Wt = np.column_stack([
                s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]),
                s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1])
            ])

            dc = f_density_corr(U, Wt, halfW, halfH)
            ci = f_coverage_iou(U, Wt, halfW, halfH)
            results.append({'s': s, 'theta': theta, 'density_corr': dc, 'coverage_iou': ci})

        if len(results) < 10:
            mode_results[mode] = None
            continue

        # θ直方图聚类
        thetas = np.array([r['theta'] for r in results])
        n_bins = 72
        theta_bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        theta_hist, _ = np.histogram(thetas, bins=theta_bins)
        peak_bin = np.argmax(theta_hist)
        peak_theta = 0.5 * (theta_bins[peak_bin] + theta_bins[peak_bin + 1])
        peak_count = theta_hist[peak_bin]
        theta_dists = angle_dist(thetas, peak_theta)

        near_peak = theta_dists < np.radians(5)
        peak_s = np.median(np.array([r['s'] for r in results])[near_peak]) if np.sum(near_peak) > 0 else 1.0

        for i, r in enumerate(results):
            d_th = theta_dists[i]
            ds = abs(r['s'] - peak_s) / peak_s
            r['is_good'] = d_th < np.radians(5) and ds < 0.03

        n_good = sum(1 for r in results if r['is_good'])
        good_dc = [r['density_corr'] for r in results if r['is_good']]
        good_ci = [r['coverage_iou'] for r in results if r['is_good']]
        bad_dc = [r['density_corr'] for r in results if not r['is_good']]
        bad_ci = [r['coverage_iou'] for r in results if not r['is_good']]

        mode_results[mode] = {
            'results': results,
            'n_good': n_good,
            'n_bad': len(results) - n_good,
            'peak_theta': peak_theta,
            'peak_count': peak_count,
            'peak_s': peak_s,
            'good_dc_med': np.median(good_dc) if good_dc else 0,
            'bad_dc_med': np.median(bad_dc) if bad_dc else 0,
            'good_ci_med': np.median(good_ci) if good_ci else 0,
            'bad_ci_med': np.median(bad_ci) if bad_ci else 0,
            'Wf': Wf,
        }

    return {
        'mode_results': mode_results,
        'U': U,
        'nsat': nsat,
        'N': N,
        'M': M_target,
        'halfW': halfW,
        'halfH': halfH,
        'star_density': star_density,
        'n_samples': n_samples,
    }

# ── 批量处理 ──
print(f"\n开始处理 {len(selected)} 帧...")
all_data = []
for idx, ((target, filt), path) in enumerate(selected):
    t0 = time.perf_counter()
    fname = os.path.basename(path)
    try:
        data = process_frame(path)
        elapsed = time.perf_counter() - t0
        if data is None:
            print(f"  [{idx+1}/{len(selected)}] {target} {filt}: 跳过")
            continue

        # 打印4模式结果
        print(f"\n  [{idx+1}/{len(selected)}] {target} {filt} "
              f"nsat={data['nsat']} N={data['N']} M={data['M']} "
              f"samples={data['n_samples']} 耗时={elapsed:.1f}s")
        print(f"  {'mode':>4} {'好':>4} {'坏':>4} {'θ_peak':>8} {'s_peak':>7} "
              f"{'好-dc':>7} {'坏-dc':>7} {'好-ci':>7} {'坏-ci':>7} {'dc_sep':>6} {'ci_sep':>6}")

        best_mode = -1
        best_score = -1
        for mode in range(4):
            mr = data['mode_results'].get(mode)
            if mr is None:
                print(f"  {mode:>4}  ---")
                continue
            g_dc = mr['good_dc_med']
            b_dc = mr['bad_dc_med']
            g_ci = mr['good_ci_med']
            b_ci = mr['bad_ci_med']
            dc_sep = g_dc / max(b_dc, 0.001)
            ci_sep = g_ci / max(b_ci, 0.001)
            score = mr['peak_count'] * g_dc
            if score > best_score:
                best_score = score
                best_mode = mode
            print(f"  {mode:>4} {mr['n_good']:>4} {mr['n_bad']:>4} "
                  f"{math.degrees(mr['peak_theta']):>7.1f}° {mr['peak_s']:>7.4f} "
                  f"{g_dc:>7.3f} {b_dc:>7.3f} {g_ci:>7.3f} {b_ci:>7.3f} "
                  f"{dc_sep:>5.1f}x {ci_sep:>5.1f}x")

        mr_best = data['mode_results'].get(best_mode)
        if mr_best:
            print(f"  → 最佳模式={best_mode} (peak×dc={best_score:.2f})")

        data['best_mode'] = best_mode
        data['target'] = target
        data['filter'] = filt
        data['fname'] = fname
        all_data.append(data)

    except Exception as e:
        import traceback
        print(f"  [{idx+1}/{len(selected)}] {target} {filt}: 错误 {e}")
        traceback.print_exc()

# ── 汇总表 ──
print(f"\n{'='*120}")
print(f"  汇总: 各帧4模式 density_corr(好变换中位)")
print(f"{'='*120}")
print(f"  {'帧':<25} {'nsat':>5} {'N':>4} {'M':>4} {'best':>4} "
      f"{'m0_dc':>7} {'m1_dc':>7} {'m2_dc':>7} {'m3_dc':>7} | "
      f"{'m0_ci':>7} {'m1_ci':>7} {'m2_ci':>7} {'m3_ci':>7}")
print(f"  {'-'*25} {'-'*5} {'-'*4} {'-'*4} {'-'*4} "
      f"{'-'*7} {'-'*7} {'-'*7} {'-'*7} | "
      f"{'-'*7} {'-'*7} {'-'*7} {'-'*7}")

for data in all_data:
    name = f"{data['target']} {data['filter']}"
    dc_m, ci_m = [], []
    for mode in range(4):
        mr = data['mode_results'].get(mode)
        if mr:
            dc_m.append(f"{mr['good_dc_med']:>7.3f}")
            ci_m.append(f"{mr['good_ci_med']:>7.3f}")
        else:
            dc_m.append(f"{'---':>7}")
            ci_m.append(f"{'---':>7}")
    print(f"  {name:<25} {data['nsat']:>5} {data['N']:>4} {data['M']:>4} {data['best_mode']:>4} "
          f"{' '.join(dc_m)} | {' '.join(ci_m)}")

# ── 向量组可视化 ──
print(f"\n生成向量组可视化...")
n_frames = len(all_data)
ncols = 4
nrows = math.ceil(n_frames / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 5*nrows))
if nrows == 1:
    axes = axes.reshape(1, -1)

for idx, data in enumerate(all_data):
    row, col = divmod(idx, ncols)
    ax = axes[row, col]
    U = data['U']
    best = data['best_mode']
    mr = data['mode_results'].get(best)
    if mr is None:
        ax.set_title(f"{data['target']} {data['filter']} (无数据)")
        continue

    # 取好变换中dc最高的
    good_results = [r for r in mr['results'] if r['is_good']]
    if good_results:
        best_r = max(good_results, key=lambda r: r['density_corr'])
        s, theta = best_r['s'], best_r['theta']
        Wf = mr['Wf']
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        Wt = np.column_stack([
            s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]),
            s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1])
        ])
    else:
        Wt = mr['Wf']

    # 裁剪Wt到FOV内
    mask = (np.abs(Wt[:, 0]) < data['halfW']) & (np.abs(Wt[:, 1]) < data['halfH'])
    Wt_clip = Wt[mask]

    ax.scatter(U[:, 0], U[:, 1], c='#4ecdc4', s=15, alpha=0.8,
               label=f'U(img,{len(U)})', zorder=2, edgecolors='white', linewidths=0.3)
    ax.scatter(Wt_clip[:, 0], Wt_clip[:, 1], c='#ff6b6b', s=8, alpha=0.4,
               label=f'Wt(cat,{len(Wt_clip)})', zorder=1)
    ax.set_aspect('equal')
    ax.set_xlim(-data['halfW']*1.1, data['halfW']*1.1)
    ax.set_ylim(-data['halfH']*1.1, data['halfH']*1.1)
    ax.set_title(f"{data['target']} {data['filter']} m={best}\n"
                 f"nsat={data['nsat']} N={data['N']} dc={mr['good_dc_med']:.3f} ci={mr['good_ci_med']:.3f}",
                 fontsize=8)
    ax.legend(fontsize=6, loc='upper right')
    ax.tick_params(labelsize=6)

for idx in range(n_frames, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row, col].set_visible(False)

plt.suptitle('U(image) vs Wt(catalog, best mode, clipped to FOV) — 向量组对比', fontsize=12, y=1.01)
plt.tight_layout()

out_path = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_0",
                         "feature_scatter_all_frames.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"图已保存: {out_path}")

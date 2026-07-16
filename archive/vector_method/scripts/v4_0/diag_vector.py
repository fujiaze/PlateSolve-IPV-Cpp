"""
V4.0 最小向量诊断脚本
功能: 只用饱和星, 绘制图像侧/星表侧向量 + 每个控制点变换结果 + SCM分数
用途: 诊断向量匹配过程, 找出CD偏差根因
"""
import sys, os, math, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import (
    GaiaClientPy, gnomonic_forward, _apply_flip, _apply_similarity,
)
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# ── 参数 ──
FITS = os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
MODE = 1  # flipX
OUT = os.path.join(PROJECT_ROOT, "overlay_output", "_v4_vector_diag.png")

# ── 读图+检测 ──
reader = ImageReader()
img = reader.read(FITS)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2
s0 = 206.265 * ps / fl  # "/px

detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)

print(f"图像: {w}x{h}, fl={fl}mm, s0={s0:.4f}\"/px")
print(f"检测星: {len(det.x)}颗, 饱和: {int(np.sum(det.saturated))}")

# ── 只取饱和星 ──
sat_mask = np.array(det.saturated, dtype=bool)
sat_x = np.array(det.x, np.float64)[sat_mask]
sat_y = np.array(det.y, np.float64)[sat_mask]
sat_flux = np.array(det.flux, np.float64)[sat_mask]
nsat = len(sat_x)
print(f"饱和星: {nsat}颗")

# ── 图像侧向量: 从中心出发 ──
cx, cy = w / 2.0, h / 2.0
U = np.column_stack([(sat_x - cx) * s0, -(sat_y - cy) * s0])  # Y翻转

# ── Gaia查询 ──
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
gaia = GaiaClientPy(gaia_dir, 1)
fov_diag = math.sqrt(w*w + h*h) * s0
qr = fov_diag * 0.5 / 3600.0
m_cut = 6.0 + 1.5 * math.log10(fl) + 2.0 * math.log10(300.0)
ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, qr, min(m_cut, 22.0))
gaia.close()

xi_all, eta_all, valid = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
halfW = w / 2.0 * s0
halfH = h / 2.0 * s0
in_fov = valid & (np.abs(xi_all) < halfW * 2) & (np.abs(eta_all) < halfH * 2)
fov_idx = np.where(in_fov)[0]
sorted_mag = np.argsort(mag_t[fov_idx])
W = np.column_stack([xi_all[fov_idx[sorted_mag]], eta_all[fov_idx[sorted_mag]]])
M_all = W.shape[0]
gaia_ra = ra_t[fov_idx[sorted_mag]]
gaia_dec = dec_t[fov_idx[sorted_mag]]
print(f"Gaia FOV内: {M_all}颗")

# ── 翻转 ──
Wf = _apply_flip(W, MODE)

# ── 初始池: 饱和星(U) + 最亮Gaia(W) ──
N_init = min(nsat, 50)
M_init = min(100, M_all)
U_init = U[:N_init]
Wf_init = Wf[:M_init]
print(f"初始池: U={N_init} W={M_init}")

# ── 1点抽样 + SCM评分 ──
from test_v40_prototype import compute_scm, compute_3nn_signatures

U_bright = U[:min(50, nsat)]
U_bright_sigs = compute_3nn_signatures(U_bright)

# 遍历所有配对, 找出SCM最高的
results = []
for ui in range(N_init):
    for wi in range(M_init):
        u_vec = U_init[ui]
        w_vec = Wf_init[wi]
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

        # SCM
        Wf_bright = Wf_init[:min(100, M_init)]
        Wt = np.column_stack([
            s * (cos_t * Wf_bright[:, 0] - sin_t * Wf_bright[:, 1]) + tx,
            s * (sin_t * Wf_bright[:, 0] + cos_t * Wf_bright[:, 1]) + ty
        ])
        scm, coverage, n_confirmed = compute_scm(U_bright, Wt, halfW, halfH, U_bright_sigs)

        if scm >= 0.35 and n_confirmed >= 3:
            results.append({
                'ui': ui, 'wi': wi,
                's': s, 'theta': theta, 'tx': tx, 'ty': ty,
                'scm': scm, 'cov': coverage, 'n_conf': n_confirmed,
            })

# 按SCM排序
results.sort(key=lambda r: -r['scm'])
print(f"\n有效配对: {len(results)} (SCM≥0.35, n_conf≥3)")

# ── 绘图 ──
fig, axes = plt.subplots(1, 2, figsize=(32, 16))
fig.patch.set_facecolor('black')

# ── 左图: 图像侧向量 ──
ax = axes[0]
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.set_title('Image side: saturated star vectors (from center)', color='white', fontsize=14)
# 画向量
for i in range(N_init):
    ax.annotate('', xy=(U[i, 0], U[i, 1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='cyan', lw=0.8, alpha=0.6))
    ax.plot(U[i, 0], U[i, 1], 'o', color='cyan', ms=4, alpha=0.8)
ax.plot(0, 0, '+', color='red', ms=15, mew=2)
ax.set_xlim(-halfW * 1.1, halfW * 1.1)
ax.set_ylim(-halfH * 1.1, halfH * 1.1)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

# ── 右图: 星表侧变换结果 ──
ax = axes[1]
ax.set_facecolor('black')
ax.set_aspect('equal')
ax.set_title(f'Catalog side: transformed Wf (mode={MODE}, SCM>=0.35)', color='white', fontsize=14)

# 原始Wf向量 (灰色)
for i in range(min(100, M_init)):
    ax.annotate('', xy=(Wf_init[i, 0], Wf_init[i, 1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.3, alpha=0.3))

# 每个高分变换: 变换后的Wt向量组
n_show = min(20, len(results))
cmap = plt.cm.hot
for rank, r in enumerate(results[:n_show]):
    s, theta, tx, ty = r['s'], r['theta'], r['tx'], r['ty']
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wf_bright = Wf_init[:min(100, M_init)]
    Wt = np.column_stack([
        s * (cos_t * Wf_bright[:, 0] - sin_t * Wf_bright[:, 1]) + tx,
        s * (sin_t * Wf_bright[:, 0] + cos_t * Wf_bright[:, 1]) + ty
    ])

    color = cmap(rank / max(n_show - 1, 1))
    # 画变换后的向量
    for j in range(len(Wt)):
        ax.annotate('', xy=(Wt[j, 0], Wt[j, 1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.3, alpha=0.3))
    # 画变换后的点
    ax.plot(Wt[:, 0], Wt[:, 1], '.', color=color, ms=2, alpha=0.5)

    # 标注: 控制点(u_idx, w_idx)的位置和分数
    u_pos = U_init[r['ui']]
    w_pos = Wf_init[r['wi']]
    w_proj = np.array([s * (cos_t * w_pos[0] - sin_t * w_pos[1]) + tx,
                       s * (sin_t * w_pos[0] + cos_t * w_pos[1]) + ty])
    ax.plot(w_proj[0], w_proj[1], 's', color=color, ms=8, alpha=0.9)
    ax.annotate(f"#{rank} SCM={r['scm']:.3f}\n"
                f"s={r['s']:.4f} θ={math.degrees(r['theta']):.1f}°\n"
                f"n_conf={r['n_conf']}",
                xy=(w_proj[0], w_proj[1]),
                xytext=(10, 10), textcoords='offset points',
                color=color, fontsize=7,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.5))

# 画U_bright参考点 (白色虚线圆)
for i in range(len(U_bright)):
    ax.plot(U_bright[i, 0], U_bright[i, 1], 'o', color='white', ms=3, alpha=0.3,
            markerfacecolor='none', markeredgewidth=0.5)

ax.plot(0, 0, '+', color='red', ms=15, mew=2)
ax.set_xlim(-halfW * 1.1, halfW * 1.1)
ax.set_ylim(-halfH * 1.1, halfH * 1.1)
ax.tick_params(colors='white')
for spine in ax.spines.values():
    spine.set_color('white')

plt.tight_layout(pad=0.5)
plt.savefig(OUT, dpi=150, bbox_inches='tight', pad_inches=0,
            facecolor='black', edgecolor='none')
plt.close()
print(f"\n保存: {OUT}")

# ── 打印详细结果 ──
print(f"\n{'='*80}")
print(f"  Top {n_show} 匹配结果 (按SCM降序)")
print(f"{'='*80}")
for rank, r in enumerate(results[:n_show]):
    u_pos = U_init[r['ui']]
    w_pos = Wf_init[r['wi']]
    # 像素坐标
    u_px_x = u_pos[0] / s0 + cx
    u_px_y = -u_pos[1] / s0 + cy
    # Gaia坐标
    g_ra = gaia_ra[r['wi']]
    g_dec = gaia_dec[r['wi']]
    print(f"  #{rank:2d} u={r['ui']:2d} w={r['wi']:3d} | "
          f"SCM={r['scm']:.3f} cov={r['cov']:.3f} n_conf={r['n_conf']:2d} | "
          f"s={r['s']:.4f} θ={math.degrees(r['theta']):+7.2f}° | "
          f"px=({u_px_x:.0f},{u_px_y:.0f}) sky=({g_ra:.4f},{g_dec:.4f}) | "
          f"tx={r['tx']:.1f} ty={r['ty']:.1f}")

# ── θ分布 ──
thetas = [math.degrees(r['theta']) for r in results]
ss = [r['s'] for r in results]
print(f"\nθ分布: min={min(thetas):.2f}° max={max(thetas):.2f}° "
      f"med={np.median(thetas):.2f}° std={np.std(thetas):.2f}°")
print(f"s分布: min={min(ss):.4f} max={max(ss):.4f} "
      f"med={np.median(ss):.4f} std={np.std(ss):.4f}")

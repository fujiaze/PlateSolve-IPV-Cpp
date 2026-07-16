"""V3.5 SIP残差分析脚本 - 诊断SIP精度问题"""
import os, sys, json, numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

# 读取WCS JSON
with open(os.path.join(PROJECT_ROOT, "vm35_wcs_output.json"), "r", encoding="utf-8") as f:
    d = json.load(f)

CD = np.array(d["CD"]).reshape(2, 2)
CRVAL = np.array(d["CRVAL"])
CRPIX = np.array(d["CRPIX"])
A = np.array(d["SIP_A"]).reshape(6, 6)
B = np.array(d["SIP_B"]).reshape(6, 6)
sip_order = d["SIP_ORDER"]
rms_px = d["RMS_PX"]

print(f"CD: {CD}")
print(f"CRVAL: {CRVAL}")
print(f"CRPIX: {CRPIX}")
print(f"SIP order: {sip_order}, RMS: {rms_px:.6f} px")

# CD逆矩阵
cdet = CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0]
CD_inv = np.array([[CD[1, 1], -CD[0, 1]], [-CD[1, 0], CD[0, 0]]]) / cdet

# 读取FITS
reader = ImageReader()
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl

# 查询Gaia
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov = np.sqrt(w * w + h * h) * s0 / 3600
_, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
    gaia, CRVAL[0], CRVAL[1], max(0.8, fov * 1.2), 500)
gaia.close()

mag_a = np.array(cat_mag, np.float64)
idx = np.argsort(mag_a)[:500]
ra_src = np.array(cat_ra, np.float64)[idx]
dec_src = np.array(cat_dec, np.float64)[idx]

# 线性投影(无SIP)
dra = ra_src - CRVAL[0]
ddec = dec_src - CRVAL[1]
xi_prime = CD_inv[0, 0] * dra + CD_inv[0, 1] * ddec
eta_prime = CD_inv[1, 0] * dra + CD_inv[1, 1] * ddec
x_lin = xi_prime + CRPIX[0]
y_lin = eta_prime + CRPIX[1]

# SIP逆投影
xi = xi_prime.copy()
eta = eta_prime.copy()
for it in range(30):
    sip_dx = np.zeros_like(xi)
    sip_dy = np.zeros_like(eta)
    for p in range(6):
        for q in range(6):
            if p + q < 2 or p + q > sip_order:
                continue
            xi_c = np.clip(xi, -1e4, 1e4)
            eta_c = np.clip(eta, -1e4, 1e4)
            term = xi_c ** p * eta_c ** q
            sip_dx += A[p, q] * term
            sip_dy += B[p, q] * term
    xi_new = xi_prime - sip_dx
    eta_new = eta_prime - sip_dy
    if np.max(np.abs(xi_new - xi)) < 1e-6 and np.max(np.abs(eta_new - eta)) < 1e-6:
        print(f"SIP迭代收敛: {it + 1}次")
        break
    xi = xi_new
    eta = eta_new

x_sip = xi + CRPIX[0]
y_sip = eta + CRPIX[1]

# 帧内星点
in_frame = (x_sip > 0) & (x_sip < w) & (y_sip > 0) & (y_sip < h)
r = np.sqrt(xi_prime ** 2 + eta_prime ** 2)  # 距CRPIX距离(px)
sip_corr = np.sqrt((x_lin - x_sip) ** 2 + (y_lin - y_sip) ** 2)  # SIP修正量

print(f"\n帧内星数: {in_frame.sum()}/500")
print(f"\nSIP修正量统计(帧内):")
print(f"  均值: {sip_corr[in_frame].mean():.3f} px")
print(f"  最大: {sip_corr[in_frame].max():.3f} px")

# 按距离分组
print(f"\n按距中心距离分组:")
for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]:
    mask = in_frame & (r >= r_lo) & (r < r_hi)
    if mask.sum() > 0:
        print(f"  r=[{r_lo},{r_hi}): n={mask.sum()}, SIP修正均值={sip_corr[mask].mean():.3f}px, "
              f"最大={sip_corr[mask].max():.3f}px")

# 检测星匹配
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
x_det = np.array(det.x, np.float64)
y_det = np.array(det.y, np.float64)

from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([x_det, y_det]))

# 线性投影匹配
gaia_lin = np.column_stack([x_lin[in_frame], y_lin[in_frame]])
dists_lin, _ = tree.query(gaia_lin, k=1)
matched_lin = dists_lin < 5.0

# SIP投影匹配
gaia_sip = np.column_stack([x_sip[in_frame], y_sip[in_frame]])
dists_sip, _ = tree.query(gaia_sip, k=1)
matched_sip = dists_sip < 5.0

r_in = r[in_frame]

print(f"\n线性投影匹配残差(按距离):")
for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]:
    mask = matched_lin & (r_in >= r_lo) & (r_in < r_hi)
    if mask.sum() > 0:
        print(f"  r=[{r_lo},{r_hi}): n={mask.sum()}, RMS={np.sqrt(np.mean(dists_lin[mask] ** 2)):.3f}px, "
              f"均值={dists_lin[mask].mean():.3f}px")

print(f"\nSIP投影匹配残差(按距离):")
for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]:
    mask = matched_sip & (r_in >= r_lo) & (r_in < r_hi)
    if mask.sum() > 0:
        print(f"  r=[{r_lo},{r_hi}): n={mask.sum()}, RMS={np.sqrt(np.mean(dists_sip[mask] ** 2)):.3f}px, "
              f"均值={dists_sip[mask].mean():.3f}px")

print(f"\n整体: 线性RMS={np.sqrt(np.mean(dists_lin[matched_lin] ** 2)):.3f}px, "
      f"SIP RMS={np.sqrt(np.mean(dists_sip[matched_sip] ** 2)):.3f}px")

# 分析残差方向模式
print(f"\n残差方向分析(SIP):")
dx_sip = x_sip[in_frame] - x_lin[in_frame]  # SIP修正的x分量
dy_sip = y_sip[in_frame] - y_lin[in_frame]  # SIP修正的y分量
for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]:
    mask = in_frame & (r >= r_lo) & (r < r_hi)
    if mask.sum() > 0:
        print(f"  r=[{r_lo},{r_hi}): dx均值={dx_sip[mask].mean():.3f}px, dy均值={dy_sip[mask].mean():.3f}px")

# 关键: 检查SIP修正量是否合理
a_vals = [abs(A[p,q]) for p in range(6) for q in range(6) if p+q>=2 and p+q<=sip_order]
print(f"\nSIP系数最大值: A_max={max(a_vals):.6e}")
print(f"在r=2000px处的SIP修正: A[0][2]*2000^2 = {A[0,2]*2000**2:.3f}px")
if sip_order >= 3:
    print(f"  A[0][3]*2000^3 = {A[0,3]*2000**3:.3f}px")
    print(f"  A[3][0]*2000^3 = {A[3,0]*2000**3:.3f}px")

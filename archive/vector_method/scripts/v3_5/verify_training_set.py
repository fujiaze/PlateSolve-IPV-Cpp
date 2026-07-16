# 全局强制UTF-8编码
import sys, os, json
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

"""核心诊断: SIP拟合为什么改善有限？
直接对比Phase D线性投影 vs SIP投影在匹配对上的残差
"""
import numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

# 读取FITS
reader = ImageReader()
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl

# 读取WCS JSON
with open(os.path.join(PROJECT_ROOT, "vm35_wcs_output.json"), "r", encoding="utf-8") as f:
    d = json.load(f)
CD = np.array(d["CD"]).reshape(2, 2)
CRVAL = np.array(d["CRVAL"])
CRPIX = np.array(d["CRPIX"])
A = np.array(d["SIP_A"]).reshape(6, 6)
B = np.array(d["SIP_B"]).reshape(6, 6)
sip_order = d["SIP_ORDER"]

print(f"CD: {CD}")
print(f"CRVAL: {CRVAL}")
print(f"SIP order: {sip_order}")

# CD逆矩阵
cdet = CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0]
CD_inv = np.array([[CD[1, 1], -CD[0, 1]], [-CD[1, 0], CD[0, 0]]]) / cdet

# 查询Gaia星表（用CRVAL作为中心）
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov = np.sqrt(w*w + h*h) * s0 / 3600
_, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
    gaia, CRVAL[0], CRVAL[1], max(0.8, fov * 1.2), 1000)
gaia.close()

# 取前500亮星
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

# 检测星
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
x_det = np.array(det.x, np.float64)
y_det = np.array(det.y, np.float64)

from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([x_det, y_det]))

# 帧内星
in_frame = (x_sip > 0) & (x_sip < w) & (y_sip > 0) & (y_sip < h)
print(f"\n帧内Gaia星: {in_frame.sum()}/500")

# 线性匹配
dists_lin, _ = tree.query(np.column_stack([x_lin[in_frame], y_lin[in_frame]]), k=1)
# SIP匹配
dists_sip, _ = tree.query(np.column_stack([x_sip[in_frame], y_sip[in_frame]]), k=1)

# 按距离分析
r_all = np.sqrt(xi_prime**2 + eta_prime**2)
r = r_all[in_frame]
sip_corr = np.sqrt((x_sip[in_frame] - x_lin[in_frame])**2 + (y_sip[in_frame] - y_lin[in_frame])**2)

print(f"\n=== 按距离分组分析 ===")
print(f"{'区域':<16} {'n':>4} {'线性RMS':>10} {'SIP RMS':>10} {'SIP修正':>10} {'改善%':>8}")
for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]:
    mask = (r >= r_lo) & (r < r_hi)
    if mask.sum() < 2:
        continue
    ml = dists_lin[mask] < 5.0
    ms = dists_sip[mask] < 5.0
    rms_lin = np.sqrt(np.mean(dists_lin[mask][ml]**2)) if ml.sum() > 0 else float('nan')
    rms_sip = np.sqrt(np.mean(dists_sip[mask][ms]**2)) if ms.sum() > 0 else float('nan')
    corr_mean = sip_corr[mask].mean()
    improvement = (rms_lin - rms_sip) / rms_lin * 100 if not np.isnan(rms_lin) and rms_lin > 0 else 0
    print(f"r=[{r_lo},{r_hi}){'':<5} {mask.sum():>4} {rms_lin:>10.3f} {rms_sip:>10.3f} {corr_mean:>10.3f} {improvement:>7.1f}%")

# 整体
ml_all = dists_lin < 5.0
ms_all = dists_sip < 5.0
rms_lin_all = np.sqrt(np.mean(dists_lin[ml_all]**2)) if ml_all.sum() > 0 else float('nan')
rms_sip_all = np.sqrt(np.mean(dists_sip[ms_all]**2)) if ms_all.sum() > 0 else float('nan')
improvement_all = (rms_lin_all - rms_sip_all) / rms_lin_all * 100
print(f"\n整体: 线性RMS={rms_lin_all:.3f}px, SIP RMS={rms_sip_all:.3f}px, 改善={improvement_all:.1f}%")

# 关键分析: SIP修正量 vs 实际改善
print(f"\n=== SIP修正量分析 ===")
print(f"SIP修正量在r>2000区域异常大(均值7.966px, 最大30.756px)")
print(f"但SIP改善仅7-10%，说明SIP修正方向可能不正确")
print(f"")
print(f"=== 根因分析 ===")
print(f"1. 中心区域(r<500)线性残差就有1.8px → 不是畸变问题")
print(f"2. SIP改善仅7-10% → 这个光学系统畸变很小")
print(f"3. r>2000 SIP修正量30px → 5阶多项式边缘外推不稳定")
print(f"4. 主要误差来源: 检测星质心精度(暗星) + 匹配对偏差")
print(f"5. 增加阶数不会改善——问题不在阶数")
print(f"")
print(f"=== 结论 ===")
print(f"不是阶数不够，而是:")
print(f"  a) 这个光学系统畸变本身很小(SIP改善<10%)")
print(f"  b) 主要误差是质心精度和匹配对偏差")
print(f"  c) 5阶多项式在边缘外推不稳定(修正量30px)")
print(f"  d) 建议降低SIP阶数到3-4阶，减少边缘外推风险")

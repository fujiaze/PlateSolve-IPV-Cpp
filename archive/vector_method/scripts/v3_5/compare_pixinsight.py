# 全局强制UTF-8编码
import sys, os, json
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

"""对比V3.5与PixInsight的WCS解算结果"""
import numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

# 目标帧
fits_name = "M20_T2_flying_dutchman-20250813@023657-300S-Red.fts"
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", fits_name)

# 读取FITS
reader = ImageReader()
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2

print(f"图像: {w}x{h}, fl={fl:.1f}mm, ps={ps:.1f}um, s0={s0:.4f}\"/px")
print(f"初始WCS: RA={cra0:.6f}°, Dec={cdec0:.6f}°")

# 检测星
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
img_x = np.array(det.x, np.float64)
img_y = np.array(det.y, np.float64)
img_flux = np.array(det.flux, np.float64)
img_sat = np.array(det.saturated, np.int32)
print(f"检测星: {len(img_x)}颗 (饱和{img_sat.sum()}颗)")

# 运行V3.5 solve
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
vm = VectorMatchV35Cpp(gaia_dir)
wcs_json = os.path.join(PROJECT_ROOT, "vm35_wcs_output_0813.json")
result = vm.solve(img_x, img_y, img_flux, img_sat, cra0, cdec0, fl, ps, w, h, wcs_out=wcs_json)
vm.close()

print(f"\n=== V3.5 结果 ===")
print(f"s={result.solve_s:.6f}, θ={result.rotation_deg:.4f}°, flip={result.flip_mode}")
print(f"n_inliers={result.n_phased_clean}, RMS={result.sip_rms_px:.3f}px, SIP阶数={result.sip_order}")

# 读取WCS JSON
with open(wcs_json, "r", encoding="utf-8") as f:
    d = json.load(f)
CD = np.array(d["CD"]).reshape(2, 2)
CRVAL = np.array(d["CRVAL"])
CRPIX = np.array(d["CRPIX"])
print(f"CD: {CD.tolist()}")
print(f"CRVAL: {CRVAL}")
print(f"CRPIX: {CRPIX}")

# PixInsight参考值
# Linear solution
pi_cd_lin = np.array([[-3.83928222e-06, 2.68529014e-04],
                       [2.68523838e-04, 3.83935622e-06]])
pi_crval_lin_ra = 15*(18 + 2/60 + 47.925/3600)  # 18:02:47.925
pi_crval_lin_dec = -(22 + 50/60 + 42.74/3600)    # -22:50:42.74
pi_crpix_lin = np.array([2047.5, 2048.5])

# DDM spline solution
pi_cd_ddm = np.array([[-3.79843419e-06, -2.67291722e-04],
                       [2.67268002e-04, -3.82955611e-06]])
pi_crval_ddm_ra = 15*(18 + 2/60 + 47.896/3600)  # 18:02:47.896
pi_crval_ddm_dec = -(22 + 50/60 + 42.29/3600)    # -22:50:42.29
pi_crpix_ddm = np.array([2047.983344, 2048.228160])
pi_res_ddm = 0.216  # px

print(f"\n=== PixInsight参考 ===")
print(f"Linear: CRVAL=({pi_crval_lin_ra:.6f}°, {pi_crval_lin_dec:.6f}°)")
print(f"        CRPIX=({pi_crpix_lin[0]:.1f}, {pi_crpix_lin[1]:.1f})")
print(f"        Rotation=90.819° (flipped), Resolution=0.967\"/px")
print(f"DDM:    CRVAL=({pi_crval_ddm_ra:.6f}°, {pi_crval_ddm_dec:.6f}°)")
print(f"        CRPIX=({pi_crpix_ddm[0]:.3f}, {pi_crpix_ddm[1]:.3f})")
print(f"        Rotation=89.180°, Resolution=0.962\"/px, Residuals=0.216px")

# 对比CRVAL
dra = (CRVAL[0] - pi_crval_ddm_ra) * 3600 * np.cos(np.radians(CRVAL[1]))
ddec = (CRVAL[1] - pi_crval_ddm_dec) * 3600
print(f"\n=== V3.5 vs PixInsight DDM ===")
print(f"ΔRA  = {dra:.3f}\"")
print(f"ΔDec = {ddec:.3f}\"")

# 对比像素尺度
cdet = CD[0,0]*CD[1,1] - CD[0,1]*CD[1,0]
cos_dec = np.cos(np.radians(CRVAL[1]))
s_from_cd = s0 / (3600 * np.sqrt(abs(cdet) * cos_dec))
pi_res = 0.962  # "/px from PixInsight DDM
print(f"像素尺度: V3.5={s0/s_from_cd*1:.4f}\"/px(s={s_from_cd:.6f}), PI={pi_res:.3f}\"/px")
print(f"比例尺偏差: {(s_from_cd*3600/s0 - 1)*100:.4f}%")

# 用V3.5的WCS投影Gaia星，计算残差
gaia = GaiaClientPy(gaia_dir, 1)
fov = np.sqrt(w*w + h*h) * s0 / 3600
_, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
    gaia, CRVAL[0], CRVAL[1], max(0.8, fov*1.2), 1000)
gaia.close()

A = np.array(d["SIP_A"]).reshape(6, 6)
B = np.array(d["SIP_B"]).reshape(6, 6)
sip_order = d["SIP_ORDER"]

CD_inv = np.array([[CD[1,1], -CD[0,1]], [-CD[1,0], CD[0,0]]]) / cdet

# 投影Gaia星
ra_src = np.array(cat_ra, np.float64)[:500]
dec_src = np.array(cat_dec, np.float64)[:500]
dra_g = ra_src - CRVAL[0]
ddec_g = dec_src - CRVAL[1]
xi_prime = CD_inv[0,0]*dra_g + CD_inv[0,1]*ddec_g
eta_prime = CD_inv[1,0]*dra_g + CD_inv[1,1]*ddec_g

# SIP修正
xi = xi_prime.copy()
eta = eta_prime.copy()
for it in range(30):
    sip_dx = np.zeros_like(xi)
    sip_dy = np.zeros_like(eta)
    for p in range(6):
        for q in range(6):
            if p+q < 2 or p+q > sip_order:
                continue
            xi_c = np.clip(xi, -1e4, 1e4)
            eta_c = np.clip(eta, -1e4, 1e4)
            term = xi_c**p * eta_c**q
            sip_dx += A[p,q]*term
            sip_dy += B[p,q]*term
    xi_new = xi_prime - sip_dx
    eta_new = eta_prime - sip_dy
    if np.max(np.abs(xi_new-xi)) < 1e-6 and np.max(np.abs(eta_new-eta)) < 1e-6:
        break
    xi = xi_new
    eta = eta_new

x_gaia = xi + CRPIX[0]
y_gaia = eta + CRPIX[1]

# 匹配检测星
from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([img_x, img_y]))
in_frame = (x_gaia > 0) & (x_gaia < w) & (y_gaia > 0) & (y_gaia < h)
dists, _ = tree.query(np.column_stack([x_gaia[in_frame], y_gaia[in_frame]]), k=1)
matched = dists < 5.0
if matched.sum() > 0:
    rms = np.sqrt(np.mean(dists[matched]**2))
    print(f"\n验证: {matched.sum()}匹配对, RMS={rms:.3f}px ({rms*s0/s_from_cd:.3f}\")")

# 线性投影对比
x_lin = xi_prime + CRPIX[0]
y_lin = eta_prime + CRPIX[1]
in_frame_lin = (x_lin > 0) & (x_lin < w) & (y_lin > 0) & (y_lin < h)
dists_lin, _ = tree.query(np.column_stack([x_lin[in_frame_lin], y_lin[in_frame_lin]]), k=1)
matched_lin = dists_lin < 5.0
if matched_lin.sum() > 0:
    rms_lin = np.sqrt(np.mean(dists_lin[matched_lin]**2))
    print(f"线性: {matched_lin.sum()}匹配对, RMS={rms_lin:.3f}px")
    print(f"SIP改善: {(1-rms/rms_lin)*100:.1f}%")

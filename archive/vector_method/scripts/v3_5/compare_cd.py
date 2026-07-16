"""对比Phase D原始CD vs Siril修正CD的精度"""
import json, numpy as np, sys, os

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from vector_match_v3_5_cpp import VectorMatchV35

# 读取FITS
reader = ImageReader()
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2

# 运行V3.5 solve
vm = VectorMatchV35()
result = vm.solve(img)
print(f"s={result.s:.6f}, theta={result.theta:.6f} deg, tx={result.tx:.3f}, ty={result.ty:.3f}")
print(f"n_inliers={result.n_inliers}, rms={result.rms:.6f}")

# 用Phase D的s/θ/tx/ty直接计算CD和CRVAL
s = result.s
theta = result.theta
tx = result.tx
ty = result.ty
flip_mode = result.best_mode  # mode 1 = flip_x

fx = (flip_mode == 1 or flip_mode == 3)
fy = (flip_mode == 2 or flip_mode == 3)
sign_x = -1.0 if fx else 1.0
sign_y = -1.0 if fy else 1.0

ct = np.cos(np.radians(theta))
st = np.sin(np.radians(theta))
cos_dec = np.cos(np.radians(cdec0))
s0_over_s_3600 = s0 / (s * 3600.0)

# Phase D原始CD
cd_pd = np.array([
    [sign_x * s0_over_s_3600 * ct / cos_dec, -sign_x * s0_over_s_3600 * st / cos_dec],
    [-sign_y * s0_over_s_3600 * st, -sign_y * s0_over_s_3600 * ct]
])
crval_pd = np.array([
    cra0 - tx / (cos_dec * 3600.0),
    cdec0 - ty / 3600.0
])
crpix = np.array([w / 2.0, h / 2.0])

print(f"\nPhase D CD: {cd_pd}")
print(f"Phase D CRVAL: {crval_pd}")

# 读取Siril修正后的CD
with open(os.path.join(PROJECT_ROOT, "vm35_wcs_output.json"), "r", encoding="utf-8") as f:
    d = json.load(f)
cd_siril = np.array(d["CD"]).reshape(2, 2)
crval_siril = np.array(d["CRVAL"])

print(f"\nSiril CD: {cd_siril}")
print(f"Siril CRVAL: {crval_siril}")

# CD差异
cd_diff = cd_siril - cd_pd
crval_diff = crval_siril - crval_pd
print(f"\nCD差异: {cd_diff}")
print(f"CD差异相对: {cd_diff / (np.abs(cd_pd) + 1e-30) * 100}%")
print(f"CRVAL差异: dra={3600*crval_diff[0]*cos_dec:.3f}\", ddec={3600*crval_diff[1]:.3f}\"")

# 用两种CD投影Gaia星，对比残差
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov = np.sqrt(w * w + h * h) * s0 / 3600
_, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
    gaia, crval_pd[0], crval_pd[1], max(0.8, fov * 1.2), 200)
gaia.close()

# 取前50亮星
mag_a = np.array(cat_mag, np.float64)
idx = np.argsort(mag_a)[:50]
ra_src = np.array(cat_ra, np.float64)[idx]
dec_src = np.array(cat_dec, np.float64)[idx]

# Phase D投影
cdet_pd = cd_pd[0, 0] * cd_pd[1, 1] - cd_pd[0, 1] * cd_pd[1, 0]
cd_inv_pd = np.array([[cd_pd[1, 1], -cd_pd[0, 1]], [-cd_pd[1, 0], cd_pd[0, 0]]]) / cdet_pd

dra = ra_src - crval_pd[0]
ddec = dec_src - crval_pd[1]
xi_pd = cd_inv_pd[0, 0] * dra + cd_inv_pd[0, 1] * ddec
eta_pd = cd_inv_pd[1, 0] * dra + cd_inv_pd[1, 1] * ddec
x_pd = xi_pd + crpix[0]
y_pd = eta_pd + crpix[1]

# Siril投影
cdet_s = cd_siril[0, 0] * cd_siril[1, 1] - cd_siril[0, 1] * cd_siril[1, 0]
cd_inv_s = np.array([[cd_siril[1, 1], -cd_siril[0, 1]], [-cd_siril[1, 0], cd_siril[0, 0]]]) / cdet_s

dra_s = ra_src - crval_siril[0]
ddec_s = dec_src - crval_siril[1]
xi_s = cd_inv_s[0, 0] * dra_s + cd_inv_s[0, 1] * ddec_s
eta_s = cd_inv_s[1, 0] * dra_s + cd_inv_s[1, 1] * ddec_s
x_s = xi_s + crpix[0]
y_s = eta_s + crpix[1]

# 检测星
from star_detector import StarDetector, SDetParamsPy
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
x_det = np.array(det.x, np.float64)
y_det = np.array(det.y, np.float64)

from scipy.spatial import cKDTree
tree = cKDTree(np.column_stack([x_det, y_det]))

# 匹配
in_frame_pd = (x_pd > 0) & (x_pd < w) & (y_pd > 0) & (y_pd < h)
in_frame_s = (x_s > 0) & (x_s < w) & (y_s > 0) & (y_s < h)

dists_pd, _ = tree.query(np.column_stack([x_pd[in_frame_pd], y_pd[in_frame_pd]]), k=1)
dists_s, _ = tree.query(np.column_stack([x_s[in_frame_s], y_s[in_frame_s]]), k=1)

matched_pd = dists_pd < 3.0
matched_s = dists_s < 3.0

print(f"\nPhase D线性投影: n={matched_pd.sum()}, RMS={np.sqrt(np.mean(dists_pd[matched_pd]**2)):.3f}px")
print(f"Siril线性投影: n={matched_s.sum()}, RMS={np.sqrt(np.mean(dists_s[matched_s]**2)):.3f}px")

# 关键: 检查Phase D投影位置是否合理
print(f"\nPhase D投影前5颗亮星:")
for i in range(min(5, in_frame_pd.sum())):
    print(f"  星{i}: x={x_pd[i]:.1f}, y={y_pd[i]:.1f}, mag={mag_a[idx[i]]:.1f}")

"""快速诊断CD矩阵精度"""
import json, numpy as np, sys, os

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit

reader = ImageReader()
img = reader.read(os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"))
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2
print(f"s0={s0:.4f}, center=({cra0:.6f}, {cdec0:.6f})")

with open(os.path.join(PROJECT_ROOT, "vm35_wcs_output.json"), "r", encoding="utf-8") as f:
    d = json.load(f)
CD = np.array(d["CD"]).reshape(2, 2)
CRVAL = np.array(d["CRVAL"])
CRPIX = np.array(d["CRPIX"])
print(f"CRVAL=({CRVAL[0]:.10f}, {CRVAL[1]:.10f})")
dra_off = 3600 * (CRVAL[0] - cra0) * np.cos(np.radians(CRVAL[1]))
ddec_off = 3600 * (CRVAL[1] - cdec0)
print(f"CRVAL偏移: dra={dra_off:.3f}arcsec, ddec={ddec_off:.3f}arcsec")

# CD正投影: 像素→天球
points = [(0, 0), (w, 0), (0, h), (w, h), (w / 2, h / 2)]
for x, y in points:
    xi = x - CRPIX[0]
    eta = y - CRPIX[1]
    dra = CD[0, 0] * xi + CD[0, 1] * eta
    ddec = CD[1, 0] * xi + CD[1, 1] * eta
    ra = CRVAL[0] + dra
    dec = CRVAL[1] + ddec
    r = np.sqrt(xi ** 2 + eta ** 2)
    print(f"  ({x:.0f},{y:.0f}) r={r:.0f}px -> RA={ra:.6f}, Dec={dec:.6f}")

# CD逆投影: 天球→像素，验证中心区域
cdet = CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0]
CD_inv = np.array([[CD[1, 1], -CD[0, 1]], [-CD[1, 0], CD[0, 0]]]) / cdet

# 用Gaia亮星验证
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov = np.sqrt(w * w + h * h) * s0 / 3600
_, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
    gaia, CRVAL[0], CRVAL[1], max(0.8, fov * 1.2), 100)
gaia.close()

print(f"\nGaia前100亮星线性投影残差:")
residuals = []
for i in range(M):
    dra = cat_ra[i] - CRVAL[0]
    ddec = cat_dec[i] - CRVAL[1]
    xi = CD_inv[0, 0] * dra + CD_inv[0, 1] * ddec
    eta = CD_inv[1, 0] * dra + CD_inv[1, 1] * ddec
    x_pix = xi + CRPIX[0]
    y_pix = eta + CRPIX[1]
    r = np.sqrt(xi ** 2 + eta ** 2)
    # 正投影回去验证
    dra_back = CD[0, 0] * xi + CD[0, 1] * eta
    ddec_back = CD[1, 0] * xi + CD[1, 1] * eta
    ra_back = CRVAL[0] + dra_back
    dec_back = CRVAL[1] + ddec_back
    err_ra = (ra_back - cat_ra[i]) * 3600 * np.cos(np.radians(cat_dec[i]))
    err_dec = (dec_back - cat_dec[i]) * 3600
    err = np.sqrt(err_ra ** 2 + err_dec ** 2)
    residuals.append((r, err, x_pix, y_pix))

# 按距离分组
for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2500)]:
    group = [(r, e) for r, e, _, _ in residuals if r_lo <= r < r_hi]
    if group:
        errs = [e for _, e in group]
        print(f"  r=[{r_lo},{r_hi}): n={len(group)}, 往返误差RMS={np.sqrt(np.mean(np.array(errs)**2)):.6f}arcsec, "
              f"均值={np.mean(errs):.6f}arcsec")

# 关键: 检查CD矩阵的往返一致性
print(f"\nCD矩阵往返一致性(应该为0):")
for i in range(min(5, M)):
    dra = cat_ra[i] - CRVAL[0]
    ddec = cat_dec[i] - CRVAL[1]
    xi = CD_inv[0, 0] * dra + CD_inv[0, 1] * ddec
    eta = CD_inv[1, 0] * dra + CD_inv[1, 1] * ddec
    dra_back = CD[0, 0] * xi + CD[0, 1] * eta
    ddec_back = CD[1, 0] * xi + CD[1, 1] * eta
    print(f"  星{i}: dra_err={3600*(dra_back-dra)*np.cos(np.radians(CRVAL[1])):.6f}\", "
          f"ddec_err={3600*(ddec_back-ddec):.6f}\"")

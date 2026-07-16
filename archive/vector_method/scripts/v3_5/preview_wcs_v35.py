"""V3.5 WCS投影测试 — 标准WCS-SIP逆投影渲染Gaia前1000亮星"""
import os, sys, math, json, numpy as np, logging
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp as VM35
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

reader = ImageReader(); detector = StarDetector(params=SDetParamsPy(fitRadius=0))

fname = "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", fname)
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen; ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1; cdec0 = img.metadata.wcs.crval2
s0 = 206.265*ps/fl

det = detector.detect_ex(img.data)
wcs_json = os.path.join(PROJECT_ROOT, "vm35_wcs_output.json")
vm = VM35(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
result = vm.solve(
    np.array(det.x, np.float64), np.array(det.y, np.float64),
    np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
    cra0, cdec0, fl, ps, w, h, wcs_out=wcs_json)
if not result: print("FAIL"); vm.close(); exit()
vm.close()

print(f"V3.5: D-clean={result.n_phased_clean} expand_mutual={result.n_expand_mutual} "
      f"expand_filtered={result.n_expand_after_filter} sip_total={result.n_sip_total} "
      f"sip_order={result.sip_order} SIP-RMS={result.sip_rms_px:.3f}px")
print(f"  s={result.solve_s:.6f} θ={result.rotation_deg:.2f}° flip={result.flip_mode}")

# 从JSON读取WCS参数
with open(wcs_json, 'r', encoding='utf-8') as f:
    wcs = json.load(f)
cd = np.array(wcs['CD'], dtype=np.float64)
crval = np.array(wcs['CRVAL'], dtype=np.float64)
crpix = np.array(wcs['CRPIX'], dtype=np.float64)
sip_A = np.array(wcs['SIP_A'], dtype=np.float64).reshape(6, 6)
sip_B = np.array(wcs['SIP_B'], dtype=np.float64).reshape(6, 6)

print(f"  CD={cd.tolist()}")
print(f"  CRVAL={crval}  CRPIX={crpix}")

# 查询Gaia前1000亮星
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov_deg = math.sqrt(w*w+h*h)*s0/3600.0
_, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
    gaia, crval[0], crval[1], max(0.8, fov_deg*1.2), 1200)
mag_a = np.array(cat_mag, np.float64); idx = np.argsort(mag_a)[:1000]
ra_src = np.array(cat_ra, np.float64)[idx]
dec_src = np.array(cat_dec, np.float64)[idx]
gaia.close()

# 标准WCS-SIP逆投影
cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
cd_inv = np.array([[cd[1,1], -cd[0,1]], [-cd[1,0], cd[0,0]]]) / cdet
dra = ra_src - crval[0]
ddec = dec_src - crval[1]
xi_prime = cd_inv[0,0]*dra + cd_inv[0,1]*ddec
eta_prime = cd_inv[1,0]*dra + cd_inv[1,1]*ddec

# 粗筛帧内星点
x_lin = xi_prime + crpix[0]
y_lin = eta_prime + crpix[1]
margin = 300
prelim = (x_lin > -margin) & (x_lin < w+margin) & (y_lin > -margin) & (y_lin < h+margin)

xi = xi_prime[prelim].copy()
eta = eta_prime[prelim].copy()
xi_prime_s = xi_prime[prelim].copy()
eta_prime_s = eta_prime[prelim].copy()

# 预计算非零SIP项
sip_terms = []
for p in range(7):
    for q in range(7):
        if p + q < 2 or p + q > 6: continue
        if p >= 6 or q >= 6: continue
        a_c = sip_A[p, q]
        b_c = sip_B[p, q]
        if abs(a_c) > 1e-30 or abs(b_c) > 1e-30:
            sip_terms.append((p, q, a_c, b_c))

for iteration in range(20):
    sip_dx = np.zeros_like(xi)
    sip_dy = np.zeros_like(eta)
    for p, q, a_c, b_c in sip_terms:
        term = xi**p * eta**q
        term = np.where(np.isfinite(term), term, 0.0)
        sip_dx += a_c * term
        sip_dy += b_c * term
    xi_new = xi_prime_s - sip_dx
    eta_new = eta_prime_s - sip_dy
    if np.max(np.abs(xi_new - xi)) < 1e-6 and np.max(np.abs(eta_new - eta)) < 1e-6:
        break
    xi = xi_new
    eta = eta_new

x_pix = np.full(len(ra_src), np.nan)
y_pix = np.full(len(ra_src), np.nan)
x_pix[prelim] = xi + crpix[0]
y_pix[prelim] = eta + crpix[1]

bt = np.isfinite(x_pix) & (x_pix>0) & (x_pix<w) & (y_pix>0) & (y_pix<h)
print(f"投影: {bt.sum()}颗在帧内 (共{len(ra_src)}颗)")

# 渲染
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = img.data.astype(np.float32); dd = data[data>0]
lo, hi = np.percentile(dd, (1, 99.5)) if len(dd)>1 else (0, 1)
img_s = np.clip((data-lo)/max(hi-lo,1), 0, 1)
DPI = 100
fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")
ax.scatter(x_pix[bt], y_pix[bt], marker="+", color="#FF0000", s=80, linewidths=2, alpha=0.9)
ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
out = os.path.join(PROJECT_ROOT, "preview_wcs_v35.png")
fig.savefig(out, dpi=DPI, pad_inches=0)
plt.close(fig)
print(f"Saved: {out}")

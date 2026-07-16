"""Single frame: full solve → project Gaia → red crosses"""
import os, sys, math, time, numpy as np, logging
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

logging.basicConfig(level=logging.WARNING)
from astro_image_io import ImageReader
from vector_match_v3_3_cpp import VectorMatch as VM33
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

DEGTORAD = math.pi / 180.0

reader = ImageReader()
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
vm = VM33(gaia_dir, db_type=1)

fname = "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", fname)

img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
cra_o = img.metadata.wcs.crval1
cdec_o = img.metadata.wcs.crval2
s0_calc = 206.265 * ps / fl

det = detector.detect_ex(img.data)
sx = np.array(det.x, dtype=np.float64); sy = np.array(det.y, dtype=np.float64)
sf = np.array(det.flux, dtype=np.float64); ss = np.array(det.saturated, dtype=np.int32)

t0 = time.time()
result = vm.solve(sx, sy, sf, ss, cra_o, cdec_o, fl, ps, w, h)
dt = time.time() - t0

if not result:
    print("FAIL"); vm.close(); exit()

tx = getattr(result, "solve_tx", 0.0)
ty = getattr(result, "solve_ty", 0.0)
s = getattr(result, "solve_s", 1.0)
s0 = getattr(result, "s0", s0_calc)
rot = result.rotation_deg; flip = result.flip_mode
cra_s, cdec_s = result.center_ra, result.center_dec
theta = rot * DEGTORAD; ct, st = math.cos(theta), math.sin(theta)

print(f"OK: mode={flip} n={result.matched_count} rms={result.rms_px:.3f}px {dt:.2f}s")
print(f"  tx={tx:.3f}\" ty={ty:.3f}\" s={s:.6f} s0={s0:.4f} rot={rot:.3f}°")
print(f"  center: {cra_s:.6f} {cdec_s:.6f}  (orig: {cra_o:.6f} {cdec_o:.6f})")

# Gaia catalog
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov_diag = math.sqrt(w*w+h*h) * s0 / 3600.0
radius = max(0.8, fov_diag)
mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, cra_s, cdec_s, radius, 600)
print(f"  Gaia: {M} stars, mag_limit={mag_limit:.1f}")

ra_a = np.array(cat_ra, dtype=np.float64); dec_a = np.array(cat_dec, dtype=np.float64)
mag_a = np.array(cat_mag, dtype=np.float64)
idx = np.argsort(mag_a)[:1000]; ra_a, dec_a = ra_a[idx], dec_a[idx]

# Project Gaia → pixel
cos_d = math.cos(cdec_o * DEGTORAD)
dx = -(ra_a - cra_o) * cos_d * 3600.0
dy = (dec_a - cdec_o) * 3600.0
if flip == 1: dx, dy = -dx, dy
elif flip == 2: dx, dy = dx, -dy
elif flip == 3: dx, dy = -dx, -dy
ux = s * ct * dx - s * st * dy + tx
uy = s * st * dx + s * ct * dy + ty
px = ux / s0 + w/2.0; py = uy / s0 + h/2.0
mk = (px > 0) & (px < w) & (py > 0) & (py < h)
print(f"  Crosses in frame: {mk.sum()}/{len(ra_a)}")

# Render
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = img.data.astype(np.float32); dd = data[data > 0]
lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 0 else (0, 1)
img_s = np.clip((data - lo) / max(hi - lo, 1), 0, 1)

DPI = 100
fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")
ax.scatter(px[mk], py[mk], marker="x", color="#FF0000",
           s=12, linewidths=0.6, alpha=0.85)
ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
out = os.path.join(PROJECT_ROOT, "preview_single.png")
fig.savefig(out, dpi=DPI, pad_inches=0)
plt.close(fig)
print(f"Saved: {out}")

vm.close(); gaia.close()

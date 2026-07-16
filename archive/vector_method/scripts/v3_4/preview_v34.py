"""V3.4 single-frame debug: U/Wt overlay + SIP fit quality"""
import os, sys, math, time, numpy as np, logging
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

logging.basicConfig(level=logging.WARNING)
from astro_image_io import ImageReader
from vector_match_v3_4_cpp import VectorMatchV34Cpp as VM34
from vector_match_v2 import (_build_catalog_vectors, _apply_flip, _apply_similarity)
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree

DEGTORAD = math.pi / 180.0
reader = ImageReader()
detector = StarDetector(params=SDetParamsPy(fitRadius=0))

fname = "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", fname)

img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen; ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1; cdec0 = img.metadata.wcs.crval2
s0 = 206.265 * ps / fl; cx, cy = w / 2.0, h / 2.0

det = detector.detect_ex(img.data)
sx = np.array(det.x, dtype=np.float64); sy = np.array(det.y, dtype=np.float64)
sf = np.array(det.flux, dtype=np.float64); ss = np.array(det.saturated, dtype=np.int32)

vm = VM34(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
t0 = time.time()
result = vm.solve(sx, sy, sf, ss, cra0, cdec0, fl, ps, w, h)
dt = time.time() - t0

if not result:
    print("FAIL"); vm.close(); exit()

tx = getattr(result, "solve_tx", 0); ty = getattr(result, "solve_ty", 0)
s_sol = getattr(result, "solve_s", 1); rot = result.rotation_deg; flip = result.flip_mode
theta = rot * DEGTORAD; ct, st = math.cos(theta), math.sin(theta)

print(f"V3.4 OK: {dt:.2f}s, n={result.matched_count}, rms={result.rms_px:.3f}px")
print(f"  AB-SNR={getattr(result,'theta_snr',0):.0f}x, C-exp={getattr(result,'n_phasec_expanded',0)}, D-clean={getattr(result,'n_phased_clean',0)}, D-iter={getattr(result,'n_phased_iterations',0)}")
sip_rms = getattr(result, 'sip_rms_px', 0)
print(f"  SIP RMS: {sip_rms:.3f}px")

# Build vectors
sat_mask = ss.astype(bool)
sat_x, sat_y = sx[sat_mask], sy[sat_mask]
U_x = (sat_x - cx) * s0; U_y = -(sat_y - cy) * s0
U = np.column_stack([U_x, U_y])

from vector_match_v2 import GaiaClientPy, bisection_mag_limit
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov_diag = math.sqrt(w*w+h*h)*s0/3600.0
radius = max(0.8, fov_diag)
mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, cra0, cdec0, radius, 500)
W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)
Wf = _apply_flip(W, flip)
Wt = _apply_similarity(Wf, s_sol, theta, tx, ty)

tree = cKDTree(Wt)
dists, idxs = tree.query(U, k=1)
ok = dists < 5.0 * s0
rms_asec = float(np.sqrt(np.mean(dists[ok]**2))) if ok.sum()>0 else 0
rms_px = rms_asec/s0
print(f"  Projection: {ok.sum()}/{U.shape[0]} pairs, RMS={rms_px:.3f}px ({rms_asec:.3f}\")")
print(f"  dist: med={np.median(dists[ok]):.2f}\" mean={np.mean(dists[ok]):.2f}\"")

# --- SIP prediction ---
if hasattr(result, 'cd') and np.any(result.cd):
    cd = result.cd
    sip_A = getattr(result, 'sip_A', np.zeros((6,6)))
    sip_B = getattr(result, 'sip_B', np.zeros((6,6)))
    print(f"  CD: [[{cd[0,0]:.6e}, {cd[0,1]:.6e}], [{cd[1,0]:.6e}, {cd[1,1]:.6e}]]")
    # Apply SIP to predict catalog pixel from detection pixel
    cat_x_pred = []; cat_y_pred = []
    cat_x_true = []; cat_y_true = []
    for k in range(U.shape[0]):
        if not ok[k]: continue
        det_px = sat_x[k]; det_py = sat_y[k]
        u = det_px; v = det_py
        du = 0; dv = 0
        for p in range(6):
            for q in range(6):
                if p+q < 2: continue
                du += sip_A[p,q] * (u**p) * (v**q)
                dv += sip_B[p,q] * (u**p) * (v**q)
        cat_x_pred.append(u + du)
        cat_y_pred.append(v + dv)
        # True catalog pixel
        j = idxs[k]
        cat_x_true.append(Wt[j,0]/s0 + cx)
        cat_y_true.append(-Wt[j,1]/s0 + cy)
    errs = np.sqrt((np.array(cat_x_pred)-np.array(cat_x_true))**2 + (np.array(cat_y_pred)-np.array(cat_y_true))**2)
    print(f"  SIP prediction RMS: {np.mean(errs):.3f}px")

# --- Render ---
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = img.data.astype(np.float32); dd = data[data>0]
lo, hi = np.percentile(dd, (1, 99.5)) if len(dd)>1 else (0,1)
img_s = np.clip((data-lo)/max(hi-lo,1), 0, 1)

def as2px(ax, ay): return ax/s0+cx, -ay/s0+cy

Wt_px, Wt_py = as2px(Wt[:,0], Wt[:,1])
gin = (Wt_px>0)&(Wt_px<w)&(Wt_py>0)&(Wt_py<h)
mp_x, mp_y = as2px(Wt[idxs,0], Wt[idxs,1])

DPI = 100
fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI, frameon=False)
ax = fig.add_axes([0,0,1,1])
ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")

for k in range(U.shape[0]):
    if not ok[k]: continue
    ax.plot([sat_x[k], mp_x[k]], [sat_y[k], mp_y[k]],
            color="#FFFF00", linewidth=0.3, alpha=0.5)

ax.scatter(sat_x, sat_y, marker="o", facecolors="none",
           edgecolors="#00FF00", s=20, linewidths=0.4, alpha=0.5)
ax.scatter(sat_x[ok], sat_y[ok], marker="o", facecolors="none",
           edgecolors="#00FF00", s=35, linewidths=0.7, alpha=0.85)
ax.scatter(Wt_px[gin], Wt_py[gin], marker="x", color="#FF0000",
           s=10, linewidths=0.4, alpha=0.5)
ax.scatter(mp_x[ok], mp_y[ok], marker="x", color="#FF0000",
           s=28, linewidths=1.0, alpha=0.9)

ax.text(10, h-30,
        f"RMS={rms_px:.2f}px SIP={sip_rms:.2f}px n={ok.sum()}",
        color="cyan", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

ax.set_xlim(0,w); ax.set_ylim(0,h); ax.axis("off")
out = os.path.join(PROJECT_ROOT, "preview_v34.png")
fig.savefig(out, dpi=DPI, pad_inches=0)
plt.close(fig)
print(f"Saved: {out}")
vm.close(); gaia.close()

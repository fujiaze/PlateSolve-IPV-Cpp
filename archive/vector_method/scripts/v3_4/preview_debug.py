"""Debug: U/Wt pairs. Circles=detected, crosses=Gaia. 1-to-1 + Umeyama refit. RMS annotated."""
import os, sys, math, time, numpy as np, logging
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

logging.basicConfig(level=logging.WARNING)
from astro_image_io import ImageReader
from vector_match_v3_3_cpp import VectorMatch as VM33
from vector_match_v2 import (GaiaClientPy, bisection_mag_limit,
    _build_catalog_vectors, _apply_flip, _apply_similarity, _umeyama)
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree

DEGTORAD = math.pi / 180.0
reader = ImageReader()
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
vm = VM33(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)

fname = "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", fname)

img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1; cdec0 = img.metadata.wcs.crval2
s0 = 206.265 * ps / fl; cx, cy = w / 2.0, h / 2.0

det = detector.detect_ex(img.data)
sx = np.array(det.x, dtype=np.float64); sy = np.array(det.y, dtype=np.float64)
sf = np.array(det.flux, dtype=np.float64); ss = np.array(det.saturated, dtype=np.int32)

result = vm.solve(sx, sy, sf, ss, cra0, cdec0, fl, ps, w, h)
if not result: print("FAIL"); exit()

tx0 = getattr(result, "solve_tx", 0.0)
ty0 = getattr(result, "solve_ty", 0.0)
s0_sol = getattr(result, "solve_s", 1.0)
rot0 = result.rotation_deg; flip = result.flip_mode
th0 = rot0 * DEGTORAD; ct0, st0 = math.cos(th0), math.sin(th0)

print(f"Solver: s={s0_sol:.6f} rot={rot0:.3f}° tx={tx0:.3f}\" ty={ty0:.3f}\" "
      f"n={result.matched_count} rms={result.rms_px:.3f}px")

# --- Build U, W, Wf ---
sat_mask = ss.astype(bool)
sat_x, sat_y = sx[sat_mask], sy[sat_mask]
U_x = (sat_x - cx) * s0; U_y = -(sat_y - cy) * s0
U = np.column_stack([U_x, U_y])

gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov_diag = math.sqrt(w*w + h*h) * s0 / 3600.0
radius = max(0.8, fov_diag)
mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, cra0, cdec0, radius, 500)
W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)
Wf = _apply_flip(W, flip)

# --- Solver projection ---
Wt0 = _apply_similarity(Wf, s0_sol, th0, tx0, ty0)
tree0 = cKDTree(Wt0)
d0, i0 = tree0.query(U, k=1)
ok0 = d0 < 5.0 * s0
rms_solver = float(np.sqrt(np.mean(d0[ok0]**2))) / s0 if ok0.sum() > 0 else 0
print(f"  Solver projection: {ok0.sum()}/{U.shape[0]} NN pairs, RMS={rms_solver:.3f}px")

# --- 1-to-1 matching on solver Wt ---
pairs = [(d0[k], k, i0[k]) for k in range(U.shape[0]) if ok0[k]]
pairs.sort()
w_used = np.zeros(Wt0.shape[0], dtype=bool)
src_idx, dst_idx = [], []
for d, k, j in pairs:
    if not w_used[j]:
        w_used[j] = True; src_idx.append(j); dst_idx.append(k)

# --- Refit Umeyama: Wf[src_idx] → U[dst_idx] ---
src_pts = Wf[src_idx]; dst_pts = U[dst_idx]
sim = _umeyama(dst_pts, src_pts)
if sim is None: print("Umeyama fail"); exit()

s1, th1, tx1, ty1 = sim
rot1 = math.degrees(th1); ct1, st1 = math.cos(th1), math.sin(th1)
print(f"Refit: s={s1:.6f} rot={rot1:.3f}° tx={tx1:.3f}\" ty={ty1:.3f}\" from {len(src_idx)} pairs")

# --- Apply refit ---
Wt1 = _apply_similarity(Wf, s1, th1, tx1, ty1)
tree1 = cKDTree(Wt1)
d1, i1 = tree1.query(U, k=1)
ok1 = d1 < 5.0 * s0
rms_refit = float(np.sqrt(np.mean(d1[ok1]**2))) / s0 if ok1.sum() > 0 else 0
print(f"  Refit projection: {ok1.sum()}/{U.shape[0]} NN pairs, RMS={rms_refit:.3f}px")
print(f"  dist: med={np.median(d1[ok1]):.2f}\" mean={np.mean(d1[ok1]):.2f}\"")

# --- Pixel conversion ---
def as2px(ax, ay):
    return ax / s0 + cx, -ay / s0 + cy

Wt1_px, Wt1_py = as2px(Wt1[:, 0], Wt1[:, 1])
gin = (Wt1_px > 0) & (Wt1_px < w) & (Wt1_py > 0) & (Wt1_py < h)

mp_x, mp_y = as2px(Wt1[i1, 0], Wt1[i1, 1])

# --- Render ---
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
data = img.data.astype(np.float32); dd = data[data > 0]
lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 1 else (0, 1)
img_s = np.clip((data - lo) / max(hi - lo, 1), 0, 1)

DPI = 100
fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, frameon=False)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")

# Yellow lines
for k in range(U.shape[0]):
    if not ok1[k]: continue
    ax.plot([sat_x[k], mp_x[k]], [sat_y[k], mp_y[k]],
            color="#FFFF00", linewidth=0.3, alpha=0.5)

# Green circles: all saturated
ax.scatter(sat_x, sat_y, marker="o", facecolors="none",
           edgecolors="#00FF00", s=20, linewidths=0.4, alpha=0.5)
# Green bold: matched
ax.scatter(sat_x[ok1], sat_y[ok1], marker="o", facecolors="none",
           edgecolors="#00FF00", s=35, linewidths=0.7, alpha=0.85)

# Red crosses: all Gaia in frame
ax.scatter(Wt1_px[gin], Wt1_py[gin], marker="x", color="#FF0000",
           s=10, linewidths=0.4, alpha=0.5)
# Red bold: matched
ax.scatter(mp_x[ok1], mp_y[ok1], marker="x", color="#FF0000",
           s=28, linewidths=1.0, alpha=0.9)

ax.text(10, h - 30, f"solver RMS={rms_solver:.2f}px → refit RMS={rms_refit:.2f}px  n={ok1.sum()}",
        color="cyan", fontsize=8, ha="left", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5))

ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
out = os.path.join(PROJECT_ROOT, "preview_debug_matches.png")
fig.savefig(out, dpi=DPI, pad_inches=0)
plt.close(fig)
print(f"Saved: {out}")
vm.close(); gaia.close()

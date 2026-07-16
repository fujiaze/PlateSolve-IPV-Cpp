import os, sys, math, numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "gaia_xpsd_client", "python"))
from astro_image_io import ImageReader
from gaia_client import GaiaClient

DEGTORAD = math.pi/180.0
reader = ImageReader()
gaia = GaiaClient(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)

import csv
csv_path = os.path.join(PROJECT_ROOT, "v33_robustness_test_results.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

row = [r for r in rows if "M20_T2_flying_dutchman-20250701@073331" in r["filename"]][0]
cra, cdec = float(row["center_ra"]), float(row["center_dec"])
scale = float(row["scale_arcsec_px"])
rot = float(row["rotation_deg"])
flip = int(row["flip_mode"])
w, h = int(row["width"]), int(row["height"])

print(f"Center: {cra:.6f}, {cdec:.6f}")
print(f"Scale: {scale:.4f} as/px, Rot: {rot:.2f}deg, Flip: {flip}, Size: {w}x{h}")

stars = gaia.query_cone(cra, cdec, 1.5, mag_limit=16)
if not stars:
    print("No Gaia stars!")
    exit()

ra = np.array([s[0] for s in stars], dtype=np.float64)
dec = np.array([s[1] for s in stars], dtype=np.float64)
mag = np.array([s[2] for s in stars], dtype=np.float64)
idx = np.argsort(mag)[:200]
ra, dec, mag = ra[idx], dec[idx], mag[idx]

cos_d0 = math.cos(cdec * DEGTORAD)
dx = -(ra - cra) * cos_d0 * 3600.0
dy = (dec - cdec) * 3600.0
theta = rot * DEGTORAD
ct, st = math.cos(theta), math.sin(theta)
if flip == 1: dx = -dx
elif flip == 2: dy = -dy
elif flip == 3: dx = -dx; dy = -dy
px = (ct * dx - st * dy) / scale + w / 2.0
py = (st * dx + ct * dy) / scale + h / 2.0

in_frame = (px >= 0) & (px < w) & (py >= 0) & (py < h)
print(f"Stars in frame: {in_frame.sum()}/{len(px)}")
print(f"px: [{px[in_frame].min():.0f}, {px[in_frame].max():.0f}]")
print(f"py: [{py[in_frame].min():.0f}, {py[in_frame].max():.0f}]")
for m, x, y in zip(mag[in_frame][:10], px[in_frame][:10], py[in_frame][:10]):
    print(f"  mag={m:.1f} px=({x:.0f},{y:.0f})")

# test image
try:
    fits_path = r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
    img = reader.read(fits_path)
    data = img.data.astype(np.float32)
    lo, hi = np.percentile(data[data>0], (2, 98))
    img_s = np.clip((data - lo) / (hi - lo), 0, 1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w/100, h/100), dpi=100)
    ax.imshow(img_s, cmap="gray", origin="lower", extent=[0, w, 0, h])
    ax.scatter(px[in_frame], py[in_frame], marker="x", color="red", s=2, linewidths=0.3, alpha=0.7)
    ax.set_xlim(0, w); ax.set_ylim(0, h)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    out_path = os.path.join(PROJECT_ROOT, "preview_test.png")
    fig.savefig(out_path, dpi=100, pad_inches=0, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
except Exception as e:
    print(f"Error: {e}")

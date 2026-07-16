"""Batch preview overlay: read solved transforms from CSV, draw Gaia crosses on FITS"""
import os, sys, math, csv, numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))

from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit, gnomonic_forward, _RADTOASEC

DEGTORAD = math.pi / 180.0
OUT_DIR = os.path.join(PROJECT_ROOT, "preview_overlay")
os.makedirs(OUT_DIR, exist_ok=True)

reader = ImageReader()
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)

csv_path = os.path.join(PROJECT_ROOT, "v33_robustness_test_results.csv")
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
ok = [r for r in rows if r["success"] == "True"]
print(f"Processing {len(ok)} frames...")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

done = 0
for row in ok:
    fname = row["filename"]
    base = os.path.splitext(fname)[0]
    out_path = os.path.join(OUT_DIR, f"{base}.png")
    if os.path.exists(out_path):
        done += 1; continue

    try:
        tx = float(row["solve_tx"]); ty = float(row["solve_ty"])
        s = float(row["solve_s"])
        rot = float(row["rotation_deg"]); flip = int(row["flip_mode"])
        s_final = float(row["scale_arcsec_px"])
        w, h = int(row["width"]), int(row["height"])
        fl = float(row["focal_length_mm"]); ps = float(row["pixel_size_um"])
        s0 = float(row["s0"])
        cra_s = float(row["center_ra"]); cdec_s = float(row["center_dec"])
        cra_o = float(row.get("original_ra", cra_s))
        cdec_o = float(row.get("original_dec", cdec_s))

        theta = rot * DEGTORAD; ct, st = math.cos(theta), math.sin(theta)

        fov_diag = math.sqrt(w*w + h*h) * s0 / 3600.0
        radius = max(0.8, fov_diag * 1.0)
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            gaia, cra_s, cdec_s, radius, 500)
        if M < 3:
            done += 1; continue

        ra_a = np.array(cat_ra, dtype=np.float64)
        dec_a = np.array(cat_dec, dtype=np.float64)
        mag_a = np.array(cat_mag, dtype=np.float64)
        idx = np.argsort(mag_a)[:1000]
        ra_a, dec_a = ra_a[idx], dec_a[idx]

        # Project Gaia → exactly same formula as solver's affine
        cos_d = math.cos(cdec_o * DEGTORAD)
        dx = -(ra_a - cra_o) * cos_d * 3600.0
        dy = (dec_a - cdec_o) * 3600.0
        if flip == 1: dx, dy = -dx, dy
        elif flip == 2: dx, dy = dx, -dy
        elif flip == 3: dx, dy = -dx, -dy
        ux = s * ct * dx - s * st * dy + tx
        uy = s * st * dx + s * ct * dy + ty
        px = ux / s0 + w / 2.0
        py = uy / s0 + h / 2.0
        mask = (px > 0) & (px < w) & (py > 0) & (py < h)
        if mask.sum() < 5:
            done += 1; continue

        fits_path = None
        for d in ["testdata", "testdata/lights", "testdata/lights1"]:
            for root, dirs, files in os.walk(os.path.join(PROJECT_ROOT, d)):
                if fname in files:
                    fits_path = os.path.join(root, fname)
                    break
            if fits_path: break
        if not fits_path:
            done += 1; continue

        img = reader.read(fits_path)
        data = img.data.astype(np.float32)
        if len(data.shape) == 3: data = data[0]
        dd = data[data > 0]
        lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) else (0, 1)
        img_s = np.clip((data - lo) / max(hi - lo, 1), 0, 1)

        DPI = 100
        fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")
        ax.scatter(px[mask], py[mask], marker="x", color="#FF0000",
                   s=4, linewidths=0.3, alpha=0.7)
        ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
        fig.savefig(out_path, dpi=DPI, pad_inches=0)
        plt.close(fig)
        done += 1
        print(f"  [{done}/{len(ok)}] {fname} ({mask.sum()} stars)")
    except Exception as e:
        done += 1
        print(f"  [{done}/{len(ok)}] SKIP {fname}: {e}")

gaia.close()
print(f"\nDone: {done} → {OUT_DIR}")

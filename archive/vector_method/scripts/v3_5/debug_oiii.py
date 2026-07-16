import sys, os
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'astro_image_io', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'star_detector', 'python'))
import numpy as np
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

# GC_P1 Oiii - 检查饱和星和普通星的位置分布
fits_path = os.path.join(PROJECT_ROOT, 'testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts')
reader = ImageReader()
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265*ps/fl

detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
img_x = np.array(det.x, np.float64)
img_y = np.array(det.y, np.float64)
img_flux = np.array(det.flux, np.float64)
img_sat = np.array(det.saturated, np.int32)

nsat = int(img_sat.sum())
print(f'nsat={nsat}, total_stars={len(img_x)}')

mask_sat = img_sat.astype(bool)
sat_idx = np.where(mask_sat)[0]
normal_idx = np.where(~mask_sat)[0]
sorted_normal = normal_idx[np.argsort(-img_flux[normal_idx])]
n_normal = 100 - nsat
top_normal = sorted_normal[:n_normal]
sel_idx = np.concatenate([sat_idx, top_normal])

cx, cy = w/2.0, h/2.0
ux = (img_x[sel_idx] - cx) * s0
uy = -(img_y[sel_idx] - cy) * s0
norm_U = np.sqrt(ux**2 + uy**2)

sat_norm = norm_U[:nsat]
normal_norm = norm_U[nsat:]
print(f'饱和星 norm_U: min={sat_norm.min():.1f} max={sat_norm.max():.1f} median={np.median(sat_norm):.1f}')
print(f'普通星 norm_U: min={normal_norm.min():.1f} max={normal_norm.max():.1f} median={np.median(normal_norm):.1f}')
print(f'全部 norm_U: min={norm_U.min():.1f} max={norm_U.max():.1f} median={np.median(norm_U):.1f}')

fov_w = w * s0 / 3600
fov_h = h * s0 / 3600
fov_diag = np.sqrt(fov_w**2 + fov_h**2)
print(f'FOV: {fov_w:.2f}deg x {fov_h:.2f}deg, diagonal={fov_diag:.2f}deg')

range_ratio = (norm_U.max() - norm_U.min()) / max(np.median(norm_U), 1e-10)
print(f'norm_U范围比: {range_ratio*100:.1f}%')

"""验证: 用solver参数直接变换Wf, 检查n_in_range是否合理"""
import os, sys, math
import numpy as np
from scipy.spatial import cKDTree

_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(_MINGW_BIN)
    except: pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy, gnomonic_forward
from v4_3.vector_match_v4_3_cpp import V43Solver

TESTDATA = os.path.join(PROJECT_ROOT, "testdata")

def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m/60.0 + sec/3600.0) * 15.0
    return float(s)

def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith("-"): sign = -1.0; s = s[1:]
    elif s.startswith("+"): s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m/60.0 + sec/3600.0)
    return float(s)

def find_fits_path(filename):
    for dirpath, _, filenames in os.walk(TESTDATA):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None

# 成功帧
fn = "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts"
fits_path = find_fits_path(fn)
reader = ImageReader()
img = reader.read(fits_path)
w, h = img.width, img.height
pixels = img.to_numpy()
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl
print(f"图像: {w}x{h}, s0={s0:.4f} arcsec/px")

kws = img.keywords
kw_dict = {k.name.upper(): k.value for k in kws}
cra0 = _parse_ra_hms(kw_dict.get("OBJCTRA") or kw_dict.get("RA"))
cdec0 = _parse_dec_dms(kw_dict.get("OBJCTDEC") or kw_dict.get("DEC"))

# 星点检测
sd = StarDetector(params=SDetParamsPy(fitRadius=0))
pixels_u16 = np.clip(pixels, 0, 65535).astype(np.uint16) if pixels.dtype != np.uint16 else pixels
det = sd.detect_ex(pixels_u16)
all_x = det.x
all_y = det.y
all_sat = det.saturated
n_total = len(all_x)
sat_indices = [i for i in range(n_total) if all_sat[i]]
u_indices = list(sat_indices[:])
N = len(u_indices)

# U向量
cx, cy = w / 2.0, h / 2.0
U = np.zeros((N, 2), dtype=np.float64)
for k, idx in enumerate(u_indices):
    U[k, 0] = (all_x[idx] - cx) * s0
    U[k, 1] = -(all_y[idx] - cy) * s0

# Gaia
gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
gaia = GaiaClientPy(gaia_dir, db_type=1)
fov_diag = math.sqrt((w*s0)**2 + (h*s0)**2)
radius = (fov_diag * 0.5) / 3600.0 * 1.2
cat_ra, cat_dec, cat_mag = gaia.cone_search(cra0, cdec0, radius, 18.0)

xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, cra0, cdec0)
fov_half_w = w / 2.0 * s0
fov_half_h = h / 2.0 * s0
fov_mask = valid & (np.abs(xi) < fov_half_w) & (np.abs(eta) < fov_half_h)
xi_f = xi[fov_mask]
eta_f = eta[fov_mask]
mag_f = cat_mag[fov_mask]
sort_idx = np.argsort(mag_f)
M_max = min(len(sort_idx), 500)
W = np.zeros((M_max, 2), dtype=np.float64)
for k in range(M_max):
    W[k, 0] = xi_f[sort_idx[k]]
    W[k, 1] = eta_f[sort_idx[k]]
print(f"U={N}, W={M_max}")

# Solver结果
solver = V43Solver(gaia_client=gaia, star_detector=sd)
import tempfile
log_dir = tempfile.mkdtemp()
result = solver.solve(image_path=fits_path, ra=cra0, dec=cdec0,
                      focal_length_mm=fl, pixel_size_um=ps, log_dir=log_dir)
sv_mode = result.get("flip_mode", -1)
sv_theta = result.get("rotation_deg", 0)
sv_s = result.get("scale", 1.0)
sv_tx = result.get("tx", 0.0)
sv_ty = result.get("ty", 0.0)
sv_matched = result.get("matched_count", 0)
print(f"Solver: mode={sv_mode} theta={sv_theta:.2f} s={sv_s:.4f} tx={sv_tx:.1f} ty={sv_ty:.1f} matched={sv_matched}")

# 用solver参数变换Wf, 验证n_in_range
fx = (sv_mode == 1 or sv_mode == 3)
fy = (sv_mode == 2 or sv_mode == 3)
Wf = np.empty_like(W)
Wf[:, 0] = -W[:, 0] if fx else W[:, 0]
Wf[:, 1] = -W[:, 1] if fy else W[:, 1]

theta_rad = sv_theta * math.pi / 180.0
ct, st = math.cos(theta_rad), math.sin(theta_rad)
match_dist = 5.0 * s0
print(f"\nmatch_dist = {match_dist:.1f} arcsec ({match_dist/s0:.1f} px)")

# 变换Wf -> Wt
Wt = np.empty_like(Wf)
Wt[:, 0] = sv_s * (ct * Wf[:, 0] - st * Wf[:, 1]) + sv_tx
Wt[:, 1] = sv_s * (st * Wf[:, 0] + ct * Wf[:, 1]) + sv_ty

# 统计 n_in_range
tree = cKDTree(Wt)
dist, nn_idx = tree.query(U, k=1)
n_match = int(np.sum(dist < match_dist))
print(f"用solver参数变换: n_in_range(dist<{match_dist:.0f}\") = {n_match}")

# 看不同匹配半径下的 n_in_range
for md in [1, 2, 3, 5, 10, 20, 50, 100]:
    md_asec = md * s0
    n = int(np.sum(dist < md_asec))
    print(f"  dist<{md_asec:.1f}\" ({md}px): n={n}")

# 看U到Wt的最近距离分布
print(f"\nU到Wt最近距离统计 (arcsec):")
print(f"  min={dist.min():.2f}  med={np.median(dist):.2f}  mean={dist.mean():.2f}  max={dist.max():.2f}")
print(f"  <1\": {np.sum(dist<1)}  <3\": {np.sum(dist<3)}  <5\": {np.sum(dist<5)}  <10\": {np.sum(dist<10)}  <30\": {np.sum(dist<30)}")

solver.close()
gaia.close()
sd.close()

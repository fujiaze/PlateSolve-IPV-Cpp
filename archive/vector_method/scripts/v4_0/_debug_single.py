"""Debug单帧: NGC4945 Lum (失败帧, Dec=-49.6)"""
import os, sys, math, json
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp
from vector_match_v2 import GaiaClientPy, gnomonic_forward, _build_catalog_vectors

# NGC4945 Lum 帧 (Dec=-49.6, 失败)
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", "T3", "lum",
                         "NGC4945_FD_T3_flying_dutchman-20250205@074722-600S-Lum.fts")

print(f"=== Debug: {os.path.basename(fits_path)} ===")

reader = ImageReader()
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2
s0 = 206.265 * ps / fl
print(f"图像: {w}x{h} fl={fl}mm ps={ps}um s0={s0:.4f}\"/px")
print(f"FITS中心: RA={cra0:.6f}° Dec={cdec0:.6f}°  cos(Dec)={math.cos(cdec0*math.pi/180):.4f}")

# 星点检测
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
print(f"检测: {len(det.x)}颗 饱和{int(np.sum(det.saturated))}颗")
detector = None

# Gaia查询
gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
fov_deg = math.sqrt(w*w + h*h) * s0 / 3600.0
query_r = max(fov_deg * 0.7, 1.0)
ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, query_r, 22.0)
print(f"Gaia查询: {len(ra_t)}颗 (r={query_r:.3f}° mag<22)")

# gnomonic投影筛选FOV内
xi, eta, valid = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
fov_half_w = w/2 * s0
fov_half_h = h/2 * s0
in_fov = valid & (np.abs(xi) < fov_half_w) & (np.abs(eta) < fov_half_h)
print(f"FOV内Gaia星: {int(np.sum(in_fov))}颗 (FOV半宽={fov_half_w:.0f}\" 半高={fov_half_h:.0f}\")")

# 检查投影坐标范围
if np.any(in_fov):
    print(f"  ξ范围: [{xi[in_fov].min():.1f}, {xi[in_fov].max():.1f}]\"")
    print(f"  η范围: [{eta[in_fov].min():.1f}, {eta[in_fov].max():.1f}]\"")

# U向量
cx, cy = w/2.0, h/2.0
det_x = np.asarray(det.x, dtype=np.float64)
det_y = np.asarray(det.y, dtype=np.float64)
ux = (det_x - cx) * s0
uy = -(det_y - cy) * s0
print(f"\nU向量范围: ux=[{ux.min():.0f}, {ux.max():.0f}]\" uy=[{uy.min():.0f}, {uy.max():.0f}]\"")

# W向量(FOV内最亮250颗)
mag_fov = mag_t[in_fov]
ra_fov = ra_t[in_fov]
dec_fov = dec_t[in_fov]
order = np.argsort(mag_fov)[:250]
W = _build_catalog_vectors(ra_fov[order], dec_fov[order], cra0, cdec0)
print(f"W向量范围(250亮星): Wx=[{W[:,0].min():.0f}, {W[:,0].max():.0f}]\" Wy=[{W[:,1].min():.0f}, {W[:,1].max():.0f}]\"")

# 检查U和W的尺度是否匹配
print(f"\nU模长: 中位={np.median(np.sqrt(ux**2+uy**2)):.0f}\" max={np.max(np.sqrt(ux**2+uy**2)):.0f}\"")
print(f"W模长: 中位={np.median(np.sqrt(W[:,0]**2+W[:,1]**2)):.0f}\" max={np.max(np.sqrt(W[:,0]**2+W[:,1]**2)):.0f}\"")

# 运行V4.0求解
print(f"\n=== 运行V4.0求解 ===")
wcs_json = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "batch_test",
                        f"debug_{os.path.basename(fits_path)}.json")
solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
result = solver.solve(
    np.array(det.x, np.float64), np.array(det.y, np.float64),
    np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
    cra0, cdec0, fl, ps, w, h,
    wcs_out=wcs_json,
    exptime=getattr(img.metadata.observation, 'exptime', 1.0),
)
solver.close()

if result:
    print(f"\n>>> 成功: mode={result.flip_mode} n={result.matched_count} RMS={result.rms_px:.3f}px")
else:
    print(f"\n>>> 失败: result=None")

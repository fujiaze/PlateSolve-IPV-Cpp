"""检查Oiii帧的RA/DEC偏移量"""
import sys, os
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'astro_image_io', 'python'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'star_detector', 'python'))
import numpy as np
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
import math

# Oiii帧
fits_path = os.path.join(PROJECT_ROOT, 'testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts')
reader = ImageReader()
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen; ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl
hdul = afits.open(fits_path); hdr = hdul[0].header; hdul.close()
ra_str = hdr.get('RA', hdr.get('OBJCTRA'))
dec_str = hdr.get('DEC', hdr.get('OBJCTDEC'))
sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
cra0, cdec0 = sc.ra.deg, sc.dec.deg

# 构建U向量
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
img_x = np.array(det.x, np.float64)
img_y = np.array(det.y, np.float64)
img_flux = np.array(det.flux, np.float64)
img_sat = np.array(det.saturated, np.int32)
nsat = int(img_sat.sum())

# 选星
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

# 查询Gaia
gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0
Ngaia = 150
maglim, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(client, cra0, cdec0, fov_diag*0.5, Ngaia)

# 构建W向量 (gnomonic投影)
from vector_match_v2 import _build_catalog_vectors
W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)
norm_W = np.sqrt(W[0,:]**2 + W[1,:]**2)

print(f'N={len(sel_idx)} M={M} s0={s0:.4f}')
print(f'FOV diagonal={fov_diag:.2f}deg = {fov_diag*3600:.0f}"')
print(f'max_t = {fov_diag*0.6*3600:.0f}" = {fov_diag*0.6:.2f}deg')
print(f'norm_U: min={norm_U.min():.1f}" max={norm_U.max():.1f}" median={np.median(norm_U):.1f}"')
print(f'norm_W: min={norm_W.min():.1f}" max={norm_W.max():.1f}" median={np.median(norm_W):.1f}"')

# 模拟1点抽样：检查正确对的比例
# 正确对：s在[0.9,1.1]内
s_matrix = norm_U[:,None] / norm_W[None,:]  # NxM
s_in_range = (s_matrix >= 0.9) & (s_matrix <= 1.1)
print(f'\n1点抽样s过滤: {s_in_range.sum()}/{s_matrix.size} = {s_in_range.sum()/s_matrix.size*100:.1f}%')

# 检查tx/ty过滤
# 对于s在范围内的对，计算tx/ty
angle_U_arr = np.arctan2(uy, ux)
angle_W_arr = np.arctan2(W[1,:], W[0,:])
max_t = fov_diag * 0.6 * 3600  # arcsec

n_tx_ok = 0
n_total = 0
for i in range(len(sel_idx)):
    for j in range(M):
        if not s_in_range[i,j]: continue
        s = s_matrix[i,j]
        theta = angle_U_arr[i] - angle_W_arr[j]
        ct, st = np.cos(theta), np.sin(theta)
        tx = ux[i] - s*(ct*W[0,j] - st*W[1,j])
        ty = uy[i] - s*(st*W[0,j] + ct*W[1,j])
        n_total += 1
        if abs(tx) <= max_t and abs(ty) <= max_t:
            n_tx_ok += 1

print(f'tx/ty过滤: {n_tx_ok}/{n_total} = {n_tx_ok/n_total*100:.1f}% 通过')
print(f'总有效对(s+tx/ty): {n_tx_ok}/{len(sel_idx)*M} = {n_tx_ok/(len(sel_idx)*M)*100:.1f}%')

# 关键：检查Gaia星是否真的在图像FOV内
# Gaia星的gnomonic投影坐标应该在FOV范围内
fov_half_w = w/2 * s0  # arcsec
fov_half_h = h/2 * s0  # arcsec
in_fov = (np.abs(W[0,:]) < fov_half_w*1.5) & (np.abs(W[1,:]) < fov_half_h*1.5)
print(f'\nGaia星在1.5xFOV内: {in_fov.sum()}/{M}')
in_fov2 = (np.abs(W[0,:]) < fov_half_w) & (np.abs(W[1,:]) < fov_half_h)
print(f'Gaia星在1.0xFOV内: {in_fov2.sum()}/{M}')

# 检查图像星和Gaia星的空间重叠
print(f'\n图像U范围: x=[{ux.min():.0f}",{ux.max():.0f}"] y=[{uy.min():.0f}",{uy.max():.0f}"]')
print(f'Gaia W范围: x=[{W[0,:].min():.0f}",{W[0,:].max():.0f}"] y=[{W[1,:].min():.0f}",{W[1,:].max():.0f}"]')

client.close()

"""检查Gaia星的投影坐标分布"""
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
from vector_match_v2 import GaiaClientPy, bisection_mag_limit, _build_catalog_vectors
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

# 查询Gaia
gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0
Ngaia = 150
maglim, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(client, cra0, cdec0, fov_diag*0.5, Ngaia)

# 构建W向量
W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)

# 图像FOV范围 (arcsec)
fov_half_w = w/2 * s0
fov_half_h = h/2 * s0
print(f'图像FOV: {w*s0:.0f}" x {h*s0:.0f}" = {w*s0/3600:.2f}deg x {h*s0/3600:.2f}deg')
print(f'图像U范围: x=[{ux.min():.0f}",{ux.max():.0f}"] y=[{uy.min():.0f}",{uy.max():.0f}"]')
print(f'Gaia W范围: x=[{W[0,:].min():.0f}",{W[0,:].max():.0f}"] y=[{W[1,:].min():.0f}",{W[1,:].max():.0f}"]')

# Gaia星在FOV内的比例
in_fov = (np.abs(W[0,:]) < fov_half_w) & (np.abs(W[1,:]) < fov_half_h)
print(f'\nGaia星在FOV内: {in_fov.sum()}/{M} = {in_fov.sum()/M*100:.1f}%')

# Gaia星到中心的距离
norm_W = np.sqrt(W[0,:]**2 + W[1,:]**2)
print(f'Gaia norm_W: min={norm_W.min():.0f}" max={norm_W.max():.0f}" median={np.median(norm_W):.0f}"')
print(f'Gaia norm_W / FOV半对角线: min={norm_W.min()/(fov_diag*3600/2):.2f} max={norm_W.max()/(fov_diag*3600/2):.2f}')

# 对比Red帧
print('\n--- Red帧对比 ---')
red_path = os.path.join(PROJECT_ROOT, 'testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts')
img2 = reader.read(red_path)
fl2 = img2.metadata.observation.focallen; ps2 = img2.metadata.observation.xpixsz
s0_2 = 206.265 * ps2 / fl2
hdul2 = afits.open(red_path); hdr2 = hdul2[0].header; hdul2.close()
ra2 = hdr2.get('RA', hdr2.get('OBJCTRA'))
dec2 = hdr2.get('DEC', hdr2.get('OBJCTDEC'))
sc2 = SkyCoord(ra2, dec2, unit=(u.hourangle, u.deg))
det2 = detector.detect_ex(img2.data)
img_sat2 = np.array(det2.saturated, np.int32)
nsat2 = int(img_sat2.sum())
Ngaia2 = math.ceil(1.5 * nsat2) if nsat2 >= 50 else 150
fov_diag2 = np.sqrt(img2.width**2+img2.height**2)*s0_2/3600.0
maglim2, M2, cat_ra2, cat_dec2, cat_mag2 = bisection_mag_limit(client, sc2.ra.deg, sc2.dec.deg, fov_diag2*0.5, Ngaia2)
W2 = _build_catalog_vectors(cat_ra2, cat_dec2, sc2.ra.deg, sc2.dec.deg)
norm_W2 = np.sqrt(W2[0,:]**2 + W2[1,:]**2)
fov_half_w2 = img2.width/2 * s0_2
fov_half_h2 = img2.height/2 * s0_2
in_fov2 = (np.abs(W2[0,:]) < fov_half_w2) & (np.abs(W2[1,:]) < fov_half_h2)
print(f'Red帧: Ngaia={Ngaia2} M={M2}')
print(f'Red Gaia W范围: x=[{W2[0,:].min():.0f}",{W2[0,:].max():.0f}"] y=[{W2[1,:].min():.0f}",{W2[1,:].max():.0f}"]')
print(f'Red Gaia星在FOV内: {in_fov2.sum()}/{M2} = {in_fov2.sum()/M2*100:.1f}%')
print(f'Red Gaia norm_W: min={norm_W2.min():.0f}" max={norm_W2.max():.0f}" median={np.median(norm_W2):.0f}"')

client.close()

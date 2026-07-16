"""直接检查Gaia返回的RA/DEC和投影坐标"""
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
from vector_match_v2 import GaiaClientPy, bisection_mag_limit, _build_catalog_vectors, gnomonic_forward

# Oiii帧
fits_path = os.path.join(PROJECT_ROOT, 'testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts')
hdul = afits.open(fits_path); hdr = hdul[0].header; hdul.close()
ra_str = hdr.get('RA', hdr.get('OBJCTRA'))
dec_str = hdr.get('DEC', hdr.get('OBJCTDEC'))
sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
cra0, cdec0 = sc.ra.deg, sc.dec.deg

fl = 200.0; ps = 6.0; s0 = 206.265*ps/fl
w, h = 4500, 3600
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0

# 查询Gaia
gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)
Ngaia = 150
maglim, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(client, cra0, cdec0, fov_diag*0.5, Ngaia)

print(f'投影中心: RA={cra0:.6f} DEC={cdec0:.6f}')
print(f'Gaia查询: maglim={maglim:.1f} M={M}')
print(f'Gaia RA范围: {cat_ra.min():.4f} ~ {cat_ra.max():.4f} (中心{cra0:.4f})')
print(f'Gaia DEC范围: {cat_dec.min():.4f} ~ {cat_dec.max():.4f} (中心{cdec0:.4f})')

# 检查Gaia星到中心的角距离
from astropy.coordinates import Angle
center = SkyCoord(ra=cra0*u.deg, dec=cdec0*u.deg)
stars = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg)
sep = center.separation(stars)
print(f'\nGaia星到中心角距: min={sep.min().arcmin:.1f}\' max={sep.max().arcmin:.1f}\' median={np.median(sep.arcmin):.1f}\'')
print(f'FOV半对角线: {fov_diag*30:.1f}\' = {fov_diag:.2f}°')

# gnomonic投影
xi, eta = gnomonic_forward(cat_ra, cat_dec, cra0, cdec0)
print(f'\ngnomonic投影 (arcsec):')
print(f'  xi范围: [{xi.min():.0f}", {xi.max():.0f}"]')
print(f'  eta范围: [{eta.min():.0f}", {eta.max():.0f}"]')

# FOV范围
fov_half_w = w/2 * s0
fov_half_h = h/2 * s0
print(f'  FOV范围: xi=[-{fov_half_w:.0f}", {fov_half_w:.0f}"] eta=[-{fov_half_h:.0f}", {fov_half_h:.0f}"]')

# 在FOV内的星
in_fov = (np.abs(xi) < fov_half_w) & (np.abs(eta) < fov_half_h)
print(f'  在FOV内: {in_fov.sum()}/{M}')

# 检查_build_catalog_vectors的输出
W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)
print(f'\n_build_catalog_vectors输出:')
print(f'  W[0]范围: [{W[0,:].min():.0f}", {W[0,:].max():.0f}"]')
print(f'  W[1]范围: [{W[1,:].min():.0f}", {W[1,:].max():.0f}"]')

# 对比xi/eta和W
print(f'\nxi vs W[0] 差异: max={np.abs(xi - W[0,:]).max():.6f}"')
print(f'eta vs W[1] 差异: max={np.abs(eta - W[1,:]).max():.6f}"')

# 检查：Gaia星是否都在查询半径内
in_radius = sep.deg < fov_diag*0.5
print(f'\nGaia星在查询半径内: {in_radius.sum()}/{M}')

# 检查：Gaia星是否都在FOV对角线内
in_diag = sep.arcsec < fov_diag*3600
print(f'Gaia星在FOV对角线内: {in_diag.sum()}/{M}')

client.close()

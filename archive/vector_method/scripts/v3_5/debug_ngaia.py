"""对比不同Ngaia的查询结果"""
import sys, os
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from vector_match_v2 import GaiaClientPy, bisection_mag_limit, _build_catalog_vectors

# Oiii帧参数
cra0, cdec0 = 272.808333, -13.176944
fl, ps = 200.0, 6.0
s0 = 206.265*ps/fl
w, h = 4500, 3600
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0
fov_half_w = w/2 * s0
fov_half_h = h/2 * s0

gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)

# 测试不同Ngaia
for Ngaia in [150, 300, 500, 1000, 2000]:
    maglim, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(client, cra0, cdec0, fov_diag*0.5, Ngaia)
    W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)
    in_fov = (np.abs(W[0,:]) < fov_half_w) & (np.abs(W[1,:]) < fov_half_h)
    center = SkyCoord(ra=cra0*u.deg, dec=cdec0*u.deg)
    stars = SkyCoord(ra=cat_ra*u.deg, dec=cat_dec*u.deg)
    sep = center.separation(stars)
    print(f'Ngaia={Ngaia:4d}: maglim={maglim:5.1f} M={M:4d} 在FOV内={in_fov.sum():3d} 角距min={sep.min().arcmin:.1f}\' median={np.median(sep.arcmin):.1f}\'')

client.close()

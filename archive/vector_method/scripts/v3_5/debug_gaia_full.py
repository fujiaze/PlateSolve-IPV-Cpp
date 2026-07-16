"""用mag=22查询，看FOV内有多少Gaia星"""
import sys, os
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
import numpy as np
from vector_match_v2 import GaiaClientPy, _build_catalog_vectors

cra0, cdec0 = 272.808333, -13.176944
fl, ps = 200.0, 6.0
s0 = 206.265*ps/fl
w, h = 4500, 3600
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0
fov_half_w = w/2 * s0
fov_half_h = h/2 * s0

gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)

# 直接用mag=22查询
ra, dec, mag = client.cone_search(cra0, cdec0, fov_diag*0.5, 22.0)
print(f'mag=22查询: {len(ra)} 颗星')

# 投影
W = _build_catalog_vectors(ra, dec, cra0, cdec0)
in_fov = (np.abs(W[0,:]) < fov_half_w) & (np.abs(W[1,:]) < fov_half_h)
print(f'在FOV内: {in_fov.sum()}/{len(ra)}')

# 按星等统计FOV内的星
for mlim in [10, 12, 14, 16, 18, 20, 22]:
    mask = mag <= mlim
    in_fov_m = in_fov & mask
    print(f'  mag<={mlim}: FOV内 {in_fov_m.sum()}/{mask.sum()}')

# FOV内星的投影坐标分布
if in_fov.sum() > 0:
    fov_W = W[:, in_fov]
    print(f'\nFOV内Gaia星投影坐标:')
    print(f'  W[0]: [{fov_W[0,:].min():.0f}", {fov_W[0,:].max():.0f}"]')
    print(f'  W[1]: [{fov_W[1,:].min():.0f}", {fov_W[1,:].max():.0f}"]')

client.close()

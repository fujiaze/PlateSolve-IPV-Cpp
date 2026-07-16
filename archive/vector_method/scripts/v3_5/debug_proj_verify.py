"""直接验证gnomonic投影"""
import sys, os
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

PROJECT_ROOT = r'F:\Astro dev\Astro CS Normalization Database'
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lib', 'plate_solve', 'python'))
import numpy as np
from vector_match_v2 import GaiaClientPy, gnomonic_forward

cra0, cdec0 = 272.808333, -13.176944
fl, ps = 200.0, 6.0
s0 = 206.265*ps/fl
w, h = 4500, 3600
fov_half_w = w/2 * s0  # arcsec
fov_half_h = h/2 * s0  # arcsec

print(f'投影中心: RA={cra0:.6f} DEC={cdec0:.6f}')
print(f'FOV: {fov_half_w*2:.0f}" x {fov_half_h*2:.0f}" = {fov_half_w*2/3600:.2f}deg x {fov_half_h*2/3600:.2f}deg')
print(f'FOV半宽: {fov_half_w:.0f}" = {fov_half_w/3600:.2f}deg')
print(f'FOV半高: {fov_half_h:.0f}" = {fov_half_h/3600:.2f}deg')

# 测试1: 投影中心本身
xi, eta, valid = gnomonic_forward(np.array([cra0]), np.array([cdec0]), cra0, cdec0)
print(f'\n投影中心: xi={xi[0]:.2f}" eta={eta[0]:.2f}" (应该≈0)')

# 测试2: 中心偏移0.1°
xi2, eta2, _ = gnomonic_forward(np.array([cra0+0.1]), np.array([cdec0]), cra0, cdec0)
print(f'RA+0.1°: xi={xi2[0]:.2f}" eta={eta2[0]:.2f}"')

xi3, eta3, _ = gnomonic_forward(np.array([cra0]), np.array([cdec0+0.1]), cra0, cdec0)
print(f'DEC+0.1°: xi={xi3[0]:.2f}" eta={eta3[0]:.2f}"')

# 测试3: 中心偏移1°
xi4, eta4, _ = gnomonic_forward(np.array([cra0+1.0]), np.array([cdec0]), cra0, cdec0)
print(f'RA+1.0°: xi={xi4[0]:.2f}" eta={eta4[0]:.2f}"')

xi5, eta5, _ = gnomonic_forward(np.array([cra0]), np.array([cdec0+1.0]), cra0, cdec0)
print(f'DEC+1.0°: xi={xi5[0]:.2f}" eta={eta5[0]:.2f}"')

# 测试4: 图像四角对应的RA/DEC
from vector_match_v2 import gnomonic_inverse
corners = [
    (-fov_half_w, -fov_half_h, "左下"),
    (fov_half_w, -fov_half_h, "右下"),
    (-fov_half_w, fov_half_h, "左上"),
    (fov_half_w, fov_half_h, "右上"),
]
print(f'\n图像四角对应的RA/DEC:')
for xi_c, eta_c, name in corners:
    ra_c, dec_c = gnomonic_inverse(xi_c, eta_c, cra0, cdec0)
    sep = np.sqrt((ra_c-cra0)**2 * np.cos(np.radians(cdec0))**2 + (dec_c-cdec0)**2) * 3600
    print(f'  {name}: RA={ra_c:.6f} DEC={dec_c:.6f} (角距≈{sep/3600:.2f}°)')

# 测试5: 查询FOV中心附近的Gaia星
gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)

# 查询1°半径内的亮星
ra_near, dec_near, mag_near = client.cone_search(cra0, cdec0, 1.0, 12.0)
print(f'\n中心1°内mag<12: {len(ra_near)} 颗星')
if len(ra_near) > 0:
    xi_near, eta_near, valid_near = gnomonic_forward(ra_near, dec_near, cra0, cdec0)
    in_fov = (np.abs(xi_near) < fov_half_w) & (np.abs(eta_near) < fov_half_h)
    print(f'  在FOV内: {in_fov.sum()}/{len(ra_near)}')
    if in_fov.sum() > 0:
        for k in range(min(5, in_fov.sum())):
            idx = np.where(in_fov)[0][k]
            print(f'  星{k}: RA={ra_near[idx]:.6f} DEC={dec_near[idx]:.6f} mag={mag_near[idx]:.1f} xi={xi_near[idx]:.0f}" eta={eta_near[idx]:.0f}"')

client.close()

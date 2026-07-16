"""验证Oiii帧的Gaia查询结果"""
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

detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
img_sat = np.array(det.saturated, np.int32)
nsat = int(img_sat.sum())

print(f'Oiii帧: RA={cra0:.4f} DEC={cdec0:.4f} nsat={nsat} s0={s0:.4f}')

# 查询Gaia
gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0
Ngaia = 150  # nsat<50时用150
maglim, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(client, cra0, cdec0, fov_diag*0.5, Ngaia)
print(f'Gaia查询: maglim={maglim:.1f} M={M} 颗星')
if M > 0:
    print(f'Gaia星等: min={cat_mag.min():.1f} max={cat_mag.max():.1f} median={np.median(cat_mag):.1f}')
    # 检查Gaia星的RA/DEC范围
    print(f'Gaia RA范围: {cat_ra.min():.4f} ~ {cat_ra.max():.4f} (中心{cra0:.4f})')
    print(f'Gaia DEC范围: {cat_dec.min():.4f} ~ {cat_dec.max():.4f} (中心{cdec0:.4f})')

# Red帧对比
print('\n--- Red帧 ---')
red_path = os.path.join(PROJECT_ROOT, 'testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts')
img2 = reader.read(red_path)
fl2 = img2.metadata.observation.focallen; ps2 = img2.metadata.observation.xpixsz
s0_2 = 206.265 * ps2 / fl2
hdul2 = afits.open(red_path); hdr2 = hdul2[0].header; hdul2.close()
ra2 = hdr2.get('RA', hdr2.get('OBJCTRA'))
dec2 = hdr2.get('DEC', hdr2.get('OBJCTDEC'))
sc2 = SkyCoord(ra2, dec2, unit=(u.hourangle, u.deg))
det2 = detector.detect_ex(img2.data)
nsat2 = int(np.array(det2.saturated, np.int32).sum())
Ngaia2 = math.ceil(1.5 * nsat2) if nsat2 >= 50 else 150
fov_diag2 = np.sqrt(img2.width**2+img2.height**2)*s0_2/3600.0
maglim2, M2, cat_ra2, cat_dec2, cat_mag2 = bisection_mag_limit(client, sc2.ra.deg, sc2.dec.deg, fov_diag2*0.5, Ngaia2)
print(f'Red帧: RA={sc2.ra.deg:.4f} DEC={sc2.dec.deg:.4f} nsat={nsat2} Ngaia={Ngaia2}')
print(f'Gaia查询: maglim={maglim2:.1f} M={M2} 颗星')
print(f'RA差异: {abs(cra0-sc2.ra.deg)*3600:.1f}"  DEC差异: {abs(cdec0-sc2.dec.deg)*3600:.1f}"')

import math
client.close()

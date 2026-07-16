"""验证Oiii帧的RA/DEC是否正确，Gaia查询区域是否包含图像中的星"""
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
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
s0 = 206.265 * ps / fl

# 从FITS头获取RA/DEC
hdul = afits.open(fits_path)
hdr = hdul[0].header
hdul.close()
ra_str = hdr.get('RA', hdr.get('OBJCTRA'))
dec_str = hdr.get('DEC', hdr.get('OBJCTDEC'))
sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
cra0, cdec0 = sc.ra.deg, sc.dec.deg

print(f'FITS头 RA={ra_str} -> {cra0:.6f}deg')
print(f'FITS头 DEC={dec_str} -> {cdec0:.6f}deg')
print(f'焦距={fl}mm 相元={ps}um s0={s0:.4f}"/px')
print(f'FOV: {w*s0/3600:.2f}deg x {h*s0/3600:.2f}deg')

# 检测星点
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)
img_x = np.array(det.x, np.float64)
img_y = np.array(det.y, np.float64)
img_sat = np.array(det.saturated, np.int32)
nsat = int(img_sat.sum())
print(f'检测到 {len(img_x)} 颗星, {nsat} 颗饱和星')

# 查询Gaia星表
gaia_dir = os.path.join(PROJECT_ROOT, 'GaiaDR3')
client = GaiaClientPy(gaia_dir)
fov_diag = np.sqrt(w*w+h*h)*s0/3600.0
radius = fov_diag * 0.5  # 候选半径
cat_ra, cat_dec, cat_mag = client.query(cra0, cdec0, radius)
print(f'Gaia查询: center=({cra0:.4f}, {cdec0:.4f}), radius={radius:.2f}deg')
print(f'Gaia返回: {len(cat_ra)} 颗星')

# 检查Gaia星的星等分布
if len(cat_mag) > 0:
    print(f'Gaia星等: min={cat_mag.min():.1f} max={cat_mag.max():.1f} median={np.median(cat_mag):.1f}')

# 用Red帧对比
print('\n--- 对比Red帧 ---')
red_path = os.path.join(PROJECT_ROOT, 'testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts')
img2 = reader.read(red_path)
fl2 = img2.metadata.observation.focallen
ps2 = img2.metadata.observation.xpixsz
s0_2 = 206.265 * ps2 / fl2
hdul2 = afits.open(red_path)
hdr2 = hdul2[0].header
hdul2.close()
ra2 = hdr2.get('RA', hdr2.get('OBJCTRA'))
dec2 = hdr2.get('DEC', hdr2.get('OBJCTDEC'))
sc2 = SkyCoord(ra2, dec2, unit=(u.hourangle, u.deg))
print(f'Red FITS头 RA={ra2} -> {sc2.ra.deg:.6f}deg')
print(f'Red FITS头 DEC={dec2} -> {sc2.dec.deg:.6f}deg')
print(f'RA差异: {abs(cra0-sc2.ra.deg)*3600:.1f}"  DEC差异: {abs(cdec0-sc2.dec.deg)*3600:.1f}"')

det2 = detector.detect_ex(img2.data)
nsat2 = int(np.array(det2.saturated, np.int32).sum())
print(f'Red帧: {len(det2.x)} 颗星, {nsat2} 颗饱和星')

# 查询Red帧的Gaia
cat_ra2, cat_dec2, cat_mag2 = client.query(sc2.ra.deg, sc2.dec.deg, radius)
print(f'Red Gaia返回: {len(cat_ra2)} 颗星')

client.close()

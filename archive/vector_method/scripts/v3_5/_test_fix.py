"""快速测试CRVAL系统性偏移修复"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from star_detector import StarDetector, SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u

fits_path = 'testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts'
reader = ImageReader()
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
hdul = afits.open(fits_path)
hdr = hdul[0].header
exptime = float(hdr.get('EXPTIME', 1.0))
ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
cra0, cdec0 = sc.ra.deg, sc.dec.deg
hdul.close()

detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)

gaia_dir = os.path.join('.', 'GaiaDR3')
vm = VectorMatchV35Cpp(gaia_dir)

result = vm.solve(
    np.array(det.x, np.float64), np.array(det.y, np.float64),
    np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
    cra0, cdec0, fl, ps, w, h,
    wcs_out='overlay_output/_test_fix.json', skip_sip=False, exptime=exptime)
vm.close()

if result:
    print(f'OK s={result.solve_s:.4f} theta={result.rotation_deg:.3f} n={result.matched_count}')
    print(f'tx={result.solve_tx:.3f} ty={result.solve_ty:.3f}')
    with open('overlay_output/_test_fix.json') as f:
        wcs = json.load(f)
    print(f'CD={wcs["CD"]}')
    print(f'CRVAL={wcs["CRVAL"]}')
    print(f'RMS_PX={wcs["RMS_PX"]}')
    print(f'SIP_ORDER={wcs.get("SIP_ORDER", 0)}')
else:
    print('FAILED')

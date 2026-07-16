# 全局强制UTF-8编码
import sys, os, json
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from star_detector import StarDetector, SDetParamsPy

reader = ImageReader()
fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
img = reader.read(fits_path)
w, h = img.width, img.height
fl = img.metadata.observation.focallen
ps = img.metadata.observation.xpixsz
cra0 = img.metadata.wcs.crval1
cdec0 = img.metadata.wcs.crval2

detector = StarDetector(params=SDetParamsPy(fitRadius=0))
det = detector.detect_ex(img.data)

gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
vm = VectorMatchV35Cpp(gaia_dir)
wcs_json = os.path.join(PROJECT_ROOT, "vm35_wcs_output.json")
result = vm.solve(
    np.array(det.x, np.float64), np.array(det.y, np.float64),
    np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
    cra0, cdec0, fl, ps, w, h, wcs_out=wcs_json
)
vm.close()

print(f"s={result.solve_s:.6f}, theta={result.rotation_deg:.6f}, n_inliers={result.n_phased_clean}, rms={result.sip_rms_px:.6f}")
print(f"best_mode={result.flip_mode}, sip_order={result.sip_order}")

# 读取生成的WCS JSON
with open(wcs_json, "r", encoding="utf-8") as f:
    d = json.load(f)
print(f"CD: {d['CD']}")
print(f"CRVAL: {d['CRVAL']}")
print(f"SIP_ORDER: {d['SIP_ORDER']}")
print(f"RMS_PX: {d['RMS_PX']}")
a_nonzero = sum(1 for x in d["SIP_A"] if abs(x) > 1e-30)
b_nonzero = sum(1 for x in d["SIP_B"] if abs(x) > 1e-30)
print(f"SIP_A nonzero: {a_nonzero}, SIP_B nonzero: {b_nonzero}")

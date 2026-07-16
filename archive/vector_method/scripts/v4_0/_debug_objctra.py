"""快速测试: 用OBJCTRA/DEC作为初始值解析单帧"""
import os, sys, time, math
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp

def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m/60.0 + sec/3600.0) * 15.0
    return float(s)

def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith('-'):
        sign = -1.0; s = s[1:]
    elif s.startswith('+'):
        s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m/60.0 + sec/3600.0)
    return float(s)

# 测试3帧: M20(Dec=-22), NGC4945(Dec=-49), NGC55(Dec=-39)
test_files = [
    os.path.join(PROJECT_ROOT, "testdata", "lights", "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"),
    os.path.join(PROJECT_ROOT, "testdata", "lights", "T3", "lum", "NGC4945_FD_T3_flying_dutchman-20250205@074722-600S-Lum.fts"),
    os.path.join(PROJECT_ROOT, "testdata", "lights", "T3", "lum", "NGC55_FD_T3_flying_dutchman-20250205@073502-600S-Lum.fts"),
]

solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)

for fits_path in test_files:
    if not os.path.exists(fits_path):
        print(f"跳过(不存在): {os.path.basename(fits_path)}")
        continue
    print(f"\n=== {os.path.basename(fits_path)} ===")
    t0 = time.time()

    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    # 用OBJCTRA/DEC作为初始指向
    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    cra0 = _parse_ra_hms(kw_dict.get('OBJCTRA') or kw_dict.get('RA'))
    cdec0 = _parse_dec_dms(kw_dict.get('OBJCTDEC') or kw_dict.get('DEC'))
    print(f"  OBJCTRA={cra0:.6f}° OBJCTDEC={cdec0:.6f}° s0={s0:.4f}\"/px")

    # 对比CRVAL
    if img.has_wcs:
        print(f"  CRVAL1={img.metadata.wcs.crval1:.6f}° CRVAL2={img.metadata.wcs.crval2:.6f}° (差距: RA={abs(cra0-img.metadata.wcs.crval1)*3600:.1f}\" Dec={abs(cdec0-img.metadata.wcs.crval2)*3600:.1f}\")")

    t1 = time.time()
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    print(f"  检测: {len(det.x)}颗 饱和{int(np.sum(det.saturated))}颗 耗时{time.time()-t1:.2f}s")

    t2 = time.time()
    wcs_json = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "batch_test",
                            f"wcs_objctra_{os.path.basename(fits_path)}.json")
    result = solver.solve(
        np.array(det.x, np.float64), np.array(det.y, np.float64),
        np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
        cra0, cdec0, fl, ps, w, h,
        wcs_out=wcs_json,
        exptime=getattr(img.metadata.observation, 'exptime', 1.0),
    )
    t3 = time.time()
    if result:
        print(f"  >>> 成功: mode={result.flip_mode} n={result.matched_count} RMS={result.rms_px:.3f}px s={result.scale_arcsec_px:.4f}\"/px 求解耗时={t3-t2:.2f}s")
    else:
        print(f"  >>> 失败 求解耗时={t3-t2:.2f}s")

solver.close()
print(f"\n总耗时: {time.time()-t0:.2f}s")

"""V4.2 Pipeline 冒烟测试

功能: 对单帧 FITS 运行 V42Pipeline.solve(), 验证 5 阶段串联 + flip 修复
用途: Task 7 验证 - 单帧冒烟测试 + 断点续跑测试
"""
import os
import sys
import time
import logging

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("V4.2冒烟")

from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy
from star_detector import StarDetector, SDetParamsPy
from v4_2.pipeline import V42Pipeline


def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m / 60.0 + sec / 3600.0) * 15.0
    return float(s)


def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith('-'):
        sign = -1.0
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m / 60.0 + sec / 3600.0)
    return float(s)


def main():
    fits_path = os.path.join(
        PROJECT_ROOT, "testdata", "lights",
        "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts")

    print(f"=== V4.2 Pipeline 冒烟测试 ===")
    print(f"FITS: {os.path.basename(fits_path)}")

    # 读取 FITS header
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    print(f"图像: {w}x{h}, focal={fl}mm, pixel={ps}um")

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    cra0 = _parse_ra_hms(kw_dict.get('OBJCTRA') or kw_dict.get('RA'))
    cdec0 = _parse_dec_dms(kw_dict.get('OBJCTDEC') or kw_dict.get('DEC'))
    print(f"中心指向: RA={cra0:.6f}° Dec={cdec0:.6f}°")

    # 实例化 pipeline (注入 Gaia 客户端和 StarDetector)
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=0)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))

    dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")

    pipeline = V42Pipeline(
        dll_dir=dll_dir,
        gaia_client=gaia_client,
        star_detector=star_detector,
    )

    # 运行求解
    force_phase = os.environ.get("V42_FORCE_PHASE", "")
    print(f"\n--- 运行 solve (resume=True, force_phase={force_phase or 'None'}) ---")
    t0 = time.time()
    result = pipeline.solve(
        image_path=fits_path,
        ra=cra0, dec=cdec0,
        focal_length_mm=fl, pixel_size_um=ps,
        resume=True,
        force_phase=force_phase if force_phase else None,
    )
    elapsed = time.time() - t0

    # 输出结果
    print(f"\n=== 结果 ===")
    print(f"耗时: {elapsed:.2f}s")
    print(f"success: {result.get('success')}")
    print(f"phase_status: {result.get('phase_status')}")
    if result.get('success'):
        print(f"matched_count: {result.get('matched_count')}")
        print(f"rms_px: {result.get('rms_px'):.4f}")
        print(f"rms_arcsec: {result.get('rms_arcsec'):.4f}")
        print(f"scale_arcsec_px: {result.get('scale_arcsec_px'):.4f}")
        print(f"rotation_deg: {result.get('rotation_deg'):.4f}")
        print(f"flip_mode: {result.get('flip_mode')}")
        print(f"s: {result.get('s'):.6f}, θ: {result.get('rotation_deg'):.4f}°")
        print(f"bayes_lnK: {result.get('bayes_lnK'):.2f}")
        print(f"triangle_pass_ratio: {result.get('triangle_pass_ratio'):.3f}")
        print(f"validated: {result.get('validated')}")
        print(f"CD: {result.get('cd')}")
        print(f"CRVAL: {result.get('crval')}")
        print(f"CRPIX: {result.get('crpix')}")
        print(f"SIP_ORDER: {result.get('sip_order')}")
    else:
        print(f"error: {result.get('error')}")

    pipeline.close()
    gaia_client.close()
    star_detector.close()

    return 0 if result.get('success') else 1


if __name__ == "__main__":
    sys.exit(main())

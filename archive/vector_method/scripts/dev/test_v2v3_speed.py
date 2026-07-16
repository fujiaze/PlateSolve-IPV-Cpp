"""
V2 vs V3 速度对比 (Gaia缓存优化后)

测试panel1所有帧, 记录:
  - solve总耗时
  - 各步骤耗时
  - 匹配结果(RMS, matched, scale, rotation)
  - WCS矩阵推导验证
"""

import os, sys, time, math, logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2
from vector_match_v3 import VectorMatch as VectorMatchV3
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

PANEL1_DIR = r"testdata\lights\panel1"
GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def extract_filter(filename):
    for f in ["H-alpha", "Oiii", "Red", "Green", "Blue"]:
        if f in filename: return f
    return "Unknown"


def get_frame_params(frame_path):
    reader = ImageReader()
    img = reader.read(frame_path)
    width, height = img.width, img.height
    center_ra, center_dec = 0.0, 0.0
    focal_length, pixel_size = 200.0, 6.0

    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        center_ra = img.metadata.wcs.crval1
        center_dec = img.metadata.wcs.crval2
    if img.metadata.observation:
        if img.metadata.observation.focallen is not None:
            focal_length = img.metadata.observation.focallen
        if img.metadata.observation.xpixsz is not None:
            pixel_size = img.metadata.observation.xpixsz

    if center_ra == 0.0 and center_dec == 0.0:
        for kw in img.keywords:
            name = kw.name.upper()
            if name in ("OBJCTRA", "RA"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                    if len(parts) >= 3:
                        center_ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
            elif name in ("OBJCTDEC", "DEC"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                    if len(parts) >= 3:
                        sign = -1 if parts[0].startswith("-") else 1
                        center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)

    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)

    return (np.array(det_result.x, dtype=np.float64),
            np.array(det_result.y, dtype=np.float64),
            np.array(det_result.flux, dtype=np.float64),
            np.array(det_result.saturated, dtype=np.int32),
            center_ra, center_dec, focal_length, pixel_size, width, height)


def result_to_wcs(result, width, height):
    """从VectorMatchResult推导WCS CD矩阵"""
    if result is None:
        return None
    s = result.scale_arcsec_px / 206265.0  # arcsec/px -> deg/px
    theta = math.radians(result.rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # CD矩阵 (考虑flip)
    flip = result.flip_mode
    if flip == 0:   # 无翻转
        cd1_1, cd1_2 = s * cos_t, -s * sin_t
        cd2_1, cd2_2 = s * sin_t,  s * cos_t
    elif flip == 1:  # 水平翻转
        cd1_1, cd1_2 = -s * cos_t, s * sin_t
        cd2_1, cd2_2 =  s * sin_t, s * cos_t
    elif flip == 2:  # 垂直翻转
        cd1_1, cd1_2 = s * cos_t, -s * sin_t
        cd2_1, cd2_2 = -s * sin_t, -s * cos_t
    else:            # 双翻转
        cd1_1, cd1_2 = -s * cos_t, s * sin_t
        cd2_1, cd2_2 = -s * sin_t, -s * cos_t

    crpix1, crpix2 = width / 2.0 + 0.5, height / 2.0 + 0.5

    return {
        'CRVAL1': result.center_ra, 'CRVAL2': result.center_dec,
        'CRPIX1': crpix1, 'CRPIX2': crpix2,
        'CD1_1': cd1_1, 'CD1_2': cd1_2,
        'CD2_1': cd2_1, 'CD2_2': cd2_2,
    }


def test_single(args):
    fname, version, vm_class = args
    frame_path = os.path.join(PROJECT_ROOT, PANEL1_DIR, fname)
    filt = extract_filter(fname)

    try:
        img_x, img_y, img_flux, img_saturated, center_ra, center_dec, \
            focal_length, pixel_size, width, height = get_frame_params(frame_path)
    except Exception as e:
        return fname, filt, version, None, 0.0, str(e), None

    vm = vm_class(GAIA_DATA_DIR, db_type=0)
    t0 = time.perf_counter()
    try:
        result = vm.solve(img_x, img_y, img_flux, img_saturated,
                          center_ra, center_dec, focal_length, pixel_size, width, height)
    except Exception as e:
        result = None
        err = str(e)
    else:
        err = None
    t_solve = time.perf_counter() - t0
    vm.close()

    wcs = result_to_wcs(result, width, height) if result else None

    return fname, filt, version, result, t_solve, err, wcs


def main():
    logging.basicConfig(level=logging.WARNING)

    panel1_dir = os.path.join(PROJECT_ROOT, PANEL1_DIR)
    files = sorted([f for f in os.listdir(panel1_dir) if f.endswith('.fts')])
    print(f"Panel1: {len(files)}帧")

    # V2测试
    print(f"\n{'='*80}")
    print("V2 测试")
    print(f"{'='*80}")
    v2_results = []
    t_v2_start = time.perf_counter()
    vm2 = VectorMatchV2(GAIA_DATA_DIR, db_type=0)
    for i, fname in enumerate(files):
        filt = extract_filter(fname)
        frame_path = os.path.join(PROJECT_ROOT, PANEL1_DIR, fname)
        img_x, img_y, img_flux, img_saturated, center_ra, center_dec, \
            focal_length, pixel_size, width, height = get_frame_params(frame_path)

        t0 = time.perf_counter()
        result = vm2.solve(img_x, img_y, img_flux, img_saturated,
                           center_ra, center_dec, focal_length, pixel_size, width, height)
        t_solve = time.perf_counter() - t0

        if result:
            wcs = result_to_wcs(result, width, height)
            v2_results.append({
                'fname': fname, 'filt': filt, 't_solve': t_solve,
                'rms_px': result.rms_px, 'matched': result.matched_count,
                'scale': result.scale_arcsec_px, 'rotation': result.rotation_deg,
                'flip': result.flip_mode, 'success': True,
                'wcs': wcs,
            })
            if i == 0:
                print(f"  WCS示例: CRVAL=({wcs['CRVAL1']:.4f},{wcs['CRVAL2']:.4f}) "
                      f"CD=({wcs['CD1_1']:.6e},{wcs['CD1_2']:.6e},{wcs['CD2_1']:.6e},{wcs['CD2_2']:.6e})")
        else:
            v2_results.append({'fname': fname, 'filt': filt, 't_solve': t_solve, 'success': False})

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] V2进度")

    t_v2_total = time.perf_counter() - t_v2_start
    vm2.close()

    # V3测试
    print(f"\n{'='*80}")
    print("V3 测试")
    print(f"{'='*80}")
    v3_results = []
    t_v3_start = time.perf_counter()
    vm3 = VectorMatchV3(GAIA_DATA_DIR, db_type=0)
    for i, fname in enumerate(files):
        filt = extract_filter(fname)
        frame_path = os.path.join(PROJECT_ROOT, PANEL1_DIR, fname)
        img_x, img_y, img_flux, img_saturated, center_ra, center_dec, \
            focal_length, pixel_size, width, height = get_frame_params(frame_path)

        t0 = time.perf_counter()
        result = vm3.solve(img_x, img_y, img_flux, img_saturated,
                           center_ra, center_dec, focal_length, pixel_size, width, height)
        t_solve = time.perf_counter() - t0

        if result:
            wcs = result_to_wcs(result, width, height)
            v3_results.append({
                'fname': fname, 'filt': filt, 't_solve': t_solve,
                'rms_px': result.rms_px, 'matched': result.matched_count,
                'scale': result.scale_arcsec_px, 'rotation': result.rotation_deg,
                'flip': result.flip_mode, 'success': True,
                'wcs': wcs,
            })
        else:
            v3_results.append({'fname': fname, 'filt': filt, 't_solve': t_solve, 'success': False})

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            print(f"  [{i+1}/{len(files)}] V3进度")

    t_v3_total = time.perf_counter() - t_v3_start
    vm3.close()

    # ============================================================
    # 对比报告
    # ============================================================
    print(f"\n{'='*80}")
    print("V2 vs V3 对比报告")
    print(f"{'='*80}")

    v2_ok = [r for r in v2_results if r['success']]
    v3_ok = [r for r in v3_results if r['success']]
    v2_fail = [r for r in v2_results if not r['success']]
    v3_fail = [r for r in v3_results if not r['success']]

    print(f"\n成功率:")
    print(f"  V2: {len(v2_ok)}/{len(files)} ({len(v2_ok)/len(files)*100:.1f}%)")
    print(f"  V3: {len(v3_ok)}/{len(files)} ({len(v3_ok)/len(files)*100:.1f}%)")

    v2_times = [r['t_solve'] for r in v2_results]
    v3_times = [r['t_solve'] for r in v3_results]

    print(f"\n耗时统计:")
    print(f"  {'':12} {'V2':>8} {'V3':>8} {'V3/V2':>7}")
    print(f"  {'中位':12} {np.median(v2_times):>7.2f}s {np.median(v3_times):>7.2f}s {np.median(v3_times)/np.median(v2_times):>6.2f}x")
    print(f"  {'均值':12} {np.mean(v2_times):>7.2f}s {np.mean(v3_times):>7.2f}s {np.mean(v3_times)/np.mean(v2_times):>6.2f}x")
    print(f"  {'P25':12} {np.percentile(v2_times,25):>7.2f}s {np.percentile(v3_times,25):>7.2f}s")
    print(f"  {'P75':12} {np.percentile(v2_times,75):>7.2f}s {np.percentile(v3_times,75):>7.2f}s")
    print(f"  {'总耗时':12} {t_v2_total:>7.1f}s {t_v3_total:>7.1f}s {t_v3_total/t_v2_total:>6.2f}x")

    # 按滤镜对比
    print(f"\n按滤镜对比:")
    print(f"  {'滤镜':<10} {'V2中位':>8} {'V3中位':>8} {'V3/V2':>7} {'V2成功率':>8} {'V3成功率':>8}")
    for filt in ["Red", "Green", "Blue", "H-alpha", "Oiii"]:
        v2f = [r for r in v2_results if r['filt'] == filt]
        v3f = [r for r in v3_results if r['filt'] == filt]
        if not v2f: continue
        v2t = np.median([r['t_solve'] for r in v2f])
        v3t = np.median([r['t_solve'] for r in v3f])
        v2r = sum(1 for r in v2f if r['success']) / len(v2f) * 100
        v3r = sum(1 for r in v3f if r['success']) / len(v3f) * 100
        print(f"  {filt:<10} {v2t:>7.2f}s {v3t:>7.2f}s {v3t/v2t:>6.2f}x {v2r:>7.1f}% {v3r:>7.1f}%")

    # RMS对比
    if v2_ok and v3_ok:
        v2_rms = [r['rms_px'] for r in v2_ok]
        v3_rms = [r['rms_px'] for r in v3_ok]
        print(f"\nRMS对比:")
        print(f"  V2: 中位={np.median(v2_rms):.3f}px 均值={np.mean(v2_rms):.3f}px")
        print(f"  V3: 中位={np.median(v3_rms):.3f}px 均值={np.mean(v3_rms):.3f}px")

    # 逐帧对比
    print(f"\n逐帧对比:")
    print(f"  {'#':>3} {'滤镜':<7} {'V2耗时':>7} {'V3耗时':>7} {'V3/V2':>6} {'V2 RMS':>7} {'V3 RMS':>7} {'V2匹配':>6} {'V3匹配':>6}")
    for i in range(len(files)):
        v2r = v2_results[i]
        v3r = v3_results[i]
        v2t = v2r['t_solve']
        v3t = v3r['t_solve']
        v2rms = f"{v2r['rms_px']:.3f}" if v2r['success'] else "---"
        v3rms = f"{v3r['rms_px']:.3f}" if v3r['success'] else "---"
        v2m = f"{v2r['matched']}" if v2r['success'] else "---"
        v3m = f"{v3r['matched']}" if v3r['success'] else "---"
        print(f"  {i+1:>3} {v2r['filt']:<7} {v2t:>6.2f}s {v3t:>6.2f}s {v3t/v2t:>5.2f}x {v2rms:>7} {v3rms:>7} {v2m:>6} {v3m:>6}")

    # WCS验证
    print(f"\nWCS矩阵验证 (V2首帧):")
    if v2_ok and v2_ok[0].get('wcs'):
        wcs = v2_ok[0]['wcs']
        print(f"  CRVAL1={wcs['CRVAL1']:.6f} CRVAL2={wcs['CRVAL2']:.6f}")
        print(f"  CRPIX1={wcs['CRPIX1']:.1f} CRPIX2={wcs['CRPIX2']:.1f}")
        print(f"  CD1_1={wcs['CD1_1']:.6e} CD1_2={wcs['CD1_2']:.6e}")
        print(f"  CD2_1={wcs['CD2_1']:.6e} CD2_2={wcs['CD2_2']:.6e}")
        det = wcs['CD1_1'] * wcs['CD2_2'] - wcs['CD1_2'] * wcs['CD2_1']
        print(f"  det(CD)={det:.6e} (负=翻转)")
        pixel_scale = math.sqrt(abs(det)) * 3600
        print(f"  像素尺度={pixel_scale:.4f} arcsec/px")


if __name__ == '__main__':
    main()

"""
Panel1批量测试 - C++ V2 vs Python V2 对比

测试panel1所有帧, 记录:
  - solve总耗时
  - 匹配结果(RMS, matched, scale, rotation)
  - 按滤镜统计成功率
  - 逐帧对比
"""

import os, sys, time, math, logging
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2Py
from vector_match_v2_cpp import VectorMatch as VectorMatchV2Cpp
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

PANEL1_DIR = os.path.join(PROJECT_ROOT, "testdata", "lights", "panel1")
GAIA_DATA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")


def extract_filter(filename):
    for f in ["H-alpha", "Oiii", "Red", "Green", "Blue"]:
        if f in filename:
            return f
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


def run_batch(files, vm_class, label):
    """批量测试"""
    results = []
    vm = vm_class(GAIA_DATA_DIR, db_type=0)
    t_total_start = time.perf_counter()

    for i, fname in enumerate(files):
        filt = extract_filter(fname)
        frame_path = os.path.join(PANEL1_DIR, fname)

        try:
            img_x, img_y, img_flux, img_saturated, center_ra, center_dec, \
                focal_length, pixel_size, width, height = get_frame_params(frame_path)
        except Exception as e:
            results.append({'fname': fname, 'filt': filt, 't_solve': 0, 'success': False, 'err': str(e)})
            continue

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

        if result:
            results.append({
                'fname': fname, 'filt': filt, 't_solve': t_solve, 'success': True,
                'rms_px': result.rms_px, 'rms_arcsec': result.rms_arcsec,
                'matched': result.matched_count, 'scale': result.scale_arcsec_px,
                'rotation': result.rotation_deg, 'flip': result.flip_mode,
                'center_ra': result.center_ra, 'center_dec': result.center_dec,
            })
        else:
            results.append({'fname': fname, 'filt': filt, 't_solve': t_solve, 'success': False, 'err': err})

        if (i + 1) % 10 == 0 or i == len(files) - 1:
            ok = sum(1 for r in results if r['success'])
            print(f"  [{i+1}/{len(files)}] {label}进度: {ok}成功")

    t_total = time.perf_counter() - t_total_start
    vm.close()
    return results, t_total


def main():
    logging.basicConfig(level=logging.WARNING)

    files = sorted([f for f in os.listdir(PANEL1_DIR) if f.endswith('.fts')])
    print(f"Panel1: {len(files)}帧")

    # Python V2
    print(f"\n{'='*80}")
    print("Python V2 测试")
    print(f"{'='*80}")
    py_results, t_py = run_batch(files, VectorMatchV2Py, "PyV2")

    # C++ V2
    print(f"\n{'='*80}")
    print("C++ V2 测试 (OpenMP多线程)")
    print(f"{'='*80}")
    cpp_results, t_cpp = run_batch(files, VectorMatchV2Cpp, "CppV2")

    # ============================================================
    # 对比报告
    # ============================================================
    print(f"\n{'='*80}")
    print("Python V2 vs C++ V2 对比报告")
    print(f"{'='*80}")

    py_ok = [r for r in py_results if r['success']]
    cpp_ok = [r for r in cpp_results if r['success']]
    py_fail = [r for r in py_results if not r['success']]
    cpp_fail = [r for r in cpp_results if not r['success']]

    print(f"\n成功率:")
    print(f"  Python V2: {len(py_ok)}/{len(files)} ({len(py_ok)/len(files)*100:.1f}%)")
    print(f"  C++    V2: {len(cpp_ok)}/{len(files)} ({len(cpp_ok)/len(files)*100:.1f}%)")

    py_times = [r['t_solve'] for r in py_results]
    cpp_times = [r['t_solve'] for r in cpp_results]

    print(f"\n耗时统计:")
    print(f"  {'':12} {'PyV2':>8} {'CppV2':>8} {'Cpp/Py':>7}")
    print(f"  {'中位':12} {np.median(py_times):>7.2f}s {np.median(cpp_times):>7.2f}s {np.median(cpp_times)/np.median(py_times):>6.2f}x")
    print(f"  {'均值':12} {np.mean(py_times):>7.2f}s {np.mean(cpp_times):>7.2f}s {np.mean(cpp_times)/np.mean(py_times):>6.2f}x")
    print(f"  {'P25':12} {np.percentile(py_times,25):>7.2f}s {np.percentile(cpp_times,25):>7.2f}s")
    print(f"  {'P75':12} {np.percentile(py_times,75):>7.2f}s {np.percentile(cpp_times,75):>7.2f}s")
    print(f"  {'总耗时':12} {t_py:>7.1f}s {t_cpp:>7.1f}s {t_cpp/t_py:>6.2f}x")

    # 按滤镜对比
    print(f"\n按滤镜对比:")
    print(f"  {'滤镜':<10} {'Py中位':>8} {'Cpp中位':>8} {'Cpp/Py':>7} {'Py成功':>8} {'Cpp成功':>8}")
    for filt in ["Red", "Green", "Blue", "H-alpha", "Oiii"]:
        pyf = [r for r in py_results if r['filt'] == filt]
        cppf = [r for r in cpp_results if r['filt'] == filt]
        if not pyf:
            continue
        pyt = np.median([r['t_solve'] for r in pyf])
        cppt = np.median([r['t_solve'] for r in cppf])
        pyr = sum(1 for r in pyf if r['success']) / len(pyf) * 100
        cppr = sum(1 for r in cppf if r['success']) / len(cppf) * 100
        print(f"  {filt:<10} {pyt:>7.2f}s {cppt:>7.2f}s {cppt/pyt:>6.2f}x {pyr:>7.1f}% {cppr:>7.1f}%")

    # RMS对比
    if py_ok and cpp_ok:
        py_rms = [r['rms_px'] for r in py_ok]
        cpp_rms = [r['rms_px'] for r in cpp_ok]
        print(f"\nRMS对比:")
        print(f"  Python V2: 中位={np.median(py_rms):.3f}px 均值={np.mean(py_rms):.3f}px")
        print(f"  C++    V2: 中位={np.median(cpp_rms):.3f}px 均值={np.mean(cpp_rms):.3f}px")

    # 逐帧对比
    print(f"\n逐帧对比:")
    print(f"  {'#':>3} {'滤镜':<7} {'Py耗时':>7} {'Cpp耗时':>7} {'Cpp/Py':>6} {'PyRMS':>7} {'CppRMS':>7} {'Py匹配':>6} {'Cpp匹配':>6}")
    for i in range(len(files)):
        pyr = py_results[i]
        cppr = cpp_results[i]
        pyt = pyr['t_solve']
        cppt = cppr['t_solve']
        pyrms = f"{pyr['rms_px']:.3f}" if pyr['success'] else "---"
        cpprms = f"{cppr['rms_px']:.3f}" if cppr['success'] else "---"
        pym = f"{pyr['matched']}" if pyr['success'] else "---"
        cppm = f"{cppr['matched']}" if cppr['success'] else "---"
        ratio = f"{cppt/pyt:.2f}x" if pyt > 0 else "---"
        print(f"  {i+1:>3} {pyr['filt']:<7} {pyt:>6.2f}s {cppt:>6.2f}s {ratio:>6} {pyrms:>7} {cpprms:>7} {pym:>6} {cppm:>6}")

    # 失败帧
    if py_fail or cpp_fail:
        print(f"\n失败帧:")
        for i in range(len(files)):
            pyr = py_results[i]
            cppr = cpp_results[i]
            if not pyr['success'] or not cppr['success']:
                py_status = "OK" if pyr['success'] else "FAIL"
                cpp_status = "OK" if cppr['success'] else "FAIL"
                print(f"  {pyr['fname']}: Py={py_status} Cpp={cpp_status}")


if __name__ == '__main__':
    main()

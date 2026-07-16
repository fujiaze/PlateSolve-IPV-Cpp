"""V4.5 快速 4 帧测试

功能: 选 4 个代表性目标各 1 帧, 快速验证 sigma_d_px + 全第三星 + 比例阈值的效果
用途: 参数调参验证, 避免每次跑 55 帧抽样测试
输出: 控制台打印每帧结果

用法: py 快速4帧测试.py
"""
import os
import sys
import time
import re
import ctypes
import functools

print = functools.partial(print, flush=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except (OSError, FileNotFoundError):
        pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.WARNING)

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from vector_match_v4_5_cpp import Vm45Solver

_V45_DLL_PATH = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_5", "vector_match_v4_5.dll")

# 4 个代表性目标 (各取第 1 帧)
_TEST_TARGETS = ["NGC6302", "NGC7293", "LDN43", "Galaxy_Center_mosaic1"]


def _parse_ra_hms(s):
    s = str(s).strip()
    # 支持 "HH:MM:SS" 和 "HH MM SS" 两种格式
    parts = s.replace(":", " ").split()
    if len(parts) == 3:
        h, m, sec = parts
        return (int(h) + int(m) / 60.0 + float(sec) / 3600.0) * 15.0
    # 纯浮点度数
    return float(s)


def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith("-"):
        sign = -1.0
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    parts = s.replace(":", " ").split()
    if len(parts) == 3:
        d, m, sec = parts
        return sign * (int(d) + int(m) / 60.0 + float(sec) / 3600.0)
    return sign * float(s)


def find_test_frames():
    root = os.path.join(PROJECT_ROOT, "testdata")
    frames = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".fts", ".fit", ".fits"):
                continue
            for t in _TEST_TARGETS:
                if fn.startswith(t + "_") and t not in frames:
                    frames[t] = os.path.join(dirpath, fn)
                    break
    return frames


def solve_single(solver, fits_path):
    t0 = time.time()
    reader = ImageReader()
    img = reader.read(fits_path)
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl if fl > 0 else 0.0

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    ra0 = _parse_ra_hms(kw_dict["OBJCTRA"])
    dec0 = _parse_dec_dms(kw_dict["OBJCTDEC"])

    result = solver.solve_theta(
        image_path=fits_path, ra=ra0, dec=dec0,
        focal_length_mm=fl, pixel_size_um=ps)
    elapsed = time.time() - t0
    return result, elapsed, s0, fl, ps


def main():
    print("=" * 70)
    print("V4.5 快速 4 帧测试 (sigma_d_px=2.0, n_third=0, ratio_min=0.05)")
    print("=" * 70)

    frames = find_test_frames()
    print(f"找到 {len(frames)}/4 个目标的首帧:")
    for t in _TEST_TARGETS:
        p = frames.get(t)
        print(f"  {t}: {'✓ ' + os.path.basename(p) if p else '✗ 未找到'}")
    print()

    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    gaia_handle = gaia_client._handle
    if isinstance(gaia_handle, ctypes.c_void_p):
        gaia_handle = gaia_handle.value
    sdet_handle = star_detector._handle
    if isinstance(sdet_handle, ctypes.c_void_p):
        sdet_handle = sdet_handle.value

    solver = Vm45Solver(dll_path=_V45_DLL_PATH)
    solver.set_gaia_client(gaia_handle)
    solver.set_star_detector(sdet_handle)

    results = []
    for t in _TEST_TARGETS:
        p = frames.get(t)
        if not p:
            continue
        base = os.path.basename(p)
        print(f"\n[{t}] {base}")
        try:
            result, elapsed, s0, fl, ps = solve_single(solver, p)
            success = bool(result.success) and result.theta_snr > 5.0
            status = "OK  " if success else "FAIL"
            print(f"  {status} θ={result.theta_peak_deg:.2f}°  SNR={result.theta_snr:.2f}  "
                  f"n_passed={result.n_passed}  n_samples={result.n_samples}  "
                  f"t={elapsed:.2f}s  FOV={result.fov_diag_deg:.3f}°  "
                  f"s0={s0:.4f}\"/px  m_lim={result.m_lim_final:.2f}  n_gaia={result.n_gaia}")
            results.append((t, success, result.theta_snr, result.n_passed, elapsed))
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            results.append((t, False, 0, 0, 0))

    print("\n" + "=" * 70)
    n_ok = sum(1 for r in results if r[1])
    print(f"汇总: {n_ok}/{len(results)} 成功")
    print(f"{'目标':<25} {'成功':<6} {'SNR':<10} {'n_passed':<10} {'耗时(s)':<8}")
    for t, ok, snr, npass, t_s in results:
        print(f"{t:<25} {'✓' if ok else '✗':<6} {snr:<10.2f} {npass:<10} {t_s:<8.2f}")


if __name__ == "__main__":
    main()

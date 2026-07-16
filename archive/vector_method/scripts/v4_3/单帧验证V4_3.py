"""V4.3 单帧验证 - 验证 V43Solver 端到端求解

用法:
    python 单帧验证V4_3.py [fits_path]

默认使用 NGC247_T2 Lum 帧 (V4.2 测试中 RMS=1.2643px, matched=67, flip=2)
"""
import os
import sys
import json
import time
import math
import re

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MSYS2 MinGW DLL 路径
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
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

from astro_image_io import ImageReader
from v4_3.vector_match_v4_3_cpp import V43Solver


def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m / 60.0 + sec / 3600.0) * 15.0
    return float(s)


def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith("-"):
        sign = -1.0
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m / 60.0 + sec / 3600.0)
    return float(s)


def main():
    # 默认测试帧 (V4.2 测试中第一帧, 成功)
    default_fits = os.path.join(
        PROJECT_ROOT, "testdata", "lights",
        "NGC247_T2_flying_dutchman-20250902@090427-600S-Lum.fts")
    fits_path = sys.argv[1] if len(sys.argv) >= 2 else default_fits

    print(f"=== V4.3 单帧验证 ===")
    print(f"FITS: {fits_path}")
    if not os.path.exists(fits_path):
        print(f"错误: 文件不存在")
        return 1

    # 读取 FITS header
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
    obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
    if not obj_ra_str or not obj_dec_str:
        print(f"错误: FITS header 缺少 OBJCTRA/OBJCTDEC")
        return 1

    cra0 = _parse_ra_hms(obj_ra_str)
    cdec0 = _parse_dec_dms(obj_dec_str)

    print(f"图像: {w}×{h}")
    print(f"焦距: {fl}mm  像元: {ps}um  s0: {s0:.4f}\"/px")
    print(f"中心指向 (FITS): RA={cra0:.6f}° Dec={cdec0:.6f}°")
    print()

    # V4.3 求解
    print("创建 V43Solver...")
    t_start = time.time()
    with V43Solver() as solver:
        init_time = time.time() - t_start
        print(f"V43Solver 初始化耗时: {init_time:.3f}s")

        print(f"\n调用 vm43_solve()...")
        t_solve = time.time()
        try:
            result = solver.solve(
                image_path=fits_path,
                ra=cra0, dec=cdec0,
                focal_length_mm=fl, pixel_size_um=ps,
            )
        except Exception as e:
            print(f"求解异常: {e}")
            import traceback
            traceback.print_exc()
            return 1
        solve_time = time.time() - t_solve

    print(f"\n求解耗时: {solve_time:.3f}s")
    print(f"\n{'=' * 60}")
    print(f"V4.3 求解结果:")
    print(f"{'=' * 60}")
    print(f"success: {result['success']}")
    if not result["success"]:
        print(f"error: {result.get('error', '(空)')}")
        return 1

    print(f"\n[WCS 参数]")
    print(f"  CD = [{result['cd'][0]:.8e}, {result['cd'][1]:.8e};")
    print(f"        {result['cd'][2]:.8e}, {result['cd'][3]:.8e}]")
    print(f"  CRVAL = [{result['crval'][0]:.6f}°, {result['crval'][1]:.6f}°]")
    print(f"  CRPIX = [{result['crpix'][0]:.2f}, {result['crpix'][1]:.2f}]")
    print(f"  SIP order = {result['sip_order']}")

    print(f"\n[精度指标]")
    print(f"  RMS = {result['rms_px']:.4f} px ({result['rms_arcsec']:.4f}\")")
    print(f"  S_robust = {result['s_robust']:.4f}\"")
    print(f"  matched_count = {result['matched_count']}")
    print(f"  n_inliers = {result['n_inliers']}")
    print(f"  n_iters = {result['n_iters']}")
    print(f"  irm_converged = {result['irm_converged']}")

    print(f"\n[变换参数]")
    print(f"  s0 = {result['s0']:.4f}\"/px")
    print(f"  s = {result['s']:.6f}")
    print(f"  θ = {result['theta']:.6f} rad ({math.degrees(result['theta']):.4f}°)")
    print(f"  tx = {result['tx']:.4f}\"  ty = {result['ty']:.4f}\"")
    print(f"  flip_mode = {result['flip_mode']}")
    print(f"  rotation_deg = {result['rotation_deg']:.4f}°")
    print(f"  center = RA {result['center_ra']:.6f}°, Dec {result['center_dec']:.6f}°")

    print(f"\n[调试信息]")
    print(f"  θ SNR = {result['theta_snr']:.2f}")
    print(f"  θ peak = {result['theta_peak_deg']:.4f}°")
    print(f"  bayes lnK = {result['bayes_lnK']:.2f}")
    print(f"  triangle pass_ratio = {result['triangle_pass_ratio']:.3f}")

    print(f"\n[元数据]")
    print(f"  img = {result['img_width']}×{result['img_height']}")
    print(f"  FOV diag = {result['fov_diag_deg']:.3f}°")
    print(f"  m_lim = {result['m_lim_final']:.2f}")
    print(f"  n_gaia = {result['n_gaia_final']}")

    # 保存完整结果
    out_dir = os.path.join(
        PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "single_test")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n完整结果已保存: {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

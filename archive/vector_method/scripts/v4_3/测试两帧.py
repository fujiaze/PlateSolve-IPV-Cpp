"""测试 LDN43@022502 和 NGC7293@081055 两帧 (Bug 4 修复后)

功能:
    对 Bug 1-3 修复后仍异常的 2 个 Type3 帧重新求解, 验证 Bug 4 修复效果

输出:
    lib/plate_solve/logs/v4_3/regression_test/two_frames_after_bug4/
"""
import os
import sys
import json
import time

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

import re
import numpy as np
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_3.vector_match_v4_3_cpp import V43Solver


_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3",
    "regression_test", "two_frames_after_bug4")

TEST_FRAMES = [
    "lights/LDN43_LRGBH_flying_dutchman-20250520@022502-600S-Lum.fts",
    "lights/NGC7293_T2_HO_flying_dutchman-20250706@081055-1200S-H-alpha.fts",
]


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


def solve_frame(fits_path, solver):
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
    cra0 = _parse_ra_hms(obj_ra_str)
    cdec0 = _parse_dec_dms(obj_dec_str)

    base = os.path.splitext(os.path.basename(fits_path))[0]
    log_dir = os.path.join(_OUTPUT_DIR, "frames", base)

    t0 = time.time()
    result = solver.solve(
        image_path=fits_path,
        ra=cra0, dec=cdec0,
        focal_length_mm=fl, pixel_size_um=ps,
        log_dir=log_dir,
    )
    dt = time.time() - t0

    return {
        "filename": os.path.basename(fits_path),
        "success": result.get("success", False),
        "matched": int(result.get("matched_count", 0)),
        "rms_px": round(float(result.get("rms_px", 0.0)), 4),
        "s_robust": round(float(result.get("s_robust", 0.0)), 4),
        "n_iters": int(result.get("n_iters", 0)),
        "irm_converged": bool(result.get("irm_converged", False)),
        "bayes_lnK": round(float(result.get("bayes_lnK", 0.0)), 2),
        "triangle_pass_ratio": round(float(result.get("triangle_pass_ratio", 0.0)), 3),
        "solve_time_s": round(dt, 3),
    }


def main():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print("=== Bug 4 修复后两帧验证 ===\n")

    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(gaia_client=gaia_client, star_detector=star_detector)

    results = []
    for rel in TEST_FRAMES:
        fits_path = os.path.join(PROJECT_ROOT, "testdata", rel)
        if not os.path.exists(fits_path):
            print(f"[跳过] 文件不存在: {fits_path}")
            continue
        print(f"测试: {os.path.basename(fits_path)}")
        r = solve_frame(fits_path, solver)
        results.append(r)
        print(f"  → success={r['success']} matched={r['matched']} "
              f"RMS={r['rms_px']:.4f}px lnK={r['bayes_lnK']} "
              f"tri={r['triangle_pass_ratio']} iters={r['n_iters']} "
              f"conv={r['irm_converged']} 耗时={r['solve_time_s']}s\n")

    solver.close()
    gaia_client.close()
    star_detector.close()

    # 保存结果
    out_json = os.path.join(_OUTPUT_DIR, "two_frames_result.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 汇总
    print("=" * 70)
    print("汇总 (Bug 4 修复前 → 修复后)")
    print("=" * 70)
    before = {
        "LDN43_LRGBH_flying_dutchman-20250520@022502-600S-Lum.fts":
            {"matched": 61, "rms_px": 0.5967, "lnK": 0.0, "tri": 0.0},
        "NGC7293_T2_HO_flying_dutchman-20250706@081055-1200S-H-alpha.fts":
            {"matched": 19, "rms_px": 0.6898, "lnK": 0.0, "tri": 0.0},
    }
    for r in results:
        b = before.get(r["filename"], {})
        lnk_fix = "✓ 已修复" if (b.get("lnK", 0) == 0 and r["bayes_lnK"] > 0) else "未变化"
        print(f"\n{r['filename']}:")
        print(f"  matched: {b.get('matched', '?')} → {r['matched']}")
        print(f"  RMS:     {b.get('rms_px', '?')} → {r['rms_px']}px")
        print(f"  lnK:     {b.get('lnK', '?')} → {r['bayes_lnK']}  {lnk_fix}")
        print(f"  tri:     {b.get('tri', '?')} → {r['triangle_pass_ratio']}")
        print(f"  收敛: {r['irm_converged']} (iters={r['n_iters']})")

    print(f"\n结果已保存: {out_json}")


if __name__ == "__main__":
    main()

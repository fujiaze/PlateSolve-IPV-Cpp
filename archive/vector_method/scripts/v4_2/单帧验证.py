"""V4.2 单帧验证 - 6 帧（3 失败 + 3 成功）对比 V4.1 vs V4.2

功能:
    对 6 帧（3 个 V4.1 失败帧 + 3 个 V4.1 成功帧）分别用 V4.1 和 V4.2 求解,
    输出对比表格（success/RMS/matched/耗时/validated）,
    保存结果到 logs/v4_2/single_frame/v42_vs_v41_comparison.json

用途:
    验证 V4.2 模块化管线相对 V4.1 的改进效果（失败帧恢复 + 成功帧保持）

帧选择:
    失败帧（V4.1 无法解析）:
      - NGC55_T3_Oiii_042902
      - Victory_mosaic1_054309_Green
      - Victory_mosaic2_062533_Lum
    成功帧:
      - NGC55_T3_Oiii_025221
      - Victory_mosaic2_062145
      - Victory_mosaic1_055047

用法:
    python 单帧验证.py
"""
import os
import sys
import json
import time
import traceback

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MSYS2 MinGW DLL 路径（vector_matcher.dll 依赖 libwinpthread-1.dll）
os.environ["PATH"] = r"C:\msys64\mingw64\bin;" + os.environ.get("PATH", "")

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_1"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import numpy as np

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("V4.2单帧验证")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from vector_match_v4_1_cpp import VectorMatchV4_1Cpp
from v4_2.pipeline import V42Pipeline


# ============================================================================
# 工具函数
# ============================================================================

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


def _read_fits_header(fits_path):
    """读取 FITS 头关键字, 返回 (ra, dec, focal_length, pixel_size, width, height, exptime)"""
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    exptime = getattr(img.metadata.observation, "exptime", 1.0)

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
    obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
    cra0 = _parse_ra_hms(obj_ra_str)
    cdec0 = _parse_dec_dms(obj_dec_str)
    return cra0, cdec0, fl, ps, w, h, exptime


# ============================================================================
# V4.1 求解（使用 VectorMatchV4_1Cpp）
# ============================================================================

def solve_v41(fits_path, solver, frame_name):
    """V4.1 求解单帧

    Returns:
        dict: {success, rms_px, matched, time_s, validated, lnK}
    """
    print(f"  [V4.1] 求解中...")
    t_start = time.time()
    try:
        cra0, cdec0, fl, ps, w, h, exptime = _read_fits_header(fits_path)

        reader = ImageReader()
        img = reader.read(fits_path)
        detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        det = detector.detect_ex(img.data)
        n_detected = len(det.x)
        if n_detected < 5:
            return {"success": False, "rms_px": None, "matched": 0,
                    "time_s": round(time.time() - t_start, 3),
                    "validated": False, "lnK": None,
                    "error": "too_few_stars"}

        wcs_json = os.path.join(_OUTPUT_DIR, f"v41_wcs_{frame_name}.json")
        result = solver.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            cra0, cdec0, fl, ps, w, h,
            wcs_out=wcs_json,
            exptime=exptime,
            img_n_target=50,
            gaia_density_ratio=1.5,
            gaia_query_radius_factor=0.55,
            verbose=False,
        )
        elapsed = round(time.time() - t_start, 3)

        if not result:
            return {"success": False, "rms_px": None, "matched": 0,
                    "time_s": elapsed, "validated": False, "lnK": None,
                    "error": "fail_solve"}

        # 读取 WCS JSON 获取 SIP RMS
        rms_px = float(getattr(result, "sip_rms_px", result.rms_px))
        if os.path.exists(wcs_json):
            try:
                with open(wcs_json, "r", encoding="utf-8") as f:
                    wcs = json.load(f)
                rms_px = float(wcs.get("RMS_PX", rms_px))
            except Exception:
                pass

        return {
            "success": True,
            "rms_px": round(rms_px, 4),
            "matched": int(result.matched_count),
            "time_s": elapsed,
            "validated": bool(getattr(result, "validated", False)),
            "lnK": float(getattr(result, "bayes_lnK", 0.0)) or None,
        }
    except Exception as e:
        return {"success": False, "rms_px": None, "matched": 0,
                "time_s": round(time.time() - t_start, 3),
                "validated": False, "lnK": None,
                "error": str(e)[:200]}


# ============================================================================
# V4.2 求解（使用 V42Pipeline）
# ============================================================================

def solve_v42(fits_path, pipeline, frame_name):
    """V4.2 求解单帧

    Returns:
        dict: {success, rms_px, matched, time_s, validated, lnK}
    """
    print(f"  [V4.2] 求解中...")
    t_start = time.time()
    try:
        cra0, cdec0, fl, ps, w, h, _ = _read_fits_header(fits_path)

        # V4.2 每帧独立日志目录（支持断点续跑）
        frame_log_dir = os.path.join(
            PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2", "single_frame", frame_name)

        result = pipeline.solve(
            image_path=fits_path,
            ra=cra0, dec=cdec0,
            focal_length_mm=fl, pixel_size_um=ps,
            log_dir=frame_log_dir,
            resume=True,
        )
        elapsed = round(time.time() - t_start, 3)

        if not result.get("success", False):
            return {"success": False, "rms_px": None, "matched": 0,
                    "time_s": elapsed, "validated": False, "lnK": None,
                    "error": result.get("error", "unknown")[:200]}

        return {
            "success": True,
            "rms_px": round(float(result.get("rms_px", 0.0)), 4),
            "matched": int(result.get("matched_count", 0)),
            "time_s": elapsed,
            "validated": bool(result.get("validated", False)),
            "lnK": float(result.get("bayes_lnK", 0.0)) or None,
        }
    except Exception as e:
        return {"success": False, "rms_px": None, "matched": 0,
                "time_s": round(time.time() - t_start, 3),
                "validated": False, "lnK": None,
                "error": str(e)[:200]}


# ============================================================================
# 帧定义
# ============================================================================

_FRAMES = [
    # 失败帧
    ("NGC55_T3_Oiii_042902", "失败",
     os.path.join(PROJECT_ROOT, "testdata", "lights",
                  "NGC55_T3_flying_dutchman-20250915@042902-1200S-Oiii.fts")),
    ("Victory_mosaic1_054309_Green", "失败",
     os.path.join(PROJECT_ROOT, "testdata", "lights",
                  "Victory_Nebula_mosaic1_flying_dutchman-20250206@054309-180S-Green.fts")),
    ("Victory_mosaic2_062533_Lum", "失败",
     os.path.join(PROJECT_ROOT, "testdata", "lights",
                  "Victory_Nebula_mosaic2_flying_dutchman-20250205@062533-180S-Lum.fts")),
    # 成功帧
    ("NGC55_T3_Oiii_025221", "成功",
     os.path.join(PROJECT_ROOT, "testdata", "lights",
                  "NGC55_T3_flying_dutchman-20250915@025221-1200S-Oiii.fts")),
    ("Victory_mosaic2_062145", "成功",
     os.path.join(PROJECT_ROOT, "testdata", "lights",
                  "Victory_Nebula_mosaic2_flying_dutchman-20250205@062145-180S-Lum.fts")),
    ("Victory_mosaic1_055047", "成功",
     os.path.join(PROJECT_ROOT, "testdata", "lights",
                  "Victory_Nebula_mosaic1_flying_dutchman-20250206@055047-180S-Blue.fts")),
]

_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2", "single_frame")
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "v42_vs_v41_comparison.json")


# ============================================================================
# 主流程
# ============================================================================

def main():
    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # 断点续跑: 加载已有结果
    frames_results = []
    done_names = set()
    if os.path.exists(_RESULT_JSON):
        try:
            with open(_RESULT_JSON, "r", encoding="utf-8") as f:
                old = json.load(f)
            frames_results = old.get("frames", [])
            done_names = {fr["frame"] for fr in frames_results}
            print(f"断点续跑: 已完成 {len(done_names)}/{len(_FRAMES)} 帧")
        except Exception:
            pass

    # 创建 solver 实例（复用）
    print("创建 V4.1 solver (VectorMatchV4_1Cpp)...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    solver_v41 = VectorMatchV4_1Cpp(gaia_dir, db_type=1)

    print("创建 V4.2 pipeline (V42Pipeline)...")
    gaia_client = GaiaClientPy(gaia_dir, db_type=0)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
    pipeline = V42Pipeline(
        dll_dir=dll_dir,
        gaia_client=gaia_client,
        star_detector=star_detector,
    )

    # 逐帧求解
    for frame_name, label, fits_path in _FRAMES:
        if frame_name in done_names:
            print(f"\n跳过（已完成）: {frame_name} [{label}]")
            continue

        print(f"\n{'=' * 70}")
        print(f"  帧: {frame_name} [{label}]")
        print(f"  路径: {os.path.basename(fits_path)}")
        print(f"{'=' * 70}")

        if not os.path.exists(fits_path):
            print(f"  文件不存在, 跳过")
            frames_results.append({
                "frame": frame_name, "label": label,
                "v41": {"success": False, "error": "file_not_found"},
                "v42": {"success": False, "error": "file_not_found"},
            })
            continue

        v41_res = solve_v41(fits_path, solver_v41, frame_name)
        v42_res = solve_v42(fits_path, pipeline, frame_name)

        frames_results.append({
            "frame": frame_name,
            "label": label,
            "v41": v41_res,
            "v42": v42_res,
        })

        # 增量保存
        _save_results(frames_results)

        # 打印对比
        _print_frame_comparison(frame_name, label, v41_res, v42_res)

    solver_v41.close()
    pipeline.close()
    gaia_client.close()
    star_detector.close()

    # 汇总
    _save_results(frames_results, with_summary=True)
    _print_summary(frames_results)


def _save_results(frames_results, with_summary=False):
    """保存结果 JSON（增量保存）"""
    data = {"frames": frames_results}
    if with_summary:
        data["summary"] = _compute_summary(frames_results)
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _compute_summary(frames_results):
    """计算汇总统计"""
    v41_succ = [fr for fr in frames_results if fr["v41"].get("success")]
    v42_succ = [fr for fr in frames_results if fr["v42"].get("success")]
    v41_rms = [fr["v41"]["rms_px"] for fr in v41_succ if fr["v41"].get("rms_px") is not None]
    v42_rms = [fr["v42"]["rms_px"] for fr in v42_succ if fr["v42"].get("rms_px") is not None]
    v41_times = [fr["v41"]["time_s"] for fr in v41_succ]
    v42_times = [fr["v42"]["time_s"] for fr in v42_succ]
    return {
        "v41_success_rate": f"{len(v41_succ)}/{len(frames_results)}",
        "v42_success_rate": f"{len(v42_succ)}/{len(frames_results)}",
        "v41_median_rms": round(float(np.median(v41_rms)), 4) if v41_rms else None,
        "v42_median_rms": round(float(np.median(v42_rms)), 4) if v42_rms else None,
        "v41_median_time": round(float(np.median(v41_times)), 3) if v41_times else None,
        "v42_median_time": round(float(np.median(v42_times)), 3) if v42_times else None,
    }


def _print_frame_comparison(frame_name, label, v41, v42):
    """打印单帧对比"""
    def _fmt(v, fmt="{}"):
        return fmt.format(v) if v is not None else "N/A"

    v41_ok = "✓" if v41.get("success") else "✗"
    v42_ok = "✓" if v42.get("success") else "✗"
    print(f"\n  对比:")
    print(f"  {'版本':<8} {'成功':<4} {'RMS(px)':<10} {'matched':<10} {'耗时(s)':<10} {'validated':<10} {'lnK':<12}")
    print(f"  {'-' * 70}")
    print(f"  {'V4.1':<8} {v41_ok:<4} {_fmt(v41.get('rms_px'), '{:.4f}'):<10} "
          f"{v41.get('matched', 0):<10} {v41.get('time_s', 0):<10.3f} "
          f"{'-' if not v41.get('validated') else '✓':<10} {_fmt(v41.get('lnK'), '{:.2f}'):<12}")
    print(f"  {'V4.2':<8} {v42_ok:<4} {_fmt(v42.get('rms_px'), '{:.4f}'):<10} "
          f"{v42.get('matched', 0):<10} {v42.get('time_s', 0):<10.3f} "
          f"{'✓' if v42.get('validated') else '✗':<10} {_fmt(v42.get('lnK'), '{:.2f}'):<12}")


def _print_summary(frames_results):
    """打印汇总表"""
    summary = _compute_summary(frames_results)
    print(f"\n{'=' * 70}")
    print(f"  V4.1 vs V4.2 单帧验证汇总")
    print(f"{'=' * 70}")
    print(f"{'帧':<35} {'标签':<6} {'V4.1':<6} {'V4.2':<6} {'结论':<20}")
    print(f"{'-' * 70}")
    for fr in frames_results:
        v41_ok = fr["v41"].get("success", False)
        v42_ok = fr["v42"].get("success", False)
        if not v41_ok and v42_ok:
            status = "✓ 恢复成功"
        elif v41_ok and not v42_ok:
            status = "✗ 退化!"
        elif v41_ok and v42_ok:
            status = "✓ 都成功"
        else:
            status = "✗ 都失败"
        v41_str = "✓" if v41_ok else "✗"
        v42_str = "✓" if v42_ok else "✗"
        print(f"  {fr['frame'][:33]:<35} {fr.get('label', ''):<6} "
              f"{v41_str:<6} {v42_str:<6} {status:<20}")

    print(f"\n  汇总:")
    print(f"    V4.1 成功率: {summary['v41_success_rate']}")
    print(f"    V4.2 成功率: {summary['v42_success_rate']}")
    print(f"    V4.1 中位 RMS: {summary['v41_median_rms']}")
    print(f"    V4.2 中位 RMS: {summary['v42_median_rms']}")
    print(f"    V4.1 中位耗时: {summary['v41_median_time']}s")
    print(f"    V4.2 中位耗时: {summary['v42_median_time']}s")
    print(f"\n  结果保存: {_RESULT_JSON}")


if __name__ == "__main__":
    main()

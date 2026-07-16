"""V4.2 50 帧批量测试 - seed=42, 对比 V4.1

功能:
    从 testdata/lights 递归收集 FITS 文件, 随机采样 50 帧（seed=42, 与 V4.1 相同）
    对每帧用 V4.2 (V42Pipeline) 求解, 记录 success/RMS/matched/耗时/validated/lnK/sip_order
    与 V4.1 batch_test_50frames_seed42.json 对比

输出:
    - lib/plate_solve/logs/v4_2/batch_test/batch_test_50frames_seed42.json
    - lib/plate_solve/logs/v4_2/batch_test/summary.csv
    - lib/plate_solve/logs/v4_2/batch_test/v41_vs_v42_comparison.json

用法:
    python 批量测试V4_2.py [帧数] [随机种子]
    默认: 50 帧, 种子=42（与 V4.1 一致, 确保帧集合相同）
"""
import os
import sys
import json
import csv
import time
import math
import random
import re

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
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import numpy as np

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("V4.2批量测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_2.pipeline import V42Pipeline


# ============================================================================
# 输出目录
# ============================================================================
_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2", "batch_test")
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "batch_test_50frames_seed42.json")
_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "summary.csv")
_COMPARISON_JSON = os.path.join(_OUTPUT_DIR, "v41_vs_v42_comparison.json")
_V41_RESULT_JSON = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_1", "batch_test",
    "batch_test_50frames_seed42.json")


# ============================================================================
# 收集 FITS 文件（与 V4.1 完全一致的逻辑）
# ============================================================================

def collect_fits_files(root_dir, max_depth=3):
    """递归收集所有 FITS 文件(.fts/.fit/.fits)"""
    fits_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_depth = os.path.relpath(dirpath, root_dir).count(os.sep)
        if rel_depth >= max_depth:
            dirnames.clear()
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".fts", ".fit", ".fits"):
                fits_files.append(os.path.join(dirpath, fn))
    return fits_files


def parse_filename(path):
    """从文件名提取目标、滤镜、曝光等信息"""
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    info = {"filename": base, "path": path}

    m = re.search(r"-(\d+)S-", name)
    info["exposure_s"] = int(m.group(1)) if m else 0

    m = re.search(r"-([A-Za-z][A-Za-z0-9_-]*)$", name)
    info["filter"] = m.group(1) if m else "unknown"

    if "_" in name:
        info["target"] = name.split("_")[0]
    elif "-" in name:
        info["target"] = name.split("-")[0]
    else:
        info["target"] = "unknown"
    return info


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


# ============================================================================
# V4.2 单帧求解
# ============================================================================

def solve_single_frame_v42(fits_path, pipeline):
    """对单帧 FITS 进行 V4.2 WCS 求解

    Returns:
        dict: 包含 WCS 参数和质量指标的结果字典
    """
    result_info = parse_filename(fits_path)
    t_start = time.time()

    try:
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
            result_info["status"] = "fail_no_objctra"
            result_info["solve_time_s"] = round(time.time() - t_start, 3)
            return result_info
        cra0 = _parse_ra_hms(obj_ra_str)
        cdec0 = _parse_dec_dms(obj_dec_str)

        result_info["width"] = w
        result_info["height"] = h
        result_info["focallen"] = fl
        result_info["pixel_size"] = ps
        result_info["s0_arcsec_px"] = round(s0, 4)
        result_info["fits_center_ra"] = cra0
        result_info["fits_center_dec"] = cdec0

        # V4.2 每帧独立日志目录（帧名用作目录名, 支持断点续跑）
        frame_base = os.path.splitext(os.path.basename(fits_path))[0]
        frame_log_dir = os.path.join(_OUTPUT_DIR, "frames", frame_base)

        # V4.2 求解
        t_solve = time.time()
        result = pipeline.solve(
            image_path=fits_path,
            ra=cra0, dec=cdec0,
            focal_length_mm=fl, pixel_size_um=ps,
            log_dir=frame_log_dir,
            resume=True,
        )
        solve_time = time.time() - t_solve

        if not result.get("success", False):
            result_info["status"] = "fail_solve"
            result_info["solve_time_s"] = round(solve_time, 3)
            result_info["error"] = result.get("error", "unknown")[:200]
            return result_info

        result_info["status"] = "success"
        result_info["solve_time_s"] = round(solve_time, 3)
        result_info["flip_mode"] = int(result.get("flip_mode", -1))
        result_info["matched_count"] = int(result.get("matched_count", 0))
        result_info["rms_px"] = round(float(result.get("rms_px", 0.0)), 4)
        result_info["scale_arcsec_px"] = round(float(result.get("scale_arcsec_px", s0)), 4)
        result_info["rotation_deg"] = round(float(result.get("rotation_deg", 0.0)), 4)
        result_info["center_ra"] = float(result.get("center_ra", cra0))
        result_info["center_dec"] = float(result.get("center_dec", cdec0))
        result_info["sip_order"] = int(result.get("sip_order", 0))
        result_info["validated"] = bool(result.get("validated", False))
        result_info["bayes_lnK"] = round(float(result.get("bayes_lnK", 0.0)), 2)
        result_info["triangle_pass_ratio"] = round(float(result.get("triangle_pass_ratio", 0.0)), 3)
        result_info["total_time_s"] = round(time.time() - t_start, 3)

    except Exception as e:
        result_info["status"] = f"error: {str(e)[:100]}"
        result_info["total_time_s"] = round(time.time() - t_start, 3)
        logger.error(f"  异常: {fits_path}: {e}")

    return result_info


# ============================================================================
# V4.1 结果加载与对比
# ============================================================================

def load_v41_results():
    """加载 V4.1 batch test 结果, 返回 {filename: result_dict}"""
    if not os.path.exists(_V41_RESULT_JSON):
        print(f"  警告: V4.1 结果文件不存在: {_V41_RESULT_JSON}")
        return {}
    try:
        with open(_V41_RESULT_JSON, "r", encoding="utf-8") as f:
            v41_list = json.load(f)
        return {r["filename"]: r for r in v41_list}
    except Exception as e:
        print(f"  警告: 加载 V4.1 结果失败: {e}")
        return {}


def build_comparison(v42_results, v41_map):
    """构建 V4.1 vs V4.2 对比"""
    comparison = {"frames": [], "summary": {}}
    for r in v42_results:
        fn = r["filename"]
        v41 = v41_map.get(fn, {})
        v41_succ = v41.get("status") == "success"
        v42_succ = r.get("status") == "success"

        entry = {
            "filename": fn,
            "target": r.get("target", ""),
            "filter": r.get("filter", ""),
            "v41": {
                "success": v41_succ,
                "rms_px": v41.get("rms_px") if v41_succ else None,
                "matched": v41.get("matched_count", 0) if v41_succ else 0,
                "time_s": v41.get("solve_time_s", 0) if v41_succ else 0,
            },
            "v42": {
                "success": v42_succ,
                "rms_px": r.get("rms_px") if v42_succ else None,
                "matched": r.get("matched_count", 0) if v42_succ else 0,
                "time_s": r.get("solve_time_s", 0) if v42_succ else 0,
                "validated": r.get("validated", False) if v42_succ else False,
                "lnK": r.get("bayes_lnK", 0) if v42_succ else 0,
                "sip_order": r.get("sip_order", 0) if v42_succ else 0,
            },
        }
        comparison["frames"].append(entry)

    # 汇总
    v41_succ_list = [e for e in comparison["frames"] if e["v41"]["success"]]
    v42_succ_list = [e for e in comparison["frames"] if e["v42"]["success"]]
    v41_rms = [e["v41"]["rms_px"] for e in v41_succ_list if e["v41"]["rms_px"] is not None]
    v42_rms = [e["v42"]["rms_px"] for e in v42_succ_list if e["v42"]["rms_px"] is not None]
    v41_times = [e["v41"]["time_s"] for e in v41_succ_list]
    v42_times = [e["v42"]["time_s"] for e in v42_succ_list]
    v41_matched = [e["v41"]["matched"] for e in v41_succ_list]
    v42_matched = [e["v42"]["matched"] for e in v42_succ_list]
    n_total = len(comparison["frames"])

    comparison["summary"] = {
        "n_total": n_total,
        "v41_success_rate": f"{len(v41_succ_list)}/{n_total}",
        "v42_success_rate": f"{len(v42_succ_list)}/{n_total}",
        "v41_success_pct": round(100.0 * len(v41_succ_list) / max(n_total, 1), 1),
        "v42_success_pct": round(100.0 * len(v42_succ_list) / max(n_total, 1), 1),
        "v41_median_rms_px": round(float(np.median(v41_rms)), 4) if v41_rms else None,
        "v42_median_rms_px": round(float(np.median(v42_rms)), 4) if v42_rms else None,
        "v41_median_time_s": round(float(np.median(v41_times)), 3) if v41_times else None,
        "v42_median_time_s": round(float(np.median(v42_times)), 3) if v42_times else None,
        "v41_median_matched": int(np.median(v41_matched)) if v41_matched else 0,
        "v42_median_matched": int(np.median(v42_matched)) if v42_matched else 0,
    }
    return comparison


# ============================================================================
# 主流程
# ============================================================================

def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) >= 2 else 50
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 42

    print(f"=== V4.2 批量测试 ===")
    print(f"帧数: {n_frames}  随机种子: {seed}")

    # 收集 FITS 文件（与 V4.1 完全一致的逻辑）
    lights_dir = os.path.join(PROJECT_ROOT, "testdata", "lights")
    all_fits = collect_fits_files(lights_dir, max_depth=3)
    print(f"找到 FITS 文件: {len(all_fits)} 个")

    if len(all_fits) < n_frames:
        print(f"警告: 只有 {len(all_fits)} 个文件, 少于请求的 {n_frames} 个, 全部使用")
        selected = all_fits
    else:
        random.seed(seed)
        selected = random.sample(all_fits, n_frames)
    print(f"随机抽取: {len(selected)} 个")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # 断点续跑: 加载已有结果
    results = []
    done_paths = set()
    if os.path.exists(_RESULT_JSON):
        try:
            with open(_RESULT_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_paths = {r["path"] for r in results if "path" in r}
            print(f"断点续跑: 已完成 {len(done_paths)}/{len(selected)} 帧")
        except Exception:
            results = []

    # 创建 V4.2 pipeline（注入 GaiaClientPy 和 StarDetector, db_type=1 与 V4.1 一致）
    print("创建 V4.2 pipeline (V42Pipeline)...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
    pipeline = V42Pipeline(
        dll_dir=dll_dir,
        gaia_client=gaia_client,
        star_detector=star_detector,
    )

    # 逐帧测试
    for i, fits_path in enumerate(selected, 1):
        base = os.path.basename(fits_path)
        if fits_path in done_paths:
            print(f"[{i}/{len(selected)}] 跳过（已完成）: {base}")
            continue

        print(f"\n[{i}/{len(selected)}] {base}")

        info = solve_single_frame_v42(fits_path, pipeline)
        results.append(info)

        # 增量保存
        with open(_RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 打印简要结果
        if info["status"] == "success":
            print(f"  → 成功: mode={info['flip_mode']} n={info['matched_count']} "
                  f"RMS={info['rms_px']:.3f}px SIP={info['sip_order']} "
                  f"validated={info['validated']} lnK={info['bayes_lnK']} "
                  f"耗时={info['solve_time_s']}s")
        else:
            print(f"  → {info['status']}")

    pipeline.close()
    gaia_client.close()
    star_detector.close()

    # 保存 JSON
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {_RESULT_JSON}")

    # 保存 CSV
    _save_csv(results)
    print(f"CSV 已保存: {_SUMMARY_CSV}")

    # 加载 V4.1 结果并对比
    v41_map = load_v41_results()
    if v41_map:
        comparison = build_comparison(results, v41_map)
        with open(_COMPARISON_JSON, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"对比 JSON 已保存: {_COMPARISON_JSON}")

    # 汇总统计
    _print_summary(results, v41_map)


def _save_csv(results):
    """保存 CSV"""
    csv_fields = [
        "filename", "target", "filter", "exposure_s",
        "width", "height", "focallen", "pixel_size", "s0_arcsec_px",
        "fits_center_ra", "fits_center_dec",
        "status", "flip_mode", "matched_count", "rms_px",
        "scale_arcsec_px", "rotation_deg",
        "center_ra", "center_dec",
        "sip_order", "validated", "bayes_lnK", "triangle_pass_ratio",
        "solve_time_s", "total_time_s",
    ]
    with open(_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _print_summary(results, v41_map):
    """打印汇总统计 + V4.1 vs V4.2 对比"""
    n_total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_fail = n_total - n_success

    print(f"\n{'=' * 70}")
    print(f"  V4.2 批量测试汇总")
    print(f"{'=' * 70}")
    print(f"总帧数: {n_total}")
    print(f"成功: {n_success} ({100 * n_success / max(n_total, 1):.1f}%)")
    print(f"失败: {n_fail} ({100 * n_fail / max(n_total, 1):.1f}%)")

    if n_success > 0:
        succ = [r for r in results if r["status"] == "success"]
        rms_list = [r["rms_px"] for r in succ]
        matched_list = [r["matched_count"] for r in succ]
        time_list = [r["solve_time_s"] for r in succ]
        lnK_list = [r["bayes_lnK"] for r in succ]

        print(f"\n成功帧统计:")
        print(f"  RMS(px): 中位={np.median(rms_list):.3f} 均值={np.mean(rms_list):.3f} "
              f"最小={np.min(rms_list):.3f} 最大={np.max(rms_list):.3f}")
        print(f"  匹配对数: 中位={np.median(matched_list):.0f} 均值={np.mean(matched_list):.0f} "
              f"最小={np.min(matched_list)} 最大={np.max(matched_list)}")
        print(f"  求解耗时(s): 中位={np.median(time_list):.3f} 均值={np.mean(time_list):.3f} "
              f"最小={np.min(time_list):.3f} 最大={np.max(time_list):.3f}")
        print(f"  贝叶斯 lnK: 中位={np.median(lnK_list):.1f} 最小={np.min(lnK_list):.1f}")

    # 失败帧列表
    if n_fail > 0:
        print(f"\n失败帧:")
        for r in results:
            if r["status"] != "success":
                print(f"  {r['filename']}: {r['status']}")

    # V4.1 vs V4.2 对比表
    if v41_map:
        print(f"\n{'=' * 70}")
        print(f"  V4.1 vs V4.2 对比 (相同 50 帧, seed=42)")
        print(f"{'=' * 70}")

        v41_succ = sum(1 for r in results if v41_map.get(r["filename"], {}).get("status") == "success")
        v42_succ = n_success
        v41_rms = [v41_map[r["filename"]]["rms_px"] for r in results
                   if r["filename"] in v41_map and v41_map[r["filename"]].get("status") == "success"]
        v42_rms = [r["rms_px"] for r in results if r["status"] == "success"]
        v41_times = [v41_map[r["filename"]]["solve_time_s"] for r in results
                     if r["filename"] in v41_map and v41_map[r["filename"]].get("status") == "success"]
        v42_times = [r["solve_time_s"] for r in results if r["status"] == "success"]
        v41_matched = [v41_map[r["filename"]]["matched_count"] for r in results
                       if r["filename"] in v41_map and v41_map[r["filename"]].get("status") == "success"]
        v42_matched = [r["matched_count"] for r in results if r["status"] == "success"]

        print(f"{'指标':<20} {'V4.1':<15} {'V4.2':<15} {'变化':<15}")
        print(f"{'-' * 65}")
        print(f"{'成功率':<20} {f'{v41_succ}/{n_total} ({100*v41_succ/n_total:.1f}%)':<15} "
              f"{f'{v42_succ}/{n_total} ({100*v42_succ/n_total:.1f}%)':<15} "
              f"{f'{v42_succ - v41_succ:+d} 帧':<15}")

        v41_med_rms = float(np.median(v41_rms)) if v41_rms else 0
        v42_med_rms = float(np.median(v42_rms)) if v42_rms else 0
        print(f"{'中位 RMS(px)':<20} {v41_med_rms:<15.4f} {v42_med_rms:<15.4f} "
              f"{f'{v42_med_rms - v41_med_rms:+.4f}':<15}")

        v41_med_time = float(np.median(v41_times)) if v41_times else 0
        v42_med_time = float(np.median(v42_times)) if v42_times else 0
        print(f"{'中位耗时(s)':<20} {v41_med_time:<15.3f} {v42_med_time:<15.3f} "
              f"{f'{v42_med_time - v41_med_time:+.3f}':<15}")

        v41_med_m = int(np.median(v41_matched)) if v41_matched else 0
        v42_med_m = int(np.median(v42_matched)) if v42_matched else 0
        print(f"{'中位 matched':<20} {v41_med_m:<15} {v42_med_m:<15} "
              f"{f'{v42_med_m - v41_med_m:+d}':<15}")

        # Spec 指标达成情况
        print(f"\n  Spec 指标达成:")
        print(f"    成功率 ≥ 96%: {'✅' if 100 * v42_succ / n_total >= 96 else '❌'} "
              f"(实际 {100 * v42_succ / n_total:.1f}%)")
        print(f"    中位 RMS ≤ 1.7px: {'✅' if v42_med_rms <= 1.7 else '❌'} "
              f"(实际 {v42_med_rms:.4f}px)")
        print(f"    中位耗时 ≤ 0.1s: {'✅' if v42_med_time <= 0.1 else '❌'} "
              f"(实际 {v42_med_time:.3f}s)")


if __name__ == "__main__":
    main()

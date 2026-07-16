"""V4.3 50 帧批量测试 - seed=42, 对比 V4.2

功能:
    从 testdata/lights 递归收集 FITS 文件, 随机采样 50 帧（seed=42, 与 V4.1/V4.2 相同）
    对每帧用 V4.3 (V43Solver, 单一 DLL + IRM 闭环) 求解
    记录 success/RMS/matched/耗时/S_robust/n_iters/irm_converged/sip_order/lnK
    与 V4.2 batch_test_50frames_seed42.json 对比

输出:
    - lib/plate_solve/logs/v4_3/batch_test/batch_test_50frames_seed42.json
    - lib/plate_solve/logs/v4_3/batch_test/summary.csv
    - lib/plate_solve/logs/v4_3/batch_test/v42_vs_v43_comparison.json

用法:
    python 批量测试V4_3.py [帧数] [随机种子]
    默认: 50 帧, 种子=42（与 V4.1/V4.2 一致, 确保帧集合相同）

Spec 验收指标:
    1. 成功率 ≥ 96%
    2. 中位 RMS ≤ 1.7 px
    3. 中位耗时 ≤ 0.1 s （单一 DLL 消除 5 次 ctypes 边界 + JSON 序列化开销）
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

# MSYS2 MinGW DLL 路径 (V4.3 DLL 依赖 libwinpthread-1.dll / libgcc_s_seh / libstdc++ / libgomp)
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

import numpy as np

import logging
# 调到 WARNING, 避免每帧大量 INFO 日志淹没批量测试输出
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("V4.3批量测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_3.vector_match_v4_3_cpp import V43Solver


# ============================================================================
# 输出目录
# ============================================================================
_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "batch_test")
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "batch_test_50frames_seed42.json")
_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "summary.csv")
_COMPARISON_JSON = os.path.join(_OUTPUT_DIR, "v42_vs_v43_comparison.json")
_V42_RESULT_JSON = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2", "batch_test",
    "batch_test_50frames_seed42.json")


# ============================================================================
# 收集 FITS 文件 (与 V4.1/V4.2 完全一致的逻辑, 保证帧集合相同)
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
# V4.3 单帧求解
# ============================================================================

def solve_single_frame_v43(fits_path, solver):
    """对单帧 FITS 进行 V4.3 WCS 求解

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

        # V4.3 每帧独立日志目录 (支持断点续跑; 与 V4.2 路径风格一致)
        frame_base = os.path.splitext(os.path.basename(fits_path))[0]
        frame_log_dir = os.path.join(_OUTPUT_DIR, "frames", frame_base)

        # V4.3 求解 (单一 DLL, 一次 vm43_solve 调用完成全流程)
        t_solve = time.time()
        result = solver.solve(
            image_path=fits_path,
            ra=cra0, dec=cdec0,
            focal_length_mm=fl, pixel_size_um=ps,
            log_dir=frame_log_dir,
        )
        solve_time = time.time() - t_solve

        if not result.get("success", False):
            result_info["status"] = "fail_solve"
            result_info["solve_time_s"] = round(solve_time, 3)
            result_info["error"] = (result.get("error", "unknown") or "")[:200]
            return result_info

        result_info["status"] = "success"
        result_info["solve_time_s"] = round(solve_time, 3)
        result_info["flip_mode"] = int(result.get("flip_mode", -1))
        result_info["matched_count"] = int(result.get("matched_count", 0))
        result_info["rms_px"] = round(float(result.get("rms_px", 0.0)), 4)
        result_info["rms_arcsec"] = round(float(result.get("rms_arcsec", 0.0)), 4)
        result_info["scale_arcsec_px"] = round(float(result.get("scale_arcsec_px", s0)), 4)
        result_info["rotation_deg"] = round(float(result.get("rotation_deg", 0.0)), 4)
        result_info["center_ra"] = float(result.get("center_ra", cra0))
        result_info["center_dec"] = float(result.get("center_dec", cdec0))
        result_info["sip_order"] = int(result.get("sip_order", 0))

        # V4.3 特有指标
        result_info["s_robust"] = round(float(result.get("s_robust", 0.0)), 4)
        result_info["n_inliers"] = int(result.get("n_inliers", 0))
        result_info["n_iters"] = int(result.get("n_iters", 0))
        result_info["irm_converged"] = bool(result.get("irm_converged", False))

        # 验证信息 (与 V4.2 兼容字段)
        result_info["bayes_lnK"] = round(float(result.get("bayes_lnK", 0.0)), 2)
        result_info["triangle_pass_ratio"] = round(float(result.get("triangle_pass_ratio", 0.0)), 3)
        result_info["theta_snr"] = round(float(result.get("theta_snr", 0.0)), 2)

        result_info["total_time_s"] = round(time.time() - t_start, 3)

    except Exception as e:
        result_info["status"] = f"error: {str(e)[:100]}"
        result_info["total_time_s"] = round(time.time() - t_start, 3)
        logger.error(f"  异常: {fits_path}: {e}")

    return result_info


# ============================================================================
# V4.2 结果加载与对比
# ============================================================================

def load_v42_results():
    """加载 V4.2 batch test 结果, 返回 {filename: result_dict}"""
    if not os.path.exists(_V42_RESULT_JSON):
        print(f"  警告: V4.2 结果文件不存在: {_V42_RESULT_JSON}")
        return {}
    try:
        with open(_V42_RESULT_JSON, "r", encoding="utf-8") as f:
            v42_list = json.load(f)
        return {r["filename"]: r for r in v42_list}
    except Exception as e:
        print(f"  警告: 加载 V4.2 结果失败: {e}")
        return {}


def build_comparison(v43_results, v42_map):
    """构建 V4.2 vs V4.3 对比"""
    comparison = {"frames": [], "summary": {}}
    for r in v43_results:
        fn = r["filename"]
        v42 = v42_map.get(fn, {})
        v42_succ = v42.get("status") == "success"
        v43_succ = r.get("status") == "success"

        entry = {
            "filename": fn,
            "target": r.get("target", ""),
            "filter": r.get("filter", ""),
            "v42": {
                "success": v42_succ,
                "rms_px": v42.get("rms_px") if v42_succ else None,
                "matched": v42.get("matched_count", 0) if v42_succ else 0,
                "time_s": v42.get("solve_time_s", 0) if v42_succ else 0,
                "sip_order": v42.get("sip_order", 0) if v42_succ else 0,
                "lnK": v42.get("bayes_lnK", 0) if v42_succ else 0,
            },
            "v43": {
                "success": v43_succ,
                "rms_px": r.get("rms_px") if v43_succ else None,
                "matched": r.get("matched_count", 0) if v43_succ else 0,
                "time_s": r.get("solve_time_s", 0) if v43_succ else 0,
                "sip_order": r.get("sip_order", 0) if v43_succ else 0,
                "lnK": r.get("bayes_lnK", 0) if v43_succ else 0,
                "s_robust": r.get("s_robust", 0) if v43_succ else 0,
                "n_iters": r.get("n_iters", 0) if v43_succ else 0,
                "irm_converged": r.get("irm_converged", False) if v43_succ else False,
            },
        }
        comparison["frames"].append(entry)

    # 汇总
    v42_succ_list = [e for e in comparison["frames"] if e["v42"]["success"]]
    v43_succ_list = [e for e in comparison["frames"] if e["v43"]["success"]]
    v42_rms = [e["v42"]["rms_px"] for e in v42_succ_list if e["v42"]["rms_px"] is not None]
    v43_rms = [e["v43"]["rms_px"] for e in v43_succ_list if e["v43"]["rms_px"] is not None]
    v42_times = [e["v42"]["time_s"] for e in v42_succ_list]
    v43_times = [e["v43"]["time_s"] for e in v43_succ_list]
    v42_matched = [e["v42"]["matched"] for e in v42_succ_list]
    v43_matched = [e["v43"]["matched"] for e in v43_succ_list]
    n_total = len(comparison["frames"])

    comparison["summary"] = {
        "n_total": n_total,
        "v42_success_rate": f"{len(v42_succ_list)}/{n_total}",
        "v43_success_rate": f"{len(v43_succ_list)}/{n_total}",
        "v42_success_pct": round(100.0 * len(v42_succ_list) / max(n_total, 1), 1),
        "v43_success_pct": round(100.0 * len(v43_succ_list) / max(n_total, 1), 1),
        "v42_median_rms_px": round(float(np.median(v42_rms)), 4) if v42_rms else None,
        "v43_median_rms_px": round(float(np.median(v43_rms)), 4) if v43_rms else None,
        "v42_median_time_s": round(float(np.median(v42_times)), 3) if v42_times else None,
        "v43_median_time_s": round(float(np.median(v43_times)), 3) if v43_times else None,
        "v42_median_matched": int(np.median(v42_matched)) if v42_matched else 0,
        "v43_median_matched": int(np.median(v43_matched)) if v43_matched else 0,
    }
    return comparison


# ============================================================================
# 主流程
# ============================================================================

def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) >= 2 else 50
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 42

    print(f"=== V4.3 批量测试 ===")
    print(f"帧数: {n_frames}  随机种子: {seed}")

    # 收集 FITS 文件 (与 V4.1/V4.2 完全一致的逻辑, 保证帧集合相同)
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

    # 创建 V4.3 Solver (注入 GaiaClientPy 和 StarDetector, db_type=1 与 V4.2 一致)
    print("创建 V4.3 V43Solver (单一 DLL + IRM 闭环)...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(
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

        info = solve_single_frame_v43(fits_path, solver)
        results.append(info)

        # 增量保存 (异常退出时已完成的帧不丢失)
        with open(_RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 打印简要结果
        if info["status"] == "success":
            print(f"  → 成功: mode={info['flip_mode']} n={info['matched_count']} "
                  f"RMS={info['rms_px']:.3f}px S_robust={info['s_robust']:.3f}\" "
                  f"iters={info['n_iters']} conv={info['irm_converged']} "
                  f"SIP={info['sip_order']} lnK={info['bayes_lnK']} "
                  f"耗时={info['solve_time_s']}s")
        else:
            print(f"  → {info['status']}")

    solver.close()
    gaia_client.close()
    star_detector.close()

    # 保存 JSON
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {_RESULT_JSON}")

    # 保存 CSV
    _save_csv(results)
    print(f"CSV 已保存: {_SUMMARY_CSV}")

    # 加载 V4.2 结果并对比
    v42_map = load_v42_results()
    if v42_map:
        comparison = build_comparison(results, v42_map)
        with open(_COMPARISON_JSON, "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        print(f"对比 JSON 已保存: {_COMPARISON_JSON}")

    # 汇总统计
    _print_summary(results, v42_map)


def _save_csv(results):
    """保存 CSV"""
    csv_fields = [
        "filename", "target", "filter", "exposure_s",
        "width", "height", "focallen", "pixel_size", "s0_arcsec_px",
        "fits_center_ra", "fits_center_dec",
        "status", "flip_mode", "matched_count", "rms_px", "rms_arcsec",
        "scale_arcsec_px", "rotation_deg",
        "center_ra", "center_dec",
        "sip_order", "bayes_lnK", "triangle_pass_ratio", "theta_snr",
        # V4.3 特有
        "s_robust", "n_inliers", "n_iters", "irm_converged",
        "solve_time_s", "total_time_s",
    ]
    with open(_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _print_summary(results, v42_map):
    """打印汇总统计 + V4.2 vs V4.3 对比"""
    n_total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_fail = n_total - n_success

    print(f"\n{'=' * 70}")
    print(f"  V4.3 批量测试汇总")
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
        s_robust_list = [r["s_robust"] for r in succ]
        n_iters_list = [r["n_iters"] for r in succ]
        n_converged = sum(1 for r in succ if r.get("irm_converged", False))

        print(f"\n成功帧统计:")
        print(f"  RMS(px): 中位={np.median(rms_list):.3f} 均值={np.mean(rms_list):.3f} "
              f"最小={np.min(rms_list):.3f} 最大={np.max(rms_list):.3f}")
        print(f"  S_robust(\"): 中位={np.median(s_robust_list):.3f} 均值={np.mean(s_robust_list):.3f} "
              f"最小={np.min(s_robust_list):.3f} 最大={np.max(s_robust_list):.3f}")
        print(f"  匹配对数: 中位={np.median(matched_list):.0f} 均值={np.mean(matched_list):.0f} "
              f"最小={np.min(matched_list)} 最大={np.max(matched_list)}")
        print(f"  求解耗时(s): 中位={np.median(time_list):.3f} 均值={np.mean(time_list):.3f} "
              f"最小={np.min(time_list):.3f} 最大={np.max(time_list):.3f}")
        print(f"  IRM 迭代次数: 中位={np.median(n_iters_list):.0f} 均值={np.mean(n_iters_list):.1f} "
              f"收敛={n_converged}/{n_success}")
        print(f"  贝叶斯 lnK: 中位={np.median(lnK_list):.1f} 最小={np.min(lnK_list):.1f}")

    # 失败帧列表
    if n_fail > 0:
        print(f"\n失败帧:")
        for r in results:
            if r["status"] != "success":
                print(f"  {r['filename']}: {r['status']}")

    # V4.2 vs V4.3 对比表
    if v42_map:
        print(f"\n{'=' * 70}")
        print(f"  V4.2 vs V4.3 对比 (相同 50 帧, seed=42)")
        print(f"{'=' * 70}")

        v42_succ = sum(1 for r in results if v42_map.get(r["filename"], {}).get("status") == "success")
        v43_succ = n_success
        v42_rms = [v42_map[r["filename"]]["rms_px"] for r in results
                   if r["filename"] in v42_map and v42_map[r["filename"]].get("status") == "success"]
        v43_rms = [r["rms_px"] for r in results if r["status"] == "success"]
        v42_times = [v42_map[r["filename"]]["solve_time_s"] for r in results
                     if r["filename"] in v42_map and v42_map[r["filename"]].get("status") == "success"]
        v43_times = [r["solve_time_s"] for r in results if r["status"] == "success"]
        v42_matched = [v42_map[r["filename"]]["matched_count"] for r in results
                       if r["filename"] in v42_map and v42_map[r["filename"]].get("status") == "success"]
        v43_matched = [r["matched_count"] for r in results if r["status"] == "success"]

        print(f"{'指标':<20} {'V4.2':<18} {'V4.3':<18} {'变化':<15}")
        print(f"{'-' * 71}")
        print(f"{'成功率':<20} "
              f"{f'{v42_succ}/{n_total} ({100*v42_succ/n_total:.1f}%)':<18} "
              f"{f'{v43_succ}/{n_total} ({100*v43_succ/n_total:.1f}%)':<18} "
              f"{f'{v43_succ - v42_succ:+d} 帧':<15}")

        v42_med_rms = float(np.median(v42_rms)) if v42_rms else 0
        v43_med_rms = float(np.median(v43_rms)) if v43_rms else 0
        print(f"{'中位 RMS(px)':<20} {v42_med_rms:<18.4f} {v43_med_rms:<18.4f} "
              f"{f'{v43_med_rms - v42_med_rms:+.4f}':<15}")

        v42_med_time = float(np.median(v42_times)) if v42_times else 0
        v43_med_time = float(np.median(v43_times)) if v43_times else 0
        print(f"{'中位耗时(s)':<20} {v42_med_time:<18.3f} {v43_med_time:<18.3f} "
              f"{f'{v43_med_time - v42_med_time:+.3f}':<15}")

        v42_med_m = int(np.median(v42_matched)) if v42_matched else 0
        v43_med_m = int(np.median(v43_matched)) if v43_matched else 0
        print(f"{'中位 matched':<20} {v42_med_m:<18} {v43_med_m:<18} "
              f"{f'{v43_med_m - v42_med_m:+d}':<15}")

        # Spec 指标达成情况
        print(f"\n  Spec 指标达成:")
        succ_pct = 100 * v43_succ / n_total if n_total > 0 else 0
        print(f"    成功率 ≥ 96%: {'✅' if succ_pct >= 96 else '❌'} "
              f"(实际 {succ_pct:.1f}%)")
        print(f"    中位 RMS ≤ 1.7px: {'✅' if v43_med_rms <= 1.7 else '❌'} "
              f"(实际 {v43_med_rms:.4f}px)")
        print(f"    中位耗时 ≤ 0.1s: {'✅' if v43_med_time <= 0.1 else '❌'} "
              f"(实际 {v43_med_time:.3f}s)")


if __name__ == "__main__":
    main()

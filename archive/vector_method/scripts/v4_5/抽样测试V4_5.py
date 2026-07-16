"""V4.5 抽样测试 (50帧代表性抽样)

功能:
    从 testdata 中按目标分层抽样, 共约 50 帧
    V4.5 仅实现 Phase A (θ 求解), 不计算 RMS/matched/CD/SIP
    记录 success/theta_peak_deg/theta_snr/n_passed/n_samples/耗时/直方图

用途:
    验证 V4.5 相对向量法 θ 求解器在实际数据上的表现
    对比 V4.4 / V4.3 的 θ 求解结果

输出:
    lib/plate_solve/logs/v4_5/batch_test/batch_results.json
    lib/plate_solve/logs/v4_5/batch_test/batch_summary.csv

用法:
    py 抽样测试V4_5.py
"""
import os
import sys
import json
import csv
import time
import re
import random
import ctypes
import functools

print = functools.partial(print, flush=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MinGW DLL 路径 (libgomp-1.dll / libstdc++ 等)
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
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("V4.5抽样测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from vector_match_v4_5_cpp import Vm45Solver


# ============================================================================
# 配置
# ============================================================================

_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_5", "batch_test")
os.makedirs(_OUTPUT_DIR, exist_ok=True)
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "batch_results.json")
_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "batch_summary.csv")

_V45_DLL_PATH = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_5", "vector_match_v4_5.dll"
)

# 11 个目标 × 每目标 5 帧 ≈ 55 帧 (实际可能少)
_TARGET_LIST = [
    "LDN43", "M20_T2", "NGC247_T2", "NGC55_T3",
    "NGC6302", "NGC4945", "NGC7293", "Victory",
    "Galaxy_Center_mosaic1", "Galaxy_Center_mosaic2", "Galaxy_Center_mosaic3",
]
_PER_TARGET = 5
SNR_THRESHOLD = 5.0


# ============================================================================
# FITS 收集 / 文件名解析
# ============================================================================

def collect_fits_files(root_dir):
    fits_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".fts", ".fit", ".fits"):
                fits_files.append(os.path.join(dirpath, fn))
    return sorted(fits_files)


def parse_filename(path, target_list=None):
    """从文件名提取目标/滤镜/曝光信息

    参数:
        target_list: 可选的目标名列表. 若提供, 用前缀匹配 (basename 以 "target_" 开头)
                     来确定目标 (兼容含下划线的目标名, 如 Galaxy_Center_mosaic1, M20_T2).
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    info = {"filename": base, "path": path}
    m = re.search(r"-(\d+)S-", name)
    info["exposure_s"] = int(m.group(1)) if m else 0
    m = re.search(r"-([A-Za-z][A-Za-z0-9_-]*)$", name)
    info["filter"] = m.group(1) if m else "unknown"
    target_found = None
    if target_list:
        for t in target_list:
            if base.startswith(t + "_"):
                target_found = t
                break
    if target_found:
        info["target"] = target_found
    elif "_" in name:
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
# 抽样
# ============================================================================

def stratified_sample_by_target(all_fits, target_list, per_target=5, seed=42):
    """按目标分层抽样, 用前缀匹配 (basename 以 "target_" 开头)

    兼容含下划线的目标名 (如 Galaxy_Center_mosaic1, M20_T2).
    """
    by_target = {t: [] for t in target_list}
    for p in all_fits:
        base = os.path.basename(p)
        for t in target_list:
            if base.startswith(t + "_"):
                by_target[t].append(p)
                break
    rng = random.Random(seed)
    sampled = []
    skipped_targets = []
    for t in target_list:
        paths = by_target.get(t, [])
        if not paths:
            skipped_targets.append(t)
            continue
        rng.shuffle(paths)
        n = min(per_target, len(paths))
        sampled.extend(paths[:n])
    return sampled, by_target, skipped_targets


# ============================================================================
# 单帧求解
# ============================================================================

def solve_single_frame_v45(fits_path, solver):
    result_info = parse_filename(fits_path, target_list=_TARGET_LIST)
    t_start = time.time()
    try:
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl if fl > 0 else 0.0

        kws = img.keywords
        kw_dict = {k.name.upper(): k.value for k in kws}
        obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
        obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
        if not obj_ra_str or not obj_dec_str:
            result_info["status"] = "fail_no_objctra"
            result_info["error_msg"] = "missing OBJCTRA/OBJCTDEC"
            result_info["elapsed_s"] = round(time.time() - t_start, 3)
            return result_info
        cra0 = _parse_ra_hms(obj_ra_str)
        cdec0 = _parse_dec_dms(obj_dec_str)

        result_info["img_width"] = w
        result_info["img_height"] = h
        result_info["focallen"] = fl
        result_info["pixel_size"] = ps
        result_info["s0_arcsec_px"] = round(s0, 4)

        t_solve = time.time()
        result = solver.solve_theta(
            image_path=fits_path,
            ra=cra0, dec=cdec0,
            focal_length_mm=fl, pixel_size_um=ps,
        )
        elapsed = time.time() - t_solve

        # rc==0 (没抛异常), success 由 theta_snr > SNR_THRESHOLD 且 result.success 决定
        rc_ok = True
        result_info["rc"] = 0
        success_flag = bool(result.success) and (result.theta_snr > SNR_THRESHOLD)
        result_info["success"] = success_flag
        result_info["theta_peak_deg"] = round(float(result.theta_peak_deg), 4)
        result_info["theta_snr"] = round(float(result.theta_snr), 4)
        result_info["n_passed"] = int(result.n_passed)
        result_info["n_samples"] = int(result.n_samples)
        result_info["fov_diag_deg"] = round(float(result.fov_diag_deg), 4)
        result_info["m_lim_final"] = round(float(result.m_lim_final), 4)
        result_info["n_img"] = int(result.n_img)
        result_info["n_gaia"] = int(result.n_gaia)
        # theta_histogram: numpy array -> list (JSON 可序列化)
        result_info["theta_histogram"] = [round(float(x), 6) for x in result.theta_histogram.tolist()]
        result_info["elapsed_s"] = round(elapsed, 3)
        result_info["error_msg"] = (result.error_msg or "")[:200]
        result_info["status"] = "success" if success_flag else "fail_low_snr"

    except Exception as e:
        result_info["status"] = "error"
        result_info["error_msg"] = str(e)[:200]
        result_info["elapsed_s"] = round(time.time() - t_start, 3)
        logger.error(f"  异常: {fits_path}: {e}")

    return result_info


# ============================================================================
# 主流程
# ============================================================================

def main():
    print("=" * 70)
    print("V4.5 抽样测试 (50帧代表性抽样)")
    print("V4.5 仅 Phase A: 相对向量法 θ 求解 (不计算 RMS/CD/SIP)")
    print(f"SNR 阈值: {SNR_THRESHOLD}")
    print("=" * 70)

    # 1. 收集 FITS
    root_testdata = os.path.join(PROJECT_ROOT, "testdata")
    all_fits = collect_fits_files(root_testdata)
    print(f"找到 FITS 文件: {len(all_fits)} 个")

    # 2. 分层抽样
    sampled, by_target, skipped = stratified_sample_by_target(
        all_fits, _TARGET_LIST, per_target=_PER_TARGET, seed=42
    )
    print(f"分层抽样: {len(sampled)} 帧 (每目标 {_PER_TARGET} 帧, 目标列表 {len(_TARGET_LIST)} 个)")
    for t in _TARGET_LIST:
        n_total = len(by_target.get(t, []))
        n_sample = min(_PER_TARGET, n_total)
        flag = "" if n_total > 0 else " [缺失]"
        print(f"  {t}: 总 {n_total} 帧, 抽样 {n_sample} 帧{flag}")
    if skipped:
        print(f"[警告] 以下目标在 testdata 中未找到: {skipped}")

    if not sampled:
        print("\n[错误] 未抽到任何帧, 退出")
        return

    # 3. 初始化依赖
    print("\n初始化 GaiaClient / StarDetector / Vm45Solver...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))

    gaia_handle = gaia_client._handle
    if isinstance(gaia_handle, ctypes.c_void_p):
        gaia_handle = gaia_handle.value
    sdet_handle = star_detector._handle
    if isinstance(sdet_handle, ctypes.c_void_p):
        sdet_handle = sdet_handle.value

    solver = Vm45Solver(dll_path=_V45_DLL_PATH, log_dir=_OUTPUT_DIR)
    solver.set_gaia_client(gaia_handle)
    solver.set_star_detector(sdet_handle)

    # 4. 求解
    results = []
    t_total_start = time.time()
    for i, fits_path in enumerate(sampled, 1):
        base = os.path.basename(fits_path)
        print(f"\n[{i}/{len(sampled)}] {base}")
        r = solve_single_frame_v45(fits_path, solver)
        status = r.get("status", "unknown")
        if status == "success":
            print(f"  OK  θ={r['theta_peak_deg']:.2f}°  SNR={r['theta_snr']:.2f}  "
                  f"n_passed={r['n_passed']}  n_samples={r['n_samples']}  "
                  f"t={r['elapsed_s']:.2f}s  FOV={r.get('fov_diag_deg', 0):.3f}°  "
                  f"m_lim={r.get('m_lim_final', 0):.2f}")
        else:
            err = r.get("error_msg", "") or r.get("error", "")
            theta_str = ""
            if "theta_peak_deg" in r:
                theta_str = f"  θ={r.get('theta_peak_deg'):.2f}°  SNR={r.get('theta_snr', 0):.2f}"
            print(f"  {status}{theta_str}  err={err[:80]}")
        results.append(r)

    t_total = time.time() - t_total_start

    # 5. 统计汇总
    n_total = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    n_fail = n_total - n_success

    snr_list = [r["theta_snr"] for r in results if "theta_snr" in r]
    npassed_list = [r["n_passed"] for r in results if "n_passed" in r]
    t_list = [r["elapsed_s"] for r in results if "elapsed_s" in r]
    theta_list = [r["theta_peak_deg"] for r in results if "theta_peak_deg" in r]

    print("\n" + "=" * 70)
    print(f"V4.5 抽样测试完成: {n_success}/{n_total} 成功 ({100*n_success/max(n_total,1):.1f}%)")
    print(f"总耗时: {t_total:.1f}s  平均: {t_total/max(n_total,1):.2f}s/帧")

    if snr_list:
        snr_arr = np.array(snr_list)
        t_arr = np.array(t_list)
        npassed_arr = np.array(npassed_list)
        theta_arr = np.array(theta_list)
        print(f"SNR:      中位={np.median(snr_arr):.2f}  均值={np.mean(snr_arr):.2f}  "
              f"max={np.max(snr_arr):.2f}  min={np.min(snr_arr):.2f}")
        print(f"n_passed: 中位={np.median(npassed_arr):.0f}  均值={np.mean(npassed_arr):.0f}  "
              f"max={np.max(npassed_arr):.0f}  min={np.min(npassed_arr):.0f}")
        print(f"耗时(s):  中位={np.median(t_arr):.2f}  均值={np.mean(t_arr):.2f}  "
              f"max={np.max(t_arr):.2f}  min={np.min(t_arr):.2f}")
        print(f"θ(°):     中位={np.median(theta_arr):.2f}  均值={np.mean(theta_arr):.2f}  "
              f"std={np.std(theta_arr):.2f}")

    # 失败帧列表
    fail_frames = [r for r in results if not r.get("success")]
    if fail_frames:
        print(f"\n失败帧 ({len(fail_frames)}):")
        for r in fail_frames:
            err = r.get("error_msg", "") or r.get("error", "")
            theta_snr_str = ""
            if "theta_snr" in r:
                theta_snr_str = f"  θ={r.get('theta_peak_deg', 0):.2f}°  SNR={r.get('theta_snr', 0):.2f}"
            print(f"  {r['filename']}: {r.get('status')}{theta_snr_str}  {err[:80]}")

    # 按目标分组统计 (θ_peak 分布)
    print("\n按目标分组统计 (θ_peak 分布):")
    by_target_results = {}
    for r in results:
        t = r.get("target", "unknown")
        by_target_results.setdefault(t, []).append(r)
    for t in _TARGET_LIST:
        rs = by_target_results.get(t, [])
        if not rs:
            continue
        ok = sum(1 for r in rs if r.get("success"))
        theta_vals = [r["theta_peak_deg"] for r in rs if "theta_peak_deg" in r]
        snr_vals = [r["theta_snr"] for r in rs if "theta_snr" in r]
        if theta_vals:
            theta_arr = np.array(theta_vals)
            theta_str = (f"θ_mean={np.mean(theta_arr):.2f}°  "
                         f"θ_med={np.median(theta_arr):.2f}°  "
                         f"θ_std={np.std(theta_arr):.2f}°")
        else:
            theta_str = "θ=N/A"
        snr_str = (f"SNR_med={np.median(snr_vals):.2f}  SNR_mean={np.mean(snr_vals):.2f}"
                   if snr_vals else "SNR=N/A")
        print(f"  {t}: {ok}/{len(rs)} 成功  {theta_str}  {snr_str}")

    # 6. 保存 JSON
    summary_obj = {
        "total": n_total, "success": n_success, "fail": n_fail,
        "total_time_s": round(t_total, 2),
        "snr_threshold": SNR_THRESHOLD,
        "per_target": _PER_TARGET,
        "target_list": _TARGET_LIST,
        "results": results,
    }
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON 已保存: {_RESULT_JSON}")

    # 7. 保存 CSV
    with open(_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "filename", "target", "filter", "exposure_s",
            "success", "theta_peak_deg", "theta_snr", "n_passed", "n_samples",
            "elapsed_s", "img_width", "img_height", "fov_diag_deg",
            "m_lim_final", "n_img", "n_gaia", "error_msg",
        ])
        for r in results:
            w.writerow([
                r.get("filename", ""), r.get("target", ""), r.get("filter", ""),
                r.get("exposure_s", ""),
                r.get("success", ""), r.get("theta_peak_deg", ""),
                r.get("theta_snr", ""), r.get("n_passed", ""),
                r.get("n_samples", ""),
                r.get("elapsed_s", ""), r.get("img_width", ""),
                r.get("img_height", ""), r.get("fov_diag_deg", ""),
                r.get("m_lim_final", ""), r.get("n_img", ""),
                r.get("n_gaia", ""),
                r.get("error_msg", "") or r.get("error", ""),
            ])
    print(f"CSV 已保存: {_SUMMARY_CSV}")

    # 释放资源
    try: solver.close()
    except Exception: pass
    try: gaia_client.close()
    except Exception: pass
    try: star_detector.close()
    except Exception: pass


if __name__ == "__main__":
    main()

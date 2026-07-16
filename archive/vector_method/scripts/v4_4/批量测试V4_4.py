"""V4.4 小批量测试 (30帧代表性抽样)

从 testdata 中按目标分层抽样, 每个目标 3-4 帧, 共约 30 帧
验证 3D 密度场 + 递归聚焦方法在真实图像上的鲁棒性

输出:
    lib/plate_solve/logs/v4_4/batch_test/batch_test_results.json
    lib/plate_solve/logs/v4_4/batch_test/batch_summary.csv

用法:
    py 批量测试V4_4.py
"""
import os
import sys
import json
import csv
import time
import re
import random
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

import numpy as np
import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("V4.4批量测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_4.vector_match_v4_4_cpp import V44Solver


_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_4", "batch_test")
os.makedirs(_OUTPUT_DIR, exist_ok=True)
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "batch_test_results.json")
_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "batch_summary.csv")


def collect_fits_files(root_dir):
    fits_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".fts", ".fit", ".fits"):
                fits_files.append(os.path.join(dirpath, fn))
    return sorted(fits_files)


def parse_filename(path):
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


def solve_single_frame_v44(fits_path, solver):
    result_info = parse_filename(fits_path)
    t_start = time.time()
    try:
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

        frame_base = os.path.splitext(os.path.basename(fits_path))[0]
        frame_log_dir = os.path.join(_OUTPUT_DIR, "frames", frame_base)

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
        result_info["s_robust"] = round(float(result.get("s_robust", 0.0)), 4)
        result_info["n_inliers"] = int(result.get("n_inliers", 0))
        result_info["n_iters"] = int(result.get("n_iters", 0))
        result_info["irm_converged"] = bool(result.get("irm_converged", False))
        result_info["bayes_lnK"] = round(float(result.get("bayes_lnK", 0.0)), 2)
        result_info["triangle_pass_ratio"] = round(float(result.get("triangle_pass_ratio", 0.0)), 3)
        result_info["theta_snr"] = round(float(result.get("theta_snr", 0.0)), 2)
        result_info["theta_peak_deg"] = round(float(result.get("theta_peak_deg", 0.0)), 2)
        result_info["total_time_s"] = round(time.time() - t_start, 3)
    except Exception as e:
        result_info["status"] = f"error: {str(e)[:100]}"
        result_info["total_time_s"] = round(time.time() - t_start, 3)
        logger.error(f"  异常: {fits_path}: {e}")
    return result_info


def stratified_sample(all_fits, per_target=4, seed=42):
    """按目标分层抽样, 每个目标 per_target 帧"""
    by_target = {}
    for p in all_fits:
        info = parse_filename(p)
        t = info["target"]
        by_target.setdefault(t, []).append(p)
    rng = random.Random(seed)
    sampled = []
    for target, paths in sorted(by_target.items()):
        rng.shuffle(paths)
        n = min(per_target, len(paths))
        sampled.extend(paths[:n])
    return sampled


def main():
    print("=" * 70)
    print("V4.4 小批量测试 (30帧代表性抽样)")
    print("3D 密度场 + 递归聚焦, dx/dy 不过滤 (让 RANSAC 处理歧义)")
    print("=" * 70)

    root_testdata = os.path.join(PROJECT_ROOT, "testdata")
    all_fits = collect_fits_files(root_testdata)
    print(f"找到 FITS 文件: {len(all_fits)} 个")

    # 分层抽样: 每个目标 4 帧 (9 个目标 × 4 = 36 帧, 接近 30)
    sampled = stratified_sample(all_fits, per_target=4, seed=42)
    print(f"分层抽样: {len(sampled)} 帧 (每目标 4 帧)")

    # 按目标分组打印
    by_target = {}
    for p in sampled:
        info = parse_filename(p)
        by_target.setdefault(info["target"], []).append(os.path.basename(p))
    for t, files in sorted(by_target.items()):
        print(f"  {t}: {len(files)} 帧")

    # 初始化
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=0)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V44Solver(gaia_client=gaia_client, star_detector=star_detector)

    # 求解
    results = []
    t_total_start = time.time()
    for i, fits_path in enumerate(sampled):
        base = os.path.basename(fits_path)
        print(f"\n[{i+1}/{len(sampled)}] {base}")
        r = solve_single_frame_v44(fits_path, solver)
        status = r.get("status", "unknown")
        if status == "success":
            print(f"  OK  RMS={r['rms_px']:.4f}px  matched={r['matched_count']}  "
                  f"SNR={r['theta_snr']:.1f}  t={r['solve_time_s']:.2f}s  "
                  f"mode={r['flip_mode']}  θ={r['rotation_deg']:.2f}°")
        else:
            print(f"  FAIL  status={status}  err={r.get('error', '')[:80]}")
        results.append(r)

    t_total = time.time() - t_total_start

    # 统计
    n_total = len(results)
    n_success = sum(1 for r in results if r.get("status") == "success")
    n_fail = n_total - n_success
    rms_list = [r["rms_px"] for r in results if r.get("status") == "success"]
    snr_list = [r["theta_snr"] for r in results if r.get("status") == "success"]
    t_list = [r["solve_time_s"] for r in results if r.get("status") == "success"]

    print("\n" + "=" * 70)
    print(f"小批量测试完成: {n_success}/{n_total} 成功 ({100*n_success/n_total:.1f}%)")
    print(f"总耗时: {t_total:.1f}s  平均: {t_total/n_total:.2f}s/帧")
    if rms_list:
        rms_arr = np.array(rms_list)
        snr_arr = np.array(snr_list)
        t_arr = np.array(t_list)
        print(f"RMS(px): 中位={np.median(rms_arr):.4f}  平均={np.mean(rms_arr):.4f}  "
              f"max={np.max(rms_arr):.4f}  min={np.min(rms_arr):.4f}")
        print(f"SNR:    中位={np.median(snr_arr):.1f}  平均={np.mean(snr_arr):.1f}  "
              f"max={np.max(snr_arr):.1f}  min={np.min(snr_arr):.1f}")
        print(f"耗时(s): 中位={np.median(t_arr):.2f}  平均={np.mean(t_arr):.2f}  "
              f"max={np.max(t_arr):.2f}  min={np.min(t_arr):.2f}")

    # 失败帧
    fail_frames = [r for r in results if r.get("status") != "success"]
    if fail_frames:
        print(f"\n失败帧 ({len(fail_frames)}):")
        for r in fail_frames:
            print(f"  {r['filename']}: {r.get('status')} {r.get('error', '')[:60]}")

    # 按目标统计
    print("\n按目标统计:")
    by_target_results = {}
    for r in results:
        t = r.get("target", "unknown")
        by_target_results.setdefault(t, []).append(r)
    for t, rs in sorted(by_target_results.items()):
        ok = sum(1 for r in rs if r.get("status") == "success")
        rms_vals = [r["rms_px"] for r in rs if r.get("status") == "success"]
        rms_str = f"RMS={np.mean(rms_vals):.4f}" if rms_vals else "N/A"
        print(f"  {t}: {ok}/{len(rs)} 成功  {rms_str}")

    # 保存 JSON
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "total": n_total, "success": n_success, "fail": n_fail,
            "total_time_s": round(t_total, 2),
            "results": results,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {_RESULT_JSON}")

    # 保存 CSV
    with open(_SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "target", "filter", "status", "rms_px", "matched_count",
                    "theta_snr", "flip_mode", "rotation_deg", "s_robust", "n_inliers",
                    "n_iters", "irm_converged", "solve_time_s", "error"])
        for r in results:
            w.writerow([
                r.get("filename", ""), r.get("target", ""), r.get("filter", ""),
                r.get("status", ""), r.get("rms_px", ""),
                r.get("matched_count", ""), r.get("theta_snr", ""),
                r.get("flip_mode", ""), r.get("rotation_deg", ""),
                r.get("s_robust", ""), r.get("n_inliers", ""),
                r.get("n_iters", ""), r.get("irm_converged", ""),
                r.get("solve_time_s", ""), r.get("error", ""),
            ])
    print(f"CSV 已保存: {_SUMMARY_CSV}")


if __name__ == "__main__":
    main()

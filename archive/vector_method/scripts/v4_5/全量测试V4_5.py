"""V4.5 全量测试 - 遍历 testdata 下所有 FITS 文件

功能:
    递归收集 testdata/ 下所有 FITS 文件 (.fts/.fit/.fits), 全部用 V4.5 求解
    V4.5 仅实现 Phase A (θ 求解), 不计算 RMS/matched/CD/SIP
    记录 success/theta_peak_deg/theta_snr/n_passed/n_samples/耗时/直方图

用途:
    验证 V4.5 相对向量法 θ 求解器在全量实际数据上的表现
    与 V4.4 全量测试对比成功率/SNR/θ 分布

输出:
    - lib/plate_solve/logs/v4_5/full_test/full_test_all.json
    - lib/plate_solve/logs/v4_5/full_test/summary.csv
    - lib/plate_solve/logs/v4_5/full_test/summary.txt

用法:
    python 全量测试V4_5.py
    python 全量测试V4_5.py --no-resume    # 忽略断点续跑, 重新测试

注:
    全量 ~790 帧, 中位耗时 ~2s/帧, 预计总耗时 ~30 分钟
    支持断点续跑, 中断后重新运行会跳过已完成的帧
"""
import os
import sys
import json
import csv
import time
import re
import argparse
import ctypes
import functools

print = functools.partial(print, flush=True)  # 强制无缓冲

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MSYS2 MinGW DLL 路径 (V4.5 DLL 依赖 libwinpthread-1.dll / libgcc_s_seh / libstdc++ / libgomp)
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
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("V4.5全量测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from vector_match_v4_5_cpp import Vm45Solver


# ============================================================================
# 配置 / 输出目录
# ============================================================================

_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_5", "full_test")
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "full_test_all.json")
_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "summary.csv")
_SUMMARY_TXT = os.path.join(_OUTPUT_DIR, "summary.txt")

_V45_DLL_PATH = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_5", "vector_match_v4_5.dll"
)

# V4.5 目标列表 (用于按目标前缀匹配, 含下划线的目标名也兼容)
_TARGET_LIST = [
    "LDN43", "M20_T2", "NGC247_T2", "NGC55_T3",
    "NGC6302", "NGC4945", "NGC7293", "Victory",
    "Galaxy_Center_mosaic1", "Galaxy_Center_mosaic2", "Galaxy_Center_mosaic3",
]

SNR_THRESHOLD = 5.0

root_testdata = os.path.join(PROJECT_ROOT, "testdata")


# ============================================================================
# 收集 FITS 文件 (递归 testdata 全量)
# ============================================================================

def collect_fits_files(root_dir):
    """递归收集所有 FITS 文件(.fts/.fit/.fits)"""
    fits_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in (".fts", ".fit", ".fits"):
                fits_files.append(os.path.join(dirpath, fn))
    return sorted(fits_files)


def parse_filename(path, target_list=None):
    """从文件名提取目标、滤镜、曝光等信息

    参数:
        target_list: 可选的目标名列表. 若提供, 用前缀匹配 (basename 以 "target_" 开头)
                     来确定目标 (兼容含下划线的目标名).
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    info = {"filename": base, "path": path}

    m = re.search(r"-(\d+)S-", name)
    info["exposure_s"] = int(m.group(1)) if m else 0

    m = re.search(r"-([A-Za-z][A-Za-z0-9_-]*)$", name)
    info["filter"] = m.group(1) if m else "unknown"

    # 目标名匹配 (优先用 target_list 前缀匹配)
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

    # 子目录 (用于按 panel 分组)
    rel = os.path.relpath(path, root_testdata)
    info["subdir"] = os.path.dirname(rel).replace(os.sep, "/") or "root"
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
# V4.5 单帧求解
# ============================================================================

def solve_single_frame_v45(fits_path, solver):
    """对单帧 FITS 进行 V4.5 Phase A (θ 求解)"""
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
        result_info["fits_center_ra"] = cra0
        result_info["fits_center_dec"] = cdec0

        t_solve = time.time()
        result = solver.solve_theta(
            image_path=fits_path,
            ra=cra0, dec=cdec0,
            focal_length_mm=fl, pixel_size_um=ps,
        )
        elapsed = time.time() - t_solve

        # success 由 theta_snr > SNR_THRESHOLD 且 result.success 决定
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
    parser = argparse.ArgumentParser(description="V4.5 全量测试 (Phase A θ 求解)")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略断点续跑, 重新测试所有帧")
    args = parser.parse_args()

    print("=" * 70)
    print("V4.5 全量测试 (Phase A 相对向量法 θ 求解)")
    print(f"SNR 阈值: {SNR_THRESHOLD}")
    print("=" * 70)

    # 1. 收集 FITS
    all_fits = collect_fits_files(root_testdata)
    print(f"找到 FITS 文件: {len(all_fits)} 个")

    # 按子目录统计
    subdir_count = {}
    for p in all_fits:
        rel = os.path.relpath(p, root_testdata)
        subdir = os.path.dirname(rel).replace(os.sep, "/") or "root"
        subdir_count[subdir] = subdir_count.get(subdir, 0) + 1
    print(f"\n按子目录分布:")
    for sd, cnt in sorted(subdir_count.items()):
        print(f"  {sd:<30} {cnt} 帧")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # 2. 断点续跑
    results = []
    done_paths = set()
    if not args.no_resume and os.path.exists(_RESULT_JSON):
        try:
            with open(_RESULT_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_paths = {r["path"] for r in results if "path" in r}
            print(f"\n断点续跑: 已完成 {len(done_paths)}/{len(all_fits)} 帧")
        except Exception as e:
            print(f"[警告] 加载断点文件失败, 从头开始: {e}")
            results = []

    # 3. 初始化依赖 (单例)
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

    # 4. 逐帧求解
    t_batch_start = time.time()
    n_total = len(all_fits)
    n_skipped = 0
    n_new_done = 0
    n_new_success = 0

    for i, fits_path in enumerate(all_fits, 1):
        base = os.path.basename(fits_path)

        # 跳过已完成帧
        if fits_path in done_paths:
            n_skipped += 1
            if i % 50 == 0 or i == n_total:
                elapsed = time.time() - t_batch_start
                print(f"[{i}/{n_total}] 跳过已完成 (累计 {n_skipped}) - 已运行 {elapsed:.1f}s")
            continue

        # 新帧打印
        print(f"\n[{i}/{n_total}] {base}")

        info = solve_single_frame_v45(fits_path, solver)
        results.append(info)
        n_new_done += 1
        if info.get("success"):
            n_new_success += 1

        # 增量保存 JSON (断点续跑用)
        try:
            with open(_RESULT_JSON, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            print(f"  [警告] 增量保存失败: {e}")

        # 简要结果
        status = info.get("status", "unknown")
        if status == "success":
            print(f"  OK  θ={info['theta_peak_deg']:.2f}°  SNR={info['theta_snr']:.2f}  "
                  f"n_passed={info['n_passed']}  n_samples={info['n_samples']}  "
                  f"t={info['elapsed_s']:.2f}s  FOV={info.get('fov_diag_deg', 0):.3f}°  "
                  f"m_lim={info.get('m_lim_final', 0):.2f}")
        else:
            err = info.get("error_msg", "") or ""
            theta_str = ""
            if "theta_peak_deg" in info:
                theta_str = f"  θ={info.get('theta_peak_deg'):.2f}°  SNR={info.get('theta_snr', 0):.2f}"
            print(f"  {status}{theta_str}  err={err[:80]}")

        # 每 10 帧打印整体进度
        if n_new_done > 0 and n_new_done % 10 == 0:
            elapsed = time.time() - t_batch_start
            avg = elapsed / n_new_done
            remaining_frames = n_total - i
            remaining_min = remaining_frames * avg / 60.0
            print(f"  --- 进度: {i}/{n_total} ({100*i/n_total:.1f}%) "
                  f"新增 {n_new_done} 帧 (成功 {n_new_success}), "
                  f"平均 {avg:.2f}s/帧, "
                  f"预计剩余 {remaining_min:.1f} 分钟 ---")

    # 5. 释放资源
    try:
        solver.close()
    except Exception:
        pass
    try:
        gaia_client.close()
    except Exception:
        pass
    try:
        star_detector.close()
    except Exception:
        pass

    total_elapsed = time.time() - t_batch_start
    print(f"\n本次新增测试: {n_new_done} 帧, 跳过: {n_skipped} 帧, "
          f"耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} 分钟)")

    # 6. 保存最终 JSON
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nJSON 已保存: {_RESULT_JSON}")

    # 7. 保存 CSV
    _save_csv(results)
    print(f"CSV 已保存: {_SUMMARY_CSV}")

    # 8. 汇总统计
    summary_lines = _build_summary(results, total_elapsed)
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n汇总已保存: {_SUMMARY_TXT}")


# ============================================================================
# 输出辅助函数
# ============================================================================

def _save_csv(results):
    """保存 CSV (列与 V4.5 抽样测试一致, 不含 theta_histogram)"""
    csv_fields = [
        "filename", "target", "filter", "exposure_s", "subdir",
        "success", "theta_peak_deg", "theta_snr", "n_passed", "n_samples",
        "elapsed_s", "img_width", "img_height", "fov_diag_deg",
        "m_lim_final", "n_img", "n_gaia", "error_msg", "status",
    ]
    with open(_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _build_summary(results, total_elapsed):
    """构建汇总统计文本 (含按目标/滤镜/子目录分组 + 失败帧列表)"""
    lines = []
    n_total = len(results)
    n_success = sum(1 for r in results if r.get("success"))
    n_fail = n_total - n_success

    lines.append("=" * 70)
    lines.append("  V4.5 全量测试汇总 (Phase A 相对向量法 θ 求解)")
    lines.append("=" * 70)
    lines.append(f"总帧数: {n_total}")
    lines.append(f"成功: {n_success} ({100 * n_success / max(n_total, 1):.1f}%)")
    lines.append(f"失败: {n_fail} ({100 * n_fail / max(n_total, 1):.1f}%)")
    lines.append(f"SNR 阈值: {SNR_THRESHOLD}")
    lines.append(f"本次运行耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} 分钟)")

    # 全帧统计
    snr_list = [r["theta_snr"] for r in results if "theta_snr" in r]
    npassed_list = [r["n_passed"] for r in results if "n_passed" in r]
    t_list = [r["elapsed_s"] for r in results if "elapsed_s" in r]
    theta_list = [r["theta_peak_deg"] for r in results if "theta_peak_deg" in r]

    if snr_list:
        snr_arr = np.array(snr_list)
        t_arr = np.array(t_list)
        npassed_arr = np.array(npassed_list)
        theta_arr = np.array(theta_list)
        lines.append("")
        lines.append("全帧统计 (含失败帧的 θ/SNR 也计入):")
        lines.append(f"  SNR:      中位={np.median(snr_arr):.2f}  均值={np.mean(snr_arr):.2f}  "
                     f"min={np.min(snr_arr):.2f}  max={np.max(snr_arr):.2f}")
        lines.append(f"  n_passed: 中位={np.median(npassed_arr):.0f}  均值={np.mean(npassed_arr):.0f}  "
                     f"min={np.min(npassed_arr):.0f}  max={np.max(npassed_arr):.0f}")
        lines.append(f"  耗时(s):  中位={np.median(t_arr):.2f}  均值={np.mean(t_arr):.2f}  "
                     f"min={np.min(t_arr):.2f}  max={np.max(t_arr):.2f}")
        lines.append(f"  θ(°):     中位={np.median(theta_arr):.2f}  均值={np.mean(theta_arr):.2f}  "
                     f"std={np.std(theta_arr):.2f}")

    # 成功帧统计
    if n_success > 0:
        succ = [r for r in results if r.get("success")]
        succ_snr = [r["theta_snr"] for r in succ if "theta_snr" in r]
        succ_npassed = [r["n_passed"] for r in succ if "n_passed" in r]
        succ_t = [r["elapsed_s"] for r in succ if "elapsed_s" in r]
        succ_theta = [r["theta_peak_deg"] for r in succ if "theta_peak_deg" in r]
        lines.append("")
        lines.append("成功帧统计:")
        if succ_snr:
            lines.append(f"  SNR:      中位={np.median(succ_snr):.2f}  均值={np.mean(succ_snr):.2f}  "
                         f"min={np.min(succ_snr):.2f}  max={np.max(succ_snr):.2f}")
            lines.append(f"  n_passed: 中位={np.median(succ_npassed):.0f}  均值={np.mean(succ_npassed):.0f}  "
                         f"min={np.min(succ_npassed):.0f}  max={np.max(succ_npassed):.0f}")
            lines.append(f"  耗时(s):  中位={np.median(succ_t):.2f}  均值={np.mean(succ_t):.2f}  "
                         f"min={np.min(succ_t):.2f}  max={np.max(succ_t):.2f}")
            lines.append(f"  θ(°):     中位={np.median(succ_theta):.2f}  均值={np.mean(succ_theta):.2f}  "
                         f"std={np.std(succ_theta):.2f}")

    # 按目标分组
    lines.append("")
    lines.append("=" * 70)
    lines.append("  按目标分组统计")
    lines.append("=" * 70)
    lines.append(f"{'目标':<28} {'总数':>6} {'成功':>6} {'成功率':>8} "
                 f"{'中位SNR':>10} {'中位θ°':>10} {'中位耗时':>10}")
    lines.append("-" * 100)
    by_target = {}
    for r in results:
        t = r.get("target", "unknown")
        by_target.setdefault(t, []).append(r)
    for t in sorted(by_target.keys()):
        rs = by_target[t]
        n = len(rs)
        s = sum(1 for r in rs if r.get("success"))
        succ_rs = [r for r in rs if r.get("success")]
        med_snr = float(np.median([r["theta_snr"] for r in succ_rs if "theta_snr" in r])) if succ_rs else 0
        med_theta = float(np.median([r["theta_peak_deg"] for r in succ_rs if "theta_peak_deg" in r])) if succ_rs else 0
        med_time = float(np.median([r["elapsed_s"] for r in succ_rs if "elapsed_s" in r])) if succ_rs else 0
        lines.append(f"{t:<28} {n:>6} {s:>6} {100*s/max(n,1):>7.1f}% "
                     f"{med_snr:>9.2f} {med_theta:>9.2f}° {med_time:>9.2f}s")

    # 按滤镜分组
    lines.append("")
    lines.append("=" * 70)
    lines.append("  按滤镜分组统计")
    lines.append("=" * 70)
    lines.append(f"{'滤镜':<15} {'总数':>6} {'成功':>6} {'成功率':>8} {'中位SNR':>10}")
    lines.append("-" * 70)
    by_filter = {}
    for r in results:
        f = r.get("filter", "unknown")
        by_filter.setdefault(f, []).append(r)
    for f in sorted(by_filter.keys()):
        rs = by_filter[f]
        n = len(rs)
        s = sum(1 for r in rs if r.get("success"))
        succ_rs = [r for r in rs if r.get("success")]
        med_snr = float(np.median([r["theta_snr"] for r in succ_rs if "theta_snr" in r])) if succ_rs else 0
        lines.append(f"{f:<15} {n:>6} {s:>6} {100*s/max(n,1):>7.1f}% {med_snr:>9.2f}")

    # 按子目录分组
    lines.append("")
    lines.append("=" * 70)
    lines.append("  按子目录分组统计")
    lines.append("=" * 70)
    lines.append(f"{'子目录':<30} {'总数':>6} {'成功':>6} {'成功率':>8}")
    lines.append("-" * 70)
    by_subdir = {}
    for r in results:
        sd = r.get("subdir", "unknown")
        by_subdir.setdefault(sd, []).append(r)
    for sd in sorted(by_subdir.keys()):
        rs = by_subdir[sd]
        n = len(rs)
        s = sum(1 for r in rs if r.get("success"))
        lines.append(f"{sd:<30} {n:>6} {s:>6} {100*s/max(n,1):>7.1f}%")

    # 失败帧列表
    if n_fail > 0:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  失败帧列表 ({n_fail} 帧, 前 50 个)")
        lines.append("=" * 70)
        fail_shown = 0
        for r in results:
            if not r.get("success"):
                err = r.get("error_msg", "") or ""
                theta_str = ""
                if "theta_peak_deg" in r:
                    theta_str = f"  θ={r.get('theta_peak_deg'):.2f}°  SNR={r.get('theta_snr', 0):.2f}"
                lines.append(f"  {r['filename']}: {r.get('status')}{theta_str}  {err[:80]}")
                fail_shown += 1
                if fail_shown >= 50:
                    break
        if n_fail > 50:
            lines.append(f"  ... 还有 {n_fail - 50} 个失败帧未显示")

    return lines


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n脚本异常: {e}")
        print(traceback.format_exc())

"""V4.3 全量测试 - 遍历 testdata 下所有 FITS 文件

功能:
    递归收集 testdata/ 下所有 FITS 文件 (.fts/.fit/.fits), 全部用 V4.3 求解
    记录 success/RMS/matched/耗时/S_robust/n_iters/irm_converged/sip_order/lnK
    按目标/滤镜分组统计, 输出失败帧列表

输出:
    - lib/plate_solve/logs/v4_3/full_test/full_test_all.json
    - lib/plate_solve/logs/v4_3/full_test/summary.csv
    - lib/plate_solve/logs/v4_3/full_test/summary.txt

用法:
    python 全量测试V4_3.py
    python 全量测试V4_3.py --no-resume    # 忽略断点续跑, 重新测试

注:
    全量 790 帧, 中位耗时 ~1.3s/帧, 预计总耗时 ~17 分钟
    支持断点续跑, 中断后重新运行会跳过已完成的帧
"""
import os
import sys
import json
import csv
import time
import re
import argparse

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
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("V4.3全量测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_3.vector_match_v4_3_cpp import V43Solver


# ============================================================================
# 输出目录
# ============================================================================
_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "full_test")
_RESULT_JSON = os.path.join(_OUTPUT_DIR, "full_test_all.json")
_SUMMARY_CSV = os.path.join(_OUTPUT_DIR, "summary.csv")
_SUMMARY_TXT = os.path.join(_OUTPUT_DIR, "summary.txt")


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

    # 子目录 (用于按 panel 分组)
    rel = os.path.relpath(path, root_testdata)
    info["subdir"] = os.path.dirname(rel).replace(os.sep, "/") or "root"
    return info


root_testdata = os.path.join(PROJECT_ROOT, "testdata")


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
    """对单帧 FITS 进行 V4.3 WCS 求解"""
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
        result_info["fits_center_ra"] = cra0
        result_info["fits_center_dec"] = cdec0

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
        result_info["center_ra"] = float(result.get("center_ra", cra0))
        result_info["center_dec"] = float(result.get("center_dec", cdec0))
        result_info["sip_order"] = int(result.get("sip_order", 0))

        result_info["s_robust"] = round(float(result.get("s_robust", 0.0)), 4)
        result_info["n_inliers"] = int(result.get("n_inliers", 0))
        result_info["n_iters"] = int(result.get("n_iters", 0))
        result_info["irm_converged"] = bool(result.get("irm_converged", False))

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
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="V4.3 全量测试")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略断点续跑, 重新测试所有帧")
    args = parser.parse_args()

    print(f"=== V4.3 全量测试 ===")

    # 收集 testdata 下所有 FITS 文件
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

    # 断点续跑
    results = []
    done_paths = set()
    if not args.no_resume and os.path.exists(_RESULT_JSON):
        try:
            with open(_RESULT_JSON, "r", encoding="utf-8") as f:
                results = json.load(f)
            done_paths = {r["path"] for r in results if "path" in r}
            print(f"\n断点续跑: 已完成 {len(done_paths)}/{len(all_fits)} 帧")
        except Exception:
            results = []

    # 创建 V4.3 Solver
    print("创建 V4.3 V43Solver (单一 DLL + IRM 闭环)...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(
        gaia_client=gaia_client,
        star_detector=star_detector,
    )

    t_batch_start = time.time()
    n_selected = len(all_fits)
    n_skipped = 0
    n_new_done = 0

    # 逐帧测试
    for i, fits_path in enumerate(all_fits, 1):
        base = os.path.basename(fits_path)
        if fits_path in done_paths:
            n_skipped += 1
            # 每 50 帧打印一次跳过进度
            if i % 50 == 0 or i == n_selected:
                elapsed = time.time() - t_batch_start
                print(f"[{i}/{n_selected}] 跳过已完成 (累计 {n_skipped}) - 已运行 {elapsed:.1f}s")
            continue

        # 新帧打印
        print(f"\n[{i}/{n_selected}] {base}")

        info = solve_single_frame_v43(fits_path, solver)
        results.append(info)
        n_new_done += 1

        # 增量保存
        with open(_RESULT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # 简要结果
        if info["status"] == "success":
            print(f"  → 成功: mode={info['flip_mode']} n={info['matched_count']} "
                  f"RMS={info['rms_px']:.3f}px S_robust={info['s_robust']:.3f}\" "
                  f"iters={info['n_iters']} conv={info['irm_converged']} "
                  f"SIP={info['sip_order']} lnK={info['bayes_lnK']} "
                  f"耗时={info['solve_time_s']}s")
        else:
            print(f"  → {info['status']}")

        # 每 20 帧打印整体进度
        if n_new_done > 0 and n_new_done % 20 == 0:
            elapsed = time.time() - t_batch_start
            avg = elapsed / n_new_done
            remaining = (n_selected - i) * avg
            print(f"  --- 进度: {i}/{n_selected} ({100*i/n_selected:.1f}%) "
                  f"新增 {n_new_done} 帧, 平均 {avg:.2f}s/帧, "
                  f"预计剩余 {remaining/60:.1f} 分钟 ---")

    solver.close()
    gaia_client.close()
    star_detector.close()

    total_elapsed = time.time() - t_batch_start
    print(f"\n本次新增测试: {n_new_done} 帧, 跳过: {n_skipped} 帧, "
          f"耗时: {total_elapsed:.1f}s ({total_elapsed/60:.1f} 分钟)")

    # 保存最终 JSON
    with open(_RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {_RESULT_JSON}")

    # 保存 CSV
    _save_csv(results)
    print(f"CSV 已保存: {_SUMMARY_CSV}")

    # 汇总统计
    summary_lines = _build_summary(results)
    summary_text = "\n".join(summary_lines)
    print(summary_text)
    with open(_SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n汇总已保存: {_SUMMARY_TXT}")


def _save_csv(results):
    """保存 CSV"""
    csv_fields = [
        "filename", "target", "filter", "exposure_s", "subdir",
        "width", "height", "focallen", "pixel_size", "s0_arcsec_px",
        "fits_center_ra", "fits_center_dec",
        "status", "flip_mode", "matched_count", "rms_px", "rms_arcsec",
        "scale_arcsec_px", "rotation_deg",
        "center_ra", "center_dec",
        "sip_order", "bayes_lnK", "triangle_pass_ratio", "theta_snr",
        "s_robust", "n_inliers", "n_iters", "irm_converged",
        "solve_time_s", "total_time_s",
    ]
    with open(_SUMMARY_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _build_summary(results):
    """构建汇总统计文本 (含按目标/滤镜分组)"""
    lines = []
    n_total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_fail = n_total - n_success

    lines.append("=" * 70)
    lines.append("  V4.3 全量测试汇总")
    lines.append("=" * 70)
    lines.append(f"总帧数: {n_total}")
    lines.append(f"成功: {n_success} ({100 * n_success / max(n_total, 1):.1f}%)")
    lines.append(f"失败: {n_fail} ({100 * n_fail / max(n_total, 1):.1f}%)")

    if n_success > 0:
        succ = [r for r in results if r["status"] == "success"]
        rms_list = [r["rms_px"] for r in succ]
        matched_list = [r["matched_count"] for r in succ]
        time_list = [r["solve_time_s"] for r in succ]
        lnK_list = [r["bayes_lnK"] for r in succ]
        s_robust_list = [r["s_robust"] for r in succ]
        n_iters_list = [r["n_iters"] for r in succ]
        n_converged = sum(1 for r in succ if r.get("irm_converged", False))

        lines.append("")
        lines.append("成功帧统计:")
        lines.append(f"  RMS(px): 中位={np.median(rms_list):.3f} 均值={np.mean(rms_list):.3f} "
                     f"最小={np.min(rms_list):.3f} 最大={np.max(rms_list):.3f}")
        lines.append(f"  S_robust(\"): 中位={np.median(s_robust_list):.3f} 均值={np.mean(s_robust_list):.3f} "
                     f"最小={np.min(s_robust_list):.3f} 最大={np.max(s_robust_list):.3f}")
        lines.append(f"  匹配对数: 中位={np.median(matched_list):.0f} 均值={np.mean(matched_list):.0f} "
                     f"最小={np.min(matched_list)} 最大={np.max(matched_list)}")
        lines.append(f"  求解耗时(s): 中位={np.median(time_list):.3f} 均值={np.mean(time_list):.3f} "
                     f"最小={np.min(time_list):.3f} 最大={np.max(time_list):.3f}")
        lines.append(f"  IRM 迭代次数: 中位={np.median(n_iters_list):.0f} 均值={np.mean(n_iters_list):.1f} "
                     f"收敛={n_converged}/{n_success}")
        lines.append(f"  贝叶斯 lnK: 中位={np.median(lnK_list):.1f} 最小={np.min(lnK_list):.1f}")

    # 按目标分组
    lines.append("")
    lines.append("=" * 70)
    lines.append("  按目标分组统计")
    lines.append("=" * 70)
    lines.append(f"{'目标':<25} {'总数':>6} {'成功':>6} {'成功率':>8} {'中位RMS':>10} {'中位耗时':>10}")
    lines.append("-" * 70)
    by_target = {}
    for r in results:
        t = r.get("target", "unknown")
        if t not in by_target:
            by_target[t] = []
        by_target[t].append(r)
    for t in sorted(by_target.keys()):
        rs = by_target[t]
        n = len(rs)
        s = sum(1 for r in rs if r["status"] == "success")
        succ_rs = [r for r in rs if r["status"] == "success"]
        med_rms = float(np.median([r["rms_px"] for r in succ_rs])) if succ_rs else 0
        med_time = float(np.median([r["solve_time_s"] for r in succ_rs])) if succ_rs else 0
        lines.append(f"{t:<25} {n:>6} {s:>6} {100*s/n:>7.1f}% {med_rms:>9.3f}p {med_time:>9.3f}s")

    # 按滤镜分组
    lines.append("")
    lines.append("=" * 70)
    lines.append("  按滤镜分组统计")
    lines.append("=" * 70)
    lines.append(f"{'滤镜':<15} {'总数':>6} {'成功':>6} {'成功率':>8} {'中位RMS':>10} {'中位耗时':>10}")
    lines.append("-" * 70)
    by_filter = {}
    for r in results:
        f = r.get("filter", "unknown")
        if f not in by_filter:
            by_filter[f] = []
        by_filter[f].append(r)
    for f in sorted(by_filter.keys()):
        rs = by_filter[f]
        n = len(rs)
        s = sum(1 for r in rs if r["status"] == "success")
        succ_rs = [r for r in rs if r["status"] == "success"]
        med_rms = float(np.median([r["rms_px"] for r in succ_rs])) if succ_rs else 0
        med_time = float(np.median([r["solve_time_s"] for r in succ_rs])) if succ_rs else 0
        lines.append(f"{f:<15} {n:>6} {s:>6} {100*s/n:>7.1f}% {med_rms:>9.3f}p {med_time:>9.3f}s")

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
        if sd not in by_subdir:
            by_subdir[sd] = []
        by_subdir[sd].append(r)
    for sd in sorted(by_subdir.keys()):
        rs = by_subdir[sd]
        n = len(rs)
        s = sum(1 for r in rs if r["status"] == "success")
        lines.append(f"{sd:<30} {n:>6} {s:>6} {100*s/n:>7.1f}%")

    # 失败帧列表
    if n_fail > 0:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  失败帧列表 ({n_fail} 帧)")
        lines.append("=" * 70)
        for r in results:
            if r["status"] != "success":
                lines.append(f"  {r['filename']}: {r['status']}")

    # 异常帧 (RMS > 5px 或 lnK < 10)
    bad_frames = [r for r in results if r["status"] == "success" and (
        r.get("rms_px", 0) > 5.0 or r.get("bayes_lnK", 0) < 10.0)]
    if bad_frames:
        lines.append("")
        lines.append("=" * 70)
        lines.append(f"  低质量帧 (RMS>5px 或 lnK<10, {len(bad_frames)} 帧)")
        lines.append("=" * 70)
        for r in bad_frames:
            lines.append(f"  {r['filename']}: RMS={r.get('rms_px', 0):.3f}px "
                         f"lnK={r.get('bayes_lnK', 0):.1f} "
                         f"matched={r.get('matched_count', 0)}")

    return lines


if __name__ == "__main__":
    main()

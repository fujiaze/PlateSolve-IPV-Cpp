"""V4.3 回归测试 - 重新测试全量测试中的低质量帧

功能:
    读取 full_test_all.json, 筛选低质量帧 (RMS>5px 或 lnK<10),
    用修复后的 V4.3 DLL 重新求解, 对比修复前后指标

输出:
    lib/plate_solve/logs/v4_3/regression_test/
    ├── regression_results.json
    ├── regression_summary.txt
    └── frames/<frame>/vm43_solve.log

用法:
    python 回归测试低质量帧.py
"""
import os
import sys
import json
import time
import re

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
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_3.vector_match_v4_3_cpp import V43Solver

# ============================================================================
# 路径
# ============================================================================
_FULL_TEST_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "full_test")
_FULL_TEST_JSON = os.path.join(_FULL_TEST_DIR, "full_test_all.json")

_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "regression_test")
_OUTPUT_JSON = os.path.join(_OUTPUT_DIR, "regression_results.json")
_OUTPUT_TXT = os.path.join(_OUTPUT_DIR, "regression_summary.txt")

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


def solve_single_frame(fits_path, solver):
    """单帧求解, 返回指标字典"""
    t_start = time.time()
    info = {"filename": os.path.basename(fits_path), "path": fits_path}
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
            info["status"] = "fail_no_objctra"
            return info
        cra0 = _parse_ra_hms(obj_ra_str)
        cdec0 = _parse_dec_dms(obj_dec_str)

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
            info["status"] = "fail_solve"
            info["solve_time_s"] = round(solve_time, 3)
            info["error"] = (result.get("error", "unknown") or "")[:200]
            return info

        info["status"] = "success"
        info["solve_time_s"] = round(solve_time, 3)
        info["matched_count"] = int(result.get("matched_count", 0))
        info["rms_px"] = round(float(result.get("rms_px", 0.0)), 4)
        info["rms_arcsec"] = round(float(result.get("rms_arcsec", 0.0)), 4)
        info["s_robust"] = round(float(result.get("s_robust", 0.0)), 4)
        info["n_inliers"] = int(result.get("n_inliers", 0))
        info["n_iters"] = int(result.get("n_iters", 0))
        info["irm_converged"] = bool(result.get("irm_converged", False))
        info["bayes_lnK"] = round(float(result.get("bayes_lnK", 0.0)), 2)
        info["triangle_pass_ratio"] = round(float(result.get("triangle_pass_ratio", 0.0)), 3)
        info["total_time_s"] = round(time.time() - t_start, 3)
    except Exception as e:
        info["status"] = f"error: {str(e)[:100]}"
        info["total_time_s"] = round(time.time() - t_start, 3)
    return info


def classify_frame(rms_px, lnK, matched):
    """分类失败类型"""
    if matched <= 2 and lnK == 0.0 and rms_px == 0.0:
        return "Type1_sparse"
    if rms_px > 50:
        return "Type2_wrong_conv"
    if lnK == 0.0 and rms_px < 1.0 and matched >= 13:
        return "Type3_verify_fail"
    if lnK < 10:
        return "low_lnK"
    if rms_px > 5:
        return "high_RMS"
    return "other"


def main():
    print("=== V4.3 回归测试 (低质量帧) ===")

    # 读取全量测试结果
    if not os.path.exists(_FULL_TEST_JSON):
        print(f"错误: 找不到全量测试结果 {_FULL_TEST_JSON}")
        return
    with open(_FULL_TEST_JSON, "r", encoding="utf-8") as f:
        full_results = json.load(f)
    print(f"全量测试结果: {len(full_results)} 帧")

    # 筛选低质量帧
    low_quality = []
    for r in full_results:
        if r.get("status") != "success":
            continue
        rms = float(r.get("rms_px", 0.0))
        lnK = float(r.get("bayes_lnK", 0.0))
        matched = int(r.get("matched_count", 0))
        if rms > 5.0 or lnK < 10.0:
            r["_fail_type"] = classify_frame(rms, lnK, matched)
            low_quality.append(r)
    print(f"低质量帧: {len(low_quality)} 帧")

    # 按失败类型统计
    type_count = {}
    for r in low_quality:
        t = r["_fail_type"]
        type_count[t] = type_count.get(t, 0) + 1
    print("\n失败类型分布:")
    for t, c in sorted(type_count.items()):
        print(f"  {t}: {c} 帧")

    os.makedirs(_OUTPUT_DIR, exist_ok=True)

    # 创建 V4.3 Solver
    print("\n创建 V4.3 V43Solver...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(gaia_client=gaia_client, star_detector=star_detector)

    # 重新求解
    new_results = []
    t_batch_start = time.time()
    for i, old_r in enumerate(low_quality, 1):
        fits_path = old_r["path"]
        base = os.path.basename(fits_path)
        print(f"\n[{i}/{len(low_quality)}] {base}")
        print(f"  旧: RMS={old_r.get('rms_px', 0)}px lnK={old_r.get('bayes_lnK', 0)} "
              f"matched={old_r.get('matched_count', 0)} type={old_r['_fail_type']}")

        new_r = solve_single_frame(fits_path, solver)
        new_r["_fail_type_old"] = old_r["_fail_type"]
        new_r["_old_rms_px"] = old_r.get("rms_px", 0)
        new_r["_old_lnK"] = old_r.get("bayes_lnK", 0)
        new_r["_old_matched"] = old_r.get("matched_count", 0)

        if new_r["status"] == "success":
            new_r["_fail_type_new"] = classify_frame(
                new_r["rms_px"], new_r["bayes_lnK"], new_r["matched_count"])
            improved = ""
            if new_r["bayes_lnK"] > old_r.get("bayes_lnK", 0) + 1:
                improved += " lnK↑"
            if new_r["rms_px"] < old_r.get("rms_px", 0) - 1:
                improved += " RMS↓"
            if new_r["matched_count"] > old_r.get("matched_count", 0) + 5:
                improved += " matched↑"
            print(f"  新: RMS={new_r['rms_px']}px lnK={new_r['bayes_lnK']} "
                  f"matched={new_r['matched_count']} type={new_r['_fail_type_new']}{improved}")
        else:
            print(f"  新: {new_r['status']} {new_r.get('error', '')}")

        new_results.append(new_r)

        # 增量保存
        with open(_OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(new_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t_batch_start
    print(f"\n=== 回归测试完成: {len(new_results)} 帧, 耗时 {elapsed:.1f}s ===")

    # 生成汇总
    write_summary(new_results, type_count)


def write_summary(results, old_type_count):
    """生成汇总报告"""
    lines = []
    lines.append("=" * 78)
    lines.append("  V4.3 回归测试汇总 (Bug 1-3 修复后)")
    lines.append("=" * 78)
    lines.append(f"测试帧数: {len(results)}")
    success = [r for r in results if r.get("status") == "success"]
    lines.append(f"成功: {len(success)} ({100.0*len(success)/max(len(results),1):.1f}%)")
    lines.append("")

    lines.append("修复前失败类型分布:")
    for t, c in sorted(old_type_count.items()):
        lines.append(f"  {t}: {c} 帧")
    lines.append("")

    # 修复后类型分布
    new_type_count = {}
    for r in results:
        if r.get("status") == "success":
            t = r.get("_fail_type_new", "unknown")
        else:
            t = "fail_solve"
        new_type_count[t] = new_type_count.get(t, 0) + 1
    lines.append("修复后类型分布:")
    for t, c in sorted(new_type_count.items()):
        lines.append(f"  {t}: {c} 帧")
    lines.append("")

    # 类型3 修复详情 (重点关注)
    type3_fixed = []
    type3_still_bad = []
    for r in results:
        if r.get("_fail_type_old") != "Type3_verify_fail":
            continue
        if r.get("status") != "success":
            type3_still_bad.append(r)
            continue
        new_t = r.get("_fail_type_new", "")
        if new_t == "Type3_verify_fail":
            type3_still_bad.append(r)
        else:
            type3_fixed.append(r)
    lines.append("-" * 78)
    lines.append(f"类型3 (lnK=0+RMS<1+matched>=13) 修复情况:")
    lines.append(f"  已修复: {len(type3_fixed)} 帧 (lnK>0 或类型改变)")
    lines.append(f"  仍异常: {len(type3_still_bad)} 帧")
    if type3_fixed:
        lines.append("")
        lines.append("  已修复帧示例 (前 10):")
        for r in type3_fixed[:10]:
            lines.append(
                f"    {r['filename']}: "
                f"lnK {r['_old_lnK']}→{r['bayes_lnK']}, "
                f"matched {r['_old_matched']}→{r['matched_count']}, "
                f"RMS {r['_old_rms_px']}→{r['rms_px']}px")
    if type3_still_bad:
        lines.append("")
        lines.append("  仍异常帧:")
        for r in type3_still_bad:
            lines.append(
                f"    {r['filename']}: "
                f"lnK={r.get('bayes_lnK', 0)}, "
                f"matched={r.get('matched_count', 0)}, "
                f"RMS={r.get('rms_px', 0)}px")
    lines.append("")

    # 其他类型改善情况
    lines.append("-" * 78)
    lines.append("其他类型改善情况:")
    for old_type in ["Type1_sparse", "Type2_wrong_conv", "low_lnK", "high_RMS"]:
        old_frames = [r for r in results if r.get("_fail_type_old") == old_type]
        if not old_frames:
            continue
        fixed = [r for r in old_frames
                 if r.get("status") == "success" and r.get("_fail_type_new", "") not in
                 ("Type1_sparse", "Type2_wrong_conv", "low_lnK", "high_RMS")]
        lines.append(f"  {old_type}: {len(old_frames)} → 修复 {len(fixed)}")

    with open(_OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n汇总已保存: {_OUTPUT_TXT}")


if __name__ == "__main__":
    main()

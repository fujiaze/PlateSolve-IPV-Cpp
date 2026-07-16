# -*- coding: utf-8 -*-
"""
DD-SPPS vs Astrometry.net 50 帧验证测试
功能: 选取 50 帧测试数据 (覆盖不同目标+滤镜), 调用 DD-SPPS 求解, 同时上传到
      astrometry.net 求解, 对比两者的 WCS (CRVAL/θ/pixscale) 一致性
用途: 用独立参考 (astrometry.net) 验证 DD-SPPS 频域盲解析算法的精度与可靠性
依赖: blind_index_v3 (pipeline/wcs/io_helpers), astrometry_client, requests

输出:
    - logs/astrometry_validation_report.txt: 统计报告
    - logs/astrometry_validation_details.json: 每帧详细数据 (便于后续分析)

运行:
    cd "f:\\Astro dev\\Astro CS Normalization Database"
    python lib/plate_solve/blind_index_v3/tests/test_astrometry_validation.py
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import numpy as np

# 项目根目录注入 sys.path
_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_MODULE_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_LOG_DIR = os.path.join(_MODULE_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_REPORT_PATH = os.path.join(_LOG_DIR, "astrometry_validation_report.txt")
_DETAILS_PATH = os.path.join(_LOG_DIR, "astrometry_validation_details.json")

# 日志
from lib.plate_solve.blind_index_v2.python.logging_setup import setup_logging, get_logger
setup_logging()
logger = get_logger("ddspps.test_astrometry")

# DD-SPPS 主管线
from lib.plate_solve.blind_index_v3.python.pipeline import solve_blind, BlindSolveResult
from lib.plate_solve.blind_index_v3.python import wcs as wcs_mod
from lib.plate_solve.blind_index_v3.python import io_helpers as io_helpers_mod

# Astrometry.net 客户端
from lib.plate_solve.blind_index_v3.python.astrometry_client import AstrometryClient, AstrometryCalibration

# IO 接口 (从 FITS 头读 s0/指向)
from lib.plate_solve.blind_index_v2.python.io_wrappers import read_image, get_s0_from_header, get_pointing_from_header

# API key (用户提供)
API_KEY = "mddxoocqojimmqfy"

# 测试参数
N_FRAMES = 50
RANDOM_SEED = 42  # 可复现
_ASTROMETRY_MAX_WAIT = 600  # 单 job 最大等待 (秒)


# ============================================================================
# 帧选择
# ============================================================================
def select_frames(lights_dir: str, n: int = 50) -> List[str]:
    """
    从 lights 目录选取 n 帧, 覆盖不同目标 + 滤镜。

    策略: 按目标分组 → 每目标按滤镜分组 → 均匀采样
    """
    # 扫描所有 .fts/.fit/.fits 文件
    patterns = ["*.fts", "*.fit", "*.fits"]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(os.path.join(lights_dir, "**", pat), recursive=True))
    # 排除 T2 子目录的 NGC4945 (单独子目录, 可能不在所有场景使用)
    # 实际保留所有, 让按目标采样来平衡
    logger.info("扫描到 %d 个 FITS 文件", len(all_files))

    # 按目标分组 (从文件名前缀提取)
    target_groups = {}
    for path in all_files:
        fname = os.path.basename(path)
        # 文件名格式: NGC4945_FD_T2_... 或 M20_T2_... 或 LDN43_... 等
        # 取第一个 _ 之前的部分作为目标
        target = fname.split("_")[0]
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(path)

    # 按目标均匀采样
    targets = sorted(target_groups.keys())
    n_targets = len(targets)
    per_target = max(1, n // n_targets)
    logger.info("目标数: %d, 每目标采样: %d", n_targets, per_target)

    random.seed(RANDOM_SEED)
    selected = []
    for target in targets:
        files = target_groups[target]
        # 按滤镜分组 (从文件名提取滤镜, 如 -Red.fts, -H-alpha.fts 等)
        filter_groups = {}
        for path in files:
            fname = os.path.basename(path)
            # 提取 -Xxx.fts 中的 Xxx
            stem = os.path.splitext(fname)[0]
            if "-" in stem:
                filt = stem.split("-")[-1]
            else:
                filt = "Unknown"
            if filt not in filter_groups:
                filter_groups[filt] = []
            filter_groups[filt].append(path)

        # 每个滤镜采样 1-2 帧
        target_selected = []
        for filt, fpaths in filter_groups.items():
            sample_n = min(max(1, per_target // len(filter_groups) + 1), len(fpaths))
            sampled = random.sample(fpaths, sample_n)
            target_selected.extend(sampled)

        # 限制每目标不超过 per_target + 2
        if len(target_selected) > per_target + 2:
            target_selected = random.sample(target_selected, per_target + 2)
        selected.extend(target_selected)

        logger.info("目标 %s: %d 帧 → 采样 %d 帧", target, len(files), len(target_selected))

    # 限制总数
    if len(selected) > n:
        selected = random.sample(selected, n)
    elif len(selected) < n:
        # 不够, 从剩余补充
        remaining = [f for f in all_files if f not in selected]
        if remaining:
            selected.extend(random.sample(remaining, min(n - len(selected), len(remaining))))

    logger.info("最终选取 %d 帧", len(selected))
    return selected


# ============================================================================
# 工具函数
# ============================================================================
def haversine_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg) -> float:
    """haversine 球面角距离 (角秒)。"""
    ra1 = math.radians(ra1_deg)
    dec1 = math.radians(dec1_deg)
    ra2 = math.radians(ra2_deg)
    dec2 = math.radians(dec2_deg)
    dra = ra2 - ra1
    ddec = dec2 - dec1
    a = math.sin(ddec / 2.0) ** 2 + math.cos(dec1) * math.cos(dec2) * math.sin(dra / 2.0) ** 2
    a = max(0.0, min(1.0, a))
    return 2.0 * math.asin(math.sqrt(a)) * (180.0 / math.pi) * 3600.0


def theta_from_cd(cd11: float, cd12: float, cd21: float, cd22: float) -> float:
    """从 CD 矩阵提取旋转角 (度), 用 atan2(cd21, cd22) 约定。"""
    return math.degrees(math.atan2(cd21, cd22))


def angular_diff_mod180(a_deg: float, b_deg: float) -> float:
    """角度差折叠到 [-90, 90] (考虑 180° 共轭对称)。"""
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    if d > 90.0:
        d -= 180.0
    elif d < -90.0:
        d += 180.0
    return d


# ============================================================================
# 单帧测试
# ============================================================================
@dataclass
class FrameValidation:
    """单帧验证结果。"""
    name: str
    path: str
    # DD-SPPS 结果
    ddspps_success: bool = False
    ddspps_theta: float = 0.0
    ddspps_crval1: float = 0.0
    ddspps_crval2: float = 0.0
    ddspps_pixscale: float = 0.0
    ddspps_rms: float = 0.0
    ddspps_n_inliers: int = 0
    ddspps_elapsed: float = 0.0
    ddspps_fail_reason: str = ""
    # Astrometry.net 结果
    astrometry_success: bool = False
    astrometry_theta: float = 0.0
    astrometry_crval1: float = 0.0
    astrometry_crval2: float = 0.0
    astrometry_pixscale: float = 0.0
    astrometry_orientation: float = 0.0
    astrometry_job_id: int = -1
    astrometry_elapsed: float = 0.0
    astrometry_error: str = ""
    # 对比指标
    theta_diff_mod180: float = 0.0
    crval_diff_arcsec: float = 0.0
    pixscale_diff_pct: float = 0.0
    # FITS 头基础信息
    s0_from_header: float = 0.0
    header_ra: float = 0.0
    header_dec: float = 0.0
    # 异常
    exception: str = ""


def validate_single_frame(
    fits_path: str,
    client: AstrometryClient,
) -> FrameValidation:
    """对单帧运行 DD-SPPS + astrometry.net 验证。"""
    name = os.path.basename(fits_path)
    # 用文件名前缀作为短名 (去掉扩展名)
    short_name = os.path.splitext(name)[0]
    # 截断过长名称
    if len(short_name) > 40:
        short_name = short_name[:37] + "..."
    result = FrameValidation(name=short_name, path=fits_path)

    logger.info("=" * 80)
    logger.info("验证帧: %s", short_name)
    logger.info("=" * 80)

    # 1. 读取 FITS 头获取 s0 + 指向 (用于 astrometry.net 上传)
    try:
        # read_image 返回 (uint16_img, ImageMetadataPy), 只需 metadata
        _, metadata = read_image(fits_path)
        s0 = get_s0_from_header(metadata)
        if s0 is None:
            result.exception = "无法从 FITS 头读取 s0"
            logger.error(result.exception)
            return result
        result.s0_from_header = s0
        pointing = get_pointing_from_header(metadata)
        if pointing is not None:
            result.header_ra, result.header_dec = pointing
        logger.info("FITS 头: s0=%.4f\"/px, 指向=(%.6f, %.6f)", s0, result.header_ra, result.header_dec)
    except Exception as e:
        result.exception = f"读取 FITS 头失败: {e}"
        logger.error(result.exception)
        return result

    # 2. DD-SPPS 求解
    t0 = time.time()
    try:
        solve_res = solve_blind(fits_path)
        result.ddspps_elapsed = time.time() - t0
        result.ddspps_success = solve_res.success
        if solve_res.best_result is not None and solve_res.best_result.wcs is not None:
            br = solve_res.best_result
            wcs = br.wcs
            result.ddspps_theta = wcs.theta_deg
            result.ddspps_crval1 = wcs.crval1
            result.ddspps_crval2 = wcs.crval2
            # pixscale 从 CD 矩阵计算: s = sqrt(|CD|) × 3600
            det = abs(wcs.cd11 * wcs.cd22 - wcs.cd12 * wcs.cd21)
            result.ddspps_pixscale = math.sqrt(det) * 3600.0
            result.ddspps_rms = br.rms_arcsec
            result.ddspps_n_inliers = br.n_inliers
            logger.info("DD-SPPS: θ=%.3f°, CRVAL=(%.6f, %.6f), s=%.4f\"/px, RMS=%.2f\", inliers=%d, 耗时=%.1fs",
                        result.ddspps_theta, result.ddspps_crval1, result.ddspps_crval2,
                        result.ddspps_pixscale, result.ddspps_rms, result.ddspps_n_inliers,
                        result.ddspps_elapsed)
        else:
            result.ddspps_fail_reason = solve_res.fail_reason or "无 best_result"
            logger.warning("DD-SPPS 失败: %s, 耗时=%.1fs", result.ddspps_fail_reason, result.ddspps_elapsed)
    except Exception as e:
        result.ddspps_elapsed = time.time() - t0
        result.exception = f"DD-SPPS 异常: {e}\n{traceback.format_exc()}"
        logger.error(result.exception)
        return result

    # 3. Astrometry.net 求解
    t0 = time.time()
    try:
        cal = client.solve(
            fits_path,
            s0=s0,
            center_ra=result.header_ra if result.header_ra != 0 else None,
            center_dec=result.header_dec if result.header_dec != 0 else None,
            radius_deg=2.0,
            max_wait=_ASTROMETRY_MAX_WAIT,
        )
        result.astrometry_elapsed = time.time() - t0
        if cal is not None and cal.success:
            result.astrometry_success = True
            result.astrometry_theta = theta_from_cd(cal.cd11, cal.cd12, cal.cd21, cal.cd22)
            result.astrometry_crval1 = cal.crval1
            result.astrometry_crval2 = cal.crval2
            result.astrometry_pixscale = cal.pixscale
            result.astrometry_orientation = cal.orientation
            result.astrometry_job_id = cal.job_id
            logger.info("Astrometry.net: θ=%.3f°, CRVAL=(%.6f, %.6f), s=%.4f\"/px, 耗时=%.1fs",
                        result.astrometry_theta, result.astrometry_crval1, result.astrometry_crval2,
                        result.astrometry_pixscale, result.astrometry_elapsed)
        else:
            result.astrometry_error = "求解失败或超时"
            logger.warning("Astrometry.net 失败: %s, 耗时=%.1fs", result.astrometry_error, result.astrometry_elapsed)
    except Exception as e:
        result.astrometry_elapsed = time.time() - t0
        result.astrometry_error = f"异常: {e}"
        logger.error("Astrometry.net 异常: %s", e)

    # 4. 对比指标 (两者都成功时)
    if result.ddspps_success and result.astrometry_success:
        result.theta_diff_mod180 = angular_diff_mod180(
            result.ddspps_theta, result.astrometry_theta
        )
        result.crval_diff_arcsec = haversine_arcsec(
            result.ddspps_crval1, result.ddspps_crval2,
            result.astrometry_crval1, result.astrometry_crval2,
        )
        if result.astrometry_pixscale > 0:
            result.pixscale_diff_pct = abs(
                result.ddspps_pixscale - result.astrometry_pixscale
            ) / result.astrometry_pixscale * 100.0
        logger.info("对比: θ_diff(mod180)=%.3f°, CRVAL_diff=%.2f\", pixscale_diff=%.2f%%",
                    result.theta_diff_mod180, result.crval_diff_arcsec, result.pixscale_diff_pct)

    return result


# ============================================================================
# 报告生成
# ============================================================================
def write_report(results: List[FrameValidation]) -> str:
    """生成统计报告并写入文件。"""
    lines = []
    lines.append("=" * 90)
    lines.append("DD-SPPS vs Astrometry.net 50 帧验证报告")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试帧数: {len(results)}")
    lines.append("=" * 90)
    lines.append("")

    # 统计
    n_ddspps_success = sum(1 for r in results if r.ddspps_success)
    n_astrometry_success = sum(1 for r in results if r.astrometry_success)
    n_both_success = sum(1 for r in results if r.ddspps_success and r.astrometry_success)
    n_pass = sum(1 for r in results if r.ddspps_success and r.astrometry_success
                 and abs(r.theta_diff_mod180) < 1.0 and r.crval_diff_arcsec < 10.0)

    lines.append("【1】总体成功率")
    lines.append("-" * 90)
    lines.append(f"  DD-SPPS 成功: %d/%d (%.1f%%)" % (n_ddspps_success, len(results), 100.0 * n_ddspps_success / len(results)))
    lines.append(f"  Astrometry.net 成功: %d/%d (%.1f%%)" % (n_astrometry_success, len(results), 100.0 * n_astrometry_success / len(results)))
    lines.append(f"  两者都成功: %d/%d (%.1f%%)" % (n_both_success, len(results), 100.0 * n_both_success / len(results)))
    lines.append(f"  验证通过 (θ_diff<1° 且 CRVAL_diff<10\"): %d/%d (%.1f%%)" % (n_pass, len(results), 100.0 * n_pass / len(results)))
    lines.append("")

    # 每帧详情表
    lines.append("【2】每帧详情")
    lines.append("-" * 90)
    header = f"  {'frame':<40} | {'DDSPPS':>6} | {'Astro':>6} | {'θ_diff':>8} | {'CRVAL_d':>8} | {'s_diff%':>7} | {'RMS':>6} | {'inl':>4}"
    lines.append(header)
    lines.append("  " + "-" * 100)
    for r in results:
        ddspps_flag = "OK" if r.ddspps_success else "FAIL"
        astro_flag = "OK" if r.astrometry_success else "FAIL"
        if r.ddspps_success and r.astrometry_success:
            theta_str = f"{r.theta_diff_mod180:+.3f}°"
            crval_str = f"{r.crval_diff_arcsec:.2f}\""
            s_str = f"{r.pixscale_diff_pct:.2f}%"
        else:
            theta_str = "-"
            crval_str = "-"
            s_str = "-"
        rms_str = f"{r.ddspps_rms:.2f}" if r.ddspps_success else "-"
        inl_str = f"{r.ddspps_n_inliers}" if r.ddspps_success else "-"
        lines.append(
            f"  {r.name:<40} | {ddspps_flag:>6} | {astro_flag:>6} | {theta_str:>8} | {crval_str:>8} | {s_str:>7} | {rms_str:>6} | {inl_str:>4}"
        )
    lines.append("")

    # 对比统计
    both_results = [r for r in results if r.ddspps_success and r.astrometry_success]
    if both_results:
        theta_diffs = [abs(r.theta_diff_mod180) for r in both_results]
        crval_diffs = [r.crval_diff_arcsec for r in both_results]
        s_diffs = [r.pixscale_diff_pct for r in both_results]
        rms_list = [r.ddspps_rms for r in both_results]

        lines.append("【3】对比统计 (两者都成功的 %d 帧)" % len(both_results))
        lines.append("-" * 90)
        lines.append(f"  θ_diff (mod180, 度):")
        lines.append(f"    平均: {np.mean(theta_diffs):.4f}°")
        lines.append(f"    中位: {np.median(theta_diffs):.4f}°")
        lines.append(f"    最大: {np.max(theta_diffs):.4f}°")
        lines.append(f"    P90:  {np.percentile(theta_diffs, 90):.4f}°")
        lines.append(f"  CRVAL_diff (角秒):")
        lines.append(f"    平均: {np.mean(crval_diffs):.2f}\"")
        lines.append(f"    中位: {np.median(crval_diffs):.2f}\"")
        lines.append(f"    最大: {np.max(crval_diffs):.2f}\"")
        lines.append(f"    P90:  {np.percentile(crval_diffs, 90):.2f}\"")
        lines.append(f"  pixscale_diff (百分比):")
        lines.append(f"    平均: {np.mean(s_diffs):.3f}%%")
        lines.append(f"    最大: {np.max(s_diffs):.3f}%%")
        lines.append(f"  DD-SPPS RMS (角秒):")
        lines.append(f"    平均: {np.mean(rms_list):.3f}\"")
        lines.append(f"    中位: {np.median(rms_list):.3f}\"")
        lines.append("")

    # 失败帧
    fail_frames = [r for r in results if not r.ddspps_success or not r.astrometry_success]
    if fail_frames:
        lines.append("【4】失败帧详情")
        lines.append("-" * 90)
        for r in fail_frames:
            lines.append(f"  {r.name}:")
            if not r.ddspps_success:
                lines.append(f"    DD-SPPS: {r.ddspps_fail_reason or '失败'}")
            if not r.astrometry_success:
                lines.append(f"    Astrometry.net: {r.astrometry_error or '失败'}")
        lines.append("")

    # 耗时统计
    ddspps_times = [r.ddspps_elapsed for r in results if r.ddspps_elapsed > 0]
    astro_times = [r.astrometry_elapsed for r in results if r.astrometry_elapsed > 0]
    lines.append("【5】耗时统计")
    lines.append("-" * 90)
    if ddspps_times:
        lines.append(f"  DD-SPPS: 平均 {np.mean(ddspps_times):.1f}s, 中位 {np.median(ddspps_times):.1f}s, 总 {np.sum(ddspps_times):.0f}s")
    if astro_times:
        lines.append(f"  Astrometry.net: 平均 {np.mean(astro_times):.1f}s, 中位 {np.median(astro_times):.1f}s, 总 {np.sum(astro_times):.0f}s")
    lines.append("")

    # 写入文件
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("报告已写入: %s", _REPORT_PATH)

    # 详细数据 JSON (便于后续分析)
    details = [asdict(r) for r in results]
    with open(_DETAILS_PATH, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logger.info("详细数据已写入: %s", _DETAILS_PATH)

    return _REPORT_PATH


# ============================================================================
# 主函数
# ============================================================================
def main():
    logger.info("DD-SPPS vs Astrometry.net 50 帧验证测试")
    logger.info("API key: %s****", API_KEY[:4])

    # 1. 选取 50 帧
    lights_dir = os.path.join(_PROJECT_ROOT, "testdata", "lights")
    if not os.path.isdir(lights_dir):
        logger.error("lights 目录不存在: %s", lights_dir)
        return
    frames = select_frames(lights_dir, N_FRAMES)
    logger.info("选取 %d 帧:", len(frames))
    for i, path in enumerate(frames, 1):
        logger.info("  %2d. %s", i, os.path.basename(path))

    # 2. 登录 astrometry.net
    client = AstrometryClient(API_KEY)
    if not client.login():
        logger.error("Astrometry.net 登录失败, 退出")
        return

    # 3. 逐帧测试
    results = []
    for i, path in enumerate(frames, 1):
        logger.info("")
        logger.info(">>> 进度: %d/%d <<<", i, len(frames))
        try:
            result = validate_single_frame(path, client)
            results.append(result)
        except Exception as e:
            logger.error("帧 %s 异常: %s\n%s", path, e, traceback.format_exc())
            results.append(FrameValidation(
                name=os.path.basename(path),
                path=path,
                exception=f"{type(e).__name__}: {e}",
            ))

        # 中间结果保存 (防止中断丢失)
        if i % 5 == 0:
            try:
                write_report(results)
                logger.info("中间结果已保存 (%d/%d)", i, len(frames))
            except Exception as e:
                logger.warning("中间结果保存失败: %s", e)

    # 4. 最终报告
    report_path = write_report(results)

    # 5. 控制台摘要
    n_ddspps = sum(1 for r in results if r.ddspps_success)
    n_astro = sum(1 for r in results if r.astrometry_success)
    n_both = sum(1 for r in results if r.ddspps_success and r.astrometry_success)
    n_pass = sum(1 for r in results if r.ddspps_success and r.astrometry_success
                 and abs(r.theta_diff_mod180) < 1.0 and r.crval_diff_arcsec < 10.0)
    logger.info("")
    logger.info("=" * 80)
    logger.info("验证完成:")
    logger.info("  DD-SPPS 成功: %d/%d", n_ddspps, len(results))
    logger.info("  Astrometry.net 成功: %d/%d", n_astro, len(results))
    logger.info("  两者都成功: %d/%d", n_both, len(results))
    logger.info("  验证通过 (θ<1° 且 CRVAL<10\"): %d/%d (%.1f%%)", n_pass, len(results), 100.0 * n_pass / len(results) if len(results) > 0 else 0.0)
    logger.info("报告: %s", report_path)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

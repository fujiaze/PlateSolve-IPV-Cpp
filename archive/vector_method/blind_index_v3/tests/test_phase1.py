# -*- coding: utf-8 -*-
"""
DD-SPPS Phase 1 测试与验证脚本 (Task 8)
功能: 对 4 帧 ADV-PA Phase 1 测试数据 (M20_T2/LDN43/NGC247_T2/NGC55_T3) 端到端调用
      solve_blind, 对比 FITS 头 WCS 与 DD-SPPS 求解结果 (旋转角/CRVAL/RMS/n_inliers)
用途: 验证 DD-SPPS 频域盲解析原型在真实天文图像上的精度与成功率, 输出统计报告
      到 logs/phase1_report.txt, 并生成诊断图像 (验证匹配图) 到 logs/

运行:
    cd "f:\\Astro dev\\Astro CS Normalization Database"
    python lib/plate_solve/blind_index_v3/tests/test_phase1.py

依赖: blind_index_v3 (pipeline/wcs/diagnostics), blind_index_v2.io_wrappers
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# 项目根目录注入 sys.path (允许独立运行)
_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 模块路径
_MODULE_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_LOG_DIR = os.path.join(_MODULE_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_REPORT_PATH = os.path.join(_LOG_DIR, "phase1_report.txt")

# 初始化日志 (复用 blind_index_v2 日志系统)
from lib.plate_solve.blind_index_v2.python.logging_setup import setup_logging, get_logger
setup_logging()
logger = get_logger("ddspps.test_phase1")

# DD-SPPS 主管线
from lib.plate_solve.blind_index_v3.python.pipeline import solve_blind, BlindSolveResult, ModeResult
from lib.plate_solve.blind_index_v3.python import wcs as wcs_mod
from lib.plate_solve.blind_index_v3.python import density as density_mod
from lib.plate_solve.blind_index_v3.python import signal as signal_mod
from lib.plate_solve.blind_index_v3.python import phase_correlation as pc_mod
from lib.plate_solve.blind_index_v3.python import diagnostics as diag_mod

# IO 接口
from lib.plate_solve.blind_index_v2.python.io_wrappers import (
    read_image, detect_stars, get_pointing_from_header, get_s0_from_header,
)


# ============================================================================
# 测试帧定义 (与 ADV-PA Phase 1 一致)
# ============================================================================
TEST_FRAMES = [
    {
        "name": "M20_T2",
        "path": os.path.join(_PROJECT_ROOT, "testdata", "lights",
                            "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts"),
        "scale": "中焦",
    },
    {
        "name": "LDN43",
        "path": os.path.join(_PROJECT_ROOT, "testdata", "lights",
                            "LDN43_LRGBH_flying_dutchman-20250503@032713-1200S-Red.fts"),
        "scale": "中焦",
    },
    {
        "name": "NGC247_T2",
        "path": os.path.join(_PROJECT_ROOT, "testdata", "lights",
                            "NGC247_T2_flying_dutchman-20250816@034607-600S-Red.fts"),
        "scale": "窄带",
    },
    {
        "name": "NGC55_T3",
        "path": os.path.join(_PROJECT_ROOT, "testdata", "lights",
                            "NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts"),
        "scale": "宽场",
    },
]

# 接受条件 (与 pipeline 内部一致)
_MIN_INLIER_RATIO = 2.0 / 3.0  # 仅用于 N_bright 比较 (不强制)
_MAX_RMS_ARCSEC = 5.0          # RMS < 5.0" (与 pipeline 一致)
_MIN_INLIER_FRAC = 0.005       # n_inliers ≥ 0.5%×min(N_stars, N_gaia)
_MIN_INLIERS_ABS = 5           # 最低绝对值
_LOW_RMS_THRESHOLD = 1.0       # 低 RMS 兜底阈值
_LOW_RMS_MIN_INLIERS = 3       # 低 RMS 兜底最低 inliers
_THETA_TOL_DEG = 1.0
_CRVAL_TOL_ARCSEC = 30.0


# ============================================================================
# 工具函数
# ============================================================================
def haversine_arcsec(ra1_deg: float, dec1_deg: float, ra2_deg: float, dec2_deg: float) -> float:
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


def extract_header_wcs(metadata) -> Optional[dict]:
    """
    从 ImageMetadataPy 提取参考 WCS 信息。

    返回 dict:
        crval1, crval2 (度)
        cd11, cd12, cd21, cd22
        theta_deg: atan2(cd21, cd22) 度 (与 DD-SPPS theta_best 同约定, spec §6.1)
        pixel_scale: arcsec/pixel
    若无 WCS 返回 None。
    """
    wcs = metadata.wcs if metadata is not None else None
    if wcs is None or not wcs.has_wcs:
        return None
    cd11 = float(wcs.cd1_1)
    cd12 = float(wcs.cd1_2)
    cd21 = float(wcs.cd2_1)
    cd22 = float(wcs.cd2_2)
    theta = math.degrees(math.atan2(cd21, cd22))
    return {
        "crval1": float(wcs.crval1),
        "crval2": float(wcs.crval2),
        "crpix1": float(wcs.crpix1),
        "crpix2": float(wcs.crpix2),
        "cd11": cd11, "cd12": cd12, "cd21": cd21, "cd22": cd22,
        "theta_deg": theta,
        "pixel_scale": float(wcs.pixel_scale),
    }


def angular_diff(a_deg: float, b_deg: float) -> float:
    """角度差折叠到 [-180, 180)。"""
    return (a_deg - b_deg + 180.0) % 360.0 - 180.0


def angular_diff_mod180(a_deg: float, b_deg: float) -> float:
    """角度差折叠到 [-90, 90] (考虑 180° 共轭对称)。"""
    d = angular_diff(a_deg, b_deg)
    if d > 90.0:
        d -= 180.0
    elif d < -90.0:
        d += 180.0
    return d


# ============================================================================
# 诊断图像生成 (重新运行 verify_wcs 获取 matched_pairs 用于 plot_verification)
# ============================================================================
def generate_verification_plot(
    image_path: str,
    best_wcs: wcs_mod.WCSResult,
    out_path: str,
    s0: float,
    query_ra: float,
    query_dec: float,
    fov_deg: float,
    g_cutoff: float,
) -> int:
    """
    重新读取图像 + 检测星 + 查询 Gaia, 调用 verify_wcs 获取 matched_pairs,
    然后绘制验证匹配图。

    Returns:
        matched_pairs 数量 (失败返回 0)
    """
    try:
        # 1. 读取图像
        uint16_img, metadata = read_image(image_path)
        image_w = uint16_img.shape[1]
        image_h = uint16_img.shape[0]
        # 2. 星点检测
        stars = detect_stars(uint16_img)
        if stars.count < 5:
            logger.warning("诊断图: 星点数不足 %d", stars.count)
            return 0
        # 3. Gaia 子集 (用相同查询参数)
        gaia_ra, gaia_dec, gaia_mag = density_mod.load_gaia_subset(
            query_ra, query_dec, fov_deg, g_cutoff,
        )
        if len(gaia_ra) < 5:
            logger.warning("诊断图: Gaia 星数不足 %d", len(gaia_ra))
            return 0
        # 4. verify_wcs
        verify = wcs_mod.verify_wcs(
            best_wcs, stars.x, stars.y, gaia_ra, gaia_dec, s0,
            sigma_pos=1.5, image_w=image_w, image_h=image_h,
        )
        # 5. 绘图
        diag_mod.plot_verification(
            best_wcs, stars.x, stars.y, gaia_ra, gaia_dec,
            verify.matched_pairs, out_path=out_path,
        )
        return len(verify.matched_pairs)
    except Exception as e:
        logger.warning("诊断图生成失败 (%s): %s", image_path, e)
        return 0


# ============================================================================
# 单帧测试
# ============================================================================
@dataclass
class FrameResult:
    """单帧测试结果汇总。"""
    name: str
    path: str
    scale: str
    file_exists: bool = False
    # 头 WCS 参考
    header_wcs: Optional[dict] = None
    # DD-SPPS 求解结果
    solve_result: Optional[BlindSolveResult] = None
    # 异常
    exception: str = ""
    # 关键对比指标 (求解成功后填充)
    theta_header: float = 0.0
    theta_best: float = 0.0
    theta_diff_deg: float = 0.0
    theta_diff_mod180_deg: float = 0.0
    crval_header_ra: float = 0.0
    crval_header_dec: float = 0.0
    crval_ddspps_ra: float = 0.0
    crval_ddspps_dec: float = 0.0
    crval_diff_arcsec: float = 0.0
    rms_arcsec: float = 0.0
    n_inliers: int = 0
    n_bright: int = 0
    inlier_ratio: float = 0.0
    elapsed_sec: float = 0.0
    best_mode: int = -1
    success: bool = False
    fail_reason: str = ""
    # 诊断图路径
    diag_plot_path: str = ""
    diag_matched_count: int = 0


def test_single_frame(frame: dict) -> FrameResult:
    """对单帧运行 solve_blind 并对比 header WCS。"""
    name = frame["name"]
    path = frame["path"]
    scale = frame["scale"]
    result = FrameResult(name=name, path=path, scale=scale)

    logger.info("=" * 70)
    logger.info("测试帧: %s [%s]", name, scale)
    logger.info("路径: %s", path)
    logger.info("=" * 70)

    if not os.path.isfile(path):
        result.file_exists = False
        result.exception = f"文件不存在: {path}"
        logger.error(result.exception)
        return result
    result.file_exists = True

    # 1. 读取 FITS 头 WCS 作为参考
    try:
        _, metadata = read_image(path)
        result.header_wcs = extract_header_wcs(metadata)
        if result.header_wcs is None:
            logger.warning("FITS 头无 WCS, 旋转角/CRVAL 对比将跳过")
        else:
            logger.info("头 WCS: CRVAL=(%.6f, %.6f), θ=%.3f°, s=%.4f\"/px",
                        result.header_wcs["crval1"], result.header_wcs["crval2"],
                        result.header_wcs["theta_deg"], result.header_wcs["pixel_scale"])
    except Exception as e:
        logger.warning("读取头 WCS 失败: %s", e)

    # 2. 调用 solve_blind (s0=None, query_ra/dec=None → 从 FITS 头读)
    try:
        solve_res = solve_blind(path)
        result.solve_result = solve_res
        result.success = solve_res.success
        result.best_mode = solve_res.best_mode
        result.elapsed_sec = solve_res.elapsed_sec
        result.n_bright = solve_res.n_bright
        result.fail_reason = solve_res.fail_reason
        if solve_res.best_result is not None:
            br = solve_res.best_result
            result.theta_best = br.theta_best
            result.rms_arcsec = br.rms_arcsec
            result.n_inliers = br.n_inliers
            if br.wcs is not None:
                result.crval_ddspps_ra = br.wcs.crval1
                result.crval_ddspps_dec = br.wcs.crval2
        # inlier ratio
        if result.n_bright > 0:
            result.inlier_ratio = float(result.n_inliers) / float(result.n_bright)
    except Exception as e:
        result.exception = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error("solve_blind 抛异常: %s", e)
        logger.error(traceback.format_exc())
        return result

    # 3. 对比指标
    if result.header_wcs is not None and solve_res.best_result is not None:
        hdr = result.header_wcs
        result.theta_header = hdr["theta_deg"]
        result.crval_header_ra = hdr["crval1"]
        result.crval_header_dec = hdr["crval2"]
        # 旋转角差 (考虑 180° 共轭对称)
        result.theta_diff_deg = angular_diff(result.theta_best, result.theta_header)
        result.theta_diff_mod180_deg = angular_diff_mod180(result.theta_best, result.theta_header)
        # CRVAL 球面距离
        result.crval_diff_arcsec = haversine_arcsec(
            result.crval_header_ra, result.crval_header_dec,
            result.crval_ddspps_ra, result.crval_ddspps_dec,
        )
        logger.info("对比: θ_diff=%.3f° (mod180=%.3f°), CRVAL_diff=%.2f\", RMS=%.2f\", "
                    "n_inliers=%d/%d (ratio=%.2f)",
                    result.theta_diff_deg, result.theta_diff_mod180_deg,
                    result.crval_diff_arcsec, result.rms_arcsec,
                    result.n_inliers, result.n_bright, result.inlier_ratio)

    # 4. 诊断图 (best WCS 存在时)
    if solve_res.best_result is not None and solve_res.best_result.wcs is not None:
        diag_path = os.path.join(_LOG_DIR, f"verification_{name}.png")
        # 用 FITS 头指向 (与 solve_blind 内部一致) 作为查询中心
        query_ra = result.header_wcs["crval1"] if result.header_wcs else solve_res.best_result.wcs.crval1
        query_dec = result.header_wcs["crval2"] if result.header_wcs else solve_res.best_result.wcs.crval2
        n_matched = generate_verification_plot(
            path, solve_res.best_result.wcs, diag_path,
            s0=solve_res.s0, query_ra=query_ra, query_dec=query_dec,
            fov_deg=solve_res.fov_deg, g_cutoff=solve_res.g_cutoff,
        )
        result.diag_plot_path = diag_path
        result.diag_matched_count = n_matched

    return result


# ============================================================================
# 报告生成
# ============================================================================
def _format_modes_table(solve_res: BlindSolveResult) -> List[str]:
    """生成 4 模式详细表格行。"""
    lines = []
    lines.append(f"    {'mode':>4} | {'theta_cand':>10} | {'theta_best':>10} | {'dx_sub':>8} | {'dy_sub':>8} | "
                 f"{'peak_snr':>9} | {'n_inliers':>9} | {'rms':>8} | {'accepted':>8}")
    lines.append("    " + "-" * 100)
    for m in solve_res.all_modes:
        lines.append(
            f"    {m.flip_mode:>4d} | {m.theta_cand:>10.3f} | {m.theta_best:>10.3f} | "
            f"{m.dx_sub:>8.3f} | {m.dy_sub:>8.3f} | {m.peak_snr:>9.3f} | "
            f"{m.n_inliers:>9d} | {m.rms_arcsec:>8.3f} | {('YES' if m.accepted else 'no'):>8}"
        )
    return lines


def write_report(frame_results: List[FrameResult]) -> str:
    """生成统计报告并写入 logs/phase1_report.txt, 返回报告路径。"""
    lines: List[str] = []

    lines.append("=" * 80)
    lines.append("DD-SPPS Phase 1 测试报告")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试帧数: {len(frame_results)}")
    lines.append("=" * 80)
    lines.append("")

    # ----- 测试帧列表 + 参数 -----
    lines.append("【1】测试帧列表与参数")
    lines.append("-" * 80)
    lines.append(f"{'name':<12} | {'scale':<8} | {'exists':<6} | {'path'}")
    lines.append("-" * 80)
    for fr in frame_results:
        lines.append(f"{fr.name:<12} | {fr.scale:<8} | {'yes' if fr.file_exists else 'NO':<6} | {fr.path}")
    lines.append("")

    # ----- 每帧详细结果 -----
    lines.append("【2】每帧详细结果")
    lines.append("-" * 80)
    for fr in frame_results:
        lines.append(f"\n>>> {fr.name} [{fr.scale}]")
        lines.append(f"  路径: {fr.path}")
        if not fr.file_exists:
            lines.append(f"  [SKIP] {fr.exception}")
            continue
        if fr.exception:
            lines.append(f"  [异常] solve_blind 抛出异常:")
            for ln in fr.exception.splitlines():
                lines.append(f"    {ln}")
            continue

        sr = fr.solve_result
        if sr is None:
            lines.append("  [异常] solve_blind 未返回结果")
            continue

        # 头 WCS
        if fr.header_wcs:
            hdr = fr.header_wcs
            lines.append(f"  头 WCS: CRVAL=({hdr['crval1']:.6f}, {hdr['crval2']:.6f}), "
                         f"θ_header=atan2(cd21,cd22)={hdr['theta_deg']:.3f}°, "
                         f"s0_header={hdr['pixel_scale']:.4f}\"/px")
            lines.append(f"          CD=[[{hdr['cd11']:.3e},{hdr['cd12']:.3e}],"
                         f"[{hdr['cd21']:.3e},{hdr['cd22']:.3e}]]")
        else:
            lines.append("  头 WCS: 无 (FITS 头未含 WCS)")

        # DD-SPPS 求解参数
        lines.append(f"  DD-SPPS: s0={sr.s0:.4f}\"/px, FOV={sr.fov_deg:.4f}°, "
                     f"ρ={sr.rho:.3f}/deg², N_bright={sr.n_bright}, "
                     f"G_cutoff={sr.g_cutoff:.2f}, N_stars={sr.n_stars}, N_gaia={sr.n_gaia}")

        # 成功状态
        lines.append(f"  状态: {'SUCCESS' if sr.success else 'FAILURE'}"
                     f" (best_mode={sr.best_mode}, elapsed={sr.elapsed_sec:.2f}s)")
        if sr.success:
            br = sr.best_result
            if br is not None and br.wcs is not None:
                lines.append(f"  最佳解: θ_best={br.theta_best:.3f}°, dx={br.dx_sub:.3f}, dy={br.dy_sub:.3f}, "
                             f"peak_snr={br.peak_snr:.3f}")
                lines.append(f"          CRVAL=({br.wcs.crval1:.6f}, {br.wcs.crval2:.6f}), "
                             f"n_inliers={br.n_inliers}, RMS={br.rms_arcsec:.3f}\", "
                             f"s_out={br.s_out:.4f}\" (consistent={br.s_consistent})")

        # 对比指标
        if fr.header_wcs:
            lines.append(f"  对比指标:")
            lines.append(f"    旋转角: θ_header={fr.theta_header:.3f}°, θ_best={fr.theta_best:.3f}°, "
                         f"diff={fr.theta_diff_deg:+.3f}° (mod180={fr.theta_diff_mod180_deg:+.3f}°)")
            theta_ok = abs(fr.theta_diff_mod180_deg) < _THETA_TOL_DEG
            lines.append(f"            [旋转角偏差 < 1°: {'PASS' if theta_ok else 'FAIL'} "
                         f"(mod180 差 {abs(fr.theta_diff_mod180_deg):.3f}°)]")
            lines.append(f"    CRVAL: header=({fr.crval_header_ra:.6f}, {fr.crval_header_dec:.6f}), "
                         f"ddspps=({fr.crval_ddspps_ra:.6f}, {fr.crval_ddspps_dec:.6f})")
            crval_ok = fr.crval_diff_arcsec < _CRVAL_TOL_ARCSEC
            lines.append(f"           [CRVAL 偏差 < 30\": {'PASS' if crval_ok else 'FAIL'} "
                         f"(实际 {fr.crval_diff_arcsec:.2f}\")]")
            rms_ok = fr.rms_arcsec < _MAX_RMS_ARCSEC
            n_for_thresh = min(fr.solve_result.n_stars, fr.solve_result.n_gaia)
            min_inliers = max(_MIN_INLIERS_ABS, int(math.ceil(_MIN_INLIER_FRAC * max(n_for_thresh, 1))))
            inl_ok = (fr.n_inliers >= min_inliers and fr.rms_arcsec < _MAX_RMS_ARCSEC) \
                     or (fr.n_inliers >= _LOW_RMS_MIN_INLIERS and fr.rms_arcsec < _LOW_RMS_THRESHOLD)
            lines.append(f"    RMS: {fr.rms_arcsec:.3f}\" [< 5.0\": {'PASS' if rms_ok else 'FAIL'}]")
            lines.append(f"    n_inliers: {fr.n_inliers}/{fr.n_bright} (ratio={fr.inlier_ratio:.2f}, "
                         f"阈值 0.5%×min(N_stars,N_gaia)={min_inliers} 或 (≥3 且 RMS<1\")) [{'PASS' if inl_ok else 'FAIL'}]")

        # 4 模式详细
        lines.append("  4 翻转模式详细:")
        lines.extend(_format_modes_table(sr))

        # 失败原因
        if not sr.success and sr.fail_reason:
            lines.append(f"  失败原因: {sr.fail_reason}")

        # 诊断图
        if fr.diag_plot_path:
            lines.append(f"  诊断图: {fr.diag_plot_path} (matched={fr.diag_matched_count})")

    lines.append("")
    lines.append("-" * 80)

    # ----- 汇总表 -----
    lines.append("")
    lines.append("【3】汇总表")
    lines.append("-" * 80)
    _crval_lbl = 'crval_diff"'
    _rms_lbl = 'rms"'
    header = (f"{'name':<12} | {'success':<7} | {'theta_diff°':>11} | {_crval_lbl:>11} | "
              f"{_rms_lbl:>7} | {'n_inliers':>9} | {'ratio':>5} | {'mode':>4} | {'elapsed':>7}")
    lines.append(header)
    lines.append("-" * 80)
    for fr in frame_results:
        if not fr.file_exists or fr.exception or fr.solve_result is None:
            lines.append(f"{fr.name:<12} | ERROR")
            continue
        lines.append(
            f"{fr.name:<12} | {'OK' if fr.success else 'FAIL':<7} | "
            f"{fr.theta_diff_mod180_deg:+11.3f} | {fr.crval_diff_arcsec:>11.2f} | "
            f"{fr.rms_arcsec:>7.3f} | {fr.n_inliers:>9d} | {fr.inlier_ratio:>5.2f} | "
            f"{fr.best_mode:>4d} | {fr.elapsed_sec:>7.2f}s"
        )
    lines.append("-" * 80)

    # ----- 汇总统计 -----
    valid_results = [fr for fr in frame_results if fr.file_exists and not fr.exception and fr.solve_result is not None]
    n_total = len(valid_results)
    n_success = sum(1 for fr in valid_results if fr.success)
    succ_rate = (n_success / n_total * 100.0) if n_total > 0 else 0.0

    # 旋转角/CRVAL/RMS 平均 (仅成功帧)
    succ_results = [fr for fr in valid_results if fr.success and fr.header_wcs]
    avg_theta_diff = (sum(abs(fr.theta_diff_mod180_deg) for fr in succ_results) / len(succ_results)) if succ_results else 0.0
    avg_crval_diff = (sum(fr.crval_diff_arcsec for fr in succ_results) / len(succ_results)) if succ_results else 0.0
    avg_rms = (sum(fr.rms_arcsec for fr in succ_results) / len(succ_results)) if succ_results else 0.0
    avg_elapsed = (sum(fr.elapsed_sec for fr in valid_results) / n_total) if n_total > 0 else 0.0

    lines.append("")
    lines.append("【4】汇总统计")
    lines.append("-" * 80)
    lines.append(f"  总帧数: {n_total}")
    lines.append(f"  成功帧数: {n_success}")
    lines.append(f"  成功率: {succ_rate:.1f}%")
    if succ_results:
        lines.append(f"  成功帧平均 |θ_diff| (mod180): {avg_theta_diff:.3f}°")
        lines.append(f"  成功帧平均 CRVAL 偏差: {avg_crval_diff:.2f}\"")
        lines.append(f"  成功帧平均 RMS: {avg_rms:.3f}\"")
    else:
        lines.append(f"  (无成功帧, 平均指标跳过)")
    lines.append(f"  全部帧平均耗时: {avg_elapsed:.2f}s")

    # ----- 各模式命中统计 -----
    lines.append("")
    lines.append("【5】各 flip_mode 命中统计")
    lines.append("-" * 80)
    mode_hits = {0: 0, 1: 0, 2: 0, 3: 0}
    for fr in valid_results:
        if fr.success and fr.best_mode in mode_hits:
            mode_hits[fr.best_mode] += 1
    for m, cnt in mode_hits.items():
        lines.append(f"  flip_mode {m}: {cnt} 命中")
    # 每个模式平均 peak_snr / n_inliers (无论是否成功)
    lines.append("")
    lines.append("  各模式平均指标 (全部帧, 含失败):")
    lines.append(f"    {'mode':>4} | {'avg_peak_snr':>12} | {'avg_n_inliers':>13} | {'avg_rms':>8} | {'best_hits':>9}")
    for m in (0, 1, 2, 3):
        m_results = [fr.solve_result.all_modes[m] for fr in valid_results if fr.solve_result and len(fr.solve_result.all_modes) > m]
        if m_results:
            avg_pk = sum(r.peak_snr for r in m_results) / len(m_results)
            avg_inl = sum(r.n_inliers for r in m_results) / len(m_results)
            avg_rm = sum(r.rms_arcsec for r in m_results if r.rms_arcsec != float("inf")) / max(1, sum(1 for r in m_results if r.rms_arcsec != float("inf")))
            lines.append(f"    {m:>4d} | {avg_pk:>12.3f} | {avg_inl:>13.2f} | {avg_rm:>8.3f} | {mode_hits[m]:>9d}")
        else:
            lines.append(f"    {m:>4d} | {'N/A':>12} | {'N/A':>13} | {'N/A':>8} | {mode_hits[m]:>9d}")

    # ----- 失败案例分析 -----
    lines.append("")
    lines.append("【6】失败案例分析")
    lines.append("-" * 80)
    n_fail = 0
    for fr in valid_results:
        if not fr.success:
            n_fail += 1
            lines.append(f"  [{fr.name}] best_mode={fr.best_mode}, fail_reason:")
            # 缩进多行
            for ln in fr.fail_reason.split(" | "):
                lines.append(f"    - {ln}")
    if n_fail == 0:
        lines.append("  (无失败案例)")

    # ----- spec checklist -----
    lines.append("")
    lines.append("【7】验证标准检查 (spec checklist)")
    lines.append("-" * 80)
    # 标准 1: 1D 相位相关旋转角偏差 < 1°
    n_theta_ok = sum(1 for fr in succ_results if abs(fr.theta_diff_mod180_deg) < _THETA_TOL_DEG)
    lines.append(f"  [标准 1] 1D 相位相关旋转角偏差 < 1°: {n_theta_ok}/{len(succ_results)} 成功帧通过")
    if succ_results and n_theta_ok < len(succ_results):
        lines.append(f"           (考虑 180° 共轭对称, 用 mod180 差评估; 不达标记录为已知限制)")
    # 标准 2: 2D 相位相关平移 CRVAL < 30"
    n_crval_ok = sum(1 for fr in succ_results if fr.crval_diff_arcsec < _CRVAL_TOL_ARCSEC)
    lines.append(f"  [标准 2] 2D 相位相关平移 CRVAL < 30\": {n_crval_ok}/{len(succ_results)} 成功帧通过")
    # 标准 3: WCS RMS < 5.0" 且 n_inliers ≥ 0.5%×min(N_stars, N_gaia) (或低 RMS 兜底)
    n_rms_inl_ok = sum(1 for fr in succ_results
                       if (fr.rms_arcsec < _MAX_RMS_ARCSEC
                           and fr.n_inliers >= max(_MIN_INLIERS_ABS, int(math.ceil(_MIN_INLIER_FRAC * max(min(fr.solve_result.n_stars, fr.solve_result.n_gaia), 1)))))
                       or (fr.n_inliers >= _LOW_RMS_MIN_INLIERS and fr.rms_arcsec < _LOW_RMS_THRESHOLD))
    lines.append(f"  [标准 3] WCS RMS < 5.0\" 且 n_inliers ≥ 0.5%×min(N_stars,N_gaia) (或 ≥3 且 RMS<1\"): {n_rms_inl_ok}/{len(succ_results)} 成功帧通过")
    # 标准 4: 4 翻转模式至少 1 种命中
    n_mode_hit = sum(1 for fr in valid_results if fr.success)
    lines.append(f"  [标准 4] 4 翻转模式至少 1 种命中正确解: {n_mode_hit}/{n_total} 帧有模式命中")
    # 标准 5: 报告输出
    lines.append(f"  [标准 5] 统计报告输出到 logs/phase1_report.txt: PASS (本文件)")

    # ----- 已知问题 -----
    lines.append("")
    lines.append("【8】已知问题")
    lines.append("-" * 80)
    lines.append("  1. 旋转角搜索 (已用全范围搜索替代 1D 相位相关):")
    lines.append("     - 1D 相位相关受方形网格 4 重对称 (0/90/180/270°) 限制, 无法恢复非 90° 旋转角")
    lines.append("     - 改为 search_rotation_full: 粗搜索 0~360° (5° 步长) + 精细搜索 (0.5° 步长)")
    lines.append("     - 直接用 2D phase correlation 峰值评分, 能求解任意角度")
    lines.append("  2. Gaia 星查询 (已修复, G_cutoff 提升):")
    lines.append("     - G_cutoff 表已提升至 12.5-15.0, 确保稀疏星场有足够 Gaia 星")
    lines.append("     - build_gaia_signal 投影后按网格范围裁剪")
    lines.append("  3. FITS 头无 WCS (NGC55_T3 场景):")
    lines.append("     - 该帧未 plate-solve, 但有 OBJCTRA/OBJCTDEC + FOCALLEN + XPIXSZ")
    lines.append("     - io_helpers.get_pointing_from_fits 从 OBJCTRA/DEC 读取指向")
    lines.append("     - s0 从 FOCALLEN/XPIXSZ 推导: s0 = 206.265 × XPIXSZ / FOCALLEN")
    lines.append("  4. 旋转角对比考虑 180° 对称 (mod180 差), 因 DD-SPPS 与 header 约定可能差 180°")
    lines.append("  5. CRVAL 偏差由旋转角误差 + 平移误差共同贡献, 旋转角偏差大时 CRVAL 偏差会放大")

    lines.append("")
    lines.append("=" * 80)
    lines.append("报告结束")
    lines.append("=" * 80)

    report_text = "\n".join(lines)
    try:
        with open(_REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info("报告已写入: %s", _REPORT_PATH)
    except OSError as e:
        logger.error("写报告失败: %s", e)
    return _REPORT_PATH


# ============================================================================
# 主入口
# ============================================================================
def main() -> int:
    logger.info("=" * 70)
    logger.info("DD-SPPS Phase 1 测试开始")
    logger.info("测试帧数: %d", len(TEST_FRAMES))
    logger.info("报告输出: %s", _REPORT_PATH)
    logger.info("诊断图目录: %s", _LOG_DIR)
    logger.info("=" * 70)

    frame_results: List[FrameResult] = []
    for frame in TEST_FRAMES:
        try:
            fr = test_single_frame(frame)
        except Exception as e:
            logger.error("测试帧 %s 顶层异常: %s", frame["name"], e)
            logger.error(traceback.format_exc())
            fr = FrameResult(name=frame["name"], path=frame["path"], scale=frame["scale"],
                             exception=f"顶层异常: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        frame_results.append(fr)

    # 生成报告
    report_path = write_report(frame_results)

    # 控制台简要输出
    print("\n" + "=" * 70)
    print("DD-SPPS Phase 1 测试完成")
    print("=" * 70)
    n_total = len([f for f in frame_results if f.file_exists and not f.exception])
    n_success = sum(1 for f in frame_results if f.success)
    print(f"成功率: {n_success}/{n_total}")
    for fr in frame_results:
        status = "OK" if fr.success else "FAIL"
        theta_d = fr.theta_diff_mod180_deg if fr.header_wcs else float("nan")
        crval_d = fr.crval_diff_arcsec if fr.header_wcs else float("nan")
        print(f"  {fr.name:<12} [{status}] mode={fr.best_mode} "
              f"θ_diff={theta_d:+.3f}° CRVAL_diff={crval_d:.2f}\" "
              f"RMS={fr.rms_arcsec:.3f}\" inliers={fr.n_inliers}/{fr.n_bright} "
              f"elapsed={fr.elapsed_sec:.2f}s")
    print(f"\n详细报告: {report_path}")
    print(f"诊断图目录: {_LOG_DIR}")
    return 0 if n_success > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

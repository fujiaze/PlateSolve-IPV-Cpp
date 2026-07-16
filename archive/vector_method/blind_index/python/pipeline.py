"""
4SADQ-KV盲解析主管线
功能: 串联星点检测→DR3查询→参考索引构建→图像四边形生成→k-vector匹配→投票→WCS求解
用途: 端到端盲解析入口, 仅以像素尺度s0为先验(测试harness可用FITS指向查询DR3构建本地索引)
依赖: numpy, 所有子模块, lib.plate_solve.python.vector_match_v2 (gnomonic_forward)

管线阶段:
    1. 读取图像 (astro_image_io)
    2. 星点检测 (star_detector)
    3. 读取s0 (FITS头或参数)
    4. 查询DR3 + gnomonic投影 + 参考四边形生成 + k-vector索引构建
    5. 图像四边形生成 (金字塔策略)
    6. k-vector查询 + 5距离验证
    7. Kolomenkin投票
    8. WCS求解 (Umeyama SVD)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .logging_setup import get_logger, setup_logging
from .io_wrappers import (
    read_image, detect_stars, query_dr3, get_pixel_scale_from_header,
    get_pointing_from_header, StarDetectionResult,
)
from .quad_geometry import Quad, ReferenceQuad
from .quad_selector import generate_image_quads, generate_reference_quads
from .kvector import KVectorIndex, build_kvector
from .matcher import match_all_quads, compute_sigma_d, DEFAULT_SIGMA_POS_PIXEL
from .wcs_solver import WCSResult
from .voting import vote, VoteResult

# 复用V3.5 gnomonic投影
from lib.plate_solve.python.vector_match_v2 import gnomonic_forward

logger = get_logger(__name__)

# 默认参数
DEFAULT_MAG_LIMIT = 12.0           # DR3查询极限星等
DEFAULT_FOV_MARGIN = 1.5           # FOV查询余量倍数


@dataclass
class SolveResult:
    """
    盲解析结果。

    Attributes:
        success: 是否成功
        wcs: WCS结果 (成功时)
        s0_arcsec_per_pixel: 使用的像素尺度
        n_detected_stars: 检测到的星点数
        n_reference_stars: DR3参考星数
        n_reference_quads: 参考四边形数
        n_image_quads: 图像四边形数
        n_candidates_total: 匹配候选总数
        best_votes: 最佳得票数
        best_rms_arcsec: 最佳RMS(arcsec)
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)
        stage_timings: 各阶段耗时(秒)字典
        message: 描述信息
    """
    success: bool = False
    wcs: Optional[WCSResult] = None
    s0_arcsec_per_pixel: float = 0.0
    n_detected_stars: int = 0
    n_reference_stars: int = 0
    n_reference_quads: int = 0
    n_image_quads: int = 0
    n_candidates_total: int = 0
    best_votes: int = 0
    best_rms_arcsec: float = float("inf")
    ra0: float = 0.0
    dec0: float = 0.0
    stage_timings: dict[str, float] = field(default_factory=dict)
    message: str = ""


def solve_blind(
    image_path: str,
    s0_arcsec_per_pixel: Optional[float] = None,
    query_center_ra: Optional[float] = None,
    query_center_dec: Optional[float] = None,
    mag_limit: float = DEFAULT_MAG_LIMIT,
    sigma_pos_pixel: float = DEFAULT_SIGMA_POS_PIXEL,
    data_dir: Optional[str] = None,
) -> SolveResult:
    """
    4SADQ-KV盲解析主管线。

    算法仅使用s0作为先验。query_center_ra/dec仅用于测试harness查询DR3构建本地索引,
    不传入匹配算法本身。

    Args:
        image_path: 图像文件路径
        s0_arcsec_per_pixel: 像素尺度(arcsec/pixel), None则从FITS头读取
        query_center_ra: DR3查询中心RA(度), 测试harness捷径
        query_center_dec: DR3查询中心Dec(度), 测试harness捷径
        mag_limit: DR3查询极限星等
        sigma_pos_pixel: 位置噪声(像素)
        data_dir: GaiaDR3数据目录

    Returns:
        SolveResult
    """
    setup_logging()
    result = SolveResult()
    timings = result.stage_timings

    logger.info("=" * 70)
    logger.info("4SADQ-KV盲解析开始: %s", image_path)
    logger.info("=" * 70)

    # ═══ 阶段1: 读取图像 ═══
    t0 = time.time()
    try:
        uint16_img, metadata = read_image(image_path)
    except Exception as e:
        result.message = f"图像读取失败: {e}"
        logger.error(result.message)
        return result
    h, w = uint16_img.shape
    timings["read_image"] = time.time() - t0
    logger.info("阶段1 完成: 图像 %dx%d (%.3fs)", w, h, timings["read_image"])

    # ═══ 阶段2: 星点检测 ═══
    t0 = time.time()
    try:
        star_result = detect_stars(uint16_img)
    except Exception as e:
        result.message = f"星点检测失败: {e}"
        logger.error(result.message)
        return result
    timings["detect_stars"] = time.time() - t0
    result.n_detected_stars = star_result.count
    logger.info("阶段2 完成: %d颗星 (%.3fs)", result.n_detected_stars, timings["detect_stars"])

    if star_result.count < 4:
        result.message = f"星点数不足: {star_result.count} < 4"
        logger.error(result.message)
        return result

    # ═══ 阶段3: 读取s0 ═══
    t0 = time.time()
    if s0_arcsec_per_pixel is None:
        s0 = get_pixel_scale_from_header(metadata)
        if s0 is None or s0 <= 0:
            result.message = "无法获取像素尺度s0"
            logger.error(result.message)
            return result
    else:
        s0 = float(s0_arcsec_per_pixel)
    result.s0_arcsec_per_pixel = s0
    timings["read_s0"] = time.time() - t0
    logger.info("阶段3 完成: s0=%.4f arcsec/pixel (%.3fs)", s0, timings["read_s0"])

    # ═══ 阶段4: 构建本地DR3索引 ═══
    t0 = time.time()

    # 4a. 确定查询中心
    if query_center_ra is not None and query_center_dec is not None:
        ra0 = float(query_center_ra)
        dec0 = float(query_center_dec)
    else:
        pointing = get_pointing_from_header(metadata)
        if pointing is None:
            result.message = "未提供query_center且FITS头无指向, 无法构建本地索引"
            logger.error(result.message)
            return result
        ra0, dec0 = pointing
    result.ra0 = ra0
    result.dec0 = dec0

    # 4b. 计算查询半径 (FOV对角线 × 余量)
    fov_diag_arcsec = float(np.sqrt(w ** 2 + h ** 2) * s0)
    fov_diag_deg = fov_diag_arcsec / 3600.0
    radius_deg = fov_diag_deg * DEFAULT_FOV_MARGIN
    logger.info("查询中心: (%.5f, %.5f), FOV对角线=%.4f°, 查询半径=%.4f°",
                 ra0, dec0, fov_diag_deg, radius_deg)

    # 4c. 查询DR3
    try:
        ra_arr, dec_arr, mag_arr = query_dr3(ra0, dec0, radius_deg, mag_limit, data_dir)
    except Exception as e:
        result.message = f"DR3查询失败: {e}"
        logger.error(result.message)
        return result
    result.n_reference_stars = len(ra_arr)
    if result.n_reference_stars < 4:
        result.message = f"参考星数不足: {result.n_reference_stars} < 4"
        logger.error(result.message)
        return result

    # 4d. gnomonic投影到切平面
    xi_arr, eta_arr, valid = gnomonic_forward(ra_arr, dec_arr, ra0, dec0)
    xi_arr = xi_arr[valid]
    eta_arr = eta_arr[valid]
    ra_arr = ra_arr[valid]
    dec_arr = dec_arr[valid]
    logger.info("gnomonic投影: %d颗有效参考星", len(ra_arr))

    # 4d-bis. 过滤参考星到 FOV 内: 仅保留落在图像范围内的星,
    # 使参考星密度与图像亮星池密度匹配(否则 margin 星污染索引,
    # 参考四边形最近邻结构与图像四边形不一致, 5距离验证无法通过)。
    half_w_arcsec = (w / 2.0) * s0
    half_h_arcsec = (h / 2.0) * s0
    fov_mask = (np.abs(xi_arr) <= half_w_arcsec) & (np.abs(eta_arr) <= half_h_arcsec)
    n_ref_in_fov = int(np.sum(fov_mask))
    xi_fov = xi_arr[fov_mask]
    eta_fov = eta_arr[fov_mask]
    ra_fov = ra_arr[fov_mask]
    dec_fov = dec_arr[fov_mask]
    logger.info("FOV过滤: %d/%d 参考星在FOV内 (半宽=%.1f\", 半高=%.1f\")",
                 n_ref_in_fov, len(xi_arr), half_w_arcsec, half_h_arcsec)

    # 4e. 参考四边形生成 (仅用 FOV 内参考星)
    ref_quads = generate_reference_quads(xi_fov, eta_fov, ra_fov, dec_fov)
    result.n_reference_quads = len(ref_quads)
    if result.n_reference_quads == 0:
        result.message = "参考四边形生成失败(全部退化)"
        logger.error(result.message)
        return result

    # 4f. k-vector索引构建
    kvector_index = build_kvector(ref_quads)
    if kvector_index is None:
        result.message = "k-vector索引构建失败"
        logger.error(result.message)
        return result
    timings["build_index"] = time.time() - t0
    logger.info("阶段4 完成: %d参考星(FOV内)→%d参考四边形, k-vector(N=%d, Δ=%.2f\", d_min=%.2f\", d_max=%.2f\") (%.3fs)",
                 n_ref_in_fov, result.n_reference_quads,
                 kvector_index.n_quads, kvector_index.delta,
                 kvector_index.d_min, kvector_index.d_max, timings["build_index"])

    # ═══ 阶段5: 图像四边形生成 ═══
    t0 = time.time()
    # 自适应亮星池大小 = FOV 内参考星数, 使图像亮星池密度≈参考星密度,
    # 保证图像四边形 d_AB 与参考四边形 d_AB 尺度匹配, 且最近邻结构一致。
    adaptive_pool = max(20, min(n_ref_in_fov, result.n_detected_stars))
    logger.info("自适应亮星池: 参考星在FOV内=%d → pool_size=%d", n_ref_in_fov, adaptive_pool)
    image_quads = generate_image_quads(
        star_result.x, star_result.y, s0,
        ref_kvector_index=kvector_index,
        pool_size=adaptive_pool,
        saturated_arr=star_result.saturated,
    )
    result.n_image_quads = len(image_quads)
    timings["gen_image_quads"] = time.time() - t0
    logger.info("阶段5 完成: %d个图像四边形 (%.3fs)", result.n_image_quads, timings["gen_image_quads"])

    if result.n_image_quads == 0:
        result.message = "图像四边形生成失败(全部退化)"
        logger.error(result.message)
        return result

    # ═══ 阶段6: k-vector查询 + 5距离验证 ═══
    t0 = time.time()
    candidates_per_quad = match_all_quads(
        image_quads, kvector_index, s0, sigma_pos_pixel
    )
    result.n_candidates_total = sum(len(c) for c in candidates_per_quad)
    timings["match"] = time.time() - t0
    logger.info("阶段6 完成: 总候选%d (%.3fs)", result.n_candidates_total, timings["match"])

    if result.n_candidates_total == 0:
        result.message = "无匹配候选"
        logger.error(result.message)
        return result

    # ═══ 阶段7: Kolomenkin投票 + WCS求解 ═══
    t0 = time.time()
    vote_result = vote(candidates_per_quad, ra0, dec0)
    timings["vote_wcs"] = time.time() - t0

    if vote_result.best_wcs is None:
        result.message = f"投票/WCS求解失败: {vote_result.message}"
        logger.error(result.message)
        return result

    result.wcs = vote_result.best_wcs
    result.best_votes = vote_result.best_votes
    result.best_rms_arcsec = vote_result.best_rms
    result.success = True
    result.message = vote_result.message
    logger.info("阶段7 完成: %s (%.3fs)", vote_result.message, timings["vote_wcs"])

    # ═══ 汇总 ═══
    total_time = sum(timings.values())
    logger.info("=" * 70)
    logger.info("4SADQ-KV盲解析完成: success=%s, RMS=%.3f\", 票数=%d, 总耗时=%.3fs",
                 result.success, result.best_rms_arcsec, result.best_votes, total_time)
    logger.info("  WCS: CRVAL=(%.5f, %.5f), CRPIX=(%.2f, %.2f), s=%.5f",
                 result.wcs.crval1, result.wcs.crval2,
                 result.wcs.crpix1, result.wcs.crpix2, result.wcs.s)
    logger.info("  CD=[%.6e, %.6e; %.6e, %.6e]",
                 result.wcs.cd[0, 0], result.wcs.cd[0, 1],
                 result.wcs.cd[1, 0], result.wcs.cd[1, 1])
    logger.info("=" * 70)
    return result

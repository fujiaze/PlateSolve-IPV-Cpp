"""
ADV-PA 盲解析主管线 (Task 7)
功能: 串联星点检测→DR3查询→星对库+k-vector索引构建→图像星对特征提取→
      (天区,旋转角)投票→峰值检测→向量法验证, 输出最终WCS
用途: 端到端盲解析入口, 仅以像素尺度s0为先验(测试harness可用FITS指向查询DR3构建本地索引)
依赖: numpy, 所有子模块, lib.plate_solve.python.vector_match_v2 (GaiaClientPy)

管线阶段:
    1. 读取图像 (astro_image_io)
    2. 星点检测 (star_detector)
    3. 读取s0 (FITS头或参数)
    4. 确定查询中心 + 查询DR3 + 构建星对库 + k-vector索引
    5. 图像星对特征提取 (限制Top-N最亮, 避免C(N,2)爆炸)
    6. k-vector查询 + (天区,旋转角)二维投票
    7. 峰值检测 → 候选(天区中心, 旋转角)
    8. 向量法验证 (Umeyama SVD + 迭代精修), 选RMS最低解
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .logging_setup import get_logger, setup_logging
from .io_wrappers import (
    read_image, detect_stars, query_dr3,
    get_s0_from_header, get_pointing_from_header, StarDetectionResult,
)
from .pair_index import (
    build_pair_library_with_kvector, PairLibrary,
    DEFAULT_K_NEIGHBORS, DEFAULT_DELTA, DEFAULT_D_MIN, DEFAULT_D_MAX,
)
from .image_features import extract_image_pairs
from .voting import (
    vote, detect_peaks, ang2pix, pix2ang, rot_bin_to_angle,
    DEFAULT_SIGMA_POS, DEFAULT_N_SIGMA, DEFAULT_TOP_K,
)
from .wcs_verify import verify_candidate, WCSResult

# 复用V3.5 GaiaClientPy (验证阶段查询局部星表)
from lib.plate_solve.python.vector_match_v2 import GaiaClientPy

logger = get_logger(__name__)

# ═══ 默认参数 ═══
DEFAULT_MAG_LIMIT = 12.0              # Phase 1 索引构建DR3查询极限星等
DEFAULT_VERIFY_MAG_LIMIT = 14.0       # 验证阶段Gaia查询极限星等
DEFAULT_FOV_MARGIN = 1.5              # 索引构建查询半径 = FOV对角线 × 1.5
DEFAULT_MAX_IMAGE_STARS = 100         # 参与配对的最大图像星数(避免C(N,2)爆炸)
DEFAULT_TOP_K = DEFAULT_TOP_K         # 候选峰值数, 默认5
DEFAULT_SIGMA_POS = DEFAULT_SIGMA_POS  # 位置噪声(像素)


@dataclass
class SolveResult:
    """
    ADV-PA 盲解析结果。

    Attributes:
        success: 是否成功求解WCS
        wcs: WCS结果(成功时), 否则None
        s0_arcsec_per_pixel: 使用的像素尺度(arcsec/pixel)
        best_rms_arcsec: 最佳RMS(arcsec), 失败时为inf
        stage_timings: 各阶段耗时(秒)字典
        n_detected: 检测到的星点数
        n_reference: DR3参考星数
        n_pairs: 星对库中星对数
        n_image_pairs: 提取的图像星对数
        n_candidates: 候选(天区,旋转角)数
        vote_peak: 投票峰值(最高票数)
        candidates_tried: 实际验证的候选数
        ra0, dec0: 索引构建中心(度, 测试harness用)
        message: 描述信息
    """
    success: bool = False
    wcs: Optional[WCSResult] = None
    s0_arcsec_per_pixel: float = 0.0
    best_rms_arcsec: float = float("inf")
    stage_timings: dict = field(default_factory=dict)
    n_detected: int = 0
    n_reference: int = 0
    n_pairs: int = 0
    n_image_pairs: int = 0
    n_candidates: int = 0
    vote_peak: int = 0
    candidates_tried: int = 0
    ra0: float = 0.0
    dec0: float = 0.0
    message: str = ""


def _select_brightest_stars(
    star_result: StarDetectionResult,
    max_stars: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从检测结果中选取Top-N最亮星点参与配对。

    优先选饱和星(位置可靠), 再按flux降序补足。饱和星flux=-1, 单独处理。

    Args:
        star_result: 检测结果
        max_stars: 最大星数

    Returns:
        (x_sel, y_sel): 选中的星点坐标
    """
    n = star_result.count
    if n <= max_stars:
        return star_result.x, star_result.y

    sat_mask = star_result.saturated > 0
    n_sat = int(np.sum(sat_mask))
    n_normal = n - n_sat

    if n_sat >= max_stars:
        # 饱和星已足够, 取前max_stars颗饱和星
        idx = np.where(sat_mask)[0][:max_stars]
        return star_result.x[idx], star_result.y[idx]

    # 取全部饱和星 + 按flux降序的正常星补足
    normal_idx = np.where(~sat_mask)[0]
    normal_flux = star_result.flux[normal_idx]
    sorted_order = np.argsort(-normal_flux)  # 降序
    n_normal_take = max_stars - n_sat
    normal_sel = normal_idx[sorted_order[:n_normal_take]]

    sat_idx = np.where(sat_mask)[0]
    sel_idx = np.concatenate([sat_idx, normal_sel])
    return star_result.x[sel_idx], star_result.y[sel_idx]


def solve_blind(
    image_path: str,
    s0_arcsec_per_pixel: Optional[float] = None,
    query_center_ra: Optional[float] = None,
    query_center_dec: Optional[float] = None,
    mag_limit: float = DEFAULT_MAG_LIMIT,
    sigma_pos: float = DEFAULT_SIGMA_POS,
    max_image_stars: int = DEFAULT_MAX_IMAGE_STARS,
    top_k: int = DEFAULT_TOP_K,
    data_dir: Optional[str] = None,
) -> SolveResult:
    """
    ADV-PA 盲解析主管线。

    算法仅使用s0作为先验。query_center_ra/dec用于测试harness查询DR3构建本地索引,
    不传入匹配算法本身。

    Args:
        image_path: 图像文件路径
        s0_arcsec_per_pixel: 像素尺度(arcsec/pixel), None则从FITS头读取
        query_center_ra: DR3查询中心RA(度), 测试harness捷径
        query_center_dec: DR3查询中心Dec(度), 测试harness捷径
        mag_limit: DR3查询极限星等(索引构建阶段)
        sigma_pos: 位置噪声(像素)
        max_image_stars: 参与配对的最大图像星数
        top_k: 峰值检测top-K
        data_dir: GaiaDR3数据目录

    Returns:
        SolveResult

    Raises:
        ValueError: 当query_center_ra/dec均为None且FITS头无指向时,
                    Phase 1无法构建本地索引
    """
    setup_logging()
    result = SolveResult()
    timings = result.stage_timings

    logger.info("=" * 70)
    logger.info("ADV-PA 盲解析开始: %s", image_path)
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
    result.n_detected = star_result.count
    logger.info("阶段2 完成: %d颗星 (%.3fs)", result.n_detected, timings["detect_stars"])

    if star_result.count < 4:
        result.message = f"星点数不足: {star_result.count} < 4"
        logger.error(result.message)
        return result

    # ═══ 阶段3: 读取s0 ═══
    t0 = time.time()
    if s0_arcsec_per_pixel is None:
        s0 = get_s0_from_header(metadata)
        if s0 is None or s0 <= 0:
            result.message = "无法获取像素尺度s0 (FITS头无WCS/FOCALLEN/XPIXSZ)"
            logger.error(result.message)
            return result
    else:
        s0 = float(s0_arcsec_per_pixel)
    result.s0_arcsec_per_pixel = s0
    timings["read_s0"] = time.time() - t0
    logger.info("阶段3 完成: s0=%.4f arcsec/pixel (%.3fs)", s0, timings["read_s0"])

    # ═══ 阶段4: 确定查询中心 + DR3查询 + 星对库 + k-vector索引 ═══
    t0 = time.time()

    # 4a. 确定查询中心 (Phase 1 测试harness)
    if query_center_ra is not None and query_center_dec is not None:
        ra0 = float(query_center_ra)
        dec0 = float(query_center_dec)
    else:
        pointing = get_pointing_from_header(metadata)
        if pointing is None:
            # Phase 1 必须 build local index, 无指向 → 抛 ValueError
            raise ValueError(
                "Phase 1 需要 query_center_ra/dec 构建本地索引, "
                "但参数未提供且FITS头无WCS指向。"
                "Phase 2 全天球索引完成后可无指向求解。"
            )
        ra0, dec0 = pointing
    result.ra0 = ra0
    result.dec0 = dec0

    # 4b. 计算FOV对角线 + 查询半径
    fov_diag_arcsec = float(np.sqrt(w ** 2 + h ** 2) * s0)
    fov_diag_deg = fov_diag_arcsec / 3600.0
    radius_deg = fov_diag_deg * DEFAULT_FOV_MARGIN
    logger.info("查询中心: (%.5f, %.5f), FOV对角线=%.4f°, 查询半径=%.4f°",
                 ra0, dec0, fov_diag_deg, radius_deg)

    # 4c. 查询DR3参考星
    try:
        ra_arr, dec_arr, mag_arr = query_dr3(ra0, dec0, radius_deg, mag_limit, data_dir)
    except Exception as e:
        result.message = f"DR3查询失败: {e}"
        logger.error(result.message)
        return result
    result.n_reference = len(ra_arr)
    if result.n_reference < 4:
        result.message = f"参考星数不足: {result.n_reference} < 4"
        logger.error(result.message)
        return result

    # 4d. 构建星对库 + k-vector索引 (一步完成)
    kv = build_pair_library_with_kvector(
        ra_arr, dec_arr, mag_arr,
        k_neighbors=DEFAULT_K_NEIGHBORS,
        delta=DEFAULT_DELTA,
        d_min=DEFAULT_D_MIN,
        d_max=DEFAULT_D_MAX,
    )
    if kv is None:
        result.message = "星对库/k-vector索引构建失败"
        logger.error(result.message)
        return result
    result.n_pairs = kv.n_pairs
    timings["build_index"] = time.time() - t0
    logger.info("阶段4 完成: %d参考星→%d星对, k-vector(N=%d, Δ=%.2f\", d∈[%.1f\", %.1f\"]) (%.3fs)",
                 result.n_reference, result.n_pairs,
                 kv.n_pairs, kv.delta, kv.d_min, kv.d_max, timings["build_index"])

    # ═══ 阶段5: 图像星对特征提取 ═══
    t0 = time.time()
    x_sel, y_sel = _select_brightest_stars(star_result, max_image_stars)
    if len(x_sel) < 4:
        result.message = f"选中星点数不足: {len(x_sel)} < 4"
        logger.error(result.message)
        return result
    d_img_arr, theta_img_arr, _, _ = extract_image_pairs(x_sel, y_sel, s0)
    result.n_image_pairs = len(d_img_arr)
    timings["extract_features"] = time.time() - t0
    logger.info("阶段5 完成: %d颗星→%d图像星对 (Top-%d最亮) (%.3fs)",
                 len(x_sel), result.n_image_pairs, max_image_stars, timings["extract_features"])

    if result.n_image_pairs == 0:
        result.message = "图像星对特征提取失败"
        logger.error(result.message)
        return result

    # ═══ 阶段6: k-vector查询 + (天区,旋转角)二维投票 ═══
    t0 = time.time()
    votes = vote(d_img_arr, theta_img_arr, kv, s0, sigma_pos, DEFAULT_N_SIGMA)
    timings["vote"] = time.time() - t0
    total_votes = sum(votes.values())
    logger.info("阶段6 完成: 总票数%d, 投票格%d (%.3fs)",
                 total_votes, len(votes), timings["vote"])

    # ═══ 阶段7: 峰值检测 ═══
    t0 = time.time()
    peaks = detect_peaks(votes, result.n_image_pairs, top_k=top_k)
    timings["detect_peaks"] = time.time() - t0

    if not peaks:
        result.message = (
            f"投票无峰值超过阈值 (总票数={total_votes}, "
            f"图像星对={result.n_image_pairs}, 噪声底线阈值法)"
        )
        logger.error(result.message)
        return result

    result.vote_peak = int(peaks[0][2])
    result.n_candidates = len(peaks)
    logger.info("阶段7 完成: %d个候选, 最高票数=%d (%.3fs)",
                 result.n_candidates, result.vote_peak, timings["detect_peaks"])

    # ═══ 阶段8: 向量法验证 (Top-K候选) ═══
    t0 = time.time()

    # 准备图像星点坐标(全量, 非Top-N, 让Umeyama有更多内点)
    image_xy = np.column_stack([star_result.x, star_result.y])

    # 打开GaiaClient用于验证阶段 (复用同一实例, 避免重复加载索引)
    if data_dir is None:
        from .io_wrappers import _DEFAULT_GAIA_DR3_DIR
        gaia_data_dir = _DEFAULT_GAIA_DR3_DIR
    else:
        gaia_data_dir = data_dir

    best_wcs: Optional[WCSResult] = None
    n_tried = 0

    try:
        gaia_client = GaiaClientPy(data_dir=gaia_data_dir, db_type=1)
    except Exception as e:
        result.message = f"GaiaClient初始化失败: {e}"
        logger.error(result.message)
        return result

    try:
        # fix-adv-pa-phase1-bugs Bug 3: 移除 early_exit (RMS<1" 即返回) 导致假阳性通过
        # 改为: 验证全部 top-K 候选, 要求 n_inliers >= max(10, 0.1*max_image_stars)
        # 在满足内点下限的解中选 RMS 最低; 若全候选均不达标则返回 FAILURE
        # 注: n_image_stars 指参与投票的图像星数 (max_image_stars), 非 len(image_xy)
        # 原 Phase 1 假阳性 n_inliers=9, threshold=10 即可拒绝
        min_inliers_required = max(10, int(0.1 * max_image_stars))
        logger.info("验证阶段: 内点下限=%d (max(10, 0.1*max_image_stars=%d))",
                     min_inliers_required, max_image_stars)

        for cand_idx, (healpix_id, rot_bin, count) in enumerate(peaks):
            n_tried += 1
            ra_center, dec_center = pix2ang(healpix_id)
            rot_angle = rot_bin_to_angle(rot_bin)
            logger.info("候选 %d/%d: 天区#%d→(%.4f,%.4f), rot=%.1f°, 票数=%d",
                         cand_idx + 1, len(peaks), healpix_id,
                         ra_center, dec_center, rot_angle, count)

            try:
                wcs_cand = verify_candidate(
                    image_xy=image_xy,
                    s0=s0,
                    ra_center=ra_center,
                    dec_center=dec_center,
                    rot_angle=rot_angle,
                    gaia_client=gaia_client,
                    fov_diag_deg=fov_diag_deg,
                    image_width=w,
                    image_height=h,
                    mag_limit=DEFAULT_VERIFY_MAG_LIMIT,
                    sigma_pos=sigma_pos,
                )
            except Exception as e:
                logger.warning("候选 %d 验证异常: %s", cand_idx + 1, e)
                continue

            if wcs_cand is None:
                logger.info("候选 %d 验证失败", cand_idx + 1)
                continue

            logger.info("候选 %d WCS: s=%.5f, RMS=%.3f\", n_inliers=%d (下限=%d)",
                         cand_idx + 1, wcs_cand.s, wcs_cand.rms_arcsec,
                         wcs_cand.n_inliers, min_inliers_required)

            # 内点不足的候选视为假阳性, 不接受
            if wcs_cand.n_inliers < min_inliers_required:
                logger.info("候选 %d 内点不足 (%d < %d), 拒绝 (假阳性抑制)",
                             cand_idx + 1, wcs_cand.n_inliers, min_inliers_required)
                continue

            # 选 RMS 最低且内点达标的解 (不再 early_exit)
            if best_wcs is None or wcs_cand.rms_arcsec < best_wcs.rms_arcsec:
                best_wcs = wcs_cand
                logger.info("候选 %d 当前最佳: RMS=%.3f\", n_inliers=%d",
                             cand_idx + 1, best_wcs.rms_arcsec, best_wcs.n_inliers)
    finally:
        try:
            gaia_client.close()
        except Exception:
            pass

    result.candidates_tried = n_tried
    timings["verify"] = time.time() - t0

    if best_wcs is None:
        result.message = (
            f"所有{n_tried}个候选验证失败或内点不足 "
            f"(下限={max(10, int(0.1 * max_image_stars))})"
        )
        logger.error(result.message)
        return result

    # ═══ 汇总 ═══
    result.wcs = best_wcs
    result.best_rms_arcsec = best_wcs.rms_arcsec
    result.success = True
    result.message = (
        f"ADV-PA 盲解析成功: RMS={best_wcs.rms_arcsec:.3f}\", "
        f"n_inliers={best_wcs.n_inliers}, s={best_wcs.s:.5f}"
    )

    total_time = sum(timings.values())
    logger.info("=" * 70)
    logger.info("ADV-PA 盲解析完成: success=%s, RMS=%.3f\", 候选尝试=%d, 总耗时=%.3fs",
                 result.success, result.best_rms_arcsec, n_tried, total_time)
    logger.info("  WCS: CRVAL=(%.5f, %.5f), CRPIX=(%.2f, %.2f), s=%.5f",
                 best_wcs.crval1, best_wcs.crval2,
                 best_wcs.crpix1, best_wcs.crpix2, best_wcs.s)
    logger.info("  CD=[%.6e, %.6e; %.6e, %.6e]",
                 best_wcs.cd[0, 0], best_wcs.cd[0, 1],
                 best_wcs.cd[1, 0], best_wcs.cd[1, 1])
    logger.info("  各阶段耗时: %s",
                 ", ".join(f"{k}={v:.3f}s" for k, v in timings.items()))
    logger.info("=" * 70)
    return result

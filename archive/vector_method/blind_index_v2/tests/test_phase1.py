# -*- coding: utf-8 -*-
"""
功能: Phase 1 区域内可靠性验证 — ADV-PA 盲解析管线多帧端到端测试
用途: 验证 ADV-PA 核心机制 (k-vector查询/投票聚集/V4收敛/鲁棒性) 在 testdata 多帧上可靠工作,
      覆盖 Task 8 全部子任务 (8.1~8.7): 成功率≥80%, SNR>10, RMS<3", CRVAL偏差<30", 鲁棒性
依赖: numpy, scipy, 项目内 lib.plate_solve.blind_index_v2 全模块 + lib.plate_solve.python.vector_match_v2

测试帧选择 (覆盖不同望远镜/指向):
    1. M20_T2   — has_wcs, s0=0.967"/px, FOV~1.1° (T2, 1917.6mm)
    2. LDN43    — has_wcs, s0=0.967"/px, FOV~1.1° (T1, 1917.6mm)
    3. NGC247_T2— has_wcs, s0=0.967"/px, FOV~1.1° (T2, 1917.4mm)
    4. NGC55_T3 — 无WCS, s0=0.989"/px(焦距公式), FOV~1.1° (T3, 1877mm), 查询中心用NGC55星表坐标
"""
from __future__ import annotations

import os
import sys
import time
import math
import io as _io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ═══ 项目路径 ═══
_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, _PROJECT_ROOT)

# 模块内导入
from lib.astro_image_io.python.astro_image_io import ImageReader
from lib.plate_solve.blind_index_v2.python.io_wrappers import (
    read_image, detect_stars, query_dr3, get_s0_from_header,
    get_pointing_from_header, StarDetectionResult,
)
from lib.plate_solve.blind_index_v2.python.pair_index import (
    build_pair_library_with_kvector, kvector_query, PairLibrary,
    DEFAULT_K_NEIGHBORS, DEFAULT_DELTA, DEFAULT_D_MIN, DEFAULT_D_MAX,
)
from lib.plate_solve.blind_index_v2.python.image_features import extract_image_pairs
from lib.plate_solve.blind_index_v2.python.voting import (
    vote, detect_peaks, pix2ang, rot_bin_to_angle,
    DEFAULT_SIGMA_POS, DEFAULT_N_SIGMA, DEFAULT_TOP_K,
)
from lib.plate_solve.blind_index_v2.python.wcs_verify import verify_candidate, WCSResult
from lib.plate_solve.blind_index_v2.python.pipeline import (
    solve_blind, SolveResult, _select_brightest_stars,
    DEFAULT_MAG_LIMIT, DEFAULT_VERIFY_MAG_LIMIT, DEFAULT_FOV_MARGIN,
    DEFAULT_MAX_IMAGE_STARS,
)
from lib.plate_solve.blind_index_v2.python import logging_setup
from lib.plate_solve.python.vector_match_v2 import (
    GaiaClientPy, gnomonic_forward, gnomonic_inverse,
)
from lib.plate_solve.blind_index_v2.python.spherical_geom import (
    angular_distance_arcsec, position_angle_deg,
)

# ═══ 常量 ═══
_DEGTORAD = math.pi / 180.0
_RADTOASEC = (180.0 / math.pi) * 3600.0

# 投票格数 (healpy不可用→等距网格回退)
# 网格: RA 429 bins × Dec 215 bins = 92235 天区格, 旋转 180 bins → 总 16,602,300 格
import lib.plate_solve.blind_index_v2.python.voting as _voting_mod
N_RA_BINS = _voting_mod._RA_BINS
N_DEC_BINS = _voting_mod._DEC_BINS
N_SKY_CELLS = N_RA_BINS * N_DEC_BINS          # 92235
N_ROT_BINS = 180
N_VOTE_CELLS = N_SKY_CELLS * N_ROT_BINS        # 16,602,300

# GaiaDR3 数据目录
_GAIA_DR3_DIR = os.path.join(_PROJECT_ROOT, "GaiaDR3")

# 日志/报告目录
_LOG_DIR = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "blind_index_v2", "logs")
_REPORT_PATH = os.path.join(_LOG_DIR, "phase1_report.txt")

# ═══ 测试帧定义 ═══
# s0=None → 从FITS头读; query_ra/dec=None → 从WCS读; expected_crval=None → 无baseline
TEST_FRAMES = [
    {
        "path": r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts",
        "name": "M20_T2",
        "s0": None,                       # 从WCS读 (0.967)
        "query_ra": None,                 # 从WCS读
        "query_dec": None,
        "expected_crval": (270.70003, -22.84992),  # WCS CRVAL作baseline
    },
    {
        "path": r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum.fts",
        "name": "LDN43",
        "s0": None,
        "query_ra": None,
        "query_dec": None,
        "expected_crval": (248.60954, -15.75894),
    },
    {
        "path": r"testdata\lights\NGC247_T2_flying_dutchman-20250816@033428-600S-Lum.fts",
        "name": "NGC247_T2",
        "s0": None,
        "query_ra": None,
        "query_dec": None,
        "expected_crval": (11.78946, -20.74164),
    },
    {
        "path": r"testdata\lights\NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts",
        "name": "NGC55_T3",
        "s0": 0.9890,                     # 无WCS, 用焦距公式 (1877mm, 9um)
        "query_ra": 3.721,                # NGC55 星系坐标 (J2000)
        "query_dec": -39.196,
        "expected_crval": None,           # 无WCS baseline
    },
]


# ═══ 工具函数 ═══

def haversine_arcsec(ra1_deg: float, dec1_deg: float,
                     ra2_deg: float, dec2_deg: float) -> float:
    """haversine 大圆角距离 (arcsec)"""
    ra1r = ra1_deg * _DEGTORAD
    dec1r = dec1_deg * _DEGTORAD
    ra2r = ra2_deg * _DEGTORAD
    dec2r = dec2_deg * _DEGTORAD
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = math.sin(ddec / 2.0) ** 2 + math.cos(dec1r) * math.cos(dec2r) * math.sin(dra / 2.0) ** 2
    a = max(0.0, min(1.0, a))
    return 2.0 * math.asin(math.sqrt(a)) * _RADTOASEC


def compute_snr(vote_peak: int, total_votes: int) -> float:
    """计算信噪比 SNR = vote_peak / (total_votes / n_cells)"""
    if total_votes <= 0 or N_VOTE_CELLS <= 0:
        return 0.0
    noise_floor = total_votes / N_VOTE_CELLS
    if noise_floor <= 0:
        return float("inf")
    return vote_peak / noise_floor


def get_total_time(result: SolveResult) -> float:
    """从stage_timings计算总耗时(秒)"""
    return float(sum(result.stage_timings.values()))


# ═══ 阶段4-8复现 (用于鲁棒性测试, 接受预检测星点) ═══

def solve_from_stars(
    star_result: StarDetectionResult,
    s0: float,
    w: int,
    h: int,
    query_ra: float,
    query_dec: float,
    mag_limit: float = DEFAULT_MAG_LIMIT,
    sigma_pos: float = DEFAULT_SIGMA_POS,
    max_image_stars: int = DEFAULT_MAX_IMAGE_STARS,
    top_k: int = DEFAULT_TOP_K,
    data_dir: Optional[str] = None,
) -> SolveResult:
    """
    用预检测星点运行管线阶段4-8 (复现 pipeline.solve_blind 的阶段4~8).
    用于鲁棒性测试: 修改星点列表后重新求解, 无需重新读图/检测.

    Returns:
        SolveResult (与 solve_blind 相同结构)
    """
    if data_dir is None:
        data_dir = _GAIA_DR3_DIR
    result = SolveResult()
    timings = result.stage_timings
    result.s0_arcsec_per_pixel = s0
    result.ra0 = query_ra
    result.dec0 = query_dec

    if star_result.count < 4:
        result.message = f"星点数不足: {star_result.count} < 4"
        return result

    # 阶段4: DR3查询 + 星对库 + k-vector
    t0 = time.time()
    fov_diag_arcsec = float(math.sqrt(w ** 2 + h ** 2) * s0)
    fov_diag_deg = fov_diag_arcsec / 3600.0
    radius_deg = fov_diag_deg * DEFAULT_FOV_MARGIN
    try:
        ra_arr, dec_arr, mag_arr = query_dr3(query_ra, query_dec, radius_deg, mag_limit, data_dir)
    except Exception as e:
        result.message = f"DR3查询失败: {e}"
        return result
    result.n_reference = len(ra_arr)
    if result.n_reference < 4:
        result.message = f"参考星数不足: {result.n_reference} < 4"
        return result
    kv = build_pair_library_with_kvector(
        ra_arr, dec_arr, mag_arr,
        k_neighbors=DEFAULT_K_NEIGHBORS, delta=DEFAULT_DELTA,
        d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX,
    )
    if kv is None:
        result.message = "星对库/k-vector索引构建失败"
        return result
    result.n_pairs = kv.n_pairs
    timings["build_index"] = time.time() - t0

    # 阶段5: 图像星对特征
    t0 = time.time()
    x_sel, y_sel = _select_brightest_stars(star_result, max_image_stars)
    if len(x_sel) < 4:
        result.message = f"选中星点数不足: {len(x_sel)} < 4"
        return result
    d_img_arr, theta_img_arr, _, _ = extract_image_pairs(x_sel, y_sel, s0)
    result.n_image_pairs = len(d_img_arr)
    timings["extract_features"] = time.time() - t0
    if result.n_image_pairs == 0:
        result.message = "图像星对特征提取失败"
        return result

    # 阶段6: 投票
    t0 = time.time()
    votes = vote(d_img_arr, theta_img_arr, kv, s0, sigma_pos, DEFAULT_N_SIGMA)
    timings["vote"] = time.time() - t0
    total_votes = sum(votes.values())

    # 阶段7: 峰值检测
    t0 = time.time()
    peaks = detect_peaks(votes, result.n_image_pairs, top_k=top_k)
    timings["detect_peaks"] = time.time() - t0
    if not peaks:
        result.message = (f"投票无峰值超过阈值 (总票数={total_votes}, "
                          f"阈值=max(3, {result.n_image_pairs}/100))")
        return result
    result.vote_peak = int(peaks[0][2])
    result.n_candidates = len(peaks)

    # 阶段8: 验证
    t0 = time.time()
    image_xy = np.column_stack([star_result.x, star_result.y])
    try:
        gaia_client = GaiaClientPy(data_dir=data_dir, db_type=1)
    except Exception as e:
        result.message = f"GaiaClient初始化失败: {e}"
        return result
    best_wcs = None
    n_tried = 0
    try:
        # fix-adv-pa-phase1-bugs Bug 3: 同步 pipeline.py 修复
        # 移除 early_exit, 加内点下限 max(10, 0.1*max_image_stars)
        min_inliers_required = max(10, int(0.1 * max_image_stars))
        for cand_idx, (healpix_id, rot_bin, count) in enumerate(peaks):
            n_tried += 1
            ra_center, dec_center = pix2ang(healpix_id)
            rot_angle = rot_bin_to_angle(rot_bin)
            try:
                wcs_cand = verify_candidate(
                    image_xy=image_xy, s0=s0,
                    ra_center=ra_center, dec_center=dec_center,
                    rot_angle=rot_angle, gaia_client=gaia_client,
                    fov_diag_deg=fov_diag_deg,
                    image_width=w, image_height=h,
                    mag_limit=DEFAULT_VERIFY_MAG_LIMIT, sigma_pos=sigma_pos,
                )
            except Exception:
                continue
            if wcs_cand is None:
                continue
            # 内点下限检查 (假阳性抑制)
            if wcs_cand.n_inliers < min_inliers_required:
                continue
            if best_wcs is None or wcs_cand.rms_arcsec < best_wcs.rms_arcsec:
                best_wcs = wcs_cand
                # 不再 early_exit (RMS<1" 即返回)
    finally:
        try:
            gaia_client.close()
        except Exception:
            pass
    result.candidates_tried = n_tried
    timings["verify"] = time.time() - t0
    if best_wcs is None:
        result.message = (f"所有{n_tried}个候选验证失败或内点不足 "
                          f"(下限={max(10, int(0.1 * max_image_stars))})")
        return result
    result.wcs = best_wcs
    result.best_rms_arcsec = best_wcs.rms_arcsec
    result.success = True
    result.message = (f"RMS={best_wcs.rms_arcsec:.3f}\", "
                      f"n_inliers={best_wcs.n_inliers}, s={best_wcs.s:.5f}")
    return result


# ═══ SubTask 8.1: 单帧测试 ═══

def run_one_frame(frame: dict) -> dict:
    """
    运行单帧盲解析, 含mag_limit重试逻辑.

    Returns:
        结果字典: name, success, s0, n_detected, n_reference, n_pairs, n_image_pairs,
                  n_candidates, vote_peak, snr, rms, crval_dev, total_time, mag_limit_used,
                  solved_crval, message, timings
    """
    name = frame["name"]
    full_path = os.path.join(_PROJECT_ROOT, frame["path"])
    print(f"\n{'=' * 70}")
    print(f"测试帧: {name}")
    print(f"路径: {frame['path']}")
    print(f"{'=' * 70}")

    mag_limit = 12.0
    s0_arg = frame["s0"]
    qra = frame["query_ra"]
    qdec = frame["query_dec"]

    t_start = time.time()
    try:
        result = solve_blind(
            image_path=full_path,
            s0_arcsec_per_pixel=s0_arg,
            query_center_ra=qra,
            query_center_dec=qdec,
            mag_limit=mag_limit,
        )
    except Exception as e:
        result = SolveResult()
        result.message = f"solve_blind异常: {e}"

    # mag_limit重试: 参考星<200 或 vote_peak<3 → 用14.0重试
    if (not result.success) and (result.n_reference < 200 or result.vote_peak < 3):
        print(f"  [重试] mag_limit=14.0 (原因: n_ref={result.n_reference}, vote_peak={result.vote_peak})")
        mag_limit = 14.0
        try:
            result = solve_blind(
                image_path=full_path,
                s0_arcsec_per_pixel=s0_arg,
                query_center_ra=qra,
                query_center_dec=qdec,
                mag_limit=mag_limit,
            )
        except Exception as e:
            result = SolveResult()
            result.message = f"solve_blind重试异常: {e}"

    elapsed = time.time() - t_start
    total_time = get_total_time(result) if result.stage_timings else elapsed

    # CRVAL偏差 — 正确方法: 比较图像中心的天空坐标
    # (solved WCS的CRVAL是候选切点, 与header WCS的CRVAL(不同切点)不可直接比较)
    # 用两个WCS分别计算图像中心的RA/Dec, 比较偏差
    crval_dev = None
    solved_crval = None
    if result.success and result.wcs is not None:
        solved_crval = (result.wcs.crval1, result.wcs.crval2)
        if frame["expected_crval"] is not None:
            # 读取header WCS
            try:
                reader = ImageReader()
                img_data = reader.read(full_path)
                meta = img_data.metadata
                if meta.wcs is not None and meta.wcs.has_wcs:
                    hdr_cd = np.array([[meta.wcs.cd1_1, meta.wcs.cd1_2],
                                        [meta.wcs.cd2_1, meta.wcs.cd2_2]])
                    hdr_crpix = (meta.wcs.crpix1, meta.wcs.crpix2)
                    hdr_crval = (meta.wcs.crval1, meta.wcs.crval2)
                    img_w = meta.geometry.width
                    img_h = meta.geometry.height
                    # 图像中心像素
                    cx, cy = img_w / 2.0, img_h / 2.0
                    # header WCS → 图像中心RA/Dec
                    dx = cx - (hdr_crpix[0] - 1.0)
                    dy = cy - (hdr_crpix[1] - 1.0)
                    xi_deg = hdr_cd[0, 0] * dx + hdr_cd[0, 1] * dy
                    eta_deg = hdr_cd[1, 0] * dx + hdr_cd[1, 1] * dy
                    hdr_ra, hdr_dec = gnomonic_inverse(
                        xi_deg * 3600.0, eta_deg * 3600.0, hdr_crval[0], hdr_crval[1])
                    # solved WCS → 图像中心RA/Dec
                    sdx = cx - (result.wcs.crpix1 - 1.0)
                    sdy = cy - (result.wcs.crpix2 - 1.0)
                    sxi_deg = result.wcs.cd[0, 0] * sdx + result.wcs.cd[0, 1] * sdy
                    seta_deg = result.wcs.cd[1, 0] * sdx + result.wcs.cd[1, 1] * sdy
                    sol_ra, sol_dec = gnomonic_inverse(
                        sxi_deg * 3600.0, seta_deg * 3600.0,
                        result.wcs.crval1, result.wcs.crval2)
                    crval_dev = haversine_arcsec(hdr_ra, hdr_dec, sol_ra, sol_dec)
                    # 也记录原始CRVAL偏差(诊断用)
                    raw_crval_dev = haversine_arcsec(
                        hdr_crval[0], hdr_crval[1],
                        result.wcs.crval1, result.wcs.crval2)
                    print(f"  图像中心天空坐标: header=({hdr_ra:.5f},{hdr_dec:.5f}) "
                          f"solved=({sol_ra:.5f},{sol_dec:.5f})")
                    print(f"  CRVAL偏差(图像中心)={crval_dev:.2f}\" "
                          f"(原始切点偏差={raw_crval_dev:.2f}\")")
                else:
                    # 无header WCS, 回退到原始CRVAL比较
                    cra, cdec = frame["expected_crval"]
                    crval_dev = haversine_arcsec(
                        result.wcs.crval1, result.wcs.crval2, cra, cdec)
                    print(f"  无header WCS, 用原始CRVAL比较: {crval_dev:.2f}\"")
            except Exception as e:
                print(f"  读取header WCS失败: {e}")
                cra, cdec = frame["expected_crval"]
                crval_dev = haversine_arcsec(
                    result.wcs.crval1, result.wcs.crval2, cra, cdec)

    # SNR (需要total_votes — pipeline未直接返回, 从日志/阶段无法获取, 用vote_peak和n_image_pairs估算)
    # 注意: SolveResult不包含total_votes, 我们用vote_peak和n_candidates估算下界
    # 真正的SNR需要在solve_from_stars中获取. 这里先用vote_peak/n_image_pairs作为代理
    # 后续k-vector检查和鲁棒性测试会用solve_from_stars获取精确total_votes
    snr = None  # 将在run_all中补充 (需要total_votes)

    rec = {
        "name": name,
        "success": result.success,
        "s0": result.s0_arcsec_per_pixel,
        "n_detected": result.n_detected,
        "n_reference": result.n_reference,
        "n_pairs": result.n_pairs,
        "n_image_pairs": result.n_image_pairs,
        "n_candidates": result.n_candidates,
        "vote_peak": result.vote_peak,
        "snr": snr,
        "rms": result.best_rms_arcsec if result.success else float("inf"),
        "crval_dev": crval_dev,
        "total_time": total_time,
        "mag_limit": mag_limit,
        "solved_crval": solved_crval,
        "expected_crval": frame["expected_crval"],
        "message": result.message,
        "candidates_tried": result.candidates_tried,
        "timings": dict(result.stage_timings),
        "result_obj": result,
    }

    # 打印单帧结果
    print(f"\n  结果: success={result.success}")
    print(f"  s0={result.s0_arcsec_per_pixel:.4f} arcsec/px, mag_limit={mag_limit}")
    print(f"  检测星点={result.n_detected}, 参考星={result.n_reference}, 星对={result.n_pairs}")
    print(f"  图像星对={result.n_image_pairs}, 候选={result.n_candidates}, 尝试={result.candidates_tried}")
    print(f"  vote_peak={result.vote_peak}, RMS={rec['rms']:.3f}\"")
    if solved_crval:
        print(f"  solved CRVAL=({solved_crval[0]:.5f}, {solved_crval[1]:.5f})")
    if crval_dev is not None:
        print(f"  CRVAL偏差={crval_dev:.2f}\" (期望<30\")")
    print(f"  总耗时={total_time:.3f}s")
    print(f"  各阶段: " + ", ".join(f"{k}={v:.3f}s" for k, v in result.stage_timings.items()))
    if not result.success:
        print(f"  失败原因: {result.message}")

    return rec


# ═══ SubTask 8.2: k-vector查询准确性验证 ═══

def check_kvector_accuracy(
    frame: dict,
    full_path: str,
    s0: float,
    query_ra: float,
    query_dec: float,
    wcs_crval: tuple[float, float],
    wcs_cd: Optional[np.ndarray] = None,
    wcs_crpix: Optional[tuple[float, float]] = None,
    mag_limit: float = 12.0,
) -> dict:
    """
    验证k-vector查询无假阴性: 对真匹配星对, 检查catalog pair在k-vector查询结果中.

    流程:
        1. 用solved/header WCS将图像星投影到RA/Dec
        2. 查询DR3 + 构建星对库 + k-vector (与管线一致)
        3. 图像星↔catalog星最近邻匹配 (阈值5*s0)
        4. 对4-6个真匹配图像星对(i,j)→catalog(I,J):
           - 计算d_img, d_cat
           - 检查|d_img-d_cat|<=3σ_d
           - kvector_query(kv, d_img, 3σ_d) → [idx_lo, idx_hi]
           - 检查catalog pair (I,J) 是否在 S[idx_lo:idx_hi+1] 中

    Returns:
        dict: n_checked, n_found, n_in_library, pairs_detail, all_found
    """
    print(f"\n  [SubTask 8.2] k-vector查询准确性验证...")
    sigma_pos = DEFAULT_SIGMA_POS
    sigma_d = sigma_pos * s0
    delta_d = DEFAULT_N_SIGMA * sigma_d

    # 1. 读图 + 检测星点
    uint16_img, metadata = read_image(full_path)
    h, w = uint16_img.shape
    star_result = detect_stars(uint16_img)
    if star_result.count < 4:
        return {"error": f"星点不足: {star_result.count}"}

    # 2. DR3查询 + 构建k-vector (与管线一致)
    fov_diag_arcsec = math.sqrt(w ** 2 + h ** 2) * s0
    fov_diag_deg = fov_diag_arcsec / 3600.0
    radius_deg = fov_diag_deg * DEFAULT_FOV_MARGIN
    ra_arr, dec_arr, mag_arr = query_dr3(query_ra, query_dec, radius_deg, mag_limit, _GAIA_DR3_DIR)
    kv = build_pair_library_with_kvector(
        ra_arr, dec_arr, mag_arr,
        k_neighbors=DEFAULT_K_NEIGHBORS, delta=DEFAULT_DELTA,
        d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX,
    )
    if kv is None:
        return {"error": "k-vector构建失败"}

    # 3. 用WCS将图像星投影到RA/Dec
    # WCS CD矩阵 (deg/px), CRPIX (1-indexed), CRVAL
    if wcs_cd is None or wcs_crpix is None:
        return {"error": "无WCS CD/CRPIX, 无法投影"}
    cd = wcs_cd
    crpix1, crpix2 = wcs_crpix
    crval1, crval2 = wcs_crval

    x_arr = star_result.x
    y_arr = star_result.y
    # 像素坐标 → intermediate world coords (linear part)
    dx = x_arr - (crpix1 - 1.0)
    dy = y_arr - (crpix2 - 1.0)
    # CD是[cd11, cd12; cd21, cd22], intermediate = CD @ [dx, dy]
    xi_deg = cd[0, 0] * dx + cd[0, 1] * dy   # RA axis
    eta_deg = cd[1, 0] * dx + cd[1, 1] * dy  # Dec axis
    # gnomonic_inverse: xi/eta (deg) → RA/Dec
    img_ra, img_dec = gnomonic_inverse(xi_deg * 3600.0, eta_deg * 3600.0, crval1, crval2)

    # 4. 图像星↔catalog星最近邻匹配 (阈值5*s0 arcsec)
    match_thresh = 5.0 * s0
    # 用catalog星构建cKDTree (在切平面arcsec)
    ra0, dec0 = crval1, crval2
    xi_cat, eta_cat, valid_cat = gnomonic_forward(ra_arr, dec_arr, ra0, dec0)
    from scipy.spatial import cKDTree
    cat_xy = np.column_stack([xi_cat[valid_cat], eta_cat[valid_cat]])
    tree = cKDTree(cat_xy)
    # 图像星在切平面
    img_xi = xi_deg * 3600.0
    img_eta = eta_deg * 3600.0
    img_xy_tan = np.column_stack([img_xi, img_eta])
    dists, idxs = tree.query(img_xy_tan, k=1)
    matched_mask = dists < match_thresh
    # 匹配的图像星下标 → catalog下标
    img_matched_idx = np.where(matched_mask)[0]
    cat_matched_idx = idxs[matched_mask]  # 在valid_cat子集中的下标

    # 还原到原始ra_arr下标
    valid_idx = np.where(valid_cat)[0]
    cat_orig_idx = valid_idx[cat_matched_idx]

    n_matched = len(img_matched_idx)
    print(f"    图像星-catalog匹配: {n_matched}/{star_result.count} (阈值={match_thresh:.1f}\")")
    if n_matched < 4:
        return {"error": f"匹配星数不足: {n_matched}"}

    # 5. 选取4-6个真匹配图像星对(i,j)
    # 从匹配的图像星中取若干对 (间隔较大, d_cat在[10", 18000"]范围内)
    pairs_to_check = []
    n_m = len(img_matched_idx)
    # 选取前6对 (i, j) 从匹配列表
    for ii in range(n_m):
        for jj in range(ii + 1, n_m):
            i_img = int(img_matched_idx[ii])
            j_img = int(img_matched_idx[jj])
            I_cat = int(cat_orig_idx[ii])
            J_cat = int(cat_orig_idx[jj])
            d_img = math.hypot(x_arr[i_img] - x_arr[j_img],
                               y_arr[i_img] - y_arr[j_img]) * s0
            if d_img < DEFAULT_D_MIN or d_img > DEFAULT_D_MAX:
                continue
            pairs_to_check.append((i_img, j_img, I_cat, J_cat, d_img))
            if len(pairs_to_check) >= 6:
                break
        if len(pairs_to_check) >= 6:
            break

    if not pairs_to_check:
        return {"error": "无可检查的真匹配星对"}

    # 6. 对每个真匹配对, 检查k-vector查询
    n_checked = 0
    n_found = 0
    n_in_library = 0
    details = []
    for (i_img, j_img, I_cat, J_cat, d_img) in pairs_to_check:
        n_checked += 1
        ra_I = ra_arr[I_cat]
        dec_I = dec_arr[I_cat]
        ra_J = ra_arr[J_cat]
        dec_J = dec_arr[J_cat]
        d_cat = angular_distance_arcsec(ra_I, dec_I, ra_J, dec_J)

        # 检查(I,J)是否在星对库S中 — 通过ra_i/dec_i或ra_j/dec_j匹配
        # S中每条记录: (d_cat, PA_cat, ra_i, dec_i, ra_j, dec_j, star_id_i)
        # 注意: 星对库中i是主星, j是K近邻. (I,J)可能以(I,J)或(J,I)形式存在
        S = kv.S
        tol = 1e-6  # 度
        # 检查 (I作为主星, J作为邻星)
        match_IJ = np.where(
            (np.abs(S['ra_i'] - ra_I) < tol) & (np.abs(S['dec_i'] - dec_I) < tol) &
            (np.abs(S['ra_j'] - ra_J) < tol) & (np.abs(S['dec_j'] - dec_J) < tol)
        )[0]
        match_JI = np.where(
            (np.abs(S['ra_i'] - ra_J) < tol) & (np.abs(S['dec_i'] - dec_J) < tol) &
            (np.abs(S['ra_j'] - ra_I) < tol) & (np.abs(S['dec_j'] - dec_I) < tol)
        )[0]
        pair_idx_in_S = -1
        if len(match_IJ) > 0:
            pair_idx_in_S = int(match_IJ[0])
        elif len(match_JI) > 0:
            pair_idx_in_S = int(match_JI[0])

        in_library = (pair_idx_in_S >= 0)
        if in_library:
            n_in_library += 1

        # k-vector查询
        idx_lo, idx_hi = kvector_query(kv, d_img, delta_d)
        found = False
        if in_library and idx_lo <= idx_hi:
            found = (idx_lo <= pair_idx_in_S <= idx_hi)

        # 也检查 |d_img - d_cat| <= 3σ_d (查询容差合理性)
        dist_diff = abs(d_img - d_cat)
        within_tol = dist_diff <= delta_d

        if found:
            n_found += 1

        details.append({
            "i_img": i_img, "j_img": j_img,
            "d_img": d_img, "d_cat": d_cat,
            "dist_diff": dist_diff, "within_tol": within_tol,
            "in_library": in_library,
            "kv_range": (idx_lo, idx_hi) if idx_lo <= idx_hi else None,
            "pair_idx_in_S": pair_idx_in_S,
            "found": found,
        })
        print(f"    对{n_checked}: d_img={d_img:.2f}\", d_cat={d_cat:.2f}\", "
              f"|Δd|={dist_diff:.2f}\" (容差{delta_d:.2f}\"), "
              f"在库={in_library}, k-vector找到={found}")

    all_found = (n_found == n_in_library) and (n_in_library >= 4)
    print(f"    汇总: 检查{n_checked}对, 在星对库中{n_in_library}对, "
          f"k-vector找到{n_found}对 → {'通过' if all_found else '未通过'}")
    return {
        "n_checked": n_checked,
        "n_found": n_found,
        "n_in_library": n_in_library,
        "all_found": all_found,
        "pairs_detail": details,
    }


# ═══ SubTask 8.5: 鲁棒性测试 ═══

def robustness_test(
    frame: dict,
    full_path: str,
    s0: float,
    query_ra: float,
    query_dec: float,
    w: int,
    h: int,
    mag_limit: float = 12.0,
) -> dict:
    """
    鲁棒性测试: 注入假星 / 移除30%星 / 标记20%饱和.

    Returns:
        dict: 三个测试的结果 (fake_star, remove_30, saturate_20)
    """
    print(f"\n  [SubTask 8.5] 鲁棒性测试...")
    # 重新检测星点 (用原始图像)
    uint16_img, _ = read_image(full_path)
    star_orig = detect_stars(uint16_img)
    print(f"    原始星点数: {star_orig.count}")
    results = {}

    # --- 测试1: 注入1颗假星 (图像边缘, 高流量) ---
    print(f"    [测试1] 注入1颗假星...")
    rng = np.random.default_rng(42)
    # 在图像边缘随机选位置
    edge = rng.choice(['top', 'bottom', 'left', 'right'])
    if edge == 'top':
        fx, fy = float(rng.integers(10, w - 10)), float(rng.integers(5, 20))
    elif edge == 'bottom':
        fx, fy = float(rng.integers(10, w - 10)), float(rng.integers(h - 20, h - 5))
    elif edge == 'left':
        fx, fy = float(rng.integers(5, 20)), float(rng.integers(10, h - 10))
    else:
        fx, fy = float(rng.integers(w - 20, w - 5)), float(rng.integers(10, h - 10))
    # 构造带假星的star_result
    x_new = np.append(star_orig.x, fx)
    y_new = np.append(star_orig.y, fy)
    flux_new = np.append(star_orig.flux, float(np.max(star_orig.flux) * 2.0))
    sat_new = np.append(star_orig.saturated, 0)
    star_fake = StarDetectionResult(x=x_new, y=y_new, flux=flux_new, saturated=sat_new)
    t0 = time.time()
    res_fake = solve_from_stars(star_fake, s0, w, h, query_ra, query_dec, mag_limit)
    t_fake = time.time() - t0
    results["fake_star"] = {
        "success": res_fake.success,
        "vote_peak": res_fake.vote_peak,
        "rms": res_fake.best_rms_arcsec if res_fake.success else float("inf"),
        "time": t_fake,
        "n_detected": star_fake.count,
        "message": res_fake.message,
    }
    print(f"      假星测试: success={res_fake.success}, vote_peak={res_fake.vote_peak}, "
          f"RMS={results['fake_star']['rms']:.3f}\", 耗时={t_fake:.2f}s")

    # --- 测试2: 移除30%星点 ---
    print(f"    [测试2] 移除30%星点...")
    n = star_orig.count
    n_keep = int(n * 0.7)
    keep_idx = rng.choice(n, size=n_keep, replace=False)
    star_remove = StarDetectionResult(
        x=star_orig.x[keep_idx], y=star_orig.y[keep_idx],
        flux=star_orig.flux[keep_idx], saturated=star_orig.saturated[keep_idx],
    )
    t0 = time.time()
    res_remove = solve_from_stars(star_remove, s0, w, h, query_ra, query_dec, mag_limit)
    t_remove = time.time() - t0
    results["remove_30"] = {
        "success": res_remove.success,
        "vote_peak": res_remove.vote_peak,
        "rms": res_remove.best_rms_arcsec if res_remove.success else float("inf"),
        "time": t_remove,
        "n_detected": star_remove.count,
        "message": res_remove.message,
    }
    print(f"      移除30%: success={res_remove.success}, vote_peak={res_remove.vote_peak}, "
          f"RMS={results['remove_30']['rms']:.3f}\", 耗时={t_remove:.2f}s")

    # --- 测试3: 标记20%星为饱和 ---
    print(f"    [测试3] 标记20%星为饱和...")
    n_sat = max(1, int(n * 0.2))
    sat_flags = star_orig.saturated.copy()
    # 随机选20%非饱和星标记为饱和
    normal_idx = np.where(sat_flags == 0)[0]
    if len(normal_idx) > 0:
        mark_idx = rng.choice(normal_idx, size=min(n_sat, len(normal_idx)), replace=False)
        sat_flags[mark_idx] = 1
    star_sat = StarDetectionResult(
        x=star_orig.x, y=star_orig.y,
        flux=star_orig.flux, saturated=sat_flags,
    )
    t0 = time.time()
    res_sat = solve_from_stars(star_sat, s0, w, h, query_ra, query_dec, mag_limit)
    t_sat = time.time() - t0
    results["saturate_20"] = {
        "success": res_sat.success,
        "vote_peak": res_sat.vote_peak,
        "rms": res_sat.best_rms_arcsec if res_sat.success else float("inf"),
        "time": t_sat,
        "n_detected": star_sat.count,
        "n_marked_saturated": int(np.sum(sat_flags)),
        "message": res_sat.message,
    }
    print(f"      标记20%饱和: success={res_sat.success}, vote_peak={res_sat.vote_peak}, "
          f"RMS={results['saturate_20']['rms']:.3f}\", 耗时={t_sat:.2f}s")

    return results


# ═══ SubTask 8.7: 报告生成 ═══

def write_report(
    frame_results: list[dict],
    kv_check: Optional[dict],
    robust: Optional[dict],
    robust_frame_name: Optional[str],
) -> str:
    """生成UTF-8报告到 phase1_report.txt, 返回报告内容"""
    lines = []
    lines.append("=" * 80)
    lines.append("ADV-PA Phase 1 可靠性验证报告")
    lines.append("生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 80)

    # 每帧结果表
    lines.append("")
    lines.append("【一】每帧测试结果")
    lines.append("-" * 80)
    header = (f"{'帧名':<12} {'s0':>8} {'Ndet':>6} {'Nref':>6} {'Npairs':>7} "
              f"{'Ncand':>6} {'Vpeak':>6} {'SNR':>8} {'RMS\"':>8} "
              f"{'CRVALdev\"':>10} {'success':>8} {'time(s)':>8}")
    lines.append(header)
    lines.append("-" * 80)
    for r in frame_results:
        snr_str = f"{r['snr']:.1f}" if r['snr'] is not None else "N/A"
        rms_str = f"{r['rms']:.3f}" if r['success'] else "inf"
        dev_str = f"{r['crval_dev']:.2f}" if r['crval_dev'] is not None else "N/A"
        succ_str = "是" if r['success'] else "否"
        lines.append(
            f"{r['name']:<12} {r['s0']:>8.4f} {r['n_detected']:>6} {r['n_reference']:>6} "
            f"{r['n_pairs']:>7} {r['n_candidates']:>6} {r['vote_peak']:>6} "
            f"{snr_str:>8} {rms_str:>8} {dev_str:>10} {succ_str:>8} {r['total_time']:>8.3f}"
        )
    lines.append("-" * 80)

    # 汇总统计
    n_frames = len(frame_results)
    n_success = sum(1 for r in frame_results if r['success'])
    success_rate = n_success / n_frames * 100 if n_frames > 0 else 0
    rms_list = [r['rms'] for r in frame_results if r['success']]
    time_list = [r['total_time'] for r in frame_results]
    vpeak_list = [r['vote_peak'] for r in frame_results]
    dev_list = [r['crval_dev'] for r in frame_results if r['crval_dev'] is not None and r['success']]

    lines.append("")
    lines.append("【二】汇总统计")
    lines.append(f"  成功率: {n_success}/{n_frames} = {success_rate:.1f}% (期望≥80%)")
    if rms_list:
        lines.append(f"  平均RMS: {np.mean(rms_list):.3f}\" (期望<3\")")
        lines.append(f"  最大RMS: {np.max(rms_list):.3f}\"")
    if time_list:
        lines.append(f"  平均耗时: {np.mean(time_list):.3f}s (期望<5s)")
        lines.append(f"  最大耗时: {np.max(time_list):.3f}s")
    if vpeak_list:
        lines.append(f"  平均vote_peak: {np.mean(vpeak_list):.1f}")
    if dev_list:
        lines.append(f"  平均CRVAL偏差: {np.mean(dev_list):.2f}\" (期望<30\")")
        lines.append(f"  最大CRVAL偏差: {np.max(dev_list):.2f}\"")

    # SNR详情
    lines.append("")
    lines.append("【三】投票信噪比 (SubTask 8.3, SNR = vote_peak / (total_votes/N_cells))")
    lines.append(f"  投票格数 N_cells = {N_SKY_CELLS}(天区) × {N_ROT_BINS}(rot) = {N_VOTE_CELLS}")
    lines.append("-" * 60)
    for r in frame_results:
        snr_str = f"{r['snr']:.2f}" if r['snr'] is not None else "N/A"
        pass_str = "通过" if (r['snr'] is not None and r['snr'] > 10) else "—"
        lines.append(f"  {r['name']:<12} vote_peak={r['vote_peak']:>4}  SNR={snr_str:>10}  {pass_str}")
    lines.append("-" * 60)

    # RMS详情
    lines.append("")
    lines.append("【四】V4收敛验证 (SubTask 8.4, RMS<3\")")
    lines.append("-" * 60)
    for r in frame_results:
        if r['success']:
            pass_str = "通过" if r['rms'] < 3.0 else "未通过"
            lines.append(f"  {r['name']:<12} RMS={r['rms']:.3f}\"  {pass_str}")
        else:
            lines.append(f"  {r['name']:<12} 未成功求解")
    lines.append("-" * 60)

    # CRVAL偏差
    lines.append("")
    lines.append("【五】CRVAL偏差验证 (SubTask 8.6, 与header WCS偏差<30\")")
    lines.append("-" * 60)
    for r in frame_results:
        if r['crval_dev'] is not None and r['success']:
            pass_str = "通过" if r['crval_dev'] < 30.0 else "未通过"
            sc = r['solved_crval']
            ec = r['expected_crval']
            lines.append(f"  {r['name']:<12} solved=({sc[0]:.5f},{sc[1]:.5f}) "
                         f"expected=({ec[0]:.5f},{ec[1]:.5f}) dev={r['crval_dev']:.2f}\" {pass_str}")
        else:
            lines.append(f"  {r['name']:<12} 无baseline或未成功")
    lines.append("-" * 60)

    # k-vector准确性
    lines.append("")
    lines.append("【六】k-vector查询准确性 (SubTask 8.2)")
    lines.append("-" * 60)
    if kv_check is None:
        lines.append("  未执行 (无成功帧)")
    elif "error" in kv_check:
        lines.append(f"  执行失败: {kv_check['error']}")
    else:
        lines.append(f"  检查星对数: {kv_check['n_checked']}")
        lines.append(f"  在星对库中: {kv_check['n_in_library']}")
        lines.append(f"  k-vector找到: {kv_check['n_found']}")
        lines.append(f"  无假阴性: {'是 (通过)' if kv_check['all_found'] else '否 (未通过)'}")
        lines.append("  详情:")
        for d in kv_check.get("pairs_detail", []):
            lines.append(
                f"    d_img={d['d_img']:.2f}\", d_cat={d['d_cat']:.2f}\", "
                f"|Δd|={d['dist_diff']:.2f}\" (容差内={d['within_tol']}), "
                f"在库={d['in_library']}, k-vector找到={d['found']}"
            )
    lines.append("-" * 60)

    # 鲁棒性
    lines.append("")
    lines.append("【七】鲁棒性测试 (SubTask 8.5)")
    if robust_frame_name:
        lines.append(f"  测试帧: {robust_frame_name}")
    lines.append("-" * 60)
    if robust is None:
        lines.append("  未执行 (无成功帧)")
    else:
        for test_name, label in [("fake_star", "注入1颗假星"),
                                  ("remove_30", "移除30%星点"),
                                  ("saturate_20", "标记20%饱和")]:
            if test_name in robust:
                tr = robust[test_name]
                succ = "成功" if tr['success'] else "失败"
                rms_str = f"{tr['rms']:.3f}\"" if tr['success'] else "inf"
                lines.append(f"  {label:<14}: {succ}  vote_peak={tr['vote_peak']}, "
                             f"RMS={rms_str}, 耗时={tr['time']:.2f}s")
    lines.append("-" * 60)

    # 结论
    lines.append("")
    lines.append("【八】结论")
    lines.append("-" * 60)
    crit_success = success_rate >= 80.0
    crit_rms = all(r['rms'] < 3.0 for r in frame_results if r['success']) if rms_list else False
    crival_pass = all(d < 30.0 for d in dev_list) if dev_list else False
    snr_pass = all(r['snr'] is not None and r['snr'] > 10 for r in frame_results if r['success']) if frame_results else False
    time_pass = all(t < 5.0 for t in time_list) if time_list else False
    lines.append(f"  成功率≥80%: {'通过' if crit_success else '未通过'} ({success_rate:.1f}%)")
    lines.append(f"  RMS<3\": {'通过' if crit_rms else '未通过'}")
    lines.append(f"  CRVAL偏差<30\": {'通过' if crival_pass else '未通过'}")
    lines.append(f"  SNR>10: {'通过' if snr_pass else '未通过'}")
    lines.append(f"  单帧<5s: {'通过' if time_pass else '未通过'}")
    overall = crit_success and crit_rms and crival_pass and snr_pass and time_pass
    lines.append("")
    if overall:
        lines.append("  → Phase 1 可靠性验证通过, ADV-PA 核心机制可靠工作, 可进入 Phase 2")
    else:
        lines.append("  → Phase 1 部分指标未达标, 需调试后复测")
    lines.append("=" * 80)

    content = "\n".join(lines)
    os.makedirs(_LOG_DIR, exist_ok=True)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    return content


# ═══ 主函数 ═══

def main():
    # 初始化日志
    logging_setup.setup_logging()
    print("ADV-PA Phase 1 可靠性验证开始")
    print(f"项目根目录: {_PROJECT_ROOT}")
    print(f"GaiaDR3目录: {_GAIA_DR3_DIR}")
    print(f"投票格数: {N_VOTE_CELLS} (grid回退 {N_SKY_CELLS}天区 × {N_ROT_BINS}rot)")
    print(f"测试帧数: {len(TEST_FRAMES)}")

    # SubTask 8.1: 逐帧测试
    frame_results = []
    for frame in TEST_FRAMES:
        rec = run_one_frame(frame)
        frame_results.append(rec)

    # 补充SNR: 对成功帧, 用solve_from_stars获取精确total_votes
    # (solve_blind不返回total_votes, 这里重跑阶段6获取)
    first_success = None
    for i, r in enumerate(frame_results):
        if r["success"]:
            first_success = i
            break

    # 为所有帧补充SNR (重跑投票阶段获取total_votes)
    print(f"\n{'=' * 70}")
    print("补充SNR计算 (重跑投票阶段获取total_votes)")
    print(f"{'=' * 70}")
    for i, (frame, r) in enumerate(zip(TEST_FRAMES, frame_results)):
        full_path = os.path.join(_PROJECT_ROOT, frame["path"])
        try:
            # 用与原run相同的s0和mag_limit
            s0 = r["s0"]
            qra = frame["query_ra"] if frame["query_ra"] is not None else r.get("result_obj").ra0
            qdec = frame["query_dec"] if frame["query_dec"] is not None else r.get("result_obj").dec0
            ml = r["mag_limit"]
            # 读图+检测 (复用)
            uint16_img, _ = read_image(full_path)
            h, w = uint16_img.shape
            star_res = detect_stars(uint16_img)
            fov_diag_deg = math.sqrt(w**2 + h**2) * s0 / 3600.0
            radius_deg = fov_diag_deg * DEFAULT_FOV_MARGIN
            ra_arr, dec_arr, mag_arr = query_dr3(qra, qdec, radius_deg, ml, _GAIA_DR3_DIR)
            kv = build_pair_library_with_kvector(
                ra_arr, dec_arr, mag_arr,
                k_neighbors=DEFAULT_K_NEIGHBORS, delta=DEFAULT_DELTA,
                d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX,
            )
            x_sel, y_sel = _select_brightest_stars(star_res, DEFAULT_MAX_IMAGE_STARS)
            d_img_arr, theta_img_arr, _, _ = extract_image_pairs(x_sel, y_sel, s0)
            votes = vote(d_img_arr, theta_img_arr, kv, s0, DEFAULT_SIGMA_POS, DEFAULT_N_SIGMA)
            total_votes = sum(votes.values())
            snr = compute_snr(r["vote_peak"], total_votes)
            r["snr"] = snr
            r["total_votes"] = total_votes
            print(f"  {r['name']:<12} total_votes={total_votes:>6}, vote_peak={r['vote_peak']}, "
                  f"SNR={snr:.2f} {'(>10 通过)' if snr > 10 else '(未通过)'}")
        except Exception as e:
            print(f"  {r['name']:<12} SNR计算失败: {e}")
            r["snr"] = None

    # SubTask 8.2: k-vector准确性 (用第一个成功帧)
    kv_check = None
    if first_success is not None:
        frame = TEST_FRAMES[first_success]
        r = frame_results[first_success]
        full_path = os.path.join(_PROJECT_ROOT, frame["path"])
        # 用solved WCS作ground truth
        wcs_obj = r["result_obj"].wcs
        if wcs_obj is not None:
            kv_check = check_kvector_accuracy(
                frame=frame,
                full_path=full_path,
                s0=r["s0"],
                query_ra=r["result_obj"].ra0,
                query_dec=r["result_obj"].dec0,
                wcs_crval=(wcs_obj.crval1, wcs_obj.crval2),
                wcs_cd=wcs_obj.cd,
                wcs_crpix=(wcs_obj.crpix1, wcs_obj.crpix2),
                mag_limit=r["mag_limit"],
            )
        else:
            # fallback: 用header WCS
            kv_check = {"error": "无solved WCS可用"}

    # SubTask 8.5: 鲁棒性测试 (用第一个成功帧)
    robust = None
    robust_frame_name = None
    if first_success is not None:
        frame = TEST_FRAMES[first_success]
        r = frame_results[first_success]
        robust_frame_name = r["name"]
        full_path = os.path.join(_PROJECT_ROOT, frame["path"])
        uint16_img, _ = read_image(full_path)
        h, w = uint16_img.shape
        robust = robustness_test(
            frame=frame, full_path=full_path,
            s0=r["s0"],
            query_ra=r["result_obj"].ra0,
            query_dec=r["result_obj"].dec0,
            w=w, h=h, mag_limit=r["mag_limit"],
        )

    # SubTask 8.7: 生成报告
    content = write_report(frame_results, kv_check, robust, robust_frame_name)
    print(f"\n{'=' * 70}")
    print("报告已写入: " + _REPORT_PATH)
    print(f"{'=' * 70}")
    print(content)


if __name__ == "__main__":
    main()

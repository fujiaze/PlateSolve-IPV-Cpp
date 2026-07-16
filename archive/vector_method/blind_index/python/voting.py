"""
Kolomenkin几何投票模块 (Task 7)
功能: 当≥3个图像四边形各有候选时, 用每个候选WCS投影其他四边形图像星到天球, 检查与该四边形候选参考星匹配(<3"), 得票最高者胜出
用途: 消除歧义候选, 从多个匹配候选中选出唯一正确WCS
依赖: numpy, scipy.spatial.cKDTree, .wcs_solver

简化版投票逻辑:
    - 若≥3个四边形各有≥1候选:
        对每个(quad_i, cand_i): 求WCS_i, 投影其他quad_j的4图像星到天球,
        在quad_j候选参考星池中最近邻<3"则投票+1
        选得票最高候选; 若最高票<2则回退到首个四边形候选列表
    - 若<3个四边形有候选: 取首个有候选的四边形, 选其RMS最低候选
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .logging_setup import get_logger
from .matcher import MatchCandidate
from .wcs_solver import WCSResult, solve_wcs_from_candidate, apply_wcs, angular_separation_arcsec

logger = get_logger(__name__)

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi

# 投票参数
VOTE_MATCH_RADIUS_ARCSEC = 3.0   # 投票匹配半径(arcsec)
MIN_VOTES_TO_ACCEPT = 2          # 接受所需最低票数
MIN_QUADS_FOR_VOTING = 3         # 启用投票所需最低四边形数


@dataclass
class VoteResult:
    """
    投票结果。

    Attributes:
        best_candidate: 最佳候选
        best_wcs: 最佳候选对应的WCS
        best_votes: 得票数
        best_rms: RMS(arcsec)
        voting_used: 是否启用了投票
        message: 描述信息
    """
    best_candidate: Optional[MatchCandidate]
    best_wcs: Optional[WCSResult]
    best_votes: int
    best_rms: float
    voting_used: bool
    message: str


def vote(
    candidates_per_quad: list[list[MatchCandidate]],
    ra0: float,
    dec0: float,
) -> VoteResult:
    """
    Kolomenkin简化版几何投票。

    Args:
        candidates_per_quad: 每个图像四边形对应的候选列表
        ra0: 切平面中心RA(度)
        dec0: 切平面中心Dec(度)

    Returns:
        VoteResult
    """
    # 统计有候选的四边形
    quads_with_candidates = [
        (i, cands) for i, cands in enumerate(candidates_per_quad) if len(cands) > 0
    ]
    n_quads_with_cands = len(quads_with_candidates)

    logger.info("投票: %d个四边形有候选 (共%d个四边形)",
                 n_quads_with_cands, len(candidates_per_quad))

    if n_quads_with_cands == 0:
        return VoteResult(
            best_candidate=None, best_wcs=None,
            best_votes=0, best_rms=float("inf"),
            voting_used=False, message="无任何候选",
        )

    # <3个四边形有候选 → 直接选首个有候选的四边形的最低RMS候选
    if n_quads_with_cands < MIN_QUADS_FOR_VOTING:
        logger.info("有候选四边形<%d, 跳过投票, 选最低RMS候选", MIN_QUADS_FOR_VOTING)
        return _pick_lowest_rms(quads_with_candidates[0][1], ra0, dec0, voting_used=False)

    # ≥3个四边形有候选 → 执行投票
    return _run_voting(candidates_per_quad, quads_with_candidates, ra0, dec0)


def _pick_lowest_rms(
    candidates: list[MatchCandidate],
    ra0: float,
    dec0: float,
    voting_used: bool,
) -> VoteResult:
    """从候选列表中选RMS最低的。"""
    best_cand = None
    best_wcs = None
    best_rms = float("inf")

    for cand in candidates:
        wcs = solve_wcs_from_candidate(cand, ra0, dec0)
        if wcs is None:
            continue
        if wcs.rms_arcsec < best_rms:
            best_rms = wcs.rms_arcsec
            best_cand = cand
            best_wcs = wcs

    if best_cand is None:
        return VoteResult(
            best_candidate=None, best_wcs=None,
            best_votes=0, best_rms=float("inf"),
            voting_used=voting_used, message="所有候选WCS求解失败",
        )

    logger.info("选最低RMS候选: RMS=%.3f\", d_AB=%.2f\"",
                 best_rms, best_cand.image_quad.d_AB)
    return VoteResult(
        best_candidate=best_cand,
        best_wcs=best_wcs,
        best_votes=0,
        best_rms=best_rms,
        voting_used=voting_used,
        message=f"选最低RMS候选 (RMS={best_rms:.3f}\")",
    )


def _run_voting(
    candidates_per_quad: list[list[MatchCandidate]],
    quads_with_candidates: list[tuple[int, list[MatchCandidate]]],
    ra0: float,
    dec0: float,
) -> VoteResult:
    """
    执行Kolomenkin几何投票。

    对每个候选cand_i: 求WCS_i, 对每个其他四边形quad_j:
    投影quad_j的4图像星到天球, 在quad_j候选参考星池中最近邻<3"→投票+1
    """
    logger.info("开始Kolomenkin投票, %d个四边形参与", len(quads_with_candidates))

    # 预计算每个候选的WCS
    all_candidates: list[tuple[int, int, MatchCandidate, WCSResult]] = []
    # (quad_idx, cand_idx, candidate, wcs)
    for quad_idx, cands in quads_with_candidates:
        for cand_idx, cand in enumerate(cands):
            wcs = solve_wcs_from_candidate(cand, ra0, dec0)
            if wcs is not None:
                all_candidates.append((quad_idx, cand_idx, cand, wcs))

    if not all_candidates:
        logger.warning("所有候选WCS求解失败, 回退到最低RMS")
        return _pick_lowest_rms(quads_with_candidates[0][1], ra0, dec0, voting_used=False)

    # 预计算每个四边形的候选参考星池 (RA/Dec), 用于投票匹配
    quad_ref_star_pools: dict[int, tuple[np.ndarray, np.ndarray, cKDTree]] = {}
    for quad_idx, cands in quads_with_candidates:
        # 收集该四边形所有候选的所有参考星
        all_ra = []
        all_dec = []
        for cand in cands:
            all_ra.extend(cand.ref_quad.ra.tolist())
            all_dec.extend(cand.ref_quad.dec.tolist())
        ra_pool = np.array(all_ra, dtype=np.float64)
        dec_pool = np.array(all_dec, dtype=np.float64)
        # 构建3D单位向量KDTree (球面最近邻)
        if len(ra_pool) > 0:
            vecs = _radec_to_unit_vectors(ra_pool, dec_pool)
            tree = cKDTree(vecs)
            quad_ref_star_pools[quad_idx] = (ra_pool, dec_pool, tree)

    # 投票
    vote_counts: list[int] = [0] * len(all_candidates)

    for i, (quad_i, _, cand_i, wcs_i) in enumerate(all_candidates):
        # 用WCS_i投影其他四边形的图像星
        for quad_j, cands_j in quads_with_candidates:
            if quad_j == quad_i:
                continue
            if quad_j not in quad_ref_star_pools:
                continue

            # quad_j的图像星 (取其第一个候选的image_quad, 因为同一四边形的图像星相同)
            img_points_j = cands_j[0].image_quad.points  # (4, 2) 像素
            ra_pool, dec_pool, tree = quad_ref_star_pools[quad_j]

            # 投影到天球
            ra_pred, dec_pred = apply_wcs(img_points_j, wcs_i)

            # 检查每个投影位置是否在3"内有参考星
            pred_vecs = _radec_to_unit_vectors(ra_pred, dec_pred)
            if len(pred_vecs) == 0:
                continue
            dists, _ = tree.query(pred_vecs, k=1)
            # dists是3D单位向量弦长, 转换为角距离(arcsec)
            # 角距离 = 2 * arcsin(d/2), 但小角度近似 dist_rad ≈ chord
            sep_arcsec = np.degrees(dists) * 3600.0  # 弦长(弧度)→度→arcsec近似

            # 所(或多数)投影位置匹配 → 投票+1
            n_matched = int(np.sum(sep_arcsec < VOTE_MATCH_RADIUS_ARCSEC))
            if n_matched >= 3:  # 4个中至少3个匹配
                vote_counts[i] += 1

    # 选得票最高
    best_i = int(np.argmax(vote_counts))
    best_votes = vote_counts[best_i]
    best_quad_idx, best_cand_idx, best_cand, best_wcs = all_candidates[best_i]

    logger.info("投票结果: 最高票=%d (四边形#%d候选#%d), RMS=%.3f\"",
                 best_votes, best_quad_idx, best_cand_idx, best_wcs.rms_arcsec)

    # 若最高票 < 2, 回退到首个四边形候选列表选最低RMS
    if best_votes < MIN_VOTES_TO_ACCEPT:
        logger.info("最高票<%d, 回退到首个四边形候选选最低RMS", MIN_VOTES_TO_ACCEPT)
        return _pick_lowest_rms(quads_with_candidates[0][1], ra0, dec0, voting_used=False)

    return VoteResult(
        best_candidate=best_cand,
        best_wcs=best_wcs,
        best_votes=best_votes,
        best_rms=best_wcs.rms_arcsec,
        voting_used=True,
        message=f"投票胜出 (票数={best_votes}, RMS={best_wcs.rms_arcsec:.3f}\")",
    )


def _radec_to_unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    将(RA,Dec)转换为3D单位向量, 用于KDTree球面最近邻。

    Args:
        ra_deg, dec_deg: (N,) 度

    Returns:
        (N, 3) 单位向量
    """
    ra = np.asarray(ra_deg) * _DEGTORAD
    dec = np.asarray(dec_deg) * _DEGTORAD
    cos_dec = np.cos(dec)
    x = cos_dec * np.cos(ra)
    y = cos_dec * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])

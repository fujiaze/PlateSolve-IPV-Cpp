"""
匹配器模块 (Task 5)
功能: 对每个图像四边形执行k-vector范围查询, 对候选参考四边形做5距离验证
用途: 从k-vector索引中检索与图像四边形几何匹配的参考四边形候选
依赖: numpy

5距离验证:
    σ_d = σ_pos × s₀ (σ_pos=0.5px默认)
    查询 d_AB ∈ [d_AB - 3σ_d, d_AB + 3σ_d]
    对每个候选验证 |d_AC_img - d_AC_cat| < 3σ_d ∧ |d_AD| < 3σ_d ∧ |d_BC| < 3σ_d ∧ |d_BD| < 3σ_d ∧ |d_CD| < 3σ_d
    全部通过 → 保留
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_setup import get_logger
from .quad_geometry import Quad, ReferenceQuad
from .kvector import KVectorIndex, kvector_query

logger = get_logger(__name__)

# 默认参数
DEFAULT_SIGMA_POS_PIXEL = 0.5    # 位置噪声(像素)
DEFAULT_N_SIGMA = 3.0            # 容差倍数 (3σ)


@dataclass
class MatchCandidate:
    """
    匹配候选: 图像四边形 ↔ 参考四边形。

    Attributes:
        image_quad: 图像四边形 (像素坐标, 距离arcsec)
        ref_quad: 参考四边形 (切平面arcsec, 含RA/Dec)
        sigma_d: 距离噪声σ_d (arcsec)
    """
    image_quad: Quad
    ref_quad: ReferenceQuad
    sigma_d: float


def compute_sigma_d(s0_arcsec_per_pixel: float, sigma_pos_pixel: float = DEFAULT_SIGMA_POS_PIXEL) -> float:
    """
    计算距离噪声 σ_d = σ_pos × s₀。

    Args:
        s0_arcsec_per_pixel: 像素尺度(arcsec/pixel)
        sigma_pos_pixel: 位置噪声(像素), 默认0.5

    Returns:
        σ_d (arcsec)
    """
    return sigma_pos_pixel * s0_arcsec_per_pixel


def verify_5_distances(
    img_distances: np.ndarray,
    ref_distances: np.ndarray,
    sigma_d: float,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> tuple[bool, float]:
    """
    5距离验证: |d_img - d_cat| < n_sigma·σ_d 对5个距离全部成立。

    Args:
        img_distances: shape (6,) 图像四边形6距离 [d_AB,d_AC,d_AD,d_BC,d_BD,d_CD]
        ref_distances: shape (6,) 参考四边形6距离
        sigma_d: 距离噪声(arcsec)
        n_sigma: 容差倍数

    Returns:
        (pass_all, max_diff): 全部通过=True, max_diff=5个距离的最大差值
    """
    tolerance = n_sigma * sigma_d
    # 验证5个距离 (索引1~5: d_AC,d_AD,d_BC,d_BD,d_CD)
    diffs = np.abs(img_distances[1:6] - ref_distances[1:6])
    max_diff = float(diffs.max())
    pass_all = bool(np.all(diffs < tolerance))
    return pass_all, max_diff


def match_image_quad(
    image_quad: Quad,
    kvector_index: KVectorIndex,
    s0_arcsec_per_pixel: float,
    sigma_pos_pixel: float = DEFAULT_SIGMA_POS_PIXEL,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> list[MatchCandidate]:
    """
    对单个图像四边形执行k-vector查询 + 5距离验证。

    Args:
        image_quad: 图像四边形 (6距离为arcsec)
        kvector_index: 参考k-vector索引
        s0_arcsec_per_pixel: 像素尺度(arcsec/pixel)
        sigma_pos_pixel: 位置噪声(像素)
        n_sigma: 容差倍数

    Returns:
        通过验证的候选列表
    """
    sigma_d = compute_sigma_d(s0_arcsec_per_pixel, sigma_pos_pixel)
    tolerance = n_sigma * sigma_d
    d_AB_img = image_quad.d_AB

    # k-vector范围查询 d_AB ∈ [d_AB - 3σ_d, d_AB + 3σ_d]
    raw_candidates = kvector_query(kvector_index, d_AB_img, tolerance)
    logger.info("图像四边形 d_AB=%.2f\" → k-vector返回%d候选 (σ_d=%.3f\", 容差±%.3f\")",
                 d_AB_img, len(raw_candidates), sigma_d, tolerance)

    # 5距离验证
    verified: list[MatchCandidate] = []
    for ref_quad in raw_candidates:
        pass_all, max_diff = verify_5_distances(
            image_quad.distances, ref_quad.distances, sigma_d, n_sigma
        )
        if pass_all:
            verified.append(MatchCandidate(
                image_quad=image_quad,
                ref_quad=ref_quad,
                sigma_d=sigma_d,
            ))

    logger.info("5距离验证通过: %d/%d", len(verified), len(raw_candidates))
    return verified


def match_all_quads(
    image_quads: list[Quad],
    kvector_index: KVectorIndex,
    s0_arcsec_per_pixel: float,
    sigma_pos_pixel: float = DEFAULT_SIGMA_POS_PIXEL,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> list[list[MatchCandidate]]:
    """
    对所有图像四边形执行匹配。

    Args:
        image_quads: 图像四边形列表
        kvector_index: 参考k-vector索引
        s0_arcsec_per_pixel: 像素尺度
        sigma_pos_pixel: 位置噪声(像素)
        n_sigma: 容差倍数

    Returns:
        candidates_per_quad: 每个图像四边形对应的候选列表
    """
    sigma_d = compute_sigma_d(s0_arcsec_per_pixel, sigma_pos_pixel)
    logger.info("匹配开始: %d个图像四边形, σ_d=%.3f\" (σ_pos=%.2fpx × s0=%.4f\"/px)",
                 len(image_quads), sigma_d, sigma_pos_pixel, s0_arcsec_per_pixel)

    candidates_per_quad: list[list[MatchCandidate]] = []
    total_verified = 0

    for i, img_quad in enumerate(image_quads):
        candidates = match_image_quad(
            img_quad, kvector_index, s0_arcsec_per_pixel, sigma_pos_pixel, n_sigma
        )
        candidates_per_quad.append(candidates)
        total_verified += len(candidates)
        if candidates:
            logger.info("  四边形#%d d_AB=%.2f\" → %d候选通过",
                         i, img_quad.d_AB, len(candidates))

    n_quads_with_candidates = sum(1 for c in candidates_per_quad if len(c) > 0)
    logger.info("匹配完成: %d/%d个四边形有候选, 总候选数=%d",
                 n_quads_with_candidates, len(image_quads), total_verified)
    return candidates_per_quad

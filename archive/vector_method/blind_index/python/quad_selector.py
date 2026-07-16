"""
四边形选择模块 (Task 4)
功能: 金字塔策略生成候选四边形——图像侧(Top-20亮星池+pivot+8最近邻)和参考侧(每颗星pivot+3最近邻)
用途: 为k-vector索引构建(参考侧)和匹配查询(图像侧)提供高质量四边形候选
依赖: numpy, scipy.spatial.cKDTree

图像侧策略:
    - Top-20最亮星为亮星池pool
    - 每颗pivot in pool[:15]: 取8最近邻(像素距离), {pivot}∪邻星[:3]→1个四边形
    - 规范化后按 uniqueness(d_AB) + geometry_quality 排序, 取前5

参考侧策略:
    - 每颗参考星作pivot, 取3最近邻(切平面距离)→1个四边形
    - 规范化 + 退化过滤

uniqueness(d_AB): 1/(±2"范围内参考四边形数), 密度越低越独特
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .logging_setup import get_logger
from .quad_geometry import (
    Quad, ReferenceQuad,
    make_quad, make_reference_quad, is_degenerate,
    MIN_INTERIOR_ANGLE_DEG, MIN_AREA_RATIO, MIN_SHORTEST_EDGE_ARCSEC,
)

logger = get_logger(__name__)

# 图像侧参数
IMAGE_POOL_SIZE = 20          # 亮星池大小
IMAGE_PIVOT_COUNT = 15        # pivot数量
IMAGE_NNEIGHBORS = 8          # 每个pivot的最近邻数
IMAGE_TOP_QUADS = 5           # 最终保留的图像四边形数

# 参考侧参数
REF_NNEIGHBORS = 3            # 每个pivot的最近邻数(组成4星四边形)

# uniqueness参数
UNIQUENESS_WINDOW = 2.0       # ±2"密度估计窗口


def _geometry_quality(points: np.ndarray, distances: np.ndarray) -> float:
    """
    四边形几何质量评分 (0~1, 越高越好)。

    综合: 最小内角 + 面积比 + 边长比

    Args:
        points: shape (4,2) 规范化坐标
        distances: shape (6,) 6距离

    Returns:
        质量分 [0, 1]
    """
    from .quad_geometry import _min_interior_angle_deg, _quad_area
    min_angle = _min_interior_angle_deg(points)
    area = _quad_area(points)
    d_AB = float(distances[0])
    d_CD = float(distances[5])
    denom = d_AB * d_CD
    if denom < 1e-15:
        return 0.0
    area_ratio = area / denom

    # 归一化到[0,1]
    angle_score = min(1.0, max(0.0, (min_angle - MIN_INTERIOR_ANGLE_DEG) / (90.0 - MIN_INTERIOR_ANGLE_DEG)))
    area_score = min(1.0, max(0.0, (area_ratio - MIN_AREA_RATIO) / (0.5 - MIN_AREA_RATIO)))

    return 0.5 * angle_score + 0.5 * area_score


def generate_image_quads(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    s0_arcsec_per_pixel: float,
    ref_kvector_index=None,
    pool_size: int = IMAGE_POOL_SIZE,
    pivot_count: int = IMAGE_PIVOT_COUNT,
    n_neighbors: int = IMAGE_NNEIGHBORS,
    top_n: int = IMAGE_TOP_QUADS,
    saturated_arr: Optional[np.ndarray] = None,
) -> list[Quad]:
    """
    金字塔策略生成图像四边形。

    Args:
        x_arr: 星点x坐标(像素)
        y_arr: 星点y坐标(像素)
        s0_arcsec_per_pixel: 像素尺度(arcsec/pixel)
        ref_kvector_index: 参考k-vector索引(用于uniqueness评分), 可选
        pool_size: 亮星池大小
        pivot_count: pivot数量
        n_neighbors: 每个pivot最近邻数
        top_n: 最终保留四边形数
        saturated_arr: 饱和标志数组(0=正常, 1=饱和), 可选; 提供时跳过饱和星

    Returns:
        规范化图像四边形列表(按质量降序), 6距离已转换为arcsec
    """
    n_stars = len(x_arr)
    if n_stars < 4:
        logger.warning("星点数%d < 4, 无法生成图像四边形", n_stars)
        return []

    # 1. 亮星池: 跳过饱和星 (flux=-1, 不在Gaia DR3中, 无法匹配参考星)
    #    star_detector 排序为饱和星优先(按r降序)+正常星(按flux降序),
    #    "Top-N最亮"应取正常星按flux排序的前N颗, 而非饱和星。
    if saturated_arr is not None:
        normal_mask = (saturated_arr == 0)
        n_sat = int(np.sum(~normal_mask))
        x_arr = x_arr[normal_mask]
        y_arr = y_arr[normal_mask]
        n_stars = len(x_arr)
        logger.info("跳过饱和星: %d颗 → %d颗正常星", n_sat, n_stars)
        if n_stars < 4:
            logger.warning("正常星数%d < 4, 无法生成图像四边形", n_stars)
            return []

    pool_n = min(pool_size, n_stars)
    pool_x = x_arr[:pool_n]
    pool_y = y_arr[:pool_n]
    pool_points = np.column_stack([pool_x, pool_y])  # (pool_n, 2) 像素

    logger.info("图像四边形生成: 亮星池=%d星 (pool_size=%d, 正常星=%d)",
                 pool_n, pool_size, n_stars)

    # 2. 构建KDTree用于最近邻搜索(像素空间, 亮星池内)
    tree = cKDTree(pool_points)

    # 3. 每个pivot生成1个四边形
    pivot_n = min(pivot_count, pool_n)
    candidates: list[tuple[float, Quad]] = []

    for pivot_idx in range(pivot_n):
        pivot_point = pool_points[pivot_idx]
        # 查询n_neighbors+1个最近邻(含自身)
        k = min(n_neighbors + 1, pool_n)
        dists_px, indices = tree.query(pivot_point, k=k)
        # 排除自身
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        neighbors = [int(i) for i in indices if int(i) != pivot_idx][:n_neighbors]

        if len(neighbors) < 3:
            continue

        # {pivot} ∪ 邻星[:3] → 4星四边形
        quad_input_idx = [pivot_idx] + neighbors[:3]
        quad_points_px = pool_points[quad_input_idx]  # (4, 2) 像素

        # 规范化 (像素空间, 然后转arcsec)
        from .quad_geometry import canonicalize_quad
        result = canonicalize_quad(quad_points_px, tuple(quad_input_idx))
        if result is None:
            continue
        canon_idx, canon_points_px, dists_px = result

        # 退化检测 (像素空间几何与arcsec空间几何一致, 比例无关)
        deg, _ = is_degenerate(canon_points_px, dists_px,
                                min_edge_arcsec=MIN_SHORTEST_EDGE_ARCSEC / s0_arcsec_per_pixel)
        if deg:
            continue

        # 转换为arcsec距离
        dists_arcsec = dists_px * s0_arcsec_per_pixel
        canon_points_arcsec = canon_points_px * s0_arcsec_per_pixel  # 仅用于评分(像素坐标也保留)

        quad = Quad(idx=canon_idx, distances=dists_arcsec, points=canon_points_px)

        # 4. 评分: uniqueness + geometry_quality
        if ref_kvector_index is not None:
            density = float(len(kvector_query_for_uniqueness(ref_kvector_index, quad.d_AB)))
            uniqueness = 1.0 / (1.0 + density)  # 密度0→1.0(最独特), 密度高→趋近0
        else:
            uniqueness = 0.5  # 无索引时中性值

        geo_q = _geometry_quality(canon_points_px, dists_px)
        score = 0.6 * uniqueness + 0.4 * geo_q

        candidates.append((score, quad))

    # 5. 按分数降序, 取前top_n
    candidates.sort(key=lambda c: c[0], reverse=True)
    quads = [c[1] for c in candidates[:top_n]]

    logger.info("图像四边形生成完成: %d个候选, 选取前%d个 (分数: %s)",
                 len(candidates), len(quads),
                 ", ".join(f"{c[0]:.3f}" for c in candidates[:len(quads)]))
    return quads


def kvector_query_for_uniqueness(index, d_ab: float, window: float = UNIQUENESS_WINDOW):
    """封装kvector查询用于uniqueness (避免循环导入)。"""
    from .kvector import kvector_query
    return kvector_query(index, d_ab, window)


def generate_reference_quads(
    xi_arr: np.ndarray,        # (N,) 切平面xi arcsec
    eta_arr: np.ndarray,       # (N,) 切平面eta arcsec
    ra_arr: np.ndarray,        # (N,) RA度
    dec_arr: np.ndarray,       # (N,) Dec度
    n_neighbors: int = REF_NNEIGHBORS,
) -> list[ReferenceQuad]:
    """
    参考侧四边形生成: 每颗星作pivot + 3最近邻。

    Args:
        xi_arr: 切平面xi坐标(arcsec)
        eta_arr: 切平面eta坐标(arcsec)
        ra_arr: RA(度)
        dec_arr: Dec(度)
        n_neighbors: 每个pivot最近邻数(默认3, 组成4星四边形)

    Returns:
        规范化参考四边形列表(已通过退化过滤)
    """
    n = len(xi_arr)
    if n < 4:
        logger.warning("参考星数%d < 4, 无法生成参考四边形", n)
        return []

    points = np.column_stack([xi_arr, eta_arr])  # (N, 2) arcsec
    tree = cKDTree(points)

    logger.info("参考四边形生成: %d颗参考星, 每颗pivot取%d最近邻", n, n_neighbors)

    ref_quads: list[ReferenceQuad] = []
    skipped = 0

    for pivot_idx in range(n):
        pivot_point = points[pivot_idx]
        k = min(n_neighbors + 1, n)
        dists, indices = tree.query(pivot_point, k=k)
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        neighbors = [int(i) for i in indices if int(i) != pivot_idx][:n_neighbors]

        if len(neighbors) < n_neighbors:
            continue

        quad_input_idx = [pivot_idx] + neighbors  # 4个索引
        quad_points = points[quad_input_idx]       # (4, 2) arcsec
        quad_ra = ra_arr[quad_input_idx]
        quad_dec = dec_arr[quad_input_idx]

        ref_quad = make_reference_quad(
            points_tangent=quad_points,
            indices=tuple(quad_input_idx),
            ra=quad_ra,
            dec=quad_dec,
            check_degenerate=True,
        )
        if ref_quad is None:
            skipped += 1
            continue
        ref_quads.append(ref_quad)

    logger.info("参考四边形生成完成: %d个 (跳过退化%d个)", len(ref_quads), skipped)
    return ref_quads

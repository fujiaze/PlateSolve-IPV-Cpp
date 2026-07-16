"""
四边形几何模块 (Task 2+4)
功能: 4星四边形的6距离计算、规范化排序(A/B/C/D)、退化构型过滤
用途: 为图像侧和参考侧提供统一的四边形几何表示，使四边形旋转/平移不变
依赖: numpy

规范化约定:
    - d_AB = 6距离中最长边, A和B为其端点
    - C/D排序: 使 d_AC ≤ d_BC; 相等则用 d_AD ≤ d_BD
    - 6距离固定顺序: [d_AB, d_AC, d_AD, d_BC, d_BD, d_CD] (arcsec)

退化过滤:
    - 最小内角 > 10°
    - 面积/(d_AB·d_CD) > 0.1
    - 最短边 > 10"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .logging_setup import get_logger

logger = get_logger(__name__)

# 退化过滤默认阈值
MIN_INTERIOR_ANGLE_DEG = 10.0       # 最小内角(度)
MIN_AREA_RATIO = 0.1                # 面积/(d_AB·d_CD)
MIN_SHORTEST_EDGE_ARCSEC = 10.0     # 最短边(arcsec)


@dataclass
class Quad:
    """
    规范化四边形。

    Attributes:
        idx: (A, B, C, D) 原始星点数组中的索引
        distances: shape (6,) [d_AB, d_AC, d_AD, d_BC, d_BD, d_CD] (arcsec)
        points: shape (4, 2) 规范化顺序的坐标 (图像=像素, 参考=切平面arcsec)
    """
    idx: tuple[int, int, int, int]
    distances: np.ndarray   # shape (6,)
    points: np.ndarray      # shape (4, 2)

    @property
    def d_AB(self) -> float:
        return float(self.distances[0])

    @property
    def d_CD(self) -> float:
        return float(self.distances[5])


@dataclass
class ReferenceQuad:
    """
    参考四边形 (含天球坐标)。

    Attributes:
        idx: (A, B, C, D) 参考星数组中的索引
        distances: shape (6,) [d_AB, d_AC, d_AD, d_BC, d_BD, d_CD] (arcsec)
        points: shape (4, 2) 切平面坐标 (xi, eta) arcsec
        ra: shape (4,) RA(度)
        dec: shape (4,) Dec(度)
        ra_center: 四边形质心RA(度)
        dec_center: 四边形质心Dec(度)
    """
    idx: tuple[int, int, int, int]
    distances: np.ndarray   # shape (6,)
    points: np.ndarray      # shape (4, 2) 切平面arcsec
    ra: np.ndarray          # shape (4,)
    dec: np.ndarray         # shape (4,)
    ra_center: float = 0.0
    dec_center: float = 0.0


def compute_pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    计算4个点的全部6条配对距离。

    Args:
        points: shape (4, 2)

    Returns:
        dist_matrix: shape (4, 4) 对称距离矩阵
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (4,4,2)
    return np.sqrt(np.sum(diff ** 2, axis=2))


def canonicalize_quad(
    points: np.ndarray,
    indices: Optional[tuple[int, int, int, int]] = None,
) -> Optional[tuple[tuple[int, int, int, int], np.ndarray, np.ndarray]]:
    """
    将4个点规范化为 (A,B,C,D) 顺序。

    规范化规则:
        1. d_AB = 最长边, A/B为其端点
        2. 剩余两星按 d_AC ≤ d_BC 排序(若相等用 d_AD ≤ d_BD)
        3. 输出6距离 [d_AB, d_AC, d_AD, d_BC, d_BD, d_CD]

    Args:
        points: shape (4, 2) 4个点坐标
        indices: 可选, 原始索引元组 (i0,i1,i2,i3) 对应points的4行

    Returns:
        (canonical_idx, canonical_points, distances) 或 None(退化)
        - canonical_idx: (A,B,C,D) 原始索引
        - canonical_points: shape (4,2) 按A,B,C,D顺序
        - distances: shape (6,) [d_AB,d_AC,d_AD,d_BC,d_BD,d_CD]
    """
    if indices is None:
        indices = (0, 1, 2, 3)

    d = compute_pairwise_distances(points)  # (4,4)

    # 1. 找最长边 → A, B
    max_d = -1.0
    A, B = 0, 1
    for i in range(4):
        for j in range(i + 1, 4):
            if d[i, j] > max_d:
                max_d = d[i, j]
                A, B = i, j

    # 2. 剩余两星
    remaining = [i for i in range(4) if i not in (A, B)]
    r1, r2 = remaining[0], remaining[1]

    # 两种分配方案:
    #   方案1: C=r1, D=r2 → key1 = (d[A,r1]-d[B,r1], d[A,r2]-d[B,r2])
    #   方案2: C=r2, D=r1 → key2 = (d[A,r2]-d[B,r2], d[A,r1]-d[B,r1])
    # 选字典序较小的方案 (优先使 d_AC ≤ d_BC, 其次 d_AD ≤ d_BD)
    diff_r1 = d[A, r1] - d[B, r1]
    diff_r2 = d[A, r2] - d[B, r2]
    key1 = (diff_r1, diff_r2)  # 方案1: C=r1, D=r2
    key2 = (diff_r2, diff_r1)  # 方案2: C=r2, D=r1
    if key1 <= key2:
        C, D = r1, r2
    else:
        C, D = r2, r1

    # 3. 构建6距离 (规范顺序)
    order = [A, B, C, D]
    canon_points = points[order].copy()
    distances = np.array([
        d[A, B],  # d_AB
        d[A, C],  # d_AC
        d[A, D],  # d_AD
        d[B, C],  # d_BC
        d[B, D],  # d_BD
        d[C, D],  # d_CD
    ], dtype=np.float64)

    canon_idx = (indices[A], indices[B], indices[C], indices[D])
    return canon_idx, canon_points, distances


def _cyclic_order(points: np.ndarray) -> np.ndarray:
    """
    获取4个点按凸包逆时针的循环顺序索引。

    Args:
        points: shape (4, 2)

    Returns:
        order: shape (4,) 循环顺序索引
    """
    centroid = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    return np.argsort(angles)


def _quad_area(points: np.ndarray) -> float:
    """
    计算4点凸包面积 (鞋带公式)。

    Args:
        points: shape (4, 2)

    Returns:
        面积(标量)
    """
    order = _cyclic_order(points)
    ordered = points[order]
    area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        area += ordered[i, 0] * ordered[j, 1]
        area -= ordered[j, 0] * ordered[i, 1]
    return abs(area) / 2.0


def _min_interior_angle_deg(points: np.ndarray) -> float:
    """
    计算4点凸包最小内角(度)。

    Args:
        points: shape (4, 2)

    Returns:
        最小内角(度)
    """
    order = _cyclic_order(points)
    ordered = points[order]
    angles = []
    for i in range(4):
        p_prev = ordered[(i - 1) % 4]
        p_curr = ordered[i]
        p_next = ordered[(i + 1) % 4]
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-15 or n2 < 1e-15:
            return 0.0
        cos_a = np.dot(v1, v2) / (n1 * n2)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        angles.append(np.degrees(np.arccos(cos_a)))
    return float(min(angles))


def is_degenerate(
    points: np.ndarray,
    distances: np.ndarray,
    min_angle_deg: float = MIN_INTERIOR_ANGLE_DEG,
    min_area_ratio: float = MIN_AREA_RATIO,
    min_edge_arcsec: float = MIN_SHORTEST_EDGE_ARCSEC,
) -> tuple[bool, str]:
    """
    退化构型检测。

    Args:
        points: shape (4, 2) 规范化坐标
        distances: shape (6,) 6距离
        min_angle_deg: 最小内角阈值(度)
        min_area_ratio: 面积/(d_AB·d_CD)阈值
        min_edge_arcsec: 最短边阈值(arcsec)

    Returns:
        (is_degenerate, reason): 退化=True, reason=原因描述
    """
    # 1. 最短边
    d_min = float(distances.min())
    if d_min < min_edge_arcsec:
        return True, f"最短边={d_min:.2f}\" < {min_edge_arcsec}\""

    # 2. 最小内角
    min_angle = _min_interior_angle_deg(points)
    if min_angle < min_angle_deg:
        return True, f"最小内角={min_angle:.1f}° < {min_angle_deg}°"

    # 3. 面积比
    area = _quad_area(points)
    d_AB = float(distances[0])
    d_CD = float(distances[5])
    denom = d_AB * d_CD
    if denom < 1e-15:
        return True, "d_AB·d_CD ≈ 0"
    area_ratio = area / denom
    if area_ratio < min_area_ratio:
        return True, f"面积比={area_ratio:.3f} < {min_area_ratio}"

    return False, ""


def make_quad(
    points: np.ndarray,
    indices: tuple[int, int, int, int],
    check_degenerate: bool = True,
) -> Optional[Quad]:
    """
    从4个点构建规范化四边形 (含退化过滤)。

    Args:
        points: shape (4, 2) 4个点
        indices: 原始索引
        check_degenerate: 是否执行退化过滤

    Returns:
        Quad 或 None(退化)
    """
    result = canonicalize_quad(points, indices)
    if result is None:
        return None
    canon_idx, canon_points, distances = result

    if check_degenerate:
        deg, reason = is_degenerate(canon_points, distances)
        if deg:
            return None

    return Quad(idx=canon_idx, distances=distances, points=canon_points)


def make_reference_quad(
    points_tangent: np.ndarray,   # (4, 2) 切平面arcsec
    indices: tuple[int, int, int, int],
    ra: np.ndarray,               # (4,) 度
    dec: np.ndarray,              # (4,) 度
    check_degenerate: bool = True,
) -> Optional[ReferenceQuad]:
    """
    从4个参考星构建规范化参考四边形 (含退化过滤)。

    Args:
        points_tangent: shape (4, 2) 切平面(xi,eta) arcsec
        indices: 参考星数组索引
        ra: shape (4,) RA(度)
        dec: shape (4,) Dec(度)
        check_degenerate: 是否执行退化过滤

    Returns:
        ReferenceQuad 或 None(退化)
    """
    result = canonicalize_quad(points_tangent, indices)
    if result is None:
        return None
    canon_idx, canon_points, distances = result

    if check_degenerate:
        deg, reason = is_degenerate(canon_points, distances)
        if deg:
            return None

    # 按规范化顺序重排ra/dec
    order_map = {orig: i for i, orig in enumerate(indices)}
    canon_order_in_input = [order_map[canon_idx[k]] for k in range(4)]
    canon_ra = ra[canon_order_in_input].copy()
    canon_dec = dec[canon_order_in_input].copy()

    # 质心
    ra_center = float(np.mean(canon_ra))
    dec_center = float(np.mean(canon_dec))

    return ReferenceQuad(
        idx=canon_idx,
        distances=distances,
        points=canon_points,
        ra=canon_ra,
        dec=canon_dec,
        ra_center=ra_center,
        dec_center=dec_center,
    )

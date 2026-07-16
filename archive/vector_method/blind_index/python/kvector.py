"""
k-vector索引模块 (Task 3)
功能: 从参考四边形库构建k-vector区间表，提供O(1)定位+O(k)扫描的范围查询
用途: 4SADQ-KV盲解析的核心索引，按最长边d_AB连续范围检索，无binning量化损失
依赖: numpy

k-vector结构:
    - S: 按d_AB升序排列的参考四边形数组
    - K[j] = S中 d_AB ≤ d_min + (j+1)·Δ 的最后一个元素下标
    - Δ = 0.5 arcsec (步长)
    - d_min = 2.0" (下限), d_max = 最大d_AB

范围查询 d_AB ∈ [d-δ, d+δ]:
    j_lo = floor((d-δ-d_min)/Δ), j_hi = floor((d+δ-d_min)/Δ)
    clamp j_lo, j_hi 到 [0, len(K)-1]
    idx_lo = K[j_lo-1]+1 if j_lo>0 else 0; idx_hi = K[j_hi]
    return S[idx_lo : idx_hi+1]
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_setup import get_logger
from .quad_geometry import ReferenceQuad

logger = get_logger(__name__)

# k-vector默认参数
DEFAULT_DELTA = 0.5          # 步长(arcsec)
DEFAULT_D_MIN = 2.0          # 最小d_AB(arcsec) 下限


@dataclass
class KVectorIndex:
    """
    k-vector索引。

    Attributes:
        S: 排序后的参考四边形列表 (按d_AB升序)
        K: 区间表, K[j] = S中d_AB ≤ d_min+(j+1)·Δ 的末下标 (int32数组)
        d_ab_sorted: S中每个四边形的d_AB值 (升序)
        delta: 步长(arcsec)
        d_min: 最小d_AB(arcsec)
        d_max: 最大d_AB(arcsec)
    """
    S: list[ReferenceQuad]
    K: np.ndarray             # shape (n_bins,) int32
    d_ab_sorted: np.ndarray   # shape (N,) float64
    delta: float
    d_min: float
    d_max: float

    @property
    def n_quads(self) -> int:
        return len(self.S)


def build_kvector(
    ref_quads: list[ReferenceQuad],
    delta: float = DEFAULT_DELTA,
    d_min: float = DEFAULT_D_MIN,
) -> Optional[KVectorIndex]:
    """
    构建k-vector索引。

    Args:
        ref_quads: 参考四边形列表
        delta: k-vector步长(arcsec), 默认0.5"
        d_min: d_AB下限(arcsec), 默认2.0"

    Returns:
        KVectorIndex 或 None(空列表)
    """
    n = len(ref_quads)
    if n == 0:
        logger.warning("参考四边形列表为空, 无法构建k-vector")
        return None

    # 1. 按d_AB升序排序
    ref_quads_sorted = sorted(ref_quads, key=lambda q: q.distances[0])
    d_ab_arr = np.array([q.distances[0] for q in ref_quads_sorted], dtype=np.float64)

    # 2. 应用d_min下限: 截断低于d_min的(理论上不应有, 因为退化过滤已排除最短边<10")
    d_max = float(d_ab_arr[-1])
    # 使用传入的d_min (可能小于实际最小d_AB, 这没关系——K表开头会有空区间)
    actual_d_min = float(d_ab_arr[0])
    effective_d_min = min(d_min, actual_d_min)

    # 3. 构建区间表 K[j] = S中 d_AB ≤ effective_d_min + (j+1)·Δ 的末下标
    n_bins = max(1, int(np.ceil((d_max - effective_d_min) / delta)) + 1)
    K = np.zeros(n_bins, dtype=np.int32)
    # 对每个j, 用二分查找确定 d_ab_arr 中 ≤ threshold 的最后一个下标
    thresholds = effective_d_min + (np.arange(n_bins) + 1) * delta  # shape (n_bins,)
    # searchsorted(side='right') 返回第一个 > threshold 的位置, -1 即为末下标
    indices = np.searchsorted(d_ab_arr, thresholds, side="right") - 1
    K[:] = indices
    # K[j] = -1 表示该区间无元素, 但查询时 K[j_lo-1]+1 会变成0, 正确处理空区间

    logger.info("k-vector构建完成: N=%d, Δ=%.2f\", d_min=%.2f\"(effective=%.2f\"), d_max=%.2f\", n_bins=%d",
                 n, delta, d_min, effective_d_min, d_max, n_bins)

    return KVectorIndex(
        S=ref_quads_sorted,
        K=K,
        d_ab_sorted=d_ab_arr,
        delta=float(delta),
        d_min=float(effective_d_min),
        d_max=d_max,
    )


def kvector_query(
    index: KVectorIndex,
    d_ab: float,
    delta_d: float,
) -> list[ReferenceQuad]:
    """
    k-vector范围查询: 返回 d_AB ∈ [d_ab - delta_d, d_ab + delta_d] 的参考四边形。

    Args:
        index: k-vector索引
        d_ab: 查询的d_AB值(arcsec)
        delta_d: 容差(arcsec), 查询区间 [d_ab-delta_d, d_ab+delta_d]

    Returns:
        候选参考四边形列表
    """
    if index.n_quads == 0:
        return []

    d_lo = d_ab - delta_d
    d_hi = d_ab + delta_d

    # 计算 j_lo, j_hi
    n_bins = len(index.K)
    j_lo = int(np.floor((d_lo - index.d_min) / index.delta))
    j_hi = int(np.floor((d_hi - index.d_min) / index.delta))

    # clamp 到 [0, n_bins-1]
    j_lo = max(0, min(j_lo, n_bins - 1))
    j_hi = max(0, min(j_hi, n_bins - 1))

    if j_lo > j_hi:
        j_lo, j_hi = j_hi, j_lo

    # idx_lo = K[j_lo-1]+1 if j_lo>0 else 0; idx_hi = K[j_hi]
    if j_lo > 0:
        idx_lo = int(index.K[j_lo - 1]) + 1
    else:
        idx_lo = 0
    idx_hi = int(index.K[j_hi])

    # clamp
    idx_lo = max(0, min(idx_lo, index.n_quads - 1))
    idx_hi = max(idx_lo - 1, min(idx_hi, index.n_quads - 1))

    if idx_lo > idx_hi:
        return []

    # 顺序扫描区间, 精确过滤(避免区间边界外的元素)
    candidates = []
    for i in range(idx_lo, idx_hi + 1):
        d_ab_i = index.d_ab_sorted[i]
        if d_lo <= d_ab_i <= d_hi:
            candidates.append(index.S[i])

    return candidates


def estimate_density(
    index: KVectorIndex,
    d_ab: float,
    window: float = 2.0,
) -> float:
    """
    估计d_AB附近的参考四边形密度 (用于uniqueness评分)。

    Args:
        index: k-vector索引
        d_ab: 查询d_AB值(arcsec)
        window: 密度估计窗口(±window, arcsec), 默认2.0"

    Returns:
        密度(区间内四边形数), 越低越独特
    """
    candidates = kvector_query(index, d_ab, window)
    return float(len(candidates))

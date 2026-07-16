"""
星对库构建与k-vector索引模块 (Task 2)
功能: 从Gaia参考星构建二星对库(每颗星取K=8最近邻)，按d_cat升序建立k-vector区间表
用途: ADV-PA盲解析的核心索引，支持O(1)定位+O(k)扫描的角距范围查询
依赖: numpy, scipy.spatial.cKDTree, lib.plate_solve.python.vector_match_v2 (gnomonic_forward)

星对库结构 (structured array):
    d_cat:     float64  球面角距(arcsec)
    PA_cat:    float64  位置角(度, 从北向东, [0,360))
    ra_i:      float64  主星RA(度)
    dec_i:     float64  主星Dec(度)
    ra_j:      float64  邻星RA(度)
    dec_j:     float64  邻星Dec(度)
    star_id_i: int32    主星在输入数组中的下标

k-vector结构:
    S: 按d_cat升序排列的星对数组
    K[j] = S中 d_cat ≤ d_min + (j+1)·Δ 的最后一个元素下标
    Δ = 0.5 arcsec (步长)
    d_min = 10.0" (下限), d_max = 最大d_cat

范围查询 d_cat ∈ [d-δ, d+δ]:
    j_lo = floor((d-δ-d_min)/Δ), j_hi = floor((d+δ-d_min)/Δ)
    clamp j_lo, j_hi 到 [0, n_bins-1]
    idx_lo = K[j_lo-1]+1 if j_lo>0 else 0; idx_hi = K[j_hi]
    顺序扫描 S[idx_lo : idx_hi+1]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .logging_setup import get_logger
from .spherical_geom import angular_distance_arcsec, position_angle_deg

# 复用V3.5 gnomonic投影
from lib.plate_solve.python.vector_match_v2 import gnomonic_forward

logger = get_logger(__name__)

# 星对structured array数据类型
PAIR_DTYPE = np.dtype([
    ('d_cat', np.float64),
    ('PA_cat', np.float64),
    ('ra_i', np.float64),
    ('dec_i', np.float64),
    ('ra_j', np.float64),
    ('dec_j', np.float64),
    ('star_id_i', np.int32),
])

# k-vector默认参数
# K=20 (fix-adv-pa-phase1-bugs Bug 1): K=8 在密集天区丢失 94.3% 真匹配对，
# 提升到 K=20 后 5000 参考星约产生 50000 星对 (≤100000 上限)。
# 双向配对 (i→j 与 j→i 的 PA_cat 差 180°) 不在星对库实现，
# 改在 voting.py 中投票 rot 和 (rot+180)%360 等价处理 (Option B, 更简单)。
DEFAULT_K_NEIGHBORS = 20          # 每颗星取K个最近邻
DEFAULT_DELTA = 0.5               # k-vector步长(arcsec)
DEFAULT_D_MIN = 10.0              # 最小d_cat(arcsec) — 过滤过近的星对(可能为重复源)
DEFAULT_D_MAX = 18000.0           # 最大d_cat(arcsec) = 5°


@dataclass
class PairLibrary:
    """
    星对库 + k-vector索引。

    Attributes:
        S: 按d_cat升序排列的星对structured array
        K: k-vector区间表, K[j] = S中d_cat ≤ d_min+(j+1)·Δ 的末下标 (int32数组)
        d_sorted: S中每个星对的d_cat值(升序, float64数组)
        delta: k-vector步长(arcsec)
        d_min: 最小d_cat(arcsec)
        d_max: 最大d_cat(arcsec)
        n_pairs: 星对总数
    """
    S: np.ndarray             # shape (N,) PAIR_DTYPE
    K: np.ndarray             # shape (n_bins,) int32
    d_sorted: np.ndarray      # shape (N,) float64
    delta: float
    d_min: float
    d_max: float

    @property
    def n_pairs(self) -> int:
        return len(self.S)


def build_pair_library(
    ra_arr: np.ndarray,
    dec_arr: np.ndarray,
    mag_arr: np.ndarray,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    d_min_arcsec: float = DEFAULT_D_MIN,
    d_max_arcsec: float = DEFAULT_D_MAX,
) -> Optional[PairLibrary]:
    """
    构建星对库: 每颗星取K个最近邻组成星对，计算(d_cat, PA_cat)。

    流程:
        1. 以参考星质心为中心做gnomonic投影到切平面(xi, eta arcsec)
        2. 用cKDTree在切平面上找每颗星的K个最近邻
        3. 对每对(i, j_neighbor)且j>i(避免重复): 计算d_cat和PA_cat(i→j)
        4. 过滤: d_cat ∈ [d_min, d_max]

    Args:
        ra_arr: 参考星RA数组(度)
        dec_arr: 参考星Dec数组(度)
        mag_arr: 参考星星等数组(度, 目前未用于过滤, 保留接口)
        k_neighbors: 每颗星取的最近邻数, 默认8
        d_min_arcsec: d_cat下限(arcsec), 默认10"
        d_max_arcsec: d_cat上限(arcsec), 默认18000"(5°)

    Returns:
        PairLibrary 或 None(空输入)
    """
    ra_arr = np.asarray(ra_arr, dtype=np.float64)
    dec_arr = np.asarray(dec_arr, dtype=np.float64)
    n_stars = len(ra_arr)
    if n_stars < 2:
        logger.warning("参考星数不足: %d < 2, 无法构建星对库", n_stars)
        return None

    # 1. 以质心为中心做gnomonic投影
    ra0 = float(np.mean(ra_arr))
    dec0 = float(np.mean(dec_arr))
    xi, eta, valid = gnomonic_forward(ra_arr, dec_arr, ra0, dec0)
    # 无效投影点(离投影中心太远)直接丢弃
    if not np.all(valid):
        n_invalid = int(np.sum(~valid))
        logger.warning("gnomonic投影丢弃%d颗无效参考星", n_invalid)
        ra_arr = ra_arr[valid]
        dec_arr = dec_arr[valid]
        xi = xi[valid]
        eta = eta[valid]
        n_stars = len(ra_arr)
        if n_stars < 2:
            return None

    # 2. cKDTree在切平面上找K最近邻
    # 切平面坐标单位是arcsec, cKDTree用欧氏距离(在此近似为角距)
    coords = np.column_stack([xi, eta])  # (N, 2)
    tree = cKDTree(coords)
    k_query = min(k_neighbors + 1, n_stars)  # +1因为最近邻包含自己
    dists_nn, idxs_nn = tree.query(coords, k=k_query)  # (N, k_query)
    # idxs_nn[:, 0] 是自己, 跳过

    logger.info("cKDTree最近邻查询完成: %d颗星 × %d邻居", n_stars, k_query - 1)

    # 3. 构建星对列表 (i, j) 其中 j > i 避免重复
    pairs_list = []
    for i in range(n_stars):
        ra_i = ra_arr[i]
        dec_i = dec_arr[i]
        for kk in range(1, k_query):  # 跳过自己(kk=0)
            j = int(idxs_nn[i, kk])
            if j <= i:
                continue  # 只保留 j > i 的星对, 避免重复
            ra_j = ra_arr[j]
            dec_j = dec_arr[j]
            d_cat = float(angular_distance_arcsec(ra_i, dec_i, ra_j, dec_j))
            if d_cat < d_min_arcsec or d_cat > d_max_arcsec:
                continue  # 过滤过近/过远的星对
            pa_cat = float(position_angle_deg(ra_i, dec_i, ra_j, dec_j))
            pairs_list.append((d_cat, pa_cat, ra_i, dec_i, ra_j, dec_j, i))

    n_pairs = len(pairs_list)
    if n_pairs == 0:
        logger.warning("星对库构建失败: 过滤后无有效星对 (d_min=%.1f\", d_max=%.1f\")",
                       d_min_arcsec, d_max_arcsec)
        return None

    # 转为structured array
    S = np.array(pairs_list, dtype=PAIR_DTYPE)

    logger.info("星对库构建完成: %d颗星 → %d星对 (K=%d, d∈[%.1f\", %.1f\"])",
                 n_stars, n_pairs, k_neighbors, d_min_arcsec, d_max_arcsec)
    return S


def build_kvector(
    pairs: np.ndarray,
    delta: float = DEFAULT_DELTA,
    d_min: float = DEFAULT_D_MIN,
) -> Optional[PairLibrary]:
    """
    从星对库构建k-vector索引。

    Args:
        pairs: 星对structured array (PAIR_DTYPE)
        delta: k-vector步长(arcsec), 默认0.5"
        d_min: d_cat下限(arcsec), 默认10"

    Returns:
        PairLibrary 或 None(空列表)
    """
    n = len(pairs)
    if n == 0:
        logger.warning("星对列表为空, 无法构建k-vector")
        return None

    # 1. 按d_cat升序排序
    sort_idx = np.argsort(pairs['d_cat'], kind='mergesort')
    S = pairs[sort_idx].copy()
    d_arr = S['d_cat'].astype(np.float64)

    # 2. 确定有效d_min/d_max
    d_max = float(d_arr[-1])
    actual_d_min = float(d_arr[0])
    effective_d_min = min(d_min, actual_d_min)

    # 3. 构建区间表 K[j] = S中 d_cat ≤ effective_d_min + (j+1)·Δ 的末下标
    n_bins = max(1, int(np.ceil((d_max - effective_d_min) / delta)) + 2)
    K = np.zeros(n_bins, dtype=np.int32)
    thresholds = effective_d_min + (np.arange(n_bins) + 1) * delta
    # searchsorted(side='right') 返回第一个 > threshold 的位置, -1 即为末下标
    indices = np.searchsorted(d_arr, thresholds, side="right") - 1
    K[:] = indices.astype(np.int32)

    logger.info("k-vector构建完成: N=%d, Δ=%.2f\", d_min=%.2f\"(effective=%.2f\"), d_max=%.2f\", n_bins=%d",
                 n, delta, d_min, effective_d_min, d_max, n_bins)

    return PairLibrary(
        S=S,
        K=K,
        d_sorted=d_arr,
        delta=float(delta),
        d_min=float(effective_d_min),
        d_max=d_max,
    )


def build_pair_library_with_kvector(
    ra_arr: np.ndarray,
    dec_arr: np.ndarray,
    mag_arr: np.ndarray,
    k_neighbors: int = DEFAULT_K_NEIGHBORS,
    delta: float = DEFAULT_DELTA,
    d_min: float = DEFAULT_D_MIN,
    d_max: float = DEFAULT_D_MAX,
) -> Optional[PairLibrary]:
    """
    便捷函数: 一步完成星对库构建 + k-vector索引构建。

    Args:
        ra_arr, dec_arr, mag_arr: 参考星坐标与星等
        k_neighbors: 每颗星取的最近邻数
        delta: k-vector步长(arcsec)
        d_min: d_cat下限(arcsec)
        d_max: d_cat上限(arcsec)

    Returns:
        PairLibrary 或 None
    """
    pairs = build_pair_library(ra_arr, dec_arr, mag_arr, k_neighbors, d_min, d_max)
    if pairs is None:
        return None
    return build_kvector(pairs, delta, d_min)


def kvector_query(
    kv: PairLibrary,
    d_query: float,
    delta_d: float,
) -> tuple[int, int]:
    """
    k-vector范围查询: 返回 d_cat ∈ [d_query - delta_d, d_query + delta_d] 的星对索引范围。

    O(1)定位 + O(k)扫描。返回的索引范围可能包含边界外的元素, 调用方需精确过滤。

    Args:
        kv: k-vector索引
        d_query: 查询的d_cat值(arcsec)
        delta_d: 容差(arcsec), 查询区间 [d_query-delta_d, d_query+delta_d]

    Returns:
        (idx_lo, idx_hi): S中的索引范围 [idx_lo, idx_hi], 若无候选返回 (0, -1)
    """
    if kv.n_pairs == 0:
        return 0, -1

    d_lo = d_query - delta_d
    d_hi = d_query + delta_d

    n_bins = len(kv.K)
    j_lo = int(np.floor((d_lo - kv.d_min) / kv.delta))
    j_hi = int(np.floor((d_hi - kv.d_min) / kv.delta))

    # clamp 到 [0, n_bins-1]
    j_lo = max(0, min(j_lo, n_bins - 1))
    j_hi = max(0, min(j_hi, n_bins - 1))

    if j_lo > j_hi:
        j_lo, j_hi = j_hi, j_lo

    # idx_lo = K[j_lo-1]+1 if j_lo>0 else 0; idx_hi = K[j_hi]
    if j_lo > 0:
        idx_lo = int(kv.K[j_lo - 1]) + 1
    else:
        idx_lo = 0
    idx_hi = int(kv.K[j_hi])

    # clamp
    idx_lo = max(0, min(idx_lo, kv.n_pairs - 1))
    idx_hi = max(idx_lo - 1, min(idx_hi, kv.n_pairs - 1))

    if idx_lo > idx_hi:
        return 0, -1

    return idx_lo, idx_hi

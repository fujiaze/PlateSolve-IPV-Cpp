"""
密度驱动的 Gaia 星等自适应选择 (Task 2)
功能: 根据图像亮星密度 ρ=N_bright/FOV² 查表选择 Gaia 截止星等 G_cutoff
用途: DD-SPPS 盲解析阶段 1, SNR∝1/ρ² 理论应用——密度越低 SNR 越高
依赖: blind_index_v2.python.io_wrappers (query_dr3, StarDetectionResult)

算法:
    1. 从图像元数据推导 FOV 对角线 (度)
    2. 按流量降序选取亮星 (饱和星优先), 计算 ρ = N_bright / FOV²
    3. 查映射表得 G_cutoff (ρ≤0.5→7.0 ... ρ>20→11.0)
    4. 以指向中心 + FOV 半对角线×1.5 余量查询 Gaia DR3
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# 复用 blind_index_v2 的 IO 接口与日志
from lib.plate_solve.blind_index_v2.python.io_wrappers import (
    query_dr3,
    StarDetectionResult,
)
from lib.plate_solve.blind_index_v2.python.logging_setup import get_logger

logger = get_logger("ddspps.density")


# 密度→星等映射表 (ρ, G_cutoff)
# Bug 10 修复: 之前的表方向反了 (密集场给更暗 G_cutoff → 更多星 → 更密集)
# 正确逻辑: 稀疏场 (低 ρ) → 更暗 G_cutoff (获取更多 Gaia 星提升信号密度)
#           密集场 (高 ρ) → 更亮 G_cutoff (减少 Gaia 星避免信号过密)
# G_cutoff 整体提高 (诊断显示 G=11.5 仅返回 125-1116 颗, 大量检测星无 Gaia 对应)
_DENSITY_MAG_TABLE: Tuple[Tuple[float, float], ...] = (
    (0.5, 15.0),   # 极稀疏 → 尽量多星
    (1.0, 14.5),
    (2.0, 14.0),
    (5.0, 13.5),
    (10.0, 13.0),
    (20.0, 12.5),
)


@dataclass
class DensityResult:
    """
    密度估计结果。

    Attributes:
        rho: 亮星密度 N_bright / FOV² (颗/度²)
        n_bright: 选取的亮星数
        bright_indices: 亮星在原始 stars 数组中的索引 (int64 数组)
        g_cutoff: Gaia 截止星等 G_cutoff
    """
    rho: float
    n_bright: int
    bright_indices: np.ndarray
    g_cutoff: float


def density_to_gmag(rho: float) -> float:
    """
    密度→Gaia 截止星等查表函数。

    映射表 (Bug 10 修复后, 稀疏场→更暗 G_cutoff):
        ρ≤0.5 → 15.0, ρ≤1.0 → 14.5, ρ≤2.0 → 14.0, ρ≤5.0 → 13.5,
        ρ≤10.0 → 13.0, ρ≤20.0 → 12.5, ρ>20.0 → 12.0

    Args:
        rho: 亮星密度 (颗/度²)

    Returns:
        G_cutoff (mag)
    """
    if rho <= 0.0:
        return 15.0
    for thresh, gmag in _DENSITY_MAG_TABLE:
        if rho <= thresh:
            return gmag
    return 12.0  # ρ > 20.0


def get_fov_from_header(metadata) -> float:
    """
    从图像元数据推导 FOV 对角线 (度)。

    优先使用 metadata.geometry.width/height; 若缺失则尝试 WCS CD 行列式 × 像素数。
    FOV_diag = sqrt((w×s0/3600)² + (h×s0/3600)²)

    Args:
        metadata: ImageMetadataPy (含 geometry 与可选 wcs/observation)

    Returns:
        FOV 对角线 (度), 无法获取时返回 0.0
    """
    w = 0
    h = 0
    # 1. geometry.width/height
    try:
        geo = metadata.geometry
        if geo is not None:
            w = int(getattr(geo, "width", 0) or 0)
            h = int(getattr(geo, "height", 0) or 0)
    except AttributeError:
        pass

    # 2. 像素尺度 s0
    s0 = None
    if metadata.wcs is not None and getattr(metadata.wcs, "has_wcs", 0):
        ps = getattr(metadata.wcs, "pixel_scale", 0.0)
        if ps and ps > 0:
            s0 = float(ps)
    if s0 is None and metadata.observation is not None:
        fl = getattr(metadata.observation, "focallen", None)
        xp = getattr(metadata.observation, "xpixsz", None)
        if fl and xp and fl > 0 and xp > 0:
            s0 = 206.265 * float(xp) / float(fl)

    if w <= 0 or h <= 0 or s0 is None or s0 <= 0:
        logger.warning("无法从元数据推导 FOV (w=%d, h=%d, s0=%s)", w, h, s0)
        return 0.0

    fov_diag = math.sqrt((w * s0 / 3600.0) ** 2 + (h * s0 / 3600.0) ** 2)
    logger.info("FOV 对角线=%.4f° (w=%d, h=%d, s0=%.4f\")", fov_diag, w, h, s0)
    return fov_diag


def estimate_density(stars: StarDetectionResult, fov_deg: float) -> DensityResult:
    """
    估计图像亮星密度并选择 Gaia 截止星等。

    选取规则:
        - 饱和星 (saturated==1) 优先, 其余按 flux 降序补足
        - ρ_target = max(2.0, 5.0 / FOV²)
        - N_bright = min(N_total, max(5, ceil(ρ_target × FOV²)))
        - ρ = N_bright / FOV²

    Args:
        stars: StarDetectionResult (x/y/flux/saturated)
        fov_deg: FOV 对角线 (度)

    Returns:
        DensityResult
    """
    n_total = int(stars.count)
    if n_total == 0 or fov_deg <= 0:
        logger.warning("密度估计输入无效: n_stars=%d, fov=%.4f", n_total, fov_deg)
        return DensityResult(rho=0.0, n_bright=0, bright_indices=np.array([], dtype=np.int64), g_cutoff=12.0)

    fov2 = fov_deg * fov_deg
    rho_target = max(2.0, 5.0 / fov2)
    target_n = int(math.ceil(rho_target * fov2))
    n_bright = min(n_total, max(5, target_n))

    # 排序: 饱和星优先, 同类按 flux 降序
    sat = np.asarray(stars.saturated, dtype=np.int32)
    flux = np.asarray(stars.flux, dtype=np.float64)
    # sort key: 饱和星 → +1e18, 然后加 flux (饱和星 flux=-1 不参与, 用大数垫底保证优先)
    # 正常星按 flux 降序, 饱和星之间按出现顺序
    sat_priority = sat.astype(np.float64) * 1e18
    # 正常星的有效 flux = flux; 饱和星 flux=-1, 用 0 填充 (已由 sat_priority 保证优先)
    eff_flux = np.where(sat == 1, 0.0, np.maximum(flux, 0.0))
    sort_key = sat_priority + eff_flux
    # 降序: 取负 argsort
    order = np.argsort(-sort_key, kind="stable")
    bright_indices = order[:n_bright].astype(np.int64)

    rho = float(n_bright) / fov2
    g_cutoff = density_to_gmag(rho)
    n_sat_used = int(np.sum(sat[bright_indices] == 1))
    logger.info("密度估计: N_total=%d, N_bright=%d (饱和=%d), FOV=%.4f°, ρ=%.4f 颗/度², G_cutoff=%.2f",
                n_total, n_bright, n_sat_used, fov_deg, rho, g_cutoff)
    return DensityResult(rho=rho, n_bright=n_bright, bright_indices=bright_indices, g_cutoff=g_cutoff)


def load_gaia_subset(
    ra_c: float,
    dec_c: float,
    fov_deg: float,
    g_cutoff: float,
    data_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    查询 Gaia DR3 局部天区参考星。

    查询半径 = FOV 对角线 / 2 × 1.5 (覆盖中心到角点 + 50% 余量, 适应翻转/旋转)。

    Args:
        ra_c: 指向中心 RA (度)
        dec_c: 指向中心 Dec (度)
        fov_deg: FOV 对角线 (度)
        g_cutoff: Gaia 截止星等
        data_dir: GaiaDR3 数据目录, 默认使用 io_wrappers 内置路径

    Returns:
        (ra, dec, mag): 三个 numpy 数组
    """
    if fov_deg <= 0:
        logger.warning("FOV 无效 (%.4f), 返回空 Gaia 子集", fov_deg)
        return np.array([]), np.array([]), np.array([])
    # 查询半径 = FOV 对角线 (覆盖完整图像区域 + 旋转余量)
    # 之前用 FOV×0.75 (半对角线×1.5) 不够覆盖边缘, 改为 FOV×1.0
    radius_deg = fov_deg * 1.0
    logger.info("加载 Gaia 子集: 中心(%.5f, %.5f), 半径=%.4f°, G_cutoff=%.2f",
                ra_c, dec_c, radius_deg, g_cutoff)
    ra, dec, mag = query_dr3(ra_c, dec_c, radius_deg, g_cutoff, data_dir=data_dir)
    logger.info("Gaia 子集加载完成: %d 颗参考星", len(ra))
    return ra, dec, mag

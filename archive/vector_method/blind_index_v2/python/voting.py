"""
k-vector查询 + (天区,旋转角)二维投票 + 峰值检测模块 (Task 4+5)
功能: 对每个图像星对(d_img, θ_img)查询k-vector候选, 计算旋转角假设rot=θ_img-PA_cat,
      投票到(HEALPix天区, 旋转角bin)二维空间, 检测top-K峰值
用途: ADV-PA盲解析的核心投票机制 — 真匹配聚集到同一(天区,rot_bin), 假匹配分散到8.86M格

投票空间:
    天区: HEALPix Nside=64 (49152格, ~0.84°分辨率) — healpy不可用时用等距网格回退
    旋转角: 180 bins (2°/bin, 覆盖0-360°)

关键参数:
    σ_pos = 0.5 pixel (星点位置噪声)
    σ_d = σ_pos × s0 (距离噪声, arcsec)
    查询容差: ±3σ_d
    rot_bin宽度: 2°
    峰值阈值: max(3, N_pairs/100)

Y翻转处理 (fix-adv-pa-phase1-bugs Bug 2):
    图像y向下(numpy数组约定), θ_img=atan2(Δy,Δx) 给出屏幕坐标系顺时针角度
    PA_cat 从北向东 (天球切平面标准约定)
    两者方向关系取决于: (a) 图像是否 Y-flip, (b) 图像星对顺序 vs catalog星对顺序
    - Y-flip: θ_img_flipped = -θ_img_normal → rot_flipped = -rot = (360-rot)%360
    - PA方向: PA_cat(I→J) 与 PA_cat(J→I) 差 180°; 星对库只存 i<j 的 PA_cat(i→j),
      但图像星对 (i,j) 顺序与 catalog 顺序可能反 → 180° 歧义
    - 4 种组合: rot, (rot+180)%360, (360-rot)%360, (360-rot+180)%360 = (180-rot)%360
    真匹配在 4 个 bin 各贡献 1 票 (1 个为真聚类, 其余 3 个分散); 假匹配也分散到 4× 噪声 bin
    SNR 仍高, 因投票空间 = 92235 × 180 = 16.6M 格, 噪声极度稀释
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .logging_setup import get_logger
from .pair_index import PairLibrary, kvector_query

logger = get_logger(__name__)

# ═══ 投票参数 ═══
DEFAULT_SIGMA_POS = 0.5           # 位置噪声(像素)
DEFAULT_N_SIGMA = 3.0             # 查询容差倍数
DEFAULT_ROT_BIN_DEG = 2.0         # 旋转角bin宽度(度)
DEFAULT_NSIDE = 64                # HEALPix Nside
DEFAULT_TOP_K = 5                 # 峰值检测top-K

# ═══ HEALPix回退网格参数 ═══
# healpy不可用时使用等距(ra,dec)网格, 分辨率近似Nside=64的0.84°
_GRID_SIZE_DEG = 0.84
_RA_BINS = int(np.ceil(360.0 / _GRID_SIZE_DEG))   # ~429
_DEC_BINS = int(np.ceil(180.0 / _GRID_SIZE_DEG))  # ~215

# 尝试导入healpy
try:
    import healpy as _hp
    _HAS_HEALPY = True
    logger.info("healpy可用 (version=%s), 使用HEALPix Nside=%d", _hp.__version__, DEFAULT_NSIDE)
except ImportError:
    _HAS_HEALPY = False
    logger.info("healpy不可用, 使用等距网格回退 (grid=%.2f°, %d×%d=%d格)",
                 _GRID_SIZE_DEG, _RA_BINS, _DEC_BINS, _RA_BINS * _DEC_BINS)


def ang2pix(ra_deg: float, dec_deg: float, nside: int = DEFAULT_NSIDE) -> int:
    """
    将(ra, dec)转换为大区pixel ID。

    优先使用healpy, 不可用时用等距(ra,dec)网格回退。

    Args:
        ra_deg: RA(度)
        dec_deg: Dec(度)
        nside: HEALPix Nside (仅healpy模式生效)

    Returns:
        pixel ID (int)
    """
    if _HAS_HEALPY:
        # healpy: lonlat=True表示(ra, dec), nest=False=RING排序
        return int(_hp.ang2pix(nside, ra_deg, dec_deg, nest=False, lonlat=True))
    # 回退: 等距网格
    ra_bin = int(ra_deg / _GRID_SIZE_DEG) % _RA_BINS
    dec_bin = int((dec_deg + 90.0) / _GRID_SIZE_DEG)
    dec_bin = max(0, min(_DEC_BINS - 1, dec_bin))
    return dec_bin * _RA_BINS + ra_bin


def pix2ang(pixel_id: int, nside: int = DEFAULT_NSIDE) -> tuple[float, float]:
    """
    将pixel ID转换回天区中心(ra, dec)。

    Args:
        pixel_id: pixel ID
        nside: HEALPix Nside (仅healpy模式生效)

    Returns:
        (ra_center, dec_center) 度
    """
    if _HAS_HEALPY:
        ra, dec = _hp.pix2ang(nside, int(pixel_id), nest=False, lonlat=True)
        return float(ra), float(dec)
    # 回退: 等距网格中心
    dec_bin = pixel_id // _RA_BINS
    ra_bin = pixel_id % _RA_BINS
    ra = (ra_bin + 0.5) * _GRID_SIZE_DEG
    dec = (dec_bin + 0.5) * _GRID_SIZE_DEG - 90.0
    return float(ra), float(dec)


def rot_bin_to_angle(rot_bin: int) -> float:
    """
    将旋转角bin转换回旋转角(度)。

    Args:
        rot_bin: 旋转角bin索引

    Returns:
        旋转角(度), = rot_bin * 2.0
    """
    return float(rot_bin) * DEFAULT_ROT_BIN_DEG


# ═══ 投票主函数 ═══

@dataclass
class VoteResult:
    """
    投票结果。

    Attributes:
        peaks: 峰值列表 [(healpix_id, rot_bin, count), ...] 按票数降序
        total_votes: 总投票数
        n_pairs: 图像星对数
        threshold: 使用的阈值
        message: 描述信息
    """
    peaks: list[tuple[int, int, int]]
    total_votes: int
    n_pairs: int
    threshold: float
    message: str


def vote(
    d_img_arr: np.ndarray,
    theta_img_arr: np.ndarray,
    kv: PairLibrary,
    s0: float,
    sigma_pos: float = DEFAULT_SIGMA_POS,
    n_sigma: float = DEFAULT_N_SIGMA,
) -> dict[tuple[int, int], int]:
    """
    对所有图像星对执行k-vector查询+(天区,rot)投票。

    对每个(d_img, θ_img):
        1. σ_d = σ_pos × s0
        2. k-vector查询 d_cat ∈ [d_img - n_sigma·σ_d, d_img + n_sigma·σ_d]
        3. 对每个候选星对: rot = (θ_img - PA_cat) % 360
        4. **4-way 投票** (fix-adv-pa-phase1-bugs Bug 2):
           - rot           (原始方向)
           - (rot+180)%360 (PA_cat(i→j) vs PA_cat(j→i) 180° 歧义)
           - (360-rot)%360 (Y-flip: θ_img 翻号)
           - (180-rot)%360 (Y-flip + 180° 歧义, = (360-(rot+180))%360)
           每个候选贡献 4 票到 4 个 (天区, rot_bin) 格
        5. 天区 = ang2pix(ra_i, dec_i)

    Args:
        d_img_arr: 图像星对绝对角距数组(arcsec)
        theta_img_arr: 图像星对方位角数组(度)
        kv: k-vector索引(PairLibrary)
        s0: 像素尺度(arcsec/pixel)
        sigma_pos: 位置噪声(像素)
        n_sigma: 查询容差倍数

    Returns:
        votes: dict {(healpix_id, rot_bin): count}
    """
    sigma_d = sigma_pos * s0
    delta_d = n_sigma * sigma_d
    votes: dict[tuple[int, int], int] = defaultdict(int)

    n_pairs = len(d_img_arr)
    total_candidates = 0

    # 4 个 rot 偏移 (fix-adv-pa-phase1-bugs Bug 2):
    # 0=原始, 180=PA方向歧义, 360-rot=Y-flip, 540-rot=Y-flip+PA歧义
    # 实际计算: 对每个 rot, 投票 rot, (rot+180)%360, (-rot)%360, (180-rot)%360
    _ROT_BIN_MAX = int(round(360.0 / DEFAULT_ROT_BIN_DEG))  # 180

    for i in range(n_pairs):
        d_img = float(d_img_arr[i])
        theta_img = float(theta_img_arr[i])

        # k-vector范围查询
        idx_lo, idx_hi = kvector_query(kv, d_img, delta_d)
        if idx_lo > idx_hi:
            continue

        # 批量处理候选星对
        candidates = kv.S[idx_lo:idx_hi + 1]
        n_cand = len(candidates)
        total_candidates += n_cand
        if n_cand == 0:
            continue

        # 向量化计算 4 个 rot 值
        pa_cat = candidates['PA_cat']
        rot = np.mod(theta_img - pa_cat, 360.0)
        # 4-way: rot, (rot+180)%360, (360-rot)%360, (180-rot)%360
        rot1 = rot
        rot2 = np.mod(rot + 180.0, 360.0)
        rot3 = np.mod(-rot, 360.0)            # = (360-rot)%360
        rot4 = np.mod(180.0 - rot, 360.0)     # = (360-(rot+180))%360

        # 4 个 rot_bin 数组
        rot_bin1 = (rot1 / DEFAULT_ROT_BIN_DEG).astype(np.int32) % _ROT_BIN_MAX
        rot_bin2 = (rot2 / DEFAULT_ROT_BIN_DEG).astype(np.int32) % _ROT_BIN_MAX
        rot_bin3 = (rot3 / DEFAULT_ROT_BIN_DEG).astype(np.int32) % _ROT_BIN_MAX
        rot_bin4 = (rot4 / DEFAULT_ROT_BIN_DEG).astype(np.int32) % _ROT_BIN_MAX

        # 计算天区pixel ID 并投票 4 次
        ra_i = candidates['ra_i']
        dec_i = candidates['dec_i']
        for k in range(n_cand):
            pix = ang2pix(float(ra_i[k]), float(dec_i[k]))
            votes[(pix, int(rot_bin1[k]))] += 1
            votes[(pix, int(rot_bin2[k]))] += 1
            votes[(pix, int(rot_bin3[k]))] += 1
            votes[(pix, int(rot_bin4[k]))] += 1

    total_votes = sum(votes.values())
    logger.info("投票完成 (4-way): %d图像星对, 总候选%d, 投票格%d, 总票数%d (4×候选数=%d)",
                 n_pairs, total_candidates, len(votes), total_votes, 4 * total_candidates)
    return dict(votes)


def detect_peaks(
    votes: dict[tuple[int, int], int],
    n_pairs: int,
    top_k: int = DEFAULT_TOP_K,
    threshold: Optional[float] = None,
) -> list[tuple[int, int, int]]:
    """
    从投票直方图检测峰值。

    取top-K个(天区, rot_bin), 票数 ≥ threshold。
    threshold默认基于噪声底线: max(3, 10×noise_floor), 其中
    noise_floor = total_votes / n_cells (每格期望票数)。
    这比固定 n_pairs/100 更合理 — 后者在 Top-N 较大时(如 N=100→4950对)
    阈值过高(49.5), 会误杀真信号(SNR可达10^4但仍低于阈值)。

    Args:
        votes: 投票字典 {(healpix_id, rot_bin): count}
        n_pairs: 图像星对数(保留接口, 不再直接用于阈值)
        top_k: 返回前K个峰值
        threshold: 手动指定阈值, None则自动计算(噪声底线法)

    Returns:
        峰值列表 [(healpix_id, rot_bin, count), ...] 按票数降序
    """
    if threshold is None:
        # 噪声底线法: total_votes / n_cells = 每格期望票数
        total_votes_local = sum(votes.values())
        if _HAS_HEALPY:
            n_sky = 12 * DEFAULT_NSIDE * DEFAULT_NSIDE  # 49152
        else:
            n_sky = _RA_BINS * _DEC_BINS  # 92235 (grid回退)
        n_rot = int(round(360.0 / DEFAULT_ROT_BIN_DEG))  # 180
        n_cells = n_sky * n_rot
        noise_floor = (total_votes_local / n_cells) if n_cells > 0 else 0.0
        # 阈值: 至少3票, 且至少10倍噪声底线 (确保非噪声)
        threshold = max(3.0, 10.0 * noise_floor)

    if not votes:
        logger.warning("投票字典为空, 无峰值")
        return []

    # 按票数降序排序
    sorted_votes = sorted(votes.items(), key=lambda x: -x[1])
    peaks = [
        (pix, rot_bin, count)
        for (pix, rot_bin), count in sorted_votes
        if count >= threshold
    ]

    # 取top-K
    peaks = peaks[:top_k]

    if peaks:
        logger.info("峰值检测: 阈值=%.1f, 检测到%d个峰值 (top-K=%d), 最高票数=%d",
                     threshold, len(peaks), top_k, peaks[0][2])
    else:
        logger.warning("峰值检测: 阈值=%.1f, 无峰值超过阈值 (最高票数=%d)",
                         threshold, sorted_votes[0][1] if sorted_votes else 0)

    return peaks


def run_voting(
    d_img_arr: np.ndarray,
    theta_img_arr: np.ndarray,
    kv: PairLibrary,
    s0: float,
    sigma_pos: float = DEFAULT_SIGMA_POS,
    n_sigma: float = DEFAULT_N_SIGMA,
    top_k: int = DEFAULT_TOP_K,
) -> VoteResult:
    """
    便捷函数: 执行投票 + 峰值检测, 返回VoteResult。

    Args:
        d_img_arr, theta_img_arr: 图像星对特征
        kv: k-vector索引
        s0: 像素尺度
        sigma_pos: 位置噪声(像素)
        n_sigma: 查询容差倍数
        top_k: 峰值检测top-K

    Returns:
        VoteResult
    """
    n_pairs = len(d_img_arr)
    votes = vote(d_img_arr, theta_img_arr, kv, s0, sigma_pos, n_sigma)
    total_votes = sum(votes.values())
    # 噪声底线法阈值 (与 detect_peaks 内部一致, 用于 VoteResult 记录)
    if _HAS_HEALPY:
        _n_sky = 12 * DEFAULT_NSIDE * DEFAULT_NSIDE
    else:
        _n_sky = _RA_BINS * _DEC_BINS
    _n_cells = _n_sky * int(round(360.0 / DEFAULT_ROT_BIN_DEG))
    _noise = (total_votes / _n_cells) if _n_cells > 0 else 0.0
    threshold = max(3.0, 10.0 * _noise)
    peaks = detect_peaks(votes, n_pairs, top_k, threshold=None)

    if not peaks:
        return VoteResult(
            peaks=[], total_votes=total_votes, n_pairs=n_pairs,
            threshold=threshold, message="无峰值超过阈值"
        )

    return VoteResult(
        peaks=peaks, total_votes=total_votes, n_pairs=n_pairs,
        threshold=threshold,
        message=f"检测到{len(peaks)}个峰值, 最高票数={peaks[0][2]}"
    )

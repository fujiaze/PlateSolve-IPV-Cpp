"""
频率域相位相关求解 (Task 4+5)
功能: 1D 相位相关估计旋转角 + 2D 相位相关估计平移 + 亚像素精化
用途: DD-SPPS 阶段 3 核心, FFT 加速的频域互相关
依赖: numpy.fft, scipy.ndimage (rotate)
理论:
    - Kuglin & Hines 1975 归一化互功率谱: R = F_f·conj(F_g) / (|F_f|·|F_g|)
    - Foroosh 2002 抛物线亚像素拟合: 3×3 邻域二次插值
    - 1D 角度签名: |FFT2(f)| 极坐标角向求和 → 旋转不变特征
    - Hann 窗: 消除方形网格边界引入的 4 重对称频谱泄漏, 使非 90° 旋转可恢复

符号约定 (重要):
    phase_correlate_1d(phi_f, phi_g) 返回 +shift 当 phi_f = roll(phi_g, shift)
    即: 图像 f 是 Gaia 模板 g 旋转 shift 后的结果时, 返回 +shift (正确旋转角)
    物理对应: 图像相对天空旋转 θ → phi_f = roll(phi_g, shift_θ) → 返回 +θ

    phase_correlate_2d(F_f, g_rot) 返回 +(dx, dy) 当 f = roll(g_rot, (dy, dx))
    即: 图像 f 是 Gaia 模板 g_rot 平移 (dy, dx) 后的结果时, 返回 +(dx, dy)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
from scipy.ndimage import rotate as _nd_rotate

from lib.plate_solve.blind_index_v2.python.logging_setup import get_logger

logger = get_logger("ddspps.phase_correlation")

_EPS = 1e-10


def hann_window(grid: int) -> np.ndarray:
    """
    生成 2D Hann 窗 (可分离: w(x,y) = w(x)·w(y))。

    Hann 窗在边界平滑衰减到 0, 消除方形网格的 4 重对称频谱泄漏,
    使 FFT 幅度谱的角度签名能正确反映非 90° 旋转。

    Args:
        grid: 网格尺寸

    Returns:
        (grid, grid) 2D Hann 窗数组
    """
    w1d = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(grid) / float(grid)))
    return np.outer(w1d, w1d)


def windowed_fft2(f: np.ndarray) -> np.ndarray:
    """
    对信号应用 Hann 窗后计算 FFT2。

    用于 1D 相位相关旋转估计 (angular_projection 输入)。
    注意: 2D 相位相关平移估计不应加窗 (会破坏平移不变性)。

    Args:
        f: (grid, grid) 2D 信号

    Returns:
        F: (grid, grid) 复数 FFT
    """
    w = hann_window(f.shape[0])
    return np.fft.fft2(f * w)


def angular_projection(M: np.ndarray, n_bins: int = 720) -> np.ndarray:
    """
    将 2D 频谱幅度 |FFT2(f)| 投影为 1D 角度签名。

    将频域 (kx, ky) 转极坐标 (rho, phi), 对 phi 分 n_bins 个 bin 做角向求和。
    旋转图像只会循环平移角度签名, 因此可用 1D 相位相关估计旋转角。

    Args:
        M: 2D 频谱幅度数组 (grid, grid), 通常 = |FFT2(f)|
        n_bins: 角度 bin 数 (默认 720, 0.5°/bin)

    Returns:
        phi_profile: (n_bins,) 角度签名, phi ∈ [0, 2π) → bin ∈ [0, n_bins)
    """
    M = np.asarray(M, dtype=np.float64)
    grid = M.shape[0]
    # 频率轴 (numpy fft 布局: [0,1,...,n/2-1,-n/2,...,-1]/n)
    freq = np.fft.fftfreq(grid)
    # meshgrid: kx 沿列 (x方向), ky 沿行 (y方向), 与 FFT2 输出 F[u,v] 一致
    kx, ky = np.meshgrid(freq, freq)
    # 极坐标角 (弧度), 映射到 [0, 2π)
    phi = np.arctan2(ky, kx) % (2.0 * np.pi)
    bin_idx = (phi / (2.0 * np.pi) * n_bins).astype(np.int64) % n_bins
    # 排除 DC 分量 (kx=ky=0): 其幅度极大且 phi 无定义, 会污染 bin 0
    dc_mask = (np.abs(kx) < 1e-15) & (np.abs(ky) < 1e-15)
    weights = M.ravel()
    mask = ~dc_mask.ravel()
    phi_profile = np.bincount(
        bin_idx.ravel()[mask],
        weights=weights[mask],
        minlength=n_bins,
    ).astype(np.float64)
    logger.debug("角度签名: n_bins=%d, 峰值=%.4f, 总和=%.4f",
                 n_bins, float(phi_profile.max()), float(phi_profile.sum()))
    return phi_profile


def phase_correlate_1d(
    phi_f: np.ndarray,
    phi_g: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """
    1D 相位相关估计旋转角。

    归一化互功率谱: R = (F_f·conj(F_g)) / (|F_f|·|F_g| + ε)
    r = |IFFT1D(R)|, 峰值位置 → 旋转角。

    符号约定 (已验证, 无 bug):
        当 phi_f = roll(phi_g, shift) 时, 返回 theta_cand = +shift
        物理含义: 图像 f 是 Gaia 模板 g 旋转 shift 后的结果 → 返回正确旋转角 +shift

        当 phi_g = roll(phi_f, shift) 时, 返回 theta_cand = -shift (即 360-shift)
        物理含义: Gaia 模板 g 是图像 f 旋转 shift 后的结果 → 返回 -shift

        这符合互相关定义: R = F_f · conj(F_g) 找的是 f 相对 g 的平移
        (即 f = roll(g, shift) 时峰值为 +shift)

    注意: 实信号 FFT 幅度谱有 180° 共轭对称 (|F(u,v)|=|F(-u,-v)|),
          所以 theta_cand 有 180° 歧义 (θ 和 θ+180° 等价)。
          调用方需用 2D 相位相关验证确定正确象限。

    Args:
        phi_f: (n_bins,) 图像角度签名
        phi_g: (n_bins,) Gaia 角度签名

    Returns:
        (theta_cand, snr, r):
            theta_cand: 候选旋转角 (度), ∈ [0, 360)
            snr: 峰值信噪比 = max(r) / (median(r) + ε)
            r: 互相关数组 (n_bins,)
    """
    phi_f = np.asarray(phi_f, dtype=np.float64)
    phi_g = np.asarray(phi_g, dtype=np.float64)
    n = len(phi_f)
    if n == 0 or len(phi_g) != n:
        logger.warning("1D 相位相关: 输入长度不匹配 (%d vs %d)", n, len(phi_g))
        return 0.0, 0.0, np.array([])

    F_f = np.fft.fft(phi_f)
    F_g = np.fft.fft(phi_g)
    denom = (np.abs(F_f) * np.abs(F_g)) + _EPS
    R = (F_f * np.conj(F_g)) / denom
    r = np.abs(np.fft.ifft(R))
    theta_idx = int(np.argmax(r))
    theta_cand = theta_idx * 360.0 / float(n)
    snr = float(r[theta_idx]) / (float(np.median(r)) + _EPS)
    logger.info("1D 相位相关: theta_cand=%.3f°, SNR=%.3f, peak=%.4f",
                theta_cand, snr, float(r[theta_idx]))
    return theta_cand, snr, r


def rotate_signal(g: np.ndarray, theta_deg: float) -> np.ndarray:
    """
    旋转 2D 信号 (双线性插值)。

    用 scipy.ndimage.rotate 旋转 Gaia 模板以匹配图像。
    reshape=False 保持尺寸, order=1 双线性, mode='constant' 边界填零。

    Args:
        g: (grid, grid) 2D 信号
        theta_deg: 旋转角 (度, 正值逆时针)

    Returns:
        g_rot: (grid, grid) 旋转后信号
    """
    g_rot = _nd_rotate(
        g, theta_deg, reshape=False, order=1, mode="constant", cval=0.0
    )
    return g_rot


def phase_correlate_2d(
    F_f: np.ndarray,
    g_rot: np.ndarray,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    2D 相位相关估计平移。

    归一化互功率谱: R = (F_f·conj(F_g_rot)) / (|F_f|·|F_g_rot| + ε)
    C = |IFFT2(R)|, 峰值位置 → (dy, dx) 平移。

    Args:
        F_f: 图像信号 FFT2 (预先计算, 避免重复)
        g_rot: 旋转后的 Gaia 信号 (时域)

    Returns:
        (dx, dy, peak_val, snr, C):
            dx: x 方向平移 (像素, 网格坐标)
            dy: y 方向平移 (像素, 网格坐标)
            peak_val: 互相关峰值
            snr: 峰值信噪比 = peak / (median(C) + ε)
            C: 互相关矩阵 (grid, grid)
    """
    g_rot = np.asarray(g_rot, dtype=np.float64)
    grid = g_rot.shape[0]
    F_g_rot = np.fft.fft2(g_rot)
    denom = (np.abs(F_f) * np.abs(F_g_rot)) + _EPS
    R = (F_f * np.conj(F_g_rot)) / denom
    C = np.abs(np.fft.ifft2(R))

    # 峰值位置 [row, col] = [y, x]
    iy, ix = np.unravel_index(int(np.argmax(C)), C.shape)
    dy = float(iy)
    dx = float(ix)
    peak_val = float(C[iy, ix])

    # 处理负位移: FFT 循环平移, 后半段表示负位移
    if dx > grid / 2.0:
        dx -= grid
    if dy > grid / 2.0:
        dy -= grid

    snr = peak_val / (float(np.median(C)) + _EPS)
    logger.info("2D 相位相关: dx=%.3f, dy=%.3f, peak=%.4f, SNR=%.3f",
                dx, dy, peak_val, snr)
    return dx, dy, peak_val, snr, C


def subpixel_refine(
    C: np.ndarray,
    dx: float,
    dy: float,
) -> Tuple[float, float]:
    """
    Foroosh 2002 抛物线亚像素精化。

    在互相关峰值 3×3 邻域做二次抛物线拟合, 提升位移精度到亚像素级。
    公式:
        dx_sub = dx + 0.5·(C[dy,dx-1] - C[dy,dx+1]) / (C[dy,dx-1] + C[dy,dx+1] - 2·C[dy,dx])
        dy_sub = dy + 0.5·(C[dy-1,dx] - C[dy+1,dx]) / (C[dy-1,dx] + C[dy+1,dx] - 2·C[dy,dx])

    注意: C 索引为 [y, x], dx 方向变化 C[dy, dx±1], dy 方向变化 C[dy±1, dx]。

    Args:
        C: 互相关矩阵 (grid, grid)
        dx: 整像素 x 平移 (网格坐标)
        dy: 整像素 y 平移 (网格坐标)

    Returns:
        (dx_sub, dy_sub): 亚像素精化后的平移
    """
    grid = C.shape[0]
    ix = int(round(dx)) % grid
    iy = int(round(dy)) % grid
    # 边界检查: 若在边界 (0 或 grid-1) 跳过精化
    if ix <= 0 or ix >= grid - 1 or iy <= 0 or iy >= grid - 1:
        logger.debug("亚像素精化: 峰值在边界 (%d,%d), 跳过", ix, iy)
        return float(dx), float(dy)

    c0 = C[iy, ix]
    cxm = C[iy, ix - 1]
    cxp = C[iy, ix + 1]
    cym = C[iy - 1, ix]
    cyp = C[iy + 1, ix]

    denom_x = (cxm + cxp - 2.0 * c0)
    denom_y = (cym + cyp - 2.0 * c0)
    dx_sub = float(dx)
    dy_sub = float(dy)
    if abs(denom_x) > _EPS:
        dx_sub = float(dx) + 0.5 * (cxm - cxp) / denom_x
    if abs(denom_y) > _EPS:
        dy_sub = float(dy) + 0.5 * (cym - cyp) / denom_y

    logger.debug("亚像素精化: dx %.3f→%.3f, dy %.3f→%.3f",
                 dx, dx_sub, dy, dy_sub)
    return dx_sub, dy_sub


def refine_rotation(
    F_f: np.ndarray,
    g: np.ndarray,
    theta_cand: float,
    search_range: float = 2.0,
    step: float = 0.5,
) -> Tuple[float, float, float, float, np.ndarray]:
    """
    旋转精化: 在 θ_cand 附近搜索最佳旋转角 (局部搜索)。

    在 [theta_cand - search_range, theta_cand + search_range] 范围, 步长 step,
    对每个 θ: g_rot = rotate_signal(g, θ), phase_correlate_2d(F_f, g_rot)。
    选 peak_val 最高的点, 再做 subpixel_refine。

    符号约定: theta_cand 是图像相对天空的旋转角 (f = rotate(g, θ))。
    要把 Gaia 模板 g 匹配图像 f, 需同向旋转 g_rot = rotate(g, θ)。

    Args:
        F_f: 图像信号 FFT2 (预先计算)
        g: Gaia 信号 (时域)
        theta_cand: 1D 相位相关候选角 (度)
        search_range: 搜索范围 (度, 默认 2.0)
        step: 搜索步长 (度, 默认 0.5)

    Returns:
        (theta_best, dx_sub, dy_sub, peak_snr, C_best):
            theta_best: 最佳旋转角 (度)
            dx_sub, dy_sub: 亚像素平移
            peak_snr: 最佳点的 SNR
            C_best: 最佳点的互相关矩阵
    """
    grid = g.shape[0]
    # 生成候选角度列表 (含 theta_cand 本身)
    angles = []
    n_steps = int(math.ceil(search_range / step)) if step > 0 else 0
    for k in range(-n_steps, n_steps + 1):
        a = theta_cand + k * step
        angles.append(a)
    # 去重 (浮点近邻)
    angles = sorted(set(round(a, 6) for a in angles))

    best_theta = float(theta_cand)
    best_peak = -1.0
    best_dx = 0.0
    best_dy = 0.0
    best_snr = 0.0
    best_C = np.zeros_like(g)

    for theta in angles:
        # 旋转 Gaia 模板以匹配图像 (同向: theta 是图像相对天空的旋转角)
        g_rot = rotate_signal(g, theta)
        dx, dy, peak_val, snr, C = phase_correlate_2d(F_f, g_rot)
        if peak_val > best_peak:
            best_peak = peak_val
            best_theta = float(theta)
            best_dx = dx
            best_dy = dy
            best_snr = snr
            best_C = C

    # 亚像素精化
    dx_sub, dy_sub = subpixel_refine(best_C, best_dx, best_dy)
    logger.info("旋转精化: theta_best=%.3f° (cand=%.3f°), dx=%.3f, dy=%.3f, peak=%.4f, SNR=%.3f",
                best_theta, theta_cand, dx_sub, dy_sub, best_peak, best_snr)
    return best_theta, dx_sub, dy_sub, best_snr, best_C


def search_rotation_full(
    f: np.ndarray,
    F_f: np.ndarray,
    g: np.ndarray,
    coarse_step: float = 5.0,
    fine_step: float = 0.5,
    fine_range: float = 5.0,
    top_k: int = 4,
) -> List[Tuple[float, float, float, float]]:
    """
    全范围旋转搜索: 粗搜索 0~360° + 精细搜索最优角度附近, 返回 top-K 候选角度。

    评分 = 2D phase correlation 峰值 (实信号 FFT 幅度谱的 180° 共轭对称使 θ 和 θ+180°
    峰值几乎相同, top-K 通常包含这两个等价解, 由调用方做 WCS 验证选优以区分 180° 歧义)。

    候选角度是数据驱动的 (从 0~360° 全范围 peak 评分排序得来), 不是固定角度,
    能求解任意角度 (用户要求: 角度可以直接求解, 0~360° 全范围)。

    流程:
        1. 粗搜索: 0~360° 步长 coarse_step, 记录所有 (θ, peak, dx, dy, C)。
        2. 按 peak 降序排序, 取 top-K 粗峰, 对每个做 ±fine_range 精细搜索。
        3. 汇总所有精细搜索局部最优, 去重 (角度相差 < 10° 视为同一, 保留 peak 高的)。
        4. 亚像素精化每个候选的 (dx, dy)。
        5. 返回 top-K 候选列表 [(theta, dx_sub, dy_sub, peak), ...]。

    Args:
        f: 图像信号 (时域, 兼容签名保留)
        F_f: 图像信号 FFT2 (预先计算, 避免重复)
        g: Gaia 信号 (时域)
        coarse_step: 粗搜索步长 (度, 默认 5°)
        fine_step: 精细搜索步长 (度, 默认 0.5°)
        fine_range: 精细搜索范围 (度, 默认 ±5°)
        top_k: 返回的候选数 (默认 4, 包含 θ 和 θ+180° 两个等价解)

    Returns:
        candidates: [(theta, dx_sub, dy_sub, peak), ...] 按 peak 降序, 长度 ≤ top_k
    """
    # 1. 粗搜索: 0~360° step=coarse_step, 收集所有 (θ, peak, dx, dy, C)
    coarse_angles = np.arange(0.0, 360.0, coarse_step)
    coarse_candidates = []  # list of (theta, peak, dx, dy, C)
    for theta in coarse_angles:
        g_rot = rotate_signal(g, theta)
        dx, dy, peak, snr, C = phase_correlate_2d(F_f, g_rot)
        coarse_candidates.append((float(theta), float(peak), float(dx), float(dy), C))

    # 按 peak 降序排序, 取 top-K 粗峰做精细搜索
    coarse_candidates.sort(key=lambda x: x[1], reverse=True)
    top_coarse = coarse_candidates[:top_k]
    logger.info("粗搜索 top-%d: %s", len(top_coarse),
                ", ".join("θ=%.1f°/peak=%.2f" % (c[0], c[1]) for c in top_coarse))

    # 2. 对每个 top 粗峰做精细搜索 ±fine_range, step=fine_step, 找局部最优
    refined_candidates = []  # list of (theta, peak, dx, dy, C)
    for theta_c, _, _, _, _ in top_coarse:
        fine_angles = np.arange(
            theta_c - fine_range,
            theta_c + fine_range + fine_step / 2.0,
            fine_step,
        )
        local_best = (float(theta_c), -1.0, 0.0, 0.0, None)
        for theta in fine_angles:
            g_rot = rotate_signal(g, theta)
            dx, dy, peak, snr, C = phase_correlate_2d(F_f, g_rot)
            if peak > local_best[1]:
                local_best = (float(theta), float(peak), float(dx), float(dy), C)
        refined_candidates.append(local_best)

    # 3. 汇总 + 去重 (角度相差 < 10° 视为同一, 保留 peak 高的)
    refined_candidates.sort(key=lambda x: x[1], reverse=True)
    deduped = []
    for cand in refined_candidates:
        theta, peak, dx, dy, C = cand
        is_dup = False
        for kept in deduped:
            d_theta = abs(theta - kept[0])
            d_theta = min(d_theta, 360.0 - d_theta)  # 考虑 0/360 边界
            if d_theta < 10.0:
                is_dup = True
                break
        if not is_dup:
            deduped.append(cand)
        if len(deduped) >= top_k:
            break

    # 4. 亚像素精化每个候选的 (dx, dy)
    final_candidates: List[Tuple[float, float, float, float]] = []
    for theta, peak, dx, dy, C in deduped:
        if C is not None:
            dx_sub, dy_sub = subpixel_refine(C, dx, dy)
        else:
            dx_sub, dy_sub = dx, dy
        final_candidates.append((theta, float(dx_sub), float(dy_sub), peak))
        logger.info("候选: θ=%.3f°, dx=%.3f, dy=%.3f, peak=%.4f",
                    theta, dx_sub, dy_sub, peak)

    return final_candidates

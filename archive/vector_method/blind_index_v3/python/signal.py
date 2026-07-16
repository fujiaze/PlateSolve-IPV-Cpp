"""
星场信号化——高斯核 2D 信号 (Task 3)
功能: 将星点坐标映射到 512×512 网格, 用等权高斯核(σ=4.5px)累加生成 2D 信号
用途: DD-SPPS 阶段 2, 星场→信号→FFT→相位相关
依赖: numpy, lib.plate_solve.python.vector_match_v2 (gnomonic_forward)

算法:
    - 图像信号: 星点 (x,y) 缩放到网格 → 高斯核累加
    - Gaia 信号: gnomonic 投影到切平面 (xi,eta arcsec) → 转网格像素 → 高斯核累加
    - flip_mode: 0=不翻, 1=x翻, 2=y翻, 3=双翻 (4 种模式独立求解)
    - 优化: 每颗星只更新 3σ 范围内像素 (局部向量化)
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from lib.plate_solve.blind_index_v2.python.logging_setup import get_logger

# 复用 vector_match_v2 的 gnomonic 投影 (接口兼容: ra,dec,ra0,dec0 → xi_arcsec,eta_arcsec,valid)
from lib.plate_solve.python.vector_match_v2 import gnomonic_forward

logger = get_logger("ddspps.signal")


def _add_gaussian_kernel(
    signal: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
) -> None:
    """
    在 signal 的 (cx, cy) 处累加一个 2D 高斯核 (in-place)。

    只更新 3σ 范围内像素。

    Args:
        signal: 2D 数组 (grid, grid), in-place 修改
        cx: 核中心 x (浮点像素坐标, 列方向)
        cy: 核中心 y (浮点像素坐标, 行方向)
        sigma: 高斯标准差 (像素)
    """
    grid = signal.shape[0]
    radius = int(math.ceil(3.0 * sigma))
    x0 = int(math.floor(cx - radius))
    x1 = int(math.floor(cx + radius)) + 1
    y0 = int(math.floor(cy - radius))
    y1 = int(math.floor(cy + radius)) + 1
    # 裁剪到有效范围
    if x1 <= 0 or y1 <= 0 or x0 >= grid or y0 >= grid:
        return
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(grid, x1)
    y1 = min(grid, y1)
    if x1 <= x0 or y1 <= y0:
        return

    # 局部坐标网格
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    kernel = np.exp(-r2 / (2.0 * sigma * sigma))
    signal[y0:y1, x0:x1] += kernel


def build_image_signal(
    stars_x: np.ndarray,
    stars_y: np.ndarray,
    image_w: float,
    image_h: float,
    grid: int = 512,
    sigma: float = 4.5,
) -> np.ndarray:
    """
    构建图像星场 2D 高斯核信号。

    坐标缩放: 用统一 scale = grid / max(image_w, image_h) (各向同性, 避免非方形图像变形)
    居中: 图像中心 (w/2, h/2) 映射到 (grid/2, grid/2), 与 build_gaia_signal 的切点对齐
    每颗星在 (x_scaled, y_scaled) 处累加高斯核 exp(-r²/2σ²)。

    Args:
        stars_x: (N,) 星点 x 像素坐标 (原图像坐标系)
        stars_y: (N,) 星点 y 像素坐标
        image_w: 原图像宽度 (像素)
        image_h: 原图像高度 (像素)
        grid: 信号网格尺寸 (默认 512)
        sigma: 高斯核标准差 (像素, 默认 4.5)

    Returns:
        f: (grid, grid) 2D 信号数组
    """
    stars_x = np.asarray(stars_x, dtype=np.float64)
    stars_y = np.asarray(stars_y, dtype=np.float64)
    n = len(stars_x)
    signal = np.zeros((grid, grid), dtype=np.float64)
    if n == 0 or image_w <= 0 or image_h <= 0:
        logger.warning("图像信号构建: 空输入 (n=%d, w=%s, h=%s)", n, image_w, image_h)
        return signal

    # 各向同性缩放: 用 max(image_w, image_h) 统一缩放, 保持旋转不变性
    image_size = max(float(image_w), float(image_h))
    scale = float(grid) / image_size
    # 居中偏移: 让图像中心 (w/2, h/2) 落在 (grid/2, grid/2), 与 Gaia 信号切点对齐
    offset_x = (grid - image_w * scale) / 2.0
    offset_y = (grid - image_h * scale) / 2.0
    sx = stars_x * scale + offset_x
    sy = stars_y * scale + offset_y

    for i in range(n):
        _add_gaussian_kernel(signal, float(sx[i]), float(sy[i]), sigma)

    logger.info("图像信号构建完成: n_stars=%d, grid=%d, sigma=%.2f, 峰值=%.4f, 总和=%.4f, "
                "offset=(%.2f, %.2f)",
                n, grid, sigma, float(signal.max()), float(signal.sum()), offset_x, offset_y)
    return signal


def gnomonic_project(
    ra: np.ndarray,
    dec: np.ndarray,
    ra_c: float,
    dec_c: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    标准 gnomonic 切平面投影 (ra,dec → xi,eta arcsec)。

    复用 lib.plate_solve.python.vector_match_v2.gnomonic_forward (接口完全兼容)。
    公式:
        cos_c = sin(dec_c)·sin(dec) + cos(dec_c)·cos(dec)·cos(ra-ra_c)
        xi  = cos(dec)·sin(ra-ra_c) / cos_c  (弧度 → 角秒)
        eta = (cos(dec_c)·sin(dec) - sin(dec_c)·cos(dec)·cos(ra-ra_c)) / cos_c
    无效点 (cos_c ≤ 0, 远离切点) 被过滤。

    Args:
        ra: RA (度), 标量或数组
        dec: Dec (度)
        ra_c: 切点 RA (度)
        dec_c: 切点 Dec (度)

    Returns:
        (xi_arcsec, eta_arcsec): 仅含有效点的切平面坐标 (角秒)
    """
    xi, eta, valid = gnomonic_forward(ra, dec, ra_c, dec_c)
    valid = np.asarray(valid, dtype=bool)
    if not np.all(valid):
        logger.debug("gnomonic 投影: 过滤 %d 个无效点 (共 %d)",
                     int(np.sum(~valid)), len(valid))
        xi = xi[valid]
        eta = eta[valid]
    return xi, eta


def build_gaia_signal(
    gaia_ra: np.ndarray,
    gaia_dec: np.ndarray,
    ra_c: float,
    dec_c: float,
    s0: float,
    image_w: float,
    image_h: float,
    grid: int = 512,
    sigma: float = 4.5,
    flip_mode: int = 0,
) -> np.ndarray:
    """
    构建 Gaia 星表 2D 高斯核信号 (模板)。

    流程:
        1. gnomonic 投影 Gaia 星到切平面 (xi, eta) arcsec
        2. 转网格像素: 用网格等效尺度 s0_grid = s0 × max(image_w,image_h) / grid
           base_x = xi/s0_grid + grid/2, base_y = eta/s0_grid + grid/2
           (确保图像信号与 Gaia 信号覆盖相同的切平面区域)
        3. flip_mode 应用 (图像 Y 朝下 / Dec 朝上, 由 flip_mode 显式控制):
            - mode 0: 不翻转
            - mode 1: x 翻转 → px_x = grid - base_x
            - mode 2: y 翻转 → px_y = grid - base_y (标准图像常用)
            - mode 3: 双翻转
        4. 相同 sigma 累加高斯核

    Args:
        gaia_ra: (M,) Gaia 星 RA (度)
        gaia_dec: (M,) Gaia 星 Dec (度)
        ra_c: 切点 RA (度)
        dec_c: 切点 Dec (度)
        s0: 原始像素尺度 (arcsec/pixel)
        image_w: 原图像宽度 (像素, 用于推导网格等效尺度)
        image_h: 原图像高度 (像素)
        grid: 信号网格尺寸 (默认 512)
        sigma: 高斯核标准差 (像素, 默认 4.5)
        flip_mode: 翻转模式 0/1/2/3

    Returns:
        g: (grid, grid) 2D 信号数组
    """
    gaia_ra = np.asarray(gaia_ra, dtype=np.float64)
    gaia_dec = np.asarray(gaia_dec, dtype=np.float64)
    signal = np.zeros((grid, grid), dtype=np.float64)
    n_in = len(gaia_ra)
    if n_in == 0 or s0 <= 0 or image_w <= 0 or image_h <= 0:
        logger.warning("Gaia 信号构建: 空输入 (n=%d, s0=%s, w=%s, h=%s)",
                       n_in, s0, image_w, image_h)
        return signal

    # 1. gnomonic 投影
    xi, eta = gnomonic_project(gaia_ra, gaia_dec, ra_c, dec_c)
    n = len(xi)
    if n == 0:
        logger.warning("Gaia 信号构建: 投影后无有效点")
        return signal

    # 2. 转网格像素 (基础坐标)
    # 网格像素等效尺度: s0_grid = s0 × max(image_w, image_h) / grid
    # 图像信号用 scale = grid / max(image_w, image_h), 覆盖区域 = max(image_w,image_h) × s0 arcsec
    # Gaia 信号需覆盖相同区域, 故 s0_grid = s0 / (grid/max(image_w,image_h)) = s0 × max / grid
    image_size = max(float(image_w), float(image_h))
    s0_grid = s0 * image_size / float(grid)
    base_x = xi / s0_grid + grid / 2.0
    base_y = eta / s0_grid + grid / 2.0

    # 3. flip_mode 应用
    flip_x = bool(flip_mode & 1)
    flip_y = bool(flip_mode & 2)
    if flip_x:
        px_x = float(grid) - base_x
    else:
        px_x = base_x
    if flip_y:
        px_y = float(grid) - base_y
    else:
        px_y = base_y

    # 4. 高斯核累加
    for i in range(n):
        _add_gaussian_kernel(signal, float(px_x[i]), float(px_y[i]), sigma)

    logger.info("Gaia 信号构建完成: n_in=%d, n_valid=%d, flip_mode=%d (x=%s,y=%s), "
                "grid=%d, sigma=%.2f, s0=%.4f\", s0_grid=%.4f\"/grid_px, 峰值=%.4f, 总和=%.4f",
                n_in, n, flip_mode, flip_x, flip_y, grid, sigma,
                s0, s0_grid, float(signal.max()), float(signal.sum()))
    return signal

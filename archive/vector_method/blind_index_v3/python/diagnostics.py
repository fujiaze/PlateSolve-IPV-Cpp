"""
诊断图像绘制 (Task 3/4/5/6 诊断函数)
功能: 可视化信号、相位相关峰值、验证匹配结果
用途: DD-SPPS 各阶段诊断, 输出 PNG 到 logs/ 目录
依赖: matplotlib (Agg 后端, 避免 GUI)

注意: matplotlib 采用延迟导入 + Agg 后端, 未安装时函数记录警告并跳过, 不影响模块导入。
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from lib.plate_solve.blind_index_v2.python.logging_setup import get_logger

logger = get_logger("ddspps.diagnostics")

# 默认输出目录: blind_index_v3/logs/
_MODULE_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_DEFAULT_OUT_DIR = os.path.join(_MODULE_ROOT, "logs")

# matplotlib 延迟导入 (Agg 后端, 避免 GUI 依赖导致导入失败)
_HAS_MPL = False
plt = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    plt = _plt
    _HAS_MPL = True
except ImportError:
    logger.warning("matplotlib 未安装, 诊断图像函数不可用")


def _ensure_dir(out_path: str) -> None:
    """确保输出目录存在。"""
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.isdir(out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as e:
            logger.warning("无法创建输出目录 %s: %s", out_dir, e)


def plot_signals(f: np.ndarray, g: np.ndarray, out_path: Optional[str] = None) -> None:
    """
    并排显示图像信号 f 和 Gaia 模板 g。

    Args:
        f: (grid, grid) 图像信号
        g: (grid, grid) Gaia 信号
        out_path: 输出 PNG 路径, None 则用默认 logs/signals.png
    """
    if not _HAS_MPL:
        logger.warning("plot_signals: matplotlib 不可用, 跳过")
        return
    if out_path is None:
        out_path = os.path.join(_DEFAULT_OUT_DIR, "signals.png")
    _ensure_dir(out_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(np.asarray(f), cmap="viridis", origin="lower")
    axes[0].set_title("Image signal f")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(np.asarray(g), cmap="viridis", origin="lower")
    axes[1].set_title("Gaia template g")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("信号图已保存: %s", out_path)


def plot_angular_profile(
    phi_f: np.ndarray,
    phi_g: np.ndarray,
    theta_cand: float,
    out_path: Optional[str] = None,
) -> None:
    """
    角度签名对比图 (图像 φ_f vs Gaia φ_g)。

    Args:
        phi_f: (n_bins,) 图像角度签名
        phi_g: (n_bins,) Gaia 角度签名
        theta_cand: 候选旋转角 (度)
        out_path: 输出 PNG 路径, None 则用默认 logs/angular_profile.png
    """
    if not _HAS_MPL:
        logger.warning("plot_angular_profile: matplotlib 不可用, 跳过")
        return
    if out_path is None:
        out_path = os.path.join(_DEFAULT_OUT_DIR, "angular_profile.png")
    _ensure_dir(out_path)

    n_bins = len(phi_f)
    angles = np.arange(n_bins) * 360.0 / max(n_bins, 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(angles, np.asarray(phi_f), label="φ_f (image)", alpha=0.8)
    ax.plot(angles, np.asarray(phi_g), label="φ_g (gaia)", alpha=0.8)
    ax.axvline(theta_cand, color="r", linestyle="--", label=f"θ_cand={theta_cand:.2f}°")
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Angular projection")
    ax.set_title("Angular signature (1D rotation estimate)")
    ax.legend(loc="best")
    ax.set_xlim(0, 360)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("角度签名图已保存: %s", out_path)


def plot_correlation_peak(
    C: np.ndarray,
    dx: float,
    dy: float,
    out_path: Optional[str] = None,
) -> None:
    """
    2D 互相关峰值图 (log scale, 标记峰值位置)。

    Args:
        C: (grid, grid) 互相关矩阵
        dx: 峰值 x 位置 (网格坐标)
        dy: 峰值 y 位置 (网格坐标)
        out_path: 输出 PNG 路径, None 则用默认 logs/correlation_peak.png
    """
    if not _HAS_MPL:
        logger.warning("plot_correlation_peak: matplotlib 不可用, 跳过")
        return
    if out_path is None:
        out_path = os.path.join(_DEFAULT_OUT_DIR, "correlation_peak.png")
    _ensure_dir(out_path)

    C = np.asarray(C, dtype=np.float64)
    C_safe = C + 1e-12  # 避免 log(0)
    grid = C.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(C_safe), cmap="hot", origin="lower",
                   extent=[0, grid, 0, grid])
    # 标记峰值 (注意 dx/dy 可能是负, 映射回 [0, grid) 显示)
    px = dx % grid
    py = dy % grid
    ax.plot(px, py, "c+", markersize=18, markeredgewidth=2.5, label=f"peak ({dx:.2f},{dy:.2f})")
    ax.plot(grid / 2.0, grid / 2.0, "g+", markersize=14, markeredgewidth=1.5, label="center")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("2D phase correlation (log scale)")
    ax.legend(loc="best")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("互相关峰值图已保存: %s", out_path)


def plot_verification(
    wcs,
    stars_x: np.ndarray,
    stars_y: np.ndarray,
    gaia_ra: np.ndarray,
    gaia_dec: np.ndarray,
    matched_pairs: List[Tuple[int, int, float]],
    out_path: Optional[str] = None,
) -> None:
    """
    验证匹配星图: 图像星投影位置(蓝圆点) vs Gaia 星(灰小点), 匹配 Gaia 星用红叉。

    Args:
        wcs: WCSResult (含 crval/cd/crpix)
        stars_x, stars_y: 图像星点坐标 (网格空间, 与 verify_wcs 一致)
        gaia_ra, gaia_dec: Gaia 参考星 (度)
        matched_pairs: list of (img_idx, gaia_idx, sep_arcsec)
        out_path: 输出 PNG 路径, None 则用默认 logs/verification.png
    """
    if not _HAS_MPL:
        logger.warning("plot_verification: matplotlib 不可用, 跳过")
        return
    if out_path is None:
        out_path = os.path.join(_DEFAULT_OUT_DIR, "verification.png")
    _ensure_dir(out_path)

    stars_x = np.asarray(stars_x, dtype=np.float64)
    stars_y = np.asarray(stars_y, dtype=np.float64)
    gaia_ra = np.asarray(gaia_ra, dtype=np.float64)
    gaia_dec = np.asarray(gaia_dec, dtype=np.float64)

    # 图像星 → 天球 (线性近似, 与 verify_wcs 一致)
    dx_pix = stars_x - wcs.crpix1
    dy_pix = stars_y - wcs.crpix2
    ra_pred = wcs.crval1 + dx_pix * wcs.cd11 + dy_pix * wcs.cd12
    dec_pred = wcs.crval2 + dx_pix * wcs.cd21 + dy_pix * wcs.cd22

    fig, ax = plt.subplots(figsize=(8, 7))
    # Gaia 星: 灰色小点
    if len(gaia_ra) > 0:
        ax.scatter(gaia_ra, gaia_dec, s=8, c="gray", alpha=0.5, label=f"Gaia ({len(gaia_ra)})")
    # 图像星投影: 蓝色圆点
    if len(ra_pred) > 0:
        ax.scatter(ra_pred, dec_pred, s=25, c="blue", marker="o",
                   edgecolors="none", alpha=0.7, label=f"Image ({len(ra_pred)})")
    # 匹配的 Gaia 星: 红色 x
    matched_gaia_idx = [p[1] for p in matched_pairs]
    if matched_gaia_idx:
        ax.scatter(gaia_ra[matched_gaia_idx], gaia_dec[matched_gaia_idx],
                   s=60, c="red", marker="x", linewidths=1.8,
                   label=f"Matched ({len(matched_gaia_idx)})")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"WCS verification (flip={wcs.flip_mode}, θ={wcs.theta_deg:.2f}°)")
    ax.legend(loc="best")
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("验证匹配图已保存: %s", out_path)

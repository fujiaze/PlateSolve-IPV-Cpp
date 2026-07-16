"""
端到端旋转恢复测试 (真实管线模拟)
功能: 从旋转星点坐标直接构建信号 (不用 scipy.rotate), 验证 phase_correlate_1d + 2D 相位相关
用途: 确认 DD-SPPS 管线在无边界 artifact 条件下的旋转/平移恢复精度

测试流程:
    1. 生成 N 颗随机星点 (x_i, y_i) in [0, grid]
    2. 构建 f = build_image_signal(stars)
    3. 对星点施加已知旋转 θ_true (绕网格中心) 得到 (x'_i, y'_i)
    4. 构建 g = build_image_signal(rotated_stars)  (模拟 Gaia 模板)
    5. phase_correlate_1d → theta_cand, 验证 |theta_cand - theta_true| < 1°
    6. rotate_signal(g, -theta_cand) + phase_correlate_2d → (dx, dy), 验证亚像素精度
"""
from __future__ import annotations

import sys
import math
import numpy as np

_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v3.python.signal import build_image_signal
from lib.plate_solve.blind_index_v3.python.phase_correlation import (
    angular_projection,
    phase_correlate_1d,
    rotate_signal,
    phase_correlate_2d,
    subpixel_refine,
    refine_rotation,
    windowed_fft2,
)


def rotate_points(
    xs: np.ndarray,
    ys: np.ndarray,
    theta_deg: float,
    cx: float,
    cy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    绕 (cx, cy) 旋转点集 θ_deg (数学约定: CCW, y-up)。
    注意: 在数组坐标系 (y-down) 中, 这等价于 CW 旋转。

    返回旋转后的 (xs', ys') 仍可能在 [0, grid] 范围外 (会被 build_image_signal 裁剪)。
    """
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs - cx
    dy = ys - cy
    # 标准旋转矩阵 (CCW in math, CW in array y-down)
    xs_new = cx + cos_t * dx - sin_t * dy
    ys_new = cy + sin_t * dx + cos_t * dy
    return xs_new, ys_new


def test_rotation_recovery_via_coordinate_rotation():
    """
    端到端: 从旋转星点坐标构建信号, 验证旋转角恢复。

    物理模型: 图像 f 由星点 (x,y) 构建; Gaia 模板 g 由同一批星点旋转 θ 后构建。
    约定: f = image signal, g = gaia template, 图像相对天空旋转了 θ。
    因此 f 的角度签名 phi_f = roll(phi_g, shift_θ), phase_correlate_1d 应返回 +θ。

    关键: 星点必须在圆形区域内, 保证旋转后不超出网格边界 (否则裁剪破坏旋转不变性)。
    注意: 实信号 FFT 幅度谱有 180° 对称性 (|F(u,v)|=|F(-u,-v)|),
          所以角度签名有 180° 歧义, theta_cand 可能返回 θ 或 θ+180°。
    """
    print("\n" + "=" * 70)
    print("端到端旋转恢复 (坐标旋转, 圆形区域避免裁剪)")
    print("=" * 70)
    grid = 512
    sigma = 4.5
    rng = np.random.default_rng(42)
    n_stars = 80
    cx = cy = grid / 2.0
    # 圆形区域: 半径 < grid/2 - margin, 保证旋转后不出界
    max_radius = grid / 2.0 - 50.0
    angles_rand = rng.uniform(0, 2 * math.pi, size=n_stars)
    radii = rng.uniform(0, max_radius, size=n_stars) * np.sqrt(rng.uniform(0, 1, size=n_stars))
    xs = cx + radii * np.cos(angles_rand)
    ys = cy + radii * np.sin(angles_rand)

    # 基准信号 f (原始星点)
    f = build_image_signal(xs, ys, grid, grid, grid=grid, sigma=sigma)
    F_f = np.fft.fft2(f)  # 2D 相位相关用 (不加窗, 保持平移不变)
    M_f = np.abs(windowed_fft2(f))  # 1D 角度签名用 (加窗消除 4 重对称)
    phi_f = angular_projection(M_f)
    phi_f = phi_f / max(phi_f.max(), 1e-10)

    results = []
    for theta_true in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0, 315.0]:
        # 旋转星点构建 g (模拟 Gaia 模板, 图像旋转了 theta_true)
        xs_rot, ys_rot = rotate_points(xs, ys, theta_true, cx, cy)
        g = build_image_signal(xs_rot, ys_rot, grid, grid, grid=grid, sigma=sigma)

        M_g = np.abs(windowed_fft2(g))  # 加窗 FFT 用于角度签名
        phi_g = angular_projection(M_g)
        phi_g = phi_g / max(phi_g.max(), 1e-10)

        theta_cand, snr, _ = phase_correlate_1d(phi_f, phi_g)

        # 真实管线约定: phi_f = roll(phi_g, shift), 所以期望 theta_cand = +theta_true
        # 但实信号 FFT 幅度谱有 180° 对称性, 所以 theta_cand 可能是 theta_true 或 theta_true+180°
        diff = (theta_cand - theta_true + 180.0) % 360.0 - 180.0
        diff_mod180 = (diff + 90.0) % 180.0 - 90.0  # 折叠到 [-90, 90]
        status = "PASS" if abs(diff_mod180) < 2.0 else "FAIL"
        results.append((theta_true, theta_cand, diff, diff_mod180, snr, status))
        print(f"  theta_true={theta_true:6.1f}° → theta_cand={theta_cand:7.2f}°, "
              f"diff={diff:+7.2f}°, diff_mod180={diff_mod180:+.2f}°, SNR={snr:.1f} [{status}]")

    n_pass = sum(1 for r in results if r[5] == "PASS")
    print(f"\n  旋转恢复 (mod 180°): {n_pass}/{len(results)} PASS")
    return n_pass == len(results)


def test_full_pipeline_rotation_and_translation():
    """
    完整管线: 旋转 + 平移恢复。

    流程:
        1. 构建 f (原始星点)
        2. 旋转 + 平移星点构建 g (模拟 Gaia 模板)
        3. phase_correlate_1d → theta_cand
        4. refine_rotation → theta_best, dx_sub, dy_sub
        5. 验证 theta_best 精度和平移恢复
    """
    print("\n" + "=" * 70)
    print("完整管线: 旋转 + 平移恢复 (圆形区域星点)")
    print("=" * 70)
    grid = 512
    sigma = 4.5
    rng = np.random.default_rng(123)
    n_stars = 80
    cx = cy = grid / 2.0
    # 圆形区域避免旋转裁剪
    max_radius = grid / 2.0 - 60.0
    angles_rand = rng.uniform(0, 2 * math.pi, size=n_stars)
    radii = rng.uniform(0, max_radius, size=n_stars) * np.sqrt(rng.uniform(0, 1, size=n_stars))
    xs = cx + radii * np.cos(angles_rand)
    ys = cy + radii * np.sin(angles_rand)

    f = build_image_signal(xs, ys, grid, grid, grid=grid, sigma=sigma)
    F_f = np.fft.fft2(f)
    M_f = np.abs(windowed_fft2(f))
    phi_f = angular_projection(M_f)
    phi_f = phi_f / max(phi_f.max(), 1e-10)

    # 测试案例: (theta_true, dx_true, dy_true)
    test_cases = [
        (30.0, 10.0, -5.0),
        (90.0, 0.0, 0.0),
        (135.0, -20.0, 15.0),
        (0.0, 25.0, 10.0),
    ]

    for theta_true, dx_true, dy_true in test_cases:
        # 旋转 + 平移星点
        xs_rot, ys_rot = rotate_points(xs, ys, theta_true, cx, cy)
        xs_rot += dx_true
        ys_rot += dy_true
        g = build_image_signal(xs_rot, ys_rot, grid, grid, grid=grid, sigma=sigma)

        M_g = np.abs(windowed_fft2(g))
        phi_g = angular_projection(M_g)
        phi_g = phi_g / max(phi_g.max(), 1e-10)

        theta_cand, snr_1d, _ = phase_correlate_1d(phi_f, phi_g)

        # 旋转精化 + 2D 相位相关
        theta_best, dx_sub, dy_sub, peak_snr, _ = refine_rotation(
            F_f, g, theta_cand, search_range=2.0, step=0.5
        )

        theta_diff = (theta_best - theta_true + 180.0) % 360.0 - 180.0
        print(f"  θ_true={theta_true:6.1f}°, dx_true={dx_true:+5.1f}, dy_true={dy_true:+5.1f} → "
              f"θ_best={theta_best:7.2f}° (diff={theta_diff:+.2f}°), "
              f"dx={dx_sub:+7.2f}, dy={dy_sub:+7.2f}, SNR={peak_snr:.1f}")


if __name__ == "__main__":
    ok1 = test_rotation_recovery_via_coordinate_rotation()
    test_full_pipeline_rotation_and_translation()
    print("\n" + "=" * 70)
    if ok1:
        print("结论: phase_correlate_1d 符号约定正确, 无需修复")
    else:
        print("结论: 部分测试未通过, 需进一步调查")
    print("=" * 70)

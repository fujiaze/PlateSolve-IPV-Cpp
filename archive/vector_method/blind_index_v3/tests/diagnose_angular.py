"""
角度签名旋转不变性诊断
功能: 诊断 angular_projection 是否产生真正的旋转不变角度签名
用途: 找出非 90° 旋转恢复失败的根因

诊断内容:
    1. 构建 f 和 g (旋转 30°), 打印角度签名峰值位置
    2. 检查 phi_g 是否是 phi_f 的循环平移
    3. 对比不同星数/分布下的表现
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
    windowed_fft2,
    hann_window,
)


def rotate_points(xs, ys, theta_deg, cx, cy):
    theta = math.radians(theta_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = xs - cx
    dy = ys - cy
    xs_new = cx + cos_t * dx - sin_t * dy
    ys_new = cy + sin_t * dx + cos_t * dy
    return xs_new, ys_new


def diagnose_angular_profile():
    """
    诊断: 检查角度签名是否真正旋转不变。
    """
    print("=" * 70)
    print("诊断: 角度签名旋转不变性")
    print("=" * 70)
    grid = 512
    sigma = 4.5
    cx = cy = grid / 2.0

    for n_stars in [50, 200, 500]:
        print(f"\n--- n_stars = {n_stars} ---")
        rng = np.random.default_rng(42)
        max_radius = grid / 2.0 - 50.0
        angles_rand = rng.uniform(0, 2 * math.pi, size=n_stars)
        radii = max_radius * np.sqrt(rng.uniform(0, 1, size=n_stars))
        xs = cx + radii * np.cos(angles_rand)
        ys = cy + radii * np.sin(angles_rand)

        f = build_image_signal(xs, ys, grid, grid, grid=grid, sigma=sigma)
        M_f = np.abs(windowed_fft2(f))
        phi_f = angular_projection(M_f)
        phi_f = phi_f / max(phi_f.max(), 1e-10)

        for theta_true in [0.0, 30.0, 90.0]:
            xs_rot, ys_rot = rotate_points(xs, ys, theta_true, cx, cy)
            g = build_image_signal(xs_rot, ys_rot, grid, grid, grid=grid, sigma=sigma)
            M_g = np.abs(windowed_fft2(g))
            phi_g = angular_projection(M_g)
            phi_g = phi_g / max(phi_g.max(), 1e-10)

            # 检查 phi_g 是否是 phi_f 的循环平移
            # 用 1D 相位相关找平移
            theta_cand, snr, r = phase_correlate_1d(phi_f, phi_g)

            # 打印角度签名 top-5 峰值位置
            top5_f = np.argsort(phi_f)[-5:][::-1]
            top5_g = np.argsort(phi_g)[-5:][::-1]
            top5_f_deg = top5_f * 360.0 / len(phi_f)
            top5_g_deg = top5_g * 360.0 / len(phi_g)

            # 检查 r 的 top-3 峰值
            top3_r = np.argsort(r)[-3:][::-1]
            top3_r_deg = top3_r * 360.0 / len(r)
            top3_r_val = r[top3_r]

            print(f"  θ_true={theta_true:5.1f}°: theta_cand={theta_cand:.2f}°, SNR={snr:.1f}")
            print(f"    phi_f top5 (deg): {top5_f_deg}")
            print(f"    phi_g top5 (deg): {top5_g_deg}")
            print(f"    r top3 (deg): {top3_r_deg}, vals: {top3_r_val}")

            # 直接验证: 如果 phi_g = roll(phi_f, shift), 则 phi_g[i] ≈ phi_f[i - shift]
            # 尝试所有可能的 shift, 找最佳相关
            n = len(phi_f)
            best_shift = 0
            best_corr = -1
            for s in range(n):
                rolled = np.roll(phi_f, s)
                corr = np.corrcoef(rolled, phi_g)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_shift = s
            best_shift_deg = best_shift * 360.0 / n
            print(f"    暴力搜索最佳 shift: {best_shift_deg:.2f}°, 相关系数: {best_corr:.4f}")


def diagnose_with_dense_uniform_stars():
    """
    用密集均匀分布的星点测试 (更接近真实星场)。
    """
    print("\n" + "=" * 70)
    print("诊断: 密集均匀星场 (1000 颗)")
    print("=" * 70)
    grid = 512
    sigma = 4.5
    cx = cy = grid / 2.0
    rng = np.random.default_rng(99)
    n_stars = 1000
    # 均匀分布在全网格
    xs = rng.uniform(20, grid - 20, size=n_stars)
    ys = rng.uniform(20, grid - 20, size=n_stars)

    f = build_image_signal(xs, ys, grid, grid, grid=grid, sigma=sigma)
    M_f = np.abs(windowed_fft2(f))
    phi_f = angular_projection(M_f)
    phi_f = phi_f / max(phi_f.max(), 1e-10)

    for theta_true in [0.0, 30.0, 45.0, 90.0, 135.0, 180.0]:
        xs_rot, ys_rot = rotate_points(xs, ys, theta_true, cx, cy)
        # 旋转后部分星点出界, 但密集分布下影响小
        g = build_image_signal(xs_rot, ys_rot, grid, grid, grid=grid, sigma=sigma)
        M_g = np.abs(windowed_fft2(g))
        phi_g = angular_projection(M_g)
        phi_g = phi_g / max(phi_g.max(), 1e-10)

        theta_cand, snr, r = phase_correlate_1d(phi_f, phi_g)

        # 暴力搜索最佳 shift
        n = len(phi_f)
        best_shift = 0
        best_corr = -1
        for s in range(0, n, 2):  # 步长 2 加速
            rolled = np.roll(phi_f, s)
            corr = np.corrcoef(rolled, phi_g)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_shift = s
        best_shift_deg = best_shift * 360.0 / n

        diff_mod180 = ((theta_cand - theta_true + 90.0) % 180.0) - 90.0
        brute_diff_mod180 = ((best_shift_deg - theta_true + 90.0) % 180.0) - 90.0
        print(f"  θ_true={theta_true:5.1f}°: phase_corr={theta_cand:.2f}° (diff_mod180={diff_mod180:+.2f}°, SNR={snr:.1f}), "
              f"brute={best_shift_deg:.2f}° (diff_mod180={brute_diff_mod180:+.2f}°, corr={best_corr:.4f})")


if __name__ == "__main__":
    diagnose_angular_profile()
    diagnose_with_dense_uniform_stars()

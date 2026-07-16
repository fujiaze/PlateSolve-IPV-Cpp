"""
phase_correlate_1d 符号约定验证测试
功能: 验证 1D 相位相关函数在不同输入约定下返回的旋转角符号
用途: 诊断 phase_correlate_1d 是否存在符号 bug, 确定 scipy.rotate + angular_projection 的整体符号约定

测试场景:
    场景 A: phi_f = roll(phi_g, shift)  (图像是 Gaia 模板的旋转版本)
    场景 B: phi_g = roll(phi_f, shift)  (Gaia 模板是图像的旋转版本)
    场景 C: 端到端 — f 为原始信号, g = scipy.rotate(f, theta_true), 验证 theta_cand
    场景 D: 端到端 — g 为原始信号, f = scipy.rotate(g, theta_true), 验证 theta_cand
"""
from __future__ import annotations

import sys
import os
import numpy as np

# 添加项目根到路径
_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v3.python.phase_correlation import (
    angular_projection,
    phase_correlate_1d,
    rotate_signal,
    phase_correlate_2d,
    subpixel_refine,
)
from lib.plate_solve.blind_index_v3.python.signal import build_image_signal


def _make_asymmetric_signal(grid: int = 512, n_stars: int = 30, seed: int = 42) -> np.ndarray:
    """
    构造非对称星场信号 (避免旋转对称导致歧义)。
    用固定 seed 保证可复现。
    """
    rng = np.random.default_rng(seed)
    xs = rng.uniform(50, grid - 50, size=n_stars)
    ys = rng.uniform(50, grid - 50, size=n_stars)
    f = build_image_signal(xs, ys, grid, grid, grid=grid, sigma=4.5)
    return f


def test_scenario_a_phi_f_is_rotated():
    """
    场景 A: phi_f = roll(phi_g, shift)
    物理含义: 图像 f 是 Gaia 模板 g 旋转 shift 后的结果
    期望: phase_correlate_1d 应返回 +shift
    """
    print("\n" + "=" * 70)
    print("场景 A: phi_f = roll(phi_g, shift)  — 图像 = Gaia 旋转 shift")
    print("=" * 70)
    f = _make_asymmetric_signal()
    M_f = np.abs(np.fft.fft2(f))
    phi_g_base = angular_projection(M_f)

    n = len(phi_g_base)
    for shift_deg in [30.0, 90.0, 135.0, 180.0, 270.0]:
        shift_bins = int(round(shift_deg * n / 360.0))
        phi_f = np.roll(phi_g_base, shift_bins)
        phi_g = phi_g_base.copy()
        theta_cand, snr, _ = phase_correlate_1d(phi_f, phi_g)
        diff = (theta_cand - shift_deg + 180.0) % 360.0 - 180.0
        status = "PASS" if abs(diff) < 1.0 else "FAIL"
        print(f"  shift={shift_deg:6.1f}° → theta_cand={theta_cand:7.2f}°, diff={diff:+7.2f}°, SNR={snr:.1f} [{status}]")


def test_scenario_b_phi_g_is_rotated():
    """
    场景 B: phi_g = roll(phi_f, shift)
    物理含义: Gaia 模板 g 是图像 f 旋转 shift 后的结果
    期望: phase_correlate_1d 应返回 -shift (即 360-shift)
    """
    print("\n" + "=" * 70)
    print("场景 B: phi_g = roll(phi_f, shift)  — Gaia = 图像 旋转 shift")
    print("=" * 70)
    f = _make_asymmetric_signal()
    M_f = np.abs(np.fft.fft2(f))
    phi_f_base = angular_projection(M_f)

    n = len(phi_f_base)
    for shift_deg in [30.0, 90.0, 135.0, 180.0, 270.0]:
        shift_bins = int(round(shift_deg * n / 360.0))
        phi_f = phi_f_base.copy()
        phi_g = np.roll(phi_f_base, shift_bins)
        theta_cand, snr, _ = phase_correlate_1d(phi_f, phi_g)
        expected = (-shift_deg) % 360.0
        diff = (theta_cand - expected + 180.0) % 360.0 - 180.0
        status = "PASS" if abs(diff) < 1.0 else "FAIL"
        print(f"  shift={shift_deg:6.1f}° → theta_cand={theta_cand:7.2f}°, expected={expected:7.2f}°, diff={diff:+7.2f}°, SNR={snr:.1f} [{status}]")


def test_scenario_c_end_to_end_f_rotated():
    """
    场景 C: 端到端 — g 为原始 Gaia 模板, f = scipy.rotate(g, theta_true)
    物理含义: 图像信号 f 是 Gaia 模板 g 旋转 theta_true 后的结果
    期望: theta_cand 应反映 theta_true (符号待验证)
    """
    from scipy.ndimage import rotate as nd_rotate
    print("\n" + "=" * 70)
    print("场景 C: 端到端 — f = scipy.rotate(g, theta_true)")
    print("=" * 70)
    g = _make_asymmetric_signal(seed=100)
    for theta_true in [30.0, 45.0, 90.0, 135.0, 180.0, 270.0]:
        f = nd_rotate(g, theta_true, reshape=False, order=1, mode="constant", cval=0.0)
        M_f = np.abs(np.fft.fft2(f))
        M_g = np.abs(np.fft.fft2(g))
        phi_f = angular_projection(M_f)
        phi_g = angular_projection(M_g)
        theta_cand, snr, _ = phase_correlate_1d(phi_f, phi_g)
        # 检查多种可能的期望值
        candidates = {
            "+theta": theta_true % 360.0,
            "-theta": (-theta_true) % 360.0,
            "+theta+90": (theta_true + 90.0) % 360.0,
            "-theta+90": (-theta_true + 90.0) % 360.0,
            "+theta+180": (theta_true + 180.0) % 360.0,
        }
        best_label = None
        best_diff = 1e9
        for label, exp in candidates.items():
            d = (theta_cand - exp + 180.0) % 360.0 - 180.0
            if abs(d) < abs(best_diff):
                best_diff = d
                best_label = label
        print(f"  theta_true={theta_true:6.1f}° → theta_cand={theta_cand:7.2f}°, "
              f"best_match={best_label} (diff={best_diff:+.2f}°), SNR={snr:.1f}")


def test_scenario_d_end_to_end_g_rotated():
    """
    场景 D: 端到端 — f 为原始图像, g = scipy.rotate(f, theta_true)
    物理含义: Gaia 模板 g 是图像 f 旋转 theta_true 后的结果
    期望: theta_cand 应反映 -theta_true (符号待验证)
    """
    from scipy.ndimage import rotate as nd_rotate
    print("\n" + "=" * 70)
    print("场景 D: 端到端 — g = scipy.rotate(f, theta_true)")
    print("=" * 70)
    f = _make_asymmetric_signal(seed=200)
    for theta_true in [30.0, 45.0, 90.0, 135.0, 180.0, 270.0]:
        g = nd_rotate(f, theta_true, reshape=False, order=1, mode="constant", cval=0.0)
        M_f = np.abs(np.fft.fft2(f))
        M_g = np.abs(np.fft.fft2(g))
        phi_f = angular_projection(M_f)
        phi_g = angular_projection(M_g)
        theta_cand, snr, _ = phase_correlate_1d(phi_f, phi_g)
        candidates = {
            "+theta": theta_true % 360.0,
            "-theta": (-theta_true) % 360.0,
            "+theta+90": (theta_true + 90.0) % 360.0,
            "-theta+90": (-theta_true + 90.0) % 360.0,
            "+theta+180": (theta_true + 180.0) % 360.0,
        }
        best_label = None
        best_diff = 1e9
        for label, exp in candidates.items():
            d = (theta_cand - exp + 180.0) % 360.0 - 180.0
            if abs(d) < abs(best_diff):
                best_diff = d
                best_label = label
        print(f"  theta_true={theta_true:6.1f}° → theta_cand={theta_cand:7.2f}°, "
              f"best_match={best_label} (diff={best_diff:+.2f}°), SNR={snr:.1f}")


def test_2d_phase_correlation_recovery():
    """
    端到端 2D 相位相关平移恢复测试。
    构造已知平移, 验证 phase_correlate_2d + subpixel_refine 的精度。
    """
    print("\n" + "=" * 70)
    print("2D 相位相关平移恢复测试")
    print("=" * 70)
    grid = 512
    f = _make_asymmetric_signal(grid=grid, seed=300)
    F_f = np.fft.fft2(f)

    for (dx_true, dy_true) in [(10, -5), (20, 15), (-30, -20), (0, 0)]:
        # 用 FFT 移位构造精确平移的 g
        g = np.roll(f, (dy_true, dx_true), axis=(0, 1))
        dx, dy, peak, snr, C = phase_correlate_2d(F_f, g)
        dx_sub, dy_sub = subpixel_refine(C, dx, dy)
        print(f"  true(dx={dx_true:+4d}, dy={dy_true:+4d}) → "
              f"raw(dx={dx:+7.3f}, dy={dy:+7.3f}), sub(dx={dx_sub:+7.3f}, dy={dy_sub:+7.3f}), "
              f"SNR={snr:.1f}")


if __name__ == "__main__":
    test_scenario_a_phi_f_is_rotated()
    test_scenario_b_phi_g_is_rotated()
    test_scenario_c_end_to_end_f_rotated()
    test_scenario_d_end_to_end_g_rotated()
    test_2d_phase_correlation_recovery()

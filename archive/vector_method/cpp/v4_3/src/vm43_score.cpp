// ============================================================================
// vm43_score.cpp - V4.3 S_robust 稳健评分模块 (IRM Step 5)
//
// 职责: 残差跳变检测外点 + 自适应内点比例估计, 替代 V4.2 普通 RMS
// V4.3 新增模块
//
// 算法:
//   Step A: 残差排序 r_(1) ≤ r_(2) ≤ ... ≤ r_(N)
//   Step B: 残差跳变检测 ratio[i] = r_(i)/r_(i-1) > 3.0 → k_cut
//           备选: MAD 方法 k_cut = count(r_i < median_r + 3.0 × MAD)
//           取两种方法的较大值 (更保守: 信任更包容的方法;
//           跳变检测在小样本+自然散布下易误触发 k_cut=1, MAD 兜底)
//   Step C: N_robust = min(k_cut, max(N/2, M₀))
//           S_robust = rms(前 N_robust 个残差)
//
// 残差计算 (与 vm43_fit Layer 1 一致, 像素空间):
//   图像像素偏移 (原点在 CRPIX):
//     ssx =  U.x / s0     (X: 右为正, U 是 arcsec, s0 是 arcsec/pixel)
//     ssy = -U.y / s0     (Y: 翻转: U_y 向上=Dec, 像素_y 向下)
//   CD^{-1} 预测像素偏移 (W arcsec → deg → pixel):
//     ddx = cdi2 · W / 3600   (cdi2 = CD^{-1}, pixel/deg)
//     ddy = cdi2 · W / 3600
//   SIP 修正 (像素空间, 在实际像素坐标处求值):
//     du, dv = sip_eval(ssx, ssy)
//     ddx_corrected = ddx - du  (SIP 建模 predicted - actual, 减去即修正预测)
//   平移估计 (median, 稳健, 因 CD 不含平移):
//     t_x = median(ddx_corrected - ssx)
//     t_y = median(ddy_corrected - ssy)
//   残差 (pixel): r_i = |(ddx_corrected - ssx) - t_x, (ddy_corrected - ssy) - t_y|
//   残差 (arcsec): r_i × s0
//
// Task 6 实现, Task 11 修复单位错误
// ============================================================================

#include "vm43_internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace v43 {

namespace {

// SIP 多项式求值 (order>=2 才有贡献, 仅累加 p+q>=2 且 p+q<=order 的项)
// 系数布局: A[p*6+q], B[p*6+q], p,q ∈ [0, order]
// 注: 系数已反归一化 (与 vm43_fit Layer 2 一致), 输入应为原始像素坐标
void sip_eval(const SIPCoeffs& sip, double x, double y,
              double& du, double& dv) {
    du = 0.0;
    dv = 0.0;
    if (sip.order < 2) return;
    double xp[5] = {1.0, x, x * x, x * x * x, x * x * x * x};
    double yq[5] = {1.0, y, y * y, y * y * y, y * y * y * y};
    for (int p = 0; p <= sip.order; ++p) {
        for (int q = 0; q <= sip.order; ++q) {
            int s = p + q;
            if (s < 2 || s > sip.order) continue;
            int idx = p * 6 + q;
            du += sip.A[idx] * xp[p] * yq[q];
            dv += sip.B[idx] * xp[p] * yq[q];
        }
    }
}

} // anonymous namespace

int vm43_compute_s_robust(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const std::vector<MatchPair>& control_points,
    const CDMatrix& cd,
    const SIPCoeffs& sip,
    double s0,
    int M0,
    SRobustResult& output,
    Logger* logger)
{
    std::memset(&output, 0, sizeof(output));

    const int N = (int)control_points.size();
    if (N <= 0) {
        if (logger) logger->warn("vm43_compute_s_robust: 控制点为空");
        return -1;
    }
    if (s0 <= 0) {
        if (logger) logger->error("vm43_compute_s_robust: s0 <= 0");
        return -1;
    }

    if (logger) {
        logger->infof("vm43_compute_s_robust: N=%d, M0=%d, s0=%.4f\", sip_order=%d",
                      N, M0, s0, sip.order);
    }

    // -------------------------------------------------------------------
    // CD 逆矩阵 (pixel/deg): 用于 W(arcsec→deg) → pixel 预测
    //   CD 是 deg/pixel, CD^{-1} 是 pixel/deg
    // -------------------------------------------------------------------
    double cd_det = cd.cd11 * cd.cd22 - cd.cd12 * cd.cd21;
    if (std::abs(cd_det) < 1e-20) {
        if (logger) logger->error("vm43_compute_s_robust: CD 行列式接近 0");
        return -1;
    }
    double cdi2[4] = {
        cd.cd22 / cd_det, -cd.cd12 / cd_det,
        -cd.cd21 / cd_det,  cd.cd11 / cd_det
    };

    // -------------------------------------------------------------------
    // Step A: 计算偏移 (像素空间, 与 vm43_fit Layer 1 一致)
    //
    //   ssx =  U.x / s0     (图像像素偏移 X, 右为正)
    //   ssy = -U.y / s0     (图像像素偏移 Y, 翻转: U_y 向上=Dec, 像素_y 向下)
    //   ddx = cdi2 · W/3600 (CD^{-1} 预测像素偏移, 无平移)
    //   ddy = cdi2 · W/3600
    //   SIP 修正: du,dv = sip_eval(ssx, ssy); ddx -= du; ddy -= dv
    //   offset = (ddx - ssx, ddy - ssy)  (含平移)
    // -------------------------------------------------------------------
    std::vector<double> offset_x, offset_y;
    offset_x.reserve(N);
    offset_y.reserve(N);

    for (int i = 0; i < N; ++i) {
        const MatchPair& mp = control_points[i];
        if (mp.u < 0 || mp.u >= (int)U.size() ||
            mp.w < 0 || mp.w >= (int)W.size()) {
            if (logger) logger->warnf("vm43_compute_s_robust: 控制点 %d 索引越界", i);
            continue;
        }

        // 图像像素偏移 (原点在 CRPIX)
        double ssx =  U[mp.u].x / s0;
        double ssy = -U[mp.u].y / s0;

        // CD^{-1} 预测像素偏移 (W arcsec → deg → pixel, 无平移)
        double Wx = W[mp.w].x, Wy = W[mp.w].y;
        double ddx = cdi2[0] * Wx / 3600.0 + cdi2[1] * Wy / 3600.0;
        double ddy = cdi2[2] * Wx / 3600.0 + cdi2[3] * Wy / 3600.0;

        // SIP 修正 (像素空间, 在实际像素坐标处求值)
        // vm43_fit: rx_h = ddx - ssx ≈ SIP(ssx, ssy) → ddx - SIP ≈ ssx
        double du, dv;
        sip_eval(sip, ssx, ssy, du, dv);
        ddx -= du;
        ddy -= dv;

        // 偏移 (预测 - 实际, 含平移)
        offset_x.push_back(ddx - ssx);
        offset_y.push_back(ddy - ssy);
    }

    int n_valid = (int)offset_x.size();
    if (n_valid <= 0) {
        if (logger) logger->error("vm43_compute_s_robust: 无有效残差");
        return -1;
    }

    // -------------------------------------------------------------------
    // 估计平移 (median, 稳健 — CD 不含平移, 需从控制点估计)
    // -------------------------------------------------------------------
    std::vector<double> ox_sorted = offset_x;
    std::vector<double> oy_sorted = offset_y;
    std::sort(ox_sorted.begin(), ox_sorted.end());
    std::sort(oy_sorted.begin(), oy_sorted.end());
    double t_x = ox_sorted[n_valid / 2];
    double t_y = oy_sorted[n_valid / 2];

    // -------------------------------------------------------------------
    // 计算残差 (arcsec): r_i = |offset_i - t_median| × s0
    // -------------------------------------------------------------------
    std::vector<double> r;
    r.reserve(n_valid);
    for (int i = 0; i < n_valid; ++i) {
        double rx = offset_x[i] - t_x;
        double ry = offset_y[i] - t_y;
        double ri_pixel = std::sqrt(rx * rx + ry * ry);
        double ri_arcsec = ri_pixel * s0;
        r.push_back(ri_arcsec);
    }

    std::sort(r.begin(), r.end());

    // -------------------------------------------------------------------
    // Step B: 残差跳变检测 (ratio > 3.0) + MAD 备选, 取较大值 (更保守)
    //   跳变检测在小样本 + 自然散布下易误触发 k_cut=1
    //   MAD 兜底, 取 max 避免虚假低 S_robust
    // -------------------------------------------------------------------
    // 方法 1: 残差比跳变
    //   0-indexed: r[k]/r[k-1] > 3.0 → 截断 k_cut_ratio = k (保留 r[0..k-1])
    int k_cut_ratio = n_valid;
    for (int k = 1; k < n_valid; ++k) {
        if (r[k - 1] > 1e-12) {
            double ratio = r[k] / r[k - 1];
            if (ratio > 3.0) {
                k_cut_ratio = k;
                break;
            }
        }
    }

    // 方法 2: MAD
    //   median_r = r[N/2], MAD = median(|r_i - median_r|) × 1.4826
    //   k_cut_mad = count(r_i < median_r + 3.0 × MAD)
    double median_r = r[n_valid / 2];
    std::vector<double> abs_dev(n_valid);
    for (int i = 0; i < n_valid; ++i) {
        abs_dev[i] = std::fabs(r[i] - median_r);
    }
    std::sort(abs_dev.begin(), abs_dev.end());
    double mad = abs_dev[n_valid / 2] * 1.4826;
    double mad_thresh = median_r + 3.0 * mad;
    int k_cut_mad = 0;
    for (int i = 0; i < n_valid; ++i) {
        if (r[i] < mad_thresh) ++k_cut_mad;
        else break;  // 已排序, 后续更大
    }

    int k_cut = std::max(k_cut_ratio, k_cut_mad);
    if (k_cut < 1) k_cut = 1;  // 至少保留 1 个
    if (k_cut > n_valid) k_cut = n_valid;

    // -------------------------------------------------------------------
    // Step C: N_robust = min(k_cut, max(N/2, M0)), S_robust = rms(前 N_robust)
    // -------------------------------------------------------------------
    int half = n_valid / 2;
    if (half < 1) half = 1;
    int lower = std::max(half, M0);
    int n_robust = std::min(k_cut, lower);
    if (n_robust < 1) n_robust = 1;
    if (n_robust > n_valid) n_robust = n_valid;

    double sum_sq = 0.0;
    for (int i = 0; i < n_robust; ++i) {
        sum_sq += r[i] * r[i];
    }
    double s_robust = std::sqrt(sum_sq / n_robust);

    // 覆盖率: n_inliers / N
    double coverage = (double)n_robust / (double)n_valid;

    // 输出
    output.s_robust  = s_robust;
    output.n_inliers = n_robust;
    output.coverage  = coverage;
    output.k_cut     = k_cut;
    output.median_r  = median_r;
    output.mad       = mad;

    if (logger) {
        logger->infof("vm43_compute_s_robust: t_median=(%.4f, %.4f)px, "
                      "k_cut_ratio=%d, k_cut_mad=%d, k_cut=%d, n_robust=%d, "
                      "S_robust=%.4f\", median=%.4f, mad=%.4f, coverage=%.3f",
                      t_x, t_y,
                      k_cut_ratio, k_cut_mad, k_cut, n_robust, s_robust,
                      median_r, mad, coverage);
    }

    return 0;
}

} // namespace v43

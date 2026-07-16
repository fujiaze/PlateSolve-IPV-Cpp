#ifndef PV_INTERNAL_H
#define PV_INTERNAL_H

// ============================================================================
// pv_internal.h - PairVerifier 内部声明（不对外暴露）
//
// 包含：工具函数(vec_median/umeyama/apply_similarity) + MAD/贝叶斯/三角形接口
// ============================================================================

#include "pv_api.h"
#include "v42_log.h"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>

// Eigen（仅用于 Umeyama SVD）
#include <Eigen/Dense>

namespace pv {

// --- 常量 ---
static constexpr double PV_PI = 3.14159265358979323846;
static constexpr double DEG_TO_ARCSEC = 3600.0;
static constexpr double SQDEG_TO_SQARCSEC = 3600.0 * 3600.0;

// ============================================================================
// 工具函数
// ============================================================================

// 中位数（会修改输入向量）
inline double vec_median(std::vector<double>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
        return (v[n / 2] + v[n / 2 - 1]) * 0.5;
    }
    return v[n / 2];
}

// 应用相似变换: Wt = s*(R×W) + t
inline void apply_similarity(const double* W, int M,
                             double s, double theta, double tx, double ty,
                             double* Wt) {
    double ct = std::cos(theta), st = std::sin(theta);
    for (int i = 0; i < M; ++i) {
        double wx = W[i * 2], wy = W[i * 2 + 1];
        Wt[i * 2]     = s * (ct * wx - st * wy) + tx;
        Wt[i * 2 + 1] = s * (st * wx + ct * wy) + ty;
    }
}

// Umeyama 2D 相似变换拟合（SVD）
//   src→dst 的最佳相似变换: dst ≈ s×R×src + t
//   尺度约束: |s-1.0| < 0.1（U 和 W 均为角秒坐标，尺度应接近1）
//   n < 2 或退化时返回 valid=false
inline v42::SimTransform umeyama(const double* src, const double* dst, int n) {
    v42::SimTransform r;
    r.valid = false; r.s = 1; r.theta = 0; r.tx = 0; r.ty = 0;
    if (n < 2) return r;

    using M2 = Eigen::Matrix2d;
    using V2 = Eigen::Vector2d;
    V2 ms = V2::Zero(), md = V2::Zero();
    for (int i = 0; i < n; ++i) {
        ms += V2(src[i * 2], src[i * 2 + 1]);
        md += V2(dst[i * 2], dst[i * 2 + 1]);
    }
    ms /= n; md /= n;

    Eigen::MatrixXd sc(2, n), dc(2, n);
    for (int i = 0; i < n; ++i) {
        sc(0, i) = src[i * 2]     - ms(0);
        sc(1, i) = src[i * 2 + 1] - ms(1);
        dc(0, i) = dst[i * 2]     - md(0);
        dc(1, i) = dst[i * 2 + 1] - md(1);
    }
    M2 H = sc * dc.transpose();
    Eigen::JacobiSVD<M2> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    V2 Sv = V2::Ones(); Sv(1) = det;
    M2 R = svd.matrixV() * Sv.asDiagonal() * svd.matrixU().transpose();
    double tr = sc.colwise().squaredNorm().sum();
    if (tr < 1e-15) return r;
    double s = svd.singularValues().dot(Sv) / tr;
    if (std::abs(s - 1.0) >= 0.1) return r;  // 尺度约束
    double th = std::atan2(R(1, 0), R(0, 0));
    V2 t = md - s * R * ms;
    r.s = s; r.theta = th; r.tx = t(0); r.ty = t(1); r.valid = true;
    return r;
}

// ============================================================================
// MAD 清洗结果
// ============================================================================
struct MadResult {
    std::vector<int> clean_u;       // 清洗后U索引
    std::vector<int> clean_w;       // 清洗后W索引
    int    n_removed;               // 剔除数
    int    iterations;              // 实际迭代次数
    double rms_arcsec;              // 清洗后RMS(角秒)
    v42::SimTransform transform;    // 最终变换参数
    bool   success;
};

// pv_mad.cpp - 3轮MAD迭代清洗
//   输入: U, W, pairs_u, pairs_w, params
//   每轮: 计算残差 → MAD阈值 → 剔除离群 → 重新Umeyama拟合
//   阈值: max(mad_min_threshold_arcsec, mad_threshold_factor × 1.4826 × MAD)
MadResult pv_mad_clean(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    const PairVerifierParams* params,
    v42::Logger* logger
);

// pv_bayes.cpp - 贝叶斯假设验证
//   lnK = Σ[-log(2πσ²) - r²/(2σ²)] + n×log(A_fov_sqsec)
//   A_fov_sqsec = π × (fov_diag_deg/2)² × 3600²
//   σ = max(sigma_min, rms_arcsec)
//   decision: lnK>lnK_accept→1, >lnK_weak→0, 否则→-1
v42::BayesResult pv_bayes_verify(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double sigma_min, double mad_rms_arcsec,
    double fov_diag_deg,
    double lnK_accept, double lnK_weak,
    v42::Logger* logger
);

// pv_triangle.cpp - 三角形双特征验证
//   特征1: 面积 A（海伦公式）
//   特征2: 极惯性矩 J = A×(a²+b²+c²)/36
//   通过条件: |A_U/A_W - 1| < eps_A && |J_U/J_W - 1| < eps_J
//   n ≤ 30: 遍历所有 C(n,3); n > 30: 随机采样 min(C(n,3), 1000)
v42::TriangleResult pv_triangle_verify(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double eps_A, double eps_J, double pass_rate_threshold,
    v42::Logger* logger
);

} // namespace pv

#endif // PV_INTERNAL_H

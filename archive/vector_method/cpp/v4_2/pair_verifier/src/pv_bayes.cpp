// ============================================================================
// pv_bayes.cpp - V4.2 PairVerifier Phase D': 贝叶斯假设验证（Task 5）
//
// 从 V4.1 vector_match_v4_1/src/vm4_bayes.cpp 迁移
// 参考: Lang 2010 Astrometry.net 综述 §5.6
//
// 核心公式:
//   匹配假设 P(数据|H) = Π_i (1/(2πσ²)) × exp(-r_i²/(2σ²))
//   零假设   P(数据|¬H) = (1/A_fov)^n  (随机分布)
//   lnK = Σ_i[-log(2πσ²) - r_i²/(2σ²)] + n×log(A_fov_sqsec)
//   A_fov_sqsec = π × (fov_diag_deg/2)² × 3600²
//
// 决策:
//   lnK > lnK_accept (20.7) → 1 (接受, 强证据)
//   lnK > lnK_weak   (6.9)  → 0 (弱证据)
//   否则                      → -1(拒绝)
//
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "pv_internal.h"

namespace pv {

// ============================================================================
// pv_bayes_verify - 贝叶斯假设验证
// ============================================================================
v42::BayesResult pv_bayes_verify(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double sigma_min, double mad_rms_arcsec,
    double fov_diag_deg,
    double lnK_accept, double lnK_weak,
    v42::Logger* logger)
{
    v42::BayesResult result;
    result.lnK = 0.0;
    result.n_match = (int)matched_pairs.size();
    result.rms_arcsec = 0.0;
    result.sigma = 0.0;
    result.decision = -1;

    // --- 边界检查 ---
    if (matched_pairs.empty()) {
        if (logger) logger->warn("pv_bayes: 匹配对为空, 返回拒绝");
        return result;
    }

    // --- sigma 估计: max(sigma_min, mad_rms) ---
    double sigma = std::max(sigma_min, mad_rms_arcsec);
    if (sigma <= 0.0) sigma = sigma_min;
    if (sigma <= 0.0) {
        if (logger) logger->warn("pv_bayes: sigma 无效, 返回拒绝");
        return result;
    }
    result.sigma = sigma;

    // --- A_fov 计算: π × (fov_diag/2)² × 3600² (平方角秒) ---
    if (fov_diag_deg <= 0.0) {
        if (logger) logger->warn("pv_bayes: fov_diag_deg 无效, 返回拒绝");
        return result;
    }
    double fov_diag_arcsec = fov_diag_deg * DEG_TO_ARCSEC;
    double A_fov_sqsec = PV_PI * (fov_diag_arcsec * 0.5) * (fov_diag_arcsec * 0.5);

    int n = (int)matched_pairs.size();

    // --- 1. 计算残差 RMS ---
    double sum_r2 = 0.0;
    std::vector<double> residuals;
    residuals.reserve(n);
    for (const auto& p : matched_pairs) {
        double dx = p[0] - p[2];  // img_x - cat_x
        double dy = p[1] - p[3];  // img_y - cat_y
        double r = std::sqrt(dx * dx + dy * dy);
        residuals.push_back(r);
        sum_r2 += r * r;
    }
    result.rms_arcsec = std::sqrt(sum_r2 / n);

    // --- 2. 匹配假设对数似然 lnL_match ---
    //    lnL_match = Σ_i [ -log(2πσ²) - r_i²/(2σ²) ]
    double sigma_sq = sigma * sigma;
    double log_2pi_sigma_sq = std::log(2.0 * PV_PI * sigma_sq);
    double lnL_match = 0.0;
    for (double r : residuals) {
        lnL_match += -log_2pi_sigma_sq - (r * r) / (2.0 * sigma_sq);
    }

    // --- 3. 零假设对数似然 lnL_null ---
    //    lnL_null = -n × log(A_fov_sqsec)
    double log_A_fov_sqsec = std::log(A_fov_sqsec);
    double lnL_null = -(double)n * log_A_fov_sqsec;

    // --- 4. 对数贝叶斯因子 ---
    result.lnK = lnL_match - lnL_null;

    // --- 5. 决策 ---
    if (result.lnK > lnK_accept) {
        result.decision = 1;  // 接受
    } else if (result.lnK > lnK_weak) {
        result.decision = 0;  // 弱证据
    } else {
        result.decision = -1; // 拒绝
    }

    if (logger) logger->info("pv_bayes: n=" + std::to_string(n) +
        " σ=" + std::to_string(sigma) +
        " RMS=" + std::to_string(result.rms_arcsec) +
        " A_fov=" + std::to_string(A_fov_sqsec) + " sqarcsec" +
        " lnK=" + std::to_string(result.lnK) +
        " decision=" + std::to_string(result.decision));

    return result;
}

} // namespace pv

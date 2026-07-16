// ============================================================================
// vm4_bayes.cpp - V4.0 贝叶斯假设验证模块实现（Task 5）
//
// 参考：Lang 2010 Astrometry.net 综述 §5.6
// 算法：用贝叶斯因子 K 替代简单阈值判定匹配成功，实现"零误报"验证
//
// 核心公式：
//   lnK = Σ_i[-log(2π·σ²) - r_i²/(2·σ²)] + n_match×log(A_fov_sqsec)
//   其中 A_fov_sqsec = A_fov_sqdeg × 3600²（平方度→平方角秒）
//
// 单线程实现，C++17
// ============================================================================

#include "../include/vm4_bayes.h"
#include <cmath>
#include <cstdio>

namespace vm4_1 {

// 圆周率
static constexpr double BAYES_PI = 3.14159265358979323846;
// 度→角秒换算因子
static constexpr double DEG_TO_ARCSEC = 3600.0;
// 平方度→平方角秒换算因子
static constexpr double SQDEG_TO_SQARCSEC = 3600.0 * 3600.0;

// 默认阈值（与 vm4_api.h 中 VM4_1SolveParams 字段一致）
static constexpr double DEFAULT_LNK_ACCEPT = 20.7;  // K > 10⁹
static constexpr double DEFAULT_LNK_WEAK   = 6.9;   // K > 10³

// ----------------------------------------------------------------------------
// compute_bayes_factor - 计算贝叶斯因子
//
// 算法步骤：
//   1. 计算残差 RMS = sqrt(Σ r_i² / n)
//   2. 计算匹配假设对数似然：
//      lnL_match = Σ_i [ -log(2π·σ²) - r_i²/(2·σ²) ]
//   3. 计算零假设对数似然（随机分布）：
//      lnL_null = -n_match × log(A_fov_sqsec)
//      其中 A_fov_sqsec = A_fov_sqdeg × 3600²
//   4. 对数贝叶斯因子：
//      lnK = lnL_match - lnL_null
//          = Σ_i[-log(2π·σ²) - r_i²/(2·σ²)] + n_match×log(A_fov_sqsec)
//   5. 决策：
//      lnK > lnK_accept → 1（接受，强证据）
//      lnK > lnK_weak   → 0（弱证据）
//      否则              → -1（拒绝）
// ----------------------------------------------------------------------------
BayesResult compute_bayes_factor(
    const std::vector<double>& residuals,
    double sigma, double A_fov_sqdeg,
    double lnK_accept, double lnK_weak)
{
    BayesResult result;
    result.lnK = 0.0;
    result.n_match = (int)residuals.size();
    result.rms_arcsec = 0.0;
    result.decision = -1;
    result.sigma = sigma;

    // --- 边界检查 ---
    if (residuals.empty()) {
        fprintf(stderr, "[vm4_1_bayes] 警告: 残差数组为空，返回拒绝\n");
        return result;
    }
    if (sigma <= 0.0) {
        fprintf(stderr, "[vm4_1_bayes] 警告: sigma=%.4f 无效（需>0），返回拒绝\n", sigma);
        return result;
    }
    if (A_fov_sqdeg <= 0.0) {
        fprintf(stderr, "[vm4_1_bayes] 警告: A_fov=%.4f 平方度无效（需>0），返回拒绝\n", A_fov_sqdeg);
        return result;
    }

    int n = (int)residuals.size();

    // 使用默认阈值（若调用方未指定，即传入0或负值）
    double thr_accept = (lnK_accept > 0.0) ? lnK_accept : DEFAULT_LNK_ACCEPT;
    double thr_weak   = (lnK_weak   > 0.0) ? lnK_weak   : DEFAULT_LNK_WEAK;

    // --- 1. 计算残差 RMS ---
    double sum_r2 = 0.0;
    for (double r : residuals) {
        sum_r2 += r * r;
    }
    result.rms_arcsec = std::sqrt(sum_r2 / n);

    // --- 2. 计算匹配假设对数似然 lnL_match ---
    //    lnL_match = Σ_i [ -log(2π·σ²) - r_i²/(2·σ²) ]
    double sigma_sq = sigma * sigma;
    double log_2pi_sigma_sq = std::log(2.0 * BAYES_PI * sigma_sq);
    double lnL_match = 0.0;
    for (double r : residuals) {
        double r_sq = r * r;
        lnL_match += -log_2pi_sigma_sq - r_sq / (2.0 * sigma_sq);
    }

    // --- 3. 计算零假设对数似然 lnL_null ---
    //    零假设：匹配对随机分布在 FOV 内，概率密度 = 1/A_fov
    //    A_fov 需从平方度转换为平方角秒，与残差单位一致
    //    lnL_null = -n_match × log(A_fov_sqsec)
    double A_fov_sqsec = A_fov_sqdeg * SQDEG_TO_SQARCSEC;
    double log_A_fov_sqsec = std::log(A_fov_sqsec);
    double lnL_null = -(double)n * log_A_fov_sqsec;

    // --- 4. 对数贝叶斯因子 lnK = lnL_match - lnL_null ---
    result.lnK = lnL_match - lnL_null;

    // --- 5. 决策 ---
    if (result.lnK > thr_accept) {
        result.decision = 1;  // 接受（强证据）
    } else if (result.lnK > thr_weak) {
        result.decision = 0;  // 弱证据
    } else {
        result.decision = -1; // 拒绝
    }

    // --- 日志输出 ---
    fprintf(stderr, "[vm4_1_bayes] n_match=%d sigma=%.3f\" RMS=%.3f\" "
            "A_fov=%.2f sqdeg (%.3e sqarcsec) | lnL_match=%.2f lnL_null=%.2f "
            "lnK=%.2f decision=%d\n",
            n, sigma, result.rms_arcsec, A_fov_sqdeg, A_fov_sqsec,
            lnL_match, lnL_null, result.lnK, result.decision);

    return result;
}

// ----------------------------------------------------------------------------
// verify_match_bayes - 从匹配对坐标计算残差并验证
//
// 输入：matched_pairs[i] = (img_x, img_y, cat_x, cat_y)，均为角秒
//   img_x, img_y: 图像星点坐标（已变换到与星表同一坐标系）
//   cat_x, cat_y: 星表星点坐标
// 残差：r_i = sqrt((img_x-cat_x)² + (img_y-cat_y)²)
//
// sigma 参数说明：
//   sigma 是预期的位置噪声标准差（角秒），典型值 1.0-3.0
//   调用方可通过 σ = max(0.5, RMS) 或用 V3.5 的 s0 作为下限来估计
//   本函数直接使用输入 sigma，不做内部估计
// ----------------------------------------------------------------------------
BayesResult verify_match_bayes(
    const std::vector<std::array<double,4>>& matched_pairs,
    double sigma, double A_fov_sqdeg,
    double lnK_accept, double lnK_weak)
{
    // --- 从匹配对坐标计算残差 ---
    std::vector<double> residuals;
    residuals.reserve(matched_pairs.size());

    for (const auto& p : matched_pairs) {
        double dx = p[0] - p[2];  // img_x - cat_x
        double dy = p[1] - p[3];  // img_y - cat_y
        double r = std::sqrt(dx * dx + dy * dy);
        residuals.push_back(r);
    }

    fprintf(stderr, "[vm4_1_bayes] verify_match_bayes: %zu 对匹配，sigma=%.3f\"，A_fov=%.2f sqdeg\n",
            matched_pairs.size(), sigma, A_fov_sqdeg);

    // --- 调用 compute_bayes_factor 进行验证 ---
    return compute_bayes_factor(residuals, sigma, A_fov_sqdeg, lnK_accept, lnK_weak);
}

// ----------------------------------------------------------------------------
// compute_bayes_increment - 计算单个候选对的贝叶斯因子增量
//
// 公式：ΔlnK = -ln(2π·σ²) - r²/(2·σ²) + ln(A_fov_sqarcsec)
//
// 数值示例（σ=1.0″, A_fov=10.6平方度=10.6×3600²=1.37×10⁸ 平方角秒）：
//   r=0.5″ → ΔlnK = -1.84 - 0.125 + 17.86 = +15.97  (强接受)
//   r=1.0″ → ΔlnK = -1.84 - 0.5   + 17.86 = +15.52  (接受)
//   r=4.0″ → ΔlnK = -1.84 - 8.0   + 17.86 = +8.02   (接受)
//   r=5.5″ → ΔlnK = -1.84 - 15.125+ 17.86 = +0.90   (弱接受，边界)
//   r=6.0″ → ΔlnK = -1.84 - 18.0  + 17.86 = -1.98   (拒绝)
//   r=10″  → ΔlnK = -1.84 - 50.0  + 17.86 = -34.0   (强烈拒绝)
// ----------------------------------------------------------------------------
BayesIncrementResult compute_bayes_increment(
    double r_arcsec, double sigma, double A_fov_sqarcsec)
{
    BayesIncrementResult result;
    result.r_arcsec = r_arcsec;
    result.delta_lnK = 0.0;
    result.accepted = false;

    // 边界检查：无效输入返回强烈拒绝
    if (sigma <= 0.0 || A_fov_sqarcsec <= 0.0 ||
        !std::isfinite(r_arcsec) || !std::isfinite(sigma) || !std::isfinite(A_fov_sqarcsec)) {
        result.delta_lnK = -1e30;
        return result;
    }

    // 拟合优度项
    double fit_term = -std::log(2.0 * BAYES_PI * sigma * sigma) -
                      (r_arcsec * r_arcsec) / (2.0 * sigma * sigma);
    // 先验惩罚项
    double prior_term = std::log(A_fov_sqarcsec);

    result.delta_lnK = fit_term + prior_term;
    result.accepted = (result.delta_lnK > 0.0);
    return result;
}

} // namespace vm4_1

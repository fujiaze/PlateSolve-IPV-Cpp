#ifndef VM4_BAYES_H
#define VM4_BAYES_H

// ============================================================================
// vm4_bayes.h - V4.0 贝叶斯假设验证模块（Task 5）
//
// 在 Phase D' 用贝叶斯因子 K 替代简单阈值判定匹配成功，实现"零误报"验证。
// 算法参考：Lang 2010 Astrometry.net 综述 §5.6
//
// 核心公式（对数贝叶斯因子）：
//   匹配似然: P(数据|H) = Π_i (1/(2π·σ²)) × exp(-r_i²/(2·σ²))
//   零假设似然: P(数据|¬H) = (1/A_fov)^n_match  (随机分布概率)
//   lnK = Σ_i[-log(2π·σ²) - r_i²/(2·σ²)] + n_match×log(A_fov_sqsec)
//
// 重要：A_fov 需从平方度转换为平方角秒，与残差单位（角秒）保持一致。
//       A_fov_sqsec = A_fov_sqdeg × 3600²
//
// 决策规则：
//   lnK > lnK_accept (20.7, K>10⁹) → 接受（强证据）
//   lnK > lnK_weak   (6.9,  K>10³) → 弱证据
//   否则                              → 拒绝
// ============================================================================

#include <vector>
#include <array>

namespace vm4 {

// 贝叶斯验证结果
struct BayesResult {
    double lnK;          // 对数贝叶斯因子
    int    n_match;      // 匹配对数
    double rms_arcsec;   // 残差RMS(角秒)
    int    decision;     // 1=接受(强证据), 0=弱证据, -1=拒绝
    double sigma;        // 位置噪声标准差(角秒)
};

// 计算贝叶斯因子
//
// 算法（参考综述 §5.6 Astrometry.net）：
//   对每个匹配对 i，残差 r_i (角秒)
//   lnK = Σ_i [-log(2π·σ²) - r_i²/(2·σ²)] + n_match×log(A_fov_sqsec)
//   其中 A_fov_sqsec = A_fov_sqdeg × 3600²
//
// 输入：
//   residuals: 每个匹配对的残差(角秒)数组
//   sigma: 位置噪声标准差(角秒)，典型值 1.0-3.0
//   A_fov_sqdeg: FOV 面积(平方度)，用于零假设
//   lnK_accept: 接受阈值(默认20.7, K>10⁹)
//   lnK_weak: 弱证据阈值(默认6.9, K>10³)
// 输出：BayesResult
BayesResult compute_bayes_factor(
    const std::vector<double>& residuals,
    double sigma, double A_fov_sqdeg,
    double lnK_accept, double lnK_weak);

// 贝叶斯增量结果
struct BayesIncrementResult {
    double delta_lnK;   // 对数贝叶斯因子增量
    double r_arcsec;    // 残差(角秒)
    bool   accepted;    // 是否接受(ΔlnK > 0)
};

// 计算单个候选匹配对的贝叶斯因子增量
//
// 用于 Phase C 扩充阶段对单个候选对做零误报筛选。
// 每个候选对的增量贡献：
//   ΔlnK = -ln(2π·σ²) - r²/(2·σ²) + ln(A_fov_sqarcsec)
//
// 解释：
//   - 拟合优度项: -ln(2π·σ²) - r²/(2·σ²)，残差越小贡献越大
//   - 先验惩罚项: +ln(A_fov_sqarcsec)，FOV越大随机碰巧概率越高
//
// 决策：
//   ΔlnK > 0  → 接受（该对提供正证据）
//   ΔlnK < -5 → 强烈拒绝（r > 3σ）
//   其他       → 弱拒绝
//
// 输入：
//   r_arcsec: 候选对残差(角秒)
//   sigma: 期望残差标准差(角秒)，取Phase B的RMS或默认1.0
//   A_fov_sqarcsec: FOV面积(平方角秒)，= W*H*s0²/3600²
// 输出：BayesIncrementResult
BayesIncrementResult compute_bayes_increment(
    double r_arcsec, double sigma, double A_fov_sqarcsec);

// 便捷函数：从匹配对坐标计算残差并验证
//
// 输入：匹配对 (u_img[i], w_cat[j]) 已变换到同一坐标系
//       即 u_img[i] 与 transform(w_cat[j]) 的差为残差
//       matched_pairs: (img_x, img_y, cat_x, cat_y) 数组，均为角秒
//       残差 r_i = sqrt((img_x-cat_x)² + (img_y-cat_y)²)
//       sigma, A_fov, 阈值
BayesResult verify_match_bayes(
    const std::vector<std::array<double,4>>& matched_pairs,
    double sigma, double A_fov_sqdeg,
    double lnK_accept, double lnK_weak);

} // namespace vm4

#endif // VM4_BAYES_H

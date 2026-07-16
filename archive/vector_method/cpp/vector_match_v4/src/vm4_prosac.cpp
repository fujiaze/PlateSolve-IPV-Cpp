// ============================================================================
// vm4_prosac.cpp - V4.0 Phase A PROSAC 优先采样模块实现（Task 4）
//
// 实现 Chum 2005 综述的 PROSAC 思想：
//   1. 按星点质量分降序排列（SNR + 稀疏度 + 饱和度加权）
//   2. 增长函数 g(t)=n×(t/T_max)^(1/3) 渐进扩大采样池
//   3. 70% 概率从前 g(t) 颗高质量星采样，30% 概率全星均匀采样
//
// 与 vm4_core.cpp 解耦：本模块仅提供质量分计算与采样器，
// Task 7 负责在 record_and_filter 中替换 V3.5 的稀疏度加权抽样逻辑。
//
// C++17，单线程（采样器内部维护 rng_）
// ============================================================================

#include "../include/vm4_prosac.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace vm4 {

// ----------------------------------------------------------------------------
// 内部工具：归一化数组到 [0,1]
// 输入: values[N]
// 输出: norm[N] = (v - min) / (max - min)，max==min 时全部置 0
// ----------------------------------------------------------------------------
static std::vector<double> normalize_minmax(const std::vector<double>& values) {
    const size_t N = values.size();
    std::vector<double> norm(N, 0.0);
    if (N == 0) return norm;

    double vmin =  std::numeric_limits<double>::infinity();
    double vmax = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < N; ++i) {
        if (values[i] < vmin) vmin = values[i];
        if (values[i] > vmax) vmax = values[i];
    }
    double range = vmax - vmin;
    if (range <= 0.0) {
        // 所有值相同：归一化为 0（避免除零）
        return norm;
    }
    for (size_t i = 0; i < N; ++i) {
        norm[i] = (values[i] - vmin) / range;
    }
    return norm;
}

// ============================================================================
// compute_quality_score - 计算每颗星的质量分并按降序排列
//
// q_i = w_snr×normalize(SNR) + w_sparse×normalize(sparsity) + w_sat×is_saturated
//   - SNR: 信噪比，越高=越亮越锐，归一化后直接加权
//   - sparsity: 第3近邻距离(角秒)，越大=越孤立=匹配歧义越小，归一化后直接加权
//   - is_saturated: 饱和星加成 w_sat×1.0
// ============================================================================
std::vector<StarQuality> compute_quality_score(
    const std::vector<double>& snr,
    const std::vector<double>& sparsity,
    const std::vector<bool>& is_saturated,
    double w_snr, double w_sparse, double w_sat)
{
    const size_t N = snr.size();
    std::vector<StarQuality> result;
    if (N == 0) return result;
    result.reserve(N);

    // 输入长度一致性校验（防御性，sparsity/is_saturated 长度应与 snr 一致）
    const size_t Ns  = sparsity.size();
    const size_t Nsa = is_saturated.size();

    // 归一化 SNR 与 sparsity 到 [0,1]
    auto norm_snr     = normalize_minmax(snr);
    auto norm_sparsity = normalize_minmax(sparsity);

    // 加权求和构建 StarQuality
    for (size_t i = 0; i < N; ++i) {
        StarQuality sq;
        sq.index        = static_cast<int>(i);
        sq.snr          = (i < N) ? snr[i] : 0.0;
        sq.sparsity     = (i < Ns) ? sparsity[i] : 0.0;
        sq.is_saturated = (i < Nsa) ? is_saturated[i] : false;

        double sat_term = sq.is_saturated ? (w_sat * 1.0) : 0.0;
        sq.quality_score = w_snr * norm_snr[i] + w_sparse * norm_sparsity[i] + sat_term;

        result.push_back(std::move(sq));
    }

    // 按质量分降序排列（稳定排序：相同质量分保持原始顺序）
    std::stable_sort(result.begin(), result.end(),
        [](const StarQuality& a, const StarQuality& b) {
            return a.quality_score > b.quality_score;
        });

    fprintf(stderr, "[vm4_prosac] compute_quality_score: N=%zu, "
            "w_snr=%.2f w_sparse=%.2f w_sat=%.2f, "
            "top_score=%.4f tail_score=%.4f\n",
            N, w_snr, w_sparse, w_sat,
            result.front().quality_score, result.back().quality_score);

    return result;
}

// ============================================================================
// prosac_pool_size - PROSAC 增长函数 g(t)
//
// g(t) = n × (t / T_max)^(1/3)
// 返回 min(n, max(1, ceil(g(t))))
// 注意: t 从 1 开始计数；t<=0 或 T_max<=0 时返回 1（安全降级）
// ============================================================================
int prosac_pool_size(int t, int n, int T_max) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (t <= 0 || T_max <= 0) return 1;

    double ratio = static_cast<double>(t) / static_cast<double>(T_max);
    if (ratio >= 1.0) return n;  // t >= T_max 时全星采样

    double g = static_cast<double>(n) * std::cbrt(ratio);  // cbrt 比 pow(x,1/3) 更稳
    int pool = static_cast<int>(std::ceil(g));
    if (pool < 1) pool = 1;
    if (pool > n) pool = n;
    return pool;
}

// ============================================================================
// ProsacSampler 实现
// ============================================================================

void ProsacSampler::init(const std::vector<StarQuality>& sorted_stars,
                         int T_max, unsigned seed) {
    const size_t N = sorted_stars.size();
    sorted_indices_.clear();
    sorted_indices_.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        sorted_indices_.push_back(sorted_stars[i].index);
    }
    n_ = static_cast<int>(N);
    T_max_ = (T_max > 0) ? T_max : 10000;
    last_pool_size_ = 1;
    rng_.seed(seed);

    // 计算质量分中位数（sorted_stars 已按 quality_score 降序，中位数取中间元素）
    quality_median_ = 0.0;
    if (N > 0) {
        if (N % 2 == 1) {
            quality_median_ = sorted_stars[N / 2].quality_score;
        } else {
            quality_median_ = 0.5 * (sorted_stars[N / 2 - 1].quality_score +
                                     sorted_stars[N / 2].quality_score);
        }
    }

    fprintf(stderr, "[vm4_prosac] ProsacSampler::init: n=%d T_max=%d "
            "quality_median=%.4f seed=%u\n",
            n_, T_max_, quality_median_, seed);
}

int ProsacSampler::sample(int t) const {
    if (n_ <= 0) return -1;
    if (n_ == 1) { last_pool_size_ = 1; return sorted_indices_[0]; }

    // 计算当前采样池大小
    int pool = prosac_pool_size(t, n_, T_max_);
    last_pool_size_ = pool;

    // p_guided=0.7: 70% 从前 pool 颗高质量星采样, 30% 从全部 n 颗均匀采样
    static constexpr double p_guided = 0.7;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double r = u01(rng_);

    int chosen_rank;  // 在 sorted_indices_ 中的下标
    if (r < p_guided) {
        // 开发：从前 pool 颗高质量星中均匀随机选 1
        std::uniform_int_distribution<int> ud(0, pool - 1);
        chosen_rank = ud(rng_);
    } else {
        // 探索：从全部 n 颗星中均匀随机选 1
        std::uniform_int_distribution<int> ud(0, n_ - 1);
        chosen_rank = ud(rng_);
    }
    return sorted_indices_[chosen_rank];
}

int ProsacSampler::last_pool_size() const {
    return last_pool_size_;
}

double ProsacSampler::quality_median() const {
    return quality_median_;
}

// ============================================================================
// prosac_sample_one - 便捷函数：从已初始化的采样器中采样一颗星
// 供 Task 7 在 record_and_filter 中替换 V3.5 的稀疏度加权抽样
// ============================================================================
int prosac_sample_one(const ProsacSampler& sampler, int t) {
    return sampler.sample(t);
}

} // namespace vm4

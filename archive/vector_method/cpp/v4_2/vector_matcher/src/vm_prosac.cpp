// ============================================================================
// vm_prosac.cpp - V4.2 VectorMatcher PROSAC 优先采样实现（Task 3）
//
// 从 V4.1 的 vm4_prosac.cpp 迁移，重命名 namespace 为 v42。
//
// 实现 Chum 2005 综述的 PROSAC 思想:
//   1. 按星点质量分降序排列(SNR + 稀疏度 + 饱和度加权)
//   2. 增长函数 g(t)=n×(t/T_max)^(1/3) 渐进扩大采样池
//   3. 70% 概率从前 g(t) 颗高质量星采样, 30% 概率全星均匀采样
//
// C++17, 单线程(采样器内部维护 rng_)
// ============================================================================

#include "../include/vm_prosac.h"
#include "../common/v42_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace v42 {

// ----------------------------------------------------------------------------
// 内部工具: 归一化数组到 [0,1]
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

    const size_t Ns  = sparsity.size();
    const size_t Nsa = is_saturated.size();

    auto norm_snr     = normalize_minmax(snr);
    auto norm_sparsity = normalize_minmax(sparsity);

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

    std::stable_sort(result.begin(), result.end(),
        [](const StarQuality& a, const StarQuality& b) {
            return a.quality_score > b.quality_score;
        });

    fprintf(stderr, "[v42_vm_prosac] compute_quality_score: N=%zu, "
            "w_snr=%.2f w_sparse=%.2f w_sat=%.2f, "
            "top_score=%.4f tail_score=%.4f\n",
            N, w_snr, w_sparse, w_sat,
            result.front().quality_score, result.back().quality_score);

    return result;
}

// ============================================================================
// prosac_pool_size - PROSAC 增长函数 g(t)
// ============================================================================
int prosac_pool_size(int t, int n, int T_max) {
    if (n <= 0) return 0;
    if (n == 1) return 1;
    if (t <= 0 || T_max <= 0) return 1;

    double ratio = static_cast<double>(t) / static_cast<double>(T_max);
    if (ratio >= 1.0) return n;

    double g = static_cast<double>(n) * std::cbrt(ratio);
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

    quality_median_ = 0.0;
    if (N > 0) {
        if (N % 2 == 1) {
            quality_median_ = sorted_stars[N / 2].quality_score;
        } else {
            quality_median_ = 0.5 * (sorted_stars[N / 2 - 1].quality_score +
                                     sorted_stars[N / 2].quality_score);
        }
    }

    fprintf(stderr, "[v42_vm_prosac] ProsacSampler::init: n=%d T_max=%d "
            "quality_median=%.4f seed=%u\n",
            n_, T_max_, quality_median_, seed);
}

int ProsacSampler::sample(int t) const {
    if (n_ <= 0) return -1;
    if (n_ == 1) { last_pool_size_ = 1; return sorted_indices_[0]; }

    int pool = prosac_pool_size(t, n_, T_max_);
    last_pool_size_ = pool;

    static constexpr double p_guided = 0.7;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    double r = u01(rng_);

    int chosen_rank;
    if (r < p_guided) {
        std::uniform_int_distribution<int> ud(0, pool - 1);
        chosen_rank = ud(rng_);
    } else {
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
// prosac_sample_one - 便捷函数
// ============================================================================
int prosac_sample_one(const ProsacSampler& sampler, int t) {
    return sampler.sample(t);
}

} // namespace v42

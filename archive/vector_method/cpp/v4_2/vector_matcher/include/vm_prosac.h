#ifndef VM42_PROSAC_H
#define VM42_PROSAC_H

// ============================================================================
// vm_prosac.h - V4.2 VectorMatcher 内部 PROSAC 接口（Task 3）
//
// 基于 Chum 2005 综述的 PROSAC 思想：
//   按星点质量分降序排列，优先从高质量星采样，加速 RANSAC 收敛。
//
// 质量分构成:
//   q_i = w_snr × normalize(SNR) + w_sparse × normalize(sparsity) + w_sat × is_saturated
//   归一化: (x - min) / (max - min) → [0,1]
//
// PROSAC 增长函数:
//   g(t) = n × (t / T_max)^(1/3)
//   第 t 次抽样时采样池大小 = min(n, ceil(g(t)))
//
// 采样策略(p_guided=0.7):
//   70% 概率从前 g(t) 颗高质量星中随机选 1 点(开发)
//   30% 概率从全部 n 颗星中均匀随机选 1 点(探索)
// ============================================================================

#include <vector>
#include <random>

namespace v42 {

// 星点质量信息
struct StarQuality {
    int    index;         // 原始索引(在 U 数组中的位置)
    double snr;           // 信噪比
    double sparsity;      // 第3近邻距离(角秒)
    bool   is_saturated;  // 是否饱和星
    double quality_score; // 综合质量分(越大越优)
};

// 计算每颗星的质量分并按降序排列
std::vector<StarQuality> compute_quality_score(
    const std::vector<double>& snr,
    const std::vector<double>& sparsity,
    const std::vector<bool>& is_saturated,
    double w_snr, double w_sparse, double w_sat);

// PROSAC 增长函数 g(t)
int prosac_pool_size(int t, int n, int T_max);

// PROSAC 采样器
class ProsacSampler {
public:
    void init(const std::vector<StarQuality>& sorted_stars, int T_max, unsigned seed);
    int  sample(int t) const;
    int  last_pool_size() const;
    double quality_median() const;

private:
    std::vector<int> sorted_indices_;
    int    T_max_       = 10000;
    int    n_           = 0;
    mutable int    last_pool_size_ = 1;
    double quality_median_ = 0.0;
    mutable std::mt19937 rng_;
};

int prosac_sample_one(const ProsacSampler& sampler, int t);

} // namespace v42

#endif // VM42_PROSAC_H

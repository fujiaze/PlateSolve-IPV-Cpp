#ifndef VM4_1_PROSAC_H
#define VM4_1_PROSAC_H

// ============================================================================
// vm4_prosac.h - V4.0 Phase A PROSAC 优先采样模块（Task 4）
//
// 基于 Chum 2005 综述的 PROSAC（Progressive Sample Consensus）思想：
//   按星点质量分降序排列，优先从高质量星采样，加速 RANSAC 收敛。
//
// 质量分构成：
//   q_i = w_snr × normalize(SNR) + w_sparse × normalize(sparsity) + w_sat × is_saturated
//   归一化: (x - min) / (max - min) → [0,1]
//
// PROSAC 增长函数：
//   g(t) = n × (t / T_max)^(1/3)
//   第 t 次抽样时采样池大小 = min(n, ceil(g(t)))
//
// 采样策略（p_guided=0.7）：
//   70% 概率从前 g(t) 颗高质量星中随机选 1 点（开发）
//   30% 概率从全部 n 颗星中均匀随机选 1 点（探索）
//
// 本模块仅实现采样逻辑，不修改 vm4_core.cpp；Task 7 负责集成替换
// V3.5 的稀疏度加权抽样。
//
// C++17，单线程（采样器内部维护 rng）
// ============================================================================

#include <vector>
#include <random>

namespace vm4_1 {

// 星点质量信息
struct StarQuality {
    int    index;         // 原始索引（在 U 数组中的位置）
    double snr;           // 信噪比
    double sparsity;      // 第3近邻距离(角秒)
    bool   is_saturated;  // 是否饱和星
    double quality_score; // 综合质量分（越大越优）
};

// 计算每颗星的质量分并按降序排列
// q_i = w_snr×normalize(SNR) + w_sparse×normalize(sparsity) + w_sat×is_saturated
// 归一化: (x - min)/(max - min) → [0,1]
// 输入: snr[N], sparsity[N], is_saturated[N], 权重 w_snr, w_sparse, w_sat
// 输出: 按 quality_score 降序排列的 StarQuality 数组
std::vector<StarQuality> compute_quality_score(
    const std::vector<double>& snr,
    const std::vector<double>& sparsity,
    const std::vector<bool>& is_saturated,
    double w_snr, double w_sparse, double w_sat);

// PROSAC 增长函数 g(t)
// g(t) = n × (t/T_max)^(1/3)
// 返回第 t 次抽样时的采样池大小: min(n, max(1, ceil(g(t))))
// 注意: t 从 1 开始计数
int prosac_pool_size(int t, int n, int T_max);

// PROSAC 采样器
class ProsacSampler {
public:
    // 初始化：传入按质量分降序排列的星索引，T_max 为最大抽样次数
    void init(const std::vector<StarQuality>& sorted_stars, int T_max, unsigned seed);

    // 第 t 次采样：返回选中的星索引（在原始 U 数组中的位置）
    // 算法:
    //   r = uniform_random(0,1)
    //   if r < p_guided (默认0.7): 从前 g(t) 颗高质量星中随机选 1 点
    //   else: 从全部 n 颗星中均匀随机选 1 点（保持探索）
    // 注: 设为 const 以支持 prosac_sample_one(const ProsacSampler&, t)，
    //     采样不改变采样器的逻辑配置（排序/T_max/n/中位数），仅推进 rng 与缓存池大小
    int sample(int t) const;

    // 调试信息：上一次采样使用的采样池大小
    int last_pool_size() const;

    // 调试信息：排序后质量分的中位数
    double quality_median() const;

private:
    std::vector<int> sorted_indices_;  // 按质量降序排列的原始索引
    int    T_max_       = 10000;       // 最大抽样次数
    int    n_           = 0;           // 星点总数
    mutable int    last_pool_size_ = 1;     // 上一次采样的采样池大小
    double quality_median_ = 0.0;      // 质量分中位数
    mutable std::mt19937 rng_;              // 随机数生成器
};

// 便捷函数：从图像星中 PROSAC 采样一颗星
// 输入: 已初始化的采样器, 当前抽样次数 t
// 输出: 选中的星在原始 U 数组中的索引
// 供 Task 7 替换 V3.5 的稀疏度加权抽样
int prosac_sample_one(const ProsacSampler& sampler, int t);

} // namespace vm4_1

#endif // VM4_1_PROSAC_H

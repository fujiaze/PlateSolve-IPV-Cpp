// ============================================================================
// vm4_kvector.cpp - V4.0 Phase C k-vector 快速角距索引实现（Task 3）
//
// 实现要点:
//   - build(): 双重循环计算所有 C(n,2) 星对角距，std::sort 排序，建立 k-vector
//   - query(): O(1) 定位 k1,k2 区间，线性扫描 sorted_pairs_ 二次过滤
//   - 角距用欧氏距离（平面坐标已投影到角秒空间，小 FOV 下近似等于球面角距）
//   - 建索引耗时用 std::chrono 测量
//   - OpenMP 可选加速 build 阶段距离计算
// ============================================================================

#include "vm4_kvector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vm4_1 {

// ============================================================================
// build - 构建 k-vector 索引
// ============================================================================
void KVectorIndex::build(const std::vector<std::pair<double,double>>& stars)
{
    auto t_start = std::chrono::high_resolution_clock::now();

    const int n = (int)stars.size();
    K_ = (n > 1) ? n * (n - 1) / 2 : 0;
    sorted_pairs_.clear();
    kvector_.clear();
    d_min_ = d_max_ = 0.0;
    a_ = b_ = 0.0;

    if (K_ <= 0) {
        build_time_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t_start).count();
        return;
    }

    // 1. 预分配并计算所有星对角距
    sorted_pairs_.reserve(K_);
    for (int i = 0; i < n - 1; ++i) {
        const double xi = stars[i].first;
        const double eta = stars[i].second;
        for (int j = i + 1; j < n; ++j) {
            const double dx = xi   - stars[j].first;
            const double dy = eta  - stars[j].second;
            StarPair sp;
            sp.i = i;
            sp.j = j;
            sp.distance = std::sqrt(dx*dx + dy*dy);
            sorted_pairs_.push_back(sp);
        }
    }

    // 2. 按角距升序排序
    std::sort(sorted_pairs_.begin(), sorted_pairs_.end(),
        [](const StarPair& a, const StarPair& b) {
            return a.distance < b.distance;
        });

    // 3. 取 d_min / d_max
    d_min_ = sorted_pairs_.front().distance;
    d_max_ = sorted_pairs_.back().distance;

    // 4. 线性映射参数: k(d) = floor(a + b*(d - d_min))
    //    使得 k(d_min) = 0, k(d_max) = K-1
    a_ = 0.0;
    if (d_max_ > d_min_ && K_ > 1) {
        b_ = (double)(K_ - 1) / (d_max_ - d_min_);
    } else {
        // 退化情况（所有角距相等或只有一对）
        b_ = 0.0;
    }

    // 5. 构建 kvector 数组
    //    kvector_[k] = 最小的 pair 索引 i，使得 k(sorted_pairs_[i].distance) >= k
    //    数组长度 K_+1，末项哨兵 = K_
    kvector_.assign(K_ + 1, K_);
    if (b_ > 0.0) {
        int k_prev = 0;
        // 第 0 项必为 0（k(d_min)=0）
        kvector_[0] = 0;
        for (int idx = 0; idx < K_; ++idx) {
            int k_cur = (int)std::floor(b_ * (sorted_pairs_[idx].distance - d_min_));
            if (k_cur < 0) k_cur = 0;
            if (k_cur > K_ - 1) k_cur = K_ - 1;
            // 填充 (k_prev, k_cur] 区间未赋值项为当前 idx（首个 >=k 的 pair）
            for (int k = k_prev + 1; k <= k_cur; ++k) {
                if (kvector_[k] == K_) kvector_[k] = idx;
            }
            k_prev = k_cur;
        }
        // 末尾哨兵保持 K_
    } else {
        // 退化：所有 pair 角距相等，全部映射到 k=0
        kvector_[0] = 0;
    }

    build_time_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - t_start).count();
}

// ============================================================================
// query - 查询角距在 [d-eps, d+eps] 的所有星对
// ============================================================================
std::vector<StarPair> KVectorIndex::query(double d, double eps) const
{
    std::vector<StarPair> result;
    if (K_ <= 0 || b_ <= 0.0) return result;

    const double d_lo = d - eps;
    const double d_hi = d + eps;

    // 早期排除：查询区间与 [d_min_, d_max_] 无交集
    if (d_lo > d_max_ || d_hi < d_min_) return result;

    // k1 = max(0, floor(b*(d-eps-d_min)))
    int k1 = (int)std::floor(b_ * (d_lo - d_min_));
    if (k1 < 0) k1 = 0;
    if (k1 > K_ - 1) k1 = K_ - 1;

    // k2 = min(K-1, floor(b*(d+eps-d_min)))
    int k2 = (int)std::floor(b_ * (d_hi - d_min_));
    if (k2 < 0) k2 = 0;
    if (k2 > K_ - 1) k2 = K_ - 1;

    // 通过 kvector_ 将 k 区间转换为 sorted_pairs_ 实际索引区间
    // pair 区间: [kvector_[k1], kvector_[k2+1])
    int pair_lo = kvector_[k1];
    int pair_hi = kvector_[k2 + 1];  // exclusive（k2+1 项为哨兵 K_）

    // 线性扫描并过滤实际角距在 [d_lo, d_hi] 的星对
    result.reserve(pair_hi - pair_lo);
    for (int p = pair_lo; p < pair_hi; ++p) {
        const double dist = sorted_pairs_[p].distance;
        if (dist >= d_lo && dist <= d_hi) {
            result.push_back(sorted_pairs_[p]);
        }
    }
    return result;
}

// ============================================================================
// kvector_prefilter - 用 k-vector 预筛选 Phase C 候选对
// 算法：
//   对每对图像星 (i, j')，计算其角距 d_ij'，用 kv_w 查询星表 W 中角距在
//   [d_ij'-eps, d_ij'+eps] 的星对 (p, q)。
//   于是 (U[i]↔W[p], U[j']↔W[q]) 或 (U[i]↔W[q], U[j']↔W[p]) 构成候选匹配。
//   为避免重复，本函数返回 (i, p) 与 (i, q) 形式的候选对（i 为图像星索引，
//   p/q 为星表索引），由后续 NN 验证阶段做精细匹配。
// ============================================================================
std::vector<std::pair<int,int>> kvector_prefilter(
    const KVectorIndex& kv_w,
    const double* U, int N,
    const double* W, int M,
    double eps)
{
    std::vector<std::pair<int,int>> candidates;
    (void)W;  // W 已编码在 kv_w 中，仅保留接口签名供 Task 7 集成使用
    if (N < 2 || M < 2 || kv_w.size() == 0) return candidates;

    // 用 unordered 去重 (i, w) 候选
    // 注意：N, M 通常 ≤2000，候选规模有限，使用排序+unique 即可去重
    for (int i = 0; i < N - 1; ++i) {
        const double uxi = U[i*2];
        const double ueta = U[i*2 + 1];
        for (int jp = i + 1; jp < N; ++jp) {
            const double dx = uxi  - U[jp*2];
            const double dy = ueta - U[jp*2 + 1];
            const double d_ij = std::sqrt(dx*dx + dy*dy);

            // 用 k-vector 在 W 中查询角距匹配的星对
            auto pairs = kv_w.query(d_ij, eps);
            for (const auto& sp : pairs) {
                // (i ↔ sp.i, jp ↔ sp.j) 或 (i ↔ sp.j, jp ↔ sp.i)
                candidates.push_back({i, sp.i});
                candidates.push_back({i, sp.j});
                candidates.push_back({jp, sp.i});
                candidates.push_back({jp, sp.j});
            }
        }
    }

    // 去重 (i, w) 候选对
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());

    return candidates;
}

} // namespace vm4_1

#ifndef VM4_1_KVECTOR_H
#define VM4_1_KVECTOR_H

// ============================================================================
// vm4_kvector.h - V4.0 Phase C k-vector 快速角距索引（Task 3）
//
// 算法原理（Mortari 1997）:
//   对排序的角距数组建立分段线性映射函数 k(d)=floor(a+b·d)，使任意角距
//   区间查询 O(k) 完成，与星表大小 n 解耦（n≤2000 时 K≈2×10⁶，内存可控）。
//
// 在 Phase C 全局 NN 匹配中用作预筛选器：先 k-vector 检索候选星对，再做
// 精细验证，将 O(n²) 全配对扫描降为 O(k) 候选+精细验证。
//
// 本模块独立于 vm4_core.cpp，Task 7 集成时通过 kvector_prefilter() 调用。
// ============================================================================

#include <vector>
#include <utility>

namespace vm4_1 {

// 星对索引：(i,j) 表示第 i 颗和第 j 颗星的星对
struct StarPair {
    int    i;           // 第一颗星索引
    int    j;           // 第二颗星索引
    double distance;    // 角距(角秒)
};

class KVectorIndex {
public:
    KVectorIndex() = default;
    ~KVectorIndex() = default;

    // ----------------------------------------------------------------------
    // 构建索引
    // 输入：stars (xi, eta) 角秒坐标数组（以中心为原点的平面坐标）
    // 算法：
    //   1. 计算所有星对角距 d_ij = sqrt((xi_i-xi_j)² + (eta_i-eta_j)²)
    //   2. 按 d 排序，得到 sorted_pairs[K], K=n*(n-1)/2
    //   3. d_min = sorted_pairs[0].distance, d_max = sorted_pairs[K-1].distance
    //   4. 建立线性映射：k(d) = floor(a + b*(d - d_min))
    //      其中 b = (K-1)/(d_max - d_min), a = 0
    //      使得 k(d_min)=0, k(d_max)=K-1
    //   5. kvector 数组：kvector[k] = 该区间内第一个 pair 的索引
    // 注意：n≤2000 时 K≈2×10⁶，内存可控；OpenMP 加速距离计算
    // ----------------------------------------------------------------------
    void build(const std::vector<std::pair<double,double>>& stars);

    // ----------------------------------------------------------------------
    // 查询角距在 [d-eps, d+eps] 的所有星对
    // 算法：
    //   k1 = max(0, floor(a + b*(d-eps-d_min)))
    //   k2 = min(K-1, floor(a + b*(d+eps-d_min)))
    //   通过 kvector[k1..k2+1] 定位 sorted_pairs 的实际索引区间
    //   线性扫描并过滤实际角距在 [d-eps, d+eps] 的星对
    //   （k-vector 是近似映射，需二次过滤）
    // 复杂度：O(k) 即候选区间大小，与星表大小 n 解耦
    // ----------------------------------------------------------------------
    std::vector<StarPair> query(double d, double eps) const;

    // 调试信息
    int    size()          const { return K_; }           // 星对总数 K
    double min_distance()  const { return d_min_; }       // d_min
    double max_distance()  const { return d_max_; }       // d_max
    double build_time_ms() const { return build_time_ms_; }// 建索引耗时(ms)

private:
    std::vector<StarPair> sorted_pairs_;  // 按角距排序的星对
    std::vector<int>      kvector_;       // k-vector 索引数组 (K+1 项, 末尾哨兵=K)
    double d_min_ = 0.0;
    double d_max_ = 0.0;
    double a_ = 0.0;       // 线性映射参数（截距，固定为0）
    double b_ = 0.0;       // 线性映射参数（斜率）
    double build_time_ms_ = 0.0;
    int    K_ = 0;         // 星对总数
};

// ============================================================================
// 便捷预筛选函数（供 Task 7 集成 Phase C 调用）
// 用 k-vector 预筛选 Phase C 候选对
// 输入：
//   kv_w  : 已构建好的星表 W 的 k-vector 索引
//   U[N*2]: 图像星点坐标 (xi,eta) 角秒，N 颗
//   W[M*2]: 星表坐标 (xi,eta) 角秒，M 颗（kv_w 即基于此构建）
//   eps   : 角距容差（角秒，建议取 params.k_vector_eps）
// 输出：候选匹配对列表 (i, j) 即 U[i] 可能对应 W[j]
// 算法：对每颗 U[i]，计算其与其他 U[j'] 的角距 d_ij'，用 kv_w 查询 W 中
//       角距在 [d_ij'-eps, d_ij'+eps] 的星对 (p,q)，于是 (i,p) 与 (j',q)
//       （或 (i,q) 与 (j',p)）构成候选匹配。
// 注意：本函数不修改 vm4_core.cpp，仅提供集成接口。
// ============================================================================
std::vector<std::pair<int,int>> kvector_prefilter(
    const KVectorIndex& kv_w,
    const double* U, int N,
    const double* W, int M,
    double eps);

} // namespace vm4_1

#endif // VM4_1_KVECTOR_H

// ============================================================================
// pv_triangle.cpp - V4.2 PairVerifier Phase D': 三角形双特征验证（Task 5）
//
// 从 V4.1 vector_match_v4_1/src/vm4_triangle.cpp 迁移
// 参考: Cole 2006（三角形不变量在 platesolve 中的应用综述）
//
// 算法:
//   对每个三元组 (i,j,k):
//     图像侧三角形: (mp[i][0],mp[i][1]), (mp[j][0],mp[j][1]), (mp[k][0],mp[k][1])
//     星表侧三角形: (mp[i][2],mp[i][3]), (mp[j][2],mp[j][3]), (mp[k][2],mp[k][3])
//     特征1: 面积 A（海伦公式）
//     特征2: 极惯性矩 J = A×(a²+b²+c²)/36
//     通过条件: |A_img-A_cat|/max(A_img,A_cat) < eps_A
//               |J_img-J_cat|/max(J_img,J_cat) < eps_J
//   退化三角形(A<1e-6)不计入总数
//   n ≤ 30: 遍历所有 C(n,3)
//   n > 30: 随机采样 min(C(n,3), 1000) 个组合
//
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "pv_internal.h"

#include <random>
#include <unordered_set>
#include <cstdint>

namespace pv {

// --- 常量 ---
static constexpr double DEGEN_AREA_EPS = 1e-6;      // 退化三角形面积阈值
static constexpr int    MAX_SAMPLES_N_LARGE = 1000; // 大n随机采样上限
static constexpr int    N_LARGE_THRESHOLD   = 30;   // 触发随机采样的n阈值
static constexpr uint32_t RNG_SEED = 0x5A5A5A5A;    // 固定随机种子

// --- 三角形特征 ---
struct TriangleFeatures {
    double area;    // 面积(角秒²)
    double moment;  // 极惯性矩 J = A×(a²+b²+c²)/36
    double a, b, c; // 三边长(角秒), a≤b≤c
};

// ----------------------------------------------------------------------------
// compute_triangle_features - 计算三角形面积和极惯性矩
// ----------------------------------------------------------------------------
static TriangleFeatures compute_triangle_features(
    double x1, double y1, double x2, double y2, double x3, double y3)
{
    TriangleFeatures f{0.0, 0.0, 0.0, 0.0, 0.0};

    // 两两欧氏距离
    double d12 = std::hypot(x1 - x2, y1 - y2);
    double d23 = std::hypot(x2 - x3, y2 - y3);
    double d13 = std::hypot(x1 - x3, y1 - y3);

    // 排序使 a ≤ b ≤ c
    double s3[3] = {d12, d23, d13};
    std::sort(s3, s3 + 3);
    f.a = s3[0]; f.b = s3[1]; f.c = s3[2];

    // 海伦公式
    double s = (f.a + f.b + f.c) * 0.5;
    double tmp = s * (s - f.a) * (s - f.b) * (s - f.c);
    if (tmp <= 0.0) return TriangleFeatures{0.0, 0.0, 0.0, 0.0, 0.0};
    f.area = std::sqrt(tmp);

    // 退化三角形跳过
    if (f.area < DEGEN_AREA_EPS)
        return TriangleFeatures{0.0, 0.0, f.a, f.b, f.c};

    // 极惯性矩 J = A×(a²+b²+c²)/36
    f.moment = f.area * (f.a * f.a + f.b * f.b + f.c * f.c) / 36.0;
    return f;
}

// ----------------------------------------------------------------------------
// 内部: 将 (i,j,k) 编码为唯一 uint64_t 键（i<j<k）
// ----------------------------------------------------------------------------
static inline uint64_t encode_triple(int i, int j, int k) {
    return ((uint64_t)i << 42) | ((uint64_t)j << 21) | (uint64_t)k;
}

// ----------------------------------------------------------------------------
// 内部: 对单个三角形组合验证
//   返回: 0=通过, 1=不通过, -1=退化(不计入)
// ----------------------------------------------------------------------------
static int check_one_triangle(
    const std::vector<std::array<double, 4>>& mp,
    int i, int j, int k, double eps_A, double eps_J)
{
    TriangleFeatures fi = compute_triangle_features(
        mp[i][0], mp[i][1], mp[j][0], mp[j][1], mp[k][0], mp[k][1]);
    TriangleFeatures fc = compute_triangle_features(
        mp[i][2], mp[i][3], mp[j][2], mp[j][3], mp[k][2], mp[k][3]);

    if (fi.area < DEGEN_AREA_EPS || fc.area < DEGEN_AREA_EPS) return -1;

    double maxA = std::max(fi.area, fc.area);
    double maxJ = std::max(fi.moment, fc.moment);
    if (maxA < DEGEN_AREA_EPS) return -1;
    if (maxJ < DEGEN_AREA_EPS) return -1;

    double rel_A = std::abs(fi.area - fc.area) / maxA;
    double rel_J = std::abs(fi.moment - fc.moment) / maxJ;

    if (rel_A < eps_A && rel_J < eps_J) return 0;
    return 1;
}

// ============================================================================
// pv_triangle_verify - 三角形双特征验证主函数
// ============================================================================
v42::TriangleResult pv_triangle_verify(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double eps_A, double eps_J, double pass_rate_threshold,
    v42::Logger* logger)
{
    v42::TriangleResult r;
    r.total = 0;
    r.passed = 0;
    r.pass_ratio = 0.0;
    r.accepted = false;

    int n = (int)matched_pairs.size();
    if (n < 3) {
        if (logger) logger->warn("pv_triangle: n=" + std::to_string(n) + " < 3, 跳过");
        return r;
    }

    int total_valid = 0;
    int passed = 0;

    if (n <= N_LARGE_THRESHOLD) {
        // 遍历所有 C(n,3) 组合
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    int rc = check_one_triangle(matched_pairs, i, j, k, eps_A, eps_J);
                    if (rc == -1) continue;
                    ++total_valid;
                    if (rc == 0) ++passed;
                }
            }
        }
    } else {
        // 随机采样
        double cn3_d = (double)n * (n - 1) * (n - 2) / 6.0;
        int sample_target = (int)(std::min)(cn3_d, (double)MAX_SAMPLES_N_LARGE);
        if (sample_target < 1) sample_target = 1;

        std::mt19937 rng(RNG_SEED);
        std::uniform_int_distribution<int> dist(0, n - 1);
        std::unordered_set<uint64_t> seen;
        seen.reserve((size_t)sample_target * 2);

        int attempts = 0;
        int max_attempts = sample_target * 20;

        while ((int)seen.size() < sample_target && attempts < max_attempts) {
            ++attempts;
            int i = dist(rng), j = dist(rng), k = dist(rng);
            if (i == j || j == k || i == k) continue;
            if (i > j) std::swap(i, j);
            if (j > k) std::swap(j, k);
            if (i > j) std::swap(i, j);

            uint64_t key = encode_triple(i, j, k);
            if (seen.count(key)) continue;
            seen.insert(key);

            int rc = check_one_triangle(matched_pairs, i, j, k, eps_A, eps_J);
            if (rc == -1) continue;
            ++total_valid;
            if (rc == 0) ++passed;
        }
    }

    r.total = total_valid;
    r.passed = passed;
    r.pass_ratio = (total_valid > 0) ? (double)passed / (double)total_valid : 0.0;
    r.accepted = (r.pass_ratio > pass_rate_threshold);

    if (logger) logger->info("pv_triangle: n=" + std::to_string(n) +
        " total=" + std::to_string(total_valid) +
        " passed=" + std::to_string(passed) +
        " ratio=" + std::to_string(r.pass_ratio) +
        " accepted=" + std::to_string(r.accepted ? 1 : 0));

    return r;
}

} // namespace pv

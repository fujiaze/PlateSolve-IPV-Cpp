// ============================================================================
// vm4_triangle.cpp - V4.0 三角形双特征二级验证（Task 6）实现
//
// Phase D' 二级验证：图像侧 vs 星表侧三角形几何一致性检验
//   特征 1: 面积 A（海伦公式）
//   特征 2: 极惯性矩 J = A·(a²+b²+c²)/36
//
// 约束: C++17，单线程；中文注释，UTF-8 编码
// ============================================================================

#include "vm4_triangle.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cstdint>

namespace vm4_1 {

// 退化三角形面积阈值（角秒²）
static constexpr double DEGEN_AREA_EPS = 1e-6;

// 大 n 时随机采样上限
static constexpr int    MAX_SAMPLES_N_LARGE = 1000;
static constexpr int    N_LARGE_THRESHOLD   = 30;

// 固定随机种子，保证结果可重现
static constexpr uint32_t RNG_SEED = 0x5A5A5A5A;

// ----------------------------------------------------------------------------
// compute_triangle_features
// ----------------------------------------------------------------------------
TriangleFeatures compute_triangle_features(
    double x1, double y1,
    double x2, double y2,
    double x3, double y3)
{
    TriangleFeatures f{0.0, 0.0, 0.0, 0.0, 0.0};

    // 两两欧氏距离
    double d12 = std::hypot(x1 - x2, y1 - y2);
    double d23 = std::hypot(x2 - x3, y2 - y3);
    double d13 = std::hypot(x1 - x3, y1 - y3);

    // 排序使 a ≤ b ≤ c
    double s3[3] = {d12, d23, d13};
    std::sort(s3, s3 + 3);
    f.a = s3[0];
    f.b = s3[1];
    f.c = s3[2];

    // 海伦公式
    double s = (f.a + f.b + f.c) * 0.5;
    double tmp = s * (s - f.a) * (s - f.b) * (s - f.c);
    if (tmp <= 0.0) {
        // 共线或退化，返回全零
        return TriangleFeatures{0.0, 0.0, 0.0, 0.0, 0.0};
    }
    f.area = std::sqrt(tmp);

    // 退化三角形（面积接近 0）跳过
    if (f.area < DEGEN_AREA_EPS) {
        return TriangleFeatures{0.0, 0.0, f.a, f.b, f.c};
    }

    // 极惯性矩 J = A·(a²+b²+c²)/36
    f.moment = f.area * (f.a * f.a + f.b * f.b + f.c * f.c) / 36.0;

    return f;
}

// ----------------------------------------------------------------------------
// 内部：将 (i,j,k) 编码为唯一 uint64_t 键（i<j<k）
// ----------------------------------------------------------------------------
static inline uint64_t encode_triple(int i, int j, int k) {
    // 假设 n < 2^21（约 200 万），三索引各占 21 位足够
    return ((uint64_t)i << 42) | ((uint64_t)j << 21) | (uint64_t)k;
}

// ----------------------------------------------------------------------------
// 内部：对单个三角形组合验证
//   返回值: 0=通过, 1=不通过, -1=退化（不计入）
// ----------------------------------------------------------------------------
static int check_one_triangle(
    const std::vector<std::array<double, 4>>& mp,
    int i, int j, int k,
    double eps_A, double eps_J)
{
    // 图像侧三角形
    TriangleFeatures fi = compute_triangle_features(
        mp[i][0], mp[i][1], mp[j][0], mp[j][1], mp[k][0], mp[k][1]);
    // 星表侧三角形
    TriangleFeatures fc = compute_triangle_features(
        mp[i][2], mp[i][3], mp[j][2], mp[j][3], mp[k][2], mp[k][3]);

    // 任一侧退化则不计入总数
    if (fi.area < DEGEN_AREA_EPS || fc.area < DEGEN_AREA_EPS) {
        return -1;
    }

    // 相对误差：用 max 避免除零
    double maxA = std::max(fi.area, fc.area);
    double maxJ = std::max(fi.moment, fc.moment);
    if (maxA < DEGEN_AREA_EPS) return -1;
    if (maxJ < DEGEN_AREA_EPS) return -1;

    double rel_A = std::abs(fi.area - fc.area) / maxA;
    double rel_J = std::abs(fi.moment - fc.moment) / maxJ;

    if (rel_A < eps_A && rel_J < eps_J) return 0;
    return 1;
}

// ----------------------------------------------------------------------------
// verify_triangles
// ----------------------------------------------------------------------------
TriangleVerifyResult verify_triangles(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double eps_A, double eps_J, double threshold)
{
    TriangleVerifyResult r;
    r.total_triangles = 0;
    r.passed          = 0;
    r.pass_ratio      = 0.0;
    r.accepted        = false;

    int n = (int)matched_pairs.size();
    if (n < 3) {
        fprintf(stderr, "[vm4_1_triangle] n=%d < 3，跳过三角形验证\n", n);
        return r;
    }

    int total_valid = 0;
    int passed      = 0;

    if (n <= N_LARGE_THRESHOLD) {
        // 枚举所有 C(n,3) 组合
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    int rc = check_one_triangle(matched_pairs, i, j, k, eps_A, eps_J);
                    if (rc == -1) continue;          // 退化，不计入
                    ++total_valid;
                    if (rc == 0) ++passed;
                }
            }
        }
    } else {
        // 随机采样 min(C(n,3), 1000) 个不同组合
        // 估算 C(n,3)，使用 double 防溢出
        double cn3_d = (double)n * (n - 1) * (n - 2) / 6.0;
        int sample_target = (int)(std::min)(cn3_d, (double)MAX_SAMPLES_N_LARGE);
        if (sample_target < 1) sample_target = 1;

        std::mt19937 rng(RNG_SEED);
        std::uniform_int_distribution<int> dist_i(0, n - 1);
        std::uniform_int_distribution<int> dist_j(0, n - 1);
        std::uniform_int_distribution<int> dist_k(0, n - 1);

        std::unordered_set<uint64_t> seen;
        seen.reserve((size_t)sample_target * 2);

        int attempts = 0;
        int max_attempts = sample_target * 20;  // 防止极端情况下死循环

        while ((int)seen.size() < sample_target && attempts < max_attempts) {
            ++attempts;
            int i = dist_i(rng);
            int j = dist_j(rng);
            int k = dist_k(rng);
            if (i == j || j == k || i == k) continue;

            // 排序为 i<j<k
            if (i > j) std::swap(i, j);
            if (j > k) std::swap(j, k);
            if (i > j) std::swap(i, j);
            // 此时 i<j<k

            uint64_t key = encode_triple(i, j, k);
            if (seen.count(key)) continue;
            seen.insert(key);

            int rc = check_one_triangle(matched_pairs, i, j, k, eps_A, eps_J);
            if (rc == -1) continue;
            ++total_valid;
            if (rc == 0) ++passed;
        }

        fprintf(stderr,
                "[vm4_1_triangle] n=%d>30，随机采样 %zu 组合（目标 %d），"
                "有效 %d，通过 %d\n",
                n, seen.size(), sample_target, total_valid, passed);
    }

    r.total_triangles = total_valid;
    r.passed          = passed;
    if (total_valid > 0) {
        r.pass_ratio = (double)passed / (double)total_valid;
    } else {
        r.pass_ratio = 0.0;
    }
    r.accepted = (r.pass_ratio > threshold);

    fprintf(stderr,
            "[vm4_1_triangle] 验证完成: total=%d passed=%d ratio=%.4f threshold=%.2f → %s\n",
            r.total_triangles, r.passed, r.pass_ratio, threshold,
            r.accepted ? "ACCEPT" : "REJECT");

    return r;
}

} // namespace vm4_1

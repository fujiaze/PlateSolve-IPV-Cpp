// ============================================================================
// test_triangle.cpp - V4.0 三角形双特征验证（Task 6）单元测试
//
// 测试 5 个场景:
//   1. 正确匹配: 30 对，微小噪声 0.1" → 通过率 > 0.95
//   2. 错误匹配: 10 对偶然匹配，几何不一致 → 通过率 < 0.3
//   3. 退化情况: 3 颗共线星 → 跳过，不计入总数
//   4. 大 n 测试: 100 对，随机采样不崩溃，通过率合理
//   5. 等边三角形: A=√3/4·a², J=A·a²/12
//
// 编译:
//   g++ -O2 -std=c++17 -I../include test_triangle.cpp ../src/vm4_triangle.cpp -o test_triangle
// 运行:
//   ./test_triangle
// ============================================================================

#include "vm4_triangle.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <string>

// --- 简单测试框架 ---
static int g_pass_count = 0;
static int g_fail_count = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { ++g_pass_count; } \
    else { ++g_fail_count; printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); } \
} while(0)

#define ASSERT_NEAR(val, expect, tol, msg) do { \
    double _d = std::abs((double)(val) - (double)(expect)); \
    if (_d < (tol)) { ++g_pass_count; } \
    else { ++g_fail_count; printf("[FAIL] %s:%d: %s (val=%.6g expect=%.6g tol=%.6g)\n", \
        __FILE__, __LINE__, msg, (double)(val), (double)(expect), (double)(tol)); } \
} while(0)

static void print_header(const std::string& name) {
    printf("\n========== %s ==========\n", name.c_str());
}

// ----------------------------------------------------------------------------
// 测试 1: 正确匹配（30 对，0.1" 噪声）→ 通过率 > 0.95
// ----------------------------------------------------------------------------
static void test_correct_match() {
    print_header("Test 1: 正确匹配 30 对 (0.1\" 噪声)");

    const int N = 30;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);
    std::normal_distribution<double> noise(0.0, 0.1);

    std::vector<std::array<double, 4>> mp(N);
    for (int i = 0; i < N; ++i) {
        double x = pos(rng);
        double y = pos(rng);
        // 图像侧 = (x, y)
        // 星表侧 = (x + 0.1" 噪声, y + 0.1" 噪声)
        mp[i] = {x, y, x + noise(rng), y + noise(rng)};
    }

    auto r = vm4::verify_triangles(mp, 0.05, 0.10, 0.8);
    printf("  total=%d passed=%d ratio=%.4f accepted=%d\n",
           r.total_triangles, r.passed, r.pass_ratio, (int)r.accepted);

    ASSERT_TRUE(r.total_triangles > 0, "应有有效三角形");
    ASSERT_TRUE(r.pass_ratio > 0.95, "正确匹配通过率应 > 0.95");
    ASSERT_TRUE(r.accepted == true, "应被接受");
}

// ----------------------------------------------------------------------------
// 测试 2: 错误匹配（10 对偶然匹配，几何不一致）→ 通过率 < 0.3
// ----------------------------------------------------------------------------
static void test_wrong_match() {
    print_header("Test 2: 错误匹配 10 对 (几何不一致)");

    const int N = 10;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);

    std::vector<std::array<double, 4>> mp(N);
    for (int i = 0; i < N; ++i) {
        // 图像侧和星表侧完全独立随机
        double ix = pos(rng), iy = pos(rng);
        double cx = pos(rng), cy = pos(rng);
        mp[i] = {ix, iy, cx, cy};
    }

    auto r = vm4::verify_triangles(mp, 0.05, 0.10, 0.8);
    printf("  total=%d passed=%d ratio=%.4f accepted=%d\n",
           r.total_triangles, r.passed, r.pass_ratio, (int)r.accepted);

    ASSERT_TRUE(r.total_triangles > 0, "应有有效三角形");
    ASSERT_TRUE(r.pass_ratio < 0.3, "错误匹配通过率应 < 0.3");
    ASSERT_TRUE(r.accepted == false, "应被拒绝");
}

// ----------------------------------------------------------------------------
// 测试 3: 退化情况（3 颗共线星）→ 跳过，不计入总数
// ----------------------------------------------------------------------------
static void test_degenerate() {
    print_header("Test 3: 退化情况 (3 颗共线星)");

    // 共线点 (0,0), (10,0), (20,0)
    auto f = vm4::compute_triangle_features(0, 0, 10, 0, 20, 0);
    printf("  area=%.6g moment=%.6g a=%.4g b=%.4g c=%.4g\n",
           f.area, f.moment, f.a, f.b, f.c);
    ASSERT_TRUE(f.area < 1e-6, "共线三角形面积应接近 0");
    ASSERT_TRUE(f.moment < 1e-6, "共线三角形极惯性矩应接近 0");

    // 3 颗共线星作为匹配对 → total 应为 0
    std::vector<std::array<double, 4>> mp = {
        {0, 0, 0, 0},
        {10, 0, 10, 0},
        {20, 0, 20, 0}
    };
    auto r = vm4::verify_triangles(mp, 0.05, 0.10, 0.8);
    printf("  total=%d passed=%d ratio=%.4f\n",
           r.total_triangles, r.passed, r.pass_ratio);
    ASSERT_TRUE(r.total_triangles == 0, "退化三角形不计入总数");
    ASSERT_TRUE(r.pass_ratio == 0.0, "无有效三角形时通过率为 0");
}

// ----------------------------------------------------------------------------
// 测试 4: 大 n 测试（100 对，随机采样不崩溃）
// ----------------------------------------------------------------------------
static void test_large_n() {
    print_header("Test 4: 大 n 测试 (100 对，随机采样)");

    const int N = 100;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> pos(-200.0, 200.0);
    std::normal_distribution<double> noise(0.0, 0.05);

    std::vector<std::array<double, 4>> mp(N);
    for (int i = 0; i < N; ++i) {
        double x = pos(rng);
        double y = pos(rng);
        mp[i] = {x, y, x + noise(rng), y + noise(rng)};
    }

    auto r = vm4::verify_triangles(mp, 0.05, 0.10, 0.8);
    printf("  total=%d passed=%d ratio=%.4f accepted=%d\n",
           r.total_triangles, r.passed, r.pass_ratio, (int)r.accepted);

    ASSERT_TRUE(r.total_triangles > 0, "应有有效三角形（采样不崩溃）");
    ASSERT_TRUE(r.total_triangles <= 1000, "采样数应 ≤ 1000");
    ASSERT_TRUE(r.pass_ratio > 0.5, "正确匹配（小噪声）通过率应 > 0.5");
}

// ----------------------------------------------------------------------------
// 测试 5: 等边三角形面积/极惯性矩计算正确性
//   边长 a，顶点 (0,0), (a,0), (a/2, a·√3/2)
//   A = √3/4·a²
//   J = A·(a²+a²+a²)/36 = A·a²/12
// ----------------------------------------------------------------------------
static void test_equilateral() {
    print_header("Test 5: 等边三角形计算正确性");

    double a = 10.0;  // 边长 10 角秒
    double sq3 = std::sqrt(3.0);
    // 三个顶点
    double x1 = 0.0,   y1 = 0.0;
    double x2 = a,     y2 = 0.0;
    double x3 = a/2.0, y3 = a * sq3 / 2.0;

    auto f = vm4::compute_triangle_features(x1, y1, x2, y2, x3, y3);
    double A_expect = sq3 / 4.0 * a * a;
    double J_expect = A_expect * a * a / 12.0;

    printf("  a=%.4g\n", a);
    printf("  area    = %.10g (expect %.10g)\n", f.area, A_expect);
    printf("  moment  = %.10g (expect %.10g)\n", f.moment, J_expect);
    printf("  sides   = a=%.6g b=%.6g c=%.6g\n", f.a, f.b, f.c);

    ASSERT_NEAR(f.area,   A_expect, 1e-9, "等边三角形面积");
    ASSERT_NEAR(f.moment, J_expect, 1e-9, "等边三角形极惯性矩");
    // 三边应相等（排序后 a≈b≈c）
    ASSERT_NEAR(f.a, f.b, 1e-9, "等边三角形 a≈b");
    ASSERT_NEAR(f.b, f.c, 1e-9, "等边三角形 b≈c");
    ASSERT_NEAR(f.a, a,   1e-9, "等边三角形边长");
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
int main() {
    printf("=== vm4_triangle 单元测试 ===\n");

    test_correct_match();
    test_wrong_match();
    test_degenerate();
    test_large_n();
    test_equilateral();

    printf("\n=== 测试汇总 ===\n");
    printf("  通过: %d\n", g_pass_count);
    printf("  失败: %d\n", g_fail_count);
    printf("  结果: %s\n", g_fail_count == 0 ? "ALL PASS" : "HAS FAILURES");

    return g_fail_count == 0 ? 0 : 1;
}

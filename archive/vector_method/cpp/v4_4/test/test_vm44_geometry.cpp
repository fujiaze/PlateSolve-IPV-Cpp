// ============================================================================
// test_geometry.cpp - V4.3 局部几何一致性过滤模块单元测试 (Task 4)
//
// 测试用例:
//   1. 真匹配场景: 邻星角距一致, consistency≥4, 保留
//   2. 假匹配场景: 邻星角距不一致, consistency<4, 丢弃
//   3. 性能测试: N=1000 候选, 过滤耗时 < 50ms
// ============================================================================

#include "vm44_internal.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstring>
#include <vector>

static const double kPi = 3.14159265358979323846;

// 构造默认参数 (仅设置 geometry 相关字段)
static v44::VM44SolveParams make_params() {
    v44::VM44SolveParams p;
    std::memset(&p, 0, sizeof(p));
    p.irm_k_geometry       = 8;    // 图像侧邻星数
    p.irm_geom_threshold   = 4;    // 一致性阈值
    p.irm_geom_dist_tol    = 3.0;  // 角距容差
    return p;
}

static v44::StarPoint make_star(double x, double y) {
    v44::StarPoint s;
    s.x = x; s.y = y; s.flux = 1.0; s.saturated = false;
    return s;
}

// ============================================================================
// 测试 1: 真匹配场景
// img_A 与 gaia_a 对齐, 图像侧邻星与星表侧邻星位置一一对应 (角距一致)
// 期望: consistency=8 ≥ 4, 候选保留
// ============================================================================
static int test_true_match() {
    std::printf("[test_true_match] 真匹配场景...\n");

    std::vector<v44::StarPoint> U, W;
    // img_A = gaia_a = (0, 0)
    U.push_back(make_star(0.0, 0.0));
    W.push_back(make_star(0.0, 0.0));
    // 8 个邻星在半径 10" 圆周上, U 与 W 位置完全对应
    for (int i = 0; i < 8; ++i) {
        double ang = i * (2.0 * kPi / 8.0);
        v44::StarPoint p = make_star(10.0 * std::cos(ang), 10.0 * std::sin(ang));
        U.push_back(p);
        W.push_back(p);
    }
    // W 侧填充远处的点, 使 knn(W, gaia_a, 15) 有足够邻居
    for (int i = 0; i < 20; ++i) {
        W.push_back(make_star(100.0 + i, 100.0));
    }

    std::vector<v44::MatchPair> cand = { {0, 0} };
    std::vector<v44::MatchPair> filtered;
    v44::VM44SolveParams params = make_params();

    int ret = v44::vm44_geometry_filter(cand, U, W, /*s0=*/1.0, /*s_robust=*/0.0,
                                        params, filtered, nullptr);
    assert(ret == 0);
    assert(filtered.size() == 1);
    std::printf("  -> 保留 %d 个候选 (期望 1)  OK\n", (int)filtered.size());
    return 0;
}

// ============================================================================
// 测试 2: 假匹配场景
// img_A 邻星在半径 10", 但 gaia_a 邻星在半径 50", 角距严重不一致
// 期望: consistency=0 < 4, 候选丢弃
// ============================================================================
static int test_false_match() {
    std::printf("[test_false_match] 假匹配场景...\n");

    std::vector<v44::StarPoint> U, W;
    U.push_back(make_star(0.0, 0.0));
    W.push_back(make_star(0.0, 0.0));
    // U 侧邻星在半径 10" 圆周
    for (int i = 0; i < 8; ++i) {
        double ang = i * (2.0 * kPi / 8.0);
        U.push_back(make_star(10.0 * std::cos(ang), 10.0 * std::sin(ang)));
    }
    // W 侧邻星在半径 50" 圆周 (角距与 U 侧差 40" >> tol=3")
    for (int i = 0; i < 8; ++i) {
        double ang = i * (2.0 * kPi / 8.0);
        W.push_back(make_star(50.0 * std::cos(ang), 50.0 * std::sin(ang)));
    }
    // W 填充
    for (int i = 0; i < 20; ++i) {
        W.push_back(make_star(200.0 + i, 200.0));
    }

    std::vector<v44::MatchPair> cand = { {0, 0} };
    std::vector<v44::MatchPair> filtered;
    v44::VM44SolveParams params = make_params();

    int ret = v44::vm44_geometry_filter(cand, U, W, /*s0=*/1.0, /*s_robust=*/0.0,
                                        params, filtered, nullptr);
    assert(ret == 0);
    assert(filtered.empty());
    std::printf("  -> 丢弃候选 (consistency < 4)  OK\n");
    return 0;
}

// ============================================================================
// 测试 3: 性能测试
// N=1000 候选, U/W 各 1100 点, 验证过滤耗时 < 50ms
// ============================================================================
static int test_performance() {
    std::printf("[test_performance] 性能测试 N=1000...\n");

    const int N = 1000;
    std::vector<v44::StarPoint> U, W;
    U.reserve(N + 100);
    W.reserve(N + 100);
    for (int i = 0; i < N + 100; ++i) {
        U.push_back(make_star(i * 10.0, 0.0));
        W.push_back(make_star(i * 10.0, 0.0));
    }

    std::vector<v44::MatchPair> cand;
    cand.reserve(N);
    for (int i = 0; i < N; ++i) {
        cand.push_back({i, i});
    }

    std::vector<v44::MatchPair> filtered;
    v44::VM44SolveParams params = make_params();

    auto t0 = std::chrono::steady_clock::now();
    int ret = v44::vm44_geometry_filter(cand, U, W, 1.0, 0.0, params, filtered, nullptr);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    assert(ret == 0);
    std::printf("  -> 耗时 %.2f ms (阈值 50 ms)\n", ms);
    assert(ms < 50.0);
    std::printf("  -> 性能达标  OK\n");
    return 0;
}

int main() {
    std::printf("=== test_geometry 开始 ===\n");
    int failed = 0;
    failed += test_true_match();
    failed += test_false_match();
    failed += test_performance();
    if (failed == 0) {
        std::printf("=== test_geometry 全部通过 ===\n");
        return 0;
    } else {
        std::printf("=== test_geometry 失败 %d 项 ===\n", failed);
        return 1;
    }
}

// ============================================================================
// test_score.cpp - V4.3 S_robust 稳健评分模块单元测试 (Task 6)
//
// 测试用例:
//   1. 外点不污染评分: 45 内点散布 ~0.1", 5 外点偏离 ~50", 验证 k_cut=45, S_robust < 1"
//   2. 全内点场景: 50 对全内点, 散布 ~0.1", S_robust ≈ 散布 RMS
//   3. 残差跳变检测准确性: 10 内点 + 5 外点(偏离~10"), 验证 ratio > 3.0 的检测
//
// 注: 新版 vm44_compute_s_robust 用 median 估计平移, 残差 = 散布 (偏离中位数的程度)
//     s0=1.0, CD=(1/3600)·I 时: offset_x = W.x - U.x, 残差 = |offset - median| × s0
// ============================================================================

#include "vm44_internal.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// s0 = 1.0 arcsec/pixel (测试用)
static constexpr double TEST_S0 = 1.0;

// 构造 CD = (1/3600) · I  (s=1, θ=0, s0=1 → cd11 = s0/(s·3600) = 1/3600)
static v44::CDMatrix make_identity_cd() {
    v44::CDMatrix cd;
    double f = 1.0 / 3600.0;
    cd.cd11 =  f; cd.cd12 = 0.0;
    cd.cd21 = 0.0; cd.cd22 =  f;
    return cd;
}

// 构造无 SIP (order=0)
static v44::SIPCoeffs make_no_sip() {
    v44::SIPCoeffs sip;
    std::memset(sip.A, 0, sizeof(sip.A));
    std::memset(sip.B, 0, sizeof(sip.B));
    sip.order = 0;
    return sip;
}

static v44::StarPoint make_star(double x, double y) {
    v44::StarPoint s;
    s.x = x; s.y = y; s.flux = 1.0; s.saturated = false;
    return s;
}

// 计算全部残差的 RMS (参考值)
static double rms_all(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s / v.size());
}

// ============================================================================
// 测试 1: 外点不污染评分
// 45 内点: offset = 10 + scatter_i (散布 ~±0.1"), 5 外点: offset = 60 (偏离 ~50")
// 期望: k_cut=45 (跳变检测), S_robust < 1.0" (不被外点污染)
// ============================================================================
static int test_outlier_no_pollution() {
    std::printf("[test_outlier_no_pollution] 外点不污染评分...\n");

    std::vector<v44::StarPoint> U, W;
    std::vector<v44::MatchPair> cps;
    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();

    // 45 内点: offset = 10 + (i%10 - 5) * 0.02 (散布 -0.1..+0.08)
    // U.x = i, W.x = i + offset → offset = W.x - U.x
    for (int i = 0; i < 45; ++i) {
        double scatter = (i % 10 - 5) * 0.02;  // -0.10, -0.08, ..., +0.08
        double offset = 10.0 + scatter;
        int idx = (int)U.size();
        U.push_back(make_star((double)i, 0.0));
        W.push_back(make_star((double)i + offset, 0.0));
        cps.push_back({idx, idx});
    }
    // 5 外点: offset = 60 (偏离 median ~50")
    for (int i = 0; i < 5; ++i) {
        int idx = (int)U.size();
        U.push_back(make_star(100.0 + i, 0.0));
        W.push_back(make_star(100.0 + i + 60.0, 0.0));
        cps.push_back({idx, idx});
    }

    v44::SRobustResult out;
    int ret = v44::vm44_compute_s_robust(U, W, cps, cd, sip, TEST_S0, /*M0=*/5, out, nullptr);
    assert(ret == 0);

    std::printf("  -> k_cut=%d (期望 45), S_robust=%.4f\" (期望 < 1.0), "
                "n_inliers=%d, coverage=%.3f\n",
                out.k_cut, out.s_robust, out.n_inliers, out.coverage);
    assert(out.k_cut == 45);
    assert(out.s_robust < 1.0);  // 不被 50" 外点污染
    std::printf("  -> 外点未污染评分  OK\n");
    return 0;
}

// ============================================================================
// 测试 2: 全内点场景
// 50 对全是内点, 散布 ~0.1", 验证 S_robust ≈ 散布 RMS
// ============================================================================
static int test_all_inliers() {
    std::printf("[test_all_inliers] 全内点场景...\n");

    std::vector<v44::StarPoint> U, W;
    std::vector<v44::MatchPair> cps;
    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();
    std::vector<double> scatter_values;

    // 50 内点: offset = 10 + scatter_i (散布 ±0.1")
    for (int i = 0; i < 50; ++i) {
        double scatter = (i % 10 - 5) * 0.02;
        double offset = 10.0 + scatter;
        int idx = (int)U.size();
        U.push_back(make_star((double)i, 0.0));
        W.push_back(make_star((double)i + offset, 0.0));
        cps.push_back({idx, idx});
        scatter_values.push_back(scatter);
    }

    v44::SRobustResult out;
    int ret = v44::vm44_compute_s_robust(U, W, cps, cd, sip, TEST_S0, /*M0=*/5, out, nullptr);
    assert(ret == 0);

    // 参考值: 散布的 RMS (偏离 median 的程度)
    std::vector<double> sc_sorted = scatter_values;
    std::sort(sc_sorted.begin(), sc_sorted.end());
    double sc_median = sc_sorted[25];
    double rss = 0;
    for (double s : scatter_values) {
        double dev = s - sc_median;
        rss += dev * dev;
    }
    double rms_ref = std::sqrt(rss / 50.0);

    std::printf("  -> S_robust=%.4f\", scatter_RMS=%.4f\", k_cut=%d, n_robust=%d\n",
                out.s_robust, rms_ref, out.k_cut, out.n_inliers);
    // 全内点时 k_cut 应为 N=50
    assert(out.k_cut == 50);
    // S_robust 应接近散布 RMS
    assert(std::fabs(out.s_robust - rms_ref) < 0.05);
    std::printf("  -> S_robust ≈ 散布 RMS  OK\n");
    return 0;
}

// ============================================================================
// 测试 3: 残差跳变检测准确性
// 10 内点 (散布 ±0.05") + 5 外点 (偏离 ~10"), ratio ≈ 10/0.1 > 3.0
// 期望: k_cut=10
// ============================================================================
static int test_jump_detection() {
    std::printf("[test_jump_detection] 残差跳变检测...\n");

    std::vector<v44::StarPoint> U, W;
    std::vector<v44::MatchPair> cps;
    v44::CDMatrix cd = make_identity_cd();
    v44::SIPCoeffs sip = make_no_sip();

    // 10 内点: offset = 10 + scatter (散布 ±0.05")
    for (int i = 0; i < 10; ++i) {
        double scatter = (i % 5 - 2) * 0.02;  // -0.04, -0.02, 0, 0.02, 0.04
        double offset = 10.0 + scatter;
        int idx = (int)U.size();
        U.push_back(make_star((double)i, 0.0));
        W.push_back(make_star((double)i + offset, 0.0));
        cps.push_back({idx, idx});
    }
    // 5 外点: offset = 20 (偏离 median ~10")
    for (int i = 0; i < 5; ++i) {
        int idx = (int)U.size();
        U.push_back(make_star(100.0 + i, 0.0));
        W.push_back(make_star(100.0 + i + 20.0, 0.0));
        cps.push_back({idx, idx});
    }

    v44::SRobustResult out;
    int ret = v44::vm44_compute_s_robust(U, W, cps, cd, sip, TEST_S0, /*M0=*/5, out, nullptr);
    assert(ret == 0);

    std::printf("  -> k_cut=%d (期望 10), median=%.4f, mad=%.4f\n",
                out.k_cut, out.median_r, out.mad);
    assert(out.k_cut == 10);
    std::printf("  -> 跳变检测准确  OK\n");
    return 0;
}

int main() {
    std::printf("=== test_score 开始 ===\n");
    int failed = 0;
    failed += test_outlier_no_pollution();
    failed += test_all_inliers();
    failed += test_jump_detection();
    if (failed == 0) {
        std::printf("=== test_score 全部通过 ===\n");
        return 0;
    } else {
        std::printf("=== test_score 失败 %d 项 ===\n", failed);
        return 1;
    }
}

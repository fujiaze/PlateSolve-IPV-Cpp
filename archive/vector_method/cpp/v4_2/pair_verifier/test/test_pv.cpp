// ============================================================================
// test_pv.cpp - V4.2 PairVerifier 单元测试（Task 5）
//
// 测试场景:
//   1. MAD 清洗: 50 对(45 正确 + 5 离群 100"), 验证剔除 >= 4, 保留 ~45
//   2. 贝叶斯接受: 50 对正确匹配, RMS~1.0", 验证 lnK > 20.7, decision=1
//   3. 贝叶斯拒绝: 50 对随机匹配, 大残差, 验证 lnK < 6.9, decision=-1
//   4. 三角形通过: 30 对正确匹配, 验证 pass_ratio > 0.8
//   5. 三角形失败: 30 对随机匹配, 验证 pass_ratio < 0.3
//
// 编译:
//   make test_pv
// 运行:
//   ./test/test_pv.exe
// ============================================================================

#include "../include/pv_api.h"

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

// --- 默认参数 ---
static PairVerifierParams make_default_params() {
    PairVerifierParams p;
    p.mad_iters = 3;
    p.mad_threshold_factor = 3.0;
    p.mad_min_threshold_arcsec = 5.0;
    p.lnK_accept = 20.7;
    p.lnK_weak = 6.9;
    p.sigma_min = 0.5;
    p.eps_A = 0.05;
    p.eps_J = 0.10;
    p.triangle_pass_rate = 0.8;
    p.fov_diag_deg = 2.0;
    p.log_file_path = nullptr;
    return p;
}

// ----------------------------------------------------------------------------
// 测试 1: MAD 清洗 — 50 对(45 正确 + 5 离群), 验证剔除离群
// ----------------------------------------------------------------------------
static void test_mad_clean() {
    print_header("Test 1: MAD 清洗 (45 正确 + 5 离群 100\")");

    const int N = 50;
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);
    std::normal_distribution<double> noise(0.0, 0.3);

    // U 和 W 数组（50 颗星，一一对应）
    std::vector<double> U(N * 2), W(N * 2);
    for (int i = 0; i < N; ++i) {
        double x = pos(rng), y = pos(rng);
        U[i * 2] = x;           U[i * 2 + 1] = y;
        W[i * 2] = x + noise(rng); W[i * 2 + 1] = y + noise(rng);
    }

    // 匹配对: 前 45 对正确匹配, 后 5 对离群(偏移 100")
    std::vector<int> pairs_u(N), pairs_w(N);
    for (int i = 0; i < 45; ++i) {
        pairs_u[i] = i; pairs_w[i] = i;
    }
    // 5 个离群: U[i] 与 W[i] 偏移 100"
    for (int i = 45; i < 50; ++i) {
        pairs_u[i] = i; pairs_w[i] = i;
        U[i * 2]     += 100.0;  // 偏移 100"
        U[i * 2 + 1] += 100.0;
    }

    auto params = make_default_params();
    VerificationResult result;
    int rc = pv_verify(U.data(), N, W.data(), N,
                       pairs_u.data(), pairs_w.data(), N,
                       2.0, &params, &result);

    printf("  rc=%d n_clean=%d n_removed=%d mad_iters=%d mad_rms=%.3f\"\n",
           rc, result.n_clean, result.n_removed, result.mad_iterations,
           result.mad_rms_arcsec);

    ASSERT_TRUE(rc == 1, "pv_verify 应返回 1");
    ASSERT_TRUE(result.success == 1, "应 success=1");
    ASSERT_TRUE(result.n_removed >= 4, "应剔除 >= 4 对离群");
    ASSERT_TRUE(result.n_clean <= 46, "清洗后应 <= 46 对");
    ASSERT_TRUE(result.n_clean >= 44, "清洗后应 >= 44 对");
    ASSERT_TRUE(result.mad_rms_arcsec < 5.0, "清洗后 RMS 应 < 5\"");

    pv_free(&result);
}

// ----------------------------------------------------------------------------
// 测试 2: 贝叶斯接受 — 50 对正确匹配, RMS~1.0"
// ----------------------------------------------------------------------------
static void test_bayes_accept() {
    print_header("Test 2: 贝叶斯接受 (50 对, RMS~1.0\")");

    const int N = 50;
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);
    std::normal_distribution<double> noise(0.0, 0.7);  // 2D RMS ~ 1.0"

    std::vector<double> U(N * 2), W(N * 2);
    std::vector<int> pairs_u(N), pairs_w(N);
    for (int i = 0; i < N; ++i) {
        double x = pos(rng), y = pos(rng);
        W[i * 2] = x;           W[i * 2 + 1] = y;
        U[i * 2] = x + noise(rng); U[i * 2 + 1] = y + noise(rng);
        pairs_u[i] = i; pairs_w[i] = i;
    }

    auto params = make_default_params();
    params.fov_diag_deg = 2.0;  // ~7200" FOV
    VerificationResult result;
    int rc = pv_verify(U.data(), N, W.data(), N,
                       pairs_u.data(), pairs_w.data(), N,
                       2.0, &params, &result);

    printf("  rc=%d n_clean=%d mad_rms=%.3f\" lnK=%.2f decision=%d\n",
           rc, result.n_clean, result.mad_rms_arcsec,
           result.bayes_lnK, result.bayes_decision);

    ASSERT_TRUE(rc == 1, "pv_verify 应返回 1");
    ASSERT_TRUE(result.bayes_decision == 1, "贝叶斯应接受 (decision=1)");
    ASSERT_TRUE(result.bayes_lnK > 20.7, "lnK 应 > 20.7");

    pv_free(&result);
}

// ----------------------------------------------------------------------------
// 测试 3: 贝叶斯拒绝 — 50 对随机匹配, 大残差
//   注: 由于"大视场贝叶斯效应"(V4.1 已记录), 大 FOV 下 50 对即使残差较大
//   仍可能给出正 lnK。此处用非常大的残差(~5000" per axis)确保 lnK < 0。
// ----------------------------------------------------------------------------
static void test_bayes_reject() {
    print_header("Test 3: 贝叶斯拒绝 (50 对, 大残差 ~5000\"/axis)");

    const int N = 50;
    std::mt19937 rng(456);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);
    // 每坐标加 5000" 随机偏移, 2D RMS ~7071"
    std::normal_distribution<double> big_noise(0.0, 5000.0);

    std::vector<double> U(N * 2), W(N * 2);
    std::vector<int> pairs_u(N), pairs_w(N);
    for (int i = 0; i < N; ++i) {
        double x = pos(rng), y = pos(rng);
        W[i * 2] = x;           W[i * 2 + 1] = y;
        // U = W + 大噪声, Umeyama 无法校正随机偏移
        U[i * 2] = x + big_noise(rng); U[i * 2 + 1] = y + big_noise(rng);
        pairs_u[i] = i; pairs_w[i] = i;
    }

    auto params = make_default_params();
    params.fov_diag_deg = 2.0;
    VerificationResult result;
    int rc = pv_verify(U.data(), N, W.data(), N,
                       pairs_u.data(), pairs_w.data(), N,
                       2.0, &params, &result);

    printf("  rc=%d n_clean=%d mad_rms=%.3f\" lnK=%.2f decision=%d\n",
           rc, result.n_clean, result.mad_rms_arcsec,
           result.bayes_lnK, result.bayes_decision);

    ASSERT_TRUE(rc == 1, "pv_verify 应返回 1");
    ASSERT_TRUE(result.bayes_decision == -1, "贝叶斯应拒绝 (decision=-1)");
    ASSERT_TRUE(result.bayes_lnK < 6.9, "lnK 应 < 6.9");

    pv_free(&result);
}

// ----------------------------------------------------------------------------
// 测试 4: 三角形通过 — 30 对正确匹配, pass_ratio > 0.8
// ----------------------------------------------------------------------------
static void test_triangle_pass() {
    print_header("Test 4: 三角形通过 (30 对, 0.1\" 噪声)");

    const int N = 30;
    std::mt19937 rng(789);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);
    std::normal_distribution<double> noise(0.0, 0.1);

    std::vector<double> U(N * 2), W(N * 2);
    std::vector<int> pairs_u(N), pairs_w(N);
    for (int i = 0; i < N; ++i) {
        double x = pos(rng), y = pos(rng);
        W[i * 2] = x;           W[i * 2 + 1] = y;
        U[i * 2] = x + noise(rng); U[i * 2 + 1] = y + noise(rng);
        pairs_u[i] = i; pairs_w[i] = i;
    }

    auto params = make_default_params();
    VerificationResult result;
    int rc = pv_verify(U.data(), N, W.data(), N,
                       pairs_u.data(), pairs_w.data(), N,
                       2.0, &params, &result);

    printf("  rc=%d tri_total=%d tri_passed=%d tri_ratio=%.4f validated=%d\n",
           rc, result.triangle_total, result.triangle_passed,
           result.triangle_pass_ratio, result.validated);

    ASSERT_TRUE(rc == 1, "pv_verify 应返回 1");
    ASSERT_TRUE(result.triangle_pass_ratio > 0.8, "三角形通过率应 > 0.8");
    ASSERT_TRUE(result.validated == 1, "应 validated=1");

    pv_free(&result);
}

// ----------------------------------------------------------------------------
// 测试 5: 三角形失败 — 30 对随机匹配, pass_ratio < 0.3
// ----------------------------------------------------------------------------
static void test_triangle_fail() {
    print_header("Test 5: 三角形失败 (30 对随机匹配)");

    const int N = 30;
    std::mt19937 rng(999);
    std::uniform_real_distribution<double> pos(-100.0, 100.0);

    // U 和 W 独立随机, 匹配对 (i,i) 为错误匹配
    std::vector<double> U(N * 2), W(N * 2);
    std::vector<int> pairs_u(N), pairs_w(N);
    for (int i = 0; i < N; ++i) {
        // U 和 W 完全独立
        U[i * 2] = pos(rng);     U[i * 2 + 1] = pos(rng);
        W[i * 2] = pos(rng);     W[i * 2 + 1] = pos(rng);
        pairs_u[i] = i; pairs_w[i] = i;
    }

    auto params = make_default_params();
    VerificationResult result;
    int rc = pv_verify(U.data(), N, W.data(), N,
                       pairs_u.data(), pairs_w.data(), N,
                       2.0, &params, &result);

    printf("  rc=%d tri_total=%d tri_passed=%d tri_ratio=%.4f validated=%d\n",
           rc, result.triangle_total, result.triangle_passed,
           result.triangle_pass_ratio, result.validated);

    ASSERT_TRUE(rc == 1, "pv_verify 应返回 1");
    ASSERT_TRUE(result.triangle_pass_ratio < 0.3, "三角形通过率应 < 0.3");

    pv_free(&result);
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("\n");
    printf("##############################################\n");
    printf("#  V4.2 PairVerifier 单元测试 (Task 5)       #\n");
    printf("##############################################\n");

    test_mad_clean();
    test_bayes_accept();
    test_bayes_reject();
    test_triangle_pass();
    test_triangle_fail();

    printf("\n==============================================\n");
    printf("  总计: %d PASS, %d FAIL\n", g_pass_count, g_fail_count);
    printf("==============================================\n");

    return (g_fail_count == 0) ? 0 : 1;
}

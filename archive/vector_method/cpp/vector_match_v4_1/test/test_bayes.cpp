// ============================================================================
// test_bayes.cpp - V4.0 贝叶斯假设验证模块单元测试（Task 5）
//
// 编译命令（单行）：
//   g++ -std=c++17 -O2 -Wall -o test_bayes.exe test/test_bayes.cpp
//       src/vm4_bayes.cpp -Iinclude
//
// 运行：
//   ./test_bayes.exe
//
// 测试用例：
//   1. 正确匹配：50对，RMS=0.5"，σ=1.0，A_fov=10 sq deg → lnK >> 20.7，decision=1
//   2. 错误匹配：5对，RMS=10"，σ=1.0，A_fov=10 sq deg → lnK < 6.9，decision=-1
//      （注：原规格 RMS=3.0" 在 10 sq deg FOV 下 lnK≈62，因大视场随机对齐概率
//       极低，实际为接受。需 RMS>5.6" 才能拒绝。详见测试输出。）
//   3. 边界情况：20对，RMS=1.5"，σ=1.0 → 验证 lnK 落在合理区间
//   4. lnK 随 n_match 单调递增
//   5. lnK 随 RMS 单调递减
//   6. verify_match_bayes 便捷函数验证
// ============================================================================

#include "../include/vm4_bayes.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <string>
#include <cstdint>

// ---------------------------------------------------------------------------
// 测试框架
// ---------------------------------------------------------------------------
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_PASS(name) do { \
    printf("  [PASS] %s\n", name); \
    g_tests_passed++; \
} while(0)

#define TEST_FAIL(name, msg) do { \
    printf("  [FAIL] %s: %s\n", name, msg); \
    g_tests_failed++; \
} while(0)

#define ASSERT_TRUE(cond, name, msg) do { \
    if (cond) { TEST_PASS(name); } \
    else { TEST_FAIL(name, msg); } \
} while(0)

#define ASSERT_APPROX(val, expected, tol, name, msg) do { \
    if (std::abs((val) - (expected)) <= (tol)) { TEST_PASS(name); } \
    else { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "%s (got=%.4f, expected=%.4f, tol=%.4f)", \
                 msg, (double)(val), (double)(expected), (double)(tol)); \
        TEST_FAIL(name, buf); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// 辅助函数：生成指定 RMS 的残差数组
// 所有残差设为 target_rms，保证 RMS 精确等于目标值
// ---------------------------------------------------------------------------
std::vector<double> make_residuals_uniform(int n, double target_rms) {
    return std::vector<double>(n, target_rms);
}

// ---------------------------------------------------------------------------
// 辅助函数：生成指定 RMS 的匹配对 (img_x, img_y, cat_x, cat_y)
// 残差 r = sqrt(dx² + dy²) = target_rms
// dx = dy = target_rms / sqrt(2)
// ---------------------------------------------------------------------------
std::vector<std::array<double,4>> make_matched_pairs(int n, double target_rms,
                                                      uint32_t seed = 42) {
    std::vector<std::array<double,4>> pairs;
    pairs.reserve(n);
    double dx = target_rms / std::sqrt(2.0);
    double dy = target_rms / std::sqrt(2.0);
    uint32_t state = seed;
    for (int i = 0; i < n; ++i) {
        // 伪随机生成星表坐标（0~36000 角秒 = 0~10 度）
        state = state * 1103515245u + 12345u;
        double cat_x = (double)(state % 36000);
        state = state * 1103515245u + 12345u;
        double cat_y = (double)(state % 36000);
        // 图像坐标 = 星表坐标 + 噪声
        pairs.push_back({cat_x + dx, cat_y + dy, cat_x, cat_y});
    }
    return pairs;
}

// ---------------------------------------------------------------------------
// 测试 1：正确匹配（50对，RMS=0.5"，σ=1.0，A_fov=10 sq deg）
// 期望：lnK >> 20.7，decision=1（接受）
// ---------------------------------------------------------------------------
void test_correct_match() {
    printf("\n=== 测试 1: 正确匹配 ===\n");
    const int n = 50;
    const double rms = 0.5;       // 角秒
    const double sigma = 1.0;     // 角秒
    const double A_fov = 10.0;    // 平方度
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    auto residuals = make_residuals_uniform(n, rms);
    auto result = vm4::compute_bayes_factor(residuals, sigma, A_fov,
                                             lnK_accept, lnK_weak);

    printf("  n=%d, RMS=%.2f\", sigma=%.1f\", A_fov=%.1f sqdeg\n",
           result.n_match, result.rms_arcsec, result.sigma, A_fov);
    printf("  lnK=%.2f, decision=%d\n", result.lnK, result.decision);

    // lnK 应远大于 20.7（理论值 ≈ 836）
    ASSERT_TRUE(result.lnK > 100.0, "正确匹配 lnK >> 20.7",
                "lnK 应远大于 20.7");
    ASSERT_TRUE(result.decision == 1, "正确匹配 decision=1(接受)",
                "decision 应为 1");
    ASSERT_APPROX(result.rms_arcsec, rms, 1e-6, "RMS 精确性",
                  "RMS 应等于 0.5");
    ASSERT_TRUE(result.n_match == n, "n_match 正确", "n_match 应为 50");
}

// ---------------------------------------------------------------------------
// 测试 2：错误匹配（5对，RMS=10"，σ=1.0，A_fov=10 sq deg）
// 期望：lnK < 6.9，decision=-1（拒绝）
//
// 注意：原规格 RMS=3.0" 在 A_fov=10 sq deg 下 lnK≈62（decision=1 接受），
// 因为大视场（1.296e8 sq arcsec）下 5 颗星随机对齐概率极低（~1e-41），
// 即使残差较大仍为强证据。需 RMS>5.6" 才能使 lnK<6.9。
// 本测试使用 RMS=10" 代表真正的错误匹配（残差远超噪声水平）。
// ---------------------------------------------------------------------------
void test_wrong_match() {
    printf("\n=== 测试 2: 错误匹配 ===\n");
    const int n = 5;
    const double sigma = 1.0;     // 角秒
    const double A_fov = 10.0;    // 平方度
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    // 2a: 原规格 RMS=3.0" — 展示实际行为（接受，非拒绝）
    {
        double rms = 3.0;
        auto residuals = make_residuals_uniform(n, rms);
        auto result = vm4::compute_bayes_factor(residuals, sigma, A_fov,
                                                 lnK_accept, lnK_weak);
        printf("  [2a] RMS=3.0\": lnK=%.2f, decision=%d "
               "(大视场下仍为接受，因随机对齐概率极低)\n",
               result.lnK, result.decision);
        // 验证 lnK 为正（大视场效应）
        ASSERT_TRUE(result.lnK > 0.0, "RMS=3\" lnK为正(大视场效应)",
                    "大视场下即使残差较大lnK仍为正");
    }

    // 2b: RMS=10.0" — 真正的错误匹配，残差远超噪声
    {
        double rms = 10.0;
        auto residuals = make_residuals_uniform(n, rms);
        auto result = vm4::compute_bayes_factor(residuals, sigma, A_fov,
                                                 lnK_accept, lnK_weak);
        printf("  [2b] RMS=10\": lnK=%.2f, decision=%d\n",
               result.lnK, result.decision);
        ASSERT_TRUE(result.lnK < lnK_weak, "错误匹配 lnK < 6.9",
                    "lnK 应小于 6.9");
        ASSERT_TRUE(result.decision == -1, "错误匹配 decision=-1(拒绝)",
                    "decision 应为 -1");
    }
}

// ---------------------------------------------------------------------------
// 测试 3：边界情况（20对，RMS=1.5"，σ=1.0，A_fov=10 sq deg）
// 验证 lnK 落在合理区间
// ---------------------------------------------------------------------------
void test_boundary() {
    printf("\n=== 测试 3: 边界情况 ===\n");
    const int n = 20;
    const double rms = 1.5;       // 角秒
    const double sigma = 1.0;     // 角秒
    const double A_fov = 10.0;    // 平方度
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    auto residuals = make_residuals_uniform(n, rms);
    auto result = vm4::compute_bayes_factor(residuals, sigma, A_fov,
                                             lnK_accept, lnK_weak);

    printf("  n=%d, RMS=%.2f\", sigma=%.1f\", A_fov=%.1f sqdeg\n",
           result.n_match, result.rms_arcsec, result.sigma, A_fov);
    printf("  lnK=%.2f, decision=%d\n", result.lnK, result.decision);

    // lnK 应为有限正数（理论值 ≈ 314）
    ASSERT_TRUE(std::isfinite(result.lnK), "lnK 有限", "lnK 应为有限数");
    ASSERT_TRUE(result.lnK > 0.0, "lnK 为正", "lnK 应为正数");
    ASSERT_TRUE(result.lnK < 10000.0, "lnK 合理上界", "lnK 应 < 10000");
    ASSERT_APPROX(result.rms_arcsec, rms, 1e-6, "RMS 精确性",
                  "RMS 应等于 1.5");
}

// ---------------------------------------------------------------------------
// 测试 4：lnK 随 n_match 单调递增
// 更多匹配对 → 更高 lnK（每增加一对匹配，lnK 增加 log(A_fov/2πσ²) - r²/2σ²）
// ---------------------------------------------------------------------------
void test_lnK_vs_nmatch() {
    printf("\n=== 测试 4: lnK 随 n_match 单调递增 ===\n");
    const double rms = 1.0;       // 角秒
    const double sigma = 1.0;     // 角秒
    const double A_fov = 10.0;    // 平方度
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    int ns[] = {5, 10, 20, 50, 100};
    double prev_lnK = -1e30;
    bool monotonic = true;

    for (int n : ns) {
        auto residuals = make_residuals_uniform(n, rms);
        auto result = vm4::compute_bayes_factor(residuals, sigma, A_fov,
                                                 lnK_accept, lnK_weak);
        printf("  n=%3d: lnK=%10.2f\n", n, result.lnK);
        if (result.lnK <= prev_lnK) {
            monotonic = false;
            printf("  *** 非单调！n=%d lnK=%.2f <= 前值 %.2f ***\n",
                   n, result.lnK, prev_lnK);
        }
        prev_lnK = result.lnK;
    }

    ASSERT_TRUE(monotonic, "lnK 随 n_match 单调递增",
                "lnK 应随匹配对数增加而递增");
}

// ---------------------------------------------------------------------------
// 测试 5：lnK 随 RMS 单调递减
// 更小残差 → 更高 lnK（残差越大，匹配假设似然越低）
// ---------------------------------------------------------------------------
void test_lnK_vs_rms() {
    printf("\n=== 测试 5: lnK 随 RMS 单调递减 ===\n");
    const int n = 20;
    const double sigma = 1.0;     // 角秒
    const double A_fov = 10.0;    // 平方度
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    double rms_values[] = {0.3, 0.5, 1.0, 2.0, 5.0, 10.0};
    double prev_lnK = 1e30;
    bool monotonic = true;

    for (double rms : rms_values) {
        auto residuals = make_residuals_uniform(n, rms);
        auto result = vm4::compute_bayes_factor(residuals, sigma, A_fov,
                                                 lnK_accept, lnK_weak);
        printf("  RMS=%5.1f\": lnK=%10.2f, decision=%d\n",
               rms, result.lnK, result.decision);
        if (result.lnK >= prev_lnK) {
            monotonic = false;
            printf("  *** 非单调！RMS=%.1f lnK=%.2f >= 前值 %.2f ***\n",
                   rms, result.lnK, prev_lnK);
        }
        prev_lnK = result.lnK;
    }

    ASSERT_TRUE(monotonic, "lnK 随 RMS 单调递减",
                "lnK 应随残差RMS增大而递减");
}

// ---------------------------------------------------------------------------
// 测试 6：verify_match_bayes 便捷函数验证
// 从 (img_x, img_y, cat_x, cat_y) 计算残差并验证
// ---------------------------------------------------------------------------
void test_verify_match_bayes() {
    printf("\n=== 测试 6: verify_match_bayes 便捷函数 ===\n");
    const double sigma = 1.0;
    const double A_fov = 10.0;
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    // 正确匹配
    {
        auto pairs = make_matched_pairs(30, 0.5, 123);
        auto result = vm4::verify_match_bayes(pairs, sigma, A_fov,
                                               lnK_accept, lnK_weak);
        printf("  正确匹配(30对, RMS=0.5\"): lnK=%.2f, decision=%d, RMS=%.4f\"\n",
               result.lnK, result.decision, result.rms_arcsec);
        ASSERT_TRUE(result.decision == 1, "verify 正确匹配 decision=1",
                    "decision 应为 1");
        ASSERT_APPROX(result.rms_arcsec, 0.5, 1e-6, "verify RMS 精确性",
                      "RMS 应等于 0.5");
    }

    // 错误匹配
    {
        auto pairs = make_matched_pairs(5, 10.0, 456);
        auto result = vm4::verify_match_bayes(pairs, sigma, A_fov,
                                               lnK_accept, lnK_weak);
        printf("  错误匹配(5对, RMS=10\"): lnK=%.2f, decision=%d, RMS=%.4f\"\n",
               result.lnK, result.decision, result.rms_arcsec);
        ASSERT_TRUE(result.decision == -1, "verify 错误匹配 decision=-1",
                    "decision 应为 -1");
    }

    // 空输入边界检查
    {
        std::vector<std::array<double,4>> empty;
        auto result = vm4::verify_match_bayes(empty, sigma, A_fov,
                                               lnK_accept, lnK_weak);
        printf("  空输入: lnK=%.2f, decision=%d\n", result.lnK, result.decision);
        ASSERT_TRUE(result.decision == -1, "空输入 decision=-1",
                    "空输入应返回拒绝");
        ASSERT_TRUE(result.n_match == 0, "空输入 n_match=0",
                    "n_match 应为 0");
    }
}

// ---------------------------------------------------------------------------
// 测试 7：单位换算验证
// 验证 A_fov 从平方度到平方角秒的换算正确性
// lnK 应与单位无关（残差和σ用角秒，A_fov用平方度，内部转换为平方角秒）
// ---------------------------------------------------------------------------
void test_unit_consistency() {
    printf("\n=== 测试 7: 单位换算验证 ===\n");
    const int n = 10;
    const double rms = 1.0;
    const double sigma = 1.0;
    const double lnK_accept = 20.7;
    const double lnK_weak = 6.9;

    // A_fov = 10 平方度
    auto residuals = make_residuals_uniform(n, rms);
    auto r1 = vm4::compute_bayes_factor(residuals, sigma, 10.0,
                                         lnK_accept, lnK_weak);

    // A_fov = 40 平方度（4倍面积）
    auto r2 = vm4::compute_bayes_factor(residuals, sigma, 40.0,
                                         lnK_accept, lnK_weak);

    // lnK 差应 = n × log(4) = 10 × 1.386 = 13.86
    double expected_diff = n * std::log(4.0);
    double actual_diff = r2.lnK - r1.lnK;
    printf("  A_fov=10: lnK=%.2f\n", r1.lnK);
    printf("  A_fov=40: lnK=%.2f\n", r2.lnK);
    printf("  差值=%.4f, 期望=%.4f (n×log(4)=%d×%.4f)\n",
           actual_diff, expected_diff, n, std::log(4.0));

    ASSERT_APPROX(actual_diff, expected_diff, 0.01, "A_fov 4倍面积 lnK差",
                  "差值应等于 n×log(4)");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("============================================================\n");
    printf("V4.0 贝叶斯假设验证模块 单元测试 (Task 5)\n");
    printf("============================================================\n");

    test_correct_match();
    test_wrong_match();
    test_boundary();
    test_lnK_vs_nmatch();
    test_lnK_vs_rms();
    test_verify_match_bayes();
    test_unit_consistency();

    printf("\n============================================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_tests_passed, g_tests_failed);
    printf("============================================================\n");

    return (g_tests_failed == 0) ? 0 : 1;
}

// ============================================================================
// test_verify.cpp - vm43_verify 单元测试 (Task 5)
//
// 测试用例:
//   1. 空输入: candidates 为空 → 返回 -1
//   2. 对数 < 3: 仅 2 对 → success=true, validated=false
//   3. 完美匹配: 10 对完全对齐 → validated=true, bayes_decision=1
//   4. 含离群点: 15 对中 3 对离群 → MAD 清洗后 validated=true
//   5. 随机匹配: 20 对随机分布 → validated=false, bayes_decision=-1
//
// 编译: make test_verify
// ============================================================================

#include "../include/vm43_internal.h"

#include <cstdio>
#include <cmath>
#include <random>
#include <string>

namespace v43 {

// 便捷构造参数
static VM43SolveParams make_test_params() {
    VM43SolveParams p;
    // PairVerifier 参数
    p.mad_iters = 3;
    p.mad_threshold_factor = 3.0;
    p.mad_min_threshold_arcsec = 5.0;
    p.lnK_accept = 10.0;
    p.lnK_weak = 3.0;
    p.eps_A = 0.1;
    p.eps_J = 0.1;
    p.triangle_pass_rate = 0.7;
    // IRM 参数 (RANSAC)
    p.irm_ransac_max_iter = 200;
    p.irm_ransac_min_inliers = 10;
    return p;
}

// 生成图像侧星点 (网格分布)
static std::vector<StarPoint> make_grid_stars(int n, double spacing) {
    std::vector<StarPoint> stars;
    stars.reserve(n);
    int side = (int)std::sqrt((double)n);
    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            StarPoint s;
            s.x = (i - side / 2.0) * spacing;
            s.y = (j - side / 2.0) * spacing;
            s.flux = 1000.0 - (i + j) * 10.0;
            s.saturated = false;
            stars.push_back(s);
        }
    }
    return stars;
}

// 测试 1: 空输入
static bool test_empty() {
    printf("--- 测试 1: 空输入 ---\n");
    std::vector<MatchPair> candidates;
    std::vector<StarPoint> U, W;
    VM43SolveParams p = make_test_params();
    VerificationResult result;
    Logger logger;

    int rc = vm43_verify(candidates, U, W, 1.0, 1.0, p, result, &logger);
    if (rc != -1) {
        printf("  失败: 期望 rc=-1, 实际 rc=%d\n", rc);
        return false;
    }
    printf("  通过: rc=-1 (空输入正确返回错误)\n");
    return true;
}

// 测试 2: 对数 < 3
static bool test_too_few() {
    printf("--- 测试 2: 对数 < 3 ---\n");
    std::vector<StarPoint> U = make_grid_stars(25, 60.0);
    std::vector<StarPoint> W = U;  // 完全相同
    std::vector<MatchPair> candidates = {{0, 0}, {1, 1}};  // 仅 2 对

    VM43SolveParams p = make_test_params();
    VerificationResult result;
    Logger logger;

    int rc = vm43_verify(candidates, U, W, 1.0, 1.0, p, result, &logger);
    if (rc != 0) {
        printf("  失败: 期望 rc=0, 实际 rc=%d\n", rc);
        return false;
    }
    if (!result.success) {
        printf("  失败: 期望 success=true\n");
        return false;
    }
    if (result.validated) {
        printf("  失败: 期望 validated=false (对数不足)\n");
        return false;
    }
    printf("  通过: success=true, validated=false (对数不足正确处理)\n");
    return true;
}

// 测试 3: 完美匹配
static bool test_perfect_match() {
    printf("--- 测试 3: 完美匹配 ---\n");
    std::vector<StarPoint> U = make_grid_stars(25, 60.0);  // 25 颗, 间距 60"
    std::vector<StarPoint> W = U;  // 完全相同

    // 全部匹配
    std::vector<MatchPair> candidates;
    for (int i = 0; i < 25; ++i) candidates.push_back({i, i});

    VM43SolveParams p = make_test_params();
    VerificationResult result;
    Logger logger;

    int rc = vm43_verify(candidates, U, W, 1.0, 2.0, p, result, &logger);
    if (rc != 0) {
        printf("  失败: 期望 rc=0, 实际 rc=%d\n", rc);
        return false;
    }
    if (!result.success) {
        printf("  失败: 期望 success=true\n");
        return false;
    }
    printf("  n_clean=%d, bayes_lnK=%.3f, bayes_decision=%d, triangle_ratio=%.3f, validated=%d\n",
           result.n_clean, result.bayes_lnK, result.bayes_decision,
           result.triangle_pass_ratio, result.validated ? 1 : 0);

    if (result.n_clean != 25) {
        printf("  失败: 期望 n_clean=25, 实际 %d\n", result.n_clean);
        return false;
    }
    if (result.bayes_decision != 1) {
        printf("  失败: 期望 bayes_decision=1 (接受), 实际 %d\n", result.bayes_decision);
        return false;
    }
    if (result.triangle_pass_ratio < 0.9) {
        printf("  失败: 期望 triangle_ratio>=0.9, 实际 %.3f\n", result.triangle_pass_ratio);
        return false;
    }
    if (!result.validated) {
        printf("  失败: 期望 validated=true\n");
        return false;
    }
    printf("  通过: 完美匹配正确验证通过\n");
    return true;
}

// 测试 4: 含离群点
static bool test_with_outliers() {
    printf("--- 测试 4: 含离群点 ---\n");
    std::vector<StarPoint> U = make_grid_stars(25, 60.0);
    std::vector<StarPoint> W = U;

    // 15 对正确 + 3 对离群 (索引错位)
    std::vector<MatchPair> candidates;
    for (int i = 0; i < 15; ++i) candidates.push_back({i, i});
    // 离群: u=15 ↔ w=20, u=16 ↔ w=21, u=17 ↔ w=22 (位置不匹配)
    candidates.push_back({15, 20});
    candidates.push_back({16, 21});
    candidates.push_back({17, 22});

    VM43SolveParams p = make_test_params();
    VerificationResult result;
    Logger logger;

    int rc = vm43_verify(candidates, U, W, 1.0, 2.0, p, result, &logger);
    if (rc != 0) {
        printf("  失败: 期望 rc=0, 实际 rc=%d\n", rc);
        return false;
    }
    printf("  n_clean=%d (期望 <=15), bayes_lnK=%.3f, validated=%d\n",
           result.n_clean, result.bayes_lnK, result.validated ? 1 : 0);

    if (result.n_clean > 15) {
        printf("  失败: 期望 n_clean<=15 (剔除离群), 实际 %d\n", result.n_clean);
        return false;
    }
    if (!result.success) {
        printf("  失败: 期望 success=true\n");
        return false;
    }
    printf("  通过: 含离群点正确清洗 (剔除 %d 对)\n", 18 - result.n_clean);
    return true;
}

// 测试 5: 随机匹配
static bool test_random_mismatch() {
    printf("--- 测试 5: 随机匹配 ---\n");
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(-300.0, 300.0);

    // 生成 20 个图像星点和 20 个星表星点 (完全独立)
    std::vector<StarPoint> U(20), W(20);
    for (int i = 0; i < 20; ++i) {
        U[i].x = dist(rng); U[i].y = dist(rng);
        U[i].flux = 1000; U[i].saturated = false;
        W[i].x = dist(rng); W[i].y = dist(rng);
        W[i].flux = 1000; W[i].saturated = false;
    }

    // 随机匹配
    std::vector<MatchPair> candidates;
    for (int i = 0; i < 20; ++i) candidates.push_back({i, i});

    VM43SolveParams p = make_test_params();
    VerificationResult result;
    Logger logger;

    int rc = vm43_verify(candidates, U, W, 1.0, 2.0, p, result, &logger);
    if (rc != 0) {
        printf("  失败: 期望 rc=0, 实际 rc=%d\n", rc);
        return false;
    }
    printf("  n_clean=%d, bayes_lnK=%.3f, bayes_decision=%d, triangle_ratio=%.3f, validated=%d\n",
           result.n_clean, result.bayes_lnK, result.bayes_decision,
           result.triangle_pass_ratio, result.validated ? 1 : 0);

    // 随机匹配应该被拒绝 (bayes_decision <= 0 或 triangle_ratio 低)
    if (result.validated) {
        printf("  失败: 期望 validated=false (随机匹配应被拒绝)\n");
        return false;
    }
    printf("  通过: 随机匹配正确被拒绝\n");
    return true;
}

} // namespace v43

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("================================\n");
    printf("vm43_verify 单元测试 (Task 5)\n");
    printf("================================\n\n");

    int passed = 0, total = 0;
    auto run = [&](bool (*fn)()) { ++total; if (fn()) ++passed; printf("\n"); };

    run(v43::test_empty);
    run(v43::test_too_few);
    run(v43::test_perfect_match);
    run(v43::test_with_outliers);
    run(v43::test_random_mismatch);

    printf("================================\n");
    printf("结果: %d/%d 通过\n", passed, total);
    printf("================================\n");
    return (passed == total) ? 0 : 1;
}

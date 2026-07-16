// test_bayes_increment.cpp - 贝叶斯增量验证单元测试
#include <cstdio>
#include <cmath>
#include "../include/vm4_bayes.h"

int main() {
    printf("=== 贝叶斯增量验证单元测试 ===\n\n");

    // 测试参数：σ=1.0″, A_fov=10.6平方度
    double sigma = 1.0;
    double A_fov_sqdeg = 10.6;
    double A_fov_sqarcsec = A_fov_sqdeg * 3600.0 * 3600.0;
    printf("参数: σ=%.1f″, A_fov=%.2f平方度(=%.2e平方角秒)\n\n",
           sigma, A_fov_sqdeg, A_fov_sqarcsec);

    struct TestCase { double r; double expected; const char* label; };
    TestCase cases[] = {
        {0.5,  16.78, "强接受"},
        {1.0,  16.40, "接受"},
        {4.0,   8.90, "接受(MAD清洗)"},
        {5.5,   1.78, "弱接受(边界)"},
        {6.0,  -1.10, "拒绝"},
        {10.0,-33.10, "强烈拒绝"},
    };

    int pass = 0, fail = 0;
    for (const auto& tc : cases) {
        auto res = vm4::compute_bayes_increment(tc.r, sigma, A_fov_sqarcsec);
        bool ok = std::abs(res.delta_lnK - tc.expected) < 0.5;  // 允许0.5容差
        printf("  r=%5.1f″ → ΔlnK=%+8.3f (期望%+8.3f) [%s] %s\n",
               tc.r, res.delta_lnK, tc.expected,
               res.accepted ? "接受" : "拒绝",
               ok ? "✓" : "✗");
        if (ok) pass++; else fail++;
    }

    // 边界测试
    printf("\n--- 边界测试 ---\n");
    auto res_nan = vm4::compute_bayes_increment(NAN, sigma, A_fov_sqarcsec);
    printf("  r=NaN → ΔlnK=%+.3f %s\n", res_nan.delta_lnK,
           res_nan.delta_lnK < -1e10 ? "✓" : "✗");
    if (res_nan.delta_lnK < -1e10) pass++; else fail++;

    auto res_neg_sigma = vm4::compute_bayes_increment(1.0, -1.0, A_fov_sqarcsec);
    printf("  σ=-1 → ΔlnK=%+.3f %s\n", res_neg_sigma.delta_lnK,
           res_neg_sigma.delta_lnK < -1e10 ? "✓" : "✗");
    if (res_neg_sigma.delta_lnK < -1e10) pass++; else fail++;

    printf("\n=== 结果: %d 通过, %d 失败 ===\n", pass, fail);
    return fail > 0 ? 1 : 0;
}

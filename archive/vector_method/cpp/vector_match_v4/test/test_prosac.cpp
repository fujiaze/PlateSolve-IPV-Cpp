// ============================================================================
// test_prosac.cpp - V4.0 PROSAC 优先采样模块单元测试（Task 4.5）
//
// 测试项：
//   1. 构造 250 颗星（前 50 饱和高 SNR，后 200 普通），计算质量分
//   2. 验证前 50 颗饱和星排名靠前
//   3. 验证 g(t): t=100, n=250, T_max=10000 → g(100)≈54
//   4. 采样 1000 次，统计前 54 颗被选频率显著高于后 200 颗
//   5. 验证探索性：后 200 颗也有一定被选概率（30% 均匀采样）
//
// 编译运行：
//   g++ -O2 -std=c++17 -Wall -I include -o test/test_prosac.exe
//       src/vm4_prosac.cpp test/test_prosac.cpp
//   ./test/test_prosac.exe
// ============================================================================

#include "../include/vm4_prosac.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

using namespace vm4;

// 测试计数器
static int g_pass = 0;
static int g_fail = 0;

// 断言宏：打印 PASS/FAIL 并计数
#define EXPECT_TRUE(cond, msg) do { \
    if (cond) { printf("  [PASS] %s\n", (msg)); ++g_pass; } \
    else      { printf("  [FAIL] %s\n", (msg)); ++g_fail; } \
} while(0)

#define EXPECT_NEAR(actual, expected, tol, msg) do { \
    double _a = (actual), _e = (expected), _t = (tol); \
    bool _ok = std::fabs(_a - _e) <= _t; \
    if (_ok) { printf("  [PASS] %s (actual=%.4f expected=%.4f)\n", (msg), _a, _e); ++g_pass; } \
    else     { printf("  [FAIL] %s (actual=%.4f expected=%.4f tol=%.4f)\n", (msg), _a, _e, _t); ++g_fail; } \
} while(0)

// ============================================================================
// 测试 1+2: 构造 250 颗星，计算质量分，验证前 50 颗排名靠前
// ============================================================================
static void test_quality_score() {
    printf("\n[测试 1+2] 质量分计算与排名\n");

    const int N = 250;
    std::vector<double> snr(N), sparsity(N);
    std::vector<bool> is_sat(N, false);

    // 前 50 颗：饱和星，高 SNR(150~174.5)，高稀疏度(20~24.9)
    for (int i = 0; i < 50; ++i) {
        snr[i] = 150.0 + i * 0.5;
        sparsity[i] = 20.0 + i * 0.1;
        is_sat[i] = true;
    }
    // 后 200 颗：普通星，低 SNR(20~39.9)，低稀疏度(5~14.95)
    for (int i = 50; i < N; ++i) {
        snr[i] = 20.0 + (i - 50) * 0.1;
        sparsity[i] = 5.0 + (i - 50) * 0.05;
        is_sat[i] = false;
    }

    double w_snr = 0.4, w_sparse = 0.4, w_sat = 0.2;
    auto sorted = compute_quality_score(snr, sparsity, is_sat, w_snr, w_sparse, w_sat);

    EXPECT_TRUE(sorted.size() == (size_t)N, "质量分结果数量=250");

    // 验证降序排列
    bool is_descending = true;
    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i].quality_score > sorted[i-1].quality_score - 1e-12) {
            // 允许相等（stable_sort），但绝不能严格大于
            if (sorted[i].quality_score > sorted[i-1].quality_score + 1e-12) {
                is_descending = false; break;
            }
        }
    }
    EXPECT_TRUE(is_descending, "质量分按降序排列");

    // 验证前 50 名全部来自原始索引 0~49（饱和星）
    int sat_in_top50 = 0;
    for (int i = 0; i < 50; ++i) {
        if (sorted[i].index < 50) ++sat_in_top50;
    }
    printf("    前 50 名中饱和星数量: %d / 50\n", sat_in_top50);
    EXPECT_TRUE(sat_in_top50 == 50, "前 50 名全部为饱和星(index<50)");

    // 验证第 1 名质量分接近 1.0（SNR/sparsity 归一化最高 + 饱和加成）
    EXPECT_NEAR(sorted[0].quality_score, 1.0, 0.05, "第 1 名质量分≈1.0");

    // 验证最后一名质量分显著低于第 1 名
    double gap = sorted[0].quality_score - sorted[N-1].quality_score;
    printf("    质量分 gap: top=%.4f tail=%.4f diff=%.4f\n",
           sorted[0].quality_score, sorted[N-1].quality_score, gap);
    EXPECT_TRUE(gap > 0.5, "质量分 top-tail 差距>0.5");
}

// ============================================================================
// 测试 3: 验证 g(t) 增长函数
// ============================================================================
static void test_pool_size() {
    printf("\n[测试 3] PROSAC 增长函数 g(t)\n");

    int n = 250, T_max = 10000;

    // g(100) = 250 × (100/10000)^(1/3) = 250 × 0.2154 = 53.85 → ceil = 54
    int g100 = prosac_pool_size(100, n, T_max);
    double expected_g = 250.0 * std::cbrt(100.0 / 10000.0);
    printf("    g(100)=%d (理论值=%.4f, ceil=%.4f)\n",
           g100, expected_g, std::ceil(expected_g));
    EXPECT_NEAR((double)g100, 54.0, 1.0, "g(100)≈54");

    // g(1) = 250 × (1/10000)^(1/3) = 250 × 0.0464 = 11.6 → ceil = 12
    int g1 = prosac_pool_size(1, n, T_max);
    double exp_g1 = 250.0 * std::cbrt(1.0 / 10000.0);
    printf("    g(1)=%d (理论值=%.4f)\n", g1, exp_g1);
    EXPECT_TRUE(g1 >= 1 && g1 <= 15, "g(1)∈[1,15]");

    // g(T_max) = n（全星采样）
    int gT = prosac_pool_size(T_max, n, T_max);
    EXPECT_TRUE(gT == n, "g(T_max)=n (全星采样)");

    // g(T_max/2) < n
    int ghalf = prosac_pool_size(T_max/2, n, T_max);
    EXPECT_TRUE(ghalf < n && ghalf > g100, "g(T_max/2) 介于 g(100) 和 n 之间");

    // 边界：t<=0 返回 1
    EXPECT_TRUE(prosac_pool_size(0, n, T_max) == 1, "g(0)=1 (安全降级)");
    EXPECT_TRUE(prosac_pool_size(-5, n, T_max) == 1, "g(-5)=1 (安全降级)");

    // 边界：n=1
    EXPECT_TRUE(prosac_pool_size(100, 1, T_max) == 1, "n=1 时返回 1");

    // 单调性：g(t) 随 t 递增
    bool monotonic = true;
    int prev = 0;
    for (int t = 1; t <= T_max; t += 100) {
        int cur = prosac_pool_size(t, n, T_max);
        if (cur < prev) { monotonic = false; break; }
        prev = cur;
    }
    EXPECT_TRUE(monotonic, "g(t) 随 t 单调不减");
}

// ============================================================================
// 测试 4+5: 采样分布统计
// ============================================================================
static void test_sampling_distribution() {
    printf("\n[测试 4+5] 采样分布统计\n");

    const int N = 250;
    std::vector<double> snr(N), sparsity(N);
    std::vector<bool> is_sat(N, false);
    for (int i = 0; i < 50; ++i) {
        snr[i] = 150.0 + i * 0.5;
        sparsity[i] = 20.0 + i * 0.1;
        is_sat[i] = true;
    }
    for (int i = 50; i < N; ++i) {
        snr[i] = 20.0 + (i - 50) * 0.1;
        sparsity[i] = 5.0 + (i - 50) * 0.05;
        is_sat[i] = false;
    }

    auto sorted = compute_quality_score(snr, sparsity, is_sat, 0.4, 0.4, 0.2);

    ProsacSampler sampler;
    sampler.init(sorted, 10000, 42u);

    // 固定 t=100（pool=54）采样 1000 次
    const int T_SAMPLE = 1000;
    const int t_fixed = 100;
    const int expected_pool = 54;

    // 验证 last_pool_size
    sampler.sample(t_fixed);
    EXPECT_TRUE(sampler.last_pool_size() == expected_pool,
                "t=100 时 last_pool_size=54");

    std::vector<int> freq(N, 0);
    for (int k = 0; k < T_SAMPLE; ++k) {
        int idx = sampler.sample(t_fixed);
        if (idx >= 0 && idx < N) freq[idx]++;
    }

    // 统计前 54 颗（按质量排名）vs 后 200 颗的被选频率
    int top_count = 0, tail_count = 0;
    for (int i = 0; i < expected_pool; ++i) {
        int orig_idx = sorted[i].index;
        top_count += freq[orig_idx];
    }
    for (int i = expected_pool; i < N; ++i) {
        int orig_idx = sorted[i].index;
        tail_count += freq[orig_idx];
    }

    double avg_top  = (double)top_count  / expected_pool;
    double avg_tail = (double)tail_count / (N - expected_pool);
    printf("    采样 %d 次 (t=%d, pool=%d):\n", T_SAMPLE, t_fixed, expected_pool);
    printf("    前 %d 颗: 总命中=%d, 平均%.2f 次/颗\n", expected_pool, top_count, avg_top);
    printf("    后 %d 颗: 总命中=%d, 平均%.2f 次/颗\n", N-expected_pool, tail_count, avg_tail);
    printf("    频率比: %.2fx\n", avg_top / std::max(0.01, avg_tail));

    // 测试 4: 前 54 颗被选频率显著高于后 200 颗（至少 3x）
    EXPECT_TRUE(avg_top > 3.0 * avg_tail,
                "前 54 颗频率 > 3x 后 200 颗频率");

    // 测试 5: 探索性 — 后 200 颗也有被选中（30% 均匀采样）
    EXPECT_TRUE(tail_count > 0,
                "后 200 颗有被选中 (探索性>0)");

    // 后 200 颗的理论期望：30% × 1000 / 250 ≈ 1.2 次/颗
    // 验证后 200 颗总命中数不低于理论值的 50%（允许随机波动）
    double expected_tail_total = 0.3 * T_SAMPLE;  // 300
    printf("    后 200 颗理论期望(30%%均匀): %.0f, 实际: %d\n",
           expected_tail_total, tail_count);
    EXPECT_TRUE(tail_count > expected_tail_total * 0.5,
                "后 200 颗总命中 > 50% 理论期望(≈300)");

    // 额外验证：每颗前 54 颗都至少被选中 1 次（1000 次采样足够覆盖）
    int uncovered_top = 0;
    for (int i = 0; i < expected_pool; ++i) {
        if (freq[sorted[i].index] == 0) ++uncovered_top;
    }
    printf("    前 54 颗中未被选中的数量: %d\n", uncovered_top);
    EXPECT_TRUE(uncovered_top <= 2, "前 54 颗几乎全部被覆盖(未覆盖<=2)");
}

// ============================================================================
// 测试 prosac_sample_one 便捷函数
// ============================================================================
static void test_convenience_function() {
    printf("\n[附加测试] prosac_sample_one 便捷函数\n");

    std::vector<double> snr = {100.0, 50.0, 20.0};
    std::vector<double> sparsity = {30.0, 15.0, 5.0};
    std::vector<bool> is_sat = {true, false, false};

    auto sorted = compute_quality_score(snr, sparsity, is_sat, 0.4, 0.4, 0.2);
    ProsacSampler sampler;
    sampler.init(sorted, 1000, 7u);

    bool all_valid = true;
    for (int t = 1; t <= 100; ++t) {
        int idx = prosac_sample_one(sampler, t);
        if (idx < 0 || idx >= 3) { all_valid = false; break; }
    }
    EXPECT_TRUE(all_valid, "prosac_sample_one 返回值全部合法");

    EXPECT_NEAR(sampler.quality_median(), sorted[1].quality_score, 1e-9,
                "quality_median 等于中间值(3颗星)");
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("============================================================\n");
    printf("V4.0 PROSAC 优先采样模块单元测试 (Task 4.5)\n");
    printf("============================================================\n");

    test_quality_score();
    test_pool_size();
    test_sampling_distribution();
    test_convenience_function();

    printf("\n============================================================\n");
    printf("测试汇总: PASS=%d  FAIL=%d\n", g_pass, g_fail);
    printf("============================================================\n");

    return (g_fail == 0) ? 0 : 1;
}

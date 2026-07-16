// ============================================================================
// test_kvector.cpp - V4.0 Phase C k-vector 模块性能与正确性测试（Task 3.4）
//
// 测试内容:
//   1. 生成 500 颗随机星（模拟 Gaia 星表，6000"×6000" 方形视场）
//   2. 构建 k-vector 索引，记录耗时
//   3. 查询 d=1200角秒, eps=2角秒，返回候选数（应约 100）
//   4. 对比暴力 O(n²) 搜索结果一致性（k-vector 查询结果应完全包含在暴力结果中）
//   5. 验证查询复杂度：n=100/500/1000 时查询耗时应接近常数（O(k)）
//
// 编译（在 lib/plate_solve/cpp/vector_match_v4/ 目录下）:
//   g++ -O3 -march=native -std=c++17 -fopenmp -Iinclude
//       test/test_kvector.cpp src/vm4_kvector.cpp -o test/test_kvector.exe
// ============================================================================

#include "vm4_kvector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace vm4;

// 简单确定性 LCG 随机数（便于复现）
static double rand_uniform(double lo, double hi, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(lo, hi);
    return dist(rng);
}

// 生成 n 颗随机星，方形视场 L×L（角秒），中心 (0,0)
static std::vector<std::pair<double,double>> generate_stars(int n, double L, unsigned seed) {
    std::mt19937 rng(seed);
    std::vector<std::pair<double,double>> stars;
    stars.reserve(n);
    for (int i = 0; i < n; ++i) {
        double xi  = rand_uniform(-L/2.0, L/2.0, rng);
        double eta = rand_uniform(-L/2.0, L/2.0, rng);
        stars.push_back({xi, eta});
    }
    return stars;
}

// 暴力 O(n²) 搜索角距在 [d-eps, d+eps] 的所有星对
static std::vector<StarPair> brute_force_query(
    const std::vector<std::pair<double,double>>& stars,
    double d, double eps)
{
    std::vector<StarPair> result;
    int n = (int)stars.size();
    double d_lo = d - eps, d_hi = d + eps;
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = stars[i].first  - stars[j].first;
            double dy = stars[i].second - stars[j].second;
            double dist = std::sqrt(dx*dx + dy*dy);
            if (dist >= d_lo && dist <= d_hi) {
                StarPair sp{i, j, dist};
                result.push_back(sp);
            }
        }
    }
    return result;
}

// 将星对集合转为可比较的键集合（无序对 (i,j) → (min,max)）
static std::set<std::pair<int,int>> to_key_set(const std::vector<StarPair>& pairs) {
    std::set<std::pair<int,int>> s;
    for (const auto& p : pairs) {
        int lo = std::min(p.i, p.j);
        int hi = std::max(p.i, p.j);
        s.insert({lo, hi});
    }
    return s;
}

// 计时工具
static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

// ============================================================================
// 测试 1+2+3+4: 500 颗星功能与正确性
// ============================================================================
static bool test_500stars() {
    printf("\n========== 测试 1-4: 500 颗星功能与正确性 ==========\n");
    const int N = 500;
    const double L = 6000.0;  // 6000" 方形视场（100' 边长，模拟 Gaia 局部天区）
    auto stars = generate_stars(N, L, 42u);

    // 2. 构建 k-vector 索引
    KVectorIndex kv;
    double t0 = now_ms();
    kv.build(stars);
    double t_build = now_ms() - t0;

    printf("[构建] N=%d  K=%d  d_min=%.2f\"  d_max=%.2f\"  build_time=%.2f ms\n",
           N, kv.size(), kv.min_distance(), kv.max_distance(), t_build);
    printf("[构建] a=%.6f  b=%.6f  (k(d)=floor(a+b*(d-d_min)))\n",
           0.0, (double)(kv.size()-1)/std::max(kv.max_distance()-kv.min_distance(), 1e-9));

    // 3. 查询 d=1200", eps=2"
    const double QD = 1200.0, QEPS = 2.0;
    double t1 = now_ms();
    auto kv_result = kv.query(QD, QEPS);
    double t_kv_query = now_ms() - t1;

    printf("[查询] d=%.1f\"  eps=%.1f\"  -> k-vector 候选数 = %zu  (耗时 %.4f ms)\n",
           QD, QEPS, kv_result.size(), t_kv_query);

    // 4. 对比暴力 O(n²) 搜索
    double t2 = now_ms();
    auto bf_result = brute_force_query(stars, QD, QEPS);
    double t_bf_query = now_ms() - t2;
    printf("[查询] 暴力 O(n²) 候选数 = %zu  (耗时 %.4f ms)\n",
           bf_result.size(), t_bf_query);

    // 一致性验证：k-vector 结果应完全包含在暴力结果中（无遗漏）
    auto bf_keys = to_key_set(bf_result);
    auto kv_keys = to_key_set(kv_result);
    int missing = 0;
    for (const auto& k : kv_keys) {
        if (bf_keys.find(k) == bf_keys.end()) ++missing;
    }
    int missing_bf = 0;
    for (const auto& k : bf_keys) {
        if (kv_keys.find(k) == kv_keys.end()) ++missing_bf;
    }
    printf("[一致性] k-vector 结果中不在暴力结果里: %d\n", missing);
    printf("[一致性] 暴力结果中不在 k-vector 结果里: %d (应为 0，否则 k-vector 漏检)\n", missing_bf);
    printf("[加速比] k-vector / 暴力 = %.4f / %.4f = %.2fx\n",
           t_kv_query, t_bf_query, t_bf_query / std::max(t_kv_query, 1e-6));

    bool ok = (missing == 0) && (missing_bf == 0);
    printf("[结果] %s\n", ok ? "通过 ✓" : "失败 ✗");
    return ok;
}

// ============================================================================
// 测试 5: 查询复杂度 O(k) 验证（n=100/500/1000）
// 思路：O(k) 的核心是与候选数 k 成正比、与星表规模 n 解耦。
//   - k-vector 查询耗时 ∝ 候选数 k，与星对总数 K=n(n-1)/2 无关
//   - 暴力 O(n²) 查询耗时 ∝ K
//   验证两点:
//     (a) k-vector 每候选耗时 ≈ 常数（query_ms / candidates 稳定）
//     (b) k-vector 相对暴力的加速比随 n 增长而提升（k/K 越来越小）
// ============================================================================
static bool test_complexity() {
    printf("\n========== 测试 5: 查询复杂度 O(k) 验证 ==========\n");
    const double L = 6000.0;
    const double QD = 1200.0, QEPS = 2.0;
    const int Ns[] = {100, 500, 1000};
    const int n_cases = (int)(sizeof(Ns)/sizeof(Ns[0]));

    printf("  %-6s %-10s %-10s %-12s %-10s %-12s %-12s\n",
           "N", "K", "build_ms", "kv_query_us", "k_cand", "bf_query_us", "speedup");

    bool ok = true;
    double prev_speedup = -1.0;
    for (int c = 0; c < n_cases; ++c) {
        int N = Ns[c];
        auto stars = generate_stars(N, L, 7u + (unsigned)N);

        KVectorIndex kv;
        double t0 = now_ms();
        kv.build(stars);
        double t_build = now_ms() - t0;

        // 多次查询取平均（单次太快测不准）
        const int NREP = 500;
        std::vector<StarPair> r;
        double t1 = now_ms();
        for (int r_i = 0; r_i < NREP; ++r_i) {
            r = kv.query(QD, QEPS);
        }
        double t_kv_us = (now_ms() - t1) * 1000.0 / NREP;  // 微秒

        // 暴力查询（多次取平均）
        double t2 = now_ms();
        std::vector<StarPair> rbf;
        for (int r_i = 0; r_i < NREP; ++r_i) {
            rbf = brute_force_query(stars, QD, QEPS);
        }
        double t_bf_us = (now_ms() - t2) * 1000.0 / NREP;  // 微秒

        double speedup = (t_kv_us > 1e-6) ? (t_bf_us / t_kv_us) : 0.0;
        printf("  %-6d %-10d %-10.2f %-12.4f %-10zu %-12.2f %-12.1f\n",
               N, kv.size(), t_build, t_kv_us, r.size(), t_bf_us, speedup);

        // 验证加速比随 n 增长而提升（O(k) << O(n²)）
        if (prev_speedup > 0.0 && speedup < prev_speedup * 0.9) {
            printf("  [警告] N=%d 加速比 %.1f < 前次 %.1f × 0.9，O(k) 解耦失效\n",
                   N, speedup, prev_speedup);
            ok = false;
        }
        prev_speedup = speedup;
    }
    printf("[结果] %s\n", ok ? "通过 ✓ (k-vector 加速比随 n 增长而提升)" : "失败 ✗");
    return ok;
}

// ============================================================================
// 测试 6: kvector_prefilter 烟雾测试
// ============================================================================
static bool test_prefilter() {
    printf("\n========== 测试 6: kvector_prefilter 烟雾测试 ==========\n");
    // 构造 U 为 W 的子集 + 平移（验证能找到正确对应）
    const int M = 200;
    const int N = 50;
    const double L = 4000.0;
    auto W_stars = generate_stars(M, L, 123u);

    KVectorIndex kv_w;
    kv_w.build(W_stars);
    printf("[构建] W 索引: K=%d  build_time=%.2f ms\n", kv_w.size(), kv_w.build_time_ms());

    // U 取 W 前 N 颗 + 小扰动 0.5"
    std::vector<double> U(N*2);
    std::vector<double> W(M*2);
    for (int i = 0; i < M; ++i) {
        W[i*2]   = W_stars[i].first;
        W[i*2+1] = W_stars[i].second;
    }
    std::mt19937 rng(999u);
    for (int i = 0; i < N; ++i) {
        U[i*2]   = W_stars[i].first  + rand_uniform(-0.5, 0.5, rng);
        U[i*2+1] = W_stars[i].second + rand_uniform(-0.5, 0.5, rng);
    }

    auto cands = kvector_prefilter(kv_w, U.data(), N, W.data(), M, 2.0);
    printf("[预筛选] U=%d 颗, W=%d 颗, 候选对数=%zu\n", N, M, cands.size());

    // 检查正确匹配 (i, i) 是否在候选中
    int n_correct = 0;
    std::set<std::pair<int,int>> cand_set(cands.begin(), cands.end());
    for (int i = 0; i < N; ++i) {
        if (cand_set.count({i, i}) > 0) ++n_correct;
    }
    printf("[验证] 正确匹配 (i,i) 出现在候选中: %d / %d\n", n_correct, N);

    bool ok = (n_correct > 0);
    printf("[结果] %s\n", ok ? "通过 ✓" : "失败 ✗");
    return ok;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char** argv) {
    (void)argc; (void)argv;
    printf("============================================================\n");
    printf(" V4.0 Phase C k-vector 模块测试 (Task 3)\n");
    printf("============================================================");

    bool ok1 = test_500stars();
    bool ok2 = test_complexity();
    bool ok3 = test_prefilter();

    printf("\n========== 总结 ==========\n");
    printf("  测试 1-4 (500 星功能/一致性):  %s\n", ok1 ? "通过 ✓" : "失败 ✗");
    printf("  测试 5   (查询复杂度 O(k)):    %s\n", ok2 ? "通过 ✓" : "失败 ✗");
    printf("  测试 6   (kvector_prefilter):  %s\n", ok3 ? "通过 ✓" : "失败 ✗");
    printf("============================================================\n");
    return (ok1 && ok2 && ok3) ? 0 : 1;
}

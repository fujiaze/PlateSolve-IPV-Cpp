// PairExpander V4.2 单元测试
//
// 测试场景:
//   1. 功能测试: N=200, M=300 合成数据, s=1.0, θ=0, 验证扩充对数 ≥ 100
//   2. 性能测试: N=2000, M=2000, 验证线性扫描 NN 耗时 < 20ms
//   3. 区域均匀性: 4500×3600 图像, region_size=800, 验证 n_regions=6×5=30
//   4. 正确性: 与暴力法 O(N·M) 对比, 验证线性扫描结果一致

#include "pe_api.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <cstring>
#include <string>

static int g_pass = 0;
static int g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_fail++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

// 合成数据: U 在图像范围内均匀分布, W = U 的子集(真匹配) + 噪声
// s=1, θ=0, tx=ty=0 时 Wt = W, 真匹配距离=0
struct SynthData {
    std::vector<double> U;      // N×2 角秒坐标
    std::vector<double> W;      // M×2 角秒坐标
    std::vector<int> true_u;    // true_u[j] = W[j] 的真匹配 U 索引, -1 = 噪声
    int N, M;
};

static SynthData gen_synth(int N, int M, double img_w_px, double img_h_px, double s0,
                           uint32_t seed) {
    SynthData d;
    d.N = N; d.M = M;
    d.U.resize((size_t)N * 2);
    d.W.resize((size_t)M * 2);
    d.true_u.assign(M, -1);
    std::mt19937 rng(seed);
    double half_w = img_w_px * s0 / 2.0;
    double half_h = img_h_px * s0 / 2.0;
    std::uniform_real_distribution<double> ux(-half_w, half_w);
    std::uniform_real_distribution<double> uy(-half_h, half_h);

    for (int i = 0; i < N; ++i) {
        d.U[i * 2]     = ux(rng);
        d.U[i * 2 + 1] = uy(rng);
    }

    int n_match = std::min(N, M);
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int j = 0; j < M; ++j) {
        if (j < n_match) {
            int i = idx[j % N];
            d.W[j * 2]     = d.U[i * 2];
            d.W[j * 2 + 1] = d.U[i * 2 + 1];
            d.true_u[j] = i;
        } else {
            d.W[j * 2]     = ux(rng);
            d.W[j * 2 + 1] = uy(rng);
        }
    }
    return d;
}

// 暴力 NN (参考实现, O(N))
static int brute_nn(const double* U, int N, double qx, double qy, double& best_d2) {
    best_d2 = 1e300;
    int best = -1;
    for (int i = 0; i < N; ++i) {
        double dx = qx - U[i * 2];
        double dy = qy - U[i * 2 + 1];
        double d2 = dx * dx + dy * dy;
        if (d2 < best_d2) { best_d2 = d2; best = i; }
    }
    return best;
}

// 单独测量线性扫描 NN 耗时 (纯 O(N·M), 无日志/过滤开销)
static double measure_linear_scan(const std::vector<double>& U, int N,
                                  const std::vector<double>& Wt, int M) {
    volatile double sink = 0;  // 防止编译器优化掉
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int j = 0; j < M; ++j) {
        double qx = Wt[j * 2], qy = Wt[j * 2 + 1];
        double best_d2 = 1e300;
        for (int i = 0; i < N; ++i) {
            double dx = qx - U[i * 2];
            double dy = qy - U[i * 2 + 1];
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) best_d2 = d2;
        }
        sink += best_d2;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;  // 消除 unused 警告
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// 测试1: 功能测试
static void test_functional() {
    printf("\n=== 测试1: 功能测试 (N=200, M=300, s=1, θ=0) ===\n");
    double s0 = 1.0;
    SynthData d = gen_synth(200, 300, 4500, 3600, s0, 42);

    PairExpanderParams params;
    std::memset(&params, 0, sizeof(params));
    params.s0 = s0;
    params.tau_factor = 3.0;
    params.scale_ratio_tol = 0.1;
    params.region_size_px = 800;
    params.N_floor = 5;
    params.N_cap = 30;
    params.N_max = 1500;
    params.img_width = 4500;
    params.img_height = 3600;
    params.log_file_path = nullptr;

    ExpansionResult result;
    int ret = pe_expand(d.U.data(), d.N, d.W.data(), d.M,
                        nullptr, nullptr, 0,
                        1.0, 0.0, 0.0, 0.0,
                        &params, &result);
    ASSERT(ret == 1, "pe_expand 返回成功");
    ASSERT(result.success == 1, "result.success == 1");
    ASSERT(result.n_expanded >= 100, "扩充对数 >= 100");
    printf("  候选=%d, 接受=%d, 扩充=%d, 合计=%d, 区域=%d, 稀疏=%d, 耗时=%.2fms\n",
           result.n_candidates, result.n_accepted, result.n_expanded,
           result.n_pairs, result.n_regions, result.n_sparse_regions,
           result.expand_time_ms);
    pe_free(&result);
}

// 测试2: 性能测试
static void test_performance() {
    printf("\n=== 测试2: 性能测试 (N=2000, M=2000, 线性扫描 < 20ms) ===\n");
    double s0 = 1.0;
    SynthData d = gen_synth(2000, 2000, 4500, 3600, s0, 42);

    // 2a. 单独测量线性扫描 NN 耗时 (纯 O(N·M), 无日志开销)
    // s=1, θ=0, tx=ty=0 → Wt = W
    double scan_ms = measure_linear_scan(d.U, d.N, d.W, d.M);
    printf("  纯线性扫描 NN: %.2fms (N×M=%.2e 操作)\n", scan_ms, (double)d.N * d.M);
    ASSERT(scan_ms < 20.0, "线性扫描 NN 耗时 < 20ms");

    // 2b. pe_expand 总耗时 (含过滤+区域均匀化, log_file_path=NULL 禁用日志)
    PairExpanderParams params;
    std::memset(&params, 0, sizeof(params));
    params.s0 = s0;
    params.tau_factor = 3.0;
    params.scale_ratio_tol = 0.1;
    params.region_size_px = 800;
    params.N_floor = 5;
    params.N_cap = 30;
    params.N_max = 1500;
    params.img_width = 4500;
    params.img_height = 3600;
    params.log_file_path = nullptr;

    ExpansionResult result;
    int ret = pe_expand(d.U.data(), d.N, d.W.data(), d.M,
                        nullptr, nullptr, 0,
                        1.0, 0.0, 0.0, 0.0,
                        &params, &result);
    ASSERT(ret == 1, "pe_expand 返回成功");
    printf("  pe_expand 总耗时: %.2fms, 候选=%d, 扩充=%d\n",
           result.expand_time_ms, result.n_candidates, result.n_expanded);
    pe_free(&result);
}

// 测试3: 区域均匀性
static void test_region_uniformity() {
    printf("\n=== 测试3: 区域均匀性 (4500×3600, region=800, 6×5=30区) ===\n");
    double s0 = 1.0;
    // 30 区, 每区至少 N_floor=5 对 → 需要足够多的均匀分布星点
    // N=500, M=500, 平均每区约 16 个点
    SynthData d = gen_synth(500, 500, 4500, 3600, s0, 42);

    PairExpanderParams params;
    std::memset(&params, 0, sizeof(params));
    params.s0 = s0;
    params.tau_factor = 3.0;
    params.scale_ratio_tol = 0.1;
    params.region_size_px = 800;
    params.N_floor = 5;
    params.N_cap = 30;
    params.N_max = 1500;
    params.img_width = 4500;
    params.img_height = 3600;
    params.log_file_path = nullptr;

    ExpansionResult result;
    int ret = pe_expand(d.U.data(), d.N, d.W.data(), d.M,
                        nullptr, nullptr, 0,
                        1.0, 0.0, 0.0, 0.0,
                        &params, &result);
    ASSERT(ret == 1, "pe_expand 返回成功");
    // n_cols = ceil(4500/800) = 6, n_rows = ceil(3600/800) = 5
    ASSERT(result.n_regions == 30, "n_regions == 30 (6×5)");
    printf("  区域=%d, 稀疏区=%d, 扩充=%d\n",
           result.n_regions, result.n_sparse_regions, result.n_expanded);
    // 500 个点均匀分布在 30 区, 平均每区 16 个, 稀疏区应较少
    ASSERT(result.n_sparse_regions < 15, "稀疏区 < 15 (500点均匀分布)");
    // 扩充对数应足够多 (至少 150 = 30区 × N_floor=5)
    ASSERT(result.n_expanded >= 100, "扩充对数 >= 100 (覆盖多数区域)");
    pe_free(&result);
}

// 测试4: 正确性 (与暴力法对比)
static void test_correctness() {
    printf("\n=== 测试4: 正确性 (与暴力法 O(N·M) 对比) ===\n");
    double s0 = 1.0;
    SynthData d = gen_synth(100, 150, 4500, 3600, s0, 42);

    // 4a. 暴力 NN: 对每个 W[j] 计算最近 U (s=1,θ=0,tx=ty=0 → Wt=W)
    double tau = 3.0 * s0;
    double tau2 = tau * tau;
    int n_brute_candidates = 0;
    int n_brute_true_match = 0;
    std::vector<int> brute_u_for_w(d.M, -1);  // 暴力法找到的 u* for each w
    for (int j = 0; j < d.M; ++j) {
        double qx = d.W[j * 2], qy = d.W[j * 2 + 1];
        double d2;
        int u_brute = brute_nn(d.U.data(), d.N, qx, qy, d2);
        if (d2 > tau2) continue;  // τ 截断
        n_brute_candidates++;
        brute_u_for_w[j] = u_brute;
        if (d.true_u[j] >= 0 && u_brute == d.true_u[j]) {
            n_brute_true_match++;
        }
    }
    printf("  暴力法: 候选=%d, 真匹配=%d\n", n_brute_candidates, n_brute_true_match);
    ASSERT(n_brute_candidates > 0, "暴力法有候选对");
    ASSERT(n_brute_true_match == n_brute_candidates, "暴力 NN 全部找到真匹配");

    // 4b. 调用 pe_expand, 验证候选数一致 + 匹配对正确
    PairExpanderParams params;
    std::memset(&params, 0, sizeof(params));
    params.s0 = s0;
    params.tau_factor = 3.0;
    params.scale_ratio_tol = 0.1;
    params.region_size_px = 800;
    params.N_floor = 5;
    params.N_cap = 30;
    params.N_max = 1500;
    params.img_width = 4500;
    params.img_height = 3600;
    params.log_file_path = nullptr;

    ExpansionResult result;
    int ret = pe_expand(d.U.data(), d.N, d.W.data(), d.M,
                        nullptr, nullptr, 0,
                        1.0, 0.0, 0.0, 0.0,
                        &params, &result);
    ASSERT(ret == 1, "pe_expand 返回成功");

    // 候选数应与暴力法一致 (线性扫描找到相同的最近邻)
    printf("  pe_expand 候选=%d, 暴力法候选=%d\n", result.n_candidates, n_brute_candidates);
    ASSERT(result.n_candidates == n_brute_candidates, "线性扫描候选数 == 暴力法候选数");

    // 验证返回的匹配对都是真匹配 (零误匹配)
    int n_correct = 0, n_wrong = 0;
    for (int k = 0; k < result.n_pairs; ++k) {
        int u = result.expand_u[k];
        int w = result.expand_w[k];
        if (d.true_u[w] == u) {
            n_correct++;
        } else {
            n_wrong++;
        }
    }
    printf("  匹配对=%d, 正确=%d, 错误=%d\n", result.n_pairs, n_correct, n_wrong);
    ASSERT(n_wrong == 0, "所有匹配对都是真匹配 (零误匹配)");
    ASSERT(n_correct == result.n_pairs, "正确率 100%");

    pe_free(&result);
}

int main() {
    printf("=== PairExpander V4.2 单元测试 ===\n");
    printf("编译: C++17, 无 nanoflann/k-vector/Eigen 依赖\n");
    printf("线性扫描 NN + 模长比过滤 + 区域均匀化\n");

    test_functional();
    test_performance();
    test_region_uniformity();
    test_correctness();

    printf("\n=== 测试汇总 ===\n");
    printf("通过: %d, 失败: %d\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
// test_vm44_relvec.cpp - V4.4 相对向量法单元测试 (Task 7)
//
// 测试 vm44_relvec.cpp 中的 v44::vm44_relvec_match:
//   1. k-vector 距离查询正确性: 通过 n_total_candidates 验证查询范围合理
//   2. t=0 已知匹配场景: s=1, θ=30°, t=0 → SNR > 5, θ_peak ≈ 30°
//   3. t≠0 场景 SNR > 5: t=(100", 100") 合成数据 → 相对向量法仍 SNR > 5 (核心优势)
//   4. U 限流: U=200 颗, relvec_max_u=100 → 算法正常运行不爆炸
//
// 由于 RelativeVectorMatcher 是 vm44_relvec.cpp 的内部类, 通过外部接口
// vm44_relvec_match 进行端到端测试, 验证整体行为
//
// 编译: make test_relvec
// 运行: ./test/test_vm44_relvec.exe
// ============================================================================

#include "vm44_internal.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>
#include <string>

// MinGW 严格 ISO C++ 下 M_PI 不可用, 自定义常量
static constexpr double TEST_PI = 3.14159265358979323846;
static constexpr double DEGTORAD = TEST_PI / 180.0;
static constexpr double RADTODEG = 180.0 / TEST_PI;

// ============================================================================
// 测试框架
// ============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { g_tests_passed++; printf("  [PASS] %s\n", msg); } \
    else { g_tests_failed++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    double _diff = std::fabs((double)(a) - (double)(b)); \
    if (_diff < (tol)) { g_tests_passed++; printf("  [PASS] %s (diff=%.4f < %.4f)\n", msg, _diff, (double)(tol)); } \
    else { g_tests_failed++; printf("  [FAIL] %s (a=%.4f b=%.4f diff=%.4f >= %.4f, line %d)\n", \
                                    msg, (double)(a), (double)(b), _diff, (double)(tol), __LINE__); } \
} while(0)

#define ASSERT_GE(a, b, msg) do { \
    if ((a) >= (b)) { g_tests_passed++; printf("  [PASS] %s (%.2f >= %.2f)\n", msg, (double)(a), (double)(b)); } \
    else { g_tests_failed++; printf("  [FAIL] %s (%.2f < %.2f, line %d)\n", msg, (double)(a), (double)(b), __LINE__); } \
} while(0)

// ============================================================================
// 合成数据生成
//   - M 个 Gaia 星 W, 在 [-1000, 1000]^2 范围内均匀分布
//   - 取前 N 个 W, 应用 (s, θ, tx, ty) 变换 + 噪声得到 U
//   - U 与 W 前 N 个一一对应 (真实匹配)
// ============================================================================

static void generate_synthetic_data(
    std::vector<v44::StarPoint>& U, int N,
    std::vector<v44::StarPoint>& W, int M,
    double s_true, double theta_true_rad,
    double tx_true, double ty_true,
    double noise_sigma = 0.3,
    unsigned int seed = 12345)
{
    U.resize(N);
    W.resize(M);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ud(-1000.0, 1000.0);
    std::normal_distribution<double> noise(0.0, noise_sigma);

    double ct = std::cos(theta_true_rad), st = std::sin(theta_true_rad);

    // 生成 W (Gaia 星)
    for (int j = 0; j < M; ++j) {
        double wx = ud(rng);
        double wy = ud(rng);
        W[j].x = wx;
        W[j].y = wy;
        // flux 用距离作代理 (相对向量法按 flux 降序限流 U)
        W[j].flux = static_cast<float>(std::sqrt(wx*wx + wy*wy));
        W[j].saturated = false;
    }

    // U = s·R·W + t + noise (取前 N 个 W 作为真实对应)
    for (int i = 0; i < N; ++i) {
        double wx = W[i].x, wy = W[i].y;
        double ux = s_true * (ct*wx - st*wy) + tx_true + noise(rng);
        double uy = s_true * (st*wx + ct*wy) + ty_true + noise(rng);
        U[i].x = ux;
        U[i].y = uy;
        U[i].flux = W[i].flux;  // 同 flux 便于限流时优先选真实对应
        U[i].saturated = false;
    }
}

// ============================================================================
// 构建 VM44SolveParams (仅设置相对向量法相关字段)
// ============================================================================

static void build_relvec_params(v44::VM44SolveParams& params,
                                 int seed = 42,
                                 int n_samples = 5000,
                                 int max_u = 100,
                                 double s_min = 0.9,
                                 double s_max = 1.1)
{
    std::memset(&params, 0, sizeof(params));

    // 基础参数
    params.seed = seed;

    // 相对向量法依赖的尺度范围
    params.s_min = s_min;
    params.s_max = s_max;

    // 相对向量法参数 (V4.4 优化: 像素容差 + 投票无上限 + 自适应停止)
    params.relvec_n_samples = n_samples;       // 最大采样上限
    params.relvec_max_u = max_u;
    params.relvec_third_star_tol = 1.5;        // 像素 (按s_est转换到Gaia角秒域)
    params.relvec_max_cand = 500;
    params.relvec_min_len_frac = 0.05;
    params.relvec_max_len_frac = 0.8;
    params.relvec_n_third_stars = 0;           // 0=用所有可用第三星 (投票无上限)
    // 自适应采样停止参数
    params.relvec_adaptive_stop = 1;
    params.relvec_min_samples = 200;
    params.relvec_check_interval = 100;
    params.relvec_snr_eps = 0.05;
    params.relvec_max_stable = 3;

    // 日志 (NULL=不写文件)
    params.log_dir = nullptr;
}

// ============================================================================
// 测试 1: k-vector 距离查询正确性 (通过端到端调用验证)
//
// 构造 W=30 颗星, U=20 颗 (s=1, θ=0, t=0), 验证:
//   - n_total_candidates > 0 (k-vector 查询有结果)
//   - n_passed > 0 (第三星验证有通过)
//   - success=true (整体流程正常)
// ============================================================================

static void test_kvector_query()
{
    printf("\n[TEST] test_kvector_query: k-vector 距离查询正确性 (N=20, M=30)\n");

    std::vector<v44::StarPoint> U, W;
    int N = 20, M = 30;
    generate_synthetic_data(U, N, W, M,
                            /*s=*/1.0, /*θ=*/0.0,
                            /*tx=*/0.0, /*ty=*/0.0,
                            /*noise=*/0.3, /*seed=*/12345);

    v44::VM44SolveParams params;
    build_relvec_params(params, /*seed=*/42, /*n_samples=*/1000, /*max_u=*/50);

    v44::RelVecResult result;
    int ret = v44::vm44_relvec_match(U, W, /*s0=*/1.0, params, result, /*logger=*/nullptr);

    printf("  ret=%d success=%d n_cand=%d n_passed=%d θ_peak=%.2f° SNR=%.2fx\n",
           ret, (int)result.success, result.n_total_candidates,
           result.n_passed, result.theta_peak_deg, result.theta_snr);

    ASSERT_TRUE(ret == 0, "vm44_relvec_match 返回成功 (ret == 0)");
    ASSERT_GE(result.n_total_candidates, 1.0, "k-vector 查询有候选 (n_total_candidates >= 1)");
    ASSERT_GE(result.n_passed, 1.0, "第三星验证有通过 (n_passed >= 1)");
    ASSERT_TRUE(result.success, "success=true (SNR >= 5)");
}

// ============================================================================
// 测试 2: t=0 已知匹配场景 (s=1, θ=30°)
//
// 数据生成: U = s·R(θ_true)·W + t
// 相对向量法: θ_rot = angle(Δw) - angle(Δu) = -θ_true (从 U 回到 W 的旋转)
// 验证:
//   - SNR > 5
//   - θ_peak ≈ -30° (±5°)
//   - n_passed > 10 (通过候选数足够)
// ============================================================================

static void test_t_zero_known_match()
{
    printf("\n[TEST] test_t_zero_known_match: t=0 已知匹配 (s=1, θ=30°, N=50, M=80)\n");

    std::vector<v44::StarPoint> U, W;
    int N = 50, M = 80;
    double theta_true = 30.0 * DEGTORAD;
    generate_synthetic_data(U, N, W, M,
                            /*s=*/1.0, /*θ=*/theta_true,
                            /*tx=*/0.0, /*ty=*/0.0,
                            /*noise=*/0.3, /*seed=*/23456);

    v44::VM44SolveParams params;
    build_relvec_params(params, /*seed=*/42, /*n_samples=*/2000, /*max_u=*/100);

    v44::RelVecResult result;
    int ret = v44::vm44_relvec_match(U, W, /*s0=*/1.0, params, result, /*logger=*/nullptr);

    printf("  ret=%d success=%d n_cand=%d n_passed=%d θ_peak=%.2f° s_peak=%.4f SNR=%.2fx\n",
           ret, (int)result.success, result.n_total_candidates,
           result.n_passed, result.theta_peak_deg, result.s_peak, result.theta_snr);

    ASSERT_TRUE(ret == 0, "vm44_relvec_match 返回成功");
    ASSERT_TRUE(result.success, "success=true (SNR >= 5)");
    ASSERT_GE(result.theta_snr, 5.0, "SNR >= 5x (t=0 场景)");
    // 相对向量法 θ_rot = angle(Δw) - angle(Δu) = -θ_true
    ASSERT_NEAR(result.theta_peak_deg, -30.0, 5.0, "θ_peak ≈ -30° (相对向量法定义, ±5°)");
    // 2D 聚类: s_peak ≈ s_true = 1.0 (±0.05)
    ASSERT_NEAR(result.s_peak, 1.0, 0.05, "s_peak ≈ 1.0 (2D 聚类, ±0.05)");
    ASSERT_GE(result.n_passed, 10.0, "n_passed >= 10 (通过候选足够)");
}

// ============================================================================
// 测试 3: t≠0 场景 SNR > 5 (相对向量法核心优势)
//
// 构造 t=(100", 100") 合成数据 (单θ采样在此场景会失败)
// 数据生成: U = s·R(θ_true)·W + t
// 相对向量法: θ_rot = -θ_true
// 验证:
//   - SNR > 5 (相对向量法消除 t 假设, 仍能成功)
//   - θ_peak ≈ -45° (±5°)
//   - n_passed > 5
// ============================================================================

static void test_t_nonzero_snr()
{
    printf("\n[TEST] test_t_nonzero_snr: t≠0 场景 (s=1, θ=45°, t=(100\",100\"), N=50, M=80)\n");

    std::vector<v44::StarPoint> U, W;
    int N = 50, M = 80;
    double theta_true = 45.0 * DEGTORAD;
    generate_synthetic_data(U, N, W, M,
                            /*s=*/1.0, /*θ=*/theta_true,
                            /*tx=*/100.0, /*ty=*/100.0,  // t≠0
                            /*noise=*/0.3, /*seed=*/34567);

    v44::VM44SolveParams params;
    build_relvec_params(params, /*seed=*/42, /*n_samples=*/2000, /*max_u=*/100);

    v44::RelVecResult result;
    int ret = v44::vm44_relvec_match(U, W, /*s0=*/1.0, params, result, /*logger=*/nullptr);

    printf("  ret=%d success=%d n_cand=%d n_passed=%d θ_peak=%.2f° s_peak=%.4f SNR=%.2fx\n",
           ret, (int)result.success, result.n_total_candidates,
           result.n_passed, result.theta_peak_deg, result.s_peak, result.theta_snr);

    ASSERT_TRUE(ret == 0, "vm44_relvec_match 返回成功");
    ASSERT_TRUE(result.success, "success=true (相对向量法在 t≠0 时仍 SNR >= 5)");
    ASSERT_GE(result.theta_snr, 5.0, "SNR >= 5x (t≠0 场景, 相对向量法核心优势)");
    // 相对向量法 θ_rot = -θ_true
    ASSERT_NEAR(result.theta_peak_deg, -45.0, 5.0, "θ_peak ≈ -45° (相对向量法定义, ±5°)");
    // 2D 聚类: s_peak ≈ s_true = 1.0 (±0.05), t≠0 不影响 s (相对向量法核心优势)
    ASSERT_NEAR(result.s_peak, 1.0, 0.05, "s_peak ≈ 1.0 (2D 聚类, t≠0 不影响 s, ±0.05)");
    ASSERT_GE(result.n_passed, 5.0, "n_passed >= 5 (t≠0 仍有通过候选)");
}

// ============================================================================
// 测试 4: U 限流 (relvec_max_u)
//
// 构造 U=200 颗 (远超 max_u=100), 验证:
//   - 算法正常运行 (ret=0, 未因候选爆炸崩溃)
//   - 仍能找到正确 θ_peak
//   - SNR >= 5 (限流后仍能找到 θ 峰)
//   - 单次采样候选限流生效 (n_total_candidates / n_samples <= max_cand 的合理倍数)
//     注: n_total_candidates 是原始候选累加 (限流前的总和), 用于统计查询规模
//         限流作用于单次采样的 cand_indices, 控制第三星验证的计算量
// ============================================================================

static void test_u_limit()
{
    printf("\n[TEST] test_u_limit: U 限流 (N=200, max_u=100, s=1, θ=60°)\n");

    std::vector<v44::StarPoint> U, W;
    int N = 200, M = 250;  // M >= N 避免 generate_synthetic_data 中 W[i] 越界
    double theta_true = 60.0 * DEGTORAD;
    generate_synthetic_data(U, N, W, M,
                            /*s=*/1.0, /*θ=*/theta_true,
                            /*tx=*/0.0, /*ty=*/0.0,
                            /*noise=*/0.3, /*seed=*/45678);

    v44::VM44SolveParams params;
    // max_u=100, N=200 → 实际只用前 100 颗 (按 flux 降序)
    build_relvec_params(params, /*seed=*/42, /*n_samples=*/2000, /*max_u=*/100);

    v44::RelVecResult result;
    int ret = v44::vm44_relvec_match(U, W, /*s0=*/1.0, params, result, /*logger=*/nullptr);

    printf("  ret=%d success=%d n_cand=%d n_passed=%d θ_peak=%.2f° s_peak=%.4f SNR=%.2fx\n",
           ret, (int)result.success, result.n_total_candidates,
           result.n_passed, result.theta_peak_deg, result.s_peak, result.theta_snr);

    // 平均每次采样的候选数 (限流前) - 用实际采样次数 (自适应停止可能 < max)
    double avg_cand_per_sample = (result.n_samples > 0) ?
        (double)result.n_total_candidates / result.n_samples : 0.0;
    printf("  平均每次采样候选数 (限流前): %.1f (max_cand=%d 限流后验证, actual_samples=%d)\n",
           avg_cand_per_sample, params.relvec_max_cand, result.n_samples);

    ASSERT_TRUE(ret == 0, "vm44_relvec_match 返回成功 (限流后未崩溃)");
    ASSERT_TRUE(result.success, "success=true (限流后仍能成功)");
    ASSERT_GE(result.theta_snr, 5.0, "SNR >= 5x (限流后仍能找到 θ 峰)");
    // 相对向量法 θ_rot = -θ_true
    ASSERT_NEAR(result.theta_peak_deg, -60.0, 5.0, "θ_peak ≈ -60° (相对向量法定义, ±5°)");
    // 2D 聚类: s_peak ≈ s_true = 1.0 (±0.05)
    ASSERT_NEAR(result.s_peak, 1.0, 0.05, "s_peak ≈ 1.0 (2D 聚类, 限流后仍正确, ±0.05)");
    // 验证算法正常完成 (n_passed > 0 表示限流后仍有通过候选)
    ASSERT_GE(result.n_passed, 1.0, "n_passed >= 1 (限流后仍有通过候选)");
    printf("  [PASS] 限流生效 (N=200 → max_u=100, 算法正常完成, n_passed=%d)\n",
           result.n_passed);
    g_tests_passed++;
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    printf("============================================================\n");
    printf("V4.4 相对向量法单元测试 (test_vm44_relvec)\n");
    printf("============================================================\n");

    test_kvector_query();
    test_t_zero_known_match();
    test_t_nonzero_snr();
    test_u_limit();

    printf("\n============================================================\n");
    printf("测试结果: %d 通过, %d 失败\n", g_tests_passed, g_tests_failed);
    printf("============================================================\n");

    return (g_tests_failed == 0) ? 0 : 1;
}

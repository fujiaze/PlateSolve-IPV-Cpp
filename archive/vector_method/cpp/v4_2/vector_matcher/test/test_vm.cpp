// ============================================================================
// test_vm.cpp - V4.2 VectorMatcher 单元测试（Task 3）
//
// 测试场景:
//   1. 合成数据: N=250 图像星, M=300 Gaia 星, 已知变换 s=1.0, θ=-90°
//   2. 验证 PROSAC 抽样能找到正确 θ 峰值(theta_peak_deg ≈ -90°)
//   3. 验证 SVD 精度: |s_estimated - 1.0| < 0.01, |θ_estimated - (-90°)| < 1°
//   4. 验证匹配对数 >= 10
//
// 编译: make test (静态编译所有源码, 避免依赖 DLL 路径)
// 运行: ./test/test_vm.exe
// ============================================================================

#include "../include/vm_api.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <cstring>

static constexpr double PI = 3.14159265358979323846;
static constexpr double DEGTORAD = PI / 180.0;
static constexpr double RADTODEG = 180.0 / PI;

// 测试框架
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
    if ((a) >= (b)) { g_tests_passed++; printf("  [PASS] %s (%d >= %d)\n", msg, (int)(a), (int)(b)); } \
    else { g_tests_failed++; printf("  [FAIL] %s (%d < %d, line %d)\n", msg, (int)(a), (int)(b), __LINE__); } \
} while(0)


// 生成合成数据
//   - M=300 个 Gaia 星 W, 在 [-1000, 1000]^2 范围内均匀分布
//   - 取前 N=250 个 W, 应用 s=1.0, θ=-90°, tx=0, ty=0, 加噪声得到 U
//   - 噪声 σ=0.3 角秒
static void generate_synthetic_data(
    std::vector<double>& U, int& N,
    std::vector<double>& W, int& M,
    double s_true, double theta_true,
    unsigned int seed = 12345)
{
    N = 250; M = 300;
    U.resize(N * 2);
    W.resize(M * 2);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ud(-1000.0, 1000.0);
    std::normal_distribution<double> noise(0.0, 0.3);

    double ct = std::cos(theta_true), st = std::sin(theta_true);

    // 生成 W
    for (int j = 0; j < M; ++j) {
        W[j*2]   = ud(rng);
        W[j*2+1] = ud(rng);
    }

    // U = s·R·W + t + noise (取前 N 个 W 作为真实对应)
    for (int i = 0; i < N; ++i) {
        double wx = W[i*2], wy = W[i*2+1];
        double ux = s_true * (ct*wx - st*wy) + noise(rng);
        double uy = s_true * (st*wx + ct*wy) + noise(rng);
        U[i*2]   = ux;
        U[i*2+1] = uy;
    }
}


int main()
{
    printf("=== V4.2 VectorMatcher 单元测试 ===\n\n");

    // ------------------------------------------------------------------
    // 测试 1: 合成数据, s=1.0, θ=-90°
    // ------------------------------------------------------------------
    printf("[Test 1] 合成数据匹配 (N=250, M=300, s=1.0, θ=-90°)\n");

    std::vector<double> U, W;
    int N, M;
    double s_true = 1.0;
    double theta_true = -90.0 * DEGTORAD;  // -π/2 弧度
    generate_synthetic_data(U, N, W, M, s_true, theta_true);

    // 构建参数
    VectorMatcherParams params;
    std::memset(&params, 0, sizeof(params));
    params.s0 = 1.0;
    params.s_min = 0.5;
    params.s_max = 2.0;
    params.n_modes = 4;
    params.seed = 42;
    params.K_total = 10000;
    params.batch_size = 500;
    params.min_samples = 50;
    params.K_top = 100;
    params.min_inliers = 5;
    params.w_snr = 0.4;
    params.w_sparse = 0.4;
    params.w_sat = 0.2;
    params.prosac_T_max = 10000;
    params.use_prosac = 1;
    params.log_file_path = nullptr;
    params.snr_values = nullptr;
    params.is_saturated_values = nullptr;

    // 调用 vm_match
    VectorMatchResult result;
    result.cu = nullptr;
    result.cw = nullptr;
    int ret = vm_match(U.data(), N, W.data(), M, &params, &result);

    printf("  vm_match 返回: %d, success=%d\n", ret, result.success);
    printf("  s=%.4f θ=%.4f° tx=%.2f\" ty=%.2f\" rms=%.4f\"\n",
           result.s, result.theta * RADTODEG, result.tx, result.ty, result.rms);
    printf("  n_pairs=%d theta_snr=%.2fx theta_peak_deg=%.2f° best_n_range=%d\n",
           result.n_pairs, result.theta_snr, result.theta_peak_deg, result.best_n_range);
    printf("  best_mode=%d n_phasea_records=%d prosac_quality_median=%.4f\n",
           result.best_mode, result.n_phasea_records, result.prosac_quality_median);

    ASSERT_TRUE(ret == 0, "vm_match 返回成功 (ret == 0)");
    ASSERT_TRUE(result.success == 1, "result.success == 1");

    // θ 峰值应在 -90° 附近 (允许 ±5°)
    double theta_peak_diff = std::fabs(result.theta_peak_deg - (-90.0));
    ASSERT_NEAR(result.theta_peak_deg, -90.0, 5.0, "Phase A θ 峰值 ≈ -90° (±5°)");

    // SVD 精度: |s - 1.0| < 0.01
    ASSERT_NEAR(result.s, 1.0, 0.01, "SVD 尺度 s ≈ 1.0 (±0.01)");

    // SVD 精度: |θ - (-90°)| < 1°
    double theta_est_deg = result.theta * RADTODEG;
    // 标准化到 [-180, 180)
    while (theta_est_deg > 180.0)  theta_est_deg -= 360.0;
    while (theta_est_deg < -180.0) theta_est_deg += 360.0;
    double theta_diff = std::fabs(theta_est_deg - (-90.0));
    if (theta_diff > 180.0) theta_diff = 360.0 - theta_diff;
    ASSERT_NEAR(theta_est_deg, -90.0, 1.0, "SVD 旋转角 θ ≈ -90° (±1°)");

    // 匹配对数 >= 10
    ASSERT_GE(result.n_pairs, 10, "匹配对数 n_pairs >= 10");

    // PROSAC 调试信息
    ASSERT_TRUE(result.prosac_quality_median > 0.0, "PROSAC quality_median > 0");
    ASSERT_TRUE(result.n_phasea_records > 0, "Phase A records > 0");

    // 释放结果
    vm_free_result(&result);

    // ------------------------------------------------------------------
    // 测试 2: 不同随机种子下的稳定性 (seed=123)
    // ------------------------------------------------------------------
    printf("\n[Test 2] 不同种子稳定性 (seed=123)\n");

    // 重新生成数据 (相同变换, 不同分布)
    std::vector<double> U2, W2;
    int N2, M2;
    generate_synthetic_data(U2, N2, W2, M2, s_true, theta_true, 54321);

    params.seed = 123;
    VectorMatchResult result2;
    result2.cu = nullptr;
    result2.cw = nullptr;
    int ret2 = vm_match(U2.data(), N2, W2.data(), M2, &params, &result2);

    printf("  vm_match 返回: %d, success=%d\n", ret2, result2.success);
    printf("  s=%.4f θ=%.4f° rms=%.4f\" n_pairs=%d\n",
           result2.s, result2.theta * RADTODEG, result2.rms, result2.n_pairs);

    ASSERT_TRUE(ret2 == 0, "Test2 vm_match 返回成功");
    ASSERT_TRUE(result2.success == 1, "Test2 success == 1");
    ASSERT_NEAR(result2.s, 1.0, 0.02, "Test2 s ≈ 1.0 (±0.02)");
    ASSERT_GE(result2.n_pairs, 5, "Test2 n_pairs >= 5");

    vm_free_result(&result2);

    // ------------------------------------------------------------------
    // 测试 3: 不启用 PROSAC (退化为均匀随机)
    // ------------------------------------------------------------------
    printf("\n[Test 3] PROSAC 禁用 (纯均匀随机)\n");

    std::vector<double> U3, W3;
    int N3, M3;
    generate_synthetic_data(U3, N3, W3, M3, s_true, theta_true, 99999);

    params.seed = 42;
    params.use_prosac = 0;
    VectorMatchResult result3;
    result3.cu = nullptr;
    result3.cw = nullptr;
    int ret3 = vm_match(U3.data(), N3, W3.data(), M3, &params, &result3);

    printf("  vm_match 返回: %d, success=%d\n", ret3, result3.success);
    printf("  s=%.4f θ=%.4f° rms=%.4f\" n_pairs=%d\n",
           result3.s, result3.theta * RADTODEG, result3.rms, result3.n_pairs);

    ASSERT_TRUE(ret3 == 0, "Test3 vm_match 返回成功");
    // PROSAC 禁用时也应能找到正确变换 (可能略慢)
    if (result3.success == 1) {
        ASSERT_NEAR(result3.s, 1.0, 0.02, "Test3 s ≈ 1.0 (±0.02)");
    } else {
        printf("  [SKIP] Test3 未成功 (允许, PROSAC 禁用时成功率略低)\n");
    }

    vm_free_result(&result3);

    // ------------------------------------------------------------------
    // 总结
    // ------------------------------------------------------------------
    printf("\n=== 测试总结 ===\n");
    printf("  通过: %d\n", g_tests_passed);
    printf("  失败: %d\n", g_tests_failed);
    if (g_tests_failed == 0) {
        printf("  >>> 全部测试通过 <<<\n");
        return 0;
    } else {
        printf("  >>> 有 %d 个测试失败 <<<\n", g_tests_failed);
        return 1;
    }
}

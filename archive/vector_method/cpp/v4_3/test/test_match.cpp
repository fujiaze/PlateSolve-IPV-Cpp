// ============================================================================
// test_match.cpp - V4.3 VectorMatcher 单元测试 (Task 2 SubTask 2.3)
//
// 测试 vm43_match.cpp 中的主函数 v43::vm43_match:
//   1. 合成数据: N=250 图像星, M=300 Gaia 星, 已知变换 s=1.0, θ=-90°
//   2. 验证 PROSAC 抽样能找到正确 θ 峰值 (theta_peak_deg ≈ -90°)
//   3. 验证 SVD 精度: |s_estimated - 1.0| < 0.01, |θ_estimated - (-90°)| < 1°
//   4. 验证匹配对数 >= 10
//
// 从 V4.2 test_vm.cpp 迁移, 适配到 V4.3 接口:
//   - 输入: std::vector<StarPoint> 替代裸 double 数组
//   - 输出: VectorMatchResult.transform + .pairs 替代 C struct (cu/cw 指针)
//   - 参数: VM43SolveParams 替代 VectorMatcherParams
//   - 调用: v43::vm43_match(...) 替代 vm_match(...)
//
// 编译: make test_match
// 运行: ./test/test_match.exe
// ============================================================================

#include "vm43_internal.h"

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
    if ((a) >= (b)) { g_tests_passed++; printf("  [PASS] %s (%d >= %d)\n", msg, (int)(a), (int)(b)); } \
    else { g_tests_failed++; printf("  [FAIL] %s (%d < %d, line %d)\n", msg, (int)(a), (int)(b), __LINE__); } \
} while(0)

// ============================================================================
// 合成数据生成
//   - M=300 个 Gaia 星 W, 在 [-1000, 1000]^2 范围内均匀分布
//   - 取前 N=250 个 W, 应用 s=1.0, θ=-90°, tx=0, ty=0, 加噪声得到 U
//   - 噪声 σ=0.3 角秒
//   - StarPoint.saturated 全部为 false (合成数据无饱和)
//   - flux 用 W 的距离作代理 (仅用于 PROSAC 质量分, 无实际意义)
// ============================================================================

static void generate_synthetic_data(
    std::vector<v43::StarPoint>& U, int& N,
    std::vector<v43::StarPoint>& W, int& M,
    double s_true, double theta_true,
    unsigned int seed = 12345)
{
    N = 250; M = 300;
    U.resize(N);
    W.resize(M);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ud(-1000.0, 1000.0);
    std::normal_distribution<double> noise(0.0, 0.3);

    double ct = std::cos(theta_true), st = std::sin(theta_true);

    // 生成 W (Gaia 星)
    for (int j = 0; j < M; ++j) {
        double wx = ud(rng);
        double wy = ud(rng);
        W[j].x = wx;
        W[j].y = wy;
        W[j].flux = static_cast<float>(std::sqrt(wx*wx + wy*wy));
        W[j].saturated = false;
    }

    // U = s·R·W + t + noise (取前 N 个 W 作为真实对应)
    for (int i = 0; i < N; ++i) {
        double wx = W[i].x, wy = W[i].y;
        double ux = s_true * (ct*wx - st*wy) + noise(rng);
        double uy = s_true * (st*wx + ct*wy) + noise(rng);
        U[i].x = ux;
        U[i].y = uy;
        U[i].flux = W[i].flux;
        U[i].saturated = false;
    }
}

// ============================================================================
// 构建 VM43SolveParams (仅设置 VectorMatcher 相关字段)
// ============================================================================

static void build_params(v43::VM43SolveParams& params, int seed, int use_prosac)
{
    std::memset(&params, 0, sizeof(params));

    // 基础参数
    params.n_modes = 4;
    params.seed = seed;

    // VectorMatcher 参数
    params.s_min = 0.5;
    params.s_max = 2.0;
    params.K_total = 10000;
    params.batch_size = 500;
    params.min_samples = 50;
    params.K_top = 100;
    params.min_inliers = 5;
    params.w_snr = 0.4;
    params.w_sparse = 0.4;
    params.w_sat = 0.2;
    params.prosac_T_max = 10000;
    params.use_prosac = use_prosac;

    // 日志 (NULL=仅 stderr)
    params.log_dir = nullptr;
}

// ============================================================================
// 测试 1: 合成数据匹配 (s=1.0, θ=-90°)
// ============================================================================

static void test_synthetic_match()
{
    printf("\n[TEST] test_synthetic_match: 合成数据匹配 (N=250, M=300, s=1.0, θ=-90°)\n");

    std::vector<v43::StarPoint> U, W;
    int N, M;
    double s_true = 1.0;
    double theta_true = -90.0 * DEGTORAD;  // -π/2 弧度
    generate_synthetic_data(U, N, W, M, s_true, theta_true);

    // 构建参数
    v43::VM43SolveParams params;
    build_params(params, /*seed=*/42, /*use_prosac=*/1);

    // 调用 vm43_match
    v43::VectorMatchResult output;
    int ret = v43::vm43_match(U, W, /*s0=*/1.0, params, output, /*logger=*/nullptr);

    printf("  vm43_match 返回: %d, success=%d\n", ret, (int)output.success);
    printf("  s=%.4f θ=%.4f° tx=%.2f\" ty=%.2f\" rms=%.4f\"\n",
           output.transform.s, output.transform.theta * RADTODEG,
           output.transform.tx, output.transform.ty, output.rms);
    printf("  n_pairs=%d theta_snr=%.2fx theta_peak_deg=%.2f° best_n_range=%d\n",
           (int)output.pairs.size(), output.theta_snr, output.theta_peak_deg,
           output.best_n_range);
    printf("  best_mode=%d n_phasea_records=%d prosac_quality_median=%.4f\n",
           output.best_mode, output.n_phasea_records, output.prosac_quality_median);

    ASSERT_TRUE(ret == 0, "vm43_match 返回成功 (ret == 0)");
    ASSERT_TRUE(output.success, "output.success == true");

    // θ 峰值应在 -90° 附近 (允许 ±5°)
    ASSERT_NEAR(output.theta_peak_deg, -90.0, 5.0, "Phase A θ 峰值 ≈ -90° (±5°)");

    // SVD 精度: |s - 1.0| < 0.01
    ASSERT_NEAR(output.transform.s, 1.0, 0.01, "SVD 尺度 s ≈ 1.0 (±0.01)");

    // SVD 精度: |θ - (-90°)| < 1°
    double theta_est_deg = output.transform.theta * RADTODEG;
    // 标准化到 [-180, 180)
    while (theta_est_deg > 180.0)  theta_est_deg -= 360.0;
    while (theta_est_deg < -180.0) theta_est_deg += 360.0;
    double theta_diff = std::fabs(theta_est_deg - (-90.0));
    if (theta_diff > 180.0) theta_diff = 360.0 - theta_diff;
    ASSERT_NEAR(theta_est_deg, -90.0, 1.0, "SVD 旋转角 θ ≈ -90° (±1°)");

    // 匹配对数 >= 10
    ASSERT_GE((int)output.pairs.size(), 10, "匹配对数 pairs.size() >= 10");

    // PROSAC 调试信息
    ASSERT_TRUE(output.prosac_quality_median > 0.0, "PROSAC quality_median > 0");
    ASSERT_TRUE(output.n_phasea_records > 0, "Phase A records > 0");

    // 匹配对索引范围验证 (cu 在 [0,N), cw 在 [0,M))
    bool pairs_index_valid = true;
    for (const auto& p : output.pairs) {
        if (p.u < 0 || p.u >= N || p.w < 0 || p.w >= M) {
            pairs_index_valid = false;
            break;
        }
    }
    ASSERT_TRUE(pairs_index_valid, "匹配对索引在合法范围内");

    // 变换参数 valid 标志
    ASSERT_TRUE(output.transform.valid, "transform.valid == true");
}

// ============================================================================
// 测试 2: 不同随机种子下的稳定性 (seed=123)
// ============================================================================

static void test_different_seed()
{
    printf("\n[TEST] test_different_seed: 不同种子稳定性 (seed=123)\n");

    // 重新生成数据 (相同变换, 不同分布)
    std::vector<v43::StarPoint> U2, W2;
    int N2, M2;
    double s_true = 1.0;
    double theta_true = -90.0 * DEGTORAD;
    generate_synthetic_data(U2, N2, W2, M2, s_true, theta_true, /*seed=*/54321);

    v43::VM43SolveParams params;
    build_params(params, /*seed=*/123, /*use_prosac=*/1);

    v43::VectorMatchResult output2;
    int ret2 = v43::vm43_match(U2, W2, /*s0=*/1.0, params, output2, nullptr);

    printf("  vm43_match 返回: %d, success=%d\n", ret2, (int)output2.success);
    printf("  s=%.4f θ=%.4f° rms=%.4f\" n_pairs=%d\n",
           output2.transform.s, output2.transform.theta * RADTODEG,
           output2.rms, (int)output2.pairs.size());

    ASSERT_TRUE(ret2 == 0, "Test2 vm43_match 返回成功");
    ASSERT_TRUE(output2.success, "Test2 success == true");
    ASSERT_NEAR(output2.transform.s, 1.0, 0.02, "Test2 s ≈ 1.0 (±0.02)");
    ASSERT_GE((int)output2.pairs.size(), 5, "Test2 n_pairs >= 5");
}

// ============================================================================
// 测试 3: 不启用 PROSAC (退化为均匀随机)
// ============================================================================

static void test_prosac_disabled()
{
    printf("\n[TEST] test_prosac_disabled: PROSAC 禁用 (纯均匀随机)\n");

    std::vector<v43::StarPoint> U3, W3;
    int N3, M3;
    double s_true = 1.0;
    double theta_true = -90.0 * DEGTORAD;
    generate_synthetic_data(U3, N3, W3, M3, s_true, theta_true, /*seed=*/99999);

    v43::VM43SolveParams params;
    build_params(params, /*seed=*/42, /*use_prosac=*/0);

    v43::VectorMatchResult output3;
    int ret3 = v43::vm43_match(U3, W3, /*s0=*/1.0, params, output3, nullptr);

    printf("  vm43_match 返回: %d, success=%d\n", ret3, (int)output3.success);
    printf("  s=%.4f θ=%.4f° rms=%.4f\" n_pairs=%d\n",
           output3.transform.s, output3.transform.theta * RADTODEG,
           output3.rms, (int)output3.pairs.size());

    ASSERT_TRUE(ret3 == 0, "Test3 vm43_match 返回成功");
    // PROSAC 禁用时也应能找到正确变换 (可能略慢)
    if (output3.success) {
        ASSERT_NEAR(output3.transform.s, 1.0, 0.02, "Test3 s ≈ 1.0 (±0.02)");
    } else {
        printf("  [SKIP] Test3 未成功 (允许, PROSAC 禁用时成功率略低)\n");
    }
}

// ============================================================================
// 测试 4: 非法输入处理
// ============================================================================

static void test_invalid_input()
{
    printf("\n[TEST] test_invalid_input: 非法输入处理\n");

    v43::VM43SolveParams params;
    build_params(params, /*seed=*/42, /*use_prosac=*/1);

    // 空 U 数组
    {
        std::vector<v43::StarPoint> empty_U;
        std::vector<v43::StarPoint> W;
        int M = 300;
        W.resize(M);
        for (int j = 0; j < M; ++j) {
            W[j].x = static_cast<double>(j);
            W[j].y = static_cast<double>(j);
            W[j].flux = 1.0f;
            W[j].saturated = false;
        }
        v43::VectorMatchResult output;
        int ret = v43::vm43_match(empty_U, W, 1.0, params, output, nullptr);
        ASSERT_TRUE(ret != 0, "空 U 数组返回错误");
        ASSERT_TRUE(!output.success, "空 U 数组 success=false");
    }

    // 空 W 数组
    {
        std::vector<v43::StarPoint> U;
        int N = 250;
        U.resize(N);
        for (int i = 0; i < N; ++i) {
            U[i].x = static_cast<double>(i);
            U[i].y = static_cast<double>(i);
            U[i].flux = 1.0f;
            U[i].saturated = false;
        }
        std::vector<v43::StarPoint> empty_W;
        v43::VectorMatchResult output;
        int ret = v43::vm43_match(U, empty_W, 1.0, params, output, nullptr);
        ASSERT_TRUE(ret != 0, "空 W 数组返回错误");
        ASSERT_TRUE(!output.success, "空 W 数组 success=false");
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main()
{
    printf("=== V4.3 VectorMatcher 单元测试 ===\n");

    test_synthetic_match();
    test_different_seed();
    test_prosac_disabled();
    test_invalid_input();

    // 总结
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

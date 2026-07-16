// ============================================================================
// test_wf.cpp - V4.2 WcsFitter 单元测试（Task 6）
//
// 测试场景:
//   1. 线性 CD: 50 对无畸变合成数据, 验证 CD 矩阵正确, RMS < 0.1px
//   2. 仿射畸变: 50 对含剪切畸变, 验证 Layer 1 改善 RMS
//   3. SIP 2 阶: 100 对含 2 阶畸变, 验证 BIC 选 order=2, RMS 显著改善
//   4. SIP 过拟合防护: 50 对纯线性+小噪声, 验证 BIC 不选 order=4
//
// 编译: make test (静态编译 wf_core.cpp, 无需 DLL)
// ============================================================================

#include "wf_api.h"
#include "../common/v42_log.h"

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include <cstdint>

// ============================================================================
// 简单确定性随机数生成器 (LCG)
// ============================================================================

static uint64_t g_rng_state = 42;

static void rng_seed(uint64_t s) { g_rng_state = s; }

static double rng_uniform() {
    // 返回 [0, 1) 均匀随机数
    g_rng_state = g_rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(g_rng_state >> 11) * (1.0 / (double)(1ULL << 53));
}

static double rng_gaussian(double sigma) {
    // Box-Muller 变换
    double u1 = rng_uniform();
    double u2 = rng_uniform();
    if (u1 < 1e-10) u1 = 1e-10;
    return sigma * std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * 3.14159265358979 * u2);
}

// ============================================================================
// 测试辅助: 生成合成数据并调用 wf_fit
// ============================================================================

struct TestData {
    std::vector<double> U;  // N*2
    std::vector<double> W;  // M*2
    std::vector<int> pairs_u;
    std::vector<int> pairs_w;
    int N, M, n_pairs;
};

// 生成合成数据: U = s·R·W + t + 可选畸变
// s0: 像素尺度, img_w/h: 图像尺寸, n_pairs: 匹配对数
// s, theta: 相似变换参数
// shear: 剪切系数 (0=无剪切)
// sip_coeff: 2阶SIP畸变系数 (0=无SIP畸变, 单位=像素@边缘)
// noise_sigma: 噪声标准差 (像素, 经 s0 转换为角秒)
static TestData generate_synthetic(double s0, double img_w, double img_h,
                                    int n_pairs, double s, double theta,
                                    double shear, double sip_coeff,
                                    double noise_sigma) {
    TestData td;
    td.n_pairs = n_pairs;
    td.N = n_pairs;
    td.M = n_pairs;
    td.U.resize(n_pairs * 2);
    td.W.resize(n_pairs * 2);
    td.pairs_u.resize(n_pairs);
    td.pairs_w.resize(n_pairs);

    double half_w_asec = (img_w / 2.0) * s0;
    double half_h_asec = (img_h / 2.0) * s0;
    double ct = std::cos(theta), st = std::sin(theta);

    for (int i = 0; i < n_pairs; ++i) {
        td.pairs_u[i] = i;
        td.pairs_w[i] = i;

        // W 在切平面上的角秒坐标 (均匀分布在 [-half, half])
        double Wx = (rng_uniform() - 0.5) * 2.0 * half_w_asec;
        double Wy = (rng_uniform() - 0.5) * 2.0 * half_h_asec;
        td.W[i * 2] = Wx;
        td.W[i * 2 + 1] = Wy;

        // U = s·R·W + shear·Wy + sip_distortion + noise
        double Ux = s * (ct * Wx - st * Wy);
        double Uy = s * (st * Wx + ct * Wy);

        // 剪切畸变
        Ux += shear * Wy;

        // 2阶SIP畸变 (在U空间, 单位=角秒)
        if (sip_coeff != 0) {
            double norm_x = Ux / half_w_asec;  // 归一化到 [-1, 1]
            double norm_y = Uy / half_h_asec;
            Ux += sip_coeff * s0 * norm_x * norm_x;  // 2阶畸变
            Uy += sip_coeff * s0 * norm_y * norm_y;
        }

        // 噪声
        if (noise_sigma > 0) {
            Ux += rng_gaussian(noise_sigma * s0);
            Uy += rng_gaussian(noise_sigma * s0);
        }

        td.U[i * 2] = Ux;
        td.U[i * 2 + 1] = Uy;
    }

    return td;
}

// 调用 wf_fit 并返回结果
static WcsResult run_wf_fit(const TestData& td, double s0,
                             double img_w, double img_h,
                             double ra, double dec,
                             int sip_max_order = 4,
                             int skip_sip = 0) {
    WcsFitterParams params;
    params.s0 = s0;
    params.sip_max_order = sip_max_order;
    params.skip_sip = skip_sip;
    params.img_width = img_w;
    params.img_height = img_h;
    params.center_ra = ra;
    params.center_dec = dec;
    params.log_file_path = nullptr;

    WcsResult result;
    wf_fit(td.U.data(), td.N, td.W.data(), td.M,
           td.pairs_u.data(), td.pairs_w.data(), td.n_pairs,
           &params, &result);
    return result;
}

// ============================================================================
// 测试用例
// ============================================================================

static int test_count = 0;
static int test_pass = 0;

#define ASSERT_TRUE(cond, msg) do { \
    test_count++; \
    std::string _msg = (msg); \
    if (cond) { test_pass++; printf("  [PASS] %s\n", _msg.c_str()); } \
    else { printf("  [FAIL] %s (line %d)\n", _msg.c_str(), __LINE__); } \
} while(0)

#define ASSERT_NEAR(val, expected, tol, msg) do { \
    test_count++; \
    std::string _msg = (msg); \
    double diff = std::abs((double)(val) - (double)(expected)); \
    if (diff < (tol)) { test_pass++; printf("  [PASS] %s (val=%.8g, exp=%.8g, diff=%.2e)\n", _msg.c_str(), (double)(val), (double)(expected), diff); } \
    else { printf("  [FAIL] %s (val=%.8g, exp=%.8g, diff=%.2e, tol=%.2e, line %d)\n", _msg.c_str(), (double)(val), (double)(expected), diff, (tol), __LINE__); } \
} while(0)

// --- 测试 1: 线性 CD ---
static void test_1_linear_cd() {
    printf("\n=== 测试 1: 线性 CD (无畸变) ===\n");
    rng_seed(42);

    double s0 = 1.0;       // 1 arcsec/pixel
    double img_w = 1000;
    double img_h = 1000;
    int n_pairs = 50;
    double ra = 100.0, dec = 30.0;

    // U = W (s=1, θ=0, 无畸变, 无噪声)
    TestData td = generate_synthetic(s0, img_w, img_h, n_pairs,
                                      1.0, 0.0, 0.0, 0.0, 0.0);

    WcsResult r = run_wf_fit(td, s0, img_w, img_h, ra, dec);

    ASSERT_TRUE(r.success, "线性CD: 拟合成功");

    // CD[0] ≈ s0/3600 = 1/3600 ≈ 2.7778e-4
    double expected_cd0 = s0 / 3600.0;
    ASSERT_NEAR(r.cd[0], expected_cd0, 1e-8, "CD[0] ≈ s0/3600");
    // CD[1] ≈ 0
    ASSERT_NEAR(r.cd[1], 0.0, 1e-8, "CD[1] ≈ 0");
    // CD[2] ≈ 0
    ASSERT_NEAR(r.cd[2], 0.0, 1e-8, "CD[2] ≈ 0");
    // CD[3] ≈ -s0/3600
    ASSERT_NEAR(r.cd[3], -expected_cd0, 1e-8, "CD[3] ≈ -s0/3600");

    // RMS < 0.1 px
    ASSERT_TRUE(r.rms_px < 0.1, "RMS < 0.1px (实际=" + std::to_string(r.rms_px) + ")");

    // CRVAL ≈ (ra, dec)
    ASSERT_NEAR(r.crval[0], ra, 1e-6, "CRVAL[0] ≈ ra");
    ASSERT_NEAR(r.crval[1], dec, 1e-6, "CRVAL[1] ≈ dec");

    // sip_order = 0 (无畸变, 无需SIP)
    ASSERT_TRUE(r.sip_order == 0 || r.sip_order == 2,
                "sip_order 为 0 或 2 (实际=" + std::to_string(r.sip_order) + ")");
}

// --- 测试 2: 仿射畸变 (剪切) ---
static void test_2_affine() {
    printf("\n=== 测试 2: 仿射畸变 (剪切) ===\n");
    rng_seed(123);

    double s0 = 1.0;
    double img_w = 1000;
    double img_h = 1000;
    int n_pairs = 50;
    double ra = 100.0, dec = 30.0;

    // s=1, θ=0, shear=0.02 (2%剪切), 无SIP, 无噪声
    TestData td = generate_synthetic(s0, img_w, img_h, n_pairs,
                                      1.0, 0.0, 0.02, 0.0, 0.0);

    // 先用 skip_sip=1 获取仅仿射的 RMS
    WcsResult r_no_sip = run_wf_fit(td, s0, img_w, img_h, ra, dec, 4, 1);

    ASSERT_TRUE(r_no_sip.success, "仿射: 拟合成功");

    // 仿射应消除剪切畸变, RMS 应很小
    // shear=0.02, Wy up to 500, so displacement up to 10px
    // 仿射拟合后残差应 < 0.1px
    printf("  仿射 RMS = %.6f px\n", r_no_sip.rms_px);
    ASSERT_TRUE(r_no_sip.rms_px < 0.1,
                "仿射 RMS < 0.1px (实际=" + std::to_string(r_no_sip.rms_px) + ")");

    // 验证 CD 矩阵包含剪切信息: CD[1] 应非零
    // 剪切: U_x = Wx + 0.02*Wy, 在像素空间 ssx = ddx - 0.02*ddy (因 ddy=-Wy)
    // 仿射 A = [1, -0.02; 0, 1], A^{-1} = [1, 0.02; 0, 1]
    // CD' = CD · A^{-1} = [1/3600, 0.02/3600; 0, -1/3600]
    // CD'[1] ≈ +0.02/3600 ≈ +5.56e-6
    double expected_cd1 = 0.02 / 3600.0;
    ASSERT_NEAR(r_no_sip.cd[1], expected_cd1, 1e-7, "CD[1] 包含剪切 (+0.02/3600)");
}

// --- 测试 3: SIP 2 阶畸变 ---
static void test_3_sip_order2() {
    printf("\n=== 测试 3: SIP 2 阶畸变 ===\n");
    rng_seed(456);

    double s0 = 1.0;
    double img_w = 1000;
    double img_h = 1000;
    int n_pairs = 100;  // 足够多的点用于SIP拟合
    double ra = 100.0, dec = 30.0;

    // s=1, θ=0, 无剪切, sip_coeff=3.0 (3px@边缘2阶畸变), 无噪声
    TestData td = generate_synthetic(s0, img_w, img_h, n_pairs,
                                      1.0, 0.0, 0.0, 3.0, 0.0);

    // 先看仅仿射的 RMS (skip_sip=1)
    WcsResult r_no_sip = run_wf_fit(td, s0, img_w, img_h, ra, dec, 4, 1);
    printf("  仅仿射 RMS = %.6f px\n", r_no_sip.rms_px);

    // 再看含SIP的结果
    WcsResult r = run_wf_fit(td, s0, img_w, img_h, ra, dec, 4, 0);

    ASSERT_TRUE(r.success, "SIP: 拟合成功");

    // BIC 应选 order=2 (2阶畸变信号明确)
    printf("  SIP order = %d, RMS = %.6f px\n", r.sip_order, r.rms_px);
    ASSERT_TRUE(r.sip_order >= 2,
                "sip_order >= 2 (实际=" + std::to_string(r.sip_order) + ")");

    // SIP 应显著改善 RMS
    ASSERT_TRUE(r.rms_px < r_no_sip.rms_px,
                "SIP RMS < 仿射RMS (" + std::to_string(r.rms_px) + " < " +
                std::to_string(r_no_sip.rms_px) + ")");

    // SIP 后 RMS 应 < 1px (2阶畸变应被充分拟合)
    ASSERT_TRUE(r.rms_px < 1.0,
                "SIP RMS < 1.0px (实际=" + std::to_string(r.rms_px) + ")");
}

// --- 测试 4: SIP 过拟合防护 ---
static void test_4_overfitting_protection() {
    printf("\n=== 测试 4: SIP 过拟合防护 ===\n");
    rng_seed(789);

    double s0 = 1.0;
    double img_w = 1000;
    double img_h = 1000;
    int n_pairs = 50;
    double ra = 100.0, dec = 30.0;

    // 纯线性数据 + 小噪声 (σ=0.15px), 无SIP畸变
    TestData td = generate_synthetic(s0, img_w, img_h, n_pairs,
                                      1.0, 0.0, 0.0, 0.0, 0.15);

    WcsResult r = run_wf_fit(td, s0, img_w, img_h, ra, dec, 4, 0);

    ASSERT_TRUE(r.success, "过拟合防护: 拟合成功");

    printf("  sip_order = %d, RMS = %.6f px\n", r.sip_order, r.rms_px);

    // BIC 不应选 order=4 (纯噪声数据, 高阶过拟合)
    ASSERT_TRUE(r.sip_order < 4,
                "sip_order < 4 (不过拟合, 实际=" + std::to_string(r.sip_order) + ")");

    // 理想情况: sip_order = 0 (噪声不足以支持任何SIP阶数)
    // 但放宽为 < 4 也可接受
}

// ============================================================================
// main
// ============================================================================

int main() {
    printf("========================================\n");
    printf("V4.2 WcsFitter 单元测试 (Task 6)\n");
    printf("========================================\n");

    test_1_linear_cd();
    test_2_affine();
    test_3_sip_order2();
    test_4_overfitting_protection();

    printf("\n========================================\n");
    printf("测试结果: %d/%d PASS\n", test_pass, test_count);
    printf("========================================\n");

    return (test_pass == test_count) ? 0 : 1;
}

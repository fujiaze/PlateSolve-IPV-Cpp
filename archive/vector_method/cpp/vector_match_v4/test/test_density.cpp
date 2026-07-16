/**
 * test_density.cpp - V4.0 Phase 0 密度匹配模块单元测试（Task 2）
 *
 * 测试三种场景：
 *   1. 银河面密集星场: n_img_bright=200, FOV_diag≈9.77° → n_target=300，验证迭代收敛到 [270,330]
 *   2. 高银纬稀疏星场: n_img_bright=30,  FOV_diag≈2.0°  → n_target=45， 验证迭代放宽 m_lim
 *   3. 窄带低星数:     n_img_bright=15,  → n_target=22，验证有足够候选
 *
 * 编译（独立可执行）：
 *   g++ -O2 -std=c++17 -Wall -Iinclude test/test_density.cpp src/vm4_density.cpp -o test_density.exe
 *
 * 运行：
 *   ./test_density.exe
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <utility>
#include <string>

#include "../include/vm4_density.h"

// 简易测试断言
static int g_test_pass = 0;
static int g_test_fail = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { ++g_test_pass; } \
    else { ++g_test_fail; std::fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); } \
} while(0)

#define ASSERT_NEAR(val, exp, tol, msg) do { \
    bool _ok = std::fabs((val) - (exp)) <= (tol); \
    if (_ok) { ++g_test_pass; } \
    else { ++g_test_fail; std::fprintf(stderr, \
        "[FAIL] %s:%d: %s (val=%.6f, exp=%.6f, tol=%.6f)\n", \
        __FILE__, __LINE__, msg, (double)(val), (double)(exp), (double)(tol)); } \
} while(0)

#define ASSERT_IN_RANGE(val, lo, hi, msg) do { \
    bool _ok = ((val) >= (lo) && (val) <= (hi)); \
    if (_ok) { ++g_test_pass; } \
    else { ++g_test_fail; std::fprintf(stderr, \
        "[FAIL] %s:%d: %s (val=%lld, range=[%lld,%lld])\n", \
        __FILE__, __LINE__, msg, (long long)(val), (long long)(lo), (long long)(hi)); } \
} while(0)

static constexpr double PI = 3.14159265358979323846;

// 模拟 Gaia 查询函数：
//   N(m_lim) = density_at_m10 × area × 10^(0.4×(m_lim - 10))
// density_at_m10 控制场密度（颗/平方度 @ m=10），area 由 query_radius 决定。
struct MockGaiaConfig {
    double density_at_m10;
    double m_ref = 10.0;
};

static auto make_mock_gaia(const MockGaiaConfig& cfg) {
    return [cfg](double ra, double dec, double radius, double m_lim) -> int {
        (void)ra; (void)dec;
        double area = PI * radius * radius;  // 平方度
        double density = cfg.density_at_m10 * std::pow(10.0, 0.4 * (m_lim - cfg.m_ref));
        double n = density * area;
        return static_cast<int>(n + 0.5);  // 四舍五入
    };
}

// ============================================================================
// 测试 1: 银河面密集星场
// ============================================================================
static void test_dense_field() {
    std::fprintf(stderr, "\n=== 测试 1: 银河面密集星场 ===\n");

    // 输入参数（FOV_diag ≈ 9.78°）
    double f_mm = 102.0;
    double pix_um = 3.0;
    double W = 4104.0, H = 4104.0;
    int    n_img_bright = 200;
    double k_match = 1.5;
    double qrf = 1.0;

    auto info = vm4::compute_fov_and_density(f_mm, pix_um, W, H, n_img_bright, k_match, qrf);

    // 期望: s0 ≈ 6.067, FOV_diag ≈ 9.78°, n_target = 300
    ASSERT_NEAR(info.s0, 6.067, 0.01, "s0");
    ASSERT_NEAR(info.fov_diag_deg, 9.77, 0.05, "FOV_diag");
    ASSERT_TRUE(info.n_target == 300, "n_target should be 300");
    ASSERT_TRUE(info.rho_target > info.rho_img, "rho_target should be > rho_img");

    // Mock Gaia: 设置使得 m_lim ≈ 14 时 N ≈ 300
    // N(14) = density × π×(9.78/2)² × 10^(0.4×4) = density × 75.0 × 39.8 = 300
    // → density_at_m10 = 300 / (75.0 × 39.8) ≈ 0.1005
    MockGaiaConfig cfg;
    cfg.density_at_m10 = 0.1005;
    auto mock = make_mock_gaia(cfg);

    // 初始 m_cut = 12.0（太低，N 不足 → 迭代放宽）
    double m_cut_initial = 12.0;
    double step = 0.5;
    int    max_iter = 8;
    double tol = 0.1;

    auto result = vm4::density_match_query(
        266.0, -29.0, info.query_radius_deg,
        info.n_target, m_cut_initial, step, max_iter, tol, mock);

    // 期望: 收敛, n_gaia ∈ [270, 330]
    ASSERT_TRUE(result.converged, "should converge");
    ASSERT_IN_RANGE(result.final_n_gaia, 270, 330, "final_n_gaia in [270,330]");
    ASSERT_TRUE(result.iterations < max_iter, "iterations < max_iter");
    ASSERT_TRUE(result.final_mag_lim > m_cut_initial, "m_lim should increase");

    std::fprintf(stderr, "[OK] 测试 1 通过: n_final=%d, m_final=%.3f, iters=%d\n",
        result.final_n_gaia, result.final_mag_lim, result.iterations);
}

// ============================================================================
// 测试 2: 高银纬稀疏星场
// ============================================================================
static void test_sparse_field() {
    std::fprintf(stderr, "\n=== 测试 2: 高银纬稀疏星场 ===\n");

    // 输入参数（FOV_diag ≈ 2.0°）
    double f_mm = 430.0;
    double pix_um = 3.0;
    double W = 4000.0, H = 3000.0;
    int    n_img_bright = 30;
    double k_match = 1.5;
    double qrf = 1.0;

    auto info = vm4::compute_fov_and_density(f_mm, pix_um, W, H, n_img_bright, k_match, qrf);

    // 期望: FOV_diag ≈ 2.0°, n_target = 45
    ASSERT_NEAR(info.fov_diag_deg, 2.0, 0.05, "FOV_diag ~ 2.0");
    ASSERT_TRUE(info.n_target == 45, "n_target should be 45");

    // Mock Gaia: 稀疏星场，N(15) = 45
    // N(15) = density × π×1² × 10^(0.4×5) = density × π × 100 = 45
    // → density_at_m10 = 45 / (π × 100) ≈ 0.1432
    MockGaiaConfig cfg;
    cfg.density_at_m10 = 0.1432;
    auto mock = make_mock_gaia(cfg);

    // 初始 m_cut = 13.0（星数不足 → 迭代放宽 m_lim）
    double m_cut_initial = 13.0;
    double step = 0.5;
    int    max_iter = 8;
    double tol = 0.1;

    auto result = vm4::density_match_query(
        180.0, 60.0, info.query_radius_deg,
        info.n_target, m_cut_initial, step, max_iter, tol, mock);

    // 期望: 收敛, m_lim 放宽（增大）
    ASSERT_TRUE(result.converged, "should converge");
    ASSERT_TRUE(result.final_mag_lim > m_cut_initial,
                "m_lim should be loosened (increased)");
    ASSERT_IN_RANGE(result.final_n_gaia,
                    static_cast<int>(45 * (1 - tol) - 0.5),
                    static_cast<int>(45 * (1 + tol) + 0.5),
                    "final_n_gaia within tolerance of n_target");

    std::fprintf(stderr, "[OK] 测试 2 通过: n_final=%d, m_final=%.3f, iters=%d\n",
        result.final_n_gaia, result.final_mag_lim, result.iterations);
}

// ============================================================================
// 测试 3: 窄带低星数
// ============================================================================
static void test_narrow_low_count() {
    std::fprintf(stderr, "\n=== 测试 3: 窄带低星数 ===\n");

    // 输入参数（窄带 FOV ≈ 0.57°）
    double f_mm = 1500.0;
    double pix_um = 3.0;
    double W = 4000.0, H = 3000.0;
    int    n_img_bright = 15;
    double k_match = 1.5;
    double qrf = 1.0;

    auto info = vm4::compute_fov_and_density(f_mm, pix_um, W, H, n_img_bright, k_match, qrf);

    // 期望: n_target = 22（round(1.5×15) = 22 或 23，依四舍五入）
    // 1.5 × 15 = 22.5 → round = 22 或 23（C++ lround 为 22.5 → 23，依实现）
    // 这里允许 22 或 23
    ASSERT_TRUE(info.n_target == 22 || info.n_target == 23,
                "n_target should be 22 or 23 (1.5×15=22.5)");
    ASSERT_TRUE(info.fov_diag_deg < 1.0, "narrow FOV < 1.0 deg");

    // Mock Gaia: 窄带低密度，目标 N(14) = 22
    // area = π×(0.573/2)² ≈ 0.257 deg²
    // N(14) = density × 0.257 × 10^1.6 = density × 0.257 × 39.8 = 22
    // → density_at_m10 = 22 / (0.257 × 39.8) ≈ 2.150
    MockGaiaConfig cfg;
    cfg.density_at_m10 = 2.150;
    auto mock = make_mock_gaia(cfg);

    // 初始 m_cut = 12.0（星数不足 → 迭代放宽）
    double m_cut_initial = 12.0;
    double step = 0.5;
    int    max_iter = 8;
    double tol = 0.1;

    auto result = vm4::density_match_query(
        100.0, 30.0, info.query_radius_deg,
        info.n_target, m_cut_initial, step, max_iter, tol, mock);

    // 期望: 有足够候选 (final_n_gaia >= n_target × (1-tol))
    int n_lo = static_cast<int>(info.n_target * (1.0 - tol));
    ASSERT_TRUE(result.final_n_gaia >= n_lo,
                "should have enough candidates (n >= n_target×(1-tol))");
    // 期望: 最终星等比初始放宽
    ASSERT_TRUE(result.final_mag_lim >= m_cut_initial,
                "m_lim should not decrease from initial (already too few)");

    std::fprintf(stderr, "[OK] 测试 3 通过: n_final=%d, m_final=%.3f, iters=%d\n",
        result.final_n_gaia, result.final_mag_lim, result.iterations);
}

// ============================================================================
// 测试 4: gnomonic 投影筛选 FOV 内星
// ============================================================================
static void test_gnomonic_projection() {
    std::fprintf(stderr, "\n=== 测试 4: gnomonic 投影 ===\n");

    // 中心 (RA=180°, Dec=30°)
    double center_ra = 180.0, center_dec = 30.0;
    double fov_diag_deg = 2.0;  // FOV 半径 1°

    // 构造一组星：中心点、FOV内、FOV外、反面（cosc<0）
    std::vector<std::pair<double,double>> stars;
    stars.emplace_back(180.0, 30.0);   // 中心
    stars.emplace_back(180.1, 30.0);   // 距中心 0.1° → 在 FOV 内
    stars.emplace_back(180.5, 30.0);   // 距中心 0.5° → 在 FOV 内
    stars.emplace_back(181.5, 30.0);   // 距中心 1.5° → 在 FOV 外
    stars.emplace_back(180.0, 31.5);   // 距中心 1.5° → 在 FOV 外
    stars.emplace_back(0.0, -30.0);    // 反面，cosc<0 → 跳过

    auto proj = vm4::gnomonic_project_fov(stars, center_ra, center_dec, fov_diag_deg);

    // 期望: 保留 3 颗（前 3 颗，后 3 颗被剔除）
    ASSERT_TRUE(proj.size() == 3, "should keep 3 stars in FOV");

    // 中心星投影后应在原点附近
    ASSERT_NEAR(proj[0].first,  0.0, 1.0, "center xi ~ 0");
    ASSERT_NEAR(proj[0].second, 0.0, 1.0, "center eta ~ 0");

    // 第一颗偏移星: xi ≈ 0.1° × cos(30°) ≈ 0.0866° → 311.77"
    // 实际上 gnomonic 在小角度时 xi ≈ Δra × cos(dec0) × 3600
    // 0.1 × cos(30°) × 3600 = 311.77"
    ASSERT_NEAR(proj[1].first, 311.77, 5.0, "xi for Δra=0.1° at dec=30°");

    std::fprintf(stderr, "[OK] 测试 4 通过: 投影 %zu 颗\n", proj.size());
}

// ============================================================================
// 测试 5: compute_initial_mag_cut 公式验证
// ============================================================================
static void test_initial_mag_cut() {
    std::fprintf(stderr, "\n=== 测试 5: 初始星等公式 ===\n");

    // m_cut = 6 + 1.5×log10(f) + 2×log10(t)
    // f=100, t=300: 6 + 1.5×2 + 2×log10(300)
    //              = 6 + 3 + 2×2.4771 = 6 + 3 + 4.9542 = 13.9542
    double m1 = vm4::compute_initial_mag_cut(100.0, 300.0);
    ASSERT_NEAR(m1, 13.9542, 0.001, "m_cut for f=100,t=300");

    // f=430, t=120: 6 + 1.5×log10(430) + 2×log10(120)
    //              = 6 + 1.5×2.6335 + 2×2.0792
    //              = 6 + 3.9502 + 4.1584 = 14.1086
    double m2 = vm4::compute_initial_mag_cut(430.0, 120.0);
    ASSERT_NEAR(m2, 14.1086, 0.001, "m_cut for f=430,t=120");

    std::fprintf(stderr, "[OK] 测试 5 通过\n");
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::fprintf(stderr, "================ vm4_density 单元测试 ================\n");

    test_dense_field();
    test_sparse_field();
    test_narrow_low_count();
    test_gnomonic_projection();
    test_initial_mag_cut();

    std::fprintf(stderr, "\n================ 测试汇总 ================\n");
    std::fprintf(stderr, "通过: %d, 失败: %d\n", g_test_pass, g_test_fail);

    return g_test_fail == 0 ? 0 : 1;
}

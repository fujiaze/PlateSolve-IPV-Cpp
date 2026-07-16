// ============================================================================
// test_select.cpp - V4.3 StarSelector 单元测试 (Task 2 SubTask 2.3)
//
// 测试 vm44_select.cpp 中的内部辅助函数:
//   1. compute_fov_density: FOV / 查询半径 / 目标星数计算
//   2. compute_initial_mag_cut: 初始极限星等公式
//   3. density_match_iterate: Mock Gaia 查询迭代收敛 + 自适应步长
//   4. select_image_stars: V4.1 不对称选星策略 (饱和>50全选 / 饱和+非饱和补足)
//   5. gnomonic_forward_proj: Gnomonic 正向投影
//
// 从 V4.2 test_ss.cpp 迁移, 适配到 V4.3 内部辅助函数接口
// 编译: make test_select
// 运行: ./test/test_select.exe
// ============================================================================

#include "vm44_internal.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <atomic>
#include <functional>
#include <vector>

// MinGW 严格 ISO C++ 下 M_PI 不可用，自定义常量
static constexpr double TEST_PI = 3.14159265358979323846;

// ============================================================================
// 测试框架
// ============================================================================

static int g_test_pass = 0;
static int g_test_fail = 0;

#define ASSERT_TRUE(cond, msg) do { \
    if (cond) { g_test_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_test_fail++; printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    double _d = std::fabs((double)(a) - (double)(b)); \
    if (_d <= (tol)) { g_test_pass++; printf("  [PASS] %s (val=%.4f, exp=%.4f, d=%.4f)\n", msg, (double)(a), (double)(b), _d); } \
    else { g_test_fail++; printf("  [FAIL] %s (val=%.4f, exp=%.4f, d=%.4f, tol=%.4f, line %d)\n", msg, (double)(a), (double)(b), _d, (double)(tol), __LINE__); } \
} while (0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) == (b)) { g_test_pass++; printf("  [PASS] %s\n", msg); } \
    else { g_test_fail++; printf("  [FAIL] %s (val=%d, exp=%d, line %d)\n", msg, (int)(a), (int)(b), __LINE__); } \
} while (0)

// ============================================================================
// Mock Gaia 查询回调
// 模型: N = density × area × 10^(0.4×(m-10))
// ============================================================================

static double g_mock_density = 1.0;     // 颗/平方度
static double g_mock_area    = 1.0;     // 平方度
static std::atomic<int> g_mock_call_count{0};

static int mock_gaia_query(double ra, double dec, double radius_deg, double mag_lim)
{
    (void)ra; (void)dec; (void)radius_deg;
    g_mock_call_count++;
    double n = g_mock_density * g_mock_area * std::pow(10.0, 0.4 * (mag_lim - 10.0));
    if (n < 0.0) n = 0.0;
    return static_cast<int>(n);
}

// ============================================================================
// 测试场景 1: FOV / 查询半径 / 目标星数 基本计算
// f=200mm, pix=3.76um, 4500×3600, n_img=50, ratio=1.5, qrf=0.55
// ============================================================================

static void test_fov_basic()
{
    printf("\n[TEST] test_fov_basic: FOV / 查询半径 / 目标星数计算\n");

    double s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target;
    int n_target;

    v44::compute_fov_density(
        200.0, 3.76, 4500, 3600, 50,
        1.5, 0.55,
        s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target, n_target,
        nullptr);

    // 期望值
    double s0_exp = 206.265 * 3.76 / 200.0;
    double diag_pix = std::sqrt(4500.0 * 4500.0 + 3600.0 * 3600.0);
    double fov_diag_exp = diag_pix * s0_exp / 3600.0;
    double query_r_exp = fov_diag_exp * 0.55;
    double img_area_exp = (4500.0 * s0_exp / 3600.0) * (3600.0 * s0_exp / 3600.0);
    double query_area_exp = TEST_PI * query_r_exp * query_r_exp;
    double n_target_dbl = 1.5 * 50.0 * (query_area_exp / img_area_exp);
    int n_target_exp = std::max(50, (int)std::lround(n_target_dbl));

    printf("    s0=%.4f, fov_diag=%.4f°, query_r=%.4f°, "
           "img_area=%.4f, query_area=%.4f, n_target=%d\n",
           s0, fov_diag, query_r, img_area, query_area, n_target);

    ASSERT_NEAR(s0, s0_exp, 1e-3, "s0 像素尺度");
    ASSERT_NEAR(fov_diag, fov_diag_exp, 1e-4, "FOV 对角线");
    ASSERT_NEAR(query_r, query_r_exp, 1e-4, "查询半径");
    ASSERT_NEAR(query_r, fov_diag * 0.55, 1e-6, "查询半径 = FOV×0.55");
    ASSERT_TRUE(n_target >= 50, "n_target >= 50");
    ASSERT_EQ(n_target, n_target_exp, "n_target 数值正确");
    ASSERT_NEAR(img_area, img_area_exp, 1e-6, "图像面积");
    ASSERT_NEAR(query_area, query_area_exp, 1e-6, "查询圆面积");
    ASSERT_NEAR(rho_img, 50.0 / img_area_exp, 1e-6, "rho_img");
    ASSERT_NEAR(rho_target, 1.5 * 50.0 / img_area_exp, 1e-6, "rho_target");
}

// ============================================================================
// 测试场景 2: Mock Gaia 查询收敛性
// 调整 density 使 m≈14 时 N≈n_target，验证迭代收敛
// ============================================================================

static void test_convergence()
{
    printf("\n[TEST] test_convergence: Mock Gaia 查询迭代收敛\n");

    // 先计算几何量
    double s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target;
    int n_target;
    v44::compute_fov_density(
        200.0, 3.76, 4500, 3600, 50,
        1.5, 0.55,
        s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target, n_target,
        nullptr);
    printf("    n_target=%d, query_area=%.4f°²\n", n_target, query_area);

    // 设 density 使 m=14 时 N=n_target
    double m_target = 14.0;
    double factor_at_m = std::pow(10.0, 0.4 * (m_target - 10.0));
    double density = n_target / (query_area * factor_at_m);
    g_mock_density = density;
    g_mock_area = query_area;
    g_mock_call_count = 0;

    // 初始 m_cut
    double m_cut = v44::compute_initial_mag_cut(200.0, 1.0, nullptr);

    // 迭代
    double m_final;
    int n_final, iterations;
    bool converged;
    v44::density_match_iterate(
        mock_gaia_query, 100.0, 30.0, query_r, n_target, m_cut,
        0.5, 15, 0.1,
        m_final, n_final, iterations, converged,
        nullptr);

    printf("    m_cut=%.4f, m_final=%.4f, n_final=%d, iters=%d, converged=%d\n",
           m_cut, m_final, n_final, iterations, (int)converged);

    ASSERT_TRUE(converged, "迭代收敛");

    // 验证最终 n_gaia 落在容差范围内
    double n_lo = n_target * (1.0 - 0.1);
    double n_hi = n_target * (1.0 + 0.1);
    ASSERT_TRUE(n_final >= (int)n_lo && n_final <= (int)n_hi,
                "n_gaia_final 在容差范围内");

    // 验证迭代次数 < max_iter
    ASSERT_TRUE(iterations < 15, "迭代次数 < max_iter");

    // 验证最终 m_lim 在 m_target 附近
    ASSERT_NEAR(m_final, m_target, 0.6, "m_lim_final 接近 m_target=14");
}

// ============================================================================
// 测试场景 3: 小视场 / 高密度 (n_target=50 下限触发)
// f=1000mm, pix=3.76um, 2000×1500, n_img=10
// ============================================================================

static void test_small_fov()
{
    printf("\n[TEST] test_small_fov: 小视场 n_target=50 下限触发\n");

    double s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target;
    int n_target;
    v44::compute_fov_density(
        1000.0, 3.76, 2000, 1500, 10,
        1.5, 0.55,
        s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target, n_target,
        nullptr);

    double s0_exp = 206.265 * 3.76 / 1000.0;
    double fov_diag_exp = std::sqrt(2000.0 * 2000.0 + 1500.0 * 1500.0) * s0_exp / 3600.0;
    double n_target_raw = 1.5 * 10.0 * (query_area / img_area);

    printf("    s0=%.4f, fov_diag=%.4f°, query_area=%.5f°², img_area=%.5f°², "
           "n_target_raw=%.2f, n_target=%d\n",
           s0, fov_diag, query_area, img_area, n_target_raw, n_target);

    ASSERT_TRUE(n_target_raw < 50.0, "n_target_raw < 50, 触发下限");
    ASSERT_EQ(n_target, 50, "n_target 下限为 50");
    ASSERT_NEAR(s0, s0_exp, 1e-3, "s0 小视场");
    ASSERT_NEAR(fov_diag, fov_diag_exp, 1e-4, "FOV 小视场");

    // 小视场也应能收敛
    double factor_at_m = std::pow(10.0, 0.4 * (14.0 - 10.0));
    double density = 50.0 / (query_area * factor_at_m);
    g_mock_density = density;
    g_mock_area = query_area;
    g_mock_call_count = 0;

    double m_cut = v44::compute_initial_mag_cut(1000.0, 1.0, nullptr);
    double m_final;
    int n_final, iterations;
    bool converged;
    v44::density_match_iterate(
        mock_gaia_query, 200.0, -30.0, query_r, n_target, m_cut,
        0.5, 15, 0.1,
        m_final, n_final, iterations, converged,
        nullptr);

    printf("    m_final=%.4f, n_final=%d, iters=%d, converged=%d\n",
           m_final, n_final, iterations, (int)converged);
    ASSERT_TRUE(converged, "小视场收敛");
}

// ============================================================================
// 测试场景 4: 初始极限星等公式
// m_cut = 6 + 1.5×log10(f_mm) + 2×log10(t_s)
// ============================================================================

static void test_initial_mag_cut()
{
    printf("\n[TEST] test_initial_mag_cut: 初始极限星等公式\n");

    // f=200mm, t=1s → m_cut = 6 + 1.5×log10(200) + 2×log10(1) = 6 + 3.4515 + 0 = 9.4515
    double m1 = v44::compute_initial_mag_cut(200.0, 1.0, nullptr);
    double m1_exp = 6.0 + 1.5 * std::log10(200.0) + 2.0 * std::log10(1.0);
    printf("    f=200mm t=1s: m_cut=%.4f (exp=%.4f)\n", m1, m1_exp);
    ASSERT_NEAR(m1, m1_exp, 1e-6, "m_cut f=200mm t=1s");

    // f=1000mm, t=60s → m_cut = 6 + 1.5×log10(1000) + 2×log10(60) = 6 + 4.5 + 3.5527 = 14.0527
    double m2 = v44::compute_initial_mag_cut(1000.0, 60.0, nullptr);
    double m2_exp = 6.0 + 1.5 * std::log10(1000.0) + 2.0 * std::log10(60.0);
    printf("    f=1000mm t=60s: m_cut=%.4f (exp=%.4f)\n", m2, m2_exp);
    ASSERT_NEAR(m2, m2_exp, 1e-6, "m_cut f=1000mm t=60s");

    // f=50mm, t=300s → 短焦长曝光
    double m3 = v44::compute_initial_mag_cut(50.0, 300.0, nullptr);
    double m3_exp = 6.0 + 1.5 * std::log10(50.0) + 2.0 * std::log10(300.0);
    printf("    f=50mm t=300s: m_cut=%.4f (exp=%.4f)\n", m3, m3_exp);
    ASSERT_NEAR(m3, m3_exp, 1e-6, "m_cut f=50mm t=300s");
}

// ============================================================================
// 测试场景 5: 图像侧选星策略 (V4.1 不对称)
//   饱和星数 > img_n_target → 全选饱和星
//   饱和星数 < img_n_target → 饱和全选 + 非饱和按 flux 降序补足
// ============================================================================

static void test_select_image_stars()
{
    printf("\n[TEST] test_select_image_stars: V4.1 不对称选星策略\n");

    // 场景 A: 饱和星 60 颗 > img_n_target=50 → 全选饱和星 (60颗)
    {
        std::vector<double> flux(100);
        std::vector<bool> sat(100, false);
        for (int i = 0; i < 60; ++i) { sat[i] = true; flux[i] = 50000.0 - i * 100; }
        for (int i = 60; i < 100; ++i) { sat[i] = false; flux[i] = 10000.0 - i * 50; }

        auto sel = v44::select_image_stars(flux, sat, 50, nullptr);
        printf("    场景A: 饱和60颗 > target=50 → 选 %d 颗\n", (int)sel.size());
        ASSERT_EQ((int)sel.size(), 60, "饱和>50: 全选饱和星 (60颗)");
        // 全部应是饱和星 (索引 0-59)
        bool all_sat = true;
        for (int idx : sel) { if (!sat[idx]) { all_sat = false; break; } }
        ASSERT_TRUE(all_sat, "饱和>50: 选中全为饱和星");
    }

    // 场景 B: 饱和星 20 颗 < img_n_target=50 → 饱和20 + 非饱和30 = 50颗
    {
        std::vector<double> flux(100);
        std::vector<bool> sat(100, false);
        for (int i = 0; i < 20; ++i) { sat[i] = true; flux[i] = 50000.0 - i * 100; }
        for (int i = 20; i < 100; ++i) { sat[i] = false; flux[i] = (100 - i) * 100.0; }

        auto sel = v44::select_image_stars(flux, sat, 50, nullptr);
        printf("    场景B: 饱和20颗 + 非饱和30颗 = 50 → 选 %d 颗\n", (int)sel.size());
        ASSERT_EQ((int)sel.size(), 50, "饱和<50: 饱和+非饱和=50颗");

        // 验证: 20 颗饱和 + 30 颗最亮非饱和
        int n_sat_sel = 0, n_nonsat_sel = 0;
        for (int idx : sel) { if (sat[idx]) n_sat_sel++; else n_nonsat_sel++; }
        ASSERT_EQ(n_sat_sel, 20, "饱和<50: 选中含 20 颗饱和");
        ASSERT_EQ(n_nonsat_sel, 30, "饱和<50: 选中含 30 颗非饱和");

        // 验证非饱和按 flux 降序选取 (flux[20] > flux[21] > ... > flux[49])
        // 选中的非饱和应是索引 20-49
        bool correct_nonsat = true;
        for (int idx : sel) {
            if (!sat[idx] && idx >= 50) { correct_nonsat = false; break; }
        }
        ASSERT_TRUE(correct_nonsat, "饱和<50: 非饱和为最亮的 30 颗");
    }

    // 场景 C: 无饱和星 → 全选非饱和前 50 颗
    {
        std::vector<double> flux(100);
        std::vector<bool> sat(100, false);
        for (int i = 0; i < 100; ++i) { flux[i] = (100 - i) * 100.0; }

        auto sel = v44::select_image_stars(flux, sat, 50, nullptr);
        printf("    场景C: 无饱和 → 选非饱和前50 → 选 %d 颗\n", (int)sel.size());
        ASSERT_EQ((int)sel.size(), 50, "无饱和: 选 50 颗非饱和");
    }

    // 场景 D: 星点总数 < img_n_target → 全选
    {
        std::vector<double> flux(10, 1000.0);
        std::vector<bool> sat(10, false);

        auto sel = v44::select_image_stars(flux, sat, 50, nullptr);
        printf("    场景D: 总星数10 < target=50 → 全选 → 选 %d 颗\n", (int)sel.size());
        ASSERT_EQ((int)sel.size(), 10, "星数<target: 全选");
    }
}

// ============================================================================
// 测试场景 6: Gnomonic 正向投影
// ============================================================================

static void test_gnomonic()
{
    printf("\n[TEST] test_gnomonic: Gnomonic 正向投影\n");

    // 中心点投影 → (0, 0)
    {
        double xi, eta;
        bool valid;
        v44::gnomonic_forward_proj(100.0, 30.0, 100.0, 30.0, xi, eta, valid);
        printf("    中心点: xi=%.4f\" eta=%.4f\" valid=%d\n", xi, eta, (int)valid);
        ASSERT_TRUE(valid, "中心点投影 valid");
        ASSERT_NEAR(xi, 0.0, 1e-6, "中心点 xi ≈ 0");
        ASSERT_NEAR(eta, 0.0, 1e-6, "中心点 eta ≈ 0");
    }

    // 已知点投影: RA 方向偏移 1 角秒
    // 在赤道 (dec=0), RA 偏移 1" → xi ≈ 1" (cos(dec)=1)
    {
        double xi, eta;
        bool valid;
        double ra0 = 100.0, dec0 = 0.0;
        // RA 偏移 1 角秒 = 1/3600 度
        v44::gnomonic_forward_proj(ra0 + 1.0/3600.0, dec0, ra0, dec0, xi, eta, valid);
        printf("    RA偏移1\" (赤道): xi=%.4f\" eta=%.4f\" valid=%d\n", xi, eta, (int)valid);
        ASSERT_TRUE(valid, "RA偏移投影 valid");
        // xi 应接近 1" (赤道处 cos(dec)=1)
        ASSERT_NEAR(xi, 1.0, 0.01, "RA偏移1\" → xi≈1\" (赤道)");
    }

    // Dec 方向偏移 1 角秒 → eta ≈ 1"
    {
        double xi, eta;
        bool valid;
        double ra0 = 100.0, dec0 = 30.0;
        v44::gnomonic_forward_proj(ra0, dec0 + 1.0/3600.0, ra0, dec0, xi, eta, valid);
        printf("    Dec偏移1\" (dec=30°): xi=%.4f\" eta=%.4f\" valid=%d\n", xi, eta, (int)valid);
        ASSERT_TRUE(valid, "Dec偏移投影 valid");
        ASSERT_NEAR(eta, 1.0, 0.01, "Dec偏移1\" → eta≈1\"");
    }

    // 高纬度 RA 偏移应被 cos(dec) 压缩
    // dec=60°, RA偏移 1" → xi ≈ cos(60°) = 0.5"
    {
        double xi, eta;
        bool valid;
        double ra0 = 100.0, dec0 = 60.0;
        v44::gnomonic_forward_proj(ra0 + 1.0/3600.0, dec0, ra0, dec0, xi, eta, valid);
        printf("    RA偏移1\" (dec=60°): xi=%.4f\" eta=%.4f\" valid=%d\n", xi, eta, (int)valid);
        ASSERT_TRUE(valid, "高纬RA偏移 valid");
        // xi 应接近 cos(60°) = 0.5"
        ASSERT_NEAR(xi, 0.5, 0.01, "RA偏移1\" dec=60° → xi≈0.5\" (cos压缩)");
    }

    // 天极附近投影应无效 (cosc < 0)
    {
        double xi, eta;
        bool valid;
        // 中心在赤道, 投影点在对侧 (RA+180, Dec=0) → cosc < 0
        v44::gnomonic_forward_proj(280.0, 0.0, 100.0, 0.0, xi, eta, valid);
        printf("    对侧点: valid=%d (应为 0)\n", (int)valid);
        ASSERT_TRUE(!valid, "对侧点投影 invalid");
    }
}

// ============================================================================
// 测试场景 7: 自适应步长验证 (前4次 step_init, 后续 step_init/2)
// ============================================================================

static void test_adaptive_step()
{
    printf("\n[TEST] test_adaptive_step: 自适应步长前4次/后续减半\n");

    double s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target;
    int n_target;
    v44::compute_fov_density(
        200.0, 3.76, 4500, 3600, 50,
        1.5, 0.55,
        s0, fov_diag, query_r, query_area, img_area, rho_img, rho_target, n_target,
        nullptr);

    // 让 mock 始终返回 0 → 持续 +step → 前4次 +0.5, 后续 +0.25
    g_mock_density = 0.0;
    g_mock_area = 1.0;
    g_mock_call_count = 0;

    double m_cut = v44::compute_initial_mag_cut(200.0, 1.0, nullptr);
    double m_final;
    int n_final, iterations;
    bool converged;
    int max_iter = 15;
    v44::density_match_iterate(
        mock_gaia_query, 100.0, 30.0, query_r, n_target, m_cut,
        0.5, max_iter, 0.05,
        m_final, n_final, iterations, converged,
        nullptr);

    // m_cut = 6 + 1.5×log10(200) = 9.4515
    double m_cut_exp = 6.0 + 1.5 * std::log10(200.0) + 2.0 * std::log10(1.0);
    // 前4次 +0.5, 后 (15-4)=11 次 +0.25
    double m_final_exp = m_cut_exp + 4 * 0.5 + (max_iter - 4) * 0.25;
    printf("    m_cut=%.4f, m_final=%.4f (exp=%.4f), iters=%d, converged=%d\n",
           m_cut, m_final, m_final_exp, iterations, (int)converged);

    ASSERT_TRUE(!converged, "mock 始终返回 0, 不收敛");
    ASSERT_EQ(iterations, max_iter, "走满 max_iter 次");
    ASSERT_NEAR(m_final, m_final_exp, 1e-6, "自适应步长累计正确");
}

// ============================================================================
// 主入口
// ============================================================================

int main()
{
    printf("=== V4.3 StarSelector 单元测试 ===\n");

    test_fov_basic();
    test_convergence();
    test_small_fov();
    test_initial_mag_cut();
    test_select_image_stars();
    test_gnomonic();
    test_adaptive_step();

    printf("\n=== 测试汇总 ===\n");
    printf("  PASS: %d\n", g_test_pass);
    printf("  FAIL: %d\n", g_test_fail);
    printf("  结果: %s\n", g_test_fail == 0 ? "ALL PASS" : "HAS FAILURES");

    return g_test_fail == 0 ? 0 : 1;
}

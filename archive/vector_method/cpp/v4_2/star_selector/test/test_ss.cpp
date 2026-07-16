/**
 * test_ss.cpp - V4.2 StarSelector 模块单元测试（Task 2 SubTask 2.5）
 *
 * 测试场景:
 *   1. FOV 计算: f=200mm, pix=3.76um, 4500×3600 → s0≈3.88"/px
 *   2. 查询半径: query_r = FOV_diag × 0.55
 *   3. 目标星数: n_img=50, ratio=1.5, query_area/img_area > 1, n_target≥50
 *   4. Mock Gaia 查询: 返回 N = density × area × 10^(0.4×(m-10))
 *   5. 验证迭代收敛
 *   6. 边界场景: 饱和>50 / 饱和<50（图像侧选星逻辑由 Python 完成，此处仅
 *      验证 C++ 端 n_img_bright 透传与密度计算正确性）
 *
 * 注意: C++ 端不实现图像侧选星，仅做密度匹配查询。本测试 mock Gaia 查询
 * 回调，验证 FOV/密度/迭代收敛逻辑。
 */

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <atomic>

#include "ss_api.h"

// MinGW 严格 ISO C++ 下 M_PI 不可用，自定义常量 TEST_PI
static constexpr double TEST_PI = 3.14159265358979323846;

// ============================================================================
// 测试辅助
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
// 通过全局变量控制 density/area，便于不同测试场景调整
// ============================================================================

static double g_mock_density = 1.0;     // 颗/平方度
static double g_mock_area    = 1.0;     // 平方度
static std::atomic<int> g_mock_call_count{0};

static int mock_gaia_query(double ra, double dec, double radius_deg, double mag_lim)
{
    (void)ra; (void)dec; (void)radius_deg;
    g_mock_call_count++;
    // N = density × area × 10^(0.4×(m-10))
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

    StarSelectorParams params{};
    params.img_n_target             = 50;
    params.gaia_density_ratio       = 1.5;
    params.gaia_query_radius_factor = 0.55;
    params.m_lim_step               = 0.5;
    params.m_lim_max_iter           = 15;
    params.density_tolerance        = 0.1;
    params.focal_length_mm          = 200.0;
    params.pixel_size_um            = 3.76;
    params.img_width                = 4500;
    params.img_height               = 3600;
    params.center_ra                = 100.0;
    params.center_dec               = 30.0;
    params.n_img_bright             = 50;
    params.exposure_time_s          = 1.0;
    params.log_file_path            = nullptr;

    StarSelectionResult result{};
    // 不调用 ss_density_match，先单独验证 FOV 公式
    // s0 = 206.265 × 3.76 / 200 = 3.8778 "/px
    double s0_exp = 206.265 * 3.76 / 200.0;
    // FOV_diag = sqrt(4500² + 3600²) × s0 / 3600
    double diag_pix = std::sqrt(4500.0 * 4500.0 + 3600.0 * 3600.0);
    double fov_diag_exp = diag_pix * s0_exp / 3600.0;
    double query_r_exp = fov_diag_exp * 0.55;
    double img_area_exp = (4500.0 * s0_exp / 3600.0) * (3600.0 * s0_exp / 3600.0);
    double query_area_exp = TEST_PI * query_r_exp * query_r_exp;
    // n_target = max(50, round(1.5 × 50 × query_area / img_area))
    double n_target_dbl = 1.5 * 50.0 * (query_area_exp / img_area_exp);
    int n_target_exp = std::max(50, (int)std::lround(n_target_dbl));

    printf("    s0_exp=%.4f, fov_diag_exp=%.4f°, query_r_exp=%.4f°, "
           "img_area=%.4f, query_area=%.4f, n_target_exp=%d\n",
           s0_exp, fov_diag_exp, query_r_exp, img_area_exp, query_area_exp, n_target_exp);

    // 通过 ss_density_match 验证（mock 不收敛也无所谓，主要看 result 中的几何量）
    g_mock_density = 0.0; // 让查询返回 0，迭代会一直 +step 直到 max_iter，不影响几何量
    g_mock_area = query_area_exp;
    g_mock_call_count = 0;

    int ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_EQ(ret, 0, "ss_density_match 返回成功");

    // s0 验证
    ASSERT_NEAR(result.s0, s0_exp, 1e-3, "s0 像素尺度");

    // FOV 对角线验证
    ASSERT_NEAR(result.fov_diag_deg, fov_diag_exp, 1e-4, "FOV 对角线");

    // 查询半径验证 = FOV_diag × 0.55
    ASSERT_NEAR(result.query_radius_deg, query_r_exp, 1e-4, "查询半径");
    ASSERT_NEAR(result.query_radius_deg, result.fov_diag_deg * 0.55, 1e-6, "查询半径 = FOV×0.55");

    // 目标星数验证 ≥ 50
    ASSERT_TRUE(result.n_target >= 50, "n_target >= 50");
    ASSERT_EQ(result.n_target, n_target_exp, "n_target 数值正确");

    // 图像面积 / 查询面积
    ASSERT_NEAR(result.img_area_sqdeg, img_area_exp, 1e-6, "图像面积");
    ASSERT_NEAR(result.query_area_sqdeg, query_area_exp, 1e-6, "查询圆面积");

    // 密度
    double rho_img_exp = 50.0 / img_area_exp;
    ASSERT_NEAR(result.rho_img, rho_img_exp, 1e-6, "rho_img");
    ASSERT_NEAR(result.rho_target, 1.5 * rho_img_exp, 1e-6, "rho_target");
}

// ============================================================================
// 测试场景 2: Mock Gaia 查询收敛性
// 调整 density 使 m≈14 时 N≈n_target，验证迭代收敛
// ============================================================================

static void test_convergence()
{
    printf("\n[TEST] test_convergence: Mock Gaia 查询迭代收敛\n");

    StarSelectorParams params{};
    params.img_n_target             = 50;
    params.gaia_density_ratio       = 1.5;
    params.gaia_query_radius_factor = 0.55;
    params.m_lim_step               = 0.5;
    params.m_lim_max_iter           = 15;
    params.density_tolerance        = 0.1;
    params.focal_length_mm          = 200.0;
    params.pixel_size_um            = 3.76;
    params.img_width                = 4500;
    params.img_height               = 3600;
    params.center_ra                = 100.0;
    params.center_dec               = 30.0;
    params.n_img_bright             = 50;
    params.exposure_time_s          = 1.0;
    params.log_file_path            = nullptr;

    // 计算几何量
    double s0 = 206.265 * 3.76 / 200.0;
    double fov_diag = std::sqrt(4500.0 * 4500.0 + 3600.0 * 3600.0) * s0 / 3600.0;
    double query_r = fov_diag * 0.55;
    double query_area = TEST_PI * query_r * query_r;
    double img_area = (4500.0 * s0 / 3600.0) * (3600.0 * s0 / 3600.0);
    int n_target = std::max(50, (int)std::lround(1.5 * 50.0 * query_area / img_area));
    printf("    n_target=%d, query_area=%.4f°²\n", n_target, query_area);

    // 设 density 使 m=14 时 N=n_target:
    // n_target = density × query_area × 10^(0.4×(14-10)) = density × query_area × 39.81
    // → density = n_target / (query_area × 39.81)
    double m_target = 14.0;
    double factor_at_m = std::pow(10.0, 0.4 * (m_target - 10.0));
    double density = n_target / (query_area * factor_at_m);
    g_mock_density = density;
    g_mock_area = query_area;
    g_mock_call_count = 0;

    StarSelectionResult result{};
    int ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_EQ(ret, 0, "ss_density_match 返回成功");

    printf("    m_lim_final=%.4f, n_gaia_final=%d, iters=%d, converged=%d, n_target=%d\n",
           result.m_lim_final, result.n_gaia_final, result.m_lim_iterations,
           (int)result.converged, result.n_target);

    // 验证收敛
    ASSERT_TRUE(result.converged, "迭代收敛");

    // 验证最终 n_gaia 落在 [n_target×(1-tol), n_target×(1+tol)] 范围内
    double n_lo = n_target * (1.0 - 0.1);
    double n_hi = n_target * (1.0 + 0.1);
    ASSERT_TRUE(result.n_gaia_final >= (int)n_lo && result.n_gaia_final <= (int)n_hi,
                "n_gaia_final 在容差范围内");

    // 验证迭代次数 < max_iter
    ASSERT_TRUE(result.m_lim_iterations < params.m_lim_max_iter, "迭代次数 < max_iter");

    // 验证最终 m_lim 在 m_target 附近（步长 0.5，收敛点应在 ±0.5 内）
    ASSERT_NEAR(result.m_lim_final, m_target, 0.6, "m_lim_final 接近 m_target=14");
}

// ============================================================================
// 测试场景 3: 小视场 / 高密度（n_target=50 下限触发）
// f=1000mm, pix=3.76um, 2000×1500, n_img=20, ratio=1.5, qrf=0.55
// ============================================================================

static void test_small_fov()
{
    printf("\n[TEST] test_small_fov: 小视场 n_target=50 下限触发\n");

    StarSelectorParams params{};
    params.img_n_target             = 50;
    params.gaia_density_ratio       = 1.5;
    params.gaia_query_radius_factor = 0.55;
    params.m_lim_step               = 0.5;
    params.m_lim_max_iter           = 15;
    params.density_tolerance        = 0.1;
    params.focal_length_mm          = 1000.0;
    params.pixel_size_um            = 3.76;
    params.img_width                = 2000;
    params.img_height               = 1500;
    params.center_ra                = 200.0;
    params.center_dec               = -30.0;
    params.n_img_bright             = 10;  // 小值使 n_target_raw < 50 触发下限
    params.exposure_time_s          = 1.0;
    params.log_file_path            = nullptr;

    // 几何计算
    double s0 = 206.265 * 3.76 / 1000.0;  // ≈0.776 "/px
    double fov_diag = std::sqrt(2000.0 * 2000.0 + 1500.0 * 1500.0) * s0 / 3600.0;
    double query_r = fov_diag * 0.55;
    double query_area = TEST_PI * query_r * query_r;
    double img_area = (2000.0 * s0 / 3600.0) * (1500.0 * s0 / 3600.0);
    // n_target_raw = 1.5 × 10 × query_area/img_area (n_img=10 触发下限)
    double n_target_raw = 1.5 * 10.0 * (query_area / img_area);
    int n_target_exp = std::max(50, (int)std::lround(n_target_raw));
    printf("    s0=%.4f, fov_diag=%.4f°, query_area=%.5f°², img_area=%.5f°², "
           "n_target_raw=%.2f, n_target_exp=%d\n",
           s0, fov_diag, query_area, img_area, n_target_raw, n_target_exp);

    // 验证 n_target_raw < 50 → 触发下限
    ASSERT_TRUE(n_target_raw < 50.0, "n_target_raw < 50, 触发下限");

    // Mock 让 m=14 时 N=50
    double factor_at_m = std::pow(10.0, 0.4 * (14.0 - 10.0));
    double density = 50.0 / (query_area * factor_at_m);
    g_mock_density = density;
    g_mock_area = query_area;
    g_mock_call_count = 0;

    StarSelectionResult result{};
    int ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_EQ(ret, 0, "ss_density_match 返回成功");

    ASSERT_EQ(result.n_target, 50, "n_target 下限为 50");
    ASSERT_NEAR(result.s0, s0, 1e-3, "s0 小视场");
    ASSERT_NEAR(result.fov_diag_deg, fov_diag, 1e-4, "FOV 小视场");
    ASSERT_TRUE(result.converged, "小视场收敛");

    printf("    m_lim_final=%.4f, n_gaia_final=%d, iters=%d, converged=%d\n",
           result.m_lim_final, result.n_gaia_final, result.m_lim_iterations,
           (int)result.converged);
}

// ============================================================================
// 测试场景 4: 参数非法处理
// ============================================================================

static void test_invalid_params()
{
    printf("\n[TEST] test_invalid_params: 非法参数处理\n");

    StarSelectorParams params{};
    StarSelectionResult result{};

    // NULL 参数
    int ret = ss_density_match(nullptr, mock_gaia_query, &result);
    ASSERT_TRUE(ret < 0, "NULL params 返回错误");

    ret = ss_density_match(&params, mock_gaia_query, nullptr);
    ASSERT_TRUE(ret < 0, "NULL result 返回错误");

    // 非法 focal_length
    params.focal_length_mm = 0.0;
    params.img_width = 4500;
    params.img_height = 3600;
    params.n_img_bright = 50;
    ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_TRUE(ret < 0, "focal_length=0 返回错误");

    // 非法 img_width
    params.focal_length_mm = 200.0;
    params.img_width = 0;
    ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_TRUE(ret < 0, "img_width=0 返回错误");

    // 非法 n_img_bright
    params.img_width = 4500;
    params.n_img_bright = 0;
    ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_TRUE(ret < 0, "n_img_bright=0 返回错误");

    // NULL gaia_query 回调
    params.n_img_bright = 50;
    ret = ss_density_match(&params, nullptr, &result);
    ASSERT_TRUE(ret < 0, "NULL gaia_query 返回错误");
}

// ============================================================================
// 测试场景 5: 自适应步长验证（前4次 step_init, 后续 step_init/2）
// 通过设置 mock 使得前4次必然朝同一方向调整，验证步长切换
// ============================================================================

static void test_adaptive_step()
{
    printf("\n[TEST] test_adaptive_step: 自适应步长前4次/后续减半\n");

    StarSelectorParams params{};
    params.img_n_target             = 50;
    params.gaia_density_ratio       = 1.5;
    params.gaia_query_radius_factor = 0.55;
    params.m_lim_step               = 0.5;
    params.m_lim_max_iter           = 15;
    params.density_tolerance        = 0.05;  // 较紧容差，迫使更多迭代
    params.focal_length_mm          = 200.0;
    params.pixel_size_um            = 3.76;
    params.img_width                = 4500;
    params.img_height               = 3600;
    params.center_ra                = 100.0;
    params.center_dec               = 30.0;
    params.n_img_bright             = 50;
    params.exposure_time_s          = 1.0;
    params.log_file_path            = nullptr;

    // 让 mock 始终返回 0 → 持续 +step → 前4次 +0.5, 后续 +0.25
    g_mock_density = 0.0;
    g_mock_area = 1.0;
    g_mock_call_count = 0;

    StarSelectionResult result{};
    int ret = ss_density_match(&params, mock_gaia_query, &result);
    ASSERT_EQ(ret, 0, "ss_density_match 返回成功");

    // 应该走满 max_iter (未收敛时 iterations == max_iter)
    ASSERT_TRUE(!result.converged, "mock 始终返回 0, 不收敛");
    ASSERT_EQ(result.m_lim_iterations, params.m_lim_max_iter, "走满 max_iter 次");

    // m_cut 初始 = 6 + 1.5×log10(200) + 2×log10(1) = 6 + 1.5×2.301 = 9.452
    double m_cut_exp = 6.0 + 1.5 * std::log10(200.0) + 2.0 * std::log10(1.0);
    // 循环执行 max_iter 次: 前4次 +0.5, 后 (max_iter-4) 次 +0.25
    double m_final_exp = m_cut_exp + 4 * 0.5 + (params.m_lim_max_iter - 4) * 0.25;
    printf("    m_cut=%.4f, m_final=%.4f (exp=%.4f), iters=%d\n",
           m_cut_exp, result.m_lim_final, m_final_exp, result.m_lim_iterations);
    ASSERT_NEAR(result.m_lim_final, m_final_exp, 1e-6, "自适应步长累计正确");
}

// ============================================================================
// 主入口
// ============================================================================

int main()
{
    printf("=== V4.2 StarSelector 单元测试 ===\n");

    test_fov_basic();
    test_convergence();
    test_small_fov();
    test_invalid_params();
    test_adaptive_step();

    printf("\n=== 测试汇总 ===\n");
    printf("  PASS: %d\n", g_test_pass);
    printf("  FAIL: %d\n", g_test_fail);
    printf("  结果: %s\n", g_test_fail == 0 ? "ALL PASS" : "HAS FAILURES");

    return g_test_fail == 0 ? 0 : 1;
}

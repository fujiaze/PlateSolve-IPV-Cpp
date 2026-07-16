// ============================================================================
// test_solve.cpp - V4.3 入口 vm44_solve() 单元测试
//
// 测试范围 (不依赖外部 GaiaClient/StarDetector):
//   1. 参数边界检查 (NULL result/image_path, 无效 focal/pixel)
//   2. 依赖注入检查 (未注入 Gaia/StarDetector)
//   3. 资源管理 (vm44_free_result NULL/空/已分配)
//   4. 默认参数 (vm44_get_default_params 字段正确)
//   5. 句柄注入 (vm44_set_gaia_client / vm44_set_star_detector)
//
// 完整端到端测试需要真实 GaiaClient + StarDetector, 在 Python 端进行
// ============================================================================

#include "../include/vm44_api.h"
#include "../include/vm44_types.h"
#include <cstring>
#include <cstdio>
#include <string>

// ============================================================================
// 测试框架
// ============================================================================
static int g_test_count = 0;
static int g_test_pass = 0;
static int g_test_fail = 0;

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("  [FAIL] %s (line %d)\n", msg, __LINE__); \
            return -1; \
        } \
    } while(0)

#define RUN_TEST(test_fn) \
    do { \
        ++g_test_count; \
        printf("[%d] %s ... ", g_test_count, #test_fn); \
        int rc = test_fn(); \
        if (rc == 0) { \
            printf("PASS\n"); \
            ++g_test_pass; \
        } else { \
            printf("FAIL\n"); \
            ++g_test_fail; \
        } \
    } while(0)

// ============================================================================
// 测试 1: NULL result 参数
// ============================================================================
static int test_null_result() {
    int rc = vm44_solve("dummy.fits", 10.0, 20.0, 1000.0, 5.0, nullptr, nullptr);
    TEST_ASSERT(rc == -1, "NULL result 应返回 -1");
    return 0;
}

// ============================================================================
// 测试 2: NULL image_path
// ============================================================================
static int test_null_image_path() {
    v44::VM44SolveResult result;
    int rc = vm44_solve(nullptr, 10.0, 20.0, 1000.0, 5.0, nullptr, &result);
    TEST_ASSERT(rc == -1, "NULL image_path 应返回 -1");
    TEST_ASSERT(result.success == false, "success 应为 false");
    TEST_ASSERT(std::strlen(result.error_msg) > 0, "error_msg 应非空");
    printf("[msg=%s] ", result.error_msg);
    return 0;
}

// ============================================================================
// 测试 3: 无效 focal_length_mm (<=0)
// ============================================================================
static int test_invalid_focal_length() {
    v44::VM44SolveResult result;
    int rc = vm44_solve("dummy.fits", 10.0, 20.0, 0.0, 5.0, nullptr, &result);
    TEST_ASSERT(rc == -1, "focal_length=0 应返回 -1");
    TEST_ASSERT(result.success == false, "success 应为 false");
    TEST_ASSERT(std::strlen(result.error_msg) > 0, "error_msg 应非空");
    printf("[msg=%s] ", result.error_msg);
    return 0;
}

// ============================================================================
// 测试 4: 无效 pixel_size_um (<=0)
// ============================================================================
static int test_invalid_pixel_size() {
    v44::VM44SolveResult result;
    int rc = vm44_solve("dummy.fits", 10.0, 20.0, 1000.0, -1.0, nullptr, &result);
    TEST_ASSERT(rc == -1, "pixel_size<0 应返回 -1");
    TEST_ASSERT(result.success == false, "success 应为 false");
    TEST_ASSERT(std::strlen(result.error_msg) > 0, "error_msg 应非空");
    printf("[msg=%s] ", result.error_msg);
    return 0;
}

// ============================================================================
// 测试 5: 未注入 GaiaClient
//   前置: focal_length/pixel_size 合法
//   期望: 返回 -1, error_msg 含 "Gaia"
// ============================================================================
static int test_no_gaia_client() {
    // 确保未注入任何句柄
    vm44_set_gaia_client(nullptr);
    vm44_set_star_detector(nullptr);

    v44::VM44SolveResult result;
    int rc = vm44_solve("dummy.fits", 10.0, 20.0, 1000.0, 5.0, nullptr, &result);
    TEST_ASSERT(rc == -1, "未注入 Gaia 应返回 -1");
    TEST_ASSERT(result.success == false, "success 应为 false");
    // error_msg 应包含 "Gaia"
    std::string err(result.error_msg);
    bool has_gaia = (err.find("Gaia") != std::string::npos) ||
                    (err.find("gaia") != std::string::npos);
    TEST_ASSERT(has_gaia, "error_msg 应提及 Gaia");
    printf("[msg=%s] ", result.error_msg);
    return 0;
}

// ============================================================================
// 测试 6: 注入 Gaia 但未注入 StarDetector
// ============================================================================
static int test_no_star_detector() {
    // 注入 Gaia (用伪造的非 NULL 句柄)
    vm44_set_gaia_client((void*)0xDEADBEEF);
    vm44_set_star_detector(nullptr);

    v44::VM44SolveResult result;
    int rc = vm44_solve("dummy.fits", 10.0, 20.0, 1000.0, 5.0, nullptr, &result);
    TEST_ASSERT(rc == -1, "未注入 StarDetector 应返回 -1");
    TEST_ASSERT(result.success == false, "success 应为 false");
    // error_msg 应包含 "StarDetector" 或 "Detector"
    std::string err(result.error_msg);
    bool has_detector = (err.find("Detector") != std::string::npos) ||
                        (err.find("detector") != std::string::npos);
    TEST_ASSERT(has_detector, "error_msg 应提及 Detector");
    printf("[msg=%s] ", result.error_msg);

    // 清理: 重置句柄
    vm44_set_gaia_client(nullptr);
    return 0;
}

// ============================================================================
// 测试 7: vm44_get_default_params 字段正确
// ============================================================================
static int test_get_default_params() {
    v44::VM44SolveParams params;
    vm44_get_default_params(&params);

    // 基础参数
    TEST_ASSERT(params.n_modes == 4, "n_modes 默认应为 4");
    TEST_ASSERT(params.seed == 42, "seed 默认应为 42");

    // StarSelector
    TEST_ASSERT(params.img_n_target == 50, "img_n_target 默认应为 50");
    TEST_ASSERT(params.gaia_density_ratio == 1.5, "gaia_density_ratio 默认应为 1.5");
    TEST_ASSERT(params.gaia_query_radius_factor == 0.55, "gaia_query_radius_factor 默认应为 0.55");

    // VectorMatcher
    TEST_ASSERT(params.s_min == 0.9, "s_min 默认应为 0.9");
    TEST_ASSERT(params.s_max == 1.1, "s_max 默认应为 1.1");
    TEST_ASSERT(params.min_inliers == 5, "min_inliers 默认应为 5");

    // PairExpander
    TEST_ASSERT(params.region_size_px == 800, "region_size_px 默认应为 800");
    TEST_ASSERT(params.N_floor == 5, "N_floor 默认应为 5");
    TEST_ASSERT(params.N_cap == 30, "N_cap 默认应为 30");
    TEST_ASSERT(params.N_max == 1500, "N_max 默认应为 1500");

    // IRM 闭环参数
    TEST_ASSERT(params.irm_max_iter == 10, "irm_max_iter 默认应为 10");
    TEST_ASSERT(params.irm_converge_eps == 0.05, "irm_converge_eps 默认应为 0.05");
    TEST_ASSERT(params.irm_diverge_factor == 1.1, "irm_diverge_factor 默认应为 1.1");
    TEST_ASSERT(params.irm_tau_min == 2.0, "irm_tau_min 默认应为 2.0");
    TEST_ASSERT(params.irm_tau_factor == 3.0, "irm_tau_factor 默认应为 3.0");
    TEST_ASSERT(params.irm_lowe_ratio == 0.7, "irm_lowe_ratio 默认应为 0.7");
    TEST_ASSERT(params.irm_k_geometry == 8, "irm_k_geometry 默认应为 8");
    TEST_ASSERT(params.irm_geom_threshold == 4, "irm_geom_threshold 默认应为 4");
    TEST_ASSERT(params.irm_ransac_max_iter == 200, "irm_ransac_max_iter 默认应为 200");
    TEST_ASSERT(params.irm_ransac_min_inliers == 10, "irm_ransac_min_inliers 默认应为 10");
    TEST_ASSERT(params.irm_huber_delta_factor == 1.345, "irm_huber_delta_factor 默认应为 1.345");
    TEST_ASSERT(params.irm_sip_min_pairs == 30, "irm_sip_min_pairs 默认应为 30");

    // WcsFitter
    TEST_ASSERT(params.sip_max_order == 4, "sip_max_order 默认应为 4");

    // 日志目录默认 NULL
    TEST_ASSERT(params.log_dir == nullptr, "log_dir 默认应为 NULL");

    printf("[IRM params OK] ");
    return 0;
}

// ============================================================================
// 测试 8: vm44_free_result(NULL) 安全
// ============================================================================
static int test_free_null_result() {
    vm44_free_result(nullptr);  // 不应崩溃
    return 0;
}

// ============================================================================
// 测试 9: vm44_free_result 对未分配 cu/cw 的 result 安全
// ============================================================================
static int test_free_empty_result() {
    v44::VM44SolveResult result;
    std::memset(&result, 0, sizeof(result));
    result.cu = nullptr;
    result.cw = nullptr;
    result.n_pairs = 0;

    vm44_free_result(&result);  // 不应崩溃

    TEST_ASSERT(result.cu == nullptr, "cu 应仍为 nullptr");
    TEST_ASSERT(result.cw == nullptr, "cw 应仍为 nullptr");
    TEST_ASSERT(result.n_pairs == 0, "n_pairs 应仍为 0");
    return 0;
}

// ============================================================================
// 测试 10: vm44_free_result 对已分配 cu/cw 的 result 正确释放
// ============================================================================
static int test_free_allocated_result() {
    v44::VM44SolveResult result;
    std::memset(&result, 0, sizeof(result));

    // 模拟内部分配 (与 vm44_entry.cpp 中相同的方式)
    result.n_pairs = 3;
    result.cu = new int[3];
    result.cw = new int[3];
    result.cu[0] = 10; result.cu[1] = 20; result.cu[2] = 30;
    result.cw[0] = 100; result.cw[1] = 200; result.cw[2] = 300;

    vm44_free_result(&result);

    TEST_ASSERT(result.cu == nullptr, "释放后 cu 应为 nullptr");
    TEST_ASSERT(result.cw == nullptr, "释放后 cw 应为 nullptr");
    TEST_ASSERT(result.n_pairs == 0, "释放后 n_pairs 应为 0");
    return 0;
}

// ============================================================================
// 测试 11: vm44_free_result 可安全重复调用
// ============================================================================
static int test_free_idempotent() {
    v44::VM44SolveResult result;
    std::memset(&result, 0, sizeof(result));
    result.n_pairs = 2;
    result.cu = new int[2];
    result.cw = new int[2];

    vm44_free_result(&result);
    vm44_free_result(&result);  // 第二次调用不应崩溃
    vm44_free_result(&result);  // 第三次也不应崩溃

    return 0;
}

// ============================================================================
// 测试 12: 句柄注入 (伪造句柄, 通过错误信息变化间接验证)
// ============================================================================
static int test_set_handles() {
    // 完全重置
    vm44_set_gaia_client(nullptr);
    vm44_set_star_detector(nullptr);

    // 仅注入 Gaia, 不注入 StarDetector → 错误应跳过 Gaia 检查
    vm44_set_gaia_client((void*)0xDEADBEEF);

    v44::VM44SolveResult result;
    int rc = vm44_solve("dummy.fits", 10.0, 20.0, 1000.0, 5.0, nullptr, &result);
    TEST_ASSERT(rc == -1, "未注入 StarDetector 应返回 -1");

    std::string err(result.error_msg);
    // 应不包含 "Gaia", 应包含 "Detector"
    bool no_gaia_err = (err.find("Gaia") == std::string::npos) &&
                       (err.find("gaia") == std::string::npos);
    TEST_ASSERT(no_gaia_err, "已注入 Gaia 后错误信息不应再提及 Gaia");
    bool has_detector = (err.find("Detector") != std::string::npos);
    TEST_ASSERT(has_detector, "应提及 Detector");
    printf("[msg=%s] ", result.error_msg);

    // 清理
    vm44_set_gaia_client(nullptr);
    return 0;
}

// ============================================================================
// 主函数
// ============================================================================
int main() {
    printf("=== V4.3 vm44_solve() 入口测试 ===\n\n");

    RUN_TEST(test_null_result);
    RUN_TEST(test_null_image_path);
    RUN_TEST(test_invalid_focal_length);
    RUN_TEST(test_invalid_pixel_size);
    RUN_TEST(test_no_gaia_client);
    RUN_TEST(test_no_star_detector);
    RUN_TEST(test_get_default_params);
    RUN_TEST(test_free_null_result);
    RUN_TEST(test_free_empty_result);
    RUN_TEST(test_free_allocated_result);
    RUN_TEST(test_free_idempotent);
    RUN_TEST(test_set_handles);

    printf("\n=== 测试汇总: %d/%d 通过, %d 失败 ===\n",
           g_test_pass, g_test_count, g_test_fail);
    return g_test_fail == 0 ? 0 : 1;
}

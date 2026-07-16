// ============================================================================
// vm45_entry.cpp - V4.5 单一入口 vm45_solve() 实现
//
// 对外暴露的 C 接口, 内部串联:
//   Phase 0: StarSelector (vm45_select) → Phase A: RelVec θ 求解 (vm45_relvec_match)
//
// **仅实现 θ 求解**, 跳过 Phase B (tx/ty 搜索), IRM 闭环, WcsFitter
// 不计算 CD 矩阵、SIP 系数、control_points, 不应用 flip
// ============================================================================

#include "vm45_api.h"
#include "vm45_internal.h"
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// 全局依赖句柄 (由 vm45_set_*_client 注入)
static void* g_gaia_client_handle = nullptr;
static void* g_star_detector_handle = nullptr;

namespace v45 {

// 内部访问器 (供 vm45_select / vm45_relvec_match 使用)
// 声明在 vm45_internal.h, 此处定义
// (放在 namespace 内, 与 V4.4 风格一致; 但 V4.4 是文件末尾定义, 此处前置以保持单文件可读)

} // namespace v45

extern "C" {

// ============================================================================
// vm45_set_gaia_client - 注入 Gaia 客户端句柄
// ============================================================================
VM45_API void vm45_set_gaia_client(void* client) {
    g_gaia_client_handle = client;
}

// ============================================================================
// vm45_set_star_detector - 注入 StarDetector 句柄
// ============================================================================
VM45_API void vm45_set_star_detector(void* detector) {
    g_star_detector_handle = detector;
}

// ============================================================================
// vm45_get_default_params - 获取默认参数
//
// 默认值依据:
//   - vm45_types.h 中各字段注释标注的默认值
//   - v4_4_relvec_sampling_design.md 设计文档 (K_total=20000, sigma_d=3.0, n_third=5)
//   - V4.4 vm44_get_default_params 中 Phase 0 / 自适应采样参数
// ============================================================================
VM45_API void vm45_get_default_params(v45::VM45SolveParams* params) {
    if (!params) return;
    std::memset(params, 0, sizeof(*params));

    // --- 基础参数 ---
    params->seed = 42;

    // --- StarSelector 参数 (Phase 0, 与 V4.4 一致) ---
    params->img_n_target              = 50;     // 图像侧目标星数
    params->gaia_density_ratio        = 1.5;    // Gaia 密度比
    params->gaia_query_radius_factor  = 0.55;   // Gaia 查询半径因子
    params->m_lim_step                = 0.5;    // 极限星等步长
    params->m_lim_max_iter            = 10;     // 极限星等最大迭代
    params->density_tolerance         = 0.1;    // 密度容差

    // --- 相对向量法参数 (Phase A, 严格按设计文档) ---
    params->K_total                   = 20000;  // 总采样次数 (设计文档值)
    params->sigma_d_px                = 2.0;    // 距离容差 σ_d (像素, 内部乘 s0 转角秒)
    params->n_third                   = 0;      // 0=用全部可用第三星 (提升SNR)
    params->third_ratio_min           = 0.05;   // 第三星通过比例阈值 (≥3颗/48颗, 兼顾噪声)
    params->theta_bw                  = 1.0;    // θ 直方图 bin 宽度 (度)
    params->snr_threshold             = 5.0;    // SNR 接受阈值
    params->relvec_max_u              = 100;    // U 组限流上限
    params->relvec_max_cand           = 500;    // 单次采样候选对上限
    params->relvec_min_len_frac       = 0.05;   // 最小星对距离比例
    params->relvec_max_len_frac       = 0.8;    // 最大星对距离比例

    // --- 自适应采样停止 (从 V4.4 沿用) ---
    params->adaptive_stop             = 1;      // 启用自适应停止
    params->min_samples               = 200;    // 最少采样次数
    params->check_interval            = 100;    // SNR 检查间隔
    params->snr_eps                   = 0.05;   // SNR 相对变化阈值
    params->max_stable                = 3;      // 连续稳定次数

    // --- 日志 ---
    params->log_dir                   = nullptr;
}

// ============================================================================
// vm45_solve - 一键求解入口 (仅 Phase 0 + Phase A)
// ============================================================================
VM45_API int vm45_solve(
    const char* image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const v45::VM45SolveParams* params,
    v45::VM45SolveResult* result)
{
    // --- 结果对象初始化 ---
    if (!result) return -1;
    std::memset(result, 0, sizeof(*result));
    result->success = false;

    // --- 获取参数 (NULL 用默认值) ---
    v45::VM45SolveParams p;
    if (params) {
        p = *params;
    } else {
        vm45_get_default_params(&p);
    }

    // --- 初始化日志 ---
    v45::Logger logger;
    std::string log_path;
    if (p.log_dir) {
        log_path = std::string(p.log_dir) + "/vm45_solve.log";
        logger.init(log_path);
    }
    logger.infof("=== vm45_solve 开始 (image=%s, RA=%.6f, Dec=%.6f) ===",
                 image_path ? image_path : "(null)", ra, dec);
    logger.infof("焦距=%.3f mm, 像元=%.3f um, K_total=%d, sigma_d_px=%.2f, n_third=%d, ratio_min=%.2f",
                 focal_length_mm, pixel_size_um, p.K_total, p.sigma_d_px, p.n_third, p.third_ratio_min);

    // --- 边界检查 ---
    if (!image_path) {
        std::strncpy(result->error_msg, "image_path 为空", sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: image_path 为空");
        return -1;
    }
    if (focal_length_mm <= 0 || pixel_size_um <= 0) {
        std::strncpy(result->error_msg, "focal_length_mm 或 pixel_size_um 无效",
                     sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: focal_length_mm 或 pixel_size_um 无效");
        return -1;
    }
    if (!g_gaia_client_handle) {
        std::strncpy(result->error_msg, "Gaia 客户端未注入",
                     sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: Gaia 客户端未注入");
        return -1;
    }
    if (!g_star_detector_handle) {
        std::strncpy(result->error_msg, "StarDetector 未注入",
                     sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: StarDetector 未注入");
        return -1;
    }

    // --- Phase 0: StarSelector ---
    logger.info("Phase 0: StarSelector 开始");
    v45::StarSelection selection;
    int rc = v45::vm45_select(std::string(image_path), ra, dec,
                               focal_length_mm, pixel_size_um,
                               p, selection, &logger);
    if (rc != 0 || !selection.success) {
        std::strncpy(result->error_msg, "Phase 0 StarSelector 失败",
                     sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: Phase 0 失败");
        return -1;
    }
    logger.infof("Phase 0 完成: N_img=%d, N_gaia=%d, fov=%.3f°, m_lim=%.2f, s0=%.4f",
                 (int)selection.U.size(), (int)selection.W.size(),
                 selection.fov_diag_deg, selection.m_lim_final, selection.s0);

    // --- s0 直接使用 selection.s0 (vm45_select 内部已计算) ---
    // 公式参考: s0 = 206.264806247 * pixel_size_um / focal_length_mm
    const double s0 = selection.s0;
    if (s0 <= 0) {
        std::strncpy(result->error_msg, "selection.s0 无效 (<=0)",
                     sizeof(result->error_msg) - 1);
        logger.errorf("vm45_solve: selection.s0 无效 (=%.6f)", s0);
        return -1;
    }

    // --- Phase A: 相对向量法 θ 求解 ---
    logger.info("Phase A: RelVec θ 求解开始");
    v45::RelVecResult relvec_result;
    rc = v45::vm45_relvec_match(selection.U, selection.W, s0, p, relvec_result, &logger);
    if (rc != 0) {
        std::strncpy(result->error_msg, "Phase A RelVec θ 求解失败",
                     sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: Phase A 失败");
        return -1;
    }
    logger.infof("Phase A 完成: θ_peak=%.4f°, SNR=%.3f, n_passed=%d, n_samples=%d",
                 relvec_result.theta_peak_deg, relvec_result.theta_snr,
                 relvec_result.n_passed, relvec_result.n_samples);

    // --- 填充 VM45SolveResult ---
    // θ 求解结果
    result->theta_peak_deg = relvec_result.theta_peak_deg;
    result->theta_snr      = relvec_result.theta_snr;
    result->peak_bin       = relvec_result.peak_bin;

    // θ 直方图: 分配 360 个 double, 从 relvec_result.smoothed_votes 拷贝
    // (VM45SolveResult.theta_histogram 字段注释: "高斯平滑后的投票")
    // 用 malloc 分配 (vm45_free_result 会 free)
    result->histogram_size = 360;
    result->theta_histogram = (double*)std::malloc(sizeof(double) * 360);
    if (!result->theta_histogram) {
        std::strncpy(result->error_msg, "theta_histogram 内存分配失败",
                     sizeof(result->error_msg) - 1);
        logger.error("vm45_solve: theta_histogram malloc 失败");
        return -1;
    }
    // 从 smoothed_votes 拷贝 (容量可能 < 360, 缺失部分填 0)
    const int n_smooth = (int)relvec_result.smoothed_votes.size();
    for (int i = 0; i < 360; ++i) {
        result->theta_histogram[i] = (i < n_smooth) ? relvec_result.smoothed_votes[i] : 0.0;
    }

    // 通过候选数 / 采样数
    result->n_passed  = relvec_result.n_passed;
    result->n_samples = relvec_result.n_samples;

    // 元数据 (从 selection 复制)
    result->img_width     = selection.img_width;
    result->img_height    = selection.img_height;
    result->fov_diag_deg  = selection.fov_diag_deg;
    result->m_lim_final   = selection.m_lim_final;
    result->n_gaia_final  = selection.n_gaia_final;
    result->s0            = selection.s0;

    // 状态
    result->success = relvec_result.success;

    // 错误信息 (无错误时保持空)
    if (!result->success) {
        std::snprintf(result->error_msg, sizeof(result->error_msg),
                      "SNR=%.3f 低于阈值 %.2f",
                      relvec_result.theta_snr, p.snr_threshold);
        logger.warnf("θ 求解未达 SNR 阈值: SNR=%.3f < %.2f",
                     relvec_result.theta_snr, p.snr_threshold);
    }

    logger.infof("=== vm45_solve 完成: θ=%.4f°, SNR=%.3f, success=%d ===",
                 result->theta_peak_deg, result->theta_snr,
                 result->success ? 1 : 0);
    return 0;
}

// ============================================================================
// vm45_free_result - 释放结果内存
// ============================================================================
VM45_API void vm45_free_result(v45::VM45SolveResult* result) {
    if (!result) return;
    if (result->theta_histogram) {
        std::free(result->theta_histogram);
        result->theta_histogram = nullptr;
    }
    result->histogram_size = 0;
}

} // extern "C"

// ============================================================================
// 内部访问器定义 (供 vm45_select / vm45_relvec_match 调用)
// 声明在 vm45_internal.h 的 namespace v45 中
// ============================================================================
namespace v45 {

void* get_gaia_client_handle() { return g_gaia_client_handle; }
void* get_star_detector_handle() { return g_star_detector_handle; }

} // namespace v45

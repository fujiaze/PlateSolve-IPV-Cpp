// ============================================================================
// vm43_entry.cpp - V4.3 单一入口 vm43_solve() 实现
//
// 对外暴露的 C 接口, 内部串联:
//   StarSelector → VectorMatcher → IRM 闭环 (Expand ↔ Verify ↔ Fit)
//
// 流程:
//   1. 获取参数 (params 为 NULL 时用默认值)
//   2. Phase 0: vm43_select(image_path, ...) → StarSelection (U + W + 元数据)
//   3. Phase A+B: vm43_match(U, W, s0, ...) → VectorMatchResult (transform + C₀ + best_mode)
//   4. 应用 flip (best_mode) 到 W
//   5. 从 SimTransform 推导 CD₀
//   6. IRM 闭环: vm43_irm_refine(C0, CD0, ...) → 最终 CD + SIP + control_points
//   7. 填充 VM43SolveResult
//
// Task 9 实现
// ============================================================================

#include "vm43_api.h"
#include "vm43_internal.h"
#include <cstring>
#include <cmath>
#include <string>

// 全局依赖句柄 (由 vm43_set_*_client 注入)
static void* g_gaia_client_handle = nullptr;
static void* g_star_detector_handle = nullptr;

namespace v43 {

// ============================================================================
// 辅助: 从 SimTransform 推导 CD 矩阵
// 公式 (与 vm43_fit Layer 0 一致):
//   s3600 = s0 / (s × 3600)   (角秒→度)
//   CD = [s3600·ct, -s3600·st, -s3600·st, -s3600·ct]
// ============================================================================
static CDMatrix sim_to_cd(const SimTransform& sim, double s0) {
    CDMatrix cd;
    double s3600 = s0 / (sim.s * 3600.0);
    double ct = std::cos(sim.theta), st = std::sin(sim.theta);
    cd.cd11 =  s3600 * ct;
    cd.cd12 = -s3600 * st;
    cd.cd21 = -s3600 * st;
    cd.cd22 = -s3600 * ct;
    return cd;
}

// ============================================================================
// 辅助: 应用 flip 到 W
// mode: 0=无, 1=flip X, 2=flip Y, 3=flip XY
// ============================================================================
static void apply_flip_to_w(std::vector<StarPoint>& W, int mode) {
    bool fx = (mode == 1 || mode == 3);
    bool fy = (mode == 2 || mode == 3);
    if (!fx && !fy) return;
    for (auto& w : W) {
        if (fx) w.x = -w.x;
        if (fy) w.y = -w.y;
    }
}

} // namespace v43

extern "C" {

// ============================================================================
// vm43_solve - 一键求解入口
// ============================================================================
VM43_API int vm43_solve(
    const char* image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const v43::VM43SolveParams* params,
    v43::VM43SolveResult* result)
{
    if (!result) return -1;
    std::memset(result, 0, sizeof(*result));
    result->success = false;

    // --- 获取参数 (NULL 用默认值) ---
    v43::VM43SolveParams p;
    if (params) {
        p = *params;
    } else {
        vm43_get_default_params(&p);
    }

    // --- 初始化日志 ---
    v43::Logger logger;
    std::string log_path;
    if (p.log_dir) {
        log_path = std::string(p.log_dir) + "/vm43_solve.log";
        logger.init(log_path);
    }
    logger.infof("=== vm43_solve 开始 (image=%s, RA=%.6f, Dec=%.6f) ===",
                 image_path ? image_path : "(null)", ra, dec);

    // --- 边界检查 ---
    if (!image_path) {
        std::strncpy(result->error_msg, "image_path 为空", sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: image_path 为空");
        return -1;
    }
    if (focal_length_mm <= 0 || pixel_size_um <= 0) {
        std::strncpy(result->error_msg, "focal_length_mm 或 pixel_size_um 无效",
                     sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: focal_length_mm 或 pixel_size_um 无效");
        return -1;
    }
    if (!g_gaia_client_handle) {
        std::strncpy(result->error_msg, "Gaia 客户端未注入",
                     sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: Gaia 客户端未注入");
        return -1;
    }
    if (!g_star_detector_handle) {
        std::strncpy(result->error_msg, "StarDetector 未注入",
                     sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: StarDetector 未注入");
        return -1;
    }

    // --- Phase 0: StarSelector ---
    logger.info("Phase 0: StarSelector 开始");
    v43::StarSelection selection;
    int rc = v43::vm43_select(image_path, ra, dec, focal_length_mm, pixel_size_um,
                               p, selection, &logger);
    if (rc != 0 || !selection.success) {
        std::strncpy(result->error_msg, "Phase 0 StarSelector 失败",
                     sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: Phase 0 失败");
        return -1;
    }
    logger.infof("Phase 0 完成: N_img=%d, M=%d, fov=%.3f°, m_lim=%.2f",
                 (int)selection.U.size(), (int)selection.W.size(),
                 selection.fov_diag_deg, selection.m_lim_final);

    // --- Phase A+B: VectorMatcher ---
    logger.info("Phase A+B: VectorMatcher 开始");
    double s0 = 206.264806247 * pixel_size_um / focal_length_mm;
    v43::VectorMatchResult vm_result;
    rc = v43::vm43_match(selection.U, selection.W, s0, p, vm_result, &logger);
    if (rc != 0 || !vm_result.success) {
        std::strncpy(result->error_msg, "Phase A+B VectorMatcher 失败",
                     sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: Phase A+B 失败");
        return -1;
    }
    logger.infof("Phase A+B 完成: s=%.4f, θ=%.4f°, N_pairs=%d, mode=%d, RMS=%.3f\"",
                 vm_result.transform.s, vm_result.transform.theta * 57.295779513082323,
                 (int)vm_result.pairs.size(), vm_result.best_mode, vm_result.rms);

    // --- 应用 flip 到 W ---
    std::vector<v43::StarPoint> W_flipped = selection.W;
    v43::apply_flip_to_w(W_flipped, vm_result.best_mode);
    if (vm_result.best_mode != 0) {
        logger.infof("应用 flip mode=%d 到 W", vm_result.best_mode);
    }

    // --- 从 SimTransform 推导 CD₀ ---
    v43::CDMatrix CD0 = v43::sim_to_cd(vm_result.transform, s0);
    logger.infof("CD0 = [%.8e, %.8e; %.8e, %.8e]",
                 CD0.cd11, CD0.cd12, CD0.cd21, CD0.cd22);

    // --- IRM 闭环 ---
    logger.info("IRM 闭环开始");
    v43::CDMatrix final_cd;
    v43::SIPCoeffs final_sip;
    std::vector<v43::MatchPair> final_control_points;
    v43::SRobustResult final_s_robust;
    int n_iters = 0;
    bool converged = false;
    double final_bayes_lnK = 0.0;
    double final_triangle_pass_ratio = 0.0;

    rc = v43::vm43_irm_refine(
        vm_result.pairs, CD0,
        selection.U, W_flipped,
        s0, focal_length_mm, pixel_size_um,
        ra, dec,
        selection.img_width, selection.img_height,
        selection.fov_diag_deg,
        p,
        final_cd, final_sip, final_control_points, final_s_robust,
        n_iters, converged,
        final_bayes_lnK, final_triangle_pass_ratio,
        &logger);
    if (rc != 0) {
        std::strncpy(result->error_msg, "IRM 闭环失败",
                     sizeof(result->error_msg) - 1);
        logger.error("vm43_solve: IRM 闭环失败");
        return -1;
    }
    logger.infof("IRM 完成: iter=%d, converged=%d, N=%d, S_robust=%.4f\"",
                 n_iters, converged ? 1 : 0,
                 (int)final_control_points.size(), final_s_robust.s_robust);

    // --- 填充 VM43SolveResult ---
    result->cd[0] = final_cd.cd11;
    result->cd[1] = final_cd.cd12;
    result->cd[2] = final_cd.cd21;
    result->cd[3] = final_cd.cd22;
    result->crval[0] = ra;
    result->crval[1] = dec;
    result->crpix[0] = selection.img_width / 2.0 + 1.0;   // 1-based
    result->crpix[1] = selection.img_height / 2.0 + 1.0;  // 1-based
    std::memcpy(result->sip_A, final_sip.A, sizeof(result->sip_A));
    std::memcpy(result->sip_B, final_sip.B, sizeof(result->sip_B));
    result->sip_order = final_sip.order;

    // 精度指标
    result->s_robust = final_s_robust.s_robust;
    result->n_inliers = final_s_robust.n_inliers;
    result->n_iters = n_iters;
    result->irm_converged = converged;

    // 变换参数
    result->scale_arcsec_px = s0;
    result->rotation_deg = vm_result.transform.theta * 57.295779513082323;
    result->flip_mode = vm_result.best_mode;
    result->center_ra = ra;
    result->center_dec = dec;
    result->s0 = s0;
    result->s = vm_result.transform.s;
    result->theta = vm_result.transform.theta;
    result->tx = vm_result.transform.tx;
    result->ty = vm_result.transform.ty;

    // 调试信息
    result->theta_snr = vm_result.theta_snr;
    result->theta_peak_deg = vm_result.theta_peak_deg;
    result->bayes_lnK = final_bayes_lnK;
    result->triangle_pass_ratio = final_triangle_pass_ratio;
    result->best_mode = vm_result.best_mode;

    // 匹配对索引 (内部分配, 由 vm43_free_result 释放)
    int n_pairs = (int)final_control_points.size();
    result->n_pairs = n_pairs;
    if (n_pairs > 0) {
        result->cu = new int[n_pairs];
        result->cw = new int[n_pairs];
        for (int i = 0; i < n_pairs; ++i) {
            result->cu[i] = final_control_points[i].u;
            result->cw[i] = final_control_points[i].w;
        }
    }
    result->matched_count = n_pairs;

    // 元数据
    result->img_width = selection.img_width;
    result->img_height = selection.img_height;
    result->fov_diag_deg = selection.fov_diag_deg;
    result->m_lim_final = selection.m_lim_final;
    result->n_gaia_final = selection.n_gaia_final;

    // RMS (用 S_robust 作为 RMS)
    result->rms_arcsec = final_s_robust.s_robust;
    result->rms_px = final_s_robust.s_robust / s0;

    result->success = true;
    logger.infof("=== vm43_solve 完成: N=%d, S_robust=%.4f\", RMS=%.4f px ===",
                 n_pairs, result->s_robust, result->rms_px);
    return 0;
}

VM43_API void vm43_set_gaia_client(void* gaia_client_handle)
{
    g_gaia_client_handle = gaia_client_handle;
}

VM43_API void vm43_set_star_detector(void* detector_handle)
{
    g_star_detector_handle = detector_handle;
}

VM43_API void vm43_free_result(v43::VM43SolveResult* result)
{
    if (!result) return;
    if (result->cu) {
        delete[] result->cu;
        result->cu = nullptr;
    }
    if (result->cw) {
        delete[] result->cw;
        result->cw = nullptr;
    }
    result->n_pairs = 0;
}

VM43_API void vm43_get_default_params(v43::VM43SolveParams* params)
{
    if (!params) return;
    std::memset(params, 0, sizeof(*params));

    // 基础参数
    params->n_modes = 4;
    params->seed = 42;

    // StarSelector
    params->img_n_target = 50;
    params->gaia_density_ratio = 1.5;
    params->gaia_query_radius_factor = 0.55;
    params->m_lim_step = 0.5;
    params->m_lim_max_iter = 10;
    params->density_tolerance = 0.1;

    // VectorMatcher
    params->s_min = 0.9;
    params->s_max = 1.1;
    params->K_total = 10000;
    params->batch_size = 500;
    params->min_samples = 50;
    params->K_top = 100;
    params->min_inliers = 5;
    params->w_snr = 0.4;
    params->w_sparse = 0.4;
    params->w_sat = 0.2;
    params->prosac_T_max = 10000;
    params->use_prosac = 1;

    // PairExpander
    params->region_size_px = 800;
    params->N_floor = 5;
    params->N_cap = 30;
    params->N_max = 1500;

    // PairVerifier
    params->mad_iters = 3;
    params->mad_threshold_factor = 3.0;
    params->mad_min_threshold_arcsec = 5.0;
    params->lnK_accept = 10.0;
    params->lnK_weak = 3.0;
    params->eps_A = 0.1;
    params->eps_J = 0.1;
    params->triangle_pass_rate = 0.7;

    // WcsFitter
    params->sip_max_order = 4;
    params->skip_sip = 0;

    // IRM 闭环参数 (V4.3 新增)
    params->irm_max_iter = 10;
    params->irm_converge_eps = 0.05;
    params->irm_diverge_factor = 1.1;
    params->irm_tau_min = 2.0;
    params->irm_tau_factor = 3.0;
    params->irm_lowe_ratio = 0.7;
    params->irm_k_geometry = 8;
    params->irm_geom_threshold = 4;
    params->irm_geom_dist_tol = 3.0;
    params->irm_ransac_max_iter = 200;
    params->irm_ransac_min_inliers = 10;
    params->irm_huber_delta_factor = 1.345;
    params->irm_sip_min_pairs = 30;
    params->irm_s_initial = 0;

    // 日志
    params->log_dir = nullptr;
}

} // extern "C"

// 全局句柄访问器 (内部模块使用)
namespace v43 {

void* get_gaia_client_handle() { return g_gaia_client_handle; }
void* get_star_detector_handle() { return g_star_detector_handle; }

} // namespace v43

// ============================================================================
// vm44_irm.cpp - V4.3 IRM 闭环主循环
//
// 职责: IRM 闭环迭代精化
//   扩增匹配 → 几何过滤 → 验证 → 模型精化 → 收敛判定 → 再扩增
// 借鉴 ICP (Besl & McKay 1992) 单调收敛保证
//
// 算法流程 (参考 lib/plate_solve/docs/iterative_model_refinement.md):
//   初始化: control_points = C0, CD = CD0, SIP = None
//   初始 S_robust: 从 C0 + CD0 计算 (或用 s0 × 0.1 兜底)
//   循环 iter = 1..max_iter:
//     Step 1: vm44_expand(U, W, CD, SIP, S_robust, ...) → 候选匹配对
//     Step 2: vm44_geometry_filter(candidates, ...) → 几何过滤后对
//     Step 3: vm44_verify(filtered, ...) → RANSAC + 贝叶斯 + 三角形 → 内点集
//     Step 4: 合并 C0 ∪ inliers → control_points (去重)
//     Step 5: vm44_fit(control_points, ...) → CD + SIP + RMS
//     Step 6: S_robust_new = vm44_compute_s_robust(...)
//     收敛判定:
//       |S_robust_new - S_robust| < converge_eps → 收敛
//       S_robust_new > S_robust × diverge_factor → 防过拟合 (回退)
//       iter ≥ max_iter → 安全上限
//
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "vm44_internal.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>

namespace v44 {

// ============================================================================
// 辅助: 合并 C0 与 inliers, 去重
// ============================================================================
static std::vector<MatchPair> merge_pairs(
    const std::vector<MatchPair>& C0,
    const std::vector<MatchPair>& inliers)
{
    // 用 set 去重 (u, w) 对
    std::set<std::pair<int, int>> seen;
    std::vector<MatchPair> merged;
    merged.reserve(C0.size() + inliers.size());

    // 先加 C0 (高置信度, 不可被移除)
    for (const auto& p : C0) {
        auto key = std::make_pair(p.u, p.w);
        if (seen.insert(key).second) {
            merged.push_back(p);
        }
    }
    // 再加 inliers
    for (const auto& p : inliers) {
        auto key = std::make_pair(p.u, p.w);
        if (seen.insert(key).second) {
            merged.push_back(p);
        }
    }
    return merged;
}

// ============================================================================
// 辅助: 创建空 SIP (order=0, 全零)
// ============================================================================
static SIPCoeffs make_empty_sip() {
    SIPCoeffs sip;
    std::memset(sip.A, 0, sizeof(sip.A));
    std::memset(sip.B, 0, sizeof(sip.B));
    sip.order = 0;
    return sip;
}

// ============================================================================
// vm44_irm_refine - IRM 闭环主循环
// ============================================================================
int vm44_irm_refine(
    const std::vector<MatchPair>& C0,
    const CDMatrix& CD0,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    double focal_length_mm,
    double pixel_size_um,
    double ra0, double dec0,
    int img_width, int img_height,
    double fov_diag_deg,
    const VM44SolveParams& params,
    CDMatrix& final_cd,
    SIPCoeffs& final_sip,
    std::vector<MatchPair>& final_control_points,
    SRobustResult& final_s_robust,
    int& n_iters,
    bool& converged,
    double& final_bayes_lnK,
    double& final_triangle_pass_ratio,
    Logger* logger)
{
    // --- 初始化输出 ---
    final_cd = CD0;
    final_sip = make_empty_sip();
    final_control_points.clear();
    std::memset(&final_s_robust, 0, sizeof(final_s_robust));
    n_iters = 0;
    converged = false;
    final_bayes_lnK = 0.0;
    final_triangle_pass_ratio = 0.0;

    // --- 边界检查 ---
    if (C0.empty()) {
        if (logger) logger->warn("vm44_irm: C0 为空, 无法启动 IRM");
        return -1;
    }
    if (U.empty() || W.empty()) {
        if (logger) logger->warn("vm44_irm: U 或 W 为空");
        return -1;
    }
    if (focal_length_mm <= 0 || pixel_size_um <= 0) {
        if (logger) logger->warn("vm44_irm: focal_length_mm 或 pixel_size_um 无效");
        return -1;
    }

    int M0 = (int)C0.size();
    int max_iter = params.irm_max_iter;
    if (max_iter < 1) max_iter = 10;
    double converge_eps = params.irm_converge_eps;
    if (converge_eps <= 0) converge_eps = 0.05;
    double diverge_factor = params.irm_diverge_factor;
    if (diverge_factor <= 1.0) diverge_factor = 1.1;

    if (logger) logger->infof("=== vm44_irm 开始 (M0=%d, max_iter=%d, s0=%.4f\") ===",
                               M0, max_iter, s0);

    // --- 初始化状态 ---
    CDMatrix CD = CD0;
    SIPCoeffs SIP = make_empty_sip();
    std::vector<MatchPair> control_points = C0;

    // 初始 S_robust: 从 C0 + CD0 计算
    SRobustResult s_robust_result;
    double s_robust;
    int rc = vm44_compute_s_robust(U, W, C0, CD, SIP, s0, M0, s_robust_result, logger);
    if (rc != 0 || s_robust_result.s_robust <= 0) {
        // 兜底: 用 s0 × 0.1 作为初始评分
        s_robust = s0 * 0.1;
        if (logger) logger->warnf("vm44_irm: 初始 S_robust 计算失败, 兜底 s_robust=%.4f\"", s_robust);
    } else {
        s_robust = s_robust_result.s_robust;
    }

    if (logger) logger->infof("vm44_irm 初始: S_robust=%.4f\", N_inliers=%d",
                               s_robust, s_robust_result.n_inliers);

    // --- IRM 闭环迭代 ---
    for (int iter = 1; iter <= max_iter; ++iter) {
        if (logger) logger->infof("--- IRM 迭代 %d/%d (S_robust=%.4f\") ---",
                                   iter, max_iter, s_robust);

        // ═══ Step 1: 投影匹配 (扩增) ═══
        ExpansionResult expand_result;
        rc = vm44_expand(U, W, CD, SIP, s_robust, s0, ra0, dec0,
                         img_width, img_height, params, expand_result, logger);
        if (rc != 0 || expand_result.candidates.empty()) {
            if (logger) logger->warnf("vm44_irm 迭代 %d: 扩增失败, 使用 C0", iter);
            expand_result.candidates = C0;
        }

        // ═══ Step 2: 局部几何一致性过滤 ═══
        std::vector<MatchPair> filtered;
        rc = vm44_geometry_filter(expand_result.candidates, U, W, s0, s_robust,
                                   params, filtered, logger);
        if (rc != 0 || filtered.empty()) {
            if (logger) logger->warnf("vm44_irm 迭代 %d: 几何过滤后为空, 使用 C0", iter);
            filtered = C0;
        }

        // ═══ Step 3: 验证 (MAD + RANSAC + 贝叶斯 + 三角形) ═══
        VerificationResult verify_result;
        rc = vm44_verify(filtered, U, W, s0, fov_diag_deg, params,
                          verify_result, logger);
        if (rc != 0 || verify_result.inliers.empty()) {
            if (logger) logger->warnf("vm44_irm 迭代 %d: 验证失败, 使用 C0", iter);
            verify_result.inliers = C0;
        }
        // 捕获验证指标 (用于最终结果输出)
        final_bayes_lnK = verify_result.bayes_lnK;
        final_triangle_pass_ratio = verify_result.triangle_pass_ratio;

        // ═══ Step 4: 合并 C0 ∪ inliers (去重) ═══
        std::vector<MatchPair> new_control_points = merge_pairs(C0, verify_result.inliers);
        int n_new = (int)new_control_points.size();
        int n_old = (int)control_points.size();
        if (logger) logger->infof("vm44_irm 迭代 %d: 控制点 %d → %d (C0=%d + inliers=%d)",
                                   iter, n_old, n_new, M0, (int)verify_result.inliers.size());

        // 控制点数未增长且已迭代 ≥ 2 次 → 收敛
        if (n_new <= n_old && iter >= 2) {
            if (logger) logger->infof("vm44_irm 迭代 %d: 控制点不再增长, 收敛", iter);
            control_points = new_control_points;
            converged = true;
            n_iters = iter;
            break;
        }
        control_points = new_control_points;

        // ═══ Step 5: 模型精化 (Huber LSQ 分层拟合) ═══
        // SIP 阶数: min(sip_max_order, iter)  (迭代中递增)
        int sip_order = params.sip_max_order;
        if (sip_order < 0) sip_order = 0;
        if (sip_order > iter) sip_order = iter;  // 前几轮用低阶
        // 控制点不足时跳过 SIP
        int sip_min = params.irm_sip_min_pairs;
        if (sip_min < 1) sip_min = 30;
        if ((int)control_points.size() < sip_min) {
            sip_order = 0;
        }
        if (params.skip_sip) sip_order = 0;

        WcsFitResult fit_result;
        rc = vm44_fit(U, W, control_points, ra0, dec0,
                      focal_length_mm, pixel_size_um,
                      img_width, img_height, sip_order,
                      params, fit_result, logger);
        if (rc != 0 || !fit_result.success) {
            if (logger) logger->warnf("vm44_irm 迭代 %d: 拟合失败, 保留上一轮模型", iter);
            // 保留上一轮 CD/SIP, 继续下一轮
            n_iters = iter;
            continue;
        }

        // 更新 CD + SIP
        CD = fit_result.cd;
        SIP = fit_result.sip;

        // ═══ Step 6: S_robust 评分 ═══
        SRobustResult s_robust_new_result;
        rc = vm44_compute_s_robust(U, W, control_points, CD, SIP, s0, M0,
                                    s_robust_new_result, logger);
        if (rc != 0 || s_robust_new_result.s_robust <= 0) {
            if (logger) logger->warnf("vm44_irm 迭代 %d: S_robust 计算失败, 保留上一轮", iter);
            n_iters = iter;
            continue;
        }
        double s_robust_new = s_robust_new_result.s_robust;

        if (logger) logger->infof("vm44_irm 迭代 %d: S_robust %.4f → %.4f (Δ=%.4f, n_inliers=%d)",
                                   iter, s_robust, s_robust_new,
                                   s_robust_new - s_robust, s_robust_new_result.n_inliers);

        // ═══ 收敛判定 ═══
        double delta = std::abs(s_robust_new - s_robust);
        if (delta < converge_eps) {
            if (logger) logger->infof("vm44_irm 迭代 %d: 收敛 (Δ=%.4f < %.4f)",
                                       iter, delta, converge_eps);
            s_robust = s_robust_new;
            s_robust_result = s_robust_new_result;
            converged = true;
            n_iters = iter;
            break;
        }

        // 防过拟合: S_robust 变差 → 回退, 停止
        if (s_robust_new > s_robust * diverge_factor) {
            if (logger) logger->warnf("vm44_irm 迭代 %d: S_robust 变差 (%.4f > %.4f×%.4f), 停止",
                                       iter, s_robust_new, s_robust, diverge_factor);
            // 保留上一轮模型 (不更新 s_robust)
            converged = false;
            n_iters = iter;
            break;
        }

        // 更新状态
        s_robust = s_robust_new;
        s_robust_result = s_robust_new_result;
        n_iters = iter;
    }

    // --- 输出最终结果 ---
    final_cd = CD;
    final_sip = SIP;
    final_control_points = control_points;
    final_s_robust = s_robust_result;

    if (logger) logger->infof("=== vm44_irm 完成: iter=%d, converged=%d, S_robust=%.4f\", N=%d ===",
                               n_iters, converged ? 1 : 0, s_robust,
                               (int)control_points.size());

    return 0;
}

} // namespace v44

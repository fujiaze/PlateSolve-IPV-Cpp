// ============================================================================
// pv_core.cpp - V4.2 PairVerifier 主入口（Task 5）
//
// 串联: Phase D(MAD清洗) → Phase D'(贝叶斯+三角形验证)
//
// 流程:
//   1. 调用 pv_mad_clean 清洗匹配对
//   2. 用清洗后对 + 变换后的W 构造 matched_pairs
//   3. 调用 pv_bayes_verify 贝叶斯验证
//   4. 调用 pv_triangle_verify 三角形验证
//   5. validated = (bayes_decision >= 0) && (triangle_pass_ratio >= threshold)
//
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "pv_internal.h"

#include <cstring>
#include <string>

namespace pv {

// ============================================================================
// pv_verify - 主入口（extern "C" 在 pv_api.h 中声明）
// ============================================================================
int pv_verify_impl(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    double /*s0*/,
    const PairVerifierParams* params,
    VerificationResult* result)
{
    // --- 初始化 result ---
    std::memset(result, 0, sizeof(VerificationResult));
    result->clean_u = nullptr;
    result->clean_w = nullptr;

    // --- 初始化日志 ---
    v42::Logger logger;
    if (params && params->log_file_path) {
        logger.init(params->log_file_path);
    }
    logger.info("=== PairVerifier 开始 (n_pairs=" + std::to_string(n_pairs) +
                ", N_img=" + std::to_string(N_img) +
                ", M=" + std::to_string(M) + ") ===");

    // --- 边界检查 ---
    if (!U || !W || !pairs_u || !pairs_w || !params || !result) {
        logger.error("pv_verify: 空指针参数");
        result->success = 0;
        return 0;
    }
    if (n_pairs <= 0) {
        logger.warn("pv_verify: n_pairs <= 0");
        result->success = 1;  // 空输入视为成功但无结果
        result->validated = 0;
        return 1;
    }

    // --- Phase D: MAD 清洗 ---
    logger.info("Phase D: MAD 清洗开始");
    MadResult mad = pv_mad_clean(U, N_img, W, M,
                                 pairs_u, pairs_w, n_pairs,
                                 params, &logger);

    result->n_clean = (int)mad.clean_u.size();
    result->n_removed = mad.n_removed;
    result->mad_iterations = mad.iterations;
    result->mad_rms_arcsec = mad.rms_arcsec;

    // 分配 clean_u/clean_w（堆内存，由 pv_free 释放）
    if (result->n_clean > 0) {
        result->clean_u = new int[result->n_clean];
        result->clean_w = new int[result->n_clean];
        for (int i = 0; i < result->n_clean; ++i) {
            result->clean_u[i] = mad.clean_u[i];
            result->clean_w[i] = mad.clean_w[i];
        }
    }

    if (result->n_clean < 3) {
        logger.warn("pv_verify: 清洗后对数 < 3, 跳过验证 (n_clean=" +
                    std::to_string(result->n_clean) + ")");
        result->bayes_decision = -1;
        result->triangle_pass_ratio = 0.0;
        result->validated = 0;
        result->success = 1;
        return 1;
    }

    // --- 构造 matched_pairs: (img_x, img_y, cat_x, cat_y) ---
    // cat 坐标 = 变换后的 Wt
    std::vector<double> Wt(M * 2);
    if (mad.transform.valid) {
        apply_similarity(W, M, mad.transform.s, mad.transform.theta,
                         mad.transform.tx, mad.transform.ty, Wt.data());
    } else {
        // Umeyama 失败，Wt = W（恒等变换）
        std::memcpy(Wt.data(), W, sizeof(double) * M * 2);
    }

    std::vector<std::array<double, 4>> matched_pairs;
    matched_pairs.reserve(result->n_clean);
    for (int i = 0; i < result->n_clean; ++i) {
        int ui = mad.clean_u[i], wi = mad.clean_w[i];
        matched_pairs.push_back({U[ui * 2], U[ui * 2 + 1],
                                  Wt[wi * 2], Wt[wi * 2 + 1]});
    }

    // --- Phase D': 贝叶斯验证 ---
    logger.info("Phase D': 贝叶斯验证开始");
    auto br = pv_bayes_verify(matched_pairs,
                              params->sigma_min, mad.rms_arcsec,
                              params->fov_diag_deg,
                              params->lnK_accept, params->lnK_weak,
                              &logger);
    result->bayes_lnK = br.lnK;
    result->bayes_n_match = br.n_match;
    result->bayes_decision = br.decision;

    // --- Phase D': 三角形验证 ---
    logger.info("Phase D': 三角形验证开始");
    auto tr = pv_triangle_verify(matched_pairs,
                                 params->eps_A, params->eps_J,
                                 params->triangle_pass_rate,
                                 &logger);
    result->triangle_total = tr.total;
    result->triangle_passed = tr.passed;
    result->triangle_pass_ratio = tr.pass_ratio;

    // --- 综合验证 ---
    // validated = (bayes_decision >= 0) && (triangle_pass_ratio >= triangle_pass_rate)
    bool bayes_ok = (br.decision >= 0);
    bool tri_ok = (tr.pass_ratio >= params->triangle_pass_rate);
    result->validated = (bayes_ok && tri_ok) ? 1 : 0;
    result->success = 1;

    logger.info("=== PairVerifier 完成: validated=" + std::to_string(result->validated) +
                " (bayes=" + std::to_string(bayes_ok) +
                " tri=" + std::to_string(tri_ok) + ") ===");

    return 1;
}

} // namespace pv

// ============================================================================
// extern "C" 入口（在 pv_api.h 中声明）
// ============================================================================
extern "C" PV_API int pv_verify(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    double s0,
    const PairVerifierParams* params,
    VerificationResult* result)
{
    return pv::pv_verify_impl(U, N_img, W, M, pairs_u, pairs_w, n_pairs,
                               s0, params, result);
}

extern "C" PV_API void pv_free(VerificationResult* result)
{
    if (!result) return;
    if (result->clean_u) { delete[] result->clean_u; result->clean_u = nullptr; }
    if (result->clean_w) { delete[] result->clean_w; result->clean_w = nullptr; }
    result->n_clean = 0;
}

// ============================================================================
// pv_mad.cpp - V4.2 PairVerifier Phase D: 3 轮 MAD 迭代清洗（Task 5）
//
// 从 V4.1 vector_match_v4_1/src/vm4_core.cpp Phase D（约1815-1900行）迁移
//
// 算法:
//   1. 用输入匹配对做初始 Umeyama 拟合 → (s, θ, tx, ty)
//   2. 变换 W → Wt
//   3. 迭代(最多 mad_iters 轮):
//      a. 收集保留点的残差 r_i = |U[ui] - Wt[wi]|
//      b. MAD = 1.4826 × median(|r_i - median(r)|)
//      c. 阈值 = max(mad_min_threshold_arcsec, mad_threshold_factor × 1.4826 × MAD)
//      d. 剔除 r_i > 阈值 的对
//      e. 若本轮无剔除 → 提前终止
//      f. 用剩余点重新 Umeyama 拟合 → 更新变换 → 重新变换 W
//   4. 计算清洗后 RMS
//
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "pv_internal.h"

namespace pv {

// ============================================================================
// pv_mad_clean - MAD 迭代清洗主函数
// ============================================================================
MadResult pv_mad_clean(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    const PairVerifierParams* params,
    v42::Logger* logger)
{
    MadResult result;
    result.n_removed = 0;
    result.iterations = 0;
    result.rms_arcsec = 0.0;
    result.success = false;

    // --- 边界检查 ---
    if (n_pairs < 3) {
        // 不足3对，无法拟合，直接返回原始对
        for (int i = 0; i < n_pairs; ++i) {
            result.clean_u.push_back(pairs_u[i]);
            result.clean_w.push_back(pairs_w[i]);
        }
        result.transform.valid = false;
        result.success = (n_pairs >= 0);
        if (logger) logger->warn("pv_mad: 对数 < 3, 跳过清洗 (n=" +
                                 std::to_string(n_pairs) + ")");
        return result;
    }

    int max_iters = params->mad_iters;
    if (max_iters < 1) max_iters = 3;
    double thresh_factor = params->mad_threshold_factor;
    if (thresh_factor <= 0) thresh_factor = 3.0;
    double min_thresh = params->mad_min_threshold_arcsec;
    if (min_thresh <= 0) min_thresh = 5.0;

    // --- 初始 Umeyama 拟合 ---
    // src = W[pairs_w], dst = U[pairs_u]
    std::vector<double> src, dst;
    src.reserve(n_pairs * 2);
    dst.reserve(n_pairs * 2);
    for (int i = 0; i < n_pairs; ++i) {
        int wi = pairs_w[i], ui = pairs_u[i];
        src.push_back(W[wi * 2]);     src.push_back(W[wi * 2 + 1]);
        dst.push_back(U[ui * 2]);     dst.push_back(U[ui * 2 + 1]);
    }
    v42::SimTransform sim = umeyama(src.data(), dst.data(), n_pairs);
    if (!sim.valid) {
        // Umeyama 失败（尺度偏差过大），退化为恒等变换
        sim.s = 1.0; sim.theta = 0.0; sim.tx = 0.0; sim.ty = 0.0; sim.valid = false;
        if (logger) logger->warn("pv_mad: 初始 Umeyama 失败, 使用恒等变换");
    }

    // 变换 W → Wt
    std::vector<double> Wt(M * 2);
    apply_similarity(W, M, sim.s, sim.theta, sim.tx, sim.ty, Wt.data());

    // --- 鲁棒预过滤 ---
    // 初始 Umeyama 用所有点拟合, 可能被离群拉偏。此时正确对残差集中且 med 较大,
    // MAD 反映的是残差集中度而非噪声, 导致阈值过低误删正确对。
    // 预过滤: 当 init_med 较大时, 用 thresh_factor × init_med 作为粗阈值剔除明显离群,
    // 重新 Umeyama 收敛后再进入标准 MAD 迭代。不改变 MAD 阈值公式 max(5", 3×1.4826×MAD)。
    std::vector<bool> keep(n_pairs, true);
    {
        std::vector<double> init_res(n_pairs);
        for (int i = 0; i < n_pairs; ++i) {
            int ui = pairs_u[i], wi = pairs_w[i];
            double dx = U[ui * 2]     - Wt[wi * 2];
            double dy = U[ui * 2 + 1] - Wt[wi * 2 + 1];
            init_res[i] = std::sqrt(dx * dx + dy * dy);
        }
        auto init_res_copy = init_res;
        double init_med = vec_median(init_res_copy);
        if (logger) logger->debug("pv_mad 预过滤检查: init_med=" +
            std::to_string(init_med) + " min_thresh=" + std::to_string(min_thresh) +
            " sim.valid=" + std::to_string(sim.valid ? 1 : 0));
        // 仅当残差中位数较大(> min_thresh)时执行预过滤, 说明初始变换被拉偏
        if (init_med > min_thresh) {
            if (logger) logger->debug("pv_mad 预过滤触发: pre_thresh=" +
                std::to_string(std::max(min_thresh, thresh_factor * init_med)));
            double pre_thresh = std::max(min_thresh, thresh_factor * init_med);
            std::vector<double> sp2, dp2;
            int n_pre_removed = 0;
            for (int i = 0; i < n_pairs; ++i) {
                if (init_res[i] > pre_thresh) {
                    keep[i] = false;
                    ++n_pre_removed;
                } else {
                    int wi = pairs_w[i], ui = pairs_u[i];
                    sp2.push_back(W[wi * 2]);  sp2.push_back(W[wi * 2 + 1]);
                    dp2.push_back(U[ui * 2]);  dp2.push_back(U[ui * 2 + 1]);
                }
            }
            int n_keep = (int)sp2.size() / 2;
            if (n_pre_removed > 0 && n_keep >= 3) {
                auto sim2 = umeyama(sp2.data(), dp2.data(), n_keep);
                if (sim2.valid) {
                    sim = sim2;
                    apply_similarity(W, M, sim.s, sim.theta, sim.tx, sim.ty, Wt.data());
                    if (logger) logger->debug("pv_mad 预过滤: init_med=" +
                        std::to_string(init_med) + " pre_thresh=" + std::to_string(pre_thresh) +
                        " removed=" + std::to_string(n_pre_removed) + " keep=" + std::to_string(n_keep) +
                        " (重新 Umeyama 收敛)");
                }
            }
        }
    }

    // --- MAD 迭代清洗 ---
    int iter = 0;
    for (; iter < max_iters; ++iter) {
        // 收集保留点的残差
        std::vector<double> res_list;
        res_list.reserve(n_pairs);
        for (int i = 0; i < n_pairs; ++i) {
            if (!keep[i]) continue;
            int ui = pairs_u[i], wi = pairs_w[i];
            double dx = U[ui * 2]     - Wt[wi * 2];
            double dy = U[ui * 2 + 1] - Wt[wi * 2 + 1];
            res_list.push_back(std::sqrt(dx * dx + dy * dy));
        }
        if (res_list.size() < 10) break;

        // MAD
        auto res_copy = res_list;
        double med = vec_median(res_copy);
        std::vector<double> dev(res_list.size());
        for (size_t i = 0; i < res_list.size(); ++i)
            dev[i] = std::abs(res_list[i] - med);
        double mad = vec_median(dev);
        double sigma = 1.4826 * mad;
        double thresh = std::max(min_thresh, thresh_factor * sigma);

        // 剔除离群
        int n_removed_this = 0;
        size_t res_idx = 0;
        for (int i = 0; i < n_pairs; ++i) {
            if (!keep[i]) continue;
            if (res_list[res_idx] > thresh) {
                keep[i] = false;
                n_removed_this++;
            }
            res_idx++;
        }

        if (logger) logger->debug("pv_mad 轮" + std::to_string(iter) +
            ": thresh=" + std::to_string(thresh) + " removed=" + std::to_string(n_removed_this) +
            " keep=" + std::to_string((int)res_list.size() - n_removed_this));

        if (n_removed_this == 0) break;

        // 用剩余点重新 Umeyama 拟合
        std::vector<double> sp, dp;
        for (int i = 0; i < n_pairs; ++i) {
            if (!keep[i]) continue;
            int wi = pairs_w[i], ui = pairs_u[i];
            sp.push_back(W[wi * 2]);  sp.push_back(W[wi * 2 + 1]);
            dp.push_back(U[ui * 2]);  dp.push_back(U[ui * 2 + 1]);
        }
        auto sim2 = umeyama(sp.data(), dp.data(), (int)sp.size() / 2);
        if (!sim2.valid) break;  // 拟合失败，保留当前变换
        sim = sim2;
        apply_similarity(W, M, sim.s, sim.theta, sim.tx, sim.ty, Wt.data());
    }
    result.iterations = iter + (iter < max_iters ? 1 : 0);
    if (iter >= max_iters) result.iterations = max_iters;
    result.transform = sim;

    // --- 收集清洗后对 + 计算 RMS ---
    double ssq = 0.0;
    int nc = 0;
    for (int i = 0; i < n_pairs; ++i) {
        if (!keep[i]) continue;
        result.clean_u.push_back(pairs_u[i]);
        result.clean_w.push_back(pairs_w[i]);
        int ui = pairs_u[i], wi = pairs_w[i];
        double dx = U[ui * 2]     - Wt[wi * 2];
        double dy = U[ui * 2 + 1] - Wt[wi * 2 + 1];
        ssq += dx * dx + dy * dy;
        nc++;
    }
    result.n_removed = n_pairs - nc;
    result.rms_arcsec = (nc > 0) ? std::sqrt(ssq / nc) : 0.0;
    result.success = true;

    if (logger) logger->info("pv_mad: " + std::to_string(result.iterations) +
        " 轮, 清洗后 " + std::to_string(nc) + " 对 (剔除 " +
        std::to_string(result.n_removed) + "), RMS=" +
        std::to_string(result.rms_arcsec) + "\"");

    return result;
}

} // namespace pv

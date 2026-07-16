// ============================================================================
// vm44_expand.cpp - V4.3 PairExpander 模块 (Phase C / IRM Step 1)
//
// 职责: 马氏距离自适应匹配 + Lowe 距离比 + 双向验证 + 区域均匀化
// 从 V4.2 pe_core.cpp 升级:
//   - 固定阈值 τ=3×s0 → 自适应 τ_i = max(tau_factor × σ_proj, tau_min)
//   - 新增 Lowe 距离比检验 (d_1/d_2 < lowe_ratio)
//   - 新增双向验证 (CD⁻¹ 反向投影一致性)
//   - 投影方向: U → W (用 CD+SIP), 与 V4.2 的 W → U 相反
//
// 算法流程:
//   1. 投影: proj_i = CD · U[i] + SIP_correction(U[i])  (U→W 空间)
//   2. 投影不确定性: σ_proj(x,y) = σ₀ × √(1 + r²/fov_half²)
//      σ₀ = max(s_robust, s0×0.1), r²=(x-cx)²+(y-cy)², cx=cy=0
//   3. 自适应匹配: τ_i = max(tau_factor × σ_proj, tau_min)
//      对每个 U[i] 线性扫描 W 找 w_1(d_1), w_2(d_2)
//      if d_1<τ_i AND d_1/d_2<lowe_ratio: 候选匹配
//   4. 双向验证: CD⁻¹ · W[w] → U 空间, 找最近邻 u_k
//      if u_k==u_idx AND d_rev<τ_i: 保留
//   5. 1对1 贪心互斥 (按距离升序)
//   6. 区域均匀化 (N_floor/N_cap/N_max, 保留 V4.2 逻辑)
//
// Task 3 实现
// ============================================================================

#include "vm44_internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>

namespace v44 {

namespace {

// SIP 多项式求值 (order>=2 才有贡献, 仅累加 p+q>=2 且 p+q<=order 的项)
// 系数布局: A[p*6+q], B[p*6+q], p,q ∈ [0, order]
// 与 vm44_score.cpp 中的实现一致, 因其位于匿名命名空间无法直接复用
void sip_eval(const SIPCoeffs& sip, double x, double y,
              double& du, double& dv) {
    du = 0.0;
    dv = 0.0;
    if (sip.order < 2) return;
    double xp[5] = {1.0, x, x * x, x * x * x, x * x * x * x};
    double yq[5] = {1.0, y, y * y, y * y * y, y * y * y * y};
    for (int p = 0; p <= sip.order; ++p) {
        for (int q = 0; q <= sip.order; ++q) {
            int s = p + q;
            if (s < 2 || s > sip.order) continue;
            int idx = p * 6 + q;
            du += sip.A[idx] * xp[p] * yq[q];
            dv += sip.B[idx] * xp[p] * yq[q];
        }
    }
}

// CD 矩阵求逆: CD⁻¹ = (1/det) × [cd22, -cd12; -cd21, cd11]
// 返回 false 表示矩阵奇异
bool cd_inverse(const CDMatrix& cd, CDMatrix& inv) {
    double det = cd.cd11 * cd.cd22 - cd.cd12 * cd.cd21;
    if (std::fabs(det) < 1e-15) return false;
    double k = 1.0 / det;
    inv.cd11 =  cd.cd22 * k;
    inv.cd12 = -cd.cd12 * k;
    inv.cd21 = -cd.cd21 * k;
    inv.cd22 =  cd.cd11 * k;
    return true;
}

// 候选匹配结构
struct Cand {
    double dist;   // d_1 最近邻距离 (角秒)
    int    u;      // U 索引
    int    w;      // W 索引
    double ux;     // U.x (用于区域索引, 角秒)
    double uy;     // U.y
};

} // anonymous namespace

int vm44_expand(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const CDMatrix& cd,
    const SIPCoeffs& sip,
    double s_robust,
    double s0,
    double ra0, double dec0,
    int img_width, int img_height,
    const VM44SolveParams& params,
    ExpansionResult& output,
    Logger* logger)
{
    std::memset(&output, 0, sizeof(output));

    const int N = (int)U.size();
    const int M = (int)W.size();
    if (N <= 0 || M <= 0) {
        if (logger) logger->warn("vm44_expand: U/W 为空");
        return -1;
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    // 参数默认值保护
    double s0_eff       = (s0 > 0) ? s0 : 1.0;
    double tau_factor   = (params.irm_tau_factor > 0) ? params.irm_tau_factor : 3.0;
    double tau_min      = params.irm_tau_min;             // 默认 2.0
    double lowe_ratio   = (params.irm_lowe_ratio > 0) ? params.irm_lowe_ratio : 0.7;
    int    region_size_px = (params.region_size_px > 0) ? params.region_size_px : 800;
    int    N_floor      = (params.N_floor > 0) ? params.N_floor : 5;
    int    N_cap        = (params.N_cap > N_floor) ? params.N_cap : 30;
    int    N_max        = (params.N_max > 0) ? params.N_max : 1500;
    double img_w        = (img_width  > 0) ? (double)img_width  : 4500.0;
    double img_h        = (img_height > 0) ? (double)img_height : 3600.0;

    // σ₀ = max(s_robust, s0 × 0.1) (首次迭代 s_robust 可能很小)
    double sigma0 = std::max(s_robust, s0_eff * 0.1);

    // fov_half (角秒) = √((img_width×s0/2)² + (img_height×s0/2)²)
    double half_w_asec = img_w * s0_eff / 2.0;
    double half_h_asec = img_h * s0_eff / 2.0;
    double fov_half    = std::sqrt(half_w_asec * half_w_asec +
                                    half_h_asec * half_h_asec);
    double fov_half_sq = fov_half * fov_half;
    if (fov_half_sq < 1e-12) fov_half_sq = 1e-12;  // 防止除零

    if (logger) {
        logger->infof("=== vm44_expand 开始 ===");
        logger->infof("N=%d M=%d s0=%.4f sigma0=%.4f fov_half=%.2f\" "
                      "tau_factor=%.2f tau_min=%.4f lowe=%.2f",
                      N, M, s0_eff, sigma0, fov_half, tau_factor, tau_min, lowe_ratio);
        logger->infof("region_size_px=%d N_floor=%d N_cap=%d N_max=%d img=%gx%g",
                      region_size_px, N_floor, N_cap, N_max, img_w, img_h);
        logger->infof("CD=[%.4f,%.4f;%.4f,%.4f] sip_order=%d s_robust=%.4f "
                      "ra0=%.4f dec0=%.4f",
                      cd.cd11, cd.cd12, cd.cd21, cd.cd22, sip.order,
                      s_robust, ra0, dec0);
    }

    // CD⁻¹ (用于双向验证)
    CDMatrix cd_inv;
    bool has_inv = cd_inverse(cd, cd_inv);
    if (!has_inv && logger) {
        logger->warn("vm44_expand: CD 矩阵奇异, 跳过双向验证");
    }

    // -------------------------------------------------------------------
    // Step 1: 投影 U[i] → W 空间
    //   U 是角秒坐标 (Y 向上), W 是角秒坐标 (Y 向上 = Dec 北)
    //   CD 是度/像素 (标准 WCS), 需要单位转换:
    //     px = U.x / s0  (角秒 → 像素偏移, X 右为正)
    //     py = -U.y / s0 (角秒 → 像素偏移, Y 翻转: U_y 上 → pixel_y 下)
    //     sky_deg = CD × [px + SIP_x; py + SIP_y]  (SIP 修正 in pixel space)
    //     W_proj = sky_deg × 3600  (度 → 角秒)
    // -------------------------------------------------------------------
    std::vector<double> proj_x(N), proj_y(N);
    for (int i = 0; i < N; ++i) {
        double px = U[i].x / s0_eff;
        double py = -U[i].y / s0_eff;  // Y 翻转
        double du, dv;
        sip_eval(sip, px, py, du, dv);  // SIP 修正 (像素空间)
        double sx_deg = cd.cd11 * (px + du) + cd.cd12 * (py + dv);
        double sy_deg = cd.cd21 * (px + du) + cd.cd22 * (py + dv);
        proj_x[i] = sx_deg * 3600.0;  // 度 → 角秒
        proj_y[i] = sy_deg * 3600.0;
    }

    // -------------------------------------------------------------------
    // Step 1.5: 估计平移 (CD 不含平移, 需从数据估计)
    //   两步法: 先用无平移投影找最近邻, 取中位偏移作平移估计
    //   然后将平移加到 proj 上 (W 空间, 角秒)
    // -------------------------------------------------------------------
    double t_W_x = 0.0, t_W_y = 0.0;
    if (N > 0 && M > 0) {
        std::vector<double> off_x, off_y;
        off_x.reserve(N);
        off_y.reserve(N);
        for (int i = 0; i < N; ++i) {
            double best_d2 = 1e300;
            int best_w = -1;
            for (int j = 0; j < M; ++j) {
                double dx = proj_x[i] - W[j].x;
                double dy = proj_y[i] - W[j].y;
                double d2 = dx * dx + dy * dy;
                if (d2 < best_d2) { best_d2 = d2; best_w = j; }
            }
            if (best_w >= 0) {
                off_x.push_back(W[best_w].x - proj_x[i]);
                off_y.push_back(W[best_w].y - proj_y[i]);
            }
        }
        if (!off_x.empty()) {
            std::nth_element(off_x.begin(), off_x.begin() + off_x.size() / 2, off_x.end());
            t_W_x = off_x[off_x.size() / 2];
            std::nth_element(off_y.begin(), off_y.begin() + off_y.size() / 2, off_y.end());
            t_W_y = off_y[off_y.size() / 2];
        }
        // 应用平移
        for (int i = 0; i < N; ++i) {
            proj_x[i] += t_W_x;
            proj_y[i] += t_W_y;
        }
        if (logger) {
            logger->infof("Step 1.5 平移估计: t_W=(%.4f, %.4f)\" (N_offset=%d)",
                          t_W_x, t_W_y, (int)off_x.size());
        }
    }

    // -------------------------------------------------------------------
    // Step 2: 计算每个 U[i] 的自适应阈值 τ_i
    //   σ_proj(x,y) = σ₀ × √(1 + ((x-cx)²+(y-cy)²) / fov_half²)
    //   τ_i = max(tau_factor × σ_proj, tau_min)
    //   cx = cy = 0 (U 坐标原点在图像中心)
    // -------------------------------------------------------------------
    std::vector<double> tau_i(N);
    for (int i = 0; i < N; ++i) {
        double r2 = U[i].x * U[i].x + U[i].y * U[i].y;
        double sigma_proj = sigma0 * std::sqrt(1.0 + r2 / fov_half_sq);
        tau_i[i] = std::max(tau_factor * sigma_proj, tau_min);
    }

    // -------------------------------------------------------------------
    // Step 3: 自适应匹配 + Lowe 距离比检验
    //   对每个 U[i], 线性扫描 W 找最近邻 w_1 (d_1) 和次近邻 w_2 (d_2)
    //   if d_1 < τ_i AND d_1/d_2 < lowe_ratio: 候选匹配 = (i, w_1_idx)
    // -------------------------------------------------------------------
    std::vector<Cand> candidates;
    candidates.reserve(N);
    int n_tau_rejected = 0, n_lowe_rejected = 0;

    for (int i = 0; i < N; ++i) {
        double px = proj_x[i];
        double py = proj_y[i];
        double tau_i_val = tau_i[i];
        double tau2 = tau_i_val * tau_i_val;

        // 线性扫描 W 找最近邻 w_1 和次近邻 w_2 (O(N·M), 无 KDTree/k-vector)
        double best_d2 = 1e300, second_d2 = 1e300;
        int best_w = -1;
        for (int j = 0; j < M; ++j) {
            double dx = px - W[j].x;
            double dy = py - W[j].y;
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                second_d2 = best_d2;
                best_d2 = d2;
                best_w = j;
            } else if (d2 < second_d2) {
                second_d2 = d2;
            }
        }

        // τ 截断
        if (best_w < 0 || best_d2 > tau2) {
            ++n_tau_rejected;
            continue;
        }

        // Lowe 距离比检验 (要求至少有次近邻)
        if (second_d2 > 1e-12) {
            double ratio = std::sqrt(best_d2 / second_d2);
            if (ratio >= lowe_ratio) {
                ++n_lowe_rejected;
                continue;
            }
        }

        candidates.push_back({std::sqrt(best_d2), i, best_w, U[i].x, U[i].y});
    }

    if (logger) {
        logger->infof("Step 3 自适应匹配: 候选 %d (τ拒 %d, lowe拒 %d)",
                      (int)candidates.size(), n_tau_rejected, n_lowe_rejected);
    }

    // -------------------------------------------------------------------
    // Step 4: 双向验证
    //   对每个候选匹配 (u_idx, w_idx):
    //     用 CD⁻¹ 将 W[w_idx] 反向投影到 U 空间
    //     单位转换 (与 Step 1 对称):
    //       W(角秒) → 度 (÷3600) → CD⁻¹·deg → 像素偏移 → U 空间角秒 (×s0, Y 翻转)
    //     在 U 中找最近邻 u_k, 距离 d_rev
    //     if u_k == u_idx AND d_rev < τ_i: 保留, 否则丢弃
    // -------------------------------------------------------------------
    std::vector<Cand> verified;
    verified.reserve(candidates.size());
    int n_reverse_rejected = 0;

    if (has_inv) {
        for (const auto& c : candidates) {
            // Bug 4 修复: t_W 是 W 空间的平移 (前向: proj = CD·U + t_W)
            // 反向必须先在 W 空间减去 t_W, 再 CD⁻¹ 反推
            // 旧代码错误地在 U 空间减 t_W, 当 CD 有旋转时引入系统性偏移
            // W(角秒) - t_W(角秒) → 度
            double wx_deg = (W[c.w].x - t_W_x) / 3600.0;
            double wy_deg = (W[c.w].y - t_W_y) / 3600.0;
            // CD⁻¹ · (W-t_W)(度) → 像素偏移 (标准 WCS: 像素 Y 向下)
            double bx_px = cd_inv.cd11 * wx_deg + cd_inv.cd12 * wy_deg;
            double by_px = cd_inv.cd21 * wx_deg + cd_inv.cd22 * wy_deg;
            // 像素 → U 空间角秒 (Y 翻转: pixel_y 向下 → U.y 向上)
            double bx = bx_px * s0_eff;
            double by = -by_px * s0_eff;

            double tau_rev = tau_i[c.u];
            double tau_rev2 = tau_rev * tau_rev;

            // 在 U 中找最近邻
            double best_d2 = 1e300;
            int best_u = -1;
            for (int k = 0; k < N; ++k) {
                double dx = bx - U[k].x;
                double dy = by - U[k].y;
                double d2 = dx * dx + dy * dy;
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_u = k;
                }
            }

            if (best_u == c.u && best_d2 < tau_rev2) {
                verified.push_back(c);
            } else {
                ++n_reverse_rejected;
            }
        }
    } else {
        // 无逆矩阵, 跳过双向验证
        verified = candidates;
    }

    if (logger) {
        logger->infof("Step 4 双向验证: 保留 %d (拒 %d)",
                      (int)verified.size(), n_reverse_rejected);
    }

    // -------------------------------------------------------------------
    // Step 5: 1对1 贪心互斥 (按距离升序)
    //   V4.3 接口未传入 Phase B 初始对, 直接对 verified 互斥
    // -------------------------------------------------------------------
    std::sort(verified.begin(), verified.end(),
              [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

    std::vector<int> u_used(N, 0), w_used(M, 0);
    std::vector<Cand> selected;
    selected.reserve(verified.size());
    for (const auto& c : verified) {
        if (u_used[c.u] || w_used[c.w]) continue;
        u_used[c.u] = 1;
        w_used[c.w] = 1;
        selected.push_back(c);
    }

    if (logger) {
        logger->infof("Step 5 1对1互斥: 选取 %d", (int)selected.size());
    }

    // -------------------------------------------------------------------
    // Step 6: 区域均匀化 (保留 V4.2 逻辑)
    //   region_size_asec = region_size_px × s0 (像素转角秒)
    //   网格: n_cols × n_rows, 原点在中心
    //   每区保底 N_floor, 上限 N_cap, 全局上限 N_max
    // -------------------------------------------------------------------
    double region_size_asec = region_size_px * s0_eff;
    if (region_size_asec < 1e-6) region_size_asec = 1.0;  // 防止除零
    int n_cols = (int)std::ceil(img_w / (double)region_size_px);
    int n_rows = (int)std::ceil(img_h / (double)region_size_px);
    if (n_cols < 1) n_cols = 1;
    if (n_rows < 1) n_rows = 1;
    int n_regions = n_cols * n_rows;

    // 区域索引: 将角秒坐标(原点在中心)映射到区域网格
    auto region_idx = [&](double x, double y) -> int {
        int cx = (int)((x + half_w_asec) / region_size_asec);
        int cy = (int)((y + half_h_asec) / region_size_asec);
        if (cx < 0) cx = 0;
        if (cx >= n_cols) cx = n_cols - 1;
        if (cy < 0) cy = 0;
        if (cy >= n_rows) cy = n_rows - 1;
        return cy * n_cols + cx;
    };

    std::vector<std::vector<int>> region_pairs(n_regions);
    for (int i = 0; i < (int)selected.size(); ++i) {
        int r = region_idx(selected[i].ux, selected[i].uy);
        region_pairs[r].push_back(i);
    }

    // 第一轮: 每区保底 N_floor (候选已按距离升序, 直接取前 N_floor)
    std::vector<int> final_idx;
    std::vector<bool> is_sparse(n_regions, false);
    std::vector<int> taken(n_regions, 0);
    for (int r = 0; r < n_regions; ++r) {
        int take = std::min((int)region_pairs[r].size(), N_floor);
        for (int k = 0; k < take; ++k) {
            final_idx.push_back(region_pairs[r][k]);
            taken[r]++;
        }
        // 候选数 < 2 视为稀疏区
        if ((int)region_pairs[r].size() < 2) {
            is_sparse[r] = true;
        }
    }

    // 第二轮: 每区补到 N_cap (候选已按距离升序, 直接追加)
    for (int r = 0; r < n_regions; ++r) {
        int already = taken[r];
        int extra = std::min((int)region_pairs[r].size() - already, N_cap - already);
        for (int k = already; k < already + extra; ++k) {
            final_idx.push_back(region_pairs[r][k]);
        }
    }

    // 全局截断 N_max (按距离升序保留最优)
    if ((int)final_idx.size() > N_max) {
        std::sort(final_idx.begin(), final_idx.end(),
                  [&](int a, int b) { return selected[a].dist < selected[b].dist; });
        final_idx.resize(N_max);
    }

    int n_sparse = 0;
    for (bool sp : is_sparse) if (sp) ++n_sparse;

    // -------------------------------------------------------------------
    // Step 7: 构造输出 (候选匹配对)
    // -------------------------------------------------------------------
    output.candidates.clear();
    output.candidates.reserve(final_idx.size());
    for (int idx : final_idx) {
        output.candidates.push_back({selected[idx].u, selected[idx].w});
    }
    output.n_expanded = (int)final_idx.size();
    output.n_pairs    = (int)final_idx.size();
    output.region_counts.assign(n_regions, 0);
    for (int idx : final_idx) {
        int r = region_idx(selected[idx].ux, selected[idx].uy);
        output.region_counts[r]++;
    }
    output.success = true;

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (logger) {
        logger->infof("Step 6 区域均匀化: 网格 %dx%d=%d区, 稀疏区 %d, "
                      "扩充 %d 对, 耗时 %.2f ms",
                      n_cols, n_rows, n_regions, n_sparse,
                      output.n_expanded, ms);
        logger->infof("=== vm44_expand 结束 ===");
    }

    return 0;
}

} // namespace v44

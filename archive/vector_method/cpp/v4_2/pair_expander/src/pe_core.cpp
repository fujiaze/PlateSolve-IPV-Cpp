// PairExpander V4.2 - Phase C 匹配对扩增器
//
// 核心简化: 移除 k-vector/KDTree/nanoflann, 改用线性扫描 NN
// 理由: CD 矩阵已知后 Wt 与 U 已接近 (<3×s0), N≤2000 下线性扫描 <20ms
//
// 算法流程:
//   1. 变换 W → U 空间: Wt[j] = s·R·W[j] + t
//   2. 对每个 Wt[j] 线性扫描 U 找最近邻 u* (O(N·M))
//   3. τ 截断: best_d2 < τ² (τ=tau_factor×s0)
//   4. 模长比过滤: |‖U[u]‖/‖W[w]‖ - 1| < scale_ratio_tol
//   5. 1对1 贪心互斥(按距离升序), 保留 Phase B 的对
//   6. 区域均匀化: region_size_px, N_floor/N_cap/N_max

#include "pe_api.h"
#include "v42_log.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cstdlib>

// 相似变换: Wt = s·R·W + t
//   Wt.x = s·(cos·Wx - sin·Wy) + tx
//   Wt.y = s·(sin·Wx + cos·Wy) + ty
static void apply_similarity(const double* W, int M,
                             double s, double theta, double tx, double ty,
                             std::vector<double>& Wt) {
    Wt.resize((size_t)M * 2);
    double c = std::cos(theta), si = std::sin(theta);
    for (int j = 0; j < M; ++j) {
        double wx = W[j * 2], wy = W[j * 2 + 1];
        Wt[j * 2]     = s * (c * wx - si * wy) + tx;
        Wt[j * 2 + 1] = s * (si * wx + c * wy) + ty;
    }
}

PE_API int pe_expand(
    const double* U, int N_img,
    const double* W, int M,
    const int* init_cu, const int* init_cw, int n_init,
    double s, double theta, double tx, double ty,
    const PairExpanderParams* params,
    ExpansionResult* result)
{
    // 初始化结果
    std::memset(result, 0, sizeof(ExpansionResult));
    result->success = 0;

    if (!U || !W || !params || !result) return 0;
    if (N_img <= 0 || M <= 0) return 0;

    // 日志 (log_file_path 为空时完全跳过日志, 避免性能测试受 stderr 影响)
    bool enable_log = (params->log_file_path && params->log_file_path[0]);
    v42::Logger logger;
    if (enable_log) {
        logger.init(params->log_file_path);
    }
    auto log_info = [&](const std::string& msg) { if (enable_log) logger.info(msg); };
    auto log_error = [&](const std::string& msg) { if (enable_log) logger.error(msg); };
    log_info("=== PairExpander V4.2 Phase C 开始 ===");
    log_info("N_img=" + std::to_string(N_img) + " M=" + std::to_string(M) +
                " n_init=" + std::to_string(n_init) +
                " s=" + std::to_string(s) + " theta=" + std::to_string(theta) +
                " tx=" + std::to_string(tx) + " ty=" + std::to_string(ty));
    log_info("s0=" + std::to_string(params->s0) +
                " tau_factor=" + std::to_string(params->tau_factor) +
                " scale_ratio_tol=" + std::to_string(params->scale_ratio_tol) +
                " region_size_px=" + std::to_string(params->region_size_px) +
                " N_floor=" + std::to_string(params->N_floor) +
                " N_cap=" + std::to_string(params->N_cap) +
                " N_max=" + std::to_string(params->N_max) +
                " img_width=" + std::to_string(params->img_width) +
                " img_height=" + std::to_string(params->img_height));

    auto t0 = std::chrono::high_resolution_clock::now();

    // 参数默认值保护
    double s0 = (params->s0 > 0) ? params->s0 : 1.0;
    double tau_factor = (params->tau_factor > 0) ? params->tau_factor : 3.0;
    double scale_ratio_tol = (params->scale_ratio_tol > 0) ? params->scale_ratio_tol : 0.1;
    int region_size_px = (params->region_size_px > 0) ? params->region_size_px : 800;
    int N_floor = (params->N_floor > 0) ? params->N_floor : 5;
    int N_cap = (params->N_cap > N_floor) ? params->N_cap : 30;
    int N_max = (params->N_max > 0) ? params->N_max : 1500;
    double img_width = (params->img_width > 0) ? params->img_width : 4500;
    double img_height = (params->img_height > 0) ? params->img_height : 3600;

    double tau = tau_factor * s0;            // 角秒
    double tau2 = tau * tau;
    double half_w_asec = img_width * s0 / 2.0;
    double half_h_asec = img_height * s0 / 2.0;
    double margin_asec = 50.0 * s0;          // 50像素 margin

    // 1. 变换 W → U 空间: Wt = s·R·W + t
    std::vector<double> Wt;
    apply_similarity(W, M, s, theta, tx, ty, Wt);

    // 2. 线性扫描 NN: 对每个 Wt[j] 找 U 最近邻 (O(N·M), 无 KDTree/k-vector)
    // 候选对: (dist, u_idx, w_idx, wt_x, wt_y)
    struct Cand {
        double dist;    // 角秒距离 (U/Wt 均为角秒坐标)
        int    u;
        int    w;
        double wx;      // Wt.x (角秒, 用于区域索引)
        double wy;      // Wt.y
    };
    std::vector<Cand> clist;
    clist.reserve(M);
    int n_out_of_range = 0;

    for (int j = 0; j < M; ++j) {
        double wtx = Wt[j * 2], wty = Wt[j * 2 + 1];
        // 图像范围过滤(角秒, 原点在中心)
        if (wtx < -half_w_asec - margin_asec || wtx > half_w_asec + margin_asec ||
            wty < -half_h_asec - margin_asec || wty > half_h_asec + margin_asec) {
            n_out_of_range++;
            continue;
        }
        // 线性扫描 U 找最近邻
        double best_d2 = 1e300;
        int best_u = -1;
        for (int i = 0; i < N_img; ++i) {
            double dx = wtx - U[i * 2];
            double dy = wty - U[i * 2 + 1];
            double d2 = dx * dx + dy * dy;
            if (d2 < best_d2) {
                best_d2 = d2;
                best_u = i;
            }
        }
        // τ 截断
        if (best_u < 0 || best_d2 > tau2) continue;
        clist.push_back({std::sqrt(best_d2), best_u, j, wtx, wty});
    }

    log_info("线性扫描 NN: 范围外 " + std::to_string(n_out_of_range) + "/" + std::to_string(M) +
                ", τ 截断后候选 " + std::to_string(clist.size()));

    // 3. 模长比过滤: |‖U[u]‖/‖W[w]‖ - 1| < scale_ratio_tol
    //    U/W 均为角秒坐标(原点在中心), 模长为到中心距离
    std::vector<Cand> accepted;
    accepted.reserve(clist.size());
    int n_ratio_rejected = 0;
    for (const auto& cd : clist) {
        double ux = U[cd.u * 2], uy = U[cd.u * 2 + 1];
        double wx = W[cd.w * 2], wy = W[cd.w * 2 + 1];
        double norm_u = std::sqrt(ux * ux + uy * uy);
        double norm_w = std::sqrt(wx * wx + wy * wy);
        if (norm_w < 1e-10) {
            // W 模长为0(星在中心), 跳过模长比检查直接接受
            accepted.push_back(cd);
            continue;
        }
        double ratio = norm_u / norm_w;
        if (std::fabs(ratio - 1.0) < scale_ratio_tol) {
            accepted.push_back(cd);
        } else {
            n_ratio_rejected++;
        }
    }
    log_info("模长比过滤: 接受 " + std::to_string(accepted.size()) +
                " (拒 " + std::to_string(n_ratio_rejected) + ")");

    // 4. 1对1 贪心互斥(按距离升序)
    std::sort(accepted.begin(), accepted.end(),
              [](const Cand& a, const Cand& b) { return a.dist < b.dist; });

    std::vector<int> u_used(N_img, 0), w_used(M, 0);
    // 先标记 Phase B 的对(保留, 不参与扩充竞争)
    for (int k = 0; k < n_init; ++k) {
        if (init_cu && init_cw) {
            int u = init_cu[k], w = init_cw[k];
            if (u >= 0 && u < N_img) u_used[u] = 1;
            if (w >= 0 && w < M) w_used[w] = 1;
        }
    }

    // 全局贪心选取(1对1互斥)
    std::vector<Cand> selected;
    selected.reserve(accepted.size());
    for (const auto& cd : accepted) {
        if (u_used[cd.u] || w_used[cd.w]) continue;
        u_used[cd.u] = 1;
        w_used[cd.w] = 1;
        selected.push_back(cd);
    }
    log_info("1对1贪心互斥: 选取 " + std::to_string(selected.size()));

    // 5. 区域均匀化
    double region_size_asec = region_size_px * s0;   // 像素转角秒
    int n_cols = (int)std::ceil(img_width / (double)region_size_px);
    int n_rows = (int)std::ceil(img_height / (double)region_size_px);
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
        int r = region_idx(selected[i].wx, selected[i].wy);
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
    for (bool sp : is_sparse) if (sp) n_sparse++;

    // 6. 构造扩充后匹配对 (Phase B + 扩充)
    std::vector<int> expand_u, expand_w;
    // Phase B 初始对
    for (int k = 0; k < n_init; ++k) {
        if (init_cu && init_cw) {
            expand_u.push_back(init_cu[k]);
            expand_w.push_back(init_cw[k]);
        }
    }
    // Phase C 扩充对
    for (int idx : final_idx) {
        expand_u.push_back(selected[idx].u);
        expand_w.push_back(selected[idx].w);
    }

    int n_expanded = (int)final_idx.size();
    int n_pairs = (int)expand_u.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    log_info("区域均匀化: 网格 " + std::to_string(n_cols) + "x" + std::to_string(n_rows) +
                "=" + std::to_string(n_regions) + "区, 稀疏区 " + std::to_string(n_sparse) +
                ", 扩充 " + std::to_string(n_expanded) + " 对, 合计 " + std::to_string(n_pairs) + " 对");
    log_info("总耗时 " + std::to_string(ms) + " ms");
    log_info("=== PairExpander V4.2 Phase C 结束 ===");

    // 填充结果
    result->n_pairs = n_pairs;
    result->n_expanded = n_expanded;
    result->n_regions = n_regions;
    result->n_sparse_regions = n_sparse;
    result->n_candidates = (int)clist.size();
    result->n_accepted = (int)accepted.size();
    result->expand_time_ms = ms;
    result->success = 1;

    // 分配并拷贝数组 (调用方需 pe_free 释放)
    size_t alloc_n = (n_pairs > 0) ? (size_t)n_pairs : 1;
    result->expand_u = (int*)std::malloc(sizeof(int) * alloc_n);
    result->expand_w = (int*)std::malloc(sizeof(int) * alloc_n);
    if (!result->expand_u || !result->expand_w) {
        if (result->expand_u) { std::free(result->expand_u); result->expand_u = nullptr; }
        if (result->expand_w) { std::free(result->expand_w); result->expand_w = nullptr; }
        result->success = 0;
        log_error("内存分配失败");
        return 0;
    }
    for (int i = 0; i < n_pairs; ++i) {
        result->expand_u[i] = expand_u[i];
        result->expand_w[i] = expand_w[i];
    }

    return 1;
}

PE_API void pe_free(ExpansionResult* result) {
    if (!result) return;
    if (result->expand_u) { std::free(result->expand_u); result->expand_u = nullptr; }
    if (result->expand_w) { std::free(result->expand_w); result->expand_w = nullptr; }
    result->n_pairs = 0;
    result->n_expanded = 0;
    result->success = 0;
}

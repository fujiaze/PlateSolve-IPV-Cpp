/**
 * vm2_core.cpp - V2向量匹配算法C++核心实现（多线程优化版）
 *
 * 与Python vector_match_v2.py完全对齐的算法逻辑
 * 多线程优化:
 *   1. 4种翻转模式OpenMP并行
 *   2. RANSAC内循环OpenMP并行（线程局部最佳，最终归约）
 *   3. KDTree复用（Wf不变时只建一次）
 *   4. 渐进tau精修（1x→2x→3x→5x s0）
 *
 * 依赖: Eigen3, nanoflann, C++17, OpenMP
 */

#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <utility>
#include <cstring>
#include <omp.h>

#include "../include/vm2_api.h"

// Eigen
#include "../third_party/Eigen/Dense"

// nanoflann
#include "../third_party/nanoflann.hpp"

namespace vm2 {

// ============================================================================
// 常量
// ============================================================================

static constexpr double PI = 3.14159265358979323846;
static constexpr double DEGTORAD = PI / 180.0;
static constexpr double RADTODEG = 180.0 / PI;
static constexpr double RADTOASEC = (180.0 / PI) * 3600.0;
static constexpr double ASECTORAD = PI / (180.0 * 3600.0);

// ============================================================================
// KDTree封装 (基于nanoflann)
// ============================================================================

struct PointCloud2D {
    std::vector<std::array<double, 2>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline double kdtree_get_pt(size_t idx, size_t dim) const { return pts[idx][dim]; }
    template <class BBOX> bool kdtree_get_bbox(BBOX&) const { return false; }
};

using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud2D>,
    PointCloud2D, 2>;
using KDTreeIndexType = KDTree::IndexType;

// ============================================================================
// apply_similarity: 对W应用相似变换(s, theta, tx, ty)
// ============================================================================

void apply_similarity(const double* W, int M, double s, double theta,
                      double tx, double ty, double* Wt)
{
    double ct = std::cos(theta);
    double st = std::sin(theta);
    for (int i = 0; i < M; ++i) {
        double wx = W[i * 2];
        double wy = W[i * 2 + 1];
        Wt[i * 2]     = s * (ct * wx - st * wy) + tx;
        Wt[i * 2 + 1] = s * (st * wx + ct * wy) + ty;
    }
}

// ============================================================================
// count_inliers_1to1: 1对1互斥内点统计（复用已有KDTree）
// ============================================================================

struct InlierResult {
    int n_inliers;
    double rms;
    std::vector<int> inlier_mask; // 长度N, 1=内点, 0=外点
};

InlierResult count_inliers_1to1(const double* U, int N, const KDTree& tree,
                                 const PointCloud2D& cloud, double tau)
{
    InlierResult result;
    result.n_inliers = 0;
    result.rms = 0.0;
    result.inlier_mask.assign(N, 0);

    if (N == 0 || (int)cloud.pts.size() == 0) return result;

    int M = (int)cloud.pts.size();

    struct Match {
        int u_idx;
        int w_idx;
        double dist;
    };
    std::vector<Match> candidates;
    candidates.reserve(N);

    double tau_sq = tau * tau;
    for (int i = 0; i < N; ++i) {
        double query[2] = {U[i * 2], U[i * 2 + 1]};
        KDTreeIndexType idx;
        double dist_sq;
        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
        rs.init(&idx, &dist_sq);
        tree.findNeighbors(rs, query);
        if (dist_sq <= tau_sq) {
            candidates.push_back({i, (int)idx, std::sqrt(dist_sq)});
        }
    }

    // 按距离排序
    std::sort(candidates.begin(), candidates.end(),
              [](const Match& a, const Match& b) { return a.dist < b.dist; });

    // 贪心1对1匹配
    std::vector<int> w_used(M, 0);
    double sum_sq = 0.0;
    for (auto& c : candidates) {
        if (w_used[c.w_idx]) continue;
        w_used[c.w_idx] = 1;
        result.inlier_mask[c.u_idx] = 1;
        sum_sq += c.dist * c.dist;
        result.n_inliers++;
    }

    if (result.n_inliers > 0) {
        result.rms = std::sqrt(sum_sq / result.n_inliers);
    }

    return result;
}

// 重载: 从原始数据构建KDTree后统计
InlierResult count_inliers_1to1(const double* U, int N, const double* Wt, int M, double tau)
{
    PointCloud2D cloud;
    cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) {
        cloud.pts[i] = {Wt[i * 2], Wt[i * 2 + 1]};
    }
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    return count_inliers_1to1(U, N, tree, cloud, tau);
}

// ============================================================================
// find_coarse_correspondences: 粗候选对构建
// ============================================================================

struct Pair {
    int u_idx;
    int w_idx;
};

std::vector<Pair> find_coarse_correspondences(const double* U, int N,
                                               const double* W, int M,
                                               double radius)
{
    std::vector<Pair> pairs;
    if (N == 0 || M == 0) return pairs;

    PointCloud2D cloud;
    cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) {
        cloud.pts[i] = {W[i * 2], W[i * 2 + 1]};
    }
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    double radius_sq = radius * radius;

    for (int i = 0; i < N; ++i) {
        double query[2] = {U[i * 2], U[i * 2 + 1]};
        std::vector<nanoflann::ResultItem<KDTreeIndexType, double>> ret_matches;
        (void)tree.radiusSearch(query, radius_sq, ret_matches);
        for (auto& m : ret_matches) {
            pairs.push_back({i, (int)m.first});
        }
    }

    return pairs;
}

// ============================================================================
// umeyama: Umeyama SVD求解最优相似变换
// 与Python _umeyama完全对齐
// dst ≈ s * R(theta) * src + (tx, ty)
// 关键: s = trace(Σ*S) / trace(W^T*W), |s-1| < 0.1
// ============================================================================

struct SimilarityTransform {
    double s;
    double theta;
    double tx;
    double ty;
    bool valid;
};

SimilarityTransform umeyama(const double* src, const double* dst, int n)
{
    SimilarityTransform result;
    result.valid = false;
    result.s = 1.0;
    result.theta = 0.0;
    result.tx = 0.0;
    result.ty = 0.0;

    if (n < 2) return result;

    using Matrix2d = Eigen::Matrix2d;
    using Vector2d = Eigen::Vector2d;

    // 去质心
    Vector2d src_mean = Vector2d::Zero();
    Vector2d dst_mean = Vector2d::Zero();
    for (int i = 0; i < n; ++i) {
        src_mean += Vector2d(src[i * 2], src[i * 2 + 1]);
        dst_mean += Vector2d(dst[i * 2], dst[i * 2 + 1]);
    }
    src_mean /= n;
    dst_mean /= n;

    // 去质心坐标
    Eigen::MatrixXd src_centered(2, n);
    Eigen::MatrixXd dst_centered(2, n);
    for (int i = 0; i < n; ++i) {
        src_centered(0, i) = src[i * 2]     - src_mean(0);
        src_centered(1, i) = src[i * 2 + 1] - src_mean(1);
        dst_centered(0, i) = dst[i * 2]     - dst_mean(0);
        dst_centered(1, i) = dst[i * 2 + 1] - dst_mean(1);
    }

    // H = src_centered * dst_centered^T (与Python w_centered.T @ u_centered对齐)
    Matrix2d H = src_centered * dst_centered.transpose();

    // SVD
    Eigen::JacobiSVD<Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    // 保证纯旋转 det(R)=+1
    double d = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    Vector2d S_vec = Vector2d::Ones(2);
    S_vec(1) = d;

    // R = V @ diag(1,d) @ U^T
    Matrix2d R = svd.matrixV() * S_vec.asDiagonal() * svd.matrixU().transpose();

    // s = trace(Σ*S) / trace(W'^T @ W')
    double trace_WtW = src_centered.colwise().squaredNorm().sum();
    double sigma_trace = (svd.singularValues().cwiseProduct(S_vec)).sum();

    if (trace_WtW < 1e-15) return result;

    double s = sigma_trace / trace_WtW;

    // s范围检查: |s-1| < 0.1
    if (std::abs(s - 1.0) >= 0.1) return result;

    double theta = std::atan2(R(1, 0), R(0, 0));

    Vector2d t = dst_mean - s * R * src_mean;

    result.s = s;
    result.theta = theta;
    result.tx = t(0);
    result.ty = t(1);
    result.valid = true;

    return result;
}

// ============================================================================
// ransac_v2: V2 RANSAC主循环（OpenMP并行版）
//
// 与Python _ransac_rigid_v2对齐:
//   1. 粗候选对构建
//   2. 按稀疏度权重抽取2个不同U点
//   3. 2点相似变换求解(s, theta, tx, ty)
//   4. s范围检查: 0.9 <= s <= 1.1
//   5. 1对1互斥内点统计
//   6. 评分: score = n_inliers - 1.0 * rms (与Python对齐)
//
// OpenMP并行: 每个线程独立运行部分迭代，最终归约选最佳
// ============================================================================

struct RansacResult {
    double s;
    double theta;
    double tx;
    double ty;
    int n_inliers;
    double rms;
    std::vector<int> inlier_mask;
    bool success;
};

RansacResult ransac_v2(const double* U, int N, const double* Wf, int M,
                        double tau, int K, int min_inliers,
                        double candidate_radius, const double* sparsity,
                        int seed, volatile bool* early_exit)
{
    RansacResult best;
    best.n_inliers = 0;
    best.rms = 1e30;
    best.s = 1.0;
    best.theta = 0.0;
    best.tx = 0.0;
    best.ty = 0.0;
    best.success = false;

    if (N < 2 || M < 2) return best;

    // 构建粗候选对
    auto pairs = find_coarse_correspondences(U, N, Wf, M, candidate_radius);
    if (pairs.size() < 2) {
        fprintf(stderr, "[vm2] ransac_v2: 候选对不足 (%zu)\n", pairs.size());
        return best;
    }

    // 预计算: u_to_pair_indices映射
    std::vector<std::vector<int>> u_to_pairs(N);
    for (int p = 0; p < (int)pairs.size(); ++p) {
        u_to_pairs[pairs[p].u_idx].push_back(p);
    }

    // 判断候选对是否过多：平均每个U点候选数 > M/4 时切换为直接选模式
    // 直接选模式：从全部Wf中随机选，不受候选对限制
    // 这对于大旋转角（90度等）的情况更有效，因为候选对中正确比例太低
    double avg_cand_per_u = (double)pairs.size() / N;
    bool use_direct_select = (avg_cand_per_u > M / 4.0);
    if (use_direct_select) {
        fprintf(stderr, "[vm2] ransac_v2: 候选对过多(%.1f/U > M/4=%d)，切换直接选模式\n",
                avg_cand_per_u, M / 4);
    } else {
        fprintf(stderr, "[vm2] ransac_v2: 候选对数=%zu (半径=%.1f角秒, %.1f/U)\n",
                pairs.size(), candidate_radius, avg_cand_per_u);
    }

    // 只保留有候选对的U点
    std::vector<int> active_u;
    std::vector<double> active_weights;
    for (int i = 0; i < N; ++i) {
        if (!u_to_pairs[i].empty()) {
            active_u.push_back(i);
            double w = (sparsity != nullptr) ? sparsity[i] : 1.0;
            active_weights.push_back(std::max(w, 1e-10));
        }
    }
    if (active_u.size() < 2) {
        fprintf(stderr, "[vm2] ransac_v2: 有候选的U点不足 (%zu)\n", active_u.size());
        return best;
    }

    // 归一化权重
    double wsum = 0.0;
    for (auto& w : active_weights) wsum += w;
    if (wsum <= 0.0) return best;
    for (auto& w : active_weights) w /= wsum;

    int n_active = (int)active_u.size();
    int actual_K = std::min(K, n_active * (n_active - 1) / 2);

    // OpenMP并行RANSAC
    int n_threads = omp_get_max_threads();
    // 每个线程维护局部最佳结果
    struct ThreadBest {
        double s, theta, tx, ty;
        int n_inliers;
        double rms;
        double score;
        int u1, w1, u2, w2; // 最佳配对索引（用于后续重建mask）
        bool valid;
    };
    std::vector<ThreadBest> thread_bests(n_threads);
    for (auto& tb : thread_bests) {
        tb.s = 1.0; tb.theta = 0.0; tb.tx = 0.0; tb.ty = 0.0;
        tb.n_inliers = 0; tb.rms = 1e30; tb.score = -1e30;
        tb.u1 = -1; tb.w1 = -1; tb.u2 = -1; tb.w2 = -1;
        tb.valid = false;
    }

    // 每个线程独立随机数生成器
    // 使用seed + thread_id确保不同线程不同序列
    // 使用离散分布实现加权采样

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        std::mt19937 rng(seed + tid * 1000);
        std::discrete_distribution<int> u_dist(active_weights.begin(), active_weights.end());
        std::uniform_int_distribution<int> w_dist(0, M - 1);

        // 线程局部缓冲区
        std::vector<double> Wt_local(M * 2);

        #pragma omp for schedule(dynamic, 64)
        for (int iter = 0; iter < actual_K; ++iter) {
            // 早退出: 其他模式已收敛
            if (early_exit && *early_exit) continue;

            // 按稀疏度权重选2个不同的U点
            int idx1 = u_dist(rng);
            int idx2 = u_dist(rng);
            int max_try = 100;
            while (idx2 == idx1 && max_try-- > 0) {
                idx2 = u_dist(rng);
            }
            if (idx1 == idx2) continue;

            int u1 = active_u[idx1];
            int u2 = active_u[idx2];

            int w1, w2;
            if (use_direct_select) {
                // 直接选模式：从全部Wf中随机选
                w1 = w_dist(rng);
                w2 = w_dist(rng);
                int max_try_w = 100;
                while (w2 == w1 && max_try_w-- > 0) {
                    w2 = w_dist(rng);
                }
                if (w1 == w2) continue;
            } else {
                // 候选对模式：从候选对中随机选
                std::uniform_int_distribution<int> p1_dist(0, (int)u_to_pairs[u1].size() - 1);
                std::uniform_int_distribution<int> p2_dist(0, (int)u_to_pairs[u2].size() - 1);
                int pair1_idx = u_to_pairs[u1][p1_dist(rng)];
                int pair2_idx = u_to_pairs[u2][p2_dist(rng)];
                w1 = pairs[pair1_idx].w_idx;
                w2 = pairs[pair2_idx].w_idx;
            }

            // 2点相似变换求解（与Python对齐）
            double du0 = U[u1 * 2]     - U[u2 * 2];
            double du1 = U[u1 * 2 + 1] - U[u2 * 2 + 1];
            double dw0 = Wf[w1 * 2]     - Wf[w2 * 2];
            double dw1 = Wf[w1 * 2 + 1] - Wf[w2 * 2 + 1];
            double norm_du = std::sqrt(du0 * du0 + du1 * du1);
            double norm_dw = std::sqrt(dw0 * dw0 + dw1 * dw1);
            if (norm_dw < 1e-12 || norm_du < 1e-12) continue;

            double s = norm_du / norm_dw;
            // s范围检查: 0.9 <= s <= 1.1
            if (s < 0.9 || s > 1.1) continue;

            double theta = std::atan2(du1, du0) - std::atan2(dw1, dw0);
            double ct = std::cos(theta), st = std::sin(theta);
            double tx = U[u1 * 2]     - s * (ct * Wf[w1 * 2]     - st * Wf[w1 * 2 + 1]);
            double ty = U[u1 * 2 + 1] - s * (st * Wf[w1 * 2]     + ct * Wf[w1 * 2 + 1]);

            // 应用变换到Wf → Wt
            apply_similarity(Wf, M, s, theta, tx, ty, Wt_local.data());

            // 用变换后的Wt统计内点（必须基于Wt建KDTree，不是Wf）
            auto inl = count_inliers_1to1(U, N, Wt_local.data(), M, tau);

            if (inl.n_inliers < min_inliers) continue;

            // 评分: score = n_inliers - 1.0 * rms（与Python对齐）
            double score = (double)inl.n_inliers - 1.0 * inl.rms;

            if (score > thread_bests[tid].score) {
                thread_bests[tid].s = s;
                thread_bests[tid].theta = theta;
                thread_bests[tid].tx = tx;
                thread_bests[tid].ty = ty;
                thread_bests[tid].n_inliers = inl.n_inliers;
                thread_bests[tid].rms = inl.rms;
                thread_bests[tid].score = score;
                thread_bests[tid].u1 = u1;
                thread_bests[tid].w1 = w1;
                thread_bests[tid].u2 = u2;
                thread_bests[tid].w2 = w2;
                thread_bests[tid].valid = true;
            }
        }
    }

    // 归约: 选全局最佳
    double global_best_score = -1e30;
    int best_tid = -1;
    for (int t = 0; t < n_threads; ++t) {
        if (thread_bests[t].valid && thread_bests[t].score > global_best_score) {
            global_best_score = thread_bests[t].score;
            best_tid = t;
        }
    }

    if (best_tid < 0 || thread_bests[best_tid].n_inliers < min_inliers) {
        fprintf(stderr, "[vm2] ransac_v2: 无有效结果\n");
        return best;
    }

    // 用最佳参数重建完整inlier_mask
    best.s = thread_bests[best_tid].s;
    best.theta = thread_bests[best_tid].theta;
    best.tx = thread_bests[best_tid].tx;
    best.ty = thread_bests[best_tid].ty;
    best.n_inliers = thread_bests[best_tid].n_inliers;
    best.rms = thread_bests[best_tid].rms;
    best.success = true;

    // 重建inlier_mask
    std::vector<double> Wt_best(M * 2);
    apply_similarity(Wf, M, best.s, best.theta, best.tx, best.ty, Wt_best.data());
    auto inl = count_inliers_1to1(U, N, Wt_best.data(), M, tau);
    best.inlier_mask = std::move(inl.inlier_mask);
    best.n_inliers = inl.n_inliers;
    best.rms = inl.rms;

    fprintf(stderr, "[vm2] ransac_v2: 最佳内点=%d rms=%.4f s=%.6f theta=%.4f° (threads=%d)\n",
            best.n_inliers, best.rms, best.s, best.theta * RADTODEG, n_threads);

    return best;
}

// ============================================================================
// iterative_svd_refine: 迭代SVD精修
// 与Python _iterative_svd_refine完全对齐:
//   1. 先用1.0*s0紧阈值重新统计内点
//   2. 不足3个时渐进放宽: 2*s0, 5*s0, 10*s0
//   3. 每次迭代: 用当前变换建立1对1配对 → Umeyama → 用1.0*s0重划内点
//   4. 收敛检查: mask不变则停止
//   5. 最终统计用1.0*s0
// ============================================================================

RansacResult iterative_svd_refine(const double* U, int N, const double* Wf, int M,
                                   double s, double theta, double tx, double ty,
                                   double s0, int max_iter)
{
    RansacResult result;
    result.s = s;
    result.theta = theta;
    result.tx = tx;
    result.ty = ty;
    result.n_inliers = 0;
    result.rms = 1e30;
    result.success = false;

    if (N < 3 || M < 3) return result;

    std::vector<double> Wt(M * 2);
    std::vector<int> prev_mask(N, 0);

    // 先用1.0*s0紧阈值重新统计内点
    double tau = 1.0 * s0;
    apply_similarity(Wf, M, s, theta, tx, ty, Wt.data());
    auto inl = count_inliers_1to1(U, N, Wt.data(), M, tau);
    fprintf(stderr, "[vm2] svd_refine: 初始紧阈值(1.0*s0=%.2f)内点=%d rms=%.4f\n",
            tau, inl.n_inliers, inl.rms);

    // 不足3个时渐进放宽
    double scale_factors[] = {2.0, 5.0, 10.0};
    for (int k = 0; k < 3 && inl.n_inliers < 3; ++k) {
        tau = scale_factors[k] * s0;
        inl = count_inliers_1to1(U, N, Wt.data(), M, tau);
        fprintf(stderr, "[vm2] svd_refine: 尝试%.0f*s0, 内点=%d\n",
                scale_factors[k], inl.n_inliers);
    }

    if (inl.n_inliers < 3) {
        fprintf(stderr, "[vm2] svd_refine: 内点不足3个 (%d), 无法精修\n",
                inl.n_inliers);
        result.inlier_mask = std::move(inl.inlier_mask);
        result.n_inliers = inl.n_inliers;
        result.rms = inl.rms;
        return result;
    }

    prev_mask = inl.inlier_mask;

    // 迭代SVD精修
    for (int iter = 0; iter < max_iter; ++iter) {
        if (result.s == 0.0) break; // 安全检查

        // 用当前变换建立1对1配对
        apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());

        // 建KDTree
        PointCloud2D cloud;
        cloud.pts.resize(M);
        for (int i = 0; i < M; ++i) {
            cloud.pts[i] = {Wt[i * 2], Wt[i * 2 + 1]};
        }
        KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

        double tau_match = 1.0 * s0;
        double tau_match_sq = tau_match * tau_match;

        // 1对1匹配
        struct Match {
            int u_idx;
            int w_idx;
            double dist;
        };
        std::vector<Match> candidates;
        for (int i = 0; i < N; ++i) {
            double query[2] = {U[i * 2], U[i * 2 + 1]};
            KDTreeIndexType idx;
            double dist_sq;
            nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
            rs.init(&idx, &dist_sq);
            tree.findNeighbors(rs, query);
            if (dist_sq <= tau_match_sq) {
                candidates.push_back({i, (int)idx, std::sqrt(dist_sq)});
            }
        }

        // 按距离排序, 贪心1对1
        std::sort(candidates.begin(), candidates.end(),
                  [](const Match& a, const Match& b) { return a.dist < b.dist; });
        std::vector<int> w_used(M, 0);
        std::vector<double> src_pts, dst_pts;
        for (auto& c : candidates) {
            if (w_used[c.w_idx]) continue;
            w_used[c.w_idx] = 1;
            // src = Wf原始坐标, dst = U坐标（与Python: umeyama(U_pts, W_pts)对齐）
            src_pts.push_back(Wf[c.w_idx * 2]);
            src_pts.push_back(Wf[c.w_idx * 2 + 1]);
            dst_pts.push_back(U[c.u_idx * 2]);
            dst_pts.push_back(U[c.u_idx * 2 + 1]);
        }

        int n_pairs = (int)src_pts.size() / 2;
        if (n_pairs < 3) {
            fprintf(stderr, "[vm2] svd_refine: 迭代%d配对不足 (%d)\n", iter, n_pairs);
            break;
        }

        // Umeyama SVD
        auto sim = umeyama(src_pts.data(), dst_pts.data(), n_pairs);
        if (!sim.valid) {
            fprintf(stderr, "[vm2] svd_refine: 迭代%d Umeyama失败\n", iter);
            break;
        }

        // 安全检查: s不应偏离1太远
        if (std::abs(sim.s - 1.0) > 0.1) {
            fprintf(stderr, "[vm2] svd_refine: 迭代%d s=%.4f偏离1太远，跳过\n",
                    iter, sim.s);
            break;
        }

        result.s = sim.s;
        result.theta = sim.theta;
        result.tx = sim.tx;
        result.ty = sim.ty;

        // 用1.0*s0重划内点
        apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());
        inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);

        fprintf(stderr, "[vm2] svd_refine: 迭代%d s=%.4f theta=%.2f° n=%d rms=%.4f\n",
                iter, result.s, result.theta * RADTODEG, inl.n_inliers, inl.rms);

        if (inl.n_inliers < 3) break;

        // 收敛检查
        bool converged = (inl.inlier_mask == prev_mask);
        prev_mask = inl.inlier_mask;

        if (converged) {
            fprintf(stderr, "[vm2] svd_refine: 收敛于迭代%d\n", iter);
            break;
        }
    }

    // 最终统计用1.0*s0
    apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());
    inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);

    result.n_inliers = inl.n_inliers;
    result.rms = inl.rms;
    result.inlier_mask = std::move(inl.inlier_mask);
    result.success = (result.n_inliers >= 3);

    return result;
}

// ============================================================================
// apply_flip: 翻转模式
// ============================================================================

void apply_flip(const double* W, int M, int mode, double* Wf)
{
    bool flip_x = (mode == 1 || mode == 3);
    bool flip_y = (mode == 2 || mode == 3);
    for (int i = 0; i < M; ++i) {
        Wf[i * 2]     = flip_x ? -W[i * 2]     : W[i * 2];
        Wf[i * 2 + 1] = flip_y ? -W[i * 2 + 1] : W[i * 2 + 1];
    }
}

// ============================================================================
// compute_normalized_score: 与Python _compute_normalized_score对齐
// norm_score = (n_inliers / min(N, M)) * (1 - rms / tau)
// ============================================================================

double compute_normalized_score(int n_inliers, double rms, int N_img, int M, double tau)
{
    double denom = std::min((double)N_img, (double)M);
    if (denom <= 0 || tau <= 0) return 0.0;
    return ((double)n_inliers / denom) * (1.0 - rms / tau);
}

// ============================================================================
// 单模式匹配: 翻转 + RANSAC + SVD精修
// ============================================================================

struct ModeResult {
    double s, theta, tx, ty;
    int n_inliers;
    double rms;
    double norm_score;
    std::vector<int> inlier_mask;
    bool success;
};

ModeResult solve_single_mode(const double* U, int N, const double* W, int M,
                              int mode, const double* sparsity,
                              double tau_coarse, int K, int min_inliers,
                              double candidate_radius, double s0, int seed,
                              volatile bool* early_exit)
{
    ModeResult mr;
    mr.success = false;
    mr.norm_score = 0.0;

    // 早退出检查: 其他模式已收敛
    if (early_exit && *early_exit) {
        fprintf(stderr, "[vm2] 模式%d: 跳过（其他模式已收敛）\n", mode);
        return mr;
    }

    // 应用翻转
    std::vector<double> Wf(M * 2);
    apply_flip(W, M, mode, Wf.data());

    fprintf(stderr, "[vm2] 模式%d: 星表向量组 %d 颗\n", mode, M);

    // RANSAC粗匹配
    auto ransac_res = ransac_v2(U, N, Wf.data(), M,
                                 tau_coarse, K, min_inliers,
                                 candidate_radius, sparsity, seed + mode,
                                 early_exit);

    if (!ransac_res.success) {
        fprintf(stderr, "[vm2] 模式%d: 粗匹配内点不足\n", mode);
        return mr;
    }

    // RANSAC后再次检查早退出
    if (early_exit && *early_exit) {
        fprintf(stderr, "[vm2] 模式%d: RANSAC后其他模式已收敛，跳过SVD精修\n", mode);
        return mr;
    }

    fprintf(stderr, "[vm2] 模式%d 粗匹配: s=%.4f theta=%.2f° n=%d rms=%.3f\n",
            mode, ransac_res.s, ransac_res.theta * RADTODEG,
            ransac_res.n_inliers, ransac_res.rms);

    // 迭代SVD精修
    auto refined = iterative_svd_refine(U, N, Wf.data(), M,
                                         ransac_res.s, ransac_res.theta,
                                         ransac_res.tx, ransac_res.ty,
                                         s0, 10);

    if (refined.success && refined.n_inliers >= min_inliers) {
        mr.s = refined.s;
        mr.theta = refined.theta;
        mr.tx = refined.tx;
        mr.ty = refined.ty;
        mr.n_inliers = refined.n_inliers;
        mr.rms = refined.rms;
        mr.inlier_mask = std::move(refined.inlier_mask);
        fprintf(stderr, "[vm2] 模式%d SVD精修: s=%.4f theta=%.2f° n=%d rms=%.3f\n",
                mode, mr.s, mr.theta * RADTODEG, mr.n_inliers, mr.rms);
    } else {
        // SVD精修失败，使用RANSAC结果
        mr.s = ransac_res.s;
        mr.theta = ransac_res.theta;
        mr.tx = ransac_res.tx;
        mr.ty = ransac_res.ty;
        mr.n_inliers = ransac_res.n_inliers;
        mr.rms = ransac_res.rms;
        mr.inlier_mask = std::move(ransac_res.inlier_mask);
    }

    // 计算norm_score（与Python对齐）
    mr.norm_score = compute_normalized_score(mr.n_inliers, mr.rms, N, M, tau_coarse);
    mr.success = true;

    // 收敛判定: norm_score >= 0.10 且 s在有效范围内 → 通知其他模式退出
    if (mr.success && mr.norm_score >= 0.10 && mr.s >= 0.9 && mr.s <= 1.1 && early_exit) {
        *early_exit = true;
        fprintf(stderr, "[vm2] 模式%d 收敛，通知其他模式退出\n", mode);
    }

    fprintf(stderr, "[vm2] 模式%d 最终: s=%.4f theta=%.2f° n=%d rms=%.3f norm_score=%.4f\n",
            mode, mr.s, mr.theta * RADTODEG, mr.n_inliers, mr.rms, mr.norm_score);

    return mr;
}

} // namespace vm2

// ============================================================================
// vm2_solve: 主入口 (C接口)
// 4种翻转模式OpenMP并行
// ============================================================================

extern "C" VM2_API int vm2_solve(
    const double* U, int N_img,
    const double* W, int M,
    const double* sparsity,
    const VM2SolveParams* params,
    VM2SolveResult* result)
{
    using namespace vm2;

    // 初始化结果
    result->s = 1.0;
    result->theta = 0.0;
    result->tx = 0.0;
    result->ty = 0.0;
    result->n_inliers = 0;
    result->rms = 1e30;
    result->best_mode = 0;
    result->norm_score = 0.0;
    result->success = 0;
    std::memset(result->inlier_mask, 0, sizeof(int) * N_img);

    if (N_img < 2 || M < 2) {
        fprintf(stderr, "[vm2] vm2_solve: 点数不足 N=%d M=%d\n", N_img, M);
        return -1;
    }

    int n_modes = params->n_modes;
    if (n_modes < 1) n_modes = 1;
    if (n_modes > 4) n_modes = 4;

    // 4种翻转模式OpenMP并行 + 早退出机制
    std::vector<ModeResult> mode_results(n_modes);
    volatile bool early_exit = false;

    #pragma omp parallel for schedule(static)
    for (int mode = 0; mode < n_modes; ++mode) {
        mode_results[mode] = solve_single_mode(
            U, N_img, W, M, mode, sparsity,
            params->tau_coarse, params->K, params->min_inliers,
            params->candidate_radius, params->s0, params->seed,
            &early_exit);
    }

    // 选最佳模式
    double best_score = -1.0;
    int best_mode = -1;
    for (int mode = 0; mode < n_modes; ++mode) {
        if (mode_results[mode].success && mode_results[mode].norm_score > best_score) {
            best_score = mode_results[mode].norm_score;
            best_mode = mode;
        }
    }

    if (best_mode < 0 || best_score < 0.10) {
        fprintf(stderr, "[vm2] vm2_solve: 所有模式失败 (best_score=%.4f)\n", best_score);
        return -1;
    }

    auto& best = mode_results[best_mode];

    // s范围检查
    if (best.s < 0.9 || best.s > 1.1) {
        fprintf(stderr, "[vm2] vm2_solve: s=%.6f 超出范围 [0.9, 1.1]\n", best.s);
        return -1;
    }

    // 填充结果
    result->s = best.s;
    result->theta = best.theta;
    result->tx = best.tx;
    result->ty = best.ty;
    result->n_inliers = best.n_inliers;
    result->rms = best.rms;
    result->best_mode = best_mode;
    result->norm_score = best_score;
    result->success = 1;
    for (int i = 0; i < N_img; ++i) {
        result->inlier_mask[i] = best.inlier_mask[i];
    }

    fprintf(stderr, "[vm2] vm2_solve: 成功 模式=%d 内点=%d rms=%.4f s=%.6f theta=%.4f° "
            "tx=%.4f ty=%.4f norm_score=%.4f\n",
            best_mode, result->n_inliers, result->rms, result->s,
            result->theta * RADTODEG, result->tx, result->ty, result->norm_score);

    return 0;
}

// ============================================================================
// vm2_svd_refine: SVD精修入口 (C接口)
// ============================================================================

extern "C" VM2_API int vm2_svd_refine(
    const double* U, int N_img,
    const double* W, int M,
    const int* inlier_mask,
    double s_init, double theta_init, double tx_init, double ty_init,
    double s0,
    int max_iter,
    VM2SolveResult* result)
{
    using namespace vm2;

    result->s = s_init;
    result->theta = theta_init;
    result->tx = tx_init;
    result->ty = ty_init;
    result->n_inliers = 0;
    result->rms = 1e30;
    result->success = 0;

    if (N_img < 3 || M < 3) return -1;

    auto refined = iterative_svd_refine(U, N_img, W, M,
                                         s_init, theta_init, tx_init, ty_init,
                                         s0, max_iter);

    result->s = refined.s;
    result->theta = refined.theta;
    result->tx = refined.tx;
    result->ty = refined.ty;
    result->n_inliers = refined.n_inliers;
    result->rms = refined.rms;
    result->success = refined.success ? 1 : 0;
    if (result->inlier_mask) {
        for (int i = 0; i < N_img; ++i) {
            result->inlier_mask[i] = refined.inlier_mask[i];
        }
    }

    return 0;
}

// ============================================================================
// vm2_count_inliers: 内点统计入口 (C接口)
// ============================================================================

extern "C" VM2_API int vm2_count_inliers(
    const double* U, int N_img,
    const double* W, int M,
    double s, double theta, double tx, double ty,
    double tau,
    int* inlier_mask,
    double* out_rms)
{
    using namespace vm2;

    std::vector<double> Wt(M * 2);
    apply_similarity(W, M, s, theta, tx, ty, Wt.data());

    auto inl = count_inliers_1to1(U, N_img, Wt.data(), M, tau);

    for (int i = 0; i < N_img; ++i) {
        inlier_mask[i] = inl.inlier_mask[i];
    }
    *out_rms = inl.rms;

    return inl.n_inliers;
}

/**
 * vm32_core.cpp - V3.2向量匹配算法C++核心实现
 *
 * 核心创新: SNR动态收紧搜索带宽
 *   - 不分固定阶段(预热/粗搜/精搜)，而是连续抽样+SNR驱动自动收紧
 *   - 初始: θ搜索范围大(±5°), s搜索范围大(±0.10)
 *   - 随着抽样进行, θ加权直方图峰值SNR提升
 *   - SNR达到阈值 → 自动收紧θ搜索范围到峰值附近
 *   - SNR继续提升 → 自动收紧s搜索范围
 *   - 带宽极窄时 → 密集抽样确定(tx,ty)
 *   - 收敛或达到N_max → 退出
 *
 * 算法原理:
 *   1点法: 1个(u_i, w_j)对即可确定变换参数(s, θ, tx, ty)
 *   - s = |u_i| / |w_j|
 *   - θ = atan2(u_i) - atan2(w_j)
 *   - tx, ty由变换方程自动确定
 *
 *   内点数加权直方图: 正确变换的内点数>>噪声变换, SNR从7x→426x
 *
 * 性能优化:
 *   1. 4种翻转模式OpenMP并行
 *   2. KDTree加速内点计数
 *   3. 预计算Wf旋转分量,避免重复三角函数
 *   4. 批量抽样+SNR评估,减少同步开销
 *   5. 早退出: SNR收敛即停止
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
#include <atomic>
#include <omp.h>

#include "../include/vm32_api.h"

// Eigen
#include "Eigen/Dense"

// nanoflann
#include "nanoflann.hpp"

namespace vm32 {

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
// count_inliers_fast: 快速内点计数（不做1对1互斥，用于抽样阶段）
// ============================================================================

struct FastInlierResult {
    int n_inliers;
    double rms;
};

FastInlierResult count_inliers_fast(const double* U, int N,
                                     const double* Wt, int M, double tau)
{
    FastInlierResult result;
    result.n_inliers = 0;
    result.rms = 0.0;

    if (N == 0 || M == 0) return result;

    // 构建KDTree
    PointCloud2D cloud;
    cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) {
        cloud.pts[i] = {Wt[i * 2], Wt[i * 2 + 1]};
    }
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    double tau_sq = tau * tau;
    double sum_sq = 0.0;
    int count = 0;

    for (int i = 0; i < N; ++i) {
        double query[2] = {U[i * 2], U[i * 2 + 1]};
        KDTreeIndexType idx;
        double dist_sq;
        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
        rs.init(&idx, &dist_sq);
        tree.findNeighbors(rs, query);
        if (dist_sq <= tau_sq) {
            sum_sq += dist_sq;
            count++;
        }
    }

    result.n_inliers = count;
    result.rms = (count > 0) ? std::sqrt(sum_sq / count) : 0.0;
    return result;
}

// ============================================================================
// count_inliers_1to1: 1对1互斥内点统计
// ============================================================================

struct InlierResult {
    int n_inliers;
    double rms;
    std::vector<int> inlier_mask;
};

InlierResult count_inliers_1to1(const double* U, int N, const double* Wt, int M, double tau)
{
    InlierResult result;
    result.n_inliers = 0;
    result.rms = 0.0;
    result.inlier_mask.assign(N, 0);

    if (N == 0 || M == 0) return result;

    PointCloud2D cloud;
    cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) {
        cloud.pts[i] = {Wt[i * 2], Wt[i * 2 + 1]};
    }
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

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

    std::sort(candidates.begin(), candidates.end(),
              [](const Match& a, const Match& b) { return a.dist < b.dist; });

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

// ============================================================================
// compute_normalized_score
// ============================================================================

double compute_normalized_score(int n_inliers, double rms, int N_img, int M, double tau)
{
    double denom = std::min((double)N_img, (double)M);
    if (denom <= 0 || tau <= 0) return 0.0;
    return ((double)n_inliers / denom) * (1.0 - rms / tau);
}

// ============================================================================
// umeyama: Umeyama SVD求解最优相似变换
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

    Vector2d src_mean = Vector2d::Zero();
    Vector2d dst_mean = Vector2d::Zero();
    for (int i = 0; i < n; ++i) {
        src_mean += Vector2d(src[i * 2], src[i * 2 + 1]);
        dst_mean += Vector2d(dst[i * 2], dst[i * 2 + 1]);
    }
    src_mean /= n;
    dst_mean /= n;

    Eigen::MatrixXd src_centered(2, n);
    Eigen::MatrixXd dst_centered(2, n);
    for (int i = 0; i < n; ++i) {
        src_centered(0, i) = src[i * 2]     - src_mean(0);
        src_centered(1, i) = src[i * 2 + 1] - src_mean(1);
        dst_centered(0, i) = dst[i * 2]     - dst_mean(0);
        dst_centered(1, i) = dst[i * 2 + 1] - dst_mean(1);
    }

    Matrix2d H = src_centered * dst_centered.transpose();

    Eigen::JacobiSVD<Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);

    double d = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    Vector2d S_vec = Vector2d::Ones(2);
    S_vec(1) = d;

    Matrix2d R = svd.matrixV() * S_vec.asDiagonal() * svd.matrixU().transpose();

    double trace_WtW = src_centered.colwise().squaredNorm().sum();
    double sigma_trace = (svd.singularValues().cwiseProduct(S_vec)).sum();

    if (trace_WtW < 1e-15) return result;

    double s = sigma_trace / trace_WtW;

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
// iterative_svd_refine: 迭代SVD精修
// ============================================================================

struct RefineResult {
    double s, theta, tx, ty;
    int n_inliers;
    double rms;
    std::vector<int> inlier_mask;
    bool success;
};

RefineResult iterative_svd_refine(const double* U, int N, const double* Wf, int M,
                                   double s, double theta, double tx, double ty,
                                   double s0, int max_iter)
{
    RefineResult result;
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
    fprintf(stderr, "[vm32] svd_refine: 初始内点=%d rms=%.4f (tau=%.2f)\n",
            inl.n_inliers, inl.rms, tau);

    // 不足3个时渐进放宽
    double scale_factors[] = {2.0, 5.0, 10.0};
    for (int k = 0; k < 3 && inl.n_inliers < 3; ++k) {
        tau = scale_factors[k] * s0;
        inl = count_inliers_1to1(U, N, Wt.data(), M, tau);
    }

    if (inl.n_inliers < 3) {
        result.inlier_mask = std::move(inl.inlier_mask);
        result.n_inliers = inl.n_inliers;
        result.rms = inl.rms;
        return result;
    }

    prev_mask = inl.inlier_mask;

    for (int iter = 0; iter < max_iter; ++iter) {
        apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());

        PointCloud2D cloud;
        cloud.pts.resize(M);
        for (int i = 0; i < M; ++i) {
            cloud.pts[i] = {Wt[i * 2], Wt[i * 2 + 1]};
        }
        KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

        double tau_match = 1.0 * s0;
        double tau_match_sq = tau_match * tau_match;

        struct Match { int u_idx; int w_idx; double dist; };
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

        std::sort(candidates.begin(), candidates.end(),
                  [](const Match& a, const Match& b) { return a.dist < b.dist; });
        std::vector<int> w_used(M, 0);
        std::vector<double> src_pts, dst_pts;
        for (auto& c : candidates) {
            if (w_used[c.w_idx]) continue;
            w_used[c.w_idx] = 1;
            src_pts.push_back(Wf[c.w_idx * 2]);
            src_pts.push_back(Wf[c.w_idx * 2 + 1]);
            dst_pts.push_back(U[c.u_idx * 2]);
            dst_pts.push_back(U[c.u_idx * 2 + 1]);
        }

        int n_pairs = (int)src_pts.size() / 2;
        if (n_pairs < 3) break;

        auto sim = umeyama(src_pts.data(), dst_pts.data(), n_pairs);
        if (!sim.valid || std::abs(sim.s - 1.0) > 0.1) break;

        result.s = sim.s;
        result.theta = sim.theta;
        result.tx = sim.tx;
        result.ty = sim.ty;

        apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());
        inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);

        if (inl.n_inliers < 3) break;

        bool converged = (inl.inlier_mask == prev_mask);
        prev_mask = inl.inlier_mask;

        if (converged) break;
    }

    apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());
    inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);

    result.n_inliers = inl.n_inliers;
    result.rms = inl.rms;
    result.inlier_mask = std::move(inl.inlier_mask);
    result.success = (result.n_inliers >= 3);

    return result;
}

// ============================================================================
// SNR动态收紧1点抽样法 - V3.2核心算法
//
// 算法流程:
//   1. 连续抽样1点(u_i, w_j)对, 计算变换参数(s, θ, tx, ty)
//   2. 统计内点数, 更新θ加权直方图和s加权直方图
//   3. 每batch_size次抽样后, 计算θ和s的峰值SNR
//   4. SNR达到θ收紧阈值 → 收紧θ搜索范围到峰值附近
//   5. SNR达到s收紧阈值 → 收紧s搜索范围到峰值附近
//   6. 带宽极窄时密集抽样确定(tx,ty)
//   7. 收敛(SNR >= snr_converge)或达到N_max → 退出
//
// 性能优化:
//   - 预计算Wf的旋转分量, 避免每次抽样重复三角函数
//   - 快速内点计数(不做1对1互斥)
//   - KDTree只在对(tx,ty)抽样时构建
//   - 批量处理减少同步开销
// ============================================================================

struct SamplingResult {
    double s, theta, tx, ty;
    int n_inliers;
    double rms;
    double peak_snr;
    int n_samples;
    std::vector<int> inlier_mask;
    bool success;
};

SamplingResult snr_adaptive_sampling(
    const double* U, int N, const double* Wf, int M,
    double tau, double s0,
    double s_min, double s_max,
    int n_max, int batch_size,
    double snr_theta_tighten, double snr_s_tighten, double snr_converge,
    double theta_band_init, double s_band_init,
    double theta_band_min, double s_band_min,
    int min_inliers,
    int seed,
    double fov_diag_asec)
{
    SamplingResult result;
    result.s = 1.0;
    result.theta = 0.0;
    result.tx = 0.0;
    result.ty = 0.0;
    result.n_inliers = 0;
    result.rms = 1e30;
    result.peak_snr = 0.0;
    result.n_samples = 0;
    result.success = false;

    if (N < 2 || M < 2) return result;

    // ── 预计算: U和Wf的模长和角度 ──
    std::vector<double> norm_U(N);
    std::vector<double> norm_Wf(M);
    std::vector<double> angle_U(N);  // atan2
    std::vector<double> angle_Wf(M);
    std::vector<bool> valid_U(N, false);
    std::vector<bool> valid_Wf(M, false);

    for (int i = 0; i < N; ++i) {
        norm_U[i] = std::sqrt(U[i*2]*U[i*2] + U[i*2+1]*U[i*2+1]);
        angle_U[i] = std::atan2(U[i*2+1], U[i*2]);
        valid_U[i] = (norm_U[i] > 1e-10);
    }
    for (int j = 0; j < M; ++j) {
        norm_Wf[j] = std::sqrt(Wf[j*2]*Wf[j*2] + Wf[j*2+1]*Wf[j*2+1]);
        angle_Wf[j] = std::atan2(Wf[j*2+1], Wf[j*2]);
        valid_Wf[j] = (norm_Wf[j] > 1e-10);
    }

    // ── 加权直方图 ──
    // θ直方图: 3600个bin, 每bin 0.1°
    static constexpr int THETA_BINS = 3600;
    static constexpr double THETA_BIN_WIDTH = 0.1; // 度
    std::vector<double> theta_hist(THETA_BINS, 0.0);

    // s直方图: 2000个bin, 每bin 0.0001
    static constexpr int S_BINS = 2000;
    static constexpr double S_BIN_WIDTH = 0.0001;
    std::vector<double> s_hist(S_BINS, 0.0);

    // ── SNR动态收紧的搜索范围 ──
    // 初始: 接受所有s∈[s_min,s_max]的对
    // 收紧后: 只接受(θ,s)在峰值附近的对
    double theta_center_deg = 0.0;    // θ搜索中心(度)
    double theta_band = theta_band_init;  // θ搜索半带宽(度)
    double s_center = 1.0;            // s搜索中心
    double s_band = s_band_init;      // s搜索半带宽

    bool theta_tightened = false;
    bool s_tightened = false;
    bool grid_mode = false;  // 是否切换到网格搜索模式

    // ── 最佳结果追踪 ──
    int best_n = 0;
    double best_s = 1.0, best_theta = 0.0, best_tx = 0.0, best_ty = 0.0;

    // ── 随机数生成器 ──
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u_dist(0, N - 1);
    std::uniform_int_distribution<int> w_dist(0, M - 1);

    // ── 预分配工作缓冲区 ──
    std::vector<double> Wt(M * 2);

    // ── 主抽样循环 ──
    // 核心逻辑:
    //   Phase A: 1点法抽样, 从(u_i,w_j)对计算(s,θ,tx,ty), SNR驱动收紧
    //   Phase B: 当θ和s都收紧后, 切换到网格搜索(θ,s), 抽样(tx,ty)
    int total_samples = 0;
    int n_valid_samples = 0;

    while (total_samples < n_max) {
        if (!grid_mode) {
            // ════════════════════════════════════════════════════════
            // Phase A: 1点法抽样 — 从(u_i,w_j)对计算变换参数
            // ════════════════════════════════════════════════════════
            int i = u_dist(rng);
            int j = w_dist(rng);
            total_samples++;

            if (!valid_U[i] || !valid_Wf[j]) continue;

            double s = norm_U[i] / norm_Wf[j];
            if (s < s_min || s > s_max) continue;

            double theta = angle_U[i] - angle_Wf[j];
            double theta_deg = std::fmod(std::fmod(theta * RADTODEG + 180.0, 360.0) + 360.0, 360.0) - 180.0;

            // SNR动态过滤
            if (theta_tightened) {
                double d_theta = std::abs(theta_deg - theta_center_deg);
                if (d_theta > 180.0) d_theta = 360.0 - d_theta;
                if (d_theta > theta_band) continue;
            }
            if (s_tightened) {
                if (std::abs(s - s_center) > s_band) continue;
            }

            double ct = std::cos(theta);
            double st = std::sin(theta);
            double tx = U[i*2] - s * (ct * Wf[j*2] - st * Wf[j*2+1]);
            double ty = U[i*2+1] - s * (st * Wf[j*2] + ct * Wf[j*2+1]);

            // tx,ty物理约束剪枝: 平移不应超过0.6倍FOV对角线
            double max_translation = fov_diag_asec * 0.6;
            if (std::abs(tx) > max_translation || std::abs(ty) > max_translation) continue;

            apply_similarity(Wf, M, s, theta, tx, ty, Wt.data());
            auto fir = count_inliers_fast(U, N, Wt.data(), M, tau);

            // 更新直方图
            int theta_bin = (int)((theta_deg + 180.0) / THETA_BIN_WIDTH);
            if (theta_bin >= 0 && theta_bin < THETA_BINS) {
                theta_hist[theta_bin] += fir.n_inliers;
            }
            int s_bin_idx = (int)((s - s_min) / S_BIN_WIDTH);
            if (s_bin_idx >= 0 && s_bin_idx < S_BINS) {
                s_hist[s_bin_idx] += fir.n_inliers;
            }

            if (fir.n_inliers > best_n) {
                best_n = fir.n_inliers;
                best_s = s;
                best_theta = theta;
                best_tx = tx;
                best_ty = ty;
            }

            n_valid_samples++;

            // 每batch_size次有效抽样后, 评估SNR并动态收紧
            if (n_valid_samples % batch_size == 0) {
                // θ峰值SNR
                int theta_peak_idx = 0;
                double theta_peak_val = 0.0;
                for (int b = 0; b < THETA_BINS; ++b) {
                    if (theta_hist[b] > theta_peak_val) {
                        theta_peak_val = theta_hist[b];
                        theta_peak_idx = b;
                    }
                }
                double theta_bg_sum = 0.0;
                int theta_bg_count = 0;
                for (int b = 0; b < THETA_BINS; ++b) {
                    if (std::abs(b - theta_peak_idx) > 5) {
                        theta_bg_sum += theta_hist[b];
                        theta_bg_count++;
                    }
                }
                double theta_bg_mean = (theta_bg_count > 10) ? theta_bg_sum / theta_bg_count : 1.0;
                double theta_snr = theta_peak_val / std::max(theta_bg_mean, 1e-10);

                // s峰值SNR
                int s_peak_idx = 0;
                double s_peak_val = 0.0;
                for (int b = 0; b < S_BINS; ++b) {
                    if (s_hist[b] > s_peak_val) {
                        s_peak_val = s_hist[b];
                        s_peak_idx = b;
                    }
                }
                double s_bg_sum = 0.0;
                int s_bg_count = 0;
                for (int b = 0; b < S_BINS; ++b) {
                    if (std::abs(b - s_peak_idx) > 5) {
                        s_bg_sum += s_hist[b];
                        s_bg_count++;
                    }
                }
                double s_bg_mean = (s_bg_count > 10) ? s_bg_sum / s_bg_count : 1.0;
                double s_snr = s_peak_val / std::max(s_bg_mean, 1e-10);

                double max_snr = std::max(theta_snr, s_snr);
                result.peak_snr = max_snr;

                double peak_theta_deg = (theta_peak_idx + 0.5) * THETA_BIN_WIDTH - 180.0;
                double peak_s_val = (s_peak_idx + 0.5) * S_BIN_WIDTH + s_min;

                // SNR驱动的动态收紧 — 只收紧θ, 不收紧s
                // (1点法的s受投影畸变影响偏差0.37%, 直方图峰值不可靠)
                if (!theta_tightened && theta_snr >= snr_theta_tighten) {
                    theta_center_deg = peak_theta_deg;
                    theta_band = std::max(theta_band_min, theta_band * 0.2);
                    theta_tightened = true;
                    fprintf(stderr, "[vm32] SNR=%.1fx → θ收紧: center=%.2f° band=%.2f°\n",
                            theta_snr, theta_center_deg, theta_band);
                }

                // 进一步收紧θ
                if (theta_tightened) {
                    theta_center_deg = peak_theta_deg;
                    if (max_snr > 3.0 * snr_theta_tighten && theta_band > theta_band_min) {
                        theta_band = std::max(theta_band_min, theta_band * 0.5);
                    }
                }

                // θ收紧后即切换到网格搜索(不依赖s直方图)
                if (theta_tightened && theta_band <= 1.0) {
                    grid_mode = true;
                    // s搜索范围: 使用全范围, 不依赖1点法s直方图
                    s_center = 1.0;
                    s_band = (s_max - s_min) / 2.0;
                    // θ搜索范围: 使用较宽的范围(±2°), 因为1点法θ峰值可能有偏差
                    theta_band = 2.0;
                    fprintf(stderr, "[vm32] 切换到网格搜索: θ=[%.2f°±%.2f°] s=[%.4f±%.4f] best_n=%d\n",
                            theta_center_deg, theta_band, s_center, s_band, best_n);
                    break;
                }

                fprintf(stderr, "[vm32] PhaseA: n_valid=%d n_total=%d θ_snr=%.1fx s_snr=%.1fx "
                        "θ_peak=%.2f° s_peak=%.4f best_n=%d θ_band=%.2f° s_band=%.4f\n",
                        n_valid_samples, total_samples, theta_snr, s_snr,
                        peak_theta_deg, peak_s_val, best_n, theta_band, s_band);
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    // Phase B: 网格搜索(θ,s) + Hough-like (tx,ty)估计
    //
    // 核心优化:
    //   1. KDTree缓存: 固定(θ,s)只建一次KDTree
    //   2. Hough-like (tx,ty): 用KDTree查询所有U点找近邻,
    //      从近邻对中估计(tx,ty), 替代K次随机抽样
    //   3. tx,ty物理约束: |tx|,|ty| > FOV对角线的变换直接跳过
    //   4. 内点数上界: 对固定(θ,s), 如果U和s*R*Wf无空间重叠则跳过
    // ════════════════════════════════════════════════════════════════
    if (grid_mode && total_samples < n_max) {
        // 网格参数
        double theta_step = 0.2;  // 粗搜θ步长(度)
        double s_step = 0.02;     // 粗搜s步长

        // 如果范围很小, 用更细的步长
        if (theta_band <= 0.3) theta_step = 0.1;
        if (s_band <= 0.005) s_step = 0.002;

        int n_theta = std::max(1, (int)(2.0 * theta_band / theta_step) + 1);
        int n_s = std::max(1, (int)(2.0 * s_band / s_step) + 1);

        // tx,ty物理约束: 平移不应超过FOV对角线
        double fov_diag_asec_local = fov_diag_asec;
        double max_translation = fov_diag_asec_local * 0.6;  // 0.6倍FOV对角线

        // (tx,ty) Hough直方图参数
        // bin大小 = tau/2, 范围 = ±max_translation
        double tx_bin_size = tau * 0.5;
        double ty_bin_size = tau * 0.5;
        int n_tx_bins = std::max(10, (int)(2.0 * max_translation / tx_bin_size) + 1);
        int n_ty_bins = std::max(10, (int)(2.0 * max_translation / ty_bin_size) + 1);

        fprintf(stderr, "[vm32] PhaseB: θ_grid=%d s_grid=%d Hough_bins=%dx%d total_combos=%d\n",
                n_theta, n_s, n_tx_bins, n_ty_bins, n_theta * n_s);

        for (int ti = 0; ti < n_theta && total_samples < n_max; ++ti) {
            double theta_deg = theta_center_deg - theta_band + ti * theta_step;
            double theta_rad = theta_deg * DEGTORAD;
            double ct = std::cos(theta_rad);
            double st = std::sin(theta_rad);

            // 预计算旋转后的Wf分量
            std::vector<double> Wf_rot_x(M), Wf_rot_y(M);
            for (int jj = 0; jj < M; ++jj) {
                Wf_rot_x[jj] = ct * Wf[jj*2] - st * Wf[jj*2+1];
                Wf_rot_y[jj] = st * Wf[jj*2] + ct * Wf[jj*2+1];
            }

            for (int si = 0; si < n_s && total_samples < n_max; ++si) {
                double s_val = s_center - s_band + si * s_step;
                if (s_val < s_min || s_val > s_max) continue;
                total_samples++;

                // ── 计算s*R*Wf (不含平移) ──
                std::vector<double> sR_Wf(M * 2);
                for (int jj = 0; jj < M; ++jj) {
                    sR_Wf[jj*2]   = s_val * Wf_rot_x[jj];
                    sR_Wf[jj*2+1] = s_val * Wf_rot_y[jj];
                }

                // ── 内点数上界检查: sR_Wf和U是否有空间重叠 ──
                double sR_min_x = 1e30, sR_max_x = -1e30;
                double sR_min_y = 1e30, sR_max_y = -1e30;
                for (int jj = 0; jj < M; ++jj) {
                    sR_min_x = std::min(sR_min_x, sR_Wf[jj*2]);
                    sR_max_x = std::max(sR_max_x, sR_Wf[jj*2]);
                    sR_min_y = std::min(sR_min_y, sR_Wf[jj*2+1]);
                    sR_max_y = std::max(sR_max_y, sR_Wf[jj*2+1]);
                }
                double U_min_x = 1e30, U_max_x = -1e30;
                double U_min_y = 1e30, U_max_y = -1e30;
                for (int ii = 0; ii < N; ++ii) {
                    U_min_x = std::min(U_min_x, U[ii*2]);
                    U_max_x = std::max(U_max_x, U[ii*2]);
                    U_min_y = std::min(U_min_y, U[ii*2+1]);
                    U_max_y = std::max(U_max_y, U[ii*2+1]);
                }
                // 检查: 是否存在(tx,ty)使sR_Wf+[tx,ty]与U重叠?
                // 需要: sR_max_x + tx >= U_min_x - tau AND sR_min_x + tx <= U_max_x + tau
                // 即: U_min_x - tau - sR_max_x <= tx <= U_max_x + tau - sR_min_x
                double tx_lo = U_min_x - tau - sR_max_x;
                double tx_hi = U_max_x + tau - sR_min_x;
                double ty_lo = U_min_y - tau - sR_max_y;
                double ty_hi = U_max_y + tau - sR_min_y;
                if (tx_lo > tx_hi || ty_lo > ty_hi) continue;  // 无重叠, 跳过

                // ── 构建KDTree (固定(θ,s)只建一次) ──
                PointCloud2D cloud_sR;
                cloud_sR.pts.resize(M);
                for (int jj = 0; jj < M; ++jj) {
                    cloud_sR.pts[jj] = {sR_Wf[jj*2], sR_Wf[jj*2+1]};
                }
                KDTree tree_sR(2, cloud_sR, nanoflann::KDTreeSingleIndexAdaptorParams(10));

                // ── Hough-like (tx,ty)估计 ──
                // 对每个U[i], 找sR_Wf中最近邻j, 如果距离<tau_large,
                // 则(tx,ty) = U[i] - sR_Wf[j] 是一个候选
                // tau_large要足够大: 1点法s偏差0.37%导致(tx,ty)偏移可达FOV*0.004
                double tau_large = std::max(tau * 5.0, fov_diag_asec * 0.005);
                double tau_large_sq = tau_large * tau_large;

                // tx,ty 1D直方图
                std::vector<double> tx_hist(n_tx_bins, 0.0);
                std::vector<double> ty_hist(n_ty_bins, 0.0);

                // 收集候选(tx,ty)及其内点数
                struct TxTyCandidate {
                    double tx, ty;
                    int n_inliers;
                };
                std::vector<TxTyCandidate> candidates;
                candidates.reserve(N);

                for (int ii = 0; ii < N; ++ii) {
                    double query[2] = {U[ii*2], U[ii*2+1]};
                    KDTreeIndexType idx;
                    double dist_sq;
                    nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
                    rs.init(&idx, &dist_sq);
                    tree_sR.findNeighbors(rs, query);

                    if (dist_sq > tau_large_sq) continue;

                    double tx = U[ii*2]   - sR_Wf[idx*2];
                    double ty = U[ii*2+1] - sR_Wf[idx*2+1];

                    // 物理约束剪枝: |tx|或|ty|超过FOV对角线
                    if (std::abs(tx) > max_translation || std::abs(ty) > max_translation) continue;

                    // 更新tx直方图
                    int tx_bin = (int)((tx + max_translation) / tx_bin_size);
                    if (tx_bin >= 0 && tx_bin < n_tx_bins) {
                        tx_hist[tx_bin] += 1.0;
                    }

                    // 更新ty直方图
                    int ty_bin = (int)((ty + max_translation) / ty_bin_size);
                    if (ty_bin >= 0 && ty_bin < n_ty_bins) {
                        ty_hist[ty_bin] += 1.0;
                    }

                    candidates.push_back({tx, ty, 0});
                }

                if (candidates.empty()) continue;

                // ── 找tx和ty的峰值 ──
                int tx_peak_bin = 0;
                double tx_peak_val = 0.0;
                for (int b = 0; b < n_tx_bins; ++b) {
                    if (tx_hist[b] > tx_peak_val) {
                        tx_peak_val = tx_hist[b];
                        tx_peak_bin = b;
                    }
                }
                double tx_peak = (tx_peak_bin + 0.5) * tx_bin_size - max_translation;

                int ty_peak_bin = 0;
                double ty_peak_val = 0.0;
                for (int b = 0; b < n_ty_bins; ++b) {
                    if (ty_hist[b] > ty_peak_val) {
                        ty_peak_val = ty_hist[b];
                        ty_peak_bin = b;
                    }
                }
                double ty_peak = (ty_peak_bin + 0.5) * ty_bin_size - max_translation;

                // ── 用峰值(tx,ty)做快速内点计数 ──
                // 不需要重建KDTree, 直接用已有的tree_sR
                // 查询: U[i] - [tx_peak, ty_peak] vs sR_Wf
                double tau_sq = tau * tau;
                int n_inliers_peak = 0;
                double sum_sq_peak = 0.0;

                for (int ii = 0; ii < N; ++ii) {
                    double query[2] = {U[ii*2] - tx_peak, U[ii*2+1] - ty_peak};
                    KDTreeIndexType idx;
                    double dist_sq;
                    nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
                    rs.init(&idx, &dist_sq);
                    tree_sR.findNeighbors(rs, query);
                    if (dist_sq <= tau_sq) {
                        n_inliers_peak++;
                        sum_sq_peak += dist_sq;
                    }
                }

                // ── 进一步精修(tx,ty): 在峰值附近搜索 ──
                // 用候选列表中接近峰值的(tx,ty)做内点计数
                double best_n_combo = n_inliers_peak;
                double best_tx_combo = tx_peak;
                double best_ty_combo = ty_peak;

                // 收集接近峰值的候选
                double refine_radius = tau * 2.0;
                for (auto& c : candidates) {
                    if (std::abs(c.tx - tx_peak) > refine_radius) continue;
                    if (std::abs(c.ty - ty_peak) > refine_radius) continue;

                    // 用这个(tx,ty)做内点计数
                    int n_inl = 0;
                    for (int ii = 0; ii < N; ++ii) {
                        double query[2] = {U[ii*2] - c.tx, U[ii*2+1] - c.ty};
                        KDTreeIndexType idx;
                        double dist_sq;
                        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
                        rs.init(&idx, &dist_sq);
                        tree_sR.findNeighbors(rs, query);
                        if (dist_sq <= tau_sq) n_inl++;
                    }

                    if (n_inl > best_n_combo) {
                        best_n_combo = n_inl;
                        best_tx_combo = c.tx;
                        best_ty_combo = c.ty;
                    }
                }

                if (best_n_combo > best_n) {
                    best_n = best_n_combo;
                    best_s = s_val;
                    best_theta = theta_rad;
                    best_tx = best_tx_combo;
                    best_ty = best_ty_combo;
                    fprintf(stderr, "[vm32] PhaseB: θ=%.2f° s=%.4f n=%d (best so far)\n",
                            theta_deg, s_val, best_n);
                }
            }
        }

        // 精搜: 在最佳(θ,s)附近更细的网格
        if (best_n >= min_inliers / 2 && total_samples < n_max) {
            double fine_theta_band = std::max(theta_band_min, 0.3);
            double fine_s_band = std::max(s_band_min, 0.005);
            double fine_theta_step = 0.1;
            double fine_s_step = 0.002;

            int n_fine_theta = std::max(1, (int)(2.0 * fine_theta_band / fine_theta_step) + 1);
            int n_fine_s = std::max(1, (int)(2.0 * fine_s_band / fine_s_step) + 1);

            double fine_theta_center = best_theta * RADTODEG;
            double fine_s_center = best_s;

            fprintf(stderr, "[vm32] PhaseB精搜: θ=[%.2f°±%.2f°] s=[%.4f±%.4f] grid=%dx%d\n",
                    fine_theta_center, fine_theta_band, fine_s_center, fine_s_band,
                    n_fine_theta, n_fine_s);

            for (int ti = 0; ti < n_fine_theta && total_samples < n_max; ++ti) {
                double theta_deg = fine_theta_center - fine_theta_band + ti * fine_theta_step;
                double theta_rad = theta_deg * DEGTORAD;
                double ct = std::cos(theta_rad);
                double st = std::sin(theta_rad);

                std::vector<double> Wf_rot_x(M), Wf_rot_y(M);
                for (int jj = 0; jj < M; ++jj) {
                    Wf_rot_x[jj] = ct * Wf[jj*2] - st * Wf[jj*2+1];
                    Wf_rot_y[jj] = st * Wf[jj*2] + ct * Wf[jj*2+1];
                }

                for (int si = 0; si < n_fine_s && total_samples < n_max; ++si) {
                    double s_val = fine_s_center - fine_s_band + si * fine_s_step;
                    if (s_val < s_min || s_val > s_max) continue;
                    total_samples++;

                    // 计算s*R*Wf
                    std::vector<double> sR_Wf(M * 2);
                    for (int jj = 0; jj < M; ++jj) {
                        sR_Wf[jj*2]   = s_val * Wf_rot_x[jj];
                        sR_Wf[jj*2+1] = s_val * Wf_rot_y[jj];
                    }

                    // 构建KDTree
                    PointCloud2D cloud_sR;
                    cloud_sR.pts.resize(M);
                    for (int jj = 0; jj < M; ++jj) {
                        cloud_sR.pts[jj] = {sR_Wf[jj*2], sR_Wf[jj*2+1]};
                    }
                    KDTree tree_sR(2, cloud_sR, nanoflann::KDTreeSingleIndexAdaptorParams(10));

                    // Hough-like (tx,ty)估计
                    double tau_large = std::max(tau * 5.0, fov_diag_asec * 0.005);
                    double tau_large_sq = tau_large * tau_large;
                    double tau_sq = tau * tau;

                    std::vector<double> tx_hist(n_tx_bins, 0.0);
                    std::vector<double> ty_hist(n_ty_bins, 0.0);
                    struct TxTyCandidate { double tx, ty; };
                    std::vector<TxTyCandidate> candidates;

                    for (int ii = 0; ii < N; ++ii) {
                        double query[2] = {U[ii*2], U[ii*2+1]};
                        KDTreeIndexType idx;
                        double dist_sq;
                        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
                        rs.init(&idx, &dist_sq);
                        tree_sR.findNeighbors(rs, query);
                        if (dist_sq > tau_large_sq) continue;

                        double tx = U[ii*2]   - sR_Wf[idx*2];
                        double ty = U[ii*2+1] - sR_Wf[idx*2+1];
                        if (std::abs(tx) > max_translation || std::abs(ty) > max_translation) continue;

                        int tx_bin = (int)((tx + max_translation) / tx_bin_size);
                        if (tx_bin >= 0 && tx_bin < n_tx_bins) tx_hist[tx_bin] += 1.0;
                        int ty_bin = (int)((ty + max_translation) / ty_bin_size);
                        if (ty_bin >= 0 && ty_bin < n_ty_bins) ty_hist[ty_bin] += 1.0;

                        candidates.push_back({tx, ty});
                    }

                    if (candidates.empty()) continue;

                    // 找峰值
                    int tx_peak_bin = 0; double tx_peak_val = 0.0;
                    for (int b = 0; b < n_tx_bins; ++b) if (tx_hist[b] > tx_peak_val) { tx_peak_val = tx_hist[b]; tx_peak_bin = b; }
                    double tx_peak = (tx_peak_bin + 0.5) * tx_bin_size - max_translation;

                    int ty_peak_bin = 0; double ty_peak_val = 0.0;
                    for (int b = 0; b < n_ty_bins; ++b) if (ty_hist[b] > ty_peak_val) { ty_peak_val = ty_hist[b]; ty_peak_bin = b; }
                    double ty_peak = (ty_peak_bin + 0.5) * ty_bin_size - max_translation;

                    // 用峰值(tx,ty)做内点计数
                    int best_n_combo = 0;
                    double best_tx_combo = tx_peak, best_ty_combo = ty_peak;

                    for (int ii = 0; ii < N; ++ii) {
                        double query[2] = {U[ii*2] - tx_peak, U[ii*2+1] - ty_peak};
                        KDTreeIndexType idx;
                        double dist_sq;
                        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
                        rs.init(&idx, &dist_sq);
                        tree_sR.findNeighbors(rs, query);
                        if (dist_sq <= tau_sq) best_n_combo++;
                    }

                    // 精修: 在峰值附近搜索候选
                    double refine_radius = tau * 2.0;
                    for (auto& c : candidates) {
                        if (std::abs(c.tx - tx_peak) > refine_radius) continue;
                        if (std::abs(c.ty - ty_peak) > refine_radius) continue;
                        int n_inl = 0;
                        for (int ii = 0; ii < N; ++ii) {
                            double query[2] = {U[ii*2] - c.tx, U[ii*2+1] - c.ty};
                            KDTreeIndexType idx;
                            double dist_sq;
                            nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
                            rs.init(&idx, &dist_sq);
                            tree_sR.findNeighbors(rs, query);
                            if (dist_sq <= tau_sq) n_inl++;
                        }
                        if (n_inl > best_n_combo) {
                            best_n_combo = n_inl;
                            best_tx_combo = c.tx;
                            best_ty_combo = c.ty;
                        }
                    }

                    if (best_n_combo > best_n) {
                        best_n = best_n_combo;
                        best_s = s_val;
                        best_theta = theta_rad;
                        best_tx = best_tx_combo;
                        best_ty = best_ty_combo;
                        fprintf(stderr, "[vm32] PhaseB精搜: θ=%.2f° s=%.4f n=%d (best)\n",
                                theta_deg, s_val, best_n);
                    }
                }
            }
        }
    }

    result.n_samples = total_samples;

    if (best_n < min_inliers) {
        fprintf(stderr, "[vm32] 抽样完成: best_n=%d < min_inliers=%d, 失败\n",
                best_n, min_inliers);
        return result;
    }

    // ── 用最佳变换做1对1互斥内点统计 ──
    apply_similarity(Wf, M, best_s, best_theta, best_tx, best_ty, Wt.data());
    auto inl = count_inliers_1to1(U, N, Wt.data(), M, tau);

    result.s = best_s;
    result.theta = best_theta;
    result.tx = best_tx;
    result.ty = best_ty;
    result.n_inliers = inl.n_inliers;
    result.rms = inl.rms;
    result.inlier_mask = std::move(inl.inlier_mask);
    result.success = (result.n_inliers >= min_inliers);

    fprintf(stderr, "[vm32] 抽样完成: s=%.4f θ=%.2f° n=%d rms=%.3f SNR=%.1fx samples=%d\n",
            result.s, result.theta * RADTODEG, result.n_inliers, result.rms,
            result.peak_snr, result.n_samples);

    return result;
}

// ============================================================================
// 单模式匹配: 翻转 + SNR动态收紧抽样 + SVD精修
// ============================================================================

struct ModeResult {
    double s, theta, tx, ty;
    int n_inliers;
    double rms;
    double norm_score;
    std::vector<int> inlier_mask;
    bool success;
    double peak_snr;
    int n_samples;
};

ModeResult solve_single_mode(const double* U, int N, const double* W, int M,
                              int mode, double tau, double s0,
                              double s_min, double s_max,
                              int n_max, int batch_size,
                              double snr_theta_tighten, double snr_s_tighten,
                              double snr_converge,
                              double theta_band_init, double s_band_init,
                              double theta_band_min, double s_band_min,
                              int min_inliers,
                              int seed,
                              double fov_diag_asec,
                              volatile std::atomic<bool>* early_exit)
{
    ModeResult mr;
    mr.success = false;
    mr.norm_score = 0.0;
    mr.peak_snr = 0.0;
    mr.n_samples = 0;

    if (early_exit && early_exit->load(std::memory_order_relaxed)) {
        fprintf(stderr, "[vm32] 模式%d: 跳过（其他模式已收敛）\n", mode);
        return mr;
    }

    // 应用翻转
    std::vector<double> Wf(M * 2);
    apply_flip(W, M, mode, Wf.data());

    fprintf(stderr, "[vm32] 模式%d: 星表向量组 %d 颗\n", mode, M);

    // SNR动态收紧1点抽样
    auto sr = snr_adaptive_sampling(
        U, N, Wf.data(), M, tau, s0,
        s_min, s_max,
        n_max, batch_size,
        snr_theta_tighten, snr_s_tighten, snr_converge,
        theta_band_init, s_band_init,
        theta_band_min, s_band_min,
        min_inliers, seed + mode,
        fov_diag_asec
    );

    mr.peak_snr = sr.peak_snr;
    mr.n_samples = sr.n_samples;

    if (!sr.success) {
        fprintf(stderr, "[vm32] 模式%d: 抽样失败\n", mode);
        return mr;
    }

    fprintf(stderr, "[vm32] 模式%d 粗匹配: s=%.4f θ=%.2f° n=%d rms=%.3f SNR=%.1fx\n",
            mode, sr.s, sr.theta * RADTODEG, sr.n_inliers, sr.rms, sr.peak_snr);

    // SVD精修
    auto refined = iterative_svd_refine(U, N, Wf.data(), M,
                                         sr.s, sr.theta, sr.tx, sr.ty,
                                         s0, 10);

    if (refined.success && refined.n_inliers >= min_inliers) {
        mr.s = refined.s;
        mr.theta = refined.theta;
        mr.tx = refined.tx;
        mr.ty = refined.ty;
        mr.n_inliers = refined.n_inliers;
        mr.rms = refined.rms;
        mr.inlier_mask = std::move(refined.inlier_mask);
        fprintf(stderr, "[vm32] 模式%d SVD精修: s=%.4f θ=%.2f° n=%d rms=%.3f\n",
                mode, mr.s, mr.theta * RADTODEG, mr.n_inliers, mr.rms);
    } else {
        mr.s = sr.s;
        mr.theta = sr.theta;
        mr.tx = sr.tx;
        mr.ty = sr.ty;
        mr.n_inliers = sr.n_inliers;
        mr.rms = sr.rms;
        mr.inlier_mask = std::move(sr.inlier_mask);
    }

    mr.norm_score = compute_normalized_score(mr.n_inliers, mr.rms, N, M, tau);
    mr.success = true;

    // 收敛判定
    if (mr.success && mr.norm_score >= 0.10 && mr.s >= 0.9 && mr.s <= 1.1 && early_exit) {
        early_exit->store(true, std::memory_order_relaxed);
        fprintf(stderr, "[vm32] 模式%d 收敛，通知其他模式退出\n", mode);
    }

    fprintf(stderr, "[vm32] 模式%d 最终: s=%.4f θ=%.2f° n=%d rms=%.3f norm_score=%.4f\n",
            mode, mr.s, mr.theta * RADTODEG, mr.n_inliers, mr.rms, mr.norm_score);

    return mr;
}

} // namespace vm32

// ============================================================================
// vm32_solve: 主入口 (C接口)
// 4种翻转模式OpenMP并行 + SNR动态收紧
// ============================================================================

extern "C" VM32_API int vm32_solve(
    const double* U, int N_img,
    const double* W, int M,
    const VM32SolveParams* params,
    VM32SolveResult* result)
{
    using namespace vm32;

    // 初始化结果
    result->s = 1.0;
    result->theta = 0.0;
    result->tx = 0.0;
    result->ty = 0.0;
    result->n_inliers = 0;
    result->rms = 1e30;
    result->best_mode = 0;
    result->norm_score = 0.0;
    result->peak_snr = 0.0;
    result->n_samples = 0;
    result->success = 0;
    std::memset(result->inlier_mask, 0, sizeof(int) * N_img);

    if (N_img < 2 || M < 2) {
        fprintf(stderr, "[vm32] vm32_solve: 点数不足 N=%d M=%d\n", N_img, M);
        return -1;
    }

    int n_modes = params->n_modes;
    if (n_modes < 1) n_modes = 1;
    if (n_modes > 4) n_modes = 4;

    // 4种翻转模式OpenMP并行 + 早退出机制
    std::vector<ModeResult> mode_results(n_modes);
    std::atomic<bool> early_exit(false);

    #pragma omp parallel for schedule(static)
    for (int mode = 0; mode < n_modes; ++mode) {
        mode_results[mode] = solve_single_mode(
            U, N_img, W, M, mode,
            params->tau, params->s0,
            params->s_min, params->s_max,
            params->n_max, params->batch_size,
            params->snr_theta_tighten, params->snr_s_tighten,
            params->snr_converge,
            params->theta_band_init, params->s_band_init,
            params->theta_band_min, params->s_band_min,
            params->min_inliers,
            params->seed,
            params->fov_diag_asec,
            &early_exit
        );
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
        fprintf(stderr, "[vm32] vm32_solve: 所有模式失败 (best_score=%.4f)\n", best_score);
        return -1;
    }

    auto& best = mode_results[best_mode];

    if (best.s < 0.9 || best.s > 1.1) {
        fprintf(stderr, "[vm32] vm32_solve: s=%.6f 超出范围 [0.9, 1.1]\n", best.s);
        return -1;
    }

    result->s = best.s;
    result->theta = best.theta;
    result->tx = best.tx;
    result->ty = best.ty;
    result->n_inliers = best.n_inliers;
    result->rms = best.rms;
    result->best_mode = best_mode;
    result->norm_score = best_score;
    result->peak_snr = best.peak_snr;
    result->n_samples = best.n_samples;
    result->success = 1;
    for (int i = 0; i < N_img; ++i) {
        result->inlier_mask[i] = best.inlier_mask[i];
    }

    fprintf(stderr, "[vm32] vm32_solve: 成功 模式=%d 内点=%d rms=%.4f s=%.6f theta=%.4f° "
            "tx=%.4f ty=%.4f norm_score=%.4f SNR=%.1fx samples=%d\n",
            best_mode, result->n_inliers, result->rms, result->s,
            result->theta * RADTODEG, result->tx, result->ty, result->norm_score,
            result->peak_snr, result->n_samples);

    return 0;
}

// ============================================================================
// vm32_svd_refine: SVD精修入口 (C接口)
// ============================================================================

extern "C" VM32_API int vm32_svd_refine(
    const double* U, int N_img,
    const double* W, int M,
    const int* inlier_mask,
    double s_init, double theta_init, double tx_init, double ty_init,
    double s0,
    int max_iter,
    VM32SolveResult* result)
{
    using namespace vm32;

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
// vm32_count_inliers: 内点统计入口 (C接口)
// ============================================================================

extern "C" VM32_API int vm32_count_inliers(
    const double* U, int N_img,
    const double* W, int M,
    double s, double theta, double tx, double ty,
    double tau,
    int* inlier_mask,
    double* out_rms)
{
    using namespace vm32;

    std::vector<double> Wt(M * 2);
    apply_similarity(W, M, s, theta, tx, ty, Wt.data());

    auto inl = count_inliers_1to1(U, N_img, Wt.data(), M, tau);

    for (int i = 0; i < N_img; ++i) {
        inlier_mask[i] = inl.inlier_mask[i];
    }
    *out_rms = inl.rms;

    return inl.n_inliers;
}

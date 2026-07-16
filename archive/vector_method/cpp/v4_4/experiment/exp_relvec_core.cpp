// ============================================================================
// exp_relvec_core.cpp - V4.4 向量法抽样核心算法 (独立实现, 重新设计)
//
// 详见 exp_relvec_core.h 算法说明
// ============================================================================

#include "exp_relvec_core.h"
#include "vm44_log.h"  // 复用主程序 v44::Logger

#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <limits>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <unordered_map>
#include <sstream>
#include <set>

#include <Eigen/Dense>

using v44::Logger;

namespace exp44 {

// ============================================================================
// 常量
// ============================================================================
static constexpr double EXP_PI = 3.14159265358979323846;
static constexpr double EXP_RADTODEG = 180.0 / EXP_PI;
static constexpr double EXP_DEGTORAD = EXP_PI / 180.0;

using Clock = std::chrono::steady_clock;
using ms_double = std::chrono::duration<double, std::milli>;

// ============================================================================
// 默认参数
// ============================================================================
RelVecParams getDefaultRelVecParams() {
    RelVecParams p;
    // 采样
    p.max_samples = 5000;
    p.max_u = 100;
    p.seed = 42;
    // s 处理 (±10% 容差)
    p.s_min = 0.9;
    p.s_max = 1.1;
    // k-vector
    p.min_len_frac = 0.05;
    p.max_len_frac = 0.8;
    // 第三星验证
    p.n_third_stars = 10;
    p.third_star_tol_px = 1.5;
    p.max_cand = 500;
    // 3D 密度场
    p.th_bins = 360;
    p.dxdy_bins = 200;
    p.peak_cluster_half = 2;
    // 递归聚焦
    p.min_samples = 200;
    p.check_interval = 100;
    p.snr_threshold = 10.0;
    p.focus_th_half = 5.0;       // θ ±5° (放宽, 容纳 θ 偏差)
    p.focus_dxdy_half = 120.0;   // tx/ty ±120" (放宽, 容纳初始中位数偏差 + s_est 误差)
    p.focus_shrink_interval = 200;
    p.focus_shrink_factor = 0.4;
    // 自适应停止
    p.adaptive_stop = 1;
    p.snr_eps = 0.05;
    p.max_stable = 3;
    p.focus_target_n_candidates = 50;  // 聚焦区候选数目标
    p.snr_final_threshold = 5.0;       // 收敛 SNR 阈值
    // 相似度加权
    p.use_similarity_weight = 1;
    p.similarity_knn = 3;
    return p;
}

// ============================================================================
// 工具函数
// ============================================================================
static inline double wrap180(double d) {
    return std::fmod(std::fmod(d + 180.0, 360.0) + 360.0, 360.0) - 180.0;
}

static inline double angle_diff_deg(double a, double b) {
    return std::abs(wrap180(a - b));
}

// 中位数
static double vec_median(std::vector<double> v) {
    size_t n = v.size();
    if (n == 0) return 0;
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
        return (v[n / 2] + v[n / 2 - 1]) * 0.5;
    }
    return v[n / 2];
}

// ============================================================================
// 相似度加权计算
//   对每个候选 (i,j,a,b,θ,s_est,tx,ty):
//   1. 构造变换后 W' = s_est·R(θ)·W + (tx, ty)
//   2. 对 U 中每颗星, 找 W' 中 KNN 邻居
//   3. 相似度 = KNN 内匹配数 / K (基于距离阈值)
//   4. 相似度高 = 变换正确 = 高投票权重
//
// 简化版 (避免全量构造 W' 的开销):
//   只对采样星对 (i,j) 验证: U[i]↔W'[a], U[j]↔W'[b] 是否在 KNN 邻域内
//   similarity = (matched_inliers / n_check) ∈ [0, 1]
// ============================================================================
static double compute_similarity(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    int i, int j, int a, int b,
    double s_est, double theta_rad, double tx, double ty,
    double s0, int knn,
    const std::vector<double>& D_U_row_i,  // U[i] 到其他星的距离
    const std::vector<double>& D_U_row_j,  // U[j] 到其他星的距离
    const std::vector<std::pair<double, int>>& D_W_sorted_a,  // W[a] 距离索引
    const std::vector<std::pair<double, int>>& D_W_sorted_b,  // W[b] 距离索引
    const std::vector<double>& D_W_row_a,  // W[a] 到其他星的距离
    const std::vector<double>& D_W_row_b,  // W[b] 到其他星的距离
    Logger* logger = nullptr)
{
    (void)logger;
    // 变换后: W'[k] = s_est·R(θ)·W[k] + (tx, ty)
    // 距离保持: |W'[k] - W'[l]| = s_est·|W[k] - W[l]| (旋转+平移保距)
    // 因此 U[i]↔W'[a] 的距离关系:
    //   |U[i] - W'[a]| = 0 (真匹配, 忽略噪声)
    //   |U[i] - W'[l]| = s_est·|W[a] - W[l]| (对任意 l)

    // 验证: U[i] 的 KNN 距离应与 W[a] 的 KNN 距离×s_est 一致
    // 取 U[i] 前 knn 个邻居, 检查 W[a] 前 knn 个邻居距离×s_est 是否匹配

    int N_u = (int)U.size();
    int N_w = (int)W.size();
    if (N_u < knn + 1 || N_w < knn + 1) return 1.0;  // 数据太少, 默认权重 1

    // U[i] 到其他星的距离 (排除 i 自身)
    std::vector<double> d_u_i;
    d_u_i.reserve(N_u - 1);
    for (int k = 0; k < N_u; ++k) {
        if (k == i) continue;
        d_u_i.push_back(D_U_row_i[k]);
    }
    std::sort(d_u_i.begin(), d_u_i.end());

    // W[a] 到其他星的距离 (排除 a 自身)
    std::vector<double> d_w_a;
    d_w_a.reserve(N_w - 1);
    for (int k = 0; k < N_w; ++k) {
        if (k == a) continue;
        d_w_a.push_back(D_W_row_a[k]);
    }
    std::sort(d_w_a.begin(), d_w_a.end());

    // 比较 U[i] 的 KNN 距离 vs s_est×W[a] 的 KNN 距离
    // 容差: 2×s0 (2 像素, 噪声容忍)
    double tol = 2.0 * s0;
    int matched = 0;
    int n_check = std::min(knn, std::min((int)d_u_i.size(), (int)d_w_a.size()));
    for (int k = 0; k < n_check; ++k) {
        double d_u = d_u_i[k];
        double d_w = d_w_a[k] * s_est;
        if (std::abs(d_u - d_w) < tol) matched++;
    }

    // 同理 U[j] ↔ W[b]
    std::vector<double> d_u_j;
    d_u_j.reserve(N_u - 1);
    for (int k = 0; k < N_u; ++k) {
        if (k == j) continue;
        d_u_j.push_back(D_U_row_j[k]);
    }
    std::sort(d_u_j.begin(), d_u_j.end());

    std::vector<double> d_w_b;
    d_w_b.reserve(N_w - 1);
    for (int k = 0; k < N_w; ++k) {
        if (k == b) continue;
        d_w_b.push_back(D_W_row_b[k]);
    }
    std::sort(d_w_b.begin(), d_w_b.end());

    int matched2 = 0;
    int n_check2 = std::min(knn, std::min((int)d_u_j.size(), (int)d_w_b.size()));
    for (int k = 0; k < n_check2; ++k) {
        double d_u = d_u_j[k];
        double d_w = d_w_b[k] * s_est;
        if (std::abs(d_u - d_w) < tol) matched2++;
    }

    // 综合: (matched + matched2) / (n_check + n_check2)
    double total_check = n_check + n_check2;
    if (total_check <= 0) return 1.0;
    double sim = (double)(matched + matched2) / total_check;
    return sim;
}

// ============================================================================
// 3D 峰值检测 (5×5×5 邻域累加, θ 环形)
// ============================================================================
static void detect_peak_3d(
    const std::unordered_map<uint64_t, int>& density3d,
    int total_votes_3d,
    int th_bins, int dxdy_bins,
    int cluster_half,
    int& peak_th, int& peak_tx, int& peak_ty,
    int& peak_cluster, double& snr,
    Logger* logger = nullptr)
{
    (void)logger;
    peak_cluster = 0;
    peak_th = 0; peak_tx = 0; peak_ty = 0;
    int h = cluster_half;

    for (auto& kv : density3d) {
        uint64_t key = kv.first;
        int th = (int)(key / ((uint64_t)dxdy_bins * dxdy_bins));
        int tx = (int)((key / (uint64_t)dxdy_bins) % (uint64_t)dxdy_bins);
        int ty = (int)(key % (uint64_t)dxdy_bins);

        int cluster = 0;
        for (int dt = -h; dt <= h; ++dt) {
            int t2 = th + dt;
            if (t2 < 0) t2 += th_bins;
            if (t2 >= th_bins) t2 -= th_bins;
            int txlo = std::max(0, tx - h), txhi = std::min(dxdy_bins - 1, tx + h);
            int tylo = std::max(0, ty - h), tyhi = std::min(dxdy_bins - 1, ty + h);
            for (int tx2 = txlo; tx2 <= txhi; ++tx2) {
                for (int ty2 = tylo; ty2 <= tyhi; ++ty2) {
                    uint64_t k2 = ((uint64_t)t2 * dxdy_bins + tx2) * dxdy_bins + ty2;
                    auto it = density3d.find(k2);
                    if (it != density3d.end()) cluster += it->second;
                }
            }
        }
        if (cluster > peak_cluster) {
            peak_cluster = cluster;
            peak_th = th; peak_tx = tx; peak_ty = ty;
        }
    }

    if (total_votes_3d > 0 && !density3d.empty()) {
        double bg = (double)total_votes_3d / (double)density3d.size();
        snr = (double)peak_cluster / std::max(bg, 1.0);
    } else {
        snr = 0;
    }
}

// ============================================================================
// 聚焦区域
// ============================================================================
struct FocusRegion {
    double th_lo, th_hi;
    double tx_lo, tx_hi;
    double ty_lo, ty_hi;
    double th_center, tx_center, ty_center;  // 固定中心 (初始确认时设置, 收紧时不移动)
    bool confirmed;
};

// ============================================================================
// Umeyama SVD 拟合 (一步求解最优 SimTransform)
//   输入: {U_k, W_k} 点对 (U = 图像星角秒, W = Gaia 星角秒)
//   模型: U = s·R(θ)·W + (tx, ty)
//   输出: SimTransform(s, θ, tx, ty)
//   参考: Umeyama 1991, "Least-squares estimation of transformation parameters
//         between two point patterns"
// ============================================================================
static SimTransform umeyama_svd(const std::vector<StarPoint>& U,
                                 const std::vector<StarPoint>& W,
                                 const std::vector<int>& u_idx,
                                 const std::vector<int>& w_idx,
                                 Logger* logger = nullptr)
{
    SimTransform result{};
    result.valid = false;
    int n = (int)u_idx.size();
    if (n < 2) return result;

    // 1. 计算质心
    Eigen::Vector2d mu_U = Eigen::Vector2d::Zero();
    Eigen::Vector2d mu_W = Eigen::Vector2d::Zero();
    for (int k = 0; k < n; ++k) {
        mu_U += Eigen::Vector2d(U[u_idx[k]].x, U[u_idx[k]].y);
        mu_W += Eigen::Vector2d(W[w_idx[k]].x, W[w_idx[k]].y);
    }
    mu_U /= n; mu_W /= n;

    // 2. 中心化协方差 Σ = Σ W'_k · U'_k^T / n  (2x2)
    Eigen::Matrix2d Sigma = Eigen::Matrix2d::Zero();
    double var_W = 0.0;
    for (int k = 0; k < n; ++k) {
        Eigen::Vector2d wp(W[w_idx[k]].x - mu_W.x(), W[w_idx[k]].y - mu_W.y());
        Eigen::Vector2d up(U[u_idx[k]].x - mu_U.x(), U[u_idx[k]].y - mu_U.y());
        Sigma += wp * up.transpose();
        var_W += wp.squaredNorm();
    }
    Sigma /= n;
    var_W /= n;
    if (var_W < 1e-12) return result;

    // 3. SVD: Σ = U_svd · S · V^T
    Eigen::JacobiSVD<Eigen::Matrix2d> svd(Sigma, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d U_svd = svd.matrixU();
    Eigen::Matrix2d V_svd = svd.matrixV();
    Eigen::Vector2d S = svd.singularValues();

    // 4. 旋转 R = V_svd · diag(1, det(V·U^T)) · U_svd^T
    //    (标准 Umeyama: R = V·diag·U^T, 使 U ≈ R·W)
    Eigen::Matrix2d diag_corr = Eigen::Matrix2d::Identity();
    double det_corr = (V_svd * U_svd.transpose()).determinant();
    if (det_corr < 0) diag_corr(1, 1) = -1.0;
    Eigen::Matrix2d R = V_svd * diag_corr * U_svd.transpose();

    // 5. 尺度 s = trace(diag_corr · S) / var_W
    double s = (diag_corr * S.asDiagonal()).trace() / var_W;

    // 6. 平移 t = μ_U - s·R·μ_W
    Eigen::Vector2d t = mu_U - s * R * mu_W;

    // 7. 提取 θ (从 R 矩阵)
    //   R = [cos -sin; sin cos] → θ = atan2(R21, R11)
    double theta = std::atan2(R(1, 0), R(0, 0));

    result.s = s;
    result.theta = theta;
    result.tx = t.x();
    result.ty = t.y();
    result.valid = true;

    if (logger) {
        logger->info("Umeyama SVD: n=" + std::to_string(n) +
                     " s=" + std::to_string(s) +
                     " θ=" + std::to_string(theta * EXP_RADTODEG) + "°" +
                     " tx=" + std::to_string(t.x()) +
                     " ty=" + std::to_string(t.y()));
    }
    return result;
}

// ============================================================================
// RANSAC 一步求解 SimTransform
//   输入: 候选点对集合 candidate_pairs (由密度场聚集区 θ 过滤得到)
//   流程:
//     1. 随机抽 2 对, 解析求 (s, θ, tx, ty)
//     2. 全部候选验证 inliers (位置残差 < tol)
//     3. 重复 N 次, 取 inliers 最多者
//     4. 对最优 inliers 做 Umeyama SVD 一步求解
//   输出: SimTransform + inliers 点对
// ============================================================================
static void ransac_solve(const std::vector<StarPoint>& U_full,
                          const std::vector<StarPoint>& W_,
                          const std::vector<PointPair>& candidate_pairs,
                          double s0, int max_iters, double tol_arcsec,
                          SimTransform& best_transform,
                          std::vector<PointPair>& best_inliers,
                          int& n_iters_actual, double& rms,
                          std::mt19937& rng,
                          Logger* logger = nullptr)
{
    best_transform.valid = false;
    best_inliers.clear();
    n_iters_actual = 0;
    rms = std::numeric_limits<double>::infinity();

    int n_cand = (int)candidate_pairs.size();
    if (n_cand < 2) {
        if (logger) logger->warn("RANSAC: 候选点对不足 (<2): " + std::to_string(n_cand));
        return;
    }

    std::uniform_int_distribution<int> ud(0, n_cand - 1);
    int best_n_inliers = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        n_iters_actual = iter + 1;

        // 随机抽 2 个不同点对
        int k1 = ud(rng), k2 = ud(rng);
        if (k1 == k2) continue;
        int i1 = candidate_pairs[k1].img_idx, a1 = candidate_pairs[k1].gaia_idx;
        int i2 = candidate_pairs[k2].img_idx, a2 = candidate_pairs[k2].gaia_idx;
        if (i1 == i2 || a1 == a2) continue;  // 同点不能成对

        // 解析求 (s, θ_true, tx, ty)
        double dUx = U_full[i1].x - U_full[i2].x;
        double dUy = U_full[i1].y - U_full[i2].y;
        double dWx = W_[a1].x - W_[a2].x;
        double dWy = W_[a1].y - W_[a2].y;
        double d_img = std::sqrt(dUx*dUx + dUy*dUy);
        double d_gaia = std::sqrt(dWx*dWx + dWy*dWy);
        if (d_gaia < 1e-6) continue;
        double s = d_img / d_gaia;
        if (s < 0.9 || s > 1.1) continue;  // s ±10% 容差

        // θ_rot (采样域) = atan2(Δw) - atan2(Δu), θ_true = -θ_rot
        double theta_rot = std::atan2(dWy, dWx) - std::atan2(dUy, dUx);
        double theta_true = -theta_rot;
        double ct = std::cos(theta_true), st = std::sin(theta_true);

        // 平移: tx = U[i1].x - s·R(θ_true)·W[a1].x
        //   R(θ_true) = [cos -sin; sin cos]
        //   R·W = (cos·wx - sin·wy, sin·wx + cos·wy)
        double tx = U_full[i1].x - s * (ct * W_[a1].x - st * W_[a1].y);
        double ty = U_full[i1].y - s * (st * W_[a1].x + ct * W_[a1].y);

        // 验证所有候选
        std::vector<PointPair> inliers;
        double sum_sq = 0.0;
        for (const auto& cp : candidate_pairs) {
            double wx = W_[cp.gaia_idx].x, wy = W_[cp.gaia_idx].y;
            double px = s * (ct * wx - st * wy) + tx;
            double py = s * (st * wx + ct * wy) + ty;
            double dx = U_full[cp.img_idx].x - px;
            double dy = U_full[cp.img_idx].y - py;
            double resid = std::sqrt(dx*dx + dy*dy);
            if (resid < tol_arcsec) {
                inliers.push_back(cp);
                sum_sq += resid * resid;
            }
        }

        if ((int)inliers.size() > best_n_inliers) {
            best_n_inliers = (int)inliers.size();
            best_inliers = inliers;
            // 临时保存解析解 (后续用 Umeyama 精化)
            best_transform.s = s;
            best_transform.theta = theta_true;
            best_transform.tx = tx;
            best_transform.ty = ty;
            best_transform.valid = true;
            rms = std::sqrt(sum_sq / std::max(1, (int)inliers.size()));
        }
    }

    // 对最优 inliers 做 Umeyama SVD 一步求解
    if (best_n_inliers >= 2) {
        std::vector<int> u_idx, w_idx;
        u_idx.reserve(best_inliers.size());
        w_idx.reserve(best_inliers.size());
        for (const auto& p : best_inliers) {
            u_idx.push_back(p.img_idx);
            w_idx.push_back(p.gaia_idx);
        }
        SimTransform refined = umeyama_svd(U_full, W_, u_idx, w_idx, logger);
        if (refined.valid) {
            best_transform = refined;

            // 重新计算 RMS
            double ct = std::cos(refined.theta), st = std::sin(refined.theta);
            double sum_sq = 0.0;
            for (const auto& p : best_inliers) {
                double px = refined.s * (ct * W_[p.gaia_idx].x - st * W_[p.gaia_idx].y) + refined.tx;
                double py = refined.s * (st * W_[p.gaia_idx].x + ct * W_[p.gaia_idx].y) + refined.ty;
                double dx = U_full[p.img_idx].x - px;
                double dy = U_full[p.img_idx].y - py;
                sum_sq += dx*dx + dy*dy;
            }
            rms = std::sqrt(sum_sq / best_inliers.size());
        }
    }

    if (logger) {
        logger->info("RANSAC: iters=" + std::to_string(n_iters_actual) +
                     " 候选=" + std::to_string(n_cand) +
                     " inliers=" + std::to_string(best_n_inliers) +
                     " RMS=" + std::to_string(rms) + "\"");
        if (best_transform.valid) {
            logger->info("RANSAC 结果: s=" + std::to_string(best_transform.s) +
                         " θ=" + std::to_string(best_transform.theta * EXP_RADTODEG) + "°" +
                         " tx=" + std::to_string(best_transform.tx) +
                         " ty=" + std::to_string(best_transform.ty));
        }
    }
}

// ============================================================================
// 主算法
// ============================================================================
int runRelVecExperiment(
    const std::vector<StarPoint>& U_full,
    const std::vector<StarPoint>& W_,
    double s0,
    const RelVecParams& params,
    const SimTransform& ground_truth,
    bool has_ground_truth,
    ExpResult& output,
    const std::string& log_dir)
{
    Logger logger;
    if (!log_dir.empty()) {
        // 日志文件路径: log_dir/exp_relvec.log
        std::string log_path = log_dir;
        if (log_path.back() != '/' && log_path.back() != '\\') log_path += "/";
        log_path += "exp_relvec.log";
        logger.init(log_path);
    }
    logger.info("=== exp44 runRelVecExperiment 开始 ===");
    logger.info("N_u=" + std::to_string(U_full.size()) +
                " N_w=" + std::to_string(W_.size()) +
                " s0=" + std::to_string(s0) +
                " s_min=" + std::to_string(params.s_min) +
                " s_max=" + std::to_string(params.s_max));

    // 初始化输出
    output = ExpResult{};
    output.density_estimate.valid = false;
    output.ransac_estimate.valid = false;
    output.success = false;
    output.snr_final = 0;
    output.n_samples_actual = 0;
    output.n_passed = 0;
    output.n_focused = 0;
    output.n_inliers = 0;
    output.n_ransac_iters = 0;
    output.ransac_rms = 0;

    int N_w = (int)W_.size();
    int N_u_full = (int)U_full.size();
    if (N_w < 3 || N_u_full < 3) {
        logger.error("数据过少: N_u=" + std::to_string(N_u_full) +
                     " N_w=" + std::to_string(N_w));
        return -1;
    }

    // ========================================================================
    // 1. U 组限流: 按 flux 降序取前 max_u 颗
    // ========================================================================
    int max_u = std::min(params.max_u, N_u_full);
    std::vector<int> u_idx(N_u_full);
    std::iota(u_idx.begin(), u_idx.end(), 0);
    std::sort(u_idx.begin(), u_idx.end(),
              [&](int a, int b){ return U_full[a].flux > U_full[b].flux; });
    std::vector<int> u_sel(u_idx.begin(), u_idx.begin() + max_u);
    int N_u = max_u;

    // 构建限流后的 U 数组
    std::vector<StarPoint> U(N_u);
    for (int k = 0; k < N_u; ++k) U[k] = U_full[u_sel[k]];

    logger.info("U 限流: " + std::to_string(N_u_full) + " → " + std::to_string(N_u));

    // ========================================================================
    // 2. 预计算 U 距离矩阵 D_U + U 距离索引 (相似度加权用)
    // ========================================================================
    std::vector<double> D_U((size_t)N_u * N_u, 0.0);
    for (int i = 0; i < N_u; ++i) {
        for (int j = i + 1; j < N_u; ++j) {
            double dx = U[i].x - U[j].x;
            double dy = U[i].y - U[j].y;
            double d = std::sqrt(dx * dx + dy * dy);
            D_U[(size_t)i * N_u + j] = d;
            D_U[(size_t)j * N_u + i] = d;
        }
    }

    // U 距离索引 (相似度加权用)
    std::vector<std::vector<double>> D_U_rows(N_u);
    for (int i = 0; i < N_u; ++i) {
        D_U_rows[i].resize(N_u);
        for (int k = 0; k < N_u; ++k) {
            D_U_rows[i][k] = D_U[(size_t)i * N_u + k];
        }
    }

    // ========================================================================
    // 3. 预计算 W 距离矩阵 D_W + W 距离索引 D_W_sorted
    // ========================================================================
    std::vector<double> D_W((size_t)N_w * N_w, 0.0);
    double d_max_global = 0.0;
    for (int i = 0; i < N_w; ++i) {
        for (int j = i + 1; j < N_w; ++j) {
            double dx = W_[i].x - W_[j].x;
            double dy = W_[i].y - W_[j].y;
            double d = std::sqrt(dx * dx + dy * dy);
            D_W[(size_t)i * N_w + j] = d;
            D_W[(size_t)j * N_w + i] = d;
            if (d > d_max_global) d_max_global = d;
        }
    }

    // W 距离索引 (D_W_sorted[a] = [(dist, c), ...] 排序, 第三星验证 + 相似度用)
    std::vector<std::vector<std::pair<double, int>>> D_W_sorted(N_w);
    for (int a = 0; a < N_w; ++a) {
        D_W_sorted[a].reserve(N_w);
        for (int c = 0; c < N_w; ++c) {
            if (c == a) continue;
            D_W_sorted[a].push_back({D_W[(size_t)a * N_w + c], c});
        }
        std::sort(D_W_sorted[a].begin(), D_W_sorted[a].end());
    }

    // W 距离行 (相似度加权用)
    std::vector<std::vector<double>> D_W_rows(N_w);
    for (int a = 0; a < N_w; ++a) {
        D_W_rows[a].resize(N_w);
        for (int k = 0; k < N_w; ++k) {
            D_W_rows[a][k] = D_W[(size_t)a * N_w + k];
        }
    }

    // ========================================================================
    // 4. 预构建 Gaia 星对数组 (i<j, 按距离排序, k-vector 查询用)
    // ========================================================================
    double d_min = d_max_global * params.min_len_frac;
    double d_max = d_max_global * params.max_len_frac;

    struct RawPair { double dist, angle; int a, b; };
    std::vector<RawPair> raw_pairs;
    raw_pairs.reserve((size_t)N_w * N_w / 4);
    for (int i = 0; i < N_w; ++i) {
        for (int j = i + 1; j < N_w; ++j) {
            double d = D_W[(size_t)i * N_w + j];
            if (d >= d_min && d <= d_max) {
                double dx = W_[i].x - W_[j].x;
                double dy = W_[i].y - W_[j].y;
                raw_pairs.push_back({d, std::atan2(dy, dx), i, j});
            }
        }
    }
    std::sort(raw_pairs.begin(), raw_pairs.end(),
              [](const RawPair& a, const RawPair& b){ return a.dist < b.dist; });

    std::vector<double> pair_dist(raw_pairs.size());
    std::vector<double> pair_angle(raw_pairs.size());
    std::vector<int> pair_a(raw_pairs.size());
    std::vector<int> pair_b(raw_pairs.size());
    for (size_t k = 0; k < raw_pairs.size(); ++k) {
        pair_dist[k]  = raw_pairs[k].dist;
        pair_angle[k] = raw_pairs[k].angle;
        pair_a[k]     = raw_pairs[k].a;
        pair_b[k]     = raw_pairs[k].b;
    }

    logger.info("Gaia 星对数: " + std::to_string(raw_pairs.size()) +
                " 距离范围[" + std::to_string(d_min) + "," +
                std::to_string(d_max) + "]\"");

    if (raw_pairs.empty()) {
        logger.error("无 Gaia 星对");
        return -1;
    }

    // ========================================================================
    // 5. 确定 tx, ty 动态范围
    //    tx = U[i].x - s·R(θ)·W[a].x, 真匹配时 tx ≈ const (平移分量)
    //    范围: [ux_min - s_max·wx_max, ux_max - s_min·wx_min] + margin
    //    (旋转不改变坐标绝对值范围, 只交换 x/y, 所以用 |wx| 最大值即可)
    // ========================================================================
    double ux_min = std::numeric_limits<double>::infinity();
    double ux_max = -std::numeric_limits<double>::infinity();
    double uy_min = std::numeric_limits<double>::infinity();
    double uy_max = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < N_u; ++i) {
        ux_min = std::min(ux_min, U[i].x); ux_max = std::max(ux_max, U[i].x);
        uy_min = std::min(uy_min, U[i].y); uy_max = std::max(uy_max, U[i].y);
    }
    // W 坐标绝对值最大 (旋转后 x/y 互换, 取 max(|wx|,|wy|) 作为单边范围)
    double w_abs_max = 0.0;
    for (int a = 0; a < N_w; ++a) {
        w_abs_max = std::max(w_abs_max, std::max(std::abs(W_[a].x), std::abs(W_[a].y)));
    }
    double margin = 200.0;  // 角秒余量
    double tx_range_lo = ux_min - params.s_max * w_abs_max - margin;
    double tx_range_hi = ux_max - params.s_min * (-w_abs_max) + margin;
    double ty_range_lo = uy_min - params.s_max * w_abs_max - margin;
    double ty_range_hi = uy_max - params.s_min * (-w_abs_max) + margin;
    double tx_bw = (tx_range_hi - tx_range_lo) / params.dxdy_bins;
    double ty_bw = (ty_range_hi - ty_range_lo) / params.dxdy_bins;
    double th_bw = 360.0 / params.th_bins;  // θ: [-180, 180) → 360 bin

    logger.info("动态范围: θ∈[-180,180)° tx∈[" +
                std::to_string(tx_range_lo) + "," + std::to_string(tx_range_hi) +
                "]\" ty∈[" + std::to_string(ty_range_lo) + "," +
                std::to_string(ty_range_hi) + "]\"");

    // ========================================================================
    // 6. 3D 密度场 (稀疏存储)
    // ========================================================================
    std::unordered_map<uint64_t, int> density3d;
    density3d.reserve(100000);
    int total_votes_3d = 0;

    // ========================================================================
    // 7. 采样循环
    // ========================================================================
    std::mt19937 rng(params.seed);
    std::uniform_int_distribution<int> ud_u(0, N_u - 1);

    int max_samples = params.max_samples;
    double third_star_tol_px = params.third_star_tol_px;
    int max_cand = params.max_cand;
    int n_third_stars = params.n_third_stars;
    int n_total_cand = 0;

    // 自适应停止
    bool adaptive_stop = (params.adaptive_stop != 0);
    int min_samples = std::max(params.min_samples, 1);
    int check_interval = std::max(params.check_interval, 1);
    double snr_eps = params.snr_eps;
    int max_stable = std::max(params.max_stable, 1);
    double prev_snr = 0.0;
    int stable_count = 0;

    // 递归聚焦
    FocusRegion region{};
    region.confirmed = false;
    int n_focused = 0;
    int n_discarded = 0;

    output.passed_pairs.clear();
    output.passed_pairs.reserve(10000);
    output.focus_history.clear();

    bool use_sim_weight = (params.use_similarity_weight != 0);
    int sim_knn = params.similarity_knn;

    int actual_samples = 0;
    for (int s = 0; s < max_samples; ++s) {
        actual_samples = s + 1;
        auto t0 = Clock::now();

        // --- 7a. 单次采样 + 验证 ---
        do {
            int i = ud_u(rng), j = ud_u(rng);
            if (i == j) { break; }
            double d_img = D_U[(size_t)i * N_u + j];
            if (d_img < 1.0) break;
            double theta_img = std::atan2(U[i].y - U[j].y, U[i].x - U[j].x);

            // --- 7b. k-vector 距离查询 (±10% s 过滤) ---
            // d_gaia ∈ [d_img/s_max, d_img/s_min] (s_min=0.9, s_max=1.1)
            double d_lo = d_img / params.s_max;
            double d_hi = d_img / params.s_min;
            int i_lo = (int)(std::lower_bound(pair_dist.begin(), pair_dist.end(), d_lo) - pair_dist.begin());
            int i_hi = (int)(std::upper_bound(pair_dist.begin(), pair_dist.end(), d_hi) - pair_dist.begin());
            int n_cand = i_hi - i_lo;
            n_total_cand += n_cand;
            if (n_cand == 0) break;

            // 候选限流
            std::vector<int> cand_indices;
            if (n_cand > max_cand) {
                std::vector<int> all_idx(n_cand);
                std::iota(all_idx.begin(), all_idx.end(), i_lo);
                std::shuffle(all_idx.begin(), all_idx.end(), rng);
                cand_indices.assign(all_idx.begin(), all_idx.begin() + max_cand);
            } else {
                cand_indices.resize(n_cand);
                std::iota(cand_indices.begin(), cand_indices.end(), i_lo);
            }

            // 第三星 k 列表
            std::vector<int> k_list;
            if (n_third_stars <= 0) {
                k_list.reserve(N_u - 2);
                for (int k = 0; k < N_u; ++k) {
                    if (k != i && k != j) k_list.push_back(k);
                }
            } else {
                k_list.reserve(n_third_stars);
                for (int t = 0; t < n_third_stars * 2 && (int)k_list.size() < n_third_stars; ++t) {
                    int k = ud_u(rng);
                    if (k != i && k != j) k_list.push_back(k);
                }
            }

            // --- 7c. 对每个候选 (a,b) 验证 ---
            for (int ci : cand_indices) {
                int a = pair_a[ci], b = pair_b[ci];
                double d_gaia_ab = pair_dist[ci];
                if (d_gaia_ab < 1e-6) continue;

                // s_est 每对星估计 (核心: 补偿实际 s 偏差)
                double s_est = d_img / d_gaia_ab;

                // s ±10% 过滤 (不在 [0.9, 1.1] 直接丢弃)
                if (s_est < params.s_min || s_est > params.s_max) continue;

                // 第三星验证容差 (像素 → 角秒)
                double tol_arcsec = third_star_tol_px * s0 / s_est;
                double s_rel_err = tol_arcsec / std::max(d_gaia_ab, 1.0);

                int n_k_passed = 0;
                for (int k : k_list) {
                    double d_ik_img = D_U[(size_t)i * N_u + k];
                    double d_jk_img = D_U[(size_t)j * N_u + k];
                    double d_ik_exp = d_ik_img / s_est;
                    double d_jk_exp = d_jk_img / s_est;
                    double d_ik_tol = d_ik_exp * s_rel_err + tol_arcsec;
                    double d_jk_tol = d_jk_exp * s_rel_err + tol_arcsec;

                    const auto& da = D_W_sorted[a];
                    auto lo_it = std::lower_bound(da.begin(), da.end(), d_ik_exp - d_ik_tol,
                        [](const std::pair<double, int>& p, double v){ return p.first < v; });
                    auto hi_it = std::upper_bound(da.begin(), da.end(), d_ik_exp + d_ik_tol,
                        [](double v, const std::pair<double, int>& p){ return v < p.first; });

                    bool k_passed = false;
                    for (auto it = lo_it; it != hi_it; ++it) {
                        int c = it->second;
                        if (c == b) continue;
                        if (std::abs(D_W[(size_t)b * N_w + c] - d_jk_exp) < d_jk_tol) {
                            k_passed = true; break;
                        }
                    }
                    if (k_passed) n_k_passed++;
                }

                if (n_k_passed <= 0) continue;

                // --- 7d. 计算 θ, tx, ty (单点法) ---
                // 采样域 θ_rot = angle(Δw) - angle(Δu) = -θ_true
                //   (因 ΔU = s·R(θ_true)·ΔW → angle(Δu) = θ_true + angle(Δw))
                // 变换关系: U = s·R(θ_true)·W + (tx,ty), θ_true = -θ_rot
                //   R(θ_true) = R(-θ_rot) = [cos(θ_rot)  sin(θ_rot); -sin(θ_rot) cos(θ_rot)]
                //   W'[a] = s·R(θ_true)·W[a] = (s·(cos·wx + sin·wy), s·(-sin·wx + cos·wy))
                //   tx = U[i].x - W'[a].x = U[i].x - s·(cos·wx + sin·wy)
                //   ty = U[i].y - W'[a].y = U[i].y - s·(-sin·wx + cos·wy)
                double theta_rot = (pair_angle[ci] - theta_img) * EXP_RADTODEG;
                theta_rot = wrap180(theta_rot);

                double th_rad = theta_rot * EXP_DEGTORAD;
                double ct_r = std::cos(th_rad), st_r = std::sin(th_rad);
                double ux = U[i].x, uy = U[i].y;
                double wx = W_[a].x, wy = W_[a].y;
                // θ_true = -θ_rot → R(θ_true) 用 cos(θ_rot), -sin(θ_rot)
                double tx_est = ux - s_est * (ct_r * wx + st_r * wy);
                double ty_est = uy - s_est * (-st_r * wx + ct_r * wy);

                // 聚焦模式: 丢弃区外候选
                if (region.confirmed) {
                    if (theta_rot < region.th_lo || theta_rot > region.th_hi ||
                        tx_est < region.tx_lo || tx_est > region.tx_hi ||
                        ty_est < region.ty_lo || ty_est > region.ty_hi) {
                        n_discarded++;
                        continue;
                    }
                    n_focused++;
                }

                // --- 7e. 相似度加权 ---
                double similarity = 1.0;
                if (use_sim_weight) {
                    similarity = compute_similarity(
                        U, W_, i, j, a, b,
                        s_est, th_rad, tx_est, ty_est,
                        s0, sim_knn,
                        D_U_rows[i], D_U_rows[j],
                        D_W_sorted[a], D_W_sorted[b],
                        D_W_rows[a], D_W_rows[b],
                        &logger);
                }
                int vote = (int)std::round(n_k_passed * similarity);
                if (vote < 1) vote = 1;

                // --- 7f. 投入 3D 密度场 ---
                int th3 = (int)((theta_rot + 180.0) / th_bw);
                if (th3 < 0) th3 = 0;
                if (th3 >= params.th_bins) th3 = params.th_bins - 1;
                int tx3 = (int)((tx_est - tx_range_lo) / tx_bw);
                if (tx3 < 0) tx3 = 0;
                if (tx3 >= params.dxdy_bins) tx3 = params.dxdy_bins - 1;
                int ty3 = (int)((ty_est - ty_range_lo) / ty_bw);
                if (ty3 < 0) ty3 = 0;
                if (ty3 >= params.dxdy_bins) ty3 = params.dxdy_bins - 1;
                uint64_t key3 = ((uint64_t)th3 * params.dxdy_bins + tx3) * params.dxdy_bins + ty3;
                density3d[key3] += vote;
                total_votes_3d += vote;

                // 记录 passed_pair
                output.passed_pairs.push_back({
                    u_sel[i], u_sel[j], a, b,
                    theta_rot, s_est, tx_est, ty_est,
                    n_k_passed, similarity, vote
                });
            }
        } while (false);

        // --- 7g. 递归聚焦检查 (每 check_interval 次, 达 min_samples 后) ---
        if ((s + 1) >= min_samples && (s + 1) % check_interval == 0) {
            int pk_th, pk_tx, pk_ty, pk_cluster;
            double snr;
            detect_peak_3d(density3d, total_votes_3d,
                            params.th_bins, params.dxdy_bins, params.peak_cluster_half,
                            pk_th, pk_tx, pk_ty, pk_cluster, snr, &logger);

            double pk_theta = (pk_th + 0.5) * th_bw - 180.0;
            double pk_tx_val = tx_range_lo + (pk_tx + 0.5) * tx_bw;
            double pk_ty_val = ty_range_lo + (pk_ty + 0.5) * ty_bw;

            // 记录聚焦历史快照
            FocusSnapshot snap;
            snap.sample_idx = s + 1;
            snap.total_votes = total_votes_3d;
            snap.n_nonzero_bins = (int)density3d.size();
            snap.peak_cluster = pk_cluster;
            snap.snr = snr;
            snap.peak_theta = pk_theta;
            snap.peak_tx = pk_tx_val;
            snap.peak_ty = pk_ty_val;
            snap.confirmed = region.confirmed;
            snap.n_focused = n_focused;
            snap.n_discarded = n_discarded;
            snap.focus_th_lo = region.th_lo; snap.focus_th_hi = region.th_hi;
            snap.focus_tx_lo = region.tx_lo; snap.focus_tx_hi = region.tx_hi;
            snap.focus_ty_lo = region.ty_lo; snap.focus_ty_hi = region.ty_hi;
            output.focus_history.push_back(snap);

            if (!region.confirmed) {
                // 阶段 2: 识别 - SNR > 阈值则确认聚焦区
                if (snr > params.snr_threshold) {
                    // θ 范围 (bin 宽 1° 精确)
                    double th_lo0 = pk_theta - params.focus_th_half;
                    double th_hi0 = pk_theta + params.focus_th_half;

                    // ⭐ tx/ty bin 宽大 (≈140"), bin 中心偏离真值最多 70"
                    //   用 3D bin 中心 (θ+tx+ty) 过滤 passed_pairs
                    //   tx/ty 范围: bin 中心 ± 1.5 * bin 宽 (覆盖 bin 中心偏差 + 噪声)
                    double tx_filter_half = 1.5 * tx_bw;
                    double ty_filter_half = 1.5 * ty_bw;
                    double tx_lo0 = pk_tx_val - tx_filter_half;
                    double tx_hi0 = pk_tx_val + tx_filter_half;
                    double ty_lo0 = pk_ty_val - ty_filter_half;
                    double ty_hi0 = pk_ty_val + ty_filter_half;

                    // 用 3D bin 中心过滤, 统计 tx/ty 中位数
                    std::vector<double> in_tx, in_ty;
                    for (const auto& p : output.passed_pairs) {
                        if (p.theta_rot_deg < th_lo0 || p.theta_rot_deg > th_hi0) continue;
                        if (p.tx < tx_lo0 || p.tx > tx_hi0) continue;
                        if (p.ty < ty_lo0 || p.ty > ty_hi0) continue;
                        in_tx.push_back(p.tx);
                        in_ty.push_back(p.ty);
                    }
                    double tx_center = pk_tx_val, ty_center = pk_ty_val;
                    if (!in_tx.empty()) {
                        std::sort(in_tx.begin(), in_tx.end());
                        std::sort(in_ty.begin(), in_ty.end());
                        tx_center = in_tx[in_tx.size() / 2];
                        ty_center = in_ty[in_ty.size() / 2];
                    }

                    // 聚焦区: θ 用精确范围, tx/ty 用中位数 ± half
                    region.th_lo = th_lo0; region.th_hi = th_hi0;
                    region.tx_lo = tx_center - params.focus_dxdy_half;
                    region.tx_hi = tx_center + params.focus_dxdy_half;
                    region.ty_lo = ty_center - params.focus_dxdy_half;
                    region.ty_hi = ty_center + params.focus_dxdy_half;
                    // ⭐ 固定中心 (初始确认时设置, 后续收紧时不移动)
                    //   避免区内 passed_pairs 很少时中位数漂移导致真匹配被排除
                    region.th_center = pk_theta;
                    region.tx_center = tx_center;
                    region.ty_center = ty_center;
                    region.confirmed = true;

                    logger.info("3D聚焦确认: s=" + std::to_string(s + 1) +
                                " θ=" + std::to_string(pk_theta) + "°" +
                                " tx_bin=" + std::to_string(pk_tx_val) +
                                " ty_bin=" + std::to_string(pk_ty_val) +
                                " → tx_med=" + std::to_string(tx_center) +
                                " ty_med=" + std::to_string(ty_center) +
                                " (3D区内 " + std::to_string(in_tx.size()) + " 对)" +
                                " SNR=" + std::to_string(snr) + "x" +
                                " cluster=" + std::to_string(pk_cluster) +
                                " 非零bin=" + std::to_string(density3d.size()));
                }
            } else {
                // 阶段 3: 聚焦区收敛检查 + 定期收紧
                // 收敛条件: n_focused <= N_target 且 SNR > snr_final_threshold
                if (n_focused > 0 &&
                    n_focused <= params.focus_target_n_candidates &&
                    snr > params.snr_final_threshold) {
                    logger.info("目标收敛停止: s=" + std::to_string(s + 1) +
                                " n_focused=" + std::to_string(n_focused) +
                                " <= " + std::to_string(params.focus_target_n_candidates) +
                                " SNR=" + std::to_string(snr) +
                                " > " + std::to_string(params.snr_final_threshold));
                    break;
                }

                // 未达标: 定期收紧区域 (有下限, 避免收紧到 bin 级别导致 n_focused=0)
                bool shrunk = false;
                if ((s + 1) % params.focus_shrink_interval == 0) {
                    // ⭐ 用固定中心 (初始确认时的中位数), 不重新计算
                    //   避免区内 passed_pairs 很少时中位数漂移导致真匹配被排除
                    //   (NGC55 案例: 区内 2 对时 tx 中位数从 -0.85 漂移到 7.90, 真匹配被排除)
                    double th_mid = region.th_center;
                    double tx_mid = region.tx_center;
                    double ty_mid = region.ty_center;
                    // 统计区内 passed_pairs (仅用于日志)
                    int n_in_region = 0;
                    for (const auto& p : output.passed_pairs) {
                        if (p.theta_rot_deg < region.th_lo || p.theta_rot_deg > region.th_hi) continue;
                        if (p.tx < region.tx_lo || p.tx > region.tx_hi) continue;
                        if (p.ty < region.ty_lo || p.ty > region.ty_hi) continue;
                        n_in_region++;
                    }
                    double th_half = params.focus_shrink_factor * 0.5 * (region.th_hi - region.th_lo);
                    double tx_half = params.focus_shrink_factor * 0.5 * (region.tx_hi - region.tx_lo);
                    double ty_half = params.focus_shrink_factor * 0.5 * (region.ty_hi - region.ty_lo);
                    // 下限: 避免收紧到噪声级别以下 (θ >= 0.5°, tx/ty >= 30")
                    //   30" 容纳质心误差 + s_est 误差, 避免真匹配被排除
                    double th_half_min = 0.5;
                    double dxdy_half_min = 30.0;
                    if (th_half < th_half_min) th_half = th_half_min;
                    if (tx_half < dxdy_half_min) tx_half = dxdy_half_min;
                    if (ty_half < dxdy_half_min) ty_half = dxdy_half_min;
                    region.th_lo = th_mid - th_half; region.th_hi = th_mid + th_half;
                    region.tx_lo = tx_mid - tx_half; region.tx_hi = tx_mid + tx_half;
                    region.ty_lo = ty_mid - ty_half; region.ty_hi = ty_mid + ty_half;
                    shrunk = true;
                    // 区域收紧后 SNR 会变化, 重置 stable_count 避免过早停止
                    // 同时重置 n_focused, 统计新区域内的采样数
                    stable_count = 0;
                    int n_focused_before_shrink = n_focused;
                    n_focused = 0;
                    logger.info("聚焦收紧: s=" + std::to_string(s + 1) +
                                " θ∈[" + std::to_string(region.th_lo) + "," + std::to_string(region.th_hi) + "]" +
                                " tx∈[" + std::to_string(region.tx_lo) + "," + std::to_string(region.tx_hi) + "]" +
                                " ty∈[" + std::to_string(region.ty_lo) + "," + std::to_string(region.ty_hi) + "]" +
                                " n_focused(收紧前)=" + std::to_string(n_focused_before_shrink) +
                                " (区内 " + std::to_string(n_in_region) + " 对)");
                }

                // 阶段 4b: SNR 稳定兜底 (区域已到下限, 无法再收紧)
                if (adaptive_stop && !shrunk && prev_snr > 0.0 && snr > params.snr_final_threshold) {
                    double rel_change = std::abs(snr - prev_snr) / prev_snr;
                    if (rel_change < snr_eps) {
                        stable_count++;
                        if (stable_count >= max_stable) {
                            logger.info("SNR 收敛停止(兜底): s=" + std::to_string(s + 1) +
                                        " SNR=" + std::to_string(snr) +
                                        " n_focused=" + std::to_string(n_focused) +
                                        " (连续 " + std::to_string(stable_count) + " 次)");
                            break;
                        }
                    } else {
                        stable_count = 0;
                    }
                }
            }
            prev_snr = snr;
        }
    }

    output.n_samples_actual = actual_samples;
    output.n_passed = (int)output.passed_pairs.size();
    output.n_focused = n_focused;

    // ========================================================================
    // 8. 最终 3D 峰值检测 (粗略, 仅用于定位聚集区 θ)
    // ========================================================================
    int pk_th, pk_tx, pk_ty, pk_cluster;
    double snr_final;
    detect_peak_3d(density3d, total_votes_3d,
                    params.th_bins, params.dxdy_bins, params.peak_cluster_half,
                    pk_th, pk_tx, pk_ty, pk_cluster, snr_final, &logger);

    double theta_rot_peak = (pk_th + 0.5) * th_bw - 180.0;  // 采样域 θ_rot (bin 精确 1°)
    double theta_true_peak = wrap180(-theta_rot_peak);       // 物理域 θ_true
    // 注意: tx/ty bin 宽大 (≈100"), bin 中心不可靠, 不直接用
    double tx_peak = tx_range_lo + (pk_tx + 0.5) * tx_bw;
    double ty_peak = ty_range_lo + (pk_ty + 0.5) * ty_bw;

    output.density_estimate.s = 1.0;  // 密度场不估 s
    output.density_estimate.theta = theta_true_peak * EXP_DEGTORAD;
    output.density_estimate.tx = tx_peak;   // 粗略
    output.density_estimate.ty = ty_peak;   // 粗略
    output.density_estimate.valid = true;
    output.snr_final = snr_final;

    logger.info("=== 密度场峰值 (粗略) ===");
    logger.info("θ_rot=" + std::to_string(theta_rot_peak) + "°" +
                " (θ_true=" + std::to_string(theta_true_peak) + "°)" +
                " tx_bin=" + std::to_string(tx_peak) + " (粗略)" +
                " ty_bin=" + std::to_string(ty_peak) + " (粗略)" +
                " SNR=" + std::to_string(snr_final) + "x" +
                " cluster=" + std::to_string(pk_cluster));

    // ========================================================================
    // 9. ⭐ 提取候选点对关系 (供 RANSAC 剔除外点)
    //    设计意图 (v4_4_relvec_sampling_design.md 第 6.2/7.1 节):
    //    3D 密度场 θ 峰值可靠 (1° bin), 但 tx/ty bin 中心不可靠 (bin 宽 ~40")。
    //    θ 过滤后 passed_pairs 中错误匹配占多数, tx/ty 中位数偏离真值
    //    (LDN43@022502: tx_med=213.8 但真值 tx=413.88, 偏差 200")。
    //
    //    新方案 (2D tx/ty 密度场找真匹配聚集区):
    //    1. 用 θ 峰值 ±2° 过滤 passed_pairs (θ 峰值可靠)
    //    2. 在 θ 过滤后的 passed_pairs 中构建 2D tx/ty 直方图 (400×400 bin)
    //       真匹配聚集在一个小区域, 错误匹配随机分布
    //    3. 找 2D 峰值 (5×5 邻域累加), 用峰值 bin 中心 ±50" 过滤
    //    4. 送 RANSAC 剔除外点
    // ========================================================================
    std::vector<PointPair> candidate_pairs;
    std::set<std::pair<int,int>> seen;  // 去重 (img_idx, gaia_idx)

    // 1. θ 过滤 (3D 密度场 θ 峰值可靠)
    double theta_filter_half = 2.0;
    std::vector<const PassedPair*> theta_filtered;
    int n_outside_theta = 0;
    for (const auto& p : output.passed_pairs) {
        if (std::abs(wrap180(p.theta_rot_deg - theta_rot_peak)) <= theta_filter_half) {
            theta_filtered.push_back(&p);
        } else {
            n_outside_theta++;
        }
    }

    // 2. 构建 2D tx/ty 直方图 (θ 过滤后, passed_pairs 多)
    //    bin 宽 = 动态范围 / 400 (约 10-20"), 比中位数更精确
    double tx_peak_2d = 0.0, ty_peak_2d = 0.0;
    int peak_cluster_2d = 0;
    int bins_2d = 400;
    if (!theta_filtered.empty()) {
        double tx_min = +1e9, tx_max = -1e9, ty_min = +1e9, ty_max = -1e9;
        for (const auto* pp : theta_filtered) {
            if ((*pp).tx < tx_min) tx_min = (*pp).tx;
            if ((*pp).tx > tx_max) tx_max = (*pp).tx;
            if ((*pp).ty < ty_min) ty_min = (*pp).ty;
            if ((*pp).ty > ty_max) ty_max = (*pp).ty;
        }
        double tx_range_2d = (tx_max - tx_min) + 1.0;
        double ty_range_2d = (ty_max - ty_min) + 1.0;
        double tx_bw_2d = tx_range_2d / bins_2d;
        double ty_bw_2d = ty_range_2d / bins_2d;

        std::unordered_map<uint64_t, int> density2d;
        density2d.reserve(theta_filtered.size());
        for (const auto* pp : theta_filtered) {
            int tx_bin = (int)(((*pp).tx - tx_min) / tx_bw_2d);
            int ty_bin = (int)(((*pp).ty - ty_min) / ty_bw_2d);
            if (tx_bin < 0 || tx_bin >= bins_2d || ty_bin < 0 || ty_bin >= bins_2d) continue;
            uint64_t key = (uint64_t)tx_bin * bins_2d + ty_bin;
            density2d[key]++;
        }

        // 找 2D 峰值 (9×9 邻域累加, 容纳 s_est 误差导致的 tx/ty 分散)
        for (auto& kv : density2d) {
            uint64_t key = kv.first;
            int tx_bin = (int)(key / bins_2d);
            int ty_bin = (int)(key % bins_2d);
            int cluster = 0;
            for (int dx = -4; dx <= 4; ++dx) {
                for (int dy = -4; dy <= 4; ++dy) {
                    int tx2 = tx_bin + dx, ty2 = ty_bin + dy;
                    if (tx2 < 0 || tx2 >= bins_2d || ty2 < 0 || ty2 >= bins_2d) continue;
                    uint64_t k2 = (uint64_t)tx2 * bins_2d + ty2;
                    auto it = density2d.find(k2);
                    if (it != density2d.end()) cluster += it->second;
                }
            }
            if (cluster > peak_cluster_2d) {
                peak_cluster_2d = cluster;
                tx_peak_2d = tx_min + (tx_bin + 0.5) * tx_bw_2d;
                ty_peak_2d = ty_min + (ty_bin + 0.5) * ty_bw_2d;
            }
        }
    }

    logger.info("候选提取: θ峰值=" + std::to_string(theta_rot_peak) + "°" +
                " ±" + std::to_string(theta_filter_half) + "°" +
                " 通过=" + std::to_string(theta_filtered.size()) +
                " (区外=" + std::to_string(n_outside_theta) + ")" +
                " 2D峰值: tx=" + std::to_string(tx_peak_2d) +
                " ty=" + std::to_string(ty_peak_2d) +
                " cluster=" + std::to_string(peak_cluster_2d) +
                " (来自 " + std::to_string(output.passed_pairs.size()) + " passed_pairs)");

    // 3. 用 2D 峰值 ±100" 过滤 + 提取候选点对
    //    ±100" 容纳 s_est 误差 (±50-100") + 质心误差
    //    RANSAC 容差收紧到 3 像素, 精确剔除大部分外点
    double dxdy_filter_half = 100.0;
    int n_outside_dxdy = 0;
    for (const auto* pp : theta_filtered) {
        if (std::abs((*pp).tx - tx_peak_2d) > dxdy_filter_half ||
            std::abs((*pp).ty - ty_peak_2d) > dxdy_filter_half) {
            n_outside_dxdy++;
            continue;
        }
        // 提取 (img_i ↔ gaia_a)
        auto k1 = std::make_pair((*pp).img_i, (*pp).gaia_a);
        if (seen.insert(k1).second) {
            candidate_pairs.push_back({(*pp).img_i, (*pp).gaia_a, (*pp).s_est,
                                       (*pp).theta_rot_deg, (*pp).tx, (*pp).ty});
        }
        // 提取 (img_j ↔ gaia_b)
        auto k2 = std::make_pair((*pp).img_j, (*pp).gaia_b);
        if (seen.insert(k2).second) {
            candidate_pairs.push_back({(*pp).img_j, (*pp).gaia_b, (*pp).s_est,
                                       (*pp).theta_rot_deg, (*pp).tx, (*pp).ty});
        }
    }

    logger.info("候选点对: " + std::to_string(candidate_pairs.size()) +
                " (去重后, θ±" + std::to_string(theta_filter_half) + "° + 2D峰值 ±" +
                std::to_string(dxdy_filter_half) + "\")" +
                " 区外=" + std::to_string(n_outside_dxdy));

    // ========================================================================
    // 10. ⭐ RANSAC 剔除外点 + Umeyama SVD 一步求解
    //     候选已通过 θ ±2° + 2D 峰值 ±100" 过滤 (含真匹配 + 部分外点)
    //     容差 3 像素 (精确剔除大部分外点), 10000 次迭代
    //     (候选 10-50 对, 真匹配 3-5 个, 10000 次足够采样到 2 真匹配)
    // ========================================================================
    int ransac_iters = 0;
    double ransac_rms = std::numeric_limits<double>::infinity();
    std::mt19937 ransac_rng(params.seed + 12345);
    ransac_solve(U_full, W_, candidate_pairs,
                 s0, 10000, 5.0 * s0,  // 10000 次迭代, 容差 5 像素
                 output.ransac_estimate, output.inlier_pairs,
                 ransac_iters, ransac_rms, ransac_rng, &logger);
    output.n_inliers = (int)output.inlier_pairs.size();
    output.n_ransac_iters = ransac_iters;
    output.ransac_rms = ransac_rms;
    output.success = output.ransac_estimate.valid && output.n_inliers >= 3;

    logger.info("=== RANSAC 最终结果 ===");
    if (output.ransac_estimate.valid) {
        logger.info("s=" + std::to_string(output.ransac_estimate.s) +
                    " θ=" + std::to_string(output.ransac_estimate.theta * EXP_RADTODEG) + "°" +
                    " tx=" + std::to_string(output.ransac_estimate.tx) +
                    " ty=" + std::to_string(output.ransac_estimate.ty) +
                    " inliers=" + std::to_string(output.n_inliers) +
                    " RMS=" + std::to_string(ransac_rms) + "\"" +
                    " iters=" + std::to_string(ransac_iters));
    } else {
        logger.warn("RANSAC 求解失败");
    }

    // 误差 (对比 RANSAC 结果 vs 真值, 仅模拟数据)
    if (has_ground_truth && ground_truth.valid && output.ransac_estimate.valid) {
        double theta_gt_deg = ground_truth.theta * EXP_RADTODEG;
        double theta_ransac_deg = output.ransac_estimate.theta * EXP_RADTODEG;
        output.err_theta_deg = angle_diff_deg(theta_ransac_deg, theta_gt_deg);
        output.err_tx = std::abs(output.ransac_estimate.tx - ground_truth.tx);
        output.err_ty = std::abs(output.ransac_estimate.ty - ground_truth.ty);
        output.err_s = std::abs(output.ransac_estimate.s - ground_truth.s);

        logger.info("真值: θ=" + std::to_string(theta_gt_deg) + "°" +
                    " tx=" + std::to_string(ground_truth.tx) +
                    " ty=" + std::to_string(ground_truth.ty) +
                    " s=" + std::to_string(ground_truth.s));
        logger.info("误差: Δθ=" + std::to_string(output.err_theta_deg) + "°" +
                    " Δtx=" + std::to_string(output.err_tx) + "\"" +
                    " Δty=" + std::to_string(output.err_ty) + "\"" +
                    " Δs=" + std::to_string(output.err_s));
    }

    // ========================================================================
    // 9. 构造 3D 密度场切片 (供 Python 可视化)
    // ========================================================================
    DensitySlice& slice = output.density_final;
    slice.th_bins = params.th_bins;
    slice.dxdy_bins = params.dxdy_bins;
    slice.th_lo = -180.0; slice.th_hi = 180.0;
    slice.tx_lo = tx_range_lo; slice.tx_hi = tx_range_hi;
    slice.ty_lo = ty_range_lo; slice.ty_hi = ty_range_hi;
    slice.theta_tx.assign((size_t)params.th_bins * params.dxdy_bins, 0.0);
    slice.theta_ty.assign((size_t)params.th_bins * params.dxdy_bins, 0.0);
    slice.tx_ty.assign((size_t)params.dxdy_bins * params.dxdy_bins, 0.0);

    for (auto& kv : density3d) {
        uint64_t key = kv.first;
        int th = (int)(key / ((uint64_t)params.dxdy_bins * params.dxdy_bins));
        int tx = (int)((key / (uint64_t)params.dxdy_bins) % (uint64_t)params.dxdy_bins);
        int ty = (int)(key % (uint64_t)params.dxdy_bins);
        int v = kv.second;
        slice.theta_tx[(size_t)th * params.dxdy_bins + tx] += v;
        slice.theta_ty[(size_t)th * params.dxdy_bins + ty] += v;
        slice.tx_ty[(size_t)tx * params.dxdy_bins + ty] += v;
    }

    logger.info("=== exp44 runRelVecExperiment 完成 ===");
    return 0;
}

} // namespace exp44

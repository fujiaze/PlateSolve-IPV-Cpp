// ============================================================================
// vm_core.cpp - V4.2 VectorMatcher 核心实现（Task 3）
//
// 从 V4.1 的 record_and_filter 迁移核心逻辑:
//   Phase A: PROSAC 优先采样 + 1 点抽样 + θ 直方图 + 5N/10N 停止条件
//   Phase B: 三级过滤 → 1对1互斥 → Umeyama SVD → 迭代精修
//   4 模式并行(OpenMP), 选择最优模式(归一化评分最高)
//
// V4.2 改动:
//   - 移除 nanoflann 依赖, 改为线性扫描 NN(简化模块独立依赖)
//   - 移除 V4.1 的 Phase C/D/D'/E 逻辑(由其他模块负责)
//   - namespace 改为 v42
//   - 输出 cu/cw 匹配对 + 初始变换(s,θ,tx,ty) + 调试信息
//   - 详细日志输出(便于分析)
//
// 依赖: Eigen3, OpenMP
// ============================================================================

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
#include <unordered_set>
#include <chrono>
#include <omp.h>

#include "../include/vm_api.h"
#include "../include/vm_prosac.h"
#include "../common/v42_log.h"
#include "Eigen/Dense"

namespace v42 {

static constexpr double PI = 3.14159265358979323846;
static constexpr double DEGTORAD = PI / 180.0;
static constexpr double RADTODEG = 180.0 / PI;

// ============================================================================
// 基础几何工具
// ============================================================================

// 应用相似变换: Wt = s·R(θ)·W + (tx,ty)
static void apply_similarity(const double* W, int M, double s, double theta,
                             double tx, double ty, double* Wt) {
    double ct = std::cos(theta), st = std::sin(theta);
    for (int i = 0; i < M; ++i) {
        double wx = W[i*2], wy = W[i*2+1];
        Wt[i*2]     = s*(ct*wx - st*wy) + tx;
        Wt[i*2+1]   = s*(st*wx + ct*wy) + ty;
    }
}

// 应用翻转模式: 0=无翻转, 1=X翻转, 2=Y翻转, 3=XY翻转
static void apply_flip(const double* W, int M, int mode, double* Wf) {
    bool fx = (mode==1 || mode==3), fy = (mode==2 || mode==3);
    for (int i = 0; i < M; ++i) {
        Wf[i*2]   = fx ? -W[i*2]   : W[i*2];
        Wf[i*2+1] = fy ? -W[i*2+1] : W[i*2+1];
    }
}

static inline double angle_diff_deg(double a, double b) {
    double d = std::fmod(std::fmod(a-b+180.0,360.0)+360.0,360.0)-180.0;
    return std::abs(d);
}

static inline double wrap180(double d) {
    return std::fmod(std::fmod(d+180.0,360.0)+360.0,360.0)-180.0;
}

// 中位数
static double vec_median(std::vector<double> v) {
    size_t n = v.size();
    if (n == 0) return 0;
    std::nth_element(v.begin(), v.begin()+n/2, v.end());
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin()+n/2-1, v.end());
        return (v[n/2] + v[n/2-1]) * 0.5;
    }
    return v[n/2];
}

// ============================================================================
// 线性扫描 NN (替代 V4.1 的 nanoflann KDTree)
//   复杂度 O(N*M), 对 N,M ≤ 数千规模足够快
// ============================================================================

// 返回查询点 q 在 Wt 中的最近邻索引和距离平方
static inline std::pair<int,double> nn_query(const double* Wt, int M,
                                              double qx, double qy) {
    int best_idx = -1;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (int j = 0; j < M; ++j) {
        double dx = Wt[j*2]   - qx;
        double dy = Wt[j*2+1] - qy;
        double d2 = dx*dx + dy*dy;
        if (d2 < best_d2) { best_d2 = d2; best_idx = j; }
    }
    return {best_idx, best_d2};
}

// 统计: 对每颗 U[k], 找 Wt 中最近邻, 若距离 < max_dist 且 norm_U[k]/norm_Wf[idx] 在 [s_min,s_max] 内则计数
static int count_s_in_range(const double* U, int N, const double* Wt, int M,
                            const double* norm_U, const double* norm_Wf,
                            double s_min, double s_max, double max_dist) {
    if (N == 0 || M == 0) return 0;
    double d2max = max_dist * max_dist;
    int c = 0;
    for (int k = 0; k < N; ++k) {
        auto pr = nn_query(Wt, M, U[k*2], U[k*2+1]);
        if (pr.second > d2max) continue;
        int idx = pr.first;
        double sr = norm_U[k] / norm_Wf[idx];
        if (sr >= s_min && sr <= s_max) c++;
    }
    return c;
}

// 1对1 互斥匹配: 距离 < tau 的对按距离升序贪心选
struct InlierResult { int n_inliers; double rms; std::vector<int> inlier_mask; };
static InlierResult count_inliers_1to1(const double* U, int N,
                                        const double* Wt, int M, double tau) {
    InlierResult r; r.n_inliers = 0; r.rms = 0; r.inlier_mask.assign(N, 0);
    if (N == 0 || M == 0) return r;

    struct Match { int u, w; double d; };
    std::vector<Match> cand; cand.reserve(N);
    double t2 = tau * tau;
    for (int i = 0; i < N; ++i) {
        auto pr = nn_query(Wt, M, U[i*2], U[i*2+1]);
        if (pr.second <= t2) cand.push_back({i, pr.first, std::sqrt(pr.second)});
    }
    std::sort(cand.begin(), cand.end(),
              [](const Match& a, const Match& b){ return a.d < b.d; });
    std::vector<int> wu(M, 0); double ss = 0;
    for (auto& c : cand) {
        if (wu[c.w]) continue;
        wu[c.w] = 1; r.inlier_mask[c.u] = 1; ss += c.d * c.d; r.n_inliers++;
    }
    if (r.n_inliers > 0) r.rms = std::sqrt(ss / r.n_inliers);
    return r;
}

// 1对1 互斥匹配并返回 (u,w) 对集(用于重建 cu/cw)
struct PairResult { int n_inliers; double rms; std::vector<int> pairs_u, pairs_w; };
static PairResult build_pairs_1to1(const double* U, int N,
                                    const double* Wt, int M, double tau) {
    PairResult r; r.n_inliers = 0; r.rms = 0;
    if (N == 0 || M == 0) return r;
    struct Match { int u, w; double d; };
    std::vector<Match> cand; cand.reserve(N);
    double t2 = tau * tau;
    for (int i = 0; i < N; ++i) {
        auto pr = nn_query(Wt, M, U[i*2], U[i*2+1]);
        if (pr.second <= t2) cand.push_back({i, pr.first, std::sqrt(pr.second)});
    }
    std::sort(cand.begin(), cand.end(),
              [](const Match& a, const Match& b){ return a.d < b.d; });
    std::vector<int> wu(M, 0); double ss = 0;
    for (auto& c : cand) {
        if (wu[c.w]) continue;
        wu[c.w] = 1; ss += c.d * c.d; r.n_inliers++;
        r.pairs_u.push_back(c.u); r.pairs_w.push_back(c.w);
    }
    if (r.n_inliers > 0) r.rms = std::sqrt(ss / r.n_inliers);
    return r;
}

// 归一化评分: n/d × (1 - rms/tau)
static double compute_normalized_score(int n, double rms, int N, int M, double tau) {
    double d = std::min((double)N, (double)M);
    if (d <= 0 || tau <= 0) return 0;
    return ((double)n / d) * (1.0 - rms / tau);
}

// ============================================================================
// Umeyama SVD (2D 相似变换) — 使用 v42_types.h 中的 SimTransform
// ============================================================================

static SimTransform umeyama(const double* src, const double* dst, int n) {
    SimTransform r; r.valid = false; r.s = 1; r.theta = 0; r.tx = 0; r.ty = 0;
    if (n < 2) return r;
    using M2 = Eigen::Matrix2d; using V2 = Eigen::Vector2d;
    V2 ms = V2::Zero(), md = V2::Zero();
    for (int i = 0; i < n; ++i) {
        ms += V2(src[i*2], src[i*2+1]);
        md += V2(dst[i*2], dst[i*2+1]);
    }
    ms /= n; md /= n;
    Eigen::MatrixXd sc(2, n), dc(2, n);
    for (int i = 0; i < n; ++i) {
        sc(0,i) = src[i*2]   - ms(0); sc(1,i) = src[i*2+1] - ms(1);
        dc(0,i) = dst[i*2]   - md(0); dc(1,i) = dst[i*2+1] - md(1);
    }
    M2 H = sc * dc.transpose();
    Eigen::JacobiSVD<M2> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    V2 Sv = V2::Ones(); Sv(1) = det;
    M2 R = svd.matrixV() * Sv.asDiagonal() * svd.matrixU().transpose();
    double tr = sc.colwise().squaredNorm().sum();
    if (tr < 1e-15) return r;
    double s = svd.singularValues().dot(Sv) / tr;
    if (std::abs(s - 1.0) >= 0.1) return r;  // 尺度偏离过大
    double th = std::atan2(R(1,0), R(0,0));
    V2 t = md - s * R * ms;
    r.s = s; r.theta = th; r.tx = t(0); r.ty = t(1); r.valid = true;
    return r;
}

// ============================================================================
// 迭代 SVD 精修 (Phase B)
// ============================================================================
struct RefineResult {
    double s, theta, tx, ty; int n_inliers; double rms;
    std::vector<int> inlier_mask; bool success;
};

static RefineResult iterative_svd_refine(
    const double* U, int N, const double* Wf, int M,
    double s, double theta, double tx, double ty,
    double s0, int max_iter)
{
    RefineResult res; res.s = s; res.theta = theta; res.tx = tx; res.ty = ty;
    res.n_inliers = 0; res.rms = 1e30; res.success = false;
    if (N < 3 || M < 3) return res;

    std::vector<double> Wt(M * 2);
    std::vector<int> prev(N, 0);
    // 初始用宽松阈值(3*s0)捕获 s 偏差大的情况, 逐步放宽直到找到足够 inliers
    double tau = 3.0 * s0;
    apply_similarity(Wf, M, s, theta, tx, ty, Wt.data());
    auto inl = count_inliers_1to1(U, N, Wt.data(), M, tau);
    double sf[] = {5.0, 10.0, 20.0, 50.0, 100.0};
    for (int k = 0; k < 5 && inl.n_inliers < 3; ++k) {
        tau = sf[k] * s0;
        inl = count_inliers_1to1(U, N, Wt.data(), M, tau);
    }
    if (inl.n_inliers < 3) {
        res.inlier_mask = std::move(inl.inlier_mask);
        res.n_inliers = inl.n_inliers; res.rms = inl.rms;
        return res;
    }
    // 记录初始放宽到的阈值, 迭代时复用(随s收敛逐步收紧)
    double tau_iter = tau;
    prev = inl.inlier_mask;

    for (int iter = 0; iter < max_iter; ++iter) {
        apply_similarity(Wf, M, res.s, res.theta, res.tx, res.ty, Wt.data());
        struct M2 { int u, w; double d; };
        std::vector<M2> cand; cand.reserve(N);
        // 迭代时用当前 tau_iter 收集 inliers, 让 SVD 用足够点收敛
        double t2 = tau_iter * tau_iter;
        for (int i = 0; i < N; ++i) {
            auto pr = nn_query(Wt.data(), M, U[i*2], U[i*2+1]);
            if (pr.second <= t2) cand.push_back({i, pr.first, std::sqrt(pr.second)});
        }
        std::sort(cand.begin(), cand.end(),
                  [](const M2& a, const M2& b){ return a.d < b.d; });
        std::vector<int> wu(M, 0);
        std::vector<double> sp, dp;
        for (auto& c : cand) {
            if (wu[c.w]) continue;
            wu[c.w] = 1;
            sp.push_back(Wf[c.w*2]);   sp.push_back(Wf[c.w*2+1]);
            dp.push_back(U[c.u*2]);    dp.push_back(U[c.u*2+1]);
        }
        int np = (int)sp.size() / 2;
        if (np < 3) break;
        auto sim = umeyama(sp.data(), dp.data(), np);
        if (!sim.valid || std::abs(sim.s - 1.0) > 0.1) break;
        res.s = sim.s; res.theta = sim.theta; res.tx = sim.tx; res.ty = sim.ty;
        apply_similarity(Wf, M, res.s, res.theta, res.tx, res.ty, Wt.data());
        // 迭代评估用当前 tau_iter, 避免s未完全收敛时丢失inliers
        inl = count_inliers_1to1(U, N, Wt.data(), M, tau_iter);
        if (inl.n_inliers < 3) break;
        // 随 s 收敛, 逐步收紧 tau_iter (趋向 3*s0)
        if (inl.rms < tau_iter * 0.5 && tau_iter > 3.0 * s0) {
            tau_iter = std::max(3.0 * s0, inl.rms * 3.0);
        }
        if (inl.inlier_mask == prev) break;
        prev = inl.inlier_mask;
    }
    apply_similarity(Wf, M, res.s, res.theta, res.tx, res.ty, Wt.data());
    inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);
    res.n_inliers = inl.n_inliers; res.rms = inl.rms;
    res.inlier_mask = std::move(inl.inlier_mask);
    res.success = (res.n_inliers >= 3);
    return res;
}

// ============================================================================
// θ_SNR 计算
// ============================================================================
struct ThetaSNRResult { int peak_idx; double peak_deg, snr; };

static ThetaSNRResult compute_theta_snr(const double* hist, int nb, double bw) {
    ThetaSNRResult r; r.peak_idx = 0; r.peak_deg = 0; r.snr = 0;
    double pv = 0;
    for (int i = 0; i < nb; ++i) if (hist[i] > pv) { pv = hist[i]; r.peak_idx = i; }
    r.peak_deg = (r.peak_idx + 0.5) * bw - 180.0;
    double bs = 0; int bc = 0;
    for (int i = 0; i < nb; ++i) if (std::abs(i - r.peak_idx) > 5) { bs += hist[i]; bc++; }
    double bm = (bc > 10) ? bs / bc : 1.0;
    r.snr = pv / std::max(bm, 1e-10);
    return r;
}

// ============================================================================
// Phase A + Phase B 实现
// ============================================================================
struct PairRecord { int u_idx, w_idx; double theta_deg; int n_in_range_s; };

struct PhaseABResult {
    double s, theta, tx, ty; int n_inliers; double rms;
    double peak_snr; int n_samples; bool success;
    double theta_peak_deg; int best_n_range; double median_noise;
    int n_phasea_records; int n_phaseb_corr;
    std::vector<int> cu, cw;          // Phase B 匹配对的 U 和 Wf 索引
    double prosac_quality_median;
    int    prosac_pool_final;
};

static PhaseABResult record_and_filter(
    const double* U, int N, const double* Wf, int M,
    double s0, double s_min, double s_max,
    int K_total, int batch_size, int min_samples,
    int min_inliers, int seed, double fov_diag_asec,
    const double* snr_values, const int* is_saturated_values,
    double w_snr, double w_sparse, double w_sat,
    int prosac_T_max, int use_prosac,
    Logger& logger)
{
    PhaseABResult res; res.s = 1; res.theta = 0; res.tx = 0; res.ty = 0;
    res.n_inliers = 0; res.rms = 1e30; res.peak_snr = 0; res.n_samples = 0;
    res.success = false; res.theta_peak_deg = 0; res.best_n_range = 0;
    res.median_noise = 0; res.n_phasea_records = 0; res.n_phaseb_corr = 0;
    res.prosac_quality_median = 0.0; res.prosac_pool_final = 0;

    if (N < 2 || M < 2) return res;

    // 计算每颗星的模长与角度
    std::vector<double> norm_U(N), angle_U(N), norm_Wf(M), angle_Wf(M);
    std::vector<bool> valid_U(N, false), valid_Wf(M, false);
    for (int i = 0; i < N; ++i) {
        norm_U[i] = std::sqrt(U[i*2]*U[i*2] + U[i*2+1]*U[i*2+1]);
        angle_U[i] = std::atan2(U[i*2+1], U[i*2]);
        valid_U[i] = norm_U[i] > 1e-10;
    }
    for (int j = 0; j < M; ++j) {
        norm_Wf[j] = std::sqrt(Wf[j*2]*Wf[j*2] + Wf[j*2+1]*Wf[j*2+1]);
        angle_Wf[j] = std::atan2(Wf[j*2+1], Wf[j*2]);
        valid_Wf[j] = norm_Wf[j] > 1e-10;
    }

    // 稀疏度: 第3近邻距离(线性扫描)
    auto compute_sparsity3 = [](const double* pts, int NP) -> std::vector<double> {
        std::vector<double> sp(NP, 0.0);
        if (NP < 2) return sp;
        int kk = std::min(3, NP - 1);
        for (int i = 0; i < NP; ++i) {
            double xi = pts[i*2], yi = pts[i*2+1];
            std::vector<double> dists;
            dists.reserve(NP - 1);
            for (int j = 0; j < NP; ++j) {
                if (j == i) continue;
                double dx = pts[j*2] - xi, dy = pts[j*2+1] - yi;
                dists.push_back(std::sqrt(dx*dx + dy*dy));
            }
            std::sort(dists.begin(), dists.end());
            sp[i] = dists[kk - 1];
        }
        return sp;
    };
    auto sparsity_U = compute_sparsity3(U, N);
    auto sparsity_W = compute_sparsity3(Wf, M);

    double sp_med_u = vec_median(sparsity_U);
    double sp_med_w = vec_median(sparsity_W);
    logger.info("PhaseA 稀疏度: U中位=" + std::to_string(sp_med_u) +
                " W中位=" + std::to_string(sp_med_w));

    // ========================================================================
    // Phase A: PROSAC 优先采样
    // ========================================================================
    ProsacSampler prosac_sampler;
    bool use_prosac_local = (use_prosac != 0);
    if (use_prosac_local) {
        std::vector<double> snr_vec(N, 0.0);
        std::vector<bool> sat_vec(N, false);
        if (snr_values) {
            for (int i = 0; i < N; ++i) snr_vec[i] = snr_values[i];
        } else {
            // 无 SNR 时用 sparsity 倒数作代理
            for (int i = 0; i < N; ++i) {
                snr_vec[i] = sparsity_U[i] > 1e-10 ? 1.0 / sparsity_U[i] : 0.0;
            }
        }
        if (is_saturated_values) {
            for (int i = 0; i < N; ++i) sat_vec[i] = (is_saturated_values[i] != 0);
        }
        auto quality = compute_quality_score(snr_vec, sparsity_U, sat_vec,
                                             w_snr, w_sparse, w_sat);
        prosac_sampler.init(quality, prosac_T_max > 0 ? prosac_T_max : 10000,
                            static_cast<unsigned>(seed));
        res.prosac_quality_median = prosac_sampler.quality_median();
        logger.info("PhaseA PROSAC 启用: N=" + std::to_string(N) +
                    " T_max=" + std::to_string(prosac_T_max) +
                    " quality_median=" + std::to_string(res.prosac_quality_median));
    } else {
        logger.info("PhaseA PROSAC 禁用, 退化为纯随机均匀抽样");
    }

    // θ 直方图: 3600 个 bin, 每bin 0.1°, 范围[-180°, 180°)
    static constexpr int THB = 3600;
    static constexpr double THBW = 0.1;
    std::vector<double> th_hist(THB, 0);
    std::unordered_set<uint64_t> sampled;
    std::vector<PairRecord> records; records.reserve(K_total);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> ud(0, N-1), wd(0, M-1);
    std::vector<double> Wt(M * 2);
    double max_t = fov_diag_asec * 0.6;
    uint64_t tp = (uint64_t)N * M;
    int Kmax = std::min(K_total, (int)std::min((uint64_t)K_total, tp));

    int n_val = 0, best_n = 0, last_snr = 0;
    // 跟踪 Phase A best 样本的完整变换(作为 PhaseB 互斥对数不足时的优质初值)
    double best_s_a = 1.0, best_theta_a = 0.0, best_tx_a = 0.0, best_ty_a = 0.0;
    for (int iter = 0; iter < Kmax; ++iter) {
        int i, j;
        if (use_prosac_local) {
            i = prosac_sampler.sample(iter + 1);  // PROSAC t 从 1 开始
            j = wd(rng);
        } else {
            i = ud(rng); j = wd(rng);
        }
        if (i < 0 || i >= N || j < 0 || j >= M) continue;  // 防御性
        uint64_t key = (uint64_t)i * M + j;
        if (sampled.count(key)) continue;
        sampled.insert(key);
        if (!valid_U[i] || !valid_Wf[j]) continue;

        double s = norm_U[i] / norm_Wf[j];
        if (s < s_min || s > s_max) continue;
        double theta = angle_U[i] - angle_Wf[j];
        double td = wrap180(theta * RADTODEG);
        double ct = std::cos(theta), st = std::sin(theta);
        double tx = U[i*2]   - s*(ct*Wf[j*2]   - st*Wf[j*2+1]);
        double ty = U[i*2+1] - s*(st*Wf[j*2]   + ct*Wf[j*2+1]);
        if (std::abs(tx) > max_t || std::abs(ty) > max_t) continue;
        apply_similarity(Wf, M, s, theta, tx, ty, Wt.data());
        int nr = count_s_in_range(U, N, Wt.data(), M,
                                   norm_U.data(), norm_Wf.data(),
                                   s_min, s_max, 5.0 * s0);
        int tb = (int)((td + 180.0) / THBW);
        if (tb >= 0 && tb < THB) th_hist[tb] += nr;
        records.push_back({i, j, td, nr});
        if (nr > best_n) {
            best_n = nr;
            best_s_a = s; best_theta_a = theta;
            best_tx_a = tx; best_ty_a = ty;
            logger.debug("PhaseA new best_n=" + std::to_string(nr) +
                         " i=" + std::to_string(i) + " j=" + std::to_string(j) +
                         " s=" + std::to_string(s) + " θ=" + std::to_string(td) + "°");
        }
        n_val++;

        // 5N/10N 动态阈值停止
        if (n_val >= min_samples && iter - last_snr >= batch_size) {
            last_snr = iter;
            auto snr = compute_theta_snr(th_hist.data(), THB, THBW);
            double t5 = std::min(5.0 * N, 500.0);
            double t10 = std::min(10.0 * N, 1000.0);
            logger.info("PhaseA n=" + std::to_string(n_val) +
                        " best=" + std::to_string(best_n) +
                        " peak=" + std::to_string(th_hist[snr.peak_idx]) +
                        " θ=" + std::to_string(snr.peak_deg) + "°" +
                        " SNR=" + std::to_string(snr.snr) + "x" +
                        " (5N=" + std::to_string(t5) + " 10N=" + std::to_string(t10) + ")");
            if (snr.snr >= t10) { logger.info("PhaseA ≥10N stop"); break; }
            if (snr.snr >= t5)  { logger.info("PhaseA ≥5N stop");  break; }
        }
    }

    if (use_prosac_local) {
        res.prosac_pool_final = prosac_sampler.last_pool_size();
    }
    res.n_samples = n_val;
    res.n_phasea_records = (int)records.size();

    // 计算 n_in_range 中位数
    std::vector<int> an; an.reserve(records.size());
    for (auto& r : records) an.push_back(r.n_in_range_s);
    std::sort(an.begin(), an.end());
    double mdn = an.empty() ? 0 : an[an.size() / 2];
    auto fsnr = compute_theta_snr(th_hist.data(), THB, THBW);
    res.peak_snr = fsnr.snr;
    res.theta_peak_deg = fsnr.peak_deg;
    res.best_n_range = best_n;
    res.median_noise = mdn;

    logger.info("PhaseA done: n=" + std::to_string(n_val) +
                " best=" + std::to_string(best_n) +
                " med=" + std::to_string(mdn) +
                " peak=" + std::to_string(th_hist[fsnr.peak_idx]) +
                " θ=" + std::to_string(fsnr.peak_deg) + "°" +
                " SNR=" + std::to_string(fsnr.snr) + "x" +
                " rec=" + std::to_string(records.size()));

    // ========================================================================
    // Phase B: 三级过滤 → 1对1互斥 → Umeyama SVD → 迭代精修
    // ========================================================================
    double tp_deg = fsnr.peak_deg;
    int n_rec = (int)records.size();
    double nthr, tband;
    std::vector<PairRecord> filt;

    if (n_rec >= 10) {
        // 严格过滤
        nthr = std::max(2.0, 1.5 * mdn); tband = 2.0;
        for (auto& r : records) {
            if (r.n_in_range_s <= (int)nthr) continue;
            if (angle_diff_deg(r.theta_deg, tp_deg) > tband) continue;
            filt.push_back(r);
        }
    }
    if (filt.size() < 2) {
        // 放宽: 降低阈值, 扩大 θ 带宽
        nthr = std::max(1.0, 1.0 * mdn); tband = 4.0;
        filt.clear();
        for (auto& r : records) {
            if (r.n_in_range_s < (int)nthr) continue;
            if (angle_diff_deg(r.theta_deg, tp_deg) > tband) continue;
            filt.push_back(r);
        }
    }
    if (filt.size() < 2) {
        // 最终放宽: n_in_range≥1, θ带宽 8°
        nthr = 1; tband = 8.0;
        filt.clear();
        for (auto& r : records) {
            if (r.n_in_range_s < (int)nthr) continue;
            if (angle_diff_deg(r.theta_deg, tp_deg) > tband) continue;
            filt.push_back(r);
        }
    }
    if (filt.size() < 2) {
        logger.warn("PhaseB: <2 pairs after filter");
        return res;
    }

    // 按 n_in_range 降序, 1对1互斥贪心选取
    std::sort(filt.begin(), filt.end(),
              [](const PairRecord& a, const PairRecord& b){ return a.n_in_range_s > b.n_in_range_s; });
    std::vector<int> uu(N, 0), wuu(M, 0);
    std::vector<int> cu, cw;
    for (auto& r : filt) {
        if (!uu[r.u_idx] && !wuu[r.w_idx]) {
            cu.push_back(r.u_idx); cw.push_back(r.w_idx);
            uu[r.u_idx] = 1; wuu[r.w_idx] = 1;
        }
    }
    res.cu = cu; res.cw = cw;
    logger.info("PhaseB: " + std::to_string(cu.size()) + " corr (filtered " +
                std::to_string(filt.size()) + " pairs)");
    if (cu.size() < 2) {
        logger.warn("PhaseB: <2 corr after 1to1");
        return res;
    }

    // Umeyama SVD (从互斥对集拟合)
    std::vector<double> sp(cu.size() * 2), dp(cu.size() * 2);
    for (size_t k = 0; k < cu.size(); ++k) {
        sp[k*2]   = Wf[cw[k]*2];   sp[k*2+1] = Wf[cw[k]*2+1];
        dp[k*2]   = U[cu[k]*2];    dp[k*2+1] = U[cu[k]*2+1];
    }
    auto sim = umeyama(sp.data(), dp.data(), (int)cu.size());
    if (!sim.valid) {
        // SVD 无效(对数太少或退化), 回退到 PhaseA best 样本变换
        logger.warn("PhaseB: SVD invalid, fallback to PhaseA best sample");
        sim.s = best_s_a; sim.theta = best_theta_a;
        sim.tx = best_tx_a; sim.ty = best_ty_a; sim.valid = true;
    }

    apply_similarity(Wf, M, sim.s, sim.theta, sim.tx, sim.ty, Wt.data());
    auto inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);

    // 选择迭代精修的初值:
    //   互斥对数充足(>=5)时用 SVD 结果; 不足时用 PhaseA best 样本变换
    //   (PhaseA best 样本已通过 n_in_range 验证, 变换基本正确,
    //    避免少量对的 SVD 偏差导致精修陷入局部最优)
    double init_s = sim.s, init_th = sim.theta, init_tx = sim.tx, init_ty = sim.ty;
    if (cu.size() < 5 && best_n >= 3) {
        logger.info("PhaseB: 互斥对数=" + std::to_string(cu.size()) +
                    " <5, 使用 PhaseA best 样本变换作初值 (best_n=" +
                    std::to_string(best_n) + ")");
        init_s = best_s_a; init_th = best_theta_a;
        init_tx = best_tx_a; init_ty = best_ty_a;
    }

    // 迭代 SVD 精修
    auto ref = iterative_svd_refine(U, N, Wf, M, init_s, init_th, init_tx, init_ty, s0, 10);
    if (ref.success) {
        res.s = ref.s; res.theta = ref.theta;
        res.tx = ref.tx; res.ty = ref.ty;
        res.n_inliers = ref.n_inliers; res.rms = ref.rms;
    } else {
        res.s = sim.s; res.theta = sim.theta;
        res.tx = sim.tx; res.ty = sim.ty;
        res.n_inliers = inl.n_inliers; res.rms = inl.rms;
    }

    // 用精修后的变换重建 cu/cw 对集(反映真实 inliers, 而非 PhaseB 互斥对数)
    apply_similarity(Wf, M, res.s, res.theta, res.tx, res.ty, Wt.data());
    auto pr_final = build_pairs_1to1(U, N, Wt.data(), M, 1.0 * s0);
    if (pr_final.n_inliers > 0) {
        cu = std::move(pr_final.pairs_u);
        cw = std::move(pr_final.pairs_w);
    }

    res.cu = cu; res.cw = cw;
    res.success = true;
    res.n_phaseb_corr = (int)cu.size();
    logger.info("PhaseB OK: s=" + std::to_string(res.s) +
                " θ=" + std::to_string(res.theta * RADTODEG) + "°" +
                " n=" + std::to_string(res.n_inliers) +
                " rms=" + std::to_string(res.rms) +
                " corr=" + std::to_string(cu.size()));
    return res;
}

// ============================================================================
// 单模式求解
// ============================================================================
struct ModeRes {
    bool success;
    int  mode;
    double s, theta, tx, ty;
    int n_inliers; double rms;
    double peak_snr; int n_samples;
    double norm_score;
    PhaseABResult ab;
};

static ModeRes solve_single_mode(
    const double* U, int N, const double* W, int M,
    int mode, double s0, double s_min, double s_max,
    int K_total, int batch_size, int min_samples,
    int min_inliers, int seed, double fov_diag_asec,
    const double* snr_values, const int* is_saturated_values,
    double w_snr, double w_sparse, double w_sat,
    int prosac_T_max, int use_prosac,
    Logger& logger)
{
    ModeRes mr; mr.success = false; mr.norm_score = 0; mr.peak_snr = 0;
    mr.n_samples = 0; mr.mode = mode;
    mr.s = 1; mr.theta = 0; mr.tx = 0; mr.ty = 0;
    mr.n_inliers = 0; mr.rms = 1e30;

    logger.info("mode" + std::to_string(mode) + ": N=" + std::to_string(N) +
                " M=" + std::to_string(M));
    std::vector<double> Wf(M * 2);
    apply_flip(W, M, mode, Wf.data());

    mr.ab = record_and_filter(U, N, Wf.data(), M, s0, s_min, s_max,
                               K_total, batch_size, min_samples, min_inliers,
                               seed + mode, fov_diag_asec,
                               snr_values, is_saturated_values,
                               w_snr, w_sparse, w_sat, prosac_T_max, use_prosac,
                               logger);
    mr.peak_snr = mr.ab.peak_snr; mr.n_samples = mr.ab.n_samples;
    if (!mr.ab.success) {
        logger.warn("mode" + std::to_string(mode) + " Phase A/B 失败");
        return mr;
    }
    mr.s = mr.ab.s; mr.theta = mr.ab.theta;
    mr.tx = mr.ab.tx; mr.ty = mr.ab.ty;
    mr.n_inliers = mr.ab.n_inliers; mr.rms = mr.ab.rms;
    mr.norm_score = compute_normalized_score(mr.n_inliers, mr.rms, N, M, 1.0 * s0);
    mr.success = true;
    return mr;
}

} // namespace v42

// ============================================================================
// C 接口实现
// ============================================================================

extern "C" VM_API int vm_match(
    const double* U, int N_img,
    const double* W, int M,
    const VectorMatcherParams* params,
    VectorMatchResult* result)
{
    using namespace v42;

    // 初始化 result(不动 cu/cw 指针)
    int* saved_cu = result->cu;
    int* saved_cw = result->cw;
    std::memset(result, 0, sizeof(VectorMatchResult));
    result->cu = saved_cu; result->cw = saved_cw;
    result->rms = 1e30;

    // 初始化日志
    Logger logger;
    bool log_enabled = false;
    if (params->log_file_path && params->log_file_path[0]) {
        logger.init(std::string(params->log_file_path));
        log_enabled = true;
        logger.info("=== vm_match 开始 ===");
        logger.info("N_img=" + std::to_string(N_img) + " M=" + std::to_string(M) +
                    " s0=" + std::to_string(params->s0) +
                    " s_min=" + std::to_string(params->s_min) +
                    " s_max=" + std::to_string(params->s_max) +
                    " n_modes=" + std::to_string(params->n_modes));
    }

    if (N_img < 2 || M < 2) {
        fprintf(stderr, "[v42_vm] N=%d M=%d too few\n", N_img, M);
        logger.error("N 或 M 过少");
        if (log_enabled) logger.close();
        return -1;
    }

    int n_modes = std::max(1, std::min(params->n_modes, 4));
    double fov_diag_asec = 0.0;
    if (params->s0 > 0) {
        // 估算 FOV 对角线(用 max_t = fov_diag*0.6 ≈ s0×5000 经验值)
        fov_diag_asec = std::max(1000.0, 5000.0 * params->s0);
    }
    logger.info("开始 " + std::to_string(n_modes) + " 模式并行求解");

    std::vector<ModeRes> mres(n_modes);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < n_modes; ++m) {
        mres[m] = solve_single_mode(U, N_img, W, M, m, params->s0,
                     params->s_min, params->s_max,
                     params->K_total, params->batch_size,
                     params->min_samples, params->min_inliers,
                     params->seed, fov_diag_asec,
                     params->snr_values, params->is_saturated_values,
                     params->w_snr, params->w_sparse, params->w_sat,
                     params->prosac_T_max, params->use_prosac,
                     logger);
    }

    // 模式选择: 归一化评分最高(与 V4.1 一致, 不强制 n_inliers >= min_inliers)
    //   V4.1 用 bayes_lnK - 10*rms 评分, V4.2 无贝叶斯, 用 norm_score
    //   min_inliers 仅作为 PhaseB success 的预留参数(实际 success 由 n_inliers>=3 决定)
    int best_mode = -1;
    double best_score = -1e30;
    for (int m = 0; m < n_modes; ++m) {
        if (!mres[m].success) continue;
        double score = mres[m].norm_score;
        logger.info("mode" + std::to_string(m) + " 评分: n_inliers=" +
                    std::to_string(mres[m].n_inliers) +
                    " rms=" + std::to_string(mres[m].rms) + "\"" +
                    " score=" + std::to_string(score));
        if (score > best_score) { best_score = score; best_mode = m; }
    }

    if (best_mode < 0) {
        fprintf(stderr, "[v42_vm] all modes failed\n");
        logger.error("所有模式均失败");
        if (log_enabled) logger.close();
        return -1;
    }

    const auto& best = mres[best_mode];
    result->s = best.s;
    result->theta = best.theta;
    result->tx = best.tx;
    result->ty = best.ty;
    result->rms = best.rms;
    result->best_n_range = best.ab.best_n_range;
    result->theta_snr = best.ab.peak_snr;
    result->theta_peak_deg = best.ab.theta_peak_deg;
    result->n_phasea_records = best.ab.n_phasea_records;
    result->prosac_quality_median = best.ab.prosac_quality_median;
    result->prosac_pool_final = best.ab.prosac_pool_final;
    result->best_mode = best_mode;
    result->success = 1;

    // 输出 cu/cw (调用方需调用 vm_free_result 释放)
    int np = (int)best.ab.cu.size();
    result->n_pairs = np;
    if (np > 0) {
        int* cu_arr = (int*)std::malloc(np * sizeof(int));
        int* cw_arr = (int*)std::malloc(np * sizeof(int));
        if (cu_arr && cw_arr) {
            std::memcpy(cu_arr, best.ab.cu.data(), np * sizeof(int));
            std::memcpy(cw_arr, best.ab.cw.data(), np * sizeof(int));
            result->cu = cu_arr;
            result->cw = cw_arr;
        } else {
            result->n_pairs = 0;
            if (cu_arr) std::free(cu_arr);
            if (cw_arr) std::free(cw_arr);
        }
    } else {
        result->cu = nullptr;
        result->cw = nullptr;
    }

    logger.info("=== vm_match 完成 === mode=" + std::to_string(best_mode) +
                " s=" + std::to_string(best.s) +
                " θ=" + std::to_string(best.theta * RADTODEG) + "°" +
                " n_inliers=" + std::to_string(best.n_inliers) +
                " rms=" + std::to_string(best.rms) + "\"" +
                " pairs=" + std::to_string(np));
    if (log_enabled) logger.close();
    return 0;
}

extern "C" VM_API void vm_free_result(VectorMatchResult* result) {
    if (!result) return;
    if (result->cu) { std::free(result->cu); result->cu = nullptr; }
    if (result->cw) { std::free(result->cw); result->cw = nullptr; }
    result->n_pairs = 0;
}

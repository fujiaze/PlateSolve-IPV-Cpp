// ============================================================================
// vm43_verify.cpp - V4.3 PairVerifier 模块 (Phase D / IRM Step 3)
//
// 职责: MAD 清洗 → RANSAC → 贝叶斯验证 → 三角形验证
// 从 V4.2 pv_core.cpp + pv_mad.cpp + pv_bayes.cpp + pv_triangle.cpp 迁移
// 新增: RANSAC 全局一致性 (4 点抽样 → 仿射拟合 → 全量验证 → 保留最大内点集)
//
// 接口:
//   int vm43_verify(candidates, U, W, s0, fov_diag_deg, params, output, logger)
//
// 算法流程:
//   1. Phase D-1: 3 轮 MAD 迭代清洗 (阈值 max(5", 3×1.4826×MAD))
//      - 初始 Umeyama 拟合 → 变换 W→Wt
//      - 鲁棒预过滤 (init_med > min_thresh 时剔除明显离群)
//      - 迭代: 收集残差 → MAD → 剔除 → 重新 Umeyama
//   2. Phase D-2: RANSAC 全局一致性 (V4.3 新增)
//      - 4 对抽样 → 仿射 LSQ 拟合 → 全量验证 → 最大内点集
//   3. Phase D-3: 贝叶斯假设验证
//      - lnK = Σ[-log(2πσ²) - r²/(2σ²)] + n×log(A_fov_sqsec)
//   4. Phase D-4: 三角形双特征验证
//      - 面积 A (海伦公式) + 极惯性矩 J = A×(a²+b²+c²)/36
//
// 约束: C++17, 单线程; 中文注释, UTF-8 编码
// ============================================================================

#include "vm43_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <unordered_set>
#include <cstdint>
#include <Eigen/Dense>

namespace v43 {

// --- 常量 ---
static constexpr double PV_PI = 3.14159265358979323846;
static constexpr double DEG_TO_ARCSEC = 3600.0;
static constexpr double DEGEN_AREA_EPS = 1e-6;       // 退化三角形面积阈值
static constexpr int    N_LARGE_THRESHOLD = 30;       // 触发随机采样的 n 阈值
static constexpr int    MAX_SAMPLES_N_LARGE = 1000;  // 大 n 随机采样上限
static constexpr uint32_t RNG_SEED = 0x5A5A5A5A;      // 固定随机种子

// ============================================================================
// 内部工具函数
// ============================================================================

// 中位数（会修改输入向量）
static double vec_median(std::vector<double>& v) {
    size_t n = v.size();
    if (n == 0) return 0.0;
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
        return (v[n / 2] + v[n / 2 - 1]) * 0.5;
    }
    return v[n / 2];
}

// 应用相似变换: Wt = s×(R×W) + t
static void apply_similarity(const std::vector<StarPoint>& W,
                             double s, double theta, double tx, double ty,
                             std::vector<StarPoint>& Wt) {
    double ct = std::cos(theta), st = std::sin(theta);
    Wt.resize(W.size());
    for (size_t i = 0; i < W.size(); ++i) {
        double wx = W[i].x, wy = W[i].y;
        Wt[i].x = s * (ct * wx - st * wy) + tx;
        Wt[i].y = s * (st * wx + ct * wy) + ty;
        Wt[i].flux = W[i].flux;
        Wt[i].saturated = W[i].saturated;
    }
}

// Umeyama 2D 相似变换拟合（SVD）
//   src→dst 的最佳相似变换: dst ≈ s×R×src + t
//   尺度约束: |s-1.0| < 0.1 (U 和 W 均为角秒坐标, 尺度应接近 1)
//   n < 2 或退化时返回 valid=false
static SimTransform umeyama_2d(const double* src, const double* dst, int n) {
    SimTransform r;
    r.valid = false; r.s = 1; r.theta = 0; r.tx = 0; r.ty = 0;
    if (n < 2) return r;

    using M2 = Eigen::Matrix2d;
    using V2 = Eigen::Vector2d;
    V2 ms = V2::Zero(), md = V2::Zero();
    for (int i = 0; i < n; ++i) {
        ms += V2(src[i * 2], src[i * 2 + 1]);
        md += V2(dst[i * 2], dst[i * 2 + 1]);
    }
    ms /= n; md /= n;

    Eigen::MatrixXd sc(2, n), dc(2, n);
    for (int i = 0; i < n; ++i) {
        sc(0, i) = src[i * 2]     - ms(0);
        sc(1, i) = src[i * 2 + 1] - ms(1);
        dc(0, i) = dst[i * 2]     - md(0);
        dc(1, i) = dst[i * 2 + 1] - md(1);
    }
    M2 H = sc * dc.transpose();
    Eigen::JacobiSVD<M2> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double det = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    V2 Sv = V2::Ones(); Sv(1) = det;
    M2 R = svd.matrixV() * Sv.asDiagonal() * svd.matrixU().transpose();
    double tr = sc.colwise().squaredNorm().sum();
    if (tr < 1e-15) return r;
    double s = svd.singularValues().dot(Sv) / tr;
    if (std::abs(s - 1.0) >= 0.1) return r;  // 尺度约束
    double th = std::atan2(R(1, 0), R(0, 0));
    V2 t = md - s * R * ms;
    r.s = s; r.theta = th; r.tx = t(0); r.ty = t(1); r.valid = true;
    return r;
}

// ============================================================================
// 三角形特征计算
// ============================================================================
struct TriangleFeatures {
    double area;    // 面积(角秒²)
    double moment;  // 极惯性矩 J = A×(a²+b²+c²)/36
    double a, b, c; // 三边长(角秒), a≤b≤c
};

static TriangleFeatures compute_triangle_features(
    double x1, double y1, double x2, double y2, double x3, double y3)
{
    TriangleFeatures f{0.0, 0.0, 0.0, 0.0, 0.0};
    double d12 = std::hypot(x1 - x2, y1 - y2);
    double d23 = std::hypot(x2 - x3, y2 - y3);
    double d13 = std::hypot(x1 - x3, y1 - y3);
    double s3[3] = {d12, d23, d13};
    std::sort(s3, s3 + 3);
    f.a = s3[0]; f.b = s3[1]; f.c = s3[2];
    double s = (f.a + f.b + f.c) * 0.5;
    double tmp = s * (s - f.a) * (s - f.b) * (s - f.c);
    if (tmp <= 0.0) return TriangleFeatures{0.0, 0.0, 0.0, 0.0, 0.0};
    f.area = std::sqrt(tmp);
    if (f.area < DEGEN_AREA_EPS)
        return TriangleFeatures{0.0, 0.0, f.a, f.b, f.c};
    f.moment = f.area * (f.a * f.a + f.b * f.b + f.c * f.c) / 36.0;
    return f;
}

// 将 (i,j,k) 编码为唯一 uint64_t 键 (i<j<k)
static inline uint64_t encode_triple(int i, int j, int k) {
    return ((uint64_t)i << 42) | ((uint64_t)j << 21) | (uint64_t)k;
}

// 对单个三角形组合验证
//   返回: 0=通过, 1=不通过, -1=退化(不计入)
static int check_one_triangle(
    const std::vector<std::array<double, 4>>& mp,
    int i, int j, int k, double eps_A, double eps_J)
{
    TriangleFeatures fi = compute_triangle_features(
        mp[i][0], mp[i][1], mp[j][0], mp[j][1], mp[k][0], mp[k][1]);
    TriangleFeatures fc = compute_triangle_features(
        mp[i][2], mp[i][3], mp[j][2], mp[j][3], mp[k][2], mp[k][3]);

    if (fi.area < DEGEN_AREA_EPS || fc.area < DEGEN_AREA_EPS) return -1;

    double maxA = std::max(fi.area, fc.area);
    double maxJ = std::max(fi.moment, fc.moment);
    if (maxA < DEGEN_AREA_EPS) return -1;
    if (maxJ < DEGEN_AREA_EPS) return -1;

    double rel_A = std::abs(fi.area - fc.area) / maxA;
    double rel_J = std::abs(fi.moment - fc.moment) / maxJ;

    if (rel_A < eps_A && rel_J < eps_J) return 0;
    return 1;
}

// ============================================================================
// Phase D-1: MAD 清洗 (3 轮迭代)
// ============================================================================
struct MadResult {
    std::vector<int> clean_u;       // 清洗后 U 索引
    std::vector<int> clean_w;       // 清洗后 W 索引
    int    n_removed;               // 剔除数
    int    iterations;              // 实际迭代次数
    double rms_arcsec;              // 清洗后 RMS(角秒)
    SimTransform transform;         // 最终变换参数
    bool   success;
};

static MadResult mad_clean(
    const std::vector<MatchPair>& candidates,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const VM43SolveParams& params,
    Logger* logger)
{
    MadResult result;
    result.n_removed = 0;
    result.iterations = 0;
    result.rms_arcsec = 0.0;
    result.success = false;

    int n_pairs = (int)candidates.size();

    // 不足 3 对, 无法拟合, 直接返回原始对
    if (n_pairs < 3) {
        for (const auto& p : candidates) {
            result.clean_u.push_back(p.u);
            result.clean_w.push_back(p.w);
        }
        result.transform.valid = false;
        result.success = (n_pairs >= 0);
        if (logger) logger->warnf("vm43_verify mad: 对数 < 3, 跳过清洗 (n=%d)", n_pairs);
        return result;
    }

    int max_iters = params.mad_iters;
    if (max_iters < 1) max_iters = 3;
    double thresh_factor = params.mad_threshold_factor;
    if (thresh_factor <= 0) thresh_factor = 3.0;
    double min_thresh = params.mad_min_threshold_arcsec;
    if (min_thresh <= 0) min_thresh = 5.0;

    // --- 初始 Umeyama 拟合 ---
    // src = W[pairs_w], dst = U[pairs_u]
    std::vector<double> src(n_pairs * 2), dst(n_pairs * 2);
    for (int i = 0; i < n_pairs; ++i) {
        int wi = candidates[i].w, ui = candidates[i].u;
        src[i * 2]     = W[wi].x;
        src[i * 2 + 1] = W[wi].y;
        dst[i * 2]     = U[ui].x;
        dst[i * 2 + 1] = U[ui].y;
    }
    SimTransform sim = umeyama_2d(src.data(), dst.data(), n_pairs);
    if (!sim.valid) {
        sim.s = 1.0; sim.theta = 0.0; sim.tx = 0.0; sim.ty = 0.0; sim.valid = false;
        if (logger) logger->warn("vm43_verify mad: 初始 Umeyama 失败, 使用恒等变换");
    }

    // 变换 W → Wt
    std::vector<StarPoint> Wt;
    apply_similarity(W, sim.s, sim.theta, sim.tx, sim.ty, Wt);

    // --- 鲁棒预过滤 ---
    // 初始 Umeyama 用所有点拟合, 可能被离群拉偏。当 init_med 较大时,
    // 用 thresh_factor × init_med 作为粗阈值剔除明显离群, 重新 Umeyama 收敛后
    // 再进入标准 MAD 迭代。不改变 MAD 阈值公式 max(5", 3×1.4826×MAD)。
    std::vector<bool> keep(n_pairs, true);
    {
        std::vector<double> init_res(n_pairs);
        for (int i = 0; i < n_pairs; ++i) {
            int ui = candidates[i].u, wi = candidates[i].w;
            double dx = U[ui].x - Wt[wi].x;
            double dy = U[ui].y - Wt[wi].y;
            init_res[i] = std::sqrt(dx * dx + dy * dy);
        }
        auto init_res_copy = init_res;
        double init_med = vec_median(init_res_copy);
        if (logger) logger->debugf("vm43_verify mad 预过滤检查: init_med=%.3f min_thresh=%.3f valid=%d",
                                    init_med, min_thresh, sim.valid ? 1 : 0);
        // 仅当残差中位数较大 (> min_thresh) 时执行预过滤, 说明初始变换被拉偏
        if (init_med > min_thresh) {
            double pre_thresh = std::max(min_thresh, thresh_factor * init_med);
            std::vector<double> sp2, dp2;
            int n_pre_removed = 0;
            for (int i = 0; i < n_pairs; ++i) {
                if (init_res[i] > pre_thresh) {
                    keep[i] = false;
                    ++n_pre_removed;
                } else {
                    int wi = candidates[i].w, ui = candidates[i].u;
                    sp2.push_back(W[wi].x);  sp2.push_back(W[wi].y);
                    dp2.push_back(U[ui].x);  dp2.push_back(U[ui].y);
                }
            }
            int n_keep = (int)sp2.size() / 2;
            if (n_pre_removed > 0 && n_keep >= 3) {
                auto sim2 = umeyama_2d(sp2.data(), dp2.data(), n_keep);
                if (sim2.valid) {
                    sim = sim2;
                    apply_similarity(W, sim.s, sim.theta, sim.tx, sim.ty, Wt);
                    if (logger) logger->debugf("vm43_verify mad 预过滤触发: init_med=%.3f pre_thresh=%.3f removed=%d keep=%d",
                                                init_med, pre_thresh, n_pre_removed, n_keep);
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
            int ui = candidates[i].u, wi = candidates[i].w;
            double dx = U[ui].x - Wt[wi].x;
            double dy = U[ui].y - Wt[wi].y;
            res_list.push_back(std::sqrt(dx * dx + dy * dy));
        }
        if (res_list.size() < 10) break;

        // MAD = 1.4826 × median(|r_i - median(r)|)
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

        if (logger) logger->debugf("vm43_verify mad 轮%d: thresh=%.3f removed=%d keep=%d",
                                    iter, thresh, n_removed_this, (int)res_list.size() - n_removed_this);

        if (n_removed_this == 0) break;

        // 用剩余点重新 Umeyama 拟合
        std::vector<double> sp, dp;
        for (int i = 0; i < n_pairs; ++i) {
            if (!keep[i]) continue;
            int wi = candidates[i].w, ui = candidates[i].u;
            sp.push_back(W[wi].x);  sp.push_back(W[wi].y);
            dp.push_back(U[ui].x);  dp.push_back(U[ui].y);
        }
        auto sim2 = umeyama_2d(sp.data(), dp.data(), (int)sp.size() / 2);
        if (!sim2.valid) break;  // 拟合失败, 保留当前变换
        sim = sim2;
        apply_similarity(W, sim.s, sim.theta, sim.tx, sim.ty, Wt);
    }
    result.iterations = (iter < max_iters) ? iter + 1 : max_iters;
    result.transform = sim;

    // --- 收集清洗后对 + 计算 RMS ---
    double ssq = 0.0;
    int nc = 0;
    for (int i = 0; i < n_pairs; ++i) {
        if (!keep[i]) continue;
        result.clean_u.push_back(candidates[i].u);
        result.clean_w.push_back(candidates[i].w);
        int ui = candidates[i].u, wi = candidates[i].w;
        double dx = U[ui].x - Wt[wi].x;
        double dy = U[ui].y - Wt[wi].y;
        ssq += dx * dx + dy * dy;
        nc++;
    }
    result.n_removed = n_pairs - nc;
    result.rms_arcsec = (nc > 0) ? std::sqrt(ssq / nc) : 0.0;
    result.success = true;

    if (logger) logger->infof("vm43_verify mad: %d 轮, 清洗后 %d 对 (剔除 %d), RMS=%.3f\"",
                               result.iterations, nc, result.n_removed, result.rms_arcsec);

    return result;
}

// ============================================================================
// Phase D-2: RANSAC 全局一致性 (V4.3 新增)
// ============================================================================
// 抽样 4 对 → 仿射 LSQ 拟合 → 全量验证 → 保留最大内点集
struct RansacResult {
    std::vector<int> inlier_indices;  // 在 candidates 中的索引
    int n_inliers;
    bool success;
};

static RansacResult ransac_global(
    const std::vector<MatchPair>& candidates,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const VM43SolveParams& params,
    Logger* logger)
{
    RansacResult r;
    r.n_inliers = 0;
    r.success = false;

    int n = (int)candidates.size();
    int min_inliers = params.irm_ransac_min_inliers;
    if (min_inliers < 4) min_inliers = 4;
    int max_iter = params.irm_ransac_max_iter;
    if (max_iter < 10) max_iter = 200;

    // 不足 4 对, 无法拟合仿射, 直接全部作为内点
    if (n < 4) {
        for (int i = 0; i < n; ++i) r.inlier_indices.push_back(i);
        r.n_inliers = n;
        r.success = true;
        if (logger) logger->warnf("vm43_verify ransac: n=%d < 4, 跳过", n);
        return r;
    }

    // RANSAC 容差: 用 mad_min_threshold_arcsec
    double tol = params.mad_min_threshold_arcsec;
    if (tol <= 0) tol = 5.0;

    std::mt19937 rng(42);  // 固定种子保证可复现
    std::uniform_int_distribution<int> dist(0, n - 1);

    int best_n_inliers = 0;
    std::vector<int> best_inliers;

    for (int it = 0; it < max_iter; ++it) {
        // 随机抽样 4 对 (互异)
        int idx[4];
        for (int k = 0; k < 4; ++k) {
            idx[k] = dist(rng);
            for (int j = 0; j < k; ++j) {
                if (idx[j] == idx[k]) { idx[k] = dist(rng); j = -1; }
            }
        }

        // 用 4 对拟合仿射变换: [u_x, u_y] = A×[w_x, w_y] + b
        // 6 参数 (a11, a12, a21, a22, bx, by), 4 对 = 8 方程, 超定
        Eigen::MatrixXd A(8, 6);
        Eigen::VectorXd b(8);
        for (int k = 0; k < 4; ++k) {
            int ui = candidates[idx[k]].u, wi = candidates[idx[k]].w;
            double wx = W[wi].x, wy = W[wi].y;
            double ux = U[ui].x, uy = U[ui].y;
            A.row(k * 2)     << wx, wy, 0,  0,  1, 0;
            A.row(k * 2 + 1) << 0,  0,  wx, wy, 0, 1;
            b(k * 2)     = ux;
            b(k * 2 + 1) = uy;
        }
        Eigen::VectorXd sol = A.colPivHouseholderQr().solve(b);
        if (!sol.allFinite()) continue;
        double a11 = sol(0), a12 = sol(1), a21 = sol(2), a22 = sol(3);
        double bx = sol(4), by = sol(5);

        // 全量验证
        std::vector<int> inliers;
        for (int i = 0; i < n; ++i) {
            int ui = candidates[i].u, wi = candidates[i].w;
            double wx = W[wi].x, wy = W[wi].y;
            double ux_pred = a11 * wx + a12 * wy + bx;
            double uy_pred = a21 * wx + a22 * wy + by;
            double dx = U[ui].x - ux_pred;
            double dy = U[ui].y - uy_pred;
            double rr = std::sqrt(dx * dx + dy * dy);
            if (rr < tol) inliers.push_back(i);
        }

        if ((int)inliers.size() > best_n_inliers) {
            best_n_inliers = (int)inliers.size();
            best_inliers = inliers;
        }
        // 几乎全部内点, 提前终止
        if (best_n_inliers >= n - 1) break;
    }

    if (best_n_inliers >= min_inliers) {
        r.inlier_indices = best_inliers;
        r.n_inliers = best_n_inliers;
        r.success = true;
    } else {
        // RANSAC 未找到足够内点, 全部保留 (后续贝叶斯/三角形兜底)
        for (int i = 0; i < n; ++i) r.inlier_indices.push_back(i);
        r.n_inliers = n;
        r.success = false;
    }

    if (logger) logger->infof("vm43_verify ransac: best_inliers=%d/%d (min=%d) success=%d",
                               best_n_inliers, n, min_inliers, r.success ? 1 : 0);

    return r;
}

// ============================================================================
// Phase D-3: 贝叶斯假设验证
// ============================================================================
// 参考: Lang 2010 Astrometry.net 综述 §5.6
//   匹配假设 P(数据|H) = Π_i (1/(2πσ²)) × exp(-r_i²/(2σ²))
//   零假设   P(数据|¬H) = (1/A_fov)^n  (随机分布)
//   lnK = Σ_i[-log(2πσ²) - r_i²/(2σ²)] + n×log(A_fov_sqsec)
//   A_fov_sqsec = π × (fov_diag_deg/2)² × 3600²
//   决策: lnK > lnK_accept → 1 (接受); > lnK_weak → 0 (弱证据); 否则 → -1 (拒绝)
struct BayesResult {
    double lnK;
    int    n_match;
    double rms_arcsec;
    double sigma;
    int    decision;  // 1=接受, 0=弱证据, -1=拒绝
};

static BayesResult bayes_verify(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double sigma_min, double mad_rms_arcsec,
    double fov_diag_deg,
    double lnK_accept, double lnK_weak,
    Logger* logger)
{
    BayesResult result;
    result.lnK = 0.0;
    result.n_match = (int)matched_pairs.size();
    result.rms_arcsec = 0.0;
    result.sigma = 0.0;
    result.decision = -1;

    if (matched_pairs.empty()) {
        if (logger) logger->warn("vm43_verify bayes: 匹配对为空, 返回拒绝");
        return result;
    }

    // sigma 估计: max(sigma_min, mad_rms)
    double sigma = std::max(sigma_min, mad_rms_arcsec);
    if (sigma <= 0.0) sigma = sigma_min;
    if (sigma <= 0.0) {
        if (logger) logger->warn("vm43_verify bayes: sigma 无效, 返回拒绝");
        return result;
    }
    result.sigma = sigma;

    // A_fov 计算: π × (fov_diag/2)² × 3600² (平方角秒)
    if (fov_diag_deg <= 0.0) {
        if (logger) logger->warn("vm43_verify bayes: fov_diag_deg 无效, 返回拒绝");
        return result;
    }
    double fov_diag_arcsec = fov_diag_deg * DEG_TO_ARCSEC;
    double A_fov_sqsec = PV_PI * (fov_diag_arcsec * 0.5) * (fov_diag_arcsec * 0.5);

    int n = (int)matched_pairs.size();

    // 1. 计算残差 RMS
    double sum_r2 = 0.0;
    std::vector<double> residuals;
    residuals.reserve(n);
    for (const auto& p : matched_pairs) {
        double dx = p[0] - p[2];  // img_x - cat_x
        double dy = p[1] - p[3];  // img_y - cat_y
        double r = std::sqrt(dx * dx + dy * dy);
        residuals.push_back(r);
        sum_r2 += r * r;
    }
    result.rms_arcsec = std::sqrt(sum_r2 / n);

    // 2. 匹配假设对数似然 lnL_match
    double sigma_sq = sigma * sigma;
    double log_2pi_sigma_sq = std::log(2.0 * PV_PI * sigma_sq);
    double lnL_match = 0.0;
    for (double r : residuals) {
        lnL_match += -log_2pi_sigma_sq - (r * r) / (2.0 * sigma_sq);
    }

    // 3. 零假设对数似然 lnL_null
    double log_A_fov_sqsec = std::log(A_fov_sqsec);
    double lnL_null = -(double)n * log_A_fov_sqsec;

    // 4. 对数贝叶斯因子
    result.lnK = lnL_match - lnL_null;

    // 5. 决策
    if (result.lnK > lnK_accept) {
        result.decision = 1;  // 接受
    } else if (result.lnK > lnK_weak) {
        result.decision = 0;  // 弱证据
    } else {
        result.decision = -1; // 拒绝
    }

    if (logger) logger->infof("vm43_verify bayes: n=%d σ=%.3f RMS=%.3f A_fov=%.0f lnK=%.3f decision=%d",
                               n, sigma, result.rms_arcsec, A_fov_sqsec, result.lnK, result.decision);

    return result;
}

// ============================================================================
// Phase D-4: 三角形双特征验证
// ============================================================================
// 参考: Cole 2006 (三角形不变量在 platesolve 中的应用综述)
//   特征1: 面积 A (海伦公式)
//   特征2: 极惯性矩 J = A×(a²+b²+c²)/36
//   通过条件: |A_img-A_cat|/max < eps_A && |J_img-J_cat|/max < eps_J
//   退化三角形 (A<1e-6) 不计入总数
//   n ≤ 30: 遍历所有 C(n,3); n > 30: 随机采样 min(C(n,3), 1000)
struct TriangleResult {
    int    total;
    int    passed;
    double pass_ratio;
    bool   accepted;
};

static TriangleResult triangle_verify(
    const std::vector<std::array<double, 4>>& matched_pairs,
    double eps_A, double eps_J, double pass_rate_threshold,
    Logger* logger)
{
    TriangleResult r;
    r.total = 0;
    r.passed = 0;
    r.pass_ratio = 0.0;
    r.accepted = false;

    int n = (int)matched_pairs.size();
    if (n < 3) {
        if (logger) logger->warnf("vm43_verify triangle: n=%d < 3, 跳过", n);
        return r;
    }

    int total_valid = 0;
    int passed = 0;

    if (n <= N_LARGE_THRESHOLD) {
        // 遍历所有 C(n,3) 组合
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    int rc = check_one_triangle(matched_pairs, i, j, k, eps_A, eps_J);
                    if (rc == -1) continue;
                    ++total_valid;
                    if (rc == 0) ++passed;
                }
            }
        }
    } else {
        // 随机采样
        double cn3_d = (double)n * (n - 1) * (n - 2) / 6.0;
        int sample_target = (int)(std::min)(cn3_d, (double)MAX_SAMPLES_N_LARGE);
        if (sample_target < 1) sample_target = 1;

        std::mt19937 rng(RNG_SEED);
        std::uniform_int_distribution<int> dist(0, n - 1);
        std::unordered_set<uint64_t> seen;
        seen.reserve((size_t)sample_target * 2);

        int attempts = 0;
        int max_attempts = sample_target * 20;

        while ((int)seen.size() < sample_target && attempts < max_attempts) {
            ++attempts;
            int i = dist(rng), j = dist(rng), k = dist(rng);
            if (i == j || j == k || i == k) continue;
            if (i > j) std::swap(i, j);
            if (j > k) std::swap(j, k);
            if (i > j) std::swap(i, j);

            uint64_t key = encode_triple(i, j, k);
            if (seen.count(key)) continue;
            seen.insert(key);

            int rc = check_one_triangle(matched_pairs, i, j, k, eps_A, eps_J);
            if (rc == -1) continue;
            ++total_valid;
            if (rc == 0) ++passed;
        }
    }

    r.total = total_valid;
    r.passed = passed;
    r.pass_ratio = (total_valid > 0) ? (double)passed / (double)total_valid : 0.0;
    r.accepted = (r.pass_ratio > pass_rate_threshold);

    if (logger) logger->infof("vm43_verify triangle: n=%d total=%d passed=%d ratio=%.3f accepted=%d",
                               n, total_valid, passed, r.pass_ratio, r.accepted ? 1 : 0);

    return r;
}

// ============================================================================
// vm43_verify - 主入口
// ============================================================================
int vm43_verify(
    const std::vector<MatchPair>& candidates,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    double fov_diag_deg,
    const VM43SolveParams& params,
    VerificationResult& output,
    Logger* logger)
{
    (void)s0;  // 当前未直接使用 (MAD 内部通过 Umeyama 估计尺度)

    // --- 初始化 output ---
    output.inliers.clear();
    output.n_clean = 0;
    output.bayes_lnK = 0.0;
    output.bayes_n_match = 0;
    output.bayes_decision = -1;
    output.triangle_total = 0;
    output.triangle_passed = 0;
    output.triangle_pass_ratio = 0.0;
    output.ransac_n_inliers = 0;
    output.n_before_geometry = (int)candidates.size();
    output.n_after_geometry = (int)candidates.size();
    output.validated = false;
    output.success = false;

    if (logger) logger->infof("=== vm43_verify 开始 (n_candidates=%d, N_img=%d, M=%d, fov=%.3f°) ===",
                               (int)candidates.size(), (int)U.size(), (int)W.size(), fov_diag_deg);

    // --- 边界检查 ---
    if (candidates.empty()) {
        if (logger) logger->warn("vm43_verify: candidates 为空");
        return -1;
    }

    // --- Phase D-1: MAD 清洗 ---
    if (logger) logger->info("Phase D-1: MAD 清洗开始");
    MadResult mad = mad_clean(candidates, U, W, params, logger);

    output.n_clean = (int)mad.clean_u.size();

    if (output.n_clean < 3) {
        if (logger) logger->warnf("vm43_verify: 清洗后对数 < 3, 跳过验证 (n_clean=%d)",
                                   output.n_clean);
        output.validated = false;
        output.success = true;
        return 0;
    }

    // 用清洗后对构造 RANSAC 候选
    std::vector<MatchPair> clean_pairs;
    clean_pairs.reserve(mad.clean_u.size());
    for (size_t i = 0; i < mad.clean_u.size(); ++i) {
        clean_pairs.push_back({mad.clean_u[i], mad.clean_w[i]});
    }

    // --- Phase D-2: RANSAC 全局一致性 ---
    if (logger) logger->info("Phase D-2: RANSAC 全局一致性开始");
    auto ransac = ransac_global(clean_pairs, U, W, params, logger);
    output.ransac_n_inliers = ransac.n_inliers;

    // 取 RANSAC 内点作为后续验证候选 (失败则用 clean_pairs)
    std::vector<MatchPair> verify_pairs;
    if (ransac.success && ransac.n_inliers >= 3) {
        verify_pairs.reserve(ransac.inlier_indices.size());
        for (int idx : ransac.inlier_indices) {
            verify_pairs.push_back(clean_pairs[idx]);
        }
    } else {
        verify_pairs = clean_pairs;
    }

    // --- 构造 matched_pairs: (img_x, img_y, cat_x, cat_y) ---
    // cat 坐标 = 变换后的 Wt
    std::vector<StarPoint> Wt;
    if (mad.transform.valid) {
        apply_similarity(W, mad.transform.s, mad.transform.theta,
                         mad.transform.tx, mad.transform.ty, Wt);
    } else {
        // Umeyama 失败, Wt = W (恒等变换)
        Wt = W;
    }

    std::vector<std::array<double, 4>> matched_pairs;
    matched_pairs.reserve(verify_pairs.size());
    for (const auto& p : verify_pairs) {
        matched_pairs.push_back({U[p.u].x, U[p.u].y, Wt[p.w].x, Wt[p.w].y});
    }

    // --- Phase D-3: 贝叶斯验证 ---
    if (logger) logger->info("Phase D-3: 贝叶斯验证开始");
    double sigma_min = params.mad_min_threshold_arcsec * 0.1;  // σ 下限
    if (sigma_min <= 0) sigma_min = 0.5;
    auto br = bayes_verify(matched_pairs, sigma_min, mad.rms_arcsec,
                            fov_diag_deg, params.lnK_accept, params.lnK_weak, logger);
    output.bayes_lnK = br.lnK;
    output.bayes_n_match = br.n_match;
    output.bayes_decision = br.decision;

    // --- Phase D-4: 三角形验证 ---
    if (logger) logger->info("Phase D-4: 三角形验证开始");
    auto tr = triangle_verify(matched_pairs, params.eps_A, params.eps_J,
                               params.triangle_pass_rate, logger);
    output.triangle_total = tr.total;
    output.triangle_passed = tr.passed;
    output.triangle_pass_ratio = tr.pass_ratio;

    // --- 综合验证 ---
    // validated = (bayes_decision >= 0) && (triangle_pass_ratio >= threshold)
    bool bayes_ok = (br.decision >= 0);
    bool tri_ok = (tr.pass_ratio >= params.triangle_pass_rate);
    output.validated = (bayes_ok && tri_ok);

    // 填充内点集 (验证后的匹配对)
    output.inliers = verify_pairs;
    output.success = true;

    if (logger) logger->infof("=== vm43_verify 完成: validated=%d (bayes=%d tri=%d ransac=%d) ===",
                               output.validated ? 1 : 0, bayes_ok ? 1 : 0, tri_ok ? 1 : 0,
                               ransac.success ? 1 : 0);

    return 0;
}

} // namespace v43

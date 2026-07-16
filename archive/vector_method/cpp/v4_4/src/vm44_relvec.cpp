// vm44_relvec.cpp - V4.4 相对向量法 (DMPDV) Phase A 实现
//
// V4.3 的单θ采样假设 t=0, 在 t≠0 (Type3失败帧) 时 SNR<5 失败。
// V4.4 相对向量法 (Δu_ij = U[i]-U[j]) 消除平移 t, 完全替代单θ Phase A。
//
// 算法 (DMPDV):
//   1. 预计算 W 距离矩阵 D_W (N_w × N_w)
//   2. 预构建 Gaia 星对数组 (距离/角度/索引), 按距离排序 (k-vector)
//   3. 预构建 W 距离索引 D_W_sorted_ (每颗星 a 的距离排序数组, 第三星验证用)
//   4. 图像星对采样 (i,j): d_img, θ_img
//   5. k-vector 距离查询: d_gaia ∈ [d_img/s_max, d_img/s_min]
//   6. 多第三星加权验证 (投票无上限, 重叠越多票数越高)
//   7. 3D (θ, dx, dy) 密度场投票 + 递归聚焦 (单点法, 歧义待解决)
//   8. SNR = peak_cluster / background_mean
//
// V4.4 优化 (2026-06-30):
//   - 3D 密度场: (θ, dx, dy) 稀疏直方图, 真阳性形成密集簇
//   - 单点法: dx,dy 用 U[i]-s·R(θ)·W[a] 计算 (歧义问题后续解决)
//   - s_est 每对星独立估计: s_est = d_img / d_gaia_ab (无量纲, 真匹配≈1.0)
//     每对星 s_est 补偿实际 s 偏差 (如 s=0.9823), 使 dx/dy 聚集在 (tx,ty);
//     曾尝试定死 s_est=1.0 → 失败 (s 偏差×W范围≈±224" 使 dx/dy 分散, 3D 峰值模糊)
//   - 递归聚焦: 探索→识别→聚焦→收敛, 高 SNR 区域到达置信区间后丢弃噪声
//   - 聚焦区内候选 = 高可靠真匹配, 直接用于 Phase B RANSAC 精修
//   - 自适应采样停止: SNR 收敛即停止

#include "vm44_internal.h"
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

namespace v44 {

static constexpr double VM44_PI = 3.14159265358979323846;
static constexpr double VM44_RADTODEG = 180.0 / VM44_PI;
static constexpr double VM44_DEGTORAD = VM44_PI / 180.0;

// 计时工具
using Clock = std::chrono::steady_clock;
using ms_double = std::chrono::duration<double, std::milli>;

// ============================================================================
// 3D (θ, dx, dy) 密度场参数 (V4.4 递归聚焦, 替代 2D 直方图)
//   - θ:  360 bin × 1° → [-180, 180)°
//   - dx: 200 bin × (dx_range/200) → 动态范围 (根据 U 坐标确定)
//   - dy: 200 bin × (dy_range/200) → 动态范围
// 真阳性在 (θ_peak, dx_peak, dy_peak) 形成密集簇 (单点法, 歧义待解决)
// 假阳性在 3D 空间分散 → 3D 聚类比 2D 更精确
// 内存: 360×200×200 = 14.4M bin, 用 unordered_map 稀疏存储 (只存非零 bin)
// ============================================================================
static constexpr int RELVEC_TH_BINS_3D = 360;
static constexpr double RELVEC_TH_BW_3D = 1.0;   // 度/bin
static constexpr int RELVEC_DXDY_BINS = 200;

// 2D (θ, s) 直方图参数 (兼容保留, 用于 s_peak 估计)
static constexpr int RELVEC_TH_BINS = 3600;
static constexpr double RELVEC_TH_BW = 0.1;   // 度/bin
static constexpr int RELVEC_S_BINS = 200;

// 聚焦区域 (递归聚焦状态机)
struct FocusRegion {
    double th_lo, th_hi;   // θ 范围 (度)
    double dx_lo, dx_hi;   // dx 范围 (角秒)
    double dy_lo, dy_hi;   // dy 范围 (角秒)
    bool   confirmed;      // 是否已确认 (SNR > threshold)
};

// 3D 聚焦输出 (match 方法通过引用输出)
struct MatchOutput3D {
    int    peak_th_idx;        // 3D 峰值 θ bin
    int    peak_dx_idx;        // 3D 峰值 dx bin
    int    peak_dy_idx;        // 3D 峰值 dy bin
    int    peak_cluster_val;   // 5×5×5 邻域累加最大值
    int    n_focused;          // 聚焦区内候选数 (高可靠)
    double snr_3d;             // 3D SNR
    FocusRegion region;        // 聚焦区域
    double dx_range_lo, dx_range_hi;  // dx 动态范围
    double dy_range_lo, dy_range_hi;  // dy 动态范围
};

// 从 3D 密度场计算 SNR (峰值簇 / 背景均值)
static double compute_snr_3d(int peak_cluster, int total_votes, int n_nonzero) {
    if (total_votes <= 0 || n_nonzero <= 0) return 0.0;
    double bg_est = (double)total_votes / (double)n_nonzero;
    return (double)peak_cluster / std::max(bg_est, 1.0);
}

// 3D 峰值检测 (5×5×5 邻域累加, 只扫非零 bin)
// θ 维度环形 (±180° 等价), dx/dy 维度有边界
static void detect_peak_3d(
    const std::unordered_map<uint64_t, int>& density3d,
    int total_votes_3d,
    int& peak_th, int& peak_dx, int& peak_dy,
    int& peak_cluster, double& snr)
{
    peak_cluster = 0;
    peak_th = 0; peak_dx = 0; peak_dy = 0;
    for (auto& kv : density3d) {
        uint64_t key = kv.first;
        int th = (int)(key / ((uint64_t)RELVEC_DXDY_BINS * RELVEC_DXDY_BINS));
        int dx = (int)((key / (uint64_t)RELVEC_DXDY_BINS) % (uint64_t)RELVEC_DXDY_BINS);
        int dy = (int)(key % (uint64_t)RELVEC_DXDY_BINS);
        int cluster = 0;
        for (int dt = -2; dt <= 2; ++dt) {
            int t2 = th + dt;
            if (t2 < 0) t2 += RELVEC_TH_BINS_3D;
            if (t2 >= RELVEC_TH_BINS_3D) t2 -= RELVEC_TH_BINS_3D;
            int dxlo = std::max(0, dx - 2), dxhi = std::min(RELVEC_DXDY_BINS - 1, dx + 2);
            int dylo = std::max(0, dy - 2), dyhi = std::min(RELVEC_DXDY_BINS - 1, dy + 2);
            for (int dx2 = dxlo; dx2 <= dxhi; ++dx2) {
                for (int dy2 = dylo; dy2 <= dyhi; ++dy2) {
                    uint64_t k2 = ((uint64_t)t2 * RELVEC_DXDY_BINS + dx2) * RELVEC_DXDY_BINS + dy2;
                    auto it = density3d.find(k2);
                    if (it != density3d.end()) cluster += it->second;
                }
            }
        }
        if (cluster > peak_cluster) {
            peak_cluster = cluster;
            peak_th = th; peak_dx = dx; peak_dy = dy;
        }
    }
    snr = compute_snr_3d(peak_cluster, total_votes_3d, (int)density3d.size());
}

// ============================================================================
// RelativeVectorMatcher: 相对向量法核心类
// ============================================================================

class RelativeVectorMatcher {
public:
    RelativeVectorMatcher(const std::vector<StarPoint>& Wf,
                          double s_min, double s_max,
                          double min_len_frac, double max_len_frac,
                          Logger& logger)
        : W_(Wf), s_min_(s_min), s_max_(s_max), logger_(logger)
    {
        N_w_ = (int)Wf.size();
        if (N_w_ < 3) {
            logger.warn("RelVecMatcher: N_w=" + std::to_string(N_w_) + " 过少");
            return;
        }

        // 1. 预计算 W 距离矩阵 D_W[i*N_w+j] = |W[i]-W[j]|
        D_W_.assign((size_t)N_w_ * N_w_, 0.0);
        double d_max_global = 0.0;
        for (int i = 0; i < N_w_; ++i) {
            for (int j = i + 1; j < N_w_; ++j) {
                double dx = Wf[i].x - Wf[j].x;
                double dy = Wf[i].y - Wf[j].y;
                double d = std::sqrt(dx*dx + dy*dy);
                D_W_[(size_t)i * N_w_ + j] = d;
                D_W_[(size_t)j * N_w_ + i] = d;
                if (d > d_max_global) d_max_global = d;
            }
        }

        // 2. 预构建 Gaia 星对数组 (i<j), 按距离排序
        double d_min = d_max_global * min_len_frac;
        double d_max = d_max_global * max_len_frac;

        struct RawPair { double dist, angle; int a, b; };
        std::vector<RawPair> raw_pairs;
        raw_pairs.reserve((size_t)N_w_ * N_w_ / 4);
        for (int i = 0; i < N_w_; ++i) {
            for (int j = i + 1; j < N_w_; ++j) {
                double d = D_W_[(size_t)i * N_w_ + j];
                if (d >= d_min && d <= d_max) {
                    double dx = Wf[i].x - Wf[j].x;
                    double dy = Wf[i].y - Wf[j].y;
                    raw_pairs.push_back({d, std::atan2(dy, dx), i, j});
                }
            }
        }
        std::sort(raw_pairs.begin(), raw_pairs.end(),
                  [](const RawPair& a, const RawPair& b){ return a.dist < b.dist; });

        pair_dist_.resize(raw_pairs.size());
        pair_angle_.resize(raw_pairs.size());
        pair_a_.resize(raw_pairs.size());
        pair_b_.resize(raw_pairs.size());
        for (size_t k = 0; k < raw_pairs.size(); ++k) {
            pair_dist_[k]  = raw_pairs[k].dist;
            pair_angle_[k] = raw_pairs[k].angle;
            pair_a_[k]     = raw_pairs[k].a;
            pair_b_[k]     = raw_pairs[k].b;
        }

        // 3. 预构建 W 距离索引 D_W_sorted_[a] = [(dist, c), ...] 按 dist 排序
        D_W_sorted_.resize(N_w_);
        for (int a = 0; a < N_w_; ++a) {
            D_W_sorted_[a].reserve(N_w_);
            for (int c = 0; c < N_w_; ++c) {
                if (c == a) continue;
                D_W_sorted_[a].push_back({D_W_[(size_t)a * N_w_ + c], c});
            }
            std::sort(D_W_sorted_[a].begin(), D_W_sorted_[a].end(),
                      [](const std::pair<double,int>& p1, const std::pair<double,int>& p2){
                          return p1.first < p2.first;
                      });
        }

        logger.info("RelVecMatcher: N_w=" + std::to_string(N_w_) +
                    " Gaia星对=" + std::to_string(pair_dist_.size()) +
                    " 距离范围[" + std::to_string(d_min) + "," +
                    std::to_string(d_max) + "]\"");
    }

    // 运行匹配, 返回 2D (θ,s) 直方图 (兼容 s_peak) + 3D 聚焦信息
    //   3D 密度场 (θ, dx, dy) 稀疏存储, 真阳性形成密集簇
    //   递归聚焦: 探索→识别→聚焦→收敛, 高 SNR 区域到达置信区间后丢弃噪声
    //   单点法: dx,dy 用 U[i]-s·R(θ)·W[a] 计算 (歧义问题后续解决)
    std::vector<int> match(
        const std::vector<StarPoint>& U_full,
        double s0,
        const VM44SolveParams& params,
        unsigned seed,
        int& n_total_cand, int& n_passed,
        std::vector<RelVecPair>& passed_pairs,
        double& t_sample_ms, double& t_verify_ms,
        int& actual_samples,
        MatchOutput3D& out3d)
    {
        // 2D (θ,s) 直方图 (兼容, 用于 s_peak 估计)
        std::vector<int> hist2d(RELVEC_TH_BINS * RELVEC_S_BINS, 0);
        int peak_val_2d = 0;
        int total_votes_2d = 0;
        double s_bw = (s_max_ - s_min_) / (double)RELVEC_S_BINS;

        // 3D (θ,dx,dy) 密度场 (稀疏存储, 真阳性形成密集簇)
        std::unordered_map<uint64_t, int> density3d;
        density3d.reserve(100000);
        int total_votes_3d = 0;

        n_total_cand = 0; n_passed = 0;
        passed_pairs.clear();
        t_sample_ms = 0; t_verify_ms = 0;
        actual_samples = 0;
        out3d = {};
        out3d.region.confirmed = false;
        out3d.peak_cluster_val = 0;
        out3d.snr_3d = 0;
        out3d.n_focused = 0;

        if (N_w_ < 3 || pair_dist_.empty()) return hist2d;

        int max_u = params.relvec_max_u;
        int max_samples = params.relvec_n_samples;
        double third_star_tol_px = params.relvec_third_star_tol;
        int max_cand = params.relvec_max_cand;
        int n_third_stars = params.relvec_n_third_stars;

        // U 组限流: 按 flux 降序取前 max_u 颗
        int N_u_full = (int)U_full.size();
        std::vector<int> u_idx(N_u_full);
        std::iota(u_idx.begin(), u_idx.end(), 0);
        std::sort(u_idx.begin(), u_idx.end(),
                  [&](int a, int b){ return U_full[a].flux > U_full[b].flux; });
        int N_u = std::min(max_u, N_u_full);
        std::vector<int> u_sel(u_idx.begin(), u_idx.begin() + N_u);

        if (N_u < 3) {
            logger_.warn("RelVecMatcher: 限流后 N_u=" + std::to_string(N_u) + " 过少");
            return hist2d;
        }

        // 预计算 U 限流后距离矩阵
        std::vector<double> D_U((size_t)N_u * N_u, 0.0);
        for (int i = 0; i < N_u; ++i) {
            for (int j = i + 1; j < N_u; ++j) {
                double dx = U_full[u_sel[i]].x - U_full[u_sel[j]].x;
                double dy = U_full[u_sel[i]].y - U_full[u_sel[j]].y;
                double d = std::sqrt(dx*dx + dy*dy);
                D_U[(size_t)i * N_u + j] = d;
                D_U[(size_t)j * N_u + i] = d;
            }
        }

        // 确定 dx,dy 动态范围 (根据 U 坐标, 留 margin 余量)
        double ux_min = std::numeric_limits<double>::infinity();
        double ux_max = -std::numeric_limits<double>::infinity();
        double uy_min = std::numeric_limits<double>::infinity();
        double uy_max = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < N_u; ++i) {
            const auto& u = U_full[u_sel[i]];
            ux_min = std::min(ux_min, u.x); ux_max = std::max(ux_max, u.x);
            uy_min = std::min(uy_min, u.y); uy_max = std::max(uy_max, u.y);
        }
        double margin = 200.0;
        out3d.dx_range_lo = ux_min - margin;
        out3d.dx_range_hi = ux_max + margin;
        out3d.dy_range_lo = uy_min - margin;
        out3d.dy_range_hi = uy_max + margin;
        double dx_bw = (out3d.dx_range_hi - out3d.dx_range_lo) / (double)RELVEC_DXDY_BINS;
        double dy_bw = (out3d.dy_range_hi - out3d.dy_range_lo) / (double)RELVEC_DXDY_BINS;

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> ud_u(0, N_u - 1);

        // 自适应停止参数
        bool adaptive_stop = (params.relvec_adaptive_stop != 0);
        int min_samples = std::max(params.relvec_min_samples, 1);
        int check_interval = std::max(params.relvec_check_interval, 1);
        double snr_eps = params.relvec_snr_eps;
        int max_stable = std::max(params.relvec_max_stable, 1);
        double prev_snr = 0.0;
        int stable_count = 0;

        // 递归聚焦状态
        FocusRegion& region = out3d.region;
        int n_focused = 0;
        int n_discarded = 0;

        for (int s = 0; s < max_samples; ++s) {
            actual_samples = s + 1;
            auto t0 = Clock::now();

            // 单次采样 + 验证 (do-while(false) + break 替代 goto)
            do {
                int i = ud_u(rng), j = ud_u(rng);
                if (i == j) { t_sample_ms += ms_double(Clock::now() - t0).count(); break; }
                double d_img = D_U[(size_t)i * N_u + j];
                if (d_img < 1.0) { t_sample_ms += ms_double(Clock::now() - t0).count(); break; }
                double theta_img = std::atan2(
                    U_full[u_sel[i]].y - U_full[u_sel[j]].y,
                    U_full[u_sel[i]].x - U_full[u_sel[j]].x);

                // k-vector 距离查询
                double d_lo = d_img / s_max_;
                double d_hi = d_img / s_min_;
                int i_lo = (int)(std::lower_bound(pair_dist_.begin(), pair_dist_.end(), d_lo) - pair_dist_.begin());
                int i_hi = (int)(std::upper_bound(pair_dist_.begin(), pair_dist_.end(), d_hi) - pair_dist_.begin());
                int n_cand = i_hi - i_lo;
                n_total_cand += n_cand;
                if (n_cand == 0) { t_sample_ms += ms_double(Clock::now() - t0).count(); break; }

                // 候选限流
                std::vector<int> cand_indices;
                if (n_cand > max_cand) {
                    std::vector<int> all_idx(n_cand);
                    std::iota(all_idx.begin(), all_idx.end(), i_lo);
                    std::shuffle(all_idx.begin(), all_idx.end(), rng);
                    cand_indices.assign(all_idx.begin(), all_idx.begin() + max_cand);
                    n_cand = max_cand;
                } else {
                    cand_indices.resize(n_cand);
                    std::iota(cand_indices.begin(), cand_indices.end(), i_lo);
                }

                t_sample_ms += ms_double(Clock::now() - t0).count();

                // 第三星 k 列表
                auto t_v0 = Clock::now();
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

                // 对每个候选 (a,b) 做多第三星加权验证
                for (int ci : cand_indices) {
                    int a = pair_a_[ci], b = pair_b_[ci];
                    double d_gaia_ab = pair_dist_[ci];
                    if (d_gaia_ab < 1e-6) continue;
                    // s_est 每对星独立估计 (U/W 都是角秒, s_est 无量纲, 真匹配≈1.0)
                    //   - 每对星 s_est 补偿实际 s 偏差 (如 s=0.9823), 使 dx/dy 聚集在 (tx,ty)
                    //   - 定死 s_est=1.0 会导致 dx/dy 分散 (s 偏差×W范围≈±224"), 3D 峰值模糊
                    double s_est = d_img / d_gaia_ab;

                    // 像素容差转换 + 距离反比相对容差
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

                        const auto& da = D_W_sorted_[a];
                        auto lo_it = std::lower_bound(da.begin(), da.end(), d_ik_exp - d_ik_tol,
                            [](const std::pair<double,int>& p, double v){ return p.first < v; });
                        auto hi_it = std::upper_bound(da.begin(), da.end(), d_ik_exp + d_ik_tol,
                            [](double v, const std::pair<double,int>& p){ return v < p.first; });

                        bool k_passed = false;
                        for (auto it = lo_it; it != hi_it; ++it) {
                            int c = it->second;
                            if (c == b) continue;
                            if (std::abs(D_W_[(size_t)b * N_w_ + c] - d_jk_exp) < d_jk_tol) {
                                k_passed = true; break;
                            }
                        }
                        if (k_passed) n_k_passed++;
                    }

                    if (n_k_passed > 0) {
                        double theta_rot = (pair_angle_[ci] - theta_img) * VM44_RADTODEG;
                        theta_rot = std::fmod(theta_rot + 180.0, 360.0);
                        if (theta_rot < 0) theta_rot += 360.0;
                        theta_rot -= 180.0;

                        // 单点法计算 dx, dy (U[i] - s·R(θ)·W[a] = t, 真匹配时)
                        // 注: 不使用中点法, 歧义问题后续解决
                        double th_rad = theta_rot * VM44_DEGTORAD;
                        double ct_r = std::cos(th_rad), st_r = std::sin(th_rad);
                        double ux = U_full[u_sel[i]].x;
                        double uy = U_full[u_sel[i]].y;
                        double wx = W_[a].x;
                        double wy = W_[a].y;
                        double dx_est = ux - s_est * (ct_r * wx - st_r * wy);
                        double dy_est = uy - s_est * (st_r * wx + ct_r * wy);

                        // 聚焦模式: 候选不在聚焦区域内 → 丢弃噪声
                        if (region.confirmed) {
                            if (theta_rot < region.th_lo || theta_rot > region.th_hi ||
                                dx_est < region.dx_lo || dx_est > region.dx_hi ||
                                dy_est < region.dy_lo || dy_est > region.dy_hi) {
                                n_discarded++;
                                continue;
                            }
                            n_focused++;
                        }

                        // 填入 2D 直方图 (兼容 s_peak)
                        int th_idx = (int)((theta_rot + 180.0) / RELVEC_TH_BW);
                        if (th_idx < 0) th_idx = 0;
                        if (th_idx >= RELVEC_TH_BINS) th_idx = RELVEC_TH_BINS - 1;
                        int s_idx = (int)((s_est - s_min_) / s_bw);
                        if (s_idx < 0) s_idx = 0;
                        if (s_idx >= RELVEC_S_BINS) s_idx = RELVEC_S_BINS - 1;
                        int bin2d = th_idx * RELVEC_S_BINS + s_idx;
                        hist2d[bin2d] += n_k_passed;
                        if (hist2d[bin2d] > peak_val_2d) peak_val_2d = hist2d[bin2d];
                        total_votes_2d += n_k_passed;

                        // 填入 3D 密度场 (稀疏, 单点法)
                        int th3 = (int)((theta_rot + 180.0) / RELVEC_TH_BW_3D);
                        if (th3 < 0) th3 = 0;
                        if (th3 >= RELVEC_TH_BINS_3D) th3 = RELVEC_TH_BINS_3D - 1;
                        int dx3 = (int)((dx_est - out3d.dx_range_lo) / dx_bw);
                        if (dx3 < 0) dx3 = 0;
                        if (dx3 >= RELVEC_DXDY_BINS) dx3 = RELVEC_DXDY_BINS - 1;
                        int dy3 = (int)((dy_est - out3d.dy_range_lo) / dy_bw);
                        if (dy3 < 0) dy3 = 0;
                        if (dy3 >= RELVEC_DXDY_BINS) dy3 = RELVEC_DXDY_BINS - 1;
                        uint64_t key3 = ((uint64_t)th3 * RELVEC_DXDY_BINS + dx3) * RELVEC_DXDY_BINS + dy3;
                        density3d[key3] += n_k_passed;
                        total_votes_3d += n_k_passed;

                        n_passed++;
                        passed_pairs.push_back({u_sel[i], u_sel[j], a, b, theta_rot, s_est, dx_est, dy_est});
                    }
                }

                t_verify_ms += ms_double(Clock::now() - t_v0).count();
            } while (false);

            // 进度日志 (每 1000 samples)
            if ((s + 1) % 1000 == 0) {
                logger_.info("  RelVec 进度 " + std::to_string(s + 1) + "/" +
                             std::to_string(max_samples) + ": 候选=" +
                             std::to_string(n_total_cand) + " 通过=" +
                             std::to_string(n_passed) + " 3D非零bin=" +
                             std::to_string(density3d.size()) +
                             (region.confirmed ? " 聚焦内=" + std::to_string(n_focused) +
                              " 丢弃=" + std::to_string(n_discarded) : ""));
            }

            // 递归聚焦检查 (每 check_interval 次, 达到 min_samples 后)
            if ((s + 1) >= min_samples && (s + 1) % check_interval == 0) {
                if (!region.confirmed) {
                    // 阶段 2: 识别 - 3D 峰值检测
                    int pk_th, pk_dx, pk_dy, pk_cluster;
                    double snr;
                    detect_peak_3d(density3d, total_votes_3d, pk_th, pk_dx, pk_dy, pk_cluster, snr);

                    if (snr > 10.0) {
                        // 确认聚焦区域 (±3°, ±30")
                        double pk_th_deg = (pk_th + 0.5) * RELVEC_TH_BW_3D - 180.0;
                        double pk_dx_val = out3d.dx_range_lo + (pk_dx + 0.5) * dx_bw;
                        double pk_dy_val = out3d.dy_range_lo + (pk_dy + 0.5) * dy_bw;
                        region.th_lo = pk_th_deg - 3.0; region.th_hi = pk_th_deg + 3.0;
                        region.dx_lo = pk_dx_val - 30.0; region.dx_hi = pk_dx_val + 30.0;
                        region.dy_lo = pk_dy_val - 30.0; region.dy_hi = pk_dy_val + 30.0;
                        region.confirmed = true;

                        out3d.peak_th_idx = pk_th;
                        out3d.peak_dx_idx = pk_dx;
                        out3d.peak_dy_idx = pk_dy;
                        out3d.peak_cluster_val = pk_cluster;
                        out3d.snr_3d = snr;

                        logger_.info("RelVec 3D聚焦确认: s=" + std::to_string(s + 1) +
                                     " θ=" + std::to_string(pk_th_deg) + "°" +
                                     " dx=" + std::to_string(pk_dx_val) +
                                     " dy=" + std::to_string(pk_dy_val) +
                                     " SNR=" + std::to_string(snr) + "x" +
                                     " cluster=" + std::to_string(pk_cluster) +
                                     " 非零bin=" + std::to_string(density3d.size()));
                    }
                    prev_snr = snr;
                } else {
                    // 阶段 3: 聚焦 - 逐步缩小区域 (每 200 次收紧 40%)
                    if ((s + 1) % 200 == 0) {
                        double th_mid = 0.5 * (region.th_lo + region.th_hi);
                        double dx_mid = 0.5 * (region.dx_lo + region.dx_hi);
                        double dy_mid = 0.5 * (region.dy_lo + region.dy_hi);
                        double th_half = 0.4 * (region.th_hi - region.th_lo);
                        double dx_half = 0.4 * (region.dx_hi - region.dx_lo);
                        double dy_half = 0.4 * (region.dy_hi - region.dy_lo);
                        region.th_lo = th_mid - th_half; region.th_hi = th_mid + th_half;
                        region.dx_lo = dx_mid - dx_half; region.dx_hi = dx_mid + dx_half;
                        region.dy_lo = dy_mid - dy_half; region.dy_hi = dy_mid + dy_half;
                    }

                    // 阶段 4: 收敛检查 - 重新检测峰值 SNR
                    int pk_th, pk_dx, pk_dy, pk_cluster;
                    double snr;
                    detect_peak_3d(density3d, total_votes_3d, pk_th, pk_dx, pk_dy, pk_cluster, snr);
                    out3d.peak_cluster_val = pk_cluster;
                    out3d.snr_3d = snr;

                    if (adaptive_stop && prev_snr > 0.0 && snr > 5.0) {
                        double rel_change = std::abs(snr - prev_snr) / prev_snr;
                        if (rel_change < snr_eps) {
                            stable_count++;
                            if (stable_count >= max_stable) {
                                logger_.info("RelVec 3D SNR收敛停止: s=" + std::to_string(s + 1) +
                                             "/" + std::to_string(max_samples) +
                                             " SNR=" + std::to_string(snr) +
                                             " 聚焦内=" + std::to_string(n_focused) +
                                             " 丢弃=" + std::to_string(n_discarded) +
                                             " (连续" + std::to_string(stable_count) + "次)");
                                break;
                            }
                        } else {
                            stable_count = 0;
                        }
                    }
                    prev_snr = snr;
                }
            }
        }

        // 最终 3D 峰值检测 (若聚焦未确认, 仍需获取最佳峰值)
        if (!region.confirmed) {
            int pk_th, pk_dx, pk_dy, pk_cluster;
            double snr;
            detect_peak_3d(density3d, total_votes_3d, pk_th, pk_dx, pk_dy, pk_cluster, snr);
            out3d.peak_th_idx = pk_th;
            out3d.peak_dx_idx = pk_dx;
            out3d.peak_dy_idx = pk_dy;
            out3d.peak_cluster_val = pk_cluster;
            out3d.snr_3d = snr;
        }

        out3d.n_focused = n_focused;
        return hist2d;
    }

private:
    const std::vector<StarPoint>& W_;
    int N_w_ = 0;
    double s_min_, s_max_;
    Logger& logger_;
    std::vector<double> D_W_;
    std::vector<double> pair_dist_;
    std::vector<double> pair_angle_;
    std::vector<int>    pair_a_, pair_b_;
    std::vector<std::vector<std::pair<double,int>>> D_W_sorted_;
};

// ============================================================================
// 外部接口 vm44_relvec_match
// ============================================================================

int vm44_relvec_match(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& Wf,
    double s0,
    const VM44SolveParams& params,
    RelVecResult& output,
    Logger* logger)
{
    Logger default_logger;
    Logger& log = logger ? *logger : default_logger;
    log.info("=== vm44_relvec_match 开始 (3D递归聚焦) ===");

    auto t_total_start = Clock::now();

    output.success = false;
    output.theta_peak_deg = 0;
    output.s_peak = 0;
    output.dx_peak = 0;
    output.dy_peak = 0;
    output.theta_snr = 0;
    output.n_samples = 0;
    output.n_total_candidates = 0;
    output.n_passed = 0;
    output.n_focused = 0;

    if ((int)U.size() < 3 || (int)Wf.size() < 3) {
        log.warn("vm44_relvec_match: U(" + std::to_string(U.size()) +
                 ") 或 Wf(" + std::to_string(Wf.size()) + ") 过少");
        return -1;
    }

    // 构造匹配器
    auto t_construct_start = Clock::now();
    RelativeVectorMatcher matcher(Wf, params.s_min, params.s_max,
                                   params.relvec_min_len_frac,
                                   params.relvec_max_len_frac, log);
    double t_construct_ms = ms_double(Clock::now() - t_construct_start).count();
    log.info("RelVec 构造耗时: " + std::to_string(t_construct_ms) + " ms");

    // 匹配 (返回 2D 直方图 + 3D 聚焦信息)
    int n_total_cand = 0, n_passed = 0;
    double t_sample_ms = 0, t_verify_ms = 0;
    int actual_samples = 0;
    MatchOutput3D out3d;
    auto hist2d = matcher.match(U, s0, params,
                                (unsigned)params.seed + 100,
                                n_total_cand, n_passed,
                                output.passed_pairs,
                                t_sample_ms, t_verify_ms,
                                actual_samples, out3d);

    output.n_samples = actual_samples;
    output.n_total_candidates = n_total_cand;
    output.n_passed = n_passed;
    output.n_focused = out3d.n_focused;

    log.info("RelVec 采样耗时: " + std::to_string(t_sample_ms) + " ms");
    log.info("RelVec 第三星验证耗时: " + std::to_string(t_verify_ms) + " ms");
    log.info("RelVec 总采样+验证: " + std::to_string(t_sample_ms + t_verify_ms) + " ms");
    log.info("RelVec 统计: N_u=" + std::to_string((int)U.size()) +
             " N_w=" + std::to_string((int)Wf.size()) +
             " max_u=" + std::to_string(params.relvec_max_u) +
             " actual_samples=" + std::to_string(actual_samples) +
             "/" + std::to_string(params.relvec_n_samples) +
             " n_third_stars=" + std::to_string(params.relvec_n_third_stars) +
             " tol=" + std::to_string(params.relvec_third_star_tol) + "px");
    log.info("RelVec 结果: n_total_cand=" + std::to_string(n_total_cand) +
             " n_passed=" + std::to_string(n_passed) +
             " n_focused=" + std::to_string(out3d.n_focused) +
             " 3D聚焦=" + (out3d.region.confirmed ? "已确认" : "未确认"));

    if (n_passed == 0) {
        log.warn("vm44_relvec_match: 无通过候选");
        return -1;
    }

    // 2D (θ,s) 直方图峰值检测 (用于 s_peak)
    auto t_hist_start = Clock::now();
    double s_bw = (params.s_max - params.s_min) / (double)RELVEC_S_BINS;

    int peak_th_idx = 0, peak_s_idx = 0;
    int peak_cluster_val = 0;
    for (int th = 0; th < RELVEC_TH_BINS; ++th) {
        for (int si = 0; si < RELVEC_S_BINS; ++si) {
            int cluster_val = 0;
            int th_lo = std::max(0, th - 2), th_hi = std::min(RELVEC_TH_BINS - 1, th + 2);
            int s_lo = std::max(0, si - 2), s_hi = std::min(RELVEC_S_BINS - 1, si + 2);
            for (int t2 = th_lo; t2 <= th_hi; ++t2) {
                for (int s2 = s_lo; s2 <= s_hi; ++s2) {
                    cluster_val += hist2d[(size_t)t2 * RELVEC_S_BINS + s2];
                }
            }
            if (cluster_val > peak_cluster_val) {
                peak_cluster_val = cluster_val;
                peak_th_idx = th;
                peak_s_idx = si;
            }
        }
    }
    double s_peak = params.s_min + (peak_s_idx + 0.5) * s_bw;

    // 3D 峰值 (θ_peak, dx_peak, dy_peak) 从 out3d 获取
    double theta_peak_deg = (out3d.peak_th_idx + 0.5) * RELVEC_TH_BW_3D - 180.0;
    double dx_bw = (out3d.dx_range_hi - out3d.dx_range_lo) / (double)RELVEC_DXDY_BINS;
    double dy_bw = (out3d.dy_range_hi - out3d.dy_range_lo) / (double)RELVEC_DXDY_BINS;
    double dx_peak = out3d.dx_range_lo + (out3d.peak_dx_idx + 0.5) * dx_bw;
    double dy_peak = out3d.dy_range_lo + (out3d.peak_dy_idx + 0.5) * dy_bw;
    double snr_3d = out3d.snr_3d;

    double t_hist_ms = ms_double(Clock::now() - t_hist_start).count();

    output.theta_peak_deg = theta_peak_deg;
    output.s_peak = s_peak;
    output.dx_peak = dx_peak;
    output.dy_peak = dy_peak;
    output.theta_snr = snr_3d;
    output.success = (snr_3d >= 5.0);

    double t_total_ms = ms_double(Clock::now() - t_total_start).count();
    log.info("RelVec 直方图耗时: " + std::to_string(t_hist_ms) + " ms");
    log.info("RelVec 总耗时: " + std::to_string(t_total_ms) + " ms");
    log.info("vm44_relvec_match: n_passed=" + std::to_string(n_passed) +
             " n_focused=" + std::to_string(out3d.n_focused) +
             " θ=" + std::to_string(theta_peak_deg) + "°" +
             " s=" + std::to_string(s_peak) +
             " dx=" + std::to_string(dx_peak) +
             " dy=" + std::to_string(dy_peak) +
             " 3D_SNR=" + std::to_string(snr_3d) + "x" +
             " 2D_peak=" + std::to_string(peak_cluster_val) +
             " (success=" + (output.success ? "true" : "false") + ")" +
             " 聚焦=" + (out3d.region.confirmed ? "已确认" : "未确认"));

    // 导出 passed_pairs (θ, s, dx, dy) 到 CSV, 用于 3D 聚类可视化
    if (params.log_dir && params.log_dir[0] != '\0') {
        std::string csv_path = std::string(params.log_dir) + "/relvec_pairs_3d.csv";
        std::ofstream ofs(csv_path);
        if (ofs.is_open()) {
            ofs << "theta_deg,s_est,dx,dy,is_near_peak,is_focused\n";
            for (const auto& p : output.passed_pairs) {
                bool near_peak = (std::abs(p.theta_rot_deg - theta_peak_deg) < 3.0 &&
                                  std::abs(p.dx - dx_peak) < 30.0 &&
                                  std::abs(p.dy - dy_peak) < 30.0);
                bool focused = (out3d.region.confirmed &&
                                p.theta_rot_deg >= out3d.region.th_lo &&
                                p.theta_rot_deg <= out3d.region.th_hi &&
                                p.dx >= out3d.region.dx_lo &&
                                p.dx <= out3d.region.dx_hi &&
                                p.dy >= out3d.region.dy_lo &&
                                p.dy <= out3d.region.dy_hi);
                ofs << std::fixed << std::setprecision(6)
                    << p.theta_rot_deg << "," << p.s_est << ","
                    << p.dx << "," << p.dy << ","
                    << (near_peak ? 1 : 0) << ","
                    << (focused ? 1 : 0) << "\n";
            }
            ofs.close();
            log.info("导出 passed_pairs 3D 数据: " + csv_path +
                     " (" + std::to_string(output.passed_pairs.size()) + " 行)");
        }
    }

    return 0;
}

} // namespace v44

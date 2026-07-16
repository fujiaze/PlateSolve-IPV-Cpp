// vm45_relvec.cpp - V4.5 相对向量法 Phase A (θ 求解) 实现
//
// 严格按设计文档 v4_4_relvec_sampling_design.md 实现 Phase A:
//   - 绝对距离 k-vector 查询: d_gaia ∈ [d_img - σ_d, d_img + σ_d] (σ_d = 3.0", 绝对值范围)
//   - 第三星三角形全等验证 (距离容差 σ_d)
//   - 1D θ 直方图 (360 bins × 1°) + 高斯平滑 (σ=1 bin, 环形)
//   - 加权投票: votes[bin] += 1 + log2(1 + n_passed)
//   - 背景估计 (去峰值 ±5° 中位数) + SNR 判定 + 抛物线亚 bin 精化
//
// 与 V4.4 (namespace v44) 并存, 不冲突。
// V4.4 偏离设计 (比例距离 d_img/s_max + 3D (θ,dx,dy) 密度场 + 递归聚焦 + 单点法),
// V4.5 严格回归设计文档 (绝对距离 + 1D θ 直方图 + 抛物线亚 bin 精化)。
//
// 注: U, W 输入均为角秒坐标 (U 在 vm45_select 中已乘 s₀ 转为角秒),
//     故 d_img = |U[j]-U[i]| 直接为绝对角距, 不再乘 s₀。
//
// 仅实现 Phase A (θ 求解), 不含 Phase B (tx/ty) / IRM / WcsFitter。

#include "vm45_internal.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace v45 {

static constexpr double VM45_PI = 3.14159265358979323846;
static constexpr double VM45_RADTODEG = 180.0 / VM45_PI;
static constexpr int    TH_BINS = 360;   // 1D θ 直方图 bin 数 (360 × 1° → [-180, 180))

// 计时工具
using Clock = std::chrono::steady_clock;
using ms_double = std::chrono::duration<double, std::milli>;

// ============================================================================
// 内部辅助函数
// ============================================================================

// 把角度 wrap 到 [-180, 180)
static double wrap180(double x) {
    double r = std::fmod(x, 360.0);
    if (r >= 180.0)  r -= 360.0;
    if (r < -180.0)  r += 360.0;
    return r;
}

// 中位数 (会排序输入向量)
static double median_sorted(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ============================================================================
// RelativeVectorMatcher: 相对向量法核心类 (严格按设计文档)
// ============================================================================
class RelativeVectorMatcher {
public:
    RelativeVectorMatcher(
        const std::vector<StarPoint>& U,
        const std::vector<StarPoint>& W,
        double s0,
        const VM45SolveParams& params,
        Logger* logger)
        : U_(U), W_(W), s0_(s0), params_(params), logger_(logger)
    {
        votes_.assign(TH_BINS, 0.0);
        smoothed_.assign(TH_BINS, 0.0);
    }

    // 执行采样+投票
    void match(std::vector<RelVecPair>& passed_pairs) {
        passed_pairs.clear();
        precompute();
        if (gaia_pairs_.empty()) {
            log_warn("RelVec: gaia_pairs 为空 (N_w<2), 无法匹配");
            return;
        }
        std::mt19937 rng((unsigned)params_.seed + 100);
        sample_and_vote(passed_pairs, rng);
    }

    // 峰值检测 (在 match 后调用)
    void detect_peak(RelVecResult& result) {
        // 最终峰检测 (高斯平滑 + argmax + 背景中位数 + SNR)
        compute_peak_internal(peak_bin_, peak_val_, bg_median_, snr_);

        result.peak_bin  = peak_bin_;
        result.bg_median = bg_median_;

        double theta_bw       = params_.theta_bw;
        double snr_threshold  = params_.snr_threshold;

        if (snr_ > snr_threshold && peak_val_ > 0.0) {
            // bin 中心
            double theta_peak = (peak_bin_ + 0.5) * theta_bw - 180.0;

            // 抛物线亚 bin 精化: 用 peak_bin-1, peak_bin, peak_bin+1 三点拟合顶点偏移
            int prev_idx = (peak_bin_ - 1 + TH_BINS) % TH_BINS;
            int next_idx = (peak_bin_ + 1) % TH_BINS;
            double y_prev = smoothed_[prev_idx];
            double y_peak = smoothed_[peak_bin_];
            double y_next = smoothed_[next_idx];
            double denom  = y_prev - 2.0 * y_peak + y_next;
            double offset = 0.0;
            if (std::abs(denom) > 1e-12) {
                offset = 0.5 * (y_prev - y_next) / denom;
                if (offset < -0.5) offset = -0.5;
                if (offset >  0.5) offset =  0.5;
            }
            theta_peak += offset * theta_bw;

            result.theta_peak_deg = theta_peak;
            result.theta_snr      = snr_;
            result.success        = true;
        } else {
            result.theta_peak_deg = 0.0;
            result.theta_snr      = snr_;
            result.success        = false;
        }
    }

    // 暴露给外部用于填充 result
    const std::vector<double>& votes()    const { return votes_; }
    const std::vector<double>& smoothed()  const { return smoothed_; }
    int n_samples()     const { return n_samples_; }
    int n_total_cand()  const { return n_total_cand_; }
    int n_passed()      const { return n_passed_; }
    int peak_bin()      const { return peak_bin_; }

private:
    // ---- 输入 ----
    const std::vector<StarPoint>& U_;   // 图像侧星 (角秒)
    const std::vector<StarPoint>& W_;   // Gaia 侧星 (角秒)
    double s0_;
    const VM45SolveParams& params_;
    Logger* logger_;

    // ---- 预计算数据 ----
    std::vector<std::vector<double>> D_W_;                       // N_w × N_w 距离矩阵 (角秒)
    struct GaiaPair { double dist; int a; int b; };              // a < b
    std::vector<GaiaPair> gaia_pairs_;                            // 按距离升序排序 (k-vector)
    std::vector<std::vector<std::pair<double,int>>> D_W_sorted_;  // 每颗 a 的邻星距离排序表

    // ---- 投票直方图 ----
    std::vector<double> votes_;      // 360 bins (加权投票)
    std::vector<double> smoothed_;   // 高斯平滑后

    // ---- 统计 ----
    int n_samples_    = 0;
    int n_total_cand_ = 0;
    int n_passed_     = 0;

    // ---- 峰值结果 ----
    int    peak_bin_  = 0;
    double peak_val_  = 0.0;
    double bg_median_ = 0.0;
    double snr_       = 0.0;

    // ========================================================================
    // 内部方法
    // ========================================================================

    // 预计算 W 距离矩阵 + Gaia 星对 (按距离排序, k-vector) + 每颗星邻星排序表
    void precompute() {
        int N_w = (int)W_.size();

        // 1. D_W_ 距离矩阵
        D_W_.assign(N_w, std::vector<double>(N_w, 0.0));
        for (int a = 0; a < N_w; ++a) {
            for (int b = a + 1; b < N_w; ++b) {
                double dx = W_[a].x - W_[b].x;
                double dy = W_[a].y - W_[b].y;
                double d  = std::sqrt(dx * dx + dy * dy);
                D_W_[a][b] = d;
                D_W_[b][a] = d;
            }
        }

        // 2. gaia_pairs_: 所有 a<b 对 (按距离升序排序, k-vector 索引)
        gaia_pairs_.clear();
        gaia_pairs_.reserve((size_t)N_w * (N_w - 1) / 2);
        for (int a = 0; a < N_w; ++a) {
            for (int b = a + 1; b < N_w; ++b) {
                gaia_pairs_.push_back({D_W_[a][b], a, b});
            }
        }
        std::sort(gaia_pairs_.begin(), gaia_pairs_.end(),
                  [](const GaiaPair& p, const GaiaPair& q){ return p.dist < q.dist; });

        // 3. D_W_sorted_: 每颗 a 的邻星距离排序表 [(d(a,c), c), ...] 按 d 升序
        D_W_sorted_.assign(N_w, {});
        for (int a = 0; a < N_w; ++a) {
            D_W_sorted_[a].reserve(N_w - 1);
            for (int c = 0; c < N_w; ++c) {
                if (c == a) continue;
                D_W_sorted_[a].push_back({D_W_[a][c], c});
            }
            std::sort(D_W_sorted_[a].begin(), D_W_sorted_[a].end(),
                      [](const std::pair<double,int>& p, const std::pair<double,int>& q){
                          return p.first < q.first;
                      });
        }

        log_info("RelVec precompute: N_w=" + std::to_string(N_w) +
                 " gaia_pairs=" + std::to_string(gaia_pairs_.size()) +
                 " s0=" + std::to_string(s0_) + " arcsec/pixel");
    }

    // 采样 + 投票主循环 (K_total 次)
    void sample_and_vote(std::vector<RelVecPair>& passed_pairs, std::mt19937& rng) {
        int    N_u       = (int)U_.size();
        int    K_total   = params_.K_total;
        // sigma_d_px (像素) → sigma_d (角秒): 内部所有距离比较仍用角秒
        double sigma_d   = params_.sigma_d_px * s0_;   // 距离容差 (arcsec)
        int    n_third   = params_.n_third;        // 第三星验证颗数 (0=用全部)
        double ratio_min = params_.third_ratio_min; // 第三星通过比例阈值
        double theta_bw  = params_.theta_bw;
        int    max_cand  = params_.relvec_max_cand;

        log_info("RelVec 采样参数: sigma_d_px=" + std::to_string(params_.sigma_d_px) +
                 " s0=" + std::to_string(s0_) +
                 " sigma_d_asec=" + std::to_string(sigma_d) +
                 " n_third=" + std::to_string(n_third) +
                 " ratio_min=" + std::to_string(ratio_min));

        std::uniform_int_distribution<int> ud_u(0, N_u - 1);

        // 自适应停止参数 (可选, 沿用 V4.4)
        bool   adaptive_stop  = (params_.adaptive_stop != 0);
        int    min_samples     = std::max(params_.min_samples, 1);
        int    check_interval  = std::max(params_.check_interval, 1);
        double snr_eps         = params_.snr_eps;
        int    max_stable      = std::max(params_.max_stable, 1);
        double prev_snr        = 0.0;
        int    stable_count    = 0;

        auto t_start = Clock::now();

        for (int s = 0; s < K_total; ++s) {
            n_samples_ = s + 1;

            // ----------------------------------------------------------
            // 1. 随机采样图像星对 (i, j), i ≠ j
            // ----------------------------------------------------------
            int i = ud_u(rng);
            int j = ud_u(rng);
            if (i == j) continue;

            double dxi = U_[j].x - U_[i].x;
            double dyi = U_[j].y - U_[i].y;
            double d_img = std::sqrt(dxi * dxi + dyi * dyi);   // 角秒 (绝对距离)
            if (d_img < 1e-6) continue;
            double angle_img = std::atan2(dyi, dxi);           // 弧度

            // ----------------------------------------------------------
            // 2. k-vector 绝对距离查询: d_lo = d_img - σ_d, d_hi = d_img + σ_d
            //    (绝对值范围, NOT 比例距离)
            // ----------------------------------------------------------
            double d_lo = d_img - sigma_d;
            double d_hi = d_img + sigma_d;
            auto lo_it = std::lower_bound(gaia_pairs_.begin(), gaia_pairs_.end(), d_lo,
                [](const GaiaPair& p, double v){ return p.dist < v; });
            auto hi_it = std::upper_bound(gaia_pairs_.begin(), gaia_pairs_.end(), d_hi,
                [](double v, const GaiaPair& p){ return v < p.dist; });
            int idx_lo = (int)(lo_it - gaia_pairs_.begin());
            int idx_hi = (int)(hi_it - gaia_pairs_.begin());
            int n_cand = idx_hi - idx_lo;
            n_total_cand_ += n_cand;
            if (n_cand <= 0) continue;

            // 候选限流 (n_cand > max_cand 时随机抽样 max_cand 个)
            std::vector<int> cand_idx;
            if (n_cand > max_cand) {
                cand_idx.resize(n_cand);
                std::iota(cand_idx.begin(), cand_idx.end(), idx_lo);
                std::shuffle(cand_idx.begin(), cand_idx.end(), rng);
                cand_idx.resize(max_cand);
            } else {
                cand_idx.resize(n_cand);
                std::iota(cand_idx.begin(), cand_idx.end(), idx_lo);
            }

            // ----------------------------------------------------------
            // 3. 取第三星 k 列表 (图像侧, k ≠ i, k ≠ j)
            //    n_third=0 → 用全部可用第三星; 否则随机取 n_third 颗
            // ----------------------------------------------------------
            std::vector<int> k_list;
            k_list.reserve(N_u - 2);
            for (int k = 0; k < N_u; ++k) {
                if (k != i && k != j) k_list.push_back(k);
            }
            int n_k_avail = (int)k_list.size();
            int n_k_use;
            if (n_third <= 0) {
                n_k_use = n_k_avail;
            } else {
                n_k_use = std::min(n_third, n_k_avail);
                // 部分洗牌 (Fisher-Yates): 取前 n_k_use 个随机 k
                if (n_k_use < n_k_avail) {
                    for (int t = 0; t < n_k_use; ++t) {
                        std::uniform_int_distribution<int> ud_k(t, n_k_avail - 1);
                        int r = ud_k(rng);
                        std::swap(k_list[t], k_list[r]);
                    }
                }
            }

            // ----------------------------------------------------------
            // 4. 对每个候选对 (a, b) 做第三星三角形全等验证
            // ----------------------------------------------------------
            for (int ci : cand_idx) {
                const GaiaPair& gp = gaia_pairs_[ci];
                int a = gp.a;
                int b = gp.b;

                int n_passed_this = 0;
                for (int tk = 0; tk < n_k_use; ++tk) {
                    int k = k_list[tk];

                    // 图像侧第三星距离 (角秒)
                    double dx_ik = U_[k].x - U_[i].x;
                    double dy_ik = U_[k].y - U_[i].y;
                    double d_ik  = std::sqrt(dx_ik * dx_ik + dy_ik * dy_ik);
                    double dx_jk = U_[k].x - U_[j].x;
                    double dy_jk = U_[k].y - U_[j].y;
                    double d_jk  = std::sqrt(dx_jk * dx_jk + dy_jk * dy_jk);

                    // 在 D_W_sorted_[a] 中二分查找 c, 使 |D_W_[a][c] - d_ik| < σ_d
                    //   (c ≠ a 已由 D_W_sorted_ 构造保证; 需 c ≠ b)
                    const auto& da = D_W_sorted_[a];
                    auto c_lo = std::lower_bound(da.begin(), da.end(), d_ik - sigma_d,
                        [](const std::pair<double,int>& p, double v){ return p.first < v; });
                    auto c_hi = std::upper_bound(da.begin(), da.end(), d_ik + sigma_d,
                        [](double v, const std::pair<double,int>& p){ return v < p.first; });

                    bool k_passed = false;
                    for (auto it = c_lo; it != c_hi; ++it) {
                        int c = it->second;
                        if (c == b) continue;   // c ≠ b
                        // 验证 |D_W_[b][c] - d_jk| < σ_d (三角形全等)
                        if (std::abs(D_W_[b][c] - d_jk) < sigma_d) {
                            k_passed = true;
                            break;   // 此 k 已找到匹配, 不再尝试其他 c
                        }
                    }
                    if (k_passed) {
                        n_passed_this++;
                    }
                }

                // ----------------------------------------------------------
                // 5. 通过比例 ≥ ratio_min → 投票 + 记录候选对
                //   - 真匹配: 多颗第三星一致通过, ratio 通常 >0.5
                //   - 假匹配: 仅个别碰巧通过, ratio <0.05
                //   - 线性加权 ratio*n_k_use: 真匹配权重高, 假匹配权重接近 0
                // ----------------------------------------------------------
                double ratio = (n_k_use > 0) ? (double)n_passed_this / n_k_use : 0.0;
                if (ratio >= ratio_min) {
                    double angle_gaia = std::atan2(W_[b].y - W_[a].y, W_[b].x - W_[a].x);
                    double delta_theta = wrap180((angle_gaia - angle_img) * VM45_RADTODEG);

                    int bin = (int)std::round((delta_theta + 180.0) / theta_bw);
                    bin = ((bin % TH_BINS) + TH_BINS) % TH_BINS;

                    // 线性加权: 真匹配 (ratio≈1.0, n_k_use≈50) weight≈50
                    //          临界匹配 (ratio≈0.3, n_k_use≈50) weight≈15
                    //   真匹配权重是临界匹配的 3+ 倍, SNR 区分度强于对数加权
                    double weight = ratio * (double)n_k_use;
                    votes_[bin] += weight;

                    n_passed_++;
                    passed_pairs.push_back({i, j, a, b, delta_theta, n_passed_this});
                }
            }

            // 进度日志
            if ((s + 1) % 2000 == 0) {
                log_info("  RelVec 进度 " + std::to_string(s + 1) + "/" +
                         std::to_string(K_total) + ": 候选=" + std::to_string(n_total_cand_) +
                         " 通过=" + std::to_string(n_passed_));
            }

            // ----------------------------------------------------------
            // 自适应停止 (可选): SNR 收敛即停止
            // ----------------------------------------------------------
            if (adaptive_stop && (s + 1) >= min_samples && (s + 1) % check_interval == 0) {
                int pk_bin; double pk_val, pk_bg, pk_snr;
                compute_peak_internal(pk_bin, pk_val, pk_bg, pk_snr);
                if (pk_snr > params_.snr_threshold && prev_snr > 0.0) {
                    double rel = std::abs(pk_snr - prev_snr) / std::max(prev_snr, 1.0);
                    if (rel < snr_eps) {
                        stable_count++;
                        if (stable_count >= max_stable) {
                            log_info("RelVec SNR 收敛停止: s=" + std::to_string(s + 1) +
                                     "/" + std::to_string(K_total) + " SNR=" + std::to_string(pk_snr) +
                                     " (连续 " + std::to_string(stable_count) + " 次稳定)");
                            break;
                        }
                    } else {
                        stable_count = 0;
                    }
                }
                prev_snr = pk_snr;
            }
        }

        double t_ms = ms_double(Clock::now() - t_start).count();
        log_info("RelVec 采样完成: samples=" + std::to_string(n_samples_) +
                 " n_total_cand=" + std::to_string(n_total_cand_) +
                 " n_passed=" + std::to_string(n_passed_) +
                 " 耗时=" + std::to_string(t_ms) + " ms");
    }

    // 高斯平滑 (σ=1 bin, 环形) + 峰值检测 + 背景估计 + SNR
    void compute_peak_internal(int& peak_bin, double& peak_val,
                               double& bg_median, double& snr)
    {
        // 高斯平滑 (σ=1 bin, 环形): smoothed[i] = 0.3*votes[i-1] + 0.4*votes[i] + 0.3*votes[i+1]
        for (int i = 0; i < TH_BINS; ++i) {
            int prev = (i - 1 + TH_BINS) % TH_BINS;
            int next = (i + 1) % TH_BINS;
            smoothed_[i] = 0.3 * votes_[prev] + 0.4 * votes_[i] + 0.3 * votes_[next];
        }

        // peak_bin = argmax(smoothed)
        peak_bin = 0;
        peak_val = smoothed_[0];
        for (int i = 1; i < TH_BINS; ++i) {
            if (smoothed_[i] > peak_val) {
                peak_val = smoothed_[i];
                peak_bin = i;
            }
        }

        // 背景估计: 去掉 [peak_bin-5, peak_bin+5] (mod 360) 区域, 取剩余 bins 的中位数 (用 raw votes)
        std::vector<double> bg_vals;
        bg_vals.reserve(TH_BINS - 11);
        for (int i = 0; i < TH_BINS; ++i) {
            int diff       = std::abs(i - peak_bin);
            int diff_wrap  = std::min(diff, TH_BINS - diff);
            if (diff_wrap <= 5) continue;   // 去掉峰值 ±5°
            bg_vals.push_back(votes_[i]);
        }
        bg_median = median_sorted(bg_vals);

        // SNR = peak_val / max(bg_median, 1.0)
        snr = peak_val / std::max(bg_median, 1.0);
    }

    void log_info(const std::string& msg) { if (logger_) logger_->info(msg); }
    void log_warn(const std::string& msg) { if (logger_) logger_->warn(msg); }
};

// ============================================================================
// 调试 CSV 输出
// ============================================================================
static void write_csv_files(const std::string& log_dir,
                            const RelVecResult& result,
                            double theta_bw)
{
    // theta_histogram.csv: bin_center_deg, votes, smoothed_votes, is_peak_region
    {
        std::string path = log_dir + "/theta_histogram.csv";
        std::ofstream ofs(path);
        if (ofs.is_open()) {
            ofs << "bin_center_deg,votes,smoothed_votes,is_peak_region\n";
            for (int i = 0; i < TH_BINS; ++i) {
                double center = (i + 0.5) * theta_bw - 180.0;
                int diff       = std::abs(i - result.peak_bin);
                int diff_wrap  = std::min(diff, TH_BINS - diff);
                int is_peak    = (diff_wrap <= 5) ? 1 : 0;
                ofs << std::fixed << std::setprecision(6)
                    << center << ","
                    << result.votes[i] << ","
                    << result.smoothed_votes[i] << ","
                    << is_peak << "\n";
            }
            ofs.close();
        }
    }
    // relvec_passed_pairs.csv: img_i, img_j, gaia_a, gaia_b, theta_deg, n_third_passed
    {
        std::string path = log_dir + "/relvec_passed_pairs.csv";
        std::ofstream ofs(path);
        if (ofs.is_open()) {
            ofs << "img_i,img_j,gaia_a,gaia_b,theta_deg,n_third_passed\n";
            for (const auto& p : result.passed_pairs) {
                ofs << p.img_i << ","
                    << p.img_j << ","
                    << p.gaia_a << ","
                    << p.gaia_b << ","
                    << std::fixed << std::setprecision(6)
                    << p.theta_rot_deg << ","
                    << p.n_third_passed << "\n";
            }
            ofs.close();
        }
    }
}

// ============================================================================
// 外部接口 vm45_relvec_match
// ============================================================================
int vm45_relvec_match(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    const VM45SolveParams& params,
    RelVecResult& output,
    Logger* logger)
{
    Logger default_logger;
    Logger* log = logger ? logger : &default_logger;
    log->info("=== vm45_relvec_match 开始 (V4.5 严格按设计文档, 绝对距离+1D直方图) ===");

    auto t_total_start = Clock::now();

    // 初始化 output
    output.theta_peak_deg       = 0;
    output.theta_snr            = 0;
    output.peak_bin             = 0;
    output.bg_median            = 0;
    output.n_samples            = 0;
    output.n_total_candidates   = 0;
    output.n_passed             = 0;
    output.success              = false;
    output.votes.assign(TH_BINS, 0.0);
    output.smoothed_votes.assign(TH_BINS, 0.0);
    output.passed_pairs.clear();

    // 输入校验
    if ((int)U.size() < 3 || (int)W.size() < 3) {
        log->warn("vm45_relvec_match: U(" + std::to_string(U.size()) +
                  ") 或 W(" + std::to_string(W.size()) + ") 过少 (<3)");
        return -1;
    }

    log->info("参数: K_total=" + std::to_string(params.K_total) +
             " sigma_d_px=" + std::to_string(params.sigma_d_px) +
             " s0=" + std::to_string(s0) +
             " sigma_d_asec=" + std::to_string(params.sigma_d_px * s0) +
             " n_third=" + std::to_string(params.n_third) +
             " ratio_min=" + std::to_string(params.third_ratio_min) +
             " theta_bw=" + std::to_string(params.theta_bw) +
             " snr_threshold=" + std::to_string(params.snr_threshold) +
             " max_cand=" + std::to_string(params.relvec_max_cand) +
             " N_u=" + std::to_string(U.size()) +
             " N_w=" + std::to_string(W.size()) +
             " s0=" + std::to_string(s0));

    // 构造匹配器
    auto t_construct_start = Clock::now();
    RelativeVectorMatcher matcher(U, W, s0, params, log);
    double t_construct_ms = ms_double(Clock::now() - t_construct_start).count();
    log->info("RelVec 构造耗时: " + std::to_string(t_construct_ms) + " ms");

    // 采样 + 投票
    matcher.match(output.passed_pairs);

    // 峰值检测 (含亚 bin 精化)
    matcher.detect_peak(output);

    // 填充直方图 (360 元素)
    const std::vector<double>& v  = matcher.votes();
    const std::vector<double>& sv = matcher.smoothed();
    std::copy(v.begin(),  v.end(),  output.votes.begin());
    std::copy(sv.begin(), sv.end(), output.smoothed_votes.begin());

    output.n_samples          = matcher.n_samples();
    output.n_total_candidates = matcher.n_total_cand();
    output.n_passed           = matcher.n_passed();
    output.peak_bin           = matcher.peak_bin();

    double t_total_ms = ms_double(Clock::now() - t_total_start).count();

    // 日志: theta_peak, SNR, peak_bin, bg_median, n_passed, n_samples, 耗时
    log->info("vm45_relvec_match 结果: theta_peak=" + std::to_string(output.theta_peak_deg) +
             " deg, SNR=" + std::to_string(output.theta_snr) +
             ", peak_bin=" + std::to_string(output.peak_bin) +
             ", bg_median=" + std::to_string(output.bg_median) +
             ", n_passed=" + std::to_string(output.n_passed) +
             ", n_samples=" + std::to_string(output.n_samples) +
             ", 耗时=" + std::to_string(t_total_ms) + " ms" +
             " (success=" + (output.success ? "true" : "false") + ")");

    // 调试 CSV 输出
    if (params.log_dir && params.log_dir[0] != '\0') {
        write_csv_files(std::string(params.log_dir), output, params.theta_bw);
        log->info("已输出 CSV 调试文件到: " + std::string(params.log_dir));
    }

    return 0;
}

} // namespace v45

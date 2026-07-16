/**
 * vm33_core.cpp - V3.3 Record-and-Filter
 *
 * Phase A: U, Wf
 *   s=|u_i|/|w_j|, θ=atan2(u_i)-atan2(w_j)
 *   s[0.9,1.1], |tx|/|ty|<0.6FOV
 *   WfWt, U[k]Wt[j_k]
 *   s_ratio[k] = |U[k]| / |Wf[j_k]|
 *   s_ratio[0.9,1.1]
 *   θ =  / θ_SNR
 *
 * Phase B: θ + n_in_range
 *   11
 *   Umeyama SVD +
 *
 * Eigen3, nanoflann, C++17, OpenMP
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
#include <unordered_set>
#include <omp.h>

#include "../include/vm33_api.h"

#include "Eigen/Dense"
#include "nanoflann.hpp"

namespace vm33 {

static constexpr double PI = 3.14159265358979323846;
static constexpr double DEGTORAD = PI / 180.0;
static constexpr double RADTODEG = 180.0 / PI;

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

void apply_similarity(const double* W, int M, double s, double theta,
                      double tx, double ty, double* Wt)
{
    double ct = std::cos(theta), st = std::sin(theta);
    for (int i = 0; i < M; ++i) {
        double wx = W[i * 2], wy = W[i * 2 + 1];
        Wt[i * 2]     = s * (ct * wx - st * wy) + tx;
        Wt[i * 2 + 1] = s * (st * wx + ct * wy) + ty;
    }
}

void apply_flip(const double* W, int M, int mode, double* Wf)
{
    bool flip_x = (mode == 1 || mode == 3);
    bool flip_y = (mode == 2 || mode == 3);
    for (int i = 0; i < M; ++i) {
        Wf[i * 2]     = flip_x ? -W[i * 2]     : W[i * 2];
        Wf[i * 2 + 1] = flip_y ? -W[i * 2 + 1] : W[i * 2 + 1];
    }
}

static inline double angle_diff_deg(double a, double b)
{
    double d = std::fmod(std::fmod(a - b + 180.0, 360.0) + 360.0, 360.0) - 180.0;
    return std::abs(d);
}

static inline double wrap180(double deg)
{
    return std::fmod(std::fmod(deg + 180.0, 360.0) + 360.0, 360.0) - 180.0;
}

// ============================================================================
// s-in-range: WfWt, U[k]WtNN, NN<max_dist, s_ratio=|U[k]|/|Wf[j]|[0.9,1.1]
int count_s_in_range(const double* U, int N,
                      const double* Wt, int M,
                      const double* norm_U,
                      const double* norm_Wf,
                      double s_min, double s_max,
                      double max_dist)
{
    if (N == 0 || M == 0) return 0;

    PointCloud2D cloud; cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) cloud.pts[i] = {Wt[i*2], Wt[i*2+1]};
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    double max_dist_sq = max_dist * max_dist;
    int count = 0;
    for (int k = 0; k < N; ++k) {
        double query[2] = {U[k*2], U[k*2+1]};
        KDTreeIndexType idx; double dist_sq;
        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
        rs.init(&idx, &dist_sq);
        tree.findNeighbors(rs, query);
        if (dist_sq > max_dist_sq) continue;
        double s_ratio = norm_U[k] / norm_Wf[idx];
        if (s_ratio >= s_min && s_ratio <= s_max) count++;
    }
    return count;
}

// ============================================================================
// SVD1
// ============================================================================

struct InlierResult { int n_inliers; double rms; std::vector<int> inlier_mask; };

InlierResult count_inliers_1to1(const double* U, int N, const double* Wt, int M, double tau)
{
    InlierResult result; result.n_inliers = 0; result.rms = 0.0;
    result.inlier_mask.assign(N, 0);
    if (N == 0 || M == 0) return result;

    PointCloud2D cloud; cloud.pts.resize(M);
    for (int i = 0; i < M; ++i) cloud.pts[i] = {Wt[i*2], Wt[i*2+1]};
    KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

    struct Match { int u_idx; int w_idx; double dist; };
    std::vector<Match> candidates; candidates.reserve(N);
    double tau_sq = tau * tau;
    for (int i = 0; i < N; ++i) {
        double query[2] = {U[i*2], U[i*2+1]};
        KDTreeIndexType idx; double dist_sq;
        nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
        rs.init(&idx, &dist_sq);
        tree.findNeighbors(rs, query);
        if (dist_sq <= tau_sq) candidates.push_back({i, (int)idx, std::sqrt(dist_sq)});
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Match& a, const Match& b) { return a.dist < b.dist; });
    std::vector<int> w_used(M, 0); double sum_sq = 0.0;
    for (auto& c : candidates) {
        if (w_used[c.w_idx]) continue;
        w_used[c.w_idx] = 1; result.inlier_mask[c.u_idx] = 1;
        sum_sq += c.dist * c.dist; result.n_inliers++;
    }
    if (result.n_inliers > 0) result.rms = std::sqrt(sum_sq / result.n_inliers);
    return result;
}

double compute_normalized_score(int n_inliers, double rms, int N_img, int M, double tau)
{
    double denom = std::min((double)N_img, (double)M);
    if (denom <= 0 || tau <= 0) return 0.0;
    return ((double)n_inliers / denom) * (1.0 - rms / tau);
}

struct SimilarityTransform { double s, theta, tx, ty; bool valid; };

SimilarityTransform umeyama(const double* src, const double* dst, int n)
{
    SimilarityTransform result; result.valid = false; result.s = 1.0;
    result.theta = 0.0; result.tx = 0.0; result.ty = 0.0;
    if (n < 2) return result;

    using Matrix2d = Eigen::Matrix2d;
    using Vector2d = Eigen::Vector2d;

    Vector2d src_mean = Vector2d::Zero(), dst_mean = Vector2d::Zero();
    for (int i = 0; i < n; ++i) {
        src_mean += Vector2d(src[i*2], src[i*2+1]);
        dst_mean += Vector2d(dst[i*2], dst[i*2+1]);
    }
    src_mean /= n; dst_mean /= n;

    Eigen::MatrixXd src_c(2, n), dst_c(2, n);
    for (int i = 0; i < n; ++i) {
        src_c(0,i)=src[i*2]-src_mean(0); src_c(1,i)=src[i*2+1]-src_mean(1);
        dst_c(0,i)=dst[i*2]-dst_mean(0); dst_c(1,i)=dst[i*2+1]-dst_mean(1);
    }

    Matrix2d H = src_c * dst_c.transpose();
    Eigen::JacobiSVD<Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double d = (svd.matrixV().transpose() * svd.matrixU().transpose()).determinant();
    Vector2d S_vec = Vector2d::Ones(); S_vec(1) = d;
    Matrix2d R = svd.matrixV() * S_vec.asDiagonal() * svd.matrixU().transpose();

    double trace_WtW = src_c.colwise().squaredNorm().sum();
    double sigma_trace = (svd.singularValues().cwiseProduct(S_vec)).sum();
    if (trace_WtW < 1e-15) return result;

    double s = sigma_trace / trace_WtW;
    if (std::abs(s - 1.0) >= 0.1) return result;

    double theta = std::atan2(R(1,0), R(0,0));
    Vector2d t = dst_mean - s * R * src_mean;

    result.s = s; result.theta = theta; result.tx = t(0); result.ty = t(1);
    result.valid = true;
    return result;
}

struct RefineResult {
    double s, theta, tx, ty; int n_inliers; double rms;
    std::vector<int> inlier_mask; bool success;
};

RefineResult iterative_svd_refine(const double* U, int N, const double* Wf, int M,
                                   double s, double theta, double tx, double ty,
                                   double s0, int max_iter)
{
    RefineResult result;
    result.s = s; result.theta = theta; result.tx = tx; result.ty = ty;
    result.n_inliers = 0; result.rms = 1e30; result.success = false;
    if (N < 3 || M < 3) return result;

    std::vector<double> Wt(M * 2);
    std::vector<int> prev_mask(N, 0);

    double tau = 1.0 * s0;
    apply_similarity(Wf, M, s, theta, tx, ty, Wt.data());
    auto inl = count_inliers_1to1(U, N, Wt.data(), M, tau);

    double scale_factors[] = {2.0, 5.0, 10.0};
    for (int k = 0; k < 3 && inl.n_inliers < 3; ++k) {
        tau = scale_factors[k] * s0;
        inl = count_inliers_1to1(U, N, Wt.data(), M, tau);
    }
    if (inl.n_inliers < 3) {
        result.inlier_mask = std::move(inl.inlier_mask);
        result.n_inliers = inl.n_inliers; result.rms = inl.rms;
        return result;
    }
    prev_mask = inl.inlier_mask;

    for (int iter = 0; iter < max_iter; ++iter) {
        apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());

        PointCloud2D cloud; cloud.pts.resize(M);
        for (int i = 0; i < M; ++i) cloud.pts[i] = {Wt[i*2], Wt[i*2+1]};
        KDTree tree(2, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));

        struct Match { int u_idx; int w_idx; double dist; };
        std::vector<Match> candidates; candidates.reserve(N);
        double tau_sq = (1.0 * s0) * (1.0 * s0);
        for (int i = 0; i < N; ++i) {
            double query[2] = {U[i*2], U[i*2+1]};
            KDTreeIndexType idx; double dist_sq;
            nanoflann::KNNResultSet<double, KDTreeIndexType> rs(1);
            rs.init(&idx, &dist_sq);
            tree.findNeighbors(rs, query);
            if (dist_sq <= tau_sq) candidates.push_back({i, (int)idx, std::sqrt(dist_sq)});
        }
        std::sort(candidates.begin(), candidates.end(),
            [](const Match& a, const Match& b) { return a.dist < b.dist; });
        std::vector<int> w_used(M, 0);
        std::vector<double> src_pts, dst_pts;
        for (auto& c : candidates) {
            if (w_used[c.w_idx]) continue;
            w_used[c.w_idx] = 1;
            src_pts.push_back(Wf[c.w_idx*2]); src_pts.push_back(Wf[c.w_idx*2+1]);
            dst_pts.push_back(U[c.u_idx*2]); dst_pts.push_back(U[c.u_idx*2+1]);
        }
        int n_pairs = (int)src_pts.size() / 2;
        if (n_pairs < 3) break;

        auto sim = umeyama(src_pts.data(), dst_pts.data(), n_pairs);
        if (!sim.valid || std::abs(sim.s - 1.0) > 0.1) break;

        result.s = sim.s; result.theta = sim.theta;
        result.tx = sim.tx; result.ty = sim.ty;

        apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());
        inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);
        if (inl.n_inliers < 3) break;
        bool converged = (inl.inlier_mask == prev_mask);
        prev_mask = inl.inlier_mask;
        if (converged) break;
    }

    apply_similarity(Wf, M, result.s, result.theta, result.tx, result.ty, Wt.data());
    inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);
    result.n_inliers = inl.n_inliers; result.rms = inl.rms;
    result.inlier_mask = std::move(inl.inlier_mask);
    result.success = (result.n_inliers >= 3);
    return result;
}

struct ThetaSNRResult { int peak_idx; double peak_deg, snr; };

ThetaSNRResult compute_theta_snr(const double* theta_hist, int n_bins, double bin_width)
{
    ThetaSNRResult result; result.peak_idx = 0; result.peak_deg = 0.0; result.snr = 0.0;
    double peak_val = 0.0;
    for (int b = 0; b < n_bins; ++b) {
        if (theta_hist[b] > peak_val) { peak_val = theta_hist[b]; result.peak_idx = b; }
    }
    result.peak_deg = (result.peak_idx + 0.5) * bin_width - 180.0;
    double bg_sum = 0.0; int bg_count = 0;
    for (int b = 0; b < n_bins; ++b) {
        if (std::abs(b - result.peak_idx) > 5) { bg_sum += theta_hist[b]; bg_count++; }
    }
    double bg_mean = (bg_count > 10) ? bg_sum / bg_count : 1.0;
    result.snr = peak_val / std::max(bg_mean, 1e-10);
    return result;
}

struct PairRecord {
    int u_idx, w_idx;
    double theta_deg;
    int n_in_range_s;
};

struct V33Result {
    double s, theta, tx, ty; int n_inliers; double rms;
    double peak_snr; int n_samples;
    std::vector<int> inlier_mask; bool success;
    double theta_peak_deg; int best_n_range; double median_noise;
    int n_phaseb_pairs; int n_phaseb_corr; int n_phasea_records;
};

// ============================================================================
// Phase A+B:   → θ+n_in_range_s → 11 → SVD
// ============================================================================

V33Result record_and_filter(
    const double* U, int N, const double* Wf, int M,
    double s0,
    double s_min, double s_max,
    int K_total, int batch_size, int min_samples,
    int min_inliers, int seed, double fov_diag_asec)
{
    V33Result result;
    result.s = 1.0; result.theta = 0.0; result.tx = 0.0; result.ty = 0.0;
    result.n_inliers = 0; result.rms = 1e30; result.peak_snr = 0.0;
    result.n_samples = 0; result.success = false;
    result.theta_peak_deg = 0.0; result.best_n_range = 0;
    result.median_noise = 0.0; result.n_phaseb_pairs = 0;
    result.n_phaseb_corr = 0; result.n_phasea_records = 0;

    if (N < 2 || M < 2) return result;

    std::vector<double> norm_U(N), angle_U(N), norm_Wf(M), angle_Wf(M);
    std::vector<bool> valid_U(N, false), valid_Wf(M, false);
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

    static constexpr int THETA_BINS = 3600;
    static constexpr double THETA_BIN_WIDTH = 0.1;
    std::vector<double> theta_hist(THETA_BINS, 0.0);

    std::unordered_set<uint64_t> sampled_pairs;
    std::vector<PairRecord> records;
    records.reserve(K_total);

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> u_dist(0, N - 1);
    std::uniform_int_distribution<int> w_dist(0, M - 1);

    std::vector<double> Wt(M * 2);
    double max_translation = fov_diag_asec * 0.6;
    uint64_t total_possible = (uint64_t)N * M;
    int K_max = std::min(K_total, (int)std::min((uint64_t)K_total, total_possible));

    int n_valid = 0;
    int best_n_range = 0;
    int last_snr_check = 0;

    // ════════════════════════════════════════════════════════════════
    // Phase A: Record
    // ════════════════════════════════════════════════════════════════
    for (int iter = 0; iter < K_max; ++iter) {
        int i = u_dist(rng);
        int j = w_dist(rng);

        uint64_t key = (uint64_t)i * M + j;
        if (sampled_pairs.count(key)) continue;
        sampled_pairs.insert(key);

        if (!valid_U[i] || !valid_Wf[j]) continue;

        double s = norm_U[i] / norm_Wf[j];
        if (s < s_min || s > s_max) continue;

        double theta = angle_U[i] - angle_Wf[j];
        double theta_deg = wrap180(theta * RADTODEG);

        double ct = std::cos(theta), st = std::sin(theta);
        double tx = U[i*2]   - s * (ct * Wf[j*2] - st * Wf[j*2+1]);
        double ty = U[i*2+1] - s * (st * Wf[j*2] + ct * Wf[j*2+1]);
        if (std::abs(tx) > max_translation || std::abs(ty) > max_translation) continue;

        apply_similarity(Wf, M, s, theta, tx, ty, Wt.data());

        int n_range = count_s_in_range(U, N, Wt.data(), M,
                                        norm_U.data(), norm_Wf.data(),
                                        s_min, s_max,
                                        5.0 * s0);

        int theta_bin = (int)((theta_deg + 180.0) / THETA_BIN_WIDTH);
        if (theta_bin >= 0 && theta_bin < THETA_BINS)
            theta_hist[theta_bin] += n_range;

        records.push_back({i, j, theta_deg, n_range});
        if (n_range > best_n_range) best_n_range = n_range;
        n_valid++;

        if (n_valid >= min_samples && iter - last_snr_check >= batch_size) {
            last_snr_check = iter;
            auto snr_result = compute_theta_snr(theta_hist.data(), THETA_BINS, THETA_BIN_WIDTH);
            result.peak_snr = snr_result.snr;

            double peak_weight = theta_hist[snr_result.peak_idx];
            double threshold_5N = std::min(5.0 * N, 500.0);
            double threshold_10N = std::min(10.0 * N, 1000.0);

            fprintf(stderr, "[vm33] PhaseA: n_valid=%d best_n=%d peak_w=%.0f"
                    " θ_peak=%.2f θ_SNR=%.1fx (5N=%.0f 10N=%.0f)\n",
                    n_valid, best_n_range, peak_weight,
                    snr_result.peak_deg, snr_result.snr,
                    threshold_5N, threshold_10N);

            if (snr_result.snr >= threshold_10N) {
                fprintf(stderr, "[vm33] PhaseA: θ_SNR=%.1f≥10N\n", snr_result.snr);
                break;
            }
            if (snr_result.snr >= threshold_5N) {
                fprintf(stderr, "[vm33] PhaseA: θ_SNR=%.1f≥5N\n", snr_result.snr);
                break;
            }
            fprintf(stderr, "[vm33] PhaseA: θ_SNR=%.1f<5N\n", snr_result.snr);
        }
    }

    result.n_samples = n_valid;

    std::vector<int> all_nr; all_nr.reserve(records.size());
    for (auto& r : records) all_nr.push_back(r.n_in_range_s);
    std::sort(all_nr.begin(), all_nr.end());
    double median_noise = all_nr[all_nr.size() / 2];

    auto final_snr = compute_theta_snr(theta_hist.data(), THETA_BINS, THETA_BIN_WIDTH);
    result.peak_snr = final_snr.snr;
    result.theta_peak_deg = final_snr.peak_deg;
    result.best_n_range = best_n_range;
    result.median_noise = median_noise;
    result.n_phasea_records = (int)records.size();

    double final_5N = std::min(5.0 * N, 500.0);
    double final_10N = std::min(10.0 * N, 1000.0);
    fprintf(stderr, "[vm33] PhaseA done: n_valid=%d best_n=%d median=%.1f peak_w=%.0f"
            " θ_peak=%.2f θ_SNR=%.1fx (5N=%.0f) records=%zu\n",
            n_valid, best_n_range, median_noise,
            theta_hist[final_snr.peak_idx], final_snr.peak_deg,
            final_snr.snr, final_5N, records.size());

    // ════════════════════════════════════════════════════════════════
    // Phase B: Filter → 11 → SVD
    // ════════════════════════════════════════════════════════════════
    double theta_peak = final_snr.peak_deg;
    double noise_threshold = std::max(2.0, 1.5 * median_noise);
    double theta_band_filter = 2.0;

    fprintf(stderr, "[vm33] PhaseB: θ_peak=%.2f noise_thr=%.1f (median=%.1f) θ_band=%.0f\n",
            theta_peak, noise_threshold, median_noise, theta_band_filter);

    std::vector<PairRecord> filtered;
    for (auto& r : records) {
        if (r.n_in_range_s <= (int)noise_threshold) continue;
        if (angle_diff_deg(r.theta_deg, theta_peak) > theta_band_filter) continue;
        filtered.push_back(r);
    }
    fprintf(stderr, "[vm33] PhaseB: %zu pairs (from %zu)\n", filtered.size(), records.size());

    if (filtered.size() < 2) {
        theta_band_filter = 4.0;
        noise_threshold = std::max(1.0, 1.0 * median_noise);
        fprintf(stderr, "[vm33] PhaseB retry: θ_band=%.0f noise_thr=%.1f\n",
                theta_band_filter, noise_threshold);
        filtered.clear();
        for (auto& r : records) {
            if (r.n_in_range_s <= (int)noise_threshold) continue;
            if (angle_diff_deg(r.theta_deg, theta_peak) > theta_band_filter) continue;
            filtered.push_back(r);
        }
        fprintf(stderr, "[vm33] PhaseB retry: %zu pairs\n", filtered.size());
    }

    if (filtered.size() < 2) {
        fprintf(stderr, "[vm33] PhaseB: <2 pairs\n");
        return result;
    }

    std::sort(filtered.begin(), filtered.end(),
        [](const PairRecord& a, const PairRecord& b) {
            return a.n_in_range_s > b.n_in_range_s;
        });

    std::vector<int> u_used(N, 0), w_used(M, 0);
    std::vector<int> corr_u, corr_w;
    for (auto& r : filtered) {
        if (!u_used[r.u_idx] && !w_used[r.w_idx]) {
            corr_u.push_back(r.u_idx);
            corr_w.push_back(r.w_idx);
            u_used[r.u_idx] = 1;
            w_used[r.w_idx] = 1;
        }
    }
    fprintf(stderr, "[vm33] PhaseB: 11 → %zu corr\n", corr_u.size());

    result.n_phaseb_pairs = (int)filtered.size();
    result.n_phaseb_corr = (int)corr_u.size();

    if (corr_u.size() < 2) {
        fprintf(stderr, "[vm33] PhaseB: <2 corr\n");
        return result;
    }

    std::vector<double> src_pts(corr_u.size() * 2), dst_pts(corr_u.size() * 2);
    for (size_t k = 0; k < corr_u.size(); ++k) {
        src_pts[k*2]=Wf[corr_w[k]*2]; src_pts[k*2+1]=Wf[corr_w[k]*2+1];
        dst_pts[k*2]=U[corr_u[k]*2]; dst_pts[k*2+1]=U[corr_u[k]*2+1];
    }

    auto sim = umeyama(src_pts.data(), dst_pts.data(), (int)corr_u.size());
    if (!sim.valid) {
        fprintf(stderr, "[vm33] PhaseB: SVD invalid\n");
        return result;
    }

    apply_similarity(Wf, M, sim.s, sim.theta, sim.tx, sim.ty, Wt.data());
    auto inl = count_inliers_1to1(U, N, Wt.data(), M, 1.0 * s0);
    fprintf(stderr, "[vm33] PhaseB SVD: s=%.4f θ=%.2f n=%d rms=%.3f (from %zu corr)\n",
            sim.s, sim.theta * RADTODEG, inl.n_inliers, inl.rms, corr_u.size());

    auto refined = iterative_svd_refine(U, N, Wf, M,
        sim.s, sim.theta, sim.tx, sim.ty, s0, 10);

    if (refined.success) {
        result.s = refined.s; result.theta = refined.theta;
        result.tx = refined.tx; result.ty = refined.ty;
        result.n_inliers = refined.n_inliers; result.rms = refined.rms;
        result.inlier_mask = std::move(refined.inlier_mask);
    } else {
        result.s = sim.s; result.theta = sim.theta;
        result.tx = sim.tx; result.ty = sim.ty;
        result.n_inliers = inl.n_inliers; result.rms = inl.rms;
        result.inlier_mask = std::move(inl.inlier_mask);
    }
    result.success = true;

    fprintf(stderr, "[vm33] PhaseB OK: s=%.4f θ=%.2f n=%d rms=%.3f (corr=%zu filtered=%zu)\n",
            result.s, result.theta * RADTODEG, result.n_inliers, result.rms,
            corr_u.size(), filtered.size());
    return result;
}

struct ModeResult {
    double s, theta, tx, ty; int n_inliers; double rms; double norm_score;
    std::vector<int> inlier_mask; bool success;
    double peak_snr; int n_samples;
    double theta_peak_deg; int best_n_range; double median_noise;
    int n_phaseb_pairs; int n_phaseb_corr; int n_phasea_records;
};

ModeResult solve_single_mode(const double* U, int N, const double* W, int M,
                              int mode, double s0,
                              double s_min, double s_max,
                              int K_total, int batch_size,
                              int min_samples,
                              int min_inliers, int seed,
                              double fov_diag_asec,
                              volatile std::atomic<bool>* early_exit)
{
    ModeResult mr;
    mr.success = false; mr.norm_score = 0.0; mr.peak_snr = 0.0; mr.n_samples = 0;

    if (early_exit && early_exit->load(std::memory_order_relaxed)) {
        fprintf(stderr, "[vm33] mode%d: skip\n", mode);
        return mr;
    }

    std::vector<double> Wf(M * 2);
    apply_flip(W, M, mode, Wf.data());
    fprintf(stderr, "[vm33] mode%d: M=%d\n", mode, M);

    auto abr = record_and_filter(U, N, Wf.data(), M, s0,
        s_min, s_max, K_total, batch_size, min_samples,
        min_inliers, seed + mode, fov_diag_asec);

    mr.peak_snr = abr.peak_snr;
    mr.n_samples = abr.n_samples;
    mr.theta_peak_deg = abr.theta_peak_deg;
    mr.best_n_range = abr.best_n_range;
    mr.median_noise = abr.median_noise;
    mr.n_phaseb_pairs = abr.n_phaseb_pairs;
    mr.n_phaseb_corr = abr.n_phaseb_corr;
    mr.n_phasea_records = abr.n_phasea_records;

    if (!abr.success) {
        fprintf(stderr, "[vm33] mode%d: fail\n", mode);
        return mr;
    }

    mr.s = abr.s; mr.theta = abr.theta;
    mr.tx = abr.tx; mr.ty = abr.ty;
    mr.n_inliers = abr.n_inliers; mr.rms = abr.rms;
    mr.inlier_mask = std::move(abr.inlier_mask);
    mr.norm_score = compute_normalized_score(mr.n_inliers, mr.rms, N, M, 1.0 * s0);
    mr.success = true;

    if (mr.success && mr.s >= 0.9 && mr.s <= 1.1 && early_exit) {
        early_exit->store(true, std::memory_order_relaxed);
        fprintf(stderr, "[vm33] mode%d converged\n", mode);
    }

    fprintf(stderr, "[vm33] mode%d final: s=%.4f θ=%.2f n=%d rms=%.3f norm=%.4f\n",
            mode, mr.s, mr.theta * RADTODEG, mr.n_inliers, mr.rms, mr.norm_score);
    return mr;
}

} // namespace vm33

// ============================================================================
// vm33_solve: main entry
// ============================================================================

extern "C" VM33_API int vm33_solve(
    const double* U, int N_img,
    const double* W, int M,
    const VM33SolveParams* params,
    VM33SolveResult* result)
{
    using namespace vm33;

    result->s = 1.0; result->theta = 0.0; result->tx = 0.0; result->ty = 0.0;
    result->n_inliers = 0; result->rms = 1e30; result->best_mode = 0;
    result->norm_score = 0.0; result->peak_snr = 0.0; result->n_samples = 0;
    result->success = 0;
    std::memset(result->inlier_mask, 0, sizeof(int) * N_img);

    if (N_img < 2 || M < 2) {
        fprintf(stderr, "[vm33] vm33_solve: N=%d M=%d insufficient\n", N_img, M);
        return -1;
    }

    int n_modes = std::max(1, std::min(params->n_modes, 4));

    std::vector<ModeResult> mode_results(n_modes);
    std::atomic<bool> early_exit(false);

    #pragma omp parallel for schedule(static)
    for (int mode = 0; mode < n_modes; ++mode) {
        mode_results[mode] = solve_single_mode(
            U, N_img, W, M, mode, params->s0,
            params->s_min, params->s_max, params->K_total, params->batch_size,
            params->min_samples,
            params->min_inliers, params->seed, params->fov_diag_asec,
            &early_exit);
    }

    double best_score = -1.0; int best_mode = -1;
    for (int mode = 0; mode < n_modes; ++mode) {
        if (mode_results[mode].success && mode_results[mode].norm_score > best_score) {
            best_score = mode_results[mode].norm_score; best_mode = mode;
        }
    }

    if (best_mode < 0) {
        fprintf(stderr, "[vm33] vm33_solve: all modes fail (best=%.4f)\n", best_score);
        return -1;
    }

    auto& best = mode_results[best_mode];
    if (best.s < 0.9 || best.s > 1.1) {
        fprintf(stderr, "[vm33] vm33_solve: s=%.6f out of range\n", best.s);
        return -1;
    }

    result->s = best.s; result->theta = best.theta;
    result->tx = best.tx; result->ty = best.ty;
    result->n_inliers = best.n_inliers; result->rms = best.rms;
    result->best_mode = best_mode; result->norm_score = best_score;
    result->peak_snr = best.peak_snr; result->n_samples = best.n_samples;
    result->debug.theta_snr = best.peak_snr;
    result->debug.theta_peak_deg = best.theta_peak_deg;
    result->debug.best_n_range = best.best_n_range;
    result->debug.median_noise = best.median_noise;
    result->debug.n_phaseb_pairs = best.n_phaseb_pairs;
    result->debug.n_phaseb_corr = best.n_phaseb_corr;
    result->debug.n_phasea_records = best.n_phasea_records;
    result->success = 1;
    for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = best.inlier_mask[i];

    fprintf(stderr, "[vm33] vm33_solve: OK mode=%d n=%d rms=%.4f s=%.6f θ=%.4f "
            "norm=%.4f SNR=%.1fx samples=%d\n",
            best_mode, result->n_inliers, result->rms, result->s,
            result->theta * RADTODEG, result->norm_score,
            result->peak_snr, result->n_samples);
    return 0;
}

extern "C" VM33_API int vm33_svd_refine(
    const double* U, int N_img, const double* W, int M,
    const int* inlier_mask,
    double s_init, double theta_init, double tx_init, double ty_init,
    double s0, int max_iter, VM33SolveResult* result)
{
    using namespace vm33;
    result->s = s_init; result->theta = theta_init;
    result->tx = tx_init; result->ty = ty_init;
    result->n_inliers = 0; result->rms = 1e30; result->success = 0;
    if (N_img < 3 || M < 3) return -1;

    auto refined = iterative_svd_refine(U, N_img, W, M,
                                         s_init, theta_init, tx_init, ty_init, s0, max_iter);
    result->s = refined.s; result->theta = refined.theta;
    result->tx = refined.tx; result->ty = refined.ty;
    result->n_inliers = refined.n_inliers; result->rms = refined.rms;
    result->success = refined.success ? 1 : 0;
    if (result->inlier_mask) {
        for (int i = 0; i < N_img; ++i) result->inlier_mask[i] = refined.inlier_mask[i];
    }
    return 0;
}

extern "C" VM33_API int vm33_count_inliers(
    const double* U, int N_img, const double* W, int M,
    double s, double theta, double tx, double ty, double tau,
    int* inlier_mask, double* out_rms)
{
    using namespace vm33;
    std::vector<double> Wt(M * 2);
    apply_similarity(W, M, s, theta, tx, ty, Wt.data());
    auto inl = count_inliers_1to1(U, N_img, Wt.data(), M, tau);
    for (int i = 0; i < N_img; ++i) inlier_mask[i] = inl.inlier_mask[i];
    *out_rms = inl.rms;
    return inl.n_inliers;
}

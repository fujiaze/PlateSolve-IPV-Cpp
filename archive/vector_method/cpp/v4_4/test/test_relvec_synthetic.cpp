// test_relvec_synthetic.cpp - 相对向量法 3D 递归聚焦 合成数据实验
//
// 目标: 验证 3D (θ,dx,dy) 密度场 + 递归聚焦在不同噪声水平下的鲁棒性
//
// 实验维度:
//   1. 位置抖动 σ_pos (角秒): 0.0, 0.5, 1.0, 2.0, 3.0
//   2. 漏检率 drop_rate: 0%, 10%, 20%, 30%
//   3. 假阳性率 extra_rate: 0%, 10%, 20%, 50%
//
// 每个组合测试 N=50 颗 W, M=80 颗 U, 真实变换 (s=1, θ=30°, tx=500", ty=300")
// 输出 CSV: 噪声参数, SNR, θ误差, dx/dy误差, n_focused, n_passed, 耗时
//
// 编译:
//   g++ -O3 -std=c++17 -fopenmp -Iinclude -I../vector_match_v2/third_party \
//       -o test/test_relvec_synthetic.exe test/test_relvec_synthetic.cpp \
//       src/vm44_relvec.cpp -static-libgcc -static-libstdc++ -fopenmp

#include "vm44_internal.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace v44;

// 生成均匀分布的星点 (在 [-W/2, W/2] × [-H/2, H/2] 角秒范围)
static std::vector<StarPoint> generate_stars(int n, double width_asec, double height_asec,
                                              unsigned seed, bool is_gaia)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(-width_asec / 2, width_asec / 2);
    std::uniform_real_distribution<double> uy(-height_asec / 2, height_asec / 2);
    std::uniform_real_distribution<double> uflux(1000.0, 50000.0);

    std::vector<StarPoint> stars(n);
    for (int i = 0; i < n; ++i) {
        stars[i].x = ux(rng);
        stars[i].y = uy(rng);
        stars[i].flux = uflux(rng);
        stars[i].saturated = false;
    }
    return stars;
}

// 应用相似变换 U = s·R(θ)·W + t
static std::vector<StarPoint> apply_transform(const std::vector<StarPoint>& W,
                                               double s, double theta_deg,
                                               double tx, double ty)
{
    double th = theta_deg * M_PI / 180.0;
    double ct = cos(th), st = sin(th);
    std::vector<StarPoint> U(W.size());
    for (size_t i = 0; i < W.size(); ++i) {
        U[i].x = s * (ct * W[i].x - st * W[i].y) + tx;
        U[i].y = s * (st * W[i].x + ct * W[i].y) + ty;
        U[i].flux = W[i].flux;
        U[i].saturated = W[i].saturated;
    }
    return U;
}

// 添加高斯位置噪声
static void add_position_noise(std::vector<StarPoint>& stars, double sigma_asec, unsigned seed)
{
    if (sigma_asec <= 0) return;
    std::mt19937 rng(seed);
    std::normal_distribution<double> noise(0.0, sigma_asec);
    for (auto& s : stars) {
        s.x += noise(rng);
        s.y += noise(rng);
    }
}

// 随机漏检 (移除 drop_rate 比例的星)
static std::vector<StarPoint> drop_stars(const std::vector<StarPoint>& stars,
                                          double drop_rate, unsigned seed)
{
    if (drop_rate <= 0) return stars;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<StarPoint> result;
    result.reserve(stars.size());
    for (const auto& s : stars) {
        if (u01(rng) >= drop_rate) result.push_back(s);
    }
    return result;
}

// 添加假阳性星 (在图像范围内随机位置)
static std::vector<StarPoint> add_false_positives(const std::vector<StarPoint>& stars,
                                                   double extra_rate,
                                                   double width_asec, double height_asec,
                                                   unsigned seed)
{
    if (extra_rate <= 0) return stars;
    int n_extra = (int)(stars.size() * extra_rate);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ux(-width_asec / 2, width_asec / 2);
    std::uniform_real_distribution<double> uy(-height_asec / 2, height_asec / 2);
    std::uniform_real_distribution<double> uflux(500.0, 30000.0);

    // 偏离中心范围 (含 tx, ty 偏移)
    std::vector<StarPoint> result = stars;
    for (int i = 0; i < n_extra; ++i) {
        StarPoint s;
        s.x = ux(rng) + 500.0;  // 加上 tx 偏移
        s.y = uy(rng) + 300.0;  // 加上 ty 偏移
        s.flux = uflux(rng);
        s.saturated = false;
        result.push_back(s);
    }
    return result;
}

// 单次实验
struct ExperimentResult {
    double sigma_pos;
    double drop_rate;
    double extra_rate;
    int    n_u;
    int    n_w;
    bool   success;
    double theta_peak;
    double s_peak;
    double dx_peak;
    double dy_peak;
    double snr_3d;
    int    n_focused;
    int    n_passed;
    int    n_samples;
    double t_ms;
    double theta_err_deg;
    double dx_err;
    double dy_err;
};

static ExperimentResult run_experiment(double sigma_pos, double drop_rate, double extra_rate,
                                        unsigned seed, Logger& log)
{
    ExperimentResult r = {};
    r.sigma_pos = sigma_pos;
    r.drop_rate = drop_rate;
    r.extra_rate = extra_rate;

    // 真实变换参数
    const double s_true = 1.0;
    const double theta_true_deg = 30.0;
    const double tx_true = 500.0;
    const double ty_true = 300.0;
    // 注: 相对向量法定义 theta_rot = angle(Δw) - angle(Δu), 应为 -theta_true = -30°

    // 生成 W (Gaia, 80 颗, 在 ±2000" 范围)
    auto W = generate_stars(80, 4000.0, 4000.0, seed, true);

    // 生成 U = s·R(θ)·W + t (50 颗, 用前 50 颗 W)
    std::vector<StarPoint> W_subset(W.begin(), W.begin() + 50);
    auto U = apply_transform(W_subset, s_true, theta_true_deg, tx_true, ty_true);

    // 添加噪声到 U
    add_position_noise(U, sigma_pos, seed + 1);
    auto U_dropped = drop_stars(U, drop_rate, seed + 2);
    auto U_final = add_false_positives(U_dropped, extra_rate, 4000.0, 4000.0, seed + 3);

    r.n_u = (int)U_final.size();
    r.n_w = (int)W.size();

    // 运行 vm44_relvec_match
    VM44SolveParams params = {};
    params.s_min = 0.95;
    params.s_max = 1.05;
    params.relvec_n_samples = 2000;
    params.relvec_max_u = 100;
    params.relvec_max_cand = 500;
    params.relvec_n_third_stars = 10;
    params.relvec_third_star_tol = 1.5;
    params.relvec_min_len_frac = 0.05;
    params.relvec_max_len_frac = 0.8;
    params.relvec_adaptive_stop = 1;
    params.relvec_min_samples = 200;
    params.relvec_check_interval = 50;
    params.relvec_snr_eps = 0.05;
    params.relvec_max_stable = 3;
    params.seed = (int)seed;

    RelVecResult output;
    auto t0 = std::chrono::steady_clock::now();
    int ret = vm44_relvec_match(U_final, W, s_true, params, output, &log);
    auto t1 = std::chrono::steady_clock::now();
    r.t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    r.success = (ret == 0 && output.success);
    r.theta_peak = output.theta_peak_deg;
    r.s_peak = output.s_peak;
    r.dx_peak = output.dx_peak;
    r.dy_peak = output.dy_peak;
    r.snr_3d = output.theta_snr;
    r.n_focused = output.n_focused;
    r.n_passed = output.n_passed;
    r.n_samples = output.n_samples;

    // 误差 (注意: θ_rot = -θ_true, 因为 angle(Δw) - angle(Δu) = -θ)
    double theta_expected = -theta_true_deg;
    r.theta_err_deg = r.theta_peak - theta_expected;
    // 归一到 [-180, 180]
    while (r.theta_err_deg > 180) r.theta_err_deg -= 360;
    while (r.theta_err_deg < -180) r.theta_err_deg += 360;
    r.dx_err = r.dx_peak - tx_true;
    r.dy_err = r.dy_peak - ty_true;

    return r;
}

int main()
{
    Logger log;
    log.info("=== 相对向量法 3D 递归聚焦 合成数据实验 ===");

    // 噪声水平
    std::vector<double> sigmas = {0.0, 0.5, 1.0, 2.0, 3.0};
    std::vector<double> drops  = {0.0, 0.10, 0.20, 0.30};
    std::vector<double> extras = {0.0, 0.10, 0.20, 0.50};

    std::vector<ExperimentResult> results;

    // 实验 1: 单维度扫描 (固定其他两个为 0)
    log.info("--- 实验 1: 位置抖动 σ 扫描 (drop=0, extra=0) ---");
    for (double sigma : sigmas) {
        auto r = run_experiment(sigma, 0.0, 0.0, 42, log);
        results.push_back(r);
        printf("σ=%.1f\" drop=0%% extra=0%%: success=%d SNR=%.1f θ_err=%.2f° dx_err=%.1f dy_err=%.1f focused=%d passed=%d t=%.0fms\n",
               sigma, r.success, r.snr_3d, r.theta_err_deg, r.dx_err, r.dy_err,
               r.n_focused, r.n_passed, r.t_ms);
    }

    log.info("--- 实验 2: 漏检率扫描 (σ=1.0, extra=0) ---");
    for (double drop : drops) {
        auto r = run_experiment(1.0, drop, 0.0, 42, log);
        results.push_back(r);
        printf("σ=1.0\" drop=%.0f%% extra=0%%: success=%d SNR=%.1f θ_err=%.2f° dx_err=%.1f dy_err=%.1f focused=%d passed=%d t=%.0fms\n",
               drop * 100, r.success, r.snr_3d, r.theta_err_deg, r.dx_err, r.dy_err,
               r.n_focused, r.n_passed, r.t_ms);
    }

    log.info("--- 实验 3: 假阳性率扫描 (σ=1.0, drop=0) ---");
    for (double extra : extras) {
        auto r = run_experiment(1.0, 0.0, extra, 42, log);
        results.push_back(r);
        printf("σ=1.0\" drop=0%% extra=%.0f%%: success=%d SNR=%.1f θ_err=%.2f° dx_err=%.1f dy_err=%.1f focused=%d passed=%d t=%.0fms\n",
               extra * 100, r.success, r.snr_3d, r.theta_err_deg, r.dx_err, r.dy_err,
               r.n_focused, r.n_passed, r.t_ms);
    }

    log.info("--- 实验 4: 综合噪声 (σ=1.0, drop=10%, extra=10%) ---");
    for (int trial = 0; trial < 5; ++trial) {
        auto r = run_experiment(1.0, 0.10, 0.10, 100 + trial, log);
        results.push_back(r);
        printf("trial=%d σ=1.0\" drop=10%% extra=10%%: success=%d SNR=%.1f θ_err=%.2f° dx_err=%.1f dy_err=%.1f focused=%d passed=%d t=%.0fms\n",
               trial, r.success, r.snr_3d, r.theta_err_deg, r.dx_err, r.dy_err,
               r.n_focused, r.n_passed, r.t_ms);
    }

    log.info("--- 实验 5: 高噪声场景 (σ=2.0, drop=20%, extra=20%) ---");
    for (int trial = 0; trial < 5; ++trial) {
        auto r = run_experiment(2.0, 0.20, 0.20, 200 + trial, log);
        results.push_back(r);
        printf("trial=%d σ=2.0\" drop=20%% extra=20%%: success=%d SNR=%.1f θ_err=%.2f° dx_err=%.1f dy_err=%.1f focused=%d passed=%d t=%.0fms\n",
               trial, r.success, r.snr_3d, r.theta_err_deg, r.dx_err, r.dy_err,
               r.n_focused, r.n_passed, r.t_ms);
    }

    log.info("--- 实验 6: 极端噪声场景 (σ=3.0, drop=30%, extra=50%) ---");
    for (int trial = 0; trial < 5; ++trial) {
        auto r = run_experiment(3.0, 0.30, 0.50, 300 + trial, log);
        results.push_back(r);
        printf("trial=%d σ=3.0\" drop=30%% extra=50%%: success=%d SNR=%.1f θ_err=%.2f° dx_err=%.1f dy_err=%.1f focused=%d passed=%d t=%.0fms\n",
               trial, r.success, r.snr_3d, r.theta_err_deg, r.dx_err, r.dy_err,
               r.n_focused, r.n_passed, r.t_ms);
    }

    // 输出 CSV
    std::string csv_path = "synthetic_results.csv";
    std::ofstream ofs(csv_path);
    if (ofs.is_open()) {
        ofs << "sigma_pos,drop_rate,extra_rate,n_u,n_w,success,theta_peak,s_peak,dx_peak,dy_peak,"
            << "snr_3d,n_focused,n_passed,n_samples,t_ms,theta_err_deg,dx_err,dy_err\n";
        for (const auto& r : results) {
            ofs << std::fixed << std::setprecision(4)
                << r.sigma_pos << "," << r.drop_rate << "," << r.extra_rate << ","
                << r.n_u << "," << r.n_w << ","
                << (r.success ? 1 : 0) << ","
                << std::setprecision(2) << r.theta_peak << ","
                << std::setprecision(4) << r.s_peak << ","
                << std::setprecision(2) << r.dx_peak << "," << r.dy_peak << ","
                << r.snr_3d << ","
                << r.n_focused << "," << r.n_passed << "," << r.n_samples << ","
                << std::setprecision(1) << r.t_ms << ","
                << std::setprecision(3) << r.theta_err_deg << ","
                << std::setprecision(2) << r.dx_err << "," << r.dy_err << "\n";
        }
        ofs.close();
        printf("\nCSV 报告已保存: %s (%d 行)\n", csv_path.c_str(), (int)results.size());
    }

    // 统计摘要
    int n_total = (int)results.size();
    int n_success = 0;
    double sum_theta_err = 0, sum_dx_err = 0, sum_dy_err = 0;
    double max_theta_err = 0, max_dx_err = 0, max_dy_err = 0;
    int n_focused_total = 0;
    for (const auto& r : results) {
        if (r.success) {
            n_success++;
            sum_theta_err += std::abs(r.theta_err_deg);
            sum_dx_err += std::abs(r.dx_err);
            sum_dy_err += std::abs(r.dy_err);
            max_theta_err = std::max(max_theta_err, std::abs(r.theta_err_deg));
            max_dx_err = std::max(max_dx_err, std::abs(r.dx_err));
            max_dy_err = std::max(max_dy_err, std::abs(r.dy_err));
            n_focused_total += r.n_focused;
        }
    }

    printf("\n=== 实验摘要 ===\n");
    printf("总实验数: %d\n", n_total);
    printf("成功率: %d/%d (%.1f%%)\n", n_success, n_total, 100.0 * n_success / n_total);
    if (n_success > 0) {
        printf("平均|θ误差|: %.3f° (max %.3f°)\n", sum_theta_err / n_success, max_theta_err);
        printf("平均|dx误差|: %.2f\" (max %.2f\")\n", sum_dx_err / n_success, max_dx_err);
        printf("平均|dy误差|: %.2f\" (max %.2f\")\n", sum_dy_err / n_success, max_dy_err);
        printf("聚焦候选总数: %d\n", n_focused_total);
    }

    return 0;
}

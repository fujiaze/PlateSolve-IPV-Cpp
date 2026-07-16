// ============================================================================
// exp_data_source.cpp - V4.4 向量法抽样独立验证实验 - 数据源
// ============================================================================

#include "exp_data_source.h"
#include "vm44_api.h"
#include "vm44_internal.h"
#include "vm44_log.h"

#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
#include <sstream>
#include <cstring>

namespace exp44 {

static constexpr double EXP_PI = 3.14159265358979323846;

// ============================================================================
// 默认模拟数据参数
// ============================================================================
SyntheticParams getDefaultSyntheticParams() {
    SyntheticParams p;
    p.n_stars = 100;
    p.fov_diag_asec = 12600.0;   // ≈ 3.5° (Galaxy_Center FOV)
    p.s_true = 0.9823;           // 模拟实际偏差
    p.theta_true_deg = 30.0;
    p.tx_true = 50.0;
    p.ty_true = -30.0;
    p.noise_sigma = 0.5;         // 角秒
    p.outlier_ratio = 0.3;       // 30% 外点
    p.seed = 42;
    return p;
}

// ============================================================================
// 生成模拟数据
// ============================================================================
int generateSyntheticData(
    const SyntheticParams& params,
    ExpInput& output,
    const std::string& data_name)
{
    output = ExpInput{};
    output.source = DataSource::SYNTHETIC;
    output.data_name = data_name;
    output.has_ground_truth = true;

    // 真值
    output.ground_truth.s = params.s_true;
    output.ground_truth.theta = params.theta_true_deg * EXP_PI / 180.0;
    output.ground_truth.tx = params.tx_true;
    output.ground_truth.ty = params.ty_true;
    output.ground_truth.valid = true;

    // s0 (标称像素尺度, 用 1.0 因为模拟数据无实际像素)
    output.s0 = 1.0;

    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> ud_pos(-params.fov_diag_asec / 2.0,
                                                    params.fov_diag_asec / 2.0);
    std::normal_distribution<double> nd_noise(0.0, params.noise_sigma);
    std::uniform_real_distribution<double> ud01(0.0, 1.0);

    // 1. 生成 W (天空星, 在 FOV 范围内)
    int N = params.n_stars;
    output.W.resize(N);
    for (int k = 0; k < N; ++k) {
        output.W[k].x = ud_pos(rng);
        output.W[k].y = ud_pos(rng);
        output.W[k].flux = 1000.0 - k * 5.0;  // 递减 flux (模拟亮度排序)
        output.W[k].saturated = (k < 5);      // 前 5 颗饱和
    }

    // 2. 应用变换 U = s·R(θ)·W + (tx, ty) 得到 U_true
    double s = params.s_true;
    double th = params.theta_true_deg * EXP_PI / 180.0;
    double ct = std::cos(th), st = std::sin(th);
    double tx = params.tx_true, ty = params.ty_true;

    output.U.resize(N);
    for (int k = 0; k < N; ++k) {
        double wx = output.W[k].x, wy = output.W[k].y;
        output.U[k].x = s * (ct * wx - st * wy) + tx;
        output.U[k].y = s * (st * wx + ct * wy) + ty;
        output.U[k].flux = output.W[k].flux;
        output.U[k].saturated = output.W[k].saturated;
    }

    // 3. 加入位置噪声 (高斯)
    for (int k = 0; k < N; ++k) {
        output.U[k].x += nd_noise(rng);
        output.U[k].y += nd_noise(rng);
    }

    // 4. 加入外点 (outlier_ratio 比例的 U 星随机移到 FOV 内其他位置)
    int n_outliers = (int)(N * params.outlier_ratio);
    for (int k = 0; k < n_outliers; ++k) {
        int idx = (int)(ud01(rng) * N);
        if (idx >= N) idx = N - 1;
        output.U[idx].x = ud_pos(rng);
        output.U[idx].y = ud_pos(rng);
    }

    return 0;
}

// ============================================================================
// 获取真实数据 (调用 vm44_select)
// ============================================================================
int loadRealData(
    const RealDataParams& params,
    ExpInput& output,
    const std::string& data_name)
{
    output = ExpInput{};
    output.source = DataSource::REAL;
    output.data_name = data_name.empty() ? params.image_path : data_name;
    output.has_ground_truth = false;
    output.ground_truth.valid = false;

    // 调用 vm44_select 获取 U/W
    // 手动初始化 VM44SolveParams (避免依赖 vm44_entry.cpp 的 vm44_get_default_params)
    v44::VM44SolveParams vm_params;
    std::memset(&vm_params, 0, sizeof(vm_params));
    vm_params.n_modes = 4;
    vm_params.seed = 42;
    vm_params.img_n_target = 50;
    vm_params.gaia_density_ratio = 1.5;
    vm_params.gaia_query_radius_factor = 0.55;
    vm_params.m_lim_step = 0.5;
    vm_params.m_lim_max_iter = 10;
    vm_params.density_tolerance = 0.1;
    vm_params.s_min = 0.9;
    vm_params.s_max = 1.1;
    vm_params.K_total = 10000;
    vm_params.batch_size = 500;
    vm_params.min_samples = 50;
    vm_params.K_top = 100;
    vm_params.min_inliers = 5;
    vm_params.w_snr = 0.4;
    vm_params.w_sparse = 0.4;
    vm_params.w_sat = 0.2;
    vm_params.prosac_T_max = 10000;
    vm_params.use_prosac = 1;
    vm_params.region_size_px = 800;
    vm_params.N_floor = 5;
    vm_params.N_cap = 30;
    vm_params.N_max = 1500;
    vm_params.mad_iters = 3;
    vm_params.mad_threshold_factor = 3.0;
    vm_params.mad_min_threshold_arcsec = 5.0;
    vm_params.lnK_accept = 10.0;
    vm_params.lnK_weak = 3.0;
    vm_params.eps_A = 0.1;
    vm_params.eps_J = 0.1;
    vm_params.triangle_pass_rate = 0.7;
    vm_params.sip_max_order = 4;
    vm_params.skip_sip = 0;
    vm_params.irm_max_iter = 10;
    vm_params.irm_converge_eps = 0.05;
    vm_params.irm_diverge_factor = 1.1;
    vm_params.irm_tau_min = 2.0;
    vm_params.irm_tau_factor = 3.0;
    vm_params.irm_lowe_ratio = 0.7;
    vm_params.irm_k_geometry = 8;
    vm_params.irm_geom_threshold = 4;
    vm_params.irm_geom_dist_tol = 3.0;
    vm_params.irm_ransac_max_iter = 200;
    vm_params.irm_ransac_min_inliers = 10;
    vm_params.irm_huber_delta_factor = 1.345;
    vm_params.irm_sip_min_pairs = 30;
    vm_params.irm_s_initial = 0;
    vm_params.relvec_n_samples = 5000;
    vm_params.relvec_max_u = 100;
    vm_params.relvec_third_star_tol = 1.5;
    vm_params.relvec_max_cand = 500;
    vm_params.relvec_min_len_frac = 0.05;
    vm_params.relvec_max_len_frac = 0.8;
    vm_params.relvec_n_third_stars = 10;
    vm_params.relvec_adaptive_stop = 1;
    vm_params.relvec_min_samples = 200;
    vm_params.relvec_check_interval = 100;
    vm_params.relvec_snr_eps = 0.05;
    vm_params.relvec_max_stable = 3;

    v44::StarSelection sel;
    v44::Logger logger;  // 默认 logger (输出到 stderr)

    // 计算中心 RA/DEC
    double ra = params.ra, dec = params.dec;

    // 调用 vm44_select (内部会读取图像 + 查询 Gaia)
    int rc = v44::vm44_select(
        params.image_path,
        ra, dec,
        params.focal_length_mm,
        params.pixel_size_um,
        vm_params,
        sel,
        &logger
    );

    if (rc != 0 || !sel.success) {
        return -1;
    }

    // 转换 v44::StarPoint → exp44::StarPoint
    output.U.resize(sel.U.size());
    for (size_t k = 0; k < sel.U.size(); ++k) {
        output.U[k].x = sel.U[k].x;
        output.U[k].y = sel.U[k].y;
        output.U[k].flux = sel.U[k].flux;
        output.U[k].saturated = sel.U[k].saturated;
    }
    output.W.resize(sel.W.size());
    for (size_t k = 0; k < sel.W.size(); ++k) {
        output.W[k].x = sel.W[k].x;
        output.W[k].y = sel.W[k].y;
        output.W[k].flux = sel.W[k].flux;
        output.W[k].saturated = sel.W[k].saturated;
    }

    // s0 = 206.264806247 × pixel_size_um / focal_length_mm
    output.s0 = 206.264806247 * params.pixel_size_um / params.focal_length_mm;

    return 0;
}

} // namespace exp44

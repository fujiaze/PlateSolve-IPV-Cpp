#ifndef VM4_DENSITY_H
#define VM4_DENSITY_H

// ============================================================================
// vm4_density.h - V4.0 Phase 0 密度匹配迭代星等查询模块（Task 2）
//
// 在 V3.5 抽样投票法基础上新增密度匹配查询，替代 V3.5 的 bisection_mag_limit。
// 通过图像亮星密度 ρ_img 反推目标星表密度 ρ_target = k_match × ρ_img，
// 迭代调整 Gaia 查询极限星等 m_lim，使 N_gaia 收敛到 [N_target×(1-tol), N_target×(1+tol)]。
//
// 模块构成：
//   1) compute_fov_and_density   - 从 FITS 头参数计算 FOV 与密度
//   2) compute_initial_mag_cut   - V3.5 m_cut 公式给出初始极限星等
//   3) density_match_query       - 密度匹配迭代查询（gaia_query_func 回调解耦客户端）
//   4) gnomonic_project_fov      - 标准 gnomonic 投影筛选 FOV 内星
//
// 不修改 vm4_api.h（字段已由 Task 1 定义）；不修改 vm4_core.cpp（Task 7 集成时再调用）。
// ============================================================================

#include <vector>
#include <utility>
#include <functional>

namespace vm4 {

// FOV 和密度计算结果
struct FovDensityInfo {
    double s0;              // 像素尺度(角秒/像素)
    double fov_diag_deg;    // FOV 对角线(度)
    double query_radius_deg;// 查询半径(度) = fov_diag_deg/2 * query_radius_factor
    double query_area_sqdeg;// 查询面积(平方度) = π × query_radius²
    double rho_img;         // 图像亮星密度(颗/平方度)
    double rho_target;      // 目标星表密度 = k_match × rho_img
    int    n_target;        // 目标星数 = round(k_match × n_img_bright)
};

// 密度匹配迭代查询结果
struct DensityMatchResult {
    double final_mag_lim;   // 最终极限星等
    int    final_n_gaia;    // 最终 Gaia 星数
    int    iterations;      // 实际迭代次数
    bool   converged;       // 是否收敛
};

// 从 FITS 头参数计算 FOV 和密度
// 输入：focal_length_mm, pixel_size_um, img_width, img_height, n_img_bright, k_match, query_radius_factor
// 输出：FovDensityInfo
// 公式：
//   s0            = 206.265 × pixel_size_um / focal_length_mm
//   fov_diag_deg  = sqrt(W² + H²) × s0 / 3600
//   query_radius  = fov_diag_deg / 2 × query_radius_factor
//   query_area    = π × query_radius²
//   rho_img       = n_img_bright / query_area
//   rho_target    = k_match × rho_img
//   n_target      = round(k_match × n_img_bright)
FovDensityInfo compute_fov_and_density(
    double focal_length_mm, double pixel_size_um,
    double img_width, double img_height,
    int n_img_bright, double k_match, double query_radius_factor);

// 初始极限星等估计（V3.5 的 m_cut 公式）
// m_cut = 6 + 1.5×log10(f_mm) + 2×log10(t_s)
double compute_initial_mag_cut(double focal_length_mm, double exposure_time_s);

// 密度匹配迭代星等查询
// 输入：center_ra, center_dec, query_radius, n_target, m_cut_initial, step, max_iter, tolerance
//       gaia_query_func: 回调函数，输入(ra, dec, radius, mag_lim)返回星数
// 输出：final_mag_lim, final_n_gaia, iterations, converged
// 算法：
//   m = m_cut_initial
//   for i in 0..max_iter:
//     n = gaia_query_func(ra, dec, radius, m)
//     if n < n_target×(1-tolerance): m += step  (放宽星等)
//     elif n > n_target×(1+tolerance): m -= step (收紧星等)
//     else: break (收敛)
//   return m, n, i
DensityMatchResult density_match_query(
    double center_ra, double center_dec, double query_radius_deg,
    int n_target, double m_cut_initial,
    double step, int max_iter, double tolerance,
    std::function<int(double,double,double,double)> gaia_query_func);

// gnomonic 投影筛选 FOV 内星（投影到以 center 为原点的切平面）
// 输入：gaia_stars (ra,dec 数组, 单位:度), center_ra, center_dec, fov_diag_deg
// 输出：投影后的 (xi, eta) 角秒坐标数组，仅保留距中心 < fov_diag_deg/2 的星
// 公式：标准 gnomonic 投影（参考 lib/plate_solve/python/vector_match_v2.py::gnomonic_forward）
std::vector<std::pair<double,double>> gnomonic_project_fov(
    const std::vector<std::pair<double,double>>& gaia_stars,
    double center_ra, double center_dec, double fov_diag_deg);

} // namespace vm4

#endif // VM4_DENSITY_H

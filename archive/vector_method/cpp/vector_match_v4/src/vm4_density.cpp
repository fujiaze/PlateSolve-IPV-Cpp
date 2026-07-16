/**
 * vm4_density.cpp - V4.0 Phase 0 密度匹配迭代星等查询模块实现（Task 2）
 *
 * 实现细节：
 *   - compute_fov_and_density   基于 FITS 头参数推导像素尺度、FOV、查询面积、密度
 *   - compute_initial_mag_cut   复用 V3.5 m_cut 经验公式给出迭代起点
 *   - density_match_query       通过 std::function 回调解耦具体 Gaia 客户端，
 *                               按固定步长调整 m_lim，使 N_gaia 落入 [N_t×(1-tol), N_t×(1+tol)]
 *   - gnomonic_project_fov      标准 gnomonic (TAN) 投影，仅保留距中心 < FOV_diag/2 的星
 *
 * 编译：C++17，单线程（密度匹配无需并行）。
 * 日志：通过 stderr 输出关键步骤，Task 8 接入 vm4_log 后可替换为日志文件。
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>

#include "../include/vm4_density.h"

namespace vm4 {

// 物理常量
static constexpr double PI = 3.14159265358979323846;
static constexpr double DEGTORAD = PI / 180.0;
static constexpr double RADTODEG = 180.0 / PI;
static constexpr double RADTOASEC = RADTODEG * 3600.0;
static constexpr double ASECTORAD = 1.0 / RADTOASEC;
// 206.265 = (180×3600)/π，把 um/mm 直接转为 角秒/像素
static constexpr double ARCSEC_PER_UM_PER_MM = 206.265;

// ----------------------------------------------------------------------------
// compute_fov_and_density
// ----------------------------------------------------------------------------
FovDensityInfo compute_fov_and_density(
    double focal_length_mm, double pixel_size_um,
    double img_width, double img_height,
    int n_img_bright, double k_match, double query_radius_factor)
{
    FovDensityInfo info{};

    // 防御：焦距必须为正
    if (focal_length_mm <= 0.0) {
        std::fprintf(stderr, "[vm4_density] 警告: focal_length_mm=%.4f 非法, 返回零值\n",
                     focal_length_mm);
        return info;
    }

    // 像素尺度（角秒/像素）
    info.s0 = ARCSEC_PER_UM_PER_MM * pixel_size_um / focal_length_mm;

    // FOV 对角线（度）
    double diag_pix = std::sqrt(img_width * img_width + img_height * img_height);
    double fov_diag_asec = diag_pix * info.s0;
    info.fov_diag_deg = fov_diag_asec / 3600.0;

    // 查询半径（度）与查询面积（平方度）
    info.query_radius_deg = info.fov_diag_deg * 0.5 * query_radius_factor;
    info.query_area_sqdeg = PI * info.query_radius_deg * info.query_radius_deg;

    // 图像亮星密度与目标密度
    if (info.query_area_sqdeg > 0.0) {
        info.rho_img = static_cast<double>(n_img_bright) / info.query_area_sqdeg;
    }
    info.rho_target = k_match * info.rho_img;

    // 目标星数（四舍五入）
    double n_target_dbl = k_match * static_cast<double>(n_img_bright);
    info.n_target = static_cast<int>(std::lround(n_target_dbl));

    std::fprintf(stderr,
        "[vm4_density] FOV计算: s0=%.4f\"/px, FOV_diag=%.4f°, query_r=%.4f°, "
        "area=%.5f deg², rho_img=%.2f, rho_target=%.2f, n_target=%d\n",
        info.s0, info.fov_diag_deg, info.query_radius_deg,
        info.query_area_sqdeg, info.rho_img, info.rho_target, info.n_target);

    return info;
}

// ----------------------------------------------------------------------------
// compute_initial_mag_cut
// ----------------------------------------------------------------------------
double compute_initial_mag_cut(double focal_length_mm, double exposure_time_s)
{
    // m_cut = 6 + 1.5×log10(f_mm) + 2×log10(t_s)
    double m_cut = 6.0
                 + 1.5 * std::log10(focal_length_mm)
                 + 2.0 * std::log10(exposure_time_s);

    std::fprintf(stderr,
        "[vm4_density] 初始星等: m_cut=%.4f (f=%.2fmm, t=%.2fs)\n",
        m_cut, focal_length_mm, exposure_time_s);
    return m_cut;
}

// ----------------------------------------------------------------------------
// density_match_query
// ----------------------------------------------------------------------------
DensityMatchResult density_match_query(
    double center_ra, double center_dec, double query_radius_deg,
    int n_target, double m_cut_initial,
    double step, int max_iter, double tolerance,
    std::function<int(double,double,double,double)> gaia_query_func)
{
    DensityMatchResult result{};
    result.final_mag_lim = m_cut_initial;
    result.final_n_gaia  = 0;
    result.iterations    = 0;
    result.converged     = false;

    if (!gaia_query_func) {
        std::fprintf(stderr, "[vm4_density] 错误: gaia_query_func 为空\n");
        return result;
    }
    if (n_target <= 0) {
        std::fprintf(stderr, "[vm4_density] 警告: n_target=%d 非正, 跳过迭代\n", n_target);
        return result;
    }

    // 容差上下界（按 spec: ±tolerance × N_target）
    double n_lo = n_target * (1.0 - tolerance);
    double n_hi = n_target * (1.0 + tolerance);

    double m = m_cut_initial;
    int    n = 0;
    int    i = 0;
    bool   converged = false;

    std::fprintf(stderr,
        "[vm4_density] 迭代开始: ra=%.4f, dec=%.4f, r=%.4f°, N_target=%d, "
        "tol=%.2f, 范围=[%.1f, %.1f], m0=%.3f, step=%.3f, max_iter=%d\n",
        center_ra, center_dec, query_radius_deg, n_target,
        tolerance, n_lo, n_hi, m_cut_initial, step, max_iter);

    for (i = 0; i < max_iter; ++i) {
        n = gaia_query_func(center_ra, center_dec, query_radius_deg, m);
        std::fprintf(stderr,
            "[vm4_density] iter=%d  m_lim=%.3f  n_gaia=%d  (target=%d, 范围=[%.1f,%.1f])\n",
            i, m, n, n_target, n_lo, n_hi);

        if (n < n_lo) {
            // 星数不足 → 放宽星等（提升 m_lim 上限）
            m += step;
        } else if (n > n_hi) {
            // 星数过多 → 收紧星等（降低 m_lim 上限）
            m -= step;
        } else {
            // 收敛
            converged = true;
            break;
        }
    }

    result.final_mag_lim = m;
    result.final_n_gaia  = n;
    result.iterations    = i;
    result.converged     = converged;

    std::fprintf(stderr,
        "[vm4_density] 迭代结束: %s  m_final=%.3f  n_final=%d  iters=%d\n",
        converged ? "收敛" : "未收敛(达到max_iter)",
        result.final_mag_lim, result.final_n_gaia, result.iterations);

    return result;
}

// ----------------------------------------------------------------------------
// gnomonic_project_fov
// ----------------------------------------------------------------------------
std::vector<std::pair<double,double>> gnomonic_project_fov(
    const std::vector<std::pair<double,double>>& gaia_stars,
    double center_ra, double center_dec, double fov_diag_deg)
{
    std::vector<std::pair<double,double>> out;
    if (gaia_stars.empty()) return out;

    // 切点坐标（弧度）
    double ra0  = center_ra  * DEGTORAD;
    double dec0 = center_dec * DEGTORAD;
    double sin_dec0 = std::sin(dec0);
    double cos_dec0 = std::cos(dec0);

    // FOV 半径（角秒）
    double radius_asec = (fov_diag_deg * 0.5) * 3600.0;
    double radius_asec_sq = radius_asec * radius_asec;

    out.reserve(gaia_stars.size());
    int n_in_fov = 0;

    for (const auto& star : gaia_stars) {
        double ra_deg  = star.first;
        double dec_deg = star.second;

        double ra  = ra_deg  * DEGTORAD;
        double dec = dec_deg * DEGTORAD;
        double delta_ra = ra - ra0;

        double sin_dec = std::sin(dec);
        double cos_dec = std::cos(dec);
        double cos_delta_ra = std::cos(delta_ra);

        // cos(c) = sin(dec0)sin(dec) + cos(dec0)cos(dec)cos(Δra)
        double cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_delta_ra;

        // 投影有效性：cosc > 1e-10 表示星位于切平面正面
        if (cosc <= 1e-10) continue;

        // 标准 gnomonic 投影
        double xi_rad  = cos_dec * std::sin(delta_ra) / cosc;
        double eta_rad = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_delta_ra) / cosc;

        // 转为角秒
        double xi_asec  = xi_rad  * RADTOASEC;
        double eta_asec = eta_rad * RADTOASEC;

        // 仅保留距中心 < FOV_diag/2 的星（圆形 FOV 过滤）
        double r_sq = xi_asec * xi_asec + eta_asec * eta_asec;
        if (r_sq > radius_asec_sq) continue;

        out.emplace_back(xi_asec, eta_asec);
        ++n_in_fov;
    }

    std::fprintf(stderr,
        "[vm4_density] 投影: 输入=%zu 颗, FOV内=%d 颗 (center=(%.4f,%.4f), r=%.2f\")\n",
        gaia_stars.size(), n_in_fov, center_ra, center_dec, radius_asec);

    return out;
}

} // namespace vm4

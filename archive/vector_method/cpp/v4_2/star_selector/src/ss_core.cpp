/**
 * ss_core.cpp - V4.2 StarSelector 模块核心实现（Task 2）
 *
 * 从 V4.1 的 vm4_density.cpp 抽取 compute_fov_and_density_asym 与
 * density_match_query_asym 逻辑，封装为独立 DLL，使用 v42::Logger 记录日志。
 *
 * 实现:
 *   1. compute_fov_and_density_asym: 计算像素尺度、FOV对角线、查询半径、
 *      图像面积、目标星数
 *   2. compute_initial_mag_cut: V3.5 m_cut 经验公式
 *   3. density_match_query_asym: 自适应步长迭代极限星等
 *      (前4次 step_init, 后续 step_init/2)
 *
 * 编译: C++17 单线程（密度匹配无需并行）
 * 日志: 通过 v42::Logger 输出 UTF-8 BOM 日志文件 + stderr
 */

#include <cmath>
#include <cstdio>
#include <string>

#include "ss_api.h"
#include "v42_log.h"

// 物理常量
static constexpr double PI = 3.14159265358979323846;
// 206.265 = (180×3600)/π，把 um/mm 直接转为 角秒/像素
static constexpr double ARCSEC_PER_UM_PER_MM = 206.265;

namespace {

// 模块内全局 Logger（由 ss_density_match 初始化）
v42::Logger g_logger;

// FOV 和密度计算结果（内部结构，对应 V4.1 FovDensityInfo）
struct FovDensityInfo {
    double s0;              // 像素尺度(角秒/像素)
    double fov_diag_deg;    // FOV 对角线(度)
    double query_radius_deg;// 查询半径(度) = fov_diag_deg × gaia_query_radius_factor
    double query_area_sqdeg;// 查询面积(平方度) = π × query_radius²
    double img_area_sqdeg;  // 图像面积(平方度)
    double rho_img;         // 图像亮星密度(颗/平方度)
    double rho_target;      // 目标星表密度 = gaia_density_ratio × rho_img
    int    n_target;        // 目标星数 = max(50, round(gaia_density_ratio × n_img × query_area/img_area))
};

// 密度匹配迭代查询结果（内部结构）
struct DensityMatchResult {
    double final_mag_lim;   // 最终极限星等
    int    final_n_gaia;    // 最终 Gaia 星数
    int    iterations;      // 实际迭代次数
    bool   converged;       // 是否收敛
};

// ----------------------------------------------------------------------------
// compute_fov_and_density_asym
// 公式（V4.1 不对称密度匹配）:
//   s0            = 206.265 × pixel_size_um / focal_length_mm
//   fov_diag_deg  = sqrt(W² + H²) × s0 / 3600
//   query_radius  = fov_diag_deg × gaia_query_radius_factor
//   query_area    = π × query_radius²
//   img_area      = (W × s0/3600) × (H × s0/3600)
//   rho_img       = n_img_bright / img_area
//   rho_target    = gaia_density_ratio × rho_img
//   n_target      = max(50, round(gaia_density_ratio × n_img × query_area/img_area))
// ----------------------------------------------------------------------------
FovDensityInfo compute_fov_and_density_asym(
    double focal_length_mm, double pixel_size_um,
    double img_width, double img_height,
    int n_img_bright, double gaia_density_ratio, double gaia_query_radius_factor)
{
    FovDensityInfo info{};

    if (focal_length_mm <= 0.0) {
        g_logger.error("compute_fov_and_density_asym: focal_length_mm=" +
                       std::to_string(focal_length_mm) + " 非法");
        return info;
    }

    // 像素尺度（角秒/像素）
    info.s0 = ARCSEC_PER_UM_PER_MM * pixel_size_um / focal_length_mm;

    // FOV 对角线（度）
    double diag_pix = std::sqrt(img_width * img_width + img_height * img_height);
    double fov_diag_asec = diag_pix * info.s0;
    info.fov_diag_deg = fov_diag_asec / 3600.0;

    // 查询半径（度）与查询面积（平方度）
    // V4.1: 查询半径 = fov_diag × gaia_query_radius_factor (默认0.55)
    info.query_radius_deg = info.fov_diag_deg * gaia_query_radius_factor;
    info.query_area_sqdeg = PI * info.query_radius_deg * info.query_radius_deg;

    // 图像面积(平方度)
    info.img_area_sqdeg = (img_width * info.s0 / 3600.0) * (img_height * info.s0 / 3600.0);
    if (info.img_area_sqdeg <= 0.0) info.img_area_sqdeg = info.query_area_sqdeg;

    // 图像面密度
    info.rho_img = (info.img_area_sqdeg > 0.0)
                   ? static_cast<double>(n_img_bright) / info.img_area_sqdeg
                   : 0.0;

    // V4.1: Gaia目标密度 = gaia_density_ratio × 图像密度
    info.rho_target = gaia_density_ratio * info.rho_img;

    // V4.1: 目标星数 = gaia_density_ratio × n_img × (查询圆面积/图像面积), 下限50
    double img_area_safe = std::max(info.img_area_sqdeg, 1e-10);
    double n_target_dbl = gaia_density_ratio * static_cast<double>(n_img_bright)
                        * (info.query_area_sqdeg / img_area_safe);
    info.n_target = std::max(50, static_cast<int>(std::lround(n_target_dbl)));

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "FOV计算: s0=%.4f\"/px, FOV_diag=%.4f°, query_r=%.4f°, "
        "img_area=%.5f°², query_area=%.5f°², rho_img=%.2f, rho_target=%.2f, n_target=%d",
        info.s0, info.fov_diag_deg, info.query_radius_deg,
        info.img_area_sqdeg, info.query_area_sqdeg,
        info.rho_img, info.rho_target, info.n_target);
    g_logger.info(buf);

    return info;
}

// ----------------------------------------------------------------------------
// compute_initial_mag_cut
// m_cut = 6 + 1.5×log10(f_mm) + 2×log10(t_s)
// ----------------------------------------------------------------------------
double compute_initial_mag_cut(double focal_length_mm, double exposure_time_s)
{
    double f_safe = std::max(focal_length_mm, 1.0);
    double t_safe = std::max(exposure_time_s, 0.1);
    double m_cut = 6.0
                 + 1.5 * std::log10(f_safe)
                 + 2.0 * std::log10(t_safe);

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "初始星等: m_cut=%.4f (f=%.2fmm, t=%.2fs)",
        m_cut, focal_length_mm, exposure_time_s);
    g_logger.info(buf);
    return m_cut;
}

// ----------------------------------------------------------------------------
// density_match_query_asym (自适应步长)
// 算法:
//   m = m_cut_initial
//   for i in 0..max_iter:
//     n = gaia_query_func(ra, dec, radius, m)
//     step = (i < 4) ? step_init : step_init * 0.5
//     if n < n_target×(1-tolerance): m += step  (放宽星等)
//     elif n > n_target×(1+tolerance): m -= step (收紧星等)
//     else: break (收敛)
//   return m, n, i, converged
// ----------------------------------------------------------------------------
DensityMatchResult density_match_query_asym(
    double center_ra, double center_dec, double query_radius_deg,
    int n_target, double m_cut_initial,
    double step_init, int max_iter, double tolerance,
    GaiaQueryFunc gaia_query_func)
{
    DensityMatchResult result{};
    result.final_mag_lim = m_cut_initial;
    result.final_n_gaia  = 0;
    result.iterations    = 0;
    result.converged     = false;

    if (!gaia_query_func) {
        g_logger.error("density_match_query_asym: gaia_query_func 为空");
        return result;
    }
    if (n_target <= 0) {
        g_logger.error("density_match_query_asym: n_target=" +
                       std::to_string(n_target) + " 非正");
        return result;
    }

    // 容差上下界
    double n_lo = n_target * (1.0 - tolerance);
    double n_hi = n_target * (1.0 + tolerance);

    double m = m_cut_initial;
    int    n = 0;
    int    i = 0;
    bool   converged = false;

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "迭代开始: ra=%.4f, dec=%.4f, r=%.4f°, N_target=%d, "
        "tol=%.2f, 范围=[%.1f, %.1f], m0=%.3f, step_init=%.3f, max_iter=%d",
        center_ra, center_dec, query_radius_deg, n_target,
        tolerance, n_lo, n_hi, m_cut_initial, step_init, max_iter);
    g_logger.info(buf);

    for (i = 0; i < max_iter; ++i) {
        n = gaia_query_func(center_ra, center_dec, query_radius_deg, m);

        // V4.1 自适应步长: 前4次 step_init, 后续 step_init/2
        double step = (i < 4) ? step_init : step_init * 0.5;

        std::snprintf(buf, sizeof(buf),
            "iter=%d  m_lim=%.3f  n_gaia=%d  (target=%d, 范围=[%.1f,%.1f], step=%.3f)",
            i, m, n, n_target, n_lo, n_hi, step);
        g_logger.info(buf);

        if (n < n_lo) {
            // 星数不足 → 放宽星等
            m += step;
        } else if (n > n_hi) {
            // 星数过多 → 收紧星等
            m -= step;
        } else {
            converged = true;
            break;
        }
    }

    result.final_mag_lim = m;
    result.final_n_gaia  = n;
    result.iterations    = i;
    result.converged     = converged;

    std::snprintf(buf, sizeof(buf),
        "迭代结束: %s  m_final=%.3f  n_final=%d  iters=%d",
        converged ? "收敛" : "未收敛(达到max_iter)",
        result.final_mag_lim, result.final_n_gaia, result.iterations);
    g_logger.info(buf);

    return result;
}

} // anonymous namespace

// ----------------------------------------------------------------------------
// ss_density_match: 主入口
// ----------------------------------------------------------------------------
SS_API int ss_density_match(
    const StarSelectorParams* params,
    GaiaQueryFunc gaia_query,
    StarSelectionResult* result)
{
    if (!params || !result) {
        return -1;
    }
    // 清零结果
    *result = StarSelectionResult{};

    // 初始化日志
    if (params->log_file_path != nullptr && params->log_file_path[0] != '\0') {
        g_logger.init(std::string(params->log_file_path));
    }
    g_logger.info("=== StarSelector ss_density_match 启动 ===");

    // 参数校验
    if (params->focal_length_mm <= 0.0) {
        g_logger.error("focal_length_mm 非法: " +
                       std::to_string(params->focal_length_mm));
        return -2;
    }
    if (params->img_width <= 0 || params->img_height <= 0) {
        g_logger.error("img_width/img_height 非法");
        return -3;
    }
    if (params->n_img_bright <= 0) {
        g_logger.error("n_img_bright 非正: " +
                       std::to_string(params->n_img_bright));
        return -4;
    }
    if (!gaia_query) {
        g_logger.error("gaia_query 回调为空");
        return -5;
    }

    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "输入参数: img_n_target=%d, gaia_density_ratio=%.3f, "
        "gaia_query_radius_factor=%.3f, m_lim_step=%.3f, m_lim_max_iter=%d, "
        "density_tolerance=%.3f, f=%.2fmm, pix=%.3fum, W=%d, H=%d, "
        "ra=%.4f, dec=%.4f, n_img_bright=%d, exptime=%.2fs",
        params->img_n_target, params->gaia_density_ratio,
        params->gaia_query_radius_factor, params->m_lim_step,
        params->m_lim_max_iter, params->density_tolerance,
        params->focal_length_mm, params->pixel_size_um,
        (int)params->img_width, (int)params->img_height,
        params->center_ra, params->center_dec,
        params->n_img_bright, params->exposure_time_s);
    g_logger.info(buf);

    // Step 1: 计算FOV与密度
    FovDensityInfo info = compute_fov_and_density_asym(
        params->focal_length_mm, params->pixel_size_um,
        params->img_width, params->img_height,
        params->n_img_bright, params->gaia_density_ratio,
        params->gaia_query_radius_factor);

    if (info.s0 <= 0.0) {
        g_logger.error("FOV计算失败");
        return -6;
    }

    // Step 2: 计算初始极限星等
    double m_cut = compute_initial_mag_cut(
        params->focal_length_mm, params->exposure_time_s);

    // Step 3: 自适应步长迭代极限星等
    DensityMatchResult dm = density_match_query_asym(
        params->center_ra, params->center_dec, info.query_radius_deg,
        info.n_target, m_cut,
        params->m_lim_step, params->m_lim_max_iter, params->density_tolerance,
        gaia_query);

    // 填充结果
    result->s0                = info.s0;
    result->fov_diag_deg      = info.fov_diag_deg;
    result->query_radius_deg  = info.query_radius_deg;
    result->m_lim_final       = dm.final_mag_lim;
    result->n_gaia_final      = dm.final_n_gaia;
    result->m_lim_iterations  = dm.iterations;
    result->converged         = dm.converged;
    result->n_target          = info.n_target;
    result->n_img_selected    = params->n_img_bright;
    result->n_gaia_selected   = dm.final_n_gaia;
    result->rho_img           = info.rho_img;
    result->rho_target        = info.rho_target;
    result->query_area_sqdeg  = info.query_area_sqdeg;
    result->img_area_sqdeg    = info.img_area_sqdeg;

    std::snprintf(buf, sizeof(buf),
        "ss_density_match 完成: s0=%.4f, fov_diag=%.4f°, query_r=%.4f°, "
        "m_lim_final=%.3f, n_gaia_final=%d, iters=%d, converged=%d, n_target=%d",
        result->s0, result->fov_diag_deg, result->query_radius_deg,
        result->m_lim_final, result->n_gaia_final, result->m_lim_iterations,
        (int)result->converged, result->n_target);
    g_logger.info(buf);
    g_logger.info("=== StarSelector ss_density_match 结束 ===");
    g_logger.close();

    return 0;
}

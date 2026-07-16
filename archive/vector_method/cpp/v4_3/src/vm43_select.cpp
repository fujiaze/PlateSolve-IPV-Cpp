// ============================================================================
// vm43_select.cpp - V4.3 StarSelector 模块 (Phase 0)
//
// 职责: 图像侧选星 + Gaia 侧不对称密度匹配查询
// 从 V4.2 ss_core.cpp 迁移核心算法, 并将 V4.2 Python 端逻辑整合到 C++:
//   - 图像读取 (astro_image_io.dll 动态加载)
//   - 星点检测 (star_detector.dll, 句柄由外部注入)
//   - 图像侧选星 (饱和>50全选 / 饱和+非饱和补足50)
//   - FOV/密度计算 + 自适应步长迭代极限星等 (V4.2 ss_core.cpp)
//   - Gaia 锥形查询 (gaia_client.dll, 句柄由外部注入)
//   - Gnomonic 投影 + FOV 内过滤
//
// 接口: 内部 C++ 函数, 无 ctypes 边界, 无 JSON 序列化
// 句柄: 通过 get_gaia_client_handle() / get_star_detector_handle() 获取
// ============================================================================

#include "vm43_internal.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>

#ifdef _WIN32
#include <windows.h>
#endif

namespace v43 {

// ============================================================================
// 物理常量
// ============================================================================

static constexpr double VM43_PI = 3.14159265358979323846;
// 206.265 = (180×3600)/π, 把 um/mm 直接转为 角秒/像素
static constexpr double VM43_ARCSEC_PER_UM_PER_MM = 206.265;
static constexpr double VM43_DEGTORAD = VM43_PI / 180.0;
static constexpr double VM43_RADTODEG = 180.0 / VM43_PI;
// 角秒/弧度 = 180×3600/π
static constexpr double VM43_ASEC_PER_RAD = 206264.80624709636;

// ============================================================================
// 内部 DLL 动态加载 (匿名命名空间, 仅本文件可见)
// ============================================================================

#ifdef _WIN32

// --- astro_image_io.dll 函数指针类型 ---
typedef struct AIOImageData AIOImageData;
typedef AIOImageData* (*aio_read_fn)(const char*);
typedef float* (*aio_get_pixel_data_fn)(const AIOImageData*);
typedef int (*aio_get_width_fn)(const AIOImageData*);
typedef int (*aio_get_height_fn)(const AIOImageData*);
typedef void (*aio_free_image_data_fn)(AIOImageData*);

// --- star_detector.dll 函数指针类型 ---
// StarDetectorHandle 是不透明指针
typedef int (*sdet_detect_ex_fn)(
    void* handle,
    const uint16_t* image, int width, int height,
    double** out_x, double** out_y, float** out_flux, int** out_saturated, int* out_count,
    const char** extra_names, int extra_count, float*** out_extras);
typedef void (*sdet_free_detect_ex_fn)(
    double* x, double* y, float* flux, int* saturated,
    float** extras, int extra_count);

// --- gaia_client.dll 函数指针类型 ---
typedef int (*gaia_cone_search_for_solver_fn)(
    void* client,
    double ra, double dec, double radius_deg,
    double mag_high,
    double** out_ra, double** out_dec, float** out_mag,
    int* out_count);

// 缓存的 DLL 函数指针
struct DllApi {
    bool loaded = false;
    bool load_failed = false;

    HMODULE aio_dll = nullptr;
    aio_read_fn aio_read = nullptr;
    aio_get_pixel_data_fn aio_get_pixel_data = nullptr;
    aio_get_width_fn aio_get_width = nullptr;
    aio_get_height_fn aio_get_height = nullptr;
    aio_free_image_data_fn aio_free = nullptr;

    HMODULE sdet_dll = nullptr;
    sdet_detect_ex_fn sdet_detect_ex = nullptr;
    sdet_free_detect_ex_fn sdet_free_ex = nullptr;

    HMODULE gaia_dll = nullptr;
    gaia_cone_search_for_solver_fn gaia_cone_search = nullptr;
};

static DllApi g_dll;

// 加载所有依赖 DLL (首次调用时加载, 后续复用)
// 返回 true 表示全部加载成功
static bool load_dlls(Logger* logger) {
    if (g_dll.loaded) return true;
    if (g_dll.load_failed) return false;

    // astro_image_io.dll
    g_dll.aio_dll = LoadLibraryA("astro_image_io.dll");
    if (!g_dll.aio_dll) {
        if (logger) logger->error("vm43_select: 无法加载 astro_image_io.dll");
        g_dll.load_failed = true;
        return false;
    }
    g_dll.aio_read = (aio_read_fn)GetProcAddress(g_dll.aio_dll, "aio_read");
    g_dll.aio_get_pixel_data = (aio_get_pixel_data_fn)GetProcAddress(g_dll.aio_dll, "aio_get_pixel_data");
    g_dll.aio_get_width = (aio_get_width_fn)GetProcAddress(g_dll.aio_dll, "aio_get_width");
    g_dll.aio_get_height = (aio_get_height_fn)GetProcAddress(g_dll.aio_dll, "aio_get_height");
    g_dll.aio_free = (aio_free_image_data_fn)GetProcAddress(g_dll.aio_dll, "aio_free_image_data");
    if (!g_dll.aio_read || !g_dll.aio_get_pixel_data ||
        !g_dll.aio_get_width || !g_dll.aio_get_height || !g_dll.aio_free) {
        if (logger) logger->error("vm43_select: astro_image_io 函数符号解析失败");
        g_dll.load_failed = true;
        return false;
    }

    // star_detector.dll
    g_dll.sdet_dll = LoadLibraryA("star_detector.dll");
    if (!g_dll.sdet_dll) {
        if (logger) logger->error("vm43_select: 无法加载 star_detector.dll");
        g_dll.load_failed = true;
        return false;
    }
    g_dll.sdet_detect_ex = (sdet_detect_ex_fn)GetProcAddress(g_dll.sdet_dll, "sdet_detect_ex");
    g_dll.sdet_free_ex = (sdet_free_detect_ex_fn)GetProcAddress(g_dll.sdet_dll, "sdet_free_detect_ex");
    if (!g_dll.sdet_detect_ex || !g_dll.sdet_free_ex) {
        if (logger) logger->error("vm43_select: star_detector 函数符号解析失败");
        g_dll.load_failed = true;
        return false;
    }

    // gaia_client.dll
    g_dll.gaia_dll = LoadLibraryA("gaia_client.dll");
    if (!g_dll.gaia_dll) {
        if (logger) logger->error("vm43_select: 无法加载 gaia_client.dll");
        g_dll.load_failed = true;
        return false;
    }
    g_dll.gaia_cone_search = (gaia_cone_search_for_solver_fn)GetProcAddress(
        g_dll.gaia_dll, "gaia_client_cone_search_for_solver");
    if (!g_dll.gaia_cone_search) {
        if (logger) logger->error("vm43_select: gaia_client_cone_search_for_solver 符号解析失败");
        g_dll.load_failed = true;
        return false;
    }

    g_dll.loaded = true;
    if (logger) logger->info("vm43_select: 依赖 DLL 全部加载成功");
    return true;
}

// 通过 GaiaClient 句柄查询星数 (用于密度迭代)
// 返回: 星数 (查询失败返回 0)
static int gaia_query_count(void* gaia_handle, double ra, double dec,
                             double radius_deg, double mag_lim) {
    if (!g_dll.gaia_cone_search || !gaia_handle) return 0;
    double *out_ra = nullptr, *out_dec = nullptr;
    float *out_mag = nullptr;
    int out_count = 0;
    int ret = g_dll.gaia_cone_search(gaia_handle, ra, dec, radius_deg, mag_lim,
                                      &out_ra, &out_dec, &out_mag, &out_count);
    if (ret != 0) return 0;
    // gaia_client 内部用 malloc 分配, 用 free 释放
    if (out_ra) free(out_ra);
    if (out_dec) free(out_dec);
    if (out_mag) free(out_mag);
    return out_count;
}

// 通过 GaiaClient 句柄查询星表 (返回 ra/dec/mag 数组)
// 返回: 0=成功, -1=失败
static int gaia_query_stars(void* gaia_handle, double ra, double dec,
                             double radius_deg, double mag_lim,
                             std::vector<double>& out_ra,
                             std::vector<double>& out_dec,
                             std::vector<float>& out_mag) {
    out_ra.clear(); out_dec.clear(); out_mag.clear();
    if (!g_dll.gaia_cone_search || !gaia_handle) return -1;
    double *ra_ptr = nullptr, *dec_ptr = nullptr;
    float *mag_ptr = nullptr;
    int count = 0;
    int ret = g_dll.gaia_cone_search(gaia_handle, ra, dec, radius_deg, mag_lim,
                                      &ra_ptr, &dec_ptr, &mag_ptr, &count);
    if (ret != 0) return -1;
    if (count > 0 && ra_ptr && dec_ptr && mag_ptr) {
        out_ra.assign(ra_ptr, ra_ptr + count);
        out_dec.assign(dec_ptr, dec_ptr + count);
        out_mag.assign(mag_ptr, mag_ptr + count);
    }
    if (ra_ptr) free(ra_ptr);
    if (dec_ptr) free(dec_ptr);
    if (mag_ptr) free(mag_ptr);
    return 0;
}

#else
// 非 Windows 平台 stub
static bool load_dlls(Logger* logger) {
    if (logger) logger->error("vm43_select: 非 Windows 平台不支持 DLL 动态加载");
    return false;
}
static int gaia_query_count(void*, double, double, double, double) { return 0; }
static int gaia_query_stars(void*, double, double, double, double,
                             std::vector<double>&, std::vector<double>&,
                             std::vector<float>&) { return -1; }
#endif // _WIN32

// ============================================================================
// 内部辅助函数 (外部链接, 供单元测试调用)
// ============================================================================

// ----------------------------------------------------------------------------
// compute_fov_density - 计算 FOV 与密度 (从 V4.2 ss_core.cpp 迁移)
// ----------------------------------------------------------------------------
void compute_fov_density(
    double focal_length_mm, double pixel_size_um,
    double img_width, double img_height,
    int n_img_bright,
    double gaia_density_ratio, double gaia_query_radius_factor,
    double& s0, double& fov_diag_deg,
    double& query_radius_deg, double& query_area_sqdeg,
    double& img_area_sqdeg, double& rho_img,
    double& rho_target, int& n_target,
    Logger* logger)
{
    // 初始化输出
    s0 = 0.0; fov_diag_deg = 0.0; query_radius_deg = 0.0;
    query_area_sqdeg = 0.0; img_area_sqdeg = 0.0;
    rho_img = 0.0; rho_target = 0.0; n_target = 0;

    if (focal_length_mm <= 0.0) {
        if (logger) logger->error("compute_fov_density: focal_length_mm 非法");
        return;
    }

    // 像素尺度 (角秒/像素)
    s0 = VM43_ARCSEC_PER_UM_PER_MM * pixel_size_um / focal_length_mm;

    // FOV 对角线 (度)
    double diag_pix = std::sqrt(img_width * img_width + img_height * img_height);
    double fov_diag_asec = diag_pix * s0;
    fov_diag_deg = fov_diag_asec / 3600.0;

    // 查询半径 (度) 与查询面积 (平方度)
    query_radius_deg = fov_diag_deg * gaia_query_radius_factor;
    query_area_sqdeg = VM43_PI * query_radius_deg * query_radius_deg;

    // 图像面积 (平方度)
    img_area_sqdeg = (img_width * s0 / 3600.0) * (img_height * s0 / 3600.0);
    if (img_area_sqdeg <= 0.0) img_area_sqdeg = query_area_sqdeg;

    // 图像面密度
    rho_img = (img_area_sqdeg > 0.0)
              ? static_cast<double>(n_img_bright) / img_area_sqdeg
              : 0.0;

    // Gaia 目标密度 = gaia_density_ratio × 图像密度
    rho_target = gaia_density_ratio * rho_img;

    // 目标星数 = gaia_density_ratio × n_img × (查询圆面积/图像面积), 下限 50
    double img_area_safe = std::max(img_area_sqdeg, 1e-10);
    double n_target_dbl = gaia_density_ratio * static_cast<double>(n_img_bright)
                        * (query_area_sqdeg / img_area_safe);
    n_target = std::max(50, static_cast<int>(std::lround(n_target_dbl)));

    if (logger) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "FOV计算: s0=%.4f\"/px, FOV_diag=%.4f°, query_r=%.4f°, "
            "img_area=%.5f°², query_area=%.5f°², rho_img=%.2f, rho_target=%.2f, n_target=%d",
            s0, fov_diag_deg, query_radius_deg,
            img_area_sqdeg, query_area_sqdeg,
            rho_img, rho_target, n_target);
        logger->info(buf);
    }
}

// ----------------------------------------------------------------------------
// compute_initial_mag_cut - 计算初始极限星等 (V4.2 公式)
//   m_cut = 6 + 1.5×log10(f_mm) + 2×log10(t_s)
// ----------------------------------------------------------------------------
double compute_initial_mag_cut(
    double focal_length_mm, double exposure_time_s,
    Logger* logger)
{
    double f_safe = std::max(focal_length_mm, 1.0);
    double t_safe = std::max(exposure_time_s, 0.1);
    double m_cut = 6.0
                 + 1.5 * std::log10(f_safe)
                 + 2.0 * std::log10(t_safe);

    if (logger) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "初始星等: m_cut=%.4f (f=%.2fmm, t=%.2fs)",
            m_cut, focal_length_mm, exposure_time_s);
        logger->info(buf);
    }
    return m_cut;
}

// ----------------------------------------------------------------------------
// density_match_iterate - 自适应步长迭代极限星等 (从 V4.2 ss_core.cpp 迁移)
//   前4次 step_init, 后续 step_init/2
// ----------------------------------------------------------------------------
void density_match_iterate(
    std::function<int(double, double, double, double)> query_func,
    double center_ra, double center_dec, double query_radius_deg,
    int n_target, double m_cut_initial,
    double step_init, int max_iter, double tolerance,
    double& final_mag_lim, int& final_n_gaia,
    int& iterations, bool& converged,
    Logger* logger)
{
    final_mag_lim = m_cut_initial;
    final_n_gaia = 0;
    iterations = 0;
    converged = false;

    if (!query_func) {
        if (logger) logger->error("density_match_iterate: query_func 为空");
        return;
    }
    if (n_target <= 0) {
        if (logger) logger->error("density_match_iterate: n_target 非正");
        return;
    }

    // 容差上下界
    double n_lo = n_target * (1.0 - tolerance);
    double n_hi = n_target * (1.0 + tolerance);

    double m = m_cut_initial;
    int n = 0;
    int i = 0;

    if (logger) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "迭代开始: ra=%.4f, dec=%.4f, r=%.4f°, N_target=%d, "
            "tol=%.2f, 范围=[%.1f, %.1f], m0=%.3f, step_init=%.3f, max_iter=%d",
            center_ra, center_dec, query_radius_deg, n_target,
            tolerance, n_lo, n_hi, m_cut_initial, step_init, max_iter);
        logger->info(buf);
    }

    for (i = 0; i < max_iter; ++i) {
        n = query_func(center_ra, center_dec, query_radius_deg, m);

        // V4.1 自适应步长: 前4次 step_init, 后续 step_init/2
        double step = (i < 4) ? step_init : step_init * 0.5;

        if (logger) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "iter=%d  m_lim=%.3f  n_gaia=%d  (target=%d, 范围=[%.1f,%.1f], step=%.3f)",
                i, m, n, n_target, n_lo, n_hi, step);
            logger->info(buf);
        }

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

    final_mag_lim = m;
    final_n_gaia = n;
    iterations = i;
    converged = converged;

    if (logger) {
        char buf[512];
        std::snprintf(buf, sizeof(buf),
            "迭代结束: %s  m_final=%.3f  n_final=%d  iters=%d",
            converged ? "收敛" : "未收敛(达到max_iter)",
            final_mag_lim, final_n_gaia, iterations);
        logger->info(buf);
    }
}

// ----------------------------------------------------------------------------
// select_image_stars - 图像侧选星: V4.1 不对称策略
//   饱和星数 > img_n_target → 全选饱和星
//   否则 → 饱和全选 + 非饱和按 flux 降序补足到 img_n_target
// ----------------------------------------------------------------------------
std::vector<int> select_image_stars(
    const std::vector<double>& flux,
    const std::vector<bool>& saturated,
    int img_n_target,
    Logger* logger)
{
    std::vector<int> sel_idx;
    int n_total = static_cast<int>(flux.size());
    if (n_total == 0 || img_n_target <= 0) return sel_idx;

    // 分离饱和与非饱和索引
    std::vector<int> sat_idx, non_sat_idx;
    for (int i = 0; i < n_total; ++i) {
        if (saturated[i]) sat_idx.push_back(i);
        else non_sat_idx.push_back(i);
    }
    int nsat = static_cast<int>(sat_idx.size());

    // 非饱和按 flux 降序排列
    std::sort(non_sat_idx.begin(), non_sat_idx.end(),
              [&](int a, int b) { return flux[a] > flux[b]; });

    if (nsat > img_n_target) {
        // 饱和星数 > 目标数 → 全选饱和星
        sel_idx = sat_idx;
        if (logger) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "选星: 饱和星 %d 颗 > img_n_target=%d, 全选饱和星",
                nsat, img_n_target);
            logger->info(buf);
        }
    } else {
        // 饱和全选 + 非饱和补足
        int n_needed = std::max(0, img_n_target - nsat);
        sel_idx.reserve(nsat + n_needed);
        for (int i = 0; i < nsat; ++i) sel_idx.push_back(sat_idx[i]);
        for (int i = 0; i < n_needed && i < static_cast<int>(non_sat_idx.size()); ++i) {
            sel_idx.push_back(non_sat_idx[i]);
        }
        if (logger) {
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "选星: 饱和 %d + 非饱和 %d = %d 颗 (img_n_target=%d)",
                nsat, n_needed, static_cast<int>(sel_idx.size()), img_n_target);
            logger->info(buf);
        }
    }
    return sel_idx;
}

// ----------------------------------------------------------------------------
// gnomonic_forward_proj - Gnomonic 正向投影 (从 V4.2 Python 迁移)
//   将天球坐标 (ra, dec) 投影到以 (ra0, dec0) 为中心的切平面
//   输出: xi, eta (角秒), valid (cosc > 1e-10 时有效)
// ----------------------------------------------------------------------------
void gnomonic_forward_proj(
    double ra_deg, double dec_deg,
    double ra0_deg, double dec0_deg,
    double& xi_asec, double& eta_asec, bool& valid)
{
    double ra = ra_deg * VM43_DEGTORAD;
    double dec = dec_deg * VM43_DEGTORAD;
    double ra0 = ra0_deg * VM43_DEGTORAD;
    double dec0 = dec0_deg * VM43_DEGTORAD;

    double sin_dec0 = std::sin(dec0), cos_dec0 = std::cos(dec0);
    double delta_ra = ra - ra0;
    double sin_dec = std::sin(dec), cos_dec = std::cos(dec);
    double cos_delta_ra = std::cos(delta_ra);

    // cosc = sin(dec0)×sin(dec) + cos(dec0)×cos(dec)×cos(delta_ra)
    double cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_delta_ra;
    valid = (cosc > 1e-10);
    double cosc_safe = valid ? cosc : 1.0;

    // xi = cos(dec)×sin(delta_ra) / cosc  (弧度)
    // eta = (cos(dec0)×sin(dec) - sin(dec0)×cos(dec)×cos(delta_ra)) / cosc  (弧度)
    double xi_rad = cos_dec * std::sin(delta_ra) / cosc_safe;
    double eta_rad = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_delta_ra) / cosc_safe;

    // 转换为角秒
    xi_asec = valid ? xi_rad * VM43_ASEC_PER_RAD : 0.0;
    eta_asec = valid ? eta_rad * VM43_ASEC_PER_RAD : 0.0;
}

// ============================================================================
// vm43_select 主实现
// ============================================================================

int vm43_select(
    const std::string& image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const VM43SolveParams& params,
    StarSelection& output,
    Logger* logger)
{
    // 清零输出
    output = StarSelection{};

    // --- 参数校验 ---
    if (focal_length_mm <= 0.0) {
        if (logger) logger->error("vm43_select: focal_length_mm 非法");
        return -1;
    }
    if (pixel_size_um <= 0.0) {
        if (logger) logger->error("vm43_select: pixel_size_um 非法");
        return -1;
    }
    if (image_path.empty()) {
        if (logger) logger->error("vm43_select: image_path 为空");
        return -1;
    }

    if (logger) logger->info("=== vm43_select 启动 ===");

    // --- 获取注入的句柄 ---
    void* gaia_handle = get_gaia_client_handle();
    void* detector_handle = get_star_detector_handle();
    if (!gaia_handle) {
        if (logger) logger->error("vm43_select: GaiaClient 句柄未注入");
        return -1;
    }
    if (!detector_handle) {
        if (logger) logger->error("vm43_select: StarDetector 句柄未注入");
        return -1;
    }

    // --- 加载依赖 DLL ---
    if (!load_dlls(logger)) {
        if (logger) logger->error("vm43_select: 依赖 DLL 加载失败");
        return -1;
    }

#ifdef _WIN32
    // --- Step 1: 读取图像 ---
    if (logger) logger->info("Step 1: 读取图像 " + image_path);
    AIOImageData* img_data = g_dll.aio_read(image_path.c_str());
    if (!img_data) {
        if (logger) logger->error("vm43_select: 图像读取失败");
        return -1;
    }
    int img_w = g_dll.aio_get_width(img_data);
    int img_h = g_dll.aio_get_height(img_data);
    float* pixel_data = g_dll.aio_get_pixel_data(img_data);
    if (img_w <= 0 || img_h <= 0 || !pixel_data) {
        if (logger) logger->error("vm43_select: 图像数据非法");
        g_dll.aio_free(img_data);
        return -1;
    }
    if (logger) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  图像尺寸: %d×%d", img_w, img_h);
        logger->info(buf);
    }

    // 转 uint16 (star_detector 需要 uint16_t*)
    std::vector<uint16_t> img_u16(static_cast<size_t>(img_w) * img_h);
    for (size_t i = 0; i < img_u16.size(); ++i) {
        float v = pixel_data[i];
        if (v < 0.0f) v = 0.0f;
        if (v > 65535.0f) v = 65535.0f;
        img_u16[i] = static_cast<uint16_t>(v);
    }
    g_dll.aio_free(img_data);

    // --- Step 2: 星点检测 ---
    if (logger) logger->info("Step 2: 星点检测");
    double *det_x = nullptr, *det_y = nullptr;
    float *det_flux = nullptr;
    int *det_sat = nullptr;
    int det_count = 0;
    int det_ret = g_dll.sdet_detect_ex(
        detector_handle, img_u16.data(), img_w, img_h,
        &det_x, &det_y, &det_flux, &det_sat, &det_count,
        nullptr, 0, nullptr);
    if (det_ret != 0 || det_count <= 0) {
        if (logger) logger->error("vm43_select: 星点检测失败或未检测到星");
        if (det_x || det_y || det_flux || det_sat) {
            g_dll.sdet_free_ex(det_x, det_y, det_flux, det_sat, nullptr, 0);
        }
        return -1;
    }
    if (logger) {
        int n_sat = 0;
        for (int i = 0; i < det_count; ++i) if (det_sat[i]) n_sat++;
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  检测到星点: %d 颗 (饱和 %d, 正常 %d)",
                      det_count, n_sat, det_count - n_sat);
        logger->info(buf);
    }

    // --- Step 3: 图像侧选星 (V4.1 不对称策略) ---
    if (logger) logger->info("Step 3: 图像侧选星");
    std::vector<double> flux_vec(det_flux, det_flux + det_count);
    std::vector<bool> sat_vec(det_count);
    for (int i = 0; i < det_count; ++i) sat_vec[i] = (det_sat[i] != 0);

    std::vector<int> sel_idx = select_image_stars(
        flux_vec, sat_vec, params.img_n_target, logger);
    int N = static_cast<int>(sel_idx.size());
    if (N < 2) {
        if (logger) logger->error("vm43_select: 图像侧选星数过少");
        g_dll.sdet_free_ex(det_x, det_y, det_flux, det_sat, nullptr, 0);
        return -1;
    }

    // --- Step 4: 构建 U 向量组 (角秒坐标, 原点在图像中心, Y 轴向上) ---
    double s0 = VM43_ARCSEC_PER_UM_PER_MM * pixel_size_um / focal_length_mm;
    double cx = img_w / 2.0, cy = img_h / 2.0;
    output.U.resize(N);
    for (int i = 0; i < N; ++i) {
        int idx = sel_idx[i];
        output.U[i].x = (det_x[idx] - cx) * s0;
        output.U[i].y = -(det_y[idx] - cy) * s0;  // Y 轴向上 (图像 Y 向下)
        output.U[i].flux = static_cast<double>(det_flux[idx]);
        output.U[i].saturated = (det_sat[idx] != 0);
    }
    output.img_width = img_w;
    output.img_height = img_h;

    // 释放星点检测内存
    g_dll.sdet_free_ex(det_x, det_y, det_flux, det_sat, nullptr, 0);

    if (logger) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "  U 向量组: %d×2 (s0=%.4f\"/px)", N, s0);
        logger->info(buf);
    }

    // --- Step 5: 计算 FOV 与密度 ---
    if (logger) logger->info("Step 4: FOV/密度计算");
    double fov_diag_deg, query_radius_deg, query_area_sqdeg, img_area_sqdeg;
    double rho_img, rho_target;
    int n_target;
    compute_fov_density(
        focal_length_mm, pixel_size_um,
        static_cast<double>(img_w), static_cast<double>(img_h),
        N, params.gaia_density_ratio, params.gaia_query_radius_factor,
        s0, fov_diag_deg, query_radius_deg, query_area_sqdeg,
        img_area_sqdeg, rho_img, rho_target, n_target, logger);

    output.fov_diag_deg = fov_diag_deg;
    output.rho_img = rho_img;
    output.rho_target = rho_target;

    // --- Step 6: 计算初始极限星等 ---
    // 曝光时间默认 1.0s (V4.3 不从 FITS header 读取, 由调用方在 params 中指定)
    double exposure_time_s = 1.0;  // 默认值, 后续可扩展从 params 读取
    double m_cut = compute_initial_mag_cut(focal_length_mm, exposure_time_s, logger);

    // --- Step 7: 自适应步长迭代极限星等 ---
    if (logger) logger->info("Step 5: Gaia 密度匹配迭代");
    auto query_func = [gaia_handle](double q_ra, double q_dec, double q_r, double q_m)
                      -> int {
        return gaia_query_count(gaia_handle, q_ra, q_dec, q_r, q_m);
    };

    double m_lim_final;
    int n_gaia_final, m_lim_iters;
    bool m_lim_converged;
    density_match_iterate(
        query_func, ra, dec, query_radius_deg,
        n_target, m_cut,
        params.m_lim_step, params.m_lim_max_iter, params.density_tolerance,
        m_lim_final, n_gaia_final, m_lim_iters, m_lim_converged, logger);

    output.m_lim_final = m_lim_final;
    output.n_gaia_final = n_gaia_final;
    output.m_lim_iterations = m_lim_iters;

    // --- Step 8: 用最终极限星等查询 Gaia 星表 ---
    if (logger) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "Step 6: 用 m_lim=%.3f 查询 Gaia 星表", m_lim_final);
        logger->info(buf);
    }
    std::vector<double> cat_ra, cat_dec;
    std::vector<float> cat_mag;
    int q_ret = gaia_query_stars(gaia_handle, ra, dec, query_radius_deg, m_lim_final,
                                  cat_ra, cat_dec, cat_mag);
    if (q_ret != 0 || cat_ra.size() < 2) {
        // 兜底: 放宽到 mag=22
        if (logger) logger->warn("Gaia 返回过少, 启用 mag=22 兜底查询");
        q_ret = gaia_query_stars(gaia_handle, ra, dec, query_radius_deg, 22.0,
                                  cat_ra, cat_dec, cat_mag);
    }
    if (q_ret != 0 || cat_ra.size() < 2) {
        if (logger) logger->error("vm43_select: Gaia 星表查询星数过少");
        return -1;
    }

    // --- Step 9: Gnomonic 投影 + FOV 内过滤 ---
    if (logger) logger->info("Step 7: Gnomonic 投影 + FOV 过滤");
    double fov_half_w = img_w / 2.0 * s0;  // 角秒
    double fov_half_h = img_h / 2.0 * s0;

    std::vector<int> fov_idx;
    for (size_t i = 0; i < cat_ra.size(); ++i) {
        double xi, eta;
        bool valid;
        gnomonic_forward_proj(cat_ra[i], cat_dec[i], ra, dec, xi, eta, valid);
        if (!valid) continue;
        if (std::abs(xi) < fov_half_w && std::abs(eta) < fov_half_h) {
            fov_idx.push_back(static_cast<int>(i));
        }
    }
    if (fov_idx.size() < 2) {
        // 放宽到 1.5×FOV
        if (logger) logger->warn("FOV 内星数过少, 放宽到 1.5×FOV");
        fov_idx.clear();
        for (size_t i = 0; i < cat_ra.size(); ++i) {
            double xi, eta;
            bool valid;
            gnomonic_forward_proj(cat_ra[i], cat_dec[i], ra, dec, xi, eta, valid);
            if (!valid) continue;
            if (std::abs(xi) < fov_half_w * 1.5 && std::abs(eta) < fov_half_h * 1.5) {
                fov_idx.push_back(static_cast<int>(i));
            }
        }
    }
    if (fov_idx.size() < 2) {
        if (logger) logger->error("vm43_select: FOV 内 Gaia 星数过少");
        return -1;
    }

    // --- Step 10: 按星等升序 (最亮优先) 取前 n_target 颗 ---
    std::sort(fov_idx.begin(), fov_idx.end(),
              [&](int a, int b) { return cat_mag[a] < cat_mag[b]; });
    int M = std::min(n_target, static_cast<int>(fov_idx.size()));

    output.W.resize(M);
    for (int i = 0; i < M; ++i) {
        int idx = fov_idx[i];
        double xi, eta;
        bool valid;
        gnomonic_forward_proj(cat_ra[idx], cat_dec[idx], ra, dec, xi, eta, valid);
        output.W[i].x = xi;
        output.W[i].y = eta;
        output.W[i].flux = 0.0;  // Gaia 星表无 flux, 用星等代理 (后续模块不依赖)
        output.W[i].saturated = false;
    }

    if (logger) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "  W 向量组: %d×2 (FOV 内 %d, 取最亮 %d)",
            M, static_cast<int>(fov_idx.size()), M);
        logger->info(buf);
    }
#else
    // 非 Windows 平台不支持
    if (logger) logger->error("vm43_select: 非 Windows 平台不支持");
    return -1;
#endif // _WIN32

    output.success = true;
    if (logger) logger->info("=== vm43_select 完成 ===");
    return 0;
}

} // namespace v43

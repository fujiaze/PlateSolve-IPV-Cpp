#ifndef SS_API_H
#define SS_API_H

// ============================================================================
// ss_api.h - V4.2 StarSelector 模块（Task 2）
//
// Phase 0 不对称选星 + Gaia 密度匹配查询的独立 C++ DLL 模块。
// 从 V4.1 的 vm4_density.cpp 抽取 compute_fov_and_density_asym 与
// density_match_query_asym 逻辑，封装为独立可调用模块。
//
// 设计:
//   - C++ 端仅负责密度匹配查询逻辑（FOV 计算 + 自适应步长迭代极限星等）
//   - Gaia 实际查询通过回调函数 GaiaQueryFunc 由 Python 端注入
//   - 图像侧选星在 Python 端完成（需调用 star_detector）
//   - 日志通过 v42::Logger 记录（共享日志接口）
//
// 公式:
//   s0            = 206.265 × pixel_size_um / focal_length_mm
//   FOV_diag      = sqrt(W² + H²) × s0 / 3600
//   query_radius  = FOV_diag × gaia_query_radius_factor
//   img_area      = W × s0/3600 × H × s0/3600
//   rho_img       = n_img_bright / img_area
//   rho_target    = gaia_density_ratio × rho_img
//   n_target      = max(50, round(gaia_density_ratio × n_img × query_area/img_area))
//   迭代: m=m_cut; 前4次步长=m_lim_step, 后续 m_lim_step/2;
//         n<n_lo→m+=step; n>n_hi→m-=step; 否则收敛
// ============================================================================

#include "v42_types.h"

#ifdef _WIN32
#define SS_API __declspec(dllexport)
#else
#define SS_API __attribute__((visibility("default")))
#endif

// StarSelector 输入参数
struct StarSelectorParams {
    int    img_n_target;              // 图像侧目标星数(默认50)
    double gaia_density_ratio;        // Gaia面密度/图像面密度(默认1.5)
    double gaia_query_radius_factor;  // Gaia查询半径因子(默认0.55)
    double m_lim_step;                // 极限星等迭代步长(默认0.5)
    int    m_lim_max_iter;            // 极限星等迭代最大次数(默认15)
    double density_tolerance;         // 密度匹配容差(默认0.1)
    double focal_length_mm;           // 焦距(mm)
    double pixel_size_um;             // 像元尺寸(um)
    double img_width;                 // 图像宽度(像素)
    double img_height;                // 图像高度(像素)
    double center_ra;                 // 中心赤经(度)
    double center_dec;                // 中心赤纬(度)
    int    n_img_bright;              // 图像侧亮星数(用于密度计算)
    double exposure_time_s;           // 曝光时间(s) for m_cut 初值估计
    const char* log_file_path;        // 日志文件路径(可选, NULL时不写)
};

// StarSelector 输出结果
struct StarSelectionResult {
    double s0;                        // 像素尺度(角秒/像素)
    double fov_diag_deg;              // FOV对角线(度)
    double query_radius_deg;          // 查询半径(度)
    double m_lim_final;               // 最终极限星等
    int    n_gaia_final;              // 最终Gaia星数
    int    m_lim_iterations;          // 星等迭代次数
    bool   converged;                 // 是否收敛
    int    n_target;                  // 目标星数
    int    n_img_selected;            // 图像侧选中星数(回显 params.n_img_bright)
    int    n_gaia_selected;           // Gaia侧选中星数(= n_gaia_final)
    double rho_img;                   // 图像亮星密度(颗/平方度)
    double rho_target;                // 目标星表密度(颗/平方度)
    double query_area_sqdeg;          // 查询圆面积(平方度)
    double img_area_sqdeg;            // 图像面积(平方度)
};

#ifdef __cplusplus
extern "C" {
#endif

// Gaia 查询回调类型: (ra, dec, radius_deg, mag_lim) -> 星数
// 由 Python 端传入具体实现（封装 GaiaClientPy.cone_search）
typedef int (*GaiaQueryFunc)(double, double, double, double);

// 执行不对称密度匹配查询
// 返回: 0=成功, <0=失败
SS_API int ss_density_match(
    const StarSelectorParams* params,
    GaiaQueryFunc gaia_query,
    StarSelectionResult* result
);

#ifdef __cplusplus
}
#endif

#endif // SS_API_H

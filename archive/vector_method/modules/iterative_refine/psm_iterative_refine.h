/**
 * @file psm_iterative_refine.h
 * @brief 第二步迭代精化模块 - SIP畸变拟合
 * 
 * 功能概述:
 *   在第一步粗匹配获得精确变换参数后，进行全星点精细匹配和畸变拟合。
 *   输出完整的WCS参数和SIP畸变系数。
 * 
 * 算法流程:
 *   1. 应用第一步变换（翻转+旋转）到Gaia像素坐标
 *   2. FOV裁剪（保留图像范围内的Gaia星）
 *   3. 近邻匹配（TOP 2000图像星 vs TOP 3000 Gaia星）
 *   4. 残差-距离分析（诊断畸变模式）
 *   5. SIP多项式拟合（最高5阶）
 *   6. SIP系数计算（正向A/B和逆向AP/BP）
 *   7. 更新中心坐标和CD矩阵
 * 
 * 待完善:
 *   - 迭代优化循环（当前单次拟合）
 *   - 分区域异常值剔除
 *   - 三角形特征匹配验证
 * 
 * 参考:
 *   - Siril astrometry_solver.c
 *   - WCSLIB SIP定义
 */

#ifndef PSM_ITERATIVE_REFINE_H
#define PSM_ITERATIVE_REFINE_H

#include "../common/psm_common.h"

#ifdef _WIN32
#define IR_EXPORT __declspec(dllexport)
#else
#define IR_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define IR_MAX_STARS_TRIANGLE    500
#define IR_TRI_RATIO_RADIUS      0.002
#define IR_TRI_MIN_AREA          100.0
#define IR_TRI_MAX_BA_RATIO      0.95
#define IR_TRI_EQUILATERAL_THRESH 0.1
#define IR_GRID_SIZE             5
#define IR_OUTLIER_ANGLE_THRESH  1.57079632679
#define IR_OUTLIER_MAG_RATIO     3.0
#define IR_MAX_ITERATIONS        5
#define IR_CONVERGE_THRESH       0.01

/**
 * @brief 第一步输出的初始变换参数
 * 
 * 从star_alignment模块获得，用于初始化第二步的坐标变换
 */
typedef struct {
    double center_ra;          /**< 中心RA（度） */
    double center_dec;         /**< 中心Dec（度） */
    double rotation_deg;       /**< 旋转角（度） */
    double scale_arcsec_px;    /**< 像素尺度（角秒/像素） */
    int flip_mode;             /**< 翻转模式：0=无, 1=X, 2=Y, 3=XY */
    int img_width;             /**< 图像宽度 */
    int img_height;            /**< 图像高度 */
} IRInitialTransform;

/**
 * @brief 图像星点输入数据
 */
typedef struct {
    const double *img_x;       /**< X坐标数组（相对于CRPIX） */
    const double *img_y;       /**< Y坐标数组（相对于CRPIX） */
    const double *img_flux;    /**< Flux数组（用于排序） */
    int img_count;             /**< 星点总数 */
    const int *img_saturated;  /**< 饱和标记数组（1=饱和） */
    int n_saturated;           /**< 饱和星数量 */
} IRImageStars;

/**
 * @brief 星表星点输入数据
 */
typedef struct {
    const double *cat_ra;      /**< RA数组（度） */
    const double *cat_dec;     /**< Dec数组（度） */
    const double *cat_mag;     /**< 星等数组（用于排序） */
    const double *cat_x_px;    /**< 像素X坐标（Gnomonic投影后） */
    const double *cat_y_px;    /**< 像素Y坐标（Gnomonic投影后） */
    int cat_count;             /**< 星表星点总数 */
} IRCatalogStars;

/**
 * @brief SIP畸变系数
 * 
 * 正向变换（像素→中间坐标）：
 *   u = x + Σ A[i][j] * x^i * y^j
 *   v = y + Σ B[i][j] * x^i * y^j
 * 
 * 逆向变换（中间坐标→像素）：
 *   x = u + Σ AP[i][j] * u^i * v^j
 *   y = v + Σ BP[i][j] * u^i * v^j
 */
typedef struct {
    double A[6][6];            /**< 正向X系数 */
    double B[6][6];            /**< 正向Y系数 */
    double AP[6][6];           /**< 逆向X系数 */
    double BP[6][6];           /**< 逆向Y系数 */
    int sip_order;             /**< SIP阶数（最大5） */
    int sip_valid;             /**< 系数有效标记 */
} IRSipCoefficients;

/**
 * @brief 迭代精化结果
 */
typedef struct {
    double final_ra;           /**< 精确中心RA（度） */
    double final_dec;          /**< 精确中心Dec（度） */
    double final_rotation;     /**< 精确旋转角（度） */
    double final_scale;        /**< 精确像素尺度（角秒/像素） */
    double dist_a0, dist_a1, dist_a2, dist_a3, dist_a4, dist_a5;  /**< X畸变系数 */
    double dist_b0, dist_b1, dist_b2, dist_b3, dist_b4, dist_b5;  /**< Y畸变系数 */
    int distortion_valid;      /**< 畸变有效标记 */
    int matched_count;         /**< 匹配星对数 */
    double rms_x;              /**< X方向RMS（像素） */
    double rms_y;              /**< Y方向RMS（像素） */
    double rms_total;          /**< 总RMS（像素） */
    double rms_arcsec;         /**< 总RMS（角秒） */
    int iteration_count;       /**< 迭代次数 */
    int triangle_matches;      /**< 三角形匹配数 */
    int *img_indices;          /**< 匹配的图像星索引 */
    int *cat_indices;          /**< 匹配的星表星索引 */
    double *residual_x;        /**< X残差数组 */
    double *residual_y;        /**< Y残差数组 */
    IRSipCoefficients sip;     /**< SIP系数 */
    double cd[2][2];           /**< CD矩阵 */
    double crpix[2];           /**< CRPIX */
    double crval[2];           /**< CRVAL */
} IRRefineResult;

/**
 * @brief 迭代精化配置参数
 */
typedef struct {
    int max_stars_triangle;       /**< 构建三角形的最大星数 */
    double tri_ratio_radius;      /**< 三角形特征匹配半径 */
    double tri_min_area;          /**< 最小三角形面积（px²） */
    double tri_max_ba_ratio;      /**< 最大ba比（剔除扁平三角形） */
    double tri_equilateral_thresh; /**< 等边三角形阈值 */
    int grid_size;                /**< 异常检测网格数 */
    double outlier_angle_thresh;  /**< 方向异常阈值（弧度） */
    double outlier_mag_ratio;     /**< 大小异常比例 */
    int max_iterations;           /**< 最大迭代次数 */
    double converge_thresh;       /**< 收敛阈值 */
    double match_threshold;       /**< 匹配阈值（像素） */
    int sip_order;                /**< SIP阶数 */
} IRConfig;

/**
 * @brief 执行迭代精化
 * 
 * @param img_stars 图像星点数据
 * @param cat_stars 星表星点数据
 * @param init_transform 第一步输出的初始变换
 * @param config 配置参数
 * @param out_result 输出结果
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_iterative_refine(
    const IRImageStars *img_stars,
    const IRCatalogStars *cat_stars,
    const IRInitialTransform *init_transform,
    const IRConfig *config,
    IRRefineResult *out_result);

/**
 * @brief 释放迭代精化结果
 * @param result 结果指针
 */
IR_EXPORT void psm_free_refine_result(IRRefineResult *result);

/**
 * @brief 二分法迭代极限星等
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_bisection_mag_limit(
    void *gaia_client,
    double center_ra, double center_dec,
    double radius_deg, int target_count,
    double mag_low, double mag_high, double tolerance,
    double *out_mag, int *out_count);

/**
 * @brief 矩形FOV裁剪
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_rect_fov_filter(
    const double *cat_x, const double *cat_y, int cat_count,
    double half_w, double half_h,
    int **out_indices, int *out_count);

/**
 * @brief 构建剪枝后的三角形
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_build_triangles_pruned(
    const double *x, const double *y, int n,
    int max_stars, double min_area, double max_ba_ratio,
    double equilateral_thresh,
    void **out_triangles, int *out_count);

/**
 * @brief 三角形局部匹配
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_triangle_match_local(
    const void *img_tris, int n_img_tris,
    const void *cat_tris, int n_cat_tris,
    double ratio_radius, double centroid_threshold,
    int **out_pairs, int *out_pair_count);

/**
 * @brief 分区域异常值过滤
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_regional_outlier_filter(
    const double *dx, const double *dy, int n,
    const double *cx, const double *cy,
    int grid_size, double angle_thresh, double mag_ratio,
    int **out_mask, int *out_kept);

/**
 * @brief 拟合畸变模型
 * @return 0成功，非0失败
 */
IR_EXPORT int psm_fit_distortion_model(
    const double *x, const double *y,
    const double *dx, const double *dy, int n,
    double *out_dist_a, double *out_dist_b);

#ifdef __cplusplus
}
#endif

#endif

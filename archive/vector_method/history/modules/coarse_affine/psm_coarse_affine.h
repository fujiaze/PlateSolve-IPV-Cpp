#ifndef PSM_COARSE_AFFINE_H
#define PSM_COARSE_AFFINE_H

#include "../common/psm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

PSM_EXPORT int psm_triangle_build(const double *x, const double *y, int n,
    int nbright, PSMTriangle **out_tris, int *out_count);

PSM_EXPORT int psm_triangle_match(const PSMTriangle *tris_a, int na,
    const PSMTriangle *tris_b, int nb, double radius,
    double min_scale, double max_scale,
    PSMStarPair **out_pairs, int *out_pair_count);

PSM_EXPORT int psm_affine_compute(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst,
    int n, PSMAffine *out_affine);

PSM_EXPORT void psm_affine_apply(const PSMAffine *affine,
    double x, double y, double *out_x, double *out_y);

PSM_EXPORT void psm_free_triangles(PSMTriangle *tris);

PSM_EXPORT void psm_free_pairs(PSMStarPair *pairs);

/* ─── 新增：中心扩散搜索+三角匹配 ───
 * 从初始中心开始，按多级网格向外扩散搜索，
 * 找到与图像星点最佳匹配的天球中心位置。
 * 
 * 参数:
 *   det_x, det_y, ndet  — 检测星点坐标（已居中到图像中心）
 *   cat_ra, cat_dec, ncat — Gaia星表 RA/Dec (度)
 *   init_ra, init_dec    — 初始中心猜测 (度)
 *   nbright              — 三角匹配用亮星数 (建议60)
 *   match_radius         — 边长比匹配半径 (建议0.002)
 *   min_scale, max_scale — 尺度范围限制
 *   deg_to_px            — 度→像素转换因子 (3600/scale_arcsec_px)
 *   half_w, half_h       — 图像半宽/半高
 *   out_center_ra, out_center_dec — 输出最佳中心(度)
 *   out_pairs            — 输出匹配星对 (内部malloc，调用者psm_free_pairs释放)
 *   out_pair_count       — 输出匹配对数
 * 
 * 返回: PSM_OK 或错误码
 */
PSM_EXPORT int psm_search_center_and_match(
    const double *det_x, const double *det_y, int ndet,
    const double *cat_ra, const double *cat_dec, int ncat,
    double init_ra, double init_dec,
    int nbright, double match_radius,
    double min_scale, double max_scale,
    double deg_to_px, double half_w, double half_h,
    double *out_center_ra, double *out_center_dec,
    PSMStarPair **out_pairs, int *out_pair_count);

#ifdef __cplusplus
}
#endif

#endif

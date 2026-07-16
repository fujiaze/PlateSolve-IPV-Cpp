#ifndef PSM_AFFINE_DISTORTION_H
#define PSM_AFFINE_DISTORTION_H

#include "../common/psm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

PSM_EXPORT int psm_ransac_filter(const double *img_x, const double *img_y,
    const double *cat_x, const double *cat_y, int n,
    double threshold_px, int max_trials,
    PSMAffine *out_affine, int **out_inlier_mask, int *out_inlier_count);

PSM_EXPORT int psm_affine_compute(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst, int n, PSMAffine *out_affine);

PSM_EXPORT void psm_affine_apply(const PSMAffine *affine,
    double x, double y, double *out_x, double *out_y);

PSM_EXPORT int psm_distortion_compute(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst, int n, PSMDistortion *out_dist);

PSM_EXPORT void psm_distortion_apply(const PSMDistortion *dist,
    double x, double y, double *out_x, double *out_y);

PSM_EXPORT int psm_check_affine(const PSMAffine *affine,
    double scale_min, double scale_max);

PSM_EXPORT void psm_free_mask(int *mask);

#ifdef __cplusplus
}
#endif

#endif

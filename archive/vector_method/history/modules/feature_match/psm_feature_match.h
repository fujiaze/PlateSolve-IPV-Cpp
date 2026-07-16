#ifndef PSM_FEATURE_MATCH_H
#define PSM_FEATURE_MATCH_H

#include "../common/psm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double x, y;
    double flux;
    int orig_idx;
} PSMFeatureStar;

typedef struct {
    int img_idx;
    int cat_idx;
    double dist;
} PSMMatchPair;

typedef struct {
    PSMMatchPair *pairs;
    int pair_count;
    double mean_dist;
    double a0, a1, a2;
    double b0, b1, b2;
    int flip_x;
    int flip_y;
} PSMDirectAlignResult;

typedef struct {
    int n_bright;
    double max_dist;
    int max_iterations;
} PSMDirectAlignConfig;

PSM_EXPORT int psm_direct_align(
    const double *img_x, const double *img_y, int n_img,
    const double *cat_x, const double *cat_y, int n_cat,
    const PSMDirectAlignConfig *config,
    PSMDirectAlignResult *out_result);

PSM_EXPORT void psm_free_direct_align_result(PSMDirectAlignResult *result);

#ifdef __cplusplus
}
#endif

#endif

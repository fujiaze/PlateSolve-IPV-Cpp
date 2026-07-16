#ifndef PSM_GRID_MATCH_H
#define PSM_GRID_MATCH_H

#include "../common/psm_common.h"

#ifdef _WIN32
#define GM_EXPORT __declspec(dllexport)
#else
#define GM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GM_DEFAULT_GRID_SIZE        50
#define GM_DEFAULT_MAX_CANDIDATES   10
#define GM_DEFAULT_MATCH_TOLERANCE  5.0
#define GM_DEFAULT_RANSAC_ITER      500
#define GM_DEFAULT_RANSAC_SIGMA     3.0
#define GM_DEFAULT_SIP_ORDER        5
#define GM_DEFAULT_CENTROID_RADIUS  20.0
#define GM_DEFAULT_RATIO_THRESHOLD  0.07
#define GM_DEFAULT_N_IMG_BRIGHT     1000
#define GM_DEFAULT_N_CAT_BRIGHT     1500

typedef struct {
    int grid_row;
    int grid_col;
    int img_star_idx;
    int cat_star_idx;
    double img_x;
    double img_y;
    double cat_x;
    double cat_y;
    double residual_x;
    double residual_y;
    int valid;
} GMControlPoint;

typedef struct {
    int grid_size;
    int max_cat_candidates;
    double match_tolerance;
    int max_ransac_iter;
    double ransac_sigma;
    int sip_order;
    double centroid_radius;
    double ratio_threshold;
    int n_img_bright;
    int n_cat_bright;
} GMConfig;

typedef struct {
    int n_control_points;
    int n_grids_matched;
    int n_grids_total;
    double rms_x;
    double rms_y;
    double rms_total;
    double rms_arcsec;
    int n_ransac_removed;
    GMControlPoint *control_points;
    double sip_A[6][6];
    double sip_B[6][6];
    double sip_AP[6][6];
    double sip_BP[6][6];
    int sip_order;
    int sip_valid;
    double cd[2][2];
    double crpix[2];
    double crval[2];
} GMResult;

typedef struct {
    const double *img_x;
    const double *img_y;
    const double *img_flux;
    const double *img_mag;
    int img_count;
} GMImageStars;

typedef struct {
    const double *cat_ra;
    const double *cat_dec;
    const double *cat_mag;
    int cat_count;
} GMCatalogStars;

typedef struct {
    double crval1;
    double crval2;
    double crpix1;
    double crpix2;
    double cd1_1;
    double cd1_2;
    double cd2_1;
    double cd2_2;
    double scale_arcsec_px;
    int img_width;
    int img_height;
} GMInitialTransform;

GM_EXPORT int psm_grid_match_perform(
    const GMImageStars *img_stars,
    const GMCatalogStars *cat_stars,
    const GMInitialTransform *init_transform,
    const GMConfig *config,
    GMResult *out_result);

GM_EXPORT void psm_grid_match_free_result(GMResult *result);

GM_EXPORT int psm_grid_ransac_filter(
    GMControlPoint *points,
    int n_points,
    int max_iter,
    double sigma_thresh,
    int *out_n_valid);

GM_EXPORT int psm_grid_fit_affine(
    const GMControlPoint *points,
    int n_points,
    double *out_a0,
    double *out_a1,
    double *out_a2,
    double *out_b0,
    double *out_b1,
    double *out_b2);

#ifdef __cplusplus
}
#endif

#endif

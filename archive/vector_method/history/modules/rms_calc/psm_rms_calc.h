#ifndef PSM_RMS_CALC_H
#define PSM_RMS_CALC_H

#include "../common/psm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

PSM_EXPORT int psm_sigma_clip_2d(const double *res_x, const double *res_y, int count,
    double sigma, int *mask);

PSM_EXPORT int psm_rms_compute_affine(const double *img_x, const double *img_y,
    const double *cat_x, const double *cat_y, int n,
    const PSMAffine *affine, double *rms_x, double *rms_y);

PSM_EXPORT int psm_rms_compute_distortion(const double *cat_x, const double *cat_y,
    const double *img_x, const double *img_y, int n,
    const PSMDistortion *dist, double *rms_x, double *rms_y);

PSM_EXPORT void psm_residual_stats(const double *residuals, int n,
    double *out_mean, double *out_median, double *out_mad, double *out_rms);

PSM_EXPORT int psm_rms_clipped(const double *img_x, const double *img_y,
    const double *cat_x, const double *cat_y, int n,
    const PSMAffine *affine, double sigma,
    double *rms_x, double *rms_y, int *out_kept);

typedef struct {
    double ra_center, dec_center;
    double scale_arcsec_px;
    double rotation_deg;
    const PSMAffine *affine;
    const PSMDistortion *distortion;
    double query_radius_deg;
    double mag_limit;
    double sigma_clip;
} PSMModelEvalInput;

typedef struct {
    int total_gaia_stars;
    int matched_count;
    int clipped_count;
    double rms_x_px;
    double rms_y_px;
    double rms_total_px;
    double rms_arcsec;
    double mean_residual_px;
    double median_residual_px;
    double mad_px;
    double score;
    double score_density;
    double score_precision;
    double score_coverage;
} PSMModelEvalResult;

PSM_EXPORT int psm_rms_evaluate_model(
    void *gaia_client,
    const PSMModelEvalInput *model,
    const double *img_x, const double *img_y, int img_count,
    PSMModelEvalResult *out_result);

PSM_EXPORT void psm_free_model_eval_result(PSMModelEvalResult *result);

#ifdef __cplusplus
}
#endif

#endif

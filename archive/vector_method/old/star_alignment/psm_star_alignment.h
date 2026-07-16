#ifndef PSM_STAR_ALIGNMENT_H
#define PSM_STAR_ALIGNMENT_H

#ifdef _WIN32
#define PSM_EXPORT __declspec(dllexport)
#else
#define PSM_EXPORT __attribute__((visibility("default")))
#endif

typedef enum {
    PSM_OK = 0,
    PSM_ERR_INVALID_PARAM = -1,
    PSM_ERR_NO_MATCH = -2,
    PSM_ERR_TRANS_FAILED = -3
} PSMError;

typedef struct {
    double a0, a1, a2;
    double b0, b1, b2;
    int matched_count;
    double rms_arcsec;
    double center_ra;
    double center_dec;
    int *img_indices;
    int *cat_indices;
} PSMStarAlignmentResult;

#ifdef __cplusplus
extern "C" {
#endif

PSM_EXPORT int psm_star_alignment(
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int n_img,
    const double *cat_x, const double *cat_y, const double *cat_mag, int n_cat,
    double scale_arcsec_px,
    double percent_scale_range,
    double center_ra,
    double center_dec,
    const double *cat_ra,
    const double *cat_dec,
    double cd1_1, double cd1_2, double cd2_1, double cd2_2,
    PSMStarAlignmentResult *result);

PSM_EXPORT void psm_free_result(PSMStarAlignmentResult *result);

#ifdef __cplusplus
}
#endif

#endif

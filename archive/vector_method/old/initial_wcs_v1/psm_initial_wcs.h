#ifndef PSM_INITIAL_WCS_H
#define PSM_INITIAL_WCS_H

#include "../common/psm_common.h"

#ifdef _WIN32
#define PSM_IW_EXPORT __declspec(dllexport)
#else
#define PSM_IW_EXPORT __attribute__((visibility("default")))
#endif

typedef struct {
    double center_ra;
    double center_dec;
    double rotation_deg;
    double scale_arcsec_px;
    int flip_mode;
    double affine[6];
    int matched_count;
    double rms_px;
    double rms_arcsec;
} InitialWCSResult;

#ifdef __cplusplus
extern "C" {
#endif

PSM_IW_EXPORT int psm_initial_wcs_solve(
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int n_stars,
    double center_ra, double center_dec,
    double focal_length_mm, double pixel_size_um,
    int width, int height,
    const char *gaia_db_path, int db_type,
    InitialWCSResult *result);

PSM_IW_EXPORT void psm_initial_wcs_free_result(InitialWCSResult *result);

#ifdef __cplusplus
}
#endif

#endif

#ifndef PSM_ITERATIVE_H
#define PSM_ITERATIVE_H

#include "../common/psm_common.h"

#ifdef __cplusplus
extern "C" {
#endif

PSM_EXPORT void psm_sky_to_plane(double ra, double dec, double ra0, double dec0,
    double *x, double *y);

PSM_EXPORT void psm_plane_to_sky(double x, double y, double ra0, double dec0,
    double *ra, double *dec);

PSM_EXPORT int psm_project_stars(const double *ra_arr, const double *dec_arr, int count,
    double ra0, double dec0, double **out_x, double **out_y);

PSM_EXPORT void psm_free_projected(double *arr);

PSM_EXPORT int psm_match_nearest(const double *det_x, const double *det_y, int ndet,
    const double *cat_x, const double *cat_y, int ncat, double match_radius,
    int **out_match_idx, double **out_match_dist);

PSM_EXPORT int psm_iterate_center(const double *det_x, const double *det_y, int ndet,
    const double *cat_ra, const double *cat_dec, int ncat,
    double *io_center_ra, double *io_center_dec,
    double scale_arcsec_px, int half_w, int half_h,
    PSMAffine *io_affine, int max_iter);

PSM_EXPORT void psm_free_match(int *match_idx, double *match_dist);

#ifdef __cplusplus
}
#endif

#endif

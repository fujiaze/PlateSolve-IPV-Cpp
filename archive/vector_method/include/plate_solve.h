#ifndef PLATE_SOLVE_H
#define PLATE_SOLVE_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define PSOLVE_EXPORT __declspec(dllexport)
#else
#define PSOLVE_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PSOLVE_OK                0
#define PSOLVE_ERR_NO_MATCH      1
#define PSOLVE_ERR_NOT_ENOUGH    2
#define PSOLVE_ERR_INVALID_TRANS 3
#define PSOLVE_ERR_NO_FOV        4
#define PSOLVE_ERR_PROJ          5
#define PSOLVE_ERR_CONVERGE      6
#define PSOLVE_ERR_CANCELLED     7
#define PSOLVE_ERR_INTERNAL      8

typedef enum {
    PSOLVE_DB_AUTO = 0,
    PSOLVE_DB_DR3 = 1,
    PSOLVE_DB_DR3SP = 2
} PSolveDbType;

typedef enum {
    PSOLVE_FLIP_NONE = 0,
    PSOLVE_FLIP_X = 1,
    PSOLVE_FLIP_Y = 2,
    PSOLVE_FLIP_XY = 3
} PSolveFlipMode;

typedef struct {
    double a0, a1, a2;
    double b0, b1, b2;
} PSolveAffine;

typedef struct {
    double a0, a1, a2, a3, a4, a5;
    double b0, b1, b2, b3, b4, b5;
} PSolveDistortion;

typedef struct {
    double crpix1, crpix2;
    double crval1, crval2;
    double cd1_1, cd1_2, cd2_1, cd2_2;
    double cdelt1, cdelt2;
    char ctype1[32];
    char ctype2[32];
    char radesys[32];
    double equinox;
} PSolveWCS;

typedef struct {
    double A[6][6];
    double B[6][6];
    double AP[6][6];
    double BP[6][6];
    int order;
    int valid;
} PSolveSIPCoeffs;

typedef struct {
    double img_x;
    double img_y;
    double cat_ra;
    double cat_dec;
    double cat_mag;
    double residual_x;
    double residual_y;
} PSolveMatchedStar;

typedef struct {
    double focal_length_mm;
    double pixel_size_um;
    double center_ra;
    double center_dec;
    int width;
    int height;
    int has_coords;
    double exposure_time_s;
    double scale_arcsec_px;
} PSolveImageData;

typedef struct {
    int use_saturated_priority;
    int n_img_bright;
    int n_cat_bright;
    double max_match_dist_px;
    int max_iterations;
    double match_threshold;
    int sip_order;
    double converge_thresh;
} PSolveConfig;

typedef struct {
    double scale_arcsec_px;
    double fov_w_arcmin;
    double fov_h_arcmin;
    double fov_radius_deg;
    double limit_mag;
    int gaia_star_count;
    int detected_star_count;
    PSolveAffine affine;
    double rms_x;
    double rms_y;
    double rms_total;
    int iteration_count;
    int matched_count;
    PSolveMatchedStar *matched_stars;
} PSolveCoarseResult;

typedef struct {
    PSolveWCS wcs;
    PSolveSIPCoeffs sip;
    double rms_x;
    double rms_y;
    double rms_total;
    double rms_arcsec;
    int matched_count;
    int subdomain_count;
    double *residual_grid_x;
    double *residual_grid_y;
    int grid_w;
    int grid_h;
} PSolveFineResult;

typedef struct {
    double center_ra;
    double center_dec;
    double rotation_deg;
    double scale_arcsec_px;
    int flip_mode;
    int matched_count;
    double rms_px;
    double step1_time_sec;
    double step2_time_sec;
    PSolveWCS wcs;
    PSolveSIPCoeffs sip;
    int sip_valid;
} PSolveResult;

typedef struct PSolveHandle_s *PSolveHandle;

PSOLVE_EXPORT PSolveHandle psolve_create(const char *gaia_data_dir);
PSOLVE_EXPORT PSolveHandle psolve_create_ex(const char *gaia_data_dir, PSolveDbType db_type);
PSOLVE_EXPORT void psolve_destroy(PSolveHandle handle);
PSOLVE_EXPORT int psolve_get_db_type(PSolveHandle handle);

PSOLVE_EXPORT int psolve_solve(
    PSolveHandle handle,
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int img_count, int n_saturated,
    const PSolveImageData *img_data,
    const PSolveConfig *config,
    PSolveResult *result);

PSOLVE_EXPORT int psolve_solve_with_image(
    PSolveHandle handle,
    const uint16_t *image, int width, int height,
    const PSolveImageData *img_data,
    const PSolveConfig *config,
    PSolveResult *result);

PSOLVE_EXPORT int psolve_solve_with_file(
    PSolveHandle handle,
    const char *file_path,
    const PSolveConfig *config,
    PSolveResult *result);

PSOLVE_EXPORT void psolve_free_result(PSolveResult *result);

PSOLVE_EXPORT int psolve_coarse(
    PSolveHandle handle,
    const uint16_t *image, int width, int height,
    const PSolveImageData *img_data,
    const double *det_x, const double *det_y, int det_count,
    PSolveCoarseResult *result);

PSOLVE_EXPORT int psolve_fine(
    PSolveHandle handle,
    const uint16_t *image, int width, int height,
    const PSolveCoarseResult *coarse,
    const PSolveImageData *img_data,
    PSolveFineResult *result);

PSOLVE_EXPORT void psolve_free_coarse_result(PSolveCoarseResult *result);
PSOLVE_EXPORT void psolve_free_fine_result(PSolveFineResult *result);

PSOLVE_EXPORT int psolve_get_matched_stars(
    PSolveHandle handle,
    double **out_x, double **out_y, int *out_count);

PSOLVE_EXPORT int psolve_get_wcs(
    PSolveHandle handle,
    PSolveWCS *out_wcs);

#ifdef __cplusplus
}
#endif

#endif

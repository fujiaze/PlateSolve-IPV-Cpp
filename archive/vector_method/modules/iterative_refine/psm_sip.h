#ifndef PSM_SIP_H
#define PSM_SIP_H

#include "../common/psm_common.h"

#ifdef _WIN32
#define SIP_EXPORT __declspec(dllexport)
#else
#define SIP_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SIP_MAX_ORDER 5
#define SIP_GRID_SIZE 7

typedef struct {
    int order;
    int match_count;
    double rms;
    double dummy;
    double x00, x10, x01, x20, x11, x02, x30, x21, x12, x03, x40, x31, x22, x13, x04, x50, x41, x32, x23, x14, x05;
    double y00, y10, y01, y20, y11, y02, y30, y21, y12, y03, y40, y31, y22, y13, y04, y50, y41, y32, y23, y14, y05;
} SIP_Transform;

typedef struct {
    double x, y;
} SIP_Point;

typedef struct {
    double A[6][6];
    double B[6][6];
    double AP[6][6];
    double BP[6][6];
    int order;
    int valid;
} SIP_Coeffs;

typedef struct {
    double crpix[2];
    double crval[2];
    double cd[2][2];
    SIP_Coeffs sip;
    int has_sip;
} SIP_WCS;

SIP_EXPORT void sip_trans_init(SIP_Transform *t, int order);

SIP_EXPORT void sip_trans_copy(SIP_Transform *dst, const SIP_Transform *src);

SIP_EXPORT void sip_trans_apply(const SIP_Transform *t, double x, double y, double *out_x, double *out_y);

SIP_EXPORT void sip_trans_apply_array(const SIP_Transform *t, SIP_Point *points, int n);

SIP_EXPORT double sip_trans_det(const SIP_Transform *t);

SIP_EXPORT void sip_trans_get_cd(const SIP_Transform *t, double cd[2][2]);

SIP_EXPORT void sip_trans_invert_cd(const SIP_Transform *t, double cd_inv[2][2]);

SIP_EXPORT SIP_Point *sip_create_grid(int rx, int ry, int n_points);

SIP_EXPORT void sip_free_grid(SIP_Point *grid);

SIP_EXPORT int sip_trans_fit(SIP_Transform *t, const SIP_Point *src, const SIP_Point *dst, int n,
                             int max_iter, double tolerance);

SIP_EXPORT int sip_coeffs_compute(const SIP_Transform *t, int rx, int ry, SIP_Coeffs *sip);

SIP_EXPORT void sip_apply_forward(const SIP_Coeffs *sip, double x, double y, double *out_x, double *out_y);

SIP_EXPORT void sip_apply_inverse(const SIP_Coeffs *sip, double u, double v, double *out_x, double *out_y);

SIP_EXPORT int sip_fit_refine(
    const double *img_x, const double *img_y, int n_img,
    const double *cat_x, const double *cat_y, int n_cat,
    double center_x, double center_y,
    int img_w, int img_h,
    int sip_order,
    SIP_Transform *out_trans,
    SIP_Coeffs *out_sip,
    double *out_rms);

#ifdef __cplusplus
}
#endif

#endif

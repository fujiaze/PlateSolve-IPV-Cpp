#include "psolve_projection.h"
#include <cmath>
#include <cstdlib>

#define PSOLVE_PI 3.14159265358979323846
#define PSOLVE_DEG2RAD (PSOLVE_PI / 180.0)
#define PSOLVE_RAD2DEG (180.0 / PSOLVE_PI)

void psolve_sky_to_plane(double ra, double dec, double ra0, double dec0,
                          double *x, double *y) {
    double ra_rad = ra * PSOLVE_DEG2RAD;
    double dec_rad = dec * PSOLVE_DEG2RAD;
    double ra0_rad = ra0 * PSOLVE_DEG2RAD;
    double dec0_rad = dec0 * PSOLVE_DEG2RAD;

    double dra = ra_rad - ra0_rad;
    double sin_dec0 = sin(dec0_rad);
    double cos_dec0 = cos(dec0_rad);
    double sin_dec = sin(dec_rad);
    double cos_dec = cos(dec_rad);
    double cos_dra = cos(dra);
    double sin_dra = sin(dra);

    double cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_dra;

    if (cos_c < 1e-10) {
        *x = 1e30;
        *y = 1e30;
        return;
    }

    *x = cos_dec * sin_dra / cos_c * PSOLVE_RAD2DEG;
    *y = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / cos_c * PSOLVE_RAD2DEG;
}

void psolve_plane_to_sky(double x, double y, double ra0, double dec0,
                          double *ra, double *dec) {
    double x_rad = x * PSOLVE_DEG2RAD;
    double y_rad = y * PSOLVE_DEG2RAD;
    double ra0_rad = ra0 * PSOLVE_DEG2RAD;
    double dec0_rad = dec0 * PSOLVE_DEG2RAD;

    double rho = sqrt(x_rad * x_rad + y_rad * y_rad);

    if (rho < 1e-10) {
        *ra = ra0;
        *dec = dec0;
        return;
    }

    double c = atan(rho);
    double sin_c = sin(c);
    double cos_c = cos(c);
    double sin_dec0 = sin(dec0_rad);
    double cos_dec0 = cos(dec0_rad);

    double ra_rad = ra0_rad + atan2(x_rad * sin_c,
                                     rho * cos_dec0 * cos_c - y_rad * sin_dec0 * sin_c);
    double dec_rad = asin(cos_c * sin_dec0 + y_rad * sin_c * cos_dec0 / rho);

    *ra = ra_rad * PSOLVE_RAD2DEG;
    *dec = dec_rad * PSOLVE_RAD2DEG;

    while (*ra < 0.0) *ra += 360.0;
    while (*ra >= 360.0) *ra -= 360.0;
}

void psolve_project_stars(double *ra_arr, double *dec_arr, int count,
                           double ra0, double dec0,
                           double **out_x, double **out_y) {
    *out_x = (double *)malloc(count * sizeof(double));
    *out_y = (double *)malloc(count * sizeof(double));

    for (int i = 0; i < count; i++) {
        psolve_sky_to_plane(ra_arr[i], dec_arr[i], ra0, dec0,
                            &(*out_x)[i], &(*out_y)[i]);
    }
}

void psolve_free_projected(double *arr) {
    if (arr) {
        free(arr);
    }
}

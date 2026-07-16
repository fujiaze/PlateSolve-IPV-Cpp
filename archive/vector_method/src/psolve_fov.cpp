#include "psolve_fov.h"
#include "psolve_log.h"
#include "../../gaia_xpsd_client/src/gaia_client.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PSOLVE_DEG2RAD (M_PI / 180.0)
#define PSOLVE_RAD2DEG (180.0 / M_PI)

double psolve_compute_scale(double focal_mm, double pixel_um) {
    if (focal_mm <= 0.0 || pixel_um <= 0.0)
        return 0.0;
    return 206.265 * pixel_um / focal_mm;
}

double psolve_compute_fov_w(double scale_arcsec, int width_px) {
    return scale_arcsec * width_px / 60.0;
}

double psolve_compute_fov_h(double scale_arcsec, int height_px) {
    return scale_arcsec * height_px / 60.0;
}

double psolve_compute_fov_radius(double scale_arcsec, int width_px, int height_px) {
    double sw = scale_arcsec * (double)width_px;
    double sh = scale_arcsec * (double)height_px;
    return sqrt(sw * sw + sh * sh) / 7200.0;
}

double psolve_compute_mag_limit(double ra, double dec, double fov_deg, int nstars) {
    double l0 = 122.9320 * PSOLVE_DEG2RAD;
    double a0 = 192.8595 * PSOLVE_DEG2RAD;
    double d0 = 27.1284 * PSOLVE_DEG2RAD;
    double ra_rad = ra * PSOLVE_DEG2RAD;
    double dec_rad = dec * PSOLVE_DEG2RAD;

    double ml = (l0 - atan2(cos(dec_rad) * sin(ra_rad - a0),
                            sin(dec_rad) * cos(d0) - cos(dec_rad) * sin(d0) * cos(ra_rad - a0))) * PSOLVE_RAD2DEG;
    double mb = asin(sin(dec_rad) * sin(d0) + cos(dec_rad) * cos(d0) * cos(ra_rad - a0)) * PSOLVE_RAD2DEG;

    if (ml > 180.0)
        ml -= 360.0;

    double S = 2.0 * (1.0 - cos(0.5 * fov_deg * PSOLVE_DEG2RAD)) * 180.0 * 180.0 / M_PI;
    double m0 = 11.68 + 2.66 * sin(fabs(mb) * PSOLVE_DEG2RAD);
    double a = 2.36 + (fabs(ml) - 90.0) * 0.0073 * (fabs(ml) < 90.0);
    double b = 0.88 - (fabs(ml) - 90.0) * 0.0065 * (fabs(ml) < 90.0);
    double s = a + b * sin(fabs(mb) * PSOLVE_DEG2RAD);
    double limit = m0 + s * (log10((double)nstars / S) - 2.0);

    if (limit < 7.0)
        limit = 7.0;
    return limit;
}

double psolve_iterate_mag_limit(void *gaia_client, double ra, double dec, double radius_deg,
                                 int target_count, double *out_mag, int *out_count) {
    double mag = psolve_compute_mag_limit(ra, dec, radius_deg * 2.0, target_count);
    int count = 0;

    PSLOG_I("psolve_iterate_mag_limit: initial mag=%.2f ra=%.4f dec=%.4f radius=%.4f target=%d",
            mag, ra, dec, radius_deg, target_count);

    for (int i = 0; i < 5; i++) {
        double *qry_ra = NULL;
        double *qry_dec = NULL;
        float *qry_mag = NULL;

        int ret = gaia_client_cone_search_for_solver(
            (GaiaClient *)gaia_client,
            ra, dec, radius_deg,
            mag,
            &qry_ra, &qry_dec, &qry_mag,
            &count);

        if (qry_ra) free(qry_ra);
        if (qry_dec) free(qry_dec);
        if (qry_mag) free(qry_mag);

        if (ret != 0) {
            PSLOG_W("psolve_iterate_mag_limit: gaia query failed at iter %d, ret=%d", i, ret);
            break;
        }

        PSLOG_I("psolve_iterate_mag_limit: iter %d mag=%.2f count=%d target=%d",
                i, mag, count, target_count);

        if (target_count <= 0) break;

        double ratio = (double)count / (double)target_count;
        if (ratio > 1.2) {
            mag -= 0.5;
        } else if (ratio < 0.8) {
            mag += 0.5;
        } else {
            break;
        }
    }

    if (out_mag) *out_mag = mag;
    if (out_count) *out_count = count;

    PSLOG_I("psolve_iterate_mag_limit: final mag=%.2f count=%d", mag, count);

    return mag;
}

double psolve_bisection_mag_limit(void *gaia_client, double ra, double dec, double radius_deg,
                                   int target_count, double *out_mag, int *out_count) {
    double mag_low = 6.0;
    double mag_high = 22.0;
    double best_mag = 15.0;
    int best_count = 0;
    const double tolerance = 0.05;
    const int max_iterations = 10;

    PSLOG_I("psolve_bisection_mag_limit: ra=%.4f dec=%.4f radius=%.4f target=%d",
            ra, dec, radius_deg, target_count);

    for (int i = 0; i < max_iterations; i++) {
        double mag_mid = (mag_low + mag_high) / 2.0;
        
        double *qry_ra = NULL;
        double *qry_dec = NULL;
        float *qry_mag = NULL;
        int count = 0;

        int ret = gaia_client_cone_search_for_solver(
            (GaiaClient *)gaia_client,
            ra, dec, radius_deg,
            mag_mid,
            &qry_ra, &qry_dec, &qry_mag,
            &count);

        if (qry_ra) free(qry_ra);
        if (qry_dec) free(qry_dec);
        if (qry_mag) free(qry_mag);

        if (ret != 0) {
            PSLOG_W("psolve_bisection_mag_limit: gaia query failed at iter %d", i);
            break;
        }

        PSLOG_I("psolve_bisection_mag_limit: iter %d mag=%.3f count=%d target=%d",
                i, mag_mid, count, target_count);

        best_mag = mag_mid;
        best_count = count;

        if (count == target_count) {
            break;
        } else if (count > target_count) {
            mag_high = mag_mid;
        } else {
            mag_low = mag_mid;
        }

        if (mag_high - mag_low < tolerance) {
            break;
        }
    }

    if (out_mag) *out_mag = best_mag;
    if (out_count) *out_count = best_count;

    PSLOG_I("psolve_bisection_mag_limit: final mag=%.3f count=%d", best_mag, best_count);

    return best_mag;
}

double psolve_estimate_mag_limit(double focal_length_mm, double exposure_time_s) {
    if (focal_length_mm <= 0.0 || exposure_time_s <= 0.0) {
        PSLOG_W("psolve_estimate_mag_limit: invalid focal=%.2f exposure=%.2f",
                focal_length_mm, exposure_time_s);
        return 15.0;
    }

    double mag_limit = 6.0 + 1.5 * log10(focal_length_mm) + 2.0 * log10(exposure_time_s);

    if (mag_limit < 8.0) mag_limit = 8.0;
    if (mag_limit > 20.0) mag_limit = 20.0;

    PSLOG_I("psolve_estimate_mag_limit: focal=%.1fmm exposure=%.1fs mag_limit=%.2f",
            focal_length_mm, exposure_time_s, mag_limit);

    return mag_limit;
}

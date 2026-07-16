#ifndef PSOLVE_FOV_H
#define PSOLVE_FOV_H

double psolve_compute_scale(double focal_mm, double pixel_um);
double psolve_compute_fov_w(double scale_arcsec, int width_px);
double psolve_compute_fov_h(double scale_arcsec, int height_px);
double psolve_compute_fov_radius(double scale_arcsec, int width_px, int height_px);
double psolve_compute_mag_limit(double ra, double dec, double fov_deg, int nstars);
double psolve_iterate_mag_limit(void *gaia_client, double ra, double dec, double radius_deg,
                                 int target_count, double *out_mag, int *out_count);
double psolve_bisection_mag_limit(void *gaia_client, double ra, double dec, double radius_deg,
                                   int target_count, double *out_mag, int *out_count);
double psolve_estimate_mag_limit(double focal_length_mm, double exposure_time_s);

#endif

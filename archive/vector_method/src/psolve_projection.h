#ifndef PSOLVE_PROJECTION_H
#define PSOLVE_PROJECTION_H

void psolve_sky_to_plane(double ra, double dec, double ra0, double dec0,
                          double *x, double *y);
void psolve_plane_to_sky(double x, double y, double ra0, double dec0,
                          double *ra, double *dec);
void psolve_project_stars(double *ra_arr, double *dec_arr, int count,
                           double ra0, double dec0,
                           double **out_x, double **out_y);
void psolve_free_projected(double *arr);

#endif

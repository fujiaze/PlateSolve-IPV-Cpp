#ifndef PSOLVE_RANSAC_H
#define PSOLVE_RANSAC_H

#include "plate_solve.h"

int psolve_compute_affine(const double *x_src, const double *y_src,
                           const double *x_dst, const double *y_dst,
                           int n, PSolveAffine *affine);
int psolve_apply_affine(const PSolveAffine *affine, double x, double y,
                         double *out_x, double *out_y);
int psolve_ransac_filter(const double *img_x, const double *img_y,
                          const double *cat_x, const double *cat_y,
                          int n,
                          double threshold_px, int max_trials,
                          PSolveAffine *out_affine,
                          int **out_inlier_mask, int *out_inlier_count);
double psolve_compute_rms(const double *img_x, const double *img_y,
                           const double *cat_x, const double *cat_y,
                           int n, const PSolveAffine *affine,
                           double *rms_x, double *rms_y);
int psolve_check_affine(const PSolveAffine *affine, double scale_min, double scale_max);

int psolve_compute_distortion(const double *x_src, const double *y_src,
                               const double *x_dst, const double *y_dst,
                               int n, PSolveDistortion *dist);
int psolve_apply_distortion(const PSolveDistortion *dist, double x, double y,
                             double *out_x, double *out_y);
double psolve_compute_rms_distortion(const double *cat_x, const double *cat_y,
                                      const double *img_x, const double *img_y,
                                      int n, const PSolveDistortion *dist,
                                      double *rms_x, double *rms_y);
int psolve_sigma_clip_2d(const double *res_x, const double *res_y, int count,
                          double sigma, int *mask);

#endif

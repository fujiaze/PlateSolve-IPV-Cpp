#include "psolve_ransac.h"
#include "psolve_log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int gauss_solve_3x3(double A[3][3], double b[3], double x[3]) {
    double aug[3][4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) aug[i][j] = A[i][j];
        aug[i][3] = b[i];
    }
    for (int col = 0; col < 3; col++) {
        int max_row = col;
        double max_val = fabs(aug[col][col]);
        for (int row = col + 1; row < 3; row++) {
            double v = fabs(aug[row][col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-15) return -1;
        if (max_row != col) {
            for (int j = 0; j < 4; j++) {
                double tmp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }
        double pivot = aug[col][col];
        for (int j = col; j < 4; j++) aug[col][j] /= pivot;
        for (int row = 0; row < 3; row++) {
            if (row == col) continue;
            double factor = aug[row][col];
            for (int j = col; j < 4; j++) aug[row][j] -= factor * aug[col][j];
        }
    }
    for (int i = 0; i < 3; i++) x[i] = aug[i][3];
    return 0;
}

int psolve_compute_affine(const double *x_src, const double *y_src,
                           const double *x_dst, const double *y_dst,
                           int n, PSolveAffine *affine) {
    if (n < 3 || !x_src || !y_src || !x_dst || !y_dst || !affine) {
        PSLOG_E("psolve_compute_affine: invalid input (n=%d)", n);
        return PSOLVE_ERR_NOT_ENOUGH;
    }
    double s1 = 0, sx = 0, sy = 0, sxx = 0, sxy = 0, syy = 0;
    double sxd = 0, syd = 0, sxxd = 0, sxyd = 0, syxd = 0, syyd = 0;
    for (int i = 0; i < n; i++) {
        double xi = x_src[i], yi = y_src[i];
        double xd = x_dst[i], yd = y_dst[i];
        s1 += 1;
        sx += xi; sy += yi;
        sxx += xi * xi; sxy += xi * yi; syy += yi * yi;
        sxd += xd; syd += yd;
        sxxd += xi * xd; sxyd += yi * xd;
        syxd += xi * yd; syyd += yi * yd;
    }
    double ATA[3][3] = {
        {s1,  sx,  sy },
        {sx,  sxx, sxy},
        {sy,  sxy, syy}
    };
    double ATbx[3] = {sxd, sxxd, sxyd};
    double ATby[3] = {syd, syxd, syyd};
    double cx[3], cy[3];
    if (gauss_solve_3x3(ATA, ATbx, cx) != 0) {
        PSLOG_E("psolve_compute_affine: singular matrix for x");
        return PSOLVE_ERR_INVALID_TRANS;
    }
    if (gauss_solve_3x3(ATA, ATby, cy) != 0) {
        PSLOG_E("psolve_compute_affine: singular matrix for y");
        return PSOLVE_ERR_INVALID_TRANS;
    }
    affine->a0 = cx[0]; affine->a1 = cx[1]; affine->a2 = cx[2];
    affine->b0 = cy[0]; affine->b1 = cy[1]; affine->b2 = cy[2];
    PSLOG_D("psolve_compute_affine: a0=%.6f a1=%.6f a2=%.6f b0=%.6f b1=%.6f b2=%.6f n=%d",
            affine->a0, affine->a1, affine->a2, affine->b0, affine->b1, affine->b2, n);
    return PSOLVE_OK;
}

int psolve_apply_affine(const PSolveAffine *affine, double x, double y,
                         double *out_x, double *out_y) {
    if (!affine || !out_x || !out_y) return PSOLVE_ERR_INTERNAL;
    *out_x = affine->a0 + affine->a1 * x + affine->a2 * y;
    *out_y = affine->b0 + affine->b1 * x + affine->b2 * y;
    return PSOLVE_OK;
}

int psolve_check_affine(const PSolveAffine *affine, double scale_min, double scale_max) {
    if (!affine) return PSOLVE_ERR_INTERNAL;
    double det = affine->a1 * affine->b2 - affine->a2 * affine->b1;
    if (fabs(det) < 1e-10) {
        PSLOG_D("psolve_check_affine: det=%.2e too small", det);
        return PSOLVE_ERR_INVALID_TRANS;
    }
    double scale = sqrt(affine->a1 * affine->a1 + affine->b1 * affine->b1);
    if (scale < 1e-15) {
        PSLOG_D("psolve_check_affine: scale=%.2e too small", scale);
        return PSOLVE_ERR_INVALID_TRANS;
    }
    double inv_scale = 1.0 / scale;
    if (inv_scale < scale_min || inv_scale > scale_max) {
        PSLOG_D("psolve_check_affine: inv_scale=%.4f out of range [%.4f, %.4f]",
                inv_scale, scale_min, scale_max);
        return PSOLVE_ERR_INVALID_TRANS;
    }
    double abs_a1 = fabs(affine->a1), abs_b2 = fabs(affine->b2);
    double abs_a2 = fabs(affine->a2), abs_b1 = fabs(affine->b1);
    double max_diag = (abs_a1 > abs_b2) ? abs_a1 : abs_b2;
    if (max_diag > 1e-10) {
        double ratio_diag = fabs(abs_a1 - abs_b2) / max_diag;
        if (ratio_diag > 0.8) {
            PSLOG_D("psolve_check_affine: diagonal ratio %.3f too large (|a1|=%.4f |b2|=%.4f)",
                    ratio_diag, abs_a1, abs_b2);
            return PSOLVE_ERR_INVALID_TRANS;
        }
    }
    double max_off = (abs_a2 > abs_b1) ? abs_a2 : abs_b1;
    if (max_off > 1e-10) {
        double ratio_off = fabs(abs_a2 - abs_b1) / max_off;
        if (ratio_off > 0.8) {
            PSLOG_D("psolve_check_affine: off-diagonal ratio %.3f too large (|a2|=%.4f |b1|=%.4f)",
                    ratio_off, abs_a2, abs_b1);
            return PSOLVE_ERR_INVALID_TRANS;
        }
    }
    return PSOLVE_OK;
}

double psolve_compute_rms(const double *img_x, const double *img_y,
                           const double *cat_x, const double *cat_y,
                           int n, const PSolveAffine *affine,
                           double *rms_x, double *rms_y) {
    if (!img_x || !img_y || !cat_x || !cat_y || !affine || n <= 0) {
        if (rms_x) *rms_x = 0;
        if (rms_y) *rms_y = 0;
        return 0;
    }
    double sum_dx2 = 0, sum_dy2 = 0;
    for (int i = 0; i < n; i++) {
        double tx, ty;
        psolve_apply_affine(affine, img_x[i], img_y[i], &tx, &ty);
        double dx = tx - cat_x[i];
        double dy = ty - cat_y[i];
        sum_dx2 += dx * dx;
        sum_dy2 += dy * dy;
    }
    double rx = sqrt(sum_dx2 / n);
    double ry = sqrt(sum_dy2 / n);
    if (rms_x) *rms_x = rx;
    if (rms_y) *rms_y = ry;
    double rms_total = sqrt((rx * rx + ry * ry) / 2.0);
    PSLOG_D("psolve_compute_rms: rms_x=%.4f rms_y=%.4f rms_total=%.4f n=%d",
            rx, ry, rms_total, n);
    return rms_total;
}

static unsigned int lcg_next(unsigned int *state) {
    *state = (unsigned int)((unsigned long long)(*state) * 1103515245ULL + 12345ULL) & 0x7FFFFFFF;
    return *state;
}

int psolve_ransac_filter(const double *img_x, const double *img_y,
                          const double *cat_x, const double *cat_y,
                          int n,
                          double threshold_px, int max_trials,
                          PSolveAffine *out_affine,
                          int **out_inlier_mask, int *out_inlier_count) {
    if (!img_x || !img_y || !cat_x || !cat_y || n < 3 || !out_affine ||
        !out_inlier_mask || !out_inlier_count) {
        PSLOG_E("psolve_ransac_filter: invalid input (n=%d)", n);
        return PSOLVE_ERR_NOT_ENOUGH;
    }
    PSLOG_I("psolve_ransac_filter: start n=%d threshold=%.2f max_trials=%d",
            n, threshold_px, max_trials);

    int best_inlier_count = 0;
    PSolveAffine best_affine;
    memset(&best_affine, 0, sizeof(best_affine));
    int *best_mask = (int *)calloc(n, sizeof(int));
    int *cur_mask = (int *)calloc(n, sizeof(int));
    if (!best_mask || !cur_mask) {
        free(best_mask); free(cur_mask);
        PSLOG_E("psolve_ransac_filter: alloc failed");
        return PSOLVE_ERR_INTERNAL;
    }

    double scale_min = 0.1;
    double scale_max = 10.0;

    for (int trial = 0; trial < max_trials; trial++) {
        unsigned int rng_state = (unsigned int)(trial + 1) * 2654435761U;
        int idx[3];
        idx[0] = (int)(lcg_next(&rng_state) % (unsigned int)n);
        idx[1] = (int)(lcg_next(&rng_state) % (unsigned int)n);
        while (idx[1] == idx[0]) idx[1] = (int)(lcg_next(&rng_state) % (unsigned int)n);
        idx[2] = (int)(lcg_next(&rng_state) % (unsigned int)n);
        while (idx[2] == idx[0] || idx[2] == idx[1]) idx[2] = (int)(lcg_next(&rng_state) % (unsigned int)n);

        double sx[3], sy[3], dx[3], dy[3];
        for (int j = 0; j < 3; j++) {
            sx[j] = img_x[idx[j]]; sy[j] = img_y[idx[j]];
            dx[j] = cat_x[idx[j]]; dy[j] = cat_y[idx[j]];
        }

        PSolveAffine trial_affine;
        int ret = psolve_compute_affine(sx, sy, dx, dy, 3, &trial_affine);
        if (ret != PSOLVE_OK) continue;

        if (psolve_check_affine(&trial_affine, scale_min, scale_max) != PSOLVE_OK) continue;

        int cur_inlier_count = 0;
        for (int i = 0; i < n; i++) {
            double tx, ty;
            psolve_apply_affine(&trial_affine, img_x[i], img_y[i], &tx, &ty);
            double res_x = tx - cat_x[i];
            double res_y = ty - cat_y[i];
            double dist = sqrt(res_x * res_x + res_y * res_y);
            if (dist < threshold_px) {
                cur_mask[i] = 1;
                cur_inlier_count++;
            } else {
                cur_mask[i] = 0;
            }
        }

        if (cur_inlier_count > best_inlier_count) {
            best_inlier_count = cur_inlier_count;
            best_affine = trial_affine;
            memcpy(best_mask, cur_mask, n * sizeof(int));
            PSLOG_D("psolve_ransac_filter: trial %d new best inliers=%d", trial, cur_inlier_count);
        }
    }

    free(cur_mask);

    if (best_inlier_count < 3) {
        free(best_mask);
        PSLOG_W("psolve_ransac_filter: only %d inliers found (need >=3)", best_inlier_count);
        return PSOLVE_ERR_NO_MATCH;
    }

    double *in_img_x = (double *)malloc(best_inlier_count * sizeof(double));
    double *in_img_y = (double *)malloc(best_inlier_count * sizeof(double));
    double *in_cat_x = (double *)malloc(best_inlier_count * sizeof(double));
    double *in_cat_y = (double *)malloc(best_inlier_count * sizeof(double));
    if (!in_img_x || !in_img_y || !in_cat_x || !in_cat_y) {
        free(best_mask);
        free(in_img_x); free(in_img_y); free(in_cat_x); free(in_cat_y);
        PSLOG_E("psolve_ransac_filter: alloc inlier arrays failed");
        return PSOLVE_ERR_INTERNAL;
    }

    int k = 0;
    for (int i = 0; i < n; i++) {
        if (best_mask[i]) {
            in_img_x[k] = img_x[i]; in_img_y[k] = img_y[i];
            in_cat_x[k] = cat_x[i]; in_cat_y[k] = cat_y[i];
            k++;
        }
    }

    int ret = psolve_compute_affine(in_img_x, in_img_y, in_cat_x, in_cat_y,
                                     best_inlier_count, out_affine);
    free(in_img_x); free(in_img_y); free(in_cat_x); free(in_cat_y);

    if (ret != PSOLVE_OK) {
        free(best_mask);
        PSLOG_E("psolve_ransac_filter: final affine compute failed");
        return ret;
    }

    int final_inlier_count = 0;
    for (int i = 0; i < n; i++) {
        double tx, ty;
        psolve_apply_affine(out_affine, img_x[i], img_y[i], &tx, &ty);
        double res_x = tx - cat_x[i];
        double res_y = ty - cat_y[i];
        double dist = sqrt(res_x * res_x + res_y * res_y);
        if (dist < threshold_px) {
            best_mask[i] = 1;
            final_inlier_count++;
        } else {
            best_mask[i] = 0;
        }
    }

    if (final_inlier_count >= 3 && final_inlier_count != best_inlier_count) {
        double *fin_img_x = (double *)malloc(final_inlier_count * sizeof(double));
        double *fin_img_y = (double *)malloc(final_inlier_count * sizeof(double));
        double *fin_cat_x = (double *)malloc(final_inlier_count * sizeof(double));
        double *fin_cat_y = (double *)malloc(final_inlier_count * sizeof(double));
        if (fin_img_x && fin_img_y && fin_cat_x && fin_cat_y) {
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (best_mask[i]) {
                    fin_img_x[m] = img_x[i]; fin_img_y[m] = img_y[i];
                    fin_cat_x[m] = cat_x[i]; fin_cat_y[m] = cat_y[i];
                    m++;
                }
            }
            psolve_compute_affine(fin_img_x, fin_img_y, fin_cat_x, fin_cat_y,
                                   final_inlier_count, out_affine);
            int recount = 0;
            for (int i = 0; i < n; i++) {
                double tx, ty;
                psolve_apply_affine(out_affine, img_x[i], img_y[i], &tx, &ty);
                double res_x = tx - cat_x[i];
                double res_y = ty - cat_y[i];
                double dist = sqrt(res_x * res_x + res_y * res_y);
                if (dist < threshold_px) {
                    best_mask[i] = 1;
                    recount++;
                } else {
                    best_mask[i] = 0;
                }
            }
            final_inlier_count = recount;
        }
        free(fin_img_x); free(fin_img_y); free(fin_cat_x); free(fin_cat_y);
    }

    *out_inlier_mask = best_mask;
    *out_inlier_count = final_inlier_count;

    PSLOG_I("psolve_ransac_filter: done inliers=%d/%d affine=[a0=%.4f a1=%.6f a2=%.6f b0=%.4f b1=%.6f b2=%.6f]",
            final_inlier_count, n,
            out_affine->a0, out_affine->a1, out_affine->a2,
            out_affine->b0, out_affine->b1, out_affine->b2);

    return PSOLVE_OK;
}

static int gauss_solve_n(int n, double *A, double *b, double *x) {
    double *aug = (double *)malloc(n * (n + 1) * sizeof(double));
    if (!aug) return -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[i * (n + 1) + j] = A[i * n + j];
        aug[i * (n + 1) + n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        int max_row = col;
        double max_val = fabs(aug[col * (n + 1) + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(aug[row * (n + 1) + col]);
            if (v > max_val) { max_val = v; max_row = row; }
        }
        if (max_val < 1e-15) { free(aug); return -1; }
        if (max_row != col) {
            for (int j = 0; j <= n; j++) {
                double tmp = aug[col * (n + 1) + j];
                aug[col * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }
        double pivot = aug[col * (n + 1) + col];
        for (int j = col; j <= n; j++) aug[col * (n + 1) + j] /= pivot;
        for (int row = 0; row < n; row++) {
            if (row == col) continue;
            double factor = aug[row * (n + 1) + col];
            for (int j = col; j <= n; j++) aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
        }
    }
    for (int i = 0; i < n; i++) x[i] = aug[i * (n + 1) + n];
    free(aug);
    return 0;
}

int psolve_compute_distortion(const double *x_src, const double *y_src,
                               const double *x_dst, const double *y_dst,
                               int n, PSolveDistortion *dist) {
    if (n < 6 || !x_src || !y_src || !x_dst || !y_dst || !dist) {
        PSLOG_E("psolve_compute_distortion: invalid input (n=%d, need>=6)", n);
        return PSOLVE_ERR_NOT_ENOUGH;
    }

    const int NP = 6;
    double ATA[36] = {0};
    double ATbx[6] = {0};
    double ATby[6] = {0};

    for (int i = 0; i < n; i++) {
        double xi = x_src[i], yi = y_src[i];
        double xd = x_dst[i], yd = y_dst[i];
        double basis[6] = {1.0, xi, yi, xi * xi, xi * yi, yi * yi};

        for (int r = 0; r < NP; r++) {
            for (int c = 0; c < NP; c++) {
                ATA[r * NP + c] += basis[r] * basis[c];
            }
            ATbx[r] += basis[r] * xd;
            ATby[r] += basis[r] * yd;
        }
    }

    double cx[6], cy[6];
    if (gauss_solve_n(NP, ATA, ATbx, cx) != 0) {
        PSLOG_E("psolve_compute_distortion: singular matrix for x");
        return PSOLVE_ERR_INVALID_TRANS;
    }
    if (gauss_solve_n(NP, ATA, ATby, cy) != 0) {
        PSLOG_E("psolve_compute_distortion: singular matrix for y");
        return PSOLVE_ERR_INVALID_TRANS;
    }

    dist->a0 = cx[0]; dist->a1 = cx[1]; dist->a2 = cx[2];
    dist->a3 = cx[3]; dist->a4 = cx[4]; dist->a5 = cx[5];
    dist->b0 = cy[0]; dist->b1 = cy[1]; dist->b2 = cy[2];
    dist->b3 = cy[3]; dist->b4 = cy[4]; dist->b5 = cy[5];

    PSLOG_D("psolve_compute_distortion: a=[%.4f %.6f %.6f %.2e %.2e %.2e] b=[%.4f %.6f %.6f %.2e %.2e %.2e] n=%d",
            dist->a0, dist->a1, dist->a2, dist->a3, dist->a4, dist->a5,
            dist->b0, dist->b1, dist->b2, dist->b3, dist->b4, dist->b5, n);

    return PSOLVE_OK;
}

int psolve_apply_distortion(const PSolveDistortion *dist, double x, double y,
                             double *out_x, double *out_y) {
    if (!dist || !out_x || !out_y) return PSOLVE_ERR_INTERNAL;
    *out_x = dist->a0 + dist->a1 * x + dist->a2 * y +
             dist->a3 * x * x + dist->a4 * x * y + dist->a5 * y * y;
    *out_y = dist->b0 + dist->b1 * x + dist->b2 * y +
             dist->b3 * x * x + dist->b4 * x * y + dist->b5 * y * y;
    return PSOLVE_OK;
}

double psolve_compute_rms_distortion(const double *cat_x, const double *cat_y,
                                      const double *img_x, const double *img_y,
                                      int n, const PSolveDistortion *dist,
                                      double *rms_x, double *rms_y) {
    if (!cat_x || !cat_y || !img_x || !img_y || !dist || n <= 0) {
        if (rms_x) *rms_x = 0;
        if (rms_y) *rms_y = 0;
        return 0;
    }
    double sum_dx2 = 0, sum_dy2 = 0;
    for (int i = 0; i < n; i++) {
        double tx, ty;
        psolve_apply_distortion(dist, cat_x[i], cat_y[i], &tx, &ty);
        double dx = tx - img_x[i];
        double dy = ty - img_y[i];
        sum_dx2 += dx * dx;
        sum_dy2 += dy * dy;
    }
    double rx = sqrt(sum_dx2 / n);
    double ry = sqrt(sum_dy2 / n);
    if (rms_x) *rms_x = rx;
    if (rms_y) *rms_y = ry;
    double rms_total = sqrt((rx * rx + ry * ry) / 2.0);
    PSLOG_D("psolve_compute_rms_distortion: rms_x=%.4f rms_y=%.4f rms_total=%.4f n=%d",
            rx, ry, rms_total, n);
    return rms_total;
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int psolve_sigma_clip_2d(const double *res_x, const double *res_y, int count,
                          double sigma, int *mask) {
    if (!res_x || !res_y || !mask || count <= 0) return 0;

    double *dist = (double *)malloc(count * sizeof(double));
    if (!dist) return 0;
    for (int i = 0; i < count; i++) {
        dist[i] = sqrt(res_x[i] * res_x[i] + res_y[i] * res_y[i]);
    }

    double *sorted = (double *)malloc(count * sizeof(double));
    if (!sorted) { free(dist); return 0; }
    memcpy(sorted, dist, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_double);

    double med = sorted[count / 2];
    double *abs_dev = (double *)malloc(count * sizeof(double));
    if (!abs_dev) { free(dist); free(sorted); return 0; }
    for (int i = 0; i < count; i++) abs_dev[i] = fabs(dist[i] - med);
    qsort(abs_dev, count, sizeof(double), compare_double);
    double mad = abs_dev[count / 2];

    double threshold = med + sigma * 1.4826 * mad;

    int kept = 0;
    for (int i = 0; i < count; i++) {
        if (dist[i] <= threshold) {
            mask[i] = 1;
            kept++;
        } else {
            mask[i] = 0;
        }
    }

    PSLOG_I("psolve_sigma_clip_2d: med=%.4f mad=%.4f threshold=%.4f kept=%d/%d",
            med, mad, threshold, kept, count);

    free(dist);
    free(sorted);
    free(abs_dev);
    return kept;
}

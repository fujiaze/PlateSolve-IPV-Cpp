#include "psm_affine_distortion.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int psm_gauss_3x3(double A[3][3], double b[3], double x[3])
{
    double M[3][4];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M[i][j] = A[i][j];
        }
        M[i][3] = b[i];
    }
    for (int col = 0; col < 3; col++) {
        int pivot = col;
        double max_val = fabs(M[col][col]);
        for (int row = col + 1; row < 3; row++) {
            if (fabs(M[row][col]) > max_val) {
                max_val = fabs(M[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-15) {
            return PSM_ERR_SINGULAR;
        }
        if (pivot != col) {
            for (int j = col; j < 4; j++) {
                double tmp = M[col][j];
                M[col][j] = M[pivot][j];
                M[pivot][j] = tmp;
            }
        }
        double inv_pivot = 1.0 / M[col][col];
        for (int row = col + 1; row < 3; row++) {
            double factor = M[row][col] * inv_pivot;
            for (int j = col; j < 4; j++) {
                M[row][j] -= factor * M[col][j];
            }
        }
    }
    for (int i = 2; i >= 0; i--) {
        double sum = M[i][3];
        for (int j = i + 1; j < 3; j++) {
            sum -= M[i][j] * x[j];
        }
        x[i] = sum / M[i][i];
    }
    return PSM_OK;
}

static int psm_gauss_n(int n, double *A, double *b, double *x)
{
    int stride = n + 1;
    double *aug = (double *)malloc((size_t)n * (size_t)stride * sizeof(double));
    if (!aug) {
        return PSM_ERR_ALLOC;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * stride + j] = A[i * n + j];
        }
        aug[i * stride + n] = b[i];
    }
    for (int col = 0; col < n; col++) {
        int pivot = col;
        double max_val = fabs(aug[col * stride + col]);
        for (int row = col + 1; row < n; row++) {
            double v = fabs(aug[row * stride + col]);
            if (v > max_val) {
                max_val = v;
                pivot = row;
            }
        }
        if (max_val < 1e-15) {
            free(aug);
            return PSM_ERR_SINGULAR;
        }
        if (pivot != col) {
            for (int j = col; j <= n; j++) {
                double tmp = aug[col * stride + j];
                aug[col * stride + j] = aug[pivot * stride + j];
                aug[pivot * stride + j] = tmp;
            }
        }
        double inv_pivot = 1.0 / aug[col * stride + col];
        for (int row = col + 1; row < n; row++) {
            double factor = aug[row * stride + col] * inv_pivot;
            for (int j = col; j <= n; j++) {
                aug[row * stride + j] -= factor * aug[col * stride + j];
            }
        }
    }
    for (int i = n - 1; i >= 0; i--) {
        double sum = aug[i * stride + n];
        for (int j = i + 1; j < n; j++) {
            sum -= aug[i * stride + j] * x[j];
        }
        x[i] = sum / aug[i * stride + i];
    }
    free(aug);
    return PSM_OK;
}

PSM_EXPORT int psm_affine_compute(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst, int n, PSMAffine *out_affine)
{
    if (!x_src || !y_src || !x_dst || !y_dst || n < 3 || !out_affine) {
        return PSM_ERR_INVALID_PARAM;
    }
    double sx = 0.0, sy = 0.0, sx2 = 0.0, sy2 = 0.0, sxy = 0.0;
    double sdx = 0.0, sdy = 0.0, sdx_x = 0.0, sdy_x = 0.0, sdx_y = 0.0, sdy_y = 0.0;
    for (int i = 0; i < n; i++) {
        double xs = x_src[i];
        double ys = y_src[i];
        double xd = x_dst[i];
        double yd = y_dst[i];
        sx += xs;
        sy += ys;
        sx2 += xs * xs;
        sy2 += ys * ys;
        sxy += xs * ys;
        sdx += xd;
        sdy += yd;
        sdx_x += xd * xs;
        sdy_x += yd * xs;
        sdx_y += xd * ys;
        sdy_y += yd * ys;
    }
    double A[3][3] = {
        { (double)n, sx, sy },
        { sx, sx2, sxy },
        { sy, sxy, sy2 }
    };
    double bx[3] = { sdx, sdx_x, sdx_y };
    double by[3] = { sdy, sdy_x, sdy_y };
    double ax[3], ay[3];
    int ret = psm_gauss_3x3(A, bx, ax);
    if (ret != PSM_OK) {
        return ret;
    }
    ret = psm_gauss_3x3(A, by, ay);
    if (ret != PSM_OK) {
        return ret;
    }
    out_affine->a0 = ax[0];
    out_affine->a1 = ax[1];
    out_affine->a2 = ax[2];
    out_affine->b0 = ay[0];
    out_affine->b1 = ay[1];
    out_affine->b2 = ay[2];
    return PSM_OK;
}

PSM_EXPORT void psm_affine_apply(const PSMAffine *affine,
    double x, double y, double *out_x, double *out_y)
{
    if (!affine || !out_x || !out_y) return;
    *out_x = affine->a0 + affine->a1 * x + affine->a2 * y;
    *out_y = affine->b0 + affine->b1 * x + affine->b2 * y;
}

PSM_EXPORT int psm_check_affine(const PSMAffine *affine,
    double scale_min, double scale_max)
{
    if (!affine) {
        return PSM_ERR_INVALID_PARAM;
    }
    double det = affine->a1 * affine->b2 - affine->a2 * affine->b1;
    if (fabs(det) < 1e-10) {
        return PSM_ERR_SINGULAR;
    }
    double scale = sqrt(affine->a1 * affine->a1 + affine->b1 * affine->b1);
    if (scale < 1e-15) {
        return PSM_ERR_INVALID_PARAM;
    }
    double inv_scale = 1.0 / scale;
    if (inv_scale < scale_min || inv_scale > scale_max) {
        return PSM_ERR_INVALID_PARAM;
    }
    double abs_a1 = fabs(affine->a1), abs_b2 = fabs(affine->b2);
    double abs_a2 = fabs(affine->a2), abs_b1 = fabs(affine->b1);
    double max_diag = (abs_a1 > abs_b2) ? abs_a1 : abs_b2;
    if (max_diag > 1e-10) {
        double ratio_diag = fabs(abs_a1 - abs_b2) / max_diag;
        if (ratio_diag > 0.8) {
            return PSM_ERR_INVALID_PARAM;
        }
    }
    double max_off = (abs_a2 > abs_b1) ? abs_a2 : abs_b1;
    if (max_off > 1e-10) {
        double ratio_off = fabs(abs_a2 - abs_b1) / max_off;
        if (ratio_off > 0.8) {
            return PSM_ERR_INVALID_PARAM;
        }
    }
    return PSM_OK;
}

static unsigned int psm_lcg_next(unsigned int *state)
{
    *state = (unsigned int)((unsigned long long)(*state) * 1103515245ULL + 12345ULL) & 0x7FFFFFFF;
    return *state;
}

PSM_EXPORT int psm_ransac_filter(const double *img_x, const double *img_y,
    const double *cat_x, const double *cat_y, int n,
    double threshold_px, int max_trials,
    PSMAffine *out_affine, int **out_inlier_mask, int *out_inlier_count)
{
    if (!img_x || !img_y || !cat_x || !cat_y || n < 3 || !out_affine ||
        !out_inlier_mask || !out_inlier_count) {
        return PSM_ERR_INVALID_PARAM;
    }
    int best_inlier_count = 0;
    PSMAffine best_affine;
    memset(&best_affine, 0, sizeof(best_affine));
    int *best_mask = (int *)calloc((size_t)n, sizeof(int));
    int *cur_mask = (int *)calloc((size_t)n, sizeof(int));
    if (!best_mask || !cur_mask) {
        free(best_mask);
        free(cur_mask);
        return PSM_ERR_ALLOC;
    }
    double scale_min = 0.1;
    double scale_max = 10.0;
    for (int trial = 0; trial < max_trials; trial++) {
        unsigned int rng_state = (unsigned int)(trial + 1) * 2654435761U;
        int idx[3];
        idx[0] = (int)(psm_lcg_next(&rng_state) % (unsigned int)n);
        idx[1] = (int)(psm_lcg_next(&rng_state) % (unsigned int)n);
        while (idx[1] == idx[0]) {
            idx[1] = (int)(psm_lcg_next(&rng_state) % (unsigned int)n);
        }
        idx[2] = (int)(psm_lcg_next(&rng_state) % (unsigned int)n);
        while (idx[2] == idx[0] || idx[2] == idx[1]) {
            idx[2] = (int)(psm_lcg_next(&rng_state) % (unsigned int)n);
        }
        double sx[3], sy[3], dx[3], dy[3];
        for (int j = 0; j < 3; j++) {
            sx[j] = img_x[idx[j]];
            sy[j] = img_y[idx[j]];
            dx[j] = cat_x[idx[j]];
            dy[j] = cat_y[idx[j]];
        }
        PSMAffine trial_affine;
        int ret = psm_affine_compute(sx, sy, dx, dy, 3, &trial_affine);
        if (ret != PSM_OK) continue;
        if (psm_check_affine(&trial_affine, scale_min, scale_max) != PSM_OK) continue;
        int cur_inlier_count = 0;
        for (int i = 0; i < n; i++) {
            double tx, ty;
            psm_affine_apply(&trial_affine, img_x[i], img_y[i], &tx, &ty);
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
            memcpy(best_mask, cur_mask, (size_t)n * sizeof(int));
        }
    }
    free(cur_mask);
    if (best_inlier_count < 3) {
        free(best_mask);
        return PSM_ERR_NO_MATCH;
    }
    double *in_img_x = (double *)malloc((size_t)best_inlier_count * sizeof(double));
    double *in_img_y = (double *)malloc((size_t)best_inlier_count * sizeof(double));
    double *in_cat_x = (double *)malloc((size_t)best_inlier_count * sizeof(double));
    double *in_cat_y = (double *)malloc((size_t)best_inlier_count * sizeof(double));
    if (!in_img_x || !in_img_y || !in_cat_x || !in_cat_y) {
        free(best_mask);
        free(in_img_x);
        free(in_img_y);
        free(in_cat_x);
        free(in_cat_y);
        return PSM_ERR_ALLOC;
    }
    int k = 0;
    for (int i = 0; i < n; i++) {
        if (best_mask[i]) {
            in_img_x[k] = img_x[i];
            in_img_y[k] = img_y[i];
            in_cat_x[k] = cat_x[i];
            in_cat_y[k] = cat_y[i];
            k++;
        }
    }
    int ret = psm_affine_compute(in_img_x, in_img_y, in_cat_x, in_cat_y,
        best_inlier_count, out_affine);
    free(in_img_x);
    free(in_img_y);
    free(in_cat_x);
    free(in_cat_y);
    if (ret != PSM_OK) {
        free(best_mask);
        return ret;
    }
    int final_inlier_count = 0;
    for (int i = 0; i < n; i++) {
        double tx, ty;
        psm_affine_apply(out_affine, img_x[i], img_y[i], &tx, &ty);
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
        double *fin_img_x = (double *)malloc((size_t)final_inlier_count * sizeof(double));
        double *fin_img_y = (double *)malloc((size_t)final_inlier_count * sizeof(double));
        double *fin_cat_x = (double *)malloc((size_t)final_inlier_count * sizeof(double));
        double *fin_cat_y = (double *)malloc((size_t)final_inlier_count * sizeof(double));
        if (fin_img_x && fin_img_y && fin_cat_x && fin_cat_y) {
            int m = 0;
            for (int i = 0; i < n; i++) {
                if (best_mask[i]) {
                    fin_img_x[m] = img_x[i];
                    fin_img_y[m] = img_y[i];
                    fin_cat_x[m] = cat_x[i];
                    fin_cat_y[m] = cat_y[i];
                    m++;
                }
            }
            psm_affine_compute(fin_img_x, fin_img_y, fin_cat_x, fin_cat_y,
                final_inlier_count, out_affine);
            int recount = 0;
            for (int i = 0; i < n; i++) {
                double tx, ty;
                psm_affine_apply(out_affine, img_x[i], img_y[i], &tx, &ty);
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
        free(fin_img_x);
        free(fin_img_y);
        free(fin_cat_x);
        free(fin_cat_y);
    }
    *out_inlier_mask = best_mask;
    *out_inlier_count = final_inlier_count;
    return PSM_OK;
}

PSM_EXPORT int psm_distortion_compute(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst, int n, PSMDistortion *out_dist)
{
    if (!x_src || !y_src || !x_dst || !y_dst || n < 6 || !out_dist) {
        return PSM_ERR_INVALID_PARAM;
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
    int ret = psm_gauss_n(NP, ATA, ATbx, cx);
    if (ret != PSM_OK) {
        return ret;
    }
    ret = psm_gauss_n(NP, ATA, ATby, cy);
    if (ret != PSM_OK) {
        return ret;
    }
    out_dist->a0 = cx[0]; out_dist->a1 = cx[1]; out_dist->a2 = cx[2];
    out_dist->a3 = cx[3]; out_dist->a4 = cx[4]; out_dist->a5 = cx[5];
    out_dist->b0 = cy[0]; out_dist->b1 = cy[1]; out_dist->b2 = cy[2];
    out_dist->b3 = cy[3]; out_dist->b4 = cy[4]; out_dist->b5 = cy[5];
    return PSM_OK;
}

PSM_EXPORT void psm_distortion_apply(const PSMDistortion *dist,
    double x, double y, double *out_x, double *out_y)
{
    if (!dist || !out_x || !out_y) return;
    *out_x = dist->a0 + dist->a1 * x + dist->a2 * y +
        dist->a3 * x * x + dist->a4 * x * y + dist->a5 * y * y;
    *out_y = dist->b0 + dist->b1 * x + dist->b2 * y +
        dist->b3 * x * x + dist->b4 * x * y + dist->b5 * y * y;
}

PSM_EXPORT void psm_free_mask(int *mask)
{
    free(mask);
}

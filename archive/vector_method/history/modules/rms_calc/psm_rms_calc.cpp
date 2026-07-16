#include "psm_rms_calc.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define EVAL_LOG(fmt, ...) fprintf(stderr, "[EVAL] " fmt "\n", ##__VA_ARGS__)

extern "C" {
    typedef struct GaiaClient GaiaClient;
    int gaia_client_cone_search_for_solver(
        GaiaClient *client,
        double ra, double dec, double radius_deg,
        double mag_high,
        double **out_ra, double **out_dec, float **out_mag,
        int *out_count);
}

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int psm_apply_affine_inline(const PSMAffine *affine, double x, double y,
    double *out_x, double *out_y)
{
    if (!affine || !out_x || !out_y) return PSM_ERR_INVALID_PARAM;
    *out_x = affine->a0 + affine->a1 * x + affine->a2 * y;
    *out_y = affine->b0 + affine->b1 * x + affine->b2 * y;
    return PSM_OK;
}

static int psm_apply_distortion_inline(const PSMDistortion *dist, double x, double y,
    double *out_x, double *out_y)
{
    if (!dist || !out_x || !out_y) return PSM_ERR_INVALID_PARAM;
    *out_x = dist->a0 + dist->a1 * x + dist->a2 * y +
             dist->a3 * x * x + dist->a4 * x * y + dist->a5 * y * y;
    *out_y = dist->b0 + dist->b1 * x + dist->b2 * y +
             dist->b3 * x * x + dist->b4 * x * y + dist->b5 * y * y;
    return PSM_OK;
}

PSM_EXPORT int psm_sigma_clip_2d(const double *res_x, const double *res_y, int count,
    double sigma, int *mask)
{
    if (!res_x || !res_y || !mask || count <= 0) {
        return 0;
    }

    double *dist = (double *)malloc((size_t)count * sizeof(double));
    if (!dist) return 0;
    for (int i = 0; i < count; i++) {
        dist[i] = sqrt(res_x[i] * res_x[i] + res_y[i] * res_y[i]);
    }

    double *sorted = (double *)malloc((size_t)count * sizeof(double));
    if (!sorted) { free(dist); return 0; }
    memcpy(sorted, dist, (size_t)count * sizeof(double));
    qsort(sorted, (size_t)count, sizeof(double), compare_double);

    double med = sorted[count / 2];

    double *abs_dev = (double *)malloc((size_t)count * sizeof(double));
    if (!abs_dev) { free(dist); free(sorted); return 0; }
    for (int i = 0; i < count; i++) {
        abs_dev[i] = fabs(dist[i] - med);
    }
    qsort(abs_dev, (size_t)count, sizeof(double), compare_double);
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

    free(dist);
    free(sorted);
    free(abs_dev);
    return kept;
}

PSM_EXPORT int psm_rms_compute_affine(const double *img_x, const double *img_y,
    const double *cat_x, const double *cat_y, int n,
    const PSMAffine *affine, double *rms_x, double *rms_y)
{
    if (!img_x || !img_y || !cat_x || !cat_y || !affine || n <= 0) {
        if (rms_x) *rms_x = 0.0;
        if (rms_y) *rms_y = 0.0;
        return PSM_ERR_INVALID_PARAM;
    }

    double sum_dx2 = 0.0, sum_dy2 = 0.0;
    for (int i = 0; i < n; i++) {
        double tx, ty;
        psm_apply_affine_inline(affine, img_x[i], img_y[i], &tx, &ty);
        double dx = tx - cat_x[i];
        double dy = ty - cat_y[i];
        sum_dx2 += dx * dx;
        sum_dy2 += dy * dy;
    }

    double rx = sqrt(sum_dx2 / (double)n);
    double ry = sqrt(sum_dy2 / (double)n);
    if (rms_x) *rms_x = rx;
    if (rms_y) *rms_y = ry;
    return PSM_OK;
}

PSM_EXPORT int psm_rms_compute_distortion(const double *cat_x, const double *cat_y,
    const double *img_x, const double *img_y, int n,
    const PSMDistortion *dist, double *rms_x, double *rms_y)
{
    if (!cat_x || !cat_y || !img_x || !img_y || !dist || n <= 0) {
        if (rms_x) *rms_x = 0.0;
        if (rms_y) *rms_y = 0.0;
        return PSM_ERR_INVALID_PARAM;
    }

    double sum_dx2 = 0.0, sum_dy2 = 0.0;
    for (int i = 0; i < n; i++) {
        double tx, ty;
        psm_apply_distortion_inline(dist, cat_x[i], cat_y[i], &tx, &ty);
        double dx = tx - img_x[i];
        double dy = ty - img_y[i];
        sum_dx2 += dx * dx;
        sum_dy2 += dy * dy;
    }

    double rx = sqrt(sum_dx2 / (double)n);
    double ry = sqrt(sum_dy2 / (double)n);
    if (rms_x) *rms_x = rx;
    if (rms_y) *rms_y = ry;
    return PSM_OK;
}

PSM_EXPORT void psm_residual_stats(const double *residuals, int n,
    double *out_mean, double *out_median, double *out_mad, double *out_rms)
{
    if (!residuals || n <= 0) {
        if (out_mean) *out_mean = 0.0;
        if (out_median) *out_median = 0.0;
        if (out_mad) *out_mad = 0.0;
        if (out_rms) *out_rms = 0.0;
        return;
    }

    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        sum += residuals[i];
        sum_sq += residuals[i] * residuals[i];
    }
    double mean = sum / (double)n;
    double rms = sqrt(sum_sq / (double)n);

    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) {
        if (out_mean) *out_mean = mean;
        if (out_median) *out_median = 0.0;
        if (out_mad) *out_mad = 0.0;
        if (out_rms) *out_rms = rms;
        return;
    }
    memcpy(sorted, residuals, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), compare_double);
    double median = sorted[n / 2];

    double *abs_dev = (double *)malloc((size_t)n * sizeof(double));
    if (!abs_dev) {
        free(sorted);
        if (out_mean) *out_mean = mean;
        if (out_median) *out_median = median;
        if (out_mad) *out_mad = 0.0;
        if (out_rms) *out_rms = rms;
        return;
    }
    for (int i = 0; i < n; i++) {
        abs_dev[i] = fabs(residuals[i] - median);
    }
    qsort(abs_dev, (size_t)n, sizeof(double), compare_double);
    double mad = abs_dev[n / 2];

    free(sorted);
    free(abs_dev);

    if (out_mean) *out_mean = mean;
    if (out_median) *out_median = median;
    if (out_mad) *out_mad = mad;
    if (out_rms) *out_rms = rms;
}

PSM_EXPORT int psm_rms_clipped(const double *img_x, const double *img_y,
    const double *cat_x, const double *cat_y, int n,
    const PSMAffine *affine, double sigma,
    double *rms_x, double *rms_y, int *out_kept)
{
    if (!img_x || !img_y || !cat_x || !cat_y || !affine || n <= 0 ||
        !rms_x || !rms_y || !out_kept) {
        if (rms_x) *rms_x = 0.0;
        if (rms_y) *rms_y = 0.0;
        if (out_kept) *out_kept = 0;
        return PSM_ERR_INVALID_PARAM;
    }

    double *res_x = (double *)malloc((size_t)n * sizeof(double));
    double *res_y = (double *)malloc((size_t)n * sizeof(double));
    if (!res_x || !res_y) {
        free(res_x); free(res_y);
        *rms_x = 0.0;
        *rms_y = 0.0;
        *out_kept = 0;
        return PSM_ERR_ALLOC;
    }

    for (int i = 0; i < n; i++) {
        double tx, ty;
        psm_apply_affine_inline(affine, img_x[i], img_y[i], &tx, &ty);
        res_x[i] = tx - cat_x[i];
        res_y[i] = ty - cat_y[i];
    }

    int *mask = (int *)malloc((size_t)n * sizeof(int));
    if (!mask) {
        free(res_x); free(res_y);
        *rms_x = 0.0;
        *rms_y = 0.0;
        *out_kept = 0;
        return PSM_ERR_ALLOC;
    }

    int kept = psm_sigma_clip_2d(res_x, res_y, n, sigma, mask);
    *out_kept = kept;

    if (kept < 3) {
        free(res_x); free(res_y); free(mask);
        *rms_x = 0.0;
        *rms_y = 0.0;
        return PSM_OK;
    }

    double sum_dx2 = 0.0, sum_dy2 = 0.0;
    for (int i = 0; i < n; i++) {
        if (mask[i]) {
            sum_dx2 += res_x[i] * res_x[i];
            sum_dy2 += res_y[i] * res_y[i];
        }
    }

    free(res_x);
    free(res_y);
    free(mask);

    *rms_x = sqrt(sum_dx2 / (double)kept);
    *rms_y = sqrt(sum_dy2 / (double)kept);
    return PSM_OK;
}

PSM_EXPORT int psm_rms_evaluate_model(
    void *gaia_client,
    const PSMModelEvalInput *model,
    const double *img_x, const double *img_y, int img_count,
    PSMModelEvalResult *out_result)
{
    if (!gaia_client || !model || !img_x || !img_y || img_count <= 0 || !out_result) {
        return PSM_ERR_INVALID_PARAM;
    }

    memset(out_result, 0, sizeof(PSMModelEvalResult));

    EVAL_LOG("center=(%.6f,%.6f) scale=%.3f rot=%.2f radius=%.4f mag=%.1f sigma=%.1f",
             model->ra_center, model->dec_center, model->scale_arcsec_px,
             model->rotation_deg, model->query_radius_deg, model->mag_limit, model->sigma_clip);

    double *gaia_ra = NULL, *gaia_dec = NULL;
    float *gaia_mag = NULL;
    int gaia_count = 0;

    int rc = gaia_client_cone_search_for_solver(
        (GaiaClient *)gaia_client,
        model->ra_center, model->dec_center,
        model->query_radius_deg,
        model->mag_limit,
        &gaia_ra, &gaia_dec, &gaia_mag, &gaia_count);

    if (rc != 0 || gaia_count <= 0) {
        EVAL_LOG("Gaia query failed rc=%d count=%d", rc, gaia_count);
        return PSM_ERR_NO_MATCH;
    }

    out_result->total_gaia_stars = gaia_count;
    EVAL_LOG("Gaia returned %d stars", gaia_count);

    double *gaia_px = (double *)malloc((size_t)gaia_count * sizeof(double));
    double *gaia_py = (double *)malloc((size_t)gaia_count * sizeof(double));
    if (!gaia_px || !gaia_py) {
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        return PSM_ERR_ALLOC;
    }

    double cdec_rad = model->dec_center * M_PI / 180.0;
    double cra_rad = model->ra_center * M_PI / 180.0;
    double rad_to_px = 180.0 / M_PI * 3600.0 / model->scale_arcsec_px;

    for (int i = 0; i < gaia_count; i++) {
        double dec_rad = gaia_dec[i] * M_PI / 180.0;
        double ra_rad = gaia_ra[i] * M_PI / 180.0;
        double cos_c = sin(cdec_rad) * sin(dec_rad) + cos(cdec_rad) * cos(dec_rad) * cos(ra_rad - cra_rad);
        if (cos_c <= 1e-10) {
            gaia_px[i] = 1e30;
            gaia_py[i] = 1e30;
            continue;
        }
        double xi = cos(dec_rad) * sin(ra_rad - cra_rad) / cos_c;
        double eta = (cos(cdec_rad) * sin(dec_rad) - sin(cdec_rad) * cos(dec_rad) * cos(ra_rad - cra_rad)) / cos_c;
        gaia_px[i] = xi * rad_to_px;
        gaia_py[i] = eta * rad_to_px;
    }

    double *map_x = (double *)malloc((size_t)img_count * sizeof(double));
    double *map_y = (double *)malloc((size_t)img_count * sizeof(double));
    if (!map_x || !map_y) {
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        free(map_x); free(map_y);
        return PSM_ERR_ALLOC;
    }

    if (model->affine) {
        for (int i = 0; i < img_count; i++) {
            psm_apply_affine_inline(model->affine, img_x[i], img_y[i], &map_x[i], &map_y[i]);
        }
        EVAL_LOG("Using affine mapping");
    } else if (model->distortion) {
        for (int i = 0; i < img_count; i++) {
            psm_apply_distortion_inline(model->distortion, img_x[i], img_y[i], &map_x[i], &map_y[i]);
        }
        EVAL_LOG("Using distortion mapping");
    } else {
        double rot_rad = model->rotation_deg * M_PI / 180.0;
        double cos_r = cos(rot_rad);
        double sin_r = sin(rot_rad);
        for (int i = 0; i < img_count; i++) {
            map_x[i] = cos_r * img_x[i] - sin_r * img_y[i];
            map_y[i] = sin_r * img_x[i] + cos_r * img_y[i];
        }
        EVAL_LOG("Using simple rotation mapping rot=%.2f", model->rotation_deg);
    }

    double *match_dx = (double *)malloc((size_t)img_count * sizeof(double));
    double *match_dy = (double *)malloc((size_t)img_count * sizeof(double));
    if (!match_dx || !match_dy) {
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        free(map_x); free(map_y);
        free(match_dx); free(match_dy);
        return PSM_ERR_ALLOC;
    }

    double max_dist_sq = 25.0 * 25.0;
    int matched = 0;

    for (int i = 0; i < img_count; i++) {
        double best_dist_sq = 1e30;
        int best_j = -1;
        for (int j = 0; j < gaia_count; j++) {
            if (gaia_px[j] > 1e29) continue;
            double dx = map_x[i] - gaia_px[j];
            double dy = map_y[i] - gaia_py[j];
            double d2 = dx * dx + dy * dy;
            if (d2 < best_dist_sq) {
                best_dist_sq = d2;
                best_j = j;
            }
        }
        if (best_j >= 0 && best_dist_sq < max_dist_sq) {
            match_dx[matched] = map_x[i] - gaia_px[best_j];
            match_dy[matched] = map_y[i] - gaia_py[best_j];
            matched++;
        }
    }

    EVAL_LOG("Initial matches: %d / %d img_stars", matched, img_count);

    if (matched < 3) {
        out_result->matched_count = matched;
        out_result->clipped_count = 0;
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        free(map_x); free(map_y);
        free(match_dx); free(match_dy);
        return PSM_OK;
    }

    int *clip_mask = (int *)malloc((size_t)matched * sizeof(int));
    if (!clip_mask) {
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        free(map_x); free(map_y);
        free(match_dx); free(match_dy);
        return PSM_ERR_ALLOC;
    }

    int kept = psm_sigma_clip_2d(match_dx, match_dy, matched, model->sigma_clip, clip_mask);
    int clipped = matched - kept;

    EVAL_LOG("Sigma-clip: kept=%d clipped=%d", kept, clipped);

    if (kept < 3) {
        out_result->matched_count = matched;
        out_result->clipped_count = clipped;
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        free(map_x); free(map_y);
        free(match_dx); free(match_dy);
        free(clip_mask);
        return PSM_OK;
    }

    double sum_dx2 = 0.0, sum_dy2 = 0.0;
    double *residuals = (double *)malloc((size_t)kept * sizeof(double));
    if (!residuals) {
        free(gaia_ra); free(gaia_dec); free(gaia_mag);
        free(gaia_px); free(gaia_py);
        free(map_x); free(map_y);
        free(match_dx); free(match_dy);
        free(clip_mask);
        return PSM_ERR_ALLOC;
    }

    int ki = 0;
    for (int i = 0; i < matched; i++) {
        if (clip_mask[i]) {
            sum_dx2 += match_dx[i] * match_dx[i];
            sum_dy2 += match_dy[i] * match_dy[i];
            residuals[ki] = sqrt(match_dx[i] * match_dx[i] + match_dy[i] * match_dy[i]);
            ki++;
        }
    }

    double rms_x = sqrt(sum_dx2 / (double)kept);
    double rms_y = sqrt(sum_dy2 / (double)kept);
    double rms_total = sqrt((sum_dx2 + sum_dy2) / (double)kept);
    double rms_arcsec_val = rms_total * model->scale_arcsec_px;

    double mean_res, median_res, mad_res, rms_res;
    psm_residual_stats(residuals, kept, &mean_res, &median_res, &mad_res, &rms_res);

    double density_score = fmin((double)kept / fmax((double)img_count * 0.7, 1.0), 1.0);
    double precision_score = exp(-rms_total / 10.0);
    double coverage_score = 1.0 - fabs((double)clipped / (double)matched - 0.95);
    double score = 40.0 * density_score + 40.0 * precision_score + 20.0 * coverage_score;

    out_result->matched_count = matched;
    out_result->clipped_count = clipped;
    out_result->rms_x_px = rms_x;
    out_result->rms_y_px = rms_y;
    out_result->rms_total_px = rms_total;
    out_result->rms_arcsec = rms_arcsec_val;
    out_result->mean_residual_px = mean_res;
    out_result->median_residual_px = median_res;
    out_result->mad_px = mad_res;
    out_result->score = score;
    out_result->score_density = density_score;
    out_result->score_precision = precision_score;
    out_result->score_coverage = coverage_score;

    EVAL_LOG("matched=%d clipped=%d rms=%.3fpx %.3f\" score=%.2f d=%.2f p=%.2f c=%.2f",
             matched, clipped, rms_total, rms_arcsec_val, score,
             density_score, precision_score, coverage_score);

    free(gaia_ra); free(gaia_dec); free(gaia_mag);
    free(gaia_px); free(gaia_py);
    free(map_x); free(map_y);
    free(match_dx); free(match_dy);
    free(clip_mask);
    free(residuals);

    return PSM_OK;
}

PSM_EXPORT void psm_free_model_eval_result(PSMModelEvalResult *result)
{
    if (result) {
        memset(result, 0, sizeof(PSMModelEvalResult));
    }
}

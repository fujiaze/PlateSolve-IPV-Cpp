#include "psolve_fine.h"
#include "psolve_log.h"
#include "psolve_ransac.h"
#include "psolve_projection.h"
#include "psolve_fov.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PSOLVE_FINE_ITER_RADIUS 2.0
#define PSOLVE_FINE_SIGMA_CLIP 3.0
#define PSOLVE_FINE_MAX_ITER 5

static double tps_phi(double r) {
    if (r < 1e-10) return 0.0;
    return r * r * log(r);
}

static int gauss_solve(int n, double *A, double *b, double *x) {
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

typedef struct {
    double cx, cy;
    double x_min, x_max, y_min, y_max;
    double half_size;
    int star_count;
    int *star_indices;
    double *rbf_wx;
    double *rbf_wy;
    double poly_cx[3];
    double poly_cy[3];
    int solved;
} SubDomain;

static int solve_subdomain_rbf(
    const double *img_x, const double *img_y,
    const double *res_x, const double *res_y,
    const int *indices, int count,
    double *wx, double *wy, double *cx_out, double *cy_out) {
    if (count < 3) return -1;
    int n = count;
    int sz = n + 3;
    double *A = (double *)calloc(sz * sz, sizeof(double));
    double *bx = (double *)calloc(sz, sizeof(double));
    double *by = (double *)calloc(sz, sizeof(double));
    double *sol_x = (double *)calloc(sz, sizeof(double));
    double *sol_y = (double *)calloc(sz, sizeof(double));
    if (!A || !bx || !by || !sol_x || !sol_y) {
        free(A); free(bx); free(by); free(sol_x); free(sol_y);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double dx = img_x[indices[i]] - img_x[indices[j]];
            double dy = img_y[indices[i]] - img_y[indices[j]];
            double r = sqrt(dx * dx + dy * dy);
            A[i * sz + j] = tps_phi(r);
        }
        A[i * sz + n] = 1.0;
        A[i * sz + n + 1] = img_x[indices[i]];
        A[i * sz + n + 2] = img_y[indices[i]];
        A[n * sz + i] = 1.0;
        A[(n + 1) * sz + i] = img_x[indices[i]];
        A[(n + 2) * sz + i] = img_y[indices[i]];
        bx[i] = res_x[indices[i]];
        by[i] = res_y[indices[i]];
    }
    if (gauss_solve(sz, A, bx, sol_x) != 0) {
        free(A); free(bx); free(by); free(sol_x); free(sol_y);
        return -1;
    }
    if (gauss_solve(sz, A, by, sol_y) != 0) {
        free(A); free(bx); free(by); free(sol_x); free(sol_y);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        wx[i] = sol_x[i];
        wy[i] = sol_y[i];
    }
    cx_out[0] = sol_x[n]; cx_out[1] = sol_x[n + 1]; cx_out[2] = sol_x[n + 2];
    cy_out[0] = sol_y[n]; cy_out[1] = sol_y[n + 1]; cy_out[2] = sol_y[n + 2];
    free(A); free(bx); free(by); free(sol_x); free(sol_y);
    return 0;
}

static double eval_subdomain_rbf(
    const SubDomain *sd,
    const double *img_x, const double *img_y,
    double px, double py, int comp) {
    if (!sd->solved || sd->star_count < 3) return 0.0;
    const double *w = (comp == 0) ? sd->rbf_wx : sd->rbf_wy;
    const double *c = (comp == 0) ? sd->poly_cx : sd->poly_cy;
    double val = c[0] + c[1] * px + c[2] * py;
    for (int i = 0; i < sd->star_count; i++) {
        double dx = px - img_x[sd->star_indices[i]];
        double dy = py - img_y[sd->star_indices[i]];
        double r = sqrt(dx * dx + dy * dy);
        val += w[i] * tps_phi(r);
    }
    return val;
}

static void eval_rbf_blended(
    const SubDomain *subdomains, int subdomain_count,
    const double *img_x, const double *img_y,
    double px, double py,
    double *out_rx, double *out_ry) {
    double sum_wx = 0, sum_wy = 0, sum_w = 0;
    for (int s = 0; s < subdomain_count; s++) {
        const SubDomain *sd = &subdomains[s];
        if (!sd->solved) continue;
        if (px < sd->x_min || px > sd->x_max || py < sd->y_min || py > sd->y_max) continue;
        double dx = px - sd->cx;
        double dy = py - sd->cy;
        double d = sqrt(dx * dx + dy * dy) / sd->half_size;
        double w = (d < 1.0) ? (1.0 - d) : 0.0;
        if (w > 0) {
            sum_wx += w * eval_subdomain_rbf(sd, img_x, img_y, px, py, 0);
            sum_wy += w * eval_subdomain_rbf(sd, img_x, img_y, px, py, 1);
            sum_w += w;
        }
    }
    if (sum_w > 1e-15) {
        *out_rx = sum_wx / sum_w;
        *out_ry = sum_wy / sum_w;
    } else {
        *out_rx = 0.0;
        *out_ry = 0.0;
    }
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static int sigma_clip_filter(
    const double *residuals, int count,
    double sigma, int *mask) {
    double *sorted = (double *)malloc(count * sizeof(double));
    memcpy(sorted, residuals, count * sizeof(double));
    qsort(sorted, count, sizeof(double), compare_double);
    double med = sorted[count / 2];
    double *abs_dev = (double *)malloc(count * sizeof(double));
    for (int i = 0; i < count; i++) abs_dev[i] = fabs(residuals[i] - med);
    qsort(abs_dev, count, sizeof(double), compare_double);
    double mad = abs_dev[count / 2];
    double threshold = med + sigma * 1.4826 * mad;
    double threshold_low = med - sigma * 1.4826 * mad;
    int kept = 0;
    for (int i = 0; i < count; i++) {
        if (residuals[i] >= threshold_low && residuals[i] <= threshold) {
            mask[i] = 1;
            kept++;
        } else {
            mask[i] = 0;
        }
    }
    free(sorted);
    free(abs_dev);
    return kept;
}

static int spatial_uniform_select(
    const double *x, const double *y, int count,
    const int *valid_mask, int target_count,
    int *selected) {
    int *indices = (int *)malloc(count * sizeof(int));
    int n_valid = 0;
    for (int i = 0; i < count; i++) {
        if (valid_mask[i]) indices[n_valid++] = i;
    }
    if (n_valid <= target_count) {
        for (int i = 0; i < n_valid; i++) selected[i] = indices[i];
        free(indices);
        return n_valid;
    }
    double x_min = x[indices[0]], x_max = x[indices[0]];
    double y_min = y[indices[0]], y_max = y[indices[0]];
    for (int i = 1; i < n_valid; i++) {
        if (x[indices[i]] < x_min) x_min = x[indices[i]];
        if (x[indices[i]] > x_max) x_max = x[indices[i]];
        if (y[indices[i]] < y_min) y_min = y[indices[i]];
        if (y[indices[i]] > y_max) y_max = y[indices[i]];
    }
    int grid_n = (int)sqrt((double)target_count);
    if (grid_n < 1) grid_n = 1;
    double cell_w = (x_max - x_min + 1.0) / grid_n;
    double cell_h = (y_max - y_min + 1.0) / grid_n;
    int total_cells = grid_n * grid_n;
    int *cell_count = (int *)calloc(total_cells, sizeof(int));
    int *cell_first = (int *)malloc(total_cells * sizeof(int));
    for (int i = 0; i < total_cells; i++) cell_first[i] = -1;
    for (int i = 0; i < n_valid; i++) {
        int gx = (int)((x[indices[i]] - x_min) / cell_w);
        int gy = (int)((y[indices[i]] - y_min) / cell_h);
        if (gx >= grid_n) gx = grid_n - 1;
        if (gy >= grid_n) gy = grid_n - 1;
        int ci = gy * grid_n + gx;
        cell_count[ci]++;
        if (cell_first[ci] == -1) cell_first[ci] = indices[i];
    }
    int sel_count = 0;
    for (int ci = 0; ci < total_cells && sel_count < target_count; ci++) {
        if (cell_first[ci] >= 0) {
            selected[sel_count++] = cell_first[ci];
        }
    }
    if (sel_count < target_count) {
        for (int i = 0; i < n_valid && sel_count < target_count; i++) {
            int already = 0;
            for (int j = 0; j < sel_count; j++) {
                if (selected[j] == indices[i]) { already = 1; break; }
            }
            if (!already) selected[sel_count++] = indices[i];
        }
    }
    free(indices);
    free(cell_count);
    free(cell_first);
    return sel_count;
}

int psolve_fine_solve(
    PSolveHandleInternal *handle,
    const uint16_t *image, int width, int height,
    const PSolveCoarseResult *coarse,
    const PSolveImageData *img_data,
    PSolveFineResult *result) {
    if (!coarse || !img_data || !result) {
        PSLOG_E("psolve_fine_solve: null input");
        return PSOLVE_ERR_INTERNAL;
    }
    if (!coarse->matched_stars || coarse->matched_count < 6) {
        PSLOG_W("psolve_fine_solve: matched=%d need>=6", coarse->matched_count);
        return PSOLVE_ERR_NOT_ENOUGH;
    }
    int matched = coarse->matched_count;
    PSLOG_I("psolve_fine_solve: start matched=%d", matched);
    memset(result, 0, sizeof(PSolveFineResult));

    double *cat_plane_x = (double *)malloc(matched * sizeof(double));
    double *cat_plane_y = (double *)malloc(matched * sizeof(double));
    double *img_x = (double *)malloc(matched * sizeof(double));
    double *img_y = (double *)malloc(matched * sizeof(double));
    double *res_x = (double *)malloc(matched * sizeof(double));
    double *res_y = (double *)malloc(matched * sizeof(double));
    if (!cat_plane_x || !cat_plane_y || !img_x || !img_y || !res_x || !res_y) {
        free(cat_plane_x); free(cat_plane_y);
        free(img_x); free(img_y);
        free(res_x); free(res_y);
        PSLOG_E("psolve_fine_solve: alloc star arrays failed");
        return PSOLVE_ERR_INTERNAL;
    }

    double ra0 = img_data->center_ra;
    double dec0 = img_data->center_dec;
    double scale_arcsec = psolve_compute_scale(img_data->focal_length_mm, img_data->pixel_size_um);
    double deg_to_px = 3600.0 / scale_arcsec;
    double half_w = width / 2.0;
    double half_h = height / 2.0;

    for (int i = 0; i < matched; i++) {
        img_x[i] = coarse->matched_stars[i].img_x - half_w;
        img_y[i] = half_h - coarse->matched_stars[i].img_y;
        psolve_sky_to_plane(coarse->matched_stars[i].cat_ra,
                            coarse->matched_stars[i].cat_dec,
                            ra0, dec0,
                            &cat_plane_x[i], &cat_plane_y[i]);
        cat_plane_x[i] *= deg_to_px;
        cat_plane_y[i] *= deg_to_px;
    }

    PSolveAffine affine;
    psolve_compute_affine(cat_plane_x, cat_plane_y, img_x, img_y, matched, &affine);

    double *total_res_x = (double *)malloc(matched * sizeof(double));
    double *total_res_y = (double *)malloc(matched * sizeof(double));
    for (int i = 0; i < matched; i++) {
        double pred_x = affine.a0 + affine.a1 * cat_plane_x[i] + affine.a2 * cat_plane_y[i];
        double pred_y = affine.b0 + affine.b1 * cat_plane_x[i] + affine.b2 * cat_plane_y[i];
        total_res_x[i] = img_x[i] - pred_x;
        total_res_y[i] = img_y[i] - pred_y;
    }

    double *total_dist = (double *)malloc(matched * sizeof(double));
    for (int i = 0; i < matched; i++) {
        total_dist[i] = sqrt(total_res_x[i] * total_res_x[i] + total_res_y[i] * total_res_y[i]);
    }

    int *sigma_mask = (int *)malloc(matched * sizeof(int));
    int sigma_kept = sigma_clip_filter(total_dist, matched, PSOLVE_FINE_SIGMA_CLIP, sigma_mask);
    PSLOG_I("Sigma clip (%.1f sigma): %d / %d kept", PSOLVE_FINE_SIGMA_CLIP, sigma_kept, matched);

    int rbf_target = 500;
    if (rbf_target > sigma_kept) rbf_target = sigma_kept;

    int *selected_indices = (int *)malloc(matched * sizeof(int));
    int selected_count = spatial_uniform_select(img_x, img_y, matched, sigma_mask, rbf_target, selected_indices);
    PSLOG_I("Spatial uniform select: %d stars from %d sigma-clipped (target=%d)",
            selected_count, sigma_kept, rbf_target);

    int *final_mask = (int *)calloc(matched, sizeof(int));
    for (int i = 0; i < selected_count; i++) {
        final_mask[selected_indices[i]] = 1;
    }

    int rbf_count = selected_count;
    double *rbf_img_x = (double *)malloc(rbf_count * sizeof(double));
    double *rbf_img_y = (double *)malloc(rbf_count * sizeof(double));
    double *rbf_res_x = (double *)malloc(rbf_count * sizeof(double));
    double *rbf_res_y = (double *)malloc(rbf_count * sizeof(double));
    for (int i = 0; i < rbf_count; i++) {
        int idx = selected_indices[i];
        rbf_img_x[i] = img_x[idx];
        rbf_img_y[i] = img_y[idx];
        rbf_res_x[i] = total_res_x[idx];
        rbf_res_y[i] = total_res_y[idx];
    }

    free(selected_indices);
    free(sigma_mask);
    free(total_dist);

    for (int i = 0; i < rbf_count; i++) {
        res_x[i] = rbf_res_x[i];
        res_y[i] = rbf_res_y[i];
    }

    PSLOG_I("psolve_fine_solve: residuals computed for %d RBF stars", rbf_count);

    int grid_nx = (int)sqrt((double)(rbf_count / 15));
    if (grid_nx < 1) grid_nx = 1;
    int grid_ny = grid_nx;
    double cell_w = (double)width / grid_nx;
    double cell_h = (double)height / grid_ny;
    double overlap = cell_w * 0.3;
    int subdomain_count = grid_nx * grid_ny;
    SubDomain *subdomains = (SubDomain *)calloc(subdomain_count, sizeof(SubDomain));
    if (!subdomains) {
        free(cat_plane_x); free(cat_plane_y);
        free(img_x); free(img_y);
        free(res_x); free(res_y);
        free(total_res_x); free(total_res_y);
        free(final_mask);
        free(rbf_img_x); free(rbf_img_y);
        free(rbf_res_x); free(rbf_res_y);
        PSLOG_E("psolve_fine_solve: alloc subdomains failed");
        return PSOLVE_ERR_INTERNAL;
    }

    for (int gy = 0; gy < grid_ny; gy++) {
        for (int gx = 0; gx < grid_nx; gx++) {
            int idx = gy * grid_nx + gx;
            SubDomain *sd = &subdomains[idx];
            sd->x_min = gx * cell_w - overlap;
            sd->x_max = (gx + 1) * cell_w + overlap;
            sd->y_min = gy * cell_h - overlap;
            sd->y_max = (gy + 1) * cell_h + overlap;
            sd->cx = (gx + 0.5) * cell_w;
            sd->cy = (gy + 0.5) * cell_h;
            sd->half_size = sqrt(cell_w * cell_w + cell_h * cell_h) * 0.5;
            sd->star_count = 0;
            sd->star_indices = NULL;
            sd->rbf_wx = NULL;
            sd->rbf_wy = NULL;
            sd->solved = 0;
        }
    }

    for (int i = 0; i < rbf_count; i++) {
        for (int s = 0; s < subdomain_count; s++) {
            SubDomain *sd = &subdomains[s];
            if (rbf_img_x[i] >= sd->x_min && rbf_img_x[i] <= sd->x_max &&
                rbf_img_y[i] >= sd->y_min && rbf_img_y[i] <= sd->y_max) {
                sd->star_count++;
            }
        }
    }

    for (int s = 0; s < subdomain_count; s++) {
        if (subdomains[s].star_count > 0) {
            subdomains[s].star_indices = (int *)malloc(subdomains[s].star_count * sizeof(int));
        }
    }

    {
        int *counters = (int *)calloc(subdomain_count, sizeof(int));
        for (int i = 0; i < rbf_count; i++) {
            for (int s = 0; s < subdomain_count; s++) {
                SubDomain *sd = &subdomains[s];
                if (rbf_img_x[i] >= sd->x_min && rbf_img_x[i] <= sd->x_max &&
                    rbf_img_y[i] >= sd->y_min && rbf_img_y[i] <= sd->y_max) {
                    sd->star_indices[counters[s]++] = i;
                }
            }
        }
        free(counters);
    }

    PSLOG_I("psolve_fine_solve: domain decomposition grid=%dx%d subdomains=%d rbf_stars=%d",
            grid_nx, grid_ny, subdomain_count, rbf_count);

    for (int s = 0; s < subdomain_count; s++) {
        SubDomain *sd = &subdomains[s];
        if (sd->star_count < 3) {
            sd->solved = 0;
            PSLOG_D("psolve_fine_solve: subdomain %d skipped stars=%d", s, sd->star_count);
            continue;
        }
        sd->rbf_wx = (double *)calloc(sd->star_count, sizeof(double));
        sd->rbf_wy = (double *)calloc(sd->star_count, sizeof(double));
        if (!sd->rbf_wx || !sd->rbf_wy) {
            free(sd->rbf_wx); free(sd->rbf_wy);
            sd->rbf_wx = NULL; sd->rbf_wy = NULL;
            sd->solved = 0;
            continue;
        }
        int ret = solve_subdomain_rbf(rbf_img_x, rbf_img_y, res_x, res_y,
                                       sd->star_indices, sd->star_count,
                                       sd->rbf_wx, sd->rbf_wy,
                                       sd->poly_cx, sd->poly_cy);
        if (ret == 0) {
            sd->solved = 1;
            PSLOG_D("psolve_fine_solve: subdomain %d solved stars=%d", s, sd->star_count);
        } else {
            sd->solved = 0;
            PSLOG_W("psolve_fine_solve: subdomain %d RBF solve failed stars=%d", s, sd->star_count);
        }
    }

    int res_grid_w = 64;
    int res_grid_h = 64;
    double *residual_grid_x = (double *)malloc(res_grid_w * res_grid_h * sizeof(double));
    double *residual_grid_y = (double *)malloc(res_grid_w * res_grid_h * sizeof(double));
    if (!residual_grid_x || !residual_grid_y) {
        free(residual_grid_x); free(residual_grid_y);
        for (int s = 0; s < subdomain_count; s++) {
            free(subdomains[s].star_indices);
            free(subdomains[s].rbf_wx);
            free(subdomains[s].rbf_wy);
        }
        free(subdomains);
        free(cat_plane_x); free(cat_plane_y);
        free(img_x); free(img_y);
        free(res_x); free(res_y);
        free(total_res_x); free(total_res_y);
        free(final_mask);
        free(rbf_img_x); free(rbf_img_y);
        free(rbf_res_x); free(rbf_res_y);
        PSLOG_E("psolve_fine_solve: alloc residual grid failed");
        return PSOLVE_ERR_INTERNAL;
    }

    for (int gy = 0; gy < res_grid_h; gy++) {
        for (int gx = 0; gx < res_grid_w; gx++) {
            double px = (gx + 0.5) * width / res_grid_w - half_w;
            double py = half_h - (gy + 0.5) * height / res_grid_h;
            double rx, ry;
            eval_rbf_blended(subdomains, subdomain_count,
                             rbf_img_x, rbf_img_y, px, py, &rx, &ry);
            int gi = gy * res_grid_w + gx;
            residual_grid_x[gi] = rx;
            residual_grid_y[gi] = ry;
        }
    }

    PSLOG_I("psolve_fine_solve: residual grid generated %dx%d", res_grid_w, res_grid_h);

    double sum_dx2 = 0, sum_dy2 = 0;
    int rms_count = 0;
    for (int i = 0; i < matched; i++) {
        double pred_x = affine.a0 + affine.a1 * cat_plane_x[i] + affine.a2 * cat_plane_y[i];
        double pred_y = affine.b0 + affine.b1 * cat_plane_x[i] + affine.b2 * cat_plane_y[i];
        double rx, ry;
        eval_rbf_blended(subdomains, subdomain_count,
                         rbf_img_x, rbf_img_y, pred_x, pred_y, &rx, &ry);
        double corrected_x = pred_x + rx;
        double corrected_y = pred_y + ry;
        double err_x = img_x[i] - corrected_x;
        double err_y = img_y[i] - corrected_y;
        sum_dx2 += err_x * err_x;
        sum_dy2 += err_y * err_y;
        rms_count++;
    }
    result->rms_x = sqrt(sum_dx2 / rms_count);
    result->rms_y = sqrt(sum_dy2 / rms_count);
    result->rms_total = sqrt((result->rms_x * result->rms_x + result->rms_y * result->rms_y) / 2.0);
    result->matched_count = matched;
    result->subdomain_count = subdomain_count;
    result->residual_grid_x = residual_grid_x;
    result->residual_grid_y = residual_grid_y;
    result->grid_w = res_grid_w;
    result->grid_h = res_grid_h;

    PSLOG_I("psolve_fine_solve: rms_x=%.4f rms_y=%.4f rms_total=%.4f (all %d stars for RMS)",
            result->rms_x, result->rms_y, result->rms_total, rms_count);

    PSolveWCS *wcs = &result->wcs;
    wcs->crpix1 = width / 2.0 + 0.5;
    wcs->crpix2 = height / 2.0 + 0.5;
    wcs->crval1 = img_data->center_ra;
    wcs->crval2 = img_data->center_dec;
    double scale_arcsec_px = sqrt(affine.a1 * affine.a1 + affine.b1 * affine.b1);
    double scale_deg_px = scale_arcsec_px / 3600.0;
    wcs->cd1_1 = affine.a1 * scale_deg_px;
    wcs->cd1_2 = affine.a2 * scale_deg_px;
    wcs->cd2_1 = affine.b1 * scale_deg_px;
    wcs->cd2_2 = affine.b2 * scale_deg_px;
    wcs->cdelt1 = scale_deg_px;
    wcs->cdelt2 = scale_deg_px;
    strcpy(wcs->ctype1, "RA---TAN");
    strcpy(wcs->ctype2, "DEC--TAN");
    strcpy(wcs->radesys, "ICRS");
    wcs->equinox = 2000.0;

    PSLOG_I("psolve_fine_solve: WCS crpix=(%.2f,%.2f) crval=(%.6f,%.6f) scale=%.4f arcsec/px",
            wcs->crpix1, wcs->crpix2, wcs->crval1, wcs->crval2, scale_arcsec_px);

    for (int s = 0; s < subdomain_count; s++) {
        free(subdomains[s].star_indices);
        free(subdomains[s].rbf_wx);
        free(subdomains[s].rbf_wy);
    }
    free(subdomains);
    free(cat_plane_x); free(cat_plane_y);
    free(img_x); free(img_y);
    free(res_x); free(res_y);
    free(total_res_x); free(total_res_y);
    free(final_mask);
    free(rbf_img_x); free(rbf_img_y);
    free(rbf_res_x); free(rbf_res_y);

    return PSOLVE_OK;
}

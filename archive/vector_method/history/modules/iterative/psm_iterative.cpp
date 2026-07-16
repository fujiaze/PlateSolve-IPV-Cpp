#include "psm_iterative.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEG2RAD (M_PI / 180.0)
#define RAD2DEG (180.0 / M_PI)

typedef struct {
    int *heads;
    int *next;
    int gw, gh;
    double x0, y0;
    double cell_size;
} MatchGrid;

static void build_grid(const double *x, const double *y, int n,
    double cell_size, MatchGrid *g)
{
    double xmin = x[0], xmax = x[0], ymin = y[0], ymax = y[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < xmin) xmin = x[i];
        if (x[i] > xmax) xmax = x[i];
        if (y[i] < ymin) ymin = y[i];
        if (y[i] > ymax) ymax = y[i];
    }
    g->x0 = xmin - cell_size * 0.1;
    g->y0 = ymin - cell_size * 0.1;
    g->cell_size = cell_size;
    g->gw = (int)((xmax - g->x0) / cell_size) + 1;
    g->gh = (int)((ymax - g->y0) / cell_size) + 1;
    if (g->gw < 1) g->gw = 1;
    if (g->gh < 1) g->gh = 1;
    int tc = g->gw * g->gh;
    g->heads = (int *)malloc((size_t)tc * sizeof(int));
    g->next = (int *)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < tc; i++) g->heads[i] = -1;
    for (int i = 0; i < n; i++) {
        int cx = (int)((x[i] - g->x0) / cell_size);
        int cy = (int)((y[i] - g->y0) / cell_size);
        if (cx < 0) cx = 0;
        if (cx >= g->gw) cx = g->gw - 1;
        if (cy < 0) cy = 0;
        if (cy >= g->gh) cy = g->gh - 1;
        g->next[i] = g->heads[cy * g->gw + cx];
        g->heads[cy * g->gw + cx] = i;
    }
}

static void free_grid(MatchGrid *g)
{
    free(g->heads);
    free(g->next);
}

static int nearest_in_grid(const MatchGrid *g, const double *cat_x, const double *cat_y,
    double qx, double qy, double max_dist2, double *out_dist2)
{
    int cx = (int)((qx - g->x0) / g->cell_size);
    int cy = (int)((qy - g->y0) / g->cell_size);
    double best_d2 = max_dist2;
    int best_i = -1;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= g->gw || ny < 0 || ny >= g->gh) continue;
            for (int k = g->heads[ny * g->gw + nx]; k != -1; k = g->next[k]) {
                double d2 = (cat_x[k] - qx) * (cat_x[k] - qx)
                          + (cat_y[k] - qy) * (cat_y[k] - qy);
                if (d2 < best_d2) {
                    best_d2 = d2;
                    best_i = k;
                }
            }
        }
    }
    if (out_dist2) *out_dist2 = best_d2;
    return best_i;
}

static int fit_affine(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst,
    int n, PSMAffine *out)
{
    if (n < 3) return PSM_ERR_NOT_ENOUGH;
    double sx = 0, sy = 0, sx2 = 0, sy2 = 0, sxy = 0;
    double sdx = 0, sdy = 0, sdx_x = 0, sdy_x = 0, sdx_y = 0, sdy_y = 0;
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
    for (int col = 0; col < 3; col++) {
        int pivot = col;
        double max_val = fabs(A[col][col]);
        for (int row = col + 1; row < 3; row++) {
            if (fabs(A[row][col]) > max_val) {
                max_val = fabs(A[row][col]);
                pivot = row;
            }
        }
        if (max_val < 1e-20) return PSM_ERR_SINGULAR;
        if (pivot != col) {
            for (int j = 0; j < 3; j++) {
                double tmp = A[col][j];
                A[col][j] = A[pivot][j];
                A[pivot][j] = tmp;
            }
            double tmp = bx[col]; bx[col] = bx[pivot]; bx[pivot] = tmp;
            tmp = by[col]; by[col] = by[pivot]; by[pivot] = tmp;
        }
        double inv_pivot = 1.0 / A[col][col];
        for (int row = col + 1; row < 3; row++) {
            double factor = A[row][col] * inv_pivot;
            for (int j = col; j < 3; j++) {
                A[row][j] -= factor * A[col][j];
            }
            bx[row] -= factor * bx[col];
            by[row] -= factor * by[col];
        }
    }
    double ax[3], ay[3];
    for (int i = 2; i >= 0; i--) {
        double sumx = bx[i], sumy = by[i];
        for (int j = i + 1; j < 3; j++) {
            sumx -= A[i][j] * ax[j];
            sumy -= A[i][j] * ay[j];
        }
        ax[i] = sumx / A[i][i];
        ay[i] = sumy / A[i][i];
    }
    out->a0 = ax[0];
    out->a1 = ax[1];
    out->a2 = ax[2];
    out->b0 = ay[0];
    out->b1 = ay[1];
    out->b2 = ay[2];
    return PSM_OK;
}

PSM_EXPORT void psm_sky_to_plane(double ra, double dec, double ra0, double dec0,
    double *x, double *y)
{
    double ra_rad = ra * DEG2RAD;
    double dec_rad = dec * DEG2RAD;
    double ra0_rad = ra0 * DEG2RAD;
    double dec0_rad = dec0 * DEG2RAD;
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
    *x = cos_dec * sin_dra / cos_c * RAD2DEG;
    *y = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_dra) / cos_c * RAD2DEG;
}

PSM_EXPORT void psm_plane_to_sky(double x, double y, double ra0, double dec0,
    double *ra, double *dec)
{
    double x_rad = x * DEG2RAD;
    double y_rad = y * DEG2RAD;
    double ra0_rad = ra0 * DEG2RAD;
    double dec0_rad = dec0 * DEG2RAD;
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
    *ra = ra_rad * RAD2DEG;
    *dec = dec_rad * RAD2DEG;
    while (*ra < 0.0) *ra += 360.0;
    while (*ra >= 360.0) *ra -= 360.0;
}

PSM_EXPORT int psm_project_stars(const double *ra_arr, const double *dec_arr, int count,
    double ra0, double dec0, double **out_x, double **out_y)
{
    if (!ra_arr || !dec_arr || count < 1 || !out_x || !out_y) {
        return PSM_ERR_INVALID_PARAM;
    }
    *out_x = (double *)malloc((size_t)count * sizeof(double));
    *out_y = (double *)malloc((size_t)count * sizeof(double));
    if (!*out_x || !*out_y) {
        free(*out_x);
        free(*out_y);
        *out_x = NULL;
        *out_y = NULL;
        return PSM_ERR_ALLOC;
    }
    for (int i = 0; i < count; i++) {
        psm_sky_to_plane(ra_arr[i], dec_arr[i], ra0, dec0,
            &(*out_x)[i], &(*out_y)[i]);
    }
    return PSM_OK;
}

PSM_EXPORT void psm_free_projected(double *arr)
{
    free(arr);
}

PSM_EXPORT int psm_match_nearest(const double *det_x, const double *det_y, int ndet,
    const double *cat_x, const double *cat_y, int ncat, double match_radius,
    int **out_match_idx, double **out_match_dist)
{
    if (!det_x || !det_y || ndet < 1 || !cat_x || !cat_y || ncat < 1 ||
        match_radius <= 0.0 || !out_match_idx || !out_match_dist) {
        return PSM_ERR_INVALID_PARAM;
    }
    *out_match_idx = (int *)malloc((size_t)ndet * sizeof(int));
    *out_match_dist = (double *)malloc((size_t)ndet * sizeof(double));
    if (!*out_match_idx || !*out_match_dist) {
        free(*out_match_idx);
        free(*out_match_dist);
        *out_match_idx = NULL;
        *out_match_dist = NULL;
        return PSM_ERR_ALLOC;
    }
    double mr2 = match_radius * match_radius;
    MatchGrid grid;
    build_grid(cat_x, cat_y, ncat, match_radius, &grid);
    int nmatch = 0;
    for (int i = 0; i < ndet; i++) {
        double d2;
        int idx = nearest_in_grid(&grid, cat_x, cat_y, det_x[i], det_y[i], mr2, &d2);
        (*out_match_idx)[i] = idx;
        if (idx >= 0) {
            (*out_match_dist)[i] = sqrt(d2);
            nmatch++;
        } else {
            (*out_match_dist)[i] = 0.0;
        }
    }
    free_grid(&grid);
    return nmatch;
}

PSM_EXPORT void psm_free_match(int *match_idx, double *match_dist)
{
    free(match_idx);
    free(match_dist);
}

PSM_EXPORT int psm_iterate_center(const double *det_x, const double *det_y, int ndet,
    const double *cat_ra, const double *cat_dec, int ncat,
    double *io_center_ra, double *io_center_dec,
    double scale_arcsec_px, int half_w, int half_h,
    PSMAffine *io_affine, int max_iter)
{
    if (!det_x || !det_y || ndet < 3 || !cat_ra || !cat_dec || ncat < 1 ||
        !io_center_ra || !io_center_dec || scale_arcsec_px <= 0.0 ||
        !io_affine || max_iter < 1) {
        return PSM_ERR_INVALID_PARAM;
    }
    double dp = 3600.0 / scale_arcsec_px;
    double *dcx = (double *)malloc((size_t)ndet * sizeof(double));
    double *dcy = (double *)malloc((size_t)ndet * sizeof(double));
    for (int i = 0; i < ndet; i++) {
        dcx[i] = det_x[i] - (double)half_w;
        dcy[i] = (double)half_h - det_y[i];
    }
    double *ctx = NULL, *cty = NULL;
    psm_project_stars(cat_ra, cat_dec, ncat,
        *io_center_ra, *io_center_dec, &ctx, &cty);
    for (int i = 0; i < ncat; i++) {
        ctx[i] *= dp;
        cty[i] *= dp;
    }
    double *dtx = (double *)malloc((size_t)ndet * sizeof(double));
    double *dty = (double *)malloc((size_t)ndet * sizeof(double));
    int iter = 0;
    for (iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < ndet; i++) {
            dtx[i] = io_affine->a0 + io_affine->a1 * dcx[i] + io_affine->a2 * dcy[i];
            dty[i] = io_affine->b0 + io_affine->b1 * dcx[i] + io_affine->b2 * dcy[i];
        }
        int *match_idx = NULL;
        double *match_dist = NULL;
        int nm = psm_match_nearest(dtx, dty, ndet, ctx, cty, ncat, 50.0,
            &match_idx, &match_dist);
        if (nm < 3) {
            psm_free_match(match_idx, match_dist);
            break;
        }
        double ox = 0.0, oy = 0.0;
        int cnt = 0;
        for (int i = 0; i < ndet; i++) {
            if (match_idx[i] >= 0) {
                ox += ctx[match_idx[i]] - dtx[i];
                oy += cty[match_idx[i]] - dty[i];
                cnt++;
            }
        }
        if (cnt < 3) {
            psm_free_match(match_idx, match_dist);
            break;
        }
        ox /= (double)cnt;
        oy /= (double)cnt;
        psm_free_projected(ctx);
        psm_free_projected(cty);
        double nra, ndec;
        psm_plane_to_sky(ox / dp, oy / dp,
            *io_center_ra, *io_center_dec, &nra, &ndec);
        *io_center_ra = nra;
        *io_center_dec = ndec;
        psm_project_stars(cat_ra, cat_dec, ncat,
            *io_center_ra, *io_center_dec, &ctx, &cty);
        for (int i = 0; i < ncat; i++) {
            ctx[i] *= dp;
            cty[i] *= dp;
        }
        double *xs = (double *)malloc((size_t)cnt * sizeof(double));
        double *ys = (double *)malloc((size_t)cnt * sizeof(double));
        double *xd = (double *)malloc((size_t)cnt * sizeof(double));
        double *yd = (double *)malloc((size_t)cnt * sizeof(double));
        int j = 0;
        for (int i = 0; i < ndet; i++) {
            if (match_idx[i] >= 0) {
                xs[j] = dcx[i];
                ys[j] = dcy[i];
                xd[j] = ctx[match_idx[i]];
                yd[j] = cty[match_idx[i]];
                j++;
            }
        }
        psm_free_match(match_idx, match_dist);
        fit_affine(xs, ys, xd, yd, cnt, io_affine);
        free(xs);
        free(ys);
        free(xd);
        free(yd);
        double offset_arcsec = sqrt(ox * ox + oy * oy) * scale_arcsec_px;
        if (offset_arcsec < 0.01) {
            break;
        }
    }
    free(dcx);
    free(dcy);
    free(dtx);
    free(dty);
    psm_free_projected(ctx);
    psm_free_projected(cty);
    return iter;
}

#include "psm_coarse_affine.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double psm_dist2(double x1, double y1, double x2, double y2)
{
    double dx = x1 - x2;
    double dy = y1 - y2;
    return dx * dx + dy * dy;
}

static int psm_compare_triangle(const void *a, const void *b)
{
    const PSMTriangle *ta = (const PSMTriangle *)a;
    const PSMTriangle *tb = (const PSMTriangle *)b;
    if (ta->ba_ratio < tb->ba_ratio) return -1;
    if (ta->ba_ratio > tb->ba_ratio) return 1;
    return 0;
}

static int psm_compare_pair_img(const void *a, const void *b)
{
    const PSMStarPair *pa = (const PSMStarPair *)a;
    const PSMStarPair *pb = (const PSMStarPair *)b;
    if (pa->img_idx < pb->img_idx) return -1;
    if (pa->img_idx > pb->img_idx) return 1;
    return 0;
}

static int psm_compare_pair_cat(const void *a, const void *b)
{
    const PSMStarPair *pa = (const PSMStarPair *)a;
    const PSMStarPair *pb = (const PSMStarPair *)b;
    if (pa->cat_idx < pb->cat_idx) return -1;
    if (pa->cat_idx > pb->cat_idx) return 1;
    return 0;
}

static int psm_binary_search_first(const PSMTriangle *tris, int n, double val, double radius)
{
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (tris[mid].ba_ratio < val - radius) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

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
        for (int row = col + 1; row < 3; row++) {
            double factor = M[row][col] / M[col][col];
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

PSM_EXPORT int psm_triangle_build(const double *x, const double *y, int n,
    int nbright, PSMTriangle **out_tris, int *out_count)
{
    if (!x || !y || n < 3 || nbright < 3 || !out_tris || !out_count) {
        return PSM_ERR_INVALID_PARAM;
    }
    int m = (nbright < n) ? nbright : n;
    int max_tris = m * (m - 1) * (m - 2) / 6;
    if (max_tris == 0) {
        return PSM_ERR_NOT_ENOUGH;
    }
    PSMTriangle *tris = (PSMTriangle *)malloc((size_t)max_tris * sizeof(PSMTriangle));
    if (!tris) {
        return PSM_ERR_ALLOC;
    }
    int count = 0;
    for (int i = 0; i < m - 2; i++) {
        for (int j = i + 1; j < m - 1; j++) {
            for (int k = j + 1; k < m; k++) {
                double dx_ij = x[i] - x[j];
                double dy_ij = y[i] - y[j];
                double dx_ik = x[i] - x[k];
                double dy_ik = y[i] - y[k];
                double dx_jk = x[j] - x[k];
                double dy_jk = y[j] - y[k];
                double len_c2 = dx_ij * dx_ij + dy_ij * dy_ij;
                double len_b2 = dx_ik * dx_ik + dy_ik * dy_ik;
                double len_a2 = dx_jk * dx_jk + dy_jk * dy_jk;
                if (len_a2 < 1e-12 || len_b2 < 1e-12 || len_c2 < 1e-12) {
                    continue;
                }
                double len_a = sqrt(len_a2);
                double len_b = sqrt(len_b2);
                double len_c = sqrt(len_c2);
                double ba_ratio = len_b / len_a;
                double ca_ratio = len_c / len_a;
                double cos_angle_a = (len_b2 + len_c2 - len_a2) / (2.0 * len_b * len_c);
                if (cos_angle_a < -1.0) cos_angle_a = -1.0;
                if (cos_angle_a > 1.0) cos_angle_a = 1.0;
                double side_a_angle = acos(cos_angle_a);
                tris[count].a_idx = i;
                tris[count].b_idx = j;
                tris[count].c_idx = k;
                tris[count].ba_ratio = ba_ratio;
                tris[count].ca_ratio = ca_ratio;
                tris[count].side_a_angle = side_a_angle;
                tris[count].side_a_length = len_a;
                count++;
            }
        }
    }
    if (count == 0) {
        free(tris);
        return PSM_ERR_NOT_ENOUGH;
    }
    qsort(tris, (size_t)count, sizeof(PSMTriangle), psm_compare_triangle);
    *out_tris = tris;
    *out_count = count;
    return PSM_OK;
}

PSM_EXPORT int psm_triangle_match(const PSMTriangle *tris_a, int na,
    const PSMTriangle *tris_b, int nb, double radius,
    double min_scale, double max_scale,
    PSMStarPair **out_pairs, int *out_pair_count)
{
    if (!tris_a || na < 1 || !tris_b || nb < 1 || radius <= 0.0 ||
        !out_pairs || !out_pair_count) {
        return PSM_ERR_INVALID_PARAM;
    }
    if (min_scale <= 0.0) min_scale = 0.1;
    if (max_scale <= 0.0 || max_scale < min_scale) max_scale = 10.0;

    int max_img = 0, max_cat = 0;
    for (int i = 0; i < na; i++) {
        if (tris_a[i].a_idx > max_img) max_img = tris_a[i].a_idx;
        if (tris_a[i].b_idx > max_img) max_img = tris_a[i].b_idx;
        if (tris_a[i].c_idx > max_img) max_img = tris_a[i].c_idx;
    }
    for (int i = 0; i < nb; i++) {
        if (tris_b[i].a_idx > max_cat) max_cat = tris_b[i].a_idx;
        if (tris_b[i].b_idx > max_cat) max_cat = tris_b[i].b_idx;
        if (tris_b[i].c_idx > max_cat) max_cat = tris_b[i].c_idx;
    }
    max_img++;
    max_cat++;

    int *votes = (int *)calloc((size_t)(max_img * max_cat), sizeof(int));
    if (!votes) {
        return PSM_ERR_ALLOC;
    }
    int max_pairs_per_tri = 64;
    PSMStarPair *temp_pairs = (PSMStarPair *)malloc(
        (size_t)na * (size_t)max_pairs_per_tri * sizeof(PSMStarPair));
    if (!temp_pairs) {
        free(votes);
        return PSM_ERR_ALLOC;
    }
    int temp_count = 0;
    double angle_tol = 0.2;
    double ratio_tol = radius;

    for (int i = 0; i < na; i++) {
        double ba = tris_a[i].ba_ratio;
        int start = psm_binary_search_first(tris_b, nb, ba, ratio_tol);
        int candidates = 0;
        for (int j = start; j < nb; j++) {
            if (tris_b[j].ba_ratio > ba + ratio_tol) break;
            double db = fabs(tris_b[j].ba_ratio - ba);
            double dc = fabs(tris_b[j].ca_ratio - tris_a[i].ca_ratio);
            if (db > ratio_tol || dc > ratio_tol) continue;
            double da = fabs(tris_b[j].side_a_angle - tris_a[i].side_a_angle);
            if (da > angle_tol) continue;
            double scale = tris_b[j].side_a_length / tris_a[i].side_a_length;
            if (scale < min_scale || scale > max_scale) continue;
            temp_pairs[temp_count].img_idx = tris_a[i].a_idx;
            temp_pairs[temp_count].cat_idx = tris_b[j].a_idx;
            temp_count++;
            temp_pairs[temp_count].img_idx = tris_a[i].b_idx;
            temp_pairs[temp_count].cat_idx = tris_b[j].b_idx;
            temp_count++;
            temp_pairs[temp_count].img_idx = tris_a[i].c_idx;
            temp_pairs[temp_count].cat_idx = tris_b[j].c_idx;
            temp_count++;
            candidates++;
            if (temp_count >= na * max_pairs_per_tri) break;
        }
    }
    if (temp_count == 0) {
        free(votes);
        free(temp_pairs);
        if (out_pairs) *out_pairs = NULL;
        if (out_pair_count) *out_pair_count = 0;
        return PSM_ERR_NO_MATCH;
    }
    for (int i = 0; i < temp_count; i++) {
        int idx = temp_pairs[i].img_idx * max_cat + temp_pairs[i].cat_idx;
        votes[idx]++;
    }

    PSMStarPair *uniq = (PSMStarPair *)malloc((size_t)temp_count * sizeof(PSMStarPair));
    if (!uniq) {
        free(votes);
        free(temp_pairs);
        return PSM_ERR_ALLOC;
    }
    int uniq_count = 0;
    for (int i = 0; i < temp_count; i++) {
        int idx = temp_pairs[i].img_idx * max_cat + temp_pairs[i].cat_idx;
        int already = 0;
        for (int k = 0; k < uniq_count; k++) {
            if (uniq[k].img_idx == temp_pairs[i].img_idx &&
                uniq[k].cat_idx == temp_pairs[i].cat_idx) {
                already = 1;
                break;
            }
        }
        if (!already) {
            uniq[uniq_count] = temp_pairs[i];
            uniq_count++;
        }
    }
    free(temp_pairs);

    qsort(uniq, (size_t)uniq_count, sizeof(PSMStarPair), psm_compare_pair_img);
    PSMStarPair *bidirectional = (PSMStarPair *)malloc(
        (size_t)uniq_count * sizeof(PSMStarPair));
    if (!bidirectional) {
        free(votes);
        free(uniq);
        return PSM_ERR_ALLOC;
    }
    int bi_count = 0;
    {
        int i = 0;
        while (i < uniq_count) {
            int img_i = uniq[i].img_idx;
            int best_j = i;
            int best_votes = 0;
            int start = i;
            while (i < uniq_count && uniq[i].img_idx == img_i) {
                int idx = uniq[i].img_idx * max_cat + uniq[i].cat_idx;
                if (votes[idx] > best_votes) {
                    best_votes = votes[idx];
                    best_j = i;
                }
                i++;
            }
            bidirectional[bi_count] = uniq[best_j];
            bi_count++;
        }
    }

    qsort(bidirectional, (size_t)bi_count, sizeof(PSMStarPair), psm_compare_pair_cat);
    PSMStarPair *final = (PSMStarPair *)malloc((size_t)bi_count * sizeof(PSMStarPair));
    if (!final) {
        free(votes);
        free(uniq);
        free(bidirectional);
        return PSM_ERR_ALLOC;
    }
    int final_count = 0;
    {
        int i = 0;
        while (i < bi_count) {
            int cat_i = bidirectional[i].cat_idx;
            int best_j = i;
            int best_votes = 0;
            while (i < bi_count && bidirectional[i].cat_idx == cat_i) {
                int idx = bidirectional[i].img_idx * max_cat + bidirectional[i].cat_idx;
                if (votes[idx] > best_votes) {
                    best_votes = votes[idx];
                    best_j = i;
                }
                i++;
            }
            final[final_count] = bidirectional[best_j];
            final_count++;
        }
    }

    free(votes);
    free(uniq);
    free(bidirectional);

    if (final_count < 3) {
        free(final);
        if (out_pairs) *out_pairs = NULL;
        if (out_pair_count) *out_pair_count = 0;
        return PSM_ERR_NO_MATCH;
    }

    PSMStarPair *result = (PSMStarPair *)realloc(final, (size_t)final_count * sizeof(PSMStarPair));
    if (!result) result = final;

    *out_pairs = result;
    *out_pair_count = final_count;
    return PSM_OK;
}

PSM_EXPORT int psm_affine_compute(const double *x_src, const double *y_src,
    const double *x_dst, const double *y_dst,
    int n, PSMAffine *out_affine)
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

PSM_EXPORT void psm_free_triangles(PSMTriangle *tris)
{
    free(tris);
}

PSM_EXPORT void psm_free_pairs(PSMStarPair *pairs)
{
    free(pairs);
}

/* ═══════════════════════════════════════════════════════════════
 * 中心扩散搜索 + 三角匹配
 * ═══════════════════════════════════════════════════════════════ */

/* Gnomonic TAN 正投影: (ra,dec) -> 切平面 (x,y) 度 */
static void psm_sky_to_plane(double ra, double dec,
    double ra0, double dec0, double *x, double *y)
{
    double cos_dec = cos(dec);
    double sin_dec = sin(dec);
    double cos_dec0 = cos(dec0);
    double sin_dec0 = sin(dec0);
    double ra_diff = ra - ra0;
    double cos_ra_diff = cos(ra_diff);
    double sin_ra_diff = sin(ra_diff);
    double cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff;
    if (cos_c < 1e-10) {
        *x = 1e30;
        *y = 1e30;
        return;
    }
    *x = cos_dec * sin_ra_diff / cos_c;
    *y = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c;
}

/* 在给定中心下，将cat星投影到切平面并尝试三角匹配
 * 返回匹配对数，及匹配星对数组(pairs指向内部malloc内存)
 * 注意：pairs中cat_idx为原始星表索引(非投影过滤后的索引) */
static int psm_try_center(
    const double *det_x, const double *det_y, int ndet,
    const double *cat_ra, const double *cat_dec, int ncat,
    double center_ra, double center_dec,
    int nbright, double match_radius,
    double min_scale, double max_scale,
    double deg_to_px,
    PSMStarPair **out_pairs)
{
    /* 投影所有cat星到切平面，同时记录原始索引映射 */
    double *proj_x = (double *)malloc((size_t)ncat * sizeof(double));
    double *proj_y = (double *)malloc((size_t)ncat * sizeof(double));
    int *orig_idx = (int *)malloc((size_t)ncat * sizeof(int));
    if (!proj_x || !proj_y || !orig_idx) {
        free(proj_x); free(proj_y); free(orig_idx);
        return 0;
    }
    double ra0_rad = center_ra * (M_PI / 180.0);
    double dec0_rad = center_dec * (M_PI / 180.0);
    int nproj = 0;
    for (int i = 0; i < ncat; i++) {
        double x, y;
        psm_sky_to_plane(cat_ra[i] * (M_PI / 180.0),
                         cat_dec[i] * (M_PI / 180.0),
                         ra0_rad, dec0_rad, &x, &y);
        if (x < 1e29 && y < 1e29) {
            proj_x[nproj] = x * deg_to_px;
            proj_y[nproj] = y * deg_to_px;
            orig_idx[nproj] = i;
            nproj++;
        }
    }

    if (nproj < nbright) {
        free(proj_x); free(proj_y); free(orig_idx);
        return 0;
    }

    /* 构建三角形 */
    PSMTriangle *tris_img = NULL;
    int ntris_img = 0;
    int ret = psm_triangle_build(det_x, det_y, ndet, nbright,
                                  &tris_img, &ntris_img);
    if (ret != PSM_OK || ntris_img < 10) {
        free(proj_x); free(proj_y); free(orig_idx);
        return 0;
    }

    PSMTriangle *tris_cat = NULL;
    int ntris_cat = 0;
    ret = psm_triangle_build(proj_x, proj_y, nproj, nbright,
                              &tris_cat, &ntris_cat);
    free(proj_x); free(proj_y);
    if (ret != PSM_OK || ntris_cat < 10) {
        psm_free_triangles(tris_img);
        free(orig_idx);
        return 0;
    }

    /* 三角匹配 */
    PSMStarPair *pairs = NULL;
    int npairs = 0;
    ret = psm_triangle_match(tris_img, ntris_img, tris_cat, ntris_cat,
                              match_radius, min_scale, max_scale,
                              &pairs, &npairs);

    psm_free_triangles(tris_img);
    psm_free_triangles(tris_cat);

    if (ret != PSM_OK || npairs < 3) {
        free(orig_idx);
        return 0;
    }

    /* 将cat_idx从过滤后的索引转换为原始星表索引 */
    for (int i = 0; i < npairs; i++) {
        pairs[i].cat_idx = orig_idx[pairs[i].cat_idx];
    }
    free(orig_idx);

    *out_pairs = pairs;
    return npairs;
}

PSM_EXPORT int psm_search_center_and_match(
    const double *det_x, const double *det_y, int ndet,
    const double *cat_ra, const double *cat_dec, int ncat,
    double init_ra, double init_dec,
    int nbright, double match_radius,
    double min_scale, double max_scale,
    double deg_to_px, double half_w, double half_h,
    double *out_center_ra, double *out_center_dec,
    PSMStarPair **out_pairs, int *out_pair_count)
{
    if (!det_x || !det_y || ndet < 3 ||
        !cat_ra || !cat_dec || ncat < 3 ||
        !out_center_ra || !out_center_dec ||
        !out_pairs || !out_pair_count) {
        return PSM_ERR_INVALID_PARAM;
    }

    double best_ra = init_ra;
    double best_dec = init_dec;
    PSMStarPair *best_pairs = NULL;
    int best_npairs = 0;

    /* 四级搜索网格：从粗到细 */
    struct { double step; int n; } levels[] = {
        { 2.0,  7 },   /* 2°步长，7×7=49点，覆盖±6° */
        { 0.5,  7 },   /* 0.5°步长，细化 */
        { 0.1,  5 },   /* 0.1°步长，精确定位 */
        { 0.02, 3 },   /* 0.02°步长，精确定位 */
    };
    int nlevels = sizeof(levels) / sizeof(levels[0]);

    for (int lev = 0; lev < nlevels; lev++) {
        double step = levels[lev].step;
        int n = levels[lev].n;
        int half = n / 2;

        double local_best_ra = best_ra;
        double local_best_dec = best_dec;
        int local_best_n = best_npairs;

        for (int iy = -half; iy <= half; iy++) {
            for (int ix = -half; ix <= half; ix++) {
                double try_ra = best_ra + (double)ix * step;
                double try_dec = best_dec + (double)iy * step;

                PSMStarPair *pairs = NULL;
                int npairs = psm_try_center(
                    det_x, det_y, ndet,
                    cat_ra, cat_dec, ncat,
                    try_ra, try_dec,
                    nbright, match_radius,
                    min_scale, max_scale,
                    deg_to_px, &pairs);

                if (npairs > local_best_n) {
                    local_best_n = npairs;
                    local_best_ra = try_ra;
                    local_best_dec = try_dec;
                    /* 释放旧的best_pairs */
                    if (best_pairs) psm_free_pairs(best_pairs);
                    best_pairs = pairs;
                    best_npairs = npairs;
                } else {
                    if (pairs) psm_free_pairs(pairs);
                }
            }
        }

        best_ra = local_best_ra;
        best_dec = local_best_dec;
        best_npairs = local_best_n;
    }

    if (best_npairs < 3) {
        if (best_pairs) psm_free_pairs(best_pairs);
        *out_pairs = NULL;
        *out_pair_count = 0;
        return PSM_ERR_NO_MATCH;
    }

    *out_center_ra = best_ra;
    *out_center_dec = best_dec;
    *out_pairs = best_pairs;
    *out_pair_count = best_npairs;
    return PSM_OK;
}

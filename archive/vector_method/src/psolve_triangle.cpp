#include "psolve_triangle.h"
#include "psolve_log.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PSOLVE_TRI_RATIO 0.9
#define PSOLVE_TRI_RADIUS 0.01
#define PSOLVE_TRI_MIN_BA 0.1

static int compare_triangles_by_ba(const void *a, const void *b) {
    double diff = ((const PSolveTriangle *)a)->ba_ratio - ((const PSolveTriangle *)b)->ba_ratio;
    return (diff > 0) - (diff < 0);
}

static int find_ba_start(const PSolveTriangle *tris, int count, double ba_min) {
    int lo = 0, hi = count;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (tris[mid].ba_ratio < ba_min)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

int psolve_build_triangles(const double *x, const double *y, int n,
                            int nbright,
                            PSolveTriangle **out_tris, int *out_count) {
    int m = n < nbright ? n : nbright;
    int max_tri = m * (m - 1) * (m - 2) / 6;

    PSolveTriangle *tris = (PSolveTriangle *)malloc(max_tri * sizeof(PSolveTriangle));
    if (!tris) {
        *out_tris = NULL;
        *out_count = 0;
        return -1;
    }

    int count = 0;
    for (int i = 0; i < m; i++) {
        for (int j = i + 1; j < m; j++) {
            for (int k = j + 1; k < m; k++) {
                double d_ij = sqrt((x[i]-x[j])*(x[i]-x[j]) + (y[i]-y[j])*(y[i]-y[j]));
                double d_ik = sqrt((x[i]-x[k])*(x[i]-x[k]) + (y[i]-y[k])*(y[i]-y[k]));
                double d_jk = sqrt((x[j]-x[k])*(x[j]-x[k]) + (y[j]-y[k])*(y[j]-y[k]));

                double sides[3] = {d_ij, d_ik, d_jk};
                int idx_a = -1, idx_b = -1, idx_c = -1;
                int vi_a, vi_b, vi_c;

                if (sides[0] >= sides[1] && sides[0] >= sides[2]) {
                    idx_a = 0; vi_a = k;
                    if (sides[1] >= sides[2]) { idx_b = 1; idx_c = 2; vi_b = j; vi_c = i; }
                    else { idx_b = 2; idx_c = 1; vi_b = i; vi_c = j; }
                } else if (sides[1] >= sides[0] && sides[1] >= sides[2]) {
                    idx_a = 1; vi_a = j;
                    if (sides[0] >= sides[2]) { idx_b = 0; idx_c = 2; vi_b = k; vi_c = i; }
                    else { idx_b = 2; idx_c = 0; vi_b = i; vi_c = k; }
                } else {
                    idx_a = 2; vi_a = i;
                    if (sides[0] >= sides[1]) { idx_b = 0; idx_c = 1; vi_b = k; vi_c = j; }
                    else { idx_b = 1; idx_c = 0; vi_b = j; vi_c = k; }
                }

                double a_len = sides[idx_a];
                if (a_len < 1e-10) continue;

                double ba = sides[idx_b] / a_len;
                double ca = sides[idx_c] / a_len;

                if (ba > PSOLVE_TRI_RATIO) continue;
                if (ba < PSOLVE_TRI_MIN_BA) continue;

                tris[count].a_idx = vi_a;
                tris[count].b_idx = vi_b;
                tris[count].c_idx = vi_c;
                tris[count].ba_ratio = ba;
                tris[count].ca_ratio = ca;
                tris[count].side_a_angle = atan2(y[vi_c] - y[vi_b], x[vi_c] - x[vi_b]);
                tris[count].side_a_length = a_len;
                count++;
            }
        }
    }

    qsort(tris, count, sizeof(PSolveTriangle), compare_triangles_by_ba);

    *out_tris = tris;
    *out_count = count;
    PSLOG_I("Built %d triangles from %d stars (nbright=%d, pruned ba>%.1f)",
            count, m, nbright, PSOLVE_TRI_RATIO);
    return 0;
}

int psolve_match_triangles(
    const PSolveTriangle *tris_a, int na,
    const PSolveTriangle *tris_b, int nb,
    double radius, double min_scale, double max_scale,
    PSolveStarPair **out_pairs, int *out_pair_count) {
    if (!tris_a || !tris_b || na == 0 || nb == 0) {
        *out_pairs = NULL;
        *out_pair_count = 0;
        return 1;
    }

    double rad2 = radius * radius;
    int max_idx_a = 0, max_idx_b = 0;
    for (int i = 0; i < na; i++) {
        if (tris_a[i].a_idx > max_idx_a) max_idx_a = tris_a[i].a_idx;
        if (tris_a[i].b_idx > max_idx_a) max_idx_a = tris_a[i].b_idx;
        if (tris_a[i].c_idx > max_idx_a) max_idx_a = tris_a[i].c_idx;
    }
    for (int i = 0; i < nb; i++) {
        if (tris_b[i].a_idx > max_idx_b) max_idx_b = tris_b[i].a_idx;
        if (tris_b[i].b_idx > max_idx_b) max_idx_b = tris_b[i].b_idx;
        if (tris_b[i].c_idx > max_idx_b) max_idx_b = tris_b[i].c_idx;
    }

    int dim_a = max_idx_a + 1;
    int dim_b = max_idx_b + 1;
    int *vote_matrix = (int *)calloc(dim_a * dim_b, sizeof(int));
    if (!vote_matrix) {
        *out_pairs = NULL;
        *out_pair_count = 0;
        return 2;
    }

    int tri_match_count = 0;

    for (int j = 0; j < nb; j++) {
        double ba_B = tris_b[j].ba_ratio;
        double ca_B = tris_b[j].ca_ratio;
        double ba_min = ba_B - radius;
        double ba_max = ba_B + radius;

        int start = find_ba_start(tris_a, na, ba_min);

        for (int i = start; i < na; i++) {
            double ba_A = tris_a[i].ba_ratio;
            if (ba_A > ba_max) break;

            double ca_A = tris_a[i].ca_ratio;
            double dist2 = (ba_A - ba_B) * (ba_A - ba_B) + (ca_A - ca_B) * (ca_A - ca_B);
            if (dist2 >= rad2) continue;

            if (min_scale > 0 && max_scale > 0) {
                double ratio = tris_a[i].side_a_length / tris_b[j].side_a_length;
                if (ratio < min_scale || ratio > max_scale) continue;
            }

            tri_match_count++;

            int a_idx[3] = {tris_a[i].a_idx, tris_a[i].b_idx, tris_a[i].c_idx};
            int b_idx[3] = {tris_b[j].a_idx, tris_b[j].b_idx, tris_b[j].c_idx};

            for (int p = 0; p < 3; p++) {
                vote_matrix[a_idx[p] * dim_b + b_idx[p]]++;
            }
        }
    }

    PSLOG_I("Triangle matches: %d, vote matrix: %dx%d", tri_match_count, dim_a, dim_b);

    int *best_b_for_a = (int *)malloc(dim_a * sizeof(int));
    int *best_a_votes = (int *)malloc(dim_a * sizeof(int));
    for (int i = 0; i < dim_a; i++) {
        best_b_for_a[i] = -1;
        best_a_votes[i] = 0;
    }

    int *best_a_for_b = (int *)malloc(dim_b * sizeof(int));
    int *best_b_votes = (int *)malloc(dim_b * sizeof(int));
    for (int j = 0; j < dim_b; j++) {
        best_a_for_b[j] = -1;
        best_b_votes[j] = 0;
    }

    int min_votes = 2;

    for (int i = 0; i < dim_a; i++) {
        for (int j = 0; j < dim_b; j++) {
            int v = vote_matrix[i * dim_b + j];
            if (v < min_votes) continue;
            if (v > best_a_votes[i]) {
                best_a_votes[i] = v;
                best_b_for_a[i] = j;
            }
            if (v > best_b_votes[j]) {
                best_b_votes[j] = v;
                best_a_for_b[j] = i;
            }
        }
    }

    free(vote_matrix);

    int result_count = 0;
    for (int i = 0; i < dim_a; i++) {
        if (best_b_for_a[i] >= 0) {
            int j = best_b_for_a[i];
            if (best_a_for_b[j] == i) {
                result_count++;
            }
        }
    }

    if (result_count == 0) {
        PSLOG_W("No unique star pairs after deduplication");
        free(best_b_for_a); free(best_a_votes);
        free(best_a_for_b); free(best_b_votes);
        *out_pairs = NULL;
        *out_pair_count = 0;
        return 4;
    }

    PSolveStarPair *pairs = (PSolveStarPair *)malloc(result_count * sizeof(PSolveStarPair));
    if (!pairs) {
        free(best_b_for_a); free(best_a_votes);
        free(best_a_for_b); free(best_b_votes);
        *out_pairs = NULL;
        *out_pair_count = 0;
        return 5;
    }

    int pidx = 0;
    for (int i = 0; i < dim_a; i++) {
        if (best_b_for_a[i] >= 0) {
            int j = best_b_for_a[i];
            if (best_a_for_b[j] == i) {
                pairs[pidx].img_idx = i;
                pairs[pidx].cat_idx = j;
                pidx++;
            }
        }
    }

    free(best_b_for_a); free(best_a_votes);
    free(best_a_for_b); free(best_b_votes);

    *out_pairs = pairs;
    *out_pair_count = result_count;

    PSLOG_I("Matched star pairs: %d (deduplicated, from %d triangle matches)", result_count, tri_match_count);
    return 0;
}

void psolve_free_triangles(PSolveTriangle *tris) {
    free(tris);
}

void psolve_free_pairs(PSolveStarPair *pairs) {
    free(pairs);
}

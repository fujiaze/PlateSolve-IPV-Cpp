#include "psm_star_alignment.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdio.h>
#include <stdarg.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define AT_MATCH_RATIO 0.9
#define AT_TRIANGLE_RADIUS 0.002
#define AT_MATCH_RADIUS 2.0
#define AT_MATCH_MAXDIST 50.0
#define AT_MATCH_MINVOTES 2
#define AT_MATCH_NSIGMA 10.0
#define AT_MATCH_PERCENTILE 0.35
#define AT_MATCH_REQUIRE 3
#define AT_MATCH_REQUIRE_LINEAR 3
#define AT_MATCH_STARTN_LINEAR 6
#define AT_MATCH_MAXITER 10
#define AT_MATCH_HALTSIGMA 1.0e-1
#define CONV_TOLERANCE 1.0e-2
#define MAX_REPROJ_TRIALS 5
#define SH_SUCCESS 0
#define SH_GENERIC_ERROR -1
#define RECALC_NO 0
#define RECALC_YES 1
#define DEGTORAD (M_PI / 180.0)
#define RADTODEG (180.0 / M_PI)
#define RADTOASEC (RADTODEG * 3600.0)
#define ASECTODEG (1.0 / 3600.0)
#define MIN_SAT_FOR_PRIORITY 10
#define RETRY_COUNTS_LEN 5
#define RANSAC_ITER 1000
#define RANSAC_THRESH_PX 5.0

struct SAStar {
    int id;
    double x;
    double y;
    double mag;
    double ra;
    double dec;
    int saturated;
};

struct SATriangle {
    double a_length;
    double ba;
    double ca;
    int a_index;
    int b_index;
    int c_index;
};

static void log_debug(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[StarAlign] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

static double calc_dist(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return sqrt(dx*dx + dy*dy);
}

static void gnomonic_projection(double ra_deg, double dec_deg,
    double ra0_deg, double dec0_deg,
    double *xi_asec, double *eta_asec) {
    double ra_rad = ra_deg * DEGTORAD;
    double dec_rad = dec_deg * DEGTORAD;
    double ra0_rad = ra0_deg * DEGTORAD;
    double dec0_rad = dec0_deg * DEGTORAD;
    double cos_dec = cos(dec_rad); double sin_dec = sin(dec_rad);
    double cos_dec0 = cos(dec0_rad); double sin_dec0 = sin(dec0_rad);
    double ra_diff = ra_rad - ra0_rad;
    double cos_ra_diff = cos(ra_diff); double sin_ra_diff = sin(ra_diff);
    double cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff;
    if (cos_c < 1e-10) { *xi_asec = 0.0; *eta_asec = 0.0; return; }
    double xi = cos_dec * sin_ra_diff / cos_c;
    double eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c;
    *xi_asec = xi * RADTOASEC;
    *eta_asec = eta * RADTOASEC;
}

static void gnomonic_deproject(double xi_asec, double eta_asec,
    double ra0_deg, double dec0_deg,
    double *ra_deg, double *dec_deg) {
    double delta_ra = (xi_asec * ASECTODEG) * DEGTORAD;
    double delta_dec = (eta_asec * ASECTODEG) * DEGTORAD;
    double r_dec = dec0_deg * DEGTORAD;
    double z = cos(r_dec) - delta_dec * sin(r_dec);
    double zz = atan2(delta_ra, z) * RADTODEG;
    double alpha = zz + ra0_deg;
    double sin_r = sin(r_dec); double cos_r = cos(r_dec);
    double denom = sqrt(1.0 + delta_ra*delta_ra + delta_dec*delta_dec);
    double delta = asin((sin_r + delta_dec * cos_r) / denom) * RADTODEG;
    if (alpha < 0.0) alpha += 360.0;
    if (alpha >= 360.0) alpha -= 360.0;
    *ra_deg = alpha;
    *dec_deg = delta;
}

static void set_triangle(const SAStar *stars, int s1, int s2, int s3, SATriangle &tri) {
    double d12 = calc_dist(stars[s1].x, stars[s1].y, stars[s2].x, stars[s2].y);
    double d23 = calc_dist(stars[s2].x, stars[s2].y, stars[s3].x, stars[s3].y);
    double d13 = calc_dist(stars[s1].x, stars[s1].y, stars[s3].x, stars[s3].y);
    double a, b, c; int ai, bi, ci;
    if (d12 >= d23 && d12 >= d13) {
        ai = s3; a = d12;
        if (d23 >= d13) { bi = s1; b = d23; ci = s2; c = d13; }
        else { bi = s2; b = d13; ci = s1; c = d23; }
    } else if (d23 > d12 && d23 >= d13) {
        ai = s1; a = d23;
        if (d12 > d13) { bi = s3; b = d12; ci = s2; c = d13; }
        else { bi = s2; b = d13; ci = s3; c = d12; }
    } else {
        ai = s2; a = d13;
        if (d12 > d23) { bi = s3; b = d12; ci = s1; c = d23; }
        else { bi = s1; b = d23; ci = s3; c = d12; }
    }
    tri.a_length = a;
    tri.ba = (a > 0.0) ? b / a : 1.0;
    tri.ca = (a > 0.0) ? c / a : 1.0;
    tri.a_index = ai; tri.b_index = bi; tri.c_index = ci;
}

static void stars_to_triangles(const SAStar *stars, int nbright, std::vector<SATriangle> &tris) {
    tris.clear();
    for (int i = 0; i < nbright; i++) {
        for (int j = i + 1; j < nbright; j++) {
            for (int k = j + 1; k < nbright; k++) {
                SATriangle tri;
                set_triangle(stars, i, j, k, tri);
                if (tri.ba <= AT_MATCH_RATIO) tris.push_back(tri);
            }
        }
    }
}

static bool compare_triangle_ba(const SATriangle &a, const SATriangle &b) {
    if (a.ba < b.ba) return true;
    if (a.ba > b.ba) return false;
    return a.ca < b.ca;
}

static int find_ba_start(const std::vector<SATriangle> &tris, double ba0) {
    int lo = 0, hi = (int)tris.size() - 1;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (tris[mid].ba < ba0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static int gauss_matrix_3x3(double m[3][3], double v[3]) {
    for (int col = 0; col < 3; col++) {
        int max_row = col;
        for (int row = col + 1; row < 3; row++)
            if (fabs(m[row][col]) > fabs(m[max_row][col])) max_row = row;
        if (fabs(m[max_row][col]) < 1e-12) return -1;
        if (max_row != col) {
            for (int j = 0; j < 3; j++) { double tmp = m[col][j]; m[col][j] = m[max_row][j]; m[max_row][j] = tmp; }
            double tmp = v[col]; v[col] = v[max_row]; v[max_row] = tmp;
        }
        for (int row = col + 1; row < 3; row++) {
            double factor = m[row][col] / m[col][col];
            for (int j = col; j < 3; j++) m[row][j] -= factor * m[col][j];
            v[row] -= factor * v[col];
        }
    }
    for (int i = 2; i >= 0; i--) {
        v[i] /= m[i][i];
        for (int j = 0; j < i; j++) v[j] -= m[j][i] * v[i];
    }
    return 0;
}

static int calc_trans_linear(const SAStar *stars_a, const SAStar *stars_b,
    int *wia, int *wib, int npairs,
    double &a0, double &a1, double &a2, double &b0, double &b1, double &b2) {
    if (npairs < AT_MATCH_REQUIRE_LINEAR) return -1;
    double s00=0,s10=0,s01=0,s20=0,s11=0,s02=0;
    double sx0=0,sx1=0,sx2=0,sy0=0,sy1=0,sy2=0;
    for (int i = 0; i < npairs; i++) {
        double x1 = stars_a[wia[i]].x, y1 = stars_a[wia[i]].y;
        double x2 = stars_b[wib[i]].x, y2 = stars_b[wib[i]].y;
        s00+=1; s10+=x1; s01+=y1; s20+=x1*x1; s11+=x1*y1; s02+=y1*y1;
        sx0+=x2; sx1+=x2*x1; sx2+=x2*y1;
        sy0+=y2; sy1+=y2*x1; sy2+=y2*y1;
    }
    double m[3][3]; double v[3];
    m[0][0]=s00; m[0][1]=s10; m[0][2]=s01;
    m[1][0]=s10; m[1][1]=s20; m[1][2]=s11;
    m[2][0]=s01; m[2][1]=s11; m[2][2]=s02;
    v[0]=sx0; v[1]=sx1; v[2]=sx2;
    if (gauss_matrix_3x3(m, v) < 0) return -1;
    a0=v[0]; a1=v[1]; a2=v[2];
    m[0][0]=s00; m[0][1]=s10; m[0][2]=s01;
    m[1][0]=s10; m[1][1]=s20; m[1][2]=s11;
    m[2][0]=s01; m[2][1]=s11; m[2][2]=s02;
    v[0]=sy0; v[1]=sy1; v[2]=sy2;
    if (gauss_matrix_3x3(m, v) < 0) return -1;
    b0=v[0]; b1=v[1]; b2=v[2];
    return 0;
}

static double find_percentile(double *array, int num, double perc) {
    if (num <= 0) return 0.0;
    int index = (int)floor(num * perc + 0.5);
    if (index >= num) index = num - 1;
    return array[index];
}

static int compare_double(const void *a, const void *b) {
    double da = *(const double*)a; double db = *(const double*)b;
    if (da < db) return -1; if (da > db) return 1; return 0;
}

static int iter_trans(const SAStar *stars_a, const SAStar *stars_b,
    int *votes, int *wia, int *wib, int nbright,
    int recalc_flag, int max_iter, double halt_sigma,
    double &a0, double &a1, double &a2, double &b0, double &b1, double &b2) {
    if (nbright < AT_MATCH_REQUIRE_LINEAR) return -1;
    int initial_pairs = (recalc_flag == RECALC_YES) ? nbright : AT_MATCH_STARTN_LINEAR;
    if (calc_trans_linear(stars_a, stars_b, wia, wib, initial_pairs, a0,a1,a2,b0,b1,b2) < 0) return -1;
    int nr = nbright;
    double *dist2 = (double*)malloc(nbright * sizeof(double));
    double *dist2_sorted = (double*)malloc(nbright * sizeof(double));
    double max_dist2 = AT_MATCH_MAXDIST * AT_MATCH_MAXDIST;
    int iters = 0; int is_ok = 1;
    while (iters < max_iter) {
        int nb = 0;
        for (int i = 0; i < nr; i++) {
            double sx=stars_a[wia[i]].x, sy=stars_a[wia[i]].y;
            double nx=a0+a1*sx+a2*sy, ny=b0+b1*sx+b2*sy;
            double bx=stars_b[wib[i]].x, by=stars_b[wib[i]].y;
            dist2[i] = (nx-bx)*(nx-bx)+(ny-by)*(ny-by);
            dist2_sorted[i] = dist2[i];
        }
        for (int i = 0; i < nr; i++) {
            if (dist2[i] > max_dist2) {
                for (int j=i+1;j<nr;j++) { votes[j-1]=votes[j]; wia[j-1]=wia[j]; wib[j-1]=wib[j]; dist2[j-1]=dist2[j]; }
                nr--; nb++; i--;
            }
        }
        if (nr < AT_MATCH_REQUIRE_LINEAR) { is_ok=0; break; }
        qsort(dist2_sorted, nr, sizeof(double), compare_double);
        double sigma = (nr >= 2) ? find_percentile(dist2_sorted, nr, AT_MATCH_PERCENTILE) : 0.0;
        if (sigma <= halt_sigma) is_ok = 1;
        int nb_sigma = 0;
        for (int i = 0; i < nr; i++) {
            if (dist2[i] > AT_MATCH_NSIGMA * sigma) {
                for (int j=i+1;j<nr;j++) { votes[j-1]=votes[j]; wia[j-1]=wia[j]; wib[j-1]=wib[j]; dist2[j-1]=dist2[j]; }
                nr--; nb_sigma++; i--;
            }
        }
        if (nr < AT_MATCH_REQUIRE_LINEAR) { is_ok=0; break; }
        if (nb==0 && nb_sigma==0) { is_ok=1; break; }
        if (calc_trans_linear(stars_a, stars_b, wia, wib, nr, a0,a1,a2,b0,b1,b2) < 0) { is_ok=0; break; }
        iters++;
        if (is_ok) break;
    }
    free(dist2); free(dist2_sorted);
    return is_ok ? nr : -1;
}

static int siril_triangle_match(
    const SAStar *stars_a, int na,
    const SAStar *stars_b, int nb,
    int nobj, double radius,
    double min_scale, double max_scale,
    double &a0, double &a1, double &a2,
    double &b0, double &b1, double &b2) {

    int nbright = std::min(nobj, std::max(na, nb));
    if (nbright < AT_MATCH_STARTN_LINEAR) return SH_GENERIC_ERROR;

    std::vector<SATriangle> tris_a, tris_b;
    stars_to_triangles(stars_a, nbright, tris_a);
    stars_to_triangles(stars_b, nbright, tris_b);
    std::sort(tris_a.begin(), tris_a.end(), compare_triangle_ba);
    std::sort(tris_b.begin(), tris_b.end(), compare_triangle_ba);

    log_debug("    triangle_match: nbright=%d tris_a=%d tris_b=%d", nbright, tris_a.size(), tris_b.size());

    int **vote = (int**)malloc(nbright * sizeof(int*));
    for (int i = 0; i < nbright; i++) vote[i] = (int*)calloc(nbright, sizeof(int));
    double rad2 = radius * radius;
    int match_count = 0;

    for (int j = 0; j < (int)tris_b.size(); j++) {
        const SATriangle &tb = tris_b[j];
        if (tb.a_index >= nbright || tb.b_index >= nbright || tb.c_index >= nbright) continue;
        double ba_min = tb.ba - radius, ba_max = tb.ba + radius;
        int start = find_ba_start(tris_a, ba_min);
        for (int i = start; i < (int)tris_a.size(); i++) {
            const SATriangle &ta = tris_a[i];
            if (ta.a_index >= nbright || ta.b_index >= nbright || ta.c_index >= nbright) continue;
            if (ta.ba > ba_max) break;
            double dba = ta.ba - tb.ba, dca = ta.ca - tb.ca;
            if (dba*dba + dca*dca < rad2) {
                double ratio = ta.a_length / tb.a_length;
                if (ratio < min_scale || ratio > max_scale) continue;
                vote[ta.a_index][tb.a_index]++;
                vote[ta.b_index][tb.b_index]++;
                vote[ta.c_index][tb.c_index]++;
                match_count++;
            }
        }
    }

    log_debug("    triangle_match: %d matching pairs", match_count);

    int *wv = (int*)malloc(nbright * sizeof(int));
    int *wia = (int*)malloc(nbright * sizeof(int));
    int *wib = (int*)malloc(nbright * sizeof(int));
    for (int i = 0; i < nbright; i++) { wv[i]=0; wia[i]=-1; wib[i]=-1; }

    for (int i = 0; i < nbright; i++) {
        for (int j = 0; j < nbright; j++) {
            if (vote[i][j] > wv[nbright-1]) {
                for (int k = 0; k < nbright; k++) {
                    if (vote[i][j] > wv[k]) {
                        for (int l = nbright-2; l >= k; l--) { wv[l+1]=wv[l]; wia[l+1]=wia[l]; wib[l+1]=wib[l]; }
                        wv[k]=vote[i][j]; wia[k]=i; wib[k]=j;
                        break;
                    }
                }
            }
        }
    }
    for (int i = 0; i < nbright; i++) free(vote[i]);
    free(vote);

    for (int i = 0; i < nbright; i++) {
        if (wv[i] < AT_MATCH_MINVOTES) { nbright = i; break; }
    }
    log_debug("    valid pairs >=%d votes: %d", AT_MATCH_MINVOTES, nbright);

    if (nbright >= 3) {
        int n_print = std::min(5, nbright);
        for (int i = 0; i < n_print; i++) {
            log_debug("      [%d] A[%d](%.1f,%.1f) -> B[%d](%.1f,%.1f) votes=%d",
                      i, wia[i], stars_a[wia[i]].x, stars_a[wia[i]].y,
                      wib[i], stars_b[wib[i]].x, stars_b[wib[i]].y, wv[i]);
        }
    }

    if (nbright < AT_MATCH_STARTN_LINEAR) {
        log_debug("    too few valid pairs: %d < %d", nbright, AT_MATCH_STARTN_LINEAR);
        free(wv); free(wia); free(wib);
        return SH_GENERIC_ERROR;
    }

    int nr = iter_trans(stars_a, stars_b, wv, wia, wib, nbright,
                        RECALC_NO, AT_MATCH_MAXITER, AT_MATCH_HALTSIGMA,
                        a0,a1,a2,b0,b1,b2);
    if (nr < AT_MATCH_REQUIRE) {
        log_debug("    iter_trans failed: %d pairs", nr);
        free(wv); free(wia); free(wib);
        return SH_GENERIC_ERROR;
    }
    log_debug("    triangle_match OK: %d pairs", nr);
    {
        int n_print = std::min(10, nr);
        for (int i = 0; i < n_print; i++) {
            double sx=stars_a[wia[i]].x, sy=stars_a[wia[i]].y;
            double tx=a0+a1*sx+a2*sy, ty=b0+b1*sx+b2*sy;
            double bx=stars_b[wib[i]].x, by=stars_b[wib[i]].y;
            log_debug("      pair[%d] A(%.1f,%.1f)->B(%.1f,%.1f) pred(%.1f,%.1f) err=%.1f",
                      i, sx, sy, bx, by, tx, ty, calc_dist(tx,ty,bx,by));
        }
    }
    free(wv); free(wia); free(wib);
    return SH_SUCCESS;
}

static void apply_trans_to_stars(const SAStar *src, int n,
    double a0, double a1, double a2, double b0, double b1, double b2,
    SAStar *dst) {
    for (int i = 0; i < n; i++) {
        dst[i].x = a0 + a1*src[i].x + a2*src[i].y;
        dst[i].y = b0 + b1*src[i].x + b2*src[i].y;
        dst[i].id = src[i].id;
        dst[i].mag = src[i].mag;
        dst[i].ra = src[i].ra;
        dst[i].dec = src[i].dec;
        dst[i].saturated = src[i].saturated;
    }
}

static int match_lists_fast(const SAStar *sa, int na, const SAStar *sb, int nb,
    double radius, std::vector<int> &idx_a, std::vector<int> &idx_b) {
    idx_a.clear(); idx_b.clear();
    double limit2 = radius * radius;

    std::vector<int> order_b(nb);
    for (int i = 0; i < nb; i++) order_b[i] = i;
    std::sort(order_b.begin(), order_b.end(), [&](int p, int q) { return sb[p].x < sb[q].x; });

    std::vector<int> tmp_a, tmp_b;
    std::vector<double> tmp_d;

    for (int i = 0; i < na; i++) {
        double ax = sa[i].x, ay = sa[i].y;
        double axm = ax - radius, axp = ax + radius;

        int lo = 0, hi = nb - 1;
        while (lo < hi) { int mid = (lo+hi)/2; if (sb[order_b[mid]].x < axm) lo = mid+1; else hi = mid; }
        int start = lo;
        lo = 0; hi = nb - 1;
        while (lo < hi) { int mid = (lo+hi+1)/2; if (sb[order_b[mid]].x <= axp) lo = mid; else hi = mid-1; }
        int end = lo;

        for (int j = start; j <= end; j++) {
            int bj = order_b[j];
            double bx = sb[bj].x, by = sb[bj].y;
            double dy = ay - by;
            if (dy > radius || dy < -radius) continue;
            double dx = ax - bx;
            double d2 = dx*dx + dy*dy;
            if (d2 < limit2) {
                tmp_a.push_back(i);
                tmp_b.push_back(bj);
                tmp_d.push_back(d2);
            }
        }
    }

    int n_raw = tmp_a.size();
    if (n_raw == 0) return 0;

    std::vector<int> order_a(n_raw);
    for (int i = 0; i < n_raw; i++) order_a[i] = i;
    std::sort(order_a.begin(), order_a.end(), [&](int p, int q) { return tmp_a[p] < tmp_a[q]; });
    std::vector<int> keep(n_raw, 1);
    for (int k = 0; k < n_raw; ) {
        int cur = tmp_a[order_a[k]]; int best_k = k; double best_d = tmp_d[order_a[k]];
        int m = k + 1;
        while (m < n_raw && tmp_a[order_a[m]] == cur) { if (tmp_d[order_a[m]] < best_d) { best_d = tmp_d[order_a[m]]; best_k = m; } m++; }
        for (int p = k; p < m; p++) if (p != best_k) keep[order_a[p]] = 0;
        k = m;
    }
    std::vector<int> order_b2(n_raw);
    for (int i = 0; i < n_raw; i++) order_b2[i] = i;
    std::sort(order_b2.begin(), order_b2.end(), [&](int p, int q) { return tmp_b[p] < tmp_b[q]; });
    for (int k = 0; k < n_raw; ) {
        int cur = tmp_b[order_b2[k]]; int best_k = k; double best_d = tmp_d[order_b2[k]];
        int m = k + 1;
        while (m < n_raw && tmp_b[order_b2[m]] == cur) { if (tmp_d[order_b2[m]] < best_d) { best_d = tmp_d[order_b2[m]]; best_k = m; } m++; }
        for (int p = k; p < m; p++) if (p != best_k) keep[order_b2[p]] = 0;
        k = m;
    }
    for (int i = 0; i < n_raw; i++) if (keep[i]) { idx_a.push_back(tmp_a[i]); idx_b.push_back(tmp_b[i]); }
    return idx_a.size();
}

static int recalc_trans(const SAStar *sa, const SAStar *sb,
    const std::vector<int> &ia, const std::vector<int> &ib, int nm,
    double &a0,double &a1,double &a2,double &b0,double &b1,double &b2) {
    if (nm < AT_MATCH_REQUIRE) return SH_GENERIC_ERROR;
    int *votes=(int*)malloc(nm*sizeof(int));
    int *wia=(int*)malloc(nm*sizeof(int));
    int *wib=(int*)malloc(nm*sizeof(int));
    for(int i=0;i<nm;i++){votes[i]=100;wia[i]=ia[i];wib[i]=ib[i];}
    int nr=iter_trans(sa,sb,votes,wia,wib,nm,RECALC_YES,AT_MATCH_MAXITER,AT_MATCH_HALTSIGMA,a0,a1,a2,b0,b1,b2);
    free(votes);free(wia);free(wib);
    return (nr>=AT_MATCH_REQUIRE)?SH_SUCCESS:SH_GENERIC_ERROR;
}

static int ransac_affine(const SAStar *sa, const SAStar *sb,
    const std::vector<int> &ia, const std::vector<int> &ib, int nm,
    double thresh_px, int max_iter,
    double &best_a0,double &best_a1,double &best_a2,
    double &best_b0,double &best_b1,double &best_b2,
    std::vector<int> &best_inliers_a, std::vector<int> &best_inliers_b) {
    if (nm < 3) return SH_GENERIC_ERROR;
    double thresh2 = thresh_px * thresh_px;
    int best_count = 0;
    srand(42);
    for (int iter = 0; iter < max_iter; iter++) {
        int i0 = rand() % nm, i1 = rand() % nm, i2 = rand() % nm;
        if (i0==i1||i1==i2||i0==i2) continue;
        int si[3] = {ia[i0], ia[i1], ia[i2]};
        int ci[3] = {ib[i0], ib[i1], ib[i2]};
        double a0t,a1t,a2t,b0t,b1t,b2t;
        int wia3[3]={si[0],si[1],si[2]}, wib3[3]={ci[0],ci[1],ci[2]};
        if (calc_trans_linear(sa, sb, wia3, wib3, 3, a0t,a1t,a2t,b0t,b1t,b2t) < 0) continue;
        int count = 0;
        for (int i = 0; i < nm; i++) {
            double sx=sa[ia[i]].x, sy=sa[ia[i]].y;
            double nx=a0t+a1t*sx+a2t*sy, ny=b0t+b1t*sx+b2t*sy;
            double dx=nx-sb[ib[i]].x, dy=ny-sb[ib[i]].y;
            if (dx*dx+dy*dy < thresh2) count++;
        }
        if (count > best_count) {
            best_count = count;
            best_a0=a0t; best_a1=a1t; best_a2=a2t;
            best_b0=b0t; best_b1=b1t; best_b2=b2t;
        }
    }
    if (best_count < 3) return SH_GENERIC_ERROR;
    best_inliers_a.clear(); best_inliers_b.clear();
    for (int i = 0; i < nm; i++) {
        double sx=sa[ia[i]].x, sy=sa[ia[i]].y;
        double nx=best_a0+best_a1*sx+best_a2*sy, ny=best_b0+best_b1*sx+best_b2*sy;
        double dx=nx-sb[ib[i]].x, dy=ny-sb[ib[i]].y;
        if (dx*dx+dy*dy < thresh2) {
            best_inliers_a.push_back(ia[i]);
            best_inliers_b.push_back(ib[i]);
        }
    }
    log_debug("    RANSAC: %d/%d inliers (thresh=%.1f px)", best_inliers_a.size(), nm, thresh_px);
    if (calc_trans_linear(sa, sb, best_inliers_a.data(), best_inliers_b.data(),
                          best_inliers_a.size(), best_a0,best_a1,best_a2,best_b0,best_b1,best_b2) < 0)
        return SH_GENERIC_ERROR;
    return SH_SUCCESS;
}

extern "C" {

int psm_star_alignment(
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int n_img,
    const double *cat_x, const double *cat_y, const double *cat_mag, int n_cat,
    double scale_arcsec_px,
    double percent_scale_range,
    double center_ra,
    double center_dec,
    const double *cat_ra,
    const double *cat_dec,
    double cd1_1, double cd1_2, double cd2_1, double cd2_2,
    PSMStarAlignmentResult *result) {

    if (!img_x||!img_y||!cat_x||!cat_y||!result) return PSM_ERR_INVALID_PARAM;
    if (n_img < 10 || n_cat < 10) return PSM_ERR_NO_MATCH;
    memset(result, 0, sizeof(PSMStarAlignmentResult));

    log_debug("=== Step1: saturated-priority star alignment ===");
    log_debug("  n_img=%d n_cat=%d scale=%.3f\"/px range=%.1f%%", n_img, n_cat, scale_arcsec_px, percent_scale_range);
    log_debug("  center RA=%.4f Dec=%.4f", center_ra, center_dec);

    std::vector<SAStar> img_all(n_img);
    int n_sat = 0;
    for (int i = 0; i < n_img; i++) {
        img_all[i].id = i;
        img_all[i].x = img_x[i];
        img_all[i].y = img_y[i];
        img_all[i].mag = -img_flux[i];
        img_all[i].ra = 0.0;
        img_all[i].dec = 0.0;
        img_all[i].saturated = (img_saturated && img_saturated[i]) ? 1 : 0;
        if (img_all[i].saturated) n_sat++;
    }

    std::vector<SAStar> cat_all(n_cat);
    for (int i = 0; i < n_cat; i++) {
        cat_all[i].id = i;
        cat_all[i].x = cat_x[i];
        cat_all[i].y = cat_y[i];
        cat_all[i].mag = cat_mag[i];
        cat_all[i].ra = cat_ra ? cat_ra[i] : 0.0;
        cat_all[i].dec = cat_dec ? cat_dec[i] : 0.0;
        cat_all[i].saturated = 0;
    }

    log_debug("  saturated img stars: %d", n_sat);

    std::sort(img_all.begin(), img_all.end(), [](const SAStar &a, const SAStar &b) { return a.mag < b.mag; });
    std::sort(cat_all.begin(), cat_all.end(), [](const SAStar &a, const SAStar &b) { return a.mag < b.mag; });

    std::vector<int> sat_img_idx, norm_img_idx;
    for (int i = 0; i < n_img; i++) {
        if (img_all[i].saturated) sat_img_idx.push_back(i);
        else norm_img_idx.push_back(i);
    }
    log_debug("  sorted: %d saturated, %d normal", sat_img_idx.size(), norm_img_idx.size());

    int dup_img = 0;
    {
        int n_check = std::min(n_img, 500);
        std::vector<int> remove_flags(n_check, 0);
        for (int i = 0; i < n_check; i++) {
            if (remove_flags[i]) continue;
            for (int j = i+1; j < n_check; j++) {
                if (remove_flags[j]) continue;
                if (fabs(img_all[i].x-img_all[j].x)<2.0 && fabs(img_all[i].y-img_all[j].y)<2.0) {
                    remove_flags[j] = 1; dup_img++;
                }
            }
        }
        if (dup_img > 0) {
            int write = 0;
            for (int i = 0; i < n_img; i++) {
                if (i < n_check) { if (!remove_flags[i]) img_all[write++] = img_all[i]; }
                else img_all[write++] = img_all[i];
            }
            n_img = write; img_all.resize(n_img);
            sat_img_idx.clear(); norm_img_idx.clear();
            for (int i = 0; i < n_img; i++) {
                if (img_all[i].saturated) sat_img_idx.push_back(i);
                else norm_img_idx.push_back(i);
            }
        }
    }
    if (dup_img > 0) log_debug("  removed %d duplicate img stars", dup_img);

    double a_scale = 1.0 + percent_scale_range / 100.0;
    double b_scale = 1.0 - percent_scale_range / 100.0;
    double scale_min = 1.0 / (scale_arcsec_px * a_scale);
    double scale_max = 1.0 / (scale_arcsec_px * b_scale);

    int retry_img_counts[RETRY_COUNTS_LEN] = {100, 200, 400, 800, 800};
    int n_retries = (n_sat >= MIN_SAT_FOR_PRIORITY) ? 1 : RETRY_COUNTS_LEN;

    double best_a0=0,best_a1=0,best_a2=0,best_b0=0,best_b1=0,best_b2=0;
    int best_matched = 0;
    std::vector<int> best_ia, best_ib;

    for (int retry = 0; retry < n_retries; retry++) {
        int n_img_bright, n_cat_bright;

        if (n_sat >= MIN_SAT_FOR_PRIORITY) {
            n_img_bright = std::min((int)sat_img_idx.size(), 200);
            n_cat_bright = std::min((int)(n_img_bright * 1.5), n_cat);
            log_debug("  --- Attempt %d: saturated-priority (n_sat=%d) ---", retry+1, n_sat);
        } else {
            n_img_bright = std::min(retry_img_counts[retry], n_img);
            n_cat_bright = std::min((int)(n_img_bright * 1.5), n_cat);
            log_debug("  --- Attempt %d: %d img / %d cat ---", retry+1, n_img_bright, n_cat_bright);
        }

        std::vector<SAStar> sel_img, sel_cat;
        if (n_sat >= MIN_SAT_FOR_PRIORITY) {
            for (int i = 0; i < n_img_bright; i++) sel_img.push_back(img_all[sat_img_idx[i]]);
            for (int i = 0; i < n_cat_bright; i++) sel_cat.push_back(cat_all[i]);
        } else {
            for (int i = 0; i < n_img_bright; i++) sel_img.push_back(img_all[i]);
            for (int i = 0; i < n_cat_bright; i++) sel_cat.push_back(cat_all[i]);
        }

        {
            int dc = 0;
            int nc = std::min((int)sel_img.size(), 300);
            std::vector<int> rf(nc, 0);
            for (int i = 0; i < nc; i++) { if(rf[i]) continue;
                for (int j = i+1; j < nc; j++) { if(rf[j]) continue;
                    if (fabs(sel_img[i].x-sel_img[j].x)<2.0 && fabs(sel_img[i].y-sel_img[j].y)<2.0) { rf[j]=1; dc++; }
                }
            }
            if (dc > 0) {
                int w=0;
                for (int i=0;i<(int)sel_img.size();i++) {
                    if (i<nc) { if(!rf[i]) sel_img[w++]=sel_img[i]; }
                    else sel_img[w++]=sel_img[i];
                }
                sel_img.resize(w);
                log_debug("    removed %d dup from sel_img", dc);
            }
        }

        int si = sel_img.size(), sc = sel_cat.size();
        log_debug("    sel_img=%d sel_cat=%d", si, sc);

        double a0,a1,a2,b0,b1,b2;
        int ret = siril_triangle_match(sel_img.data(), si, sel_cat.data(), sc,
                    60, AT_TRIANGLE_RADIUS, scale_min, scale_max,
                    a0,a1,a2,b0,b1,b2);

        if (ret != SH_SUCCESS) {
            log_debug("    triangle_match failed, trying RANSAC fallback");
            std::vector<int> all_ia, all_ib;
            for (int i = 0; i < si; i++) all_ia.push_back(i);
            for (int i = 0; i < sc; i++) all_ib.push_back(i);
            std::vector<int> ria, rib;
            ret = ransac_affine(sel_img.data(), sel_cat.data(), all_ia, all_ib, si,
                                RANSAC_THRESH_PX, RANSAC_ITER,
                                a0,a1,a2,b0,b1,b2, ria, rib);
            if (ret != SH_SUCCESS) {
                log_debug("    RANSAC also failed");
                continue;
            }
            log_debug("    RANSAC OK with %d inliers", ria.size());
        }

        double det = a1*b2 - a2*b1;
        if (fabs(det) < 1e-10) {
            log_debug("    TRANS degenerate (det=%.8f)", det);
            continue;
        }

        std::vector<SAStar> img_trans(si);
        apply_trans_to_stars(sel_img.data(), si, a0,a1,a2,b0,b1,b2, img_trans.data());

        double match_radii[] = {50.0, 30.0, 10.0, AT_MATCH_RADIUS};
        int n_rounds = 4;
        std::vector<int> ma, mb;
        int nm = 0;

        for (int round = 0; round < n_rounds; round++) {
            ma.clear(); mb.clear();
            nm = match_lists_fast(img_trans.data(), si, sel_cat.data(), sc, match_radii[round], ma, mb);
            log_debug("    round %d (radius=%.0f\"): %d pairs", round, match_radii[round], nm);
            if (nm < AT_MATCH_REQUIRE) break;

            ret = recalc_trans(sel_img.data(), sel_cat.data(), ma, mb, nm, a0,a1,a2,b0,b1,b2);
            if (ret != SH_SUCCESS) { log_debug("    recalc round %d failed", round); nm = 0; break; }

            apply_trans_to_stars(sel_img.data(), si, a0,a1,a2,b0,b1,b2, img_trans.data());

            double cur_scale = sqrt(a1*a1 + a2*a2);
            log_debug("    round %d: %d pairs, scale=%.6f", round, nm, cur_scale);
        }

        if (nm < AT_MATCH_REQUIRE) {
            log_debug("    matching failed after %d rounds", n_rounds);
            continue;
        }

        det = a1*b2 - a2*b1;
        if (fabs(det) < 1e-10) { log_debug("    final TRANS degenerate"); continue; }

        double scale = sqrt(a1*a1 + a2*a2);
        double expected_scale = scale_arcsec_px;
        double scale_ratio = scale / expected_scale;
        if (scale_ratio < 0.5 || scale_ratio > 2.0) {
            log_debug("    scale %.6f (expected ~%.6f arcsec/px, ratio=%.2f) out of range", scale, expected_scale, scale_ratio);
            continue;
        }

        log_debug("    match OK: %d pairs, scale=%.6f", nm, scale);

        if (nm > best_matched) {
            best_matched = nm;
            best_a0=a0; best_a1=a1; best_a2=a2;
            best_b0=b0; best_b1=b1; best_b2=b2;
            best_ia = ma; best_ib = mb;
        }

        if (best_matched > 10) break;
    }

    if (best_matched < AT_MATCH_REQUIRE) {
        log_debug("  all attempts failed");
        return PSM_ERR_NO_MATCH;
    }

    double a0=best_a0,a1=best_a1,a2=best_a2,b0=best_b0,b1=best_b1,b2=best_b2;
    int num_matched = best_matched;

    double conv = sqrt(a0*a0 + b0*b0);
    log_debug("  coarse match done: %d pairs, conv=%.4f arcsec", num_matched, conv);

    double ra0 = center_ra, dec0 = center_dec;

    if (cat_ra && cat_dec && conv > CONV_TOLERANCE) {
        log_debug("  === iterative reprojection ===");
        int trial = 0;
        while (conv > CONV_TOLERANCE && trial < MAX_REPROJ_TRIALS) {
            double new_ra, new_dec;
            gnomonic_deproject(a0, b0, ra0, dec0, &new_ra, &new_dec);
            ra0 = new_ra; dec0 = new_dec;
            log_debug("  reproj %d: RA=%.6f Dec=%.6f (offset=%.4f\")", trial, ra0, dec0, conv);

            for (int i = 0; i < n_cat; i++) {
                double nx, ny;
                gnomonic_projection(cat_all[i].ra, cat_all[i].dec, ra0, dec0, &nx, &ny);
                cat_all[i].x = nx; cat_all[i].y = ny;
            }

            a0 = 0; a1 = scale_arcsec_px; a2 = 0;
            b0 = 0; b1 = 0; b2 = scale_arcsec_px;
            if (cd1_1 < 0) { a1 = -scale_arcsec_px; }
            if (cd2_2 < 0) { b2 = -scale_arcsec_px; }

            double reproj_radii[] = {100.0, 50.0, 30.0, 10.0, 5.0};
            int reproj_ok = 0;
            for (int r = 0; r < 5; r++) {
                std::vector<SAStar> img_trans(n_img);
                apply_trans_to_stars(img_all.data(), n_img, a0,a1,a2,b0,b1,b2, img_trans.data());

                std::vector<int> ma, mb;
                int nm = match_lists_fast(img_trans.data(), n_img, cat_all.data(), n_cat, reproj_radii[r], ma, mb);
                log_debug("  reproj %d round %d (radius=%.0f\"): %d pairs", trial, r, reproj_radii[r], nm);
                if (nm < AT_MATCH_REQUIRE) break;

                int ret = recalc_trans(img_all.data(), cat_all.data(), ma, mb, nm, a0,a1,a2,b0,b1,b2);
                if (ret != SH_SUCCESS) { log_debug("  recalc failed"); break; }

                num_matched = nm;
                double cur_conv = sqrt(a0*a0 + b0*b0);
                double cur_scale = sqrt(a1*a1 + a2*a2);
                log_debug("  reproj %d round %d: conv=%.4f\" scale=%.6f", trial, r, cur_conv, cur_scale);
                if (r == 4) reproj_ok = 1;
            }

            if (!reproj_ok) { log_debug("  reproj match failed at trial %d", trial); break; }

            conv = sqrt(a0*a0 + b0*b0);
            trial++;
            log_debug("  reproj %d: conv=%.4f arcsec, %d pairs", trial, conv, num_matched);
        }
        if (conv <= CONV_TOLERANCE) log_debug("  converged after %d trials", trial);
        else log_debug("  not fully converged (conv=%.4f\")", conv);
    }

    {
        int n_use_img = std::min(n_img, 3000);
        int n_use_cat = std::min(n_cat, 3000);
        std::vector<SAStar> img_trans(n_use_img);
        apply_trans_to_stars(img_all.data(), n_use_img, a0,a1,a2,b0,b1,b2, img_trans.data());
        log_debug("  transformed img[0]=(%.1f,%.1f) img[1]=(%.1f,%.1f)", img_trans[0].x, img_trans[0].y, img_trans[1].x, img_trans[1].y);
        log_debug("  cat[0]=(%.1f,%.1f) cat[1]=(%.1f,%.1f)", cat_all[0].x, cat_all[0].y, cat_all[1].x, cat_all[1].y);
        double tx_min=1e18,tx_max=-1e18,ty_min=1e18,ty_max=-1e18;
        for(int i=0;i<n_use_img;i++){tx_min=std::min(tx_min,img_trans[i].x);tx_max=std::max(tx_max,img_trans[i].x);ty_min=std::min(ty_min,img_trans[i].y);ty_max=std::max(ty_max,img_trans[i].y);}
        log_debug("  trans range: x=[%.1f,%.1f] y=[%.1f,%.1f]", tx_min,tx_max,ty_min,ty_max);
        double cx_min=1e18,cx_max=-1e18,cy_min=1e18,cy_max=-1e18;
        for(int i=0;i<n_use_cat;i++){cx_min=std::min(cx_min,cat_all[i].x);cx_max=std::max(cx_max,cat_all[i].x);cy_min=std::min(cy_min,cat_all[i].y);cy_max=std::max(cy_max,cat_all[i].y);}
        log_debug("  cat range: x=[%.1f,%.1f] y=[%.1f,%.1f]", cx_min,cx_max,cy_min,cy_max);
        std::vector<int> ma, mb;
        int nm = match_lists_fast(img_trans.data(), n_use_img, cat_all.data(), n_use_cat, 5.0, ma, mb);
        log_debug("  final match: %d pairs (radius=5\", %d img x %d cat)", nm, n_use_img, n_use_cat);
        if (nm < AT_MATCH_REQUIRE && nm > 0) {
            for (int i = 0; i < nm; i++) {
                log_debug("    pair[%d]: img(%.1f,%.1f) cat(%.1f,%.1f) d=%.1f",
                          i, img_trans[ma[i]].x, img_trans[ma[i]].y,
                          cat_all[mb[i]].x, cat_all[mb[i]].y,
                          calc_dist(img_trans[ma[i]].x,img_trans[ma[i]].y,cat_all[mb[i]].x,cat_all[mb[i]].y));
            }
        }
        nm = match_lists_fast(img_trans.data(), n_use_img, cat_all.data(), n_use_cat, 30.0, ma, mb);
        log_debug("  final match (30\"): %d pairs", nm);
        if (nm >= 3 && nm <= 20) {
            for (int i = 0; i < nm; i++) {
                log_debug("    pair[%d]: img(%.1f,%.1f) cat(%.1f,%.1f) d=%.1f",
                          i, img_trans[ma[i]].x, img_trans[ma[i]].y,
                          cat_all[mb[i]].x, cat_all[mb[i]].y,
                          calc_dist(img_trans[ma[i]].x,img_trans[ma[i]].y,cat_all[mb[i]].x,cat_all[mb[i]].y));
            }
        }
        if (nm >= AT_MATCH_REQUIRE) {
            num_matched = nm;
            double rms = 0;
            for (int i = 0; i < num_matched; i++) {
                double sx=img_all[ma[i]].x, sy=img_all[ma[i]].y;
                double tx=a0+a1*sx+a2*sy, ty=b0+b1*sx+b2*sy;
                double d=calc_dist(tx,ty,cat_all[mb[i]].x,cat_all[mb[i]].y);
                rms += d*d;
            }
            rms = sqrt(rms / num_matched);

            double scale = sqrt(a1*a1 + a2*a2);
            log_debug("  Final: %d pairs, RMS=%.4f arcsec (%.3f px), scale=%.6f arcsec/px",
                      num_matched, rms, rms/scale, scale);
            log_debug("  Final center: RA=%.6f Dec=%.6f", ra0, dec0);

            result->a0=a0; result->a1=a1; result->a2=a2;
            result->b0=b0; result->b1=b1; result->b2=b2;
            result->matched_count = num_matched;
            result->rms_arcsec = rms;
            result->center_ra = ra0;
            result->center_dec = dec0;
            result->img_indices = (int*)malloc(num_matched * sizeof(int));
            result->cat_indices = (int*)malloc(num_matched * sizeof(int));
            for (int i = 0; i < num_matched; i++) {
                result->img_indices[i] = img_all[ma[i]].id;
                result->cat_indices[i] = cat_all[mb[i]].id;
            }
            return PSM_OK;
        }
    }

    log_debug("  final match failed");
    return PSM_ERR_NO_MATCH;
}

void psm_free_result(PSMStarAlignmentResult *result) {
    if (result) {
        if (result->img_indices) free(result->img_indices);
        if (result->cat_indices) free(result->cat_indices);
        result->img_indices = NULL;
        result->cat_indices = NULL;
    }
}

}


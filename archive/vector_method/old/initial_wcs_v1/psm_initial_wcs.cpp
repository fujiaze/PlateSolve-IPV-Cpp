#include "psm_initial_wcs.h"
#include "../common/psm_common.h"
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>
#include <numeric>
#include <functional>

#include "../../gaia_xpsd_client/src/gaia_client.h"
#include "../../src/psolve_projection.h"

#define PI 3.14159265358979323846
#define DEG2RAD (PI / 180.0)
#define RAD2DEG (180.0 / PI)
#define RAD2ASEC (180.0 / PI * 3600.0)

#define AT_MATCH_RATIO        0.9
#define AT_TRIANGLE_RADIUS    0.002
#define AT_MATCH_MAXDIST      50.0
#define AT_MATCH_NSIGMA       10.0
#define AT_MATCH_PERCENTILE   0.35
#define AT_MATCH_MAXITER      10
#define AT_MATCH_HALTSIGMA    0.1
#define AT_MATCH_REQUIRE      3
#define AT_MATCH_MINVOTES     2
#define MIN_SAT_FOR_PRIORITY  10
#define MAX_BRIGHT_NORM       100
#define MAX_BRIGHT_CAT        150
#define MAX_REPROJ_TRIALS     5
#define CONV_TOLERANCE        0.01

static void log_msg(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[InitialWCS] ");
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}

static inline double compute_pixel_scale(double focal_mm, double pixel_um) {
    return 206.265 * pixel_um / focal_mm;
}

static inline void compute_fov(double scale_arcsec_px, int width, int height,
                                double *fov_w, double *fov_h, double *fov_diag) {
    *fov_w = width * scale_arcsec_px / 3600.0;
    *fov_h = height * scale_arcsec_px / 3600.0;
    *fov_diag = sqrt((*fov_w) * (*fov_w) + (*fov_h) * (*fov_h));
}

struct Triangle {
    int a_idx, b_idx, c_idx;
    double ba_ratio;
    double ca_ratio;
    double side_a_length;
};

struct FlipMatchResult {
    int flip_mode;
    double affine[6];
    int matched_count;
    double rms_arcsec;
    std::vector<int> img_indices;
    std::vector<int> cat_indices;
};

static int cmp_triangles(const void *a, const void *b) {
    const Triangle *ta = (const Triangle *)a;
    const Triangle *tb = (const Triangle *)b;
    if (ta->ba_ratio < tb->ba_ratio) return -1;
    if (ta->ba_ratio > tb->ba_ratio) return 1;
    if (ta->ca_ratio < tb->ca_ratio) return -1;
    if (ta->ca_ratio > tb->ca_ratio) return 1;
    return 0;
}

static int set_triangle(double x0, double y0, double x1, double y1,
                         double x2, double y2, int i0, int i1, int i2,
                         Triangle *tri) {
    double d01 = sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
    double d12 = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    double d02 = sqrt((x0 - x2) * (x0 - x2) + (y0 - y2) * (y0 - y2));

    if (d01 < 1e-10 || d12 < 1e-10 || d02 < 1e-10) return -1;

    double sides[3] = {d01, d12, d02};
    int verts[3][2] = {{i0, i1}, {i1, i2}, {i0, i2}};

    int a_side = 0;
    if (sides[1] > sides[a_side]) a_side = 1;
    if (sides[2] > sides[a_side]) a_side = 2;

    int b_side = -1, c_side = -1;
    for (int k = 0; k < 3; k++) {
        if (k == a_side) continue;
        if (b_side < 0) b_side = k;
        else c_side = k;
    }

    if (sides[b_side] < sides[c_side]) {
        int tmp = b_side; b_side = c_side; c_side = tmp;
    }

    double a_len = sides[a_side];
    double b_len = sides[b_side];
    double c_len = sides[c_side];

    double ba = b_len / a_len;
    double ca = c_len / a_len;

    if (ba > AT_MATCH_RATIO || ca > AT_MATCH_RATIO) return -1;

    tri->a_idx = verts[a_side][0] ^ verts[a_side][1];
    tri->b_idx = verts[b_side][0] ^ verts[b_side][1];
    tri->c_idx = verts[c_side][0] ^ verts[c_side][1];
    tri->ba_ratio = ba;
    tri->ca_ratio = ca;
    tri->side_a_length = a_len;

    return 0;
}

static std::vector<Triangle> build_triangles(const double *x, const double *y, int n) {
    std::vector<Triangle> tris;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            for (int k = j + 1; k < n; k++) {
                Triangle t;
                if (set_triangle(x[i], y[i], x[j], y[j], x[k], y[k],
                                 i, j, k, &t) == 0) {
                    tris.push_back(t);
                }
            }
        }
    }
    std::sort(tris.begin(), tris.end(), [](const Triangle &a, const Triangle &b) {
        if (a.ba_ratio != b.ba_ratio) return a.ba_ratio < b.ba_ratio;
        return a.ca_ratio < b.ca_ratio;
    });
    return tris;
}

static int calc_trans_linear(const double *x1, const double *y1,
                              const double *x2, const double *y2,
                              int n, double *a0, double *a1, double *a2,
                              double *b0, double *b1, double *b2) {
    if (n < AT_MATCH_REQUIRE) return -1;

    double sxx = 0, sxy = 0, sx = 0;
    double syx = 0, syy = 0, sy = 0;
    double sx1 = 0, sy1 = 0, s1 = (double)n;
    double sx2 = 0, sy2 = 0;

    for (int i = 0; i < n; i++) {
        sx += x1[i]; sy += y1[i];
        sx2 += x2[i]; sy2 += y2[i];
        sxx += x1[i] * x1[i]; sxy += x1[i] * y1[i];
        syx += y1[i] * x1[i]; syy += y1[i] * y1[i];
        sx1 += x1[i] * x2[i]; sy1 += y1[i] * x2[i];
    }

    double A[3][3] = {
        {sxx, sxy, sx},
        {syx, syy, sy},
        {sx,  sy,  s1}
    };
    double Bx[3] = {sx1, sy1, sx2};
    double By[3] = {0, 0, 0};
    for (int i = 0; i < n; i++) By[2] += y2[i];
    for (int i = 0; i < n; i++) By[0] += x1[i] * y2[i];
    for (int i = 0; i < n; i++) By[1] += y1[i] * y2[i];

    for (int col = 0; col < 3; col++) {
        int max_row = col;
        for (int row = col + 1; row < 3; row++) {
            if (fabs(A[row][col]) > fabs(A[max_row][col])) max_row = row;
        }
        if (max_row != col) {
            for (int j = 0; j < 3; j++) std::swap(A[col][j], A[max_row][j]);
            std::swap(Bx[col], Bx[max_row]);
            std::swap(By[col], By[max_row]);
        }
        if (fabs(A[col][col]) < 1e-12) return -1;
        for (int row = col + 1; row < 3; row++) {
            double f = A[row][col] / A[col][col];
            for (int j = col + 1; j < 3; j++) A[row][j] -= f * A[col][j];
            Bx[row] -= f * Bx[col];
            By[row] -= f * By[col];
            A[row][col] = 0;
        }
    }

    double xres[3], yres[3];
    for (int i = 2; i >= 0; i--) {
        xres[i] = Bx[i];
        yres[i] = By[i];
        for (int j = i + 1; j < 3; j++) {
            xres[i] -= A[i][j] * xres[j];
            yres[i] -= A[i][j] * yres[j];
        }
        xres[i] /= A[i][i];
        yres[i] /= A[i][i];
    }

    *a1 = xres[0]; *a2 = xres[1]; *a0 = xres[2];
    *b1 = yres[0]; *b2 = yres[1]; *b0 = yres[2];
    return 0;
}

static int iter_trans(const double *img_x, const double *img_y,
                       const double *cat_x, const double *cat_y,
                       const std::vector<int> &img_idx, const std::vector<int> &cat_idx,
                       double *a0, double *a1, double *a2,
                       double *b0, double *b1, double *b2,
                       double max_dist, int max_iter, double halt_sigma) {
    int n = (int)img_idx.size();
    if (n < AT_MATCH_REQUIRE) return -1;

    std::vector<double> ix(n), iy(n), cx(n), cy(n);
    for (int i = 0; i < n; i++) {
        ix[i] = img_x[img_idx[i]];
        iy[i] = img_y[img_idx[i]];
        cx[i] = cat_x[cat_idx[i]];
        cy[i] = cat_y[cat_idx[i]];
    }

    if (calc_trans_linear(ix.data(), iy.data(), cx.data(), cy.data(), n,
                           a0, a1, a2, b0, b1, b2) != 0) return -1;

    for (int iter = 0; iter < max_iter; iter++) {
        int nn = (int)ix.size();
        std::vector<double> dx(nn), dy(nn);
        for (int i = 0; i < nn; i++) {
            double px = *a0 + *a1 * ix[i] + *a2 * iy[i];
            double py = *b0 + *b1 * ix[i] + *b2 * iy[i];
            dx[i] = px - cx[i];
            dy[i] = py - cy[i];
        }

        std::vector<double> dist(nn);
        for (int i = 0; i < nn; i++) dist[i] = sqrt(dx[i]*dx[i] + dy[i]*dy[i]);

        std::vector<double> sorted_dist = dist;
        std::sort(sorted_dist.begin(), sorted_dist.end());
        double sig = sorted_dist[(int)(AT_MATCH_PERCENTILE * nn)];

        std::vector<int> keep;
        for (int i = 0; i < nn; i++) {
            if (dist[i] < max_dist && dist[i] < AT_MATCH_NSIGMA * sig) {
                keep.push_back(i);
            }
        }

        if ((int)keep.size() < AT_MATCH_REQUIRE) return -1;

        std::vector<double> nix, niy, ncx, ncy;
        std::vector<int> n_img_idx, n_cat_idx;
        for (int i : keep) {
            nix.push_back(ix[i]); niy.push_back(iy[i]);
            ncx.push_back(cx[i]); ncy.push_back(cy[i]);
        }
        ix = nix; iy = niy; cx = ncx; cy = ncy;

        if (calc_trans_linear(ix.data(), iy.data(), cx.data(), cy.data(),
                              (int)ix.size(), a0, a1, a2, b0, b1, b2) != 0) return -1;

        if (sig < halt_sigma) break;
    }

    return (int)ix.size();
}

static std::vector<int> triangle_match(const Triangle *tris_a, int na,
                                         const Triangle *tris_b, int nb,
                                         double radius, double scale_min, double scale_max) {
    std::vector<int> votes(na * nb, 0);

    for (int i = 0; i < na; i++) {
        double ba_lo = tris_a[i].ba_ratio - radius;
        double ba_hi = tris_a[i].ba_ratio + radius;

        int lo = 0, hi = nb - 1;
        int lo_idx = nb;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (tris_b[mid].ba_ratio >= ba_lo) { lo_idx = mid; hi = mid - 1; }
            else lo = mid + 1;
        }
        int hi_idx = -1;
        lo = 0; hi = nb - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (tris_b[mid].ba_ratio <= ba_hi) { hi_idx = mid; lo = mid + 1; }
            else hi = mid - 1;
        }

        for (int j = lo_idx; j <= hi_idx && j < nb; j++) {
            double dca = fabs(tris_a[i].ca_ratio - tris_b[j].ca_ratio);
            if (dca > radius) continue;

            double scale = tris_a[i].side_a_length / (tris_b[j].side_a_length + 1e-30);
            if (scale < scale_min || scale > scale_max) continue;

            int va = tris_a[i].a_idx ^ tris_a[i].b_idx ^ tris_a[i].c_idx;
            int vb = tris_b[j].a_idx ^ tris_b[j].b_idx ^ tris_b[j].c_idx;
            votes[va * nb + vb]++;
        }
    }

    std::vector<int> best_a(na, -1), best_b(nb, -1);
    std::vector<int> best_vote_a(na, 0), best_vote_b(nb, 0);

    for (int i = 0; i < na; i++) {
        for (int j = 0; j < nb; j++) {
            int v = votes[i * nb + j];
            if (v > best_vote_a[i]) { best_vote_a[i] = v; best_a[i] = j; }
            if (v > best_vote_b[j]) { best_vote_b[j] = v; best_b[j] = i; }
        }
    }

    std::vector<int> pairs;
    for (int i = 0; i < na; i++) {
        if (best_a[i] >= 0 && best_b[best_a[i]] == i && best_vote_a[i] >= AT_MATCH_MINVOTES) {
            pairs.push_back(i);
            pairs.push_back(best_a[i]);
        }
    }

    return pairs;
}

static void apply_flip(double *cat_px, double *cat_py, int n, int flip_mode) {
    for (int i = 0; i < n; i++) {
        if (flip_mode & 1) cat_px[i] = -cat_px[i];
        if (flip_mode & 2) cat_py[i] = -cat_py[i];
    }
}

static void select_match_stars(const double *img_x, const double *img_y,
                                const double *img_flux, const int *img_saturated,
                                int n_img,
                                const double *cat_x, const double *cat_y,
                                const double *cat_mag, int n_cat,
                                std::vector<double> &sel_img_x, std::vector<double> &sel_img_y,
                                std::vector<double> &sel_cat_x, std::vector<double> &sel_cat_y,
                                std::vector<int> &sel_img_idx, std::vector<int> &sel_cat_idx) {
    int n_saturated = 0;
    for (int i = 0; i < n_img; i++) if (img_saturated[i]) n_saturated++;

    std::vector<int> img_order(n_img);
    std::iota(img_order.begin(), img_order.end(), 0);
    std::sort(img_order.begin(), img_order.end(), [&](int a, int b) {
        return img_flux[a] > img_flux[b];
    });

    std::vector<int> cat_order(n_cat);
    std::iota(cat_order.begin(), cat_order.end(), 0);
    std::sort(cat_order.begin(), cat_order.end(), [&](int a, int b) {
        return cat_mag[a] < cat_mag[b];
    });

    if (n_saturated >= MIN_SAT_FOR_PRIORITY) {
        for (int i = 0; i < n_img; i++) {
            if (img_saturated[i]) {
                sel_img_idx.push_back(i);
            }
        }
        int n_cat_sel = std::min((int)(sel_img_idx.size() * 1.2), n_cat);
        for (int i = 0; i < n_cat_sel; i++) {
            sel_cat_idx.push_back(cat_order[i]);
        }
    } else {
        std::vector<int> sat_idx;
        for (int i = 0; i < n_img; i++) {
            if (img_saturated[i]) sat_idx.push_back(i);
        }
        int n_norm = MAX_BRIGHT_NORM - (int)sat_idx.size();
        if (n_norm < 0) n_norm = 0;

        sel_img_idx = sat_idx;
        int norm_count = 0;
        for (int i = 0; i < n_img && norm_count < n_norm; i++) {
            int idx = img_order[i];
            if (!img_saturated[idx]) {
                sel_img_idx.push_back(idx);
                norm_count++;
            }
        }

        int n_cat_sel = std::min(MAX_BRIGHT_CAT, n_cat);
        for (int i = 0; i < n_cat_sel; i++) {
            sel_cat_idx.push_back(cat_order[i]);
        }
    }

    sel_img_x.resize(sel_img_idx.size());
    sel_img_y.resize(sel_img_idx.size());
    for (int i = 0; i < (int)sel_img_idx.size(); i++) {
        sel_img_x[i] = img_x[sel_img_idx[i]];
        sel_img_y[i] = img_y[sel_img_idx[i]];
    }
    sel_cat_x.resize(sel_cat_idx.size());
    sel_cat_y.resize(sel_cat_idx.size());
    for (int i = 0; i < (int)sel_cat_idx.size(); i++) {
        sel_cat_x[i] = cat_x[sel_cat_idx[i]];
        sel_cat_y[i] = cat_y[sel_cat_idx[i]];
    }
}

static int nearest_neighbor_match(const double *x1, const double *y1, int n1,
                                   const double *x2, const double *y2, int n2,
                                   double max_dist,
                                   std::vector<int> &idx1, std::vector<int> &idx2) {
    std::vector<double> best_dist_1(n1, 1e30);
    std::vector<int> best_j_1(n1, -1);
    std::vector<double> best_dist_2(n2, 1e30);
    std::vector<int> best_i_2(n2, -1);

    for (int i = 0; i < n1; i++) {
        for (int j = 0; j < n2; j++) {
            double dx = x1[i] - x2[j];
            double dy = y1[i] - y2[j];
            double d = sqrt(dx*dx + dy*dy);
            if (d < best_dist_1[i]) { best_dist_1[i] = d; best_j_1[i] = j; }
            if (d < best_dist_2[j]) { best_dist_2[j] = d; best_i_2[j] = i; }
        }
    }

    idx1.clear(); idx2.clear();
    for (int i = 0; i < n1; i++) {
        int j = best_j_1[i];
        if (j >= 0 && best_i_2[j] == i && best_dist_1[i] < max_dist) {
            idx1.push_back(i);
            idx2.push_back(j);
        }
    }
    return (int)idx1.size();
}

static int verify_match(const double *img_x, const double *img_y, int n_img,
                         const double *cat_x, const double *cat_y, int n_cat,
                         double a0, double a1, double a2, double b0, double b1, double b2,
                         const double *radii, int n_radii,
                         double *out_a0, double *out_a1, double *out_a2,
                         double *out_b0, double *out_b1, double *out_b2,
                         std::vector<int> &final_img_idx, std::vector<int> &final_cat_idx) {
    *out_a0 = a0; *out_a1 = a1; *out_a2 = a2;
    *out_b0 = b0; *out_b1 = b1; *out_b2 = b2;

    for (int r = 0; r < n_radii; r++) {
        std::vector<double> pred_x(n_img), pred_y(n_img);
        for (int i = 0; i < n_img; i++) {
            pred_x[i] = *out_a0 + *out_a1 * img_x[i] + *out_a2 * img_y[i];
            pred_y[i] = *out_b0 + *out_b1 * img_x[i] + *out_b2 * img_y[i];
        }

        std::vector<int> m1, m2;
        int nm = nearest_neighbor_match(pred_x.data(), pred_y.data(), n_img,
                                         cat_x, cat_y, n_cat, radii[r], m1, m2);
        if (nm < AT_MATCH_REQUIRE) break;

        std::vector<double> ix(nm), iy(nm), cx(nm), cy(nm);
        for (int i = 0; i < nm; i++) {
            ix[i] = img_x[m1[i]]; iy[i] = img_y[m1[i]];
            cx[i] = cat_x[m2[i]]; cy[i] = cat_y[m2[i]];
        }

        double ta0, ta1, ta2, tb0, tb1, tb2;
        if (calc_trans_linear(ix.data(), iy.data(), cx.data(), cy.data(), nm,
                               &ta0, &ta1, &ta2, &tb0, &tb1, &tb2) != 0) break;

        *out_a0 = ta0; *out_a1 = ta1; *out_a2 = ta2;
        *out_b0 = tb0; *out_b1 = tb1; *out_b2 = tb2;

        final_img_idx = m1;
        final_cat_idx = m2;
    }

    return (int)final_img_idx.size();
}

static FlipMatchResult match_with_flip(
    const double *img_x, const double *img_y, int n_img,
    const double *img_flux, const int *img_saturated,
    const double *cat_px_orig, const double *cat_py_orig,
    const double *cat_mag, int n_cat,
    int flip_mode, double scale_arcsec_px) {

    FlipMatchResult result;
    result.flip_mode = flip_mode;
    result.matched_count = 0;
    result.rms_arcsec = 1e30;
    memset(result.affine, 0, sizeof(result.affine));

    std::vector<double> cat_px(cat_px_orig, cat_px_orig + n_cat);
    std::vector<double> cat_py(cat_py_orig, cat_py_orig + n_cat);
    apply_flip(cat_px.data(), cat_py.data(), n_cat, flip_mode);

    std::vector<double> sel_img_x, sel_img_y, sel_cat_x, sel_cat_y;
    std::vector<int> sel_img_idx, sel_cat_idx;
    select_match_stars(img_x, img_y, img_flux, img_saturated, n_img,
                        cat_px.data(), cat_py.data(), cat_mag, n_cat,
                        sel_img_x, sel_img_y, sel_cat_x, sel_cat_y,
                        sel_img_idx, sel_cat_idx);

    int n_sel_img = (int)sel_img_x.size();
    int n_sel_cat = (int)sel_cat_x.size();
    log_msg("  翻转模式%d: 选中图像星%d颗, Gaia星%d颗", flip_mode, n_sel_img, n_sel_cat);

    if (n_sel_img < AT_MATCH_REQUIRE || n_sel_cat < AT_MATCH_REQUIRE) {
        log_msg("  翻转模式%d: 星数不足，跳过", flip_mode);
        return result;
    }

    auto tris_img = build_triangles(sel_img_x.data(), sel_img_y.data(), n_sel_img);
    auto tris_cat = build_triangles(sel_cat_x.data(), sel_cat_y.data(), n_sel_cat);
    log_msg("  翻转模式%d: 图像三角形%zu个, Gaia三角形%zu个",
            flip_mode, tris_img.size(), tris_cat.size());

    if (tris_img.empty() || tris_cat.empty()) {
        log_msg("  翻转模式%d: 三角形为空，跳过", flip_mode);
        return result;
    }

    double scale_min = 0.5, scale_max = 2.0;
    auto pairs = triangle_match(tris_img.data(), (int)tris_img.size(),
                                 tris_cat.data(), (int)tris_cat.size(),
                                 AT_TRIANGLE_RADIUS, scale_min, scale_max);

    if ((int)pairs.size() < 2 * AT_MATCH_REQUIRE) {
        log_msg("  翻转模式%d: 三角匹配对数不足(%d)，跳过", flip_mode, (int)pairs.size() / 2);
        return result;
    }

    std::vector<int> match_img_idx, match_cat_idx;
    for (int i = 0; i < (int)pairs.size(); i += 2) {
        match_img_idx.push_back(sel_img_idx[pairs[i]]);
        match_cat_idx.push_back(sel_cat_idx[pairs[i+1]]);
    }

    double a0, a1, a2, b0, b1, b2;
    int n_matched = iter_trans(img_x, img_y, cat_px.data(), cat_py.data(),
                                match_img_idx, match_cat_idx,
                                &a0, &a1, &a2, &b0, &b1, &b2,
                                AT_MATCH_MAXDIST, AT_MATCH_MAXITER, AT_MATCH_HALTSIGMA);

    if (n_matched < AT_MATCH_REQUIRE) {
        log_msg("  翻转模式%d: iter_trans失败(n=%d)", flip_mode, n_matched);
        return result;
    }

    log_msg("  翻转模式%d: 三角匹配+iter_trans得到%d对", flip_mode, n_matched);

    double verify_radii[] = {50.0, 30.0, 10.0, 2.0};
    double va0, va1, va2, vb0, vb1, vb2;
    std::vector<int> v_img_idx, v_cat_idx;
    int n_verify = verify_match(img_x, img_y, n_img,
                                 cat_px.data(), cat_py.data(), n_cat,
                                 a0, a1, a2, b0, b1, b2,
                                 verify_radii, 4,
                                 &va0, &va1, &va2, &vb0, &vb1, &vb2,
                                 v_img_idx, v_cat_idx);

    if (n_verify < AT_MATCH_REQUIRE) {
        log_msg("  翻转模式%d: 验证匹配不足(n=%d)", flip_mode, n_verify);
        return result;
    }

    double rms_sum = 0;
    for (int i = 0; i < n_verify; i++) {
        double px = va0 + va1 * img_x[v_img_idx[i]] + va2 * img_y[v_img_idx[i]];
        double py = vb0 + vb1 * img_x[v_img_idx[i]] + vb2 * img_y[v_img_idx[i]];
        double dx = px - cat_px[v_cat_idx[i]];
        double dy = py - cat_py[v_cat_idx[i]];
        rms_sum += dx*dx + dy*dy;
    }
    double rms_px = sqrt(rms_sum / n_verify);
    double rms_arcsec = rms_px * scale_arcsec_px;

    log_msg("  翻转模式%d: 验证匹配%d对, RMS=%.3f px (%.3f arcsec)",
            flip_mode, n_verify, rms_px, rms_arcsec);

    result.affine[0] = va0; result.affine[1] = va1; result.affine[2] = va2;
    result.affine[3] = vb0; result.affine[4] = vb1; result.affine[5] = vb2;
    result.matched_count = n_verify;
    result.rms_arcsec = rms_arcsec;
    result.img_indices = v_img_idx;
    result.cat_indices = v_cat_idx;

    return result;
}

static FlipMatchResult select_best_flip(const FlipMatchResult *results, int n) {
    FlipMatchResult best = results[0];
    for (int i = 1; i < n; i++) {
        if (results[i].matched_count > best.matched_count ||
            (results[i].matched_count == best.matched_count &&
             results[i].rms_arcsec < best.rms_arcsec)) {
            best = results[i];
        }
    }
    return best;
}

static int bisection_mag_limit(GaiaClient *client, double ra, double dec,
                                double radius_deg, int target_count,
                                double *out_mag, double **out_ra, double **out_dec,
                                float **out_mag_arr, int *out_count) {
    double mag_low = 6.0, mag_high = 22.0;
    int target_high = (int)(target_count * 1.2);
    double tolerance = 0.1;

    double *tmp_ra = nullptr, *tmp_dec = nullptr;
    float *tmp_mag = nullptr;
    int tmp_count = 0;

    while ((mag_high - mag_low) > tolerance) {
        double mid = (mag_low + mag_high) / 2.0;
        if (tmp_ra) { free(tmp_ra); free(tmp_dec); free(tmp_mag); }
        tmp_ra = nullptr; tmp_dec = nullptr; tmp_mag = nullptr;
        tmp_count = 0;

        int rc = gaia_client_cone_search_for_solver(client, ra, dec, radius_deg,
                                                      mid, &tmp_ra, &tmp_dec, &tmp_mag, &tmp_count);
        if (rc != 0) { mag_high = mid; continue; }

        if (tmp_count < target_count) {
            mag_low = mid;
        } else if (tmp_count > target_high) {
            mag_high = mid;
        } else {
            *out_mag = mid;
            *out_ra = tmp_ra; *out_dec = tmp_dec; *out_mag_arr = tmp_mag;
            *out_count = tmp_count;
            return 0;
        }
    }

    *out_mag = (mag_low + mag_high) / 2.0;
    if (tmp_ra) { free(tmp_ra); free(tmp_dec); free(tmp_mag); }
    tmp_ra = nullptr; tmp_dec = nullptr; tmp_mag = nullptr;
    tmp_count = 0;
    int rc = gaia_client_cone_search_for_solver(client, ra, dec, radius_deg,
                                                  *out_mag, &tmp_ra, &tmp_dec, &tmp_mag, &tmp_count);
    *out_ra = tmp_ra; *out_dec = tmp_dec; *out_mag_arr = tmp_mag;
    *out_count = tmp_count;
    return rc;
}

PSM_IW_EXPORT int psm_initial_wcs_solve(
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int n_stars,
    double center_ra, double center_dec,
    double focal_length_mm, double pixel_size_um,
    int width, int height,
    const char *gaia_db_path, int db_type,
    InitialWCSResult *result) {

    log_msg("=== 初始WCS生成开始 ===");
    log_msg("图像星点: %d颗, 中心: (%.4f, %.4f)", n_stars, center_ra, center_dec);
    log_msg("焦距: %.1fmm, 像元: %.2fum, 尺寸: %dx%d", focal_length_mm, pixel_size_um, width, height);

    if (n_stars < AT_MATCH_REQUIRE) {
        log_msg("错误: 图像星点数不足(%d < %d)", n_stars, AT_MATCH_REQUIRE);
        return -1;
    }

    // Step 1: 计算像素尺度和FOV
    double scale_arcsec_px = compute_pixel_scale(focal_length_mm, pixel_size_um);
    double fov_w, fov_h, fov_diag;
    compute_fov(scale_arcsec_px, width, height, &fov_w, &fov_h, &fov_diag);
    log_msg("Step1: scale=%.3f arcsec/px, FOV=%.2fx%.2f (%.2f度对角线)",
            scale_arcsec_px, fov_w, fov_h, fov_diag);

    // Step 2: Gaia锥形查询 + 极限星等二分法
    GaiaClient *gaia = gaia_client_create_ex(gaia_db_path, (GaiaDbType)db_type);
    if (!gaia) {
        log_msg("错误: 无法创建Gaia客户端");
        return -1;
    }

    double radius_deg = fov_diag * 1.2 / 2.0;
    int target_count = (int)(n_stars * 1.5);
    double mag_limit;
    double *cat_ra = nullptr, *cat_dec = nullptr;
    float *cat_mag_f = nullptr;
    int n_cat = 0;

    log_msg("Step2: 查询半径=%.2f度, 目标星数=%d", radius_deg, target_count);
    int rc = bisection_mag_limit(gaia, center_ra, center_dec, radius_deg, target_count,
                                  &mag_limit, &cat_ra, &cat_dec, &cat_mag_f, &n_cat);
    if (rc != 0 || n_cat < AT_MATCH_REQUIRE) {
        log_msg("错误: Gaia查询失败或星数不足(n=%d)", n_cat);
        gaia_client_destroy(gaia);
        return -1;
    }
    log_msg("Step2: 极限星等=%.1f, Gaia星数=%d", mag_limit, n_cat);

    std::vector<double> cat_mag(n_cat);
    for (int i = 0; i < n_cat; i++) cat_mag[i] = (double)cat_mag_f[i];

    // Step 3: Gnomonic投影
    double *cat_px_raw = nullptr, *cat_py_raw = nullptr;
    psolve_project_stars(cat_ra, cat_dec, n_cat, center_ra, center_dec,
                          &cat_px_raw, &cat_py_raw);

    std::vector<double> cat_px(n_cat), cat_py(n_cat);
    for (int i = 0; i < n_cat; i++) {
        cat_px[i] = cat_px_raw[i] * 3600.0 / scale_arcsec_px;
        cat_py[i] = cat_py_raw[i] * 3600.0 / scale_arcsec_px;
    }
    free(cat_px_raw); free(cat_py_raw);

    log_msg("Step3: Gnomonic投影完成, %d颗Gaia星投影到像素坐标", n_cat);

    // Step 4: 4种翻转模式饱和星优先三角匹配
    log_msg("Step4: 开始4种翻转模式匹配...");
    FlipMatchResult flip_results[4];
    for (int flip_mode = 0; flip_mode < 4; flip_mode++) {
        flip_results[flip_mode] = match_with_flip(
            img_x, img_y, n_stars, img_flux, img_saturated,
            cat_px.data(), cat_py.data(), cat_mag.data(), n_cat,
            flip_mode, scale_arcsec_px);
    }

    // Step 5: 选择最佳翻转模式
    FlipMatchResult best = select_best_flip(flip_results, 4);
    log_msg("Step5: 最佳翻转模式=%d, 匹配数=%d, RMS=%.3f arcsec",
            best.flip_mode, best.matched_count, best.rms_arcsec);

    if (best.matched_count < AT_MATCH_REQUIRE) {
        log_msg("错误: 所有翻转模式匹配数不足");
        free(cat_ra); free(cat_dec); free(cat_mag_f);
        gaia_client_destroy(gaia);
        return -1;
    }

    // Step 6: 迭代重投影收敛
    double a0 = best.affine[0], a1 = best.affine[1], a2 = best.affine[2];
    double b0 = best.affine[3], b1 = best.affine[4], b2 = best.affine[5];
    double ra0 = center_ra, dec0 = center_dec;

    log_msg("Step6: 迭代重投影收敛...");
    for (int trial = 0; trial < MAX_REPROJ_TRIALS; trial++) {
        double offset = sqrt(a0*a0 + b0*b0) * scale_arcsec_px;
        log_msg("  迭代%d: offset=%.4f arcsec", trial, offset);
        if (offset < CONV_TOLERANCE) break;

        double new_x = a0 * scale_arcsec_px / 3600.0;
        double new_y = b0 * scale_arcsec_px / 3600.0;
        double new_ra, new_dec;
        psolve_plane_to_sky(new_x, new_y, ra0, dec0, &new_ra, &new_dec);
        ra0 = new_ra; dec0 = new_dec;

        free(cat_ra); free(cat_dec); free(cat_mag_f);
        cat_ra = nullptr; cat_dec = nullptr; cat_mag_f = nullptr;
        rc = gaia_client_cone_search_for_solver(gaia, ra0, dec0, radius_deg,
                                                  mag_limit, &cat_ra, &cat_dec, &cat_mag_f, &n_cat);
        if (rc != 0 || n_cat < AT_MATCH_REQUIRE) break;

        cat_mag.resize(n_cat);
        for (int i = 0; i < n_cat; i++) cat_mag[i] = (double)cat_mag_f[i];

        double *new_px_raw = nullptr, *new_py_raw = nullptr;
        psolve_project_stars(cat_ra, cat_dec, n_cat, ra0, dec0, &new_px_raw, &new_py_raw);
        cat_px.resize(n_cat); cat_py.resize(n_cat);
        for (int i = 0; i < n_cat; i++) {
            cat_px[i] = new_px_raw[i] * 3600.0 / scale_arcsec_px;
            cat_py[i] = new_py_raw[i] * 3600.0 / scale_arcsec_px;
        }
        free(new_px_raw); free(new_py_raw);

        apply_flip(cat_px.data(), cat_py.data(), n_cat, best.flip_mode);

        std::vector<double> pred_x(n_stars), pred_y(n_stars);
        for (int i = 0; i < n_stars; i++) {
            pred_x[i] = a0 + a1 * img_x[i] + a2 * img_y[i];
            pred_y[i] = b0 + b1 * img_x[i] + b2 * img_y[i];
        }

        std::vector<int> m1, m2;
        int nm = nearest_neighbor_match(pred_x.data(), pred_y.data(), n_stars,
                                         cat_px.data(), cat_py.data(), n_cat, 5.0, m1, m2);
        if (nm < AT_MATCH_REQUIRE) break;

        std::vector<double> ix(nm), iy(nm), cx(nm), cy(nm);
        for (int i = 0; i < nm; i++) {
            ix[i] = img_x[m1[i]]; iy[i] = img_y[m1[i]];
            cx[i] = cat_px[m2[i]]; cy[i] = cat_py[m2[i]];
        }

        double na0, na1, na2, nb0, nb1, nb2;
        if (calc_trans_linear(ix.data(), iy.data(), cx.data(), cy.data(), nm,
                               &na0, &na1, &na2, &nb0, &nb1, &nb2) != 0) break;
        a0 = na0; a1 = na1; a2 = na2;
        b0 = nb0; b1 = nb1; b2 = nb2;
    }

    // 计算最终RMS
    double rms_sum = 0;
    std::vector<double> pred_x(n_stars), pred_y(n_stars);
    for (int i = 0; i < n_stars; i++) {
        pred_x[i] = a0 + a1 * img_x[i] + a2 * img_y[i];
        pred_y[i] = b0 + b1 * img_x[i] + b2 * img_y[i];
    }
    std::vector<int> fm1, fm2;
    int final_n = nearest_neighbor_match(pred_x.data(), pred_y.data(), n_stars,
                                          cat_px.data(), cat_py.data(), n_cat, 5.0, fm1, fm2);
    double rms_px = 0;
    if (final_n > 0) {
        for (int i = 0; i < final_n; i++) {
            double dx = pred_x[fm1[i]] - cat_px[fm2[i]];
            double dy = pred_y[fm1[i]] - cat_py[fm2[i]];
            rms_sum += dx*dx + dy*dy;
        }
        rms_px = sqrt(rms_sum / final_n);
    }

    // 计算旋转角
    double rotation_deg = atan2(a2, a1) * RAD2DEG;

    // 填充结果
    result->center_ra = ra0;
    result->center_dec = dec0;
    result->rotation_deg = rotation_deg;
    result->scale_arcsec_px = scale_arcsec_px;
    result->flip_mode = best.flip_mode;
    result->affine[0] = a0; result->affine[1] = a1; result->affine[2] = a2;
    result->affine[3] = b0; result->affine[4] = b1; result->affine[5] = b2;
    result->matched_count = final_n;
    result->rms_px = rms_px;
    result->rms_arcsec = rms_px * scale_arcsec_px;

    log_msg("=== 初始WCS生成完成 ===");
    log_msg("中心: (%.6f, %.6f), 旋转: %.3f度, 比例尺: %.3f arcsec/px",
            result->center_ra, result->center_dec, result->rotation_deg, result->scale_arcsec_px);
    log_msg("翻转模式: %d, 匹配数: %d, RMS: %.3f px (%.3f arcsec)",
            result->flip_mode, result->matched_count, result->rms_px, result->rms_arcsec);

    if (cat_ra) free(cat_ra);
    if (cat_dec) free(cat_dec);
    if (cat_mag_f) free(cat_mag_f);
    gaia_client_destroy(gaia);

    return 0;
}

PSM_IW_EXPORT void psm_initial_wcs_free_result(InitialWCSResult *result) {
    // 无需释放，所有字段为值类型
}

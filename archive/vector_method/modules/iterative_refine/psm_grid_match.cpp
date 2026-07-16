#include "psm_grid_match.h"
#include "psm_sip.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define GM_LOG(fmt, ...) fprintf(stderr, "[GM] " fmt "\n", ##__VA_ARGS__)

static void gm_sky_to_plane(double ra, double dec, double ra0, double dec0, double *x, double *y)
{
    double cos_dec = cos(dec), sin_dec = sin(dec);
    double cos_dec0 = cos(dec0), sin_dec0 = sin(dec0);
    double ra_diff = ra - ra0;
    double cos_ra_diff = cos(ra_diff), sin_ra_diff = sin(ra_diff);
    double cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff;
    if (cos_c < 1e-10) { *x = 1e30; *y = 1e30; return; }
    *x = cos_dec * sin_ra_diff / cos_c;
    *y = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c;
}

class GMKDTree {
    struct Node {
        double x, y;
        int idx;
        int left, right;
    };

    std::vector<Node> nodes_;
    int root_;

    int build_r(int *indices, int n, int depth,
                const double *x, const double *y) {
        if (n <= 0) return -1;
        int id = (int)nodes_.size();
        nodes_.push_back({0, 0, 0, -1, -1});

        if (n == 1) {
            nodes_[id].x = x[indices[0]];
            nodes_[id].y = y[indices[0]];
            nodes_[id].idx = indices[0];
            return id;
        }

        int mid = n / 2;
        if (depth % 2 == 0) {
            std::nth_element(indices, indices + mid, indices + n,
                [&](int a, int b) { return x[a] < x[b]; });
        } else {
            std::nth_element(indices, indices + mid, indices + n,
                [&](int a, int b) { return y[a] < y[b]; });
        }

        nodes_[id].x = x[indices[mid]];
        nodes_[id].y = y[indices[mid]];
        nodes_[id].idx = indices[mid];
        nodes_[id].left = build_r(indices, mid, depth + 1, x, y);
        nodes_[id].right = build_r(indices + mid + 1, n - mid - 1, depth + 1, x, y);
        return id;
    }

    void range_r(int nid, double qx, double qy, double r2, int depth,
                 std::vector<int> &result) const {
        if (nid < 0) return;
        const Node &nd = nodes_[nid];
        double dx = qx - nd.x;
        double dy = qy - nd.y;
        double d2 = dx * dx + dy * dy;
        if (d2 < r2) result.push_back(nd.idx);

        int axis = depth % 2;
        double diff = (axis == 0) ? dx : dy;
        double r_axis = sqrt(r2);

        if (diff < r_axis) {
            range_r(nd.left, qx, qy, r2, depth + 1, result);
        }
        if (diff > -r_axis) {
            range_r(nd.right, qx, qy, r2, depth + 1, result);
        }
    }

public:
    void build(const double *x, const double *y, int n) {
        nodes_.clear();
        nodes_.reserve(n);
        std::vector<int> indices(n);
        for (int i = 0; i < n; i++) indices[i] = i;
        root_ = build_r(indices.data(), n, 0, x, y);
    }

    std::vector<int> range_query(double qx, double qy, double radius) const {
        std::vector<int> result;
        range_r(root_, qx, qy, radius * radius, 0, result);
        return result;
    }
};

struct GMTriangle {
    int i0, i1, i2;
    double cx, cy;
    double ba_ratio, ca_ratio;
    double area;
    double d01, d02, d12;
    double rotation_angle;
    int handedness;
};

static void gm_build_triangles_radius(
    const double *x, const double *y, int n,
    double radius,
    std::vector<GMTriangle> &tris,
    int max_triangles)
{
    tris.clear();
    if (n < 3) return;

    GMKDTree kdtree;
    kdtree.build(x, y, n);

    for (int i = 0; i < n && (int)tris.size() < max_triangles; i++) {
        auto nearby = kdtree.range_query(x[i], y[i], radius);
        if ((int)nearby.size() < 2) continue;

        std::sort(nearby.begin(), nearby.end());

        int n_nb = (int)nearby.size();
        for (int jj = 0; jj < n_nb && (int)tris.size() < max_triangles; jj++) {
            int i1 = nearby[jj];
            if (i1 <= i) continue;
            double dx1 = x[i1] - x[i], dy1 = y[i1] - y[i];
            double d01_sq = dx1*dx1 + dy1*dy1;
            if (d01_sq < 100.0) continue;

            for (int kk = jj + 1; kk < n_nb && (int)tris.size() < max_triangles; kk++) {
                int i2 = nearby[kk];
                if (i2 <= i) continue;

                double x0 = x[i], y0 = y[i];
                double x1 = x[i1], y1 = y[i1];
                double x2 = x[i2], y2 = y[i2];

                double signed_area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
                double area = fabs(signed_area) * 0.5;
                if (area < 100.0) continue;

                double da = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
                double db = sqrt((x2-x0)*(x2-x0) + (y2-y0)*(y2-y0));
                double dc = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));

                if (da < 10.0 || db < 10.0 || dc < 10.0) continue;

                double a = da, b = db, c = dc;
                if (a < b) std::swap(a, b);
                if (b < c) std::swap(b, c);
                if (a < b) std::swap(a, b);

                if (a < 1e-6) continue;
                double ba = b / a;
                double ca = c / a;

                if (ba > 0.95) continue;
                if (ba - ca < 0.02) continue;

                double v01_x = x1 - x0, v01_y = y1 - y0;
                double v02_x = x2 - x0, v02_y = y2 - y0;
                double rotation_angle = atan2(v01_y, v01_x);

                GMTriangle tri;
                tri.i0 = i;
                tri.i1 = i1;
                tri.i2 = i2;
                tri.cx = (x0 + x1 + x2) / 3.0;
                tri.cy = (y0 + y1 + y2) / 3.0;
                tri.ba_ratio = ba;
                tri.ca_ratio = ca;
                tri.area = area;
                tri.d01 = da;
                tri.d02 = db;
                tri.d12 = dc;
                tri.rotation_angle = rotation_angle;
                tri.handedness = (signed_area > 0) ? 1 : -1;
                tris.push_back(tri);
            }
        }
    }
}

static void gm_fit_affine(const double *img_x, const double *img_y,
                           const double *cat_x, const double *cat_y, int n,
                           double &a0, double &a1, double &a2,
                           double &b0, double &b1, double &b2) {
    double S1 = n;
    double Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0;
    double Sxa = 0, Sya = 0, Sxb = 0, Syb = 0;
    double Sa = 0, Sb = 0;

    for (int i = 0; i < n; i++) {
        double xi = img_x[i], yi = img_y[i];
        double xa = cat_x[i], yb = cat_y[i];
        Sx += xi; Sy += yi;
        Sxx += xi * xi; Syy += yi * yi; Sxy += xi * yi;
        Sa += xa; Sb += yb;
        Sxa += xi * xa; Sya += yi * xa;
        Sxb += xi * yb; Syb += yi * yb;
    }

    double M[3][3] = {{S1, Sx, Sy}, {Sx, Sxx, Sxy}, {Sy, Sxy, Syy}};
    double bx[3] = {Sa, Sxa, Sya};
    double by[3] = {Sb, Sxb, Syb};

    double det = M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1])
               - M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0])
               + M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]);

    if (fabs(det) < 1e-15) { a0=a1=a2=b0=b1=b2=0; return; }

    double inv[3][3];
    inv[0][0] = (M[1][1]*M[2][2] - M[1][2]*M[2][1]) / det;
    inv[0][1] = (M[0][2]*M[2][1] - M[0][1]*M[2][2]) / det;
    inv[0][2] = (M[0][1]*M[1][2] - M[0][2]*M[1][1]) / det;
    inv[1][0] = (M[1][2]*M[2][0] - M[1][0]*M[2][2]) / det;
    inv[1][1] = (M[0][0]*M[2][2] - M[0][2]*M[2][0]) / det;
    inv[1][2] = (M[0][2]*M[1][0] - M[0][0]*M[1][2]) / det;
    inv[2][0] = (M[1][0]*M[2][1] - M[1][1]*M[2][0]) / det;
    inv[2][1] = (M[0][1]*M[2][0] - M[0][0]*M[2][1]) / det;
    inv[2][2] = (M[0][0]*M[1][1] - M[0][1]*M[1][0]) / det;

    a0 = inv[0][0]*bx[0] + inv[0][1]*bx[1] + inv[0][2]*bx[2];
    a1 = inv[1][0]*bx[0] + inv[1][1]*bx[1] + inv[1][2]*bx[2];
    a2 = inv[2][0]*bx[0] + inv[2][1]*bx[1] + inv[2][2]*bx[2];

    b0 = inv[0][0]*by[0] + inv[0][1]*by[1] + inv[0][2]*by[2];
    b1 = inv[1][0]*by[0] + inv[1][1]*by[1] + inv[1][2]*by[2];
    b2 = inv[2][0]*by[0] + inv[2][1]*by[1] + inv[2][2]*by[2];
}

GM_EXPORT int psm_grid_ransac_filter(
    GMControlPoint *points,
    int n_points,
    int max_iter,
    double sigma_thresh,
    int *out_n_valid)
{
    if (!points || n_points < 10) {
        *out_n_valid = 0;
        return PSM_ERR_INVALID_PARAM;
    }

    for (int i = 0; i < n_points; i++) {
        points[i].valid = 1;
    }

    int n_valid = n_points;

    for (int iter = 0; iter < max_iter; iter++) {
        double a0, a1, a2, b0, b1, b2;

        std::vector<double> vx, vy;
        for (int i = 0; i < n_points; i++) {
            if (!points[i].valid) continue;
            vx.push_back(points[i].img_x);
            vy.push_back(points[i].img_y);
        }
        std::vector<double> vcx, vcy;
        for (int i = 0; i < n_points; i++) {
            if (!points[i].valid) continue;
            vcx.push_back(points[i].cat_x);
            vcy.push_back(points[i].cat_y);
        }

        if ((int)vx.size() < 10) break;

        gm_fit_affine(vx.data(), vy.data(), vcx.data(), vcy.data(), (int)vx.size(),
                       a0, a1, a2, b0, b1, b2);

        std::vector<double> residuals;
        for (int i = 0; i < n_points; i++) {
            if (!points[i].valid) continue;

            double pred_x = a0 + a1 * points[i].img_x + a2 * points[i].img_y;
            double pred_y = b0 + b1 * points[i].img_x + b2 * points[i].img_y;

            points[i].residual_x = points[i].cat_x - pred_x;
            points[i].residual_y = points[i].cat_y - pred_y;

            double r = sqrt(points[i].residual_x * points[i].residual_x +
                           points[i].residual_y * points[i].residual_y);
            residuals.push_back(r);
        }

        if (residuals.empty()) break;

        std::sort(residuals.begin(), residuals.end());
        double median = residuals[residuals.size() / 2];

        std::vector<double> abs_dev;
        for (double r : residuals) {
            abs_dev.push_back(fabs(r - median));
        }
        std::sort(abs_dev.begin(), abs_dev.end());
        double mad = abs_dev[abs_dev.size() / 2];

        double threshold = median + sigma_thresh * mad * 1.4826;

        int removed = 0;
        for (int i = 0; i < n_points; i++) {
            if (!points[i].valid) continue;

            double r = sqrt(points[i].residual_x * points[i].residual_x +
                           points[i].residual_y * points[i].residual_y);

            if (r > threshold) {
                points[i].valid = 0;
                removed++;
            }
        }

        n_valid -= removed;

        GM_LOG("RANSAC iter %d: median=%.3f mad=%.3f thresh=%.3f removed=%d valid=%d",
               iter, median, mad, threshold, removed, n_valid);

        if (removed < n_points * 0.05 || n_valid < 10) break;
    }

    *out_n_valid = n_valid;
    return PSM_OK;
}

GM_EXPORT int psm_grid_fit_affine(
    const GMControlPoint *points,
    int n_points,
    double *out_a0, double *out_a1, double *out_a2,
    double *out_b0, double *out_b1, double *out_b2)
{
    if (!points || n_points < 10) return PSM_ERR_INVALID_PARAM;

    std::vector<double> vx, vy, vcx, vcy;
    for (int i = 0; i < n_points; i++) {
        if (!points[i].valid) continue;
        vx.push_back(points[i].img_x);
        vy.push_back(points[i].img_y);
        vcx.push_back(points[i].cat_x);
        vcy.push_back(points[i].cat_y);
    }

    if ((int)vx.size() < 10) return PSM_ERR_NO_DATA;

    gm_fit_affine(vx.data(), vy.data(), vcx.data(), vcy.data(), (int)vx.size(),
                   *out_a0, *out_a1, *out_a2, *out_b0, *out_b1, *out_b2);
    return PSM_OK;
}

GM_EXPORT int psm_grid_match_perform(
    const GMImageStars *img_stars,
    const GMCatalogStars *cat_stars,
    const GMInitialTransform *init_transform,
    const GMConfig *config,
    GMResult *out_result)
{
    if (!img_stars || !cat_stars || !init_transform || !config || !out_result) {
        return PSM_ERR_INVALID_PARAM;
    }

    memset(out_result, 0, sizeof(GMResult));

    int sip_order = config->sip_order > 0 ? config->sip_order : GM_DEFAULT_SIP_ORDER;
    double match_tol = config->match_tolerance > 0 ? config->match_tolerance : GM_DEFAULT_MATCH_TOLERANCE;
    int max_ransac = config->max_ransac_iter > 0 ? config->max_ransac_iter : GM_DEFAULT_RANSAC_ITER;
    double ransac_sigma = config->ransac_sigma > 0 ? config->ransac_sigma : GM_DEFAULT_RANSAC_SIGMA;
    double centroid_radius = config->centroid_radius > 0 ? config->centroid_radius : GM_DEFAULT_CENTROID_RADIUS;
    double ratio_thresh = config->ratio_threshold > 0 ? config->ratio_threshold : GM_DEFAULT_RATIO_THRESHOLD;
    int n_img_bright = config->n_img_bright > 0 ? config->n_img_bright : GM_DEFAULT_N_IMG_BRIGHT;
    int n_cat_bright = config->n_cat_bright > 0 ? config->n_cat_bright : GM_DEFAULT_N_CAT_BRIGHT;

    int img_w = init_transform->img_width;
    int img_h = init_transform->img_height;

    double crval1_rad = init_transform->crval1 * M_PI / 180.0;
    double crval2_rad = init_transform->crval2 * M_PI / 180.0;
    double rad_to_px = 180.0 / M_PI * 3600.0 / init_transform->scale_arcsec_px;

    double cd[2][2] = {
        {init_transform->cd1_1, init_transform->cd1_2},
        {init_transform->cd2_1, init_transform->cd2_2}
    };
    double det_cd = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];

    GM_LOG("=== Triangle Match Start ===");
    GM_LOG("  Image: %d x %d  scale=%.3f arcsec/px  rad_to_px=%.3f",
           img_w, img_h, init_transform->scale_arcsec_px, rad_to_px);

    int n_img_use = std::min(n_img_bright, img_stars->img_count);
    int n_cat_use = std::min(n_cat_bright, cat_stars->cat_count);

    std::vector<double> use_img_x(n_img_use), use_img_y(n_img_use);
    for (int i = 0; i < n_img_use; i++) {
        use_img_x[i] = img_stars->img_x[i];
        use_img_y[i] = img_stars->img_y[i];
    }

    std::vector<double> cat_px(n_cat_use), cat_py(n_cat_use);
    for (int i = 0; i < n_cat_use; i++) {
        double ra_rad = cat_stars->cat_ra[i] * M_PI / 180.0;
        double dec_rad = cat_stars->cat_dec[i] * M_PI / 180.0;

        double xi, eta;
        gm_sky_to_plane(ra_rad, dec_rad, crval1_rad, crval2_rad, &xi, &eta);

        cat_px[i] = xi * rad_to_px;
        cat_py[i] = -eta * rad_to_px;
    }

    double half_w = img_w / 2.0;
    double half_h = img_h / 2.0;
    double margin = 100.0;

    std::vector<int> valid_cat_idx;
    for (int i = 0; i < n_cat_use; i++) {
        if (cat_px[i] >= -half_w - margin && cat_px[i] <= half_w + margin &&
            cat_py[i] >= -half_h - margin && cat_py[i] <= half_h + margin) {
            valid_cat_idx.push_back(i);
        }
    }

    GM_LOG("  Using %d img stars, %d Gaia stars (in range: %d)", n_img_use, n_cat_use, (int)valid_cat_idx.size());

    std::vector<double> vcat_px(valid_cat_idx.size()), vcat_py(valid_cat_idx.size());
    for (size_t k = 0; k < valid_cat_idx.size(); k++) {
        int i = valid_cat_idx[k];
        vcat_px[k] = cat_px[i];
        vcat_py[k] = cat_py[i];
    }

    {
        GMKDTree vtree;
        vtree.build(use_img_x.data(), use_img_y.data(), n_img_use);
        int nn5=0, nn10=0, nn20=0, nn50=0;
        for (size_t k = 0; k < valid_cat_idx.size(); k++) {
            auto nb = vtree.range_query(vcat_px[k], vcat_py[k], 50.0);
            double bd = 1e30;
            for (int j : nb) {
                double dd = sqrt((use_img_x[j]-vcat_px[k])*(use_img_x[j]-vcat_px[k])+(use_img_y[j]-vcat_py[k])*(use_img_y[j]-vcat_py[k]));
                if (dd < bd) bd = dd;
            }
            if (bd < 50) nn50++;
            if (bd < 20) nn20++;
            if (bd < 10) nn10++;
            if (bd < 5) nn5++;
        }
        GM_LOG("  VERIFY: Gaia->Img: within5=%d within10=%d within20=%d within50=%d / %d",
               nn5, nn10, nn20, nn50, (int)valid_cat_idx.size());
    }

    double tri_radius = 100.0;
    int max_tris = 50000;

    GM_LOG("  Building image triangles (radius=%.0f)...", tri_radius);
    std::vector<GMTriangle> img_tris;
    gm_build_triangles_radius(use_img_x.data(), use_img_y.data(), n_img_use, tri_radius, img_tris, max_tris);
    GM_LOG("  Image triangles: %d", (int)img_tris.size());

    int n_vcat = (int)valid_cat_idx.size();
    GM_LOG("  Building Gaia triangles (radius=%.0f)...", tri_radius);
    std::vector<GMTriangle> cat_tris;
    if (n_vcat >= 3) {
        gm_build_triangles_radius(vcat_px.data(), vcat_py.data(), n_vcat, tri_radius, cat_tris, max_tris);
    }
    GM_LOG("  Gaia triangles: %d", (int)cat_tris.size());

    if (img_tris.size() < 3 || cat_tris.size() < 3) {
        GM_LOG("  Too few triangles, aborting");
        return PSM_ERR_NO_DATA;
    }

    std::vector<double> cat_tri_cx(cat_tris.size()), cat_tri_cy(cat_tris.size());
    for (size_t i = 0; i < cat_tris.size(); i++) {
        cat_tri_cx[i] = cat_tris[i].cx;
        cat_tri_cy[i] = cat_tris[i].cy;
    }

    GMKDTree cat_tri_pos_tree;
    cat_tri_pos_tree.build(cat_tri_cx.data(), cat_tri_cy.data(), (int)cat_tris.size());

    double search_r = 50.0;
    double area_thresh = 0.25;
    double rotation_thresh = 15.0 * M_PI / 180.0;
    double ratio_thresh2 = ratio_thresh * ratio_thresh;

    GM_LOG("  Matching triangles (search_r=%.0f, ratio_thresh=%.3f, area_thresh=%.2f, rotation_thresh=%.1f°)...",
           search_r, ratio_thresh, area_thresh, rotation_thresh * 180.0 / M_PI);

    struct TriMatch { int img_ti; int cat_ti; };
    std::vector<TriMatch> tri_matches;

    for (size_t i = 0; i < img_tris.size(); i++) {
        const auto &it = img_tris[i];

        auto nearby = cat_tri_pos_tree.range_query(it.cx, it.cy, search_r);

        for (int j : nearby) {
            const auto &ct = cat_tris[j];

            double dba = it.ba_ratio - ct.ba_ratio;
            double dca = it.ca_ratio - ct.ca_ratio;
            if (dba*dba + dca*dca > ratio_thresh2) continue;

            double area_ratio = (it.area > ct.area) ? ct.area / it.area : it.area / ct.area;
            if (area_ratio < 1.0 - area_thresh) continue;

            if (it.handedness != ct.handedness) continue;

            double drot = fabs(it.rotation_angle - ct.rotation_angle);
            if (drot > rotation_thresh) continue;

            tri_matches.push_back({(int)i, j});
        }
    }

    GM_LOG("  Triangle matches: %d", (int)tri_matches.size());

    if (tri_matches.size() < 3) {
        GM_LOG("  Too few triangle matches, aborting");
        return PSM_ERR_NO_DATA;
    }

    double best_aff_a0=0, best_aff_a1=1, best_aff_a2=0;
    double best_aff_b0=0, best_aff_b1=0, best_aff_b2=1;
    int best_aff_inliers = 0;
    double aff_thresh2 = 20.0 * 20.0;

    int n_sample = std::min((int)tri_matches.size(), 2000);
    srand(42);

    GMKDTree cat_ransac_tree;
    cat_ransac_tree.build(vcat_px.data(), vcat_py.data(), n_vcat);

    for (int t = 0; t < n_sample; t++) {
        int mi = rand() % (int)tri_matches.size();
        const auto &tm = tri_matches[mi];
        const auto &it = img_tris[tm.img_ti];
        const auto &ct = cat_tris[tm.cat_ti];

        int img_v[3] = {it.i0, it.i1, it.i2};
        int cat_v[3] = {ct.i0, ct.i1, ct.i2};

        double img_d[3] = {it.d12, it.d02, it.d01};
        double cat_d[3] = {ct.d12, ct.d02, ct.d01};

        int img_ord[3] = {0, 1, 2};
        int cat_ord[3] = {0, 1, 2};
        std::sort(img_ord, img_ord + 3, [&](int a, int b) { return img_d[a] > img_d[b]; });
        std::sort(cat_ord, cat_ord + 3, [&](int a, int b) { return cat_d[a] > cat_d[b]; });

        int perm[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};

        for (int p = 0; p < 6; p++) {
            int co[3] = {cat_ord[perm[p][0]], cat_ord[perm[p][1]], cat_ord[perm[p][2]]};

            double ix[3], iy[3], cx[3], cy[3];
            for (int k = 0; k < 3; k++) {
                ix[k] = use_img_x[img_v[img_ord[k]]];
                iy[k] = use_img_y[img_v[img_ord[k]]];
                cx[k] = vcat_px[cat_v[co[k]]];
                cy[k] = vcat_py[cat_v[co[k]]];
            }

            double ta0, ta1, ta2, tb0, tb1, tb2;
            gm_fit_affine(ix, iy, cx, cy, 3, ta0, ta1, ta2, tb0, tb1, tb2);

            if (fabs(ta1) > 2.0 || fabs(ta2) > 2.0 || fabs(tb1) > 2.0 || fabs(tb2) > 2.0) continue;

            int inliers = 0;
            for (int s = 0; s < n_img_use; s++) {
                double px = ta0 + ta1 * use_img_x[s] + ta2 * use_img_y[s];
                double py = tb0 + tb1 * use_img_x[s] + tb2 * use_img_y[s];
                auto nb = cat_ransac_tree.range_query(px, py, sqrt(aff_thresh2));
                if (!nb.empty()) inliers++;
            }

            if (inliers > best_aff_inliers) {
                best_aff_inliers = inliers;
                best_aff_a0 = ta0; best_aff_a1 = ta1; best_aff_a2 = ta2;
                best_aff_b0 = tb0; best_aff_b1 = tb1; best_aff_b2 = tb2;
            }
        }
    }

    GM_LOG("  Best Affine from triangles: inliers=%d a0=%.2f a1=%.6f a2=%.6f b0=%.2f b1=%.6f b2=%.6f",
           best_aff_inliers, best_aff_a0, best_aff_a1, best_aff_a2, best_aff_b0, best_aff_b1, best_aff_b2);

    if (best_aff_inliers < 10) {
        GM_LOG("  Too few Affine inliers, aborting");
        return PSM_ERR_NO_DATA;
    }

    GMKDTree cat_nn_tree;
    cat_nn_tree.build(vcat_px.data(), vcat_py.data(), n_vcat);

    std::vector<double> refine_ix, refine_iy, refine_cx, refine_cy;
    double nn_radius = 20.0;

    for (int i = 0; i < n_img_use; i++) {
        double px = best_aff_a0 + best_aff_a1 * use_img_x[i] + best_aff_a2 * use_img_y[i];
        double py = best_aff_b0 + best_aff_b1 * use_img_x[i] + best_aff_b2 * use_img_y[i];

        auto nearby = cat_nn_tree.range_query(px, py, nn_radius);
        double best_d = 1e30;
        int best_j = -1;
        for (int j : nearby) {
            double ddx = px - vcat_px[j];
            double ddy = py - vcat_py[j];
            double d = sqrt(ddx*ddx + ddy*ddy);
            if (d < best_d) { best_d = d; best_j = j; }
        }
        if (best_j >= 0) {
            refine_ix.push_back(use_img_x[i]);
            refine_iy.push_back(use_img_y[i]);
            refine_cx.push_back(vcat_px[best_j]);
            refine_cy.push_back(vcat_py[best_j]);
        }
    }

    GM_LOG("  NN pairs after Affine projection (radius=%.0f): %d", nn_radius, (int)refine_ix.size());

    for (int iter = 0; iter < 3; iter++) {
        if ((int)refine_ix.size() < 10) break;

        double ia0, ia1, ia2, ib0, ib1, ib2;
        gm_fit_affine(refine_ix.data(), refine_iy.data(),
                       refine_cx.data(), refine_cy.data(), (int)refine_ix.size(),
                       ia0, ia1, ia2, ib0, ib1, ib2);

        std::vector<double> res((int)refine_ix.size());
        for (size_t k = 0; k < refine_ix.size(); k++) {
            double px = ia0 + ia1*refine_ix[k] + ia2*refine_iy[k];
            double py = ib0 + ib1*refine_ix[k] + ib2*refine_iy[k];
            res[k] = sqrt((px-refine_cx[k])*(px-refine_cx[k])+(py-refine_cy[k])*(py-refine_cy[k]));
        }
        std::sort(res.begin(), res.end());
        double med = res[res.size()/2];
        std::vector<double> dev(res.size());
        for (size_t k = 0; k < res.size(); k++) dev[k] = fabs(res[k]-med);
        std::sort(dev.begin(), dev.end());
        double mad = dev[dev.size()/2];
        double thresh = med + 3.0 * mad * 1.4826;

        std::vector<double> clean_ix, clean_iy, clean_cx, clean_cy;
        for (size_t k = 0; k < refine_ix.size(); k++) {
            if (res[k] < thresh) {
                clean_ix.push_back(refine_ix[k]);
                clean_iy.push_back(refine_iy[k]);
                clean_cx.push_back(refine_cx[k]);
                clean_cy.push_back(refine_cy[k]);
            }
        }

        gm_fit_affine(clean_ix.data(), clean_iy.data(),
                       clean_cx.data(), clean_cy.data(), (int)clean_ix.size(),
                       ia0, ia1, ia2, ib0, ib1, ib2);

        refine_ix.clear(); refine_iy.clear(); refine_cx.clear(); refine_cy.clear();
        for (int i = 0; i < n_img_use; i++) {
            double px = ia0 + ia1 * use_img_x[i] + ia2 * use_img_y[i];
            double py = ib0 + ib1 * use_img_x[i] + ib2 * use_img_y[i];

            auto nearby = cat_nn_tree.range_query(px, py, nn_radius);
            double best_d = 1e30;
            int best_j = -1;
            for (int j : nearby) {
                double ddx = px - vcat_px[j];
                double ddy = py - vcat_py[j];
                double d = sqrt(ddx*ddx + ddy*ddy);
                if (d < best_d) { best_d = d; best_j = j; }
            }
            if (best_j >= 0) {
                refine_ix.push_back(use_img_x[i]);
                refine_iy.push_back(use_img_y[i]);
                refine_cx.push_back(vcat_px[best_j]);
                refine_cy.push_back(vcat_py[best_j]);
            }
        }

        GM_LOG("  Iter %d: %d pairs (med=%.2f mad=%.2f)", iter, (int)refine_ix.size(), med, mad);
    }

    if ((int)refine_ix.size() < 10) {
        GM_LOG("  Too few refined pairs, aborting");
        return PSM_ERR_NO_DATA;
    }

    double aff_a0, aff_a1, aff_a2, aff_b0, aff_b1, aff_b2;
    gm_fit_affine(refine_ix.data(), refine_iy.data(),
                   refine_cx.data(), refine_cy.data(), (int)refine_ix.size(),
                   aff_a0, aff_a1, aff_a2, aff_b0, aff_b1, aff_b2);

    GM_LOG("  Step2 Affine: a0=%.3f a1=%.6f a2=%.6f b0=%.3f b1=%.6f b2=%.6f",
           aff_a0, aff_a1, aff_a2, aff_b0, aff_b1, aff_b2);

    {
        std::vector<double> residuals;
        for (size_t k = 0; k < refine_ix.size(); k++) {
            double px = aff_a0 + aff_a1 * refine_ix[k] + aff_a2 * refine_iy[k];
            double py = aff_b0 + aff_b1 * refine_ix[k] + aff_b2 * refine_iy[k];
            residuals.push_back(sqrt((px-refine_cx[k])*(px-refine_cx[k])+(py-refine_cy[k])*(py-refine_cy[k])));
        }
        std::sort(residuals.begin(), residuals.end());
        GM_LOG("  Affine residual: median=%.3f max=%.3f", residuals[residuals.size()/2], residuals.back());
    }

    int n_control_points = (int)refine_ix.size();
    out_result->control_points = (GMControlPoint *)malloc(n_control_points * sizeof(GMControlPoint));
    if (!out_result->control_points) return PSM_ERR_NO_MEMORY;

    for (int i = 0; i < n_control_points; i++) {
        GMControlPoint *cp = &out_result->control_points[i];
        cp->grid_row = 0; cp->grid_col = 0;
        cp->img_star_idx = i; cp->cat_star_idx = i;
        cp->img_x = refine_ix[i]; cp->img_y = refine_iy[i];
        cp->cat_x = refine_cx[i]; cp->cat_y = refine_cy[i];
        double pred_x = aff_a0 + aff_a1 * refine_ix[i] + aff_a2 * refine_iy[i];
        double pred_y = aff_b0 + aff_b1 * refine_ix[i] + aff_b2 * refine_iy[i];
        cp->residual_x = refine_cx[i] - pred_x;
        cp->residual_y = refine_cy[i] - pred_y;
        cp->valid = 1;
    }

    out_result->n_control_points = n_control_points;
    out_result->n_grids_matched = n_control_points;
    out_result->n_grids_total = n_control_points;

    int n_valid;
    psm_grid_ransac_filter(out_result->control_points, n_control_points,
                            5, ransac_sigma, &n_valid);
    out_result->n_ransac_removed = n_control_points - n_valid;

    GM_LOG("  After MAD filter: %d valid, %d removed", n_valid, out_result->n_ransac_removed);

    double sum_rx = 0, sum_ry = 0, sum_r2 = 0;
    int nn = 0;
    for (int i = 0; i < n_control_points; i++) {
        if (!out_result->control_points[i].valid) continue;
        double rx = out_result->control_points[i].residual_x;
        double ry = out_result->control_points[i].residual_y;
        sum_rx += rx * rx; sum_ry += ry * ry; sum_r2 += rx*rx + ry*ry;
        nn++;
    }

    if (nn > 0) {
        out_result->rms_x = sqrt(sum_rx / nn);
        out_result->rms_y = sqrt(sum_ry / nn);
        out_result->rms_total = sqrt(sum_r2 / nn);
        out_result->rms_arcsec = out_result->rms_total * init_transform->scale_arcsec_px;
    }

    GM_LOG("  RMS: %.3f px (%.3f arcsec)", out_result->rms_total, out_result->rms_arcsec);

    std::vector<double> fit_x(nn), fit_y(nn), fit_cx(nn), fit_cy(nn);
    int kk = 0;
    for (int i = 0; i < n_control_points; i++) {
        if (!out_result->control_points[i].valid) continue;
        fit_x[kk] = out_result->control_points[i].img_x;
        fit_y[kk] = out_result->control_points[i].img_y;
        fit_cx[kk] = out_result->control_points[i].cat_x;
        fit_cy[kk] = out_result->control_points[i].cat_y;
        kk++;
    }

    SIP_Transform trans;
    SIP_Coeffs sip_coeff;
    double sip_rms;

    int sip_status = sip_fit_refine(
        fit_x.data(), fit_y.data(), kk,
        fit_cx.data(), fit_cy.data(), kk,
        0.0, 0.0,
        img_w, img_h,
        sip_order,
        &trans,
        &sip_coeff,
        &sip_rms);

    if (sip_status == PSM_OK && sip_coeff.valid) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                out_result->sip_A[i][j] = sip_coeff.A[i][j];
                out_result->sip_B[i][j] = sip_coeff.B[i][j];
                out_result->sip_AP[i][j] = sip_coeff.AP[i][j];
                out_result->sip_BP[i][j] = sip_coeff.BP[i][j];
            }
        }
        out_result->sip_order = sip_order;
        out_result->sip_valid = 1;

        double sip_cd[2][2];
        sip_trans_get_cd(&trans, sip_cd);
        out_result->cd[0][0] = sip_cd[0][0];
        out_result->cd[0][1] = sip_cd[0][1];
        out_result->cd[1][0] = sip_cd[1][0];
        out_result->cd[1][1] = sip_cd[1][1];

        out_result->crpix[0] = init_transform->crpix1;
        out_result->crpix[1] = init_transform->crpix2;
        out_result->crval[0] = init_transform->crval1;
        out_result->crval[1] = init_transform->crval2;

        GM_LOG("  SIP fitted: order=%d rms=%.3f px", sip_order, sip_rms);
        GM_LOG("  WCS: CRVAL=(%.6f, %.6f) CD=[%.6f,%.6f;%.6f,%.6f]",
               out_result->crval[0], out_result->crval[1],
               out_result->cd[0][0], out_result->cd[0][1],
               out_result->cd[1][0], out_result->cd[1][1]);
    } else {
        out_result->crpix[0] = init_transform->crpix1;
        out_result->crpix[1] = init_transform->crpix2;
        out_result->crval[0] = init_transform->crval1;
        out_result->crval[1] = init_transform->crval2;
        out_result->cd[0][0] = init_transform->cd1_1;
        out_result->cd[0][1] = init_transform->cd1_2;
        out_result->cd[1][0] = init_transform->cd2_1;
        out_result->cd[1][1] = init_transform->cd2_2;
        GM_LOG("  SIP fit failed, using initial WCS");
    }

    GM_LOG("=== Triangle Match Complete ===");
    return PSM_OK;
}

GM_EXPORT void psm_grid_match_free_result(GMResult *result)
{
    if (result) {
        free(result->control_points);
        result->control_points = NULL;
    }
}

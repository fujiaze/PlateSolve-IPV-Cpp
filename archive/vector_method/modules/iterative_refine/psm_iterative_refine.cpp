#include "psm_iterative_refine.h"
#include "psm_sip.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IR_LOG(fmt, ...) fprintf(stderr, "[IR] " fmt "\n", ##__VA_ARGS__)

static void ir_sky_to_plane(double ra, double dec, double ra0, double dec0, double *x, double *y)
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

static void ir_plane_to_sky(double x, double y, double ra0, double dec0, double *ra, double *dec)
{
    double r = sqrt(x * x + y * y);
    if (r < 1e-10) { *ra = ra0; *dec = dec0; return; }
    double c = atan(r);
    double sin_c = sin(c), cos_c = cos(c);
    double sin_dec0 = sin(dec0), cos_dec0 = cos(dec0);
    *dec = asin(cos_c * sin_dec0 + sin_c * cos_dec0 * y / r);
    *ra = ra0 + atan2(x * sin_c / r, cos_c * cos_dec0 - sin_c * sin_dec0 * y / r);
}

IR_EXPORT int psm_iterative_refine(
    const IRImageStars *img_stars,
    const IRCatalogStars *cat_stars,
    const IRInitialTransform *init_transform,
    const IRConfig *config,
    IRRefineResult *out_result)
{
    if (!img_stars || !cat_stars || !init_transform || !config || !out_result) {
        return PSM_ERR_INVALID_PARAM;
    }

    memset(out_result, 0, sizeof(IRRefineResult));

    double center_ra = init_transform->center_ra * M_PI / 180.0;
    double center_dec = init_transform->center_dec * M_PI / 180.0;
    double rotation = init_transform->rotation_deg * M_PI / 180.0;
    double scale = init_transform->scale_arcsec_px;
    int flip_mode = init_transform->flip_mode;
    int img_w = init_transform->img_width;
    int img_h = init_transform->img_height;

    int flip_x = (flip_mode == 1 || flip_mode == 3) ? 1 : 0;
    int flip_y = (flip_mode == 2 || flip_mode == 3) ? 1 : 0;

    double half_w = img_w / 2.0;
    double half_h = img_h / 2.0;
    double rad_to_px = 180.0 / M_PI * 3600.0 / scale;

    int sip_order = config->sip_order > 0 ? config->sip_order : 5;

    IR_LOG("=== Step 2: SIP Fitting (order %d) ===", sip_order);
    IR_LOG("  Initial: RA=%.6f Dec=%.6f rot=%.3f scale=%.3f",
           init_transform->center_ra, init_transform->center_dec,
           init_transform->rotation_deg, scale);

    std::vector<double> cat_px(cat_stars->cat_count);
    std::vector<double> cat_py(cat_stars->cat_count);
    for (int i = 0; i < cat_stars->cat_count; i++) {
        double px = cat_stars->cat_x_px[i];
        double py = cat_stars->cat_y_px[i];
        if (flip_x) px = -px;
        if (flip_y) py = -py;
        double cos_r = cos(rotation), sin_r = sin(rotation);
        cat_px[i] = cos_r * px - sin_r * py;
        cat_py[i] = sin_r * px + cos_r * py;
    }

    std::vector<int> cat_valid_indices;
    for (int i = 0; i < cat_stars->cat_count; i++) {
        if (fabs(cat_px[i]) < half_w && fabs(cat_py[i]) < half_h) {
            cat_valid_indices.push_back(i);
        }
    }
    int n_cat_valid = (int)cat_valid_indices.size();

    std::vector<int> img_indices;
    for (int i = 0; i < img_stars->img_count; i++) {
        if (!img_stars->img_saturated[i]) img_indices.push_back(i);
    }
    int n_img_top = std::min(2000, (int)img_indices.size());
    std::partial_sort(img_indices.begin(), img_indices.begin() + n_img_top, img_indices.end(),
        [&](int a, int b) { return img_stars->img_flux[a] > img_stars->img_flux[b]; });
    img_indices.resize(n_img_top);

    int n_cat_top = std::min(3000, n_cat_valid);
    std::partial_sort(cat_valid_indices.begin(), cat_valid_indices.begin() + n_cat_top, cat_valid_indices.end(),
        [&](int a, int b) { return cat_stars->cat_mag[a] < cat_stars->cat_mag[b]; });
    cat_valid_indices.resize(n_cat_top);

    std::vector<std::pair<int, int>> matches;
    double max_match_dist = 20.0;
    for (int i = 0; i < n_img_top; i++) {
        int img_idx = img_indices[i];
        double ix = img_stars->img_x[img_idx];
        double iy = img_stars->img_y[img_idx];
        double best_dist = max_match_dist * max_match_dist;
        int best_cat = -1;
        for (int j = 0; j < n_cat_top; j++) {
            int cat_idx = cat_valid_indices[j];
            double dx = ix - cat_px[cat_idx];
            double dy = iy - cat_py[cat_idx];
            double dist2 = dx * dx + dy * dy;
            if (dist2 < best_dist) { best_dist = dist2; best_cat = cat_idx; }
        }
        if (best_cat >= 0) matches.push_back({img_idx, best_cat});
    }
    IR_LOG("  Matches: %zu pairs", matches.size());

    if (matches.size() < 100) {
        out_result->final_ra = init_transform->center_ra;
        out_result->final_dec = init_transform->center_dec;
        out_result->final_rotation = init_transform->rotation_deg;
        out_result->final_scale = scale;
        return PSM_OK;
    }

    int n_match = (int)matches.size();
    std::vector<double> match_img_x(n_match), match_img_y(n_match);
    std::vector<double> match_cat_x(n_match), match_cat_y(n_match);
    for (int i = 0; i < n_match; i++) {
        match_img_x[i] = img_stars->img_x[matches[i].first];
        match_img_y[i] = img_stars->img_y[matches[i].first];
        match_cat_x[i] = cat_px[matches[i].second];
        match_cat_y[i] = cat_py[matches[i].second];
    }

    IR_LOG("\n  === Residual vs Distance Analysis ===");
    std::vector<double> dist_to_center(n_match), residual_mag(n_match);
    for (int i = 0; i < n_match; i++) {
        double dx = match_cat_x[i] - match_img_x[i];
        double dy = match_cat_y[i] - match_img_y[i];
        dist_to_center[i] = sqrt(match_img_x[i] * match_img_x[i] + match_img_y[i] * match_img_y[i]);
        residual_mag[i] = sqrt(dx * dx + dy * dy);
    }
    
    std::vector<std::pair<double, double>> dist_resid(n_match);
    for (int i = 0; i < n_match; i++) dist_resid[i] = {dist_to_center[i], residual_mag[i]};
    std::sort(dist_resid.begin(), dist_resid.end());
    
    IR_LOG("  Distance bins: residual (median)");
    for (int bin = 0; bin < 10; bin++) {
        int start = bin * n_match / 10;
        int end = (bin + 1) * n_match / 10;
        double dist_med = dist_resid[(start + end) / 2].first;
        std::vector<double> bin_resid;
        for (int i = start; i < end; i++) bin_resid.push_back(dist_resid[i].second);
        std::nth_element(bin_resid.begin(), bin_resid.begin() + bin_resid.size() / 2, bin_resid.end());
        IR_LOG("    r=%6.0f px: resid=%6.3f px", dist_med, bin_resid[bin_resid.size() / 2]);
    }

    SIP_Transform trans;
    SIP_Coeffs sip_coeff;
    double rms;
    
    int status = sip_fit_refine(
        match_img_x.data(), match_img_y.data(), n_match,
        match_cat_x.data(), match_cat_y.data(), n_match,
        0.0, 0.0,
        img_w, img_h,
        sip_order,
        &trans,
        &sip_coeff,
        &rms);
    
    if (status) {
        IR_LOG("  SIP fitting failed, using fallback");
        out_result->final_ra = init_transform->center_ra;
        out_result->final_dec = init_transform->center_dec;
        out_result->final_rotation = init_transform->rotation_deg;
        out_result->final_scale = scale;
        return PSM_OK;
    }
    
    IR_LOG("  Transform: x00=%.4f x10=%.6f x01=%.6f", trans.x00, trans.x10, trans.x01);
    IR_LOG("             y00=%.4f y10=%.6f y01=%.6f", trans.y00, trans.y10, trans.y01);
    IR_LOG("  RMS after fit: %.4f px", rms);
    
    if (sip_coeff.valid) {
        IR_LOG("  SIP coefficients computed:");
        IR_LOG("    A_20=%.2e A_11=%.2e A_02=%.2e", 
               sip_coeff.A[2][0], sip_coeff.A[1][1], sip_coeff.A[0][2]);
        if (sip_order >= 3) {
            IR_LOG("    A_30=%.2e A_21=%.2e A_12=%.2e A_03=%.2e",
                   sip_coeff.A[3][0], sip_coeff.A[2][1], sip_coeff.A[1][2], sip_coeff.A[0][3]);
        }
        if (sip_order >= 4) {
            IR_LOG("    A_40=%.2e A_31=%.2e A_22=%.2e A_13=%.2e A_04=%.2e",
                   sip_coeff.A[4][0], sip_coeff.A[3][1], sip_coeff.A[2][2], 
                   sip_coeff.A[1][3], sip_coeff.A[0][4]);
        }
        if (sip_order >= 5) {
            IR_LOG("    A_50=%.2e A_41=%.2e A_32=%.2e A_23=%.2e A_14=%.2e A_05=%.2e",
                   sip_coeff.A[5][0], sip_coeff.A[4][1], sip_coeff.A[3][2],
                   sip_coeff.A[2][3], sip_coeff.A[1][4], sip_coeff.A[0][5]);
        }
        
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                out_result->sip.A[i][j] = sip_coeff.A[i][j];
                out_result->sip.B[i][j] = sip_coeff.B[i][j];
                out_result->sip.AP[i][j] = sip_coeff.AP[i][j];
                out_result->sip.BP[i][j] = sip_coeff.BP[i][j];
            }
        }
        out_result->sip.sip_order = sip_order;
        out_result->sip.sip_valid = 1;
    }

    double xi_rad = trans.x00 / rad_to_px;
    double eta_rad = trans.y00 / rad_to_px;
    double new_ra, new_dec;
    ir_plane_to_sky(xi_rad, eta_rad, center_ra, center_dec, &new_ra, &new_dec);

    double cd[2][2];
    sip_trans_get_cd(&trans, cd);
    
    double det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    double new_scale = sqrt(fabs(det)) * scale;

    out_result->final_ra = new_ra * 180.0 / M_PI;
    out_result->final_dec = new_dec * 180.0 / M_PI;
    
    double new_rot = atan2(cd[1][0], cd[0][0]);
    out_result->final_rotation = (rotation + new_rot) * 180.0 / M_PI;
    out_result->final_scale = new_scale;

    out_result->cd[0][0] = cd[0][0] * scale / 3600.0;
    out_result->cd[0][1] = cd[0][1] * scale / 3600.0;
    out_result->cd[1][0] = cd[1][0] * scale / 3600.0;
    out_result->cd[1][1] = cd[1][1] * scale / 3600.0;
    
    out_result->crpix[0] = half_w;
    out_result->crpix[1] = half_h;
    out_result->crval[0] = out_result->final_ra;
    out_result->crval[1] = out_result->final_dec;

    out_result->dist_a0 = trans.x00;
    out_result->dist_b0 = trans.y00;
    out_result->dist_a1 = cd[0][0] - 1;
    out_result->dist_a2 = cd[0][1];
    out_result->dist_b1 = cd[1][0];
    out_result->dist_b2 = cd[1][1] - 1;
    out_result->distortion_valid = 1;

    double delta_ra = (out_result->final_ra - init_transform->center_ra) * 3600 * cos(center_dec);
    double delta_dec = (out_result->final_dec - init_transform->center_dec) * 3600;
    IR_LOG("\n  Final correction: ΔRA=%.2f\" ΔDec=%.2f\" Δrot=%.4f° Δscale=%.6f",
           delta_ra, delta_dec, new_rot * 180.0 / M_PI, new_scale / scale - 1.0);

    out_result->matched_count = n_match;
    out_result->triangle_matches = n_match;
    out_result->iteration_count = 1;
    out_result->rms_total = rms;
    out_result->rms_x = rms / sqrt(2);
    out_result->rms_y = rms / sqrt(2);
    out_result->rms_arcsec = rms * scale;

    IR_LOG("=== Step 2 Complete ===");
    return PSM_OK;
}

IR_EXPORT void psm_free_refine_result(IRRefineResult *result)
{
    if (result) {
        free(result->img_indices);
        free(result->cat_indices);
        free(result->residual_x);
        free(result->residual_y);
        result->img_indices = NULL;
        result->cat_indices = NULL;
        result->residual_x = NULL;
        result->residual_y = NULL;
    }
}

IR_EXPORT int psm_rect_fov_filter(const double *cat_x, const double *cat_y, int cat_count,
    double half_w, double half_h, int **out_indices, int *out_count)
{
    if (!cat_x || !cat_y || !out_indices || !out_count) return PSM_ERR_INVALID_PARAM;
    std::vector<int> valid;
    for (int i = 0; i < cat_count; i++) {
        if (fabs(cat_x[i]) < half_w && fabs(cat_y[i]) < half_h) valid.push_back(i);
    }
    *out_count = (int)valid.size();
    if (valid.empty()) { *out_indices = NULL; return PSM_OK; }
    *out_indices = (int *)malloc(valid.size() * sizeof(int));
    memcpy(*out_indices, valid.data(), valid.size() * sizeof(int));
    return PSM_OK;
}

IR_EXPORT int psm_regional_outlier_filter(const double *dx, const double *dy, int n,
    const double *cx, const double *cy, int grid_size, double angle_thresh, double mag_ratio,
    int **out_mask, int *out_kept)
{
    if (!dx || !dy || !cx || !cy || !out_mask || !out_kept) return PSM_ERR_INVALID_PARAM;
    *out_mask = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) (*out_mask)[i] = 1;
    *out_kept = n;
    return PSM_OK;
}

IR_EXPORT int psm_fit_distortion_model(const double *x, const double *y,
    const double *dx, const double *dy, int n, double *out_dist_a, double *out_dist_b)
{
    if (n < 10) return -1;
    std::vector<double> sorted_dx(dx, dx + n), sorted_dy(dy, dy + n);
    std::nth_element(sorted_dx.begin(), sorted_dx.begin() + n/2, sorted_dx.end());
    std::nth_element(sorted_dy.begin(), sorted_dy.begin() + n/2, sorted_dy.end());
    out_dist_a[0] = sorted_dx[n/2];
    out_dist_b[0] = sorted_dy[n/2];
    return 0;
}

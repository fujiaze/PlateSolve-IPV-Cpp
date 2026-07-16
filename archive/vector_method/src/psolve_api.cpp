#include "plate_solve.h"
#include "psolve_log.h"
#include "psolve_coarse.h"
#include "psolve_fine.h"
#include "../modules/star_alignment/psm_star_alignment.h"
#include "../modules/iterative_refine/psm_iterative_refine.h"
#include "../modules/iterative_refine/psm_grid_match.h"
#include "../../gaia_xpsd_client/src/gaia_client.h"
#include "../../star_detector/include/star_detector.h"
#include "../../astro_image_io/include/astro_image_io.h"
#include <stdlib.h>
#include <string.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct PSolveHandle_s {
    void *gaia_client;
    double last_scale;
    PSolveAffine last_affine;
    PSolveCoarseResult last_coarse;
    int has_coarse;
    PSolveWCS last_wcs;
    int has_wcs;
    PSolveResult last_result;
    int has_result;
};

typedef struct PSolveHandle_s PSolveHandleInternal;

static PSolveHandle psolve_create_impl(const char *gaia_data_dir, int db_type) {
    if (!gaia_data_dir) return nullptr;

    PSolveHandleInternal *handle = (PSolveHandleInternal *)calloc(1, sizeof(PSolveHandleInternal));
    if (!handle) return nullptr;

    if (db_type == GAIA_DB_AUTO) {
        handle->gaia_client = (void *)gaia_client_create(gaia_data_dir);
    } else {
        handle->gaia_client = (void *)gaia_client_create_ex(gaia_data_dir, (GaiaDbType)db_type);
    }
    
    if (!handle->gaia_client) {
        free(handle);
        return nullptr;
    }

    char log_dir[1024];
    const char *last_sep = strrchr(gaia_data_dir, '/');
    if (!last_sep) last_sep = strrchr(gaia_data_dir, '\\');
    if (last_sep) {
        size_t parent_len = last_sep - gaia_data_dir;
        memcpy(log_dir, gaia_data_dir, parent_len);
        log_dir[parent_len] = '\0';
    } else {
        strcpy(log_dir, ".");
    }
    size_t dlen = strlen(log_dir);
#ifdef _WIN32
    snprintf(log_dir + dlen, sizeof(log_dir) - dlen, "\\plate_solve\\logs");
#else
    snprintf(log_dir + dlen, sizeof(log_dir) - dlen, "/plate_solve/logs");
#endif
    psolve_log_init(log_dir);
    
    int db_type_detected = gaia_client_get_db_type((GaiaClient *)handle->gaia_client);
    const char *db_type_name = "UNKNOWN";
    if (db_type_detected == GAIA_DB_DR3) db_type_name = "DR3";
    else if (db_type_detected == GAIA_DB_DR3SP) db_type_name = "DR3SP";
    else if (db_type_detected == GAIA_DB_AUTO) db_type_name = "AUTO";
    
    PSLOG_I("Plate solver created, gaia_data_dir=%s, db_type=%s", gaia_data_dir, db_type_name);

    return (PSolveHandle)handle;
}

PSOLVE_EXPORT PSolveHandle psolve_create(const char *gaia_data_dir) {
    return psolve_create_impl(gaia_data_dir, GAIA_DB_AUTO);
}

PSOLVE_EXPORT PSolveHandle psolve_create_ex(const char *gaia_data_dir, PSolveDbType db_type) {
    return psolve_create_impl(gaia_data_dir, (int)db_type);
}

PSOLVE_EXPORT int psolve_get_db_type(PSolveHandle handle) {
    if (!handle) return PSOLVE_DB_AUTO;
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;
    return gaia_client_get_db_type((GaiaClient *)h->gaia_client);
}

PSOLVE_EXPORT void psolve_destroy(PSolveHandle handle) {
    if (!handle) return;
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;

    if (h->gaia_client) {
        gaia_client_destroy((GaiaClient *)h->gaia_client);
        h->gaia_client = nullptr;
    }

    psolve_log_close();
    free(h);
}

static void gnomonic_projection(
    const double *ra_rad, const double *dec_rad, int n,
    double center_ra_rad, double center_dec_rad,
    double *out_x, double *out_y)
{
    double cos_dec0 = cos(center_dec_rad);
    double sin_dec0 = sin(center_dec_rad);
    
    for (int i = 0; i < n; i++) {
        double cos_dec = cos(dec_rad[i]);
        double sin_dec = sin(dec_rad[i]);
        double ra_diff = ra_rad[i] - center_ra_rad;
        double cos_ra_diff = cos(ra_diff);
        double sin_ra_diff = sin(ra_diff);
        double cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff;
        
        if (cos_c > 1e-10) {
            out_x[i] = cos_dec * sin_ra_diff / cos_c;
            out_y[i] = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_ra_diff) / cos_c;
        } else {
            out_x[i] = 1e30;
            out_y[i] = 1e30;
        }
    }
}

static int query_gaia_stars(
    GaiaClient *client,
    double center_ra, double center_dec,
    double radius_deg, double limit_mag,
    std::vector<double> &out_ra, std::vector<double> &out_dec, std::vector<double> &out_mag)
{
    double *ra_ptr = nullptr, *dec_ptr = nullptr;
    float *mag_ptr = nullptr;
    int n_stars = 0;
    
    int rc = gaia_client_cone_search_for_solver(
        client, center_ra, center_dec, radius_deg, limit_mag,
        &ra_ptr, &dec_ptr, &mag_ptr, &n_stars);
    
    if (rc != 0 || n_stars <= 0) {
        if (ra_ptr) free(ra_ptr);
        if (dec_ptr) free(dec_ptr);
        if (mag_ptr) free(mag_ptr);
        return -1;
    }
    
    out_ra.resize(n_stars);
    out_dec.resize(n_stars);
    out_mag.resize(n_stars);
    
    for (int i = 0; i < n_stars; i++) {
        out_ra[i] = ra_ptr[i];
        out_dec[i] = dec_ptr[i];
        out_mag[i] = mag_ptr[i];
    }
    
    free(ra_ptr);
    free(dec_ptr);
    free(mag_ptr);
    
    return n_stars;
}

PSOLVE_EXPORT int psolve_solve(
    PSolveHandle handle,
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int img_count, int n_saturated,
    const PSolveImageData *img_data,
    const PSolveConfig *config,
    PSolveResult *result)
{
    if (!handle || !img_x || !img_y || !img_data || !config || !result) {
        return PSOLVE_ERR_INTERNAL;
    }
    
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;
    GaiaClient *gaia = (GaiaClient *)h->gaia_client;
    
    memset(result, 0, sizeof(PSolveResult));
    
    PSLOG_I("=== psolve_solve: Two-step Plate Solving ===");
    
    double t_start = 0.0;
#ifdef _WIN32
    t_start = (double)GetTickCount() / 1000.0;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    t_start = ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
    
    int width = img_data->width;
    int height = img_data->height;
    double center_ra = img_data->center_ra;
    double center_dec = img_data->center_dec;
    
    double scale_arcsec_px;
    if (img_data->scale_arcsec_px > 0) {
        scale_arcsec_px = img_data->scale_arcsec_px;
    } else {
        scale_arcsec_px = 206.265 * img_data->pixel_size_um / img_data->focal_length_mm;
    }
    double fov_diag = sqrt((double)width * width + (double)height * height) * scale_arcsec_px / 3600.0;
    double query_radius = fov_diag * 1.2 / 2.0;
    
    PSLOG_I("  Image: %dx%d, scale=%.3f\"/px, FOV=%.2f°", width, height, scale_arcsec_px, fov_diag);
    PSLOG_I("  Center: RA=%.6f°, Dec=%.6f°", center_ra, center_dec);
    
    int target_count = img_count * 3 / 2;
    double mag_low = 6.0, mag_high = 22.0;
    
    for (int iter = 0; iter < 10; iter++) {
        double mag_mid = (mag_low + mag_high) / 2.0;
        std::vector<double> tmp_ra, tmp_dec, tmp_mag;
        int n = query_gaia_stars(gaia, center_ra, center_dec, query_radius, mag_mid, tmp_ra, tmp_dec, tmp_mag);
        if (n < 0) { mag_high = mag_mid; continue; }
        if (n > target_count) mag_high = mag_mid;
        else mag_low = mag_mid;
        if (mag_high - mag_low < 0.1) break;
    }
    
    double limit_mag = (mag_low + mag_high) / 2.0;
    std::vector<double> cat_ra, cat_dec, cat_mag;
    int n_cat = query_gaia_stars(gaia, center_ra, center_dec, query_radius, limit_mag, cat_ra, cat_dec, cat_mag);
    
    if (n_cat <= 0) {
        PSLOG_E("  Gaia query failed");
        return PSOLVE_ERR_NO_MATCH;
    }
    
    PSLOG_I("  Gaia stars: %d, limit_mag=%.2f", n_cat, limit_mag);
    
    std::vector<int> sort_idx(n_cat);
    for (int i = 0; i < n_cat; i++) sort_idx[i] = i;
    std::sort(sort_idx.begin(), sort_idx.end(), [&](int a, int b) { return cat_mag[a] < cat_mag[b]; });
    
    std::vector<double> cat_ra_sorted(n_cat), cat_dec_sorted(n_cat), cat_mag_sorted(n_cat);
    for (int i = 0; i < n_cat; i++) {
        cat_ra_sorted[i] = cat_ra[sort_idx[i]];
        cat_dec_sorted[i] = cat_dec[sort_idx[i]];
        cat_mag_sorted[i] = cat_mag[sort_idx[i]];
    }
    cat_ra = std::move(cat_ra_sorted);
    cat_dec = std::move(cat_dec_sorted);
    cat_mag = std::move(cat_mag_sorted);
    
    std::vector<double> cat_ra_rad(n_cat), cat_dec_rad(n_cat);
    double center_ra_rad = center_ra * M_PI / 180.0;
    double center_dec_rad = center_dec * M_PI / 180.0;
    for (int i = 0; i < n_cat; i++) {
        cat_ra_rad[i] = cat_ra[i] * M_PI / 180.0;
        cat_dec_rad[i] = cat_dec[i] * M_PI / 180.0;
    }
    
    std::vector<double> cat_xi(n_cat), cat_eta(n_cat);
    gnomonic_projection(cat_ra_rad.data(), cat_dec_rad.data(), n_cat, center_ra_rad, center_dec_rad, cat_xi.data(), cat_eta.data());
    
    double rad_to_px = 180.0 / M_PI * 3600.0 / scale_arcsec_px;
    std::vector<double> cat_x_px(n_cat), cat_y_px(n_cat);
    for (int i = 0; i < n_cat; i++) {
        cat_x_px[i] = cat_xi[i] * rad_to_px;
        cat_y_px[i] = -cat_eta[i] * rad_to_px;
    }
    
    PSLOG_I("\n=== Step 1: Coarse Matching ===");
    
    int n_cat_bright_list[] = {150, 300, 600, 1200, 2400};
    int n_cat_bright_count = 5;
    double rms_threshold = 5.0;
    
    PSMStarAlignmentResult sa_result;
    memset(&sa_result, 0, sizeof(sa_result));
    int best_matched = 0;
    double best_rms = 1e30;
    int best_n_cat_bright = 0;
    int prev_matched = 0;
    
    double t1_start = 0.0;
#ifdef _WIN32
    t1_start = (double)GetTickCount() / 1000.0;
#endif
    
    for (int retry_idx = 0; retry_idx < n_cat_bright_count; retry_idx++) {
        int n_cat_bright = n_cat_bright_list[retry_idx];
        if (n_cat_bright > n_cat) n_cat_bright = n_cat;
        
        if (n_saturated >= 10 && retry_idx > 0) {
            break;
        }
        
        PSLOG_I("  Attempt %d: n_cat_bright=%d", retry_idx + 1, n_cat_bright);
        
        PSMStarAlignmentInput sa_input;
        memset(&sa_input, 0, sizeof(sa_input));
        sa_input.img_x = img_x;
        sa_input.img_y = img_y;
        sa_input.img_count = img_count;
        sa_input.cat_x = cat_x_px.data();
        sa_input.cat_y = cat_y_px.data();
        sa_input.cat_mag = cat_mag.data();
        sa_input.cat_count = n_cat;
        sa_input.n_img_bright = config->n_img_bright > 0 ? config->n_img_bright : std::min(500, img_count);
        sa_input.n_cat_bright = n_cat_bright;
        sa_input.max_dist_px = config->max_match_dist_px > 0 ? config->max_match_dist_px : 25.0;
        sa_input.max_iterations = config->max_iterations > 0 ? config->max_iterations : 5;
        sa_input.match_threshold = config->match_threshold > 0 ? config->match_threshold : 10.0;
        sa_input.img_saturated = img_saturated;
        sa_input.n_saturated = n_saturated;
        
        PSMStarAlignmentResult try_result;
        memset(&try_result, 0, sizeof(try_result));
        
        int rc = psm_star_align(&sa_input, &try_result);
        
        if (rc == 0 && try_result.matched_count > 0 && try_result.rms_px < rms_threshold) {
            bool is_better = (try_result.matched_count > best_matched) ||
                             (try_result.matched_count == best_matched && try_result.rms_px < best_rms);
            
            if (is_better) {
                if (sa_result.matched_count > 0) {
                    psm_free_star_alignment_result(&sa_result);
                }
                sa_result = try_result;
                best_matched = try_result.matched_count;
                best_rms = try_result.rms_px;
                best_n_cat_bright = n_cat_bright;
            } else if (try_result.matched_count > 0) {
                psm_free_star_alignment_result(&try_result);
            }
            
            PSLOG_I("  matched=%d, RMS=%.3fpx", try_result.matched_count, try_result.rms_px);
            
            if (retry_idx > 0 && try_result.matched_count == prev_matched) {
                PSLOG_I("  Match count unchanged (%d), using best result to continue", best_matched);
                break;
            }
            
            prev_matched = try_result.matched_count;
        } else {
            if (try_result.matched_count > 0) {
                psm_free_star_alignment_result(&try_result);
            }
            
            if (retry_idx > 0 && best_matched > 0 && best_matched == prev_matched) {
                PSLOG_I("  No improvement, using best result (%d matches) to continue", best_matched);
                break;
            }
        }
    }
    
    if (best_matched == 0 || best_rms >= rms_threshold) {
        PSLOG_E("  Step 1 failed: no valid match (matched=%d, rms=%.3f)", best_matched, best_rms);
        return PSOLVE_ERR_NO_MATCH;
    }
    
    double t1_end = 0.0;
#ifdef _WIN32
    t1_end = (double)GetTickCount() / 1000.0;
#endif
    double step1_time = t1_end - t1_start;
    
    PSLOG_I("  Step 1 final: matched=%d, RMS=%.3fpx, n_cat_bright=%d, time=%.2fs", 
            sa_result.matched_count, sa_result.rms_px, best_n_cat_bright, step1_time);
    PSLOG_I("  Offset: (%.2f, %.2f)px, rotation=%.3f°, flip=%d",
            sa_result.offset_x, sa_result.offset_y, sa_result.rotation_deg, sa_result.flip_mode);
    
    double step1_ra = center_ra + sa_result.offset_x * scale_arcsec_px / 3600.0 / cos(center_dec_rad);
    double step1_dec = center_dec + sa_result.offset_y * scale_arcsec_px / 3600.0;
    double step1_rotation = sa_result.rotation_deg;
    int step1_flip = sa_result.flip_mode;
    int step1_matched = sa_result.matched_count;
    double step1_rms = sa_result.rms_px;
    
    double aff_a0 = sa_result.a0, aff_a1 = sa_result.a1, aff_a2 = sa_result.a2;
    double aff_b0 = sa_result.b0, aff_b1 = sa_result.b1, aff_b2 = sa_result.b2;
    
    PSLOG_I("  Step 1 affine: a0=%.2f a1=%.6f a2=%.6f b0=%.2f b1=%.6f b2=%.6f",
            aff_a0, aff_a1, aff_a2, aff_b0, aff_b1, aff_b2);
    
    FILE *step1_fp = fopen("F:\\Astro dev\\Astro CS Normalization Database\\debug_step1_predictions.csv", "w");
    if (step1_fp) {
        fprintf(step1_fp, "img_x,img_y,cat_x,cat_y,pred_x,pred_y,mag,flip_mode\n");
        int flip_x = (step1_flip == PSM_FLIP_X || step1_flip == PSM_FLIP_XY);
        int flip_y = (step1_flip == PSM_FLIP_Y || step1_flip == PSM_FLIP_XY);
        for (int i = 0; i < step1_matched; i++) {
            int img_idx = sa_result.img_indices[i];
            int cat_idx = sa_result.cat_indices[i];
            double img_x_val = img_x[img_idx];
            double img_y_val = img_y[img_idx];
            double cat_x_val = cat_x_px[cat_idx];
            double cat_y_val = cat_y_px[cat_idx];
            double mag_val = cat_mag[cat_idx];
            double flipped_x = flip_x ? -img_x_val : img_x_val;
            double flipped_y = flip_y ? -img_y_val : img_y_val;
            double pred_x = aff_a0 + aff_a1 * flipped_x + aff_a2 * flipped_y;
            double pred_y = aff_b0 + aff_b1 * flipped_x + aff_b2 * flipped_y;
            double img_x_abs = img_x_val + width / 2.0;
            double img_y_abs = height / 2.0 - img_y_val;
            double cat_x_abs = cat_x_val + width / 2.0;
            double cat_y_abs = height / 2.0 - cat_y_val;
            double pred_x_abs = pred_x + width / 2.0;
            double pred_y_abs = height / 2.0 - pred_y;
            fprintf(step1_fp, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                    img_x_abs, img_y_abs, cat_x_abs, cat_y_abs, pred_x_abs, pred_y_abs, mag_val, step1_flip);
        }
        fclose(step1_fp);
        PSLOG_I("  Debug: Step1 predictions saved to debug_step1_predictions.csv (%d matches)", step1_matched);
    }
    
    psm_free_star_alignment_result(&sa_result);
    
    if (config->sip_order == 0) {
        PSLOG_I("  Step 2 skipped (sip_order=0)");
        result->center_ra = step1_ra;
        result->center_dec = step1_dec;
        result->rotation_deg = step1_rotation;
        result->scale_arcsec_px = scale_arcsec_px;
        result->flip_mode = step1_flip;
        result->matched_count = step1_matched;
        result->rms_px = step1_rms;
        result->step1_time_sec = step1_time;
        result->step2_time_sec = 0.0;
        result->sip_valid = 0;
        
        result->wcs.crpix1 = width / 2.0;
        result->wcs.crpix2 = height / 2.0;
        result->wcs.crval1 = step1_ra;
        result->wcs.crval2 = step1_dec;
        double cd_scale = scale_arcsec_px / 3600.0;
        double cos_r = cos(step1_rotation * M_PI / 180.0);
        double sin_r = sin(step1_rotation * M_PI / 180.0);
        result->wcs.cd1_1 = -cd_scale * cos_r;
        result->wcs.cd1_2 = -cd_scale * sin_r;
        result->wcs.cd2_1 = cd_scale * sin_r;
        result->wcs.cd2_2 = -cd_scale * cos_r;
        strcpy(result->wcs.ctype1, "RA---TAN");
        strcpy(result->wcs.ctype2, "DEC--TAN");
        strcpy(result->wcs.radesys, "ICRS");
        result->wcs.equinox = 2000.0;
        
        h->last_result = *result;
        h->has_result = 1;
        h->last_wcs = result->wcs;
        h->has_wcs = 1;
        
        return PSOLVE_OK;
    }
    
    PSLOG_I("\n=== Step 2: Triangle Match ===");
    PSLOG_I("  Using step1 center: RA=%.6f Dec=%.6f", step1_ra, step1_dec);
    
    std::vector<double> img_mag(img_count);
    double max_flux = 0;
    for (int i = 0; i < img_count; i++) {
        if (img_flux[i] > max_flux) max_flux = img_flux[i];
    }
    for (int i = 0; i < img_count; i++) {
        img_mag[i] = -2.5 * log10(img_flux[i] / max_flux) + 0.0;
    }
    
    GMImageStars gm_img;
    memset(&gm_img, 0, sizeof(gm_img));
    gm_img.img_x = img_x;
    gm_img.img_y = img_y;
    gm_img.img_flux = img_flux;
    gm_img.img_mag = img_mag.data();
    gm_img.img_count = img_count;
    
    GMCatalogStars gm_cat;
    memset(&gm_cat, 0, sizeof(gm_cat));
    gm_cat.cat_ra = cat_ra.data();
    gm_cat.cat_dec = cat_dec.data();
    gm_cat.cat_mag = cat_mag.data();
    gm_cat.cat_count = n_cat;
    
    GMInitialTransform gm_init;
    memset(&gm_init, 0, sizeof(gm_init));
    gm_init.crval1 = center_ra;
    gm_init.crval2 = center_dec;
    gm_init.crpix1 = width / 2.0;
    gm_init.crpix2 = height / 2.0;
    {
        double cd_scale = scale_arcsec_px / 3600.0;
        double cos_r = cos(step1_rotation * M_PI / 180.0);
        double sin_r = sin(step1_rotation * M_PI / 180.0);
        gm_init.cd1_1 = cd_scale * cos_r;
        gm_init.cd1_2 = cd_scale * sin_r;
        gm_init.cd2_1 = -cd_scale * sin_r;
        gm_init.cd2_2 = cd_scale * cos_r;
        if (step1_flip == PSM_FLIP_X || step1_flip == PSM_FLIP_XY) {
            gm_init.cd1_1 = -gm_init.cd1_1;
            gm_init.cd1_2 = -gm_init.cd1_2;
        }
        if (step1_flip == PSM_FLIP_Y || step1_flip == PSM_FLIP_XY) {
            gm_init.cd2_1 = -gm_init.cd2_1;
            gm_init.cd2_2 = -gm_init.cd2_2;
        }
    }
    gm_init.scale_arcsec_px = scale_arcsec_px;
    gm_init.img_width = width;
    gm_init.img_height = height;
    
    GMConfig gm_config;
    memset(&gm_config, 0, sizeof(gm_config));
    gm_config.match_tolerance = 5.0;
    gm_config.max_ransac_iter = 500;
    gm_config.ransac_sigma = 3.0;
    gm_config.sip_order = config->sip_order > 0 ? config->sip_order : 5;
    gm_config.centroid_radius = 50.0;
    gm_config.ratio_threshold = 0.07;
    gm_config.n_img_bright = 1000;
    gm_config.n_cat_bright = 1500;
    
    GMResult gm_result;
    memset(&gm_result, 0, sizeof(gm_result));
    
    double t2_start = 0.0;
#ifdef _WIN32
    t2_start = (double)GetTickCount() / 1000.0;
#endif
    
    int rc2 = psm_grid_match_perform(&gm_img, &gm_cat, &gm_init, &gm_config, &gm_result);
    
    double t2_end = 0.0;
#ifdef _WIN32
    t2_end = (double)GetTickCount() / 1000.0;
#endif
    double step2_time = t2_end - t2_start;
    
    if (rc2 != 0) {
        PSLOG_E("  Step 2 failed: %d", rc2);
        result->center_ra = step1_ra;
        result->center_dec = step1_dec;
        result->rotation_deg = step1_rotation;
        result->scale_arcsec_px = scale_arcsec_px;
        result->flip_mode = step1_flip;
        result->matched_count = step1_matched;
        result->rms_px = step1_rms;
        result->step1_time_sec = step1_time;
        result->step2_time_sec = step2_time;
        return PSOLVE_OK;
    }
    
    PSLOG_I("  Step 2: control_points=%d, grids_matched=%d/%d, RMS=%.3fpx, time=%.2fs",
            gm_result.n_control_points, gm_result.n_grids_matched, gm_result.n_grids_total,
            gm_result.rms_total, step2_time);
    PSLOG_I("  RANSAC removed: %d points", gm_result.n_ransac_removed);
    
    result->center_ra = gm_result.crval[0];
    result->center_dec = gm_result.crval[1];
    result->rotation_deg = step1_rotation;
    result->scale_arcsec_px = scale_arcsec_px;
    result->flip_mode = step1_flip;
    result->matched_count = gm_result.n_control_points;
    result->rms_px = gm_result.rms_total;
    result->step1_time_sec = step1_time;
    result->step2_time_sec = step2_time;
    
    result->wcs.crpix1 = gm_result.crpix[0];
    result->wcs.crpix2 = gm_result.crpix[1];
    result->wcs.crval1 = gm_result.crval[0];
    result->wcs.crval2 = gm_result.crval[1];
    result->wcs.cd1_1 = gm_result.cd[0][0];
    result->wcs.cd1_2 = gm_result.cd[0][1];
    result->wcs.cd2_1 = gm_result.cd[1][0];
    result->wcs.cd2_2 = gm_result.cd[1][1];
    strcpy(result->wcs.ctype1, "RA---TAN-SIP");
    strcpy(result->wcs.ctype2, "DEC--TAN-SIP");
    strcpy(result->wcs.radesys, "ICRS");
    result->wcs.equinox = 2000.0;
    
    if (gm_result.sip_valid) {
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                result->sip.A[i][j] = gm_result.sip_A[i][j];
                result->sip.B[i][j] = gm_result.sip_B[i][j];
                result->sip.AP[i][j] = gm_result.sip_AP[i][j];
                result->sip.BP[i][j] = gm_result.sip_BP[i][j];
            }
        }
        result->sip.order = gm_result.sip_order;
        result->sip.valid = 1;
        result->sip_valid = 1;
    }
    
    FILE *debug_fp = fopen("F:\\Astro dev\\Astro CS Normalization Database\\debug_control_points.csv", "w");
    if (debug_fp) {
        fprintf(debug_fp, "img_x,img_y,cat_x,cat_y,residual_x,residual_y,valid\n");
        for (int i = 0; i < gm_result.n_control_points; i++) {
            GMControlPoint *cp = &gm_result.control_points[i];
            double img_x_abs = cp->img_x + width/2.0;
            double img_y_abs = -(cp->img_y - height/2.0);
            double cat_x_abs = cp->cat_x + width/2.0;
            double cat_y_abs = -(cp->cat_y - height/2.0);
            double res_x_abs = cat_x_abs - img_x_abs;
            double res_y_abs = cat_y_abs - img_y_abs;
            fprintf(debug_fp, "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d\n",
                    img_x_abs, img_y_abs, cat_x_abs, cat_y_abs,
                    res_x_abs, res_y_abs, cp->valid);
        }
        fclose(debug_fp);
        PSLOG_I("  Debug: control points saved to debug_control_points.csv");
    }
    
    psm_grid_match_free_result(&gm_result);
    
    h->last_result = *result;
    h->has_result = 1;
    h->last_wcs = result->wcs;
    h->has_wcs = 1;
    
    PSLOG_I("=== psolve_solve Complete ===");
    return PSOLVE_OK;
}

PSOLVE_EXPORT void psolve_free_result(PSolveResult *result)
{
    if (result) {
        memset(result, 0, sizeof(PSolveResult));
    }
}

PSOLVE_EXPORT int psolve_solve_with_image(
    PSolveHandle handle,
    const uint16_t *image, int width, int height,
    const PSolveImageData *img_data,
    const PSolveConfig *config,
    PSolveResult *result)
{
    if (!handle || !image || !img_data || !config || !result) {
        return PSOLVE_ERR_INTERNAL;
    }
    
    PSLOG_I("\n=== psolve_solve_with_image: Auto Star Detection ===");
    
    double t_detect_start = 0.0;
#ifdef _WIN32
    t_detect_start = (double)GetTickCount() / 1000.0;
#endif
    
    SDetParams det_params;
    memset(&det_params, 0, sizeof(det_params));
    det_params.structureLayers = 5;
    det_params.hotPixelFilterRadius = 1;
    det_params.iterativeClipSigma = 9.0f;
    det_params.iterativeMaxRounds = 5;
    det_params.medianFilterDetail = 1;
    det_params.maxStars = 0;
    det_params.fitRadius = 8;
    det_params.fwhmClipSigma = 3.0f;
    det_params.maxAxisRatio = 2.0f;
    
    StarDetectorHandle det_handle = sdet_create(&det_params);
    if (!det_handle) {
        PSLOG_E("  Failed to create star detector");
        return PSOLVE_ERR_INTERNAL;
    }
    
    double *det_x = nullptr, *det_y = nullptr;
    float *det_flux = nullptr;
    int *det_saturated = nullptr;
    int det_count = 0;
    
    int rc_det = sdet_detect_ex(det_handle, image, width, height,
                                 &det_x, &det_y, &det_flux, &det_saturated, &det_count);
    
    sdet_destroy(det_handle);
    
    double t_detect_end = 0.0;
#ifdef _WIN32
    t_detect_end = (double)GetTickCount() / 1000.0;
#endif
    double detect_time = t_detect_end - t_detect_start;
    
    if (rc_det != 0 || det_count <= 0) {
        PSLOG_E("  Star detection failed: rc=%d, count=%d", rc_det, det_count);
        if (det_x) sdet_free_detect_ex(det_x, det_y, det_flux, det_saturated);
        return PSOLVE_ERR_NO_MATCH;
    }
    
    int n_saturated = 0;
    for (int i = 0; i < det_count; i++) {
        if (det_saturated[i]) n_saturated++;
    }
    
    PSLOG_I("  Detected %d stars (saturated=%d, normal=%d) in %.2fs",
            det_count, n_saturated, det_count - n_saturated, detect_time);
    
    std::vector<double> img_x(det_count), img_y(det_count), img_flux(det_count);
    std::vector<int> img_sat(det_count);
    
    for (int i = 0; i < det_count; i++) {
        img_x[i] = det_x[i] - width / 2.0;
        img_y[i] = -(det_y[i] - height / 2.0);
        img_flux[i] = det_flux[i];
        img_sat[i] = det_saturated[i];
    }
    
    sdet_free_detect_ex(det_x, det_y, det_flux, det_saturated);
    
    PSolveConfig solve_config = *config;
    
    int rc = psolve_solve(handle,
                          img_x.data(), img_y.data(), img_flux.data(),
                          img_sat.data(), det_count, n_saturated,
                          img_data, &solve_config, result);
    
    return rc;
}

PSOLVE_EXPORT int psolve_coarse(
    PSolveHandle handle,
    const uint16_t *image, int width, int height,
    const PSolveImageData *img_data,
    const double *det_x, const double *det_y, int det_count,
    PSolveCoarseResult *result) {
    if (!handle || !img_data || !result) return PSOLVE_ERR_INTERNAL;
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;
    return psolve_coarse_solve(h, image, width, height, img_data, det_x, det_y, det_count, result);
}

PSOLVE_EXPORT int psolve_fine(
    PSolveHandle handle,
    const uint16_t *image, int width, int height,
    const PSolveCoarseResult *coarse,
    const PSolveImageData *img_data,
    PSolveFineResult *result) {
    if (!handle || !coarse || !img_data || !result) return PSOLVE_ERR_INTERNAL;
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;
    int rc = psolve_fine_solve(h, image, width, height, coarse, img_data, result);
    if (rc == PSOLVE_OK) {
        h->last_wcs = result->wcs;
        h->has_wcs = 1;
    }
    return rc;
}

PSOLVE_EXPORT void psolve_free_coarse_result(PSolveCoarseResult *result) {
    if (!result) return;
    free(result->matched_stars);
    result->matched_stars = nullptr;
    memset(result, 0, sizeof(PSolveCoarseResult));
}

PSOLVE_EXPORT void psolve_free_fine_result(PSolveFineResult *result) {
    if (!result) return;
    free(result->residual_grid_x);
    free(result->residual_grid_y);
    result->residual_grid_x = nullptr;
    result->residual_grid_y = nullptr;
    memset(result, 0, sizeof(PSolveFineResult));
}

PSOLVE_EXPORT int psolve_get_matched_stars(
    PSolveHandle handle,
    double **out_x, double **out_y, int *out_count) {
    if (!handle || !out_x || !out_y || !out_count) return PSOLVE_ERR_INTERNAL;
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;

    if (!h->has_coarse || h->last_coarse.matched_count <= 0) {
        *out_x = nullptr;
        *out_y = nullptr;
        *out_count = 0;
        return PSOLVE_ERR_NO_MATCH;
    }

    int count = h->last_coarse.matched_count;
    double *x = (double *)malloc(count * sizeof(double));
    double *y = (double *)malloc(count * sizeof(double));
    if (!x || !y) {
        free(x);
        free(y);
        return PSOLVE_ERR_INTERNAL;
    }

    for (int i = 0; i < count; i++) {
        x[i] = h->last_coarse.matched_stars[i].img_x;
        y[i] = h->last_coarse.matched_stars[i].img_y;
    }

    *out_x = x;
    *out_y = y;
    *out_count = count;
    return PSOLVE_OK;
}

PSOLVE_EXPORT int psolve_get_wcs(PSolveHandle handle, PSolveWCS *out_wcs) {
    if (!handle || !out_wcs) return PSOLVE_ERR_INTERNAL;
    PSolveHandleInternal *h = (PSolveHandleInternal *)handle;

    if (!h->has_wcs) return PSOLVE_ERR_NO_MATCH;

    memcpy(out_wcs, &h->last_wcs, sizeof(PSolveWCS));
    return PSOLVE_OK;
}

static double parse_hms_ra(const char *value) {
    if (!value || strlen(value) == 0) return -1.0;
    char buf[128];
    strncpy(buf, value, 127);
    buf[127] = '\0';
    
    char *parts[3] = {nullptr, nullptr, nullptr};
    char *p = buf;
    int idx = 0;
    while (*p && idx < 3) {
        while (*p == ' ' || *p == ':') p++;
        if (*p == '\0') break;
        parts[idx++] = p;
        while (*p && *p != ' ' && *p != ':') p++;
        if (*p) *p++ = '\0';
    }
    
    if (idx < 3) return -1.0;
    double h = atof(parts[0]);
    double m = atof(parts[1]);
    double s = atof(parts[2]);
    return (h + m/60.0 + s/3600.0) * 15.0;
}

static double parse_dms_dec(const char *value) {
    if (!value || strlen(value) == 0) return -999.0;
    char buf[128];
    strncpy(buf, value, 127);
    buf[127] = '\0';
    
    int sign = 1;
    char *start = buf;
    if (buf[0] == '-') { sign = -1; start = buf + 1; }
    else if (buf[0] == '+') { start = buf + 1; }
    
    char *parts[3] = {nullptr, nullptr, nullptr};
    char *p = start;
    int idx = 0;
    while (*p && idx < 3) {
        while (*p == ' ' || *p == ':') p++;
        if (*p == '\0') break;
        parts[idx++] = p;
        while (*p && *p != ' ' && *p != ':') p++;
        if (*p) *p++ = '\0';
    }
    
    if (idx < 3) return -999.0;
    double d = atof(parts[0]);
    double m = atof(parts[1]);
    double s = atof(parts[2]);
    return sign * (d + m/60.0 + s/3600.0);
}

PSOLVE_EXPORT int psolve_solve_with_file(
    PSolveHandle handle,
    const char *file_path,
    const PSolveConfig *config,
    PSolveResult *result) {
    
    if (!handle || !file_path || !config || !result) {
        return PSOLVE_ERR_INTERNAL;
    }
    
    PSLOG_I("\n=== psolve_solve_with_file: %s ===", file_path);
    
    AIOImageData *img_data = aio_read(file_path);
    if (!img_data) {
        PSLOG_E("  Failed to read image file");
        return PSOLVE_ERR_INTERNAL;
    }
    
    int width = aio_get_width(img_data);
    int height = aio_get_height(img_data);
    float *pixels = aio_get_pixel_data(img_data);
    
    if (!pixels || width <= 0 || height <= 0) {
        PSLOG_E("  Invalid image data");
        aio_free_image_data(img_data);
        return PSOLVE_ERR_INTERNAL;
    }
    
    PSLOG_I("  Image: %dx%d", width, height);
    
    AIOImageMetadata meta = aio_get_metadata(img_data);
    
    double center_ra = -1.0;
    double center_dec = -999.0;
    double focal_mm = 200.0;
    double pixel_um = 6.0;
    
    int kw_count = aio_get_keyword_count(img_data);
    for (int i = 0; i < kw_count; i++) {
        AIOFITSKeyword kw = aio_get_keyword(img_data, i);
        const char *name = kw.name;
        const char *val = kw.value;
        
        if (strcmp(name, "RA") == 0 || strcmp(name, "OBJCTRA") == 0) {
            double ra = parse_hms_ra(val);
            if (ra > 0) center_ra = ra;
        }
        else if (strcmp(name, "DEC") == 0 || strcmp(name, "OBJCTDEC") == 0) {
            double dec = parse_dms_dec(val);
            if (dec > -900) center_dec = dec;
        }
        else if (strcmp(name, "FOCALLEN") == 0) {
            focal_mm = atof(val);
        }
        else if (strcmp(name, "XPIXSZ") == 0) {
            pixel_um = atof(val);
        }
    }
    
    if (meta.observation.has_focallen) focal_mm = meta.observation.focallen;
    if (meta.observation.has_xpixsz) pixel_um = meta.observation.xpixsz;
    
    if (center_ra < 0 || center_dec < -900) {
        PSLOG_E("  No valid RA/DEC in header");
        aio_free_image_data(img_data);
        return PSOLVE_ERR_NO_MATCH;
    }
    
    PSLOG_I("  Header: RA=%.6f Dec=%.6f focal=%.1fmm pixel=%.2fum",
            center_ra, center_dec, focal_mm, pixel_um);
    
    std::vector<uint16_t> img_u16(width * height);
    for (int i = 0; i < width * height; i++) {
        img_u16[i] = (uint16_t)(pixels[i] + 32768.0);
    }
    
    aio_free_image_data(img_data);
    
    PSolveImageData psolve_img_data;
    memset(&psolve_img_data, 0, sizeof(psolve_img_data));
    psolve_img_data.focal_length_mm = focal_mm;
    psolve_img_data.pixel_size_um = pixel_um;
    psolve_img_data.center_ra = center_ra;
    psolve_img_data.center_dec = center_dec;
    psolve_img_data.width = width;
    psolve_img_data.height = height;
    psolve_img_data.has_coords = 1;
    
    int rc = psolve_solve_with_image(handle, img_u16.data(), width, height,
                                      &psolve_img_data, config, result);
    
    return rc;
}

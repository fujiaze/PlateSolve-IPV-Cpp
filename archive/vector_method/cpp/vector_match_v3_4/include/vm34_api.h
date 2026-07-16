#ifndef VM34_API_H
#define VM34_API_H

#ifdef _WIN32
#define VM34_API __declspec(dllexport)
#else
#define VM34_API __attribute__((visibility("default")))
#endif

struct VM34SolveParams {
    double s0;
    double s_min;
    double s_max;
    int    n_modes;
    int    seed;
    int    K_total;
    int    batch_size;
    int    min_samples;
    int    K_top;
    int    min_inliers;
    double fov_diag_asec;
    double img_width;
    double img_height;
    double center_ra;    // 图像中心赤经(度)
    double center_dec;   // 图像中心赤纬(度)
    const char* wcs_out_path;
};

struct VM34DebugInfo {
    double theta_snr;
    double theta_peak_deg;
    int    best_n_range;
    double median_noise;
    int    n_phaseb_pairs;
    int    n_phaseb_corr;
    int    n_phasea_records;
    int    n_phasec_expanded;
    int    n_phased_clean;
    int    n_phased_iterations;
    double mad_rms_arcsec;
};

struct VM34SolveResult {
    double s;
    double theta;
    double tx;
    double ty;
    int    n_inliers;
    double rms;
    int    best_mode;
    double norm_score;
    int*   inlier_mask;
    int    success;
    double peak_snr;
    int    n_samples;
    VM34DebugInfo debug;

    double sip_A[36];
    double sip_B[36];
    double cd[4];
    double crval[2];
    double crpix[2];
};

#ifdef __cplusplus
extern "C" {
#endif

VM34_API int vm34_solve(
    const double* U,
    int N_img,
    const double* W,
    int M,
    const VM34SolveParams* params,
    VM34SolveResult* result
);

VM34_API int vm34_count_inliers(
    const double* U, int N_img, const double* W, int M,
    double s, double theta, double tx, double ty,
    double s0, int* inlier_mask, double* out_rms
);

VM34_API int vm34_write_wcs_file(
    const VM34SolveResult* result,
    const char* path
);

#ifdef __cplusplus
}
#endif

#endif

#ifndef VM33_API_H
#define VM33_API_H

#ifdef _WIN32
#define VM33_API __declspec(dllexport)
#else
#define VM33_API __attribute__((visibility("default")))
#endif

struct VM33SolveParams {
    double tau;
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
};

struct VM33DebugInfo {
    double theta_snr;
    double theta_peak_deg;
    int    best_n_range;
    double median_noise;
    int    n_phaseb_pairs;
    int    n_phaseb_corr;
    int    n_phasea_records;
};

struct VM33SolveResult {
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
    VM33DebugInfo debug;
};

#ifdef __cplusplus
extern "C" {
#endif

VM33_API int vm33_solve(
    const double* U,
    int N_img,
    const double* W,
    int M,
    const VM33SolveParams* params,
    VM33SolveResult* result
);

VM33_API int vm33_svd_refine(
    const double* U, int N_img,
    const double* W, int M,
    const int* inlier_mask,
    double s_init, double theta_init, double tx_init, double ty_init,
    double s0,
    int max_iter,
    VM33SolveResult* result
);

VM33_API int vm33_count_inliers(
    const double* U, int N_img,
    const double* W, int M,
    double s, double theta, double tx, double ty,
    double tau,
    int* inlier_mask,
    double* out_rms
);

#ifdef __cplusplus
}
#endif

#endif

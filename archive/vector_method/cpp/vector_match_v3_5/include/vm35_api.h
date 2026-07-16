#ifndef VM35_API_H
#define VM35_API_H

#ifdef _WIN32
#define VM35_API __declspec(dllexport)
#else
#define VM35_API __attribute__((visibility("default")))
#endif

struct VM35SolveParams {
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

    // V3.5新增参数（snr_stop已移除，回退5N/10N停止）
    int    skip_sip;           // 跳过SIP拟合(Phase C/D/D'/E)，默认0
    int    expand_n_gaia;    // 扩增Gaia亮星数，默认1500
    int    expand_n_img;     // 扩增图像星点数，默认1000
    int    radial_n_bins;    // 径向幅度过滤分bin数，默认20
    int    radial_fit_order; // 径向幅度过滤拟合多项式阶数，默认3
    int    radial_n_iters;   // 径向幅度过滤迭代次数，默认3
};

struct VM35DebugInfo {
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

    // V3.5新增调试字段
    int    n_expand_mutual;     // 双向NN匹配对数
    int    n_expand_after_filter; // 径向过滤后对数
    int    n_sip_total;         // SIP拟合总对数
    int    sip_order;           // 实际SIP阶数
};

struct VM35SolveResult {
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
    VM35DebugInfo debug;

    double sip_A[36];
    double sip_B[36];
    double cd[4];
    double crval[2];
    double crpix[2];
};

#ifdef __cplusplus
extern "C" {
#endif

VM35_API int vm35_solve(
    const double* U,
    int N_img,
    const double* W,
    int M,
    const VM35SolveParams* params,
    VM35SolveResult* result
);

VM35_API int vm35_count_inliers(
    const double* U, int N_img, const double* W, int M,
    double s, double theta, double tx, double ty,
    double s0, int* inlier_mask, double* out_rms
);

VM35_API int vm35_write_wcs_file(
    const VM35SolveResult* result,
    const char* path
);

#ifdef __cplusplus
}
#endif

#endif

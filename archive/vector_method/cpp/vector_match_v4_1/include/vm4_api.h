#ifndef VM4_1_API_H
#define VM4_1_API_H

// ============================================================================
// vm4_api.h - V4.0 抽样投票向量法 C 接口
//
// 在 V3.5 抽样投票核心架构基础上新增 5 大优化：
//   Phase 0 : 密度匹配迭代星等查询（Task 2）
//   Phase C : k-vector 快速角距索引（Task 3）
//   Phase A : PROSAC 优先采样（Task 4）
//   Phase D': 贝叶斯假设验证（Task 5）
//   Phase D': 三角形双特征二级验证（Task 6）
//
// 本头文件保持 V3.5 继承字段原样，新增 V4.0 字段用于后续 Task 2-6 增强。
// ============================================================================

#ifdef _WIN32
#define VM4_1_API __declspec(dllexport)
#else
#define VM4_1_API __attribute__((visibility("default")))
#endif

struct VM4_1SolveParams {
    // === V3.5 继承字段（保留原样）===
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
    int    skip_sip;           // 跳过SIP拟合(Phase C/D/D'/E)，默认0
    int    expand_n_gaia;      // 扩增Gaia亮星数，默认1500
    int    expand_n_img;       // 扩增图像星点数，默认1000
    int    radial_n_bins;      // 径向幅度过滤分bin数，默认20
    int    radial_fit_order;   // 径向幅度过滤拟合多项式阶数，默认3
    int    radial_n_iters;     // 径向幅度过滤迭代次数，默认3

    // === V4.0 新增字段 ===
    // Phase 0: 密度匹配查询参数（Task 2）
    double k_match;           // 星表密度匹配系数，默认1.5
    double query_radius_factor; // 查询半径因子，默认1.0
    double m_lim_step;        // 极限星等迭代步长，默认0.5
    int    m_lim_max_iter;    // 极限星等迭代最大次数，默认15
    double density_tolerance; // 密度匹配容差，默认0.1
    int    n_img_bright;      // 图像亮星数（用于密度计算）
    double focal_length_mm;   // 焦距(mm)
    double pixel_size_um;     // 像元尺寸(um)
    double exposure_time_s;   // 曝光时间(s)

    // === V4.1 新增字段：不对称选星 ===
    int    img_n_target;              // 图像侧目标星数(默认50): 饱和>此值→全选饱和; 否则饱和+非饱和补足到此值
    double gaia_density_ratio;        // Gaia面密度/图像面密度(默认1.5)
    double gaia_query_radius_factor;  // Gaia查询半径因子(默认0.55, 即直径=1.1×FOV对角线)

    // Phase C: k-vector 参数（Task 3）
    double k_vector_eps;      // k-vector 角距查询容差(角秒)，默认2.0
    int    use_kvector;       // 是否启用 k-vector 预筛选，默认1

    // Phase A: PROSAC 参数（Task 4）
    double w_snr;             // SNR 权重，默认0.4
    double w_sparse;          // 稀疏度权重，默认0.4
    double w_sat;             // 饱和度权重，默认0.2
    int    prosac_T_max;      // PROSAC 最大抽样次数，默认10000
    int    use_prosac;        // 是否启用 PROSAC，默认1

    // Phase D': 贝叶斯验证参数（Task 5）
    double lnK_accept;        // 贝叶斯因子接受阈值，默认20.7
    double lnK_weak;          // 弱证据阈值，默认6.9
    int    use_bayes;         // 是否启用贝叶斯验证，默认1

    // Phase D': 三角形验证参数（Task 6）
    double eps_A;             // 三角形面积相对误差阈值，默认0.05
    double eps_J;             // 极惯性矩相对误差阈值，默认0.10
    double triangle_pass_rate; // 通过率阈值，默认0.8
    int    use_triangle;      // 是否启用三角形验证，默认1

    // === Task 7 集成新增可选输入字段（PROSAC 质量分用）===
    // 注: NULL 时退化为纯 sparsity 抽样，保持向后兼容
    const double* snr_values;          // 图像星 SNR 数组（长度=N_img，可选）
    const int*    is_saturated_values; // 图像星饱和标志数组（长度=N_img，1=饱和，0=正常，可选）
    const char*   log_file_path;       // 日志文件路径（UTF-8，可选；NULL 时不写文件仅 stderr）
};

struct VM4_1DebugInfo {
    // === V3.5 继承字段 ===
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
    int    n_expand_mutual;     // 双向NN匹配对数
    int    n_expand_after_filter; // 径向过滤后对数
    int    n_sip_total;         // SIP拟合总对数
    int    sip_order;           // 实际SIP阶数

    // === V4.0 新增字段 ===
    // Phase 0
    double rho_img;           // 图像亮星密度
    double rho_target;        // 目标星表密度
    double m_lim_final;       // 最终极限星等
    int    n_gaia_final;      // 最终 Gaia 星数
    int    m_lim_iterations;  // 星等迭代次数

    // Phase C
    double kvector_build_ms;  // k-vector 建索引耗时
    int    kvector_queries;   // k-vector 查询次数
    double kvector_avg_candidates; // 平均候选数

    // Phase A
    double prosac_quality_median; // 质量分中位数
    int    prosac_pool_final;     // 最终采样池大小

    // Phase D'
    double bayes_lnK;         // 贝叶斯因子对数
    int    bayes_n_match;     // 匹配对数
    int    bayes_decision;    // 决策(1接受/0弱证据/-1拒绝)
    int    triangle_total;    // 三角形总数
    double triangle_pass_ratio; // 通过率
};

struct VM4_1SolveResult {
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
    VM4_1DebugInfo debug;

    double sip_A[36];
    double sip_B[36];
    double cd[4];
    double crval[2];
    double crpix[2];
};

#ifdef __cplusplus
extern "C" {
#endif

VM4_1_API int vm4_1_solve(
    const double* U,
    int N_img,
    const double* W,
    int M,
    const VM4_1SolveParams* params,
    VM4_1SolveResult* result
);

VM4_1_API int vm4_1_count_inliers(
    const double* U, int N_img, const double* W, int M,
    double s, double theta, double tx, double ty,
    double s0, int* inlier_mask, double* out_rms
);

VM4_1_API int vm4_1_write_wcs_file(
    const VM4_1SolveResult* result,
    const char* path
);

#ifdef __cplusplus
}
#endif

#endif

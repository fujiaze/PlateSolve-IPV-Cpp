#ifndef VM32_API_H
#define VM32_API_H

#ifdef _WIN32
#define VM32_API __declspec(dllexport)
#else
#define VM32_API __attribute__((visibility("default")))
#endif

// V3.2 求解参数
struct VM32SolveParams {
    double tau;              // 内点阈值(角秒)
    double s0;               // 像素尺度(角秒/像素)
    double s_min;            // 比例尺下限(默认0.90)
    double s_max;            // 比例尺上限(默认1.10)
    int    n_modes;          // 翻转模式数(4)
    int    seed;              // 随机种子
    int    n_max;             // 最大总抽样次数(默认100000)
    int    batch_size;        // 每批抽样次数(默认1000)
    double snr_theta_tighten; // θ收紧的SNR阈值(默认3.0)
    double snr_s_tighten;     // s收紧的SNR阈值(默认8.0)
    double snr_converge;      // 收敛SNR阈值(默认20.0)
    double theta_band_init;   // θ初始搜索半带宽(度, 默认5.0)
    double s_band_init;       // s初始搜索半带宽(默认0.10)
    double theta_band_min;    // θ最小搜索半带宽(度, 默认0.1)
    double s_band_min;        // s最小搜索半带宽(默认0.002)
    int    min_inliers;       // 最少内点数
    double fov_diag_asec;     // FOV对角线(角秒)
};

// V3.2 求解结果
struct VM32SolveResult {
    double s;            // 缩放因子
    double theta;        // 旋转角(弧度)
    double tx;           // 平移x(角秒)
    double ty;           // 平移y(角秒)
    int    n_inliers;    // 内点数
    double rms;         // RMS(角秒)
    int    best_mode;   // 最佳翻转模式
    double norm_score;   // 归一化得分
    int*   inlier_mask; // 内点掩码(调用方分配, 长度=N_img)
    int    success;      // 是否成功(0=失败, 1=成功)
    double peak_snr;     // 峰值信噪比
    int    n_samples;    // 实际抽样次数
};

#ifdef __cplusplus
extern "C" {
#endif

// 核心求解函数
// 输入: U(N_img×2), W(M×2), params
// 输出: result
VM32_API int vm32_solve(
    const double* U,       // 图像向量组 (N_img, 2)
    int N_img,             // U的行数
    const double* W,       // 星表向量组 (M, 2)
    int M,                 // W的行数
    const VM32SolveParams* params,
    VM32SolveResult* result
);

// SVD精修函数
VM32_API int vm32_svd_refine(
    const double* U, int N_img,
    const double* W, int M,
    const int* inlier_mask,
    double s_init, double theta_init, double tx_init, double ty_init,
    double s0,
    int max_iter,
    VM32SolveResult* result
);

// 内点统计
VM32_API int vm32_count_inliers(
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

#endif // VM32_API_H

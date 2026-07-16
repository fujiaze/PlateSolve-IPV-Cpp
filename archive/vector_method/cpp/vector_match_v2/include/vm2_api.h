#ifndef VM2_API_H
#define VM2_API_H

#ifdef _WIN32
#define VM2_API __declspec(dllexport)
#else
#define VM2_API __attribute__((visibility("default")))
#endif

// 求解参数
struct VM2SolveParams {
    double tau_coarse;        // 粗匹配内点阈值(角秒)
    int    K;                 // RANSAC最大迭代次数
    int    min_inliers;       // 最少内点数
    double candidate_radius;  // 粗候选搜索半径(角秒)
    double s0;                // 像素尺度(角秒/像素)
    double fov_diag_asec;     // FOV对角线(角秒)
    int    n_modes;           // 翻转模式数(4)
    int    seed;              // 随机种子
};

// 求解结果
struct VM2SolveResult {
    double s;           // 缩放因子
    double theta;       // 旋转角(弧度)
    double tx;          // 平移x(角秒)
    double ty;          // 平移y(角秒)
    int    n_inliers;   // 内点数
    double rms;         // RMS(角秒)
    int    best_mode;   // 最佳翻转模式
    double norm_score;  // 归一化得分
    int*   inlier_mask; // 内点掩码(调用方分配, 长度=N_img)
    int    success;     // 是否成功(0=失败, 1=成功)
};

#ifdef __cplusplus
extern "C" {
#endif

// 核心求解函数
// 输入: U(N_img×2), W(M×2), sparsity(N_img), params
// 输出: result
VM2_API int vm2_solve(
    const double* U,       // 图像向量组 (N_img, 2)
    int N_img,             // U的行数
    const double* W,       // 星表向量组 (M, 2)
    int M,                 // W的行数
    const double* sparsity,// 稀疏度权重 (N_img,)
    const VM2SolveParams* params,
    VM2SolveResult* result
);

// SVD精修函数
VM2_API int vm2_svd_refine(
    const double* U, int N_img,
    const double* W, int M,
    const int* inlier_mask,  // 输入内点掩码
    double s_init, double theta_init, double tx_init, double ty_init,
    double s0,
    int max_iter,
    VM2SolveResult* result
);

// 1对1内点统计
VM2_API int vm2_count_inliers(
    const double* U, int N_img,
    const double* W, int M,
    double s, double theta, double tx, double ty,
    double tau,
    int* inlier_mask,    // 输出内点掩码(调用方分配, 长度=N_img)
    double* out_rms      // 输出RMS
);

#ifdef __cplusplus
}
#endif

#endif // VM2_API_H

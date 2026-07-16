#ifndef VM_API_H
#define VM_API_H

// ============================================================================
// vm_api.h - V4.2 VectorMatcher 模块 C 接口（Task 3）
//
// 职责：
//   Phase A: PROSAC 优先采样 + 1 点抽样 + θ 直方图投票 + 5N/10N 停止条件
//   Phase B: 三级过滤（n_in_range / θ 一致性 / 距离）→ 1对1互斥 → Umeyama SVD → 迭代精修
//   4 模式并行（OpenMP），选择最优模式（n_inliers×(1-rms/τ) 最高）
//
// 输入: U(N×2, 图像侧角秒坐标), W(M×2, Gaia 侧角秒坐标)
// 输出: 相似变换 (s, θ, tx, ty) + 匹配对 (cu[], cw[])
//
// 依赖: Eigen3, OpenMP
// ============================================================================

#include "v42_types.h"

#ifdef _WIN32
#define VM_API __declspec(dllexport)
#else
#define VM_API __attribute__((visibility("default")))
#endif

struct VectorMatcherParams {
    // 像素尺度与尺度范围
    double s0;          // 像素尺度(角秒/像素)
    double s_min;       // 尺度下限
    double s_max;       // 尺度上限
    int    n_modes;     // 模式数(默认4: 0=无翻转, 1=X翻转, 2=Y翻转, 3=XY翻转)
    int    seed;        // 随机种子

    // Phase A 抽样
    int    K_total;     // 总抽样次数(默认10000)
    int    batch_size;  // 批大小(默认500)
    int    min_samples; // 最小样本数(默认50)
    int    K_top;       // Top-K(默认100)

    // Phase B 过滤
    int    min_inliers; // 最小内点数(默认5)

    // PROSAC 参数
    double w_snr;       // SNR权重(默认0.4)
    double w_sparse;    // 稀疏度权重(默认0.4)
    double w_sat;       // 饱和度权重(默认0.2)
    int    prosac_T_max;// PROSAC最大抽样次数(默认10000)
    int    use_prosac;  // 是否启用PROSAC(默认1)

    // 日志
    const char* log_file_path;        // 日志文件路径(UTF-8, NULL=仅 stderr)

    // 可选输入: PROSAC 质量分用(NULL 时退化为 sparsity 代理)
    const double* snr_values;          // 图像星 SNR 数组(长度=N_img)
    const int*    is_saturated_values; // 图像星饱和标志数组(长度=N_img, 1=饱和)
};

struct VectorMatchResult {
    // 初始变换(Phase B 输出)
    double s;
    double theta;      // 弧度
    double tx;
    double ty;
    double rms;        // RMS(角秒)

    // 匹配对(粗匹配, Phase B 输出)
    int*   cu;         // U索引数组
    int*   cw;         // W索引数组
    int    n_pairs;    // 匹配对数

    // 调试信息
    double theta_snr;
    double theta_peak_deg;
    int    best_n_range;
    int    n_phasea_records;
    double prosac_quality_median;
    int    prosac_pool_final;
    int    best_mode;
    int    success;
};

#ifdef __cplusplus
extern "C" {
#endif

// 执行向量匹配 (Phase A + Phase B, 4 模式并行)
// 返回: 0=成功, -1=失败
VM_API int vm_match(
    const double* U, int N_img,
    const double* W, int M,
    const VectorMatcherParams* params,
    VectorMatchResult* result
);

// 释放 result 中的 cu/cw 数组
VM_API void vm_free_result(VectorMatchResult* result);

#ifdef __cplusplus
}
#endif

#endif // VM_API_H

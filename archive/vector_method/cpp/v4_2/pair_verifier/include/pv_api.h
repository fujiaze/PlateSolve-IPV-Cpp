#ifndef PV_API_H
#define PV_API_H

// ============================================================================
// pv_api.h - V4.2 PairVerifier 模块公共 C 接口（Task 5）
//
// 职责：
//   Phase D  - 3 轮 MAD 迭代清洗（阈值 max(5", factor×1.4826×MAD)）
//   Phase D' - 贝叶斯假设验证 + 三角形双特征验证
//
// 从 V4.1 vector_match_v4_1 抽取相关逻辑，作为独立 DLL 模块。
// 约束：C++17，单线程；中文注释，UTF-8 编码
// ============================================================================

#include "v42_types.h"

#ifdef _WIN32
#define PV_API __declspec(dllexport)
#else
#define PV_API __attribute__((visibility("default")))
#endif

// --- 验证器参数 ---
struct PairVerifierParams {
    // MAD 清洗参数
    int    mad_iters;                // 迭代次数(默认3)
    double mad_threshold_factor;     // 阈值因子(默认3.0, 阈值=max(5", factor×1.4826×MAD))
    double mad_min_threshold_arcsec; // 最小阈值(默认5.0")

    // 贝叶斯验证参数
    double lnK_accept;               // 接受阈值(默认20.7)
    double lnK_weak;                 // 弱证据阈值(默认6.9)
    double sigma_min;                // σ下限(默认0.5")

    // 三角形验证参数
    double eps_A;                    // 面积相对误差阈值(默认0.05)
    double eps_J;                    // 极惯性矩相对误差阈值(默认0.10)
    double triangle_pass_rate;       // 通过率阈值(默认0.8)

    // FOV 参数(贝叶斯 A_fov 计算)
    double fov_diag_deg;             // FOV对角线(度)

    const char* log_file_path;       // 日志文件路径(NULL=仅stderr)
};

// --- 验证结果 ---
struct VerificationResult {
    // MAD 清洗后
    int*   clean_u;              // 清洗后U索引(堆分配, 需调用 pv_free 释放)
    int*   clean_w;              // 清洗后W索引(堆分配, 需调用 pv_free 释放)
    int    n_clean;              // 清洗后对数
    int    n_removed;            // 剔除数
    int    mad_iterations;       // 实际迭代次数
    double mad_rms_arcsec;       // MAD后RMS(角秒)

    // 贝叶斯
    double bayes_lnK;            // 贝叶斯因子对数
    int    bayes_n_match;        // 匹配对数
    int    bayes_decision;       // 1=接受, 0=弱证据, -1=拒绝

    // 三角形
    int    triangle_total;       // 三角形总数(剔除退化)
    int    triangle_passed;      // 通过数
    double triangle_pass_ratio;  // 通过率

    // 综合验证
    int    validated;            // 1=通过(bayes_decision≥0 && triangle_pass), 0=未通过
    int    success;              // 1=执行成功, 0=失败
};

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// pv_verify - 执行 Phase D(MAD清洗) + Phase D'(贝叶斯+三角形验证)
//
// 输入:
//   U       : 图像侧星点坐标数组 [N_img×2], 角秒
//   N_img   : 图像侧星点数
//   W       : Gaia侧星点坐标数组 [M×2], 角秒
//   M       : Gaia侧星点数
//   pairs_u : 匹配对的U索引数组 [n_pairs]
//   pairs_w : 匹配对的W索引数组 [n_pairs]
//   n_pairs : 匹配对数
//   s0      : 像素尺度(角秒/像素), 用于参考
//   params  : 验证器参数
//   result  : 验证结果输出(调用方分配)
//
// 返回: 1=成功, 0=失败
// ============================================================================
PV_API int pv_verify(
    const double* U, int N_img,
    const double* W, int M,
    const int* pairs_u, const int* pairs_w, int n_pairs,
    double s0,
    const PairVerifierParams* params,
    VerificationResult* result
);

// ============================================================================
// pv_free - 释放 VerificationResult 中的堆内存(clean_u, clean_w)
// ============================================================================
PV_API void pv_free(VerificationResult* result);

#ifdef __cplusplus
}
#endif

#endif // PV_API_H

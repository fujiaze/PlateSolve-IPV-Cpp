#ifndef PE_API_H
#define PE_API_H

#include "v42_types.h"

#ifdef _WIN32
#define PE_API __declspec(dllexport)
#else
#define PE_API __attribute__((visibility("default")))
#endif

// PairExpander 参数 (V4.2 Phase C: 线性扫描 NN + 区域均匀化)
struct PairExpanderParams {
    double s0;                  // 像素尺度(角秒/像素)
    double tau_factor;          // 距离阈值因子(默认3.0, τ=tau_factor×s0)
    double scale_ratio_tol;     // 模长比容差(默认0.1, |‖U‖/‖W‖-1|<tol)
    int    region_size_px;      // 区域尺寸(像素, 默认800)
    int    N_floor;             // 每区保底对数(默认5)
    int    N_cap;               // 每区上限对数(默认30)
    int    N_max;               // 全局上限(默认1500)
    double img_width;           // 图像宽度(像素)
    double img_height;          // 图像高度(像素)
    const char* log_file_path;  // 日志路径(可选)
};

// 扩增结果 (返回扩充后 U/W 索引数组, 需调用 pe_free 释放)
struct ExpansionResult {
    int*   expand_u;        // 扩充后U索引数组
    int*   expand_w;        // 扩充后W索引数组
    int    n_pairs;         // 总对数(Phase B + 扩充)
    int    n_expanded;      // 扩充对数(不含Phase B)
    int    n_regions;       // 区域数
    int    n_sparse_regions;// 稀疏区数
    int    n_candidates;    // 候选对数(τ截断后)
    int    n_accepted;      // 接受对数(模长比过滤后)
    double expand_time_ms;  // 扩增耗时
    int    success;
};

#ifdef __cplusplus
extern "C" {
#endif

// 执行匹配对扩增
// 输入:
//   U[N_img×2], W[M×2]: 角秒坐标数组
//   init_cu[n_init×1], init_cw[n_init×1]: Phase B 初始匹配对
//   s, theta, tx, ty: 初始变换参数(来自 VectorMatcher)
// 输出: ExpansionResult (含扩充后匹配对, 需调用 pe_free 释放)
PE_API int pe_expand(
    const double* U, int N_img,
    const double* W, int M,
    const int* init_cu, const int* init_cw, int n_init,
    double s, double theta, double tx, double ty,
    const PairExpanderParams* params,
    ExpansionResult* result
);

PE_API void pe_free(ExpansionResult* result);

#ifdef __cplusplus
}
#endif

#endif // PE_API_H

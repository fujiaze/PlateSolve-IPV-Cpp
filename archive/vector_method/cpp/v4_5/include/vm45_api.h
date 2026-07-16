#ifndef VM45_API_H
#define VM45_API_H

// ============================================================================
// vm45_api.h - V4.5 单一 C 接口 (仅 Phase A θ 求解)
//
// V4.5 严格按 v4_4_relvec_sampling_design.md 实现:
//   - Phase 0 (StarSelector): 复用 V4.4 算法
//   - Phase A (相对向量法): 绝对距离 k-vector + 1D θ 直方图 + 高斯平滑
//   - 跳过 Phase B (tx/ty 搜索), IRM 闭环, WcsFitter
//
// 编译为单一 vector_match_v4_5.dll, 与 vector_match_v4_4.dll 并存
// 对外暴露一个高级 C 接口 vm45_solve(), 内部串联:
//   StarSelector → 相对向量法 Phase A → (输出 θ + SNR + 直方图)
//
// Python 端只需 1 次 ctypes 调用, 无 JSON 序列化, 无数组边界拷贝
// ============================================================================

#include "vm45_types.h"

#ifdef _WIN32
#define VM45_API __declspec(dllexport)
#else
#define VM45_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// 主求解接口
// ===========================================================================

// 一键求解 θ: 输入 FITS 路径 + 中心指向 + 焦距/像元, 输出 θ + SNR + 直方图
// 内部串联: StarSelector → 相对向量法 Phase A
//
// 参数:
//   image_path     - FITS 图像路径 (UTF-8)
//   ra, dec        - 图像中心赤经赤纬 (度)
//   focal_length_mm- 焦距 (mm)
//   pixel_size_um  - 像元尺寸 (um)
//   params         - 求解参数 (NULL 用默认值)
//   result         - 输出结果 (调用者分配, theta_histogram 由 vm45_free_result 释放)
//
// 返回: 0=成功, -1=失败 (错误信息在 result->error_msg)
VM45_API int vm45_solve(
    const char* image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const v45::VM45SolveParams* params,
    v45::VM45SolveResult* result
);

// ===========================================================================
// 依赖注入接口
// ===========================================================================

// 注入 Gaia 客户端句柄 (复用已有 gaia_client.dll)
// gaia_client_handle: GaiaClientPy 内部句柄 (void*)
// 注意: 调用者负责管理 GaiaClientPy 生命周期, V4.5 DLL 不负责释放
VM45_API void vm45_set_gaia_client(void* gaia_client_handle);

// 注入 StarDetector 句柄 (复用已有 star_detector.dll)
// detector_handle: StarDetector 内部句柄 (void*)
// 注意: 调用者负责管理 StarDetector 生命周期
VM45_API void vm45_set_star_detector(void* detector_handle);

// ===========================================================================
// 资源释放
// ===========================================================================

// 释放 VM45SolveResult 内部分配的数组 (theta_histogram)
// 调用后 result 内指针置 NULL, 可安全重复调用
VM45_API void vm45_free_result(v45::VM45SolveResult* result);

// ===========================================================================
// 默认参数获取
// ===========================================================================

// 获取默认求解参数 (Python 端可调用以获取基础参数, 再覆盖特定字段)
VM45_API void vm45_get_default_params(v45::VM45SolveParams* params);

#ifdef __cplusplus
}
#endif

#endif // VM45_API_H

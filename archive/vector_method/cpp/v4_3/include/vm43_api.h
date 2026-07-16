#ifndef VM43_API_H
#define VM43_API_H

// ============================================================================
// vm43_api.h - V4.3 单一 C 接口
//
// V4.3 将 V4.2 的 5 个独立 DLL 合并为单一 vector_match_v4_3.dll
// 对外暴露一个高级 C 接口 vm43_solve(), 内部串联所有阶段:
//   StarSelector → VectorMatcher → IRM 闭环 (Expand ↔ Verify ↔ Fit)
//
// Python 端只需 1 次 ctypes 调用, 无 JSON 序列化, 无数组边界拷贝
// ============================================================================

#include "vm43_types.h"

#ifdef _WIN32
#define VM43_API __declspec(dllexport)
#else
#define VM43_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ===========================================================================
// 主求解接口
// ===========================================================================

// 一键求解: 输入 FITS 路径 + 中心指向 + 焦距/像元, 输出完整 WCS
// 内部串联: StarSelector → VectorMatcher → IRM 闭环 → WcsFitter
//
// 参数:
//   image_path     - FITS 图像路径 (UTF-8)
//   ra, dec        - 图像中心赤经赤纬 (度)
//   focal_length_mm- 焦距 (mm)
//   pixel_size_um  - 像元尺寸 (um)
//   params         - 求解参数 (NULL 用默认值)
//   result         - 输出结果 (调用者分配, 内部 cu/cw 由 vm43_free_result 释放)
//
// 返回: 0=成功, -1=失败 (错误信息在 result->error_msg)
VM43_API int vm43_solve(
    const char* image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const v43::VM43SolveParams* params,
    v43::VM43SolveResult* result
);

// ===========================================================================
// 依赖注入接口
// ===========================================================================

// 注入 Gaia 客户端句柄 (复用已有 gaia_client.dll)
// gaia_client_handle: GaiaClientPy 内部句柄 (void*)
// 注意: 调用者负责管理 GaiaClientPy 生命周期, V4.3 DLL 不负责释放
VM43_API void vm43_set_gaia_client(void* gaia_client_handle);

// 注入 StarDetector 句柄 (复用已有 star_detector.dll)
// detector_handle: StarDetector 内部句柄 (void*)
// 注意: 调用者负责管理 StarDetector 生命周期
VM43_API void vm43_set_star_detector(void* detector_handle);

// ===========================================================================
// 资源释放
// ===========================================================================

// 释放 VM43SolveResult 内部分配的数组 (cu/cw)
// 调用后 result 内指针置 NULL, 可安全重复调用
VM43_API void vm43_free_result(v43::VM43SolveResult* result);

// ===========================================================================
// 默认参数获取
// ===========================================================================

// 获取默认求解参数 (Python 端可调用以获取基础参数, 再覆盖特定字段)
VM43_API void vm43_get_default_params(v43::VM43SolveParams* params);

#ifdef __cplusplus
}
#endif

#endif // VM43_API_H

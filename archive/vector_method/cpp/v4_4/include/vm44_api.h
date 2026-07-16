#ifndef VM44_API_H
#define VM44_API_H

// ============================================================================
// vm44_api.h - V4.4 单一 C 接口
//
// V4.4: 相对向量法完全替代单θ Phase A, 其余与 V4.3 一致
// 编译为单一 vector_match_v4_4.dll
// 对外暴露一个高级 C 接口 vm44_solve(), 内部串联所有阶段:
//   StarSelector → VectorMatcher → IRM 闭环 (Expand ↔ Verify ↔ Fit)
//
// Python 端只需 1 次 ctypes 调用, 无 JSON 序列化, 无数组边界拷贝
// ============================================================================

#include "vm44_types.h"

#ifdef _WIN32
#define VM44_API __declspec(dllexport)
#else
#define VM44_API __attribute__((visibility("default")))
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
//   result         - 输出结果 (调用者分配, 内部 cu/cw 由 vm44_free_result 释放)
//
// 返回: 0=成功, -1=失败 (错误信息在 result->error_msg)
VM44_API int vm44_solve(
    const char* image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const v44::VM44SolveParams* params,
    v44::VM44SolveResult* result
);

// ===========================================================================
// 依赖注入接口
// ===========================================================================

// 注入 Gaia 客户端句柄 (复用已有 gaia_client.dll)
// gaia_client_handle: GaiaClientPy 内部句柄 (void*)
// 注意: 调用者负责管理 GaiaClientPy 生命周期, V4.3 DLL 不负责释放
VM44_API void vm44_set_gaia_client(void* gaia_client_handle);

// 注入 StarDetector 句柄 (复用已有 star_detector.dll)
// detector_handle: StarDetector 内部句柄 (void*)
// 注意: 调用者负责管理 StarDetector 生命周期
VM44_API void vm44_set_star_detector(void* detector_handle);

// ===========================================================================
// 资源释放
// ===========================================================================

// 释放 VM44SolveResult 内部分配的数组 (cu/cw)
// 调用后 result 内指针置 NULL, 可安全重复调用
VM44_API void vm44_free_result(v44::VM44SolveResult* result);

// ===========================================================================
// 默认参数获取
// ===========================================================================

// 获取默认求解参数 (Python 端可调用以获取基础参数, 再覆盖特定字段)
VM44_API void vm44_get_default_params(v44::VM44SolveParams* params);

#ifdef __cplusplus
}
#endif

#endif // VM44_API_H

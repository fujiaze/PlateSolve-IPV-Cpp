// ============================================================================
// exp_stub.cpp - 实验程序 stub
//
// 提供 vm44_select.cpp 需要但 vm44_entry.cpp 实现的函数 (避免链接整个 vm44_entry.cpp)
// 注入句柄: 通过 vm44_set_gaia_client / vm44_set_star_detector (vm44_api.h)
// ============================================================================

#include "vm44_internal.h"
#include "vm44_api.h"
#include <cstring>

namespace v44 {

// 句柄存储 (静态全局, 供 get_xxx_handle 读取)
namespace {
    void* g_gaia_client_handle = nullptr;
    void* g_star_detector_handle = nullptr;
}

// 供 vm44_select.cpp 调用
void* get_gaia_client_handle() {
    return g_gaia_client_handle;
}

void* get_star_detector_handle() {
    return g_star_detector_handle;
}

// 友元注入 (供 C 接口 vm44_set_xxx 写入)
void set_gaia_client_internal(void* h) { g_gaia_client_handle = h; }
void set_star_detector_internal(void* h) { g_star_detector_handle = h; }

} // namespace v44

// ============================================================================
// C 接口实现 (vm44_api.h 声明)
// ============================================================================
extern "C" {

// 注意: VM44_API 宏在 vm44_api.h 已定义
// 实验程序编译为 .exe 不需要 dllexport, 但保持声明一致
void vm44_set_gaia_client(void* gaia_client_handle) {
    v44::set_gaia_client_internal(gaia_client_handle);
}

void vm44_set_star_detector(void* detector_handle) {
    v44::set_star_detector_internal(detector_handle);
}

void vm44_free_result(v44::VM44SolveResult* result) {
    if (!result) return;
    if (result->cu) { delete[] result->cu; result->cu = nullptr; }
    if (result->cw) { delete[] result->cw; result->cw = nullptr; }
    result->n_pairs = 0;
}

void vm44_get_default_params(v44::VM44SolveParams* params) {
    if (!params) return;
    std::memset(params, 0, sizeof(*params));
    // 仅填充 StarSelector 需要的字段 (其他 Phase 不用)
    params->n_modes = 4;
    params->seed = 42;
    params->img_n_target = 50;
    params->gaia_density_ratio = 1.5;
    params->gaia_query_radius_factor = 0.55;
    params->m_lim_step = 0.5;
    params->m_lim_max_iter = 10;
    params->density_tolerance = 0.1;
    // 其余字段为 0 (实验不用)
}

} // extern "C"

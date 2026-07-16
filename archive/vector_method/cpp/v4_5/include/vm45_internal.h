#ifndef VM45_INTERNAL_H
#define VM45_INTERNAL_H

// ============================================================================
// vm45_internal.h - V4.5 内部模块互相调用的函数声明
//
// V4.5 仅含两个核心模块 (Phase 0 + Phase A):
//   vm45_select.cpp  - StarSelector (Phase 0, 复用 V4.4 算法)
//   vm45_relvec.cpp  - 相对向量法 Phase A (θ 求解, 严格按设计文档)
//   vm45_entry.cpp   - vm45_solve() 入口 (仅串联 select + relvec)
//
// 不含 Phase B / IRM / WcsFitter (用户要求 "后续步骤先不做 专注于θ求解")
//
// 编译为单一 vector_match_v4_5.dll, 无 ctypes 边界
// ============================================================================

#include "vm45_types.h"
#include "vm45_log.h"
#include <string>
#include <vector>
#include <functional>

namespace v45 {

// ===========================================================================
// 全局句柄访问器 (在 vm45_entry.cpp 中定义)
// ===========================================================================

// 获取注入的 GaiaClient 句柄 (void* 实际为 GaiaClient*)
// 返回 nullptr 表示未注入, 模块应返回错误
void* get_gaia_client_handle();

// 获取注入的 StarDetector 句柄 (void* 实际为 StarDetectorHandle)
// 返回 nullptr 表示未注入, 模块应返回错误
void* get_star_detector_handle();

// ===========================================================================
// 内部辅助函数 (vm45_select.cpp 中实现, 供单元测试调用)
// 与 V4.4 vm44_select.cpp 算法一致, 仅 namespace/前缀变化
// ===========================================================================

// 计算 FOV 与密度 (从 V4.4 迁移)
void compute_fov_density(
    double focal_length_mm, double pixel_size_um,
    double img_width, double img_height,
    int n_img_bright,
    double gaia_density_ratio, double gaia_query_radius_factor,
    double& s0, double& fov_diag_deg,
    double& query_radius_deg, double& query_area_sqdeg,
    double& img_area_sqdeg, double& rho_img,
    double& rho_target, int& n_target,
    Logger* logger = nullptr);

// 计算初始极限星等 m_cut
double compute_initial_mag_cut(
    double focal_length_mm, double exposure_time_s,
    Logger* logger = nullptr);

// 自适应步长迭代极限星等
void density_match_iterate(
    std::function<int(double, double, double, double)> query_func,
    double center_ra, double center_dec, double query_radius_deg,
    int n_target, double m_cut_initial,
    double step_init, int max_iter, double tolerance,
    double& final_mag_lim, int& final_n_gaia,
    int& iterations, bool& converged,
    Logger* logger = nullptr);

// 图像侧选星: 不对称策略
std::vector<int> select_image_stars(
    const std::vector<double>& flux,
    const std::vector<bool>& saturated,
    int img_n_target,
    Logger* logger = nullptr);

// Gnomonic 正向投影
void gnomonic_forward_proj(
    double ra_deg, double dec_deg,
    double ra0_deg, double dec0_deg,
    double& xi_asec, double& eta_asec, bool& valid);

// ===========================================================================
// Phase 0: StarSelector (vm45_select.cpp) - 复用 V4.4 算法
// ===========================================================================

// 图像侧选星 + Gaia 侧不对称密度匹配查询
// 输入: FITS 路径 + 中心指向 + 焦距/像元 + 参数
// 输出: StarSelection (U ~50 颗 + W ~75-150 颗 + 元数据)
// 返回: 0=成功, -1=失败
int vm45_select(
    const std::string& image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const VM45SolveParams& params,
    StarSelection& output,
    Logger* logger = nullptr
);

// ===========================================================================
// Phase A: 相对向量法 θ 求解 (vm45_relvec.cpp) - 严格按设计文档
// ===========================================================================

// 相对向量法 Phase A: 采样 + 第三星验证 + θ 直方图投票 + 峰值检测
//
// 算法 (来自 v4_4_relvec_sampling_design.md):
//   1. 预计算 W 距离矩阵 + Gaia 星对按距离排序 (k-vector) + 每颗星邻星表
//   2. K_total 次采样:
//      a. 随机图像星对 (i,j) → d_img (角秒, 绝对距离), angle_img
//      b. k-vector 距离查询: d_gaia ∈ [d_img - σ_d, d_img + σ_d]  ← 绝对值范围!
//      c. 第三星三角形全等验证 (距离容差 σ_d)
//      d. 通过则投票: votes[bin] += 1 + log2(1 + n_passed)
//   3. 高斯平滑 → 峰值检测 → 背景中位数 (去峰值 ±5°) → SNR
//   4. 抛物线亚 bin 精化
//
// 输入: U (图像侧, 角秒) + W (Gaia 侧, 角秒) + s0 + 参数
// 输出: RelVecResult (θ_peak + SNR + 直方图 + 通过候选对)
// 返回: 0=成功, -1=失败
int vm45_relvec_match(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    const VM45SolveParams& params,
    RelVecResult& output,
    Logger* logger = nullptr
);

} // namespace v45

#endif // VM45_INTERNAL_H

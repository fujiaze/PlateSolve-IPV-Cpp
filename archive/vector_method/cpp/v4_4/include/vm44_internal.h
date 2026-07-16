#ifndef VM44_INTERNAL_H
#define VM44_INTERNAL_H

// ============================================================================
// vm44_internal.h - V4.3 内部模块互相调用的函数声明
//
// 5 个模块的 .cpp 文件通过本头文件暴露内部函数, 互相调用
// 编译时链接为单一 DLL, 无 ctypes 边界, 无 JSON 序列化
//
// 模块组织:
//   vm44_select.cpp   - StarSelector (Phase 0)
//   vm44_match.cpp    - VectorMatcher (Phase A+B)
//   vm44_expand.cpp   - PairExpander (Phase C / IRM Step 1)
//   vm44_geometry.cpp - 局部几何一致性过滤 (IRM Step 2)
//   vm44_verify.cpp   - PairVerifier (Phase D / IRM Step 3)
//   vm44_fit.cpp      - WcsFitter (Phase E / IRM Step 4)
//   vm44_score.cpp    - S_robust 稳健评分 (IRM Step 5)
//   vm44_irm.cpp      - IRM 闭环主循环
//   vm44_entry.cpp    - vm44_solve() 入口
// ============================================================================

#include "vm44_types.h"
#include "vm44_log.h"
#include <string>
#include <vector>
#include <functional>

namespace v44 {

// ===========================================================================
// 全局句柄访问器 (在 vm44_entry.cpp 中定义)
// ===========================================================================

// 获取注入的 GaiaClient 句柄 (void* 实际为 GaiaClient*)
// 返回 nullptr 表示未注入, 模块应返回错误
void* get_gaia_client_handle();

// 获取注入的 StarDetector 句柄 (void* 实际为 StarDetectorHandle)
// 返回 nullptr 表示未注入, 模块应返回错误
void* get_star_detector_handle();

// ===========================================================================
// 内部辅助函数 (vm44_select.cpp 中实现, 供单元测试调用)
// 不在匿名命名空间, 具有外部链接; 不在 vm44_internal.h 之外的任何头文件声明
// ===========================================================================

// 计算 FOV 与密度 (从 V4.2 ss_core.cpp 迁移)
// 公式:
//   s0            = 206.265 × pixel_size_um / focal_length_mm
//   fov_diag_deg  = sqrt(W² + H²) × s0 / 3600
//   query_radius  = fov_diag_deg × gaia_query_radius_factor
//   query_area    = π × query_radius²
//   img_area      = (W × s0/3600) × (H × s0/3600)
//   rho_img       = n_img_bright / img_area
//   rho_target    = gaia_density_ratio × rho_img
//   n_target      = max(50, round(gaia_density_ratio × n_img × query_area/img_area))
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

// 计算初始极限星等 m_cut (V4.2 公式)
//   m_cut = 6 + 1.5×log10(f_mm) + 2×log10(t_s)
double compute_initial_mag_cut(
    double focal_length_mm, double exposure_time_s,
    Logger* logger = nullptr);

// 自适应步长迭代极限星等 (从 V4.2 ss_core.cpp 迁移)
// query_func: (ra, dec, radius_deg, mag_lim) → 星数
// 算法:
//   m = m_cut_initial
//   for i in 0..max_iter:
//     n = query_func(ra, dec, radius, m)
//     step = (i < 4) ? step_init : step_init × 0.5
//     if n < n_target×(1-tolerance): m += step  (放宽星等)
//     elif n > n_target×(1+tolerance): m -= step (收紧星等)
//     else: break (收敛)
void density_match_iterate(
    std::function<int(double, double, double, double)> query_func,
    double center_ra, double center_dec, double query_radius_deg,
    int n_target, double m_cut_initial,
    double step_init, int max_iter, double tolerance,
    double& final_mag_lim, int& final_n_gaia,
    int& iterations, bool& converged,
    Logger* logger = nullptr);

// 图像侧选星: V4.1 不对称策略
//   饱和星数 > img_n_target → 全选饱和星
//   否则 → 饱和全选 + 非饱和按 flux 降序补足到 img_n_target
// 返回: 选中星点在原数组中的索引
std::vector<int> select_image_stars(
    const std::vector<double>& flux,
    const std::vector<bool>& saturated,
    int img_n_target,
    Logger* logger = nullptr);

// Gnomonic 正向投影 (从 V4.2 Python 迁移)
// 输入: ra_deg, dec_deg (星表位置), ra0_deg, dec0_deg (中心)
// 输出: xi_asec, eta_asec (角秒坐标), valid (是否有效)
void gnomonic_forward_proj(
    double ra_deg, double dec_deg,
    double ra0_deg, double dec0_deg,
    double& xi_asec, double& eta_asec, bool& valid);

// ===========================================================================
// Phase 0: StarSelector (vm44_select.cpp)
// ===========================================================================

// 图像侧选星 + Gaia 侧不对称密度匹配查询
// 输入: FITS 路径 + 中心指向 + 焦距/像元 + 参数
// 输出: StarSelection (U 50 颗 + W ~75-150 颗 + 元数据)
// 返回: 0=成功, -1=失败
int vm44_select(
    const std::string& image_path,
    double ra, double dec,
    double focal_length_mm,
    double pixel_size_um,
    const VM44SolveParams& params,
    StarSelection& output,
    Logger* logger = nullptr
);

// ===========================================================================
// Phase A+B: VectorMatcher (vm44_match.cpp)
// ===========================================================================

// PROSAC 抽样 + θ 直方图投票 + 三级过滤 + Umeyama SVD 精修
// 4 模式并行 (OpenMP), 选择最优模式
// 输入: U (图像侧 50 颗) + W (Gaia 侧) + s0 + 参数
// 输出: VectorMatchResult (s/θ/tx/ty + cu/cw + best_mode)
// 返回: 0=成功, -1=失败
int vm44_match(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    const VM44SolveParams& params,
    VectorMatchResult& output,
    Logger* logger = nullptr
);

// ===========================================================================
// Phase A 路径2: 相对向量法 (vm44_relvec.cpp, V4.4 新增)
// ===========================================================================

// 相对向量法通过第三星验证的候选对
struct RelVecPair {
    int    img_i;           // 图像星 i 索引 (在 U 数组中)
    int    img_j;           // 图像星 j 索引 (在 U 数组中)
    int    gaia_a;          // Gaia 星 a 索引 (在 Wf 数组中)
    int    gaia_b;          // Gaia 星 b 索引 (在 Wf 数组中)
    double theta_rot_deg;   // 该候选的 θ_rot = angle(Δw) - angle(Δu) (度)
    double s_est;           // 该候选的尺度估计 s = d_img / d_gaia (用于 2D 聚类过滤)
    double dx;              // 平移分量 x = U[i].x - s·(cos(θ)·W[a].x - sin(θ)·W[a].y) (用于 3D 聚类可视化)
    double dy;              // 平移分量 y = U[i].y - s·(sin(θ)·W[a].x + cos(θ)·W[a].y) (用于 3D 聚类可视化)
};

// 相对向量法结果 (Phase A 输出)
struct RelVecResult {
    double theta_peak_deg;      // θ 峰值 (度)
    double s_peak;              // s 峰值 (2D 聚类, 用于 Phase B 双过滤)
    double dx_peak;             // dx 峰值 (3D 聚类中点法, 平移 x 分量)
    double dy_peak;             // dy 峰值 (3D 聚类中点法, 平移 y 分量)
    double theta_snr;           // SNR (3D 聚类峰背比)
    int    n_samples;           // 实际采样次数 (自适应停止可能 < relvec_n_samples)
    int    n_total_candidates;  // 总候选数
    int    n_passed;            // 通过第三星验证数
    int    n_focused;           // 聚焦区内候选数 (高可靠, 直接用于 RANSAC)
    bool   success;             // 是否成功 (SNR >= 5)
    std::vector<RelVecPair> passed_pairs;  // 通过第三星验证的候选对列表
};

// 相对向量法匹配 (单模式)
// 算法: 图像星对采样 → k-vector 距离查询 → 第三星交叉验证 → θ 直方图投票
// 完全替代 V4.3 的单θ Phase A (消除 t 假设, 1D 搜索 θ_true)
// 输入: U (图像侧, 角秒) + Wf (翻转后 Gaia, 角秒) + s0 + 参数
// 输出: RelVecResult (θ_peak + SNR + 通过候选列表)
// 返回: 0=成功, -1=失败
int vm44_relvec_match(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& Wf,
    double s0,
    const VM44SolveParams& params,
    RelVecResult& output,
    Logger* logger = nullptr
);

// ===========================================================================
// Phase C / IRM Step 1: PairExpander (vm44_expand.cpp)
// ===========================================================================

// 马氏距离自适应匹配 + Lowe 距离比 + 双向验证 + 区域均匀化
// 输入: U + W + CD + SIP + S_robust(上一轮) + s0 + 参数
// 输出: ExpansionResult (候选匹配对)
// 返回: 0=成功, -1=失败
//
// V4.3 改进: 固定阈值 τ=3×s0 → 自适应 τ_i = max(3.0 × σ_proj, 2.0")
//            σ_proj(x,y) = σ₀ × √(1 + ((x-cx)²+(y-cy)²) / (fov_half)²)
//            σ₀ = S_robust × cos(δ₀) (上一轮评分)
int vm44_expand(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const CDMatrix& cd,
    const SIPCoeffs& sip,
    double s_robust,           // 上一轮 S_robust (首次用 s0 × 0.1)
    double s0,
    double ra0, double dec0,   // 中心赤经赤纬(度)
    int img_width, int img_height,
    const VM44SolveParams& params,
    ExpansionResult& output,
    Logger* logger = nullptr
);

// ===========================================================================
// IRM Step 2: 局部几何一致性过滤 (vm44_geometry.cpp)
// ===========================================================================

// K=8 邻星角距一致性过滤
// 输入: 候选匹配对 + U + W + s0 + S_robust + 参数
// 输出: 过滤后候选 (consistency ≥ 4/8 才保留)
// 返回: 0=成功, -1=失败
//
// 算法:
//   对每个候选 (img_A, gaia_a):
//     neighbors_img = knn(img_A, K=8)
//     neighbors_gaia = knn(gaia_a, K=15)
//     for each img_B: d_img = |img_A - img_B| × s₀
//       检查是否存在 gaia_b 使 |d_img - d_gaia| < max(3.0, 3.0 × S_robust)
//     consistency ≥ 4 才保留
int vm44_geometry_filter(
    const std::vector<MatchPair>& candidates,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    double s_robust,
    const VM44SolveParams& params,
    std::vector<MatchPair>& filtered,
    Logger* logger = nullptr
);

// ===========================================================================
// Phase D / IRM Step 3: PairVerifier (vm44_verify.cpp)
// ===========================================================================

// 几何过滤 → MAD 清洗 → RANSAC → 贝叶斯验证 → 三角形验证
// 输入: 候选匹配对 + U + W + s0 + 参数
// 输出: VerificationResult (内点集 + 验证结果)
// 返回: 0=成功, -1=失败
int vm44_verify(
    const std::vector<MatchPair>& candidates,
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    double fov_diag_deg,
    const VM44SolveParams& params,
    VerificationResult& output,
    Logger* logger = nullptr
);

// ===========================================================================
// Phase E / IRM Step 4: WcsFitter (vm44_fit.cpp)
// ===========================================================================

// 分层 Huber 稳健 LSQ 拟合: CD → 径向畸变 → SIP (BIC 选阶)
// 输入: U + W + 匹配对 + 中心指向 + 焦距/像元 + 参数
// 输出: WcsFitResult (CD + CRVAL + CRPIX + SIP + RMS)
// 返回: 0=成功, -1=失败
//
// V4.3 改进: 普通 LSQ → Huber 稳健 LSQ (IRLS)
//   δ = 1.345 × MAD(r) × 1.4826
//   权重 w_i = min(1, δ/|r_i|)
//   3-5 轮 IRLS 收敛
int vm44_fit(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const std::vector<MatchPair>& pairs,
    double ra0, double dec0,
    double focal_length_mm,
    double pixel_size_um,
    int img_width, int img_height,
    int sip_max_order,
    const VM44SolveParams& params,
    WcsFitResult& output,
    Logger* logger = nullptr
);

// ===========================================================================
// IRM Step 5: S_robust 稳健评分 (vm44_score.cpp)
// ===========================================================================

// 计算稳健评分 S_robust (残差跳变检测 + MAD 备选)
// 输入: 控制点 + CD + SIP + s0 (像素尺度, arcsec/pixel) + 初始控制点数 M₀
// 输出: SRobustResult (S_robust + n_inliers + coverage)
// 返回: 0=成功, -1=失败
//
// 残差计算 (与 vm44_fit Layer 1 一致, 像素空间):
//   ssx =  U.x / s0, ssy = -U.y / s0   (图像像素偏移)
//   ddx = CD^{-1} · W / 3600           (CD^{-1} 预测像素偏移, 无平移)
//   SIP 修正: ddx -= sip_eval(ssx, ssy)
//   t_median = median(ddx - ssx)       (稳健平移估计)
//   r_i = |(ddx - ssx) - t_median| × s0  (残差, arcsec)
//
// 算法:
//   Step A: 残差排序 r_(1) ≤ r_(2) ≤ ... ≤ r_(N)
//   Step B: 残差跳变检测 ratio[i] = r_(i)/r_(i-1) > 3.0 → k_cut
//           备选: MAD 方法 k_cut = count(r_i < median_r + 3.0 × MAD)
//   Step C: N_robust = min(k_cut, max(N/2, M₀))
//           S_robust = rms(前 N_robust 个残差)
int vm44_compute_s_robust(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    const std::vector<MatchPair>& control_points,
    const CDMatrix& cd,
    const SIPCoeffs& sip,
    double s0,
    int M0,
    SRobustResult& output,
    Logger* logger = nullptr
);

// ===========================================================================
// IRM 闭环主循环 (vm44_irm.cpp)
// ===========================================================================

// IRM 闭环迭代精化: 扩增 → 几何过滤 → 验证 → 拟合 → 收敛判定 → 再扩增
// 输入: C₀ (初始控制点) + CD₀ (来自 VectorMatcher) + U + W + s0 + 参数
// 输出: 最终 CD + SIP + control_points + S_robust + n_iters
// 返回: 0=成功, -1=失败
//
// 收敛条件:
//   1. |S_robust_new - S_robust| < 0.05" → 收敛
//   2. S_robust_new > S_robust × 1.1 → 防过拟合
//   3. iter ≥ 10 → 安全上限
int vm44_irm_refine(
    const std::vector<MatchPair>& C0,           // 初始控制点 (Phase B 输出)
    const CDMatrix& CD0,                         // 初始 CD (从 Phase B 变换推导)
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    double focal_length_mm,                      // 焦距 (mm, vm44_fit 需要)
    double pixel_size_um,                        // 像元尺寸 (um, vm44_fit 需要)
    double ra0, double dec0,
    int img_width, int img_height,
    double fov_diag_deg,
    const VM44SolveParams& params,
    CDMatrix& final_cd,
    SIPCoeffs& final_sip,
    std::vector<MatchPair>& final_control_points,
    SRobustResult& final_s_robust,
    int& n_iters,
    bool& converged,
    double& final_bayes_lnK,                     // 最终贝叶斯 lnK (来自最后一轮 verify)
    double& final_triangle_pass_ratio,           // 最终三角形通过率 (来自最后一轮 verify)
    Logger* logger = nullptr
);

} // namespace v44

#endif // VM44_INTERNAL_H

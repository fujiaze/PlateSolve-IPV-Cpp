#ifndef VM45_TYPES_H
#define VM45_TYPES_H

// ============================================================================
// vm45_types.h - V4.5 共享数据结构 (仅 Phase A θ 求解, 精简自 V4.4)
//
// V4.5 严格按 v4_4_relvec_sampling_design.md 实现 Phase A:
//   - 绝对距离 k-vector 查询 (d_gaia ∈ [d_img - σ_d, d_img + σ_d])
//   - 1D θ 直方图 (360 bins × 1°) + 高斯平滑 + 抛物线亚 bin 精化
//   - 砍掉 Phase B (tx/ty), IRM 闭环, WcsFitter, 3D 密度场, 递归聚焦
//
// 与 V4.4 (namespace v44) 并存, 不冲突
// ============================================================================

#include <vector>
#include <string>
#include <cstdint>

namespace v45 {

// ===========================================================================
// 基础数据结构
// ===========================================================================

// 星点 (角秒坐标, 原点在图像中心 / Gaia 切点)
struct StarPoint {
    double x;          // X坐标(角秒)
    double y;          // Y坐标(角秒)
    double flux;       // 流量
    bool   saturated;  // 是否饱和
};

// ===========================================================================
// Phase 0: StarSelector 输出 (与 V4.4 一致, 复用算法)
// ===========================================================================

struct StarSelection {
    std::vector<StarPoint> U;   // 图像侧星点 (角秒坐标, ~50 颗)
    std::vector<StarPoint> W;   // Gaia 侧星点 (角秒坐标, ~75-150 颗)
    // 元数据
    int    img_width;        // 图像宽度(像素)
    int    img_height;       // 图像高度(像素)
    double fov_diag_deg;     // FOV 对角线(度)
    double m_lim_final;      // 最终极限星等
    int    n_gaia_final;     // 最终 Gaia 星数
    int    m_lim_iterations; // 极限星等迭代次数
    double rho_img;          // 图像侧星密度
    double rho_target;       // 目标星密度
    double s0;               // 像素尺度 (arcsec/pixel)
    bool   success;
};

// ===========================================================================
// Phase A: 相对向量法数据结构
// ===========================================================================

// 通过第三星验证的候选对 (i, j, a, b)
// 注: V4.5 不计算 s_est/dx/dy (那是 Phase B 的职责)
struct RelVecPair {
    int    img_i;             // 图像星 i 索引 (在 U 数组中)
    int    img_j;             // 图像星 j 索引 (在 U 数组中)
    int    gaia_a;            // Gaia 星 a 索引 (在 W 数组中)
    int    gaia_b;            // Gaia 星 b 索引 (在 W 数组中)
    double theta_rot_deg;     // 该候选的 Δθ = angle(Δw) - angle(Δu) (度, [-180, 180))
    int    n_third_passed;    // 通过第三星验证的颗数
};

// 相对向量法结果 (Phase A 输出)
struct RelVecResult {
    double theta_peak_deg;       // θ 峰值 (度, [-180, 180))
    double theta_snr;            // SNR = peak / max(bg_median, 1.0)
    int    peak_bin;             // 峰值 bin 索引 [0, 360)
    double bg_median;            // 背景中位数 (去峰值 ±5°)
    int    n_samples;           // 实际采样次数
    int    n_total_candidates;   // 总候选对数
    int    n_passed;             // 通过第三星验证数
    bool   success;              // SNR > 5.0 时为 true

    // θ 直方图 (360 bins × 1°, [-180, 180))
    // 索引 i 对应角度 [i - 180, i - 180 + 1) 度
    std::vector<double> votes;          // 原始投票 (加权)
    std::vector<double> smoothed_votes; // 高斯平滑后 (σ=1 bin, 环形)

    std::vector<RelVecPair> passed_pairs;  // 通过第三星验证的候选对列表
};

// ===========================================================================
// 求解参数 (仅 Phase 0 + Phase A, 砍掉 V4.4 的 IRM/Phase B 参数)
// ===========================================================================

struct VM45SolveParams {
    // --- 基础参数 ---
    int    seed;               // 随机种子(默认42)

    // --- StarSelector 参数 (Phase 0, 与 V4.4 一致) ---
    int    img_n_target;              // 图像侧目标星数(默认50)
    double gaia_density_ratio;        // Gaia 密度比(默认1.5)
    double gaia_query_radius_factor;  // Gaia 查询半径因子(默认0.55)
    double m_lim_step;                // 极限星等步长(默认0.5)
    int    m_lim_max_iter;            // 极限星等最大迭代(默认10)
    double density_tolerance;         // 密度容差(默认0.1)

    // --- 相对向量法参数 (Phase A, 严格按设计文档) ---
    int    K_total;               // 总采样次数(默认20000, 设计文档值)
    double sigma_d_px;            // 距离容差 σ_d (像素, 默认2.0; 内部乘 s0 转角秒)
                                 //   px 单位使算法行为在不同焦距下一致:
                                 //   - 短焦(200mm): 2px ≈ 6.2"  (旧 3" 过严导致宽场失败)
                                 //   - 长焦(2000mm): 2px ≈ 0.6" (旧 3" 过松导致密场爆炸)
    int    n_third;               // 第三星验证颗数(默认0=用全部可用第三星, 提升SNR)
    double third_ratio_min;       // 第三星通过比例阈值(默认0.3, 真匹配>0.5, 假匹配<0.05)
    double theta_bw;              // θ 直方图 bin 宽度(度, 默认1.0)
    double snr_threshold;         // SNR 接受阈值(默认5.0)
    int    relvec_max_u;          // U 组限流上限(默认100)
    int    relvec_max_cand;       // 单次采样候选对上限(默认500)
    double relvec_min_len_frac;   // 最小星对距离比例(默认0.05, 相对 FOV)
    double relvec_max_len_frac;   // 最大星对距离比例(默认0.8)

    // --- 自适应采样停止 (可选, 从 V4.4 沿用) ---
    int    adaptive_stop;         // 启用自适应停止(默认1)
    int    min_samples;            // 最少采样次数(默认200)
    int    check_interval;         // SNR 检查间隔(默认100)
    double snr_eps;               // SNR 相对变化阈值(默认0.05)
    int    max_stable;             // 连续稳定次数(默认3)

    // --- 日志 ---
    const char* log_dir;          // 日志目录(UTF-8, NULL=不写日志)
};

// ===========================================================================
// 求解结果 (仅 Phase A, 砍掉 CD/SIP/tx/ty)
// ===========================================================================

struct VM45SolveResult {
    // θ 求解结果
    double theta_peak_deg;       // θ 峰值 (度, [-180, 180))
    double theta_snr;            // SNR
    int    peak_bin;             // 峰值 bin 索引 [0, 360)

    // θ 直方图 (360 元素, 由 vm45_free_result 释放)
    double* theta_histogram;     // 高斯平滑后的投票 (360 元素)
    int     histogram_size;      // = 360

    // 通过候选数
    int    n_passed;             // 通过第三星验证数
    int    n_samples;            // 实际采样次数

    // 元数据
    int    img_width;
    int    img_height;
    double fov_diag_deg;
    double m_lim_final;
    int    n_gaia_final;
    double s0;                   // 像素尺度 (arcsec/pixel)

    // 状态
    bool   success;
    char   error_msg[256];       // 错误信息
};

} // namespace v45

#endif // VM45_TYPES_H

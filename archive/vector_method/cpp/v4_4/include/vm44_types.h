#ifndef VM44_TYPES_H
#define VM44_TYPES_H

// ============================================================================
// vm44_types.h - V4.4 共享数据结构
//
// V4.4 相对向量法完全替代单θ Phase A, 其余结构与 V4.3 一致
// 本头文件定义所有模块共享的数据结构, 无 ctypes 边界开销
// ============================================================================

#include <vector>
#include <string>
#include <cstdint>

namespace v44 {

// ===========================================================================
// 基础数据结构
// ===========================================================================

// 星点 (角秒坐标, 原点在图像中心)
struct StarPoint {
    double x;          // X坐标(角秒)
    double y;          // Y坐标(角秒)
    double flux;       // 流量
    bool   saturated;  // 是否饱和
};

// 匹配对 (图像星索引 u, 星表星索引 w)
struct MatchPair {
    int u;
    int w;
};

// 相似变换参数 (s, θ, tx, ty)
struct SimTransform {
    double s;       // 尺度
    double theta;   // 旋转角(弧度)
    double tx;      // X平移(角秒)
    double ty;      // Y平移(角秒)
    bool   valid;   // 是否有效
};

// CD 矩阵 (2x2, 标准 WCS 格式)
struct CDMatrix {
    double cd11, cd12, cd21, cd22;  // CD矩阵元素
};

// SIP 多项式系数 (最多 4 阶, 6x6=36 项)
struct SIPCoeffs {
    double A[36];   // SIP A 多项式
    double B[36];   // SIP B 多项式
    int    order;   // 实际 SIP 阶数 (0=无 SIP, 2/3/4)
};

// ===========================================================================
// 模块间传递的中间结果结构体
// ===========================================================================

// StarSelector 输出 (Phase 0)
struct StarSelection {
    std::vector<StarPoint> U;   // 图像侧星点 (角秒坐标, 50 颗)
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
    bool   success;
};

// VectorMatcher 输出 (Phase A+B)
struct VectorMatchResult {
    SimTransform transform;          // 初始变换 (s, θ, tx, ty)
    std::vector<MatchPair> pairs;    // 粗匹配对 (cu, cw)
    double rms;                      // RMS(角秒)
    // 调试信息
    double theta_snr;
    double theta_peak_deg;
    int    best_n_range;
    int    n_phasea_records;
    double prosac_quality_median;
    int    prosac_pool_final;
    int    best_mode;                // flip 模式 (0=无, 1=X, 2=Y, 3=XY)
    bool   success;
};

// PairExpander 输出 (Phase C / IRM Step 1)
struct ExpansionResult {
    std::vector<MatchPair> candidates;   // 候选匹配对 (含 Phase B 初始对)
    int n_expanded;                       // 本轮扩增数
    int n_pairs;                          // 总匹配对数
    // 区域统计
    std::vector<int> region_counts;       // 各区域匹配对数
    bool success;
};

// PairVerifier 输出 (Phase D / IRM Step 3)
struct VerificationResult {
    std::vector<MatchPair> inliers;   // 验证后内点集
    int    n_clean;                   // MAD 清洗后对数
    // 贝叶斯验证
    double bayes_lnK;
    int    bayes_n_match;
    int    bayes_decision;            // 1=接受, 0=弱证据, -1=拒绝
    // 三角形验证
    int    triangle_total;
    int    triangle_passed;
    double triangle_pass_ratio;
    // RANSAC
    int    ransac_n_inliers;
    // 几何过滤
    int    n_before_geometry;
    int    n_after_geometry;
    bool   validated;
    bool   success;
};

// WcsFitter 输出 (Phase E / IRM Step 4)
struct WcsFitResult {
    CDMatrix cd;             // CD 矩阵
    double   crval[2];       // 中心赤经赤纬(度)
    double   crpix[2];       // 参考像素(1-based)
    SIPCoeffs sip;           // SIP 系数
    double   rms_px;         // RMS(像素)
    double   rms_arcsec;     // RMS(角秒)
    int      n_pairs;        // 拟合用对数
    bool     success;
};

// ===========================================================================
// S_robust 稳健评分 (新增)
// ===========================================================================

struct SRobustResult {
    double s_robust;      // 稳健 RMS(角秒)
    int    n_inliers;     // 内点数
    double coverage;      // 覆盖率 (n_inliers / N_img)
    int    k_cut;         // 残差跳变检测的截断点
    double median_r;      // 残差中位数
    double mad;           // MAD (1.4826 标准化)
};

// ===========================================================================
// IRM 闭环状态
// ===========================================================================

struct IRMState {
    int    iter;              // 当前迭代轮数
    double s_robust_prev;     // 上一轮 S_robust
    double s_robust_curr;     // 当前轮 S_robust
    int    n_inliers_prev;    // 上一轮内点数
    int    n_inliers_curr;    // 当前轮内点数
    int    n_candidates;      // 本轮候选匹配对数
    bool   converged;         // 是否收敛
    std::string stop_reason;  // 停止原因
};

// ===========================================================================
// 求解参数 (含 IRM 参数表)
// ===========================================================================

struct VM44SolveParams {
    // --- 基础参数 ---
    int    n_modes;            // flip 模式数(默认4)
    int    seed;               // 随机种子(默认42)

    // --- StarSelector 参数 ---
    int    img_n_target;              // 图像侧目标星数(默认50)
    double gaia_density_ratio;        // Gaia 密度比(默认1.5)
    double gaia_query_radius_factor;  // Gaia 查询半径因子(默认0.55)
    double m_lim_step;                // 极限星等步长(默认0.5)
    int    m_lim_max_iter;            // 极限星等最大迭代(默认10)
    double density_tolerance;         // 密度容差(默认0.1)

    // --- VectorMatcher 参数 ---
    double s_min;             // 尺度下限(默认0.9)
    double s_max;             // 尺度上限(默认1.1)
    int    K_total;           // 总抽样次数(默认10000)
    int    batch_size;        // 批大小(默认500)
    int    min_samples;       // 最小样本数(默认50)
    int    K_top;             // Top-K(默认100)
    int    min_inliers;       // 最小内点数(默认5)
    double w_snr;             // SNR权重(默认0.4)
    double w_sparse;          // 稀疏度权重(默认0.4)
    double w_sat;             // 饱和度权重(默认0.2)
    int    prosac_T_max;      // PROSAC最大抽样(默认10000)
    int    use_prosac;        // 启用PROSAC(默认1)

    // --- PairExpander 参数 ---
    double region_size_px;    // 区域大小(像素, 默认800)
    int    N_floor;           // 每区最少对数(默认5)
    int    N_cap;             // 每区最多对数(默认30)
    int    N_max;             // 全局最多对数(默认1500)

    // --- PairVerifier 参数 ---
    int    mad_iters;                 // MAD 迭代次数(默认3)
    double mad_threshold_factor;      // MAD 阈值因子(默认3.0)
    double mad_min_threshold_arcsec;  // MAD 最小阈值(默认5.0)
    double lnK_accept;                // 贝叶斯接受阈值(默认10)
    double lnK_weak;                  // 贝叶斯弱证据阈值(默认3)
    double eps_A;                     // 三角形面积容差(默认0.1)
    double eps_J;                     // 三角形雅可比容差(默认0.1)
    double triangle_pass_rate;        // 三角形通过率(默认0.7)

    // --- WcsFitter 参数 ---
    int    sip_max_order;     // SIP 最大阶数(默认4)
    int    skip_sip;          // 跳过 SIP(默认0)

    // --- IRM 闭环参数 (V4.3 新增) ---
    int    irm_max_iter;            // IRM 最大迭代(默认10)
    double irm_converge_eps;        // S_robust 收敛阈值(默认0.05)
    double irm_diverge_factor;      // S_robust 变差因子(默认1.1)
    double irm_tau_min;             // 自适应容差下限(默认2.0)
    double irm_tau_factor;          // 自适应容差因子(默认3.0)
    double irm_lowe_ratio;          // Lowe 距离比(默认0.7)
    int    irm_k_geometry;          // 几何邻星数(默认8)
    int    irm_geom_threshold;      // 几何一致性阈值(默认4)
    double irm_geom_dist_tol;       // 几何角距容差(默认3.0)
    int    irm_ransac_max_iter;     // RANSAC 最大迭代(默认200)
    int    irm_ransac_min_inliers;  // RANSAC 最小内点(默认10)
    double irm_huber_delta_factor;  // Huber δ 因子(默认1.345)
    int    irm_sip_min_pairs;       // SIP 最少对数(默认30)
    int    irm_s_initial;           // 初始控制点保护(默认0)

    // --- 相对向量法参数 (V4.4 新增, 完全替代单θ Phase A) ---
    int    relvec_n_samples;        // 最大采样上限(默认5000, 自适应停止可能提前结束)
    int    relvec_max_u;            // U组限流上限(默认100, 解决LDN43候选爆炸)
    double relvec_third_star_tol;   // 第三星验证容差(像素, 默认1.5; 按s_est转换到Gaia角秒域)
    int    relvec_max_cand;         // 单次采样候选上限(默认500)
    double relvec_min_len_frac;     // 最小星对距离比例(默认0.05)
    double relvec_max_len_frac;     // 最大星对距离比例(默认0.8)
    int    relvec_n_third_stars;    // 第三星验证颗数(默认0=用所有可用第三星, 投票无上限; >0=随机采样上限)
    // 自适应采样停止参数 (V4.4 优化, 替代固定次数)
    int    relvec_adaptive_stop;    // 启用自适应停止(默认1)
    int    relvec_min_samples;      // 最少采样次数(默认200, 到达后才开始检查收敛)
    int    relvec_check_interval;   // SNR检查间隔(默认100次采样)
    double relvec_snr_eps;          // SNR相对变化阈值(默认0.05=5%)
    int    relvec_max_stable;       // 连续稳定次数(默认3, 连续3次变化<eps则停止)

    // --- 日志 ---
    const char* log_dir;      // 日志目录(UTF-8, NULL=不写日志)
};

// ===========================================================================
// 求解结果 (对外输出, 与 V4.1/V4.2 兼容)
// ===========================================================================

struct VM44SolveResult {
    // WCS 参数
    double cd[4];             // CD矩阵 [cd11, cd12, cd21, cd22]
    double crval[2];          // 中心赤经赤纬(度)
    double crpix[2];          // 参考像素(1-based)
    double sip_A[36];         // SIP A 多项式
    double sip_B[36];         // SIP B 多项式
    int    sip_order;         // SIP 阶数

    // 精度指标
    double rms_px;            // RMS(像素)
    double rms_arcsec;        // RMS(角秒)
    double s_robust;          // 稳健 RMS(角秒) - V4.3 新增
    int    matched_count;     // 匹配对数
    int    n_inliers;         // 内点数 - V4.3 新增
    int    n_iters;           // IRM 迭代轮数 - V4.3 新增
    bool   irm_converged;     // IRM 是否收敛 - V4.3 新增

    // 变换参数
    double scale_arcsec_px;   // 像素尺度(角秒/像素)
    double rotation_deg;      // 旋转角(度)
    int    flip_mode;         // flip 模式
    double center_ra;         // 实际中心 RA
    double center_dec;        // 实际中心 Dec
    double s0;                // 初始像素尺度
    double s;                 // 尺度因子
    double theta;             // 旋转角(弧度)
    double tx;                // X平移
    double ty;                // Y平移

    // 调试信息
    double theta_snr;
    double theta_peak_deg;
    double bayes_lnK;
    double triangle_pass_ratio;
    int    best_mode;

    // 匹配对索引 (内部分配, 由 vm44_free_result 释放)
    int*   cu;                // U 索引数组
    int*   cw;                // W 索引数组
    int    n_pairs;           // 匹配对数

    // 元数据
    int    img_width;
    int    img_height;
    double fov_diag_deg;
    double m_lim_final;
    int    n_gaia_final;

    // 状态
    bool   success;
    char   error_msg[256];    // 错误信息
};

} // namespace v44

#endif // VM44_TYPES_H

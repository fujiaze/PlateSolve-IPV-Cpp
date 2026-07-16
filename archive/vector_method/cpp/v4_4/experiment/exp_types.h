// ============================================================================
// exp_types.h - V4.4 向量法抽样独立验证实验 - 共享数据结构
//
// 完全脱离主程序 (vm44_match.cpp / vm44_relvec.cpp), 重新实现向量法抽样核心
// 星点选取逻辑保留: 通过 exp_select_adapter 调用 vm44_select 获取真实 U/W
//
// 实验目标 (4 项):
//   1. 3D 聚集能力: (θ, tx, ty) 密度场能否形成聚集簇
//   2. 递归聚焦收敛性: 递归聚焦能否收敛到真值簇 (非镜像簇)
//   3. s 处理有效性: s ±10% 过滤 + 每对星 s_est 估计, (tx,ty) 是否足够聚集
//   4. 相似度加权效果: 变换后 W' 与 U 的相似度加权能否提高真簇 SNR
// ============================================================================

#ifndef EXP_TYPES_H
#define EXP_TYPES_H

#include <vector>
#include <string>
#include <cstdint>

namespace exp44 {

// ============================================================================
// 基础数据结构 (与主程序一致, 但独立定义, 避免耦合)
// ============================================================================

// 星点 (角秒坐标, 原点在图像中心)
struct StarPoint {
    double x;          // X坐标(角秒)
    double y;          // Y坐标(角秒)
    double flux;       // 流量
    bool   saturated;  // 是否饱和
};

// 相似变换参数 (s, θ, tx, ty) - 真值/估计值
//   U = s·R(θ)·W + (tx, ty)
//   R(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
struct SimTransform {
    double s;       // 尺度 (无量纲, 真匹配≈1.0, 容差±10%)
    double theta;   // 旋转角(弧度)
    double tx;      // X平移(角秒) - 图像中心相对天空切点的偏移
    double ty;      // Y平移(角秒)
    bool   valid;
};

// ============================================================================
// 实验数据源
// ============================================================================

// 数据源类型
enum class DataSource {
    SYNTHETIC,   // 模拟数据 (已知真值, 可控噪声)
    REAL         // 真实数据 (调用 vm44_select 获取 U/W)
};

// 模拟数据生成参数
struct SyntheticParams {
    int    n_stars;           // 星点数 (默认 100)
    double fov_diag_asec;     // FOV 对角线 (角秒, 默认 12600" ≈ 3.5°)
    double s_true;            // 真实尺度 (默认 0.9823, 模拟实际偏差)
    double theta_true_deg;    // 真实旋转角 (度, 默认 30°)
    double tx_true;           // 真实平移 X (角秒, 默认 50")
    double ty_true;           // 真实平移 Y (角秒, 默认 -30")
    double noise_sigma;       // 位置噪声 (角秒, 默认 0.5")
    double outlier_ratio;     // 外点比例 (默认 0.3, 模拟假匹配)
    int    seed;              // 随机种子 (默认 42)
};

// 真实数据参数
struct RealDataParams {
    std::string image_path;   // FITS 路径
    double ra;                // 中心 RA (度)
    double dec;               // 中心 Dec (度)
    double focal_length_mm;   // 焦距 (mm)
    double pixel_size_um;     // 像元尺寸 (um)
};

// 实验输入数据 (统一格式)
struct ExpInput {
    std::vector<StarPoint> U;   // 图像侧星点 (角秒)
    std::vector<StarPoint> W;   // Gaia 侧星点 (角秒)
    double s0;                  // 标称像素尺度 (角秒/像素)
    // 真值 (仅模拟数据有效, 真实数据置 NaN)
    SimTransform ground_truth;
    bool has_ground_truth;
    // 元数据
    std::string data_name;      // 数据名称 (用于日志/输出)
    DataSource source;
};

// ============================================================================
// 向量法核心算法参数
// ============================================================================

struct RelVecParams {
    // 采样参数
    int    max_samples;         // 最大采样次数 (默认 5000)
    int    max_u;               // U 组限流上限 (默认 100)
    int    seed;                // 随机种子 (默认 42)

    // s 处理 (核心: ±10% 容差, 每对星 s_est 估计)
    double s_min;               // s 下限 (默认 0.9, 即 -10%)
    double s_max;               // s 上限 (默认 1.1, 即 +10%)

    // k-vector 距离查询
    double min_len_frac;        // 最小星对距离比例 (默认 0.05)
    double max_len_frac;        // 最大星对距离比例 (默认 0.8)

    // 第三星验证
    int    n_third_stars;       // 第三星验证颗数 (默认 10, 0=用所有)
    double third_star_tol_px;   // 第三星容差 (像素, 默认 1.5)
    int    max_cand;            // 单次采样候选上限 (默认 500)

    // 3D 密度场 (θ, tx, ty)
    int    th_bins;             // θ bin 数 (默认 360, 1°/bin)
    int    dxdy_bins;           // tx/ty bin 数 (默认 200)
    int    peak_cluster_half;   // 峰值簇半径 (默认 2, 即 5×5×5)

    // 递归聚焦
    int    min_samples;         // 最少采样次数 (默认 200)
    int    check_interval;      // SNR 检查间隔 (默认 100)
    double snr_threshold;       // 确认聚焦的 SNR 阈值 (默认 10.0)
    double focus_th_half;       // 聚焦区 θ 半宽 (度, 默认 3.0)
    double focus_dxdy_half;     // 聚焦区 tx/ty 半宽 (角秒, 默认 30.0)
    int    focus_shrink_interval;  // 聚焦收紧间隔 (默认 200)
    double focus_shrink_factor;    // 聚焦收紧因子 (默认 0.4)

    // 自适应停止
    int    adaptive_stop;       // 启用自适应停止 (默认 1)
    double snr_eps;             // SNR 相对变化阈值 (默认 0.05)
    int    max_stable;          // 连续稳定次数 (默认 3)
    int    focus_target_n_candidates;  // 聚焦区候选数目标 (默认 50, 收紧到此值且 SNR>snr_final_threshold 即停)
    double snr_final_threshold;         // 收敛 SNR 阈值 (默认 5.0)

    // 相似度加权 (核心: 变换后 W' 与 U 的相似度作为票数权重)
    int    use_similarity_weight;  // 启用相似度加权 (默认 1)
    int    similarity_knn;         // 相似度 KNN 邻居数 (默认 3)
};

// ============================================================================
// 实验输出 (供 CSV 导出 + Python 可视化)
// ============================================================================

// 单个通过候选 (用于 3D 密度场可视化)
struct PassedPair {
    int    img_i;           // 图像星 i 索引
    int    img_j;           // 图像星 j 索引
    int    gaia_a;          // Gaia 星 a 索引
    int    gaia_b;          // Gaia 星 b 索引
    double theta_rot_deg;   // 旋转角 (度)
    double s_est;           // 尺度估计 (无量纲)
    double tx;              // 平移 X (角秒) = U[i].x - s·R(θ)·W[a].x
    double ty;              // 平移 Y (角秒)
    int    n_k_passed;      // 第三星验证通过数
    double similarity;      // 相似度权重 (变换后 W' 与 U 的相似程度)
    int    vote;            // 最终投票数 = n_k_passed × similarity (或 n_k_passed)
};

// 递归聚焦过程快照 (每个 check_interval 记录一次)
struct FocusSnapshot {
    int    sample_idx;          // 采样次数
    int    total_votes;         // 总投票数
    int    n_nonzero_bins;      // 非零 bin 数
    int    peak_cluster;        // 峰值簇累加值
    double snr;                 // SNR = peak / (total / n_nonzero)
    double peak_theta;          // 峰值 θ (度)
    double peak_tx;             // 峰值 tx (角秒)
    double peak_ty;             // 峰值 ty (角秒)
    bool   confirmed;           // 是否已确认聚焦
    int    n_focused;           // 聚焦区内候选数
    int    n_discarded;         // 丢弃数
    double focus_th_lo, focus_th_hi;  // 当前聚焦区 θ 范围
    double focus_tx_lo, focus_tx_hi;  // 当前聚焦区 tx 范围
    double focus_ty_lo, focus_ty_hi;  // 当前聚焦区 ty 范围
};

// 3D 密度场切片 (最终状态, 供 Python 可视化)
struct DensitySlice {
    // θ-tx 投影 (对 ty 求和)
    std::vector<double> theta_tx;  // [th_bins × dxdy_bins]
    // θ-ty 投影 (对 tx 求和)
    std::vector<double> theta_ty;
    // tx-ty 投影 (对 θ 求和)
    std::vector<double> tx_ty;
    int th_bins, dxdy_bins;
    double th_lo, th_hi;
    double tx_lo, tx_hi;
    double ty_lo, ty_hi;
};

// 点对关系 (聚集区内, 供 RANSAC 求解)
//   由密度场聚集区过滤得到, 一个 img_i 可能对应多个 gaia_a (RANSAC 会去重)
struct PointPair {
    int    img_idx;     // 图像星索引 (U 全集索引)
    int    gaia_idx;    // Gaia 星索引
    double s_est;       // 该配对的 s_est (参考用)
    double theta_rot;   // 该配对的 θ_rot (度, 参考用)
    double tx;          // 该配对的 tx (角秒, 来自 passed_pair)
    double ty;          // 该配对的 ty (角秒, 来自 passed_pair)
};

// 实验最终结果
struct ExpResult {
    // 密度场估计 (粗略, 仅用于定位聚集区)
    SimTransform density_estimate;  // 密度场峰值 (粗略, bin 中心)
    double snr_final;               // 最终 SNR
    int    n_samples_actual;        // 实际采样次数
    int    n_passed;                // 通过候选数
    int    n_focused;               // 聚焦区内候选数
    bool   success;                 // 是否成功 (SNR >= 5)

    // ⭐ RANSAC 一步求解结果 (最终输出, 用户纠正后的架构)
    SimTransform ransac_estimate;   // RANSAC + Umeyama SVD 求解的变换参数
    int    n_inliers;               // RANSAC inliers 数
    int    n_ransac_iters;          // RANSAC 迭代次数
    double ransac_rms;              // inliers 的 RMS (角秒)
    std::vector<PointPair> inlier_pairs;  // RANSAC inliers 点对关系

    // 过程数据 (供可视化)
    std::vector<PassedPair> passed_pairs;      // 全部通过候选
    std::vector<FocusSnapshot> focus_history;  // 递归聚焦过程快照
    DensitySlice density_final;                // 最终 3D 密度场切片

    // 误差 (对比 RANSAC 结果 vs 真值, 仅模拟数据有效)
    double err_theta_deg;       // θ 误差 (度)
    double err_tx;              // tx 误差 (角秒)
    double err_ty;              // ty 误差 (角秒)
    double err_s;               // s 误差
};

} // namespace exp44

#endif // EXP_TYPES_H

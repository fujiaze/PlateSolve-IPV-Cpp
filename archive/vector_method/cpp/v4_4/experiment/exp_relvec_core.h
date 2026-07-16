// ============================================================================
// exp_relvec_core.h - V4.4 向量法抽样核心算法 (独立实现, 重新设计)
//
// 核心思想 (用户原话):
//   "通过抽样得到的 tx, ty, θ 在空间上的聚集关系, 来得到真正的解"
//   "正确的变换下, 应该能有很高的投票数, 因此用变换后的向量组相似程度加权"
//   "s 在范围内的点保留抽样结果, s 不在范围内的直接丢弃"
//   "结果肯定是越来越集中的, 最后置信范围内的点都可以当做假阳性, 再用 ransac 快速验证"
//
// 变换模型: U = s·R(θ)·W + (tx, ty)
//   U[i] = 图像星 i 相对图像中心的角秒向量
//   W[a] = Gaia 星 a 相对图像中心(gnomonic 切点)的角秒向量
//   (s, θ, tx, ty) = 待解参数
//
// 算法流程:
//   1. 预计算 W 距离矩阵 D_W + W 距离索引 D_W_sorted (k-vector)
//   2. 预构建 Gaia 星对数组 (距离/角度/索引, 按距离排序)
//   3. 采样循环:
//      a. 随机采图像星对 (i,j) → d_img, θ_img
//      b. k-vector 查询 d_gaia ∈ [d_img/1.1, d_img/0.9] (±10% s 过滤)
//      c. 对每个候选 (a,b):
//         - s_est = d_img / d_gaia_ab (每对星估计, 补偿实际 s 偏差)
//         - θ = angle(Δw) - angle(Δu)
//         - tx = U[i].x - s_est·R(θ)·W[a].x  (单点法)
//         - ty = U[i].y - s_est·R(θ)·W[a].y
//         - 第三星 k 验证 (提高可靠性)
//         - 相似度加权 (变换后 W' 与 U 的相似程度, 提高真簇 SNR)
//      d. 投入 3D 密度场 (θ, tx, ty)
//   4. 递归聚焦: 探索→识别→聚焦→收敛
//   5. 输出: 估计的 (θ, tx, ty) + 过程数据 (供可视化)
//
// 实验目标 (4 项):
//   1. 3D 聚集能力: 真匹配聚集在 (θ_true, tx_true, ty_true)
//   2. 递归聚焦收敛性: 收敛到真值簇 (非镜像簇)
//   3. s 处理有效性: ±10% 过滤 + 每对星 s_est → (tx,ty) 足够聚集
//   4. 相似度加权效果: W' 与 U 相似度加权 → 真簇 SNR 提高
// ============================================================================

#ifndef EXP_RELVEC_CORE_H
#define EXP_RELVEC_CORE_H

#include "exp_types.h"
#include <string>

namespace exp44 {

// ============================================================================
// 默认参数获取
// ============================================================================
RelVecParams getDefaultRelVecParams();

// ============================================================================
// 主算法: 向量法抽样 + 3D 密度场 + 递归聚焦
// ============================================================================
// 输入: U (图像侧, 角秒) + W (Gaia 侧, 角秒) + s0 + 参数
// 输出: ExpResult (估计的变换 + 过程数据)
// 返回: 0=成功, -1=失败
int runRelVecExperiment(
    const std::vector<StarPoint>& U,
    const std::vector<StarPoint>& W,
    double s0,
    const RelVecParams& params,
    const SimTransform& ground_truth,  // 模拟数据真值 (真实数据置 invalid)
    bool has_ground_truth,
    ExpResult& output,
    const std::string& log_dir = ""    // 日志目录 (空=不写日志)
);

} // namespace exp44

#endif // EXP_RELVEC_CORE_H

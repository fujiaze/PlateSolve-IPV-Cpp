// ============================================================================
// exp_data_source.h - V4.4 向量法抽样独立验证实验 - 数据源
//
// 两种数据源:
//   1. 模拟数据 (SYNTHETIC): 已知真值 (s, θ, tx, ty), 可控噪声, 验证算法正确性
//   2. 真实数据 (REAL): 调用 vm44_select 获取真实 U/W, 验证实际效果
//
// 模拟数据生成:
//   1. 在 FOV 范围内随机生成 N 颗 "天空星" W
//   2. 应用变换 U = s·R(θ)·W + (tx, ty) 得到 "图像星" U
//   3. 加入位置噪声 (高斯, σ=noise_sigma)
//   4. 加入外点 (outlier_ratio 比例的星随机移到 FOV 内其他位置)
// ============================================================================

#ifndef EXP_DATA_SOURCE_H
#define EXP_DATA_SOURCE_H

#include "exp_types.h"
#include <string>

namespace exp44 {

// ============================================================================
// 默认模拟数据参数
// ============================================================================
SyntheticParams getDefaultSyntheticParams();

// ============================================================================
// 生成模拟数据
// ============================================================================
// 输入: params (星点数/真值/噪声/外点比例)
// 输出: ExpInput (U + W + ground_truth + s0)
// 返回: 0=成功, -1=失败
int generateSyntheticData(
    const SyntheticParams& params,
    ExpInput& output,
    const std::string& data_name = "synthetic"
);

// ============================================================================
// 获取真实数据 (调用 vm44_select)
// ============================================================================
// 输入: params (FITS 路径/中心指向/焦距/像元)
// 输出: ExpInput (U + W + s0, ground_truth 置 invalid)
// 返回: 0=成功, -1=失败
//
// 注意: 需要先注入 GaiaClient 和 StarDetector 句柄
//   vm44_set_gaia_client(handle);
//   vm44_set_star_detector(handle);
int loadRealData(
    const RealDataParams& params,
    ExpInput& output,
    const std::string& data_name = ""
);

} // namespace exp44

#endif // EXP_DATA_SOURCE_H

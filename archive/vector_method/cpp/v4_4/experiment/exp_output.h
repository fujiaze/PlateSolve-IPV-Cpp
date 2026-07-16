// ============================================================================
// exp_output.h - V4.4 向量法抽样独立验证实验 - CSV 输出
//
// 输出文件 (供 Python 可视化):
//   1. passed_pairs.csv     - 全部通过候选 (θ, s, tx, ty, vote, similarity)
//   2. focus_history.csv    - 递归聚焦过程快照 (sample_idx, snr, peak_*, focus_*)
//   3. density_theta_tx.csv - 3D 密度场 θ-tx 投影
//   4. density_theta_ty.csv - 3D 密度场 θ-ty 投影
//   5. density_tx_ty.csv    - 3D 密度场 tx-ty 投影
//   6. result_summary.csv   - 最终结果汇总 (估计值/真值/误差/SNR)
// ============================================================================

#ifndef EXP_OUTPUT_H
#define EXP_OUTPUT_H

#include "exp_types.h"
#include <string>

namespace exp44 {

// 导出全部 CSV 文件
// output_dir: 输出目录 (自动创建)
// result: 实验结果
// 返回: 0=成功, -1=失败
int exportAllCSV(
    const std::string& output_dir,
    const ExpResult& result,
    const ExpInput& input
);

} // namespace exp44

#endif // EXP_OUTPUT_H

# plate_solve 归档调试脚本（2026-07）

> 归档日期: 2026-07-16
> 来源: 项目根目录一次性诊断脚本

## 归档文件

| 文件 | 用途 | 归档理由 |
|------|------|----------|
| `diag_wcs.py` | WCS 基本信息诊断（CRVAL/CRPIX/CD 矩阵/四角投影） | 硬编码 Galaxy_Center_T4 路径，一次性使用 |
| `diag_wcs_offset.py` | WCS 边缘残差诊断（Gaia 投影位置 vs 实际亮峰偏移） | 针对特定帧边缘 WCS 精度问题 |

## 说明

这些脚本为开发过程中针对特定测试帧的一次性诊断工具，已归档保留供历史参考。
如需复用 WCS 诊断功能，请使用 `lib/plate_solve/tools/diag_projection_plot.py`（通用 WCS 投影精度可视化）。

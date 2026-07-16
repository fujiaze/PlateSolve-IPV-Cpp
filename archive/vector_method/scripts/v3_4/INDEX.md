# 📂 v3_4 脚本索引

> V3.4 版本 — WCS-SIP标准输出，Siril风格CD修正

| 脚本 | 功能 |
|------|------|
| `preview_v34.py` | 对测试帧运行V3.4求解并打印结果汇总 |
| `preview_single.py` | 对单帧运行V3.4求解，输出s/θ/n/RMS等指标 |
| `preview_debug.py` | 单帧求解并打印详细调试信息（各Phase统计） |
| `preview_wcs_v34.py` | 单帧求解后用WCS-SIP投影Gaia星，生成十字覆盖图预览 |
| `preview_overlay_gen.py` | 批量生成WCS-SIP覆盖图（十字标注Gaia预测位置） |
| `preview_test_frame.py` | 对指定测试帧快速运行V3.4求解 |
| `verify_cd.py` | 对比V3.4输出的CD矩阵与PixInsight参考解，验证旋转方向 |
| `analyze_fail.py` | 分析V3.4在短焦窄带帧上的求解失败原因，输出败因统计 |

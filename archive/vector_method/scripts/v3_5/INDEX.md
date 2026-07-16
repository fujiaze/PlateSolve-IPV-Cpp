# 📂 v3_5 脚本索引

> V3.5 版本 — BIC自适应SIP阶数、全仿射CD/CRVAL精修、MAD稳健拟合

### 求解与验证

| 脚本 | 功能 |
|------|------|
| `run_solve.py` | 单帧快速运行V3.5全流程求解，打印s/θ/n/SNR等结果 |
| `full_test.py` | 对多帧批量求解，统计各帧成功率、RMS、耗时等完整指标 |
| `quick_verify.py` | 对单帧求解后快速输出关键参数，验证结果合理性 |
| `verify_wcs_v35.py` | 求解后加载WCS JSON，将Gaia星投影到图像并计算投影残差RMS |
| `verify_gaia.py` | 查询Gaia星表，验证星等截断、FOV内星数等参数 |
| `verify_training_set.py` | 对训练集帧批量求解，统计各帧匹配数、RMS分布 |
| `verify_oiii_ra_dec.py` | 验证Oiii窄带帧的RA/DEC坐标解析是否正确 |

### 批处理与覆盖图

| 脚本 | 功能 |
|------|------|
| `batch_overlay.py` | 随机抽取50帧→V3.5全流程求解→WCS-SIP投影→输出十字覆盖图PNG |
| `preview_wcs_v35.py` | 单帧求解后生成WCS-SIP覆盖图，十字标注Gaia预测位置 |
| `check_headers.py` | 扫描FITS帧，检查WCS头信息（RA/DEC/焦距/像素尺寸）完整性 |

### 诊断与调试

| 脚本 | 功能 |
|------|------|
| `debug_gaia_full.py` | 详细打印单帧Gaia查询的星数、星等范围、FOV覆盖率 |
| `debug_gaia_pos.py` | 验证Gaia星位置在gnomonic投影和WCS投影下的一致性 |
| `debug_gaia_proj.py` | 对比gnomonic投影、CD投影、CD+SIP投影三种方式的差异 |
| `debug_ngaia.py` | 验证极限星等公式给出的Ngaia星数是否满足求解需求 |
| `debug_oiii.py` | 诊断Oiii窄带帧的Phase A抽样效率、θ直方图和SNR |
| `debug_proj_verify.py` | 用已知WCS参数验证正向/逆向投影的往返精度 |
| `debug_sampling.py` | 分析稀疏度加权抽样的效果，对比均匀抽样 |
| `debug_v33_oiii.py` | 用V3.3算法复现Oiii帧的求解，对比V3.5的改进 |
| `diag_cd.py` | 诊断CD矩阵精度，对比V3.5输出与Umeyama直接结果的差异 |
| `diag_sip_root_cause.py` | 分析SIP系数为零的根因（线性修正不够→残差无曲率） |
| `diagnose_nir.py` | 专门针对近红外帧的求解诊断 |
| `diagnose_overlap.py` | 诊断相邻帧重叠区的WCS一致性 |
| `compare_cd.py` | 对比两帧的CD矩阵，验证旋转角和scale的差异 |
| `compare_pixinsight.py` | 对比V3.5求解结果与PixInsight参考解 |

### 拟合与可视化

| 脚本 | 功能 |
|------|------|
| `test_fit_standalone.py` | **独立拟合pipeline测试**：生成合成控制点对→验证线性RMS<1px、SIP<0.5px |
| `test_fit_real.py` | 用真实Phase D匹配对运行独立拟合pipeline，对比C++输出 |
| `标准控制点对可视化调试示例.py` | 全尺寸输出：青圆圈=检测星，品红十字=Gaia投影，黄箭头=残差方向 |
| `_test_overlay.py` | 快速生成3帧测试覆盖图（NGC7293/GC_P2/GC_P1），十字标注Gaia预测 |
| `_test_fix.py` | 修复CRVAL偏移后的单帧覆盖图生成测试 |
| `_match_full.py` | 全尺寸输出匹配对可视化：圆圈+十字+连线，无边框无图例 |
| `_match_viz.py` | 4子图匹配对诊断：全图+3个局部放大，黄色均值箭头，连线着色 |

### 9区域残差诊断系列

| 脚本 | 功能 |
|------|------|
| `_diag_shift.py` | 对比Umeyama直投、CD线性、CD+SIP三种投影方式的残差 |
| `_diag_grid.py` | 3×3网格区域分析：各区域残差均值/方向/径向/切向/剪切分量 |
| `_diag_region.py` | 1对1匹配对在9区域的残差中位数，检测平移/旋转/剪切模式 |
| `_diag_clean.py` | 用Phase D clean对精确验证3种投影的精修后残差 |
| `_diag_dist.py` | 残差直方图分布：按bin统计各距离范围的匹配对数量 |
| `analyze_sip_residuals.py` | 分析SIP拟合前后的残差变化，验证SIP是否有效降低RMS |
| `_q.py` | 快速诊断：CD+SIP投影后1对1匹配对的残差中位数和分布 |

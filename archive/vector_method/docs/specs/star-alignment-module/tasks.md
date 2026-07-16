# Tasks

## Task 1: 创建 star_alignment 模块目录和数据结构
- [x] 创建 `lib/plate_solve/modules/star_alignment/` 目录
- [x] 创建 `psm_star_alignment.h` 头文件，定义：
  - `PSMFlipMode` 枚举 (NONE/X/Y/XY)
  - `PSMStarAlignmentInput` (两个点集指针+匹配参数)
  - `PSMStarAlignmentResult` (偏移量+旋转角+翻转+仿射+匹配质量+索引)
  - 导出函数声明: `psm_star_align()`, `psm_free_star_alignment_result()`

## Task 2: 实现四种翻转模式匹配核心逻辑
- [x] 实现 `try_flip_mode()` 内部函数：
  - 按翻转模式变换图像星点坐标 (NONE/X/Y/XY)
  - 取最亮 n_bright 颗星构建三角形特征
  - KDTree 最近邻搜索 + 最小二乘仿射拟合（max_iterations次迭代）
  - 返回匹配数、平均距离、仿射6参数
- [x] 实现 `star_align_try_all_flips()` 主匹配逻辑：
  - 依次尝试 PSM_FLIP_NONE → PSM_FLIP_X → PSM_FLIP_Y → PSM_FLIP_XY
  - 每种模式评分 = matched_count / (1 + mean_dist/10)
  - 记录最佳结果
  - 提前终止：matched ≥ 50 且 mean_dist ≤ 5px

## Task 3: 实现偏移量和旋转角提取
- [x] 从最佳仿射变换提取：
  - offset_x = a0, offset_y = b0（平移项即偏移量）
  - rotation_deg = atan2(a2, a1) × 180/π
  - scale_factor = sqrt(a1² + a2²)（验证应接近1.0）
- [x] 填充 PSMStarAlignmentResult 所有字段

## Task 4: 实现 psm_star_align 主入口
- [x] 输入验证（点集非空、count > 3）
- [x] 调用 star_align_try_all_flips()
- [x] 错误处理：所有翻转失败返回 PSM_ERR_NO_MATCH
- [x] 日志输出：每种翻转模式的匹配结果、最终偏移量和旋转角

## Task 5: 编译 star_alignment.dll
- [x] 创建 Makefile（g++ -O2 -std=c++17 -shared）
- [x] 编译成功，验证导出符号

## Task 6: 增强 rms_calc 模块
- [x] 在 `psm_rms_calc.h` 新增 `PSMModelEvalInput`, `PSMModelEvalResult` 数据结构
- [x] 在 `psm_rms_calc.h` 新增 `psm_rms_evaluate_model()`, `psm_free_model_eval_result()` 声明
- [x] 在 `psm_rms_calc.cpp` 实现评估逻辑：
  - 从模型参数构建正向映射
  - Gaia DR3 锥形查询
  - 投影到像素空间
  - KDTree 匹配
  - Sigma-clip 过滤
  - 计算 RMS + 评分
- [x] 重新编译 rms_calc.dll

## Task 7: Python 测试脚本
- [x] 创建 `lib/plate_solve/python/test_star_alignment.py`：
  - 读取 FITS 图像 + WCS（中心RA/Dec, scale, CD矩阵）
  - star_detector 检测星点
  - Gaia DR3 查询（1.2×FOV）+ 二分法迭代极限星等
  - Gnomonic 投影 + 比例尺映射到像素坐标系
  - 调用 star_alignment DLL 匹配
  - WCS修正：偏移量→ΔRA/ΔDec, 旋转角→CD矩阵, 翻转→CD符号
  - 输出修正前后WCS对比
  - 生成标注图像（检测星=绿色, Gaia星=红色, 匹配连线=黄色）

## Task 8: 端到端验证
- [x] 使用测试图像运行完整流程
- [x] 验证偏移量合理（WCS偏差通常 < 100px）
- [x] 验证旋转角合理（通常 < 5°）
- [x] 检查标注图像中匹配对视觉对齐正确
- [x] 记录性能数据

# Task Dependencies
- Task 2 依赖 Task 1 (需要数据结构)
- Task 3 依赖 Task 2 (需要匹配结果)
- Task 4 依赖 Task 2, Task 3 (完整流程)
- Task 5 依赖 Task 4 (代码完成)
- Task 6 独立，可与 Task 2~4 并行
- Task 7 依赖 Task 5, Task 6 (需要编译好的DLL)
- Task 8 依赖 Task 7 (需要测试脚本)

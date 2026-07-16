# Tasks

## Task 1: 创建模块目录和数据结构定义
- [x] 创建 `lib/plate_solve/modules/feature_match/` 目录
- [x] 创建 `psm_feature_match.h` 头文件，定义数据结构：
  - `PSMFeatureStar` (x, y, flux, orig_idx)
  - `PSMFeatureTriangle` (a_idx, b_idx, c_idx, ba_ratio, ca_ratio, side_a_angle, side_a_length)
  - `PSMMatchPair` (img_idx, cat_idx, votes, flux_ratio)
  - `PSMMatchResult` (pairs, pair_count, tri_match_count, total_votes)
- [x] 声明导出函数：
  - `psm_feature_match()` 主入口
  - `psm_free_match_result()` 释放结果

## Task 2: 实现星点筛选函数
- [x] 实现 `sort_stars_by_flux()` - flux降序排序
- [x] 实现 `select_brightest_stars()` - 取top-N并保留原始索引
- [x] 测试：验证排序正确性和索引保留

## Task 3: 实现三角形构建和剪枝
- [x] 实现 `calc_distance_matrix()` - NxN距离矩阵
- [x] 实现 `build_triangles()` - 枚举C(N,3)并计算特征
  - 边长排序 a >= b >= c
  - 特征计算 ba, ca, side_a_angle, a_length
  - 顶点索引识别 (a_idx对最长边)
- [x] 实现 `prune_triangles()` - 剔除等边三角形
  - ba > 0.9 剔除
  - a_length < 5px 剔除
- [x] 实现 `sort_triangles_by_ba()` - 按ba升序排序（用于二分查找）
- [x] 测试：验证三角形数量和剪枝效果

## Task 4: 实现三角匹配和投票矩阵
- [x] 实现 `find_ba_range()` - 二分查找ba范围
- [x] 实现 `match_triangles()` - 三角匹配主循环
  - 对每个Gaia三角形，找候选图像三角形
  - 计算(ba, ca)空间距离
  - 可选尺度过滤
- [x] 实现 `create_vote_matrix()` - 创建并填充投票矩阵
- [x] 测试：验证匹配对数和投票矩阵

## Task 5: 实现Top Vote Getters
- [x] 实现 `extract_top_pairs()` - 提取最佳匹配对
  - 找每个图像星的最佳Gaia星
  - 找每个Gaia星的最佳图像星
  - 最少票数过滤 (votes < 2)
  - 双向一致性去重
- [x] 实现 `flux_enhance_votes()` - flux辅助投票增强（可选）
- [x] 测试：验证匹配对数量和质量

## Task 6: 实现主入口和结果释放
- [x] 实现 `psm_feature_match()` 主入口
  - 输入：图像星点列表、Gaia星点列表、配置参数
  - 输出：PSMMatchResult
  - 流程：筛选→构建三角形→剪枝→匹配→投票→提取
- [x] 实现 `psm_free_match_result()` 释放结果
- [x] 添加日志输出

## Task 7: 编译和Makefile
- [x] 创建 Makefile
- [x] 编译验证 (g++ -O2 -march=native -Wall -std=c++17)
- [x] 测试 DLL 导出符号

## Task 8: Python绑定和集成测试
- [x] 创建 `python/feature_match.py` ctypes封装
- [x] 更新测试脚本，使用新的 feature_match 模块
- [x] 测试：读取图像 → 星点检测 → Gaia查询 → 特征匹配 → 仿射拟合
- [x] 对比测试结果：匹配对数应 >= 50，RMS应 < 10px

# Task Dependencies
- Task 2 依赖 Task 1
- Task 3 依赖 Task 2
- Task 4 依赖 Task 3
- Task 5 依赖 Task 4
- Task 6 依赖 Task 5
- Task 7 依赖 Task 1-6
- Task 8 依赖 Task 7

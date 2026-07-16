# Tasks

- [ ] Task 1: 实现亮星选择函数 select_bright_stars()
  - [ ] SubTask 1.1: 实现图像侧选择逻辑（饱和星+最亮正常星，总计≤50颗）
  - [ ] SubTask 1.2: 实现星表侧选择逻辑（最亮M=2×N_img颗）
  - [ ] SubTask 1.3: 返回选择结果含亮度排名信息

- [ ] Task 2: 实现四边形结构特征计算 compute_quad_features()
  - [ ] SubTask 2.1: 实现C(N,4)四边形枚举和6条边长计算
  - [ ] SubTask 2.2: 实现边长排序和5个不变比值计算
  - [ ] SubTask 2.3: 实现亮度排序约束（4颗星的亮度排名置换）
  - [ ] SubTask 2.4: 验证特征对旋转/平移/缩放/翻转的不变性（单元测试）

- [ ] Task 3: 实现结构特征匹配与投票 structural_match()
  - [ ] SubTask 3.1: 构建星表特征5维KDTree
  - [ ] SubTask 3.2: 实现特征空间近邻搜索（半径0.01）
  - [ ] SubTask 3.3: 实现亮度排序兼容性检查
  - [ ] SubTask 3.4: 实现投票累积和互为最佳约束
  - [ ] SubTask 3.5: 实现sigma-clip异常过滤
  - [ ] SubTask 3.6: 端到端测试验证匹配正确性（实际数据）

- [ ] Task 4: 实现仿射行列式翻转检测 determine_flip_from_affine()
  - [ ] SubTask 4.1: 实现行列式和系数符号判断逻辑
  - [ ] SubTask 4.2: 确定翻转后应用翻转到星表坐标并重新计算仿射
  - [ ] SubTask 4.3: 验证4种翻转模式的正确检测

- [ ] Task 5: 实现bright_star_structural_match()整合函数
  - [ ] SubTask 5.1: 整合亮星选择+特征匹配+异常过滤+仿射求解+翻转检测
  - [ ] SubTask 5.2: 实现全星点验证匹配（半径2px，sigma-clip迭代）
  - [ ] SubTask 5.3: 端到端测试验证RMS

- [ ] Task 6: 修复迭代重投影 iterative_reprojection()
  - [ ] SubTask 6.1: 将最终匹配半径从5px改为2px
  - [ ] SubTask 6.2: 将重投影半径从[100,50,30,10,5]改为[30,10,5,2]
  - [ ] SubTask 6.3: 验证收敛行为（实际数据）

- [ ] Task 7: 修复scale_arcsec_px报告
  - [ ] SubTask 7.1: 使用输入的像素尺度而非仿射线性部分模
  - [ ] SubTask 7.2: 修正rms_px计算（用scale_arcsec_px除）

- [ ] Task 8: 更新InitialWCS.solve()流程
  - [ ] SubTask 8.1: 将7步流程改为6步（移除4模式独立测试循环）
  - [ ] SubTask 8.2: 移除对旧match_with_flip/select_best_flip的调用
  - [ ] SubTask 8.3: 端到端测试RMS<2px

# Task Dependencies
- Task 2 依赖 Task 1（需要亮星选择结果）
- Task 3 依赖 Task 2（需要特征描述子）
- Task 4 独立
- Task 5 依赖 Task 3, 4（需要匹配结果和翻转检测）
- Task 6, 7 独立（可并行）
- Task 8 依赖 Task 5, 6, 7

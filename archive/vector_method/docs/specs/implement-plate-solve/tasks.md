# Tasks

- [x] Task 1: 搭建plate_solve模块骨架
  - [x] SubTask 1.1: 创建 `lib/plate_solve/` 目录结构（src/, include/, python/, logs/）
  - [x] SubTask 1.2: 编写 `include/plate_solve.h` 公共C头文件，定义数据结构和函数声明
  - [x] SubTask 1.3: 编写 `src/psolve_log.cpp/h` 日志系统
  - [x] SubTask 1.4: 编写 Makefile，编译为 plate_solve.dll

- [x] Task 2: 实现FOV计算与元数据解析
  - [x] SubTask 2.1: 编写 `src/psolve_fov.cpp/h`，实现采样率和FOV计算
  - [x] SubTask 2.2: 实现从AIOImageMetadata提取焦距、像元大小、中心坐标
  - [x] SubTask 2.3: 实现极限星等计算（参考siril的compute_mag_limit_from_position_and_fov）

- [x] Task 3: 实现Gnomonic投影
  - [x] SubTask 3.1: 编写 `src/psolve_projection.cpp/h`
  - [x] SubTask 3.2: 实现正投影 sky_to_plane(RA,Dec,RA0,Dec0) → (x,y)
  - [x] SubTask 3.3: 实现逆投影 plane_to_sky(x,y,RA0,Dec0) → (RA,Dec)
  - [x] SubTask 3.4: 单元测试：往返投影精度验证

- [x] Task 4: 实现三角匹配算法
  - [x] SubTask 4.1: 编写 `src/psolve_triangle.cpp/h`
  - [x] SubTask 4.2: 实现三角形构建：从星点列表生成三角形，计算边长比(b/a, c/a)和方向角
  - [x] SubTask 4.3: 实现三角形匹配：在三角形空间中搜索匹配对，投票机制
  - [x] SubTask 4.4: 实现从匹配三角形提取星点对应关系

- [x] Task 5: 实现RANSAC过滤
  - [x] SubTask 5.1: 编写 `src/psolve_ransac.cpp/h`
  - [x] SubTask 5.2: 实现仿射变换矩阵计算（6参数线性模型）
  - [x] SubTask 5.3: 实现RANSAC迭代：随机采样、内点统计、最优模型选择
  - [x] SubTask 5.4: 实现变换矩阵验证：行列式检查、尺度检查

- [x] Task 6: 实现粗解析主流程
  - [x] SubTask 6.1: 编写 `src/psolve_coarse.cpp/h`
  - [x] SubTask 6.2: 实现极限星等迭代：查询Gaia → 比较星点数 → 调整星等 → 重新查询
  - [x] SubTask 6.3: 实现粗解析主循环：投影参考星 → 三角匹配 → RANSAC → 获取初始变换
  - [x] SubTask 6.4: 实现迭代精化：逆映射参考星 → dynamic_psf重搜索(5px) → RANSAC → 更新变换 → 检查RMS收敛
  - [x] SubTask 6.5: 实现RMS误差计算和收敛判断

- [x] Task 7: 实现精解析（DDM-RBF残差拟合）
  - [x] SubTask 7.1: 编写 `src/psolve_fine.cpp/h`
  - [x] SubTask 7.2: 实现残差计算：粗解析模型预测位置与实际位置的偏差(Δx, Δy)
  - [x] SubTask 7.3: 实现域分解：将图像区域划分为重叠子域（网格划分），每个子域≥10颗星
  - [x] SubTask 7.4: 实现局部RBF插值：薄板样条（TPS）或高斯RBF，每个子域独立求解系数
  - [x] SubTask 7.5: 实现子域拼接：距离加权平均，平滑合并各子域残差场
  - [x] SubTask 7.6: 实现最终映射：粗解析仿射模型 + RBF残差场校正
  - [x] SubTask 7.7: 实现WCS参数输出（CRPIX, CRVAL, CD矩阵 + 残差查找表）

- [x] Task 8: 实现C API入口
  - [x] SubTask 8.1: 编写 `src/psolve_api.cpp`，实现 psolve_create/destroy/coarse/fine 等接口
  - [x] SubTask 8.2: 整合Gaia客户端调用（链接gaia_xpsd_client）
  - [x] SubTask 8.3: 整合dynamic_psf调用（链接dynamic_psf）
  - [x] SubTask 8.4: 编写完整Makefile，编译链接所有依赖

- [x] Task 9: 实现Python绑定
  - [x] SubTask 9.1: 编写 `python/plate_solve.py`，ctypes封装PlateSolver类
  - [x] SubTask 9.2: 定义Python数据类（SolveResult, WCSResult, MatchedStar等）
  - [x] SubTask 9.3: 实现完整调用流程：读取图像 → 检测星点 → 粗解析 → 精解析

- [x] Task 10: 编译验证
  - [x] SubTask 10.1: plate_solve.dll 编译成功
  - [x] SubTask 10.2: Python绑定语法验证通过

# Task Dependencies
- [Task 2] depends on [Task 1] (需要头文件和日志系统)
- [Task 3] depends on [Task 1] (需要头文件和日志系统)
- [Task 4] depends on [Task 1] (需要头文件和日志系统)
- [Task 5] depends on [Task 1] (需要头文件和日志系统)
- [Task 6] depends on [Task 2, Task 3, Task 4, Task 5] (粗解析需要所有子模块)
- [Task 7] depends on [Task 6] (精解析依赖粗解析结果)
- [Task 8] depends on [Task 6, Task 7] (API整合所有功能)
- [Task 9] depends on [Task 8] (Python绑定依赖C API)
- [Task 10] depends on [Task 9] (测试依赖完整流程)
- [Task 3, Task 4, Task 5] 可并行开发

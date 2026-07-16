# Tasks

- [x] Task 1: 4种旋转/镜像假设 + atFindTrans + iter_trans
  - [x] SubTask 1.1: 对每个mode (不变/180°/x镜像/y镜像) 做三角匹配+仿射
  - [x] SubTask 1.2: iter_trans (35%分位数sigma, 10-sigma clip, 5次迭代) 
  - [x] SubTask 1.3: 以匹配对数量+scale选最优假设
  - **结果**: 4假设都得7对三角匹配, iter_trans后245对, RMS=7.09px, 0.33s

- [x] Task 2: 内存内星等迭代 + 网格加速NN + matched_stars
  - [x] SubTask 2.1: 仿射确定后按星等选≈det_count颗Gaia星
  - [x] SubTask 2.2: 50px格网空间索引, NN搜索10px内匹配
  - [x] SubTask 2.3: 填入matched_stars (img_x/y, cat_ra/dec/mag, residual_x/y)
  - **结果**: 245对匹配, RMS=7.09px

- [x] Task 3: Python诊断图 + 向量化
  - [x] SubTask 3.1: numpy向量化距离/残差/分bin
  - [x] SubTask 3.2: 图1: 距离vs残差散点+RMS线
  - [x] SubTask 3.3: 图2: 径向直方图
  - [x] SubTask 3.4: 控制台分bin RMS表
  - **结果**: affine_diagnostics.png + 分bin统计输出

- [ ] Task 4 (待实现): 畸变模型粗迭代 (第2步)
  - 原因: 仿射无法收敛中心 (gnomonic畸变被吸收到scale=0.53, 反推中心不可靠)
  - 需先建立畸变模型才能精确中心

# Task Dependencies
- Task 2 依赖 Task 1
- Task 3 依赖 Task 2
- Task 4 独立

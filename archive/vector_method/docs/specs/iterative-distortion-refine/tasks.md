# Tasks

- [ ] Task 1: 创建 iterative_refine 模块目录和数据结构
  - [ ] SubTask 1.1: 创建 `lib/plate_solve/modules/iterative_refine/` 目录
  - [ ] SubTask 1.2: 创建 `psm_iterative_refine.h` 头文件，定义：
    - `PSMDistortion` 结构体（dx1,dx2,dx3,dy1,dy2,dy3）— 纯二次项畸变
    - `PSMIterRefineResult` 结构体（精化后的中心/比例尺/旋转/翻转/仿射/畸变/匹配信息）
    - 导出函数声明: `psm_iterative_refine()`, `psm_free_iter_refine_result()`
  - [ ] SubTask 1.3: 在 `plate_solve.h` 新增 `PSOLVE_ERR_ITER_FAILED = 9` 错误码

- [ ] Task 2: 实现仿射+畸变拟合（12参数）
  - [ ] SubTask 2.1: 实现 `psm_affine_distortion_compute()` 函数：
    - 构建线性方程组：
      - x_gaia = a0 + a1*x + a2*y + dx1*x² + dx2*x*y + dx3*y²
      - y_gaia = b0 + b1*x + b2*y + dy1*x² + dy2*x*y + dy3*y²
    - 最小二乘求解12参数（高斯消元法）
    - 匹配对数<6时返回错误
  - [ ] SubTask 2.2: 实现 `psm_affine_distortion_apply()` 函数：
    - 给定(x,y)和仿射+畸变参数，计算(x',y')
  - [ ] SubTask 2.3: 实现 `psm_extract_params()` 函数：
    - 从仿射参数提取：比例尺=sqrt(a1²+a2²), 旋转角=atan2(a2,a1), 中心偏移=(a0,b0)

- [ ] Task 3: 实现近距离多边形匹配
  - [ ] SubTask 3.1: 实现Grid哈希最近邻搜索（复用coarse_solve的Grid结构）：
    - 构建图像星点Grid（cell_size=10px）
    - 对每个Gaia映射星点搜索最近邻
    - 距离阈值：初始10px，逐步收紧
  - [ ] SubTask 3.2: 实现双向一致性检查：
    - img→cat最近邻 = cat→img最近邻
  - [ ] SubTask 3.3: 实现sigma-clip过滤（3σ剔除离群匹配对）

- [ ] Task 4: 实现Gaia星点像素映射
  - [ ] SubTask 4.1: 实现 `psm_map_gaia_to_pixel()` 函数：
    - Gnomonic投影 → 角度转像素 → 旋转 → 翻转 → 平移到图像中心
  - [ ] SubTask 4.2: 实现矩形裁剪 `psm_clip_to_fov()`：
    - 投影到切平面 → 旋转 → 丢弃|x|>W/2或|y|>H/2的星点

- [ ] Task 5: 实现迭代精化主流程
  - [ ] SubTask 5.1: 实现主入口 `psm_iterative_refine()`：
    - 输入验证（星点数≥3，参数有效）
    - 初始化：从输入参数构建初始映射
    - 迭代循环（最多5次）：
      1. 映射Gaia星点到像素空间（含仿射+畸变校正）
      2. 近距离匹配（搜索半径逐步收紧：10→8→6→5→5px）
      3. 拟合仿射+畸变模型（12参数）
      4. 从仿射参数提取更新的中心/旋转/比例尺
      5. 收敛判断：RMS变化<1%
    - 填充输出结果
  - [ ] SubTask 5.2: 实现参数提取逻辑：
    - 比例尺：scale = sqrt(a1²+a2²) × original_scale
    - 旋转角：rotation = atan2(a2,a1)
    - 中心偏移：逆投影(a0,b0)到天球坐标

- [ ] Task 6: 实现日志系统
  - [ ] SubTask 6.1: 实现 `psm_iter_refine_log.cpp/h`：
    - 日志路径: `modules/iterative_refine/logs/iterative_refine.log`
    - 输出：每次迭代的匹配数、RMS、仿射参数、畸变参数

- [ ] Task 7: 编译 iterative_refine.dll
  - [ ] SubTask 7.1: 创建 Makefile（g++ -O2 -std=c++17 -shared）
  - [ ] SubTask 7.2: 编译成功，验证导出符号

- [ ] Task 8: Python调度层实现
  - [ ] SubTask 8.1: 在 `plate_solve.py` 新增 `solve_iterative_refine()` 方法：
    - 从粗解析结果提取center_RA/Dec, scale, rotation, flip
    - 计算旋转后的FOV矩形角点天球坐标
    - 调用gaia_client锥形查询（对角线FOV + 5° margin）
    - 矩形裁剪（投影→旋转→丢弃矩形外星点）
    - 极限星等迭代（二分法，目标=det_count×1.5）
    - 调用 iterative_refine DLL
    - 用精化结果更新WCS参数
  - [ ] SubTask 8.2: 实现WCS更新逻辑：
    - 中心偏移→逆投影→ΔRA/ΔDec
    - 旋转角→CD矩阵旋转
    - 比例尺→CD矩阵缩放
    - 畸变→SIP系数（可选）

- [ ] Task 9: 端到端测试
  - [ ] SubTask 9.1: 创建测试脚本 `test_iterative_refine.py`：
    - 读取测试图像 + 粗解析结果
    - 运行完整迭代精化流程
    - 输出精化前后RMS对比
    - 输出仿射参数和畸变参数
    - 生成标注图像（匹配对连线）
  - [ ] SubTask 9.2: 验证RMS从~7px降至<1px
  - [ ] SubTask 9.3: 验证中心坐标偏差<5px
  - [ ] SubTask 9.4: 验证旋转角精度<0.1°
  - [ ] SubTask 9.5: 验证畸变参数在无畸变图像接近零

# Task Dependencies
- [Task 2] depends on [Task 1] (需要数据结构定义)
- [Task 3] depends on [Task 1] (需要数据结构定义)
- [Task 4] depends on [Task 1] (需要数据结构定义)
- [Task 5] depends on [Task 2, Task 3, Task 4] (主流程整合所有子模块)
- [Task 6] depends on [Task 1] (可与Task 2~4并行)
- [Task 7] depends on [Task 5, Task 6] (代码完成)
- [Task 8] depends on [Task 7] (需要编译好的DLL)
- [Task 9] depends on [Task 8] (需要Python绑定)
- [Task 2, Task 3, Task 4] 可并行开发

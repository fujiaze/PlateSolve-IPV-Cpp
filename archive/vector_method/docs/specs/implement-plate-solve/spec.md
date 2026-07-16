# Plate Solve 天文解析模块 Spec

## Why
天文图像处理流水线需要将图像像素坐标与天球坐标建立映射关系（plate solving），这是后续光度测量、图像对齐、标准化等步骤的基础。当前项目已有星点检测（star_detector）、PSF拟合（dynamic_psf）、Gaia星表查询（gaia_xpsd_client）和图像IO（astro_image_io）模块，需要开发plate solve模块将这些能力串联起来，实现从图像到天球坐标的自动解析。

## What Changes
- 新增 `lib/plate_solve` 模块，包含C++核心算法（编译为plate_solve.dll）和Python绑定
- 实现粗解析（coarse solve）：基于FITS头信息计算FOV，迭代极限星等从Gaia提取参考星，三角匹配+RANSAC建立初始变换模型，迭代精化直到RMS收敛
- 实现精解析（fine solve）：基于粗解析结果，使用DDM-RBF（域分解-径向基函数）拟合残差
- 实现Gnomonic投影：将天球坐标投影到切平面，用于星点匹配
- 实现三角匹配算法：参考siril的atpmatch，基于Valdes et al. 1995
- 实现RANSAC过滤：剔除误匹配星点对
- 实现迭代精化：逆映射参考星到图像，用dynamic_psf重新搜索星点，5px阈值匹配

## Impact
- Affected specs: 无已有spec受影响，这是全新模块
- Affected code: 依赖 `lib/star_detector`, `lib/dynamic_psf`, `lib/gaia_xpsd_client`, `lib/astro_image_io`
- 新增文件: `lib/plate_solve/` 目录下完整的C++源码、头文件、Python绑定、Makefile

## ADDED Requirements

### Requirement: FOV与采样率计算
系统应从FITS文件头读取焦距（FOCALLEN）和像元大小（XPIXSZ/YPIXSZ），计算：
- 采样率（arcsec/pixel）= 206.265 × pixel_size_μm / focal_length_mm
- FOV宽度（arcmin）= scale_arcsec × width_pixels / 60
- FOV高度（arcmin）= scale_arcsec × height_pixels / 60
- FOV对角线半径（deg）= sqrt((scale × rx)² + (scale × ry)²) / 7200

#### Scenario: 正常FITS头
- **WHEN** FITS头包含FOCALLEN=800mm, XPIXSZ=3.76μm, 图像4500×3600
- **THEN** 采样率≈0.97 arcsec/px, FOV≈72.8×58.2 arcmin

#### Scenario: 缺少焦距或像元信息
- **WHEN** FITS头缺少FOCALLEN或XPIXSZ
- **THEN** 返回错误码，提示用户手动输入

### Requirement: 极限星等迭代
系统应根据图像检测到的星点数量，迭代调整极限星等从Gaia提取数量相近的参考星：
1. 初始极限星等由 `compute_mag_limit_from_position_and_fov()` 计算（参考siril算法）
2. 查询Gaia，比较返回星点数与star_detector检测数
3. 若Gaia星点过多，降低极限星等；过少则提高
4. 迭代直到Gaia星点数与检测星点数比值在0.8~1.2范围内，或达到最大迭代次数（5次）

#### Scenario: 星点数量匹配
- **WHEN** star_detector检测到5000颗星，初始极限星等返回8000颗Gaia星
- **THEN** 降低极限星等，重新查询，直到Gaia星点数在4000~6000范围内

#### Scenario: 无法匹配
- **WHEN** 迭代5次后仍无法达到目标范围
- **THEN** 使用最后一次查询结果，记录警告日志

### Requirement: Gnomonic投影
系统应实现Gnomonic（TAN）投影，将天球坐标(RA,Dec)投影到以指定点(RA0,Dec0)为中心的切平面坐标(x,y)：
- 正投影: (RA,Dec) → (x,y)，x和y单位为度
- 逆投影: (x,y) → (RA,Dec)
- 投影中心为FITS头中的坐标（OBJCTRA/OBJCTDEC或WCS CRVAL）

#### Scenario: 投影精度
- **WHEN** 对已知坐标的星点进行正投影再逆投影
- **THEN** 往返误差 < 0.001角秒

### Requirement: 三角匹配算法
系统应实现基于三角形的星点匹配算法（参考siril atpmatch / Valdes et al. 1995）：
1. 对图像星点和参考星分别按星等排序，取最亮的N颗（默认60颗）
2. 构建三角形：计算所有三组合的边长比(b/a, c/a)和方向角
3. 在三角形空间中匹配：比较边长比和方向角，投票确定匹配对
4. 从匹配三角形中提取星点对应关系
5. 用匹配对计算初始仿射变换矩阵（6参数线性变换）

#### Scenario: 匹配成功
- **WHEN** 图像星点和参考星有足够重叠（≥6对匹配）
- **THEN** 返回仿射变换矩阵和匹配星点对列表

#### Scenario: 匹配失败
- **WHEN** 匹配对数 < 3
- **THEN** 返回SOLVE_NO_MATCH错误码

### Requirement: RANSAC过滤
系统应在三角匹配得到初始对应关系后，使用RANSAC算法过滤误匹配：
1. 随机选取3对星点计算变换矩阵
2. 计算所有星点对的变换残差
3. 统计内点数（残差 < 阈值）
4. 重复N次（默认1000次），保留内点最多的模型
5. 用所有内点重新计算变换矩阵

#### Scenario: 误匹配过滤
- **WHEN** 初始匹配包含30%误匹配
- **THEN** RANSAC能正确识别并剔除误匹配，保留正确对应关系

### Requirement: 粗解析迭代精化
系统应在获得初始变换模型后，进行迭代精化直到RMS误差收敛：
1. 用当前变换模型将Gaia参考星逆映射到图像坐标系
2. 对每个逆映射位置，用dynamic_psf在5px半径内重新搜索星点
3. 将搜索到的星点与参考星建立新的对应关系
4. 对新对应关系进行RANSAC过滤
5. 用过滤后的对应关系重新计算变换矩阵
6. 计算RMS误差
7. 若RMS误差变化 < 1%（或达到最大迭代10次），停止迭代

#### Scenario: 迭代收敛
- **WHEN** 初始RMS=2.0px，经过3次迭代后RMS=0.8px，第4次迭代RMS=0.79px
- **THEN** 停止迭代，记录最终变换模型和匹配星点的图像坐标

#### Scenario: 迭代不收敛
- **WHEN** 10次迭代后RMS仍在变化
- **THEN** 使用RMS最小的迭代结果，记录警告

### Requirement: 精解析（DDM-RBF残差拟合）
系统应在粗解析基础上，使用DDM-RBF（Domain Decomposition Method - Radial Basis Function）方法拟合残差，而非全局高阶多项式：
1. 计算残差：对每对匹配星点，计算粗解析模型预测位置与实际位置的偏差(Δx, Δy)
2. 域分解：将图像区域划分为重叠子域（网格划分），每个子域包含足够数量的星点（≥10颗）
3. 局部RBF插值：在每个子域内，使用薄板样条（Thin Plate Spline）或高斯RBF拟合残差场
4. 局部矩阵求解：每个子域独立求解RBF系数，矩阵规模小，复杂度低
5. 子域拼接：使用加权平均（如距离加权）将各子域的残差插值结果平滑拼接
6. 校正：将RBF残差场叠加到粗解析仿射模型上，得到最终映射
7. 计算最终RMS误差

设计要点：
- 舍弃全局三阶SIP运算，避免大矩阵求逆（O(n³)复杂度）
- DDM将全局问题分解为多个局部小问题，每个子域矩阵规模可控
- 仅拟合残差（粗解析仿射模型已捕捉主要变换），RBF只需拟合局部畸变
- 子域间重叠确保拼接平滑，避免边界不连续

#### Scenario: 光学畸变校正
- **WHEN** 图像存在场曲/畸变，粗解析RMS=0.5px，残差呈空间相关分布
- **THEN** DDM-RBF精解析后RMS < 0.2px，残差场被有效拟合

#### Scenario: 无明显畸变
- **WHEN** 图像无明显畸变，粗解析RMS已 < 0.2px，残差呈随机分布
- **THEN** RBF拟合的残差场幅值极小，精解析结果与粗解析基本一致

#### Scenario: 大视场畸变
- **WHEN** 大视场图像边缘畸变显著，中心区域良好
- **THEN** DDM-RBF在边缘子域拟合较大残差，中心子域残差接近零

### Requirement: C API接口
系统应提供C API（extern "C"导出），核心接口包括：
- `psolve_create()`: 创建解析器实例，传入Gaia数据目录路径
- `psolve_destroy()`: 销毁实例
- `psolve_coarse()`: 执行粗解析，输入图像数据+元数据，输出变换模型+匹配星点
- `psolve_fine()`: 执行精解析，输入粗解析结果，输出WCS参数
- `psolve_get_matched_stars()`: 获取匹配星点的图像坐标
- `psolve_get_wcs()`: 获取WCS参数

### Requirement: Python绑定
系统应提供Python ctypes绑定，封装C API：
- `PlateSolver` 类：管理解析器生命周期
- `solve_coarse()` 方法：执行粗解析
- `solve_fine()` 方法：执行精解析
- 返回数据类：`SolveResult`（包含WCS参数、匹配星点、RMS误差等）

### Requirement: 日志系统
系统应实现与现有模块一致的日志系统：
- 日志文件路径: `lib/plate_solve/logs/plate_solve.log`
- 日志级别: DEBUG/INFO/WARN/ERROR
- 关键步骤输出: FOV计算、星等迭代、匹配结果、每次迭代的RMS误差

# 初始WCS生成模块重写 Spec

## Why
现有 `star_alignment` 模块代码混乱（~900行单函数），缺少四种翻转模式测试，三角匹配与RANSAC回退逻辑交织，迭代重投影参数硬编码。需要按"饱和星优先"策略从零重写初始WCS生成步骤，算法核心参考siril atpmatch/Valdes 1995，物理思路采用用户的饱和星优先策略。

## What Changes
- **新增** `initial_wcs` Python原型模块（5步算法流程）
- **新增** `initial_wcs` C++ DLL模块（Python验证后重写）
- **归档** 现有 `star_alignment` 模块到 `old/` 目录
- **保留** `iterative_refine`（Step 2）不动
- **保留** Gaia客户端、Gnomonic投影等基础设施

## Impact
- Affected specs: star-alignment-module, implement-plate-solve
- Affected code:
  - `lib/plate_solve/modules/star_alignment/` → 归档到 `old/`
  - `lib/plate_solve/python/plate_solve.py` → 更新Step1调用
  - `lib/plate_solve/DESIGN.md` → 更新设计文档

## ADDED Requirements

### Requirement: 像素尺度和FOV计算
系统 SHALL 根据焦距和像元尺寸计算像素尺度（arcsec/px）和对角线FOV（度）。

#### Scenario: 标准天文图像
- **WHEN** 输入 focal_length_mm=200, pixel_size_um=6.0, width=4500, height=3600
- **THEN** scale_arcsec_px ≈ 6.188, fov_diag_deg ≈ 9.77

### Requirement: Gaia锥形查询与极限星等二分法
系统 SHALL 通过二分法迭代确定极限星等，使Gaia查询星数在 [det_count, 1.2×det_count] 范围内。

#### Scenario: 银心区域密集星场
- **WHEN** 检测到14000颗星，FOV对角线9.77度
- **THEN** 二分法在6~22mag范围内收敛，目标星数约21000

#### Scenario: 稀疏星场
- **WHEN** 检测到500颗星
- **THEN** 二分法收敛到较亮的极限星等，目标星数约750

### Requirement: Gnomonic投影
系统 SHALL 将Gaia星表(RA,Dec)投影到以初始中心为切点的切面坐标(xi, eta)，并转换为像素坐标。

#### Scenario: 投影精度
- **WHEN** 输入Gaia星的RA/Dec和中心坐标
- **THEN** 投影坐标与siril的project_catalog_stars结果一致（误差<0.01 arcsec）

### Requirement: 饱和星优先三角匹配
系统 SHALL 按以下策略选择匹配星点：
- 若饱和星≥10颗：图像侧用全部饱和星，Gaia侧用1.2×饱和星数的最亮Gaia星
- 若饱和星<10颗：图像侧用饱和星+最亮正常星共100颗，Gaia侧用150颗最亮Gaia星

三角匹配算法参考siril的atpmatch（Valdes 1995），包含：
1. 三角形构建（ba, ca空间）
2. 投票矩阵匹配（radius=0.002）
3. 互为最佳去重
4. iter_trans迭代精化（35%分位数sigma, 10-sigma clip, 最多10次迭代）

#### Scenario: 饱和星充足
- **WHEN** 检测到163颗饱和星
- **THEN** 图像侧使用163颗饱和星，Gaia侧使用约196颗最亮星进行三角匹配

#### Scenario: 饱和星不足
- **WHEN** 检测到5颗饱和星
- **THEN** 图像侧使用5颗饱和星+95颗最亮正常星=100颗，Gaia侧使用150颗最亮星

### Requirement: 四种翻转模式独立匹配
系统 SHALL 对4种翻转模式（正常/X翻转/Y翻转/XY翻转）分别独立执行三角匹配+iter_trans，选择匹配数最多且RMS最小的模式。

#### Scenario: 正常模式最佳
- **WHEN** 图像无翻转
- **THEN** 正常模式匹配数最多，RMS最小

#### Scenario: Y翻转最佳
- **WHEN** 图像存在Y翻转（如某些相机输出）
- **THEN** Y翻转模式匹配数最多，RMS最小

### Requirement: 全星点验证匹配
系统 SHALL 在三角匹配获得初始仿射变换后，用全部星点进行验证匹配：
1. 用仿射变换预测所有星的位置
2. 近邻匹配（递减半径：50→30→10→2 arcsec）
3. 每轮iter_trans精化

#### Scenario: 验证匹配收敛
- **WHEN** 三角匹配获得7对初始匹配
- **THEN** 全星点验证后匹配数增长到>1000对

### Requirement: 迭代重投影收敛
系统 SHALL 在匹配完成后进行迭代重投影收敛（siril风格）：
- 收敛条件：offset < CONV_TOLERANCE (0.01 arcsec)
- 最大迭代：5次
- 每次迭代：从仿射平移量逆投影新中心 → 重新投影Gaia → 重新匹配

#### Scenario: 一次收敛
- **WHEN** 初始中心偏差<1度
- **THEN** 1-2次迭代即收敛

#### Scenario: 大偏差收敛
- **WHEN** 初始中心偏差>5度
- **THEN** 3-5次迭代收敛

### Requirement: Python原型先行验证
系统 SHALL 先用Python实现完整5步算法，验证逻辑正确性后再用C++重写性能关键部分。

#### Scenario: Python验证通过
- **WHEN** Python原型在测试数据上RMS<2px
- **THEN** 可以开始C++重写

### Requirement: 旧代码归档
系统 SHALL 将现有 `star_alignment` 模块归档到 `old/` 目录，保留代码供参考，不删除。

#### Scenario: 归档完整性
- **WHEN** 重写完成
- **THEN** 旧代码在 `lib/plate_solve/old/star_alignment/` 中可查阅，附有用途说明

## MODIFIED Requirements

### Requirement: PlateSolve统一API
PlateSolve的`solve()`方法 SHALL 调用新的`initial_wcs`模块替代旧的`star_alignment`模块，输出数据结构保持兼容（center_ra, center_dec, rotation_deg, scale_arcsec_px, flip_mode, affine）。

## REMOVED Requirements

### Requirement: 旧star_alignment模块
**Reason**: 代码混乱，逻辑交织，缺少翻转模式测试
**Migration**: 归档到 `old/star_alignment/`，新代码在 `initial_wcs/`

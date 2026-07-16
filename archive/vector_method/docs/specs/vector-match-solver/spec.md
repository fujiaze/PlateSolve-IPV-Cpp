# 向量匹配法Plate Solver Python原型 Spec

## Why
设计文档"向量匹配法设计文档.md"提出了一种全新的解析方式：利用亮星坐标构建以各自中心为原点的二维向量组，通过RANSAC求解两组向量间的相似变换（旋转、统一缩放、平移、镜像翻转），替代旧的三角形匹配算法。旧platesolve代码已归档到`plate_solve_old/`。需要先用Python验证算法正确性，再考虑C++重写。

## What Changes
- **新增** `vector_match.py`：基于向量组对齐的完整plate solving Python原型
- **彻底移除** 三角形匹配算法（set_triangle/build_triangles/triangle_match全部丢弃）
- **新增** 向量组构建（角秒空间，以各自中心为原点）
- **新增** RANSAC相似变换求解（2点采样，4参数：s,θ,tx,ty）
- **新增** 4种翻转模式独立RANSAC + 归一化打分选择
- **复用** GaiaClientPy、gnomonic投影、二分法查询等基础设施

## Impact
- Affected specs: initial-wcs-rewrite, redesign-initial-wcs-matching
- Affected code:
  - `lib/plate_solve/python/` → 新增 `vector_match.py`
  - `lib/plate_solve_old/` → 旧代码已归档，不受影响

## ADDED Requirements

### Requirement: 像素尺度和FOV计算
系统 SHALL 根据焦距和像元尺寸计算像素尺度（arcsec/px）和对角线FOV（度）。

公式：
- s0 = 206.265 × pixel_size_um / focal_length_mm（角秒/像素）
- FOV_diag = sqrt(width² + height²) × s0 / 3600（度）

#### Scenario: 标准天文图像
- **WHEN** 输入 focal_length_mm=200, pixel_size_um=6.0, width=4500, height=3600
- **THEN** scale_arcsec_px ≈ 6.188, fov_diag_deg ≈ 9.77

### Requirement: Gaia锥形查询与极限星等二分法
系统 SHALL 通过二分法迭代确定极限星等，使Gaia查询星数接近 N_gaia = ceil(1.5 × N_img)。

- 查询半径 = FOV_diag × 1.2 / 2
- 二分范围：6.0 ~ 22.0 mag
- 收敛条件：星数在 [N_gaia × 0.9, N_gaia × 1.1] 或迭代超过10次

#### Scenario: 银心区域密集星场
- **WHEN** 检测到14000颗星，FOV对角线9.77度
- **THEN** 二分法收敛，目标星数约21000

### Requirement: 向量组构建
系统 SHALL 分别构建图像向量组U和星表向量组W，均以角秒为单位，以各自中心为原点。

**图像向量组U**：
- 原点：图像几何中心 (width/2, height/2)
- 对每颗亮星i，像素偏移 p_i = (x_i - c_x, y_i - c_y)
- 角秒偏移 u_i = s0 × p_i

**星表向量组W**：
- 原点：FITS头中的参考天球坐标 (α₀, δ₀)
- 对每颗Gaia星j，用gnomonic投影计算 (x_proj, y_proj)（弧度）
- 转角秒：w_j = (180 × 3600 / π) × (x_proj, y_proj)

#### Scenario: 向量组尺度一致
- **WHEN** 两组向量都转换为角秒单位
- **THEN** 相同天区对应的向量长度近似相等（差异仅来自像素尺度误差s0）

### Requirement: 亮星选取策略
系统 SHALL 按以下规则选取匹配用的亮星：

- 若饱和星 n_sat ≥ 10：图像侧用全部饱和星，N_img = n_sat
- 若饱和星 n_sat < 10：图像侧用饱和星 + 最亮(100 - n_sat)颗正常星，N_img = 100
- 星表侧目标星数 N_gaia = ceil(1.5 × N_img)

#### Scenario: 饱和星充足
- **WHEN** 检测到163颗饱和星
- **THEN** N_img = 163, N_gaia = 245

#### Scenario: 饱和星不足
- **WHEN** 检测到5颗饱和星
- **THEN** N_img = 100, N_gaia = 150

### Requirement: RANSAC相似变换求解
系统 SHALL 使用RANSAC框架求解图像向量组U与星表向量组W'之间的相似变换。

**变换模型**：u = s × R × w' + t
- s：统一尺度因子（理想≈1.0，吸收s0偏差）
- R：2×2旋转矩阵（det > 0，翻转由W'预处理实现）
- t = (t_x, t_y)：中心平移向量（角秒）

**RANSAC参数**：
- 最大迭代次数：K = 200
- 内点距离阈值：τ = max(1.0, 2.5 × s0) 角秒
- 最少内点数：min_inliers = max(5, N_img × 0.3)
- 缩放因子允许范围：[0.9, 1.1]

**2点采样求解**：
1. 从U中随机选2点 u_a, u_b，从W'中随机选2点 w'_a, w'_b
2. s = ‖u_a - u_b‖ / ‖w'_a - w'_b‖，若s∉[0.9,1.1]则丢弃
3. θ = angle(u_a - u_b) - angle(w'_a - w'_b)
4. t = u_a - s × R(θ) × w'_a

**numpy向量化内点统计**：
- 一次性变换所有W'点：û_j = s × R × w'_j + t
- KDTree最近邻匹配（距离 < τ）
- 1对1互斥匹配

#### Scenario: 正确变换求解
- **WHEN** U和W'有>30%的对应点
- **THEN** RANSAC在200次迭代内找到正确变换，内点数 > min_inliers

#### Scenario: 缩放因子过滤
- **WHEN** 随机2点对产生的s = 0.5
- **THEN** 该假设被丢弃（s∉[0.9,1.1]）

### Requirement: 4种翻转模式独立RANSAC
系统 SHALL 对4种星表向量预处理分别运行RANSAC：

| 模式 | x轴处理 | y轴处理 |
|------|---------|---------|
| 0    | 原样    | 原样    |
| 1    | 取反    | 原样    |
| 2    | 原样    | 取反    |
| 3    | 取反    | 取反    |

每种模式独立求解相似变换，记录最佳变换和归一化得分。

#### Scenario: 正常模式最佳
- **WHEN** 图像无翻转
- **THEN** 模式0归一化得分最高

#### Scenario: Y翻转最佳
- **WHEN** 图像存在Y翻转
- **THEN** 模式2归一化得分最高

### Requirement: 打分函数与模式选择
系统 SHALL 使用以下打分函数评估和比较变换质量：

**单模式内打分**：score = n_inliers - λ × RMS（λ = 1.0）

**模式间归一化打分**：
score_norm = n_inliers / min(N_img, M) × (1 - RMS / τ)

选择归一化得分最高的模式。若最高得分 < 0.3，判定匹配失败。

#### Scenario: 模式选择
- **WHEN** 4种模式分别得到归一化得分 [0.75, 0.12, 0.08, 0.15]
- **THEN** 选择模式0（得分0.75最高）

#### Scenario: 匹配失败
- **WHEN** 4种模式最高归一化得分为0.2
- **THEN** 判定匹配失败，返回失败状态

### Requirement: WCS参数提取
系统 SHALL 从最佳相似变换提取天文标定参数：

**中心坐标修正**：
- Δα = t_x / (cos(δ₀) × 3600)（度）
- Δδ = t_y / 3600（度）
- α_c = α₀ + Δα, δ_c = δ₀ + Δδ

**旋转角**：θ = atan2(c, a)（从R矩阵元素提取）

**最终像素尺度**：s_final = s0 × s

**翻转信息**：根据所选模式记录翻转状态

#### Scenario: 参数提取
- **WHEN** 最佳变换给出 t=(0.15, 0.10)角秒, θ=0.018°, s=1.001
- **THEN** 中心偏移约0.0024°, 最终像素尺度 ≈ s0 × 1.001

### Requirement: 迭代重投影收敛
系统 SHALL 在匹配完成后进行迭代重投影收敛（siril风格）：
- 收敛条件：offset < 0.01 arcsec
- 最大迭代：5次
- 每次迭代：从平移量逆投影新中心 → 重新投影Gaia → 重新匹配

#### Scenario: 一次收敛
- **WHEN** 初始中心偏差<1度
- **THEN** 1-2次迭代即收敛

### Requirement: 完整端到端流程
系统 SHALL 提供完整的端到端plate solving流程：

输入：FITS图像（通过astro_image_io读取）→ 星点检测（通过star_detector）→ Gaia查询（通过gaia_client）→ 向量匹配 → WCS输出

输出：center_ra, center_dec, rotation_deg, scale_arcsec_px, flip_mode, matched_count, rms_arcsec, affine

#### Scenario: 端到端测试
- **WHEN** 输入testdata中的测试FITS图像
- **THEN** 成功解析，RMS < 2px

## MODIFIED Requirements

### Requirement: PlateSolve集成
PlateSolve的`solve()`方法 SHALL 支持调用新的`vector_match`模块作为Step1替代，输出数据结构保持兼容。

## REMOVED Requirements

### Requirement: 三角形匹配算法
**Reason**: 彻底丢弃三角形匹配，用向量组RANSAC替代
**Migration**: 新代码在vector_match.py中，不包含任何三角形匹配逻辑

### Requirement: 旧star_alignment/initial_wcs模块
**Reason**: 旧代码已归档到plate_solve_old/
**Migration**: 新代码在vector_match.py中

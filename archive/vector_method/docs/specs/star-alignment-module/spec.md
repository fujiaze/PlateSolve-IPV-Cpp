# Star Alignment 模块 Spec

## Why

plate_solve 的后续求解器（畸变模型、残差拟合、RMS评估）都需要**准确的中心坐标和旋转角**作为前提。FITS头WCS虽然存在但不够精确，需要专门的模块通过图像星点与Gaia星表的匹配来确定精确的偏移量、旋转角和翻转情况。

## What Changes

- **新增** `star_alignment` 独立DLL模块（`lib/plate_solve/modules/star_alignment/`）
  - 纯匹配模块：接收两个**同尺度**的2D点集（图像星点 + Gaia像素空间星点），不负责Gaia查询/投影
  - 四种翻转预设下做中心平移+旋转匹配
  - 输出修正量：偏移量(Δx, Δy)、旋转角、翻转模式
- **增强** `rms_calc` 模块：新增 `psm_rms_evaluate_model()` 函数，接收完整模型参数，调用Gaia DR3计算裁剪后RMS并输出评分

## Impact

- 新增目录: `lib/plate_solve/modules/star_alignment/`
- 新增数据结构: `PSMStarAlignmentInput`, `PSMStarAlignmentResult`, `PSMFlipMode`
- 修改文件: `lib/plate_solve/modules/rms_calc/psm_rms_calc.h/cpp`（新增evaluate函数）
- 依赖: feature_match（三角形匹配）, rms_calc（sigma-clip）
- **不改变**现有模块接口

## 架构关系图

```
┌──────────────────────────────────────────────────────────┐
│                    Python 调度层                          │
│                                                          │
│  1. 读取FITS → WCS(不精确但存在) → 中心RA/Dec, scale    │
│  2. star_detector → 图像星点坐标                         │
│  3. Gaia DR3 查询(1.2×FOV) + 二分法迭代极限星等         │
│  4. Gnomonic投影 + 焦距/像元比例尺 → Gaia像素坐标       │
│     (尺度与图像像素同级，中心对齐WCS中心)                │
│                                                          │
│  输入A: 图像星点 (x, y)  ← 以图像中心为原点             │
│  输入B: Gaia星点 (x, y)  ← 以同一中心投影，同尺度       │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────┐
│            ★ star_alignment (新增DLL) ★                  │
│                                                          │
│  输入: 两个同尺度2D点集                                  │
│                                                          │
│  对每种翻转预设 (NONE → X → Y → XY):                    │
│    1. 翻转图像星点坐标                                   │
│    2. 三角形特征匹配 → 初始对应                          │
│    3. 中心平移+旋转 拟合 (仿射6参数)                     │
│    4. KDTree迭代精化                                     │
│    5. 评分                                               │
│                                                          │
│  输出:                                                   │
│    · 偏移量 (offset_x, offset_y) pixels                  │
│    · 旋转角 (degrees)                                    │
│    · 翻转模式                                            │
│    · 匹配质量 (matched_count, rms, mean_dist)            │
│                                                          │
└──────────────┬───────────────────────────────────────────┘
               │ 修正量
               ▼
┌──────────────────────────────────────────────────────────┐
│  Python层: 用偏移量+旋转角+翻转 更新精确WCS             │
│  offset → 逆投影 → ΔRA, ΔDec                            │
│  rotation → CD矩阵旋转                                   │
│  flip → CD矩阵符号                                       │
└──────────────┬───────────────────────────────────────────┘
               │ 精确WCS
               ▼
┌──────────────────────────────────────────────────────────┐
│               后续求解流程 (已有模块)                     │
│                                                          │
│  coarse_affine → affine_distortion → iterative(DDM-RBF)  │
│                       ↓                                  │
│              rms_calc (增强: evaluate_model)              │
└──────────────────────────────────────────────────────────┘
```

## ADDED Requirements

### Requirement: Star Alignment 纯匹配模块

系统应提供纯2D点集匹配模块，接收两个同尺度坐标系中的星点，在四种翻转预设下做中心平移+旋转匹配。

#### 前提假设
- FITS必定有WCS（不精确但存在），提供初始中心RA/Dec和比例尺
- Gaia星点已由外部投影到与图像像素同尺度的坐标系中
- 两个坐标系的中心点相同（均为WCS中心），尺度相同（均为arcsec/px换算后的像素单位）
- 唯一未知的变换是：中心偏移 + 旋转 + 可能的翻转

#### 输入数据结构
```c
typedef struct {
    // 图像星点（以图像中心为原点的坐标）
    const double *img_x;
    const double *img_y;
    int img_count;

    // Gaia星点（以WCS中心投影到像素空间，同尺度）
    const double *cat_x;
    const double *cat_y;
    const double *cat_mag;     // 星等（用于亮度排序筛选）
    int cat_count;

    // 匹配参数
    int n_bright;              // 构建三角形用的最亮星数 (默认100)
    double max_dist_px;        // 最大匹配距离 (默认25px)
    int max_iterations;        // 仿射迭代次数 (默认5)
    double match_threshold;    // 匹配成功阈值: 最少匹配对数 (默认30)
} PSMStarAlignmentInput;
```

#### 匹配流程
对每种翻转模式（顺序：NONE → X → Y → XY）：

1. **翻转图像星点**：按当前模式翻转 img_x/img_y
   - NONE: (x, y)
   - X: (-x, y)
   - Y: (x, -y)
   - XY: (-x, -y)

2. **三角形特征匹配**（复用 feature_match 逻辑）：
   - 取图像侧最亮 n_bright 颗星
   - 取Gaia侧最亮 n_bright 颗星（按cat_mag升序）
   - 构建三角形，在 (ba, ca) 空间匹配
   - 若匹配三角形数 < 阈值 → 跳过此模式

3. **仿射拟合**（中心平移+旋转+微缩放）：
   - 初始估计：质心对齐 + 单位尺度
   - KDTree最近邻搜索（max_dist_px阈值内）
   - 最小二乘6参数仿射拟合
   - 迭代精化（max_iterations次）

4. **评分**：`score = matched_count / (1 + mean_dist / 10)`

5. **提前终止**：若 matched_count ≥ 50 且 mean_dist ≤ 5px → 停止尝试

#### 输出数据结构
```c
typedef enum {
    PSM_FLIP_NONE = 0,
    PSM_FLIP_X = 1,
    PSM_FLIP_Y = 2,
    PSM_FLIP_XY = 3
} PSMFlipMode;

typedef struct {
    // 修正量（核心输出）
    double offset_x;           // X方向偏移量 (pixels), 正值=图像中心偏右
    double offset_y;           // Y方向偏移量 (pixels), 正值=图像中心偏下
    double rotation_deg;       // 旋转角 (degrees), 逆时针为正
    PSMFlipMode flip_mode;     // 翻转模式

    // 仿射变换完整参数（图像→Gaia像素空间）
    double a0, a1, a2;         // x' = a0 + a1*x + a2*y
    double b0, b1, b2;         // y' = b0 + b1*x + b2*y

    // 匹配质量
    int matched_count;         // 匹配星点对数
    double rms_px;             // RMS误差 (pixels)
    double mean_dist_px;       // 平均距离 (pixels)

    // 匹配详情（用于可视化和后续验证）
    int *img_indices;          // 图像星点索引 [matched_count]
    int *cat_indices;          // Gaia星点索引 [matched_count]
} PSMStarAlignmentResult;
```

#### 偏移量与旋转角的提取
- **偏移量**：仿射变换的平移项 (a0, b0) 即为偏移量
- **旋转角**：`rotation = atan2(a2, a1)` （从仿射矩阵的旋转分量提取）
- **尺度因子**：`scale = sqrt(a1² + a2²)` （应接近1.0，因为输入已同尺度）

#### C API
```c
PSM_EXPORT int psm_star_align(
    const PSMStarAlignmentInput *input,
    PSMStarAlignmentResult *out_result);

PSM_EXPORT void psm_free_star_alignment_result(PSMStarAlignmentResult *result);
```

#### Scenario: 正常模式成功
- **GIVEN** 图像方向与Gaia投影一致，WCS中心偏移约20px，旋转偏差约0.3°
- **WHEN** 调用 psm_star_align()
- **THEN** 返回 PSM_OK, flip_mode=PSM_FLIP_NONE, offset≈(20, 5)px, rotation≈0.3°, matched_count≥50

#### Scenario: 需要Y翻转
- **GIVEN** 相机Y轴方向与FITS标准相反
- **WHEN** NONE模式匹配失败（<30对），Y翻转模式匹配成功
- **THEN** 返回 PSM_OK, flip_mode=PSM_FLIP_Y

#### Scenario: 所有模式失败
- **WHEN** 四种模式均无法获得 ≥ match_threshold 对匹配
- **THEN** 返回 PSM_ERR_NO_MATCH, matched_count=0

### Requirement: WCS修正（Python层）
系统应在Python层提供WCS修正函数，将star_alignment输出的修正量转换为精确WCS：

#### 修正流程
1. **偏移量→天球坐标修正**：
   - offset_x, offset_y (pixels) → 逆投影 → ΔRA, ΔDec
   - `ΔRA = offset_x * scale / 3600 / cos(Dec)`
   - `ΔDec = offset_y * scale / 3600`
   - `RA_new = RA_old + ΔRA`, `Dec_new = Dec_old + ΔDec`

2. **旋转角→CD矩阵修正**：
   - 构建2×2旋转矩阵 R(θ)
   - `CD_new = R(θ) × CD_old`

3. **翻转→CD矩阵修正**：
   - X翻转: CD[0,:] *= -1
   - Y翻转: CD[1,:] *= -1
   - XY翻转: CD *= -1

#### Scenario: WCS修正
- **GIVEN** WCS中心RA=266.416°, Dec=-29.000°, offset=(20, 5)px, scale=6.294"/px, rotation=0.3°
- **WHEN** 执行WCS修正
- **THEN** RA_new ≈ 266.416 + 20×6.294/3600/cos(-29°) ≈ 266.421°, Dec_new ≈ -29.000 + 5×6.294/3600 ≈ -28.991°

### Requirement: RMS模型评估器（增强rms_calc）
系统应提供独立函数，用于评估任意给定模型的质量：

#### 功能
输入一组完整的模型参数，调用Gaia DR3查询参考星，计算裁剪后的RMS误差并输出综合评分。

#### 输入参数
```c
typedef struct {
    double ra_center, dec_center;
    double scale_arcsec_px;
    double rotation_deg;
    const PSMAffine *affine;          // 可选, NULL则从上面构建
    const PSMDistortion *distortion;  // 可选, NULL则跳过畸变校正
    double query_radius_deg;
    double mag_limit;
    double sigma_clip;                // 默认3.0
} PSMModelEvalInput;

typedef struct {
    int total_gaia_stars;
    int matched_count;
    int clipped_count;
    double rms_x_px;
    double rms_y_px;
    double rms_total_px;
    double rms_arcsec;
    double mean_residual_px;
    double median_residual_px;
    double mad_px;
    double score;
    double score_density;
    double score_precision;
    double score_coverage;
} PSMModelEvalResult;
```

#### 评分公式
```
score = 40 * density_score + 40 * precision_score + 20 * coverage_score

density_score = min(matched_count / expected_count, 1.0)
precision_score = exp(-rms_total_px / 10.0)
coverage_score = 1.0 - |clipped_ratio - 0.95|
```

#### C API
```c
PSM_EXPORT int psm_rms_evaluate_model(
    void *gaia_client,
    const PSMModelEvalInput *model,
    const double *img_x, const double *img_y, int img_count,
    PSMModelEvalResult *out_result);

PSM_EXPORT void psm_free_model_eval_result(PSMModelEvalResult *result);
```

#### Scenario: 评估良好模型
- **GIVEN** 准确的中心+旋转+仿射参数
- **WHEN** 调用 psm_rms_evaluate_model()
- **THEN** matched_count ≈ det_count × 0.7+, rms < 5px, score > 70

#### Scenario: 评估偏差模型
- **GIVEN** 中心偏离 5 arcmin
- **WHEN** 调用评估
- **THEN** matched_count 显著下降, rms 增大, score < 30

## MODIFIED Requirements

### Requirement: rms_calc 模块扩展
**Before**: rms_calc 仅支持 affine 和 distortion 两种模型的 RMS 计算。

**After**: 新增 `psm_rms_evaluate_model()` 函数，支持：
- 接收完整模型描述（中心坐标、旋转角、仿射、畸变）
- 内部自动调用 Gaia DR3 进行锥形查询
- 自动执行 sigma-clip 异常点过滤
- 输出综合评分

**Migration**: 原有函数签名不变，纯增量添加。

## REMOVED Requirements
无。

# 迭代畸变精化 Spec

## Why
当前粗解析（coarse solve）通过过曝星点匹配获得了中心坐标、旋转角度和翻转角度的近似解，但精度有限（RMS ~7px）。需要一个迭代精化步骤，利用近似解从Gaia查询精确参考星，通过近距离多边形匹配拟合六参数仿射模型和六参数畸变模型，逐步精化比例尺、中心坐标和旋转角度，将RMS降至亚像素级别，为后续DDM-RBF精解析提供高质量初始条件。

## What Changes
- **新增** `iterative_refine` 独立DLL模块（`lib/plate_solve/modules/iterative_refine/`）
- **新增** 六参数畸变模型数据结构 `PSMDistortion`（纯二次项，不含仿射）
- **新增** 迭代精化主流程：Gaia矩形区域查询 → 极限星等迭代 → 像素映射 → 近距离匹配 → 仿射+畸变拟合 → 参数精化
- **新增** Python调度层：矩形FOV计算、Gaia查询、投影映射、WCS更新
- **修改** `plate_solve.h`：新增 `PSOLVE_ERR_ITER_FAILED` 错误码

## Impact
- Affected specs: implement-plate-solve（粗解析后新增精化步骤）
- Affected code:
  - `lib/plate_solve/include/plate_solve.h` — 新增错误码
  - `lib/plate_solve/modules/iterative_refine/` — 全新模块
  - `lib/plate_solve/python/plate_solve.py` — 新增 `solve_iterative_refine()` 方法
- 依赖: gaia_xpsd_client（锥形查询）, psolve_projection（Gnomonic投影）, star_alignment（匹配逻辑参考）
- **不改变**现有粗解析和精解析接口

## ADDED Requirements

### Requirement: 六参数仿射模型
系统应使用六参数仿射模型描述像素坐标的线性变换：

```c
typedef struct {
    double a0, a1, a2;    // x' = a0 + a1*x + a2*y
    double b0, b1, b2;    // y' = b0 + b1*x + b2*y
} PSMAffine;
```

仿射模型公式：
- x' = a0 + a1*x + a2*y
- y' = b0 + b1*x + b2*y

参数物理意义：
- a0, b0：平移量（中心偏移）
- a1, b1, a2, b2：线性变换矩阵（包含旋转、缩放、剪切）

### Requirement: 六参数畸变模型
系统应使用六参数畸变模型描述像素坐标的非线性畸变（纯二次项，不含仿射）：

```c
typedef struct {
    double dx1, dx2, dx3;    // Δx = dx1*x² + dx2*xy + dx3*y²
    double dy1, dy2, dy3;    // Δy = dy1*x² + dy2*xy + dy3*y²
} PSMDistortion;
```

畸变模型公式：
- Δx = dx1*x² + dx2*x*y + dx3*y²
- Δy = dy1*x² + dy2*x*y + dy3*y²

#### 完整变换公式
仿射 + 畸变：
- x' = a0 + a1*x + a2*y + dx1*x² + dx2*x*y + dx3*y²
- y' = b0 + b1*x + b2*y + dy1*x² + dy2*x*y + dy3*y²

#### Scenario: 无畸变情况
- **WHEN** 图像无光学畸变
- **THEN** dx1≈dx2≈dx3≈0, dy1≈dy2≈dy3≈0，模型退化为纯仿射变换

#### Scenario: 桶形畸变
- **WHEN** 图像存在桶形畸变（边缘星点向内偏移）
- **THEN** dx1, dx3, dy1, dy3 显著非零，正确描述径向偏移

### Requirement: Gaia矩形区域查询
系统应支持基于当前FOV的矩形区域Gaia查询，当Gaia仅支持锥形查询时，使用对角线FOV作为查询半径，然后丢弃矩形以外的星点。

#### 查询流程
1. 根据当前中心坐标(RA, Dec)、旋转角θ、FOV宽高(W, H)计算四个角点的天球坐标
2. 计算对角线半径：r_diag = sqrt(W² + H²) / 2
3. 使用 `gaia_client_cone_search_for_solver()` 查询半径 r_diag + margin(5°) 的区域
4. 将查询结果投影到切平面，按旋转角θ旋转后，丢弃超出矩形范围的星点
5. 矩形判定：|x_rotated| ≤ W/2 且 |y_rotated| ≤ H/2

#### Scenario: 正常矩形查询
- **GIVEN** 中心RA=266.4°, Dec=-29.0°, FOV=7.7°×6.2°, 旋转角=0°
- **WHEN** 执行矩形区域查询
- **THEN** 对角线半径≈4.9°，查询半径≈9.9°，丢弃矩形外星点后剩余星点覆盖完整FOV

#### Scenario: 大旋转角
- **GIVEN** 旋转角=45°
- **WHEN** 执行矩形区域查询
- **THEN** 旋转后的矩形判定正确，不因旋转角导致星点遗漏

### Requirement: 极限星等迭代
系统应迭代调整极限星等，使Gaia查询返回的星点数量与star_detector检测的星点数量近似匹配。

#### 迭代算法
1. 目标星数：target = detected_count × 1.5（略多于检测数，确保匹配覆盖）
2. 初始范围：mag_low = 6.0, mag_high = 22.0
3. 二分法迭代（最多10次）：
   - mag_mid = (mag_low + mag_high) / 2
   - 查询Gaia，得到star_count
   - 若 star_count > target × 1.2 → mag_high = mag_mid
   - 若 star_count < target × 0.8 → mag_low = mag_mid
   - 否则 → 收敛
4. 容差：|star_count - target| / target < 0.2

#### Scenario: 星数匹配成功
- **WHEN** detected_count = 5000, target = 7500
- **THEN** 迭代后Gaia返回6000~9000颗星

#### Scenario: 银心区域星数过多
- **WHEN** 银心区域即使mag_limit=8仍有>10000颗星
- **THEN** 使用mag_limit=8的结果，记录警告日志

### Requirement: Gaia星点像素映射
系统应将Gaia星点从天球坐标映射到像素坐标系，使用当前的中心坐标、旋转角、翻转和比例尺。

#### 映射流程
1. Gnomonic投影：将(RA, Dec)投影到以(center_RA, center_Dec)为中心的切平面(x_deg, y_deg)
2. 角度转像素：x_px = x_deg × 3600 / scale_arcsec_px, y_px = y_deg × 3600 / scale_arcsec_px
3. 应用旋转：根据旋转角θ旋转坐标
4. 应用翻转：根据flip_mode翻转坐标
5. 平移到图像中心：x_img = x_rotated + width/2, y_img = y_rotated + height/2

#### Scenario: 近似解映射精度
- **WHEN** 粗解析中心偏差~50px，旋转偏差~0.5°
- **THEN** 映射后Gaia星点与图像星点偏差在~50px范围内，足以进行近距离匹配

### Requirement: 近距离多边形匹配
系统应在像素坐标系中进行近距离匹配，利用初始解较为精确的前提，将搜索半径限制在10px内减小计算量。

#### 匹配算法
1. 构建图像星点的KDTree（或Grid哈希）
2. 对每个Gaia映射星点，在KDTree中搜索最近邻
3. 距离阈值：max_dist = 10px（初始迭代），逐步收紧至5px
4. 双向一致性检查：img→cat和cat→img互为最近邻
5. Sigma-clip过滤：3σ剔除离群匹配对

#### Scenario: 初始解精确时匹配
- **WHEN** 粗解析RMS ~7px，映射后偏差~10px
- **THEN** 10px搜索半径内匹配率>60%，获得充足匹配对

#### Scenario: 匹配对不足
- **WHEN** 匹配对数 < 30
- **THEN** 逐步增大搜索半径至20px，若仍不足返回错误

### Requirement: 仿射+畸变拟合
系统应从匹配星点对拟合仿射模型（6参数）和畸变模型（6参数），共12参数。

#### 拟合算法
1. 将匹配对坐标转换为以图像中心为原点的坐标
2. 构建线性方程组：
   - 对每对匹配 (x_img, y_img) → (x_gaia, y_gaia)：
   - x_gaia = a0 + a1*x + a2*y + dx1*x² + dx2*x*y + dx3*y²
   - y_gaia = b0 + b1*x + b2*y + dy1*x² + dy2*x*y + dy3*y²
3. 最小二乘求解12个参数
4. 计算拟合残差RMS

#### Scenario: 正常拟合
- **WHEN** 匹配对数≥100
- **THEN** 拟合RMS < 2px

#### Scenario: 匹配对数不足
- **WHEN** 匹配对数 < 6
- **THEN** 返回错误 PSOLVE_ERR_NOT_ENOUGH

#### Scenario: 无明显畸变
- **WHEN** 图像无明显光学畸变
- **THEN** 畸变参数接近零（|dx1|,|dx2|,|dx3|,|dy1|,|dy2|,|dy3| < 1e-6）

### Requirement: 迭代参数精化
系统应在仿射+畸变拟合后迭代精化比例尺、中心坐标和旋转角度。

#### 迭代流程
1. 用当前仿射+畸变模型将Gaia星点映射到像素空间
2. 近距离匹配（搜索半径逐步收紧）
3. 拟合新的仿射+畸变模型
4. 从仿射参数提取更新的中心坐标、旋转角、比例尺：
   - 比例尺：scale = sqrt(a1² + a2²) × original_scale
   - 旋转角：rotation = atan2(a2, a1)
   - 中心偏移：offset = (a0, b0)，逆投影到天球坐标更新中心
5. 收敛判断：RMS变化 < 1% 或达到最大迭代次数（5次）

#### Scenario: 迭代收敛
- **WHEN** 初始RMS=7px，经过3次迭代RMS=0.8px，第4次RMS=0.79px
- **THEN** 停止迭代，返回精化后的模型参数

#### Scenario: 迭代不收敛
- **WHEN** 5次迭代后RMS仍在变化
- **THEN** 使用RMS最小的迭代结果，记录警告

### Requirement: C API接口
系统应提供C API，核心接口包括：

```c
PSM_EXPORT int psm_iterative_refine(
    const double *img_x, const double *img_y, int img_count,
    const double *cat_ra, const double *cat_dec, const float *cat_mag, int cat_count,
    double center_ra, double center_dec,
    double scale_arcsec_px, double rotation_deg, PSMFlipMode flip_mode,
    int img_width, int img_height,
    PSMIterRefineResult *out_result);

PSM_EXPORT void psm_free_iter_refine_result(PSMIterRefineResult *result);
```

#### 输出数据结构
```c
typedef struct {
    double center_ra;           // 精化后的中心RA
    double center_dec;          // 精化后的中心Dec
    double scale_arcsec_px;     // 精化后的比例尺
    double rotation_deg;        // 精化后的旋转角
    PSMFlipMode flip_mode;      // 翻转模式
    PSMAffine affine;           // 六参数仿射模型
    PSMDistortion distortion;   // 六参数畸变模型
    int matched_count;          // 匹配星点对数
    double rms_px;              // 最终RMS（像素）
    double rms_arcsec;          // 最终RMS（角秒）
    int iteration_count;        // 迭代次数
    int *img_indices;           // 图像星点匹配索引
    int *cat_indices;           // Gaia星点匹配索引
} PSMIterRefineResult;
```

#### Scenario: 正常精化
- **GIVEN** 粗解析结果：center偏差~50px, rotation偏差~0.5°, RMS~7px
- **WHEN** 调用 psm_iterative_refine()
- **THEN** 返回精化结果：RMS < 1px, matched_count > 500, iteration_count ≤ 5

### Requirement: Python绑定
系统应在Python层提供完整的调度逻辑：

#### 调度流程
1. 从粗解析结果提取：center_RA/Dec, scale, rotation, flip
2. 计算旋转后的FOV矩形四个角点天球坐标
3. 调用gaia_client锥形查询（对角线FOV + margin）
4. 矩形裁剪：投影到切平面，旋转后丢弃矩形外星点
5. 极限星等迭代：二分法使Gaia星数≈检测星数×1.5
6. 调用 iterative_refine DLL 进行迭代精化
7. 用精化结果更新WCS参数

#### Scenario: 完整Python流程
- **WHEN** 输入图像 + 粗解析结果
- **THEN** 自动完成Gaia查询→矩形裁剪→星等迭代→DLL精化→WCS更新

### Requirement: 日志系统
系统应实现与现有模块一致的日志系统：
- 日志路径: `lib/plate_solve/modules/iterative_refine/logs/iterative_refine.log`
- 关键步骤输出: Gaia查询星数、矩形裁剪后星数、每次迭代的匹配数和RMS、最终模型参数（仿射6参数+畸变6参数）

## MODIFIED Requirements

### Requirement: plate_solve.h 错误码扩展
**Before**: 错误码到 PSOLVE_ERR_INTERNAL = 8

**After**: 新增 PSOLVE_ERR_ITER_FAILED = 9（迭代精化失败）

**Migration**: 纯增量，不影响现有错误码。

## REMOVED Requirements
无。

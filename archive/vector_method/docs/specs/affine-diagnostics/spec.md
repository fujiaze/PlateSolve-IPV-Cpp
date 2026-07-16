# Affine Diagnostics Spec

## Why
仿射解析完成后需要可视化诊断：距离vs残差分布、径向星点密度。当前：
- 三角匹配用PSOLVE_GAIA_HARD_LIMIT=2000，匹配用此子集，**此限制保留不变**
- RMS只算了前2000颗Gaia星，且matched_stars=NULL
- 无诊断图输出

## What Changes
- **C代码 (psolve_coarse.cpp)**:
  - 三角匹配2000硬截断**保持不变**
  - 仿射确定后：在内存中对已加载的全部Gaia星按星等迭代→选出约det_count颗最亮星→投影+仿射变换→最近邻搜索→填入matched_stars
  - 用空间网格加速NN（~50px格网）
- **Python脚本 (test_coarse_affine.py)**:
  - 从result.matched_stars提取匹配对数据
  - **全部在Python侧用numpy向量化计算**：距离、残差、分bin
  - 图1：距离vs残差散点图+RMS线
  - 图2：径向直方图
  - 控制台径向分bin RMS表

## Impact
- Affected specs: implement-plate-solve
- Affected code: `lib/plate_solve/src/psolve_coarse.cpp`, `test_coarse_affine.py`

## ADDED Requirements

### Requirement: 内存内星等迭代选星
仿射确定后，系统SHALL在已加载的全部Gaia星中按G星等排序，迭代选择星等上限使选出的Gaia星数≈det_count（检测星数），用于RMS计算。

#### Scenario: 6918颗Gaia星，3148颗检测星
- **WHEN** Gaia已全部在内存中，仿射模型已确定
- **THEN** 按星等排序后选出mag最小的~3148颗Gaia星（或最近似数量），投影→仿射→NN匹配→填matched_stars

### Requirement: 空间网格加速NN搜索
系统SHALL将检测星按~50px格网分bin，每个Gaia星变换后只在其所在格+8邻格内搜索最近邻。

#### Scenario: 3000 Gaia × 3000 检测星
- **WHEN** 未优化时为900万次距离计算
- **THEN** 网格加速后约15万次计算

### Requirement: matched_stars输出
系统SHALL在PSolveCoarseResult.matched_stars中填入匹配对：
- img_x, img_y: 检测星坐标（图像中心系，px）
- cat_ra, cat_dec: Gaia天球坐标（度）
- cat_mag: Gaia G星等
- residual_x, residual_y: 仿射残差（px）

### Requirement: Python向量化诊断图
Python脚本SHALL用numpy向量化操作：
1. 从matched_stars构建numpy数组
2. 图1：距离=sqrt(img_x²+img_y²) vs 残差=sqrt(res_x²+res_y²) 散点+RMS线
3. 图2：距离分40bin，柱状图（每bin星点数）
4. 控制台：每bin星数+RMS+中位残差表

## MODIFIED Requirements
无。三角匹配2000硬截断不变。

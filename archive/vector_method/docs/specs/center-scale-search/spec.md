# 特征匹配模块 Spec

## Why
当前coarse_affine模块的三角匹配存在以下问题：
1. 未利用星点flux（亮度）信息进行特征增强
2. 等边/等腰三角形未被有效剔除，产生大量噪声投票
3. 星点数量控制不合理（图像侧和Gaia侧不平衡）

需要构建一个独立的特征匹配模块，参考siril的atpmatch算法，实现：
- 基于flux排序的星点筛选（图像≤2000，Gaia≤5000）
- 三角形特征构建与剪枝（ba>0.9剔除等边三角形）
- siril风格投票机制

## What Changes
- **新增** `feature_match` 独立DLL模块
- **新增** 带flux的星点数据结构 `PSMFeatureStar`
- **新增** flux降序排序和top-N筛选函数
- **新增** 三角形特征构建函数（ba, ca, side_a_angle, a_length）
- **新增** 等边三角形剪枝（ba > 0.9, a_length < 5px）
- **新增** siril风格投票矩阵和top_vote_getters
- **新增** flux辅助投票增强（可选）

## Impact
- 新增目录: `lib/plate_solve/modules/feature_match/`
- 新增数据结构: `PSMFeatureStar`, `PSMFeatureTriangle`, `PSMMatchResult`
- 依赖: `common/psm_common.h` 现有数据类型
- **不**影响现有coarse_affine/affine_distortion/iterative/rms_calc模块

## ADDED Requirements

### Requirement: 特征星点数据结构
The system SHALL define a `PSMFeatureStar` structure that includes:
```c
typedef struct {
    double x, y;    // 坐标 (图像坐标或切平面投影坐标)
    double flux;    // 亮度/流量值 (图像: ADU; Gaia: 1/mag或flux)
    int orig_idx;   // 原始列表中的索引
} PSMFeatureStar;
```

### Requirement: 三角形特征数据结构
The system SHALL define a `PSMFeatureTriangle` structure:
```c
typedef struct {
    int a_idx, b_idx, c_idx;  // 顶点索引 (a最长边对顶点)
    double ba_ratio;          // b/a (中等边/最长边)
    double ca_ratio;          // c/a (最短边/最长边)
    double side_a_angle;      // 最长边的角度 (用于旋转校验)
    double side_a_length;     // 最长边长度 (用于尺度校验)
} PSMFeatureTriangle;
```

### Requirement: 基于flux的星点筛选
The system SHALL sort stars by flux (descending) and select:
- Image stars: ≤2000 brightest (或实际检测数，取较小值)
- Gaia stars: ≤5000 brightest (或实际查询数，取较小值)

#### Scenario: Flux排序筛选
- **WHEN** given a list of detected stars with flux values
- **THEN** sort by flux descending
- **AND** keep top N (N = min(actual_count, max_count))
- **AND** preserve original indices in `orig_idx` field

### Requirement: 三角形特征构建
The system SHALL build triangles from the brightest stars following siril's method:

#### Scenario: 构建三角形
- **WHEN** given N stars
- **THEN** compute distance matrix (NxN)
- **THEN** enumerate all C(N,3) triangles
- **THEN** for each triangle:
  - Sort edges as longest(a) >= medium(b) >= shortest(c)
  - Compute features: ba=b/a, ca=c/a, side_a_angle, a_length
  - Identify vertex indices: a_idx (opposite to longest side), b_idx, c_idx
- **THEN** sort triangles by ba_ratio ascending (for binary search)

### Requirement: 等边三角形剪枝
The system SHALL prune feature-degenerate triangles (参考siril AT_MATCH_RATIO=0.9):
- ba > 0.9 → discard (near-equilateral, 特征不明显)
- a_length < 5 pixels → discard (too small, noise-prone)

#### Scenario: 剪枝效果
- **WHEN** building triangles from 60 stars
- **THEN** theoretical max = C(60,3) = 34220 triangles
- **AND** after pruning, expect ~25000-30000 valid triangles

### Requirement: 三角匹配和投票矩阵（参考siril make_vote_matrix）
The system SHALL implement siril-style triangle matching:

#### Scenario: 三角匹配
- **WHEN** matching triangles from image and Gaia
- **THEN** for each Gaia triangle:
  - Use binary search to find candidate image triangles with ba in range [ba_B - radius, ba_B + radius]
  - For each candidate, compute Euclidean distance in (ba, ca) space
  - If distance < radius (default 0.002), consider as match
  - Optionally filter by scale ratio: min_scale=0.7, max_scale=1.3
- **THEN** for each matched triangle pair, increment vote for all 3 vertex pairs

#### Scenario: 投票矩阵
- **WHEN** creating vote matrix
- **THEN** allocate nbright x nbright integer matrix (initialized to 0)
- **AND** for each matched triangle pair (tri_img, tri_gaia):
  - vote[tri_img.a_idx][tri_gaia.a_idx]++
  - vote[tri_img.b_idx][tri_gaia.b_idx]++
  - vote[tri_img.c_idx][tri_gaia.c_idx]++

### Requirement: Top Vote Getters（参考siril）
The system SHALL extract top matching star pairs from vote matrix:

#### Scenario: 提取最佳匹配
- **WHEN** extracting matched pairs
- **THEN** for each image star i, find Gaia star j with max votes
- **AND** for each Gaia star j, find image star i with max votes
- **AND** apply minimum vote filter: discard pairs with votes < 2 (AT_MATCH_MINVOTES)
- **AND** apply bidirectional consistency: keep only pairs where i's best is j AND j's best is i

### Requirement: Flux辅助投票增强（可选）
The system MAY use flux information to enhance matching:
- For matched vertex pairs, compute flux ratio similarity: `flux_sim = min(f1/f2, f2/f1)`
- If flux_sim > 0.5, boost the vote by +0.5 (fractional vote)
- This helps break symmetry in regular star patterns

### Requirement: 结果输出
The system SHALL output:
```c
typedef struct {
    int img_idx;    // 图像星点索引
    int cat_idx;    // Gaia星点索引
    int votes;      // 票数
    double flux_ratio; // flux比值 (可选)
} PSMMatchPair;

typedef struct {
    PSMMatchPair *pairs;
    int pair_count;
    int tri_match_count;  // 匹配的三角形对数
    int total_votes;      // 总票数
} PSMMatchResult;
```

## MODIFIED Requirements
无（全新模块）

## REMOVED Requirements
无

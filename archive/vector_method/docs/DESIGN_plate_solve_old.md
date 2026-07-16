# PlateSolve 模块设计文档

## 概述

PlateSolve模块负责天文图像的Plate Solving（天球坐标解析），建立像素坐标与天球坐标的映射关系。

**核心特性：**
- 整合两步解析流程为单一API
- 保持模块化设计，便于维护和扩展
- 支持5阶SIP畸变拟合
- 输出完整WCS和SIP系数

---

## 统一API

### 核心函数

```c
int psolve_solve(
    PSolveHandle handle,
    const double *img_x, const double *img_y, const double *img_flux,
    const int *img_saturated, int img_count, int n_saturated,
    const PSolveImageData *img_data,
    const PSolveConfig *config,
    PSolveResult *result);
```

### Python接口

```python
from plate_solve import PlateSolve, PlateSolveConfig

solver = PlateSolve(gaia_data_dir="/path/to/GaiaDR3")

result = solver.solve(
    img_x=img_x, img_y=img_y, img_flux=img_flux,
    img_saturated=saturated, n_saturated=n_sat,
    center_ra=266.0, center_dec=-28.0,
    focal_length_mm=200.0, pixel_size_um=6.0,
    width=4500, height=3000
)

print(f"中心: RA={result.center_ra:.6f}, Dec={result.center_dec:.6f}")
print(f"RMS: {result.rms_px:.3f} px")
print(f"WCS: {result.wcs}")
```

---

## 两步解析流程

```
┌─────────────────────────────────────────────────────────────────┐
│  输入: 星点坐标 + flux + 饱和标记 + 图像元数据                  │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  第一步: 精匹配（多阶段策略）                                   │
│                                                                 │
│  阶段A: 饱和星粗匹配                                           │
│  ├─ 使用饱和星（最亮~200颗）进行三角匹配                       │
│  ├─ 测试四种翻转模式                                           │
│  ├─ 获得初始变换：中心偏移、旋转、缩放、翻转                   │
│  └─ 容差：旋转任意，缩放0.5x~2x                                │
│                                                                 │
│  阶段B: 非饱和亮星精化                                         │
│  ├─ 筛选前200颗非饱和亮星（按flux排序）                        │
│  ├─ 用阶段A的变换预测Gaia坐标                                  │
│  ├─ 在预测位置±50px范围内搜索匹配                              │
│  ├─ 建立1对1匹配对                                             │
│  ├─ RANSAC过滤外点（阈值5px，迭代1000次）                      │
│  └─ 最小二乘拟合精确变换                                       │
│                                                                 │
│  输出（传递给第二步）:                                          │
│  ├─ center_ra, center_dec: 精确中心坐标                       │
│  ├─ rotation_deg: 旋转角（度）                                 │
│  ├─ scale_arcsec_px: 比例尺（角秒/像素）                      │
│  ├─ flip_mode: 翻转模式（0-3）                                 │
│  ├─ affine: 仿射变换系数（a0-a5, b0-b5）                       │
│  └─ rms_px: 匹配RMS                                            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  第二步: SIP畸变拟合（继承第一步结果）                          │
│                                                                 │
│  阶段A: 应用第一步变换                                          │
│  ├─ 用第一步的中心坐标重新投影Gaia星点                         │
│  ├─ 应用第一步的旋转和翻转                                      │
│  ├─ 应用第一步的比例尺                                          │
│  └─ 此时图像坐标与Gaia坐标应大致对齐                           │
│                                                                 │
│  阶段B: 近邻匹配                                                │
│  ├─ TOP 2000非饱和亮星 vs TOP 3000 Gaia亮星                    │
│  ├─ 在预测位置±20px范围内搜索匹配                              │
│  └─ 建立1对1匹配对                                             │
│                                                                 │
│  阶段C: 仿射变换拟合                                            │
│  ├─ 最小二乘拟合仿射变换（平移+旋转+缩放）                     │
│  └─ 计算残差                                                    │
│                                                                 │
│  阶段D: 5阶SIP畸变拟合                                          │
│  ├─ 归一化坐标到[-1,1]范围（数值稳定性）                       │
│  ├─ 构建正规方程矩阵                                            │
│  ├─ 选主元高斯消元求解                                          │
│  └─ 转换系数回原始坐标                                          │
│                                                                 │
│  输出:                                                          │
│  ├─ wcs: 完整WCS参数（CRVAL, CRPIX, CD矩阵）                   │
│  ├─ sip: SIP畸变系数（A/B, AP/BP）                             │
│  └─ rms_px: 最终RMS                                             │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│  输出: PSolveResult                                            │
│  ├─ center_ra, center_dec: 精确中心坐标                       │
│  ├─ rotation_deg, scale_arcsec_px: 旋转和尺度                  │
│  ├─ flip_mode: 翻转模式                                        │
│  ├─ wcs: 完整WCS参数                                          │
│  ├─ sip: SIP畸变系数                                          │
│  └─ rms_px, matched_count: 精度指标                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第一步详细设计

### 阶段A: 饱和星粗匹配

**目的**: 获取初始变换，确定翻转模式

**输入**:
- 饱和星坐标（img_x_sat, img_y_sat），约100-200颗
- Gaia亮星坐标（cat_x, cat_y），约200颗

**流程**:
1. 构建三角形哈希表（图像和目录各5000个三角形）
2. 匹配三角形（边长比例容差5%）
3. 对每种翻转模式：
   - 计算相似变换（平移+旋转+缩放）
   - 统计匹配数（距离<25px）
   - 计算RMS
4. 选择最佳翻转模式（匹配数最多，RMS最小）

**容差**:
- 旋转：任意角度（0°-360°）
- 缩放：0.5x ~ 2x（允许焦距估计误差）

**输出**:
- 粗变换：tx, ty, rotation, scale, flip_mode

### 阶段B: 非饱和亮星精化

**目的**: 精确确定变换参数

**输入**:
- 非饱和亮星（TOP 200 by flux）
- Gaia亮星（TOP 1000 by mag）
- 阶段A的粗变换

**流程**:
1. 用粗变换预测每颗图像星对应的Gaia位置
2. 在预测位置±50px范围内搜索候选匹配
3. 选择最近的候选作为匹配对
4. RANSAC过滤：
   - 随机选择3对点
   - 计算仿射变换
   - 统计内点（残差<5px）
   - 重复1000次，选择内点最多的模型
5. 用所有内点最小二乘拟合精确仿射变换

**仿射变换模型**:
```
x' = a0 + a1*x + a2*y
y' = b0 + b1*x + b2*y
```

**输出**:
- 精确变换：center_ra, center_dec, rotation_deg, scale_arcsec_px, flip_mode
- 仿射系数：a0-a5, b0-b5
- RMS

---

## 第二步详细设计

### 关键：继承第一步结果

**问题**: 第二步必须在第一步基础上进行，不能独立匹配

**正确流程**:
1. 用第一步的中心坐标(center_ra, center_dec)重新进行Gnomonic投影
2. 应用第一步的旋转和翻转到Gaia坐标
3. 应用第一步的比例尺
4. 此时图像坐标和Gaia坐标应大致对齐（残差<5px）
5. 进行近邻匹配和SIP拟合

**错误流程**（之前的实现）:
- 第二步用原始中心坐标投影
- 没有应用第一步的旋转和翻转
- 导致匹配完全错误

---

## 数据传递

### 第一步到第二步的数据

```c
typedef struct {
    // 精确中心坐标
    double center_ra;      // 度
    double center_dec;     // 度
    
    // 旋转和比例尺
    double rotation_deg;   // 度
    double scale_arcsec_px; // 角秒/像素
    
    // 翻转模式
    int flip_mode;         // 0-3
    
    // 仿射变换系数
    double a0, a1, a2;     // x' = a0 + a1*x + a2*y
    double b0, b1, b2;     // y' = b0 + b1*x + b2*y
    
    // 精度
    double rms_px;
    int matched_count;
} Step1Result;
```

### 第二步如何使用

```c
// 1. 用新中心重新投影
gnomonic_projection(cat_ra, cat_dec, step1.center_ra, step1.center_dec, &cat_x, &cat_y);

// 2. 应用旋转和翻转
for (int i = 0; i < n_cat; i++) {
    double x = cat_x[i];
    double y = cat_y[i];
    
    // 翻转
    if (step1.flip_mode & 1) x = -x;
    if (step1.flip_mode & 2) y = -y;
    
    // 旋转
    double cos_r = cos(step1.rotation_deg * M_PI / 180);
    double sin_r = sin(step1.rotation_deg * M_PI / 180);
    cat_x[i] = cos_r * x - sin_r * y;
    cat_y[i] = sin_r * x + cos_r * y;
}

// 3. 应用比例尺
double scale_factor = step1.scale_arcsec_px / initial_scale;
for (int i = 0; i < n_cat; i++) {
    cat_x[i] *= scale_factor;
    cat_y[i] *= scale_factor;
}

// 4. 现在cat_x, cat_y与img_x, img_y大致对齐，可以进行近邻匹配
```

---

## 模块架构

```
lib/plate_solve/
├── include/
│   └── plate_solve.h           # 统一API头文件
│
├── src/
│   ├── psolve_api.cpp          # 统一API实现（整合两步）
│   ├── psolve_coarse.cpp       # 第一步核心逻辑
│   ├── psolve_fine.cpp         # 第二步核心逻辑
│   └── ...                     # 其他辅助模块
│
├── modules/
│   ├── star_alignment/         # 第一步：粗匹配模块
│   │   ├── psm_star_alignment.cpp
│   │   ├── psm_star_alignment.h
│   │   └── star_alignment.dll
│   │
│   ├── iterative_refine/       # 第二步：精匹配模块
│   │   ├── psm_iterative_refine.cpp
│   │   ├── psm_iterative_refine.h
│   │   ├── psm_sip.cpp         # SIP多项式拟合
│   │   ├── psm_sip.h
│   │   └── psm_iterative_refine.dll
│   │
│   └── common/                 # 公共数据结构
│
├── python/
│   ├── plate_solve.py          # Python统一接口
│   └── test_*.py               # 测试脚本
│
└── DESIGN.md                   # 本设计文档
```

**模块化设计优势：**
1. 每个模块独立编译，可单独测试
2. 模块间通过清晰接口通信
3. 便于替换或升级单个模块
4. 统一API简化使用，内部保持模块化

---

## 第一步：粗匹配（饱和星优先策略）

### 设计原理

**饱和星的优势：**
- 饱和星是图像中最亮的星，信噪比最高
- 饱和星数量少（通常几十到几百颗），计算量小
- 饱和星在Gaia星表中对应亮星，匹配成功率高
- 饱和星位置准确（质心不受PSF形状影响）

### 算法流程

```
Step 1: 计算像素尺度和FOV
  scale_arcsec_px = 206.265 × pixel_size_um / focal_length_mm
  fov_diag = sqrt(width² + height²) × scale_arcsec_px / 3600

Step 2: Gaia锥形查询 + 极限星等二分法
  查询半径 = fov_diag × 1.2 / 2
  目标星数 = 检测星数 × 1.5
  二分法确定极限星等

Step 3: Gnomonic投影
  将Gaia星表坐标投影到像素坐标

Step 4: 饱和星优先三角匹配
  if n_saturated >= 10:
      使用饱和星进行三角匹配
  else:
      使用饱和星 + 最亮正常星共100颗

Step 5: 四种翻转模式测试
  选择匹配数最多、RMS最小的模式
```

---

## 第二步：SIP畸变拟合

### 设计原理

在第一步获得精确中心坐标、旋转角和翻转模式后，进行精细匹配和畸变拟合。

### SIP畸变模型

使用5阶多项式描述像素坐标到中间世界坐标的映射：

```
u = Σ x[i,j] × x^i × y^j  (i+j ≤ 5)
v = Σ y[i,j] × x^i × y^j  (i+j ≤ 5)
```

**坐标归一化：**
```
max_radius = sqrt((width/2)² + (height/2)²)
norm_x = x / max_radius
norm_y = y / max_radius
```

**SIP正向变换：**
```
u = x + Σ Aij × x^i × y^j  (i+j ≥ 2)
v = y + Σ Bij × x^i × y^j  (i+j ≥ 2)
```

**SIP逆向变换：**
```
x = u + Σ APij × u^i × v^j  (i+j ≥ 2)
y = v + Σ BPij × u^i × v^j  (i+j ≥ 2)
```

---

## 数据结构

### PSolveConfig

```c
typedef struct {
    int use_saturated_priority;   // 使用饱和星优先策略
    int n_img_bright;             // 图像侧亮星数量
    int n_cat_bright;             // 星表侧亮星数量
    double max_match_dist_px;     // 最大匹配距离（像素）
    int max_iterations;           // 最大迭代次数
    double match_threshold;       // 匹配阈值
    int sip_order;                // SIP阶数（默认5）
    double converge_thresh;       // 收敛阈值
} PSolveConfig;
```

### PSolveResult

```c
typedef struct {
    double center_ra;             // 精确中心RA（度）
    double center_dec;            // 精确中心Dec（度）
    double rotation_deg;          // 旋转角（度）
    double scale_arcsec_px;       // 像素尺度（角秒/像素）
    int flip_mode;                // 翻转模式
    int matched_count;            // 匹配星对数
    double rms_px;                // RMS（像素）
    double step1_time_sec;        // 第一步耗时（秒）
    double step2_time_sec;        // 第二步耗时（秒）
    PSolveWCS wcs;                // WCS参数
    PSolveSIPCoeffs sip;          // SIP系数
    int sip_valid;                // SIP系数有效标记
} PSolveResult;
```

---

## 性能指标

| 步骤 | 耗时 | RMS | 匹配数 |
|------|------|-----|--------|
| 第一步 | ~2s | ~1.4px | ~3000 |
| 第二步 | <0.1s | ~2px | ~1000 |
| **总计** | **~2.1s** | - | - |

---

## 待完善事项

1. **核心功能**：✅ 已完成
   - 第一步粗匹配（star_alignment.dll）
   - 第二步SIP拟合框架（iterative_refine.dll）
   - 统一API
   - Python接口

2. **第二步畸变模型**：🔄 进行中
   - ✅ 接收第一步变换参数
   - ✅ 应用翻转和旋转到Gaia坐标
   - ✅ 近邻匹配（TOP 2000图像星 vs TOP 3000 Gaia星，20px范围）
   - ✅ 残差-距离分析（10个距离分箱）
   - ✅ SIP多项式拟合（最高5阶，坐标归一化）
   - ✅ SIP系数计算（正向A/B和逆向AP/BP）
   - 📝 迭代优化过程（当前单次拟合，需多轮迭代）
   - 📝 分区域异常值剔除（grid_size=5，5×5区域）
   - 📝 三角形特征匹配验证
   - 📝 中心/比例尺/旋转角联合更新

3. **扩展功能**：📝 待实现
   - Sigma-Clip异常剔除
   - RMS评分系统
   - 迭代收敛判断

4. **输出**：📝 待实现
   - WCS头文件生成
   - SIP系数写入FITS

---

## 第二步详细算法流程

### 输入数据
- 图像星点：`img_x`, `img_y`, `img_flux`, `img_saturated`
- Gaia星表：`cat_ra`, `cat_dec`, `cat_mag`, `cat_x_px`, `cat_y_px`
- 第一步变换：`center_ra`, `center_dec`, `rotation_deg`, `scale_arcsec_px`, `flip_mode`

### 处理流程

```
Step 2.1: 应用第一步变换到Gaia坐标
  for each Gaia star:
    px = cat_x_px[i]
    py = cat_y_px[i]
    if flip_x: px = -px
    if flip_y: py = -py
    cat_px[i] = cos(rot) * px - sin(rot) * py
    cat_py[i] = sin(rot) * px + cos(rot) * py

Step 2.2: FOV裁剪
  保留 |cat_px| < width/2 且 |cat_py| < height/2 的星

Step 2.3: 选择亮星
  图像侧：TOP 2000 非饱和亮星（按flux排序）
  Gaia侧：TOP 3000 亮星（按mag排序）

Step 2.4: 近邻匹配
  max_match_dist = 20.0 px
  for each image star:
    在Gaia星中找最近邻
    if distance < max_match_dist:
      加入匹配对

Step 2.5: 残差-距离分析
  计算每个匹配点到图像中心的距离
  计算 残差 = |cat_pos - img_pos|
  按10个距离分箱统计残差中位数
  用于诊断畸变模式

Step 2.6: SIP多项式拟合
  坐标归一化：
    x_scale = max|src_x - x_center|
    y_scale = max|src_y - y_center|
    u = (src_x - x_center) / x_scale
    v = (src_y - y_center) / y_scale
  
  构建正规方程：
    XtX * cx = Xtdx
    XtX * cy = Xtdy
  
  选主元高斯消元求解
  
  反归一化得到原始坐标系数

Step 2.7: 计算SIP系数
  正向变换：A, B（像素→中间坐标）
  逆向变换：AP, BP（中间坐标→像素）
  
  使用7×7网格点计算逆变换

Step 2.8: 更新中心坐标
  从拟合的平移量(x00, y00)计算新中心：
    xi_rad = x00 / rad_to_px
    eta_rad = y00 / rad_to_px
    ir_plane_to_sky(xi, eta, center_ra, center_dec, &new_ra, &new_dec)

Step 2.9: 更新CD矩阵
  cd[0][0] = trans.x10
  cd[0][1] = trans.x01
  cd[1][0] = trans.y10
  cd[1][1] = trans.y01
  
  新比例尺 = sqrt(|det(cd)|) * 原比例尺
  新旋转角 = atan2(cd[1][0], cd[0][0])
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_stars_triangle` | 500 | 构建三角形的最大星数 |
| `tri_ratio_radius` | 0.002 | 三角形特征匹配半径 |
| `tri_min_area` | 100 px² | 最小三角形面积 |
| `grid_size` | 5 | 异常检测网格数（5×5） |
| `max_iterations` | 5 | 最大迭代次数 |
| `match_threshold` | 50.0 px | 匹配阈值 |
| `sip_order` | 5 | SIP阶数 |

### 输出数据

```c
IRRefineResult:
  final_ra, final_dec      // 精确中心坐标
  final_rotation           // 精确旋转角
  final_scale              // 精确比例尺
  dist_a0-a5, dist_b0-b5   // 畸变系数
  distortion_valid         // 畸变有效标记
  matched_count            // 匹配星对数
  rms_total                // 总RMS（像素）
  rms_arcsec               // 总RMS（角秒）
  sip.A, sip.B             // SIP正向系数
  sip.AP, sip.BP           // SIP逆向系数
  cd[2][2]                 // CD矩阵
  crpix[2], crval[2]       // WCS参考点
```

---

## 参考资料

- SIP畸变模型实现参考了 [Siril](https://gitlab.com/free-astro/siril) 项目的astrometry_solver.c模块
- SIP系数定义参考 [WCSLIB](https://www.atnf.csiro.au/people/mcalabre/WCS/wcslib/index.html) 文档

---

## 更新日志

- **2026-06-02**: 重写初始WCS生成模块（Step1）
  - 创建 `initial_wcs` 模块替代旧的 `star_alignment`
  - Python原型 (`initial_wcs.py`): 完整5步算法实现
  - C++ DLL (`psm_initial_wcs.dll`): 基于Python验证后的算法重写
  - 算法核心参考siril atpmatch/Valdes 1995
  - 饱和星优先策略：≥10颗用饱和星，<10颗用饱和+亮星共100颗
  - 4种翻转模式独立匹配
  - 迭代重投影收敛（siril风格）
  - 旧 `star_alignment` 模块归档到 `old/star_alignment/`
  - Python绑定 (`initial_wcs_ctypes.py`)
  - PlateSolve新增 `solve_step1_initial_wcs()` 方法
- **2026-05-29**: 完善二阶段畸变模型设计文档
  - 详细记录算法流程
  - 添加关键参数表
  - 明确待完善事项
- **2026-05-28**: 整合两步流程为统一API
  - 创建`psolve_solve`函数整合第一步和第二步
  - 创建Python封装`PlateSolve`类
  - 保持模块化设计，便于维护
- **2026-05-28**: 完成5阶SIP畸变拟合实现
  - 实现多项式变换结构SIP_Transform
  - 实现坐标归一化防止数值溢出
  - 实现网格点逆变换计算逆向SIP系数
- **2026-05-28**: 完成第二步实现和测试
  - 实现仿射变换拟合（平移+旋转+比例尺）
  - 实现残差-距离分析
- **2026-05-28**: 完成第一步实现和测试
  - 饱和星优先三角匹配策略
  - 四种翻转模式测试

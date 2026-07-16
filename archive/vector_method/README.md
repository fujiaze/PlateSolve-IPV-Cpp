# Vector Match V2 — 向量匹配法 Plate Solving算法

基于向量组对齐的天文图像Plate Solving算法，通过RANSAC+Umeyama SVD精修实现图像星点到Gaia星表的匹配，求解WCS参数。

## 测试结果

157帧全量测试（银心3面板×5滤镜）：

| 滤镜 | 帧数 | RMS中位数 | 成功率 |
|------|------|-----------|--------|
| H-alpha | 26 | 0.484px | 100% |
| Blue | 32 | 0.470px | 100% |
| Green | 34 | 0.495px | 100% |
| Oiii | 31 | 0.423px | 93.5% |
| Red | 32 | 0.504px | 100% |
| **总计** | **155/157** | **0.47px** | **98.7%** |

---

## 算法流程

```
输入: FITS图像 + 初始WCS（粗略RA/Dec/像素尺度）
  │
  ├─ Step 1: 从FITS头提取像素尺度s0和FOV
  ├─ Step 2: 星点检测（StarDetector, fitRadius=0自适应）
  ├─ Step 3: 亮星选取策略
  │     饱和星≥50 → 全部饱和 + 1.5×Gaia星
  │     饱和星<50 → 100饱和/亮 + 150 Gaia星
  ├─ Step 4: Gaia锥形查询 + 二分法极限星等
  ├─ Step 5: 向量组构建 + 稀疏度权重
  │     U: 图像向量组（像素坐标→gnomonic投影角秒，Y取反）
  │     W: 星表向量组（Gaia RA/Dec→gnomonic投影角秒）
  ├─ Step 6: 4种翻转模式独立匹配:
  │     ├─ 6a: 粗候选对构建（0.5×FOV KDTree搜索）
  │     ├─ 6b: RANSAC粗匹配（稀疏度加权采样 + 尺度预检|dU/dW-1|<0.05）
  │     ├─ 6c: 精细候选对重建（粗变换投影重建候选）
  │     └─ 6d: 迭代SVD精修（Umeyama + 动态MAD阈值）
  ├─ Step 7: 归一化打分选最佳模式
  └─ Step 8: WCS参数提取 + 中心修正 + RANSAC+SVD精修
  │
输出: WCS参数（CRVAL, CD矩阵, 仿射6参数）+ RMS
```

---

## 五项核心优化

| # | 优化 | 说明 | 解决的问题 |
|---|------|------|-----------|
| 1 | 稀疏度加权采样 | 孤立星优先采样，概率∝最近邻距离 | 纯随机RANSAC命中率极低（~8×10⁻⁶） |
| 2 | 尺度预检 | \|dU/dW-1\|>0.05的候选对直接丢弃 | 错误配对浪费RANSAC迭代 |
| 3 | 迭代SVD精修 | Umeyama算法替代第二次RANSAC | 第二次RANSAC不稳定且慢 |
| 4 | 动态MAD阈值 | 3.0×1.4826×MAD自适应内点阈值 | 固定阈值无法适应不同噪声 |
| 5 | 两阶段候选对 | 粗候选0.5×FOV + 投影重建精细候选 | 粗候选精度不够，精细候选半径太小会漏真 |

---

## 核心算法详解

### 1. 像素尺度与FOV计算

从FITS头或望远镜参数计算像素尺度s₀（角秒/像素）：

```
s₀ = 206.265 × pixel_size_μm / focal_length_mm
```

FOV对角线（度）：
```
FOV_diag = √(width² + height²) × s₀ / 3600
```

Gaia查询半径 = 1.2 × FOV_diag / 2（略大于半FOV，确保边缘星不遗漏）。

---

### 2. 亮星选取策略

图像星点分为两类：**饱和星**（像素值超过半阈值，用连通分量质心定位）和**正常星**（Moffat4拟合定位）。选取策略根据饱和星数量自适应：

```
if 饱和星数 ≥ 50:
    选取: 全部饱和星
    Gaia目标星数 = ⌈1.5 × 饱和星数⌉
else:
    选取: 全部饱和星 + 最亮的(100-饱和星数)颗正常星
    Gaia目标星数 = 150
```

**设计理由**：饱和星定位精度高且通量大，是匹配的锚点。窄带滤镜饱和星少时，补充正常星维持匹配所需的星数下限。

---

### 3. Gaia星表查询（二分法极限星等）

给定目标星数N_gaia，用二分法搜索极限星等mag_limit，使Gaia返回星数落在[N_gaia, 1.1×N_gaia]区间：

```
mag_low = 6.0, mag_high = 22.0
for i in range(30):
    mid = (mag_low + mag_high) / 2
    count = gaia_cone_search(ra, dec, radius, mid)
    if count < N_gaia:   mag_low = mid   # 星太少，放宽星等
    elif count > 1.1×N:  mag_high = mid   # 星太多，收紧星等
    else: break
    if mag_high - mag_low ≤ 0.1: break
```

**设计理由**：固定星等限制在不同天区返回星数差异巨大（银心vs极区），二分法自适应控制星数，平衡匹配精度和计算量。

---

### 4. 向量组构建

#### 图像向量组U

将选取的图像星点像素坐标转换为以图像中心为原点的角秒坐标：

```
U_x[i] = (x[i] - width/2) × s₀
U_y[i] = -(y[i] - height/2) × s₀    ← Y取反！
```

**Y取反原因**：图像像素Y轴向下（行号增大），天球赤纬Dec向上。gnomonic投影的η轴向上对应Dec增大，因此图像向量的Y分量必须取反才能与星表向量对齐。

#### 星表向量组W

将Gaia星表的(RA, Dec)通过gnomonic投影转换为以(RA₀, Dec₀)为中心的切面坐标(ξ, η)：

```
cos(c) = sin(Dec₀)·sin(Dec) + cos(Dec₀)·cos(Dec)·cos(RA-RA₀)
ξ = cos(Dec)·sin(RA-RA₀) / cos(c) × 206265″
η = [cos(Dec₀)·sin(Dec) - sin(Dec₀)·cos(Dec)·cos(RA-RA₀)] / cos(c) × 206265″
```

其中206265 = 180/π × 3600是弧度到角秒的转换因子。

#### 稀疏度权重

对U中每个点计算到最近邻U点的距离作为稀疏度权重：

```
tree = KDTree(U)
sparsity[i] = distance(U[i], nearest_neighbor(U[i]))
```

孤立星（sparsity大）在RANSAC中被优先采样，因为孤立星被误匹配的概率更低。

#### 4种翻转模式

天文图像可能存在4种朝向（镜像翻转），对W施加翻转：

```
mode=0: W不变
mode=1: W_x取反（左右镜像）
mode=2: W_y取反（上下镜像）
mode=3: W_x和W_y都取反（180°旋转+镜像）
```

每种模式独立匹配，最终选得分最高的。

---

### 5. 两阶段候选对构建（优化5）

#### 阶段一：粗候选对

用KDTree在W中搜索每个U点距离 < candidate_radius 的所有W点，构成候选对列表：

```
candidate_radius = FOV_diag × 3600 × 0.1    (FOV对角线的10%)
pairs = [(u_idx, w_idx) for each U[i] where ||U[i] - W[j]|| < candidate_radius]
```

**半径选择**：10% FOV是经验最优值。50% FOV导致每个U点平均113个候选（命中率极低），5% FOV可能遗漏真匹配。10% FOV平均6个候选/U点，平衡命中率和计算量。

#### 阶段二：精细候选对重建

粗匹配得到变换(s, θ, tx, ty)后，将W投影到U空间，重建高纯度候选池：

```
W' = s·R(θ)·W + t
for each U[i]:
    find nearest W'[j] where ||U[i] - W'[j]|| < 2 × tau_coarse
    add (i, j) to fine_pairs
```

**设计理由**：粗候选基于位置相近，但U和W可能有系统性偏移导致真匹配不在候选中。用粗变换投影后重建候选，消除了系统性偏移，大幅提高候选纯度。

---

### 6. RANSAC粗匹配（优化1+2）

#### 采样方式

从候选对中按稀疏度加权抽取2个**不同U点**，每个U点从其候选列表中随机选1个W配对：

```
1. 按稀疏度权重概率选择2个不同的U点: u_idx_a, u_idx_b
   prob[i] = sparsity[i] / Σsparsity
2. 从u_idx_a的候选W列表中随机选1个 → (u_idx_a, w_idx_a)
3. 从u_idx_b的候选W列表中随机选1个 → (u_idx_b, w_idx_b)
```

**为什么不用纯随机**：从N_img个U点和M个W点独立随机选2对，正确配对的概率≈(N_img/C(N_img,2))²×(1/M)² ≈ 8×10⁻⁶，几乎不可能命中。稀疏度加权让孤立星优先被选中，大幅提高命中率。

#### 2点相似变换求解

给定2对配对(u_a, w_a)和(u_b, w_b)，求解相似变换(s, θ, tx, ty)：

```
du = u_a - u_b
dw = w_a - w_b

s = ||du|| / ||dw||                   ← 缩放因子
θ = atan2(du_y, du_x) - atan2(dw_y, dw_x)   ← 旋转角
tx = u_a_x - s·(cos(θ)·w_a_x - sin(θ)·w_a_y)  ← 平移x
ty = u_a_y - s·(sin(θ)·w_a_x + cos(θ)·w_a_y)  ← 平移y
```

#### 尺度预检（优化2）

在RANSAC循环中，2点求解后立即检查s是否在有效范围：

```
if s < 0.9 or s > 1.1: skip    ← s必须在1.0±10%以内
```

**设计理由**：正确匹配的s≈0.983（像素尺度估算误差），错误匹配的s可能为0.1或10.0。预检直接过滤99%以上的无效假设，避免浪费后续内点统计。

#### 1对1互斥内点统计

应用变换后统计内点，采用**贪心1对1匹配**避免多对一：

```
1. W' = s·R(θ)·W + t
2. 对每个U[i]，找W'中最近邻: dist[i] = ||U[i] - W'[nearest[i]]||
3. 按dist从小到大排序
4. 贪心分配: 遍历排序后的U，如果dist < tau且W'[nearest[i]]未被占用，标记为内点
```

#### RANSAC参数

| 参数 | 值 | 说明 |
|------|-----|------|
| tau_coarse | max(1.0, 2.5×s₀) | 粗匹配内点阈值(角秒) |
| K | 3000 | 最大迭代次数 |
| min_inliers | max(5, 20%×N_img) | 最少内点数 |
| candidate_radius | 10%×FOV | 粗候选搜索半径 |
| s范围 | [0.9, 1.1] | 缩放因子有效范围 |

打分函数：`score = n_inliers - 1.0 × rms`（内点数多且RMS小的变换得分高）

---

### 7. 迭代SVD精修（优化3+4）

RANSAC粗匹配后，用Umeyama SVD精修获得统计最优变换。

#### 步骤1：紧阈值重新统计内点

RANSAC的tau_coarse太松（~15角秒），内点含大量误匹配。用紧阈值1.0×s₀重新统计：

```
tau_fine = 1.0 × s₀
W' = s·R(θ)·W + t
inliers = 1对1匹配(U, W', tau_fine)
```

如果内点不足3个，逐步放宽到2×s₀、5×s₀、10×s₀。

#### 步骤2：Umeyama SVD求解

在内点上建立1对1配对，用Umeyama算法求解最优相似变换：

```
输入: {u_i} (图像内点), {w_i} (对应星表内点), i=1..n

1. 去质心:
   μ_u = mean(u), μ_w = mean(w)
   u' = u - μ_u, w' = w - μ_w

2. 协方差矩阵:
   H = w'ᵀ @ u'     (2×2矩阵)
   注意: Python是w_centered.T @ u_centered
         C++是src_centered * dst_centered.transpose()

3. SVD分解:
   H = U_svd · Σ · V_svdᵀ

4. 保证纯旋转(det(R)=+1):
   d = det(V_svd · U_svdᵀ)
   S = diag(1.0, d)

5. 旋转矩阵:
   R = V_svd · S · U_svdᵀ

6. 缩放因子:
   s = trace(Σ · S) / trace(w'ᵀ · w')
   ⚠️ 关键: 分母是 Σ||w'_i||² = trace(w'ᵀ·w')
      不是除以n的方差 var_w = trace(w'ᵀ·w')/n
      错误写法会使s被放大n倍!

7. 平移向量:
   t = μ_u - s · R · μ_w

8. 提取旋转角:
   θ = atan2(R[1,0], R[0,0])
```

#### 步骤3：迭代精修

```
for iteration in range(max_iter=10):
    1. 用当前(s, θ, tx, ty)变换W，1对1匹配统计内点(tau=1.0×s₀)
    2. 在内点上建立配对，Umeyama求解新的(s, θ, tx, ty)
    3. 安全检查: |s - 1.0| > 0.1 则中止
    4. 收敛检查: 内点集合不再变化则停止
```

#### 动态MAD阈值（优化4）

迭代精修中使用的内点阈值可以自适应：

```
1. 对所有U[i]计算到最近W'[j]的距离d[i]
2. med = median(d)
3. MAD = median(|d[i] - med|)
4. tau_mad = 3.0 × 1.4826 × MAD
5. tau = max(min_tau, max(base_tau, tau_mad))
```

其中1.4826是正态分布下MAD到标准差的换算系数（σ ≈ 1.4826 × MAD），3.0σ对应99.7%置信区间。MAD对离群值鲁棒，不会像标准差那样被少数大残差拉高。

**注意**：实际迭代精修中使用固定紧阈值1.0×s₀（MAD阈值太松会导致迭代不收敛），MAD阈值仅用于最终统计。

---

### 8. 归一化打分与模式选择

4种翻转模式各自独立匹配后，用归一化得分选择最佳模式：

```
norm_score = (n_inliers / min(N_img, M)) × (1.0 - rms / tau)
```

得分越高表示：内点占比高且RMS小。选择norm_score最高的模式，且要求norm_score > 0.10（防止低质量匹配被误判为成功）。

最终还要检查s是否在[0.9, 1.1]范围内，超出则判定整个匹配失败。

---

### 9. WCS参数提取与中心修正

#### 中心修正

粗匹配得到的平移(tx, ty)反映了图像中心与星表参考点的偏移，用其修正参考点：

```
cos_dec0 = cos(Dec₀)
ΔRA  = tx / (cos_dec0 × 3600)    ← 角秒→度，考虑赤纬cos修正
ΔDec = ty / 3600                  ← 角秒→度
RA_new  = RA₀  + ΔRA
Dec_new = Dec₀ + ΔDec
```

#### 重新投影 + RANSAC+SVD精修

中心修正后，以新参考点(RA_new, Dec_new)重新对星表做gnomonic投影，再执行一轮RANSAC+SVD精修：

```
1. W_new = gnomonic_project(cat_ra, cat_dec, RA_new, Dec_new)
2. Wf_new = apply_flip(W_new, best_mode)
3. RANSAC粗匹配(tau=1.0×s₀, K=2000)
4. 迭代SVD精修(max_iter=10)
```

**为什么用RANSAC而不是纯SVD**：中心修正消除了大部分偏移，但重新投影后U和W仍有亚角秒级的残余偏移。纯SVD从tx=ty=0开始搜索，可能找不到足够的初始内点。RANSAC可以容忍初始偏移，找到正确配对后再用SVD精修到最优。

#### WCS参数输出

最终从(s, θ, tx, ty)提取WCS参数：

```
CRVAL1 = RA_new        ← 参考点赤经(度)
CRVAL2 = Dec_new       ← 参考点赤纬(度)
CD1_1 = s × cos(θ)    ← CD矩阵元素
CD1_2 = -s × sin(θ)
CD2_1 = s × sin(θ)
CD2_2 = s × cos(θ)

像素尺度 = s₀ × s_final (角秒/像素)
旋转角 = θ (度)
仿射6参数 = (tx, s×cos(θ), -s×sin(θ), ty, s×sin(θ), s×cos(θ))

RMS(像素) = RMS(角秒) / s₀
```

---

### 10. 逆Gnomonic投影

从切面坐标(ξ, η)反算天球坐标(RA, Dec)：

```
ρ = √(ξ² + η²)
c = arctan(ρ)
Dec = arcsin(cos(c)·sin(Dec₀) + η·sin(c)·cos(Dec₀)/ρ)
RA  = RA₀ + arctan2(ξ·sin(c), ρ·cos(Dec₀)·cos(c) - η·sin(Dec₀)·sin(c))
```

---

## 文件结构

```
lib/plate_solve/
├── python/
│   ├── vector_match_v2.py          # 纯Python实现（完整算法）
│   └── vector_match_v2_cpp.py      # C++加速版（Python I/O + C++核心计算）
├── cpp/
│   └── vector_match_v2/
│       ├── include/vm2_api.h       # C++ API头文件
│       ├── src/vm2_core.cpp        # C++核心实现
│       ├── third_party/
│       │   ├── Eigen/              # Eigen3（SVD）
│       │   ├── eigen-3.4.0/        # Eigen3完整源码
│       │   └── nanoflann-master/   # nanoflann（KDTree）
│       ├── Makefile
│       └── vector_match_v2.dll     # 编译产物
└── plate_solve_old/                # 旧版三角形匹配（已废弃）
```

---

## C++ API

### 数据结构

```c
// 求解参数
struct VM2SolveParams {
    double tau_coarse;        // 粗匹配内点阈值(角秒)
    int    K;                 // RANSAC最大迭代次数
    int    min_inliers;       // 最少内点数
    double candidate_radius;  // 粗候选搜索半径(角秒)
    double s0;                // 像素尺度(角秒/像素)
    double fov_diag_asec;     // FOV对角线(角秒)
    int    n_modes;           // 翻转模式数(4)
    int    seed;              // 随机种子
};

// 求解结果
struct VM2SolveResult {
    double s;           // 缩放因子
    double theta;       // 旋转角(弧度)
    double tx;          // 平移x(角秒)
    double ty;          // 平移y(角秒)
    int    n_inliers;   // 内点数
    double rms;         // RMS(角秒)
    int    best_mode;   // 最佳翻转模式
    double norm_score;  // 归一化得分
    int*   inlier_mask; // 内点掩码(调用方分配, 长度=N_img)
    int    success;     // 0=失败, 1=成功
};
```

### 导出函数

```c
// 核心求解: RANSAC粗匹配 + 内点统计
// 输入: U(N_img×2)图像向量, W(M×2)星表向量, sparsity(N_img)稀疏度权重
int vm2_solve(const double* U, int N_img,
              const double* W, int M,
              const double* sparsity,
              const VM2SolveParams* params,
              VM2SolveResult* result);

// SVD精修: 在给定内点上迭代Umeyama
int vm2_svd_refine(const double* U, int N_img,
                    const double* W, int M,
                    const int* inlier_mask,
                    double s_init, double theta_init,
                    double tx_init, double ty_init,
                    double s0, int max_iter,
                    VM2SolveResult* result);

// 内点统计: 1对1对应+RMS计算
int vm2_count_inliers(const double* U, int N_img,
                      const double* W, int M,
                      double s, double theta,
                      double tx, double ty,
                      double tau, int* inlier_mask,
                      double* out_rms);
```

### 编译

```bash
cd lib/plate_solve/cpp/vector_match_v2
make
# 产物: vector_match_v2.dll
# 依赖: Eigen3(已含), nanoflann(已含)
# 编译器: g++ -O3 -march=native -std=c++17
```

---

## Python接口

### 纯Python版

```python
from vector_match_v2 import VectorMatch

vm = VectorMatch(gaia_dir="GaiaDR3SP")
result = vm.solve(image_data, header)

if result.success:
    print(f"RA: {result.ra:.6f}°, Dec: {result.dec:.6f}°")
    print(f"Scale: {result.scale:.4f}\"/px, Rot: {result.rotation:.2f}°")
    print(f"RMS: {result.rms:.3f}px, Matched: {result.matched_count}")
```

### C++加速版

```python
from vector_match_v2_cpp import VectorMatchCpp

vm = VectorMatchCpp(gaia_dir="GaiaDR3SP")
result = vm.solve(image_data, header)
# Python层: Gaia查询、星点选取、WCS参数提取
# C++ DLL层: RANSAC、SVD精修（核心计算）
```

**C++ vs Python性能对比**：
- C++比Python快1.23x（星点检测占80%时间，DLL部分加速有限）
- RMS精度一致（平均差0.022px，无出错帧）

### ctypes结构体

```python
class VM2SolveParamsC(ctypes.Structure):
    _fields_ = [
        ("tau_coarse", c_double),
        ("K", c_int),
        ("min_inliers", c_int),
        ("candidate_radius", c_double),
        ("s0", c_double),
        ("fov_diag_asec", c_double),
        ("n_modes", c_int),
        ("seed", c_int),
    ]

class VM2SolveResultC(ctypes.Structure):
    _fields_ = [
        ("s", c_double),
        ("theta", c_double),
        ("tx", c_double),
        ("ty", c_double),
        ("n_inliers", c_int),
        ("rms", c_double),
        ("best_mode", c_int),
        ("norm_score", c_double),
        ("inlier_mask", POINTER(c_int)),
        ("success", c_int),
    ]
```

---

## 依赖模块

| 模块 | 用途 | 路径 |
|------|------|------|
| `star_detector` | 星点检测（Moffat4+饱和星） | `lib/star_detector/` |
| `astro_image_io` | FITS/XISF图像读写 | `lib/astro_image_io/` |
| `gaia_client` | Gaia DR3/DR3SP星表查询 | `lib/gaia_xpsd_client/` |
| `dynamic_psf` | Moffat4 PSF拟合 | `lib/dynamic_psf/` |

---

## 批量测试

```powershell
python test_vector_match.py
# 16线程并发，输出:
#   output/vector_match_report.txt  - 文本报告
#   output/debug/                    - 每帧调试图像（Gaia星标注红色十字）
```

---

## 已知限制

1. **窄带Oiii成功率略低**（93.5%）：极低信噪比帧可能匹配失败
2. **饱和星<30时成功率显著下降**：特别是H-alpha/Oiii窄带短曝光帧
3. **需要初始WCS**：依赖FITS头中的粗略RA/Dec和像素尺度
4. **s范围约束**：s必须在s₀±10%内，超出判定无效（实际s≈0.983时有效）

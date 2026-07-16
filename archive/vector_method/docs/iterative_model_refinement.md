# 控制点扩增与模型迭代精化设计

## 闭环迭代：初始CD → 扩增匹配 → 模型精化 → 扩增匹配 → 收敛

---

## 0. 问题形式化

```
已知:
  - M₀ 对高置信度控制点 C₀ = {(x_i, α_i, δ_i)}, M₀ 通常 4~15 对 (向量法 Phase B 输出)
  - 初始 CD₀ (从 C₀ 最小二乘拟合)
  - 图像侧: N_img 颗检测星 U = {(x_i, y_i)}
  - 星表侧: Gaia 锥形查询结果 G = {(α_j, δ_j)} (几千颗)

目标: 通过闭环迭代, 同时扩增控制点到 ≥50 对, 并精化 CD + SIP

核心约束:
  1. 每轮迭代后模型精度必须提高 (单调收敛)
  2. 错误点必须自动剔除, 不污染模型
  3. 评分机制必须真实反映精度, 对外点鲁棒
```

---

## 1. 整体架构

```
═══════════════════════════════════════════════════════════════
  IRM (Iterative Refinement Model) — 闭环迭代精化
═══════════════════════════════════════════════════════════════

  C₀ (初始控制点, M₀对) + CD₀
    │
    ▼
  ┌─────────────────────────────────────────────────────────┐
  │  循环 (iter = 1, 2, ...):                               │
  │                                                         │
  │  Step 1: 投影匹配                                       │
  │    用 CD+SIP 将全部图像星投影到天球                     │
  │    马氏距离自适应匹配 → 候选匹配对                       │
  │                                                         │
  │  Step 2: 局部几何一致性过滤                              │
  │    每对候选验证局部星对结构一致性                        │
  │    剔除歧义和假匹配                                     │
  │                                                         │
  │  Step 3: 全局 RANSAC 一致性                              │
  │    抽样 4 对 → 拟合 CD → 全量验证 → 内点集              │
  │    保留最大一致集                                       │
  │                                                         │
  │  Step 4: 模型精化                                       │
  │    内点集 + 初始控制点 C₀ → 联合 LSQ → CD + SIP         │
  │    稳健损失函数 (Huber) 降低外点影响                     │
  │                                                         │
  │  Step 5: 收敛判定                                       │
  │    S_robust 评分变化 < ε  或  S_robust 开始下降          │
  │    或  内点数不再增长                                   │
  │                                                         │
  │  若不收敛 → iter++, 回到 Step 1                         │
  └─────────────────────────────────────────────────────────┘
    │
    ▼
  最终 CD + SIP + 全部内点
```

---

## 2. 稳健评分机制 S_robust

### 2.1 问题：为什么 RMS 不可靠

```
普通 RMS:
  RMS = √(1/N Σ r_i²)

外点影响:
  假设 50 对控制点，其中 5 对是外点 (残差 ~50")
  内点残差 ~0.5"
  
  r_i²: 45×0.25 + 5×2500 = 11.25 + 12500 = 12511.25
  RMS = √(12511.25/50) ≈ 15.8"
  
  实际精度 0.5" 的模型，RMS 被报告为 15.8"  ← 毫无意义
```

### 2.2 S_robust 定义

```
S_robust = 模型对"最可信的 K% 控制点"的拟合精度

算法:

Step A: 残差排序
  for each 控制点 i:
    (ra_proj, dec_proj) = project(x_i, y_i, CD, SIP)
    r_i = angular_distance(ra_proj, dec_proj, α_i, δ_i)  // arcsec
  排序: r_(1) ≤ r_(2) ≤ ... ≤ r_(N)

Step B: 自适应内点比例估计
  // 利用残差跳变检测内点/外点分界
  ratio[i] = r_(i) / r_(i-1)   // 相邻残差比
  // 从 i=3 开始扫描, 找第一个 ratio > 3.0 的位置
  k_cut = 第一个满足 r_(i)/r_(i-1) > 3.0 的 i
  若找不到 → k_cut = N (全部内点)
  
  // 备选: 使用 MAD 方法
  median_r = r_(N/2)
  MAD = median(|r_i - median_r|) × 1.4826
  k_cut = count(r_i < median_r + 3.0 × MAD)

Step C: 计算稳健评分
  N_robust = min(k_cut, max(N/2, M₀))  // 至少用一半的点, 至少包含初始控制点
  
  S_robust = rms({r_(1), ..., r_(N_robust)})   // arcsec
  N_inliers = N_robust
  
  // 附加: 覆盖率评分
  coverage = N_robust / N_img
  S_robust_weighted = S_robust / min(coverage, 1.0)

输出: (S_robust, N_inliers, coverage)
```

### 2.3 为什么 S_robust 能真实反映精度

```
外点不参与计算:
  5个外点残差 ~50", 45个内点残差 ~0.5"
  残差跳变: r_(45)=0.6", r_(46)=47" → ratio=78 → k_cut=45
  S_robust = rms(前45个) ≈ 0.5"  ✓ 真实精度

迭代中可以正确比较:
  iter 1: S_robust = 2.1", N_inliers = 18 → 模型还粗
  iter 2: S_robust = 1.2", N_inliers = 32 → 模型改善 ✓
  iter 3: S_robust = 0.6", N_inliers = 45 → 模型精化 ✓
  iter 4: S_robust = 0.6", N_inliers = 45 → 收敛 ✓

评分单调性:
  只要模型改善, 内点的残差一定降低
  外点残差始终高, 不影响评分
  → S_robust 单调递减直到收敛
```

---

## 3. 马氏距离自适应匹配

### 3.1 动机

```
固定容差 NN 匹配问题:
  图像中心: CD 投影精度 ~0.3"
  图像边缘: CD 投影精度 ~1.5" (畸变 + CD 外推误差)
  固定容差 1.0" → 中心匹配过严 (可能遗漏), 边缘匹配过松 (可能假匹配)
```

### 3.2 方法

```
对每个图像星 (x_i, y_i):

Step 1: 投影
  (ra_i, dec_i) = gnomonic_inverse(CD · [x_i-cx, y_i-cy]^T + SIP_correction)

Step 2: 投影不确定性
  像素位置越远离中心, CD 外推误差越大
  近似: σ_proj(x,y) = σ₀ × √(1 + ((x-cx)²+(y-cy)²) / (fov_half)²)
  其中 σ₀ = S_robust × cos(δ₀) (上一轮评分, cos δ 校正天球投影)

Step 3: 自适应匹配
  τ_i = max(3.0 × σ_proj(x_i,y_i), 2.0")  // 至少 2" 容差
  
  在 Gaia KD-tree 中:
    找 nearest neighbor gaia_1 (d_1) 和 second nearest gaia_2 (d_2)
    
    if d_1 < τ_i AND d_1/d_2 < 0.7:  // Lowe 距离比
      候选匹配 = (img_i, gaia_1)
    else:
      候选匹配 = None

Step 4: 双向验证
  对每个候选匹配 (img_i, gaia_j):
    将 gaia_j 用 CD+SIP 反向投影到图像: (x_proj, y_proj)
    在图像星中找最近邻 img_k, 距离 d_rev
    if img_k == img_i AND d_rev < τ_i:
      保留
    else:
      丢弃

输出: 候选匹配对列表 (通常 N_img × 30-60% ≈ 数十到上百对)
```

### 3.3 自适应容差的物理直觉

```
迭代中的 τ 变化:

iter 0 (CD₀ from M₀=8 pairs, σ₀≈2"):
  中心 τ ≈ 3×2 = 6"   ← 宽松, CD 精度低
  边缘 τ ≈ 3×2×√(1+1) ≈ 8.5"
  → 尽量多捞候选, 宁滥勿缺 (后续几何过滤会清洗)

iter 2 (CD from 20 pairs, σ₀≈1"):
  中心 τ ≈ 3×1 = 3"   ← 收紧
  边缘 τ ≈ 4.2"
  → CD 改善, 容差自动收紧

iter 5 (CD+SIP from 45 pairs, σ₀≈0.5"):
  中心 τ ≈ 1.5"
  边缘 τ ≈ 2.1"
  → 模型精确, 匹配非常干净
```

---

## 4. 局部几何一致性过滤

### 4.1 原理

借鉴 Heyl [2013] k-d match 和 Kolomenkin [2008] 几何投票：

```
核心思想: 真匹配的星对, 其周围星的相对几何关系也是正确的
         假匹配只碰巧位置接近, 无法解释邻域的几何结构

不依赖当前模型 (此检查独立于投影精度)
仅使用星间角距 (旋转/平移不变量)
```

### 4.2 算法

```
对每个候选匹配 (img_A, gaia_a):

  // 图像侧: 取 img_A 的 K 个最近邻
  neighbors_img = knn(img_A, K=8)

  // 星表侧: 
  // 方案 A: 这 K 个邻星也在候选匹配中有对应
  // 方案 B: 直接在 gaia_a 周围找最近邻 (更快)
  neighbors_gaia = knn(gaia_a, K=15)  // 多取一些, 增加匹配机会

  consistency = 0

  for each img_B in neighbors_img:
    d_img = |img_A - img_B| × s₀  // 图像侧角距 (arcsec)
    
    // 检查: 是否存在 gaia_b 使得 d_gaia ≈ d_img?
    for each gaia_b in neighbors_gaia[:5]:  // 只看最近的 5 个
      d_gaia = angular_distance(gaia_a, gaia_b)
      
      if |d_img - d_gaia| < max(3.0, 3.0 × S_robust):
        // 进一步检查: 三角形闭合
        // 检查第三颗星的一致性 (可选, 更强)
        consistency += 1
        break  // 找到一个匹配就跳出内层循环

  // 判定
  if consistency >= K * 0.5:  // ≥4/8
    保留候选匹配 (img_A, gaia_a)
    同时: 将 consistency_count > 0 的邻星对也加入"可信对"缓存
  else:
    丢弃
```

### 4.3 为什么几何过滤鲁棒

```
真匹配 (img_A = gaia_a):
  img_A 周围 8 颗邻星 → 角距图像中已知
  gaia_a 周围 15 颗 Gaia 星 → 角距天球上已知
  由于 density matching (§1.3 DD-SPPS设计), 密度相当
  → 8 颗图像邻星中预计 ≥4 颗有对应 Gaia 星
  → consistency ≥ 4 ✓

假匹配 (img_A 的投影碰巧落在 gaia_b 附近):
  img_A 的 8 颗邻星 → 角距是真实天文星场的
  gaia_b 周围星的距离与图像中的完全不同 (随机天球位置)
  → consistency ≈ 0~1 ✗
```

---

## 5. RANSAC 全局一致性

### 5.1 为什么需要 RANSAC

```
几何过滤是 "局部" 检查 → 残留少量假匹配可能通过
RANSAC 是 "全局" 检查 → 所有正确匹配必须满足同一变换模型

互补: 几何过滤删除了 80-90% 假匹配
      RANSAC 删除剩余 10-20%
```

### 5.2 实现

```
输入: 候选匹配对 (经过 Step 2 几何过滤)
      N_candidates ≈ 30-100 对

RANSAC 参数:
  最小子集: n = 4 对
  最大迭代: N_ransac = min(1000, ceil(log(1-0.999)/log(1-w⁴)))
            w ≈ 0.6 (内点率估计)
            ≈ log(0.001)/log(1-0.13) ≈ 53 次 → 取 200
  内点阈值: τ_ransac = max(2×S_robust, 1.5")

算法:
  best_inliers = []
  best_CD = None

  for iter in 1..200:
    sample = random 4 pairs from candidates
    
    // 检查: 4 个采样的图像侧是否共线/近共线
    if 最小三角形面积 < 10 px²: continue
    
    CD_sample = LSQ_4pairs(sample)
    
    // 全量验证 (只做 Gnomonic 投影, 不做 SIP)
    inliers = []
    for each cand in candidates:
      (ra_proj, dec_proj) = project(cand.img, CD_sample)
      error = angular_dist(ra_proj, dec_proj, cand.gaia_ra, cand.gaia_dec)
      if error < τ_ransac:
        inliers.append(cand)
    
    if len(inliers) > len(best_inliers):
      best_inliers, best_CD = inliers, CD_sample

  // 尾端处理 (PROSAC 思想, Chum 2005)
  if len(best_inliers) < 10:
    // 放宽阈值
    τ_ransac *= 1.5
    重新 RANSAC
```

### 5.3 为什么 RANSAC 有效（Kumar 2010 理论）

```
Kumar [2010] 闭式解:
  4 星多边形误匹配频率 f_false ≈ N_cat⁴×(Kσ)^5/π²

带入数值: N_cat=50 (候选数), K=6, σ=1"
  f_false ≈ 50⁴ × 6⁵ / π² ≈ 6.25×10⁶ × 7776 / 9.87 ≈ 4.9×10⁹

但在 RANSAC 框架下:
  仅 4 个采样可能全是外点的概率:
    P_4_outliers = (1-w)⁴ ≈ (0.4)⁴ ≈ 0.026
  
  RANSAC 做了 200 次 → 200 × 0.026 ≈ 5 次可能抽到全外点组
  
  但全外点组找不到足够的 inliers → 被淘汰
  
  真内点组 w⁴ ≈ 0.6⁴ ≈ 0.13 → 200 × 0.13 ≈ 26 次抽到
  → 轻松找到足够数量的内点
```

---

## 6. 稳健模型精化

### 6.1 Huber 损失 LSQ

```
问题: 普通 LSQ 对任何外点都敏感
     Huber 损失对大残差点的贡献做截断

Huber 损失:
  ρ(r) =     r²           if |r| ≤ δ
        = 2δ|r| - δ²      if |r| > δ

  δ = 1.345 × σ̂  (95% 渐近效率)
  其中 σ̂ = MAD(r) × 1.4826 (稳健尺度估计)

Huber LSQ 求解:
  等价于迭代加权最小二乘 (IRLS):
    权重 w_i = min(1, δ / |r_i|)
    
  iter 0: 初始权重全为 1 (普通 LSQ)
  iter 1: 计算残差 → 更新权重 → 加权 LSQ
  iter 2: 重复直到收敛
  
  → 3-5 轮即收敛
```

### 6.2 分层求解: CD → 径向畸变 → SIP

```
控制点: N 对 (来自 RANSAC 内点 + 初始控制点 C₀)

Step 1: 仅 CDs (4 参数)
  待求: a, b, c, d (CD 矩阵元素)
  方程: (ra - ra₀)·cos(dec₀) = a·dx + b·dy
        (dec - dec₀)          = c·dx + d·dy
  Huber LSQ → CD

Step 2: 径向畸变 (额外 K 参数)
  残差 = 投影误差 (径向部分)
  model: dr(r) = K₁·r³ + K₂·r⁵ + K₃·r⁷
        r = sqrt(dx² + dy²)  (相对于 CRPIX)
  Huber LSQ → K₁, K₂, K₃

Step 3: SIP (可选, 需 ≥30 对)
  残差 = 投影误差 (剩余部分, 非径向)
  model: du = Σ A_{pq}·u^p·v^q, dv = Σ B_{pq}·u^p·v^q
        (u,v) 归一化到 [-1, 1]
  选阶: BIC = N·log(RMS²) + k_poly·log(N)
  选 BIC 最小的阶数 (最多 4 阶, 15~21 项)
  Huber LSQ → SIP 系数
```

---

## 7. 完整算法伪代码

```
═══════════════════════════════════════════════════════════════
  IRM — Iterative Refinement Model
═══════════════════════════════════════════════════════════════

function irm_refine(C0, CD0, U_img, G_gaia, s0):

  // 初始化
  control_points = C0  // 高置信度对, 不可被移除
  CD = CD0
  SIP = None
  S_robust_prev = inf
  S_robust = compute_s_robust(control_points, CD, SIP)
  
  iter = 0
  
  while True:
    iter += 1
    
    // ═══ Step 1: 投影匹配 ═══
    candidates = []
    for each img in U_img:
      proj = gnomonic_inverse(CD · img + SIP_correction)
      τ_i = max(3.0 × σ_proj(img), 2.0")
      nn1, nn2 = gaia_kdtree.find_2nn(proj)
      if nn1.dist < τ_i and nn1.dist/nn2.dist < 0.7:
        candidates.append((img, nn1.star, nn1.dist))
    
    // 双向验证
    candidates = bidirectional_filter(candidates, CD, SIP, U_img)
    
    // ═══ Step 2: 局部几何一致性 ═══
    filtered = []
    for each (img_i, gaia_j) in candidates:
      consistency = local_geometry_check(img_i, gaia_j, U_img, G_gaia, s0)
      if consistency >= 4:  // K=8, 阈值 4
        filtered.append((img_i, gaia_j, consistency))
    
    // ═══ Step 3: RANSAC 全局一致性 ═══
    all_candidates = filtered 推断的邻星对 + filtered 自身
    ransac_result = ransac_fit(all_candidates, τ=max(2×S_robust, 1.5"))
    
    // ═══ Step 4: 模型精化 ═══
    merged_points = C0 ∪ ransac_result.inliers  // 初始控制点永不丢弃
    
    if len(merged_points) > len(control_points):
      control_points = merged_points
    
    // Huber LSQ 分层拟合
    CD = huber_lsq_cd(control_points)
    SIP = huber_lsq_sip(control_points, CD, max_order=min(4, iter+1))
    
    // ═══ Step 5: 收敛判定 ═══
    S_robust_new = compute_s_robust(control_points, CD, SIP)
    
    if abs(S_robust_new - S_robust) < 0.05:
      break  // S_robust 不再变化 → 收敛
    if S_robust_new > S_robust * 1.1:
      break  // 评分变差 → 模型过拟合, 停止
    
    S_robust = S_robust_new
    S_robust_prev = S_robust
    
    if iter >= 10:
      break  // 安全上限
  
  return (CD, SIP, control_points, S_robust)
```

---

## 8. 收敛性分析

### 8.1 单调性

```
引理: 在第 t 轮选出的内点集 I_t 上,
      LSQ 解 CD_{t+1} 的 S_robust 不高于 CD_t 的 S_robust

证明:
  I_t 的定义 = 在 CD_t 投影下, 误差 < τ 的点
  CD_{t+1} = argmin_CD Huber_loss(CD | I_t)
  
  CD_t 是可行解 (误差 < τ, 所以误差 < δ, Huber=LSQ)
  CD_{t+1} 必定是更好的或等价的解
  
  因此 Huber_loss(CD_{t+1}) ≤ Huber_loss(CD_t)
  → S_robust 非增
  
  但并非严格递减: 若模型已经最优, 则持平

下降量分析 (借鉴 ICP Besl & McKay 1992 单调收敛定理):
  均方误差序列 {S_robust^(t)} 非增且有下界 (0)
  → 必然收敛 ✓
```

### 8.2 典型收敛曲线

```
iter  S_robust  N_inliers  notes
────────────────────────────────────────
  0     2.50"       8      CD₀ from Phase B
  1     1.80"      18      首次扩增, 匹配对增加
  2     1.20"      28      模型改善, 更多点能匹配
  3     0.70"      42      SIP 加入, 畸变补偿
  4     0.55"      48      接近收敛
  5     0.52"      50      收敛 ✓
  6     0.52"      50      稳定 (停止)
```

---

## 9. 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| K_geometry | 8 | 局部几何检查的邻星数 |
| τ_geometry | 0.5×K | 几何一致性阈值 (≥4/8) |
| Lowe_ratio | 0.7 | 距离比检验阈值 |
| RANSAC_n | 4 | RANSAC 最小子集 |
| RANSAC_max_iter | 200 | RANSAC 最大迭代 |
| τ_ransac_base | 1.5" | RANSAC 内点阈值 (min) |
| Huber_δ | 1.345×MAD | Huber 损失转折点 |
| SIP_max_order | min(4, iter+1) | SIP 最大阶数 (迭代中递增) |
| SIP_min_pairs | 15 | SIP 最少控制点数 |
| converge_eps | 0.05" | S_robust 收敛阈值 |
| max_iter | 10 | 最大迭代次数 |

---

## 10. 与 V3.5 的差异

| 维度 | V3.5 | IRM |
|------|------|-----|
| 扩增方式 | Phase C 全局 NN (一次性) | 渐进: 投影→过滤→RANSAC→精化→循环 |
| 匹配容差 | 固定 (1×s₀) | 自适应 (马氏距离, 随 σ_proj 变化) |
| 误匹配过滤 | Pre-filter (MAD 3σ) | 三重: 距离比 + 几何一致性 + RANSAC |
| 模型拟合 | 普通 LSQ | Huber 稳健 LSQ |
| SIP 阶选择 | BIC | BIC (同上) |
| 评分 | RMS (被外点拉高) | S_robust (对外点鲁棒) |
| 收敛 | 无 (一次尝试) | 循环直到 S_robust 稳定 |
| 控制点管理 | Phase B 输出固定 | 动态增长, 外点自动剔除 |
```

---

## 11. 实现文件规划

```
lib/plate_solve/
├── cpp/
│   └── vector_match_v5/
│       ├── include/
│       │   └── vm5_api.h           # IRM 接口
│       ├── src/
│       │   ├── vm5_core.cpp        # 主迭代循环
│       │   ├── vm5_match.cpp       # 自适应投影匹配 + 距离比
│       │   ├── vm5_geometry.cpp    # 局部几何一致性过滤
│       │   ├── vm5_ransac.cpp      # RANSAC 全局一致性
│       │   ├── vm5_huber.cpp       # Huber LSQ 分层拟合
│       │   └── vm5_score.cpp       # S_robust 评分
│       └── Makefile
└── python/
    └── vector_match_v5_cpp.py      # ctypes 封装
```

---

## 12. 总结

IRM 的核心设计哲学：

1. **闭环正反馈**：模型改善 → 匹配更准 → 控制点增多 → 模型更好 → 匹配更准。借鉴 ICP (Besl & McKay 1992) 的单调收敛保证。

2. **三重过滤**：Lowe 距离比 (歧义淘汰) → 几何一致性 (局部结构) → RANSAC (全局模型)。参考 Heyl [2013] k-d match、Kolomenkin [2008] 几何投票、PROSAC [Chum 2005]。

3. **稳健评分 S_robust**：利用残差跳变检测外点分界，只对内点计算 RMS。借鉴 MAD 稳健统计 (Hampel 1974)。

4. **Huber 损失拟合**：大残差点贡献线性截断，保护模型不被外点扭曲。

5. **自适应容差**：随着模型精度提升，匹配容差自动收紧，淘汰更多假匹配。

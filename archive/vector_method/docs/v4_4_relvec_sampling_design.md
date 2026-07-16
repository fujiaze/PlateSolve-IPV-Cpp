# V4.4 相对向量法 — 系统设计

## 基于绝对距离的 θ 分离求解 + 小范围 tx/ty 搜索

---

## 0. 问题定义

```
已知:
  U = {u_i}, i=1..N_u    图像星向量 (角秒, 相对图像中心)
  W = {w_j}, j=1..N_w    Gaia 星向量 (角秒, 同一切点)
  s₀                     像素比例尺 (arcsec/pixel)

变换模型:
  u = s · R(θ) · w + t

  s ∈ [1-δ_s, 1+δ_s]  (δ_s ≈ 0.1)
  θ ∈ [-180°, 180°)
  t = (tx, ty)         平移 (角秒)

目标: 求 θ, t, s
约束: 不分先后步骤依赖, θ 求解阶段不依赖 t 的任何近似值
```

---

## 1. 核心原理：相对向量消去平移

### 1.1 差分变换

对任意两颗图像星 i, j 的向量差：

```
Δu_ij = u_j - u_i
```

若 u_i = s·R(θ)·w_a + t,  u_j = s·R(θ)·w_b + t，则：

```
Δu_ij = s·R(θ)·(w_b - w_a) = s·R(θ)·Δw_ab
```

**t 完全消去。** 这是整个方法的基础。

### 1.2 利用 s₀：绝对距离已知

图像星对的距离是绝对已知量：

```
d_img = |p_j - p_i| × s₀   (arcsec)
```

不需要用 s 去缩放——这是直接可算的。

这意味着在 Gaia 侧匹配时，搜索条件变为精确的绝对值范围查询，而非比例范围：

```
查询: d_gaia ∈ [d_img - 3σ_d, d_img + 3σ_d]

其中 σ_d = σ_pos × s₀  ≈ 1.0" (典型值)
     3σ_d ≈ 3.0"
```

### 1.3 角度差 — 纯旋转信号

```
Δθ = angle(Δu_ij) - angle(Δw_ab)
```

真匹配时 Δθ = θ，与 s 无关，与 t 无关，仅依赖方向差。

---

## 2. 阶段 A：θ 分离求解

### 2.1 数据结构

```
Gaia 侧预计算 (一次性, 不改 V4.4 现有逻辑):

1. 全配对距离矩阵 D_W[i][j] = |W[i] - W[j]|

2. k-vector 索引:
   - 按距离升序排列所有 Gaia 星对 (i, j, i<j)
   - 建立 k-vector 映射: K[d_bin] → 该距离区间的最后一个星对下标
   - Δ_kvec = 0.5" (步长)
   - 查询复杂度: O(1) 定位 + O(k) 扫描, k = 候选数

3. 每颗 Gaia 星的邻星距离排序表:
   D_W_sorted[a] = [(d(a,c₁), c₁), (d(a,c₂), c₂), ...] 按距离升序
   用于第三星交叉验证的快速查找
```

### 2.2 采样与投票

```
参数:
  K_total = 20000       总采样次数
  σ_d = 3.0"            距离容差 (3σ)
  n_third = 5           第三星验证数

for each sample k in 1..K_total:

  1. 图像星对随机采样:
     i = random(N_u), j = random(N_u), i≠j
     d_img = |U[j] - U[i]|  (已知! 用 s₀ 换算的角秒)
     angle_img = atan2(U[j].y-U[i].y, U[j].x-U[i].x)

  2. k-vector 距离查询:
     d_lo = d_img - σ_d, d_hi = d_img + σ_d
     idx_lo, idx_hi = kvector_query(d_lo, d_hi)
     gaia_pairs = [(a,b) for idx in idx_lo..idx_hi]

  3. 第三星交叉验证 (降噪):
     for each (a,b) in gaia_pairs:
       // 随机取 n_third 颗第三星 k
       n_passed = 0
       for each 第三星 k:
         d_ik = |U[k] - U[i]|
         d_jk = |U[k] - U[j]|
         // 在 D_W_sorted[a] 中二分查找距离 ≈ d_ik 的 c
         // 验证 |d(b,c) - d_jk| < σ_d
         if 存在这样的 c: n_passed++
       
       if n_passed >= 1:
         angle_gaia = atan2(W[b].y-W[a].y, W[b].x-W[a].x)
         Δθ = wrap180((angle_gaia - angle_img) × 180/π)
         
         // 加权投票
         bin = round((Δθ + 180°) / θ_bw)   // θ_bw = 1°
         votes[bin] += 1 + log2(1 + n_passed)  // 通过数越多, 权重越高
```

### 2.3 θ 峰值检测

```
θ 直方图: 360 bins × 1°

// 高斯平滑 (σ=1 bin)
for i in 0..359:
  smoothed[i] = 0.3×votes[i-1] + 0.4×votes[i] + 0.3×votes[i+1]

peak_bin = argmax(smoothed)
peak_val = smoothed[peak_bin]

// 背景估计: 去掉峰值 ±5° 区域后取中位数
bg_median = median(votes[exclude peak_bin ±5°])
bg_mad = MAD(votes[exclude peak_bin ±5°]) × 1.4826

SNR = peak_val / max(bg_median, 1.0)

// 判定
if SNR > 5.0:
  θ_peak = (peak_bin + 0.5) × 1° - 180°
  // 亚 bin 精化: 抛物线拟合
  用 peak_bin-1, peak_bin, peak_bin+1 三点的抛物线顶点
  θ_peak += sub_bin_offset
else:
  标记为失败 (SNR 不足)
```

### 2.4 为什么 θ 不依赖 t

```
Δθ = angle(Δu) - angle(Δw)

其中:
  Δu = u_j - u_i = s·R(θ)·(w_b - w_a)
  arg(Δu) = θ + arg(w_b - w_a)            ← s 和 t 都不在 arg 中!
  arg(Δw) = arg(w_b - w_a)

  Δθ = θ + arg(w_b-w_a) - arg(w_b-w_a) = θ   ← 纯净旋转

无论 t 多大, 这个等式都成立。
只要 Δu 和 Δw 源于同一组星, θ 信号就是干净的。
```

---

## 3. 阶段 B：小范围 tx/ty 搜索

### 3.1 已知 θ 下的约束

```
θ 已知 (从阶段 A) → 剩下的未知量: s, tx, ty

对每对候选 (U[i] 和 W[a]):
  若为真匹配: tx = U[i].x - s·R(θ)·W[a].x
              ty = U[i].y - s·R(θ)·W[a].y
```

### 3.2 候选提取

```
从阶段 A 的 passed_pairs 中:
  筛选: |Δθ - θ_peak| < 2°
  
  提取候选点对:
    每对 (i,j,a,b) 生成两种假设的点对应:
      假设 A: (U[i]↔W[a], U[j]↔W[b])
      假设 B: (U[i]↔W[b], U[j]↔W[a])
    
    对每个点对 (U, W):
      记录: (U 坐标, W 坐标, s_est)

  去重: (U_idx, W_idx) → 保留 s_est 最接近 1.0 的
```

### 3.3 RANSAC + Umeyama 求解

```
输入: candidate_pairs (去重后的候选点对)
参数: n_sample=2,  max_iter=5000,  tol=5px×s₀

for iter in 1..5000:
  随机抽 2 个点对
  解析求 (θ, s, tx, ty) 或直接用 Umeyama 求 (s, θ, tx, ty)
  
  全量验证:
    对每个候选点对:
      w_proj = s·R(θ)·W + (tx,ty)
      error = |U - w_proj|
      if error < tol: inlier
  
  保留 inlier 最多的变换

用最佳 inlier 集重做 Umeyama SVD → 最终 (s, θ, tx, ty)
```

### 3.4 为什么这一步容错

```
输入 candidate_pairs 已由阶段 A 的 θ 过滤:
  - 假匹配中, 角度差偏离 θ_peak 的已经排除
  - 剩余的是满足旋转约束的候选
  
  RANSAC 再进一步:
    - 2 对抽样 → 少量假匹配不影响 (真匹配率 >5% 时收敛)
    - Umeyama 最优解加最小二乘 → 输出精度高
```

---

## 4. 完整数据流

```
═══════════════════════════════════════════════════════════════
  相对向量法 (V4.4 RelVec) — 完整数据流
═══════════════════════════════════════════════════════════════

输入: N_u 颗图像星 U, N_w 颗 Gaia 星 W, s₀

离线:
  ├─ Gaia 星对全距离矩阵 D_W[i][j]
  ├─ Gaia 星对 k-vector 索引 (按距离)
  └─ 每颗 Gaia 星的邻星距离排序表

在线:
  ┌─────────────────────────────────────────────────────────┐
  │ 阶段 A: θ 分离求解 (不依赖 t)                           │
  │                                                         │
  │ K=20000 次图像星对采样                                  │
  │   for each (i,j):                                       │
  │     d_img = |U[j]-U[i]| × s₀    ← 已知绝对角距!        │
  │     k-vector: 找 |Δw| ∈ [d_img±3"] 的 Gaia 候选        │
  │     第三星验证: 三角形全等 → 通过则投票                 │
  │                                                         │
  │ θ 直方图 → 高斯平滑 → 峰值检测 → SNR 判定              │
  │                                                         │
  │ 输出: θ_peak (±0.5°), SNR, passed_pairs                │
  └─────────────────────────────────────────────────────────┘
    │
    ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 阶段 B: tx/ty 求解 (θ 已知)                             │
  │                                                         │
  │ passed_pairs 中筛选 |Δθ - θ_peak| < 2°                 │
  │ 提取候选点对 (img↔gaia) → 去重                         │
  │                                                         │
  │ RANSAC (5000次, 2对/次):                                │
  │   → Umeyama SVD → (s, θ, tx, ty) + inliers              │
  │                                                         │
  │ 输出: SimTransform + 内点列表                           │
  └─────────────────────────────────────────────────────────┘
```

---

## 5. 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| K_total | 20000 | 阶段 A 总采样次数 |
| σ_d | 3.0" | 距离容差 (3σ, 覆盖位置噪声) |
| n_third | 5 | 第三星验证数 |
| θ_bw | 1° | θ 直方图 bin 宽度 |
| θ_filter | 2° | 阶段 B θ 过滤半宽 |
| SNR_threshold | 5.0 | θ 峰值接受阈值 |
| RANSAC_n | 2 | RANSAC 最小采样对 |
| RANSAC_iter | 5000 | RANSAC 最大迭代 |
| RANSAC_tol | 5px × s₀ | RANSAC 内点容差 |

---

## 6. 与 V3.5 单点法的对比

| 维度 | V3.5 单点法 | V4.4 相对向量法 |
|------|------------|----------------|
| 信号量 | Δθ = angle(u_i) - angle(w_j) | Δθ = angle(Δu) - angle(Δw) |
| 平移依赖 | 强 (t≈0 假设) | **无** (差分消去) |
| s₀ 利用 | 间接 (通过 s = \|u\|/\|w\|) | **直接** (d_img = \|Δp\|×s₀ 绝对角距) |
| 距离查询 | 比例范围 ±10% | **绝对值范围 ±3"** |
| 采样单元 | 1 图像星 + 1 Gaia 星 | 2 图像星 + 2 Gaia 星 |
| 第三星验证 | 无 | 三角形全等 (距离约束) |
| SNR 对 t 敏感 | 是 | **否** |

---

## 7. 预期性能

| 场景 | θ 正确率 | 阶段A耗时 | 阶段B耗时 |
|------|---------|----------|----------|
| t 小 (<100") | >99% | <5ms | <2ms |
| t 中 (100"-1000") | >99% | <5ms | <2ms |
| t 大 (1000"-5000") | >98% | <5ms | <3ms |

θ 求解对 t 不敏感，仅受采样数和噪声影响。

---

## 8. 实现状态

`cpp/v4_4/` 已实现 Phase A 相对向量 + 第三星验证 + θ 投票核心。
待确认：当前 k-vector 距离查询是否使用了绝对距离（d_img 直接换算），还是仍用比例范围。

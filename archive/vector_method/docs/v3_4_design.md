# 向量匹配算法 V3.4 设计文档 (v4 — WCS-SIP标准输出)

## 1. 动机

V3.3 Record-and-Filter达到83.1%成功率，算法失败0帧，但存在2px系统性偏差。V3.4通过复用Phase A记录的采样对+全局NN匹配，快速扩充对应关系（17对→199对），中位数+3σ迭代剔除离群，一次性拟合6阶SIP，输出标准WCS格式。

实测: T2 Red帧 SNR=4360x → 17对 → 扩充至202对 → 清洗得199对 → SIP RMS=0.375px

## 2. 算法原理

### 2.1 Phase A+B (V3.3保留，不变)

```
Phase A: 无放回抽样(u_i,w_j) → s=|u_i|/|w_j|, θ=atan2(u_i)-atan2(w_j)
         → Wf→Wt → NN匹配U → s_ratio=|U[k]|/|Wf[l]| (NN距离<5×s₀)
         → θ加权直方图 → θ_SNR监控 → 达标停止
         → records: {u_idx, w_idx, θ, n_in_range_s}

Phase B: θ峰值(±2°) + n_in_range_s过滤 → 1对1互斥分配 → Umeyama SVD
         → 初始相似变换 (s₀, θ₀, tx₀, ty₀), flip_mode
```

### 2.2 Phase C: 双源扩充对应关系

**核心思想**: Phase A已经用10000次抽样投入了大量成本，每一条record都是经过s∈[0.9,1.1]和|tx,ty|<0.6·FOV预筛选的(u_i,w_j)对。Phase B只在θ峰值选了17对。现在用精确变换回验这些记录，同时补充全局NN匹配来找遗漏的对应关系。

```
输入: Phase A records[~900], Phase B精确解(s₀,θ₀,tx₀,ty₀), 全部向量U, Wf

Step 1: Source 1 — 复用Phase A记录
  Wt = s₀·R(θ₀)·Wf + (tx₀,ty₀)
  for each record (u_idx, w_idx):
    去重
    dist² = |U[u_idx] - Wt[w_idx]|²
    if dist² > (5×s₀)²: continue
    s_ratio = |U[u_idx]| / |Wf[w_idx]|
    if s_ratio < 0.9 or > 1.1: continue
    → candidates

Step 2: Source 2 — 全局NN匹配(扩大匹配池)
  对每个U[k] (k=0..N-1):
    j = KDTree最近邻Wt[j]
    if dist > 5×s₀: continue
    s_ratio = |U[k]| / |Wf[j]|
    if s_ratio < 0.9 or > 1.1: continue
    去重 → candidates

Step 3: 1对1互斥分配 (按距离升序贪心，每个U和W各用一次)
  → expanded: 100-200对确定对应关系
```

**为什么有效**: 精确变换后Wt≈U，全局NN+模长比过滤几乎全是真匹配。两种来源互补——records覆盖Phase A已验证的配对，全局NN补充遗漏。

**实测**: T2 Red帧 records去重+全局NN → 202唯一点对 → 1对1后202对

### 2.3 Phase D: 迭代中位数+3σ剔除离群

```
输入: 扩充对应关系 {U[k]↔Wf[j], 100-200对}

Step 1: Umeyama SVD → 初始刚体变换 A₀
  Wt = A₀(Wf)

Step 2: 迭代循环 (最多10轮):
  a) 计算每对残差: (dx, dy) = U[k] - Wt[j]
  b) 中位数+MAD:
     med_dx = median(dx),  σ_dx = 1.4826 × median(|dx - med_dx|)
     med_dy = median(dy),  σ_dy = 1.4826 × median(|dy - med_dy|)
  c) 3σ剔除: |dx - med_dx| < 3×σ_dx  ∧  |dy - med_dy| < 3×σ_dy
  d) 收敛: 本轮剔除数=0 → 停止
  e) 重拟合: 用保留点重做Umeyama SVD

Step 3: 最终精修 — 用所有干净对应做Umeyama SVD
  → 最终变换 (s, θ, tx, ty) + 干净对应关系
```

**实测**: T2 Red帧 202对→第1轮剔除3个→199对干净，MAD-RMS=1.675"

### 2.4 Phase E: WCS-SIP标准输出

Phase E输出标准FITS WCS格式，包含CD矩阵、CRVAL/CRPIX、6阶SIP畸变系数。SIP修正量按WCS标准定义：给定检测像素(x,y)，SIP修正后经CD矩阵投影到天球坐标。

#### 2.4.1 CD矩阵计算 (pixel → sky)

CD矩阵直接从相似变换参数推导，不依赖多项式拟合的线性项：

```
映射链: pixel(ξ,η) → U(弧秒) → Wf(弧秒) → W(弧秒) → sky(度)

其中:
  U_x = ξ·s₀,  U_y = -η·s₀          (Y翻转: 图像Y向下, 天球Dec向上)
  Wf = R(-θ)/s · U                    (逆相似变换)
  W = unflip(Wf)                      (根据flip_mode还原)
  Δα = W_x / (3600·cos(δ₀))          (弧秒→度, cos(δ)修正)
  Δδ = W_y / 3600                     (弧秒→度)

flip_mode: 0=无, 1=flip_x, 2=flip_y, 3=flip_both
  unflip: W_x = (fx ? -Wf_x : Wf_x),  W_y = (fy ? -Wf_y : Wf_y)

合并为CD矩阵:
  s0_s_3600 = s₀ / (s·3600)
  sign_x = (fx ? -1 : 1),  sign_y = (fy ? -1 : 1)

  CD1_1 = sign_x · s0_s_3600 · cos(θ) / cos(δ₀)    (ξ→Δα)
  CD1_2 = -sign_x · s0_s_3600 · sin(θ) / cos(δ₀)   (η→Δα)
  CD2_1 = -sign_y · s0_s_3600 · sin(θ)              (ξ→Δδ)
  CD2_2 = -sign_y · s0_s_3600 · cos(θ)              (η→Δδ)
```

#### 2.4.2 CRVAL计算 (天球中心)

```
CRVAL1 = center_ra - tx / (3600·cos(center_dec))
CRVAL2 = center_dec - ty / 3600
CRPIX1 = w/2,  CRPIX2 = h/2
```

其中center_ra/center_dec为图像FITS头中的原始中心坐标，tx/ty为Phase D精修后的弧秒平移。

#### 2.4.3 SIP修正量计算 (WCS标准方向)

SIP修正量定义为：给定检测像素偏移(ξ,η)，经CD逆投影得到的中间坐标(ξ',η')与(ξ,η)的差值。

```
对每对匹配 {U[k]↔Wf[j]}:

1) 检测像素偏移:
   x_det = U_x/s₀ + w/2,  y_det = -U_y/s₀ + h/2
   ξ = x_det - CRPIX1,  η = y_det - CRPIX2

2) 天球坐标差 (从Wf unflip回W):
   W_x = (fx ? -Wf_x : Wf_x),  W_y = (fy ? -Wf_y : Wf_y)
   Δα = W_x / (3600·cos(δ₀)),  Δδ = W_y / 3600

3) 中间坐标:
   [ξ', η'] = CD⁻¹ · [Δα, Δδ]

4) SIP修正目标:
   sip_dx = ξ' - ξ    (中间坐标 - 像素偏移)
   sip_dy = η' - η
```

**关键**: 必须从Wf unflip回W再计算天球坐标差，否则SIP修正量会错误放大（实测从1-3px变为-1000~-3000px）。

#### 2.4.4 SIP多项式拟合

```
Step 1: 归一化 (数值稳定性)
  x_n = ξ / (w/2),  y_n = η / (h/2)

Step 2: 构建设计矩阵 (6阶完整2D多项式 = 28项)
  A_row = [1, x_n, y_n, x_n², x_n·y_n, y_n², ..., x_n⁶, ..., y_n⁶]

Step 3: 最小二乘求解
  β_x = (AᵀA)⁻¹ Aᵀ·sip_target_x
  β_y = (AᵀA)⁻¹ Aᵀ·sip_target_y

Step 4: 提取SIP系数 (仅高阶项 p+q≥2)
  归一化反推: A_pq = β_x(idx) / ((w/2)^p · (h/2)^q)
  同理: B_pq = β_y(idx) / ((w/2)^p · (h/2)^q)

输出: CD矩阵 + CRVAL/CRPIX + 6阶SIP A/B系数
```

线性项(p+q≤1)由CD矩阵覆盖，SIP仅拟合高阶畸变修正量。实测SIP修正量1-3px，SIP RMS=0.375px。

#### 2.4.5 WCS-SIP逆投影 (sky → pixel)

标准WCS-SIP逆投影用于验证和渲染：

```
输入: 天球坐标(α, δ)

1) Δα = α - CRVAL1,  Δδ = δ - CRVAL2
2) [ξ', η'] = CD⁻¹ · [Δα, Δδ]    (天球→中间坐标)
3) 迭代求解 (通常5-7次收敛):
   ξ = ξ' - ΣA_pq·ξ^p·η^q
   η = η' - ΣB_pq·ξ^p·η^q
4) x = ξ + CRPIX1,  y = η + CRPIX2
```

## 3. 完整数据流

```
Phase A: 10000次抽样 → θ_SNR=4360x → 856条records
    ↓
Phase B: θ峰值+n_range过滤 → 17对 → SVD → (s₀,θ₀,tx₀,ty₀)
    ↓
Phase C:
  Source 1: records去重+精确变换验证 → 约15对
  Source 2: 全局NN匹配(全部U×Wt) + 模长比 → 约190对
  1对1互斥 → 202对
    ↓
Phase D: 迭代MAD3σ → 剔除3个 → 199对干净 → 最终Umeyama精修
    ↓
Phase E:
  CD矩阵 ← 相似变换参数(s,θ,flip_mode,cos(δ))
  CRVAL ← 原始中心 + 平移偏移
  SIP修正量 ← CD⁻¹·[Δα,Δδ] - [ξ,η] (Wf unflip回W)
  6阶多项式拟合 → SIP A/B系数
    ↓
输出: WCS参数(CRVAL, CRPIX, CD矩阵) + 6阶SIP畸变模型
      → JSON文件 (绕开ctypes对齐问题)
```

## 4. 关键设计决策

### 4.1 双源扩充

Source 1 (records)利用Phase A的投资，Source 2 (全局NN)覆盖遗漏。两者互补，通常可得100-200对对应关系。

### 4.2 零额外抽样成本

Phase C不需要新的随机抽样。records复用+全局NN匹配的KDTree操作是O(N log M)，几乎瞬间完成。

### 4.3 迭代3σ剔除

每轮剔除离群→重拟合→残差更集中→σ下降→暴露新离群。通常1-2轮收敛，剔除率<5%。

### 4.4 CD矩阵直接计算 vs 多项式提取

CD矩阵直接从相似变换参数推导，而非从多项式线性项提取。原因：
- 多项式线性项受归一化影响，提取CD时容易出错（之前linear-only RMS=1827px）
- 直接计算保证CD矩阵与相似变换参数完全一致
- CD⁻¹用于计算SIP修正目标值，CD精度直接影响SIP质量

### 4.5 SIP修正量按WCS标准定义

SIP修正量 = CD⁻¹·[Δα,Δδ] - [ξ,η]，而非det_pixel→cat_pixel的完整映射残差。原因：
- WCS-SIP标准定义：ξ' = ξ + ΣA_pq·ξ^p·η^q，SIP修正量 = ξ' - ξ
- 完整映射残差包含线性分量（已被CD覆盖），会导致SIP系数包含线性项
- 标准定义下SIP仅拟合非线性畸变，线性项(p+q≤1)自动为零

### 4.6 JSON文件传递WCS参数

C++输出JSON文件传递CD/SIP/CRVAL/CRPIX，Python从JSON读取。原因：
- Windows x64 + MSVC下ctypes结构体字节对齐复杂（double 8字节对齐、int 4字节对齐、指针8字节对齐）
- 之前CD矩阵通过ctypes读取为-3.9e-21（实际2.686e-4），JSON完全规避此问题
- JSON可读性好，便于调试和验证

### 4.7 Wf unflip回W

计算天球坐标差时，必须从Wf unflip回W（原始gnomonic投影坐标）。Wf是flip后的坐标，直接用于计算Δα/Δδ会导致SIP修正量错误放大（实测从1-3px变为-1000~-3000px）。

## 5. 参数表

| 参数 | 默认值 | 说明 | 来源 |
|------|--------|------|------|
| K_total | min(10000, N×M) | Phase A抽样次数 | V3.3 |
| batch_size | 1000 | SNR检查间隔 | V3.3 |
| min_samples | 2000 | 最少抽样 | V3.3 |
| s_min/max | 0.90/1.10 | 模长比有效范围 | V3.3 |
| max_dist | 5×s₀ | NN距离上限 | V3.3 |
| θ_band | 2.0°/4.0° | Phase B过滤 | V3.3 |
| sigma_factor | 3.0 | Phase D剔除倍数 | V3.4 |
| sip_order | 6 | 多项式阶数 | V3.4 |

## 6. 实测结果 (T2 Red帧, M20, 1.56° FOV)

| 指标 | V3.3 | V3.4 |
|------|------|------|
| Phase B对应关系 | 17对 | 17对 |
| Phase C扩充 | — | **202对** |
| Phase D清洗后 | — | **199对** |
| 投影RMS | 2.71px | — |
| SIP拟合RMS | — | **0.375px** |
| SIP修正量范围 | — | 1-3px |
| 刚体残差(Phase D) | — | 1.68" |
| 解析时间 | 0.02s | 0.07s |
| CD矩阵 | — | 直接计算(含flip+cos(δ)) |
| WCS输出 | — | **标准FITS WCS+SIP** |

## 7. WCS输出格式

### 7.1 JSON文件 (C++ → Python)

```json
{
  "CD": [[CD1_1, CD1_2], [CD2_1, CD2_2]],
  "CRVAL": [RA_center, Dec_center],
  "CRPIX": [x_ref, y_ref],
  "SIP_A": [36个A系数, p*6+q索引],
  "SIP_B": [36个B系数, p*6+q索引],
  "RMS_PX": 0.375
}
```

### 7.2 FITS WCS头关键字 (待实现)

```
CRVAL1  = RA中心 (度)
CRVAL2  = Dec中心 (度)
CRPIX1  = 参考像素x
CRPIX2  = 参考像素y
CD1_1   = ∂RA/∂x (度/像素)
CD1_2   = ∂RA/∂y (度/像素)
CD2_1   = ∂Dec/∂x (度/像素)
CD2_2   = ∂Dec/∂y (度/像素)
A_ORDER = 6
A_2_0   = SIP系数
...
B_ORDER = 6
B_2_0   = SIP系数
...
```

## 8. 实现文件

- `cpp/vector_match_v3_4/include/vm34_api.h` — C API头文件 (含center_ra/center_dec, wcs_out_path)
- `cpp/vector_match_v3_4/src/vm34_core.cpp` — C++核心 (~850行)
- `cpp/vector_match_v3_4/Makefile` — 编译脚本
- `python/vector_match_v3_4_cpp.py` — Python ctypes封装 (_pack_=8对齐, JSON读取WCS)
- `docs/v3_4_design.md` — 本文档

## 9. 已解决的技术问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| CD矩阵ctypes读取为-3.9e-21 | 结构体字节对齐不一致 | JSON文件传递 + _pack_=8 |
| CRVAL=[0,0] | 未传入图像中心坐标 | VM34SolveParams添加center_ra/center_dec |
| SIP修正量-3000px | Wf未unflip回W | unflip后再计算Δα/Δδ |
| CD矩阵从多项式提取RMS=1827px | 归一化空间反推出错 | CD直接从相似变换参数计算 |
| SIP方向错误 | 拟合det→cat完整映射 | 改为WCS标准: SIP=CD⁻¹·[Δα,Δδ]-[ξ,η] |
| SIP迭代overflow | 远离帧的星点ξ/η高次幂溢出 | 粗筛帧内星点再做SIP迭代 |

# 向量匹配算法 V3.5 设计文档

## 1. 动机

V3.4实现了完整的WCS-SIP标准输出。V3.5发现拟合精度糟糕且有系统性误差，根因是Phase C全局NN扩充和Phase D'星点扩增引入了大量假匹配对。V3.5核心改动：

1. **移除Phase C/D/D'所有NN扩充环节**：这些环节在变换不完全精确时引入大量假匹配对，导致拟合精度恶化和系统性偏移
2. **仅保留Phase A+B信噪比验证的匹配对**：经过s∈[0.9,1.1]、|tx,ty|<0.6·FOV、s-in-range统计、θ峰值验证四重筛选，是唯一可信的对应关系
3. **新增向量残差中值离群预过滤**：Phase B验证对中仍有少量假匹配，用Umeyama→残差→MAD 3σ过滤剔除
4. **修正仿射CD/CRVAL更新公式**：原公式拟合方向反转导致CRVAL修正符号错误，已修正为det = A·gaia + t → CD' = CD·A⁻¹ → CRVAL' = center − CD'·t
5. **始终执行Layer 0+1（甚至<30对）**：少对场景跳过SIP但必须跑完仿射精修CD/CRVAL，否则CRVAL无修正导致系统偏移

**核心原则：Phase A+B SNR验证对 → 向量离群过滤 → Layer 0(CD) → Layer 1(仿射) → [Layer 2(SIP)]。零NN引入。**

## 2. 算法原理

### 2.1 Phase A: 抽样+θ直方图+SNR监控

V3.5最终版：N=250 + 稀疏度加权抽样 + 5N/10N SNR停止。

```
Phase A: 抽样(u_i,w_j) → s=|u_i|/|w_j|, θ=atan2(u_i)-atan2(w_j)
         → Wf→Wt → NN匹配U → s_ratio=|U[k]|/|Wf[l]| (NN距离<5×s₀)
         → θ加权直方图 → θ_SNR监控 → 达标停止(≥5N或≥10N)
         → records: {u_idx, w_idx, θ, n_in_range_s}
```

**稀疏度加权**: 计算U和W的局部星点密度(sparsity=第3近邻距离)，70%概率选密度排名相近的对，30%均匀随机。正确θ对应的U/W在对应区域密度相似，加权后正确对的θ峰值更高。

**参数**: N=250(饱和星+亮星补足), K_total=10000, batch_size=1000, min_samples=2000。

### 2.2 Phase B: 三级放宽过滤 + SVD

```
Phase B: θ峰值 + n_in_range_s过滤 → 1对1互斥分配 → cu/cw匹配对 + Umeyama SVD
         → 相似变换 + flip_mode + M_D对验证匹配对

三级放宽过滤:
  Level 1: n_in_range_s > 1.5×median, θ在峰值±2°内
  Level 2: n_in_range_s > 1.0×median, θ在峰值±4°内 (若Level 1 < 2对)
  Level 3: n_in_range_s ≥ 1, θ在峰值±8°内 (若Level 2 < 2对)
```

**输出**: M_D对cu/cw匹配对（通常20~30对）+ 相似变换参数。这些对经过四重筛选，是唯一可信的对应关系。

### 2.3 Phase C: 分层拟合（vm35_core.cpp fit_affine_sip_adaptive）

#### 2.3.0 Pre-filter: 向量残差中值离群排异

Phase B 的 M_D 对中仍有少量假匹配。用 Umeyama 求初始变换 → 算每对残差向量 → dx/dy 分别 MAD 3σ 过滤：

```
Input: clean_u, clean_w (M_D对)
Step 1: Umeyama(Wf[clean_w], U[clean_u]) → (s_pre, θ_pre, tx_pre, ty_pre)
Step 2: 残差向量: res_dx[i]=U[ui]-Wt_x, res_dy[i]=U[ui]-Wt_y
Step 3: med_dx = median(res_dx), σ_dx = MAD(res_dx)×1.4826 (dy同理)
Step 4: for each i: if max(|res_dx[i]-med_dx|/σ_dx, |res_dy[i]-med_dy|/σ_dy) ≥ 3 → 剔除
Output: clean_u_f, clean_w_f (过滤后对)
```

**原理**: 正确匹配的残差向量分布集中，假匹配的残留向量方向/幅度异常。MAD统计自然将其滤出。实验中NGC7293 30→27对，GC_P1 28→20对（29%假匹配被清除）。

#### 2.3.1 Layer 0: Umeyama弧秒初值 + 符号验正 → CD

```
Step 1: Umeyama(Wf[clean_w_f], U[clean_u_f]) → (s,θ,tx,ty)
Step 2: 符号验正 — θ有±π歧义，两种变体各计算CD投影RMS，选RMS更低的
  变体A: θ         → CD_A, rms_A
  变体B: θ+π       → CD_B, rms_B  选rms更低的
Step 3: CD矩阵 (pixel → sky):
  CD1_1 = sign_x · s₀/(s·3600) · cos(θ) / cos(δ₀)
  CD1_2 = -sign_x · s₀/(s·3600) · sin(θ) / cos(δ₀)
  CD2_1 = -sign_y · s₀/(s·3600) · sin(θ)
  CD2_2 = -sign_y · s₀/(s·3600) · cos(θ)
```

#### 2.3.2 Layer 1: MAD迭代剔除outlier + 全6-DOF仿射 → 更新CD/CRVAL

```
Step 4: 像素投影:
  src[i] = 检测星像素(相对CRPIX) = U[ui]/s0 + crpix
  dst[i] = CD⁻¹ · [Δα, Δδ] (Gaia星线性投影, Wf[wi]→弧秒→CD⁻¹)
Step 5: MAD迭代剔除 (最多3轮):
  MAD = median(|残差|) × 1.4826
  threshold = max(5px, 3×MAD)
  剔除残差 > threshold的点
Step 6: 全6-DOF仿射: det = A · gaia + t
  L = [gaia_x, gaia_y, 1, 0, 0, 0; 0, 0, 0, gaia_x, gaia_y, 1]
  R = [det_x; det_y]
  ab = (LᵀL)⁻¹LᵀR → a00,a01,tx,a10,a11,ty
Step 7: CD/CRVAL更新 (关键修正):
  CD' · gaia = CD · gaia  (同一颗星的天空坐标不变)
  CD' · (A⁻¹ · (det - t)) = CD · gaia  → CD' = CD · A⁻¹
  CRVAL' = center − CD' · t
```

**V3.5关键修正**: 原公式拟合方向为gaia=A·det+t，导致CD'=CD·A与CRVAL'=center+CD'·t符号错误。修正后仿射残差在匹配点处均值归零（GC_P1: mean=0.00±0.45px, GC_P2: mean=0.00±0.57px）。

#### 2.3.3 Layer 2: SIP BIC逐阶拟合（≥2阶，仅sip_order>0且M_clean≥5时执行）

```
Step 8: 仿射后残差(dst_new - src) = 纯非线性畸变
Step 9: BIC选阶 (2~max_order=4):
  归一化: u=(x-CRPIX)/(w/2), v=(y-CRPIX)/(h/2)
  设计矩阵: A(i, col) = u^p · v^q  (p+q≥2)
  最小二乘: β_x, β_y
  RMS = √(SSQ/M_clean)
  BIC = M_clean·log(RMS²) + 2·nterms·log(M_clean)
  选BIC最低的阶数
Step 10: SIP系数反归一化:
  A[p][q] = β_x(idx) / ((w/2)^p · (h/2)^q)
  B[p][q] = β_y(idx) / ((w/2)^p · (h/2)^q)
```

#### 2.3.4 少对场景（n_pairs < 30）

```
if n_pairs < 30: sip_order = 0
→ 只执行 Layer 0 + Layer 1（CD精修 + 仿射CRVAL修正），跳过Layer 2 SIP
→ 理由: 20-30对匹配点不足以稳健估计高阶SIP系数（BIC选阶不可靠）
→ 仿射层始终保持执行，确保CRVAL正确吸收系统偏移
```

#### 2.3.5 s从CD行列式提取

```
s = s₀ / (3600 × √(|CD| × cos(δ)))
```

### 2.4 模式选择策略

```
4个模式并行运行，选择最优:
  n≥5: 选norm_score最高  (匹配对更多更可靠)
  n<5: 选SNR最高       (θ峰值更可靠，避免n=2且θ错误的模式)
```

### 2.5 early_exit条件

```
s在[0.9,1.1]内 且 n_inliers ≥ min_inliers 才触发
```

## 3. 完整数据流

```
Phase A (4mode并行): 1点SNR抽样 + 稀疏度加权 + θ直方图 + 5N/10N停止
    ↓
Phase B: 三级放宽过滤 → 1对1互斥 → cu/cw匹配对 + Umeyama SVD
    → M_D对验证匹配对 + (s,θ,tx,ty,flip)
    ↓
Phase C (vm35_core.cpp fit_affine_sip_adaptive):
  Pre-filter: 向量残差中值离群(MAD 3σ) → 剔除少量假匹配
  Layer 0: Umeyama弧秒 → CD + 符号验正(θ/θ+π选RMS更低)
  Layer 1: 像素MAD剔除outlier(3轮) → 全6-DOF仿射 → CD'=CD·A⁻¹, CRVAL'=center-CD'·t
  Layer 2: (仅sip_order>0且M_clean≥5) 仿射残差上SIP BIC选阶(2~4阶)
  s ← 从CD行列式提取
    ↓
模式选择: n≥5选norm_score最高; n<5选SNR最高
    ↓
输出: WCS参数(CD/CRVAL/CRPIX) + SIP畸变系数(A/B) + MATCH_PAIRS → JSON文件
```

## 4. 关键修正记录

| 修正 | 根因 | 影响 |
|------|------|------|
| 仿射拟合方向反转 | 原公式 `gaia=A·det+t`，CD'/CRVAL推导错误 | CD'/CRVAL方向反，残差系统性+24px |
| `n<30→order=0`跳过全流程 | order=0触发Layer0/1/2全程跳过 | CRVAL无修正，宽视场数十px偏移保留 |
| Pre-filter加入 | Phase B验证对中仍有假匹配 | 少对场景假匹配占比高(GC_P1 29%)，污染拟合 |
| MATCH_PAIRS存绝对RA/Dec | 存Wf弧秒，Python当作相对CRVAL偏移投影 | 可视化黄色箭头全部偏向同一方向 |
| 网格均匀取星 | 按星等全局Top-N致红色十字集中在右下 | 可视化无法覆盖全幅面 |

## 5. 可视化调试脚本

### 5.1 标准控制点对可视化调试示例.py

**路径**: [`lib/plate_solve/scripts/v3_5/标准控制点对可视化调试示例.py`](file:///f:/Astro%20dev/Astro%20CS%20Normalization%20Database/lib/plate_solve/scripts/v3_5/标准控制点对可视化调试示例.py)

**三层叠加**:

| 图层 | 颜色/样式 | 数据源 | 含义 |
|------|----------|--------|------|
| 红色十字 `+` | `color='red'` | Gaia前1000亮星 → CD⁻¹+SIP投影 | WCS模型在天球上的**纯预测**，与检测/匹配无关 |
| 蓝色圆圈 `○` | `fc='none', ec='cyan'` | WCS JSON `MATCH_PAIRS` 中的检测星 | Phase B信噪比验证+Pre-filter过滤后的**真实匹配星** |
| 黄色箭头 `→` | `fc='yellow', ec='yellow'` | Gaia投影位置 → 检测星位置 | 残差向量（预测偏差） |

**Gaia查询策略**:
```
查询半径: max(FOV×0.65, 3.5°)
查询星等: mag=22
查询中心: CRVAL（精修后WCS参考点，确保查询区与投影区对齐）
```

**网格均匀取星** (32×32网格，每格取最亮星):
```
for gi in 0..31:
    x0,x1 = gi*w/32, (gi+1)*w/32
    for gj in 0..31:
        y0,y1 = gj*h/32, (gj+1)*h/32
        取该网格内mag最小的星
→ 保证全幅面均匀覆盖（最多1000个），而非集中在亮星密集区
```

**依赖**: ImageReader, StarDetector, VectorMatchV35Cpp, GaiaClientPy, CD⁻¹+SIP投影，matplotlib

**输出**: `overlay_output/_cp_{label}.png` — 全尺寸无边框灰度底图叠加三层标注

### 5.2 独立拟合验证

**路径**: [`lib/plate_solve/scripts/v3_5/test_fit_standalone.py`](file:///f:/Astro%20dev/Astro%20CS%20Normalization%20Database/lib/plate_solve/scripts/v3_5/test_fit_standalone.py)

将Phase C管线剥离为独立Python实现，用于合成数据验证：
- 合成200对+σ=0.3px → 线性RMS=0.42px ✓
- 合成110对+σ=0.5px → 线性RMS=0.68px ✓
- 与C++输出逐项对比验证一致性

## 6. V3.5 vs V3.4 改动汇总

| 改动 | V3.4 | V3.5最终版 | 原因 |
|------|------|-----------|------|
| Phase C全局NN扩充 | 有 | **移除** | 变换不完全精确时NN引入大量假匹配 |
| Phase D迭代MAD清洗 | 有 | **移除** | 清洗也无法根除假匹配污染 |
| Phase D'星点扩增 | 无 | **移除** | 双向NN同样引入大量假匹配 |
| 匹配对来源 | Phase C+D扩充(100-200对) | Phase B验证(20~30对) | 纯度>数量 |
| Pre-filter(向量离群) | 无 | **新增** | 20-30对中仍有个别假匹配 |
| 仿射更新公式 | gaia=A·det+t, CD'=CD·A | **修正** det=A·gaia+t, CD'=CD·A⁻¹ | 方向反转导致CRVAL符号错误 |
| <30对order=0 | 跳过全流程(CD/CRVAL无修正) | **修正** 始终执行Layer0+1，仅跳SIP | CRVAL无修正导致数十px系统偏移 |
| SIP阶数 | 固定6阶(28项) | BIC选阶(2~4阶) | 6阶过拟合 |
| Phase B过滤 | 两级 | 三级(最终±8°, n≥1) | 确保总有足够对 |
| N_img | 100 | 250 | 饱和星+亮星满250 |
| 抽样策略 | 均匀随机 | 稀疏度加权(70%)+均匀(30%) | 优先密度近似对 |
| early_exit | s在范围内即触发 | s在范围内且n≥min_inliers | 防止n=0误触发 |
| 模式选择 | norm_score最高 | n≥5选norm_score; n<5选SNR | 低匹配时θ峰值更可靠 |
| MATCH_PAIRS | 存Wf弧秒偏移 | **存绝对RA/Dec** | Python端用WCS标准投影 |
| 可视化 | 5px单方向NN(数千假对) | 仅PhaseB验证对 + 网格取星 | 精确显示真实匹配 |

### 不改的部分

- 4个mode并行（覆盖360°）
- Phase A核心逻辑（抽样+θ直方图+SNR）
- Phase B匹配对提取（三级过滤+1对1互斥+SVD）
- JSON文件传递方式
- C++核心+Python ctypes架构

## 7. 参数表

| 参数 | 默认 | 说明 |
|------|------|------|
| K_total | 10000 | 最大抽样次数 |
| batch_size | 1000 | SNR检查间隔 |
| SNR停止 | 5N/10N | SNR≥5N或10N时停止 |
| min_samples | 2000 | 最少抽样 |
| s_min/max | 0.90/1.10 | 模长比范围 |
| N_img | 250 | 图像星点数 |
| θ_band | 2.0°/4.0°/8.0° | Phase B三级过滤 |
| sip_order | 4 (n≥30), 0 (n<30) | 最大SIP阶数 |
| MAD 3σ in pre-filter | 3.0 | 向量残差离群阈值 |

## 8. 测试结果

### 8.1 3帧验证集

| 帧 | 望远镜 | 滤镜 | 原始对 | Pre-filter后 | 拟合 | 残差均值 | 残差std |
|-----|--------|------|--------|-------------|------|---------|---------|
| NGC7293 | 1917mm T2 | H-alpha | 30 | 27 | SIP阶4 RMS=2.0px | ~0.0 | ~0.3px |
| GC_P2 | 200mm T4 | Oiii | 25 | 22 | 仿射(无SIP) RMS=13.4px | 0.00 | 0.57px |
| GC_P1 | 200mm T4 | Red | 28 | 20 | 仿射(无SIP) RMS=31.3px | 0.00 | 0.45px |

### 8.2 已知局限

- **GC_P2 全幅面覆盖不足**: 22个匹配点全部集中在帧右侧x=[3774,4496]（像素宽4500的16%），CD外推到左侧不可靠。需更多匹配点或放宽Phase B过滤条件
- **少对场景SIP跳过**: <30对时不拟合SIP，用纯仿射CD投影。对光学畸变大的系统可能残留非线性残差
- **SIP系数归一化**: C++和Python的SIP计算使用不同坐标系统，当前跳过SIP的帧不存在此问题

## 9. 实现文件

- `cpp/vector_match_v3_5/include/vm35_api.h` — C API头文件
- `cpp/vector_match_v3_5/src/vm35_core.cpp` — C++核心
- `cpp/vector_match_v3_5/Makefile` — 编译脚本
- `python/vector_match_v3_5_cpp.py` — Python ctypes封装
- `scripts/v3_5/标准控制点对可视化调试示例.py` — 可视化调试（三层叠加）
- `scripts/v3_5/test_fit_standalone.py` — 独立拟合验证
- `docs/v3_5_design.md` — 本文档

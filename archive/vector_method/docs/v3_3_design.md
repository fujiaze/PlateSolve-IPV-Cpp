# 向量匹配算法 V3.3 设计文档 (v5 final)

## 1. 动机

V3.2用1点法建θ直方图+GridSearch+Hough估计(tx,ty)，成功定位θ但(tx,ty)精度不足。

**核心洞察**：1点法抽样(u_i,w_j)→计算变换→变换Wf→NN匹配U→统计s_in_range。正确变换的s_in_range远高于噪声。**θ_SNR达标后，θ过滤+s_in_range过滤留下的配对就是正确的U↔Wf对应关系，直接SVD得精确解。**

## 2. 算法原理

### 2.1 一次抽样的流程

```
1. 无放回选 (u_i, w_j), i∈[0,N), j∈[0,M)
2. s = |u_i| / |w_j|, 必须在[0.9, 1.1]内
3. θ = atan2(u_i) - atan2(w_j)
4. tx,ty 由变换方程确定, 物理约束|tx|,|ty| < 0.6×FOV
5. 用(s,θ,tx,ty)变换星表Wf → Wt
6. 对每个图像向量U[k], KDTree找Wt最近邻→对应Wf[l]
   需满足NN距离 < 5×s0（确保空间位置接近）
   计算s_ratio[k] = |U[k]| / |Wf[l]|
   统计s_ratio落在[0.9, 1.1]内的数量 = n_in_range
7. 记录配对(i,j, θ, n_in_range)
8. θ直方图[θ_bin] += n_in_range  (内点数加权)
```

### 2.2 为什么有效

- **正确配对**：s正确→Wt≈U→NN命中正确Wf→s_ratio≈1→n_in_range≈min(N,M)
- **错误配对**：s偏差大→Wt整体缩放错误→NN可能命中错误Wf且距离大→NN距离<5×s0过滤大量噪声→s_ratio偏离
- **θ加权直方图放大信号**：正确配对贡献n_in_range≈200到峰值bin，噪声贡献1-2到随机bin
- **θ_SNR = peak_val / bg_mean**：正确模式的θ_SNR可达4000x

### 2.3 停止条件

```
θ_SNR ≥ min(10N, 1000) → 提前结束
θ_SNR ≥ min(5N, 500)  → 正常停止
θ_SNR < min(5N, 500)  → 继续(至K_total)
```

上限设计：宽视场N小时防止5N过高无法触发。

## 3. 完整流程

```
Phase A: Record — 无放回抽样, 加权θ直方图, SNR停止
  precompute: norm_U[], angle_U[], norm_Wf[], angle_Wf[]
  theta_hist[3600] = {0}
  sampled_pairs = HashSet (uint64 key = i*M+j)
  records = []

  loop (max = min(K_total, N×M)):
    无放回抽(i,j), s∈[0.9,1.1], |tx|,|ty|<0.6×FOV
    Wt = s·R(θ)·Wf + (tx,ty)
    n_range = count_s_in_range(U, Wt, norm_U, norm_Wf, max_dist=5×s0)
    theta_hist[(θ+180)/0.1] += n_range
    records.push({i, j, θ, n_range})
    每batch_size次: 检查θ_SNR, 达标则停止

Phase B: Filter → 对应关系 → SVD
  median_noise = median(records[].n_range)
  θ_peak = theta_hist峰值位置
  noise_thr = max(2.0, 1.5×median), θ_band = 2°

  filtered = {n_range > noise_thr && |θ - θ_peak| < θ_band}
  if len < 2: 回退(θ_band=4°, noise_thr=median)

  1对1互斥分配(按n_range降序贪心)
  Umeyama SVD (from Wf↔U correspondences)
  迭代SVD精修 (≤10 iter)

输出: (s, θ, tx, ty) + inlier_mask + RMS
```

## 4. 关键设计决策

### 4.1 s-in-range vs KDTree距离计数

核心差异：不是看变换后Wt与U的距离，而是看**每个U点找最近Wt对应的Wf，算模长比s_ratio**。s_ratio需要结合NN距离约束（<5×s0）防止噪声匹配。

### 4.2 单阶段，无二阶段精修

Phase B从配对中提取的U↔Wf对应关系是真正的点对点匹配。SVD直接一次给出精确解，Python端无需中心修正→重投影→再求解的二阶段。

### 4.3 无min_inliers门槛

θ_SNR达标意味着5-10对正确配对已被确认，Phase B过滤后的对应关系不可能为假。SVD结果直接信任。

### 4.4 无放回抽样

`uint64_t key = i*M+j` 存入`unordered_set`，防重复。K_max = min(K_total, N×M)。

## 5. 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| K_total | min(10000, N×M) | 总有效抽样次数 |
| batch_size | 1000 | SNR评估间隔 |
| min_samples | 2000 | 最少抽样次数 |
| s_min / s_max | 0.90 / 1.10 | s有效范围 |
| max_dist | 5×s0 | NN距离上限(角秒) |
| min_inliers | max(5, 0.1×N) | (API保留) |
| fov_diag_asec | FOV对角线 | 物理约束 |

**θ_SNR阈值**：≥min(10N,1000)→提前, ≥min(5N,500)→正常, <min(5N,500)→继续

**Phase B过滤**：noise_thr=max(2,1.5×median) θ_band=2°；回退：noise_thr=max(1,1×median) θ_band=4°

## 6. 架构对比

| 特性 | V2 | V3.2 | V3.3 |
|------|-----|------|------|
| 抽样 | 2点有放回 | 1点有放回 | **1点无放回** |
| 度量 | KDTree距离 | KDTree距离 | **s-in-range+NN距离** |
| Phase B | 无 | GridSearch+Hough | **θ+度量双过滤→SVD** |
| 二阶段 | 无 | 中心修正重投影 | **不需要** |
| 停止条件 | RANSAC迭代 | SNR收紧→GridSearch | **θ_SNR vs 5N/10N** |
| 对应关系 | 无 | 无 | **抽到的配对本身就是** |

## 7. 测试结果（562有效帧, GaiaDR3, s-in-range版）

| 望远镜 | V2 | V3.2 | V3.3 |
|--------|-----|------|------|
| T1 | ~95% | 53.8% | **100%** |
| T2 | 0% | 73.2% | **76.3%** |
| T3 | 0% | 58.3% | **62.2%** |
| T4 | ~95% | 27.4% | **100%** |
| **总计** | — | 55.2% | **83.1%** |

| 指标 | 值 |
|------|-----|
| RMS中位 | 0.51px |
| 解析中位 | 0.02s |
| 算法失败 | 0帧 |

## 8. 失败分析

算法失败0帧。467帧有Gaia数据的全部成功。剩余95帧Gaia查询M=0（NGC4945/NGC247/NGC7293等南天目标或天区边缘）。

## 9. 实现文件

- `cpp/vector_match_v3_3/src/vm33_core.cpp` — C++核心 (~700行)
- `cpp/vector_match_v3_3/include/vm33_api.h` — C API头文件
- `python/vector_match_v3_3_cpp.py` — Python ctypes包装 (~340行)
- `scripts/test_v33_cpp_single.py` — 单帧测试
- `scripts/test_v33_cpp_batch.py` — 批量测试

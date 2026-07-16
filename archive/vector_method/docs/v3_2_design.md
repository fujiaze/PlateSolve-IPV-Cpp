# 向量匹配算法 V3.2 设计文档

## 1. 问题背景

V2算法在T4望远镜（WCS旋转≈0°）上成功率98.1%，但在T2/T3望远镜（WCS旋转≈±90°）上成功率0%。

V3.1用蒙特卡洛1点法替代RANSAC 2点法，T2帧成功但T4帧（宽视场FOV=9.9°）失败。
V3.2引入SNR动态收紧1点抽样法，解决宽视场帧问题。

## 2. 关键发现

### 发现1：RANSAC 2点法在大旋转角时失效
- **原因**：RANSAC需要同时选到2个正确匹配对，P(两点都正确) ≈ 10⁻⁵
- **结论**：1点法（只需1个正确对）是更好的选择

### 发现2：1点法+内点数加权θ直方图可以检测正确变换
- 加权后SNR从7x提升到426x（V3.1已验证）

### 发现3：θ精度是宽视场帧的关键瓶颈
- **验证**：T4帧（FOV=9.91°），θ=0°时max_n=83，θ=0.5°时max_n=6
- **结论**：θ精度要求<0.5°，预热直方图1°分辨率不够，需要θ精搜索

### 发现4：s精度同样关键
- **验证**：T4帧，s=0.982时max_n=83，s=0.984时max_n=53
- **结论**：s精度要求<0.4%（0.004），1点法s不够准，需要s精搜索

### 发现5：固定(θ,s)后，1点法只需1个正确对计算(tx,ty)
- **结论**：将4D搜索(s,θ,tx,ty)降为2D搜索(θ,s)，tx/ty由抽样自动确定

### 发现6：Mode 1和Mode 2在θ≈±90°时数学等价
- R(-90°)·flip_x = R(+90°)·flip_y，产生相同的变换矩阵

### 发现7：1点法s直方图峰值不可靠
- 1点法的s受投影畸变影响偏差0.37%，s直方图峰值偏离真实s
- **结论**：不能依赖1点法s直方图来收紧s搜索范围，s应使用全范围网格搜索

## 3. V3.2核心思路：SNR动态收紧1点抽样法

### 3.1 算法思想

将1点法的4D参数搜索(s,θ,tx,ty)分解为两个阶段，**由SNR动态驱动**：

1. **Phase A（1点法抽样+SNR动态收紧θ）**：
   - 随机选(u_i, w_j)对，计算变换参数(s, θ, tx, ty)
   - 建立θ加权直方图，计算峰值SNR
   - SNR达到阈值（3x）→ 收紧θ搜索范围到峰值附近
   - θ收紧到1°以内 → 切换到Phase B

2. **Phase B（网格搜索(θ,s) + Hough-like (tx,ty)估计）**：
   - θ范围：Phase A峰值±2°（因为1点法θ峰值可能有0.5-1°偏差）
   - s范围：全范围[s_min, s_max]（不依赖1点法s直方图）
   - **Hough-like (tx,ty)估计**：对固定(θ,s)，用KDTree查询所有U点找近邻，
     从近邻对中估计(tx,ty)，替代K次随机抽样
   - 粗搜：θ步长0.2°, s步长0.02
   - 精搜：θ步长0.1°, s步长0.002

### 3.2 Phase B的Hough-like (tx,ty)估计

核心优化：替代K次随机抽样(tx,ty)，用KDTree直接估计最优(tx,ty)。

**算法**：
1. 对固定(θ,s)，计算sR_Wf = s·R(θ)·Wf（不含平移）
2. 构建KDTree（固定(θ,s)只建一次）
3. 对每个U[i]，找sR_Wf中最近邻j，如果距离<tau_large：
   - (tx,ty) = U[i] - sR_Wf[j] 是一个候选
4. 用1D直方图找tx和ty的峰值
5. 用峰值(tx,ty)做内点计数
6. 在峰值附近精修(tx,ty)

**tau_large选择**：max(5×tau, FOV×0.005)
- 1点法s偏差0.37%导致(tx,ty)偏移可达FOV×0.004
- 需要足够大的tau_large覆盖这种偏移

### 3.3 剪枝优化

| 优化 | 描述 | 效果 |
|------|------|------|
| tx,ty物理约束 | \|tx\|或\|ty\|>0.6×FOV对角线→跳过 | Phase A+Phase B |
| 空间重叠检查 | sR_Wf和U无空间重叠→跳过该(θ,s) | Phase B |
| KDTree缓存 | 固定(θ,s)只建一次KDTree | Phase B |
| Hough-like | KDTree近邻估计(tx,ty)替代K次随机抽样 | Phase B核心加速 |

### 3.4 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `snr_theta_tighten` | 3.0x | θ收紧的SNR阈值 |
| `theta_band_init` | 5.0° | θ初始搜索半带宽 |
| `theta_band_min` | 0.1° | θ最小搜索半带宽 |
| `n_max` | 500000 | 最大总抽样次数 |
| `batch_size` | 1000 | Phase A每批抽样次数 |
| `theta_step_coarse` | 0.2° | Phase B粗搜θ步长 |
| `s_step_coarse` | 0.02 | Phase B粗搜s步长 |
| `theta_step_fine` | 0.1° | Phase B精搜θ步长 |
| `s_step_fine` | 0.002 | Phase B精搜s步长 |
| `tau_large` | max(5×tau, FOV×0.005) | Hough-like搜索范围 |

## 4. V3.2算法流程

```
输入: U(N×2), W(M×2), s0, tau, n_max
输出: 最佳变换参数(s, θ, tx, ty)

对每种翻转模式 mode=0..3 (OpenMP并行):
  1. Wf = apply_flip(W, mode)

  2. Phase A: 1点法抽样 + SNR动态收紧θ
     while n_samples < n_max:
       随机选(u_i, w_j), 计算(s, θ, tx, ty)
       s∈[0.9,1.1]筛选, |tx|,|ty|<0.6×FOV筛选
       快速内点计数, 更新θ加权直方图
       每batch_size次:
         计算θ峰值SNR
         SNR >= snr_theta_tighten → 收紧θ范围
         θ_band <= 1° → 切换到Phase B

  3. Phase B: 网格搜索(θ,s) + Hough-like (tx,ty)
     对每组(θ,s):
       空间重叠检查: sR_Wf和U是否有重叠?
       构建KDTree (固定(θ,s)只建一次)
       Hough-like: 对每个U[i]找sR_Wf近邻, 估计(tx,ty)
       tx,ty物理约束剪枝
       1D直方图找tx/ty峰值
       用峰值(tx,ty)做内点计数
       在峰值附近精修(tx,ty)

  4. 用最佳变换做1对1内点统计 + SVD精修

选择4种模式中norm_score最高的作为最终结果
```

## 5. 验证结果

### 验证1-3: θ精度/s精度/1点法s直方图不可靠
（同前，见第2节发现3/4/7）

### 验证4：T4帧完整测试（V3.2 C++ Hough-like优化后）
- FOV=9.91°, N_img=163, M=259
- Phase A: θ_peak=179.35° SNR=24.7x
- Phase B Hough-like: θ=179.95° s=0.9830 n=80
- **SVD精修后**: s=0.9821 θ=179.99° n=66 rms=3.410
- **耗时: 1.04s**

### 验证5：T2帧完整测试（V3.2 C++ Hough-like优化后）
- FOV=1.56°, N_img=311, M=512
- Phase A: θ_peak=-89.15° SNR=2031.8x
- Phase B Hough-like: θ=270.90° s=1.0048 n=169
- **SVD精修后**: s=1.0035 θ=-89.11° n=44 rms=0.515
- **耗时: 0.31s**

### 验证6：Mode等价性确认
T4帧中Mode 1(θ=180°)和Mode 2(θ=0°)给出完全一致的结果

## 6. 性能分析

### V3.2性能演进

| 版本 | T2帧 | T4帧 | 加速比(vs Python) |
|------|------|------|-------------------|
| V3.2 Python | 685s | 379s | 1x |
| V3.2 C++ (随机抽样) | 10.48s | 5.48s | 65-69x |
| V3.2 C++ (Hough-like) | **0.31s** | **1.04s** | **365-2200x** |

### Hough-like优化加速比

| 帧 | 优化前 | 优化后 | 加速比 |
|---|---|---|---|
| T2 (FOV=1.56°) | 10.48s | 0.31s | **34x** |
| T4 (FOV=9.91°) | 5.48s | 1.04s | **5x** |

### 瓶颈分析（Hough-like优化后）
- Phase A: ~1000次1点法抽样，耗时<0.1s
- Phase B粗搜: Hough-like O(N×logM) per (θ,s)，21θ×11s
- Phase B精搜: 6θ×6s
- SVD精修: <0.1s
- 中心修正后精搜: 1-2次重试

## 7. 文件结构

```
lib/plate_solve/cpp/vector_match_v3_2/
├── include/vm32_api.h          # C接口头文件
├── src/vm32_core.cpp           # 核心C++实现
├── Makefile                    # 编译脚本
└── vector_match_v3_2.dll       # 编译产物

lib/plate_solve/python/
├── vector_match_v3_2.py        # Python纯版（参考）
└── vector_match_v3_2_cpp.py   # Python ctypes包装器

lib/plate_solve/scripts/
├── test_v32_cpp_single.py      # 单帧测试脚本
└── test_v32_cpp_batch.py       # 全量测试脚本
```

## 8. 待完成

- [x] T2帧：SNR动态收紧算法成功（rms=0.64px n=37）
- [x] T4帧：SNR动态收紧算法成功（rms=0.55px n=66）
- [x] C++加速：0.3-1.0s/帧（目标<10s ✓）
- [x] Hough-like (tx,ty)优化：34x加速
- [ ] 全量测试：200+帧成功率对比V2
- [ ] 参数调优

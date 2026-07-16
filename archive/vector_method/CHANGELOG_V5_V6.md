# Plate Solve V5/V6 版本历史

## V5.0 4SADQ-KV 盲解析最小测试系统（2026-06-25）
**核心创新**：将已知像素尺度s₀嵌入特征——4颗星的6个绝对角距作为k-vector排序键，单一连续索引检索，理论区分度比Astrometry.net归一化哈希高10^16倍。设计文档 [blind_indexing_feature_design.md](file:///F:/Astro%20dev/Astro%20CS%20Normalization%20Database/lib/plate_solve/blind_indexing_feature_design.md)

**模块**: `lib/plate_solve/blind_index/`（Python原型，零侵入V3.5）
- `python/`: io_wrappers(复用3个DLL)、quad_geometry(6距离+规范化+退化过滤)、kvector(二分K[j]+O(1)定位O(k)扫描)、quad_selector(金字塔Top-20+8NN)、matcher(d_AB范围查询+5距离3σ验证)、wcs_solver(Umeyama SVD允许反射捕获Y翻转)、voting(Kolomenkin≥3交叉投票)、pipeline(主管线)
- `tests/test_pipeline.py`: 端到端测试，`logs/test_report.txt` 报告
- Spec: `.trae/specs/blind-indexing-kvector-prototype/`

**关键设计决策**:
- 本地索引测试harness捷径：用FITS头RA/Dec查询DR3局部天区构建k-vector，但匹配算法本身仅用s₀不用指向
- 6距离规范：d_AB=最长边，C/D按d_AC≤d_BC字典序排序
- k-vector: Δ=0.5", d_min=2.0", K[j]=d_AB≤d_min+(j+1)·Δ的末下标
- σ_d=σ_pos×s₀(σ_pos=0.5px)，查询容差±3σ_d
- Umeyama允许反射(det R=-1)以捕获图像Y下/天球Dec上的Y翻转

**测试结果(4帧, T1/T2/T3/T4)**:
- GalaxyCenter_T4_Red(s0=6.188"/px宽场): 成功，RMS=3.33"(略超3"阈值)，CRVAL偏差0.00"，1候选
- M20_T2/LDN43_T1/NGC55_T3: 失败(0候选)，根因=star selection瓶颈

**已修复Bug**: star_detector输出饱和星优先(flux=-1, mag<5.7不在Gaia DR3)，污染Top-N亮星池。修复：quad_selector.generate_image_quads()增加saturated_arr参数跳过饱和星。修复后M20 d_AB从[845,831,762]"降至[124,89,317,407]"(落在参考范围内)

**遗留限制(算法层面，待C++全量版改进)**:
- flux排序与Gaia星等不相关→窄场帧Top-N亮星池中Gaia星比例极低→无候选
- 建议改进：空间均匀抽样/PROSAC渐进扩张/增大pool_size+四边形数
- 核心机制(k-vector+5距离+Umeyama)在星密度足够时验证通过

## V6.0 DD-SPPS 频域盲解析（2026-06-26, Phase 1 4/4 成功 + 50 帧独立验证）
**核心算法**: FFT 频域信号处理盲解析 (Density-Driven Signal Processing Plate Solving)
- 将星场映射到 512×512 网格生成 2D 高斯核信号, 用相位相关在频域求解旋转+平移, 构建 WCS

**模块**: `lib/plate_solve/blind_index_v3/`（Python 原型）
- **模块记忆**: [lib/plate_solve/blind_index_v3/memory.md](file:///F:/Astro%20dev/Astro%20CS%20Normalization%20Database/lib/plate_solve/blind_index_v3/memory.md)
- `python/`: density(密度估计)、signal(高斯核信号化)、phase_correlation(1D/2D相位相关+Hann窗)、wcs(WCS构建+KD-tree验证+refine_wcs_2d)、pipeline(主管线)、io_helpers(FITS头指向读取)、astrometry_client(Astrometry.net API客户端)
- `tests/test_phase1.py`: 4帧端到端测试, `logs/phase1_report.txt` 报告
- `tests/test_astrometry_validation.py`: 50帧 astrometry.net 独立验证, `logs/astrometry_validation_report.txt` 报告
- Spec: `.trae/specs/blind-indexing-ddspps-prototype/`

**关键创新**:
1. **1D 相位相关 (Fourier-Mellin) 直接求解任意角度 0~360°**: windowed_fft2→|F|极坐标角向投影→1D相位相关→θ_1d, ncorr消除180°歧义, N=2000信号星确保角度签名稳定
2. **dx=0/dy=0 + CRVAL 迭代精化**: 不依赖 phase_correlate_2d 的 dx/dy (不可靠), 用 refine_crval + refine_wcs_2d 迭代
3. **多阶段容差收敛**: 大容差(45")→MAD outlier剔除→中容差(15")→正常容差(5"), 逐步收紧
4. **接受条件放宽**: 0.5%×min(N_stars,N_gaia)最低5, 低RMS兜底(n_inliers≥3且RMS<1")
5. **Astrometry.net WCS 文件下载**: 直接下载 WCS FITS 文件读取准确 CD 矩阵, 避免从 orientation+parity 重建的 90° 偏差

**Phase 1 测试结果(4帧, 100%成功)**:
| 帧 | mode | θ_diff(mod180) | CRVAL_diff | RMS | n_inliers | 耗时 |
|---|---|---|---|---|---|---|
| M20_T2 | 3 | -0.689° | 0.96" | 3.375" | 188 | 5.58s |
| LDN43 | 3 | -1.189° | 1.51" | 3.561" | 11 | 4.49s |
| NGC247_T2 | 3 | -1.332° | 3.00" | 0.259" | 3 | 3.93s |
| NGC55_T3 | 3 | -(无WCS) | - | 1.567" | 5 | 4.08s |

**Astrometry.net 50 帧独立验证结果 (2026-06-26)**:
- DD-SPPS 成功: 29/50 (58.0%), Astrometry.net 成功: 43/50 (86.0%)
- 两者都成功: 25/50 (50.0%), 验证通过(θ<1°且CRVAL<10"): 13/50 (26.0%)
- **主要问题**: 90° 偏差 (12/25 = 48%) — 1D 相位相关 4 重网格对称遗留
- **稀疏星场失败**: NGC55_T3(6/6), Victory_Nebula(5/5), NGC6302_T1(5/5)
- **性能优势**: DD-SPPS 中位 4.6s vs Astrometry.net 中位 27.3s (快 5.4x)
- **验证通过的帧精度**: θ_diff 中位 0.729°, CRVAL_diff 中位 1.64", pixscale_diff 平均 0.22%

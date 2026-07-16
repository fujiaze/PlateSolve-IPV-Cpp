# Plate Solve 历史版本迭代文档

> 本文档记录 plate solving 模块从向量法到三角法的完整演进脉络。
> 所有历史代码、文档、日志均完整归档于 `archive/` 目录，不做删除。

---

## 概述

本模块经历三个研发阶段：

| 阶段 | 时间范围 | 算法路线 | 最终状态 |
|------|----------|----------|----------|
| 第一阶段 | 2026-05 ~ 2026-06 | 向量法（V2-V4.5） | 790帧成功率33.8%，算法理论完整但工程实用性不足 |
| 第二阶段 | 2026-06 | 盲解析实验（V5-V6） | 探索性研究，独立验证通过率26% |
| 第三阶段 | 2026-07 ~ 至今 | 三角法（IPV，V4.6-V4.28） | 790帧成功率99.87%，当前生产版本 |

**重要说明**: 当前三角法实现是为推进项目进度的被迫选择。向量法作为算法研发主线，其理论完整、资料保存完整，待未来条件成熟时可继续深入研究。

---

## 第一阶段：向量法时代（V2-V4.5）

> 归档位置: `archive/vector_method/`
> 算法文档: `archive/vector_method/README.md`（Vector Match V2 详细设计）

### 核心算法思想

向量匹配法的核心是将图像星点和星表星点表示为向量，通过向量间的几何关系（模长比、夹角）建立对应关系，再通过 Umeyama SVD 求解相似变换（尺度 s、旋转角 θ、平移 tx/ty），最终构建 WCS。

### V2 - Python 纯实现

- **算法**: RANSAC + Umeyama SVD
- **关键设计**:
  - 稀疏加权采样：饱和星优先 + 空间均匀化
  - 4 flip 模式扫描（NONE / FLIP_X / FLIP_Y / FLIP_XY）
  - 两阶段候选对：粗筛（距离比） + 精筛（NN 匹配）
  - 5 大核心优化（详见 README.md）
- **测试结果**: T1/T4 ~95%，T2/T3 ~0%（南天目标 Gaia 查询失败）

### V3.x - Record-and-Filter + C++ 加速

#### V3.0 - 早期三阶段匹配
- 粗筛 / 精筛 / 验证三阶段流水线

#### V3.2 - C++ 加速版本（初版）
- 核心算法移植到 C++，Python 通过 ctypes 调用
- 引入 Eigen3（SVD）和 nanoflann（KDTree）

#### V3.3 - Record-and-Filter（核心创新）
- **1点法无放回抽样**: (u_i, w_j) -> 变换 -> NN 匹配 U -> 统计 s_ratio -> θ 加权直方图 -> θ_SNR 达标后双过滤 -> 配对即对应关系 -> SVD 精确求解
- **关键度量**: s-in-range（模长比）+ NN 距离约束，不是 KDTree 距离内点
- **测试结果**（562 有效帧）: 总成功率 83.1%，RMS 中位 0.51px

#### V3.4 - WCS-SIP 标准输出
- Siril 风格 CD 修正
- WCS-SIP 标准输出格式

#### V3.5 - 分层拟合
- **BIC 自适应 SIP 阶数**
- **全仿射 CD/CRVAL**
- **MAD 稳健拟合**
- **Umeyama 符号验正**
- 流程: Phase A (SNR 抽样) -> Phase B (过滤+SVD) -> Phase C (分层拟合: Layer0 Umeyama->CD + Layer1 MAD+全仿射 + Layer2 BIC SIP)
- **关键教训**: 全局 NN 扩充 + 星点扩增导致精度恶化（移除后仅保留 Phase A+B 验证的 5~50 对零假匹配）

### V4.0 - 5 大优化

在 V3.5 抽样投票核心架构基础上新增 5 大优化：

| 优化 | 定位 | 核心算法 |
|------|------|----------|
| 密度匹配 | Phase 0 | 图像亮星密度反推目标星表密度，迭代调整 Gaia 查询极限星等 |
| k-vector | Phase C | O(k) 快速角距索引，与星表大小 n 解耦（Mortari 1997） |
| PROSAC | Phase A | 按星点质量分降序优先采样，加速 RANSAC 收敛（Chum 2005） |
| 贝叶斯 | Phase D' | 贝叶斯因子 K 替代简单阈值，实现"零误报"验证（Lang 2010） |
| 三角形 | Phase D' | 双特征验证（面积 A + 极惯性矩 J） |

- **测试结果**（10 帧对比）: 成功率 100%，max RMS 从 22.22px 降至 8.84px

### V4.1 - 不对称选星策略

- **图像侧**: 50 颗（饱和优先 + flux 补足）
- **Gaia 侧**: 1.5x 密度匹配
- **效果**: 失败帧恢复率 2/3，成功率提升 2 个百分点

### V4.2 - 模块化管线重构

将单体 C++ DLL 拆分为 5 个独立模块 DLL + Python 编排器：

| 模块 | 功能 | 外部依赖 |
|------|------|----------|
| StarSelector | Phase 0 选星 + 密度匹配 | 无 |
| VectorMatcher | Phase A+B 抽样投票 + SVD | 无 |
| PairExpander | Phase C 匹配对扩增 | 零外部依赖 |
| PairVerifier | Phase D+D' MAD 清洗 + 贝叶斯 + 三角形验证 | Eigen3 |
| WcsFitter | Phase E WCS 拟合 | Eigen3 |

- **关键改进**: 支持断点续跑、每阶段独立 checkpoint JSON、模块独立可调用

### V4.3 - 相对向量法实验（DMPDV）

- **核心创新**: 相对向量 Δu_ij = U[i]-U[j] 消除平移 t，将 4D (s,θ,tx,ty) 降为 1D θ 搜索
- **效果**: t≠0 场景（Galaxy_Center）SNR 从 2.67/4.00 提升至 8.4/31.0
- **合成数据实验**: 28/28 成功，θ 准确

### V4.4 - 相对向量法 C++ 实现 + 3D 密度场

#### V4.4 相对向量法 C++ 实现
- k-vector 距离索引 + 第三星交叉验证 + θ 直方图投票
- **第三星验证关键**: 假匹配通过率 ≈ 1.4×10⁻⁶，背景降低 2300 倍
- 4/4 成功，RMS 与 V4.3 完全一致

#### V4.4 3D (θ,dx,dy) 密度场
- 360×200×200 bin 稀疏直方图
- 递归聚焦状态机：探索 -> 识别 -> 聚焦 -> 收敛
- 36/36 success=True，但 6 帧 RMS 异常（dx/dy 镜像对称歧义）
- **遗留**: dx/dy 歧义问题待解决

### V4.5 - 相对向量法 θ 求解器
- V4.4 的最终优化版本
- 790 帧全量测试成功率 33.8%（宽 FOV 瓶颈）

### 向量法总结

**成就**:
- 理论体系完整：从 V2 Python 原型到 V4.5 C++ 优化，涵盖 RANSAC、Umeyama SVD、k-vector、PROSAC、贝叶斯验证、相对向量法、3D 密度场等多种算法
- 算法创新多：Record-and-Filter、相对向量法消除平移、3D 密度场递归聚焦等均为原创设计
- 资料保存完整：设计文档、CHANGELOG、测试脚本、实验数据齐全

**瓶颈**:
- 宽 FOV（>3°）成功率低：星场密集导致向量匹配区分度不足
- 790 帧全量测试成功率仅 33.8%，无法满足项目需求
- dx/dy 镜像对称歧义问题未解决

**未来研究方向**:
- 解决 3D 密度场的 dx/dy 歧义
- 宽 FOV 星场密集场景的向量匹配优化
- 相对向量法 + 深度学习特征提取

---

## 第二阶段：盲解析实验（V5-V6）

> 归档位置: `archive/blind_solving/`

### V5.0 - 4SADQ-KV 四元组盲解析

- **核心创新**: 4 颗星的 6 个绝对角距作为 k-vector 排序键，单一连续索引检索
- **理论区分度**: 比 Astrometry.net 归一化哈希高 10^16 倍
- **测试结果**: 4 帧中 1 帧成功（GalaxyCenter_T4_Red），3 帧失败（star selection 瓶颈）

### V6.0 - DD-SPPS 频域盲解析

- **核心算法**: FFT 频域信号处理盲解析
  - 星场映射到 512×512 网格生成 2D 高斯核信号
  - 相位相关在频域求解旋转 + 平移
  - 1D 相位相关（Fourier-Mellin）直接求解任意角度 0~360°
- **Phase 1 测试**: 4 帧 100% 成功
- **50 帧独立验证**: DD-SPPS 成功 58%，Astrometry.net 成功 86%，两者都成功验证通过率 26%
- **主要问题**: 90° 偏差（1D 相位相关 4 重网格对称遗留）

### 盲解析总结

**成就**:
- 探索了频域盲解析和四元组盲解析两条技术路线
- DD-SPPS 速度优势显著（中位 4.6s vs Astrometry.net 27.3s）
- 集成了 Astrometry.net 完整 C 代码库作为独立验证基准

**瓶颈**:
- 90° 偏差问题未解决
- 稀疏星场成功率低
- 独立验证通过率仅 26%

---

## 第三阶段：三角法时代（IPV，当前）

> 代码位置: `lib/plate_solve/cpp/ipv/`
> 当前版本: V4.28

### 背景

向量法 790 帧全量测试成功率 33.8% 无法满足项目需求。为推进项目进度，转向借鉴 PixInsight ImageSolver 的三角法作为工程实现。**此为推进项目进度的被迫选择，向量法算法价值保留待未来研究。**

### V4.6 - 六边形描述符 + 几何投票 + PROSAC 验证（Phase I MVP）

- **六边形描述符**: 1 中心星 + 5 邻星组成六边形
- **几何投票**: 全星对 pairwise 距离对称投票
- **PROSAC 验证**: 按 vote 降序优先采样，2 对解析求解相似变换
- **20 帧测试**: 100% 成功

### V4.22 - 崩溃修复 + Siril 对齐

- **崩溃修复**: `ipv_solve` 入口添加 `std::memset(result, 0, sizeof(IpvWcsResult))`，解决 0xC0000005 访问违例
- **Siril 对齐**: 5 个 P1 级修复（n_target=20, MINVOTES=2, 2D 投票矩阵, 双向匹配, SIP 清零）
- **结果**: 成功率从 91.5% 跌至 54.3%（Siril 参数不适用），但零崩溃

### V4.23-V4.24 - scale 约束 + 三角形匹配星数对齐

- **V4.23**: iter_trans 工作集扩展 + make_vote_matrix scale 约束 + 子模块日志器初始化
- **V4.24**: 三角形匹配星数 20->60（对齐 Siril AT_MATCH_CATALOG_NBRIGHT=60）
- **memmove 修复**: 用 ctypes 直接赋值替代 memmove，解决 0xC0000005 崩溃
- **结果**: 成功率恢复至 89.1%，RMS 中位 0.559px

### V4.25-V4.27 - 精度对标 Siril + OpenMP 并行化

#### V4.25 - 算法对齐 Siril
- iter_trans 默认 order=3（支持三次多项式）
- SIP order 默认 3
- iterative_reproject 高阶重匹配
- atRecalcTrans 二轮精化
- **结果**: 97.34%，RMS 中位 0.487"

#### V4.26 - order fallback
- order=3 失败时自动降级 order=2 -> order=1
- **结果**: 97.59%（+2 帧），完全匹配 Siril 成功率

#### V4.27 - OpenMP 16 线程并行化
- StarSelector Gnomonic 投影并行
- triangle_match stars_to_triangles + make_vote_matrix 并行
- iterative_reproject 投影并行
- **结果**: 耗时中位 1.744s -> 1.250s（-28%），比 Siril 快 16%

### V4.28 - 成功率 + 精度 + 速度提升（当前版本）

#### Phase A - 成功率修复
- **wcs_check 像素阈值**: 固定 600" -> 250px（FOV 自适应）+ RMS<3" + n_pairs≥10
- **Oiii 帧诊断**: 真难解帧（Siril 也失败），非算法 bug

#### Phase B - 精度修复
- **iter_trans sigma-clip 修复**: tol 预过滤 + sigma 钳制
- 5/5 帧 RMS>2" 改善，15 帧无回归

#### Phase C - 速度修复
- **Gaia 缓存预热**: 正式运行前预热 block_cache（4GB 持久）
- **编译优化**: -O2 -> -O3 -ffast-math -funroll-loops
- 0 精度回归，中位加速 7.4%

#### 最终结果

| 指标 | V4.27（基线） | V4.28（最终） | 变化 |
|------|-------------|-------------|------|
| 总成功率 | 97.59% (771/790) | **99.87% (789/790)** | +18 帧 |
| RMS 中位 | 0.489" | **0.489"** | 持平 |
| RMS max | 5.089" | **2.849"** | -44% |
| 耗时中位 | 1.250s | **1.199s** | -4% |
| 14s 异常帧 | 14.187s | **1.960s** | -86% |
| 异常/崩溃 | 0 | **0** | 无崩溃 |

### 当前状态

- **成功率**: 99.87%（789/790），唯一失败帧为 Oiii 真难解帧（Siril 也失败）
- **精度**: RMS 中位 0.489"，max 2.849"
- **速度**: 耗时中位 1.199s，比 Siril CLI 快 16%
- **稳定性**: 0 崩溃 / 0 异常

---

## 算法路线对比

| 维度 | 向量法（V2-V4.5） | 三角法（IPV，当前） |
|------|-------------------|---------------------|
| 核心思想 | 向量模长比 + 夹角投票 + SVD | 三角形投票 + iter_trans sigma-clip + SIP |
| 优势 | 理论完整，算法创新多 | 成功率高（99.87%），工程稳定 |
| 劣势 | 宽 FOV 成功率低（33.8%） | 借鉴 Siril，算法原创性较低 |
| 适用场景 | 中窄 FOV，理论研究 | 全 FOV 范围，生产环境 |
| 依赖 | Eigen3 + nanoflann + OpenMP | 无外部依赖（手写 SVD） |
| 归档状态 | 完整归档于 archive/vector_method/ | 当前生产版本 |

---

## 资产索引

| 资产 | 归档位置 | 说明 |
|------|----------|------|
| 向量法 V2 详细设计 | `archive/vector_method/README.md` | RANSAC + Umeyama SVD 完整文档 |
| V3 版本历史 | `archive/vector_method/CHANGELOG_V3.md` | V3.0-V3.5 |
| V4 版本历史 | `archive/vector_method/CHANGELOG_V4.md` | V4.0-V4.4（含 5 大优化详细记录） |
| V5/V6 版本历史 | `archive/vector_method/CHANGELOG_V5_V6.md` | 盲解析实验 |
| 向量法 Python 实现 | `archive/vector_method/python/` | 各版本求解器 + Gaia 接口 |
| 向量法 C++ 实现 | `archive/vector_method/cpp/` | V3.2-V4.5 C++ 加速版本 |
| 向量法设计文档 | `archive/vector_method/docs/` | 各版本设计文档 |
| 向量法实验脚本 | `archive/vector_method/scripts/` | V3.3-V4.5 测试脚本 |
| Astrometry.net 客户端 | `archive/blind_solving/astrometry/` | Python 客户端 |
| Astrometry.net C 代码库 | `archive/blind_solving/astrometry_net/` | 完整 C 代码库 |
| 盲解析实验 | `archive/blind_solving/blind_index*/` | V5 四元组 + V6 频域 |
| 历史测试日志 | `archive/historical_logs/` | V4.13-V4.27 全部日志 |
| 历史测试脚本 | `archive/historical_scripts/` | crash 日志 + 诊断脚本 |
| 压缩归档 | `archive/plate_solve_old.rar` | 原始 plate_solve_old 压缩包 |

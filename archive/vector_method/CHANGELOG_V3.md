# Plate Solve V3 版本历史

## V3.3 Record-and-Filter（当前版本）

**核心创新**：1点法无放回抽样(u_i,w_j)→变换→NN匹配U→统计s_ratio→θ加权直方图→θ_SNR达标后θ+度量双过滤→配对即对应关系→SVD精确求解

**流程**:
- Phase A: 无放回随机抽样, 计算s_ratio=|U[k]|/|Wf[l]|(NN距离<5×s0), θ直方图加权, θ_SNR监控停止
- Phase B: θ峰值+n_in_range双过滤→1对1互斥→Umeyama SVD→迭代精修

**关键设计**:
- 度量: s-in-range (模长比) + NN距离约束, 不是KDTree距离内点
- 单阶段: Python端一次C++调用得精确解, 无二阶段精修
- 无门槛: θ_SNR达标后SVD结果直接信任, 不设min_inliers
- 无放回: unordered_set去重, K_max=min(10000,N×M)

**测试结果（562有效帧, GaiaDR3）**:
| 望远镜 | V2 | V3.2 | V3.3 |
|--------|-----|------|------|
| T1 | ~95% | 53.8% | 100% |
| T2 | 0% | 73.2% | 76.3% |
| T3 | 0% | 58.3% | 62.2% |
| T4 | ~95% | 27.4% | 100% |
| 总计 | — | 55.2% | 83.1% |
- RMS中位: 0.51px, 解析中位: 0.02s, 算法失败: 0帧
- 剩余失败：Gaia查询M=0（南天目标/天区边缘）

**实现文件**:
- `cpp/vector_match_v3_3/src/vm33_core.cpp` (~700行, C++17+OpenMP)
- `cpp/vector_match_v3_3/include/vm33_api.h` (C接口)
- `python/vector_match_v3_3_cpp.py` (~340行)
- `docs/v3_3_design.md` (设计文档v5)

## V3.5 分层拟合（2026-06-09）

C++代码已重构，移除Phase C全局NN扩充 + Phase D迭代清洗 + Phase D'星点扩增
- 新流程：Phase A(SNR抽样) → Phase B(过滤+SVD, 输出cu/cw匹配对) → Phase C(分层拟合: Layer0 Umeyama→CD + Layer1 MAD+全仿射 + Layer2 BIC SIP)
- DLL已编译，编译需添加PATH: `$env:PATH = "C:\msys64\mingw64\bin;" + $env:PATH`
- 测试结果：NGC7293(30对, SIP-RMS=2.007px, BIC选4阶), GC_P2(25对, 仅线性CD), GC_P1(28对, 仅线性CD)
- 可视化：`标准控制点对可视化调试示例.py` → 输出 734/6379/11224 对匹配到 PNG
- 设计文档已更新：[docs/v3_5_design.md](file:///F:/Astro%20dev/Astro%20CS%20Normalization%20Database/lib/plate_solve/docs/v3_5_design.md)

## 已解决的重大问题（V3时期）

8. **plate_solve三角匹配失败**: 旧代码用200检测星+2000 catalog星构建三角形(catalog侧~1.3B三角形OOM), 且未启用scale过滤。修复为nbright=60双方+scale过滤+radius=0.002

12. **三角形匹配在噪声多时失败**: 饱和星+亮星场景下三角形匹配只找到6对（全是噪声），粗拟合完全依赖恒等变换回退。修复：用KDTree+sigma-clip剪枝替代三角形匹配，利用WCS近似正确的假设直接匹配

13. **二次畸变过拟合**: 训练RMS=0.931px但最终评估RMS=1.290px。修复：最终比较时使用sigma-clip+更小阈值评估，且二次畸变需显著优于仿射(<0.95x)才启用

14. **最终评估阈值不一致**: pipeline内部用max_dist*0.5=12.5px评估但外部用max_dist*0.2=5.0px，导致模型选择偏差。修复：统一使用max_dist*0.3+sigma-clip评估

15. **V3.2 Phase B GridSearch+Hough (tx,ty)估计失败**: 1D直方图笛卡尔积丢失2D峰值。V3.3用s-in-range+θ双过滤→配对即对应关系→SVD替代

16. **V3.3开发过程中对s-in-range度量的误解**: 初始用KDTree距离作为内点度量，后纠正为用户的原始设计——模长比s_ratio+NN距离过滤。关键区别在于s-in-range不受星等均匀分布影响，且正确/噪声区分度极高（θ_SNR 4360x vs 22x）

17. **V3.5全局NN扩充+星点扩增导致精度恶化 (2026-06-09)**: Phase C全局NN扩充在变换不完全精确时KDTree最近邻会匹配到错误星点，Phase D'星点扩增的双向NN同样引入大量假匹配，混入拟合数据导致系统性误差和精度变差。解决方案：移除Phase C全局NN、Phase D迭代清洗、Phase D'星点扩增，仅保留Phase A+B信噪比方法验证的匹配对（5~50对，零假匹配），直送Phase C分层拟合。

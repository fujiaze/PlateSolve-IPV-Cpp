# 📂 dev 脚本索引

> 早期开发与调试脚本 — T4望远镜向量匹配算法的开发期测试代码

### T4三阶段匹配调试 (debug_t4_*)

| 脚本 | 功能 |
|------|------|
| `debug_t4_3stage.py` | T4三阶段（粗筛→精筛→验证）匹配算法的端到端实验 |
| `debug_t4_3stage_kdtree.py` | 将三阶段匹配中的线性扫描替换为KDTree加速，对比速度 |
| `debug_t4_3stage_opt.py` | 三阶段匹配的参数优化：搜索半径、候选数、投票阈值等 |
| `debug_t4_3stage_vote.py` | 投票机制的独立实验：不同投票策略对匹配精度的影响 |
| `debug_t4_correct_pairs.py` | 检验三阶段算法能否正确识别已知ground truth的匹配对 |
| `debug_t4_correct_s.py` | 验证算法恢复scale参数的精度（s误差≤0.1%） |
| `debug_t4_theta_impact.py` | 分析旋转角θ的采样密度对匹配成功率的影响 |
| `debug_t4_theta_s_search.py` | 联合搜索θ和s参数的最优组合 |
| `debug_t4_s_analysis.py` | 对s参数分布的详细统计分析 |
| `debug_t4_s_search.py` | 单独搜索s参数最优值 |
| `debug_t4_fixed_st.py` | 固定s和θ搜索平移量tx/ty |
| `debug_t4_largetau.py` | 测试较大τ（匹配阈值）下的搜索策略 |
| `debug_t4_tau.py` | 不同τ值对匹配结果的影响分析 |
| `debug_t4_1point.py` | 单点抽样的基础实验（1对Gaia星→1个s/θ估计） |
| `debug_t4_2point_s.py` | 两点抽样的s估计精度实验 |
| `debug_t4_warmup.py` | 多组候选s/θ的预热搜索策略 |
| `debug_t4_strategies.py` | 对比多种搜索策略（穷举/vs/贪心/vs/随机等） |
| `debug_t4_verify.py` | 对单帧运行T4全套求解并验证结果 |
| `debug_t4_vote_fine.py` | 精细化投票：在粗糙匹配基础上做局部精细搜索 |

### V2/WCS早期调试

| 脚本 | 功能 |
|------|------|
| `debug_vm2.py` | V2向量匹配算法的单帧调试实验 |
| `debug_gold_pool_distribution.py` | 分析黄金池（gold pool）中候选匹配的分布特性 |
| `debug_mc_sampling.py` | Monte Carlo采样的收敛性实验 |
| `debug_mc_weighted.py` | 加权MC采样对匹配精度的影响 |

### 版本对比与性能测试 (test_*)

| 脚本 | 功能 |
|------|------|
| `test_vector_match.py` | 原始向量匹配算法的基本功能测试 |
| `test_full_solve.py` | 端到端全流程求解的性能测试 |
| `test_v2v3_compare.py` | V2与V3算法在同一数据上的对比 |
| `test_v2v3_full.py` | V2与V3算法多帧批量对比 |
| `test_v2v3_speed.py` | V2与V3的纯速度benchmark |
| `test_v31_single.py` | V3.1单帧求解功能测试 |
| `test_v32_single.py` | V3.2 Python纯实现单帧测试 |
| `test_v32_cpp_single.py` | V3.2 C++加速版单帧测试 |
| `test_v32_cpp_batch.py` | V3.2 C++加速版批量测试 |
| `test_v33_cpp_single.py` | V3.3 C++单帧求解功能测试 |
| `test_v33_cpp_batch.py` | V3.3 C++批量求解成功率统计 |
| `test_cpp_v2_single.py` | V2 C++加速版的单帧功能测试 |
| `test_panel1_cpp_v2.py` | Panel1数据用V2 C++版求解的批量测试 |
| `test_panel1_debug.py` | Panel1特定帧的详细调试 |
| `test_robustness.py` | 参数扰动下的鲁棒性测试 |
| `test_slow_frame.py` | 慢帧（大量星点）的耗时分析和优化 |
| `test_timing.py` | 各阶段耗时分析（星点检测/投影/匹配/SIP） |
| `test_step_timing.py` | 逐子步骤的精细耗时测量 |
| `test_thread_io.py` | 多线程I/O的性能测试 |
| `test_cache_effect.py` | 缓存命中率对Gaia查询速度的影响 |
| `test_copy_bottleneck.py` | 测量数据拷贝是否为性能瓶颈 |
| `test_dbtype_compare.py` | 对比SQLite vs CSV文件的Gaia查询速度 |
| `test_param_analysis.py` | 各求解参数对结果影响的敏感性分析 |
| `test_projection_debug.py` | 投影坐标转换的精度验证 |

### Gaia查询调试 (test_gaia_*)

| 脚本 | 功能 |
|------|------|
| `test_gaia_cache.py` | Gaia缓存机制的命中率和加速比 |
| `test_gaia_cache_detail.py` | 缓存逐次访问的详细trace |
| `test_gaia_optimize.py` | Gaia查询参数的自动优化 |
| `test_gaia_radius.py` | 不同查询半径对星数和求解速度的影响 |

### 其他诊断与工具

| 脚本 | 功能 |
|------|------|
| `batch_scheduler.py` | 批量求解任务调度器：按批次处理大量帧，输出汇总csv |
| `diagnose_failed_frames.py` | 分析全部失败帧的败因分布（超时/vs/匹配不足/vs/SNR低） |
| `gen_panel1_report.py` | 为Panel1数据生成带统计图的PDF/HTML报告 |
| `run_debug_visualize.py` | 调试模式下的可视化输出：匹配对连线、残差热图 |
| `scan_coordinates.py` | 扫描FOV内坐标的RA/DEC变化范围 |
| `verify_coord_fix.py` | 验证坐标修正后的精度改善 |
| `verify_coords.py` | 验证原始FITS头中的坐标与实际求解坐标的一致性 |
| `verify_csv_coords.py` | 从CSV文件读取坐标进行批量验证 |
| `analyze_v32_results.py` | 分析V3.2版本批量求解的输出统计 |

# Tasks

- [x] Task 1: 实现向量组构建模块
  - [x] SubTask 1.1: 实现图像向量组构建 build_image_vectors()：以图像中心为原点，像素偏移×s0→角秒偏移（Y取反匹配天球convention）
  - [x] SubTask 1.2: 实现星表向量组构建 build_catalog_vectors()：gnomonic投影→角秒偏移
  - [x] SubTask 1.3: 实现亮星选取策略（饱和星≥10用全部饱和星，<10用饱和+亮星共100颗）
  - [x] SubTask 1.4: 实现4种翻转预处理 apply_flip()

- [x] Task 2: 实现RANSAC相似变换求解核心
  - [x] SubTask 2.1: 实现2点采样求解相似变换 solve_similarity_2pt()
  - [x] SubTask 2.2: 实现相似变换应用 apply_similarity()
  - [x] SubTask 2.3: 实现numpy向量化内点统计 count_inliers_1to1()：KDTree+1对1互斥
  - [x] SubTask 2.4: 实现两阶段RANSAC：KDTree候选对应+RANSAC过滤离群点
  - [x] SubTask 2.5: 实现缩放因子过滤：s∈[0.9,1.1]

- [x] Task 3: 实现打分函数与模式选择
  - [x] SubTask 3.1: 实现单模式打分 score = n_inliers - λ×RMS
  - [x] SubTask 3.2: 实现归一化打分 score_norm = n_inliers/min(N,M) × (1 - RMS/τ)
  - [x] SubTask 3.3: 实现4模式综合决策：选归一化得分最高者，<0.15判定失败

- [x] Task 4: 实现4种翻转模式独立RANSAC匹配
  - [x] SubTask 4.1: 每种模式独立翻转+RANSAC+打分
  - [x] SubTask 4.2: coarse-to-fine精化：粗匹配(tau_coarse)后在内点上精化(tau_fine)
  - [x] SubTask 4.3: 最佳模式选择

- [x] Task 5: 实现WCS参数提取
  - [x] SubTask 5.1: 中心坐标修正：平移量→ΔRA/ΔDec
  - [x] SubTask 5.2: 旋转角提取：atan2
  - [x] SubTask 5.3: 最终像素尺度：s_final = s0 × s
  - [x] SubTask 5.4: 翻转信息记录

- [x] Task 6: 实现中心修正+精化
  - [x] SubTask 6.1: 从平移量修正中心坐标
  - [x] SubTask 6.2: 新中心下重新投影+RANSAC精化(tau_fine)

- [x] Task 7: 实现完整端到端VectorMatch类
  - [x] SubTask 7.1: 整合为VectorMatch.solve()方法
  - [x] SubTask 7.2: 集成GaiaClientPy
  - [x] SubTask 7.3: 集成astro_image_io和star_detector
  - [x] SubTask 7.4: 实现solve_with_file()便捷接口
  - [x] SubTask 7.5: 输出VectorMatchResult数据结构

- [x] Task 8: 端到端测试验证
  - [x] SubTask 8.1: 使用testdata中的FITS图像进行端到端测试
  - [x] SubTask 8.2: 验证RMS < 2px（实际0.613px ✅）
  - [x] SubTask 8.3: 验证4种翻转模式正确识别（模式2=Y翻转 ✅）
  - [x] SubTask 8.4: 验证像素尺度报告正确（6.077角秒/px ✅）

# Task Dependencies
- Task 2 依赖 Task 1
- Task 3 依赖 Task 2
- Task 4 依赖 Task 1, 2, 3
- Task 5 依赖 Task 4
- Task 6 依赖 Task 5
- Task 7 依赖 Task 1-6
- Task 8 依赖 Task 7

# 实现过程中的关键发现
1. **Y轴翻转**：图像像素Y向下，天球Dec向上，图像向量Y分量必须取反
2. **纯随机RANSAC不可行**：从U和W独立随机采样2点，正确配对概率≈8×10⁻⁶，200次迭代不可能命中
3. **两阶段RANSAC**：先用KDTree建立候选对应（半径300角秒），再在候选对上RANSAC
4. **Coarse-to-fine**：粗匹配tau=2.5×s0，精化tau=1.0×s0，RMS从1.5px降到0.6px
5. **迭代重投影发散**：中心修正后重新RANSAC会导致偏移累积，改为单次中心修正+精化

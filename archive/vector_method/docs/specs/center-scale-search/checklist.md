# Checklist

## 数据结构定义
- [x] `PSMFeatureStar` 结构体定义正确，包含 x, y, flux, orig_idx
- [x] `PSMFeatureTriangle` 结构体定义正确，包含顶点索引和特征
- [x] `PSMMatchPair` 结构体包含 img_idx, cat_idx, votes, flux_ratio
- [x] `PSMMatchResult` 结构体包含 pairs, pair_count, tri_match_count, total_votes
- [x] 头文件使用 PSM_EXPORT / extern "C" 导出

## 星点筛选
- [x] flux降序排序正确
- [x] top-N筛选正确（取min(actual, max)）
- [x] 原始索引保留正确
- [x] 图像星取 ≤2000 颗
- [x] Gaia星取 ≤5000 颗

## 三角形构建
- [x] 距离矩阵计算正确 (NxN)
- [x] 三角形枚举覆盖所有 C(N,3) 组合
- [x] 边长排序正确 a >= b >= c
- [x] 特征计算正确：ba, ca, side_a_angle, a_length
- [x] 顶点索引识别正确（a_idx对最长边）
- [x] 按ba升序排序正确

## 三角形剪枝
- [x] ba > 0.9 剔除正确
- [x] a_length < 5px 剔除正确
- [x] 剪枝后三角形数量合理（~70-80%保留）

## 三角匹配
- [x] 二分查找ba范围正确
- [x] (ba, ca)空间距离计算正确
- [x] 半径过滤正确（默认0.002）
- [x] 尺度过滤正确（可选）

## 投票矩阵
- [x] 投票矩阵创建正确 (nbright x nbright)
- [x] 投票填充正确（每个匹配三角形+3票）
- [x] flux辅助投票增强正确（可选）

## Top Vote Getters
- [x] 每个图像星的最佳Gaia星识别正确
- [x] 每个Gaia星的最佳图像星识别正确
- [x] 最少票数过滤正确 (votes < 2)
- [x] 双向一致性去重正确

## 编译
- [x] Makefile 正确
- [x] g++ 编译无错误（警告可接受）
- [x] DLL 生成成功
- [x] DLL 导出符号正确

## 集成测试
- [x] Python 绑定正确调用 DLL
- [x] 全流程：读取 → 检测 → Gaia查询 → 特征匹配 → 仿射拟合
- [ ] 匹配对数 >= 50 — 需要优化坐标转换
- [ ] 最终 RMS < 10px — 需要优化坐标转换

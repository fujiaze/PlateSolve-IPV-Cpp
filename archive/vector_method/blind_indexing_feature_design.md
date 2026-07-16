# 天文盲解析算法综合设计

## 取百家之长：k-vector索引 · 金字塔选星 · Kolomenkin投票 · 贝叶斯验证

---

## 0. 开篇澄清：这不是"Astometry.net + s₀"

### 0.1 Astrometry.net 的尺度处理机制

```
Astrometry.net 使用 s₀ 的方式:

离线阶段:
  对每个离散的像素尺度 s_k (如 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 arcsec/pix):
    构建一个独立的索引文件 index_s_k.bin
    每颗参考星 → 取邻星 → 归一化四边形 → 4D哈希码
    (归一化消去了尺度: d_AB→1, 其他距离相对于 d_AB)

在线阶段:
  根据输入图像的 s₀ → 选择最接近的 s_k → 加载 index_s_k.bin
  在此索引中搜索 4D 归一化hash → 候选四边形列表
  贝叶斯验证 → WCS

关键点:
  ✓ s₀ 被用来选择"哪个索引文件"
  ✗ s₀ 没有被嵌入特征本身
  ✗ 归一化消去了绝对尺度信息 → 4D hash 区分度受限
  ✗ 多索引需要额外存储和维护
```

### 0.2 本方法的根本差异：从"哈希"到"连续k-vector"

```
本方法使用 s₀ 的方式:

离线阶段:
  构建单一索引文件（与尺度无关）
  每颗参考星 → 取邻星四边形 → 6个绝对角距 (arcsec)
  以 d_AB (最长边) 为排序键 → 建立 1D k-vector
  k-vector 是连续函数，允许任意精度范围查询

在线阶段:
  图像4星 → 6个绝对角距 d = pixel_dist × s₀
  查询: d_AB ∈ [d_AB - δ, d_AB + δ] → k-vector O(k) 检索 → 候选列表
  余下5个距离做精确匹配验证

关键差异:
  ✓ s₀ 直接嵌入特征: d = |p_i - p_j| × s₀ → 绝对角距(arcsec)
  ✓ k-vector 替代哈希表: 连续范围查询 → 无量化binen流失
  ✓ 单一日志索引: 不需要多个尺度版本
  ✓ 6D vs 4D: 绝对距离比归一化坐标多2个自由度的信息
```

### 0.3 信息论对比

```
Astrometry.net (单一尺度索引内的信息量):
  4颗星 → 归一化 → 4个坐标值 ∈ [0,1], 受几何约束 → ~3.5 有效自由度
  每自由度分辨 ~50个可区分水平 (bin大小 ~0.02)
  总区分度: ~50^3.5 ≈ 10^6 种四边形模式
  → 每个hash桶预期有 ~10^3-10^4 个候选（需贝叶斯验证消除）

本方法 (k-vector + 6D绝对距离):
  4颗星 → 6个绝对角距 ∈ [1", 7200"], 受5个几何约束 → ~5 有效自由度
  每距离分辨 ~0.2" (位置噪声量级)
  总区分度: ~(7200/0.2)^5 ≈ 6×10^22 种四边形模式
  → 每次查询预期 ~10^-6 个假匹配（几乎无需验证）

信息量差距: 10^16 倍
来源: 已知 s₀ 将距离从无量纲比值变成了有量纲绝对值
      这比 Astrometry.net 多利用了 log₂(7200"/0.2"/50) ≈ 8.5 bit/dim × 5dim ≈ 42 bit
      即 ~2.5×10^12 倍的区分度提升
```

---

## 1. 理论框架：不变量统一理论下的特征选择

### 1.1 Christian 2021 框架的应用

Christian和Crassidis [2021]建立了星点识别的不变量统一理论：

```
星点成像模型:
  \bar{u}_i ∝ K · T · e_i

  K ∈ R^{3×3}: 相机标定矩阵 (5个内参数)
  T ∈ SO(3): 姿态旋转矩阵 (3个参数)
  e_i: 单位方向向量 (天球上的星)

本方法的已知条件:
  ✓ K 已知 (s₀ + 主点 + 畸变已标定)
  ✗ T 未知 (旋转 = 图像旋转角未知)
  
→ K 已知意味着相机是"标定相机" (calibrated camera) 类别
→ Christian 框架结论: 标定相机下，星间角距是 SO(3) 不变量
→ d 颗星产生 (2d-3) 个独立不变量

对于 4 颗星: 2×4-3 = 5 个独立不变量
→ 恰好对应 4SADQ 的 6 个距离中的 5 个独立分量
→ 特征空间维度 = 理论最小充分统计量维度
→ 任何进一步降维都会损失信息！
```

### 1.2 信息完备性证明——为什么 4 颗星已经理论最优

```
Christian 框架给出的独立不变量数: I(d) = 2d - 3

| d (星数) | I(d) | 含义 |
|----------|------|------|
| 2        | 1    | 只有角距，无法确定姿态 |
| 3        | 3    | 三角形，但球面三角形内角只有2个独立分量 |
| 4        | 5    | ★ 恰好够确定 4 参数相似变换 (s,θ,tx,ty) |
| 5        | 7    | 信息冗余，可用于一致性检验 |

结论: 4 颗星是标定相机盲解析的理论下限
      5 颗星提供 2 个冗余自由度用于验证
      4SADQ 的 6D 距离特征 = 5 维独立 + 1 维冗余 (满足几何约束的代价)
```

### 1.3 特征稳定性理论

```
问题: 给定噪声 σ_pos，6 个距离的测量精度如何?

对距离 d_ij = sqrt((x_i-x_j)² + (y_i-y_j)²) × s₀:
  ∂d_ij/∂x_i = (x_i-x_j) × s₀² / d_ij × s₀ = (x_i-x_j) × s₀ / d_ij
  ∂d_ij/∂y_i = (y_i-y_j) × s₀ / d_ij

  协方差: σ²(d_ij) = [(x_i-x_j)² + (y_i-y_j)²] × σ²_pos × s₀² / d_ij²
                     = σ²_pos × s₀²

→ σ_d = σ_pos × s₀   (与距离本身无关！所有距离测量精度相同！)

数值: σ_pos = 0.5px, s₀ = 2"/px → σ_d = 1.0"

这意味着:
  ✓ 近距和远距的测量精度相同 → 不存在"小距离精度差"的问题
  ✓ 6 个距离的噪声独立且同分布 (i.i.d. N(0, σ_d²))
  ✓ k-vector 查询容差可以统一设为 ±3σ_d ≈ ±3"

与 Astrometry.net 归一化hash的对比:
  归一化后最远对固定为1 → 近距的归一化值被放大 → 噪声也被放大
  例如 d_AC=10", d_AB=1000" → norm = 0.01, 噪声被放大100倍
  4SADQ 的绝对距离避免了这个问题
```

---

## 2. 索引结构：k-vector 范式下的四边形检索

### 2.1 从哈希桶到 k-vector 连续检索

Mortari [1997]的k-vector技术是这个领域最重要的索引创新。传统哈希需要精确量化→bin→查找，而k-vector提供连续的范围查询能力。

```
传统哈希索引 (Astrometry.net, 前一版4SADQ):

  特征: 4D/6D 量化值 → uint64 hash_key
  查询: hash_key ∈ {key-δ, key, key+δ}
  问题:
    - 固定binning → 距离在bin边界时"跨bin"问题
    - 必须枚举所有 δ 组合 (3^6=729 → 去重后~200个hash)
    - 200次随机内存访问 → cache miss 严重

k-vector 索引 (本方案):

  特征: d_AB = 6个距离中选的最长边 (arcsec)
  索引构建:
    (1) 计算所有参考四边形的 d_AB
    (2) 按 d_AB 升序排序
    (3) 存储排序数组 S[d_AB_sorted]
    (4) 建立 k-vector 辅助数组: K[j] = index_of_last_element ≤ (j+1)*Δ
  
  查询 d_AB ∈ [d - δ, d + δ]:
    j_lo = floor((d-δ)/Δ), j_hi = floor((d+δ)/Δ)
    idx_lo = K[j_lo], idx_hi = K[j_hi]
    → 直接获得候选区间 [idx_lo, idx_hi]  → O(1) 定位 + O(k) 扫描

  优势:
    - 连续范围查询 → 无binning损失 → 可用更大的容差而不增加假匹配
    - 顺序内存访问 (区间扫描) → cache友好
    - 单次查询 vs 200次hash → ~200x 更快(cache效应)

k-vector 参数:
  N_entries = 4.5×10^7 (总四边形数)
  d_AB_range = [2", 7200"]  (0.0006° ~ 2°)
  Δ = 0.5" (k-vector 步长)
  K数组大小 = 7200/0.5 = 14400 个元素
  k ≈ N_entries / 14400 ≈ 3125 (平均每bin的候选数)

  3125 个候选还需用剩余5个距离过滤 → 最终到0-5个
```

### 2.2 多层索引加速：双级k-vector + 低维过滤

```
为了将 k≈3125 降到可直接处理的程度，采用双级索引:

Level 1 — d_AB k-vector:
  查询: d_AB ∈ [d-3σ_d, d+3σ_d] → ~3125 候选
  目的: 粗筛，去除非该尺度的四边形

Level 2 — 5D 距离验证 (在3125候选上):
  对每个候选:
    验证 |d_AC - d_AC_cat| < 3σ_d
    验证 |d_AD - d_AD_cat| < 3σ_d
    验证 |d_BC - d_BC_cat| < 3σ_d
    验证 |d_BD - d_BD_cat| < 3σ_d
    验证 |d_CD - d_CD_cat| < 3σ_d
    
  全部通过 → 保留，否则 → 丢弃

  期望通过率: P_pass = Π_i P(|Δd_i| < 3σ_d) 
                        = [Φ(3)]^5 ≈ (0.9973)^5 ≈ 98.7% (对真四边形)
                        = [3σ_d / range_d]^5 ≈ (6"/3600")^5 ≈ 10^-14 (对随机四边形)

  期望候选数: 3125 × 10^-14 ≈ 3×10^-11 → 0 (随机假四边形全部消除)
             + 真四边形: 3125 × 98.7% ≈ 3083, 但索引中该天区只有~10个
                        实际该查询区间内含该天区的四边形 ~10-30个
             → 最终候选: 0-3个

问题: 3083 次验证仍较多

加速 — 用第二个独立k-vector:
  Level 2a — d_BC k-vector (次长边):
    在 S 数组的 [idx_lo, idx_hi] 区间子集中，
    再按 d_BC 建立局部 k-vector
    → 查询 d_BC ∈ [d_BC-δ, d_BC+δ] → 再缩小到 ~50 候选
    → 然后做余下4距离验证
  
  或更简单的方案:
    在 L1 的 3125 候选上，先过滤 d_BC (这个维度"散度"最大)
    → 简单循环 + 浮点比较，3125次检查 < 0.01ms
    → 剩余 ~50 候选，再做全5距离验证
```

### 2.3 索引文件格式 (最终设计)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 4SADQ 索引文件 (binary, little-endian)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Header (64 bytes):
  magic:          uint64  0x345144514B564543  ("4SADQ-KVEC")
  version:        uint32  1
  num_quads:      uint64  N (总四边形数, ~4.5×10^7)
  kvector_delta:  float32 Δ = 0.5  (k-vector步长, arcsec)
  d_min:          float32 2.0  (最小 d_AB, arcsec)
  d_max:          float32 7200.0  (最大 d_AB, arcsec)
  quant_step:     float32 0.2  (距离存储量化步长, arcsec)
  reserved:       uint8[28]

K-vector 区间表 (inline after header):
  kvec[j]:  uint32  S 数组中第 j*Δ 到 (j+1)*Δ arcsec 区间的结束下标
  kvec_size = ceil((d_max - d_min) / Δ) ≈ 14400
  kvec_bytes = 14400 × 4 = 57.6 KB

排序四边形数组 S (inline after k-vector):
  按 d_AB_quantized 升序排列
  每条目 (20 bytes):
    d_AB:        uint16  量化值 = round(d_AB / quant_step)
    d_AC:        uint16
    d_AD:        uint16
    d_BC:        uint16
    d_BD:        uint16
    d_CD:        uint16
    ra_center:   int32   RA × 10^7 (微角秒) — 四边形质心
    dec_center:  int32   Dec × 10^7

  S_bytes = N × 20 ≈ 4.5×10^7 × 20 ≈ 900 MB

总文件大小: 64 + 57600 + 900,000,000 ≈ 900 MB
(mmap加载，OS按需分页，实际内存占用 ~100 MB 热数据)
```

---

## 3. 星选择策略：金字塔最优排列 + 亮度先验

### 3.1 Mortari 金字塔的启示

金字塔算法 [Mortari et al., 2004] 的核心洞察：

```
误匹配频率的闭式解 (Kumar 2010):
  M颗星闭合多边形:
    f_false ≈ N_cat^M × (K·σ)^(2M-3) / π^(M-2)

  其中 σ 是角距测量噪声标准差，K≈6 (3σ范围)

对于 4 星四边形 (M=4):
  f_false ≈ N_cat^4 × (K·σ)^5 / π^2

关键推论: 
  - 误匹配率 ∝ σ^5 → 选择角距测量精度更高的星对
  - 角距测量噪声 σ_d ∝ σ_pos (对所有距离一样)
  - 但 d_AB 是特性区分度的主要贡献者 → 选择"更独特"的 d_AB
  - 独特性 = d_AB 附近密度低 → 在 k-vector 稀疏区间

金字塔的最优排列:
  优先选择角距最小的星对 (而非最亮)
  理由: 近距离星对误匹配概率更低
        (小角距的四边形容差空间更小)
```

### 3.2 融合方案：亮度优先 + 金字塔角度约束

```
4SADQ 星选择策略 (融合金字塔 + 亮度 + 空间覆盖):

已知: 亮星池 pool (Top-20 最亮星), 预计算了全配对距离

Step 1 — 候选四边形生成 (从 pool 中):
  对 pool 中每颗星作为候选 pivot:
    取 pivot 的8颗最近邻星 (按像素距离排序) → local_pool
    从 {pivot} ∪ local_pool[:3] 中取4星组合 (C(3,3)=1种 per pivot)
  
  最多 20×1 = 20 个候选四边形
  → 比 C(20,4)=4845 少 240 倍
  
Step 2 — 四边形质量排序:
  对每个候选四边形:
    score = α × uniqueness(d_AB) + β × geometry_quality(quad)
    
    uniqueness(d_AB):
      检查 d_AB 在 k-vector 的局部密度
      uniqueness ∝ 1 / local_density(d_AB)
      优先选"该尺度上四边形少"的特征
    
    geometry_quality(quad):
      最小角 > 10° (避免退化)
      四边形面积 / (d_AB × d_CD) > 0.1 (形状非退化)
      最短边 > 10" (避免极近星对噪声放大)
  
  按 score 降序排列 → 取前 5 个

Step 3 — 亮度作为 tie-breaker:
  相同 uniqueness 时，优先选包含更亮星的四边形
  (亮星 = 更高概率出现在索引中)

Step 4 — 空间覆盖约束:
  确保 5 个 quad 的质心在图像中分布均匀
  → 若 5 个都集中在同一区域 → 替换最差的那个
```

---

## 4. 验证框架：几何投票 + 贝叶斯双重保险

### 4.1 Kolomenkin 几何投票应用于四边形验证

Kolomenkin [2008]的几何投票方法利用星对角距对称性进行全局一致性验证。我们将其推广到四边形匹配：

```
传统方法 (逐个quad独立匹配):
  quad_1 → k-vector → candidate_list_1 → WCS_1
  quad_2 → k-vector → candidate_list_2 → WCS_2
  ...
  
  问题: 各quad独立决策，相互不通信
        一个quad的假匹配可能不会被其他quad发现

Kolomenkin投票方法:
  quad_1 → candidate_1 = {星 (a₁,b₁,c₁,d₁), (a₂,b₂,c₂,d₂), ...}
  quad_2 → candidate_2 = {星 (e₁,f₁,g₁,h₁), (e₂,f₂,g₂,h₂), ...}
  
  投票机制:
    对每个 cand in candidate_1:
      用 cand 的 4 颗星求解 WCS_cand
      用 WCS_cand 将 quad_2 的 4 颗图像星投影到天球
      在天球上检查这 4 个投影位置是否与 candidate_2 中的星匹配
      若能匹配 → cand 得 1 票
      (本质: 两个独立quad "共识"于同一个 WCS)

  几何解释:
    若 quad_1 的候选是真 → 导出的 WCS 是正确的
    → quad_2 在正确 WCS 下的投影应该匹配星表中正确的星
    → 两个正确答案互洽

    若 quad_1 的候选是假 → 导出的 WCS 是随机的
    → quad_2 在随机 WCS 下匹配到正确星的概率 ≈ 0
    → 假候选不得票

  消除率分析:
    假设 quad_1 产生 3 个候选 (1真2假)
    假设 quad_2 产生 2 个候选 (1真1假)
    
    真真配对: 1对 → 1票
    真假配对: 1×1 = 1对 → 每个假WCS下quad_2投影随机 → 0票 (概率 ≈0)
    假真配对: 2×1 = 2对 → 同上
    假假配对: 2×1 = 2对 → 同上

    最终: 真候选得 1票, 假候选得 0票 → 唯一确定!
    
    更一般地: N 个quad各产生 K_i 个候选
    真候选期望得票: (N-1)票 (所有其他quad的正确答案都投票)
    假候选期望得票: 0票 (其他quad的投影随机)
```

### 4.2 贝叶斯验证

采纳 Astrometry.net [Lang et al., 2010] 的贝叶斯因子框架，但使用四边形匹配的特化形式：

```
假设检验:
  H₀: 该四边形匹配是随机的 (虚警)
  H₁: 该四边形匹配是正确的

贝叶斯因子:
  K = P(data | H₁) / P(data | H₀)

data = {n_matched_quads} (共识投票中得票数)

P(data | H₁) = 所有quad都有正确候选但位置噪声导致部分不匹配

  简化计算 (类似 Lang 2010 的泊松近似):
    设期望得票数 = N_quads - 1 (排除自己)
    实际得票数 = n_votes
    
    ln K = n_votes × ln(1 / P_false) 
           - (N_quads - 1 - n_votes) × ln(1 / (1 - P_false))
    
    其中 P_false ≈ d_max² / (360° × 3600" × FOV_sr)
                ≈ (7200")² / (4π × (180/π)² × 3600² × FOV_rad²)
                ≈ 10^-9  (假WCS下随机匹配概率)

  典型值:
    N_quads = 5, n_votes = 4
    ln K ≈ 4·ln(10^9) - 1·ln(1) ≈ 4×20.7 ≈ 82.8
    K ≈ e^82.8 ≈ 10^36 >> 10^9 ✓

  阈值: 同 Lang 2010, K > 10^9 → 接受
        或 ln K > 20.7

  与 Lang 2010 的区别:
    Lang: 计算单个quad的贝叶斯因子 (基于所有星的一致性)
    我们: 计算跨quad的贝叶斯因子 (基于独立quad的共识)
    
    我们的方法: 每个quad提供独立证据 → 因子相乘 → ln K 相加
```

### 4.3 Heyl k-d match 假阳性消除

Heyl [2013]采用双 k-d 树进行四边形变换参数的一致性验证。我们将其融入投票后的精验证：

```
投票后，我们有 1 个或少数几个候选 WCS 解。

Heyl 式验证:
  1. 用候选 WCS 将 pool 中全部亮星投影到天球
  2. 在候选天区构建 Gaia 星的 KD-tree
  3. 批量最近邻搜索 → 获得匹配对列表
  4. 用这些匹配对重新拟合 WCS (Umeyama SVD)
  5. 计算残差 RMS
  
  若 RMS < 阈值 且 内点数 > 阈值 → 接受
  否则 → 拒绝

  与 Heyl 的对应:
    Heyl: 第一棵KDtree存储候选变换参数
          第二棵KDtree验证几何一致性
    我们: 第一级=投票得出的WCS候选
          第二级=全量内点验证 + 残差分析
```

---

## 5. 完整算法

### 5.1 伪代码

```
══════════════════════════════════════════════════════════════
  算法: 4SADQ-KV (4-Star Absolute Distance Quad with K-Vector)
  输入: 图像星点 {p_i} (N≥4), 像素尺度 s₀
  输出: WCS (CRVAL, CD, CRPIX) 或 FAILURE
══════════════════════════════════════════════════════════════

// ═══ 阶段0: 初始化 ═══
亮星池 = 按亮度取 Top-min(N, 20)
预计算 pool 内全配对距离矩阵

// ═══ 阶段1: 四边形生成 (金字塔约束) ═══
quads = []
for each pivot in pool[:15]:
  邻星 = pivot的8最近邻 (按像素距离)
  从 {pivot} ∪ 邻星[:3] 中取唯一四星组合
  几何排序 → (A,B,C,D) 有序组
  quality = uniqueness(d_AB) + geometry_score
  quads.append((quality, A,B,C,D))
quads 按 quality 降序 → 取前 5

// ═══ 阶段2: k-vector查询 + 候选收集 ═══
all_candidates = []  // (quad_idx, candidate_list)

for each quad in quads[:5]:
  6个角距: d = pixel_dist × s₀
  
  // k-vector Level 1: d_AB 范围查询
  idx_lo, idx_hi = kvector_query(d[AB] - 3σ, d[AB] + 3σ)
  
  // Level 2: 5距离顺序验证
  candidates = []
  for i in idx_lo .. idx_hi:
    cat = S[i]  // 索引条目
    if |d[AC]-cat.AC| < 3σ and |d[AD]-cat.AD| < 3σ and
       |d[BC]-cat.BC| < 3σ and |d[BD]-cat.BD| < 3σ and
       |d[CD]-cat.CD| < 3σ:
      candidates.append(cat)
    if len(candidates) > 10: break  // 早停, 假四边形太多

  all_candidates.append((quad, candidates))

// ═══ 阶段3: Kolomenkin几何投票 (N_quads≥3时) ═══
if len(quads) >= 3:
  vote_counts = {}  // WCS_hash → count
  
  for (quad_A, cands_A) in all_candidates:
    for cand in cands_A:
      WCS_A = solve_wcs(quad_A.points, cand.stars)
      WCS_A_hash = discretize(WCS_A)  // 量化CRVAL到~0.1°
      
      // 用 WCS_A 验证其他 quad
      for (quad_B, cands_B) in all_candidates:
        if quad_A == quad_B: continue
        proj_B = project(quad_B.points, WCS_A)  // 投影到天球
        for cand_B in cands_B:
          if max(|proj_B[k] - cand_B.stars[k]|) < 3":
            vote_counts[WCS_A_hash] += 1
            break

  // 选出得票最高的 WCS
  best_WCS_hash = argmax(vote_counts)
  if vote_counts[best_WCS_hash] < 2:
    goto stage_4  // 投票未达共识
  best_candidates = [c for c in all_candidates if WCS(c) ≈ best_WCS_hash]

else:  // N_quads < 3 (星太少)
  best_candidates = 第一个quad的所有候选

// ═══ 阶段4: 贝叶斯验证 ═══
for each cand in best_candidates:
  WCS = solve_wcs(cand.quad_points, cand.stars)
  
  // 计算贝叶斯因子
  n_matched = 用 WCS 验证 pool 中全部星 → 内点计数
  n_quads_voted = vote_counts[WCS_hash]  // 或 N-1 若直接验证
  
  ln_K = compute_bayes_factor(n_matched, n_quads_voted, N_quads)
  
  if ln_K > 20.7:  // K > 10^9
    // Heyl式精验证
    inliers = 全量匹配对 (WCS投影 + KDtree邻近)
    WCS_refined = Umeyama_LSQ(inliers)
    RMS = compute_rms(WCS_refined, inliers)
    
    if RMS < 3.0" and len(inliers) ≥ 2·|pool|/3:
      return WCS_refined  // ← 成功!

// ═══ 阶段5: PROSAC 渐进扩张 (阶段2-4未命中时) ═══
pool_expand: Top-20 → Top-30 → Top-50
for each expansion:
  生成新的5个quad (用金字塔约束)
  重复阶段2-4
  if 命中 → return WCS

// ═══ 阶段6: 终极回退 ═══
if 全部失败:
  回退到 V4.0 SCM 方法 (对极端少量星有独特鲁棒性)
```

### 5.2 各阶段计算量

```
阶段 0 (初始化):       O(N log N) + O(20²), <0.01ms
阶段 1 (四边形生成):    O(15 × sorting) ≈ O(15), <0.01ms
阶段 2 (k-vector查询):  
  L1: O(1) index + O(k) scan, k≈3000 → ~3000次内存读, <0.01ms
  L2: 3000 × 5次浮点比较 → <0.01ms
  5 quads × 2 step: <0.1ms
阶段 3 (几何投票):     
  最坏: 5 quads × 3 cands × 4 other_quads × 3 cands = 180次WCS验证
  每WCS验证: 4点投影 + 索引查找 → <0.01ms
  总: <2ms
阶段 4 (贝叶斯+精验):  
  K个WCS候选 (K≤3): K × (pool投影 + Gaia KDTree + LSQ) ≈ 3×1ms → <3ms
阶段 5 (渐进扩张):     
  最多触发2次 × 阶段2-4 → <10ms (极少触发)

总耗时: 典型 <5ms, 最坏 <20ms
```

---

## 6. 误匹配概率上界——Kumar范式下的理论保证

### 6.1 随机四边形误匹配概率

```
Kumar [2010] 的M星闭合多边形误匹配频率通式:

  f_false(M) ≈ N_cat^M × (K·σ_d)^(2M-3) / π^(M-2)

  其中:
    N_cat ≈ 4.5×10^7 (索引四边形总数)
    σ_d ≈ 1" (距离测量噪声标准差)
    K = 6 (3σ范围)
    M = 4 (四边形)

  f_false(4) ≈ (4.5×10^7)^4 × (6×1")^5 / π^2
              ≈ 4.1×10^30 × 7776 / 9.87
              ≈ 3.2×10^33  (!!!)

  但这只考虑距离随机匹配，没考虑6D全距离约束。
  
修正: 距离不是独立的 → 有效自由度=5

  f_false_eff ≈ N_cat^4 × (6σ/range)^5 / (2π)²
                ≈ 10^30 × (6"/7200")^5
                ≈ 10^30 × (8.3×10^-4)^5
                ≈ 10^30 × 4.0×10^-16
                ≈ 4×10^14

  仍然很大 — 因为 N_cat^4 主导
  
但这是"全索引随机匹配"的期望，实际查询是：
  k-vector d_AB 约束 → 只有 ~3000 候选 → 相当于 N_cat_effective = 3000

  f_false_query ≈ 3000 × (6"/7200")^4 / 2π
                 ≈ 3000 × 4.8×10^-13 / 6.28
                 ≈ 2.3×10^-10

  单次查询的假匹配概率 < 10^-9 ✓
```

### 6.2 几何投票的误匹配消除率

```
设: 
  N_quads 个独立四边形参与投票
  每个四边形在查询后产生 K 个候选 (K≈2, 含 1真+1假)
  假候选的 WCS 是随机在天球上分布的
  
投票消除:
  P(假候选得 ≥2 票) = P(≥2 个独立quad的假WCS恰好指向同一位置)
                     ≈ C(N_quads, 2) × (ΔRA×ΔDec / 全天球面积)
                     ≈ C(5,2) × (0.1°×0.1° / 41253 deg²)
                     ≈ 10 × 2.4×10^-7
                     ≈ 2.4×10^-6

贝叶斯消除:
  ln K > 20.7 → 额外降至 ~10^-9

联合:
  P(误匹配输出) < 3×10^-15  (每帧)
  → 天文尺度: 100万帧中误报 < 1次
  → 满足科学零误报需求
```

---

## 7. 消融分析：每个"借来的方法"贡献了什么

### 7.1 方法贡献矩阵

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 组件           借自           贡献                移除后果
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 已知s₀(+绝对距离) Christian 2021 信息量+10^18        退化为Astrometry
    
 k-vector索引     Mortari 1997  O(k)连续查询,        200次hash查询
                                消除量化损失          cache miss严重

 金字塔选星       Mortari 2004  优化四边形选择,       Top-4全退化时
                               避免退化构型           无回退机制

 Kolomenkin投票   Kolomenkin    独立quad交叉验证,     假匹配候选混入
                  2008          消除歧义候选          WCS输出错误

 贝叶斯验证       Lang 2010     理论的零误报保证      n_matched阈值
                                                     可能被突破

 PROSAC渐进       Chum 2005     亮星→暗星有序采样     Top-20失败后
                                                    无后续尝试

 n维k-vector理论  Arnas 2020    多维正交范围搜索      降为1D k-vector
                                                     候选数×100

 Heyl双k-d精验    Heyl 2013     变换参数一致性验证     WCS精度粗
                                                     无亚像素精度

 Kumar误匹配界    Kumar 2010    预期误匹配率闭式解    无法理论保证
                                  → 参数优化依据
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 7.2 为什么这确实不同——与 Astrometry.net 的根本区别

```
                     Astrometry.net           4SADQ-KV (本方法)

索引范式           哈希表 (离散)             k-vector (连续)
尺度处理           多版本索引选择            嵌入特征(d = px×s₀)
特征空间           4D归一化                 6D绝对量
唯一性来源         归一化形状                绝对距离 + 形状
旋转不变           最远对强制对齐            距离天然不变
索引复杂度         O(1) hash lookup          O(k) range scan
查询容差处理       bin ±1 枚举              连续容差, 无binning损失
验证机制           贝叶斯单因子             Kolomenkin投票+贝叶斯
候选数             10²-10⁴                  0-5
误匹配率           贝叶斯保证               Kumar界+投票+贝叶斯保证
选星策略           无优化                   Pyramid约束+亮度+空间覆盖
尺度鲁棒性         需多索引(离散)           单索引(连续容差自适应)
失败回退           无                       PROSAC渐进+V4互补
```

---

## 8. 实现路径

### Phase 1: k-vector 原型 (Python, 1天)
```
1. 从测试帧取4颗亮星 → 计算6距离
2. 在本地构建小规模 k-vector (1000个参考四边形)
3. 验证: 查询返回候选数 < 5
4. 对比: hash vs k-vector 的查询性能
```

### Phase 2: 索引构建 (Python+C, 2天)
```
1. 从 Gaia DR3 G<10.5 建四边形库 → 排序 → k-vector
2. 二进制索引文件写入
3. 随机查询验证区分度
```

### Phase 3: C++ 集成 (3天)
```
1. mmap 索引加载器 + k-vector 查询
2. 金字塔选星器 + 几何投票引擎
3. 贝叶斯因子计算
4. WCS 直接求解 + Heyl精验
```

### Phase 4: 562帧测试 (2天)
```
盲解析成功率 + WCS精度 + 耗时 + 失败分析
```

---

## 9. 总结

**本方案 = Astrometry.net的结构框架 + Mortari/Arnas的k-vector索引范式 + Mortari/Du的金字塔选星理论 + Kolomenkin的全局投票机制 + Kumar的误匹配理论界 + Lang/Clouse的贝叶斯验证 + Chum的PROSAC渐进策略 + Heyl的双树精验**

而核心创新——**将绝对距离（利用已知s₀）作为k-vector的排序键，把"多尺度离散索引选择"升级为"单一连续索引检索"**——才是本方案超越 Astrometry.net 的根本原因。

这不是简单的追加优化，而是**索引范式的根本改变**：从"我有多个尺度索引，选一个查找"到"我有连续索引，任何尺度都直接检索到"，就像从图书馆的"按书架号查找"升级到"全文搜索引擎"。

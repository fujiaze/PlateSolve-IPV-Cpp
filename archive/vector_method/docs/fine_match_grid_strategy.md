# Step2 三角形匹配精细拟合策略设计

## 1. 概述

在Step1粗匹配获得初始Affine变换后，Step2使用三角形匹配替代网格就近匹配，解决畸变导致就近检索错误匹配的问题。三角形边长比是旋转/缩放不变量，天然抗畸变。

## 2. 核心流程

```
Step1 Affine变换 → Gaia星点投影到图像坐标
                    ↓
取前1000亮图像星点 + 前1500亮Gaia投影星点
                    ↓
图像星点构建三角形 + Gaia投影星点构建三角形
                    ↓
三角形匹配：边长比(ba_ratio, ca_ratio) + 重心距离20px (KDTree)
                    ↓
提取星点匹配对（三角形顶点对应）
                    ↓
RANSAC过滤错误匹配
                    ↓
拟合SIP多项式 + Affine参数
                    ↓
输出完整WCS：准确中心RA/Dec、比例尺、翻转、SIP多项式
```

## 3. 输入数据

| 数据 | 来源 | 数量 |
|------|------|------|
| 图像星点 | star detector返回 | 前1000亮度 |
| Gaia星点 | Gaia数据库查询 | 前1500亮度 |
| Step1变换 | Step1输出的Affine 6参数 | — |

## 4. 坐标预处理

### 4.1 图像星点
直接使用相对于图像中心的坐标 `(img_x, img_y)`，已处理翻转。

### 4.2 Gaia星点
1. Gnomonic投影：`(ra, dec) → (xi, eta)`
2. 转像素：`px = xi * rad_to_px, py = -eta * rad_to_px`
3. 用Step1的Affine逆变换映射到图像坐标：
   ```
   fimg_x = inv_a0 + inv_a1 * px + inv_a2 * py
   fimg_y = inv_b0 + inv_b1 * px + inv_b2 * py
   ```
4. 应用Flip：`cat_px = flip_x ? -fimg_x : fimg_x`

## 5. 三角形构建

复用Step1的 `sa_build_triangles` 函数：
- 每个点取最近15个邻居
- 遍历步长 `step = max(1, n/800)`
- 三角形过滤条件：
  - 最小面积：50px²
  - 最小角度：10°
  - 最大角度：70°
  - ba_ratio < 0.92（排除等腰三角形）
  - ba - ca > 0.03（排除退化三角形）
- 最大三角形数：20000

## 6. 三角形匹配（核心）

### 6.1 匹配条件
1. **边长比匹配**：`|ba_ratio_img - ba_ratio_cat| < 0.07` 且 `|ca_ratio_img - ca_ratio_cat| < 0.07`
2. **重心距离约束**：用Step1的Affine变换预测图像三角形重心对应的Gaia位置，在20px内搜索

### 6.2 KDTree优化
- 对Gaia三角形的重心位置建KDTree
- 对Gaia三角形的边长比建KDTree
- 先用重心距离20px做range_query缩小候选集
- 再在候选集中找边长比最近的

### 6.3 匹配流程
```
对每个图像三角形:
  1. 用Step1 Affine变换预测重心在Gaia坐标系的位置
  2. 在Gaia三角形重心KDTree中range_query(20px)
  3. 在候选集中找边长比最近的三角形
  4. 若边长比距离 < 0.07²，记录匹配对
```

## 7. 星点匹配对提取

匹配的三角形对 `(img_tri, cat_tri)` 中，3个顶点需要对应：
- 按边长排序：最长边对最长边，次长边对次长边，最短边对最短边
- 具体实现：对三角形的3个顶点按到重心的距离排序，距离最近的对应距离最近的

### 7.1 去重
- 同一个图像星点可能出现在多个三角形中
- 使用 `img_used` 标记避免重复匹配
- 保留匹配质量最好的配对

## 8. RANSAC过滤

### 8.1 算法
- 随机选3个匹配对，拟合Affine变换
- 统计内点（残差 < 5px）
- 迭代500次，保留内点最多的
- 用内点重新拟合

### 8.2 迭代MAD过滤
- 用内点拟合Affine
- 计算残差的median和MAD
- 阈值：`threshold = median + 3 * MAD * 1.4826`
- 剔除超限点
- 迭代直到剔除数量 < 5%

## 9. WCS参数计算

### 9.1 Affine拟合
用匹配对拟合6参数Affine变换：
```
cat_x = a0 + a1 * img_x + a2 * img_y
cat_y = b0 + b1 * img_x + b2 * img_y
```

从Affine系数提取物理参数：
- **偏移**：a0, b0
- **旋转角**：`rotation = atan2(b1, a1)`
- **比例尺**：`scale = sqrt(a1² + b1²) * original_scale`
- **翻转**：从Step1继承

### 9.2 准确中心计算
```
offset_x_px = a0  (像素偏移)
offset_y_px = b0  (像素偏移)
offset_ra = offset_x_px * scale_arcsec_px / 3600.0 / cos(center_dec_rad)
offset_dec = offset_y_px * scale_arcsec_px / 3600.0
center_ra_final = step1_ra + offset_ra
center_dec_final = step1_dec + offset_dec
```

### 9.3 SIP多项式拟合
用匹配对拟合SIP畸变系数：
- 输入：`(img_x, img_y) → (cat_x, cat_y)` 匹配对
- 输出：SIP A/B/AP/BP系数矩阵

### 9.4 WCS构建
- `CRPIX` = 图像中心
- `CRVAL` = 准确中心RA/Dec
- `CD矩阵` = 从比例尺+旋转角构建
- `CTYPE` = RA---TAN-SIP / DEC--TAN-SIP
- `SIP系数` = 拟合得到的多项式

## 10. 对外接口

保持 `psm_grid_match_perform` 的签名不变，内部实现替换为三角形匹配。

```cpp
GM_EXPORT int psm_grid_match_perform(
    const GMImageStars *img_stars,
    const GMCatalogStars *cat_stars,
    const GMInitialTransform *init_transform,
    const GMConfig *config,
    GMResult *out_result);
```

### 10.1 GMConfig新增参数
```cpp
typedef struct {
    int grid_size;            // 保留兼容，三角形匹配中用作最大三角形数/1000
    int max_cat_candidates;   // 保留兼容
    double match_tolerance;   // 匹配容差 (默认5px)
    int max_ransac_iter;      // RANSAC最大迭代次数 (默认500)
    double ransac_sigma;      // RANSAC剔除阈值系数 (默认3.0)
    int sip_order;            // SIP阶数 (默认5)
    double centroid_radius;   // 三角形重心搜索半径 (默认20px)
    double ratio_threshold;   // 边长比匹配阈值 (默认0.07)
    int n_img_bright;         // 图像星点取前N亮 (默认1000)
    int n_cat_bright;         // Gaia星点取前N亮 (默认1500)
} GMConfig;
```

### 10.2 GMResult保持不变
```cpp
typedef struct {
    int n_control_points;
    int n_grids_matched;
    int n_grids_total;
    double rms_x, rms_y, rms_total, rms_arcsec;
    int n_ransac_removed;
    GMControlPoint *control_points;
    double sip_A[6][6], sip_B[6][6], sip_AP[6][6], sip_BP[6][6];
    int sip_order, sip_valid;
    double cd[2][2];
    double crpix[2], crval[2];
} GMResult;
```

## 11. 调试输出

### 11.1 日志输出
- 三角形构建数量（图像/Gaia）
- 匹配三角形对数量
- 星点匹配对数量
- RANSAC前后匹配数
- Affine参数
- SIP拟合RMS
- WCS参数

### 11.2 CSV文件输出
- `debug_step2_predictions.csv`: Gaia预测位置（前1000亮星）
- `debug_control_points.csv`: 控制点匹配对

## 12. 预期效果

- 匹配对数量：500~1500（取决于星点密度）
- 匹配精度：RMS < 3px
- 畸变校正：有效校正光学畸变和场曲
- 抗畸变能力：三角形边长比是旋转缩放不变量，不受畸变影响

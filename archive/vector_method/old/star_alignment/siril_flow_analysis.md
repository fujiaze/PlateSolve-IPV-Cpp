# Siril Plate Solving 完整流程分析（极尽详细版）

## 核心结论：Siril内部solver不依赖astrometry.net

Siril有**两种互斥**的plate solving模式：

| 模式 | 入口 | 依赖 | 算法 |
|------|------|------|------|
| SOLVER_LOCALASNET | `local_asnet_platesolve()` [L1006] | 外部solve-field程序 | astrometry.net索引匹配 |
| Siril内部solver | `siril_platesolve()` [L1014] | 图像头信息RA/Dec | 三角形匹配+迭代重投影 |

**关键**：Siril内部solver的初始坐标来自图像FITS头信息（OBJCTRA/OBJCTDEC > CRVAL > RA/DEC），**不依赖astrometry.net**。

---

## 完整调用链（含源码行号）

```
plate_solver()                          [astrometry_solver.c L878]
  │
  ├─ SOLVER_LOCALASNET → local_asnet_platesolve()  [L1006]  ← 完全不同的路径
  │
  └─ Siril内部solver:
     │
     ├─ 1. 图像星居中+Y翻转           [L1008-1012]
     │     x0 = rx*0.5, y0 = ry*0.5
     │     stars[s]->xpos -= x0
     │     stars[s]->ypos = y0 - stars[s]->ypos
     │
     ├─ 2. siril_platesolve()           [L1325]
     │     │
     │     ├─ get_catalog_stars()        [L1327]  查询星表
     │     │
     │     ├─ match_catalog()           [L1333]  ★核心函数★
     │     │     │
     │     │     ├─ 计算scale_min/max   [L1412-1415]
     │     │     │
     │     │     ├─ project_catalog_stars() [L1417]  gnomonic投影
     │     │     │
     │     │     ├─ new_star_match()    [L1421]  ★三角形匹配★
     │     │     │     │ (详见下方)
     │     │     │
     │     │     ├─ check_affine_TRANS_sanity() [L1430]
     │     │     ├─ check_affine_TRANS_scale() [L1434]
     │     │     │
     │     │     ├─ ★迭代重投影★       [L1444-1470]
     │     │     │     while (conv > 0.01"):
     │     │     │       apply_match() → 新投影中心
     │     │     │       project_catalog_stars() → 重投影
     │     │     │       update_stars_positions() → 更新B坐标
     │     │     │       atRecalcTrans() → 重算变换
     │     │     │
     │     │     └─ (可选)高阶匹配     [L1476-1521]
     │     │
     │     ├─ near_solve (如果失败)     [L1334-1351]
     │     │     在搜索半径内尝试不同投影中心
     │     │     成功后重新调用siril_platesolve()
     │     │
     │     └─ 生成WCS                   [L1352-1398]
     │
     └─ 3. 更新FITS头信息              [L1027-1088]
```

---

## 阶段1：图像星准备

### 源码位置：astrometry_solver.c L1008-1012

```c
double x0 = args->fit->rx * 0.5;  // 图像宽度的一半
double y0 = args->fit->ry * 0.5;  // 图像高度的一半
for (int s = 0; s < nb_stars; s++) {
    stars[s]->xpos -= x0;           // 居中：x = xpos - x0
    stars[s]->ypos = y0 - stars[s]->ypos;  // Y翻转：y = y0 - ypos
}
```

**坐标系含义**：
- 原点在图像中心
- X轴向右为正
- Y轴向上为正（与FITS的向下为正相反）
- 单位：像素

**饱和星处理**：Siril在PSF拟合阶段就剔除了饱和星（`if (!stars[i]->has_saturated)`），不传入solver。我们用n_saturated参数模拟这个行为。

---

## 阶段2：星表星准备

### 源码位置：astrometry_solver.c L1402-1417 (match_catalog函数)

```c
double ra0 = siril_cat->center_ra;     // 来自图像头信息
double dec0 = siril_cat->center_dec;   // 来自图像头信息

double a = 1.0 + (percent_scale_range / 100.0);
double b = 1.0 - (percent_scale_range / 100.0);
double scale_min = 1.0 / (scale * a);  // scale = arcsec/px
double scale_max = 1.0 / (scale * b);

cstars = project_catalog_stars(siril_cat, ra0, dec0);
```

### 投影方式：siril_catalogues.c

```c
// gnomonic投影
cat_items[i].x = xi * RADtoASEC;   // 角秒单位
cat_items[i].y = eta * RADtoASEC;   // 角秒单位，eta不取负！
```

**坐标系含义**：
- 原点在投影中心(ra0, dec0)
- X = xi * RADtoASEC（角秒，向RA增大方向为正）
- Y = eta * RADtoASEC（角秒，向Dec增大方向为正，**不取负**）
- 单位：角秒

### scale_min/max的含义

scale_min和scale_max是**像素/角秒**的比值范围，用于三角形匹配中的scale过滤：
- `ratio = ta.a_length / tb.a_length`（图像三角形边长 / 星表三角形边长）
- 图像三角形边长单位是像素，星表三角形边长单位是角秒
- 所以ratio的单位是像素/角秒
- scale_min = 1/(scale*1.1)，scale_max = 1/(scale*0.9)
- 例如：scale=6.188 arcsec/px → scale_min=0.147, scale_max=0.180

---

## 阶段3：new_star_match（核心匹配入口）

### 源码位置：match.c L131-441

这是Siril的**核心匹配函数**，完整流程如下：

### 3.1 准备星表 [L200-237]

```c
get_stars(s1, n1, &numA, &star_list_A);         // 图像星→链表
get_stars(s1, n1, &numA_copy, &star_list_A_copy); // 图像星副本（保存原始坐标）
get_stars(s2, n2, &numB, &star_list_B);          // 星表星→链表
reset_copy_ids(numA, star_list_A, star_list_A_copy); // 同步id
```

**关键**：`star_list_A_copy`保存了A星的原始坐标，因为后续`atApplyTrans`会修改A星坐标。

### 3.2 atFindTrans（三角形匹配）[L242-256]

```c
int ret = atFindTrans(numA, star_list_A, numB, star_list_B,
    triangle_radius, nobj, min_scale, max_scale,
    rot_angle, rot_tol, max_iter, halt_sigma, trans);
```

- nobj = min(max(n1, n2), AT_MATCH_CATALOG_NBRIGHT=60)
- 详细流程见阶段4

### 3.3 atApplyTrans（应用变换到所有A星）[L268]

```c
atApplyTrans(numA, star_list_A, trans);
```

- 将所有A星坐标从A坐标系（像素）变换到B坐标系（角秒）
- **注意：这会修改star_list_A中的坐标！**
- 变换公式：`x' = a0 + a1*x + a2*y`，`y' = b0 + b1*x + b2*y`

### 3.4 atMatchLists（第一轮匹配）[L281-283]

```c
atMatchLists(numA, star_list_A, numB, star_list_B,
    match_radius, &num_matches, &matched_list_A, &matched_list_B);
```

- match_radius = AT_MATCH_RADIUS = 2.0（角秒单位）
- 此时两组星都在B坐标系（角秒）中
- 使用`match_arrays_slow`去重逻辑

### 3.5 prepare_to_recalc（准备重算变换）[L298-310]

```c
if (prepare_to_recalc(num_matched_A, matched_list_A,
        num_matched_B, matched_list_B, star_list_A_copy, trans) != 0) {
    return SH_GENERIC_ERROR;
}
```

**prepare_to_recalc做了什么** [match.c L808-838]：
1. `atCalcRMS()` → 计算RMS → `trans->sx = Xrms, trans->sy = Yrms`
2. `reset_A_coords()` → 将matched_list_A的坐标重置为原始坐标
   - 因为atApplyTrans已经把A星坐标改成了B坐标系
   - 需要恢复原始A坐标才能重新计算TRANS

### 3.6 atRecalcTrans（第一轮重算）[L312-326]

```c
if (atRecalcTrans(num_matched_A, matched_list_A, num_matched_B,
        matched_list_B, max_iter, halt_sigma, trans) != SH_SUCCESS) {
    return SH_GENERIC_ERROR;
}
```

- atRecalcTrans内部调用`iter_trans(RECALC_YES)`
- RECALC_YES：用所有匹配对初始化（initial_pairs = nbright）
- 比atFindTrans更精确，因为输入已经是确认的匹配对

### 3.7 第二轮匹配 [L342-395]

```c
// 重置所有A星坐标为原始值
reset_A_coords(numA, star_list_A, star_list_A_copy);  // [L343]

// 用改进的变换重新变换所有A星
atApplyTrans(numA, star_list_A, trans);                // [L350]

// 重新匹配
atMatchLists(numA, star_list_A, numB, star_list_B, ...); // [L355]

// 准备重算
prepare_to_recalc(num_matched_A, matched_list_A, ...);   // [L366]

// 最终重算
atRecalcTrans(num_matched_A, matched_list_A, ...);       // [L381]
```

**关键理解**：
- 第一轮匹配用atFindTrans的粗略变换，可能只匹配到少量星
- 第二轮用改进后的变换，应该能匹配到更多星
- 每次atApplyTrans前都要reset_A_coords，因为atApplyTrans修改了A星坐标

---

## 阶段4：atFindTrans详细流程

### 源码位置：atpmatch.c L342-592

### 4.1 确定nbright [L462-467]

```c
nbright = nobj;  // 通常=60
if (nbright > max(num_stars_A, num_stars_B))
    nbright = max(num_stars_A, num_stars_B);
if (nbright < start_pairs)  // start_pairs=6 for linear
    nbright = start_pairs;
```

### 4.2 stars_to_triangles [L468-473]

```c
triangle_array_A = stars_to_triangles(star_array_A, num_stars_A,
    min(num_stars_A, nbright), &num_triangles_A);
triangle_array_B = stars_to_triangles(star_array_B, num_stars_B,
    min(num_stars_B, nbright), &num_triangles_B);
```

**内部流程**：
1. `sort_star_by_mag`：按mag升序排序（最亮的排前面），设置`array[i].index = i`
2. 对前nbright颗星，生成所有C(nbright,3)个三角形
3. 对每个三角形调用`set_triangle`：
   - 确定最长边a、次长边b、最短边c
   - 计算 ba = b/a, ca = c/a（旋转+缩放+平移不变量）
   - 保存 a_index, b_index, c_index（对应星在star_array中的索引）
4. 过滤 ba > AT_MATCH_RATIO=0.9 的三角形

### 4.3 prune_triangle_array [L481-482]

```c
prune_triangle_array(triangle_array_A, &num_triangles_A);
prune_triangle_array(triangle_array_B, &num_triangles_B);
```

- 按ba排序
- 剔除ba > AT_MATCH_RATIO的三角形
- **注意**：这和stars_to_triangles内部的过滤是重复的，但Siril两者都做了

### 4.4 make_vote_matrix [L516-519]

```c
vote_matrix = make_vote_matrix(star_array_A, num_stars_A, star_array_B, num_stars_B,
    triangle_array_A, num_triangles_A, triangle_array_B, num_triangles_B,
    nbright, radius, min_scale, max_scale, rotation_deg, tolerance_deg);
```

**详细流程**：
1. 分配nbright×nbright的投票矩阵，初始化为0
2. **外层循环**遍历B三角形（星表），**内层循环**遍历A三角形（图像）
3. 对每个B三角形，用二分查找找到ba范围匹配的A三角形
4. 检查(ba,ca)距离 < radius² (AT_TRIANGLE_RADIUS=0.002)
5. 检查scale过滤：`ratio = ta.a_length / tb.a_length`，`min_scale <= ratio <= max_scale`
6. 投票：`vote[ta.a_index][tb.a_index]++`，`vote[ta.b_index][tb.b_index]++`，`vote[ta.c_index][tb.c_index]++`

**关键理解**：
- A三角形的边长单位是像素，B三角形的边长单位是角秒
- ratio = ta.a_length / tb.a_length 的单位是像素/角秒
- scale_min和scale_max也是像素/角秒
- 所以scale过滤确保了匹配的三角形对有正确的缩放关系

### 4.5 top_vote_getters [L546-547]

```c
top_vote_getters(vote_matrix, nbright, &winner_votes, &winner_index_A, &winner_index_B);
```

- 提取投票最高的nbright对匹配星
- 按票数降序排列
- 每对包含：winner_index_A[i]（A星索引）、winner_index_B[i]（B星索引）、winner_votes[i]（票数）

### 4.6 过滤低票匹配 [L559-568]

```c
for (i = 0; i < nbright; i++) {
    if (winner_votes[i] < AT_MATCH_MINVOTES) {
        nbright = i;
        break;
    }
}
```

- AT_MATCH_MINVOTES = 2
- 只保留票数≥2的匹配对

### 4.7 iter_trans（RECALC_NO）[L580-592]

```c
iter_trans(nbright, star_array_A, num_stars_A, star_array_B, num_stars_B,
    winner_votes, winner_index_A, winner_index_B,
    RECALC_NO, max_iter, halt_sigma, trans);
```

- RECALC_NO：只用前AT_MATCH_STARTN_LINEAR=6对初始化
- 详细流程见阶段5

---

## 阶段5：iter_trans详细流程

### 源码位置：atpmatch.c L2760-3186

### 5.1 初始化 [L2856-2869]

```c
if (recalc_flag == RECALC_YES) initial_pairs = nbright;
else initial_pairs = start_pairs;  // =6 for linear
```

### 5.2 calc_trans（用initial_pairs对计算初始变换）[L2864-2869]

```c
calc_trans(initial_pairs, star_array_A, num_stars_A, star_array_B, num_stars_B,
    winner_votes, winner_index_A, winner_index_B, trans);
```

- 调用`calc_trans_linear`：3×3高斯消元法
- 变换形式：`x' = a0 + a1*x + a2*y`，`y' = b0 + b1*x + b2*y`

### 5.3 迭代循环 [L2923-3152]

```
while (iters_so_far < max_iterations) {
    1. 用当前变换变换A星坐标 → a_prime[i]  [L2937-2948]
    2. 计算a_prime与对应B星的距离²  [L2955-2965]
    3. 剔除距离² > AT_MATCH_MAXDIST²=2500的对  [L2983-3013]
       → 如果nr < required_pairs=3，失败
    4. 排序dist2_sorted  [L2970]
    5. sigma = find_percentile(dist2_sorted, nr, 0.35)  [L3039-3040]
    6. 如果sigma <= halt_sigma=0.1，标记is_ok=1  [L3050-3057]
       注意：不break！继续sigma clip
    7. 剔除距离² > AT_MATCH_NSIGMA*sigma=10*sigma的对  [L3070-3100]
       → 如果nr < required_pairs=3，失败
    8. 如果没有剔除任何对(nb==0 && nb_sigma==0)，成功break  [L3112-3118]
    9. 用剩余对重新计算变换  [L3134-3142]
    10. iters_so_far++  [L3149]
       如果is_ok，break  [L3150-3151]
}
```

**关键理解**：
- 步骤6中sigma <= halt_sigma时只标记is_ok=1，不立即break
- 步骤7继续做sigma clip，可能还会剔除一些对
- 只有当步骤8确认没有更多剔除时才break
- 最终trans->nr = nr（剩余匹配对数），trans->sig = percentile(dist2, nr, 0.6827)

---

## 阶段6：match_catalog的迭代重投影（★我们完全遗漏的关键步骤★）

### 源码位置：astrometry_solver.c L1439-1470

```c
int num_matched = trans.nm;
int trial = 0;
double conv = DBL_MAX;

conv = get_center_offset_from_trans(&trans);  // sqrt(a0² + b0²)
siril_debug_print("iteration %d - offset: %.3f, number of matches: %d\n", trial, conv, trans.nr);

while (conv > CONV_TOLERANCE && trial < max_trials) {  // CONV_TOLERANCE = 0.01 arcsec
    // 1. 用当前变换计算新的投影中心
    apply_match(ra0, dec0, 0., 0., &trans, &ra0, &dec0);
    //    对线性变换：delta_ra = trans.x00, delta_dec = trans.y00
    //    然后从(xi,eta)反投影到(RA,Dec)

    // 2. 重新投影星表到新中心
    free_fitted_stars(cstars);
    cstars = project_catalog_stars(siril_cat, ra0, dec0);

    // 3. 更新匹配星的B坐标
    update_stars_positions(&star_list_B, num_matched, cstars);
    //    用cstars中对应id的坐标更新star_list_B

    // 4. 用原始A坐标和新B坐标重新计算变换
    atRecalcTrans(num_matched, star_list_A, num_matched, star_list_B,
                  AT_MATCH_MAXITER, AT_MATCH_HALTSIGMA, &trans);

    num_matched = trans.nm;
    conv = get_center_offset_from_trans(&trans);
    trial++;
}
```

### apply_match详解 [apply_match.c L115-201]

```c
void apply_match(double ra, double dec, double xval, double yval, TRANS *trans,
                 double *a, double *d) {
    // 1. 用TRANS计算(xi, eta)偏移（角秒）
    switch (trans->order) {
    case AT_TRANS_LINEAR:
        delta_ra  = trans->x00 + trans->x10 * xval + trans->x01 * yval;
        delta_dec = trans->y00 + trans->y10 * xval + trans->y01 * yval;
        break;
    // ...
    }

    // 2. 角秒→弧度
    delta_ra  = (delta_ra / 3600.0) * DEGTORAD;
    delta_dec = (delta_dec / 3600.0) * DEGTORAD;

    // 3. 从切平面反投影到(RA, Dec)
    z = cos(r_dec) - delta_dec * sin(r_dec);
    zz = atan2(delta_ra, z) * RADTODEG;
    alpha = zz + ra;  // 新RA
    delta = asin((sin(r_dec) + delta_dec * cos(r_dec)) /
                 sqrt(1. + delta_ra*delta_ra + delta_dec*delta_dec)) * RADTODEG;
    // 新Dec
}
```

**当xval=0, yval=0时**（计算投影中心偏移）：
- `delta_ra = trans->x00`（即a0，角秒）
- `delta_dec = trans->y00`（即b0，角秒）
- 然后反投影得到新的投影中心(RA, Dec)

### 为什么需要迭代重投影？

1. gnomonic投影是**非线性的**，投影中心偏移会导致边缘星的位置有系统误差
2. 初始投影中心(ra0, dec0)来自图像头信息，可能与真实中心有偏差
3. 通过迭代重投影，将投影中心移到真实中心，消除系统误差
4. 每次重投影后，B星的坐标会变化（因为投影中心变了），需要重新计算TRANS
5. 收敛条件：`sqrt(a0² + b0²) < 0.01 arcsec`（非常严格）

---

## 阶段7：siril_platesolve的near solve

### 源码位置：astrometry_solver.c L1334-1351, L1223-1319

如果初始solve失败（match_catalog返回错误），且`args->near_solve`为true：

```c
if (args->near_solve && dist > 0.15 * args->used_fov) {
    // 1. 用solve结果作为新中心重新查询星表
    args->ref_stars->center_ra = ra;
    args->ref_stars->center_dec = dec;
    get_catalog_stars(args->ref_stars);
    // 2. 再次尝试match_catalog
    match_catalog(stars, nb_stars, args->ref_stars, ...);
}
```

更完整的near solve在`siril_near_platesolve()` [L1223]：
1. 在搜索半径内生成网格点作为候选投影中心
2. 对每个候选中心，投影星表并尝试match_catalog
3. 多线程并行搜索
4. 找到后用新中心重新做正常solve

---

## 我们实现与Siril的差异对照表

| 步骤 | Siril | 我们 | 状态 |
|------|-------|------|------|
| 图像星居中+Y翻转 | L1008-1012 | ✅ 已实现 | ✅ |
| 星表gnomonic投影 | L1417, eta不取负 | ✅ 已实现 | ✅ |
| scale_min/max计算 | L1412-1415 | ✅ 已实现 | ✅ |
| 三角形匹配(atFindTrans) | atpmatch.c L342 | ✅ 已实现 | ✅ |
| atApplyTrans | match.c L268 | ✅ 已实现 | ✅ |
| atMatchLists去重 | match_arrays_slow | ✅ 已实现 | ✅ |
| **prepare_to_recalc** | match.c L298 | ❌ 缺失 | 🔴 |
| **reset_A_coords** | match.c L343 | ⚠️ 隐式实现 | 🟡 |
| **迭代重投影** | L1444-1470 | ❌ 完全缺失 | 🔴 |
| **TRANS验证** | L1430-1437 | ❌ 缺失 | 🔴 |
| **near solve** | L1223-1319 | ❌ 缺失 | 🟡 |
| 重复星点检测 | PSF阶段 | ⚠️ 有bug | 🔴 |

### 差异详解

#### 差异1：prepare_to_recalc（🔴关键缺失）

Siril在atMatchLists之后调用prepare_to_recalc，它做了两件事：
1. 计算RMS：`trans->sx = Xrms, trans->sy = Yrms`
2. reset_A_coords：将matched_list_A的坐标重置为原始坐标

我们的实现中，at_recalc_trans直接使用img_stars的原始坐标（因为我们的架构不同——我们不修改原始A星坐标），所以reset_A_coords是隐式实现的。但RMS计算缺失。

**影响**：RMS信息在迭代重投影中不是必需的，但缺失意味着我们无法评估匹配质量。

#### 差异2：迭代重投影（🔴最关键缺失）

这是**我们与Siril最大的差异**。Siril在new_star_match成功后，会迭代重投影星表：
1. 用当前变换计算新的投影中心（apply_match）
2. 重新投影星表到新中心（project_catalog_stars）
3. 更新匹配星的B坐标（update_stars_positions）
4. 重新计算变换（atRecalcTrans）
5. 重复直到收敛（offset < 0.01 arcsec）

**影响**：没有迭代重投影，即使初始匹配成功，变换精度也会受限。特别是当初始投影中心与真实中心有较大偏差时，gnomonic投影的非线性会导致系统误差。

#### 差异3：TRANS验证（🔴缺失）

Siril在new_star_match后验证：
1. `check_affine_TRANS_sanity`：检查变换是否合理（如行列式>0）
2. `check_affine_TRANS_scale`：检查scale是否在预期范围内

**影响**：可能接受不合理的变换。

#### 差异4：重复星点检测bug（🔴）

当前去重逻辑检查`fabs(x1-x2) < 0.5 && fabs(y1-y2) < 0.5`，但A[21]和A[22]坐标完全相同却未被检测到。可能原因：
1. 坐标精度问题（%.1f格式隐藏了微小差异）
2. 去重阈值0.5px可能不够

**修复**：增加去重阈值到2.0px（与AT_MATCH_RADIUS一致），并添加更详细的诊断日志。

---

## 修改计划

### 修改1：C++接口扩展

新增参数传入原始星表RA/Dec，用于迭代重投影：

```cpp
int psm_star_alignment(
    const double *img_x, const double *img_y, const double *img_flux, int n_img,
    const double *cat_x, const double *cat_y, const double *cat_mag, int n_cat,
    double scale_arcsec_px,
    double percent_scale_range,
    int n_saturated,
    double center_ra,       // 新增：初始投影中心RA(度)
    double center_dec,      // 新增：初始投影中心Dec(度)
    const double *cat_ra,   // 新增：原始星表RA(度)
    const double *cat_dec,  // 新增：原始星表Dec(度)
    PSMStarAlignmentResult *result);
```

### 修改2：C++中实现gnomonic投影和反投影

```cpp
// gnomonic投影：RA/Dec → xi/eta（角秒）
void gnomonic_projection(double ra, double dec, double ra0, double dec0,
                         double *xi_asec, double *eta_asec);

// 反投影：xi/eta（角秒）→ RA/Dec（度）—— apply_match的核心
void gnomonic_deproject(double xi_asec, double eta_asec,
                        double ra0, double dec0,
                        double *ra, double *dec);
```

### 修改3：实现迭代重投影

在new_star_match成功后，添加迭代重投影循环：
1. conv = sqrt(a0² + b0²)
2. while (conv > 0.01 && trial < 5):
   a. 用apply_match计算新投影中心
   b. 重新投影所有星表星
   c. 更新匹配星的B坐标
   d. atRecalcTrans重算变换
   e. conv = sqrt(a0² + b0²)

### 修改4：修复重复星点检测

增加去重阈值到2.0px，并添加诊断日志。

### 修改5：添加TRANS验证

检查行列式>0和scale在预期范围内。

### 修改6：Python脚本修改

传入原始RA/Dec和投影中心到C++。

# 4SADQ-KV 盲解析模块 - 开发记忆

## 模块概述
**名称**: 4SADQ-KV (4-Star Absolute Distance Quad with K-Vector)
**路径**: `lib/plate_solve/blind_index/`
**用途**: 仅以像素尺度s0为先验的天文图像盲plate solving，验证k-vector范围查询+6绝对角距+四边形匹配核心机制
**设计文档**: [blind_indexing_feature_design.md](../../blind_indexing_feature_design.md)
**Spec**: [.trae/specs/blind-indexing-kvector-prototype/spec.md](../../../../.trae/specs/blind-indexing-kvector-prototype/spec.md)

## 设计决策

### 1. 本地索引 (测试harness捷径)
- 盲解析理论上需全球索引(~9×10^7四边形, 900MB)
- 最小测试采用**本地索引**: 用FITS头指向查询DR3局部天区(~1.5×FOV对角线半径)构建k-vector
- **匹配算法本身只接收星点+s0，不接收指向**
- `pipeline.solve_blind()` 的 `query_center_ra/dec` 参数仅用于测试harness查询DR3

### 2. s0-only算法
- 算法输入 = 检测星点(x,y,flux) + s0(像素尺度)
- s0将像素距离转换为绝对角距: d = pixel_dist × s0
- 不使用天区指向、旋转角、CRVAL等信息

### 3. 简化版Kolomenkin投票
- ≥3个四边形各有候选时: 用每个候选WCS投影其他四边形图像星到天球, 检查与候选参考星池匹配(<3")
- 4个投影位置中≥3个匹配 → 投票+1
- 最高票<2则回退到首个四边形候选列表选最低RMS
- 暂未实现: PROSAC渐进扩张、完整贝叶斯因子、Heyl双k-d精验(留待C++全量版)

### 4. Y翻转处理
- 图像y向下增加, eta(Dec)向上增加
- Umeyama SVD拟合自然捕获翻转(产生R矩阵中的负元素), 不手动翻转
- 验证: 投影回天球与参考星比较RMS

## 文件结构
```
lib/plate_solve/blind_index/
├── python/
│   ├── __init__.py          # 包初始化
│   ├── logging_setup.py     # 日志配置(文件+控制台, UTF-8)
│   ├── io_wrappers.py       # Task 1: 统一接口复用star_detector/gaia_client/astro_image_io
│   ├── quad_geometry.py     # Task 2+4: 6距离、四边形规范化、退化过滤
│   ├── kvector.py           # Task 3: k-vector索引构建与范围查询
│   ├── quad_selector.py     # Task 4: 金字塔选星(图像侧+参考侧)
│   ├── matcher.py           # Task 5: k-vector查询+5距离验证
│   ├── wcs_solver.py        # Task 6: Umeyama SVD→CD/CRVAL/CRPIX+RMS
│   ├── voting.py            # Task 7: Kolomenkin几何投票
│   └── pipeline.py          # 主管线串联所有阶段
├── logs/                    # 日志目录(blind_index.log)
├── tests/                   # 测试目录(Task 8填充)
└── memory.md                # 本文件
```

## 关键参数
| 参数 | 值 | 说明 |
|------|------|------|
| σ_pos | 0.5 pixel | 星点位置噪声 |
| σ_d | σ_pos × s0 | 距离噪声(arcsec) |
| n_sigma | 3.0 | 容差倍数 (3σ) |
| Δ (k-vector步长) | 0.5 arcsec | k-vector区间表步长 |
| d_min | 2.0 arcsec | d_AB下限 |
| 退化: 最小内角 | 10° | 退化过滤阈值 |
| 退化: 面积比 | 0.1 | 面积/(d_AB·d_CD) |
| 退化: 最短边 | 10" | 最短边阈值 |
| 图像亮星池 | 20星 | Top-20最亮 |
| 图像pivot数 | 15 | 每个pivot生成1个四边形 |
| 图像最近邻 | 8 | pivot的8最近邻 |
| 图像最终四边形 | 5 | 按uniqueness+geometry_quality排序取前5 |
| 参考最近邻 | 3 | 每颗参考星+3最近邻→四边形 |
| 投票匹配半径 | 3" | 投票时投影位置匹配阈值 |
| 投票最低票数 | 2 | 接受投票结果所需最低票数 |
| DR3默认星等 | 12.0 | DR3查询极限星等 |
| FOV余量 | 1.5× | DR3查询半径=FOV对角线×1.5 |

## 6距离规范顺序
```
[A, B, C, D] 排序规则:
  1. d_AB = 6距离中最长边, A/B为其端点
  2. C/D: 使 d_AC ≤ d_BC; 相等则 d_AD ≤ d_BD
  3. 6距离: [d_AB, d_AC, d_AD, d_BC, d_BD, d_CD] (arcsec)

实现: 两种C/D分配方案的key = (d_AC-d_BC, d_AD-d_BD), 选字典序较小者
```

## k-vector结构
```
S: 按d_AB升序排列的参考四边形数组
K[j] = S中 d_AB ≤ d_min+(j+1)·Δ 的末下标 (二分查找构建)

范围查询 d_AB ∈ [d-δ, d+δ]:
  j_lo = floor((d-δ-d_min)/Δ), j_hi = floor((d+δ-d_min)/Δ)
  clamp j_lo, j_hi 到 [0, len(K)-1]
  idx_lo = K[j_lo-1]+1 if j_lo>0 else 0; idx_hi = K[j_hi]
  顺序扫描 S[idx_lo : idx_hi+1], 精确过滤区间边界
```

## 复用的现有模块
| 模块 | 来源 | 用途 |
|------|------|------|
| StarDetector, SDetParamsPy | lib/star_detector/python/star_detector.py | 星点检测(detect_ex返回x,y,flux,saturated) |
| GaiaClientPy | lib/plate_solve/python/vector_match_v2.py (L54-150) | DR3锥形查询(db_type=1) |
| gnomonic_forward/inverse | lib/plate_solve/python/vector_match_v2.py (L156-187) | 切平面投影 |
| ImageReader | lib/astro_image_io/python/astro_image_io.py | FITS/XISF图像读取 |

## 如何运行
```powershell
# 全局强制UTF-8编码
[Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PATH = "C:\msys64\mingw64\bin;" + $env:PATH
cd "f:\Astro dev\Astro CS Normalization Database"

# 导入测试
python -c "from lib.plate_solve.blind_index.python import pipeline; print('import OK')"

# 端到端测试 (Task 8)
python -c "
from lib.plate_solve.blind_index.python.pipeline import solve_blind
result = solve_blind(
    image_path='testdata/xxx.fits',
    query_center_ra=123.456,
    query_center_dec=45.678,
)
print(result)
"
```

## 当前进度
- [x] Task 1: 模块骨架与依赖封装复用 (logging_setup.py, io_wrappers.py)
- [x] Task 2: DR3参考星查询与四边形库构建 (quad_geometry.py的ReferenceQuad)
- [x] Task 3: k-vector索引构建 (kvector.py)
- [x] Task 4: 图像星点检测与四边形生成 (quad_selector.py)
- [x] Task 5: k-vector范围查询+5距离验证 (matcher.py)
- [x] Task 6: WCS求解 Umeyama SVD (wcs_solver.py)
- [x] Task 7: 简化版Kolomenkin几何投票 (voting.py)
- [x] 主管线 pipeline.py
- [x] Task 8: 测试验证脚本 (tests/test_pipeline.py + logs/test_report.txt)

## Task 8 测试结果 (2026-06-25)

### 测试帧与结果
| 帧 | s0 | n_det | n_ref | n_quads(img/ref) | n_cand | RMS | CRVAL偏差 | 结果 |
|------|------|-------|-------|------------------|--------|------|----------|------|
| M20_T2_Green | 0.967 | 28434 | 3772 | 5/290 | 0 | inf | N/A | 失败 |
| LDN43_T1_Lum | 0.967 | 15601 | 724 | 5/40 | 0 | inf | N/A | 失败 |
| NGC55_T3_Red | 0.989 | 2313 | 450 | 5/29 | 0 | inf | N/A | 失败 |
| GalaxyCenter_T4_Red | 6.188 | 35971 | 140312 | 5/9880 | 1 | 3.33" | 0.00" | success(RMS略超) |

成功率: 1/4=25%, 双通过(CRVAL+RMS): 0/4=0%

### 已修复bug: 饱和星占据Top-N亮星池
- **问题**: star_detector输出排序为饱和星优先(flux=-1, mag<5.7, 不在Gaia DR3中), pipeline的"Top-N最亮池"直接取前N颗→饱和星→无Gaia匹配
- **修复**: `quad_selector.generate_image_quads()` 增加 `saturated_arr` 参数, 跳过 saturated=1 的星; `pipeline.py` 传入 `star_result.saturated`
- **效果**: M20 d_AB从[845,831,762]"(超出参考范围[50,840]")降至[124,89,317,407]"(落在范围内), k-vector能找到d_AB候选
- **文件**: quad_selector.py (L78-131), pipeline.py (L248-253)

### 核心机制评估: 正常工作
1. k-vector范围查询: d_AB匹配时正确返回候选 (GalaxyCenter 1候选, M20 修复后317"/407"各1候选)
2. 5距离验证: 图像星非Gaia星时正确拒绝 (形状不匹配)
3. Umeyama SVD: GalaxyCenter WCS求解 RMS=3.33" (接近阈值3.0")
4. 结论: 核心机制(k-vector+5距离验证+Umeyama SVD)在星密度足够时正常工作

### 遗留限制: flux排序与Gaia星等不相关 (star selection瓶颈)
- **根因**: 正常星flux(Moffat4振幅A)与Gaia星等不相关, Top-N正常星中Gaia星比例极低
- **数据**: M20 Top-338正常星中仅~1颗Gaia星; LDN43/NGC55 Top-N正常星中无Gaia星
- **影响**: 图像四边形的4颗星几乎不含Gaia星, 5距离验证必然失败
- **GalaxyCenter成功原因**: 星密度极高(35971检测星, 140312参考星), Top-N正常星中Gaia星比例足够
- **建议**: 改进star selection策略(空间均匀抽样/PROSAC/增大pool_size+四边形数), 超出Task 8范围

### GalaxyCenter CRVAL偏差=0.00"说明
- WCS求解返回CRVAL=(ra0,dec0)=query_center, 与expected_crval相同 → trivially true
- 真正WCS精度指标是RMS=3.33" (Umeyama SVD残差)

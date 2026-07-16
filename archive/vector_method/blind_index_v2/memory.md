# ADV-PA 盲解析模块记忆 (Absolute Distance Voting with Position Angle)

## 模块概述
- **位置**: `lib/plate_solve/blind_index_v2/`
- **算法**: 二星绝对角距对 + k-vector 索引 + (天区,旋转角)二维投票 + 向量法验证
- **先验**: 仅像素尺度 s₀ (arcsec/pixel)，旋转/平移不变
- **理论**: Christian & Crassidis [2021] SO(3) 不变量框架 — 标定相机下星间角距是 SO(3) 不变量，2星→1个不变量
- **状态**: Phase 1 (区域内索引验证) Task 1-8 已执行, Task 8 验证失败 — 算法bug待修复 (新 spec "fix-adv-pa-phase1-bugs")
- **Spec**: [.trae/specs/blind-indexing-distance-voting-prototype/spec.md](file:///F:/Astro%20dev/Astro%20CS%20Normalization%20Database/.trae/specs/blind-indexing-distance-voting-prototype/spec.md)

## 设计决策

### 1. 特征选择：二星绝对角距对
- **特征**: d_cat (球面角距 arcsec) + PA_cat (位置角 度, 从北向东)
- **理由**:
  - 4颗星=6对→6票聚集 vs 4SADQ的4星仅1四边形
  - 投票与匹配对数正相关 (用户需求)
  - d=pixel_dist×s₀ 是 SO(3) 不变量，无需多尺度遍历
- **对比 Astrometry.net**: 归一化哈希需多尺度，本方案 s₀ 已知→单尺度

### 2. k-vector 索引 (单维)
- **结构**: 星对按 d_cat 升序排列，区间表 K[j] = S中 d_cat ≤ d_min+(j+1)·Δ 的末下标
- **参数**: Δ=0.5", d_min=10", d_max=18000"(5°), K=8 最近邻
- **查询**: O(1) 定位 + O(k) 扫描，无 binning 量化损失
- **实现**: `np.searchsorted(side='right')-1` 批量构建 K[j]

### 3. (天区, 旋转角) 二维投票
- **天区**: HEALPix Nside=64 (49152格, ~0.84°) — healpy 不可用时回退等距网格 (0.84°×429×215)
- **旋转角**: 180 bins (2°/bin, 覆盖0-360°)
- **投票公式**: rot = (θ_img - PA_cat) % 360, rot_bin = floor(rot/2°)
- **峰值阈值**: max(3, N_pairs/100)
- **区分度**: 4星6对→真天区得票6, 假票分散到 8.86M 格 (期望 1.5×10⁻⁴/格), SNR≈40000

### 4. Y 翻转处理 (关键)
- **推导**: y-down 图像中 theta_img = PA_cat + R - 90° (R=图像旋转角)
- **投票峰值**: rot = theta_img - PA_cat = R - 90°
- **图像旋转角**: R = rot_angle + 90° (在 wcs_verify.py 中应用)
- **Umeyama 反射**: 不强制 det(R)=+1，允许 det(R)=-1 捕获 Y 翻转

### 5. 向量法验证 (复用 V3.5 引擎)
- **Gaia 查询**: 锥角 1.5×FOV, 极限星等 14
- **gnomonic 投影**: 复用 `vector_match_v2.gnomonic_forward/inverse`
- **旋转星表到图像帧**: M = [[cos_R, sin_R], [sin_R, -cos_R]] (含 Y 翻转, M²=I)
- **匹配**: cKDTree 最近邻, 初始阈值 5*s0, 精修阈值 2*σ_pos*s0
- **迭代精修**: 3 次 Umeyama SVD 重匹配
- **RMS**: haversine 角距离, 投影图像像素→切平面→RA/Dec→与参考星比较

### 6. 测试 Harness 捷径 (Phase 1)
- 用 FITS 头 RA/Dec 查询 DR3 局部天区 (3×FOV) 构建本地 k-vector
- **匹配算法本身只接收星点 + s₀**，不接收指向
- Phase 2 完成全天球索引后可无指向求解

## 文件结构
```
lib/plate_solve/blind_index_v2/
├── memory.md              # 本文件
├── python/
│   ├── __init__.py        # 包初始化, __all__
│   ├── logging_setup.py   # 日志 (UTF-8 文件+控制台)
│   ├── io_wrappers.py     # 复用 star_detector/gaia_client/astro_image_io
│   ├── spherical_geom.py  # 球面角距 + 位置角 (向量化)
│   ├── pair_index.py      # 星对库 + k-vector 索引 (cKDTree+searchsorted)
│   ├── image_features.py  # 图像星对 (d_img, θ_img) 提取
│   ├── voting.py          # k-vector查询+(天区,rot)投票+峰值检测
│   ├── wcs_verify.py      # Umeyama SVD (允许反射) + RMS
│   └── pipeline.py        # 主管线 solve_blind()
├── logs/
│   └── .gitkeep
└── tests/
    └── .gitkeep
```

## 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| K (最近邻) | 8 | 每颗参考星取 K 个最近邻组成星对 |
| Δ (k-vector步长) | 0.5" | 区间表精度 |
| d_min | 10" | 星对最小角距 (过滤重复源) |
| d_max | 18000" (5°) | 星对最大角距 |
| σ_pos | 0.5 px | 星点位置噪声 |
| n_sigma | 3.0 | k-vector 查询容差倍数 (±3σ_d) |
| rot_bin | 2° | 旋转角 bin 宽度 |
| Nside | 64 | HEALPix Nside (49152 格) |
| top_k | 5 | 峰值检测 top-K |
| mag_limit (索引) | 12.0 | DR3 索引构建极限星等 |
| mag_limit (验证) | 14.0 | 验证阶段 Gaia 查询极限星等 |
| max_image_stars | 100 | 参与配对的最大图像星数 (避免 C(N,2) 爆炸) |
| FOV margin | 1.5 | 索引构建查询半径 = FOV对角线 × 1.5 |

## 运行方式

### 冒烟测试 (验证导入)
```powershell
cd "f:\Astro dev\Astro CS Normalization Database"
python -c "from lib.plate_solve.blind_index_v2.python import pipeline; print('import OK')"
```

### 端到端调用
```python
from lib.plate_solve.blind_index_v2.python.pipeline import solve_blind

result = solve_blind(
    image_path="testdata/Light/M20_T2/red/M20_001.fits",
    s0_arcsec_per_pixel=None,        # None则从FITS头读
    query_center_ra=None,            # None则从FITS WCS读
    query_center_dec=None,
    mag_limit=12.0,
)
if result.success:
    print(f"RMS={result.best_rms_arcsec:.3f}\", n_inliers={result.wcs.n_inliers}")
```

## 已解决的重大问题
1. **healpy 不可用**: Windows 缺 C 编译器无法 `pip install healpy`。spec 允许回退，实现等距网格 (0.84°×429×215=92235格，近似 Nside=64)
2. **PowerShell UTF-8 设置**: `[System.Text.Encoding]::Default` 是只读属性，仅设置 InputEncoding/OutputEncoding
3. **mingw python 缺 numpy**: 改用用户 Python 3.12 (C:\Users\fujia\AppData\Local\Programs\Python\Python312\python.exe) 验证导入
4. **Y 翻转推导**: 通过 theta_img = PA_cat + R - 90° 推导出 R = rot_angle + 90°，旋转矩阵 M=[[cos,sin],[sin,-cos]] 含 Y 翻转
5. **Umeyama 反射**: 标准 Umeyama 强制 det(R)=+1 会阻止 Y 翻转捕获，改为不强制，允许 det(R)=-1
6. **k-vector 区间表**: 用 `np.searchsorted(side='right')-1` 批量计算 K[j]，避免 Python 循环
7. **图像星数爆炸**: C(N,2) 在 N=41041 时达 8.4×10⁸ 对，限制 Top-100 最亮星 (饱和星优先 + flux 降序补足)

## Phase 1 测试结果 (Task 8, 2026-06-26)

### 测试帧
| 帧名 | 望远镜 | s0 (") | 检测星 | 参考星 | 星对数 |
|------|--------|--------|--------|--------|--------|
| M20_T2 | T2 (1917.6mm) | 0.9669 | 28434 | 3772 | 14959 |
| LDN43 | T1 (1917.6mm) | 0.9668 | 15601 | 724 | 2907 |
| NGC247_T2 | T2 (1917.4mm) | 0.9670 | 2325 | 430 | 1671 |
| NGC55_T3 | T3 (1877mm) | 0.9890 | 2313 | 450 | 1820 |

### 验证结果
| 指标 | 期望 | 实际 | 状态 |
|------|------|------|------|
| 成功率 | ≥80% | 75% (3/4) | ❌ 失败 |
| 平均RMS | <3" | 1.259" | ✅ 通过 |
| 平均耗时 | <5s | 1.984s | ✅ 通过 |
| SNR | >10 | >18000 | ✅ 通过 |
| CRVAL偏差 | <30" | 8565" (假阳性) | ❌ 失败 |
| k-vector准确率 | 无遗漏 | 5.7% (1/6) | ❌ 失败 |
| 鲁棒性 | 通过 | 3/3 通过 (但基于假阳性) | ⚠️ 存疑 |

### 每帧详情
| 帧名 | success | RMS" | CRVALdev" | SNR | time(s) |
|------|---------|------|-----------|-----|---------|
| M20_T2 | 是 | 0.549 | 9187 (假阳性) | 18375 | 3.238 |
| LDN43 | 是 | 2.975 | 7945 (假阳性) | 38745 | 1.919 |
| NGC247_T2 | 否 | inf | N/A | 68013 | 1.378 |
| NGC55_T3 | 是 | 0.252 | N/A (无baseline) | 37421 | 1.403 |

### 已修复的Bug (Task 8期间)
1. **峰值检测阈值过激** (genuine bug, 已修复): 原阈值 max(3, N_pairs/100)=49.5 导致 vote_peak=44 被拒绝. 改为噪声基底阈值 max(3, 10×total_votes/n_cells)
2. **CRVAL对比方法错误** (设计问题, 已修复): WCSResult.crval 是候选切点, 不同WCS用不同切点. 改为比较图像中心天空坐标 (pixel(w/2,h/2) → RA/Dec → haversine)

### 未修复的根因 (需新spec)
1. **K=8邻居限制过严**: k-vector仅找到5.7%真匹配 (42/741). 真匹配对要求两星都是对方的K=8最近邻, 但亮星密集区此条件过严
2. **真匹配rot值分散**: std=103.8° (应聚集在1-2个值). 怀疑PA_cat/θ_img方向性定义不一致或Y-flip处理有问题
3. **假阳性峰值**: 错误天区(距真天区2.5°)的catalog星偶然形成一致rot, 产生44票峰值超过真天区

## 待解决问题 (Task 8) — 已执行, 部分未通过
- [x] Phase 1 多帧测试 (4帧: M20_T2/LDN43/NGC247_T2/NGC55_T3)
- [x] k-vector 查询准确性验证 — 失败 (5.7%真匹配率)
- [x] 投票聚集信噪比验证 — 通过 (SNR > 18000)
- [x] V4 Umeyama SVD 收敛验证 — 通过 (RMS < 3")
- [x] 鲁棒性测试 — 通过 (但基于假阳性)
- [x] 与 header WCS 对比 — 失败 (CRVAL偏差 8565")
- [x] 成功率 75% (未达 80%), 单帧耗时 1.984s (通过)
- [ ] 算法bug修复 (新 spec "fix-adv-pa-phase1-bugs")

## Phase 2 (后续独立 spec)
- 全天球 G<12 亮星对 k-vector 索引 (~256 MB, ~8×10⁶ 星对)
- 真正盲解析 (无指向先验)
- 前置条件: Phase 1 可靠性验证通过

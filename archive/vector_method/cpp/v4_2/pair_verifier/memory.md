# V4.2 PairVerifier 模块开发记忆

## 模块定位
V4.2 Phase D（MAD 清洗）+ Phase D'（贝叶斯 + 三角形验证）独立模块，从 V4.1 `vector_match_v4_1/src/vm4_core.cpp` 的 Phase D/D' 迁移。C++17 单线程，依赖 Eigen3（SVD），无 OpenMP。

## 文件结构
```
lib/plate_solve/cpp/v4_2/pair_verifier/
├── include/
│   └── pv_api.h              # 公共 C 接口（PairVerifierParams + VerificationResult + pv_verify/pv_free）
├── src/
│   ├── pv_internal.h         # 内部命名空间 pv（vec_median/apply_similarity/umeyama 内联 + MadResult + 三个子函数签名）
│   ├── pv_mad.cpp            # Phase D: 3 轮 MAD 迭代清洗 + 鲁棒预过滤
│   ├── pv_bayes.cpp          # Phase D': 贝叶斯假设验证（lnK 决策）
│   ├── pv_triangle.cpp       # Phase D': 三角形双特征验证（面积 A + 极惯性矩 J）
│   └── pv_core.cpp           # 主入口 pv_verify：串联 MAD → 贝叶斯 → 三角形
├── test/
│   └── test_pv.cpp           # 5 个测试场景 17 项断言
├── Makefile                  # g++ C++17 -O2 -march=native，-static 完全静态链接
└── pair_verifier.dll         # 编译产物（仅依赖 KERNEL32.dll + msvcrt.dll）
```

Python 绑定: `lib/plate_solve/python/v4_2/pair_verifier.py`（PairVerifier 类，ctypes 封装，verify() 方法）

## 核心算法

### Phase D: MAD 清洗（pv_mad.cpp）
1. 初始 Umeyama 拟合（src=W[pairs_w], dst=U[pairs_u]）→ (s, θ, tx, ty)
2. 变换 W → Wt
3. **鲁棒预过滤**（关键修复）：当 init_med > min_thresh 时，用 thresh_factor × init_med 作为粗阈值剔除明显离群，重新 Umeyama 收敛后再进入标准 MAD 迭代
4. MAD 迭代（最多 mad_iters 轮）：
   - 收集保留点残差 r_i = |U[ui] - Wt[wi]|
   - MAD = 1.4826 × median(|r_i - median(r)|)
   - 阈值 = max(mad_min_threshold_arcsec, mad_threshold_factor × 1.4826 × MAD)
   - 剔除 r_i > 阈值 的对，无剔除则提前终止
   - 用剩余点重新 Umeyama 拟合 → 更新变换 → 重新变换 W
5. 计算清洗后 RMS

### Phase D' 贝叶斯验证（pv_bayes.cpp）
- 公式：lnK = Σ[-log(2πσ²) - r²/(2σ²)] + n×log(A_fov_sqsec)
- σ = max(sigma_min, 1.4826×MAD(r))
- A_fov_sqsec = (fov_diag_deg × 3600)²
- 决策：lnK > lnK_accept(20.7) → 接受(1)；lnK > lnK_weak(6.9) → 弱证据(0)；否则 → 拒绝(-1)

### Phase D' 三角形验证（pv_triangle.cpp）
- 双特征：面积 A（海伦公式）+ 极惯性矩 J = A×(a²+b²+c²)/36
- 阈值：eps_A=0.05, eps_J=0.10
- 随机采样三角形（seed 固定可复现），统计通过率
- 通过率 ≥ triangle_pass_rate(0.8) → validated=1

### pv_core.cpp 主流程
1. pv_mad_clean → 获得 clean_u/clean_w + transform + RMS
2. 构造 matched_pairs（用于贝叶斯/三角形）
3. pv_bayes_verify → lnK + decision
4. pv_triangle_verify → pass_ratio + validated
5. 填充 VerificationResult，由 pv_verify 返回

## 关键设计决策

### 1. 鲁棒预过滤修复（核心 bug fix）
**问题**：Test 1 中 5 个 100" 偏移的离群拉偏初始 Umeyama，导致正确对残差 ~14"，MAD 阈值 ~10" 误删正确对（移除 49 对而非 5 对）。
**修复**：在标准 MAD 迭代前插入预过滤——当 `init_med > min_thresh` 时，用 `thresh_factor × init_med` 作为粗阈值剔除明显离群，重新 Umeyama 收敛。
**约束**：不改变 MAD 阈值公式 `max(5", 3×1.4826×MAD)`，仅在初始变换被拉偏时启用。

### 2. DLL 完全静态链接
`-static` 标志替代 `-static-libgcc -static-libstdc++`，消除 libwinpthread-1.dll 依赖，仅剩系统库（KERNEL32.dll, msvcrt.dll），Python ctypes 加载无需额外 PATH 配置。

### 3. 尺度约束 |s-1|<0.1
Umeyama 拟合后检查尺度因子 s，超出 [0.9, 1.1] 范围则标记 sim.valid=false，退化为恒等变换（避免错误变换传播）。

### 4. 三角形采样可复现
使用固定 seed（std::mt19937），确保相同输入产生相同结果，便于调试和回归测试。

## 参数结构体（PairVerifierParams，11 字段）
| 字段 | 默认值 | 说明 |
|------|--------|------|
| mad_iters | 3 | MAD 迭代轮数 |
| mad_threshold_factor | 3.0 | MAD 阈值因子 |
| mad_min_threshold_arcsec | 5.0 | MAD 最小阈值（角秒） |
| lnK_accept | 20.7 | 贝叶斯接受阈值 |
| lnK_weak | 6.9 | 贝叶斯弱证据阈值 |
| sigma_min | 1.0 | σ 下限（角秒） |
| eps_A | 0.05 | 三角形面积容差 |
| eps_J | 0.10 | 三角形极惯性矩容差 |
| triangle_pass_rate | 0.8 | 三角形通过率阈值 |
| fov_diag_deg | 6.0 | 视场对角线（度） |
| log_file_path | NULL | 日志路径（NULL 禁用日志） |

## 验证结果结构体（VerificationResult，14 字段）
clean_u/clean_w 指针 + n_clean/n_removed/mad_iterations/mad_rms_arcsec/bayes_lnK/bayes_n_match/bayes_decision/triangle_total/triangle_passed/triangle_pass_ratio/validated/success

## 编译验证
- `make all`：DLL 编译成功（pair_verifier.dll），零警告零错误
- `make test_pv`：17/17 PASS，退出码 0
- DLL 依赖：仅 KERNEL32.dll + msvcrt.dll（系统库），无外部依赖

## 测试结果（5 个场景 17 项断言）
| 测试场景 | 结果 | 关键指标 |
|---------|------|---------|
| Test 1: MAD 清洗 (50对+5离群) | PASS | n_clean=45, n_removed=5, RMS=0.40" |
| Test 2: 贝叶斯接受 (30对正确) | PASS | lnK=750.53, decision=1（接受） |
| Test 3: 贝叶斯拒绝 (30对随机) | PASS | lnK=-80.50, decision=-1（拒绝） |
| Test 4: 三角形通过 (50对正确) | PASS | ratio=0.9663, validated=1 |
| Test 5: 三角形失败 (50对随机) | PASS | ratio=0.0044, validated=0 |

## Python 端验证
- `from lib.plate_solve.python.v4_2 import PairVerifier` 导入成功
- 合成数据 verify() 冒烟测试通过：n_clean=45, validated=True, lnK=755.21

## 依赖
- Eigen3: `lib/plate_solve/cpp/vector_match_v2/third_party/Eigen`（SVD 求解）
- 共享头文件: `cpp/v4_2/common/v42_types.h`（SimTransform/MatchPair/BayesResult/TriangleResult）+ `cpp/v4_2/common/v42_log.h`（Logger）

## 编译命令
```powershell
# 全局强制UTF-8编码
[Console]::InputEncoding = [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$mingwBin = "C:\msys64\mingw64\bin"
$env:Path = "$mingwBin;$env:Path"
cd "f:\Astro dev\Astro CS Normalization Database\lib\plate_solve\cpp\v4_2\pair_verifier"
make all
make test_pv
```

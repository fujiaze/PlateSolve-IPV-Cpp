# 归档索引

> 本目录归档了 plate solving 模块的历史代码、文档和日志。
> 所有内容完整保留，不做删除。归档日期: 2026-07-09。

## 目录结构

```
archive/
├── HISTORY.md                          # 历史版本迭代文档（V2-V4.28 完整演进）
├── ARCHIVE_INDEX.md                    # 本文件
├── plate_solve_old.rar                 # 原始 plate_solve_old 压缩包
├── vector_method/                      # 向量法系列 V2-V4.5
│   ├── README.md                       # Vector Match V2 详细算法文档
│   ├── CHANGELOG_V3.md                 # V3 版本历史
│   ├── CHANGELOG_V4.md                 # V4 版本历史
│   ├── CHANGELOG_V5_V6.md              # V5/V6 版本历史
│   ├── INDEX.md                        # 原始索引
│   ├── python/                         # Python 实现（astap, plate_solve, feature_match 等）
│   ├── src/                            # C++ 实现（psolve_api, psolve_coarse, psolve_ransac 等）
│   ├── include/                        # C++ 头文件
│   ├── modules/                        # 历史模块（initial_wcs, iterative_refine, star_alignment）
│   ├── history/                        # 更早期模块（coarse_affine, feature_match, iterative, rms_calc）
│   ├── old/                            # 早期版本（initial_wcs_v1, specs_v1, star_alignment）
│   ├── config/                         # 配置文件
│   ├── cpp/                            # V3.2-V4.5 C++ 加速版本
│   ├── docs/                           # 设计文档
│   ├── scripts/                        # 测试脚本（v3_3-v4_5 各版本）
│   ├── design/                         # 设计资料
│   ├── logs/                           # 向量法时期日志
│   └── build_files/                    # 编译产物（Makefile, *.o, *.dll, build 脚本）
├── blind_solving/                      # 盲解析实验 V5-V6
│   ├── astrometry/                     # Astrometry.net Python 客户端
│   ├── astrometry_net/                 # Astrometry.net 完整 C 代码库
│   ├── astrometry.net-main/            # Astrometry.net 主线代码
│   ├── blind_index/                    # V5.0 4SADQ-KV 四元组盲解析
│   ├── blind_index_v2/                 # V5 迭代版本
│   └── blind_index_v3/                 # V6.0 DD-SPPS 频域盲解析
├── historical_logs/                    # 历史测试日志
│   ├── ipv_diag/                       # 早期诊断
│   ├── ipv_full_test/                  # 790 帧全量测试（V4.7-V4.9）
│   ├── ipv_real_test/                  # 真实数据测试
│   ├── v422_diag/                      # V4.22 诊断
│   ├── v423_dbg/                       # V4.23 调试
│   ├── v423_focus/                     # V4.23 聚焦分析
│   ├── v423_m20_green/                 # V4.23 M20 Green 分析
│   ├── v424_fail/                      # V4.24 失败帧分析
│   ├── v424_focus/                     # V4.24 聚焦分析
│   ├── v424_m20_green/                 # V4.24 M20 Green
│   ├── v425_diag/                      # V4.25 诊断
│   ├── v426_diag/                      # V4.26 诊断
│   ├── v433_mismatch/                  # V4.33 不匹配分析
│   ├── v433_galaxy_test/               # V4.33 Galaxy 测试
│   ├── bclass_diag/                    # B 类帧诊断
│   ├── test_force/                     # 强制测试
│   ├── test_order3/                    # order=3 测试
│   ├── rms_score/                      # RMS 评分
│   ├── siril_compare/                  # Siril 对比结果（V4.25-V4.27）
│   │   ├── Galaxy_Center_Oiii/         # Siril 诊断数据
│   │   ├── NGC4945_Lum/                # Siril 诊断数据
│   │   ├── NGC6302_narrow/             # Siril 诊断数据
│   │   ├── all_frames/                 # 全帧 stars.csv/lst
│   │   ├── compare/                    # 对比结果
│   │   ├── ipv_baseline_790/           # IPv 基线
│   │   ├── siril_baseline_790/         # Siril 基线
│   │   ├── v425_20frame/               # V4.25 20 帧测试
│   │   ├── v425_5frame_fix/            # V4.25 5 帧修复
│   │   ├── v425_790/                   # V4.25 790 帧
│   │   ├── v426_790/                   # V4.26 790 帧
│   │   ├── v427_5frame/                # V4.27 5 帧
│   │   └── v427_790/                   # V4.27 790 帧
│   └── *.md, *.log, *.txt, *.json      # 散落分析文件
└── historical_scripts/                 # 历史测试/诊断脚本
    ├── crash_logs/                     # crash_*.log（18 个）
    ├── diag_scripts/                   # 诊断脚本（diag_*.py, diag_*.log）
    ├── test_scripts/                   # 测试脚本（test_*.py, test_*.log, rms_score.py）
    └── siril_compare_diag/             # siril_compare 下的诊断脚本
```

## 归档规则

1. **不删除任何内容** - 所有历史代码、文档、日志完整保留
2. **按算法路线分类** - 向量法 / 盲解析 / 历史日志 / 历史脚本
3. **保持原始结构** - 各子目录内部结构不变
4. **命名归一化** - 统一归档目录命名，去除嵌套归档

## 当前版本（不在归档中）

当前生产版本 V4.28 位于 `lib/plate_solve/`：
- C++ 实现: `cpp/ipv/`
- Python 绑定: `python/ipv_solver.py`
- 基线脚本: `python/siril_compare/run_ipv_baseline.py`, `run_siril_baseline.py`, `siril_runner.py`
- 当前日志: `logs/siril_compare/v428_*/`

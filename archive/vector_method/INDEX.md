# 📂 plate_solve — 天文图像星图匹配/WCS求解库

> 基于向量匹配的plate solving引擎，从V2到V3.5持续演进

## 目录结构

```
lib/plate_solve/
├── INDEX.md              ← 本索引
├── python/               ← 核心Python模块（各版本求解+Gaia接口）
├── cpp/                  ← C++加速实现（V3.2起，当前V3.5）
├── docs/                 ← 设计文档与规格
└── scripts/              ← 脚本[→](scripts/INDEX.md)
    ├── v3_5/             ← 当前版本 [→](scripts/v3_5/INDEX.md)
    ├── v3_4/             ← [→](scripts/v3_4/INDEX.md)
    ├── v3_3/             ← [→](scripts/v3_3/INDEX.md)
    └── dev/              ← 早期开发脚本 [→](scripts/dev/INDEX.md)
```

## 版本演进

| 版本 | 核心改进 | 设计文档 | 脚本 |
|------|----------|----------|------|
| **V3.5** | BIC自适应SIP阶数、全仿射CD/CRVAL、MAD稳健拟合、Umeyama符号验正 | [v3_5_design.md](docs/v3_5_design.md) | [→](scripts/v3_5/INDEX.md) |
| V3.4 | WCS-SIP标准输出、Siril风格CD修正 | [v3_4_design.md](docs/v3_4_design.md) | [→](scripts/v3_4/INDEX.md) |
| V3.3 | 单点抽样向量匹配、极限星等公式 | [v3_3_design.md](docs/v3_3_design.md) | [→](scripts/v3_3/INDEX.md) |
| V3.2 | C++加速版本（初版） | [v3_2_design.md](docs/v3_2_design.md) | — |
| V3.1 | 迭代SVD精修 | [v3_1_design.md](docs/v3_1_design.md) | — |
| V3.0- | 早期三阶段匹配、粗筛/精筛/验证 | [vector_match_analysis.md](docs/vector_match_analysis.md) | [→](scripts/dev/INDEX.md) |

## 核心模块 (python/)

| 模块 | 功能 |
|------|------|
| `vector_match_v3_5_cpp.py` | **当前版本**：V3.5 C++求解器的Python绑定 |
| `vector_match_v3_4_cpp.py` | V3.4 C++求解器Python绑定（用于对比） |
| `vector_match_v2.py` | Gaia星表查询、gnomonic投影、flip等基础工具 |
| `vector_match.py` | 最早期Python纯实现 |

## C++实现 (cpp/)

| 目录 | 功能 |
|------|------|
| `vector_match_v3_5/` | **当前版本**：`vm35_core.cpp`（全流程+Phase E分层拟合） |
| `vector_match_v3_4/` | V3.4实现（用于回归对比） |
| `vector_match_v2/` | V2实现 + third_party（Eigen, nanoflann） |

## 设计文档 (docs/)

| 文档 | 内容 |
|------|------|
| `v3_5_design.md` | V3.5完整设计：分层拟合、MAD稳健、符号验正、相位精度 |
| `v3_4_design.md` | V3.4设计：SIP输出标准、Siril CD修正 |
| `v3_3_design.md` | V3.3设计：单点抽样、SNR累积分离、极限星等公式 |
| `vector_match_analysis.md` | 早期分析和实验总结 |
| `specs/` | 子模块规格：affine-diagnostics, star-alignment-module 等8个模块 |

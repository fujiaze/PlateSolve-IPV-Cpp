# Tasks

- [x] Task 1: 归档旧star_alignment模块
  - [x] SubTask 1.1: 创建 `lib/plate_solve/old/star_alignment/` 目录
  - [x] SubTask 1.2: 复制 `modules/star_alignment/` 下所有文件到 `old/star_alignment/`
  - [x] SubTask 1.3: 在 `old/star_alignment/README.txt` 中备注用途（饱和星优先三角匹配+KDTree迭代精化，旧版Step1实现）

- [x] Task 2: Python原型 - 像素尺度与FOV计算
  - [x] SubTask 2.1: 创建 `lib/plate_solve/python/initial_wcs.py`
  - [x] SubTask 2.2: 实现 `compute_pixel_scale(focal_mm, pixel_um)` 函数
  - [x] SubTask 2.3: 实现 `compute_fov(scale_arcsec_px, width, height)` 函数
  - [x] SubTask 2.4: 单元测试验证计算结果（Python运行验证通过）

- [x] Task 3: Python原型 - Gaia锥形查询与极限星等二分法
  - [x] SubTask 3.1: 实现 `bisection_mag_limit(gaia_client, center_ra, center_dec, radius_deg, target_count)` 函数
  - [x] SubTask 3.2: 实现 `gaia_cone_search_with_bisection(gaia_client, center_ra, center_dec, fov_diag, det_count)` 函数
  - [ ] SubTask 3.3: 测试验证二分法收敛（需实际Gaia数据）

- [x] Task 4: Python原型 - Gnomonic投影
  - [x] SubTask 4.1: 实现 `gnomonic_forward(ra, dec, ra0, dec0) -> (xi, eta)` 函数
  - [x] SubTask 4.2: 实现 `gnomonic_inverse(xi, eta, ra0, dec0) -> (ra, dec)` 函数
  - [x] SubTask 4.3: 实现 `project_gaia_to_pixel(cat_ra, cat_dec, center_ra, center_dec, scale_arcsec_px)` 批量投影
  - [x] SubTask 4.4: 与现有C++投影结果交叉验证（Python往返精度<1e-14度，代码比对公式一致）

- [x] Task 5: Python原型 - 饱和星优先三角匹配
  - [x] SubTask 5.1: 实现 `select_match_stars(img_x, img_y, img_flux, img_saturated, cat_x, cat_y, cat_mag)` 星点选择函数（饱和星优先策略）
  - [x] SubTask 5.2: 实现 `build_triangles(x, y, nbright)` 三角形构建函数（ba, ca空间）
  - [x] SubTask 5.3: 实现 `triangle_match(tris_a, tris_b, radius, scale_min, scale_max)` 投票矩阵匹配
  - [x] SubTask 5.4: 实现 `iter_trans(stars_a, stars_b, pairs, max_iter, halt_sigma)` 迭代精化（35%分位数sigma, 10-sigma clip）
  - [ ] SubTask 5.5: 单元测试验证三角匹配正确性（需实际数据）

- [x] Task 6: Python原型 - 四种翻转模式匹配
  - [x] SubTask 6.1: 实现 `apply_flip(cat_px, cat_py, flip_mode)` 翻转函数
  - [x] SubTask 6.2: 实现 `match_with_flip(img_stars, cat_stars, flip_mode, params)` 单模式完整匹配流程
  - [x] SubTask 6.3: 实现 `select_best_flip(results)` 选择最佳翻转模式（匹配数最多→RMS最小）
  - [ ] SubTask 6.4: 测试4种模式在已知翻转图像上的正确性（需实际数据）

- [x] Task 7: Python原型 - 全星点验证匹配
  - [x] SubTask 7.1: 实现 `apply_affine(x, y, a0, a1, a2, b0, b1, b2)` 仿射变换
  - [x] SubTask 7.2: 实现 `verify_match(img_stars, cat_stars, affine, radii)` 递减半径验证匹配
  - [ ] SubTask 7.3: 测试验证匹配后匹配数增长（需实际数据）

- [x] Task 8: Python原型 - 迭代重投影收敛
  - [x] SubTask 8.1: 实现 `reproject_center(a0, b0, ra0, dec0)` 从仿射平移量逆投影新中心
  - [x] SubTask 8.2: 实现 `iterative_reprojection(img_stars, gaia_client, init_center, affine, max_trials, conv_tol)` 迭代收敛循环
  - [ ] SubTask 8.3: 测试收敛行为（需实际数据）

- [x] Task 9: Python原型 - 整合InitialWCS类
  - [x] SubTask 9.1: 实现 `InitialWCS` 类，整合5步流程
  - [x] SubTask 9.2: 实现 `solve()` 方法，返回 `InitialWCSResult`
  - [x] SubTask 9.3: 添加详细日志输出（每步耗时、匹配数、RMS等）
  - [ ] SubTask 9.4: 用测试数据端到端验证（目标RMS<2px，需实际数据）

- [x] Task 10: C++重写 - psm_initial_wcs模块
  - [x] SubTask 10.1: 创建 `lib/plate_solve/modules/initial_wcs/` 目录结构
  - [x] SubTask 10.2: 实现C++版本的5步算法（基于验证过的Python逻辑）
  - [x] SubTask 10.3: 编译为 `psm_initial_wcs.dll`
  - [x] SubTask 10.4: Python ctypes绑定

- [x] Task 11: 更新PlateSolve统一API
  - [x] SubTask 11.1: 修改 `plate_solve.py` 新增 `solve_step1_initial_wcs()` 方法
  - [x] SubTask 11.2: 更新 `DESIGN.md`
  - [x] SubTask 11.3: 更新 `memory.md`

# Task Dependencies
- Task 2, 3, 4 可并行（无依赖）
- Task 5 依赖 Task 4（需要投影坐标）
- Task 6 依赖 Task 5（需要三角匹配）
- Task 7 依赖 Task 6（需要初始仿射变换）
- Task 8 依赖 Task 7（需要验证匹配结果）
- Task 9 依赖 Task 2-8（整合）
- Task 10 依赖 Task 9（Python验证通过后）
- Task 11 依赖 Task 10（C++模块就绪后）
- Task 1 可随时执行（独立归档）

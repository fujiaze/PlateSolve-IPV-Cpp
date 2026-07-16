旧版 star_alignment 模块 - 归档

用途: PlateSolve第一步粗匹配的旧版实现

核心功能:
- 饱和星优先三角匹配 (siril atpmatch / Valdes 1995)
- RANSAC仿射回退
- iter_trans迭代精化 (35%分位数sigma, 10-sigma clip)
- 迭代重投影收敛

关键文件:
- psm_star_alignment.cpp: 主实现 (~900行单函数)
- psm_star_alignment.h: C API头文件
- psm_star_alignment.dll: 编译后的DLL
- star_alignment.dll: 旧版DLL

归档原因:
- 代码混乱，~900行单函数逻辑交织
- 缺少四种翻转模式测试
- 三角匹配与RANSAC回退逻辑混杂
- 迭代重投影参数硬编码

替代: 新的 initial_wcs 模块 (Python原型 + C++重写)
归档日期: 2026-06-02

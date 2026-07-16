# Checklist

- [x] 三角匹配仍用2000硬截断，未改变
- [x] 仿射确定后日志输出"RMS mag: 12.14 (target 3148 of 6918)"
- [x] 空间网格NN搜索耗时 ~0.2s（全流程含投影+仿射+NN=0.23s）
- [x] matched_stars数组非空，matched_count=245 > 0
- [x] 每对matched_star含img_x, img_y, cat_ra, cat_dec, cat_mag, residual_x, residual_y
- [x] psolve_free_coarse_result正确释放matched_stars（psolve_api.cpp:86）
- [x] plate_solve.dll 编译通过无错误
- [x] Python脚本运行输出affine_diagnostics.png
- [x] 图1：距离vs残差散点图，含全星RMS(7.09px)和前200亮星RMS参考线
- [x] 图2：径向直方图40bin，含图像角半径(2880px)标注
- [x] 控制台输出分bin RMS统计表（星数、RMS、中位残差）

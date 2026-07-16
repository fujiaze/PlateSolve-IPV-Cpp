"""
DD-SPPS 盲解析模块 (Density-Driven Signal Processing Plate Solving)
功能: 基于 FFT 频域信号处理的盲 plate solving 算法
用途: 仅以像素尺度 s0 为先验, 通过密度自适应星等选择→高斯核信号化→
      1D/2D 相位相关求解旋转与平移→WCS 构建+KD-tree 验证完成盲解析
依赖: blind_index_v2.python (io_wrappers, logging_setup), lib.plate_solve.python.vector_match_v2
"""

__all__ = [
    "density",
    "signal",
    "phase_correlation",
    "wcs",
    "diagnostics",
    "pipeline",
]

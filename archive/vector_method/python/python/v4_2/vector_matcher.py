"""
VectorMatcher 模块 - V4.2 Phase A/B 向量匹配器

功能:
    V4.2 模块化管线中的向量匹配器, 从 V4.1 抽样投票核心抽取并独立封装。
    负责 Phase A (PROSAC 抽样 + θ 直方图投票) 与 Phase B (三级过滤 + SVD 精修)。
    输出初始相似变换 (s, θ, tx, ty) 与粗匹配对 (cu, cw), 供下游 PairExpander 扩增。

用途:
    Python 端调用 V4.2 C++ DLL 的接口, 封装输入输出结构体, 实现 match() 方法。
    输入 U(N×2, 图像侧角秒坐标), W(M×2, Gaia 侧角秒坐标)。
    输出 dict: {cu, cw, s, theta, tx, ty, rms, theta_snr, success, ...}。

依赖:
    - vector_matcher.dll (C++ 编译产物)
    - numpy, ctypes
"""

import ctypes
import numpy as np
import os
from typing import Dict, Any, Optional, List


# ============================================================================
# ctypes 结构体定义 (与 vm_api.h 严格对应, 默认对齐)
# ============================================================================

class VectorMatcherParamsC(ctypes.Structure):
    _fields_ = [
        # 像素尺度与尺度范围
        ("s0", ctypes.c_double),
        ("s_min", ctypes.c_double),
        ("s_max", ctypes.c_double),
        ("n_modes", ctypes.c_int),
        ("seed", ctypes.c_int),
        # Phase A 抽样
        ("K_total", ctypes.c_int),
        ("batch_size", ctypes.c_int),
        ("min_samples", ctypes.c_int),
        ("K_top", ctypes.c_int),
        # Phase B 过滤
        ("min_inliers", ctypes.c_int),
        # PROSAC 参数
        ("w_snr", ctypes.c_double),
        ("w_sparse", ctypes.c_double),
        ("w_sat", ctypes.c_double),
        ("prosac_T_max", ctypes.c_int),
        ("use_prosac", ctypes.c_int),
        # 日志
        ("log_file_path", ctypes.c_char_p),
        # 可选输入
        ("snr_values", ctypes.POINTER(ctypes.c_double)),
        ("is_saturated_values", ctypes.POINTER(ctypes.c_int)),
    ]


class VectorMatchResultC(ctypes.Structure):
    _fields_ = [
        # 初始变换
        ("s", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("tx", ctypes.c_double),
        ("ty", ctypes.c_double),
        ("rms", ctypes.c_double),
        # 匹配对
        ("cu", ctypes.POINTER(ctypes.c_int)),
        ("cw", ctypes.POINTER(ctypes.c_int)),
        ("n_pairs", ctypes.c_int),
        # 调试信息
        ("theta_snr", ctypes.c_double),
        ("theta_peak_deg", ctypes.c_double),
        ("best_n_range", ctypes.c_int),
        ("n_phasea_records", ctypes.c_int),
        ("prosac_quality_median", ctypes.c_double),
        ("prosac_pool_final", ctypes.c_int),
        ("best_mode", ctypes.c_int),
        ("success", ctypes.c_int),
    ]


class VectorMatcher:
    """V4.2 向量匹配器: PROSAC 抽样 + SVD 精修"""

    def __init__(self, dll_path: Optional[str] = None):
        """加载 vector_matcher.dll"""
        if dll_path is None:
            # 默认路径: cpp/v4_2/vector_matcher/vector_matcher.dll
            here = os.path.dirname(os.path.abspath(__file__))
            # python/v4_2 -> ../../cpp/v4_2/vector_matcher
            project_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
            dll_path = os.path.join(
                project_root, "lib", "plate_solve", "cpp", "v4_2",
                "vector_matcher", "vector_matcher.dll"
            )
        self._dll_path = dll_path
        try:
            self._dll = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"无法加载 vector_matcher.dll: {dll_path}, 错误: {e}")

        # 配置函数签名
        self._dll.vm_match.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,   # U, N_img
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,   # W, M
            ctypes.POINTER(VectorMatcherParamsC),            # params
            ctypes.POINTER(VectorMatchResultC),              # result
        ]
        self._dll.vm_match.restype = ctypes.c_int

        self._dll.vm_free_result.argtypes = [ctypes.POINTER(VectorMatchResultC)]
        self._dll.vm_free_result.restype = None

    def match(self, U: np.ndarray, W: np.ndarray,
              s0: float = 0.0,
              s_min: float = 0.0,
              s_max: float = 0.0,
              n_modes: int = 4,
              seed: int = 42,
              K_total: int = 10000,
              batch_size: int = 500,
              min_samples: int = 50,
              K_top: int = 100,
              min_inliers: int = 5,
              w_snr: float = 0.4,
              w_sparse: float = 0.4,
              w_sat: float = 0.2,
              prosac_T_max: int = 10000,
              use_prosac: bool = True,
              snr_values: Optional[np.ndarray] = None,
              is_saturated_values: Optional[np.ndarray] = None,
              log_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        执行向量匹配 (Phase A PROSAC + Phase B SVD)

        Args:
            U: 图像侧星点数组 (N, 2) 角秒坐标
            W: Gaia 侧星点数组 (M, 2) 角秒坐标
            s0: 像素尺度(角秒/像素)
            s_min, s_max: 尺度搜索范围
            n_modes: 翻转模式数(默认4: 0=无翻转, 1=X, 2=Y, 3=XY)
            seed: 随机种子
            K_total: 总抽样次数上限
            batch_size: SNR 检查批大小
            min_samples: 启动 SNR 检查前最小抽样数
            K_top: Top-K 候选(预留)
            min_inliers: 模式选用的最小内点数
            w_snr, w_sparse, w_sat: PROSAC 质量分权重
            prosac_T_max: PROSAC 增长函数 T_max
            use_prosac: 是否启用 PROSAC
            snr_values: 图像星 SNR 数组(可选, 长度=N)
            is_saturated_values: 图像星饱和标志(可选, 长度=N, 1=饱和)
            log_file_path: 日志文件路径(可选)

        Returns:
            dict: {cu, cw, s, theta, tx, ty, rms, theta_snr, theta_peak_deg,
                   best_n_range, n_phasea_records, prosac_quality_median,
                   prosac_pool_final, best_mode, success}
        """
        # 输入校验
        U = np.ascontiguousarray(U, dtype=np.float64)
        W = np.ascontiguousarray(W, dtype=np.float64)
        if U.ndim != 2 or U.shape[1] != 2:
            raise ValueError(f"U 必须是 (N, 2) 形状, 实际: {U.shape}")
        if W.ndim != 2 or W.shape[1] != 2:
            raise ValueError(f"W 必须是 (M, 2) 形状, 实际: {W.shape}")
        N_img = U.shape[0]
        M = W.shape[0]

        # 默认尺度范围
        if s0 <= 0:
            # 自动估计: 用 U 中位模长 / 中位模长 (粗略估计 1")
            s0 = 1.0
        if s_min <= 0:
            s_min = s0 * 0.5
        if s_max <= 0:
            s_max = s0 * 2.0

        # 构建 ctypes 参数
        params = VectorMatcherParamsC()
        params.s0 = float(s0)
        params.s_min = float(s_min)
        params.s_max = float(s_max)
        params.n_modes = int(n_modes)
        params.seed = int(seed)
        params.K_total = int(K_total)
        params.batch_size = int(batch_size)
        params.min_samples = int(min_samples)
        params.K_top = int(K_top)
        params.min_inliers = int(min_inliers)
        params.w_snr = float(w_snr)
        params.w_sparse = float(w_sparse)
        params.w_sat = float(w_sat)
        params.prosac_T_max = int(prosac_T_max)
        params.use_prosac = 1 if use_prosac else 0

        # 日志路径
        if log_file_path:
            params.log_file_path = log_file_path.encode("utf-8")
        else:
            params.log_file_path = None

        # 可选 SNR / 饱和数组
        snr_ptr = None
        sat_ptr = None
        snr_arr = None
        sat_arr = None
        if snr_values is not None:
            snr_arr = np.ascontiguousarray(snr_values, dtype=np.float64)
            if snr_arr.shape[0] != N_img:
                raise ValueError(f"snr_values 长度 {snr_arr.shape[0]} != N {N_img}")
            snr_ptr = snr_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        if is_saturated_values is not None:
            sat_arr = np.ascontiguousarray(is_saturated_values, dtype=np.int32)
            if sat_arr.shape[0] != N_img:
                raise ValueError(f"is_saturated_values 长度 {sat_arr.shape[0]} != N {N_img}")
            sat_ptr = sat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        params.snr_values = snr_ptr
        params.is_saturated_values = sat_ptr

        # 调用 DLL
        result = VectorMatchResultC()
        # 预置 cu/cw 为 NULL (C 端会分配)
        result.cu = None
        result.cw = None

        U_ptr = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        W_ptr = W.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        try:
            ret = self._dll.vm_match(U_ptr, N_img, W_ptr, M,
                                     ctypes.byref(params),
                                     ctypes.byref(result))
        finally:
            # 保证 snr_arr/sat_arr 不被提前 GC
            _ = snr_arr
            _ = sat_arr

        if ret != 0:
            return {
                "success": False,
                "cu": [], "cw": [],
                "s": 0.0, "theta": 0.0, "tx": 0.0, "ty": 0.0, "rms": 0.0,
                "theta_snr": 0.0, "theta_peak_deg": 0.0,
                "best_n_range": 0, "n_phasea_records": 0,
                "prosac_quality_median": 0.0, "prosac_pool_final": 0,
                "best_mode": -1, "n_pairs": 0,
                "error": f"vm_match 返回错误码: {ret}",
            }

        # 提取结果
        n_pairs = result.n_pairs
        cu_list: List[int] = []
        cw_list: List[int] = []
        if n_pairs > 0 and result.cu and result.cw:
            cu_list = [result.cu[i] for i in range(n_pairs)]
            cw_list = [result.cw[i] for i in range(n_pairs)]

        out = {
            "success": bool(result.success),
            "cu": cu_list,
            "cw": cw_list,
            "n_pairs": n_pairs,
            "s": result.s,
            "theta": result.theta,
            "tx": result.tx,
            "ty": result.ty,
            "rms": result.rms,
            "theta_snr": result.theta_snr,
            "theta_peak_deg": result.theta_peak_deg,
            "best_n_range": result.best_n_range,
            "n_phasea_records": result.n_phasea_records,
            "prosac_quality_median": result.prosac_quality_median,
            "prosac_pool_final": result.prosac_pool_final,
            "best_mode": result.best_mode,
        }

        # 释放 C 端分配的内存
        self._dll.vm_free_result(ctypes.byref(result))
        return out

    def close(self):
        """释放 DLL 句柄(可选)"""
        self._dll = None

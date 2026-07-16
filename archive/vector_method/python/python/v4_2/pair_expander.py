"""
PairExpander 模块 - V4.2 Phase C 匹配对扩增器

功能: 用初始 CD 矩阵变换 W → U 空间, 线性扫描 NN 找匹配对, 区域均匀化
用途: V4.2 模块化管线 Phase C, 替代 V4.1 的 KDTree/nanoflann 实现
依赖: pair_expander.dll (C++17 编译, 无 nanoflann/k-vector/Eigen)

核心简化:
  - 移除 k-vector/KDTree/nanoflann, 改用线性扫描 NN
  - 理由: CD 矩阵已知后 Wt 与 U 已接近 (<3×s0), N≤2000 下线性扫描 <20ms
  - 模长比过滤替代 V4.1 贝叶斯增量过滤

算法流程:
  1. 变换 W → U 空间: Wt[j] = s·R·W[j] + t
  2. 对每个 Wt[j] 线性扫描 U 找最近邻 u* (O(N·M))
  3. τ 截断: best_d2 < τ² (τ=tau_factor×s0)
  4. 模长比过滤: |‖U[u]‖/‖W[w]‖ - 1| < scale_ratio_tol
  5. 1对1 贪心互斥(按距离升序), 保留 Phase B 的对
  6. 区域均匀化: region_size_px, N_floor/N_cap/N_max

DLL: pair_expander.dll
"""

import ctypes
import numpy as np
import os
from typing import Dict, Any, Optional, List


class _PairExpanderParamsC(ctypes.Structure):
    """ctypes 对应 C++ PairExpanderParams"""
    _fields_ = [
        ("s0", ctypes.c_double),
        ("tau_factor", ctypes.c_double),
        ("scale_ratio_tol", ctypes.c_double),
        ("region_size_px", ctypes.c_int),
        ("N_floor", ctypes.c_int),
        ("N_cap", ctypes.c_int),
        ("N_max", ctypes.c_int),
        ("img_width", ctypes.c_double),
        ("img_height", ctypes.c_double),
        ("log_file_path", ctypes.c_char_p),
    ]


class _ExpansionResultC(ctypes.Structure):
    """ctypes 对应 C++ ExpansionResult"""
    _fields_ = [
        ("expand_u", ctypes.POINTER(ctypes.c_int)),
        ("expand_w", ctypes.POINTER(ctypes.c_int)),
        ("n_pairs", ctypes.c_int),
        ("n_expanded", ctypes.c_int),
        ("n_regions", ctypes.c_int),
        ("n_sparse_regions", ctypes.c_int),
        ("n_candidates", ctypes.c_int),
        ("n_accepted", ctypes.c_int),
        ("expand_time_ms", ctypes.c_double),
        ("success", ctypes.c_int),
    ]


class PairExpander:
    """V4.2 匹配对扩增器: 线性扫描 NN + 模长比过滤 + 区域均匀化"""

    def __init__(self, dll_path: Optional[str] = None):
        """加载 pair_expander.dll

        Args:
            dll_path: DLL 路径, None 时使用默认路径
        """
        if dll_path is None:
            # 默认: lib/plate_solve/cpp/v4_2/pair_expander/pair_expander.dll
            here_dir = os.path.dirname(os.path.abspath(__file__))
            cpp_dir = os.path.normpath(os.path.join(here_dir, '..', '..', 'cpp', 'v4_2', 'pair_expander'))
            dll_path = os.path.join(cpp_dir, 'pair_expander.dll')

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"pair_expander.dll 未找到: {dll_path}")

        self._dll = ctypes.CDLL(dll_path)

        # 设置函数签名
        self._dll.pe_expand.restype = ctypes.c_int
        self._dll.pe_expand.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,           # U, N_img
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,           # W, M
            ctypes.POINTER(ctypes.c_int),                            # init_cu
            ctypes.POINTER(ctypes.c_int),                            # init_cw
            ctypes.c_int,                                            # n_init
            ctypes.c_double, ctypes.c_double,                        # s, theta
            ctypes.c_double, ctypes.c_double,                        # tx, ty
            ctypes.POINTER(_PairExpanderParamsC),                    # params
            ctypes.POINTER(_ExpansionResultC),                       # result
        ]

        self._dll.pe_free.restype = None
        self._dll.pe_free.argtypes = [ctypes.POINTER(_ExpansionResultC)]

    def expand(self, U: np.ndarray, W: np.ndarray,
               T: Dict[str, Any],
               init_cu: Optional[List[int]] = None,
               init_cw: Optional[List[int]] = None,
               s0: float = 1.0,
               img_width: float = 4500,
               img_height: float = 3600,
               tau_factor: float = 3.0,
               scale_ratio_tol: float = 0.1,
               region_size_px: int = 800,
               N_floor: int = 5,
               N_cap: int = 30,
               N_max: int = 1500,
               log_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行匹配对扩增 (Phase C 线性扫描 NN + 区域均匀化)

        Args:
            U: 图像侧星点数组 (N, 2) 角秒坐标, 原点在图像中心
            W: Gaia 侧星点数组 (M, 2) 角秒坐标, 原点在图像中心
            T: 初始相似变换 {s, theta, tx, ty} 来自 VectorMatcher
            init_cu: Phase B 初始匹配对 U 索引列表 (可选)
            init_cw: Phase B 初始匹配对 W 索引列表 (可选)
            s0: 像素尺度 (角秒/像素), 用于 τ=tau_factor×s0 和区域尺寸
            img_width: 图像宽度 (像素), 用于区域划分
            img_height: 图像高度 (像素), 用于区域划分
            tau_factor: 距离阈值因子 τ=tau_factor×s0 (默认3.0)
            scale_ratio_tol: 模长比容差 (默认0.1)
            region_size_px: 区域网格大小 (像素, 默认800)
            N_floor: 每区最少对数 (稀疏区除外, 默认5)
            N_cap: 每区最多对数 (默认30)
            N_max: 全局最多对数 (默认1500)
            log_dir: 日志目录 (可选, None 时禁用日志)

        Returns:
            dict: {
                success: bool,
                expand_u: list[int],      # 扩充后 U 索引 (Phase B + 扩充)
                expand_w: list[int],      # 扩充后 W 索引
                n_pairs: int,             # 总对数
                n_expanded: int,          # 扩充对数 (不含 Phase B)
                meta: {
                    n_regions, n_sparse_regions, n_candidates,
                    n_accepted, expand_time_ms
                }
            }
        """
        # 准备数据 (确保连续 float64)
        U_arr = np.ascontiguousarray(U, dtype=np.float64)
        W_arr = np.ascontiguousarray(W, dtype=np.float64)
        if U_arr.ndim != 2 or U_arr.shape[1] != 2:
            raise ValueError(f"U 必须是 (N, 2) 数组, 实际 shape={U_arr.shape}")
        if W_arr.ndim != 2 or W_arr.shape[1] != 2:
            raise ValueError(f"W 必须是 (M, 2) 数组, 实际 shape={W_arr.shape}")

        N_img = U_arr.shape[0]
        M = W_arr.shape[0]

        # Phase B 初始匹配对
        if init_cu is not None and init_cw is not None and len(init_cu) > 0:
            init_cu_arr = np.ascontiguousarray(init_cu, dtype=np.int32)
            init_cw_arr = np.ascontiguousarray(init_cw, dtype=np.int32)
            n_init = len(init_cu_arr)
            cu_ptr = init_cu_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            cw_ptr = init_cw_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        else:
            n_init = 0
            cu_ptr = None
            cw_ptr = None

        # 参数结构体
        params = _PairExpanderParamsC()
        params.s0 = float(s0)
        params.tau_factor = float(tau_factor)
        params.scale_ratio_tol = float(scale_ratio_tol)
        params.region_size_px = int(region_size_px)
        params.N_floor = int(N_floor)
        params.N_cap = int(N_cap)
        params.N_max = int(N_max)
        params.img_width = float(img_width)
        params.img_height = float(img_height)
        params.log_file_path = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'phase_c_pair_expander.log')
            params.log_file_path = log_path.encode('utf-8')

        # 调用 C++ DLL
        result = _ExpansionResultC()
        ret = self._dll.pe_expand(
            U_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N_img,
            W_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
            cu_ptr, cw_ptr, n_init,
            float(T['s']), float(T['theta']), float(T['tx']), float(T['ty']),
            ctypes.byref(params),
            ctypes.byref(result),
        )

        if ret != 1 or result.success != 1:
            self._dll.pe_free(ctypes.byref(result))
            return {
                'success': False,
                'expand_u': [],
                'expand_w': [],
                'n_pairs': 0,
                'n_expanded': 0,
                'meta': {'error': f'pe_expand 返回 ret={ret}, success={result.success}'},
            }

        # 提取结果 (在 free 之前拷贝)
        n_pairs = result.n_pairs
        expand_u = [int(result.expand_u[i]) for i in range(n_pairs)]
        expand_w = [int(result.expand_w[i]) for i in range(n_pairs)]

        meta = {
            'n_pairs': int(result.n_pairs),
            'n_expanded': int(result.n_expanded),
            'n_regions': int(result.n_regions),
            'n_sparse_regions': int(result.n_sparse_regions),
            'n_candidates': int(result.n_candidates),
            'n_accepted': int(result.n_accepted),
            'expand_time_ms': float(result.expand_time_ms),
        }

        # 释放 C++ 内存
        self._dll.pe_free(ctypes.byref(result))

        return {
            'success': True,
            'expand_u': expand_u,
            'expand_w': expand_w,
            'n_pairs': n_pairs,
            'n_expanded': meta['n_expanded'],
            'meta': meta,
        }

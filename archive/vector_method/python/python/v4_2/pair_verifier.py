"""
PairVerifier 模块 - V4.2 Phase D/D' 验证器

功能: 对匹配对执行 MAD 清洗 + 贝叶斯假设验证 + 三角形双特征验证
用途: V4.2 模块化管线 Phase D/D', 从 V4.1 vm4_core/vm4_bayes/vm4_triangle 迁移
依赖: pair_verifier.dll (C++17 编译, 依赖 Eigen3, 无 OpenMP)

算法流程:
  1. Phase D: 3 轮 MAD 迭代清洗, 阈值 max(5", 3×1.4826×MAD)
     - 含鲁棒预过滤: 初始 Umeyama 被离群拉偏时(init_med>min_thresh),
       用 thresh_factor×init_med 粗过滤剔除明显离群, 重新 Umeyama 收敛
  2. Phase D' 贝叶斯: lnK = Σ[-log(2πσ²) - r²/(2σ²)] + n×log(A_fov_sqsec)
     - 决策: lnK>20.7→接受, >6.9→弱证据, 否则→拒绝
  3. Phase D' 三角形: 面积 A + 极惯性矩 J 双特征, 通过率阈值 0.8
  4. validated = (bayes_decision >= 0) && (triangle_pass_ratio >= threshold)

DLL: pair_verifier.dll
"""

import ctypes
import numpy as np
import os
from typing import Dict, Any, Optional, List


class _PairVerifierParamsC(ctypes.Structure):
    """ctypes 对应 C++ PairVerifierParams"""
    _fields_ = [
        ("mad_iters", ctypes.c_int),
        ("mad_threshold_factor", ctypes.c_double),
        ("mad_min_threshold_arcsec", ctypes.c_double),
        ("lnK_accept", ctypes.c_double),
        ("lnK_weak", ctypes.c_double),
        ("sigma_min", ctypes.c_double),
        ("eps_A", ctypes.c_double),
        ("eps_J", ctypes.c_double),
        ("triangle_pass_rate", ctypes.c_double),
        ("fov_diag_deg", ctypes.c_double),
        ("log_file_path", ctypes.c_char_p),
    ]


class _VerificationResultC(ctypes.Structure):
    """ctypes 对应 C++ VerificationResult"""
    _fields_ = [
        ("clean_u", ctypes.POINTER(ctypes.c_int)),
        ("clean_w", ctypes.POINTER(ctypes.c_int)),
        ("n_clean", ctypes.c_int),
        ("n_removed", ctypes.c_int),
        ("mad_iterations", ctypes.c_int),
        ("mad_rms_arcsec", ctypes.c_double),
        ("bayes_lnK", ctypes.c_double),
        ("bayes_n_match", ctypes.c_int),
        ("bayes_decision", ctypes.c_int),
        ("triangle_total", ctypes.c_int),
        ("triangle_passed", ctypes.c_int),
        ("triangle_pass_ratio", ctypes.c_double),
        ("validated", ctypes.c_int),
        ("success", ctypes.c_int),
    ]


class PairVerifier:
    """V4.2 验证器: MAD 清洗 + 贝叶斯 + 三角形验证"""

    def __init__(self, dll_path: Optional[str] = None):
        """加载 pair_verifier.dll

        Args:
            dll_path: DLL 路径, None 时使用默认路径
        """
        if dll_path is None:
            # 默认: lib/plate_solve/cpp/v4_2/pair_verifier/pair_verifier.dll
            here_dir = os.path.dirname(os.path.abspath(__file__))
            cpp_dir = os.path.normpath(os.path.join(here_dir, '..', '..', 'cpp', 'v4_2', 'pair_verifier'))
            dll_path = os.path.join(cpp_dir, 'pair_verifier.dll')

        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"pair_verifier.dll 未找到: {dll_path}")

        self._dll = ctypes.CDLL(dll_path)

        # 设置函数签名
        self._dll.pv_verify.restype = ctypes.c_int
        self._dll.pv_verify.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,           # U, N_img
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,           # W, M
            ctypes.POINTER(ctypes.c_int),                            # pairs_u
            ctypes.POINTER(ctypes.c_int),                            # pairs_w
            ctypes.c_int,                                            # n_pairs
            ctypes.c_double,                                         # s0
            ctypes.POINTER(_PairVerifierParamsC),                    # params
            ctypes.POINTER(_VerificationResultC),                    # result
        ]

        self._dll.pv_free.restype = None
        self._dll.pv_free.argtypes = [ctypes.POINTER(_VerificationResultC)]

    def verify(self, U: np.ndarray, W: np.ndarray,
               pairs: list,
               s0: float = 2.0,
               mad_iters: int = 3,
               mad_threshold_factor: float = 3.0,
               mad_min_threshold_arcsec: float = 5.0,
               lnK_accept: float = 20.7,
               lnK_weak: float = 6.9,
               sigma_min: float = 0.5,
               eps_A: float = 0.05,
               eps_J: float = 0.10,
               triangle_pass_rate: float = 0.8,
               fov_diag_deg: float = 2.0,
               log_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行验证 (Phase D MAD 清洗 + Phase D' 贝叶斯+三角形)

        Args:
            U: 图像侧星点数组 (N, 2) 角秒坐标, 原点在图像中心
            W: Gaia 侧星点数组 (M, 2) 角秒坐标, 原点在图像中心
            pairs: 匹配对列表 [[u, w], ...] 来自 PairExpander
            s0: 像素尺度 (角秒/像素, 默认2.0)
            mad_iters: MAD 清洗轮数 (默认3)
            mad_threshold_factor: MAD 阈值因子 (默认3.0, 阈值=max(5", factor×1.4826×MAD))
            mad_min_threshold_arcsec: MAD 最小阈值 (角秒, 默认5.0)
            lnK_accept: 贝叶斯接受阈值 (默认20.7)
            lnK_weak: 贝叶斯弱证据阈值 (默认6.9)
            sigma_min: 位置噪声σ下限 (角秒, 默认0.5)
            eps_A: 三角形面积相对容差 (默认0.05)
            eps_J: 三角形极惯性矩相对容差 (默认0.10)
            triangle_pass_rate: 三角形通过率阈值 (默认0.8)
            fov_diag_deg: FOV对角线 (度, 默认2.0, 用于贝叶斯 A_fov 计算)
            log_dir: 日志目录 (可选, None 时禁用日志)

        Returns:
            dict: {
                success: bool,             # 执行是否成功
                validated: bool,           # 综合验证是否通过
                pairs: list[[u, w]],       # 清洗后匹配对
                n_clean: int,              # 清洗后对数
                n_removed: int,            # 剔除数
                mad: {
                    iterations: int,       # MAD 迭代次数
                    rms_arcsec: float,     # 清洗后 RMS (角秒)
                },
                bayes: {
                    lnK: float,            # 贝叶斯因子对数
                    n_match: int,          # 匹配对数
                    decision: int,         # 1=接受, 0=弱证据, -1=拒绝
                },
                triangle: {
                    total: int,            # 三角形总数
                    passed: int,           # 通过数
                    pass_ratio: float,     # 通过率
                },
                meta: dict                 # 其他元数据
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

        # 匹配对转两个 int32 数组
        if len(pairs) == 0:
            return {
                'success': True, 'validated': False,
                'pairs': [], 'n_clean': 0, 'n_removed': 0,
                'mad': {'iterations': 0, 'rms_arcsec': 0.0},
                'bayes': {'lnK': 0.0, 'n_match': 0, 'decision': -1},
                'triangle': {'total': 0, 'passed': 0, 'pass_ratio': 0.0},
                'meta': {'error': '空匹配对列表'},
            }
        pairs_arr = np.ascontiguousarray(pairs, dtype=np.int32)
        if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
            raise ValueError(f"pairs 必须是 (n, 2) 数组, 实际 shape={pairs_arr.shape}")
        pairs_u_arr = np.ascontiguousarray(pairs_arr[:, 0], dtype=np.int32)
        pairs_w_arr = np.ascontiguousarray(pairs_arr[:, 1], dtype=np.int32)
        n_pairs = pairs_arr.shape[0]

        # 参数结构体
        params = _PairVerifierParamsC()
        params.mad_iters = int(mad_iters)
        params.mad_threshold_factor = float(mad_threshold_factor)
        params.mad_min_threshold_arcsec = float(mad_min_threshold_arcsec)
        params.lnK_accept = float(lnK_accept)
        params.lnK_weak = float(lnK_weak)
        params.sigma_min = float(sigma_min)
        params.eps_A = float(eps_A)
        params.eps_J = float(eps_J)
        params.triangle_pass_rate = float(triangle_pass_rate)
        params.fov_diag_deg = float(fov_diag_deg)
        params.log_file_path = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'phase_d_pair_verifier.log')
            params.log_file_path = log_path.encode('utf-8')

        # 调用 C++ DLL
        result = _VerificationResultC()
        ret = self._dll.pv_verify(
            U_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N_img,
            W_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
            pairs_u_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            pairs_w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_pairs,
            float(s0),
            ctypes.byref(params),
            ctypes.byref(result),
        )

        if ret != 1 or result.success != 1:
            self._dll.pv_free(ctypes.byref(result))
            return {
                'success': False, 'validated': False,
                'pairs': [], 'n_clean': 0, 'n_removed': 0,
                'mad': {'iterations': 0, 'rms_arcsec': 0.0},
                'bayes': {'lnK': 0.0, 'n_match': 0, 'decision': -1},
                'triangle': {'total': 0, 'passed': 0, 'pass_ratio': 0.0},
                'meta': {'error': f'pv_verify 返回 ret={ret}, success={result.success}'},
            }

        # 提取结果 (在 free 之前拷贝)
        n_clean = int(result.n_clean)
        clean_pairs = []
        if n_clean > 0 and result.clean_u and result.clean_w:
            for i in range(n_clean):
                clean_pairs.append([int(result.clean_u[i]), int(result.clean_w[i])])

        out = {
            'success': True,
            'validated': bool(result.validated),
            'pairs': clean_pairs,
            'n_clean': n_clean,
            'n_removed': int(result.n_removed),
            'mad': {
                'iterations': int(result.mad_iterations),
                'rms_arcsec': float(result.mad_rms_arcsec),
            },
            'bayes': {
                'lnK': float(result.bayes_lnK),
                'n_match': int(result.bayes_n_match),
                'decision': int(result.bayes_decision),
            },
            'triangle': {
                'total': int(result.triangle_total),
                'passed': int(result.triangle_passed),
                'pass_ratio': float(result.triangle_pass_ratio),
            },
            'meta': {
                'n_input_pairs': n_pairs,
                'n_clean': n_clean,
            },
        }

        # 释放 C++ 堆内存
        self._dll.pv_free(ctypes.byref(result))
        return out

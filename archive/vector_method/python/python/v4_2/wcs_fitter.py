"""
WcsFitter 模块 - V4.2 Phase E WCS 拟合器

功能:
    V4.2 模块化管线中的 WCS 拟合器, 从 V4.1 Phase E 分层 SIP 拟合逻辑迁移。
    负责将匹配对 (U, W, pairs) 转换为标准 WCS 参数 (CD, CRVAL, CRPIX, SIP)。
    分三层:
      Layer 0: Umeyama SVD → CD 矩阵
      Layer 1: 像素残差 MAD 剔除 outlier → 6 参数全仿射 → 更新 CD/CRVAL
      Layer 2: BIC 选择 SIP 阶数 (2-4 阶, 高阶需 BIC 差 > 2 才选)

用途:
    Python 端调用 V4.2 C++ DLL 的接口, 封装输入输出结构体, 实现 fit() 方法。
    输入 U(N×2, 图像侧角秒坐标), W(M×2, 星表侧角秒坐标), pairs[[u,w],...]。
    输出 dict: {cd, crval, crpix, sip_A, sip_B, sip_order, rms_px, n_pairs, success}。

依赖:
    - wcs_fitter.dll (C++ 编译产物)
    - numpy, ctypes
"""

import ctypes
import numpy as np
import os
from typing import Dict, Any, Optional, List


# ============================================================================
# ctypes 结构体定义 (与 wf_api.h / v42_types.h 严格对应)
# ============================================================================

class WcsFitterParamsC(ctypes.Structure):
    """与 wf_api.h 的 WcsFitterParams 对应 (默认对齐, 无 _pack_)"""
    _fields_ = [
        ("s0", ctypes.c_double),
        ("sip_max_order", ctypes.c_int),
        ("skip_sip", ctypes.c_int),
        ("img_width", ctypes.c_double),
        ("img_height", ctypes.c_double),
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("log_file_path", ctypes.c_char_p),
    ]


class WcsResultC(ctypes.Structure):
    """与 v42_types.h 的 v42::WcsResult 对应 (默认对齐, 无 _pack_)"""
    _fields_ = [
        ("cd", ctypes.c_double * 4),
        ("crval", ctypes.c_double * 2),
        ("crpix", ctypes.c_double * 2),
        ("sip_A", ctypes.c_double * 36),
        ("sip_B", ctypes.c_double * 36),
        ("sip_order", ctypes.c_int),
        ("rms_px", ctypes.c_double),
        ("n_pairs", ctypes.c_int),
        ("success", ctypes.c_bool),
    ]


class WcsFitter:
    """V4.2 WCS 拟合器: 分层 SIP 拟合"""

    def __init__(self, dll_path: Optional[str] = None):
        """加载 wcs_fitter.dll"""
        if dll_path is None:
            # 默认路径: cpp/v4_2/wcs_fitter/wcs_fitter.dll
            here = os.path.dirname(os.path.abspath(__file__))
            # python/v4_2 -> ../../cpp/v4_2/wcs_fitter
            project_root = os.path.normpath(os.path.join(here, "..", "..", "..", ".."))
            dll_path = os.path.join(
                project_root, "lib", "plate_solve", "cpp", "v4_2",
                "wcs_fitter", "wcs_fitter.dll"
            )
        self._dll_path = dll_path
        try:
            self._dll = ctypes.CDLL(dll_path)
        except OSError as e:
            raise RuntimeError(f"无法加载 wcs_fitter.dll: {dll_path}, 错误: {e}")

        # 配置函数签名
        self._dll.wf_fit.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,   # U, N_img
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,   # W, M
            ctypes.POINTER(ctypes.c_int),                     # pairs_u
            ctypes.POINTER(ctypes.c_int),                     # pairs_w
            ctypes.c_int,                                     # n_pairs
            ctypes.POINTER(WcsFitterParamsC),                 # params
            ctypes.POINTER(WcsResultC),                       # result
        ]
        self._dll.wf_fit.restype = ctypes.c_int

    def fit(self, U: np.ndarray, W: np.ndarray,
            pairs: list,
            ra: float, dec: float,
            focal_length_mm: float, pixel_size_um: float,
            img_width: int = 0, img_height: int = 0,
            sip_max_order: int = 4,
            skip_sip: bool = False,
            log_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        执行 WCS 拟合 (Phase E 分层 SIP)

        Args:
            U: 图像侧星点数组 (N, 2) 角秒坐标
            W: Gaia 侧星点数组 (M, 2) 角秒坐标
            pairs: 匹配对列表 [[u, w], ...] 来自 PairVerifier
            ra, dec: 图像中心赤经赤纬(度)
            focal_length_mm: 焦距(mm)
            pixel_size_um: 像元尺寸(um)
            img_width: 图像宽度(像素)
            img_height: 图像高度(像素)
            sip_max_order: SIP 最大阶数(默认4)
            skip_sip: 是否跳过SIP拟合(仅线性CD)
            log_dir: 日志目录(可选)

        Returns:
            dict: {cd: list[4], crval: list[2], crpix: list[2],
                   sip_A: list[36], sip_B: list[36], sip_order: int,
                   rms_px: float, n_pairs: int, success: bool}
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

        if len(pairs) == 0:
            return {
                "success": False, "cd": [0.0]*4, "crval": [ra, dec],
                "crpix": [img_width/2.0, img_height/2.0],
                "sip_A": [0.0]*36, "sip_B": [0.0]*36,
                "sip_order": 0, "rms_px": 0.0, "n_pairs": 0,
                "error": "pairs 为空",
            }

        # 拆分匹配对
        pairs_u = np.array([p[0] for p in pairs], dtype=np.int32)
        pairs_w = np.array([p[1] for p in pairs], dtype=np.int32)
        n_pairs = len(pairs)

        # 计算像素尺度 s0 (arcsec/pixel)
        # s0 = 206.265 * pixel_size[um] / focal_length[mm]
        if focal_length_mm <= 0 or pixel_size_um <= 0:
            raise ValueError(f"焦距和像元尺寸必须 > 0: focal={focal_length_mm}, pixel={pixel_size_um}")
        s0 = 206.265 * pixel_size_um / focal_length_mm

        # 构建 ctypes 参数
        params = WcsFitterParamsC()
        params.s0 = float(s0)
        params.sip_max_order = int(sip_max_order)
        params.skip_sip = 1 if skip_sip else 0
        params.img_width = float(img_width)
        params.img_height = float(img_height)
        params.center_ra = float(ra)
        params.center_dec = float(dec)

        # 日志路径
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "phase_e_wcs_fitter.log")
            params.log_file_path = log_path.encode("utf-8")
        else:
            params.log_file_path = None

        # 调用 DLL
        result = WcsResultC()
        U_ptr = U.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        W_ptr = W.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pairs_u_ptr = pairs_u.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        pairs_w_ptr = pairs_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        ret = self._dll.wf_fit(U_ptr, N_img, W_ptr, M,
                               pairs_u_ptr, pairs_w_ptr, n_pairs,
                               ctypes.byref(params),
                               ctypes.byref(result))

        if ret == 0:
            return {
                "success": False,
                "cd": [0.0]*4,
                "crval": [ra, dec],
                "crpix": [img_width/2.0, img_height/2.0],
                "sip_A": [0.0]*36, "sip_B": [0.0]*36,
                "sip_order": 0, "rms_px": 0.0, "n_pairs": 0,
                "error": "wf_fit 返回失败",
            }

        # 提取结果 (WcsResult 是值类型, 无指针, 不需要 free)
        return {
            "success": bool(result.success),
            "cd": [result.cd[i] for i in range(4)],
            "crval": [result.crval[i] for i in range(2)],
            "crpix": [result.crpix[i] for i in range(2)],
            "sip_A": [result.sip_A[i] for i in range(36)],
            "sip_B": [result.sip_B[i] for i in range(36)],
            "sip_order": int(result.sip_order),
            "rms_px": float(result.rms_px),
            "n_pairs": int(result.n_pairs),
        }

    def close(self):
        """释放 DLL 句柄(可选)"""
        self._dll = None

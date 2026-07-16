"""
vector_match_v4_4_cpp.py - V4.4 单一 DLL ctypes 封装

功能:
    V4.4 统一管线 ctypes 封装, 在 V4.3 基础上用相对向量法 (DMPDV) 完全替代单θ Phase A
    内部串联: StarSelector → 相对向量法(Phase A) → Phase B → IRM 闭环 (Expand ↔ Verify ↔ Fit)

用途:
    Python 端调用 V4.4 求解器, 输入 FITS 路径 + 中心指向 + 焦距/像元,
    输出标准 WCS 参数 (CD/CRVAL/CRPIX/SIP) + IRM 调试信息 (S_robust/n_inliers/n_iters)

依赖:
    - vector_match_v4_4.dll (由 lib/plate_solve/cpp/v4_4/Makefile 编译)
    - gaia_client.dll (经 GaiaClientPy 封装, 可注入)
    - star_detector.dll (经 StarDetector 封装, 可注入)
    - numpy, ctypes

V4.4 vs V4.3 改进:
    1. 相对向量法 (DMPDV) 完全替代单θ Phase A
       - Δu_ij = U[i]-U[j] 消除平移 t, 把 4D (s,θ,tx,ty) 降为 1D θ 搜索
       - 在 t≠0 场景 (Type3失败帧, 如 Galaxy_Center t=100-160") 仍能成功
    2. U 组限流 (max_u=100): 解决 LDN43 (U=271) 候选爆炸问题
    3. 第三星交叉验证: 用第一对精确 s, 容差仅噪声级 ±3"
    4. k-vector 距离查询: 预排序 Gaia 星对距离, 二分查找
    5. 其余 IRM 闭环逻辑与 V4.3 一致
"""

import os
import sys
import math
import ctypes
import logging
from typing import Any, Dict, Optional

# MinGW runtime 必须在 PATH 中 (libgcc_s_seh, libstdc++, libgomp)
# V4.4 DLL 由 g++ 编译, 运行时依赖这些动态库
# Python 3.8+ 改变了 DLL 搜索行为, 必须用 os.add_dll_directory 显式添加搜索路径
_MINGW_BIN = r"C:\msys64\mingw64\bin"
if _MINGW_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except (OSError, FileNotFoundError):
        pass

import numpy as np

# 项目根与路径
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
_PLATE_SOLVE_PY = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "python")
if _PLATE_SOLVE_PY not in sys.path:
    sys.path.insert(0, _PLATE_SOLVE_PY)
_STAR_DET_PY = os.path.join(_PROJECT_ROOT, "lib", "star_detector", "python")
if _STAR_DET_PY not in sys.path:
    sys.path.insert(0, _STAR_DET_PY)

logger = logging.getLogger("v4_4_solver")


# ============================================================================
# ctypes 结构体定义 (与 C 结构体一一对应)
# ============================================================================

class VM44SolveParams(ctypes.Structure):
    """V4.4 求解参数 (对应 C 结构体 VM44SolveParams)

    字段顺序必须与 C 结构体 vm44_types.h 中的 VM44SolveParams 完全一致。
    V4.4 在 irm_s_initial 之后、log_dir 之前新增了 12 个 relvec_* 参数
    (7 个基础相对向量法参数 + 5 个自适应采样停止参数)。
    """
    _pack_ = 8
    _fields_ = [
        # 基础参数
        ("n_modes", ctypes.c_int),
        ("seed", ctypes.c_int),
        # StarSelector 参数
        ("img_n_target", ctypes.c_int),
        ("gaia_density_ratio", ctypes.c_double),
        ("gaia_query_radius_factor", ctypes.c_double),
        ("m_lim_step", ctypes.c_double),
        ("m_lim_max_iter", ctypes.c_int),
        ("density_tolerance", ctypes.c_double),
        # VectorMatcher 参数
        ("s_min", ctypes.c_double),
        ("s_max", ctypes.c_double),
        ("K_total", ctypes.c_int),
        ("batch_size", ctypes.c_int),
        ("min_samples", ctypes.c_int),
        ("K_top", ctypes.c_int),
        ("min_inliers", ctypes.c_int),
        ("w_snr", ctypes.c_double),
        ("w_sparse", ctypes.c_double),
        ("w_sat", ctypes.c_double),
        ("prosac_T_max", ctypes.c_int),
        ("use_prosac", ctypes.c_int),
        # PairExpander 参数
        ("region_size_px", ctypes.c_double),
        ("N_floor", ctypes.c_int),
        ("N_cap", ctypes.c_int),
        ("N_max", ctypes.c_int),
        # PairVerifier 参数
        ("mad_iters", ctypes.c_int),
        ("mad_threshold_factor", ctypes.c_double),
        ("mad_min_threshold_arcsec", ctypes.c_double),
        ("lnK_accept", ctypes.c_double),
        ("lnK_weak", ctypes.c_double),
        ("eps_A", ctypes.c_double),
        ("eps_J", ctypes.c_double),
        ("triangle_pass_rate", ctypes.c_double),
        # WcsFitter 参数
        ("sip_max_order", ctypes.c_int),
        ("skip_sip", ctypes.c_int),
        # IRM 闭环参数
        ("irm_max_iter", ctypes.c_int),
        ("irm_converge_eps", ctypes.c_double),
        ("irm_diverge_factor", ctypes.c_double),
        ("irm_tau_min", ctypes.c_double),
        ("irm_tau_factor", ctypes.c_double),
        ("irm_lowe_ratio", ctypes.c_double),
        ("irm_k_geometry", ctypes.c_int),
        ("irm_geom_threshold", ctypes.c_int),
        ("irm_geom_dist_tol", ctypes.c_double),
        ("irm_ransac_max_iter", ctypes.c_int),
        ("irm_ransac_min_inliers", ctypes.c_int),
        ("irm_huber_delta_factor", ctypes.c_double),
        ("irm_sip_min_pairs", ctypes.c_int),
        ("irm_s_initial", ctypes.c_int),
        # --- 相对向量法参数 (V4.4 新增, 完全替代单θ Phase A) ---
        ("relvec_n_samples", ctypes.c_int),        # 最大采样上限(默认5000, 自适应停止可能提前结束)
        ("relvec_max_u", ctypes.c_int),            # U组限流上限(默认100)
        ("relvec_third_star_tol", ctypes.c_double),# 第三星验证容差(像素, 默认1.5; 按s_est转换到Gaia角秒域)
        ("relvec_max_cand", ctypes.c_int),         # 单次采样候选上限(默认500)
        ("relvec_min_len_frac", ctypes.c_double),  # 最小星对距离比例(默认0.05)
        ("relvec_max_len_frac", ctypes.c_double),  # 最大星对距离比例(默认0.8)
        ("relvec_n_third_stars", ctypes.c_int),    # 第三星验证颗数(默认0=用所有可用, 投票无上限; >0=随机采样上限)
        # 自适应采样停止参数 (V4.4 优化, 替代固定次数)
        ("relvec_adaptive_stop", ctypes.c_int),    # 启用自适应停止(默认1)
        ("relvec_min_samples", ctypes.c_int),      # 最少采样次数(默认200)
        ("relvec_check_interval", ctypes.c_int),   # SNR检查间隔(默认100)
        ("relvec_snr_eps", ctypes.c_double),       # SNR相对变化阈值(默认0.05=5%)
        ("relvec_max_stable", ctypes.c_int),       # 连续稳定次数(默认3)
        # 日志
        ("log_dir", ctypes.c_char_p),
    ]


class _VM44SolveResult(ctypes.Structure):
    """V4.4 求解结果 (对应 C 结构体 VM44SolveResult, 内部使用)

    与 V4.3 结果结构体完全一致 (相对向量法的调试信息复用 theta_snr/theta_peak_deg 字段)。
    """
    _pack_ = 8
    _fields_ = [
        ("cd", ctypes.c_double * 4),
        ("crval", ctypes.c_double * 2),
        ("crpix", ctypes.c_double * 2),
        ("sip_A", ctypes.c_double * 36),
        ("sip_B", ctypes.c_double * 36),
        ("sip_order", ctypes.c_int),
        ("rms_px", ctypes.c_double),
        ("rms_arcsec", ctypes.c_double),
        ("s_robust", ctypes.c_double),
        ("matched_count", ctypes.c_int),
        ("n_inliers", ctypes.c_int),
        ("n_iters", ctypes.c_int),
        ("irm_converged", ctypes.c_bool),
        ("scale_arcsec_px", ctypes.c_double),
        ("rotation_deg", ctypes.c_double),
        ("flip_mode", ctypes.c_int),
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("s0", ctypes.c_double),
        ("s", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("tx", ctypes.c_double),
        ("ty", ctypes.c_double),
        ("theta_snr", ctypes.c_double),
        ("theta_peak_deg", ctypes.c_double),
        ("bayes_lnK", ctypes.c_double),
        ("triangle_pass_ratio", ctypes.c_double),
        ("best_mode", ctypes.c_int),
        ("cu", ctypes.POINTER(ctypes.c_int)),
        ("cw", ctypes.POINTER(ctypes.c_int)),
        ("n_pairs", ctypes.c_int),
        ("img_width", ctypes.c_int),
        ("img_height", ctypes.c_int),
        ("fov_diag_deg", ctypes.c_double),
        ("m_lim_final", ctypes.c_double),
        ("n_gaia_final", ctypes.c_int),
        ("success", ctypes.c_bool),
        ("error_msg", ctypes.c_char * 256),
    ]


# ============================================================================
# 默认 DLL 路径
# ============================================================================

_DEFAULT_DLL_PATH = os.path.join(
    _PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_4", "vector_match_v4_4.dll"
)
_DEFAULT_GAIA_DATA_DIR = os.path.join(_PROJECT_ROOT, "GaiaDR3")


# ============================================================================
# V44Solver 主类
# ============================================================================

class V44Solver:
    """V4.4 单一 DLL 求解器

    在 V4.3 基础上用相对向量法 (DMPDV) 完全替代单θ Phase A,
    在 t≠0 场景 (Type3失败帧) 仍能成功

    用法:
        with V44Solver() as solver:
            result = solver.solve("image.fits", ra=10.0, dec=20.0,
                                  focal_length_mm=1000.0, pixel_size_um=5.0)
            if result["success"]:
                print(f"RMS={result['rms_px']:.3f}px matched={result['matched_count']}")
    """

    def __init__(self,
                 dll_path: Optional[str] = None,
                 gaia_client: Optional[Any] = None,
                 star_detector: Optional[Any] = None):
        """初始化 V4.4 求解器

        Args:
            dll_path: vector_match_v4_4.dll 路径
                     None 时使用默认路径 lib/plate_solve/cpp/v4_4/vector_match_v4_4.dll
            gaia_client: 已实例化的 GaiaClientPy (None 时内部创建, 默认 GaiaDR3)
            star_detector: 已实例化的 StarDetector (None 时内部创建, fitRadius=0 自动)
        """
        # --- 加载 V4.4 DLL ---
        if dll_path is None:
            dll_path = _DEFAULT_DLL_PATH
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"V4.4 DLL 未找到: {dll_path}")

        self._dll = ctypes.CDLL(dll_path)
        self._dll_path = dll_path
        self._setup_argtypes()

        # --- Gaia 客户端: 注入优先, 否则内部创建 ---
        self._gaia_external = gaia_client is not None
        if gaia_client is not None:
            self._gaia_client = gaia_client
        else:
            from vector_match_v2 import GaiaClientPy  # 延迟导入
            self._gaia_client = GaiaClientPy(_DEFAULT_GAIA_DATA_DIR, db_type=0)
            logger.info("V4.4 内部创建 GaiaClientPy (GaiaDR3)")

        # --- StarDetector: 注入优先, 否则内部创建 ---
        self._star_detector_external = star_detector is not None
        if star_detector is not None:
            self._star_detector = star_detector
        else:
            try:
                from star_detector import StarDetector, SDetParamsPy
                self._star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
                logger.info("V4.4 内部创建 StarDetector (fitRadius=0 自动)")
            except Exception as e:
                logger.warning("StarDetector 内部创建失败: %s", e)
                self._star_detector = None

        # --- 注入句柄到 C++ ---
        self._inject_handles()

        self._closed = False
        logger.info("V44Solver 初始化完成 (dll=%s)", os.path.basename(dll_path))

    # ========================================================================
    # ctypes 函数签名配置
    # ========================================================================

    def _setup_argtypes(self):
        """配置 V4.4 DLL 的函数签名"""
        dll = self._dll

        # vm44_solve(image_path, ra, dec, focal_length_mm, pixel_size_um,
        #            params*, result*) -> int
        dll.vm44_solve.argtypes = [
            ctypes.c_char_p,              # image_path
            ctypes.c_double,              # ra
            ctypes.c_double,              # dec
            ctypes.c_double,              # focal_length_mm
            ctypes.c_double,              # pixel_size_um
            ctypes.POINTER(VM44SolveParams),  # params (可为 NULL)
            ctypes.POINTER(_VM44SolveResult),  # result
        ]
        dll.vm44_solve.restype = ctypes.c_int

        # vm44_set_gaia_client(handle) -> void
        dll.vm44_set_gaia_client.argtypes = [ctypes.c_void_p]
        dll.vm44_set_gaia_client.restype = None

        # vm44_set_star_detector(handle) -> void
        dll.vm44_set_star_detector.argtypes = [ctypes.c_void_p]
        dll.vm44_set_star_detector.restype = None

        # vm44_free_result(result*) -> void
        dll.vm44_free_result.argtypes = [ctypes.POINTER(_VM44SolveResult)]
        dll.vm44_free_result.restype = None

        # vm44_get_default_params(params*) -> void
        dll.vm44_get_default_params.argtypes = [ctypes.POINTER(VM44SolveParams)]
        dll.vm44_get_default_params.restype = None

    # ========================================================================
    # 句柄注入
    # ========================================================================

    def _inject_handles(self):
        """从 GaiaClientPy / StarDetector 提取内部句柄, 注入 V4.4 DLL"""
        # GaiaClient: _handle 是 c_void_p (GaiaClient*)
        gaia_handle = getattr(self._gaia_client, "_handle", None)
        if not gaia_handle:
            raise RuntimeError("GaiaClientPy._handle 为空, 无法注入")
        # c_void_p.value 为 int 或 None
        gaia_handle_int = gaia_handle if isinstance(gaia_handle, int) else gaia_handle
        self._dll.vm44_set_gaia_client(ctypes.c_void_p(gaia_handle_int))
        logger.debug("注入 GaiaClient handle=0x%x", gaia_handle_int)

        # StarDetector: _handle 是 c_void_p (StarDetectorHandle)
        if self._star_detector is None:
            raise RuntimeError("StarDetector 未初始化, 无法注入")
        sdet_handle = getattr(self._star_detector, "_handle", None)
        if not sdet_handle:
            raise RuntimeError("StarDetector._handle 为空, 无法注入")
        sdet_handle_int = sdet_handle if isinstance(sdet_handle, int) else sdet_handle
        self._dll.vm44_set_star_detector(ctypes.c_void_p(sdet_handle_int))
        logger.debug("注入 StarDetector handle=0x%x", sdet_handle_int)

    # ========================================================================
    # 公共求解接口
    # ========================================================================

    def solve(self,
              image_path: str,
              ra: float, dec: float,
              focal_length_mm: float,
              pixel_size_um: float,
              params: Optional[VM44SolveParams] = None,
              log_dir: Optional[str] = None) -> Dict[str, Any]:
        """一键求解 (与 V4.1/V4.2/V4.3 接口兼容)

        Args:
            image_path: FITS 图像路径
            ra, dec: 图像中心赤经赤纬(度)
            focal_length_mm: 焦距(mm)
            pixel_size_um: 像元尺寸(um)
            params: 求解参数 (None 用默认值)
            log_dir: 日志目录 (None 用默认 logs/v4_4/<frame>/)

        Returns:
            dict: SolveResult, 含:
                success, cd, crval, crpix, sip_A, sip_B, sip_order, rms_px,
                rms_arcsec, s_robust, matched_count, n_inliers, n_iters,
                irm_converged, scale_arcsec_px, rotation_deg, flip_mode,
                center_ra, center_dec, s0, s, theta, tx, ty,
                theta_snr (相对向量法 SNR), theta_peak_deg, bayes_lnK,
                triangle_pass_ratio, ...
        """
        if self._closed:
            raise RuntimeError("V44Solver 已关闭, 不能再调用")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        # --- 日志目录 ---
        frame_base = os.path.splitext(os.path.basename(image_path))[0]
        if log_dir is None:
            log_dir = os.path.join(
                _PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_4", frame_base)
        os.makedirs(log_dir, exist_ok=True)

        # --- 参数准备 ---
        if params is None:
            params = VM44SolveParams()
            self._dll.vm44_get_default_params(ctypes.byref(params))

        # 设置 log_dir (字节串)
        log_dir_bytes = log_dir.encode("utf-8")
        params.log_dir = ctypes.c_char_p(log_dir_bytes)

        # --- 调用 vm44_solve ---
        image_path_bytes = image_path.encode("utf-8")
        result = _VM44SolveResult()

        logger.info("[%s] vm44_solve 开始 (RA=%.6f Dec=%.6f focal=%.1fmm pixel=%.2fum)",
                    frame_base, ra, dec, focal_length_mm, pixel_size_um)

        ret = self._dll.vm44_solve(
            image_path_bytes,
            ctypes.c_double(ra), ctypes.c_double(dec),
            ctypes.c_double(focal_length_mm), ctypes.c_double(pixel_size_um),
            ctypes.byref(params),
            ctypes.byref(result),
        )

        # --- 转换结果 ---
        solve_result = self._convert_result(result, frame_base, log_dir)

        # --- 释放 C++ 内部分配的 cu/cw ---
        self._dll.vm44_free_result(ctypes.byref(result))

        if ret != 0 or not result.success:
            logger.error("[%s] vm44_solve 失败: %s", frame_base,
                         solve_result.get("error", "未知错误"))
        else:
            logger.info("[%s] vm44_solve 成功: RMS=%.4fpx S_robust=%.4f\" matched=%d iters=%d",
                        frame_base, solve_result["rms_px"], solve_result["s_robust"],
                        solve_result["matched_count"], solve_result["n_iters"])

        return solve_result

    # ========================================================================
    # 结果转换
    # ========================================================================

    @staticmethod
    def _convert_result(result: _VM44SolveResult,
                        frame_base: str,
                        log_dir: str) -> Dict[str, Any]:
        """将 C 结构体 _VM44SolveResult 转换为 Python dict (V4.1/V4.2/V4.3 兼容)"""

        # 提取匹配对索引 (cu/cw)
        n_pairs = result.n_pairs
        cu_list = []
        cw_list = []
        if n_pairs > 0 and result.cu and result.cw:
            cu_list = [result.cu[i] for i in range(n_pairs)]
            cw_list = [result.cw[i] for i in range(n_pairs)]

        # 错误信息
        error_msg = result.error_msg.decode("utf-8", errors="replace") if result.error_msg else ""

        # 与 V4.3 兼容的字段
        return {
            # 状态
            "success": bool(result.success),
            "error": error_msg,
            "frame_base": frame_base,
            "log_dir": log_dir,

            # WCS 参数
            "cd": [float(result.cd[0]), float(result.cd[1]),
                   float(result.cd[2]), float(result.cd[3])],
            "crval": [float(result.crval[0]), float(result.crval[1])],
            "crpix": [float(result.crpix[0]), float(result.crpix[1])],
            "sip_A": [float(result.sip_A[i]) for i in range(36)],
            "sip_B": [float(result.sip_B[i]) for i in range(36)],
            "sip_order": int(result.sip_order),
            "sip_rms_px": float(result.rms_px),

            # 精度指标
            "rms_px": float(result.rms_px),
            "rms_arcsec": float(result.rms_arcsec),
            "s_robust": float(result.s_robust),
            "matched_count": int(result.matched_count),
            "n_inliers": int(result.n_inliers),
            "n_iters": int(result.n_iters),
            "irm_converged": bool(result.irm_converged),

            # 变换参数
            "scale_arcsec_px": float(result.scale_arcsec_px),
            "rotation_deg": float(result.rotation_deg),
            "flip_mode": int(result.flip_mode),
            "center_ra": float(result.center_ra),
            "center_dec": float(result.center_dec),
            "original_ra": float(result.crval[0]),
            "original_dec": float(result.crval[1]),
            "s0": float(result.s0),
            "s": float(result.s),
            "theta": float(result.theta),
            "tx": float(result.tx),
            "ty": float(result.ty),

            # 调试信息 (V4.4: theta_snr/theta_peak_deg 来自相对向量法)
            "theta_snr": float(result.theta_snr),
            "theta_peak_deg": float(result.theta_peak_deg),
            "bayes_lnK": float(result.bayes_lnK),
            "triangle_pass_ratio": float(result.triangle_pass_ratio),
            "best_mode": int(result.best_mode),

            # 匹配对索引
            "cu": cu_list,
            "cw": cw_list,
            "n_pairs": n_pairs,

            # 元数据
            "img_width": int(result.img_width),
            "img_height": int(result.img_height),
            "fov_diag_deg": float(result.fov_diag_deg),
            "m_lim_final": float(result.m_lim_final),
            "n_gaia_final": int(result.n_gaia_final),
        }

    # ========================================================================
    # 资源管理
    # ========================================================================

    def close(self):
        """释放资源 (外部注入的 GaiaClient/StarDetector 不负责关闭)"""
        if self._closed:
            return

        # 释放 Python 端 ctypes DLL 引用
        self._dll = None

        # 关闭内部创建的资源
        if not self._star_detector_external and self._star_detector is not None:
            try:
                self._star_detector.close()
            except Exception as e:
                logger.warning("StarDetector 关闭失败: %s", e)
            self._star_detector = None

        if not self._gaia_external and self._gaia_client is not None:
            try:
                self._gaia_client.close()
            except Exception as e:
                logger.warning("GaiaClient 关闭失败: %s", e)
            self._gaia_client = None

        self._closed = True
        logger.info("V44Solver 已关闭")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

"""
V4.5 相对向量法 θ 求解器 - Python ctypes 封装

功能:
    调用 vector_match_v4_5.dll 执行 plate solving 的 Phase A (θ 求解)
    仅输出 θ 峰值 + SNR + 直方图, 不计算 CD/SIP (V4.5 砍掉了 Phase B/IRM/WcsFitter)

用途:
    用于验证相对向量法 θ 求解的核心算法正确性
    输入: 图像路径 + 中心坐标 (RA/Dec) + 焦距 + 像素尺寸
    输出: θ_peak (度), SNR, 直方图 (numpy array), 元数据

使用方法:
    from vector_match_v4_5_cpp import Vm45Solver

    solver = Vm45Solver(dll_path="path/to/vector_match_v4_5.dll")
    solver.set_gaia_client(gaia_handle)
    solver.set_star_detector(detector_handle)

    result = solver.solve_theta(
        image_path="image.fits",
        ra=180.0, dec=45.0,
        focal_length_mm=2000.0,
        pixel_size_um=9.0,
    )
    print(f"θ = {result.theta_peak_deg:.3f}°, SNR = {result.theta_snr:.2f}")

    # 绘制 θ 直方图
    import matplotlib.pyplot as plt
    bins = np.linspace(-180, 180, 361)
    plt.bar(bins[:-1], result.theta_histogram, width=1.0)
    plt.axvline(result.theta_peak_deg, color='r', linestyle='--', label=f'θ_peak={result.theta_peak_deg:.2f}°')
    plt.xlabel('θ (deg)')
    plt.ylabel('votes (smoothed)')
    plt.legend()
    plt.savefig('theta_histogram.png', dpi=150)
"""

import os
import sys
import ctypes
import logging
from dataclasses import dataclass
from typing import Optional

# MinGW runtime 必须在 PATH 中 (libgomp 用于 OpenMP)
# V4.5 DLL 由 g++ 编译, 链接了 -fopenmp, 运行时依赖 libgomp-1.dll
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

logger = logging.getLogger("v4_5_solver")


# ============================================================================
# ctypes 结构体定义 (严格对应 vm45_types.h, 字段顺序必须一致)
# ============================================================================

class VM45SolveParams(ctypes.Structure):
    """V4.5 求解参数 (对应 C 结构体 v45::VM45SolveParams)

    字段顺序必须与 vm45_types.h 中的 VM45SolveParams 完全一致。
    仅含 Phase 0 (StarSelector) + Phase A (相对向量法) 参数,
    砍掉了 V4.4 的 IRM/Phase B/WcsFitter 参数。

    字段顺序 (与 vm45_types.h 第 91-123 行一致):
        1.  seed (int)
        2.  img_n_target (int)
        3.  gaia_density_ratio (double)
        4.  gaia_query_radius_factor (double)
        5.  m_lim_step (double)
        6.  m_lim_max_iter (int)
        7.  density_tolerance (double)
        8.  K_total (int)
        9.  sigma_d_px (double)
        10. n_third (int)
        11. third_ratio_min (double)
        12. theta_bw (double)
        13. snr_threshold (double)
        14. relvec_max_u (int)
        15. relvec_max_cand (int)
        16. relvec_min_len_frac (double)
        17. relvec_max_len_frac (double)
        18. adaptive_stop (int)
        19. min_samples (int)
        20. check_interval (int)
        21. snr_eps (double)
        22. max_stable (int)
        23. log_dir (const char*)
    """
    _pack_ = 8
    _fields_ = [
        # --- 基础参数 ---
        ("seed", ctypes.c_int),
        # --- StarSelector 参数 (Phase 0, 与 V4.4 一致) ---
        ("img_n_target", ctypes.c_int),
        ("gaia_density_ratio", ctypes.c_double),
        ("gaia_query_radius_factor", ctypes.c_double),
        ("m_lim_step", ctypes.c_double),
        ("m_lim_max_iter", ctypes.c_int),
        ("density_tolerance", ctypes.c_double),
        # --- 相对向量法参数 (Phase A, 严格按设计文档) ---
        ("K_total", ctypes.c_int),
        ("sigma_d_px", ctypes.c_double),
        ("n_third", ctypes.c_int),
        ("third_ratio_min", ctypes.c_double),
        ("theta_bw", ctypes.c_double),
        ("snr_threshold", ctypes.c_double),
        ("relvec_max_u", ctypes.c_int),
        ("relvec_max_cand", ctypes.c_int),
        ("relvec_min_len_frac", ctypes.c_double),
        ("relvec_max_len_frac", ctypes.c_double),
        # --- 自适应采样停止 (可选, 从 V4.4 沿用) ---
        ("adaptive_stop", ctypes.c_int),
        ("min_samples", ctypes.c_int),
        ("check_interval", ctypes.c_int),
        ("snr_eps", ctypes.c_double),
        ("max_stable", ctypes.c_int),
        # --- 日志 ---
        ("log_dir", ctypes.c_char_p),
    ]


class VM45SolveResult(ctypes.Structure):
    """V4.5 求解结果 (对应 C 结构体 v45::VM45SolveResult)

    字段顺序必须与 vm45_types.h 中的 VM45SolveResult 完全一致。
    仅含 Phase A 输出 (θ + SNR + 直方图 + 元数据),
    砍掉了 CD/SIP/tx/ty 等 V4.4 字段。

    字段顺序 (与 vm45_types.h 第 129-154 行一致):
        1.  theta_peak_deg (double)
        2.  theta_snr (double)
        3.  peak_bin (int)
        4.  theta_histogram (double*)
        5.  histogram_size (int)
        6.  n_passed (int)
        7.  n_samples (int)
        8.  img_width (int)
        9.  img_height (int)
        10. fov_diag_deg (double)
        11. m_lim_final (double)
        12. n_gaia_final (int)
        13. s0 (double)
        14. success (bool)
        15. error_msg (char[256])

    注:
        - theta_histogram 由 DLL 内部分配, 必须用 vm45_free_result 释放
        - 转 numpy 时必须 copy, 因为释放后指针失效
    """
    _pack_ = 8
    _fields_ = [
        # θ 求解结果
        ("theta_peak_deg", ctypes.c_double),
        ("theta_snr", ctypes.c_double),
        ("peak_bin", ctypes.c_int),
        # θ 直方图 (360 元素, 由 vm45_free_result 释放)
        ("theta_histogram", ctypes.POINTER(ctypes.c_double)),
        ("histogram_size", ctypes.c_int),
        # 通过候选数
        ("n_passed", ctypes.c_int),
        ("n_samples", ctypes.c_int),
        # 元数据
        ("img_width", ctypes.c_int),
        ("img_height", ctypes.c_int),
        ("fov_diag_deg", ctypes.c_double),
        ("m_lim_final", ctypes.c_double),
        ("n_gaia_final", ctypes.c_int),
        ("s0", ctypes.c_double),
        # 状态
        ("success", ctypes.c_bool),
        ("error_msg", ctypes.c_char * 256),
    ]


# ============================================================================
# 默认 DLL 路径
# ============================================================================

_DEFAULT_DLL_PATH = os.path.join(
    _PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_5", "vector_match_v4_5.dll"
)
_DEFAULT_GAIA_DATA_DIR = os.path.join(_PROJECT_ROOT, "GaiaDR3")


# ============================================================================
# Python 友好的结果封装
# ============================================================================

@dataclass
class Vm45SolveOutput:
    """Python 友好的结果封装 (从 VM45SolveResult 转换而来)

    所有字段均为 Python 原生类型, theta_histogram 为 numpy 数组 (已 copy, 可安全持有)。
    字段对应 C 结构体如下:
        success         <- result.success
        theta_peak_deg  <- result.theta_peak_deg
        theta_snr       <- result.theta_snr
        n_passed        <- result.n_passed
        n_samples       <- result.n_samples
        theta_histogram <- np.copy(result.theta_histogram[:histogram_size])
        img_width       <- result.img_width
        img_height      <- result.img_height
        fov_diag_deg    <- result.fov_diag_deg
        m_lim_final     <- result.m_lim_final
        s0              <- result.s0
        n_img           <- 0 (C 结构未暴露图像侧星数, StarSelection.U.size 未导出)
        n_gaia          <- result.n_gaia_final
        error_msg       <- result.error_msg.decode('utf-8')
    """
    success: bool
    theta_peak_deg: float
    theta_snr: float
    n_passed: int
    n_samples: int
    theta_histogram: np.ndarray  # shape=(360,)
    img_width: int
    img_height: int
    fov_diag_deg: float
    m_lim_final: float
    s0: float
    n_img: int
    n_gaia: int
    error_msg: str


# ============================================================================
# Vm45Solver 主类
# ============================================================================

class Vm45Solver:
    """V4.5 Plate Solver Python 封装

    仅执行 Phase A (相对向量法 θ 求解), 不计算 CD/SIP/WCS。
    依赖外部注入 GaiaClient 与 StarDetector 句柄 (复用 V4.4 同款依赖)。

    用法:
        solver = Vm45Solver(dll_path=".../vector_match_v4_5.dll", log_dir="./logs")
        solver.set_gaia_client(gaia_handle)
        solver.set_star_detector(detector_handle)

        result = solver.solve_theta(
            image_path="image.fits",
            ra=180.0, dec=45.0,
            focal_length_mm=2000.0,
            pixel_size_um=9.0,
        )
        if result.success:
            print(f"θ = {result.theta_peak_deg:.3f}°, SNR = {result.theta_snr:.2f}")

    或使用上下文管理器:
        with Vm45Solver() as solver:
            solver.set_gaia_client(...)
            solver.set_star_detector(...)
            result = solver.solve_theta(...)
    """

    def __init__(self,
                 dll_path: Optional[str] = None,
                 log_dir: Optional[str] = None):
        """初始化 V4.5 求解器

        参数:
            dll_path: vector_match_v4_5.dll 的路径。None 时在默认位置查找。
            log_dir: 日志目录。None 时不写日志文件 (传 NULL 给 C 端)。
        """
        self.dll_path = self._find_dll(dll_path)
        if not os.path.exists(self.dll_path):
            raise FileNotFoundError(f"V4.5 DLL 未找到: {self.dll_path}")

        logger.info("加载 V4.5 DLL: %s", self.dll_path)
        self.dll = ctypes.CDLL(self.dll_path)
        self._setup_signatures()

        # 日志目录 (绝对路径, 传给 C 端 log_dir 字段)
        self.log_dir = os.path.abspath(log_dir) if log_dir else None
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info("V4.5 日志目录: %s", self.log_dir)

        self._closed = False
        logger.info("Vm45Solver 初始化完成 (dll=%s)", os.path.basename(self.dll_path))

    # ------------------------------------------------------------------
    # DLL 查找
    # ------------------------------------------------------------------

    def _find_dll(self, dll_path: Optional[str] = None) -> str:
        """查找 DLL 路径

        优先级:
            1. 显式传入 dll_path
            2. 默认位置: <PROJECT_ROOT>/lib/plate_solve/cpp/v4_5/vector_match_v4_5.dll
            3. 当前工作目录下的 vector_match_v4_5.dll
        """
        if dll_path:
            return os.path.abspath(dll_path)

        candidates = [
            _DEFAULT_DLL_PATH,
            os.path.join(os.path.dirname(__file__), "..", "cpp", "v4_5", "vector_match_v4_5.dll"),
            "vector_match_v4_5.dll",
        ]
        for p in candidates:
            if p and os.path.isfile(p):
                return os.path.abspath(p)
        # 找不到时返回默认路径, 由上层抛 FileNotFoundError
        return _DEFAULT_DLL_PATH

    # ------------------------------------------------------------------
    # 函数签名配置
    # ------------------------------------------------------------------

    def _setup_signatures(self):
        """配置 V4.5 DLL 的 C 函数签名

        对应 vm45_api.h 中导出的 5 个函数:
            - vm45_solve(image_path, ra, dec, focal, pixel, params*, result*) -> int
            - vm45_set_gaia_client(void*) -> void
            - vm45_set_star_detector(void*) -> void
            - vm45_free_result(result*) -> void
            - vm45_get_default_params(params*) -> void
        """
        dll = self.dll

        # vm45_solve: 主求解接口
        # int vm45_solve(const char* image_path,
        #                double ra, double dec,
        #                double focal_length_mm, double pixel_size_um,
        #                const VM45SolveParams* params,
        #                VM45SolveResult* result);
        dll.vm45_solve.restype = ctypes.c_int
        dll.vm45_solve.argtypes = [
            ctypes.c_char_p,                          # image_path (UTF-8)
            ctypes.c_double,                          # ra
            ctypes.c_double,                          # dec
            ctypes.c_double,                          # focal_length_mm
            ctypes.c_double,                          # pixel_size_um
            ctypes.POINTER(VM45SolveParams),          # params (可为 NULL)
            ctypes.POINTER(VM45SolveResult),          # result
        ]

        # vm45_set_gaia_client: 注入 Gaia 客户端句柄
        dll.vm45_set_gaia_client.restype = None
        dll.vm45_set_gaia_client.argtypes = [ctypes.c_void_p]

        # vm45_set_star_detector: 注入 StarDetector 句柄
        dll.vm45_set_star_detector.restype = None
        dll.vm45_set_star_detector.argtypes = [ctypes.c_void_p]

        # vm45_free_result: 释放 result 内部分配的 theta_histogram
        dll.vm45_free_result.restype = None
        dll.vm45_free_result.argtypes = [ctypes.POINTER(VM45SolveResult)]

        # vm45_get_default_params: 获取默认参数
        dll.vm45_get_default_params.restype = None
        dll.vm45_get_default_params.argtypes = [ctypes.POINTER(VM45SolveParams)]

        logger.debug("V4.5 DLL 函数签名配置完成")

    # ------------------------------------------------------------------
    # 依赖注入接口
    # ------------------------------------------------------------------

    def set_gaia_client(self, gaia_handle: int):
        """注入 Gaia 客户端句柄 (复用已有 gaia_client.dll)

        参数:
            gaia_handle: GaiaClientPy 内部句柄 (void*). 通常为 gaia_client._handle,
                         可直接传入 int 或 ctypes.c_void_p.
        """
        if self._closed:
            raise RuntimeError("Vm45Solver 已关闭, 不能再调用")
        self.dll.vm45_set_gaia_client(ctypes.c_void_p(gaia_handle))
        logger.info("已注入 GaiaClient 句柄: 0x%x", gaia_handle)

    def set_star_detector(self, detector_handle: int):
        """注入 StarDetector 句柄 (复用已有 star_detector.dll)

        参数:
            detector_handle: StarDetector 内部句柄 (void*). 通常为 detector._handle,
                              可直接传入 int 或 ctypes.c_void_p.
        """
        if self._closed:
            raise RuntimeError("Vm45Solver 已关闭, 不能再调用")
        self.dll.vm45_set_star_detector(ctypes.c_void_p(detector_handle))
        logger.info("已注入 StarDetector 句柄: 0x%x", detector_handle)

    # ------------------------------------------------------------------
    # 默认参数
    # ------------------------------------------------------------------

    def get_default_params(self) -> VM45SolveParams:
        """获取默认求解参数

        返回填充了默认值的 VM45SolveParams (调用者可修改特定字段后再传入 solve_theta)。

        默认值 (与 vm45_types.h 注释一致):
            seed=42, img_n_target=50, gaia_density_ratio=1.5,
            gaia_query_radius_factor=0.55, m_lim_step=0.5, m_lim_max_iter=10,
            density_tolerance=0.1, K_total=20000, sigma_d_px=2.0, n_third=0,
            third_ratio_min=0.3, theta_bw=1.0, snr_threshold=5.0,
            relvec_max_u=100, relvec_max_cand=500,
            relvec_min_len_frac=0.05, relvec_max_len_frac=0.8,
            adaptive_stop=1, min_samples=200, check_interval=100,
            snr_eps=0.05, max_stable=3
        """
        if self._closed:
            raise RuntimeError("Vm45Solver 已关闭, 不能再调用")
        params = VM45SolveParams()
        self.dll.vm45_get_default_params(ctypes.byref(params))
        logger.debug("已获取默认参数: seed=%d, K_total=%d, sigma_d_px=%.2f, n_third=%d, ratio_min=%.2f, snr_threshold=%.2f",
                     params.seed, params.K_total, params.sigma_d_px, params.n_third,
                     params.third_ratio_min, params.snr_threshold)
        return params

    # ------------------------------------------------------------------
    # 主求解接口
    # ------------------------------------------------------------------

    def solve_theta(self,
                    image_path: str,
                    ra: float, dec: float,
                    focal_length_mm: float,
                    pixel_size_um: float,
                    params: Optional[VM45SolveParams] = None) -> Vm45SolveOutput:
        """执行 θ 求解 (Phase 0 + Phase A)

        内部串联:
            1. Phase 0 (StarSelector): 从图像提取星点 + 从 Gaia 查询星点
            2. Phase A (相对向量法): k-vector 距离查询 + 1D θ 直方图 + 高斯平滑 + 亚 bin 精化

        参数:
            image_path: 图像文件路径 (UTF-8, 通常为 FITS)
            ra: 中心赤经 (度)
            dec: 中心赤纬 (度)
            focal_length_mm: 焦距 (mm)
            pixel_size_um: 像素尺寸 (um)
            params: 可选参数。None 时用默认值。可用 self.get_default_params() 获取后修改。

        返回:
            Vm45SolveOutput 对象, 含 theta_peak_deg, theta_snr, theta_histogram 等。
            theta_histogram 为 numpy 数组 (已 copy, 可安全持有)。

        异常:
            FileNotFoundError: 图像文件不存在
            RuntimeError: vm45_solve 返回非 0 (错误信息在异常消息中)
        """
        if self._closed:
            raise RuntimeError("Vm45Solver 已关闭, 不能再调用")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        frame_base = os.path.splitext(os.path.basename(image_path))[0]

        # --- 参数准备 ---
        if params is None:
            params = self.get_default_params()

        # 设置 log_dir (保持字节串在作用域内, 避免 ctypes 提前释放)
        if self.log_dir:
            log_dir_bytes = self.log_dir.encode("utf-8")
            params.log_dir = ctypes.c_char_p(log_dir_bytes)
        else:
            log_dir_bytes = None
            params.log_dir = None

        # --- 调用 vm45_solve ---
        image_path_bytes = image_path.encode("utf-8")
        result = VM45SolveResult()

        logger.info("[%s] vm45_solve 开始: RA=%.6f Dec=%.6f focal=%.1fmm pixel=%.2fum",
                    frame_base, ra, dec, focal_length_mm, pixel_size_um)
        logger.info("[%s] 参数: K_total=%d, sigma_d_px=%.2f, n_third=%d, ratio_min=%.2f, theta_bw=%.2f, snr_threshold=%.2f",
                    frame_base, params.K_total, params.sigma_d_px, params.n_third,
                    params.third_ratio_min, params.theta_bw, params.snr_threshold)

        ret = self.dll.vm45_solve(
            image_path_bytes,
            ctypes.c_double(ra), ctypes.c_double(dec),
            ctypes.c_double(focal_length_mm), ctypes.c_double(pixel_size_um),
            ctypes.byref(params),
            ctypes.byref(result),
        )

        # --- 提取结果 (在 free 之前 copy) ---
        try:
            if ret != 0:
                error_msg = result.error_msg.decode("utf-8", errors="replace") if result.error_msg else ""
                if not error_msg:
                    error_msg = f"vm45_solve 返回 {ret}"
                logger.error("[%s] vm45_solve 失败 (rc=%d): %s", frame_base, ret, error_msg)
                raise RuntimeError(f"V4.5 求解失败: {error_msg}")

            # 提取直方图 (必须 copy, 因为 vm45_free_result 会释放原指针)
            n_bins = result.histogram_size if result.histogram_size > 0 else 360
            if result.theta_histogram and n_bins > 0:
                try:
                    histogram = np.ctypeslib.as_array(
                        result.theta_histogram, shape=(n_bins,)
                    ).copy()
                except Exception as e:
                    logger.warning("[%s] 直方图提取失败, 用零数组替代: %s", frame_base, e)
                    histogram = np.zeros(n_bins, dtype=np.float64)
            else:
                logger.warning("[%s] theta_histogram 为空, 用零数组替代", frame_base)
                histogram = np.zeros(n_bins, dtype=np.float64)

            error_msg = result.error_msg.decode("utf-8", errors="replace") if result.error_msg else ""

            output = Vm45SolveOutput(
                success=bool(result.success),
                theta_peak_deg=float(result.theta_peak_deg),
                theta_snr=float(result.theta_snr),
                n_passed=int(result.n_passed),
                n_samples=int(result.n_samples),
                theta_histogram=histogram,
                img_width=int(result.img_width),
                img_height=int(result.img_height),
                fov_diag_deg=float(result.fov_diag_deg),
                m_lim_final=float(result.m_lim_final),
                s0=float(result.s0),
                n_img=0,  # C 结构未暴露图像侧星数 (StarSelection.U.size 未导出)
                n_gaia=int(result.n_gaia_final),
                error_msg=error_msg,
            )

            if output.success:
                logger.info("[%s] vm45_solve 成功: θ=%.4f°, SNR=%.4f, n_passed=%d, n_samples=%d, "
                            "img=%dx%d, FOV=%.3f°, m_lim=%.2f, s0=%.4f\"/px, n_gaia=%d",
                            frame_base, output.theta_peak_deg, output.theta_snr,
                            output.n_passed, output.n_samples,
                            output.img_width, output.img_height,
                            output.fov_diag_deg, output.m_lim_final,
                            output.s0, output.n_gaia)
            else:
                logger.warning("[%s] vm45_solve 完成 但 success=false: θ=%.4f°, SNR=%.4f, err=%s",
                               frame_base, output.theta_peak_deg, output.theta_snr,
                               error_msg or "(无)")

            return output
        finally:
            # 释放 DLL 内部分配的 theta_histogram 内存 (Python 已 copy)
            self.dll.vm45_free_result(ctypes.byref(result))
            logger.debug("[%s] 已调用 vm45_free_result 释放内部内存", frame_base)

    # ------------------------------------------------------------------
    # 资源管理
    # ------------------------------------------------------------------

    def close(self):
        """释放资源 (Vm45Solver 不拥有 GaiaClient/StarDetector, 仅释放 DLL 引用)"""
        if self._closed:
            return
        self.dll = None
        self._closed = True
        logger.info("Vm45Solver 已关闭")

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# 模块自测试
# ============================================================================

if __name__ == "__main__":
    import argparse

    # 配置日志到 stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="V4.5 θ 求解器测试")
    parser.add_argument("--dll", help="DLL 路径 (默认: lib/plate_solve/cpp/v4_5/vector_match_v4_5.dll)")
    parser.add_argument("--image", required=True, help="图像路径 (FITS)")
    parser.add_argument("--ra", type=float, required=True, help="中心赤经 (度)")
    parser.add_argument("--dec", type=float, required=True, help="中心赤纬 (度)")
    parser.add_argument("--focal", type=float, required=True, help="焦距 (mm)")
    parser.add_argument("--pixel", type=float, required=True, help="像素尺寸 (um)")
    parser.add_argument("--log-dir", help="日志目录 (默认: 不写日志)")
    parser.add_argument("--gaia-dir", default=_DEFAULT_GAIA_DATA_DIR,
                        help=f"Gaia 数据目录 (默认: {_DEFAULT_GAIA_DATA_DIR})")
    args = parser.parse_args()

    solver = Vm45Solver(dll_path=args.dll, log_dir=args.log_dir)

    # 注入 GaiaClient (复用 V4.2 风格的 GaiaClientPy)
    gaia = None
    try:
        from vector_match_v2 import GaiaClientPy
        gaia = GaiaClientPy(args.gaia_dir, db_type=0)
        gaia_handle = gaia._handle
        if isinstance(gaia_handle, ctypes.c_void_p):
            gaia_handle = gaia_handle.value
        solver.set_gaia_client(gaia_handle)
    except ImportError:
        print("[警告] 未找到 vector_match_v2.GaiaClientPy, 跳过 Gaia 注入")
    except Exception as e:
        print(f"[警告] GaiaClient 初始化失败: {e}")
        gaia = None

    # 注入 StarDetector (复用 star_detector 模块)
    sdet = None
    try:
        from star_detector import StarDetector, SDetParamsPy
        sdet = StarDetector(params=SDetParamsPy(fitRadius=0))
        sdet_handle = sdet._handle
        if isinstance(sdet_handle, ctypes.c_void_p):
            sdet_handle = sdet_handle.value
        solver.set_star_detector(sdet_handle)
    except ImportError:
        print("[警告] 未找到 star_detector.StarDetector, 跳过注入")
    except Exception as e:
        print(f"[警告] StarDetector 初始化失败: {e}")
        sdet = None

    try:
        result = solver.solve_theta(
            image_path=args.image,
            ra=args.ra, dec=args.dec,
            focal_length_mm=args.focal,
            pixel_size_um=args.pixel,
        )

        print()
        print("=" * 60)
        print("V4.5 求解结果")
        print("=" * 60)
        print(f"成功: {result.success}")
        print(f"θ_peak = {result.theta_peak_deg:.4f}°")
        print(f"SNR   = {result.theta_snr:.4f}")
        print(f"n_passed = {result.n_passed}")
        print(f"n_samples = {result.n_samples}")
        print(f"img: {result.img_width}x{result.img_height}, FOV_diag = {result.fov_diag_deg:.3f}°")
        print(f"s0 = {result.s0:.4f}\"/px, n_img = {result.n_img}, n_gaia = {result.n_gaia}")
        print(f"m_lim_final = {result.m_lim_final:.2f}")
        if result.error_msg:
            print(f"error_msg: {result.error_msg}")
        print(f"theta_histogram: shape={result.theta_histogram.shape}, "
              f"sum={result.theta_histogram.sum():.4f}, "
              f"max={result.theta_histogram.max():.4f}")
    finally:
        solver.close()
        if gaia is not None:
            try:
                gaia.close()
            except Exception:
                pass
        if sdet is not None:
            try:
                sdet.close()
            except Exception:
                pass

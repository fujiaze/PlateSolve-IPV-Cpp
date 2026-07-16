"""
StarSelector 模块 - V4.2 Phase 0 选星器

功能:
    V4.2 模块化管线的选星器，封装 star_selector.dll，从图像中检测星点并
    按不对称策略选取图像侧星点，调用 C++ DLL 执行 Gaia 密度匹配查询，
    最终输出图像侧 U 向量组和 Gaia 侧 W 向量组，供下游 VectorMatcher 使用。

用途:
    输入 FITS 图像路径 + 中心指向 + 焦距/像元尺寸，输出选星结果:
      U: 图像侧星点向量组 (N×2, 角秒坐标, 原点在图像中心)
      W: Gaia 侧星表向量组 (M×2, 角秒坐标, gnomonic 投影)
      meta: 元数据 (s0/FOV/query_radius/m_lim/n_gaia/...)

    图像侧选星策略 (V4.1 不对称):
      - 饱和星数 > img_n_target → 全选饱和星
      - 否则 → 饱和全选 + 非饱和按 flux 降序补足到 img_n_target

    Gaia 侧密度匹配:
      - 查询半径 = FOV_diag × gaia_query_radius_factor (默认 0.55)
      - 目标星数 = max(50, round(gaia_density_ratio × n_img × query_area/img_area))
      - 自适应步长迭代极限星等: 前4次 step_init, 后续 step_init/2

依赖:
    - star_selector.dll (C++ 编译产物)
    - star_detector.dll + star_detector.py (图像星点检测)
    - astro_image_io.dll + astro_image_io.py (FITS/XISF 读取)
    - gaia_client.dll + vector_match_v2.GaiaClientPy (Gaia 锥形查询)
    - numpy, ctypes, logging
"""

import ctypes
import logging
import math
import os
import sys
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

# 复用 V4.1 的 GaiaClientPy 和 gnomonic_forward
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
_PLATE_SOLVE_PY = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "python")
if _PLATE_SOLVE_PY not in sys.path:
    sys.path.insert(0, _PLATE_SOLVE_PY)
_STAR_DET_PY = os.path.join(_PROJECT_ROOT, "lib", "star_detector", "python")
if _STAR_DET_PY not in sys.path:
    sys.path.insert(0, _STAR_DET_PY)
_ASTRO_IO_PY = os.path.join(_PROJECT_ROOT, "lib", "astro_image_io", "python")
if _ASTRO_IO_PY not in sys.path:
    sys.path.insert(0, _ASTRO_IO_PY)

from vector_match_v2 import GaiaClientPy, gnomonic_forward  # noqa: E402

logger = logging.getLogger("v4_2_star_selector")


# ============================================================================
# ctypes 结构体定义（与 ss_api.h 严格对应）
# ============================================================================

class StarSelectorParamsC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("img_n_target", ctypes.c_int),
        ("gaia_density_ratio", ctypes.c_double),
        ("gaia_query_radius_factor", ctypes.c_double),
        ("m_lim_step", ctypes.c_double),
        ("m_lim_max_iter", ctypes.c_int),
        ("density_tolerance", ctypes.c_double),
        ("focal_length_mm", ctypes.c_double),
        ("pixel_size_um", ctypes.c_double),
        ("img_width", ctypes.c_double),
        ("img_height", ctypes.c_double),
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("n_img_bright", ctypes.c_int),
        ("exposure_time_s", ctypes.c_double),
        ("log_file_path", ctypes.c_char_p),
    ]


class StarSelectionResultC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s0", ctypes.c_double),
        ("fov_diag_deg", ctypes.c_double),
        ("query_radius_deg", ctypes.c_double),
        ("m_lim_final", ctypes.c_double),
        ("n_gaia_final", ctypes.c_int),
        ("m_lim_iterations", ctypes.c_int),
        ("converged", ctypes.c_bool),
        ("n_target", ctypes.c_int),
        ("n_img_selected", ctypes.c_int),
        ("n_gaia_selected", ctypes.c_int),
        ("rho_img", ctypes.c_double),
        ("rho_target", ctypes.c_double),
        ("query_area_sqdeg", ctypes.c_double),
        ("img_area_sqdeg", ctypes.c_double),
    ]


# Gaia 查询回调: (ra, dec, radius_deg, mag_lim) -> 星数
GaiaQueryFuncC = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
)


# ============================================================================
# 工具函数
# ============================================================================

def _find_dll(dll_path: Optional[str] = None) -> str:
    """查找 star_selector.dll 路径"""
    if dll_path and os.path.exists(dll_path):
        return dll_path
    candidates = [
        os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2",
                     "star_selector", "star_selector.dll"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"star_selector.dll 未找到，请先在 lib/plate_solve/cpp/v4_2/star_selector/ 下编译。"
        f" 已查找: {candidates}")


def _make_log_path(log_dir: Optional[str]) -> Optional[str]:
    """生成日志文件路径，log_dir=None 时返回 None（不写日志）"""
    if log_dir is None:
        return None
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, "phase_0_star_selector.log")


def _load_image(image_path: str) -> Tuple[np.ndarray, int, int]:
    """读取 FITS/XISF 图像为 uint16 数组，返回 (data, width, height)"""
    from astro_image_io import ImageReader
    reader = ImageReader()
    img = reader.read(image_path)
    data = img.data
    if data.dtype != np.uint16:
        data = np.clip(data, 0, 65535).astype(np.uint16)
    h, w = data.shape
    return data, w, h


def _detect_stars(data: np.ndarray, detector=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """检测星点，返回 (x, y, flux, saturated) 四个数组

    Args:
        data: 图像数据 (uint16)
        detector: 已有的 StarDetector 实例（None 则创建新实例）
    """
    from star_detector import StarDetector
    if detector is None:
        detector = StarDetector()
    det = detector.detect_ex(data)
    # detect_ex 返回 StarDetectionResult (含 .x/.y/.flux/.saturated 列表属性)
    if len(det.x) == 0:
        return (np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.bool_))
    img_x = np.array(det.x, dtype=np.float64)
    img_y = np.array(det.y, dtype=np.float64)
    img_flux = np.array(det.flux, dtype=np.float64)
    img_sat = np.array(det.saturated, dtype=np.bool_)
    return img_x, img_y, img_flux, img_sat


def _select_image_stars(img_x: np.ndarray, img_y: np.ndarray,
                         img_flux: np.ndarray, img_sat: np.ndarray,
                         img_n_target: int) -> np.ndarray:
    """V4.1 不对称图像侧选星策略

    - 饱和星数 > img_n_target → 全选饱和星
    - 否则 → 饱和全选 + 非饱和按 flux 降序补足到 img_n_target

    Returns:
        sel_idx: 选中星点在原数组中的索引
    """
    sat_idx = np.where(img_sat)[0]
    nsat = len(sat_idx)
    non_sat_idx = np.where(~img_sat)[0]
    if len(non_sat_idx) > 0:
        non_sat_sorted = non_sat_idx[np.argsort(-img_flux[non_sat_idx])]
    else:
        non_sat_sorted = np.array([], dtype=np.int64)

    if nsat > img_n_target:
        sel_idx = sat_idx
        logger.info("V4.2 选星: 饱和星 %d 颗 > img_n_target=%d, 全选饱和星",
                    nsat, img_n_target)
    else:
        n_needed = max(0, img_n_target - nsat)
        top_non_sat = non_sat_sorted[:n_needed]
        sel_idx = np.concatenate([sat_idx, top_non_sat]).astype(np.int64)
        logger.info("V4.2 选星: 饱和 %d + 非饱和 %d = %d 颗 (img_n_target=%d)",
                    nsat, n_needed, len(sel_idx), img_n_target)
    return sel_idx


# ============================================================================
# 主封装类
# ============================================================================

class StarSelector:
    """V4.2 选星器: 图像侧 + Gaia 侧不对称选星

    用法:
        with StarSelector(gaia_data_dir=".../GaiaDR3", db_type=1) as sel:
            result = sel.select("image.fits", ra=100.0, dec=30.0,
                                focal_length_mm=200.0, pixel_size_um=3.76)
            U, W, meta = result["U"], result["W"], result["meta"]
    """

    def __init__(self,
                 dll_path: Optional[str] = None,
                 gaia_data_dir: Optional[str] = None,
                 db_type: int = 0,
                 gaia_client: Optional[GaiaClientPy] = None,
                 star_detector: Optional[Any] = None):
        """加载 star_selector.dll 和初始化 Gaia 客户端

        Args:
            dll_path: star_selector.dll 路径（None 则自动查找）
            gaia_data_dir: Gaia 数据目录（None 则延迟到 select() 时校验）
            db_type: Gaia 数据库类型（0=auto, 1=DR3, 2=DR3SP）
            gaia_client: 已实例化的 GaiaClientPy（优先于 gaia_data_dir，外部注入时不负责关闭）
            star_detector: 已实例化的 StarDetector（None 则在 _detect_stars 中创建）
        """
        # 加载 DLL
        self._dll_path = _find_dll(dll_path)
        mingw_bin = r"C:\msys64\mingw64\bin"
        if os.path.isdir(mingw_bin):
            os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(mingw_bin)
            except OSError:
                pass
        dll_dir = os.path.dirname(os.path.abspath(self._dll_path))
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass
        self._dll = ctypes.CDLL(self._dll_path)
        self._dll.ss_density_match.argtypes = [
            ctypes.POINTER(StarSelectorParamsC),
            GaiaQueryFuncC,
            ctypes.POINTER(StarSelectionResultC),
        ]
        self._dll.ss_density_match.restype = ctypes.c_int

        # Gaia 客户端（注入优先，否则按 gaia_data_dir 创建）
        self._gaia_data_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia: Optional[GaiaClientPy] = None
        self._gaia_external = gaia_client is not None
        if gaia_client is not None:
            self._gaia = gaia_client
        elif gaia_data_dir is not None:
            self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        # 星点检测器（注入或 None）
        self._star_detector = star_detector
        self._closed = False

    def _ensure_gaia(self) -> GaiaClientPy:
        if self._gaia is None:
            raise RuntimeError(
                "Gaia 客户端未初始化，请在 __init__ 中传入 gaia_data_dir 或 gaia_client")
        return self._gaia

    def select(self, image_path: str, ra: float, dec: float,
               focal_length_mm: float, pixel_size_um: float,
               img_n_target: int = 50,
               gaia_density_ratio: float = 1.5,
               gaia_query_radius_factor: float = 0.55,
               m_lim_step: float = 0.5,
               m_lim_max_iter: int = 15,
               density_tolerance: float = 0.1,
               exposure_time_s: float = 1.0,
               log_dir: Optional[str] = None) -> Dict[str, Any]:
        """执行选星

        Args:
            image_path: FITS 图像路径
            ra, dec: 图像中心赤经赤纬(度)
            focal_length_mm: 焦距(mm)
            pixel_size_um: 像元尺寸(um)
            img_n_target: 图像侧目标星数(默认50)
            gaia_density_ratio: Gaia 面密度/图像面密度(默认1.5)
            gaia_query_radius_factor: Gaia 查询半径因子(默认0.55)
            m_lim_step: 极限星等迭代步长(默认0.5)
            m_lim_max_iter: 极限星等迭代最大次数(默认15)
            density_tolerance: 密度匹配容差(默认0.1)
            exposure_time_s: 曝光时间(s), 用于初始 m_cut 估计
            log_dir: 日志目录(可选)

        Returns:
            dict: {
                U: np.array(N×2)  图像侧向量组(角秒, 原点在图像中心),
                W: np.array(M×2)  Gaia 侧向量组(角秒, gnomonic 投影),
                meta: dict  元数据(s0/FOV/query_radius/m_lim/n_gaia/...),
            }
        """
        # ----------------------------------------------------------------
        # Step 1: 读取图像并检测星点
        # ----------------------------------------------------------------
        logger.info("Step 1: 读取图像 %s", image_path)
        img_data, img_w, img_h = _load_image(image_path)
        logger.info("  图像尺寸: %d×%d", img_w, img_h)

        img_x, img_y, img_flux, img_sat = _detect_stars(img_data, self._star_detector)
        if len(img_x) == 0:
            raise RuntimeError(f"未检测到星点: {image_path}")
        logger.info("  检测到星点: %d 颗 (饱和 %d, 正常 %d)",
                    len(img_x), int(img_sat.sum()), int((~img_sat).sum()))

        # ----------------------------------------------------------------
        # Step 2: V4.1 不对称图像侧选星
        # ----------------------------------------------------------------
        logger.info("Step 2: 图像侧选星 (img_n_target=%d)", img_n_target)
        sel_idx = _select_image_stars(img_x, img_y, img_flux, img_sat, img_n_target)
        N = len(sel_idx)
        if N < 2:
            raise RuntimeError(f"图像侧选星数 N={N} 过少，无法继续")

        # 像素尺度
        s0 = 206.265 * pixel_size_um / focal_length_mm
        cx, cy = img_w / 2.0, img_h / 2.0

        # 构建 U 向量组（角秒坐标，原点在图像中心，Y 轴向上）
        ux = (img_x[sel_idx] - cx) * s0
        uy = -(img_y[sel_idx] - cy) * s0
        U = np.column_stack([ux, uy]).astype(np.float64)
        logger.info("  U 向量组: %d×2 (s0=%.4f\"/px)", N, s0)

        # ----------------------------------------------------------------
        # Step 3: 调用 C++ DLL 执行 Gaia 密度匹配查询
        # ----------------------------------------------------------------
        logger.info("Step 3: Gaia 密度匹配查询")

        # Gaia 查询回调（Python 端实现，调用 GaiaClientPy.cone_search 返回星数）
        gaia_client = self._ensure_gaia()

        def _gaia_query_cb(ra_: float, dec_: float, radius: float,
                           mag_lim: float) -> int:
            try:
                ra_arr, _, _ = gaia_client.cone_search(ra_, dec_, radius, mag_lim)
                n = len(ra_arr)
                logger.debug("  Gaia 查询: ra=%.4f dec=%.4f r=%.4f° m=%.3f → %d 颗",
                             ra_, dec_, radius, mag_lim, n)
                return n
            except Exception as e:
                logger.error("  Gaia 查询异常: %s", e)
                return 0

        cb = GaiaQueryFuncC(_gaia_query_cb)

        log_path = _make_log_path(log_dir)
        params = StarSelectorParamsC()
        params.img_n_target             = int(img_n_target)
        params.gaia_density_ratio       = float(gaia_density_ratio)
        params.gaia_query_radius_factor = float(gaia_query_radius_factor)
        params.m_lim_step               = float(m_lim_step)
        params.m_lim_max_iter           = int(m_lim_max_iter)
        params.density_tolerance        = float(density_tolerance)
        params.focal_length_mm          = float(focal_length_mm)
        params.pixel_size_um            = float(pixel_size_um)
        params.img_width                = float(img_w)
        params.img_height               = float(img_h)
        params.center_ra                = float(ra)
        params.center_dec               = float(dec)
        params.n_img_bright             = int(N)
        params.exposure_time_s          = float(exposure_time_s)
        if log_path is not None:
            params.log_file_path = log_path.encode("utf-8")
        else:
            params.log_file_path = None

        result_c = StarSelectionResultC()
        ret = self._dll.ss_density_match(
            ctypes.byref(params), cb, ctypes.byref(result_c))
        if ret != 0:
            raise RuntimeError(f"ss_density_match 失败, ret={ret}")

        logger.info("  收敛: %s, m_lim=%.3f, n_gaia=%d, iters=%d, n_target=%d",
                    result_c.converged, result_c.m_lim_final,
                    result_c.n_gaia_final, result_c.m_lim_iterations,
                    result_c.n_target)

        # ----------------------------------------------------------------
        # Step 4: 用最终极限星等查询 Gaia 星表
        # ----------------------------------------------------------------
        logger.info("Step 4: 用 m_lim=%.3f 查询 Gaia 星表",
                    result_c.m_lim_final)
        cat_ra, cat_dec, cat_mag = gaia_client.cone_search(
            ra, dec, result_c.query_radius_deg, result_c.m_lim_final)
        if len(cat_ra) < 2:
            # 兜底：放宽到 mag=22
            logger.warning("  Gaia 返回 %d 颗，启用 mag=22 兜底查询", len(cat_ra))
            cat_ra, cat_dec, cat_mag = gaia_client.cone_search(
                ra, dec, result_c.query_radius_deg, 22.0)
        if len(cat_ra) < 2:
            raise RuntimeError(f"Gaia 星表查询星数过少: {len(cat_ra)}")

        # ----------------------------------------------------------------
        # Step 5: gnomonic 投影 + FOV 内过滤 + 按星等取最亮 N_target 颗
        # ----------------------------------------------------------------
        logger.info("Step 5: gnomonic 投影 + FOV 过滤")
        xi_all, eta_all, valid_all = gnomonic_forward(cat_ra, cat_dec, ra, dec)
        # 用图像尺寸的角秒半径作为 FOV 半宽
        fov_half_w = img_w / 2 * s0
        fov_half_h = img_h / 2 * s0
        in_fov = valid_all & (np.abs(xi_all) < fov_half_w) & (np.abs(eta_all) < fov_half_h)
        fov_idx = np.where(in_fov)[0]
        if len(fov_idx) < 2:
            # 放宽到 1.5×FOV
            logger.warning("  FOV 内仅 %d 颗，放宽到 1.5×FOV", len(fov_idx))
            in_fov = valid_all & (np.abs(xi_all) < fov_half_w * 1.5) & \
                     (np.abs(eta_all) < fov_half_h * 1.5)
            fov_idx = np.where(in_fov)[0]
        if len(fov_idx) < 2:
            raise RuntimeError(f"FOV 内 Gaia 星数过少: {len(fov_idx)}")

        # 按星等升序（最亮优先）取前 n_target 颗
        n_target = result_c.n_target
        fov_mag = cat_mag[fov_idx]
        sorted_order = np.argsort(fov_mag)
        sel_order = sorted_order[:min(n_target, len(sorted_order))]
        sel_idx_cat = fov_idx[sel_order]

        cat_ra_sel = cat_ra[sel_idx_cat]
        cat_dec_sel = cat_dec[sel_idx_cat]
        M = len(cat_ra_sel)

        # 重新投影选中的星
        xi_sel, eta_sel, _ = gnomonic_forward(cat_ra_sel, cat_dec_sel, ra, dec)
        W = np.column_stack([xi_sel, eta_sel]).astype(np.float64)
        logger.info("  W 向量组: %d×2 (FOV 内 %d, 取最亮 %d)",
                    M, len(fov_idx), M)

        # ----------------------------------------------------------------
        # Step 6: 组装结果
        # ----------------------------------------------------------------
        meta = {
            "s0": float(result_c.s0),
            "fov_diag_deg": float(result_c.fov_diag_deg),
            "query_radius_deg": float(result_c.query_radius_deg),
            "m_lim_final": float(result_c.m_lim_final),
            "n_gaia_final": int(result_c.n_gaia_final),
            "m_lim_iterations": int(result_c.m_lim_iterations),
            "converged": bool(result_c.converged),
            "n_target": int(result_c.n_target),
            "n_img_selected": int(N),
            "n_gaia_selected": int(M),
            "rho_img": float(result_c.rho_img),
            "rho_target": float(result_c.rho_target),
            "query_area_sqdeg": float(result_c.query_area_sqdeg),
            "img_area_sqdeg": float(result_c.img_area_sqdeg),
            "img_width": int(img_w),
            "img_height": int(img_h),
            "center_ra": float(ra),
            "center_dec": float(dec),
            "focal_length_mm": float(focal_length_mm),
            "pixel_size_um": float(pixel_size_um),
            "log_file": log_path,
        }
        logger.info("选星完成: U=%d×2, W=%d×2, converged=%s, m_lim=%.3f",
                    N, M, meta["converged"], meta["m_lim_final"])

        return {
            "U": U,
            "W": W,
            "meta": meta,
        }

    def close(self):
        if not self._closed:
            # 仅关闭内部创建的 Gaia 客户端，外部注入的不负责关闭
            if self._gaia is not None and not self._gaia_external:
                self._gaia.close()
            self._gaia = None
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

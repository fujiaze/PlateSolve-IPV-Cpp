"""
Iterative Refine Module - 第二步迭代精化模块

功能:
    - 极限星等二分法迭代
    - 矩形FOV裁剪
    - 三角形特征剪枝(剔除等边三角形)
    - 多边形近邻匹配
    - 分区域异常值剔除
    - 畸变模型拟合
    - 迭代优化(中心/比例尺/旋转角)

用途:
    在第一步粗匹配后,进行全星点精细匹配和畸变拟合
"""

import ctypes
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class IRInitialTransform:
    center_ra: float
    center_dec: float
    rotation_deg: float
    scale_arcsec_px: float
    flip_mode: int
    img_width: int
    img_height: int

@dataclass
class IRConfig:
    max_stars_triangle: int = 500
    tri_ratio_radius: float = 0.002
    tri_min_area: float = 100.0
    tri_max_ba_ratio: float = 0.95
    tri_equilateral_thresh: float = 0.1
    grid_size: int = 5
    outlier_angle_thresh: float = 1.57079632679
    outlier_mag_ratio: float = 3.0
    max_iterations: int = 5
    converge_thresh: float = 0.01
    match_threshold: float = 50.0

@dataclass
class IRRefineResult:
    final_ra: float
    final_dec: float
    final_rotation: float
    final_scale: float
    dist_a: List[float]
    dist_b: List[float]
    distortion_valid: int
    matched_count: int
    rms_x: float
    rms_y: float
    rms_total: float
    rms_arcsec: float
    iteration_count: int
    triangle_matches: int

class IterativeRefine:
    def __init__(self, dll_path: Optional[str] = None):
        if dll_path is None:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            dll_path = os.path.join(module_dir, "modules", "iterative_refine", "psm_iterative_refine.dll")
        
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"DLL not found: {dll_path}")
        
        self._dll = ctypes.CDLL(dll_path)
        self._setup_prototypes()
    
    def _setup_prototypes(self):
        self._dll.psm_iterative_refine.argtypes = [
            ctypes.POINTER(self._make_img_stars_struct()),
            ctypes.POINTER(self._make_cat_stars_struct()),
            ctypes.POINTER(self._make_init_transform_struct()),
            ctypes.POINTER(self._make_config_struct()),
            ctypes.POINTER(self._make_result_struct())
        ]
        self._dll.psm_iterative_refine.restype = ctypes.c_int
        
        self._dll.psm_free_refine_result.argtypes = [
            ctypes.POINTER(self._make_result_struct())
        ]
        self._dll.psm_free_refine_result.restype = None
    
    def _make_img_stars_struct(self):
        class IRImageStars(ctypes.Structure):
            _fields_ = [
                ("img_x", ctypes.POINTER(ctypes.c_double)),
                ("img_y", ctypes.POINTER(ctypes.c_double)),
                ("img_flux", ctypes.POINTER(ctypes.c_double)),
                ("img_count", ctypes.c_int),
                ("img_saturated", ctypes.POINTER(ctypes.c_int)),
                ("n_saturated", ctypes.c_int)
            ]
        return IRImageStars
    
    def _make_cat_stars_struct(self):
        class IRCatalogStars(ctypes.Structure):
            _fields_ = [
                ("cat_ra", ctypes.POINTER(ctypes.c_double)),
                ("cat_dec", ctypes.POINTER(ctypes.c_double)),
                ("cat_mag", ctypes.POINTER(ctypes.c_double)),
                ("cat_count", ctypes.c_int)
            ]
        return IRCatalogStars
    
    def _make_init_transform_struct(self):
        class IRInitialTransformC(ctypes.Structure):
            _fields_ = [
                ("center_ra", ctypes.c_double),
                ("center_dec", ctypes.c_double),
                ("rotation_deg", ctypes.c_double),
                ("scale_arcsec_px", ctypes.c_double),
                ("flip_mode", ctypes.c_int),
                ("img_width", ctypes.c_int),
                ("img_height", ctypes.c_int)
            ]
        return IRInitialTransformC
    
    def _make_config_struct(self):
        class IRConfigC(ctypes.Structure):
            _fields_ = [
                ("max_stars_triangle", ctypes.c_int),
                ("tri_ratio_radius", ctypes.c_double),
                ("tri_min_area", ctypes.c_double),
                ("tri_max_ba_ratio", ctypes.c_double),
                ("tri_equilateral_thresh", ctypes.c_double),
                ("grid_size", ctypes.c_int),
                ("outlier_angle_thresh", ctypes.c_double),
                ("outlier_mag_ratio", ctypes.c_double),
                ("max_iterations", ctypes.c_int),
                ("converge_thresh", ctypes.c_double),
                ("match_threshold", ctypes.c_double)
            ]
        return IRConfigC
    
    def _make_result_struct(self):
        class IRRefineResultC(ctypes.Structure):
            _fields_ = [
                ("final_ra", ctypes.c_double),
                ("final_dec", ctypes.c_double),
                ("final_rotation", ctypes.c_double),
                ("final_scale", ctypes.c_double),
                ("dist_a0", ctypes.c_double), ("dist_a1", ctypes.c_double),
                ("dist_a2", ctypes.c_double), ("dist_a3", ctypes.c_double),
                ("dist_a4", ctypes.c_double), ("dist_a5", ctypes.c_double),
                ("dist_b0", ctypes.c_double), ("dist_b1", ctypes.c_double),
                ("dist_b2", ctypes.c_double), ("dist_b3", ctypes.c_double),
                ("dist_b4", ctypes.c_double), ("dist_b5", ctypes.c_double),
                ("distortion_valid", ctypes.c_int),
                ("matched_count", ctypes.c_int),
                ("rms_x", ctypes.c_double),
                ("rms_y", ctypes.c_double),
                ("rms_total", ctypes.c_double),
                ("rms_arcsec", ctypes.c_double),
                ("iteration_count", ctypes.c_int),
                ("triangle_matches", ctypes.c_int),
                ("img_indices", ctypes.POINTER(ctypes.c_int)),
                ("cat_indices", ctypes.POINTER(ctypes.c_int)),
                ("residual_x", ctypes.POINTER(ctypes.c_double)),
                ("residual_y", ctypes.POINTER(ctypes.c_double))
            ]
        return IRRefineResultC
    
    def refine(self,
               img_x: np.ndarray,
               img_y: np.ndarray,
               img_flux: np.ndarray,
               img_saturated: np.ndarray,
               cat_ra: np.ndarray,
               cat_dec: np.ndarray,
               cat_mag: np.ndarray,
               init_transform: IRInitialTransform,
               config: Optional[IRConfig] = None) -> IRRefineResult:
        """
        执行迭代精化
        
        参数:
            img_x, img_y: 图像星点坐标
            img_flux: 图像星点flux
            img_saturated: 饱和标记 (1=饱和, 0=正常)
            cat_ra, cat_dec, cat_mag: 星表星点
            init_transform: 第一步输出的初始变换
            config: 配置参数
        
        返回:
            IRRefineResult: 精化结果
        """
        if config is None:
            config = IRConfig()
        
        n_img = len(img_x)
        n_cat = len(cat_ra)
        n_sat = int(np.sum(img_saturated))
        
        img_x_arr = np.ascontiguousarray(img_x, dtype=np.float64)
        img_y_arr = np.ascontiguousarray(img_y, dtype=np.float64)
        img_flux_arr = np.ascontiguousarray(img_flux, dtype=np.float64)
        img_sat_arr = np.ascontiguousarray(img_saturated, dtype=np.int32)
        
        cat_ra_arr = np.ascontiguousarray(cat_ra, dtype=np.float64)
        cat_dec_arr = np.ascontiguousarray(cat_dec, dtype=np.float64)
        cat_mag_arr = np.ascontiguousarray(cat_mag, dtype=np.float64)
        
        IRImageStars = self._make_img_stars_struct()
        IRCatalogStars = self._make_cat_stars_struct()
        IRInitialTransformC = self._make_init_transform_struct()
        IRConfigC = self._make_config_struct()
        IRRefineResultC = self._make_result_struct()
        
        img_stars = IRImageStars(
            img_x=img_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            img_y=img_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            img_flux=img_flux_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            img_count=n_img,
            img_saturated=img_sat_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n_saturated=n_sat
        )
        
        cat_stars = IRCatalogStars(
            cat_ra=cat_ra_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cat_dec=cat_dec_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cat_mag=cat_mag_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            cat_count=n_cat
        )
        
        init_c = IRInitialTransformC(
            center_ra=init_transform.center_ra,
            center_dec=init_transform.center_dec,
            rotation_deg=init_transform.rotation_deg,
            scale_arcsec_px=init_transform.scale_arcsec_px,
            flip_mode=init_transform.flip_mode,
            img_width=init_transform.img_width,
            img_height=init_transform.img_height
        )
        
        config_c = IRConfigC(
            max_stars_triangle=config.max_stars_triangle,
            tri_ratio_radius=config.tri_ratio_radius,
            tri_min_area=config.tri_min_area,
            tri_max_ba_ratio=config.tri_max_ba_ratio,
            tri_equilateral_thresh=config.tri_equilateral_thresh,
            grid_size=config.grid_size,
            outlier_angle_thresh=config.outlier_angle_thresh,
            outlier_mag_ratio=config.outlier_mag_ratio,
            max_iterations=config.max_iterations,
            converge_thresh=config.converge_thresh,
            match_threshold=config.match_threshold
        )
        
        result_c = IRRefineResultC()
        
        ret = self._dll.psm_iterative_refine(
            ctypes.byref(img_stars),
            ctypes.byref(cat_stars),
            ctypes.byref(init_c),
            ctypes.byref(config_c),
            ctypes.byref(result_c)
        )
        
        if ret != 0:
            raise RuntimeError(f"psm_iterative_refine failed with code {ret}")
        
        result = IRRefineResult(
            final_ra=result_c.final_ra,
            final_dec=result_c.final_dec,
            final_rotation=result_c.final_rotation,
            final_scale=result_c.final_scale,
            dist_a=[result_c.dist_a0, result_c.dist_a1, result_c.dist_a2,
                    result_c.dist_a3, result_c.dist_a4, result_c.dist_a5],
            dist_b=[result_c.dist_b0, result_c.dist_b1, result_c.dist_b2,
                    result_c.dist_b3, result_c.dist_b4, result_c.dist_b5],
            distortion_valid=result_c.distortion_valid,
            matched_count=result_c.matched_count,
            rms_x=result_c.rms_x,
            rms_y=result_c.rms_y,
            rms_total=result_c.rms_total,
            rms_arcsec=result_c.rms_arcsec,
            iteration_count=result_c.iteration_count,
            triangle_matches=result_c.triangle_matches
        )
        
        self._dll.psm_free_refine_result(ctypes.byref(result_c))
        
        return result


def estimate_mag_limit(focal_mm: float, exposure_s: float) -> float:
    """
    估算极限星等
    
    公式: mag = 6 + 1.5*log10(focal_mm) + 2*log10(exposure_s)
    
    参数:
        focal_mm: 焦距(mm)
        exposure_s: 曝光时间(秒)
    
    返回:
        估算的极限星等
    """
    import math
    return 6.0 + 1.5 * math.log10(focal_mm) + 2.0 * math.log10(exposure_s)


def bisection_mag_limit(gaia_client,
                        center_ra: float,
                        center_dec: float,
                        radius_deg: float,
                        target_count: int,
                        mag_low: float = 6.0,
                        mag_high: float = 22.0,
                        tolerance: float = 0.1) -> Tuple[float, int]:
    """
    二分法迭代极限星等
    
    目标: 使Gaia星数在 [target_count, 1.2*target_count] 范围内
    
    参数:
        gaia_client: Gaia客户端
        center_ra, center_dec: 中心坐标(度)
        radius_deg: 查询半径(度)
        target_count: 目标星数
        mag_low, mag_high: 搜索范围
        tolerance: 容差
    
    返回:
        (最终极限星等, 最终星数)
    """
    target_high = int(target_count * 1.2)
    
    while (mag_high - mag_low) > tolerance:
        mid = (mag_low + mag_high) / 2.0
        
        stars = gaia_client.cone_search(center_ra, center_dec, radius_deg, mag_limit=mid)
        count = len(stars)
        
        if count < target_count:
            mag_low = mid
        elif count > target_high:
            mag_high = mid
        else:
            return mid, count
    
    final_mag = (mag_low + mag_high) / 2.0
    stars = gaia_client.cone_search(center_ra, center_dec, radius_deg, mag_limit=final_mag)
    return final_mag, len(stars)

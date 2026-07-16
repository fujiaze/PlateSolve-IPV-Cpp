# -*- coding: utf-8 -*-
"""
PlateSolve - 天文图像Plate Solving模块

整合两步解析流程:
1. 第一步: 粗匹配（饱和星优先策略）
2. 第二步: SIP畸变拟合（5阶多项式）
"""
import os
import sys
import ctypes
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

_mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(_mingw_bin):
    os.environ["PATH"] = _mingw_bin + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_mingw_bin)
    except OSError:
        pass

_project_root = r"F:\Astro dev\Astro CS Normalization Database"
_dll_path = os.path.join(_project_root, "lib", "plate_solve", "plate_solve.dll")

if os.path.exists(_dll_path):
    try:
        os.add_dll_directory(os.path.dirname(_dll_path))
    except OSError:
        pass
    _dll = ctypes.CDLL(_dll_path)
else:
    _dll = None

@dataclass
class PlateSolveConfig:
    use_saturated_priority: int = 1
    n_img_bright: int = 500
    n_cat_bright: int = 600
    max_match_dist_px: float = 25.0
    max_iterations: int = 5
    match_threshold: float = 10.0
    sip_order: int = 5
    converge_thresh: float = 0.01

@dataclass
class PlateSolveResult:
    center_ra: float
    center_dec: float
    rotation_deg: float
    scale_arcsec_px: float
    flip_mode: int
    matched_count: int
    rms_px: float
    step1_time_sec: float
    step2_time_sec: float
    wcs: dict
    sip_valid: bool
    sip: Optional[np.ndarray] = None

class _PSolveImageData(ctypes.Structure):
    _fields_ = [
        ("focal_length_mm", ctypes.c_double),
        ("pixel_size_um", ctypes.c_double),
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("has_coords", ctypes.c_int),
        ("exposure_time_s", ctypes.c_double),
        ("scale_arcsec_px", ctypes.c_double),
    ]

class _PSolveConfig(ctypes.Structure):
    _fields_ = [
        ("use_saturated_priority", ctypes.c_int),
        ("n_img_bright", ctypes.c_int),
        ("n_cat_bright", ctypes.c_int),
        ("max_match_dist_px", ctypes.c_double),
        ("max_iterations", ctypes.c_int),
        ("match_threshold", ctypes.c_double),
        ("sip_order", ctypes.c_int),
        ("converge_thresh", ctypes.c_double),
    ]

class _PSolveSIPCoeffs(ctypes.Structure):
    _fields_ = [
        ("A", (ctypes.c_double * 6) * 6),
        ("B", (ctypes.c_double * 6) * 6),
        ("AP", (ctypes.c_double * 6) * 6),
        ("BP", (ctypes.c_double * 6) * 6),
        ("order", ctypes.c_int),
        ("valid", ctypes.c_int),
    ]

class _PSolveWCS(ctypes.Structure):
    _fields_ = [
        ("crpix1", ctypes.c_double), ("crpix2", ctypes.c_double),
        ("crval1", ctypes.c_double), ("crval2", ctypes.c_double),
        ("cd1_1", ctypes.c_double), ("cd1_2", ctypes.c_double),
        ("cd2_1", ctypes.c_double), ("cd2_2", ctypes.c_double),
        ("cdelt1", ctypes.c_double), ("cdelt2", ctypes.c_double),
        ("ctype1", ctypes.c_char * 32),
        ("ctype2", ctypes.c_char * 32),
        ("radesys", ctypes.c_char * 32),
        ("equinox", ctypes.c_double),
    ]

class _PSolveResult(ctypes.Structure):
    _fields_ = [
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("rotation_deg", ctypes.c_double),
        ("scale_arcsec_px", ctypes.c_double),
        ("flip_mode", ctypes.c_int),
        ("matched_count", ctypes.c_int),
        ("rms_px", ctypes.c_double),
        ("step1_time_sec", ctypes.c_double),
        ("step2_time_sec", ctypes.c_double),
        ("wcs", _PSolveWCS),
        ("sip", _PSolveSIPCoeffs),
        ("sip_valid", ctypes.c_int),
    ]

class PlateSolve:
    """
    PlateSolve - 天文图像Plate Solving
    
    整合两步解析流程:
    1. 第一步: 粗匹配（饱和星优先策略）
    2. 第二步: SIP畸变拟合（5阶多项式）
    
    使用示例:
    ```python
    solver = PlateSolve(gaia_data_dir="/path/to/GaiaDR3")
    
    result = solver.solve(
        img_x=img_x, img_y=img_y, img_flux=img_flux,
        img_saturated=saturated, n_saturated=n_sat,
        center_ra=266.0, center_dec=-28.0,
        focal_length_mm=200.0, pixel_size_um=6.0,
        width=4500, height=3000
    )
    
    print(f"中心: RA={result.center_ra:.6f}, Dec={result.center_dec:.6f}")
    print(f"RMS: {result.rms_px:.3f} px")
    ```
    """
    
    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        """
        初始化PlateSolve
        
        参数:
            gaia_data_dir: Gaia数据库目录
            db_type: 数据库类型 (0=自动, 1=DR3, 2=DR3SP)
        """
        if _dll is None:
            raise RuntimeError(f"plate_solve.dll not found: {_dll_path}")
        
        self._dll = _dll
        self._setup_api()
        
        gaia_dir_bytes = gaia_data_dir.encode("utf-8")
        if db_type == 0:
            self._handle = self._dll.psolve_create(gaia_dir_bytes)
        else:
            self._handle = self._dll.psolve_create_ex(gaia_dir_bytes, db_type)
        
        if not self._handle:
            raise RuntimeError(f"Failed to create plate solver: {gaia_data_dir}")
        
        self._gaia_dir = gaia_data_dir
    
    def _setup_api(self):
        self._dll.psolve_create.argtypes = [ctypes.c_char_p]
        self._dll.psolve_create.restype = ctypes.c_void_p
        self._dll.psolve_create_ex.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._dll.psolve_create_ex.restype = ctypes.c_void_p
        self._dll.psolve_destroy.argtypes = [ctypes.c_void_p]
        self._dll.psolve_destroy.restype = None
        
        self._dll.psolve_solve.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_PSolveImageData),
            ctypes.POINTER(_PSolveConfig),
            ctypes.POINTER(_PSolveResult),
        ]
        self._dll.psolve_solve.restype = ctypes.c_int
        
        self._dll.psolve_solve_with_image.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(_PSolveImageData),
            ctypes.POINTER(_PSolveConfig),
            ctypes.POINTER(_PSolveResult),
        ]
        self._dll.psolve_solve_with_image.restype = ctypes.c_int
        
        self._dll.psolve_solve_with_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(_PSolveConfig),
            ctypes.POINTER(_PSolveResult),
        ]
        self._dll.psolve_solve_with_file.restype = ctypes.c_int
        
        self._dll.psolve_free_result.argtypes = [ctypes.POINTER(_PSolveResult)]
        self._dll.psolve_free_result.restype = None
    
    def solve(
        self,
        img_x: np.ndarray,
        img_y: np.ndarray,
        img_flux: np.ndarray,
        img_saturated: np.ndarray,
        n_saturated: int,
        center_ra: float,
        center_dec: float,
        focal_length_mm: float,
        pixel_size_um: float,
        width: int,
        height: int,
        config: Optional[PlateSolveConfig] = None,
        scale_arcsec_px: float = 0.0,
    ) -> PlateSolveResult:
        """
        执行两步Plate Solving
        
        参数:
            img_x: 图像星点X坐标（相对于CRPIX）
            img_y: 图像星点Y坐标（相对于CRPIX）
            img_flux: 图像星点flux
            img_saturated: 饱和标记数组
            n_saturated: 饱和星数量
            center_ra: 初始中心RA（度）
            center_dec: 初始中心Dec（度）
            focal_length_mm: 焦距（mm）
            pixel_size_um: 像元尺寸（um）
            width: 图像宽度
            height: 图像高度
            config: 可选配置
            scale_arcsec_px: 可选比例尺（角秒/像素），若提供则覆盖计算值
        
        返回:
            PlateSolveResult对象
        """
        if config is None:
            config = PlateSolveConfig()
        
        img_x = np.ascontiguousarray(img_x, dtype=np.float64)
        img_y = np.ascontiguousarray(img_y, dtype=np.float64)
        img_flux = np.ascontiguousarray(img_flux, dtype=np.float64)
        img_saturated = np.ascontiguousarray(img_saturated, dtype=np.int32)
        
        img_count = len(img_x)
        
        img_data = _PSolveImageData()
        img_data.focal_length_mm = focal_length_mm
        img_data.pixel_size_um = pixel_size_um
        img_data.center_ra = center_ra
        img_data.center_dec = center_dec
        img_data.width = width
        img_data.height = height
        img_data.has_coords = 1
        img_data.exposure_time_s = 0.0
        img_data.scale_arcsec_px = scale_arcsec_px
        
        cfg = _PSolveConfig()
        cfg.use_saturated_priority = config.use_saturated_priority
        cfg.n_img_bright = config.n_img_bright
        cfg.n_cat_bright = config.n_cat_bright
        cfg.max_match_dist_px = config.max_match_dist_px
        cfg.max_iterations = config.max_iterations
        cfg.match_threshold = config.match_threshold
        cfg.sip_order = config.sip_order
        cfg.converge_thresh = config.converge_thresh
        
        result = _PSolveResult()
        
        rc = self._dll.psolve_solve(
            self._handle,
            img_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            img_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            img_flux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            img_saturated.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            img_count, n_saturated,
            ctypes.byref(img_data),
            ctypes.byref(cfg),
            ctypes.byref(result),
        )
        
        if rc != 0:
            raise RuntimeError(f"psolve_solve failed with code {rc}")
        
        wcs_dict = {
            "crpix1": result.wcs.crpix1,
            "crpix2": result.wcs.crpix2,
            "crval1": result.wcs.crval1,
            "crval2": result.wcs.crval2,
            "cd1_1": result.wcs.cd1_1,
            "cd1_2": result.wcs.cd1_2,
            "cd2_1": result.wcs.cd2_1,
            "cd2_2": result.wcs.cd2_2,
            "ctype1": result.wcs.ctype1.decode("utf-8"),
            "ctype2": result.wcs.ctype2.decode("utf-8"),
            "radesys": result.wcs.radesys.decode("utf-8"),
            "equinox": result.wcs.equinox,
        }
        
        sip_arr = None
        if result.sip_valid:
            sip_arr = np.zeros((4, 6, 6), dtype=np.float64)
            for i in range(6):
                for j in range(6):
                    sip_arr[0, i, j] = result.sip.A[i][j]
                    sip_arr[1, i, j] = result.sip.B[i][j]
                    sip_arr[2, i, j] = result.sip.AP[i][j]
                    sip_arr[3, i, j] = result.sip.BP[i][j]
        
        return PlateSolveResult(
            center_ra=result.center_ra,
            center_dec=result.center_dec,
            rotation_deg=result.rotation_deg,
            scale_arcsec_px=result.scale_arcsec_px,
            flip_mode=result.flip_mode,
            matched_count=result.matched_count,
            rms_px=result.rms_px,
            step1_time_sec=result.step1_time_sec,
            step2_time_sec=result.step2_time_sec,
            wcs=wcs_dict,
            sip_valid=bool(result.sip_valid),
            sip=sip_arr,
        )
    
    def solve_with_image(
        self,
        image: np.ndarray,
        center_ra: float,
        center_dec: float,
        focal_length_mm: float,
        pixel_size_um: float,
        config: Optional[PlateSolveConfig] = None,
        scale_arcsec_px: float = 0.0,
    ) -> PlateSolveResult:
        """
        执行Plate Solving（C++端自动检测星点）
        
        参数:
            image: uint16图像数组
            center_ra: 初始中心RA（度）
            center_dec: 初始中心Dec（度）
            focal_length_mm: 焦距（mm）
            pixel_size_um: 像元尺寸（um）
            config: 可选配置
            scale_arcsec_px: 可选比例尺
        
        返回:
            PlateSolveResult对象
        """
        if config is None:
            config = PlateSolveConfig()
        
        image = np.ascontiguousarray(image, dtype=np.uint16)
        height, width = image.shape[:2]
        
        img_data = _PSolveImageData()
        img_data.focal_length_mm = focal_length_mm
        img_data.pixel_size_um = pixel_size_um
        img_data.center_ra = center_ra
        img_data.center_dec = center_dec
        img_data.width = width
        img_data.height = height
        img_data.has_coords = 1
        img_data.exposure_time_s = 0.0
        img_data.scale_arcsec_px = scale_arcsec_px
        
        cfg = _PSolveConfig()
        cfg.use_saturated_priority = config.use_saturated_priority
        cfg.n_img_bright = config.n_img_bright
        cfg.n_cat_bright = config.n_cat_bright
        cfg.max_match_dist_px = config.max_match_dist_px
        cfg.max_iterations = config.max_iterations
        cfg.match_threshold = config.match_threshold
        cfg.sip_order = config.sip_order
        cfg.converge_thresh = config.converge_thresh
        
        result = _PSolveResult()
        
        rc = self._dll.psolve_solve_with_image(
            self._handle,
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            width, height,
            ctypes.byref(img_data),
            ctypes.byref(cfg),
            ctypes.byref(result),
        )
        
        if rc != 0:
            raise RuntimeError(f"psolve_solve_with_image failed with code {rc}")
        
        wcs_dict = {
            "crpix1": result.wcs.crpix1,
            "crpix2": result.wcs.crpix2,
            "crval1": result.wcs.crval1,
            "crval2": result.wcs.crval2,
            "cd1_1": result.wcs.cd1_1,
            "cd1_2": result.wcs.cd1_2,
            "cd2_1": result.wcs.cd2_1,
            "cd2_2": result.wcs.cd2_2,
            "ctype1": result.wcs.ctype1.decode("utf-8"),
            "ctype2": result.wcs.ctype2.decode("utf-8"),
            "radesys": result.wcs.radesys.decode("utf-8"),
            "equinox": result.wcs.equinox,
        }
        
        sip_arr = None
        if result.sip_valid:
            sip_arr = np.zeros((4, 6, 6), dtype=np.float64)
            for i in range(6):
                for j in range(6):
                    sip_arr[0, i, j] = result.sip.A[i][j]
                    sip_arr[1, i, j] = result.sip.B[i][j]
                    sip_arr[2, i, j] = result.sip.AP[i][j]
                    sip_arr[3, i, j] = result.sip.BP[i][j]
        
        return PlateSolveResult(
            center_ra=result.center_ra,
            center_dec=result.center_dec,
            rotation_deg=result.rotation_deg,
            scale_arcsec_px=result.scale_arcsec_px,
            flip_mode=result.flip_mode,
            matched_count=result.matched_count,
            rms_px=result.rms_px,
            step1_time_sec=result.step1_time_sec,
            step2_time_sec=result.step2_time_sec,
            wcs=wcs_dict,
            sip_valid=bool(result.sip_valid),
            sip=sip_arr,
        )
    
    def solve_with_file(
        self,
        file_path: str,
        config: Optional[PlateSolveConfig] = None,
    ) -> PlateSolveResult:
        """
        执行Plate Solving（C端读取文件，自动解析FITS头）
        
        参数:
            file_path: FITS文件路径
            config: 可选配置
        
        返回:
            PlateSolveResult对象
        """
        if config is None:
            config = PlateSolveConfig()
        
        cfg = _PSolveConfig()
        cfg.use_saturated_priority = config.use_saturated_priority
        cfg.n_img_bright = config.n_img_bright
        cfg.n_cat_bright = config.n_cat_bright
        cfg.max_match_dist_px = config.max_match_dist_px
        cfg.max_iterations = config.max_iterations
        cfg.match_threshold = config.match_threshold
        cfg.sip_order = config.sip_order
        cfg.converge_thresh = config.converge_thresh
        
        result = _PSolveResult()
        
        file_path_bytes = file_path.encode('utf-8')
        
        rc = self._dll.psolve_solve_with_file(
            self._handle,
            file_path_bytes,
            ctypes.byref(cfg),
            ctypes.byref(result),
        )
        
        if rc != 0:
            raise RuntimeError(f"psolve_solve_with_file failed with code {rc}")
        
        wcs_dict = {
            "crpix1": result.wcs.crpix1,
            "crpix2": result.wcs.crpix2,
            "crval1": result.wcs.crval1,
            "crval2": result.wcs.crval2,
            "cd1_1": result.wcs.cd1_1,
            "cd1_2": result.wcs.cd1_2,
            "cd2_1": result.wcs.cd2_1,
            "cd2_2": result.wcs.cd2_2,
            "ctype1": result.wcs.ctype1.decode("utf-8"),
            "ctype2": result.wcs.ctype2.decode("utf-8"),
            "radesys": result.wcs.radesys.decode("utf-8"),
            "equinox": result.wcs.equinox,
        }
        
        sip_arr = None
        if result.sip_valid:
            sip_arr = np.zeros((4, 6, 6), dtype=np.float64)
            for i in range(6):
                for j in range(6):
                    sip_arr[0, i, j] = result.sip.A[i][j]
                    sip_arr[1, i, j] = result.sip.B[i][j]
                    sip_arr[2, i, j] = result.sip.AP[i][j]
                    sip_arr[3, i, j] = result.sip.BP[i][j]
        
        return PlateSolveResult(
            center_ra=result.center_ra,
            center_dec=result.center_dec,
            rotation_deg=result.rotation_deg,
            scale_arcsec_px=result.scale_arcsec_px,
            flip_mode=result.flip_mode,
            matched_count=result.matched_count,
            rms_px=result.rms_px,
            step1_time_sec=result.step1_time_sec,
            step2_time_sec=result.step2_time_sec,
            wcs=wcs_dict,
            sip_valid=bool(result.sip_valid),
            sip=sip_arr,
        )
    
    def solve_step1_initial_wcs(
        self,
        img_x: np.ndarray,
        img_y: np.ndarray,
        img_flux: np.ndarray,
        img_saturated: np.ndarray,
        center_ra: float,
        center_dec: float,
        focal_length_mm: float,
        pixel_size_um: float,
        width: int,
        height: int,
        use_cpp: bool = True,
    ):
        """使用新的initial_wcs模块执行Step1（初始WCS生成）

        参数:
            img_x, img_y: 图像星点坐标
            img_flux: 图像星点通量
            img_saturated: 饱和标志 0/1
            center_ra, center_dec: 初始中心坐标（度）
            focal_length_mm: 焦距（mm）
            pixel_size_um: 像元尺寸（um）
            width, height: 图像尺寸
            use_cpp: True=使用C++ DLL, False=使用Python原型

        返回:
            InitialWCSResult 或 None(失败时)
        """
        if use_cpp:
            from .initial_wcs_ctypes import solve as cpp_solve
            return cpp_solve(
                img_x, img_y, img_flux, img_saturated,
                center_ra, center_dec,
                focal_length_mm, pixel_size_um,
                width, height,
                self._gaia_dir, 0
            )
        else:
            from .initial_wcs import InitialWCS
            solver = InitialWCS(self._gaia_dir)
            return solver.solve(
                img_x, img_y, img_flux, img_saturated,
                center_ra, center_dec,
                focal_length_mm, pixel_size_um,
                width, height
            )

    def close(self):
        """释放资源"""
        if self._handle:
            self._dll.psolve_destroy(self._handle)
            self._handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        self.close()

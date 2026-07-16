"""
初始WCS生成模块 Python绑定
功能: 通过ctypes调用psm_initial_wcs.dll，实现PlateSolve第一步粗匹配
用途: 替代旧的star_alignment模块，使用饱和星优先三角匹配+4种翻转模式测试
"""

import ctypes
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

_dll = None

class InitialWCSResultC(ctypes.Structure):
    _fields_ = [
        ("center_ra", ctypes.c_double),
        ("center_dec", ctypes.c_double),
        ("rotation_deg", ctypes.c_double),
        ("scale_arcsec_px", ctypes.c_double),
        ("flip_mode", ctypes.c_int),
        ("affine", ctypes.c_double * 6),
        ("matched_count", ctypes.c_int),
        ("rms_px", ctypes.c_double),
        ("rms_arcsec", ctypes.c_double),
    ]

@dataclass
class InitialWCSResult:
    center_ra: float
    center_dec: float
    rotation_deg: float
    scale_arcsec_px: float
    flip_mode: int
    affine: tuple
    matched_count: int
    rms_px: float
    rms_arcsec: float

def _load_dll():
    global _dll
    if _dll is not None:
        return _dll

    dll_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "modules", "initial_wcs")
    dll_path = os.path.join(dll_dir, "psm_initial_wcs.dll")

    if not os.path.exists(dll_path):
        raise FileNotFoundError(f"找不到DLL: {dll_path}")

    _dll = ctypes.CDLL(dll_path)
    _dll.psm_initial_wcs_solve.argtypes = [
        ctypes.POINTER(ctypes.c_double),  # img_x
        ctypes.POINTER(ctypes.c_double),  # img_y
        ctypes.POINTER(ctypes.c_double),  # img_flux
        ctypes.POINTER(ctypes.c_int),     # img_saturated
        ctypes.c_int,                     # n_stars
        ctypes.c_double,                  # center_ra
        ctypes.c_double,                  # center_dec
        ctypes.c_double,                  # focal_length_mm
        ctypes.c_double,                  # pixel_size_um
        ctypes.c_int,                     # width
        ctypes.c_int,                     # height
        ctypes.c_char_p,                  # gaia_db_path
        ctypes.c_int,                     # db_type
        ctypes.POINTER(InitialWCSResultC) # result
    ]
    _dll.psm_initial_wcs_solve.restype = ctypes.c_int

    _dll.psm_initial_wcs_free_result.argtypes = [ctypes.POINTER(InitialWCSResultC)]
    _dll.psm_initial_wcs_free_result.restype = None

    return _dll

def solve(img_x, img_y, img_flux, img_saturated,
          center_ra, center_dec,
          focal_length_mm, pixel_size_um,
          width, height,
          gaia_db_path, db_type=0):
    """调用C++ DLL执行初始WCS生成

    参数:
        img_x, img_y: 图像星点坐标 (numpy array)
        img_flux: 图像星点通量 (numpy array)
        img_saturated: 饱和标志 0/1 (numpy array)
        center_ra, center_dec: 初始中心坐标 (度)
        focal_length_mm: 焦距 (mm)
        pixel_size_um: 像元尺寸 (um)
        width, height: 图像尺寸
        gaia_db_path: Gaia数据库路径
        db_type: 数据库类型 (0=Auto, 1=DR3, 2=DR3SP)

    返回:
        InitialWCSResult 或 None(失败时)
    """
    dll = _load_dll()

    n_stars = len(img_x)
    c_x = np.ascontiguousarray(img_x, dtype=np.float64)
    c_y = np.ascontiguousarray(img_y, dtype=np.float64)
    c_flux = np.ascontiguousarray(img_flux, dtype=np.float64)
    c_sat = np.ascontiguousarray(img_saturated, dtype=np.int32)

    result_c = InitialWCSResultC()
    rc = dll.psm_initial_wcs_solve(
        c_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_flux.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        c_sat.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        n_stars,
        center_ra, center_dec,
        focal_length_mm, pixel_size_um,
        width, height,
        gaia_db_path.encode('utf-8'),
        db_type,
        ctypes.byref(result_c)
    )

    if rc != 0:
        return None

    return InitialWCSResult(
        center_ra=result_c.center_ra,
        center_dec=result_c.center_dec,
        rotation_deg=result_c.rotation_deg,
        scale_arcsec_px=result_c.scale_arcsec_px,
        flip_mode=result_c.flip_mode,
        affine=tuple(result_c.affine[i] for i in range(6)),
        matched_count=result_c.matched_count,
        rms_px=result_c.rms_px,
        rms_arcsec=result_c.rms_arcsec
    )

"""
特征匹配模块 Python绑定
基于flux排序的星点筛选和siril风格三角匹配
"""

import ctypes
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

_dll_path = os.path.join(os.path.dirname(__file__), "modules", "feature_match", "feature_match.dll")
if not os.path.exists(_dll_path):
    _dll_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules", "feature_match", "feature_match.dll")

_dll = ctypes.CDLL(_dll_path)

class _PSMFeatureStar(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("flux", ctypes.c_double),
        ("orig_idx", ctypes.c_int),
    ]

class _PSMFeatureTriangle(ctypes.Structure):
    _fields_ = [
        ("a_idx", ctypes.c_int),
        ("b_idx", ctypes.c_int),
        ("c_idx", ctypes.c_int),
        ("ba_ratio", ctypes.c_double),
        ("ca_ratio", ctypes.c_double),
        ("side_a_angle", ctypes.c_double),
        ("side_a_length", ctypes.c_double),
    ]

class _PSMMatchPair(ctypes.Structure):
    _fields_ = [
        ("img_idx", ctypes.c_int),
        ("cat_idx", ctypes.c_int),
        ("votes", ctypes.c_int),
        ("flux_ratio", ctypes.c_double),
    ]

class _PSMMatchResult(ctypes.Structure):
    _fields_ = [
        ("pairs", ctypes.POINTER(_PSMMatchPair)),
        ("pair_count", ctypes.c_int),
        ("tri_match_count", ctypes.c_int),
        ("total_votes", ctypes.c_int),
    ]

class _PSMFeatureMatchConfig(ctypes.Structure):
    _fields_ = [
        ("img_max_stars", ctypes.c_int),
        ("cat_max_stars", ctypes.c_int),
        ("nbright", ctypes.c_int),
        ("match_radius", ctypes.c_double),
        ("min_scale", ctypes.c_double),
        ("max_scale", ctypes.c_double),
        ("min_side_length", ctypes.c_double),
        ("max_ba_ratio", ctypes.c_double),
        ("min_votes", ctypes.c_int),
        ("use_flux_enhance", ctypes.c_int),
        ("flux_enhance_threshold", ctypes.c_double),
    ]

_dll.psm_feature_match.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(_PSMFeatureMatchConfig),
    ctypes.POINTER(_PSMMatchResult),
]
_dll.psm_feature_match.restype = ctypes.c_int

_dll.psm_free_match_result.argtypes = [ctypes.POINTER(_PSMMatchResult)]
_dll.psm_free_match_result.restype = None

@dataclass
class MatchPair:
    img_idx: int
    cat_idx: int
    votes: int
    flux_ratio: float

@dataclass
class MatchResult:
    pairs: List[MatchPair]
    tri_match_count: int
    total_votes: int

@dataclass
class FeatureMatchConfig:
    img_max_stars: int = 2000
    cat_max_stars: int = 5000
    nbright: int = 60
    match_radius: float = 0.002
    min_scale: float = 0.7
    max_scale: float = 1.3
    min_side_length: float = 5.0
    max_ba_ratio: float = 0.9
    min_votes: int = 2
    use_flux_enhance: bool = False
    flux_enhance_threshold: float = 0.5

def feature_match(
    img_x: np.ndarray, img_y: np.ndarray, img_flux: np.ndarray,
    cat_x: np.ndarray, cat_y: np.ndarray, cat_flux: np.ndarray,
    config: Optional[FeatureMatchConfig] = None
) -> MatchResult:
    """
    执行特征匹配
    
    参数:
        img_x, img_y, img_flux: 图像星点坐标和flux
        cat_x, cat_y, cat_flux: 星表星点坐标和flux
        config: 配置参数
        
    返回:
        MatchResult: 匹配结果
    """
    if config is None:
        config = FeatureMatchConfig()
    
    n_img = len(img_x)
    n_cat = len(cat_x)
    
    img_x_arr = np.ascontiguousarray(img_x, dtype=np.float64)
    img_y_arr = np.ascontiguousarray(img_y, dtype=np.float64)
    img_flux_arr = np.ascontiguousarray(img_flux, dtype=np.float64)
    cat_x_arr = np.ascontiguousarray(cat_x, dtype=np.float64)
    cat_y_arr = np.ascontiguousarray(cat_y, dtype=np.float64)
    cat_flux_arr = np.ascontiguousarray(cat_flux, dtype=np.float64)
    
    c_config = _PSMFeatureMatchConfig(
        img_max_stars=config.img_max_stars,
        cat_max_stars=config.cat_max_stars,
        nbright=config.nbright,
        match_radius=config.match_radius,
        min_scale=config.min_scale,
        max_scale=config.max_scale,
        min_side_length=config.min_side_length,
        max_ba_ratio=config.max_ba_ratio,
        min_votes=config.min_votes,
        use_flux_enhance=1 if config.use_flux_enhance else 0,
        flux_enhance_threshold=config.flux_enhance_threshold,
    )
    
    c_result = _PSMMatchResult()
    
    ret = _dll.psm_feature_match(
        img_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        img_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        img_flux_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_img,
        cat_x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cat_y_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        cat_flux_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_cat,
        ctypes.byref(c_config),
        ctypes.byref(c_result),
    )
    
    if ret != 0:
        _dll.psm_free_match_result(ctypes.byref(c_result))
        return MatchResult(pairs=[], tri_match_count=0, total_votes=0)
    
    pairs = []
    for i in range(c_result.pair_count):
        p = c_result.pairs[i]
        pairs.append(MatchPair(
            img_idx=p.img_idx,
            cat_idx=p.cat_idx,
            votes=p.votes,
            flux_ratio=p.flux_ratio,
        ))
    
    result = MatchResult(
        pairs=pairs,
        tri_match_count=c_result.tri_match_count,
        total_votes=c_result.total_votes,
    )
    
    _dll.psm_free_match_result(ctypes.byref(c_result))
    
    return result

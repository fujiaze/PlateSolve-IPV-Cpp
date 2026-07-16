"""
Vector Match V3.3 C++加速版 - Record-and-Refine

核心创新:
    Phase A (Record): 1点随机抽样, 记录所有有效变换(s,θ,tx,ty)和内点数
    Phase B (Refine): 对top-K最佳变换提取向量对应关系, SVD精确求解

完全不需要GridSearch、Hough、2点RANSAC、mix_ratio

Python端负责: Gaia查询、星点选取、WCS参数提取
C++端负责: Record-and-Refine、内点统计、SVD精修(OpenMP并行)

依赖: numpy, ctypes, 与vector_match_v2.py相同的GaiaClientPy
"""

import ctypes
import math
import os
import logging
import numpy as np
from typing import Optional

from vector_match_v2 import (
    GaiaClientPy, gnomonic_forward, gnomonic_inverse, bisection_mag_limit,
    VectorMatchResult, _DEGTORAD, _RADTODEG, _RADTOASEC, _ASECTORAD,
    _build_image_vectors, _build_catalog_vectors, _apply_flip,
    _apply_similarity, _count_inliers_1to1, _compute_normalized_score,
    _find_fine_correspondences, _iterative_svd_refine,
)

logger = logging.getLogger("vector_match_v3_3_cpp")

_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


class VM33SolveParamsC(ctypes.Structure):
    _fields_ = [
        ("tau", ctypes.c_double),
        ("s0", ctypes.c_double),
        ("s_min", ctypes.c_double),
        ("s_max", ctypes.c_double),
        ("n_modes", ctypes.c_int),
        ("seed", ctypes.c_int),
        ("K_total", ctypes.c_int),
        ("batch_size", ctypes.c_int),
        ("min_samples", ctypes.c_int),
        ("K_top", ctypes.c_int),
        ("min_inliers", ctypes.c_int),
        ("fov_diag_asec", ctypes.c_double),
    ]


class VM33DebugInfoC(ctypes.Structure):
    _fields_ = [
        ("theta_snr", ctypes.c_double),
        ("theta_peak_deg", ctypes.c_double),
        ("best_n_range", ctypes.c_int),
        ("median_noise", ctypes.c_double),
        ("n_phaseb_pairs", ctypes.c_int),
        ("n_phaseb_corr", ctypes.c_int),
        ("n_phasea_records", ctypes.c_int),
    ]


class VM33SolveResultC(ctypes.Structure):
    _fields_ = [
        ("s", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("tx", ctypes.c_double),
        ("ty", ctypes.c_double),
        ("n_inliers", ctypes.c_int),
        ("rms", ctypes.c_double),
        ("best_mode", ctypes.c_int),
        ("norm_score", ctypes.c_double),
        ("inlier_mask", ctypes.POINTER(ctypes.c_int)),
        ("success", ctypes.c_int),
        ("peak_snr", ctypes.c_double),
        ("n_samples", ctypes.c_int),
        ("debug", VM33DebugInfoC),
    ]


def _find_dll() -> str:
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(module_dir, "vector_match_v3_3.dll"),
        os.path.join(module_dir, "..", "..", "cpp", "vector_match_v3_3", "vector_match_v3_3.dll"),
        os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v3_3", "vector_match_v3_3.dll"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("vector_match_v3_3.dll")


def _load_dll() -> ctypes.CDLL:
    dll_path = _find_dll()
    dll_dir = os.path.dirname(os.path.abspath(dll_path))
    try:
        os.add_dll_directory(dll_dir)
    except OSError:
        pass

    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.isdir(mingw_bin):
        os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(mingw_bin)
        except OSError:
            pass

    dll = ctypes.CDLL(dll_path)

    dll.vm33_solve.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(VM33SolveParamsC),
        ctypes.POINTER(VM33SolveResultC),
    ]
    dll.vm33_solve.restype = ctypes.c_int

    dll.vm33_count_inliers.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),
    ]
    dll.vm33_count_inliers.restype = ctypes.c_int

    return dll


class VectorMatch:
    """V3.3 C++加速版向量匹配Plate Solving算法

    核心创新: Record-and-Refine
        Phase A: 1点随机抽样, 记录所有有效变换和内点数
        Phase B: 对top-K最佳变换提取向量对应关系, SVD精确求解

    Python端负责: Gaia查询、星点选取、WCS参数提取
    C++端负责: Record-and-Refine、内点统计、SVD精修(OpenMP并行)
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 1):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._dll = _load_dll()
        logger.info("VectorMatchV3.3Cpp初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

    def solve(self, img_x, img_y, img_flux, img_saturated,
              center_ra, center_dec, focal_length_mm, pixel_size_um,
              width, height):
        """V3.3 C++版主入口

        流程:
            1. 计算s0和FOV
            2. 构建U + 稀疏度权重
            3. Gaia查询
            4. 构建W, 调用C++ vm33_solve (n_modes=4, OpenMP并行4种翻转)
            5. 中心修正
            6. 重新投影W, 渐进tau精修
            7. WCS参数提取
        """
        s0 = 206.265 * pixel_size_um / focal_length_mm
        logger.info("s0=%.4f arcsec/px (FOCALLEN=%.1fmm XPIXSZ=%.1fum)",
                     s0, focal_length_mm, pixel_size_um)

        fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        logger.info("FOV diagonal=%.2f deg, query radius=%.2f deg", fov_diag, radius_deg)

        U, N_img, n_sat, sparsity = _build_image_vectors(
            np.asarray(img_x, dtype=np.float64),
            np.asarray(img_y, dtype=np.float64),
            np.asarray(img_flux, dtype=np.float64),
            np.asarray(img_saturated, dtype=np.int32),
            s0, width, height,
        )
        if N_img < 2:
            logger.error("Not enough bright stars: N_img=%d", N_img)
            raise ValueError(f"Not enough bright stars: N_img={N_img}")
        logger.info("Image vectors: N_img=%d (saturated=%d)", N_img, n_sat)

        if n_sat >= 50:
            N_gaia = math.ceil(1.5 * n_sat)
        else:
            N_gaia = 150
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            self._gaia, center_ra, center_dec, radius_deg, N_gaia
        )
        if M < 2:
            logger.error("Catalog stars insufficient: M=%d", M)
            raise ValueError(f"Catalog stars insufficient: M={M}")

        logger.info("Gaia query: mag_limit=%.2f, M=%d (target N_gaia=%d)", mag_limit, M, N_gaia)

        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)

        tau_coarse = max(1.0, 2.5 * s0)
        min_inliers = max(5, int(N_img * 0.1))

        params = VM33SolveParamsC()
        params.tau = tau_coarse
        params.s0 = s0
        params.s_min = 0.90
        params.s_max = 1.10
        params.n_modes = 4
        params.seed = 42
        params.K_total = 10000
        params.batch_size = 1000
        params.min_samples = 2000
        params.K_top = 50
        params.min_inliers = min_inliers
        params.fov_diag_asec = fov_diag * 3600.0

        inlier_mask_arr = np.zeros(N_img, dtype=np.int32)

        result = VM33SolveResultC()
        result.inlier_mask = inlier_mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        U_cont = np.ascontiguousarray(U, dtype=np.float64)
        W_cont = np.ascontiguousarray(W, dtype=np.float64)

        ret = self._dll.vm33_solve(
            U_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N_img,
            W_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
            ctypes.byref(params), ctypes.byref(result),
        )

        if ret != 0 or result.success == 0:
            logger.warning("C++ solve failed: ret=%d success=%d", ret, result.success)
            return None

        if result.s < 0.9 or result.s > 1.1:
            logger.warning("s=%.4f out of valid range [0.9, 1.1]", result.s)
            return None

        s = result.s
        theta = result.theta
        tx = result.tx
        ty = result.ty
        best_mode = result.best_mode

        logger.info("C++ solve: mode=%d s=%.4f theta=%.2f deg n=%d rms=%.3f SNR=%.1fx samples=%d",
                     best_mode, s, math.degrees(theta), result.n_inliers, result.rms,
                     result.peak_snr, result.n_samples)

        cos_d0 = math.cos(center_dec * _DEGTORAD)
        if abs(cos_d0) < 1e-10:
            cos_d0 = 1e-10
        delta_ra = -tx / (cos_d0 * 3600.0)
        delta_dec = -ty / 3600.0
        cur_ra = center_ra + delta_ra
        cur_dec = center_dec + delta_dec

        inlier_mask = inlier_mask_arr.astype(bool)
        Wf = _apply_flip(W, best_mode)

        rotation_deg = math.degrees(theta)
        s_final = s0 * s

        rms_arcsec = 0.0
        rms_px = 0.0
        if np.any(inlier_mask):
            Wt = _apply_similarity(Wf, s, theta, tx, ty)
            from scipy.spatial import cKDTree
            tree = cKDTree(Wt)
            dists, idxs = tree.query(U, k=1)
            U_in = U[inlier_mask]
            W_in = Wt[idxs[inlier_mask]]
            diffs = U_in - W_in
            rms_arcsec = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
            rms_px = rms_arcsec / s0 if s0 > 0 else 0.0

        cos_t, sin_t = math.cos(theta), math.sin(theta)
        affine = (tx, s * cos_t, -s * sin_t, ty, s * sin_t, s * cos_t)

        vm_result = VectorMatchResult(
            center_ra=cur_ra, center_dec=cur_dec,
            original_ra=center_ra, original_dec=center_dec,
            rotation_deg=rotation_deg, scale_arcsec_px=s_final,
            flip_mode=best_mode, matched_count=int(np.sum(inlier_mask)),
            rms_px=rms_px, rms_arcsec=rms_arcsec, affine=affine,
        )
        vm_result.solve_tx = tx
        vm_result.solve_ty = ty
        vm_result.solve_s = s
        vm_result.s0 = s0
        vm_result.theta_snr = float(result.debug.theta_snr)
        vm_result.theta_peak_deg = float(result.debug.theta_peak_deg)
        vm_result.best_n_range = int(result.debug.best_n_range)
        vm_result.median_noise = float(result.debug.median_noise)
        vm_result.n_phaseb_pairs = int(result.debug.n_phaseb_pairs)
        vm_result.n_phaseb_corr = int(result.debug.n_phaseb_corr)
        vm_result.n_phasea_records = int(result.debug.n_phasea_records)
        return vm_result

    def close(self):
        if not self._closed and self._gaia:
            self._gaia.close()
            self._gaia = None
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

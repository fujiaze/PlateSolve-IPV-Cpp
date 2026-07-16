"""
Vector Match V2 C++加速版（多线程优化）

通过ctypes调用C++ DLL实现V2算法的核心计算部分（RANSAC、SVD精修等），
Python端负责Gaia查询、星点选取、WCS参数提取等I/O部分。

C++端多线程优化:
    1. 4种翻转模式OpenMP并行
    2. RANSAC内循环OpenMP并行（线程局部最佳，最终归约）
    3. KDTree复用（Wf不变时只建一次）
    4. 渐进tau精修（1x→2x→3x→5x s0）

与Python V2完全对齐:
    - s0只用物理参数计算, 不接受外部覆盖
    - 中心修正: ΔRA = -tx / (cos×3600), ΔDec = -ty / 3600
    - norm_score = (n_inliers/min(N,M)) * (1-rms/tau)
    - 评分: score = n_inliers - 1.0 * rms
    - s范围限制: 0.9 <= s <= 1.1
    - norm_score阈值: 0.10

依赖: numpy, ctypes, 与vector_match_v2.py相同的GaiaClientPy
"""

import ctypes
import math
import os
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

# 复用V2 Python版的数据结构和Gaia客户端
from vector_match_v2 import (
    GaiaClientPy, gnomonic_forward, gnomonic_inverse, bisection_mag_limit,
    VectorMatchResult, _DEGTORAD, _RADTODEG, _RADTOASEC, _ASECTORAD,
    _build_image_vectors, _build_catalog_vectors, _apply_flip,
    _apply_similarity, _count_inliers_1to1, _compute_normalized_score,
    _find_fine_correspondences, _iterative_svd_refine,
)

logger = logging.getLogger("vector_match_v2_cpp")

_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


# ============================================================================
# ctypes结构体定义
# ============================================================================

class VM2SolveParamsC(ctypes.Structure):
    _fields_ = [
        ("tau_coarse", ctypes.c_double),
        ("K", ctypes.c_int),
        ("min_inliers", ctypes.c_int),
        ("candidate_radius", ctypes.c_double),
        ("s0", ctypes.c_double),
        ("fov_diag_asec", ctypes.c_double),
        ("n_modes", ctypes.c_int),
        ("seed", ctypes.c_int),
    ]


class VM2SolveResultC(ctypes.Structure):
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
    ]


# ============================================================================
# DLL查找与加载
# ============================================================================

def _find_dll() -> str:
    """查找C++ DLL"""
    module_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(module_dir, "vector_match_v2.dll"),
        os.path.join(module_dir, "..", "..", "cpp", "vector_match_v2", "vector_match_v2.dll"),
        os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v2", "vector_match_v2.dll"),
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    raise FileNotFoundError("未找到vector_match_v2.dll")


def _load_dll() -> ctypes.CDLL:
    """加载C++ DLL并设置函数签名"""
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

    # vm2_solve
    dll.vm2_solve.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # U, N_img
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # W, M
        ctypes.POINTER(ctypes.c_double),                # sparsity
        ctypes.POINTER(VM2SolveParamsC),                # params
        ctypes.POINTER(VM2SolveResultC),                # result
    ]
    dll.vm2_solve.restype = ctypes.c_int

    # vm2_count_inliers
    dll.vm2_count_inliers.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # U, N_img
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,  # W, M
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,  # s, theta, tx, ty
        ctypes.c_double,                                # tau
        ctypes.POINTER(ctypes.c_int),                   # inlier_mask
        ctypes.POINTER(ctypes.c_double),                # out_rms
    ]
    dll.vm2_count_inliers.restype = ctypes.c_int

    return dll


# ============================================================================
# 主类
# ============================================================================

class VectorMatch:
    """V2 C++加速版向量匹配Plate Solving算法（多线程优化）

    Python端负责: Gaia查询、星点选取、WCS参数提取
    C++端负责: RANSAC、SVD精修、内点统计等计算密集部分（OpenMP并行）

    与Python V2完全对齐:
        - s0只用物理参数计算, 不接受外部覆盖
        - 中心修正: ΔRA = -tx / (cos×3600), ΔDec = -ty / 3600
        - 渐进tau精修: 1x→2x→3x→5x s0
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._dll = _load_dll()
        logger.info("VectorMatchV2Cpp初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

    def solve(self, img_x, img_y, img_flux, img_saturated,
              center_ra, center_dec, focal_length_mm, pixel_size_um,
              width, height):
        """V2 C++版主入口

        流程（与Python V2完全对齐）:
            1. 计算s0和FOV（只用物理参数）
            2. 构建U + 稀疏度权重
            3. Gaia查询
            4. 构建W, 调用C++ vm2_solve (n_modes=4, OpenMP并行4种翻转)
            5. 中心修正（与Python对齐: ΔRA = -tx/(cos×3600)）
            6. 重新投影W, 渐进tau精修（1x→2x→3x→5x s0）
            7. WCS参数提取
        """
        # ── Step 1: 像素尺度s0和FOV计算（与Python V2对齐: 不接受外部覆盖）──
        s0 = 206.265 * pixel_size_um / focal_length_mm
        logger.info("像素尺度 s0=%.4f 角秒/像素 (FOCALLEN=%.1fmm XPIXSZ=%.1fμm)",
                     s0, focal_length_mm, pixel_size_um)

        fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        logger.info("FOV对角线=%.2f度, 查询半径=%.2f度", fov_diag, radius_deg)

        # ── Step 2: 构建U + 稀疏度权重 ──
        U, N_img, n_sat, sparsity = _build_image_vectors(
            np.asarray(img_x, dtype=np.float64),
            np.asarray(img_y, dtype=np.float64),
            np.asarray(img_flux, dtype=np.float64),
            np.asarray(img_saturated, dtype=np.int32),
            s0, width, height,
        )
        if N_img < 2:
            logger.error("图像亮星不足: N_img=%d", N_img)
            return None
        logger.info("图像向量组: N_img=%d (饱和星=%d)", N_img, n_sat)

        # ── Step 3: Gaia查询 ──
        if n_sat >= 50:
            N_gaia = math.ceil(1.5 * n_sat)
        else:
            N_gaia = 150
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            self._gaia, center_ra, center_dec, radius_deg, N_gaia
        )
        if M < 2:
            logger.error("星表星数不足: M=%d", M)
            return None
        logger.info("星表查询: 极限星等=%.2f, 星数=%d (目标N_gaia=%d)", mag_limit, M, N_gaia)

        # ── Step 4: 构建W, 调用C++ vm2_solve (n_modes=4, OpenMP并行) ──
        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)

        tau_coarse = max(1.0, 2.5 * s0)
        K = 3000
        min_inliers = max(5, int(N_img * 0.2))
        candidate_radius = fov_diag * 3600.0 * 0.1

        params = VM2SolveParamsC()
        params.tau_coarse = tau_coarse
        params.K = K
        params.min_inliers = min_inliers
        params.candidate_radius = candidate_radius
        params.s0 = s0
        params.fov_diag_asec = fov_diag * 3600.0
        params.n_modes = 4
        params.seed = 42

        # 分配内点掩码
        inlier_mask_arr = np.zeros(N_img, dtype=np.int32)

        result = VM2SolveResultC()
        result.inlier_mask = inlier_mask_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        U_cont = np.ascontiguousarray(U, dtype=np.float64)
        W_cont = np.ascontiguousarray(W, dtype=np.float64)
        sp_cont = np.ascontiguousarray(sparsity, dtype=np.float64)

        ret = self._dll.vm2_solve(
            U_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N_img,
            W_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
            sp_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.byref(params), ctypes.byref(result),
        )

        if ret != 0 or result.success == 0:
            logger.warning("C++求解失败: ret=%d success=%d", ret, result.success)
            return None

        # s范围检查
        if result.s < 0.9 or result.s > 1.1:
            logger.warning("s=%.4f超出有效范围[0.9, 1.1]", result.s)
            return None

        s = result.s
        theta = result.theta
        tx = result.tx
        ty = result.ty
        best_mode = result.best_mode

        logger.info("C++粗匹配: 模式=%d s=%.4f theta=%.2f° n=%d rms=%.3f score=%.4f",
                     best_mode, s, math.degrees(theta), result.n_inliers, result.rms, result.norm_score)

        # ── Step 5: 中心修正（与Python V2对齐: ΔRA = -tx/(cos×3600)）──
        cur_ra, cur_dec = center_ra, center_dec
        cos_d0 = math.cos(cur_dec * _DEGTORAD)
        if abs(cos_d0) < 1e-10:
            cos_d0 = 1e-10
        delta_ra = -tx / (cos_d0 * 3600.0)
        delta_dec = -ty / 3600.0
        cur_ra += delta_ra
        cur_dec += delta_dec
        logger.info("中心修正: ΔRA=%.6f° ΔDec=%.6f° → RA=%.6f Dec=%.6f",
                     delta_ra, delta_dec, cur_ra, cur_dec)

        # ── Step 6: 重新投影W, 渐进tau精修（与Python V2对齐: 1x→2x→3x→5x s0）──
        W_new = _build_catalog_vectors(cat_ra, cat_dec, cur_ra, cur_dec)
        Wf_new = _apply_flip(W_new, best_mode)

        refine_radius = fov_diag * 3600.0 * 0.5
        min_inliers_refine = max(5, int(N_img * 0.2))

        # 渐进放宽tau（与Python V2 _extract_wcs_and_converge对齐）
        refine_success = False
        for tau_mult in [1.0, 2.0, 3.0, 5.0]:
            tau_try = max(0.5, tau_mult * s0)

            params2 = VM2SolveParamsC()
            params2.tau_coarse = tau_try
            params2.K = 5000
            params2.min_inliers = min_inliers_refine
            params2.candidate_radius = refine_radius
            params2.s0 = s0
            params2.fov_diag_asec = fov_diag * 3600.0
            params2.n_modes = 1  # 只用最佳模式（W已翻转）
            params2.seed = 43

            inlier_mask_arr2 = np.zeros(N_img, dtype=np.int32)
            result2 = VM2SolveResultC()
            result2.inlier_mask = inlier_mask_arr2.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

            Wf_new_cont = np.ascontiguousarray(Wf_new, dtype=np.float64)
            ret2 = self._dll.vm2_solve(
                U_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N_img,
                Wf_new_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
                sp_cont.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.byref(params2), ctypes.byref(result2),
            )

            if ret2 == 0 and result2.success == 1 and result2.n_inliers >= min_inliers_refine:
                s, theta, tx, ty = result2.s, result2.theta, result2.tx, result2.ty
                inlier_mask = inlier_mask_arr2.astype(bool)
                Wf = Wf_new
                refine_success = True
                logger.info("  中心修正后C++ RANSAC+SVD(tau=%.1fx): s=%.4f theta=%.2f° n=%d rms=%.3f",
                            tau_mult, s, math.degrees(theta), result2.n_inliers, result2.rms)
                break
            else:
                logger.info("  中心修正后C++ RANSAC(tau=%.1fx): 内点不足 n=%d < min=%d",
                            tau_mult, result2.n_inliers if ret2 == 0 else -1, min_inliers_refine)

        if not refine_success:
            # 使用第一次的结果
            inlier_mask = inlier_mask_arr.astype(bool)
            Wf = _apply_flip(W, best_mode)

        # ── Step 7: 最终参数提取 ──
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

        return VectorMatchResult(
            center_ra=cur_ra, center_dec=cur_dec,
            original_ra=center_ra, original_dec=center_dec,
            rotation_deg=rotation_deg, scale_arcsec_px=s_final,
            flip_mode=best_mode, matched_count=int(np.sum(inlier_mask)),
            rms_px=rms_px, rms_arcsec=rms_arcsec, affine=affine,
        )

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

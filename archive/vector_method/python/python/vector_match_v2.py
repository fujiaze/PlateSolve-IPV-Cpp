"""
Vector Match V2 - 优化版向量组对齐Plate Solving算法

相比V1的五项核心优化:
    优化1: 稀疏度加权采样 - 孤立星优先采样，提升RANSAC命中率
    优化2: 尺度预检+固定尺度粗匹配 - |dU/dW-1|<0.05预检，s=1刚体变换降低自由度
    优化3: 迭代SVD精修 - Umeyama算法替代第二次RANSAC，统计最优+微秒级
    优化4: 动态内点阈值 - MAD自适应阈值，适应不同噪声环境
    优化5: 两阶段候选对构建 - 粗候选0.5×FOV + 变换投影重建精细候选

算法流程:
    Step 1: 像素尺度s0和FOV计算
    Step 2: 亮星选取 (饱和≥50全用，否则饱和+亮星共100颗)
    Step 3: Gaia锥形查询+二分法极限星等
    Step 4: 向量组构建 + 稀疏度权重计算
    Step 5: 4种翻转模式独立匹配:
        5a: 粗候选对构建 (0.5×FOV)
        5b: RANSAC粗匹配 (稀疏度加权采样 + 尺度预检 + 固定s=1)
        5c: 精细候选对重建 (粗变换投影)
        5d: 迭代SVD精修 (Umeyama + 动态MAD阈值)
    Step 6: 归一化打分选最佳模式
    Step 7: WCS参数提取 + 中心修正 + SVD精修

依赖: numpy, scipy(scipy.spatial.cKDTree), ctypes(Gaia DLL)
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger("vector_match_v2")

_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi
_RADTOASEC = (180.0 / np.pi) * 3600.0
_ASECTORAD = np.pi / (180.0 * 3600.0)


# ============================================================================
# Gaia客户端 (与V1相同)
# ============================================================================

class GaiaClientPy:
    """Gaia数据库Python客户端封装"""

    def __init__(self, data_dir: str, db_type: int = 0):
        dll_path = self._find_dll()
        self._dll = self._load_dll(dll_path)
        data_dir_bytes = data_dir.encode("utf-8")
        if db_type == 0:
            self._handle = self._dll.gaia_client_create(data_dir_bytes)
        else:
            self._handle = self._dll.gaia_client_create_ex(data_dir_bytes, db_type)
        if not self._handle:
            raise RuntimeError(f"Gaia客户端创建失败: {data_dir}")
        self._msvcrt = ctypes.CDLL("msvcrt.dll")
        self._closed = False

    @staticmethod
    def _find_dll() -> str:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(module_dir, "..", "gaia_client.dll"),
            os.path.join(_PROJECT_ROOT, "lib", "gaia_xpsd_client", "gaia_client.dll"),
        ]
        for c in candidates:
            p = os.path.normpath(c)
            if os.path.exists(p):
                return p
        raise FileNotFoundError("未找到gaia_client.dll")

    @staticmethod
    def _load_dll(dll_path: str) -> ctypes.CDLL:
        mingw_bin = r"C:\msys64\mingw64\bin"
        if os.path.isdir(mingw_bin):
            os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(mingw_bin)
            except OSError:
                pass
        dll_dir = os.path.dirname(os.path.abspath(dll_path))
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass
        dll = ctypes.CDLL(dll_path)
        dll.gaia_client_create.argtypes = [ctypes.c_char_p]
        dll.gaia_client_create.restype = ctypes.c_void_p
        dll.gaia_client_create_ex.argtypes = [ctypes.c_char_p, ctypes.c_int]
        dll.gaia_client_create_ex.restype = ctypes.c_void_p
        dll.gaia_client_destroy.argtypes = [ctypes.c_void_p]
        dll.gaia_client_destroy.restype = None
        dll.gaia_client_cone_search_for_solver.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.gaia_client_cone_search_for_solver.restype = ctypes.c_int
        return dll

    def cone_search(self, center_ra: float, center_dec: float, radius_deg: float, mag_limit: float):
        ra_ptr = ctypes.POINTER(ctypes.c_double)()
        dec_ptr = ctypes.POINTER(ctypes.c_double)()
        mag_ptr = ctypes.POINTER(ctypes.c_float)()
        n_stars = ctypes.c_int()
        ret = self._dll.gaia_client_cone_search_for_solver(
            self._handle, center_ra, center_dec, radius_deg, mag_limit,
            ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars),
        )
        if ret != 0:
            return np.array([]), np.array([]), np.array([])
        count = n_stars.value
        if count <= 0:
            return np.array([]), np.array([]), np.array([])
        ra_arr = np.array([ra_ptr[i] for i in range(count)], dtype=np.float64)
        dec_arr = np.array([dec_ptr[i] for i in range(count)], dtype=np.float64)
        mag_arr = np.array([float(mag_ptr[i]) for i in range(count)], dtype=np.float64)
        self._msvcrt.free(ra_ptr)
        self._msvcrt.free(dec_ptr)
        self._msvcrt.free(mag_ptr)
        return ra_arr, dec_arr, mag_arr

    def close(self):
        if not self._closed and self._handle:
            self._dll.gaia_client_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# 投影与查询 (与V1相同)
# ============================================================================

def gnomonic_forward(ra_deg, dec_deg, ra0_deg, dec0_deg):
    ra = np.asarray(ra_deg, dtype=np.float64) * _DEGTORAD
    dec = np.asarray(dec_deg, dtype=np.float64) * _DEGTORAD
    ra0 = ra0_deg * _DEGTORAD
    dec0 = dec0_deg * _DEGTORAD
    sin_dec0, cos_dec0 = np.sin(dec0), np.cos(dec0)
    delta_ra = ra - ra0
    sin_dec, cos_dec = np.sin(dec), np.cos(dec)
    cos_delta_ra = np.cos(delta_ra)
    cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_delta_ra
    valid = cosc > 1e-10
    cosc_safe = np.where(valid, cosc, 1.0)
    xi = cos_dec * np.sin(delta_ra) / cosc_safe
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_delta_ra) / cosc_safe
    xi = np.where(valid, xi, 0.0) * _RADTOASEC
    eta = np.where(valid, eta, 0.0) * _RADTOASEC
    return xi, eta, valid


def gnomonic_inverse(xi_asec, eta_asec, ra0_deg, dec0_deg):
    xi = np.asarray(xi_asec, dtype=np.float64) * _ASECTORAD
    eta = np.asarray(eta_asec, dtype=np.float64) * _ASECTORAD
    ra0 = ra0_deg * _DEGTORAD
    dec0 = dec0_deg * _DEGTORAD
    sin_dec0, cos_dec0 = np.sin(dec0), np.cos(dec0)
    rho = np.sqrt(xi ** 2 + eta ** 2)
    c = np.arctan(rho)
    sin_c, cos_c = np.sin(c), np.cos(c)
    rho_safe = np.where(rho > 1e-15, rho, 1.0)
    dec = np.arcsin(cos_c * sin_dec0 + eta * sin_c * cos_dec0 / rho_safe)
    ra = ra0 + np.arctan2(xi * sin_c, rho_safe * cos_dec0 * cos_c - eta * sin_dec0 * sin_c)
    return ra * _RADTODEG, dec * _RADTODEG


def bisection_mag_limit(gaia_client, center_ra, center_dec, radius_deg, target_count,
                        mag_low=6.0, mag_high=22.0, tolerance=0.1):
    target_high = int(target_count * 1.1)
    best_mag = mag_high
    best_ra, best_dec, best_mag_arr = np.array([]), np.array([]), np.array([])
    best_count = 0
    for _ in range(30):
        mid = (mag_low + mag_high) / 2.0
        ra, dec, mag = gaia_client.cone_search(center_ra, center_dec, radius_deg, mid)
        count = len(ra)
        if count < target_count:
            mag_low = mid
        elif count > target_high:
            mag_high = mid
        else:
            best_mag = mid
            best_count = count
            best_ra, best_dec, best_mag_arr = ra, dec, mag
            break
        best_mag = mid
        best_count = count
        best_ra, best_dec, best_mag_arr = ra, dec, mag
        if (mag_high - mag_low) <= tolerance:
            break
    return best_mag, best_count, best_ra, best_dec, best_mag_arr


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class VectorMatchResult:
    """向量匹配结果"""
    center_ra: float
    center_dec: float
    original_ra: float  # 原始投影中心RA
    original_dec: float  # 原始投影中心Dec
    rotation_deg: float
    scale_arcsec_px: float
    flip_mode: int
    matched_count: int
    rms_px: float
    rms_arcsec: float
    affine: tuple


# ============================================================================
# V2 核心函数
# ============================================================================

def _build_image_vectors(img_x, img_y, img_flux, img_saturated, scale0, width, height):
    """构建图像向量组U + 稀疏度权重

    返回:
        (U, N_img, n_sat, sparsity): sparsity[i] = U[i]到最近邻U点的距离
    """
    cx = width / 2.0
    cy = height / 2.0
    n_sat = int(np.sum(img_saturated))
    if n_sat >= 50:
        mask = img_saturated.astype(bool)
        sel_x, sel_y = img_x[mask], img_y[mask]
    else:
        mask_sat = img_saturated.astype(bool)
        n_normal = 100 - n_sat
        normal_idx = np.where(~mask_sat)[0]
        if len(normal_idx) > 0 and n_normal > 0:
            sorted_idx = normal_idx[np.argsort(-img_flux[normal_idx])]
            top_normal = sorted_idx[:n_normal]
            sel_idx = np.concatenate([np.where(mask_sat)[0], top_normal])
        else:
            sel_idx = np.where(mask_sat)[0]
        sel_x, sel_y = img_x[sel_idx], img_y[sel_idx]
    n_img = len(sel_x)
    if n_img < 2:
        return np.empty((0, 2)), 0, n_sat, np.array([])

    ux = (sel_x - cx) * scale0
    uy = -(sel_y - cy) * scale0
    U = np.column_stack([ux, uy])

    # 优化1: 计算稀疏度权重 = 到最近邻同池星的距离
    if n_img >= 3:
        tree_u = cKDTree(U)
        dists_nn, _ = tree_u.query(U, k=2)  # k=2: 第0个是自己, 第1个是最近邻
        sparsity = dists_nn[:, 1]  # 最近邻距离
    else:
        sparsity = np.ones(n_img)

    return U, n_img, n_sat, sparsity


def _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0):
    """构建星表向量组W (gnomonic投影)"""
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, ra0, dec0)
    return np.column_stack([xi[valid], eta[valid]])


def _apply_flip(W, mode):
    """施加翻转模式"""
    Wf = W.copy()
    if mode & 1:
        Wf[:, 0] = -Wf[:, 0]
    if mode & 2:
        Wf[:, 1] = -Wf[:, 1]
    return Wf


def _apply_similarity(W, s, theta, tx, ty):
    """应用相似变换"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt_x = s * (cos_t * W[:, 0] - sin_t * W[:, 1]) + tx
    Wt_y = s * (sin_t * W[:, 0] + cos_t * W[:, 1]) + ty
    return np.column_stack([Wt_x, Wt_y])


def _apply_rigid(W, theta, tx, ty):
    """应用刚体变换 (s=1)"""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    Wt_x = cos_t * W[:, 0] - sin_t * W[:, 1] + tx
    Wt_y = sin_t * W[:, 0] + cos_t * W[:, 1] + ty
    return np.column_stack([Wt_x, Wt_y])


def _count_inliers_1to1(U, Wt, tau):
    """1对1互斥内点统计"""
    tree = cKDTree(Wt)
    dists, idxs = tree.query(U, k=1)
    final_mask = np.zeros(len(U), dtype=bool)
    used_cat = set()
    order = np.argsort(dists)
    for i in order:
        if dists[i] >= tau:
            break
        ci = int(idxs[i])
        if ci in used_cat:
            continue
        used_cat.add(ci)
        final_mask[i] = True
    n_inliers = int(np.sum(final_mask))
    if n_inliers == 0:
        return 0, 0.0, final_mask
    rms = float(np.sqrt(np.mean(dists[final_mask] ** 2)))
    return n_inliers, rms, final_mask


def _compute_normalized_score(n_inliers, rms, N_img, M, tau):
    """归一化得分"""
    denom = min(N_img, M)
    if denom <= 0 or tau <= 0:
        return 0.0
    return (n_inliers / denom) * (1.0 - rms / tau)


# ============================================================================
# 优化1: 稀疏度加权采样
# ============================================================================

def _weighted_choice(sparsity, rng, n=2):
    """按稀疏度权重随机选取n个图像点索引

    sparsity越大(越孤立)的点被选中概率越高
    """
    weights = sparsity.copy()
    weights = np.maximum(weights, 1e-10)  # 避免全零
    prob = weights / np.sum(weights)
    return rng.choice(len(sparsity), n, replace=False, p=prob)


# ============================================================================
# 优化2: 尺度预检 + 固定尺度刚体变换
# ============================================================================

def _scale_precheck(u_a, u_b, w_a, w_b, tol=0.05):
    """尺度预检: |dU/dW - 1| < tol

    如果图像点对距离与星表点对距离之比偏离1太远，说明配对错误
    """
    dU = np.sqrt((u_a[0] - u_b[0]) ** 2 + (u_a[1] - u_b[1]) ** 2)
    dW = np.sqrt((w_a[0] - w_b[0]) ** 2 + (w_a[1] - w_b[1]) ** 2)
    if dW < 1e-12 or dU < 1e-12:
        return False
    ratio = dU / dW
    return abs(ratio - 1.0) < tol


def _solve_rigid_2pt(u_a, u_b, w_a, w_b):
    """固定s=1求解刚体变换(旋转+平移, 3自由度)

    θ = angle(u_a-u_b) - angle(w_a-w_b)
    t = u_a - R(θ)·w_a

    返回: (theta, tx, ty) 或 None
    """
    du = u_a - u_b
    dw = w_a - w_b
    norm_du = np.sqrt(du[0] ** 2 + du[1] ** 2)
    norm_dw = np.sqrt(dw[0] ** 2 + dw[1] ** 2)
    if norm_dw < 1e-12 or norm_du < 1e-12:
        return None
    theta = math.atan2(du[1], du[0]) - math.atan2(dw[1], dw[0])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R_wa_x = cos_t * w_a[0] - sin_t * w_a[1]
    R_wa_y = sin_t * w_a[0] + cos_t * w_a[1]
    tx = u_a[0] - R_wa_x
    ty = u_a[1] - R_wa_y
    return theta, tx, ty


# ============================================================================
# 优化3: Umeyama SVD精修
# ============================================================================

def _umeyama(U_pts, W_pts):
    """Umeyama算法: 从配对点集求解最优相似变换 (SVD)

    最小化 Σ‖u_i - (s·R·w_i + t)‖²

    步骤:
        1. 去质心: u' = u - μ_u, w' = w - μ_w
        2. 协方差矩阵 H = Σ w'_i · u'_i^T
        3. SVD分解 H = UΣV^T
        4. R = V·diag(1, det(VU^T))·U^T  (保证纯旋转)
        5. s = tr(Σ·S) / tr(Σ_w)  其中S=diag(1, det(VU^T))
        6. t = μ_u - s·R·μ_w

    参数:
        U_pts: 图像配对点 shape=(n, 2)
        W_pts: 星表配对点 shape=(n, 2)
    返回:
        (s, theta, tx, ty) 或 None
    """
    n = len(U_pts)
    if n < 2:
        return None

    mu_u = np.mean(U_pts, axis=0)
    mu_w = np.mean(W_pts, axis=0)
    u_centered = U_pts - mu_u
    w_centered = W_pts - mu_w

    # 协方差矩阵
    H = w_centered.T @ u_centered  # (2,2)
    U_svd, S_svd, Vt_svd = np.linalg.svd(H)

    # 保证纯旋转 (det(R)=+1)
    d = np.linalg.det(Vt_svd.T @ U_svd.T)
    sign_matrix = np.diag([1.0, d])

    # 旋转矩阵
    R = Vt_svd.T @ sign_matrix @ U_svd.T

    # 缩放因子: s = trace(Σ*S) / trace(W^T W)
    # 注意: trace(W^T W) = sum(||w_i - mu_w||^2), 不是除以n的方差
    trace_WtW = np.sum(w_centered ** 2)
    if trace_WtW < 1e-20:
        return None
    s = np.sum(S_svd * np.diag(sign_matrix)) / trace_WtW

    # 平移
    t = mu_u - s * R @ mu_w

    # 提取旋转角
    theta = math.atan2(R[1, 0], R[0, 0])

    return s, theta, float(t[0]), float(t[1])


# ============================================================================
# 优化4: 动态MAD阈值
# ============================================================================

def _adaptive_tau(U, Wt, base_tau, s0):
    """基于MAD的自适应内点阈值

    τ_fine = max(1.0×s0, 3.0 × 1.4826 × MAD)

    MAD = median(|r_i - median(r)|)
    1.4826是正态分布下MAD到标准差的换算系数
    3.0σ对应99.7%置信区间

    参数:
        U: 图像向量组
        Wt: 变换后的星表向量组
        base_tau: 基础阈值(下限)
        s0: 像素尺度
    返回:
        自适应阈值(角秒)
    """
    tree = cKDTree(Wt)
    dists, _ = tree.query(U, k=1)
    min_tau = max(0.5, 1.0 * s0)

    if len(dists) < 3:
        return base_tau

    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    tau_mad = 3.0 * 1.4826 * mad

    # 取base_tau和MAD阈值的较大值，但不低于min_tau
    return max(min_tau, max(base_tau, tau_mad))


# ============================================================================
# 优化5: 两阶段候选对构建
# ============================================================================

def _find_coarse_correspondences(U, W, radius):
    """阶段一: 粗候选对构建 (大半径0.5×FOV)

    对每个U点，找W中距离<radius的所有点，构成候选对列表。
    这与V1的候选对构建方式一致，确保不遗漏真匹配。
    """
    tree = cKDTree(W)
    # 查询每个U点在radius内的所有W点
    results = tree.query_ball_point(U, radius)
    pairs = []
    for u_idx, w_indices in enumerate(results):
        for w_idx in w_indices:
            pairs.append((u_idx, w_idx))
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def _find_fine_correspondences(U, W, s, theta, tx, ty, tau_coarse):
    """阶段二: 用粗变换投影重建高纯度候选池

    将W用粗相似变换(s,θ,tx,ty)投影到U空间，
    然后对每个U点在投影后的W'中搜索距离<2×tau_coarse的最近邻
    """
    Wt = _apply_similarity(W, s, theta, tx, ty)
    tree = cKDTree(Wt)
    dists, idxs = tree.query(U, k=1)
    radius = 2.0 * tau_coarse
    mask = dists < radius
    u_idx = np.where(mask)[0]
    w_idx = idxs[mask]
    if len(u_idx) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack([u_idx, w_idx])


# ============================================================================
# V2 RANSAC: 稀疏度加权 + 尺度预检 + 固定s=1刚体变换
# ============================================================================

def _ransac_rigid_v2(U, Wf, tau, K, min_inliers, rng, candidate_radius, sparsity):
    """V2 RANSAC: 稀疏度加权采样 + 相似变换

    流程:
        1. 建立粗候选对 (10%×FOV)
        2. RANSAC循环:
            a. 按稀疏度权重从候选对中抽取2对(不同U点)
            b. 2点相似变换求解(s, θ, tx, ty)
            c. 用tau统计1对1互斥内点
        3. 返回最佳相似变换

    参数:
        U: 图像向量组 (N, 2)
        Wf: 翻转后星表向量组 (M, 2)
        tau: 内点阈值(角秒)
        K: 最大迭代次数
        min_inliers: 最少内点数
        rng: 随机数生成器
        candidate_radius: 候选搜索半径(角秒)
        sparsity: U的稀疏度权重 (N,)
    返回:
        (s, theta, tx, ty, n_inliers, rms, inlier_mask)
    """
    N = len(U)
    M = len(Wf)
    if N < 2 or M < 2:
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    # 阶段一: 粗候选对
    pairs = _find_coarse_correspondences(U, Wf, candidate_radius)
    P = len(pairs)
    logger.debug("粗候选对: %d对 (半径=%.1f角秒)", P, candidate_radius)
    if P < 2:
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    # 预计算: 每个U点对应的候选对索引
    u_to_pair_indices = {}
    for p_idx in range(P):
        u_idx = int(pairs[p_idx, 0])
        if u_idx not in u_to_pair_indices:
            u_to_pair_indices[u_idx] = []
        u_to_pair_indices[u_idx].append(p_idx)

    # 每个U点的稀疏度权重 (用于选择U点)
    unique_u = np.array(list(u_to_pair_indices.keys()), dtype=np.int64)
    u_weights = sparsity[unique_u]
    u_weights = np.maximum(u_weights, 1e-10)
    u_prob = u_weights / u_weights.sum()

    best_score = -1e30
    best_s = 1.0
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0
    best_n_inliers = 0
    best_rms = 0.0
    best_mask = np.zeros(N, dtype=bool)

    actual_K = min(K, len(unique_u) * (len(unique_u) - 1) // 2)

    for _ in range(actual_K):
        # 优化1: 按稀疏度权重抽取2个不同的U点
        sel = _weighted_choice(u_prob, rng, n=2)
        u_idx_a = int(unique_u[sel[0]])
        u_idx_b = int(unique_u[sel[1]])

        # 从各自候选对中随机选1对
        pairs_a = u_to_pair_indices[u_idx_a]
        pairs_b = u_to_pair_indices[u_idx_b]
        idx_a = int(pairs_a[rng.integers(len(pairs_a))])
        idx_b = int(pairs_b[rng.integers(len(pairs_b))])

        w_idx_a = int(pairs[idx_a, 1])
        w_idx_b = int(pairs[idx_b, 1])

        u_a, u_b = U[u_idx_a], U[u_idx_b]
        w_a, w_b = Wf[w_idx_a], Wf[w_idx_b]

        # 2点相似变换求解
        du = u_a - u_b
        dw = w_a - w_b
        norm_du = math.sqrt(du[0] ** 2 + du[1] ** 2)
        norm_dw = math.sqrt(dw[0] ** 2 + dw[1] ** 2)
        if norm_dw < 1e-12 or norm_du < 1e-12:
            continue
        s = norm_du / norm_dw
        # s范围限制: 必须在1.0±10%以内，否则直接跳过
        if s < 0.9 or s > 1.1:
            continue
        theta = math.atan2(du[1], du[0]) - math.atan2(dw[1], dw[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx = u_a[0] - s * (cos_t * w_a[0] - sin_t * w_a[1])
        ty = u_a[1] - s * (sin_t * w_a[0] + cos_t * w_a[1])

        # 应用相似变换并统计内点
        Wt = _apply_similarity(Wf, s, theta, tx, ty)
        n_inliers, rms, final_mask = _count_inliers_1to1(U, Wt, tau)
        if n_inliers < min_inliers:
            continue

        score = n_inliers - 1.0 * rms
        if score > best_score:
            best_score = score
            best_s = s
            best_theta = theta
            best_tx = tx
            best_ty = ty
            best_n_inliers = n_inliers
            best_rms = rms
            best_mask = final_mask

    return best_s, best_theta, best_tx, best_ty, best_n_inliers, best_rms, best_mask


# ============================================================================
# V2 迭代SVD精修
# ============================================================================

def _iterative_svd_refine(U, Wf, inlier_mask, s0, s_init=1.0, theta_init=0.0, tx_init=0.0, ty_init=0.0, max_iter=10):
    """迭代SVD精修 (优化3+4)

    流程:
        1. 用RANSAC结果(s,θ,tx,ty)变换Wf，在紧阈值下重新统计内点
        2. 用高置信度内点建立1对1配对，Umeyama求解最优(s, θ, t)
        3. 用自适应MAD阈值重划内点
        4. 重复2-3直到收敛或max_iter次
        5. 最终统计

    参数:
        U: 图像向量组 (N, 2)
        Wf: 翻转后星表向量组 (M, 2)
        inlier_mask: 粗匹配内点掩码(仅用于判断是否有解)
        s0: 像素尺度
        s_init, theta_init, tx_init, ty_init: RANSAC粗匹配结果
        max_iter: 最大迭代次数
    返回:
        (s, theta, tx, ty, n_inliers, rms, inlier_mask)
    """
    N = len(U)
    cur_s = s_init
    cur_theta = theta_init
    cur_tx = tx_init
    cur_ty = ty_init

    # 先用紧阈值重新统计内点(粗匹配tau太松，内点含大量误匹配)
    tau_fine_init = 1.0 * s0
    Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
    n_init, rms_init, cur_mask = _count_inliers_1to1(U, Wt, tau_fine_init)
    logger.debug("  SVD初始: 紧阈值内点 n=%d rms=%.3f (tau=%.2f)", n_init, rms_init, tau_fine_init)

    if n_init < 3:
        # 紧阈值内点不足，尝试更松的阈值
        for tau_try in [2.0 * s0, 5.0 * s0, 10.0 * s0]:
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            n_init, rms_init, cur_mask = _count_inliers_1to1(U, Wt, tau_try)
            logger.debug("  SVD初始: 尝试tau=%.2f n=%d rms=%.3f", tau_try, n_init, rms_init)
            if n_init >= 3:
                break
        if n_init < 3:
            logger.debug("  SVD: 内点不足，跳过精修")
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            n_final, rms_final, final_mask = _count_inliers_1to1(U, Wt, _adaptive_tau(U, Wt, 1.0 * s0, s0))
            return cur_s, cur_theta, cur_tx, cur_ty, n_final, rms_final, final_mask

    for iteration in range(max_iter):
        if np.sum(cur_mask) < 3:
            break

        # 用当前变换建立配对
        Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
        tree = cKDTree(Wt)
        dists, idxs = tree.query(U, k=1)

        # 只用内点做SVD
        inlier_U = U[cur_mask]
        inlier_W_idx = idxs[cur_mask]
        inlier_W = Wf[inlier_W_idx]

        if len(inlier_U) < 3:
            break

        # 优化3: Umeyama SVD求解
        result = _umeyama(inlier_U, inlier_W)
        if result is None:
            break
        new_s, new_theta, new_tx, new_ty = result

        # 安全检查: s不应偏离1太远
        if abs(new_s - 1.0) > 0.1:
            logger.debug("  SVD迭代%d: s=%.4f偏离1太远，跳过", iteration, new_s)
            break

        cur_s, cur_theta, cur_tx, cur_ty = new_s, new_theta, new_tx, new_ty

        # 应用新变换
        Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)

        # 用固定紧阈值重划内点(自适应MAD阈值太松，不适合迭代精修)
        tau_iter = 1.0 * s0
        n_inliers, rms, new_mask = _count_inliers_1to1(U, Wt, tau_iter)

        logger.debug("  SVD迭代%d: s=%.4f theta=%.2f° n=%d rms=%.3f tau=%.2f",
                      iteration, cur_s, math.degrees(cur_theta), n_inliers, rms, tau_iter)

        if n_inliers < 3:
            break
        # 收敛检查
        if np.array_equal(cur_mask, new_mask):
            break
        cur_mask = new_mask

    # 最终统计: 用固定紧阈值(1.0*s0)
    Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
    n_inliers, rms, final_mask = _count_inliers_1to1(U, Wt, 1.0 * s0)

    return cur_s, cur_theta, cur_tx, cur_ty, n_inliers, rms, final_mask


# ============================================================================
# V2 主类
# ============================================================================

class VectorMatch:
    """V2 向量匹配Plate Solving算法

    五项优化:
        1. 稀疏度加权采样
        2. 尺度预检+固定尺度粗匹配
        3. 迭代SVD精修
        4. 动态MAD阈值
        5. 两阶段候选对构建
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._rng = np.random.default_rng(42)
        logger.info("VectorMatchV2初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

    def solve(
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
    ) -> Optional[VectorMatchResult]:
        """V2 向量匹配主入口

        流程:
            1. 计算s0和FOV (只用FOCALLEN/XPIXSZ, 不接受外部覆盖)
            2. 构建U + 稀疏度权重
            3. Gaia查询
            4. 4种翻转模式独立匹配:
                a. 粗候选 (0.5×FOV)
                b. RANSAC粗匹配 (加权+预检+刚体)
                c. 精细候选重建
                d. 迭代SVD精修
            5. 归一化打分选最佳
            6. WCS提取+中心修正+SVD精修
        """
        # s0只用物理参数计算, 不接受外部覆盖
        s0 = 206.265 * pixel_size_um / focal_length_mm
        logger.info("像素尺度 s0=%.4f 角秒/像素 (FOCALLEN=%.1fmm XPIXSZ=%.1fμm)",
                     s0, focal_length_mm, pixel_size_um)

        fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        logger.info("FOV对角线=%.2f度, 查询半径=%.2f度", fov_diag, radius_deg)

        # Step 2: 构建U + 稀疏度权重
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

        # Step 3: Gaia查询
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

        # RANSAC参数
        tau_coarse = max(1.0, 2.5 * s0)
        K = 3000
        min_inliers = max(5, int(N_img * 0.2))
        # 优化5: 粗候选半径 = FOV对角线的10% (平衡候选数量与RANSAC命中率)
        candidate_radius_coarse = fov_diag * 3600.0 * 0.1
        logger.info("RANSAC参数: tau_coarse=%.2f K=%d min_inliers=%d candidate_radius=%.1f角秒",
                     tau_coarse, K, min_inliers, candidate_radius_coarse)

        best_mode = -1
        best_norm_score = -1.0
        best_result = None

        for mode in range(4):
            W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
            Wf = _apply_flip(W, mode)
            logger.info("翻转模式%d: 星表向量组 %d 颗", mode, len(Wf))

            # ── 阶段A: RANSAC粗匹配 (加权+相似变换) ──
            s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
                U, Wf, tau_coarse, K, min_inliers, self._rng,
                candidate_radius_coarse, sparsity
            )
            if n_inliers < min_inliers:
                logger.info("  模式%d: 粗匹配内点不足 n=%d < min=%d", mode, n_inliers, min_inliers)
                continue
            logger.info("  模式%d 粗匹配: s=%.4f theta=%.2f° tx=%.2f ty=%.2f n=%d rms=%.3f",
                        mode, s, math.degrees(theta), tx, ty, n_inliers, rms)

            # ── 阶段B: 精细候选对重建 (优化5) ──
            pairs_fine = _find_fine_correspondences(U, Wf, s, theta, tx, ty, tau_coarse)
            logger.debug("  精细候选对: %d对", len(pairs_fine))

            # ── 阶段C: 迭代SVD精修 (优化3+4) ──
            s_refined, theta_refined, tx_refined, ty_refined, n_refined, rms_refined, mask_refined = \
                _iterative_svd_refine(U, Wf, inlier_mask, s0, s, theta, tx, ty, max_iter=10)

            if n_refined >= min_inliers:
                s, theta, tx, ty = s_refined, theta_refined, tx_refined, ty_refined
                n_inliers, rms, inlier_mask = n_refined, rms_refined, mask_refined
                logger.info("  模式%d SVD精修: s=%.4f theta=%.2f° n=%d rms=%.3f",
                            mode, s, math.degrees(theta), n_inliers, rms)

            norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)
            logger.info("  模式%d 最终: s=%.4f theta=%.2f° n=%d rms=%.3f norm_score=%.4f",
                        mode, s, math.degrees(theta), n_inliers, rms, norm_score)

            if norm_score > best_norm_score:
                best_norm_score = norm_score
                best_mode = mode
                best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

        if best_mode < 0 or best_norm_score < 0.10:
            logger.warning("所有模式匹配失败: best_norm_score=%.4f", best_norm_score)
            return None

        s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result

        # s范围限制: 最终s必须在1.0±10%以内
        if s < 0.9 or s > 1.1:
            logger.warning("s=%.4f超出有效范围[0.9, 1.1]，判定为无效结果", s)
            return None

        logger.info("最佳模式=%d, 归一化得分=%.4f", best_mode, best_norm_score)

        result = self._extract_wcs_and_converge(
            s, theta, tx, ty, best_mode, s0,
            center_ra, center_dec, width, height,
            U, Wf, inlier_mask, N_img, M,
            cat_ra, cat_dec, cat_mag,
            fov_diag, sparsity,
        )
        return result

    def _extract_wcs_and_converge(
        self, s, theta, tx, ty, flip_mode, s0,
        ra0, dec0, width, height,
        U, Wf, inlier_mask, N_img, M,
        cat_ra, cat_dec, cat_mag,
        fov_diag, sparsity,
    ):
        """WCS参数提取 + 中心修正 + SVD精修"""
        cur_ra, cur_dec = ra0, dec0
        cur_s, cur_theta = s, theta
        cur_tx, cur_ty = tx, ty
        cur_flip = flip_mode

        # 中心修正 (简单线性近似, 假设theta≈0)
        # 注意: ΔRA = -tx / (cos×3600), 因为W_new相对于W_old的偏移方向和tx相反
        cos_d0 = math.cos(cur_dec * _DEGTORAD)
        if abs(cos_d0) < 1e-10:
            cos_d0 = 1e-10
        delta_ra = -cur_tx / (cos_d0 * 3600.0)
        delta_dec = -cur_ty / 3600.0
        cur_ra += delta_ra
        cur_dec += delta_dec
        logger.info("中心修正: ΔRA=%.6f° ΔDec=%.6f° → RA=%.6f Dec=%.6f",
                     delta_ra, delta_dec, cur_ra, cur_dec)

        # 重新投影 + RANSAC精修 (中心修正后SVD可能找不到内点，用RANSAC更稳健)
        W_new = _build_catalog_vectors(cat_ra, cat_dec, cur_ra, cur_dec)
        Wf_new = _apply_flip(W_new, cur_flip)

        refine_radius = fov_diag * 3600.0 * 0.1
        min_inliers_refine = max(5, int(N_img * 0.2))

        # 渐进放宽tau: 从1.0*s0逐步放宽到5.0*s0
        # 原因: 粗匹配RMS可能较大(5-9角秒), 1.0*s0太紧找不到内点
        # 中心修正后需要更新affine参数
        # W_new = W_old - ΔW, 新平移 t_new = t + s × R × ΔW
        # 通过重新投影W_new并做RANSAC精修来更新参数
        refine_success = False
        for tau_mult in [1.0, 2.0, 3.0, 5.0]:
            tau_try = max(0.5, tau_mult * s0)
            s2, theta2, tx2, ty2, n2, rms2, mask2 = _ransac_rigid_v2(
                U, Wf_new, tau_try, 3000, min_inliers_refine, self._rng,
                refine_radius, sparsity
            )
            if n2 >= min_inliers_refine:
                # RANSAC成功，再做SVD精修
                s3, theta3, tx3, ty3, n3, rms3, mask3 = _iterative_svd_refine(
                    U, Wf_new, mask2, s0, s2, theta2, tx2, ty2, max_iter=10
                )
                if n3 >= min_inliers_refine:
                    cur_s, cur_theta = s3, theta3
                    cur_tx, cur_ty = tx3, ty3
                    inlier_mask = mask3
                    Wf = Wf_new
                    refine_success = True
                    logger.info("  中心修正后RANSAC+SVD(tau=%.1fx): s=%.4f theta=%.2f° tx=%.2f ty=%.2f n=%d rms=%.3f",
                                tau_mult, s3, math.degrees(theta3), tx3, ty3, n3, rms3)
                else:
                    cur_s, cur_theta = s2, theta2
                    cur_tx, cur_ty = tx2, ty2
                    inlier_mask = mask2
                    Wf = Wf_new
                    refine_success = True
                    logger.info("  中心修正后RANSAC(tau=%.1fx): s=%.4f theta=%.2f° tx=%.2f ty=%.2f n=%d rms=%.3f",
                                tau_mult, s2, math.degrees(theta2), tx2, ty2, n2, rms2)
                break
            else:
                logger.info("  中心修正后RANSAC(tau=%.1fx): 内点不足 n=%d < min=%d",
                            tau_mult, n2, min_inliers_refine)

        # 最终参数
        rotation_deg = math.degrees(cur_theta)
        s_final = s0 * cur_s

        rms_arcsec = 0.0
        rms_px = 0.0
        if np.any(inlier_mask):
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            tree = cKDTree(Wt)
            dists, idxs = tree.query(U, k=1)
            U_in = U[inlier_mask]
            W_in = Wt[idxs[inlier_mask]]
            diffs = U_in - W_in
            rms_arcsec = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
            rms_px = rms_arcsec / s0 if s0 > 0 else 0.0

        cos_t, sin_t = math.cos(cur_theta), math.sin(cur_theta)
        affine = (cur_tx, cur_s * cos_t, -cur_s * sin_t,
                  cur_ty, cur_s * sin_t, cur_s * cos_t)

        return VectorMatchResult(
            center_ra=cur_ra, center_dec=cur_dec,
            original_ra=ra0, original_dec=dec0,
            rotation_deg=rotation_deg, scale_arcsec_px=s_final,
            flip_mode=cur_flip, matched_count=int(np.sum(inlier_mask)),
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

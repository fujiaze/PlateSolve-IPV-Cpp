"""
Vector Match V3 - 黄金池采样+验证池验证+方向检验

相比V2的核心升级:
    优化6: 向量方向检验 - 内点判定同时要求位置对齐和方向一致
    优化7: 黄金池/验证池分层策略 - 小池采样计算变换，大池验证+精修

    分层策略:
        黄金池(Gold Pool): 饱和星+最亮星(~100颗), 质心精度高, RANSAC从中采样
        验证池(Validation Pool): 亮度前1000颗, 在更大范围搜索内点+方向检验

        流程:
        1. RANSAC从黄金池采样, 计算变换(s,θ,tx,ty)
        2. 用变换投影验证池的W, 在验证池中搜索内点(位置+方向)
        3. 验证池内点数作为RANSAC得分(比黄金池内点数更稳定)
        4. SVD精修也用验证池内点(更多配对→更精确)

    方向检验:
        内点条件: dist < tau_pos AND sin_dtheta < sin_tau
        二维叉乘: cross = |Ux*Wy - Uy*Wx|
        归一化: sin_dtheta = cross / (||U|| * ||W'||)
        对尺度变换不敏感: s0偏差只影响向量长度, 不影响方向

继承V2的五项优化:
    优化1: 稀疏度加权采样
    优化2: 尺度预检+固定尺度粗匹配
    优化3: 迭代SVD精修 (Umeyama)
    优化4: 动态MAD阈值
    优化5: 两阶段候选对构建

依赖: numpy, scipy(scipy.spatial.cKDTree), ctypes(Gaia DLL)
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger("vector_match_v3")

_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi
_RADTOASEC = (180.0 / np.pi) * 3600.0
_ASECTORAD = np.pi / (180.0 * 3600.0)


# ============================================================================
# Gaia客户端 (与V2相同)
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
# 投影与查询 (与V2相同)
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
    original_ra: float
    original_dec: float
    rotation_deg: float
    scale_arcsec_px: float
    flip_mode: int
    matched_count: int
    rms_px: float
    rms_arcsec: float
    affine: tuple


# ============================================================================
# V3 核心函数
# ============================================================================

def _build_gold_and_validation_pools(img_x, img_y, img_flux, img_saturated,
                                      scale0, width, height, n_validation=1000):
    """构建黄金池和验证池

    星选择策略(与V2一致):
        饱和≥50: 全部饱和星 + 1.2×Gaia星点
        饱和<50: 饱和+top亮星共200 + 300 Gaia星点
        星点不足: 用最多有效星点 + 1.5×Gaia星点

    黄金池: 用于RANSAC采样的星点集
    验证池: 亮度前n_validation颗(含黄金池), 用于验证和SVD精修

    返回:
        (U_gold, N_gold, n_sat, sparsity_gold, U_val, N_val, N_gaia_target)
    """
    cx = width / 2.0
    cy = height / 2.0
    n_sat = int(np.sum(img_saturated))
    n_total = len(img_x)

    # 按亮度排序所有星
    all_idx = np.arange(n_total)
    sorted_by_flux = all_idx[np.argsort(-img_flux)]

    sat_idx = np.where(img_saturated.astype(bool))[0]

    # ── 星选择策略 ──
    if n_sat >= 50:
        # 饱和≥50: 全部饱和星
        gold_idx = sat_idx
        N_gaia_target = math.ceil(1.2 * n_sat)
    else:
        # 饱和<50: 饱和+top亮星共200
        n_gold_target = min(200, n_total)
        n_normal_gold = n_gold_target - n_sat
        if n_normal_gold > 0:
            normal_sorted = sorted_by_flux[~np.isin(sorted_by_flux, sat_idx)]
            gold_idx = np.concatenate([sat_idx, normal_sorted[:n_normal_gold]])
        else:
            gold_idx = sat_idx
        N_gaia_target = 300

    # 星点不足时调整
    if len(gold_idx) < 50:
        N_gaia_target = math.ceil(1.5 * len(gold_idx))

    # 验证池: 按r由大到小, flux由高到低排序, 取前n_validation颗
    # r大的星向量更特征(信息量大), flux高的星质心更准
    r_all = np.sqrt((img_x - cx)**2 + (img_y - cy)**2)
    # lexsort: 最后一个key为主排序, 前面的为副排序
    # 主排序: -r (r由大到小), 副排序: -flux (flux由高到低)
    sort_key = np.lexsort((-img_flux, -r_all))
    val_idx = sort_key[:n_validation]

    # 确保黄金池是验证池的子集
    gold_set = set(gold_idx.tolist())
    val_set = set(val_idx.tolist())
    for idx in gold_set - val_set:
        val_idx = np.append(val_idx, idx)

    # 构建向量组
    def _to_vectors(sel_idx):
        sel_x = img_x[sel_idx]
        sel_y = img_y[sel_idx]
        ux = (sel_x - cx) * scale0
        uy = -(sel_y - cy) * scale0
        U = np.column_stack([ux, uy])
        return U

    U_gold = _to_vectors(gold_idx)
    U_val = _to_vectors(val_idx)

    N_gold = len(U_gold)
    N_val = len(U_val)

    # 稀疏度权重(基于黄金池)
    if N_gold >= 3:
        tree_u = cKDTree(U_gold)
        dists_nn, _ = tree_u.query(U_gold, k=2)
        sparsity = dists_nn[:, 1]
    else:
        sparsity = np.ones(N_gold)

    logger.info("黄金池: %d颗 (饱和=%d), 验证池: %d颗, Gaia目标: %d",
                N_gold, n_sat, N_val, N_gaia_target)

    return U_gold, N_gold, n_sat, sparsity, U_val, N_val, N_gaia_target, val_idx


def _build_catalog_vectors(cat_ra, cat_dec, ra0, dec0, cat_mag=None):
    """构建星表向量组W (gnomonic投影)

    参数:
        cat_mag: 可选, 星表星等数组。若提供, 同时返回有效投影点的星等
    返回:
        若cat_mag为None: W (N_valid, 2)
        若cat_mag提供: (W, cat_mag_valid)
    """
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, ra0, dec0)
    W = np.column_stack([xi[valid], eta[valid]])
    if cat_mag is not None:
        return W, cat_mag[valid]
    return W


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


# ============================================================================
# 方向检验内点统计
# ============================================================================

def _count_inliers_1to1(U, Wt, tau, sin_tau=1.0):
    """1对1互斥内点统计 + 分层检验(先方向后位置)

    分层检验策略:
        1. 先用廉价的方向条件过滤: sin_dtheta < sin_tau
        2. 再用昂贵的位置条件确认: dist < tau
        效果: 在密集星场，方向检验能瞬间剔除80%的邻近假配

    内点条件:
        1. 方向一致: sin_dtheta = |U x W'| / (||U|| * ||W'||) < sin_tau
        2. 位置对齐: dist = ||U[i] - W'[j]|| < tau

    参数:
        U: 图像向量组 (N, 2)
        Wt: 变换后的星表向量组 (M, 2)
        tau: 位置内点阈值(角秒)
        sin_tau: 方向偏差阈值
                 1.0 = 不做方向检验
                 0.1 = 约6度容差
                 0.05 = 约3度容差
    返回:
        (n_inliers, rms, inlier_mask)
    """
    tree = cKDTree(Wt)
    dists, idxs = tree.query(U, k=1)

    # 分层检验: 先方向检验(廉价)，再位置检验(昂贵)
    if sin_tau < 1.0:
        matched_Wt = Wt[idxs]
        # 计算叉乘(极快)
        cross = np.abs(U[:, 0] * matched_Wt[:, 1] - U[:, 1] * matched_Wt[:, 0])
        norms_u = np.sqrt(U[:, 0] ** 2 + U[:, 1] ** 2)
        norms_w = np.sqrt(matched_Wt[:, 0] ** 2 + matched_Wt[:, 1] ** 2)
        norm_prod = norms_u * norms_w
        sin_dtheta = np.where(norm_prod > 1e-10, cross / norm_prod, 0.0)

        # 先用方向检验过滤(廉价)
        direction_ok = sin_dtheta < sin_tau
        # 再用位置检验确认(昂贵)
        position_ok = dists < tau
        # 同时满足两个条件
        candidate_mask = direction_ok & position_ok
    else:
        candidate_mask = dists < tau

    # 1对1互斥匹配
    final_mask = np.zeros(len(U), dtype=bool)
    used_cat = set()
    order = np.argsort(dists)
    for i in order:
        if not candidate_mask[i]:
            continue
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
# 稀疏度加权采样
# ============================================================================

def _weighted_choice(sparsity, rng, n=2):
    """按稀疏度权重随机选取n个图像点索引"""
    weights = sparsity.copy()
    weights = np.maximum(weights, 1e-10)
    prob = weights / np.sum(weights)
    return rng.choice(len(sparsity), n, replace=False, p=prob)


# ============================================================================
# Umeyama SVD精修
# ============================================================================

def _umeyama(U_pts, W_pts):
    """Umeyama算法: 从配对点集求解最优相似变换 (SVD)"""
    n = len(U_pts)
    if n < 2:
        return None

    mu_u = np.mean(U_pts, axis=0)
    mu_w = np.mean(W_pts, axis=0)
    u_centered = U_pts - mu_u
    w_centered = W_pts - mu_w

    H = w_centered.T @ u_centered
    U_svd, S_svd, Vt_svd = np.linalg.svd(H)

    d = np.linalg.det(Vt_svd.T @ U_svd.T)
    sign_matrix = np.diag([1.0, d])

    R = Vt_svd.T @ sign_matrix @ U_svd.T

    trace_WtW = np.sum(w_centered ** 2)
    if trace_WtW < 1e-20:
        return None
    s = np.sum(S_svd * np.diag(sign_matrix)) / trace_WtW

    t = mu_u - s * R @ mu_w

    theta = math.atan2(R[1, 0], R[0, 0])

    return s, theta, float(t[0]), float(t[1])


# ============================================================================
# 动态MAD阈值
# ============================================================================

def _adaptive_tau(U, Wt, base_tau, s0):
    """基于MAD的自适应内点阈值"""
    tree = cKDTree(Wt)
    dists, _ = tree.query(U, k=1)
    min_tau = max(0.5, 1.0 * s0)

    if len(dists) < 3:
        return base_tau

    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    tau_mad = 3.0 * 1.4826 * mad

    return max(min_tau, max(base_tau, tau_mad))


# ============================================================================
# 两阶段候选对构建
# ============================================================================

def _find_coarse_correspondences(U, W, radius):
    """阶段一: 粗候选对构建"""
    tree = cKDTree(W)
    results = tree.query_ball_point(U, radius)
    pairs = []
    for u_idx, w_indices in enumerate(results):
        for w_idx in w_indices:
            pairs.append((u_idx, w_idx))
    if len(pairs) == 0:
        return np.empty((0, 2), dtype=np.int64)
    return np.array(pairs, dtype=np.int64)


def _find_fine_correspondences(U, W, s, theta, tx, ty, tau_coarse):
    """阶段二: 用粗变换投影重建高纯度候选池"""
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
# V3 RANSAC: 黄金池采样 + 验证池评分
# ============================================================================

def _ransac_v3(U_gold, U_val, Wf, tau, K, min_inliers, rng,
               candidate_radius, sparsity, sin_tau=1.0, fov_diag_arcsec=0.0):
    """V3 RANSAC: 黄金池采样 + 验证池评分 + LO优化 + 剪枝 + 提前终止

    流程:
        1. 从黄金池U_gold的粗候选对中采样2对, 计算变换(s,θ,tx,ty)
        2. 剪枝: 尺度±10%, 平移<0.5×FOV对角线
        3. 用变换投影Wf, 在验证池U_val中统计内点(位置+方向)
        4. LO-RANSAC: 找到好模型后局部优化
        5. 提前终止: 内点比例超过阈值时停止

    参数:
        U_gold: 黄金池向量组 (N_gold, 2), 用于采样
        U_val: 验证池向量组 (N_val, 2), 用于评分
        Wf: 翻转后星表向量组 (M, 2)
        tau: 位置内点阈值
        K: 最大迭代次数
        min_inliers: 最少内点数(基于验证池)
        rng: 随机数生成器
        candidate_radius: 候选搜索半径
        sparsity: 黄金池稀疏度权重
        sin_tau: 方向偏差阈值
        fov_diag_arcsec: FOV对角线(角秒), 用于平移剪枝
    """
    N_gold = len(U_gold)
    N_val = len(U_val)
    M = len(Wf)
    if N_gold < 2 or M < 2:
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N_val, dtype=bool)

    # 平移剪枝阈值: 0.5×FOV对角线
    max_tx_ty = 0.5 * fov_diag_arcsec if fov_diag_arcsec > 0 else 1e30
    logger.debug("平移剪枝阈值: tx,ty < %.1f角秒", max_tx_ty)

    # 提前终止阈值: 内点比例超过15%
    early_stop_ratio = 0.15
    early_stop_inliers = int(N_val * early_stop_ratio)

    # 粗候选对(基于黄金池)
    pairs = _find_coarse_correspondences(U_gold, Wf, candidate_radius)
    P = len(pairs)
    logger.debug("粗候选对: %d对 (半径=%.1f角秒)", P, candidate_radius)
    if P < 2:
        return 1.0, 0, 0, 0, 0, 0, np.zeros(N_val, dtype=bool)

    u_to_pair_indices = {}
    for p_idx in range(P):
        u_idx = int(pairs[p_idx, 0])
        if u_idx not in u_to_pair_indices:
            u_to_pair_indices[u_idx] = []
        u_to_pair_indices[u_idx].append(p_idx)

    unique_u = np.array(list(u_to_pair_indices.keys()), dtype=np.int64)
    u_weights = sparsity[unique_u]
    u_weights = np.maximum(u_weights, 1e-10)
    u_prob = u_weights / u_weights.sum()

    # 计算实际迭代次数
    actual_K = min(K, len(unique_u) * (len(unique_u) - 1) // 2)

    # ── 翻转快速失败机制 ──
    # 在前N次迭代后，如果内点数为0或非常低，直接终止该模式
    # 避免错误翻转模式跑满迭代次数
    fast_fail_check_iter = min(100, actual_K // 10)  # 前10%迭代或100次
    fast_fail_threshold = max(1, min_inliers // 2)  # 内点数低于min_inliers的一半视为失败

    best_score = -1e30
    best_s = 1.0
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0
    best_n_inliers = 0
    best_rms = 0.0
    best_mask = np.zeros(N_val, dtype=bool)

    # LO-RANSAC: 局部优化迭代计数
    lo_iter_count = 0
    max_lo_iter = 5

    # 剪枝统计
    prune_count = 0  # 被剪枝的采样次数
    total_samples = 0  # 总采样次数

    for iter_idx in range(actual_K):
        total_samples += 1
        sel = _weighted_choice(u_prob, rng, n=2)
        u_idx_a = int(unique_u[sel[0]])
        u_idx_b = int(unique_u[sel[1]])

        pairs_a = u_to_pair_indices[u_idx_a]
        pairs_b = u_to_pair_indices[u_idx_b]
        idx_a = int(pairs_a[rng.integers(len(pairs_a))])
        idx_b = int(pairs_b[rng.integers(len(pairs_b))])

        w_idx_a = int(pairs[idx_a, 1])
        w_idx_b = int(pairs[idx_b, 1])

        u_a, u_b = U_gold[u_idx_a], U_gold[u_idx_b]
        w_a, w_b = Wf[w_idx_a], Wf[w_idx_b]

        # ── 剪枝0: 单向量模长一致性 ──
        # 正确变换下，单个向量的模长不应出现巨大差异
        norm_u_a = math.sqrt(u_a[0] ** 2 + u_a[1] ** 2)
        norm_w_a = math.sqrt(w_a[0] ** 2 + w_a[1] ** 2)
        norm_u_b = math.sqrt(u_b[0] ** 2 + u_b[1] ** 2)
        norm_w_b = math.sqrt(w_b[0] ** 2 + w_b[1] ** 2)
        if norm_w_a > 1e-10 and norm_w_b > 1e-10:
            ratio_a = norm_u_a / norm_w_a
            ratio_b = norm_u_b / norm_w_b
            # 单向量模长比应在±30%范围内(粗匹配阶段偏差大)
            if abs(ratio_a - 1.0) > 0.30 or abs(ratio_b - 1.0) > 0.30:
                prune_count += 1
                continue

        du = u_a - u_b
        dw = w_a - w_b
        norm_du = math.sqrt(du[0] ** 2 + du[1] ** 2)
        norm_dw = math.sqrt(dw[0] ** 2 + dw[1] ** 2)
        if norm_dw < 1e-12 or norm_du < 1e-12:
            prune_count += 1
            continue

        # ── 剪枝1: 尺度约束(点对距离比) ±20% ──
        s = norm_du / norm_dw
        if s < 0.8 or s > 1.2:
            prune_count += 1
            continue

        # ── 剪枝2: 角度单调性(已移除) ──
        # 相机安装角度不固定(可360°旋转)，角度差无约束意义
        # 原逻辑: angle_diff > 90° 剪枝，但正确旋转下角度差=θ，可任意值

        theta = math.atan2(du[1], du[0]) - math.atan2(dw[1], dw[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx = u_a[0] - s * (cos_t * w_a[0] - sin_t * w_a[1])
        ty = u_a[1] - s * (sin_t * w_a[0] + cos_t * w_a[1])

        # ── 剪枝3: 平移约束 < 0.5×FOV对角线 ──
        if abs(tx) > max_tx_ty or abs(ty) > max_tx_ty:
            prune_count += 1
            continue

        # 在验证池中评分(大池评分, 但min_inliers基于黄金池)
        Wt = _apply_similarity(Wf, s, theta, tx, ty)
        n_inliers, rms, final_mask = _count_inliers_1to1(U_val, Wt, tau, sin_tau)
        if n_inliers < min_inliers:
            continue

        score = n_inliers - 1.0 * rms

        # LO-RANSAC: 局部优化
        if score > best_score * 0.9:  # 接近最佳时尝试LO优化
            s_lo, theta_lo, tx_lo, ty_lo, n_lo, rms_lo, mask_lo = _local_optimize(
                U_val, Wf, s, theta, tx, ty, tau, sin_tau, max_lo_iter, rng
            )
            if n_lo >= n_inliers:
                s, theta, tx, ty = s_lo, theta_lo, tx_lo, ty_lo
                n_inliers, rms, final_mask = n_lo, rms_lo, mask_lo
                score = n_inliers - 1.0 * rms
                lo_iter_count += 1

        if score > best_score:
            best_score = score
            best_s = s
            best_theta = theta
            best_tx = tx
            best_ty = ty
            best_n_inliers = n_inliers
            best_rms = rms
            best_mask = final_mask

            # 提前终止: 内点比例超过阈值
            if best_n_inliers >= early_stop_inliers:
                logger.debug("提前终止: iter=%d n_inliers=%d >= threshold=%d",
                            iter_idx, best_n_inliers, early_stop_inliers)
                break

        # ── 快速失败检查 ──
        # 在前N次迭代后，如果内点数为0或非常低，直接终止该模式
        if iter_idx == fast_fail_check_iter and best_n_inliers < fast_fail_threshold:
            logger.debug("快速失败: iter=%d n_inliers=%d < threshold=%d, 终止该模式",
                        iter_idx, best_n_inliers, fast_fail_threshold)
            break

    prune_ratio = prune_count / total_samples if total_samples > 0 else 0.0
    logger.info("RANSAC完成: iter=%d/%d LO优化=%d次 剪枝率=%.1f%% best_n=%d best_rms=%.3f",
                iter_idx + 1, actual_K, lo_iter_count, prune_ratio * 100, best_n_inliers, best_rms)

    return best_s, best_theta, best_tx, best_ty, best_n_inliers, best_rms, best_mask


def _local_optimize(U_val, Wf, s_init, theta_init, tx_init, ty_init,
                    tau, sin_tau, max_iter, rng):
    """LO-RANSAC局部优化: 从当前模型出发迭代精修

    流程:
        1. 用当前变换统计内点
        2. 从内点中随机采样2对重新计算变换
        3. 重复直到收敛或max_iter次
    """
    cur_s, cur_theta, cur_tx, cur_ty = s_init, theta_init, tx_init, ty_init
    best_n, best_rms, best_mask = 0, 0.0, np.zeros(len(U_val), dtype=bool)

    for _ in range(max_iter):
        Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
        n_inliers, rms, mask = _count_inliers_1to1(U_val, Wt, tau, sin_tau)

        if n_inliers < 2:
            break

        if n_inliers > best_n:
            best_n, best_rms, best_mask = n_inliers, rms, mask

        # 从内点中随机采样2对
        inlier_indices = np.where(mask)[0]
        if len(inlier_indices) < 2:
            break

        sel = rng.choice(inlier_indices, size=2, replace=False)
        u_a, u_b = U_val[sel[0]], U_val[sel[1]]

        # 找到对应的W
        tree = cKDTree(Wt)
        dists, w_indices = tree.query(U_val[mask], k=1)
        if len(w_indices) < 2:
            break

        w_a = Wf[w_indices[0]]
        w_b = Wf[w_indices[1]]

        # 重新计算变换
        du = u_a - u_b
        dw = w_a - w_b
        norm_du = math.sqrt(du[0] ** 2 + du[1] ** 2)
        norm_dw = math.sqrt(dw[0] ** 2 + dw[1] ** 2)
        if norm_dw < 1e-12 or norm_du < 1e-12:
            continue

        s_new = norm_du / norm_dw
        if s_new < 0.8 or s_new > 1.2:
            continue

        theta_new = math.atan2(du[1], du[0]) - math.atan2(dw[1], dw[0])
        cos_t, sin_t = math.cos(theta_new), math.sin(theta_new)
        tx_new = u_a[0] - s_new * (cos_t * w_a[0] - sin_t * w_a[1])
        ty_new = u_a[1] - s_new * (sin_t * w_a[0] + cos_t * w_a[1])

        cur_s, cur_theta, cur_tx, cur_ty = s_new, theta_new, tx_new, ty_new

    return cur_s, cur_theta, cur_tx, cur_ty, best_n, best_rms, best_mask


# ============================================================================
# V3 迭代SVD精修 (验证池+方向检验)
# ============================================================================

def _iterative_svd_refine(U_val, Wf, inlier_mask, s0, s_init=1.0, theta_init=0.0,
                          tx_init=0.0, ty_init=0.0, max_iter=10, sin_tau=1.0):
    """迭代SVD精修 + 方向检验 (在验证池上操作)

    参数:
        U_val: 验证池向量组
        Wf: 星表向量组
        inlier_mask: 验证池内点掩码
        sin_tau: 方向偏差阈值
    """
    cur_s = s_init
    cur_theta = theta_init
    cur_tx = tx_init
    cur_ty = ty_init

    # 先用紧阈值重新统计内点
    tau_fine_init = 1.0 * s0
    Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
    n_init, rms_init, cur_mask = _count_inliers_1to1(U_val, Wt, tau_fine_init, sin_tau)
    logger.debug("  SVD初始: 紧阈值内点 n=%d rms=%.3f (tau=%.2f sin_tau=%.3f)",
                 n_init, rms_init, tau_fine_init, sin_tau)

    if n_init < 3:
        for tau_try in [2.0 * s0, 5.0 * s0, 10.0 * s0]:
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            n_init, rms_init, cur_mask = _count_inliers_1to1(U_val, Wt, tau_try, sin_tau)
            logger.debug("  SVD初始: 尝试tau=%.2f n=%d rms=%.3f", tau_try, n_init, rms_init)
            if n_init >= 3:
                break
        if n_init < 3:
            logger.debug("  SVD: 内点不足，跳过精修")
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            n_final, rms_final, final_mask = _count_inliers_1to1(
                U_val, Wt, _adaptive_tau(U_val, Wt, 1.0 * s0, s0), sin_tau)
            return cur_s, cur_theta, cur_tx, cur_ty, n_final, rms_final, final_mask

    for iteration in range(max_iter):
        if np.sum(cur_mask) < 3:
            break

        Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
        tree = cKDTree(Wt)
        dists, idxs = tree.query(U_val, k=1)

        inlier_U = U_val[cur_mask]
        inlier_W_idx = idxs[cur_mask]
        inlier_W = Wf[inlier_W_idx]

        if len(inlier_U) < 3:
            break

        result = _umeyama(inlier_U, inlier_W)
        if result is None:
            break
        new_s, new_theta, new_tx, new_ty = result

        if abs(new_s - 1.0) > 0.2:
            logger.debug("  SVD迭代%d: s=%.4f偏离1太远，跳过", iteration, new_s)
            break

        cur_s, cur_theta, cur_tx, cur_ty = new_s, new_theta, new_tx, new_ty

        Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
        tau_iter = 1.0 * s0
        n_inliers, rms, new_mask = _count_inliers_1to1(U_val, Wt, tau_iter, sin_tau)

        logger.debug("  SVD迭代%d: s=%.4f theta=%.2f° n=%d rms=%.3f tau=%.2f",
                      iteration, cur_s, math.degrees(cur_theta), n_inliers, rms, tau_iter)

        if n_inliers < 3:
            break
        if np.array_equal(cur_mask, new_mask):
            break
        cur_mask = new_mask

    # 最终统计
    Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
    n_inliers, rms, final_mask = _count_inliers_1to1(U_val, Wt, 1.0 * s0, sin_tau)

    return cur_s, cur_theta, cur_tx, cur_ty, n_inliers, rms, final_mask


def _highdim_verify(U_val, Wf, s, theta, tx, ty, img_flux_val, cat_mag_valid, s0):
    """高维点积验证候选变换

    算法:
        1. 变换Gaia向量, KDTree最近邻匹配
        2. 距离阈值过滤
        3. 按角度逆时针排序对齐
        4. 亮度排名+模长剔除噪声
        5. 残差向量Dx, Dy高维点积
           正确变换: Dx·Dy≈0 (x/y残差独立), ||D||²很小
           错误变换: Dx·Dy大, ||D||²大

    参数:
        U_val: 验证池图像向量 (N_val, 2)
        Wf: 星表向量 (M, 2)
        s, theta, tx, ty: 候选变换参数
        img_flux_val: 验证池图像星flux (N_val,)
        cat_mag_valid: 有效投影的星表星等 (M,)
        s0: 像素尺度

    返回:
        (score, n_matched, rms_arcsec)
        score越小越好(正确变换下接近0)
    """
    N_val = len(U_val)
    M = len(Wf)

    # Step 1: 变换Gaia向量
    Wt = _apply_similarity(Wf, s, theta, tx, ty)

    # Step 2: KDTree最近邻匹配
    tree = cKDTree(Wt)
    dists, idxs = tree.query(U_val, k=1)

    # Step 3: 距离阈值过滤
    tau = max(1.0, 2.5 * s0)
    mask_dist = dists < tau
    n_matched = int(np.sum(mask_dist))

    if n_matched < 5:
        return float('inf'), n_matched, float('inf')

    # 获取匹配对
    U_matched = U_val[mask_dist]
    W_idx_matched = idxs[mask_dist]
    W_matched = Wt[W_idx_matched]
    flux_matched = img_flux_val[mask_dist]
    mag_matched = cat_mag_valid[W_idx_matched]

    # Step 4: 按角度逆时针排序
    # 对匹配的二维向量按其角度(atan2)排序, 使两组向量对齐
    angles = np.arctan2(U_matched[:, 1], U_matched[:, 0])
    sort_order = np.argsort(angles)
    U_sorted = U_matched[sort_order]
    W_sorted = W_matched[sort_order]
    flux_sorted = flux_matched[sort_order]
    mag_sorted = mag_matched[sort_order]

    # Step 5: 噪声剔除
    n = len(U_sorted)

    # 5a: 亮度排名一致性
    # 图像侧: flux越高排名越前(0), 星表侧: mag越低(越亮)排名越前(0)
    # 归一化到[0,1], 正确匹配下两侧排名应一致
    img_rank = np.argsort(np.argsort(-flux_sorted)).astype(np.float64) / max(n - 1, 1)
    cat_rank = np.argsort(np.argsort(mag_sorted)).astype(np.float64) / max(n - 1, 1)
    rank_diff = np.abs(img_rank - cat_rank)

    # 5b: 模长一致性 (位置向量模长比)
    # 正确变换下 ||U||/||W'|| ≈ 1.0
    norms_u = np.sqrt(U_sorted[:, 0]**2 + U_sorted[:, 1]**2)
    norms_w = np.sqrt(W_sorted[:, 0]**2 + W_sorted[:, 1]**2)
    mag_ratio = norms_u / np.maximum(norms_w, 1e-10)

    # 噪声剔除: 排名差异<0.4 且 模长比在0.5-2.0之间
    clean_mask = (rank_diff < 0.4) & (mag_ratio > 0.5) & (mag_ratio < 2.0)
    n_clean = int(np.sum(clean_mask))

    if n_clean < 5:
        # 噪声剔除太严, 放宽: 仅用模长比
        clean_mask = (mag_ratio > 0.3) & (mag_ratio < 3.0)
        n_clean = int(np.sum(clean_mask))

    if n_clean < 5:
        # 仍然太严, 不做剔除
        clean_mask = np.ones(n, dtype=bool)
        n_clean = n

    U_clean = U_sorted[clean_mask]
    W_clean = W_sorted[clean_mask]

    # Step 6: 残差向量
    D = U_clean - W_clean
    Dx = D[:, 0]
    Dy = D[:, 1]

    # Step 7: 高维点积验证
    # 交叉点积: 正确变换下 Dx·Dy ≈ 0 (x/y残差独立)
    cross_dot = abs(float(np.dot(Dx, Dy)))
    # 自点积: 正确变换下 ||Dx||² + ||Dy||² 很小
    self_dot = float(np.dot(Dx, Dx) + np.dot(Dy, Dy))

    n_final = len(Dx)
    rms_arcsec = math.sqrt(self_dot / n_final) if n_final > 0 else float('inf')

    # 综合得分: 交叉点积(方向独立性) + 自点积(位置精度)
    # 归一化: 除以匹配数避免维度偏差
    score = cross_dot / n_final + self_dot / n_final

    logger.debug("  高维验证: n_matched=%d n_clean=%d cross_dot=%.2f self_dot=%.2f "
                 "rms=%.3f score=%.4f", n_matched, n_clean, cross_dot, self_dot,
                 rms_arcsec, score)

    return score, n_final, rms_arcsec


def _ransac_v3_topk(U_gold, U_val, Wf, tau, K, min_inliers, rng,
                     candidate_radius, sparsity, sin_tau=1.0,
                     fov_diag_arcsec=0.0, top_k=5):
    """RANSAC返回top-K候选变换 (稀疏度加权U点采样, 与V2一致)

    采样策略: 按稀疏度权重抽取U点, 再从候选对中选配对 (与V2 _ransac_rigid_v2相同)
    评分在验证池U_val上进行

    返回:
        list of (s, theta, tx, ty, n_inliers, rms, inlier_mask, score)
        按score降序排列
    """
    N_gold = len(U_gold)
    M = len(Wf)

    if N_gold < 2 or M < 2:
        return []

    # 粗候选对(基于黄金池)
    pairs = _find_coarse_correspondences(U_gold, Wf, candidate_radius)
    if len(pairs) < 2:
        logger.debug("_ransac_v3_topk: 候选对不足 n_pairs=%d", len(pairs))
        return []

    # 预计算: 每个U点对应的候选对索引 (与V2相同)
    u_to_pair_indices = {}
    for p_idx in range(len(pairs)):
        u_idx = int(pairs[p_idx, 0])
        if u_idx not in u_to_pair_indices:
            u_to_pair_indices[u_idx] = []
        u_to_pair_indices[u_idx].append(p_idx)

    # 每个U点的稀疏度权重 (用于选择U点)
    unique_u = np.array(list(u_to_pair_indices.keys()), dtype=np.int64)
    u_weights = sparsity[unique_u]
    u_weights = np.maximum(u_weights, 1e-10)
    u_prob = u_weights / u_weights.sum()

    actual_K = min(K, len(unique_u) * (len(unique_u) - 1) // 2)

    top_results = []  # (score, result_tuple)
    n_tried = 0
    n_passed = 0

    for _ in range(actual_K):
        # 稀疏度加权U点采样 (与V2相同)
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

        u_a, u_b = U_gold[u_idx_a], U_gold[u_idx_b]
        w_a, w_b = Wf[w_idx_a], Wf[w_idx_b]

        # 剪枝0: 模长比±30%
        norm_u_a = math.sqrt(u_a[0]**2 + u_a[1]**2)
        norm_w_a = math.sqrt(w_a[0]**2 + w_a[1]**2)
        norm_u_b = math.sqrt(u_b[0]**2 + u_b[1]**2)
        norm_w_b = math.sqrt(w_b[0]**2 + w_b[1]**2)
        if norm_w_a > 1e-10 and norm_w_b > 1e-10:
            ratio_a = norm_u_a / norm_w_a
            ratio_b = norm_u_b / norm_w_b
            if abs(ratio_a - 1.0) > 0.30 or abs(ratio_b - 1.0) > 0.30:
                continue

        # 相似变换求解
        du = u_b - u_a
        dw = w_b - w_a
        norm_du = math.sqrt(du[0]**2 + du[1]**2)
        norm_dw = math.sqrt(dw[0]**2 + dw[1]**2)
        if norm_dw < 1e-10 or norm_du < 1e-10:
            continue

        s_new = norm_du / norm_dw
        # 剪枝1: 尺度约束±20%
        if s_new < 0.8 or s_new > 1.2:
            continue

        theta_est = math.atan2(du[1], du[0]) - math.atan2(dw[1], dw[0])
        cos_t = math.cos(theta_est)
        sin_t = math.sin(theta_est)
        tx_est = u_a[0] - s_new * (cos_t * w_a[0] - sin_t * w_a[1])
        ty_est = u_a[1] - s_new * (sin_t * w_a[0] + cos_t * w_a[1])

        # 剪枝3: 平移约束
        if fov_diag_arcsec > 0:
            max_shift = 0.5 * fov_diag_arcsec
            if abs(tx_est) > max_shift or abs(ty_est) > max_shift:
                continue

        # 在验证池上评分
        Wt = _apply_similarity(Wf, s_new, theta_est, tx_est, ty_est)
        n_inliers, rms, final_mask = _count_inliers_1to1(U_val, Wt, tau, sin_tau)

        n_tried += 1
        if n_inliers < min_inliers:
            continue

        n_passed += 1
        score = n_inliers / (1.0 + rms)

        # 维护top-K
        result = (s_new, theta_est, tx_est, ty_est, n_inliers, rms, final_mask, score)
        if len(top_results) < top_k:
            top_results.append((score, result))
            top_results.sort(key=lambda x: -x[0])
        elif score > top_results[-1][0]:
            top_results[-1] = (score, result)
            top_results.sort(key=lambda x: -x[0])

    logger.info("_ransac_v3_topk: n_pairs=%d n_tried=%d n_passed=%d top=%d",
                len(pairs), n_tried, n_passed, len(top_results))
    return [r[1] for r in top_results]

class VectorMatch:
    """V3 向量匹配Plate Solving算法

    V3核心策略:
        1. 黄金池RANSAC → top-K候选变换 (特征性最强的星采样)
        2. 快速筛查: 10%缩放/角度一致性
        3. 高维点积验证: 角度排序+亮度/模长剔除噪声+残差点积
           正确变换: Dx·Dy≈0 (x/y残差独立), ||D||²很小
           错误变换: Dx·Dy大, ||D||²大
        4. 4种翻转模式并行, 有收敛即终止
        5. SVD精修最佳候选
    """

    # 方向检验阈值
    SIN_TAU_COARSE = 1.0   # 粗匹配: 不做方向检验
    SIN_TAU_REFINE = 0.1   # 精修: sin(6°)≈0.1

    # 验证池大小
    N_VALIDATION = 1000

    # RANSAC top-K候选数
    TOP_K = 5

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._rng = np.random.default_rng(42)
        logger.info("VectorMatchV3初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

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
        """V3 向量匹配主入口"""
        s0 = 206.265 * pixel_size_um / focal_length_mm
        logger.info("像素尺度 s0=%.4f 角秒/像素 (FOCALLEN=%.1fmm XPIXSZ=%.1fμm)",
                     s0, focal_length_mm, pixel_size_um)

        fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        logger.info("FOV对角线=%.2f度, 查询半径=%.2f度", fov_diag, radius_deg)

        # Step 1: 构建黄金池+验证池
        img_x_a = np.asarray(img_x, dtype=np.float64)
        img_y_a = np.asarray(img_y, dtype=np.float64)
        img_flux_a = np.asarray(img_flux, dtype=np.float64)
        img_sat_a = np.asarray(img_saturated, dtype=np.int32)

        U_gold, N_gold, n_sat, sparsity, U_val, N_val, N_gaia_target, val_idx = _build_gold_and_validation_pools(
            img_x_a, img_y_a, img_flux_a, img_sat_a,
            s0, width, height, self.N_VALIDATION,
        )
        if N_gold < 2:
            logger.error("黄金池亮星不足: N_gold=%d", N_gold)
            return None

        # 验证池的flux (用于高维验证)
        img_flux_val = img_flux_a[val_idx]

        # Step 2: Gaia查询
        mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            self._gaia, center_ra, center_dec, radius_deg, N_gaia_target
        )
        if M < 2:
            logger.error("星表星数不足: M=%d", M)
            return None
        logger.info("星表查询: 极限星等=%.2f, 星数=%d (目标N_gaia=%d)", mag_limit, M, N_gaia_target)

        # RANSAC参数
        tau_coarse = max(1.0, 2.5 * s0)
        K = 3000
        min_inliers = max(5, int(N_gold * 0.2))
        candidate_radius_coarse = fov_diag * 3600.0 * 0.1
        fov_diag_arcsec = fov_diag * 3600.0
        logger.info("RANSAC参数: tau_coarse=%.2f K=%d min_inliers=%d top_k=%d",
                     tau_coarse, K, min_inliers, self.TOP_K)

        # ── 4种翻转模式并行匹配 ──
        def _run_mode(mode):
            """单模式: 黄金池RANSAC→top-K→高维验证→选最佳"""
            rng = np.random.default_rng(seed=mode * 1000 + 42)
            W, cat_mag_valid = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec, cat_mag)
            Wf = _apply_flip(W, mode)
            logger.info("翻转模式%d: 星表向量组 %d 颗", mode, len(Wf))

            # 阶段A: RANSAC获取top-K候选变换
            candidates = _ransac_v3_topk(
                U_gold, U_val, Wf, tau_coarse, K, min_inliers, rng,
                candidate_radius_coarse, sparsity,
                sin_tau=self.SIN_TAU_COARSE,
                fov_diag_arcsec=fov_diag_arcsec,
                top_k=self.TOP_K,
            )
            if not candidates:
                logger.info("  模式%d: RANSAC无候选", mode)
                return mode, None

            logger.info("  模式%d: RANSAC产生%d个候选", mode, len(candidates))

            # 阶段B: 高维点积验证每个候选
            best_verify_score = float('inf')
            best_candidate = None

            for cand in candidates:
                s_c, theta_c, tx_c, ty_c, n_c, rms_c, mask_c, score_c = cand

                # 快速筛查: 缩放±15%
                if s_c < 0.85 or s_c > 1.15:
                    continue

                # 高维点积验证 (使用cat_mag_valid: 有效投影点的星等)
                verify_score, n_verified, rms_verified = _highdim_verify(
                    U_val, Wf, s_c, theta_c, tx_c, ty_c,
                    img_flux_val, cat_mag_valid, s0,
                )

                if verify_score < best_verify_score:
                    best_verify_score = verify_score
                    best_candidate = (s_c, theta_c, tx_c, ty_c, n_verified, rms_verified, mask_c)
                    logger.debug("  模式%d 候选: s=%.4f θ=%.2f° n=%d verify=%.4f rms=%.3f",
                                 mode, s_c, math.degrees(theta_c), n_verified, verify_score, rms_verified)

            if best_candidate is None:
                logger.info("  模式%d: 高维验证无通过候选", mode)
                return mode, None

            s, theta, tx, ty, n_inliers, rms, inlier_mask = best_candidate
            logger.info("  模式%d 最佳: s=%.4f θ=%.2f° n=%d verify=%.4f rms=%.3f",
                        mode, s, math.degrees(theta), n_inliers, best_verify_score, rms)

            # 阶段C: SVD精修
            s_ref, theta_ref, tx_ref, ty_ref, n_ref, rms_ref, mask_ref = \
                _iterative_svd_refine(U_val, Wf, inlier_mask, s0, s, theta, tx, ty,
                                      max_iter=10, sin_tau=self.SIN_TAU_REFINE)
            if n_ref >= min_inliers:
                s, theta, tx, ty = s_ref, theta_ref, tx_ref, ty_ref
                n_inliers, rms, inlier_mask = n_ref, rms_ref, mask_ref
                logger.info("  模式%d SVD精修: s=%.4f θ=%.2f° n=%d rms=%.3f",
                            mode, s, math.degrees(theta), n_inliers, rms)

            result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf, best_verify_score)
            return mode, result

        # 并行执行4种模式, 有收敛即终止
        best_mode = -1
        best_verify = float('inf')
        best_result = None

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_mode, m): m for m in [2, 0, 1, 3]}

            for future in as_completed(futures):
                mode, result = future.result()
                if result is None:
                    continue

                s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf, verify_score = result

                # verify_score越小越好
                if verify_score < best_verify:
                    best_verify = verify_score
                    best_mode = mode
                    best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

                    # 高质量匹配: verify_score很小且RMS低
                    if verify_score < 0.01 and rms < 2.0 * s0:
                        logger.info("  模式%d找到高质量匹配(verify=%.4f), 取消剩余模式", mode, verify_score)
                        for f in futures:
                            f.cancel()
                        break

        if best_mode < 0 or best_verify == float('inf'):
            logger.warning("所有模式匹配失败: best_verify=%.4f", best_verify)
            return None

        s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result

        if s < 0.85 or s > 1.15:
            logger.warning("s=%.4f超出有效范围[0.85, 1.15]，判定为无效结果", s)
            return None

        logger.info("最佳模式=%d, 验证得分=%.4f", best_mode, best_verify)

        result = self._extract_wcs_and_converge(
            s, theta, tx, ty, best_mode, s0,
            center_ra, center_dec, width, height,
            U_val, Wf, inlier_mask, N_val, M,
            cat_ra, cat_dec, cat_mag,
            fov_diag, sparsity, U_gold,
        )
        return result

    def _extract_wcs_and_converge(
        self, s, theta, tx, ty, flip_mode, s0,
        ra0, dec0, width, height,
        U_val, Wf, inlier_mask, N_val, M,
        cat_ra, cat_dec, cat_mag,
        fov_diag, sparsity, U_gold,
    ):
        """WCS参数提取 + 中心修正 + SVD精修"""
        cur_ra, cur_dec = ra0, dec0
        cur_s, cur_theta = s, theta
        cur_tx, cur_ty = tx, ty
        cur_flip = flip_mode

        # 中心修正
        cos_d0 = math.cos(cur_dec * _DEGTORAD)
        if abs(cos_d0) < 1e-10:
            cos_d0 = 1e-10
        delta_ra = -cur_tx / (cos_d0 * 3600.0)
        delta_dec = -cur_ty / 3600.0
        cur_ra += delta_ra
        cur_dec += delta_dec
        logger.info("中心修正: ΔRA=%.6f° ΔDec=%.6f° → RA=%.6f Dec=%.6f",
                     delta_ra, delta_dec, cur_ra, cur_dec)

        # 重新投影 + RANSAC精修
        W_new = _build_catalog_vectors(cat_ra, cat_dec, cur_ra, cur_dec)
        Wf_new = _apply_flip(W_new, cur_flip)

        refine_radius = fov_diag * 3600.0 * 0.1
        min_inliers_refine = max(5, int(N_val * 0.03))

        # 渐进放宽tau
        for tau_mult in [1.0, 2.0, 3.0, 5.0]:
            tau_try = max(0.5, tau_mult * s0)
            s2, theta2, tx2, ty2, n2, rms2, mask2 = _ransac_v3(
                U_gold, U_val, Wf_new, tau_try, 3000, min_inliers_refine, self._rng,
                refine_radius, sparsity,
                sin_tau=self.SIN_TAU_COARSE,
            )
            if n2 >= min_inliers_refine:
                s3, theta3, tx3, ty3, n3, rms3, mask3 = _iterative_svd_refine(
                    U_val, Wf_new, mask2, s0, s2, theta2, tx2, ty2,
                    max_iter=10, sin_tau=self.SIN_TAU_REFINE,
                )
                if n3 >= min_inliers_refine:
                    cur_s, cur_theta = s3, theta3
                    cur_tx, cur_ty = tx3, ty3
                    inlier_mask = mask3
                    Wf = Wf_new
                    logger.info("  中心修正后RANSAC+SVD(tau=%.1fx): s=%.4f theta=%.2f° n=%d rms=%.3f",
                                tau_mult, s3, math.degrees(theta3), n3, rms3)
                else:
                    cur_s, cur_theta = s2, theta2
                    cur_tx, cur_ty = tx2, ty2
                    inlier_mask = mask2
                    Wf = Wf_new
                    logger.info("  中心修正后RANSAC(tau=%.1fx): s=%.4f theta=%.2f° n=%d rms=%.3f",
                                tau_mult, s2, math.degrees(theta2), n2, rms2)
                break
            else:
                logger.info("  中心修正后RANSAC(tau=%.1fx): 内点不足 n=%d < min=%d",
                            tau_mult, n2, min_inliers_refine)

        # 最终参数
        rotation_deg = math.degrees(cur_theta)
        s_final = s0 * cur_s

        rms_arcsec = 0.0
        rms_px = 0.0
        matched_count = 0
        if True:
            Wt = _apply_similarity(Wf, cur_s, cur_theta, cur_tx, cur_ty)
            # 用验证池重新计算RMS和内点数
            tau_rms = max(1.0, 2.5 * s0)
            _, rms_val, mask_val = _count_inliers_1to1(U_val, Wt, tau_rms, sin_tau=self.SIN_TAU_REFINE)
            rms_arcsec = rms_val
            rms_px = rms_arcsec / s0 if s0 > 0 else 0.0
            matched_count = int(np.sum(mask_val))

        cos_t, sin_t = math.cos(cur_theta), math.sin(cur_theta)
        affine = (cur_tx, cur_s * cos_t, -cur_s * sin_t,
                  cur_ty, cur_s * sin_t, cur_s * cos_t)

        return VectorMatchResult(
            center_ra=cur_ra, center_dec=cur_dec,
            original_ra=ra0, original_dec=dec0,
            rotation_deg=rotation_deg, scale_arcsec_px=s_final,
            flip_mode=cur_flip, matched_count=matched_count,
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

"""
Vector Match - 基于向量组对齐的Plate Solving算法

功能:
    彻底抛弃三角形匹配，使用RANSAC求解相似变换实现粗匹配。
    核心思路：将图像检测星点和星表(Gaia)参考星分别构建为以各自中心为原点
    的二维向量组(角秒空间)，通过RANSAC求解两组向量间的相似变换
    (旋转+统一缩放+平移)，一次性解出中心偏移、旋转角、像素尺度修正和翻转模式。

算法流程:
    Step 1: 像素尺度s0和FOV计算 (焦距+像元尺寸→角秒/像素)
    Step 2: 亮星选取 (饱和星≥50用全部饱和星，否则饱和+亮星共100颗)
    Step 3: Gaia锥形查询+极限星等二分法 (饱和≥50→1.5×饱和星; 饱和<50→150颗)
    Step 4: 向量组构建 (图像向量U: 像素偏移×s0→角秒, Y取反;
                        星表向量W: gnomonic投影→角秒)
    Step 5: 4种翻转模式独立RANSAC匹配 (粗匹配tau_coarse=2.5×s0)
    Step 6: Coarse-to-fine精化 (在内点上用tau_fine=1.0×s0重新RANSAC)
    Step 7: 归一化打分选择最佳翻转模式
    Step 8: WCS参数提取 (中心偏移/旋转角/像素尺度/仿射6参数)
    Step 9: 中心修正+精化 (平移量→新中心→重新投影→RANSAC)

关键设计决策:
    - Y轴翻转: 图像像素Y向下, 天球Dec向上, 图像向量Y分量必须取反
    - 两阶段RANSAC: 纯随机2点采样正确配对概率≈8×10⁻⁶, 改用KDTree候选+RANSAC
    - Coarse-to-fine: 粗匹配找初始变换, 精化用更小tau过滤, RMS从1.5px降到0.6px
    - 单次中心修正: 迭代重投影会导致偏移累积发散, 改为单次修正+精化

用途:
    替代旧版三角形匹配粗匹配模块，作为Plate Solve第一步。
    输入: 图像星点坐标+初始中心坐标+焦距/像元信息
    输出: VectorMatchResult (中心坐标/旋转角/像素尺度/翻转模式/匹配数/RMS/仿射参数)

依赖: numpy, scipy(scipy.spatial.cKDTree), ctypes(Gaia DLL)

详细算法分析: 参见同目录下 vector_match_analysis.md
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

logger = logging.getLogger("vector_match")

_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

_DEGTORAD = np.pi / 180.0
_RADTODEG = 180.0 / np.pi
_RADTOASEC = (180.0 / np.pi) * 3600.0
_ASECTORAD = np.pi / (180.0 * 3600.0)


class GaiaClientPy:
    """Gaia数据库Python客户端封装，通过ctypes调用gaia_client.dll

    封装了gaia_client.dll的C接口，提供Python级别的锥形查询功能。
    DLL搜索路径: 1) 模块同级目录 2) 项目lib/gaia_xpsd_client目录
    使用gaia_client_cone_search_for_solver接口，返回分离的ra/dec/mag数组。

    用法:
        with GaiaClientPy(data_dir, db_type=2) as client:
            ra, dec, mag = client.cone_search(center_ra, center_dec, radius, mag_limit)
    """

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

    def cone_search(
        self, center_ra: float, center_dec: float, radius_deg: float, mag_limit: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """锥形查询，返回 (ra, dec, mag) 三个numpy数组

        参数:
            center_ra: 中心赤经(度)
            center_dec: 中心赤纬(度)
            radius_deg: 查询半径(度)
            mag_limit: 极限星等(G波段)
        返回:
            (ra_array, dec_array, mag_array) 三个float64/float64/float64数组
            查询失败返回三个空数组
        """
        ra_ptr = ctypes.POINTER(ctypes.c_double)()
        dec_ptr = ctypes.POINTER(ctypes.c_double)()
        mag_ptr = ctypes.POINTER(ctypes.c_float)()
        n_stars = ctypes.c_int()
        ret = self._dll.gaia_client_cone_search_for_solver(
            self._handle, center_ra, center_dec, radius_deg, mag_limit,
            ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars),
        )
        if ret != 0:
            logger.warning("Gaia锥形查询失败: ret=%d", ret)
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


def gnomonic_forward(
    ra_deg: np.ndarray, dec_deg: np.ndarray, ra0_deg: float, dec0_deg: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准gnomonic(TAN)正投影: (RA, Dec)度 → (ξ, η)角秒

    将天球坐标投影到以(ra0, dec0)为切点的切平面上。
    ξ 向东增加(对应RA增加方向), η 向北增加(对应Dec增加方向)。

    公式:
        cosc = sin(δ₀)sin(δ) + cos(δ₀)cos(δ)cos(Δα)
        ξ = cos(δ)sin(Δα) / cosc
        η = (cos(δ₀)sin(δ) - sin(δ₀)cos(δ)cos(Δα)) / cosc

    参数:
        ra_deg: 源赤经数组(度)
        dec_deg: 源赤纬数组(度)
        ra0_deg: 投影中心赤经(度)
        dec0_deg: 投影中心赤纬(度)
    返回:
        (xi_asec, eta_asec, valid): 角秒偏移 + 有效性掩码
        valid=False表示源在对径点附近，投影无效
    """
    ra = np.asarray(ra_deg, dtype=np.float64) * _DEGTORAD
    dec = np.asarray(dec_deg, dtype=np.float64) * _DEGTORAD
    ra0 = ra0_deg * _DEGTORAD
    dec0 = dec0_deg * _DEGTORAD
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)
    delta_ra = ra - ra0
    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    cos_delta_ra = np.cos(delta_ra)
    cosc = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_delta_ra
    valid = cosc > 1e-10
    cosc_safe = np.where(valid, cosc, 1.0)
    xi = cos_dec * np.sin(delta_ra) / cosc_safe
    eta = (cos_dec0 * sin_dec - sin_dec0 * cos_dec * cos_delta_ra) / cosc_safe
    xi = np.where(valid, xi, 0.0) * _RADTOASEC
    eta = np.where(valid, eta, 0.0) * _RADTOASEC
    return xi, eta, valid


def gnomonic_inverse(
    xi_asec: np.ndarray, eta_asec: np.ndarray, ra0_deg: float, dec0_deg: float
) -> Tuple[np.ndarray, np.ndarray]:
    """标准gnomonic(TAN)逆投影: (ξ, η)角秒 → (RA, Dec)度

    将切平面坐标逆投影回天球坐标。

    公式:
        ρ = sqrt(ξ² + η²)
        c = arctan(ρ)
        δ = arcsin(cos(c)sin(δ₀) + η·sin(c)cos(δ₀)/ρ)
        α = α₀ + arctan2(ξ·sin(c), ρ·cos(δ₀)cos(c) - η·sin(δ₀)sin(c))

    参数:
        xi_asec: 切平面X偏移(角秒, 东向正)
        eta_asec: 切平面Y偏移(角秒, 北向正)
        ra0_deg: 投影中心赤经(度)
        dec0_deg: 投影中心赤纬(度)
    返回:
        (ra_deg, dec_deg): 天球坐标(度)
    """
    xi = np.asarray(xi_asec, dtype=np.float64) * _ASECTORAD
    eta = np.asarray(eta_asec, dtype=np.float64) * _ASECTORAD
    ra0 = ra0_deg * _DEGTORAD
    dec0 = dec0_deg * _DEGTORAD
    sin_dec0 = np.sin(dec0)
    cos_dec0 = np.cos(dec0)
    rho = np.sqrt(xi ** 2 + eta ** 2)
    c = np.arctan(rho)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    rho_safe = np.where(rho > 1e-15, rho, 1.0)
    dec = np.arcsin(cos_c * sin_dec0 + eta * sin_c * cos_dec0 / rho_safe)
    ra = ra0 + np.arctan2(xi * sin_c, rho_safe * cos_dec0 * cos_c - eta * sin_dec0 * sin_c)
    return ra * _RADTODEG, dec * _RADTODEG


def bisection_mag_limit(
    gaia_client: GaiaClientPy,
    center_ra: float,
    center_dec: float,
    radius_deg: float,
    target_count: int,
    mag_low: float = 6.0,
    mag_high: float = 22.0,
    tolerance: float = 0.1,
) -> Tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    """二分法确定Gaia查询的极限星等，使返回星数接近target_count

    在[mag_low, mag_high]范围内二分搜索，使查询返回的星数落在
    [target_count, 1.1×target_count]区间内。最多30次迭代，收敛
    条件为星等区间宽度≤tolerance。

    参数:
        gaia_client: Gaia客户端实例
        center_ra, center_dec: 查询中心(度)
        radius_deg: 查询半径(度)
        target_count: 目标星数
        mag_low, mag_high: 星等搜索范围
        tolerance: 收敛容差(星等)
    返回:
        (极限星等, 实际星数, ra数组, dec数组, mag数组)
    """
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


@dataclass
class VectorMatchResult:
    """向量匹配结果数据结构

    属性:
        center_ra: 修正后的中心赤经(度)
        center_dec: 修正后的中心赤纬(度)
        rotation_deg: 旋转角(度), 正值=逆时针(从星表到图像)
        scale_arcsec_px: 最终像素尺度(角秒/像素) = s0 × s
        flip_mode: 翻转模式 (0=无, 1=X翻转, 2=Y翻转, 3=XY翻转)
        matched_count: RANSAC内点数
        rms_px: 匹配RMS(像素)
        rms_arcsec: 匹配RMS(角秒)
        affine: 仿射6参数 (a0,a1,a2,b0,b1,b2), 用于天球→像素投影
            u = a0 + a1*w'_x + a2*w'_y  (图像X向量, 角秒, 东向)
            v = b0 + b1*w'_x + b2*w'_y  (图像Y向量, 角秒, 北向)
            其中w'是翻转后的星表向量
    """
    center_ra: float
    center_dec: float
    rotation_deg: float
    scale_arcsec_px: float
    flip_mode: int
    matched_count: int
    rms_px: float
    rms_arcsec: float
    affine: tuple


def _build_image_vectors(
    img_x: np.ndarray,
    img_y: np.ndarray,
    img_flux: np.ndarray,
    img_saturated: np.ndarray,
    scale0: float,
    width: int,
    height: int,
) -> Tuple[np.ndarray, int, int]:
    """构建图像向量组U(角秒空间, 天球convention: Y向上=Dec增加方向)

    亮星选取策略:
        - 饱和星≥50: 仅用全部饱和星 (质心稳定, 信噪比高)
        - 饱和星<50: 饱和星 + 最亮(100-n_sat)颗正常星, 共100颗

    坐标变换:
        ux = (x - cx) × s0    # 像素偏移→角秒偏移, X方向不变
        uy = -(y - cy) × s0   # Y取反: 图像Y向下→天球Y向上

    参数:
        img_x, img_y: 全部检测星点坐标(像素, Y向下)
        img_flux: 星点亮度
        img_saturated: 饱和标记(1=饱和, 0=正常)
        scale0: 初始像素尺度(角秒/像素)
        width, height: 图像宽高(像素)
    返回:
        (U, N_img, n_sat): U shape=(N_img, 2) 角秒, N_img=选中的亮星数, n_sat=饱和星数
    """
    cx = width / 2.0
    cy = height / 2.0
    n_sat = int(np.sum(img_saturated))
    if n_sat >= 50:
        # 饱和星≥50: 全部饱和星
        mask = img_saturated.astype(bool)
        sel_x = img_x[mask]
        sel_y = img_y[mask]
    else:
        # 饱和星<50: 饱和星 + top亮星共100颗
        mask_sat = img_saturated.astype(bool)
        n_normal = 100 - n_sat
        normal_idx = np.where(~mask_sat)[0]
        if len(normal_idx) > 0 and n_normal > 0:
            sorted_idx = normal_idx[np.argsort(-img_flux[normal_idx])]
            top_normal = sorted_idx[:n_normal]
            sel_idx = np.concatenate([np.where(mask_sat)[0], top_normal])
        else:
            sel_idx = np.where(mask_sat)[0]
        sel_x = img_x[sel_idx]
        sel_y = img_y[sel_idx]
    n_img = len(sel_x)
    if n_img < 2:
        return np.empty((0, 2)), 0, n_sat
    ux = (sel_x - cx) * scale0
    uy = -(sel_y - cy) * scale0
    U = np.column_stack([ux, uy])
    return U, n_img, n_sat


def _build_catalog_vectors(
    cat_ra: np.ndarray, cat_dec: np.ndarray, ra0: float, dec0: float
) -> np.ndarray:
    """构建星表向量组W(角秒空间, 天球convention: ξ=东增, η=北增)

    使用gnomonic投影将Gaia星的天球坐标投影到以(ra0, dec0)为切点的
    切平面上，输出角秒偏移。投影无效的点(对径点附近)被排除。

    参数:
        cat_ra, cat_dec: Gaia星的天球坐标(度)
        ra0, dec0: 投影中心(度), 通常为FITS头中的参考坐标
    返回:
        W: shape=(M_valid, 2) 角秒偏移
    """
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, ra0, dec0)
    W = np.column_stack([xi[valid], eta[valid]])
    return W


def _apply_flip(W: np.ndarray, mode: int) -> np.ndarray:
    """对星表向量组施加翻转模式(在RANSAC前预处理)

    4种翻转模式覆盖所有可能的镜像关系:
        mode & 1 (bit 0): X翻转 (东西镜像, ξ取反)
        mode & 2 (bit 1): Y翻转 (南北镜像, η取反)

    翻转在星表侧而非图像侧施加，这样RANSAC中的旋转矩阵R
    始终满足det(R)>0(纯旋转)，简化相似变换求解。

    参数:
        W: 星表向量组 shape=(M, 2)
        mode: 翻转模式 0-3
    返回:
        Wf: 翻转后的向量组(副本)
    """
    Wf = W.copy()
    if mode & 1:
        Wf[:, 0] = -Wf[:, 0]
    if mode & 2:
        Wf[:, 1] = -Wf[:, 1]
    return Wf


def _solve_similarity_2pt(
    u_a: np.ndarray, u_b: np.ndarray,
    w_a: np.ndarray, w_b: np.ndarray,
) -> Optional[Tuple[float, float, float, float]]:
    """从2对匹配点求解相似变换 u = s×R(θ)×w + t

    相似变换4参数的闭式求解:
        1. 缩放因子: s = ‖u_a - u_b‖ / ‖w_a - w_b‖
           - 若s∉[0.9, 1.1]则丢弃(物理约束: 焦距误差<10%)
        2. 旋转角: θ = angle(u_a-u_b) - angle(w_a-w_b)
        3. 平移量: t = u_a - s×R(θ)×w_a

    参数:
        u_a, u_b: 图像向量组中的2个点(角秒)
        w_a, w_b: 星表向量组中的2个对应点(角秒, 已翻转)
    返回:
        (s, theta, tx, ty) 或 None(缩放超范围/零向量)
    """
    du = u_a - u_b
    dw = w_a - w_b
    norm_du = np.sqrt(du[0] ** 2 + du[1] ** 2)
    norm_dw = np.sqrt(dw[0] ** 2 + dw[1] ** 2)
    if norm_dw < 1e-12 or norm_du < 1e-12:
        return None
    s = norm_du / norm_dw
    if s < 0.9 or s > 1.1:
        return None
    angle_u = math.atan2(du[1], du[0])
    angle_w = math.atan2(dw[1], dw[0])
    theta = angle_u - angle_w
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    R_wa_x = cos_t * w_a[0] - sin_t * w_a[1]
    R_wa_y = sin_t * w_a[0] + cos_t * w_a[1]
    tx = u_a[0] - s * R_wa_x
    ty = u_a[1] - s * R_wa_y
    return s, theta, tx, ty


def _apply_similarity(W: np.ndarray, s: float, theta: float, tx: float, ty: float) -> np.ndarray:
    """将相似变换 u = s×R(θ)×w + t 应用到整个向量组

    numpy向量化实现，一次性变换所有M个点。

    参数:
        W: 星表向量组 shape=(M, 2) 角秒
        s, theta, tx, ty: 相似变换参数
    返回:
        Wt: 变换后的向量组 shape=(M, 2) 角秒
    """
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    Wt_x = s * (cos_t * W[:, 0] - sin_t * W[:, 1]) + tx
    Wt_y = s * (sin_t * W[:, 0] + cos_t * W[:, 1]) + ty
    return np.column_stack([Wt_x, Wt_y])


def _find_putative_correspondences(
    U: np.ndarray, W: np.ndarray, radius: float
) -> np.ndarray:
    """用KDTree建立候选对应关系(RANSAC阶段1)

    对U中每个点在W中找最近邻，距离<radius的保留为候选对。
    这些候选对包含大量错误配对(离群点)，但正确配对也包含其中。
    RANSAC阶段2负责从候选对中筛选出正确的子集。

    为什么需要候选对应而非纯随机采样:
        U和W的索引没有对应关系，纯随机2点采样的正确配对概率
        ≈(N_correct/(N×M))²≈8×10⁻⁶，200次迭代几乎不可能命中。
        KDTree候选将正确配对概率提升到≈(N_correct/P)²≈0.65。

    参数:
        U: 图像向量组 shape=(N, 2)
        W: 星表向量组(已翻转) shape=(M, 2)
        radius: 候选搜索半径(角秒)
    返回:
        pairs: shape=(P, 2), pairs[i]=(u_idx, w_idx)
    """
    tree = cKDTree(W)
    dists, idxs = tree.query(U, k=1)
    mask = dists < radius
    u_idx = np.where(mask)[0]
    w_idx = idxs[mask]
    return np.column_stack([u_idx, w_idx])


def _count_inliers_1to1(
    U: np.ndarray, Wt: np.ndarray, tau: float
) -> Tuple[int, float, np.ndarray]:
    """1对1互斥内点统计(RANSAC内点评估)

    流程:
        1. 构建KDTree(Wt)
        2. 对U中每个点找Wt中最近邻，记录距离
        3. 按距离升序贪心分配: 距离小者优先匹配，每个Wt点只能被匹配一次
        4. 距离<τ的标记为内点

    1对1互斥避免多对一匹配(多个U点匹配同一个Wt点)导致虚假高分。
    贪心策略(按距离升序)保证最近邻匹配的全局最优性(贪心意义下)。

    参数:
        U: 图像向量组 shape=(N, 2)
        Wt: 变换后的星表向量组 shape=(M, 2)
        tau: 内点距离阈值(角秒)
    返回:
        (n_inliers, rms, inlier_mask): 内点数, RMS(角秒), U的内点掩码
    """
    tree = cKDTree(Wt)
    dists, idxs = tree.query(U, k=1)
    inlier_mask = dists < tau
    used_cat = set()
    final_mask = np.zeros(len(U), dtype=bool)
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
    inlier_dists = dists[final_mask]
    rms = float(np.sqrt(np.mean(inlier_dists ** 2)))
    return n_inliers, rms, final_mask


def _ransac_similarity(
    U: np.ndarray,
    W: np.ndarray,
    tau: float,
    K: int,
    min_inliers: int,
    rng: np.random.Generator,
    candidate_radius: float = 0.0,
) -> Tuple[float, float, float, float, int, float, np.ndarray]:
    """两阶段RANSAC求解相似变换 u = s×R(θ)×w + t

    阶段1 - KDTree候选对应:
        对U中每个点在W中找最近邻(距离<candidate_radius)，建立候选对。
        候选对包含正确配对和大量错误配对。

    阶段2 - RANSAC采样与过滤:
        从候选对中随机选2对，求解相似变换4参数(s,θ,tx,ty)。
        缩放因子s∈[0.9,1.1]的物理约束过滤大部分错误假设。
        用1对1互斥内点统计评估变换质量。
        保留得分(n_inliers - λ×RMS)最高的变换。

    参数:
        U: 图像向量组 shape=(N, 2) 角秒
        W: 星表向量组(已翻转) shape=(M, 2) 角秒
        tau: 内点距离阈值(角秒)
        K: 最大迭代次数
        min_inliers: 最少内点数
        rng: numpy随机数生成器
        candidate_radius: 候选对应搜索半径(角秒), 0则自动计算
    返回:
        (s, theta, tx, ty, n_inliers, rms, inlier_mask)
    """
    N = len(U)
    M = len(W)
    if N < 2 or M < 2:
        return 0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    if candidate_radius <= 0:
        candidate_radius = tau * 50.0  # 默认候选半径基于tau

    pairs = _find_putative_correspondences(U, W, candidate_radius)
    P = len(pairs)
    logger.debug("候选对应: %d对 (半径=%.1f角秒)", P, candidate_radius)
    if P < 2:
        return 0, 0, 0, 0, 0, 0, np.zeros(N, dtype=bool)

    best_score = -1e30
    best_s = 1.0
    best_theta = 0.0
    best_tx = 0.0
    best_ty = 0.0
    best_n_inliers = 0
    best_rms = 0.0
    best_mask = np.zeros(N, dtype=bool)

    actual_K = min(K, P * (P - 1) // 2)

    for _ in range(actual_K):
        pi, pj = rng.choice(P, 2, replace=False)
        u_idx_a, w_idx_a = pairs[pi]
        u_idx_b, w_idx_b = pairs[pj]

        result = _solve_similarity_2pt(
            U[u_idx_a], U[u_idx_b], W[w_idx_a], W[w_idx_b]
        )
        if result is None:
            continue
        s, theta, tx, ty = result

        Wt = _apply_similarity(W, s, theta, tx, ty)
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


def _compute_normalized_score(n_inliers: int, rms: float, N_img: int, M: int, tau: float) -> float:
    """计算归一化得分，用于跨模式比较

    score_norm = (n_inliers / min(N_img, M)) × (1 - RMS / τ)

    第一项: 内点率, 归一化到[0,1]
    第二项: 精度率, 归一化到[0,1] (RMS=0时为1, RMS=τ时为0)
    综合得分范围[0,1], 越大越好。

    参数:
        n_inliers: 内点数
        rms: 内点RMS(角秒)
        N_img: 图像侧亮星数
        M: 星表侧星数
        tau: 内点距离阈值(角秒)
    返回:
        归一化得分 [0, 1]
    """
    denom = min(N_img, M)
    if denom <= 0:
        return 0.0
    if tau <= 0:
        return 0.0
    return (n_inliers / denom) * (1.0 - rms / tau)


class VectorMatch:
    """基于向量组对齐的Plate Solving算法

    使用RANSAC求解相似变换，4种翻转模式独立匹配，coarse-to-fine精化。

    用法:
        with VectorMatch(gaia_data_dir, db_type=2) as vm:
            result = vm.solve(img_x, img_y, img_flux, img_saturated,
                              center_ra, center_dec, focal_length_mm,
                              pixel_size_um, width, height)
            if result:
                print(f"中心: RA={result.center_ra}, Dec={result.center_dec}")
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        """初始化

        参数:
            gaia_data_dir: Gaia数据库目录路径
            db_type: 数据库类型 (0=默认, 2=XPSD格式)
        """
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._closed = False
        self._rng = np.random.default_rng(42)
        logger.info("VectorMatch初始化: data_dir=%s, db_type=%d", gaia_data_dir, db_type)

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
        scale_arcsec_px: float = 0.0,
    ) -> Optional[VectorMatchResult]:
        """执行向量匹配plate solving

        完整流程:
            1. 计算像素尺度s0和FOV
            2. 构建图像向量组U(亮星选取+Y取反)
            3. Gaia查询+二分法极限星等
            4. 4种翻转模式独立RANSAC(coarse)
            5. Coarse-to-fine精化(fine)
            6. 归一化打分选最佳模式
            7. WCS参数提取+中心修正+精化

        参数:
            img_x, img_y: 图像星点坐标(像素, Y向下)
            img_flux: 星点亮度
            img_saturated: 饱和标记(1=饱和, 0=正常)
            center_ra, center_dec: 初始中心坐标(度)
            focal_length_mm: 焦距(mm)
            pixel_size_um: 像元尺寸(μm)
            width, height: 图像宽高(像素)
            scale_arcsec_px: 像素尺度(角秒/像素), 0则自动计算
        返回:
            VectorMatchResult 或 None(失败)
        """
        if scale_arcsec_px > 0:
            s0 = scale_arcsec_px
        else:
            s0 = 206.265 * pixel_size_um / focal_length_mm
        logger.info("初始像素尺度 s0=%.4f 角秒/像素", s0)

        fov_diag = math.sqrt(width ** 2 + height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        logger.info("FOV对角线=%.2f度, 查询半径=%.2f度", fov_diag, radius_deg)

        U, N_img, n_sat = _build_image_vectors(
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

        # Gaia星点选择策略: 饱和星≥50 → 1.5×饱和星; 饱和星<50 → 150颗
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

        tau_coarse = max(1.0, 2.5 * s0)
        tau_fine = max(0.5, 1.0 * s0)
        K = 500
        min_inliers = max(5, int(N_img * 0.2))
        # 候选半径 = FOV对角线的30% (角秒)，确保足够大以覆盖初始坐标误差
        candidate_radius = fov_diag * 3600.0 * 0.3
        logger.info("RANSAC参数: tau_coarse=%.2f tau_fine=%.2f K=%d min_inliers=%d candidate_radius=%.1f角秒",
                     tau_coarse, tau_fine, K, min_inliers, candidate_radius)

        best_mode = -1
        best_norm_score = -1.0
        best_result = None

        for mode in range(4):
            W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
            Wf = _apply_flip(W, mode)
            logger.info("翻转模式%d: 星表向量组 %d 颗", mode, len(Wf))

            s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_similarity(
                U, Wf, tau_coarse, K, min_inliers, self._rng, candidate_radius
            )
            if n_inliers < min_inliers:
                logger.info("  模式%d: 内点不足 n=%d < min=%d", mode, n_inliers, min_inliers)
                continue

            if n_inliers >= 3:
                inlier_U = U[inlier_mask]
                Wt = _apply_similarity(Wf, s, theta, tx, ty)
                tree = cKDTree(Wt)
                dists, idxs = tree.query(inlier_U, k=1)
                inlier_W = Wf[idxs]
                pairs_arr = np.column_stack([inlier_U, inlier_W])
                n_pairs = len(pairs_arr)
                if n_pairs >= 2:
                    best_s2 = 1.0
                    best_theta2 = 0.0
                    best_tx2 = 0.0
                    best_ty2 = 0.0
                    best_n2 = 0
                    best_rms2 = 0.0
                    best_mask2 = np.zeros(len(U), dtype=bool)
                    for _ in range(300):
                        pi, pj = self._rng.choice(n_pairs, 2, replace=False)
                        ua = pairs_arr[pi, :2]
                        ub = pairs_arr[pj, :2]
                        wa = pairs_arr[pi, 2:]
                        wb = pairs_arr[pj, 2:]
                        res2 = _solve_similarity_2pt(ua, ub, wa, wb)
                        if res2 is None:
                            continue
                        s2, theta2, tx2, ty2 = res2
                        Wt2 = _apply_similarity(Wf, s2, theta2, tx2, ty2)
                        n2, rms2, mask2 = _count_inliers_1to1(U, Wt2, tau_fine)
                        if n2 > best_n2 or (n2 == best_n2 and rms2 < best_rms2):
                            best_s2 = s2
                            best_theta2 = theta2
                            best_tx2 = tx2
                            best_ty2 = ty2
                            best_n2 = n2
                            best_rms2 = rms2
                            best_mask2 = mask2
                    if best_n2 >= min_inliers:
                        s = best_s2
                        theta = best_theta2
                        tx = best_tx2
                        ty = best_ty2
                        n_inliers = best_n2
                        rms = best_rms2
                        inlier_mask = best_mask2

            norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)
            logger.info(
                "  模式%d: s=%.4f theta=%.2f° tx=%.2f ty=%.2f n=%d rms=%.3f norm_score=%.4f",
                mode, s, math.degrees(theta), tx, ty, n_inliers, rms, norm_score,
            )

            if norm_score > best_norm_score:
                best_norm_score = norm_score
                best_mode = mode
                best_result = (s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf)

        if best_mode < 0 or best_norm_score < 0.10:
            logger.warning("所有模式匹配失败: best_norm_score=%.4f", best_norm_score)
            return None

        s, theta, tx, ty, n_inliers, rms, inlier_mask, Wf = best_result
        logger.info("最佳模式=%d, 归一化得分=%.4f", best_mode, best_norm_score)

        result = self._extract_wcs_and_converge(
            s, theta, tx, ty, best_mode, s0,
            center_ra, center_dec, width, height,
            U, Wf, inlier_mask, tau_coarse, N_img, M,
            cat_ra, cat_dec, cat_mag, radius_deg,
        )
        return result

    def _extract_wcs_and_converge(
        self,
        s: float,
        theta: float,
        tx: float,
        ty: float,
        flip_mode: int,
        s0: float,
        ra0: float,
        dec0: float,
        width: int,
        height: int,
        U: np.ndarray,
        Wf: np.ndarray,
        inlier_mask: np.ndarray,
        tau: float,
        N_img: int,
        M: int,
        cat_ra: np.ndarray,
        cat_dec: np.ndarray,
        cat_mag: np.ndarray,
        radius_deg: float,
    ) -> VectorMatchResult:
        """从变换参数提取WCS参数，并执行中心修正+精化

        流程:
            1. 平移量→中心坐标修正 (考虑赤经收缩因子cos(δ₀))
            2. 以新中心重新gnomonic投影Gaia星
            3. 施加翻转模式
            4. 用tau_fine重新RANSAC精化
            5. 计算最终RMS
            6. 提取仿射6参数

        为什么只做一次修正而非迭代:
            实测发现迭代重投影会导致偏移量累积发散。
            原因: 中心修正后重新投影改变了向量组形状，
            RANSAC可能找到不同的(错误的)匹配。
            单次修正已将中心精度提高到亚角秒级别。
        """
        cur_ra = ra0
        cur_dec = dec0
        cur_s = s
        cur_theta = theta
        cur_tx = tx
        cur_ty = ty
        cur_flip = flip_mode

        min_inliers = max(5, int(N_img * 0.2))

        cos_d0 = math.cos(cur_dec * _DEGTORAD)
        if abs(cos_d0) < 1e-10:
            cos_d0 = 1e-10
        delta_ra = cur_tx / (cos_d0 * 3600.0)
        delta_dec = cur_ty / 3600.0
        cur_ra = cur_ra + delta_ra
        cur_dec = cur_dec + delta_dec
        logger.info("中心修正: ΔRA=%.6f° ΔDec=%.6f° → RA=%.6f Dec=%.6f",
                     delta_ra, delta_dec, cur_ra, cur_dec)

        iter_candidate_radius = tau * 50.0  # 精化阶段候选半径基于tau
        W_new = _build_catalog_vectors(cat_ra, cat_dec, cur_ra, cur_dec)
        Wf_new = _apply_flip(W_new, cur_flip)
        tau_refine = max(0.5, 1.0 * s0)
        s2, theta2, tx2, ty2, n2, rms2, mask2 = _ransac_similarity(
            U, Wf_new, tau_refine, 500, min_inliers, self._rng, iter_candidate_radius
        )
        if n2 >= min_inliers:
            cur_s = s2
            cur_theta = theta2
            cur_tx = tx2
            cur_ty = ty2
            inlier_mask = mask2
            Wf = Wf_new
            logger.info("  精化: s=%.4f theta=%.2f° n=%d rms=%.3f",
                        s2, math.degrees(theta2), n2, rms2)

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

        cos_t = math.cos(cur_theta)
        sin_t = math.sin(cur_theta)
        a0 = cur_tx
        a1 = cur_s * cos_t
        a2 = -cur_s * sin_t
        b0 = cur_ty
        b1 = cur_s * sin_t
        b2 = cur_s * cos_t
        affine = (a0, a1, a2, b0, b1, b2)

        n_matched = int(np.sum(inlier_mask))

        return VectorMatchResult(
            center_ra=cur_ra,
            center_dec=cur_dec,
            rotation_deg=rotation_deg,
            scale_arcsec_px=s_final,
            flip_mode=cur_flip,
            matched_count=n_matched,
            rms_px=rms_px,
            rms_arcsec=rms_arcsec,
            affine=affine,
        )

    def solve_with_file(self, file_path: str) -> Optional[VectorMatchResult]:
        """从图像文件读取并执行plate solving

        自动从FITS头提取: 中心坐标(WCS/OBJCTRA/OBJCTDEC)、
        焦距(FOCALLEN)、像元尺寸(XPIXSZ)、像素尺度(已有WCS时)。
        然后调用solve()执行向量匹配。

        参数:
            file_path: 图像文件路径(FITS/XISF)
        返回:
            VectorMatchResult 或 None(失败)
        """
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "astro_image_io", "python"))
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "star_detector", "python"))
        from astro_image_io import ImageReader
        from star_detector import StarDetector

        reader = ImageReader()
        img = reader.read(file_path)

        detector = StarDetector()
        coords, fluxes, saturated = detector.detect_ex(img.data)

        if len(coords) == 0:
            logger.error("未检测到星点: %s", file_path)
            return None

        img_x = np.array([c[0] for c in coords], dtype=np.float64)
        img_y = np.array([c[1] for c in coords], dtype=np.float64)
        img_flux = np.array(fluxes, dtype=np.float64)
        img_saturated = np.array(saturated, dtype=np.int32)

        meta = img.metadata
        center_ra = 0.0
        center_dec = 0.0
        focal_length = 0.0
        pixel_size = 0.0
        scale = 0.0

        if meta.wcs and meta.wcs.has_wcs:
            center_ra = meta.wcs.crval1
            center_dec = meta.wcs.crval2
            scale = meta.wcs.pixel_scale

        if meta.observation:
            if meta.observation.focallen is not None:
                focal_length = meta.observation.focallen
            if meta.observation.xpixsz is not None:
                pixel_size = meta.observation.xpixsz

        if scale > 0:
            s0 = scale
        elif focal_length > 0 and pixel_size > 0:
            s0 = 206.265 * pixel_size / focal_length
        else:
            logger.error("无法确定像素尺度: 缺少焦距/像元尺寸/WCS信息")
            return None

        if center_ra == 0.0 and center_dec == 0.0:
            for kw in img.keywords:
                name = kw.name.upper()
                if name in ("OBJCTRA", "RA"):
                    val = kw.value
                    if isinstance(val, str):
                        parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                        if len(parts) >= 3:
                            center_ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
                        else:
                            parts2 = val.split()
                            if len(parts2) >= 3:
                                center_ra = (float(parts2[0]) + float(parts2[1]) / 60 + float(parts2[2]) / 3600) * 15
                    else:
                        center_ra = float(val)
                elif name in ("OBJCTDEC", "DEC"):
                    val = kw.value
                    if isinstance(val, str):
                        parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                        if len(parts) >= 3:
                            sign = -1 if parts[0].startswith("-") else 1
                            center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
                        else:
                            parts2 = val.split()
                            if len(parts2) >= 3:
                                sign = -1 if parts2[0].startswith("-") else 1
                                center_dec = sign * (abs(float(parts2[0])) + float(parts2[1]) / 60 + float(parts2[2]) / 3600)
                    else:
                        center_dec = float(val)

        if center_ra == 0.0 and center_dec == 0.0:
            logger.error("无法确定初始中心坐标: 缺少WCS/OBJCTRA/OBJCTDEC信息")
            return None

        result = self.solve(
            img_x=img_x,
            img_y=img_y,
            img_flux=img_flux,
            img_saturated=img_saturated,
            center_ra=center_ra,
            center_dec=center_dec,
            focal_length_mm=focal_length,
            pixel_size_um=pixel_size,
            width=img.width,
            height=img.height,
            scale_arcsec_px=s0,
        )
        return result

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

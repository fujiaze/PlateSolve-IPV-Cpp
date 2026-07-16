"""
InitialWCS - 初始WCS生成模块

功能:
    PlateSolve第一步，生成初始WCS解
    算法核心参考siril atpmatch/Valdes 1995
    物理思路采用饱和星优先策略

用途:
    从图像星点和Gaia星表匹配，计算初始仿射变换参数
    支持4种翻转模式（正常/X翻转/Y翻转/XY翻转）
    迭代重投影收敛至亚角秒精度
"""

import os
import sys
import ctypes
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from scipy.spatial import cKDTree

logger = logging.getLogger("InitialWCS")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    logger.addHandler(_ch)

_mingw_bin = r"C:\msys64\mingw64\bin"
if os.path.isdir(_mingw_bin):
    os.environ["PATH"] = _mingw_bin + ";" + os.environ.get("PATH", "")
    try:
        os.add_dll_directory(_mingw_bin)
    except OSError:
        pass

AT_MATCH_RATIO = 0.9
AT_TRIANGLE_RADIUS = 0.002
AT_MATCH_RADIUS = 2.0
AT_MATCH_MAXDIST = 50.0
AT_MATCH_MINVOTES = 2
AT_MATCH_NSIGMA = 10.0
AT_MATCH_PERCENTILE = 0.35
AT_MATCH_REQUIRE = 3
AT_MATCH_REQUIRE_LINEAR = 3
AT_MATCH_STARTN_LINEAR = 6
AT_MATCH_MAXITER = 10
AT_MATCH_HALTSIGMA = 0.1
CONV_TOLERANCE = 0.01
MAX_REPROJ_TRIALS = 5
MIN_SAT_FOR_PRIORITY = 10
RETRY_COUNTS = [100, 200, 400, 800, 800]
RANSAC_ITER = 1000
RANSAC_THRESH_PX = 5.0
FLIP_MODES = {0: "正常", 1: "X翻转", 2: "Y翻转", 3: "XY翻转"}


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


@dataclass
class FlipMatchResult:
    flip_mode: int
    affine: tuple
    matched_count: int
    rms_arcsec: float
    img_indices: List[int] = field(default_factory=list)
    cat_indices: List[int] = field(default_factory=list)


def compute_pixel_scale(focal_mm: float, pixel_um: float) -> float:
    """计算像素尺度 (arcsec/px)
    公式: scale = 206.265 * pixel_um / focal_mm
    """
    return 206.265 * pixel_um / focal_mm


def compute_fov(scale_arcsec_px: float, width: int, height: int) -> Tuple[float, float, float]:
    """计算FOV
    返回: (fov_width_deg, fov_height_deg, fov_diag_deg)
    """
    fov_w = width * scale_arcsec_px / 3600.0
    fov_h = height * scale_arcsec_px / 3600.0
    fov_diag = np.sqrt(fov_w ** 2 + fov_h ** 2)
    return fov_w, fov_h, fov_diag


class GaiaClientPy:
    """Gaia数据库Python客户端封装
    通过ctypes调用gaia_client.dll
    """

    def __init__(self, data_dir: str, db_type: int = 0):
        dll_path = self._find_dll()
        if dll_path is None:
            raise FileNotFoundError("找不到gaia_client.dll")
        try:
            os.add_dll_directory(os.path.dirname(os.path.abspath(dll_path)))
        except OSError:
            pass
        self._dll = ctypes.CDLL(dll_path)
        self._setup_api()

        data_dir_bytes = data_dir.encode("utf-8")
        if db_type == 0:
            self._handle = self._dll.gaia_client_create(data_dir_bytes)
        else:
            self._handle = self._dll.gaia_client_create_ex(data_dir_bytes, db_type)

        if not self._handle:
            raise RuntimeError(f"创建Gaia客户端失败: {data_dir}")

        self._msvcrt = ctypes.CDLL("msvcrt.dll")
        logger.info("Gaia客户端初始化完成: %s (type=%d)", data_dir, db_type)

    def _find_dll(self) -> Optional[str]:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "gaia_client.dll"),
            os.path.join(os.path.dirname(__file__), "..", "gaia_xpsd_client", "gaia_client.dll"),
        ]
        project_root = r"F:\Astro dev\Astro CS Normalization Database"
        candidates.extend([
            os.path.join(project_root, "lib", "gaia_xpsd_client", "gaia_client.dll"),
            os.path.join(project_root, "lib", "plate_solve", "plate_solve.dll"),
        ])
        for p in candidates:
            p = os.path.abspath(p)
            if os.path.exists(p):
                return p
        return None

    def _setup_api(self):
        self._dll.gaia_client_create.argtypes = [ctypes.c_char_p]
        self._dll.gaia_client_create.restype = ctypes.c_void_p
        self._dll.gaia_client_create_ex.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._dll.gaia_client_create_ex.restype = ctypes.c_void_p
        self._dll.gaia_client_destroy.argtypes = [ctypes.c_void_p]
        self._dll.gaia_client_destroy.restype = None
        self._dll.gaia_client_cone_search_for_solver.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._dll.gaia_client_cone_search_for_solver.restype = ctypes.c_int

    def cone_search(self, center_ra: float, center_dec: float,
                    radius_deg: float, mag_limit: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """锥形搜索Gaia星表
        返回: (ra数组, dec数组, mag数组)
        """
        ra_ptr = ctypes.POINTER(ctypes.c_double)()
        dec_ptr = ctypes.POINTER(ctypes.c_double)()
        mag_ptr = ctypes.POINTER(ctypes.c_float)()
        n_stars = ctypes.c_int()

        ret = self._dll.gaia_client_cone_search_for_solver(
            self._handle, center_ra, center_dec, radius_deg, mag_limit,
            ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars)
        )

        if ret != 0:
            logger.warning("Gaia锥形搜索失败: ret=%d", ret)
            return np.array([]), np.array([]), np.array([])

        count = n_stars.value
        if count == 0:
            return np.array([]), np.array([]), np.array([])

        ra_arr = np.array([ra_ptr[i] for i in range(count)], dtype=np.float64)
        dec_arr = np.array([dec_ptr[i] for i in range(count)], dtype=np.float64)
        mag_arr = np.array([float(mag_ptr[i]) for i in range(count)], dtype=np.float64)

        self._msvcrt.free(ra_ptr)
        self._msvcrt.free(dec_ptr)
        self._msvcrt.free(mag_ptr)

        return ra_arr, dec_arr, mag_arr

    def close(self):
        if self._handle:
            self._dll.gaia_client_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


def bisection_mag_limit(gaia_client: GaiaClientPy, center_ra: float, center_dec: float,
                         radius_deg: float, target_count: int,
                         mag_low: float = 6.0, mag_high: float = 22.0,
                         tolerance: float = 0.1) -> Tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    """二分法迭代极限星等
    目标: 使Gaia星数在 [target_count, 1.2*target_count] 范围内
    返回: (极限星等, 星数, ra数组, dec数组, mag数组)
    """
    target_high = int(target_count * 1.2)
    ra_arr, dec_arr, mag_arr = np.array([]), np.array([]), np.array([])

    for _ in range(30):
        mid = (mag_low + mag_high) / 2.0
        ra, dec, mag = gaia_client.cone_search(center_ra, center_dec, radius_deg, mid)
        count = len(ra)
        if count < target_count:
            mag_low = mid
        elif count > target_high:
            mag_high = mid
        else:
            ra_arr, dec_arr, mag_arr = ra, dec, mag
            return mid, count, ra_arr, dec_arr, mag_arr
        if mag_high - mag_low < tolerance:
            break

    final_mag = (mag_low + mag_high) / 2.0
    ra_arr, dec_arr, mag_arr = gaia_client.cone_search(center_ra, center_dec, radius_deg, final_mag)
    return final_mag, len(ra_arr), ra_arr, dec_arr, mag_arr


def gaia_cone_search_with_bisection(gaia_client: GaiaClientPy, center_ra: float,
                                     center_dec: float, fov_diag_deg: float,
                                     det_count: int) -> Tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    """Gaia锥形查询 + 极限星等二分法
    查询半径 = fov_diag * 1.2 / 2
    目标星数 = det_count * 1.5
    返回: (极限星等, 星数, ra数组, dec数组, mag数组)
    """
    radius_deg = fov_diag_deg * 1.2 / 2.0
    target_count = int(det_count * 1.5)
    t0 = time.time()
    mag_limit, count, ra, dec, mag = bisection_mag_limit(
        gaia_client, center_ra, center_dec, radius_deg, target_count
    )
    elapsed = time.time() - t0
    logger.info("Gaia查询完成: %d星, 极限星等=%.2f, 耗时%.2fs", count, mag_limit, elapsed)
    return mag_limit, count, ra, dec, mag


def gnomonic_forward(ra_deg, dec_deg, ra0_deg, dec0_deg):
    """Gnomonic正投影: (RA, Dec) -> (xi, eta) 单位: 角秒
    参考 siril 的 project_catalog_stars
    """
    DEGTORAD = np.pi / 180.0
    RADTOASEC = 180.0 / np.pi * 3600.0

    ra_rad = np.asarray(ra_deg, dtype=np.float64) * DEGTORAD
    dec_rad = np.asarray(dec_deg, dtype=np.float64) * DEGTORAD
    ra0_rad = ra0_deg * DEGTORAD
    dec0_rad = dec0_deg * DEGTORAD

    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_dec0 = np.cos(dec0_rad)
    sin_dec0 = np.sin(dec0_rad)
    ra_diff = ra_rad - ra0_rad
    cos_ra_diff = np.cos(ra_diff)
    sin_ra_diff = np.sin(ra_diff)

    cos_c = sin_dec0 * sin_dec + cos_dec0 * cos_dec * cos_ra_diff
    valid = cos_c > 1e-10
    xi = np.zeros_like(cos_c)
    eta = np.zeros_like(cos_c)
    xi[valid] = cos_dec[valid] * sin_ra_diff[valid] / cos_c[valid]
    eta[valid] = (cos_dec0 * sin_dec[valid] - sin_dec0 * cos_dec[valid] * cos_ra_diff[valid]) / cos_c[valid]

    return xi * RADTOASEC, eta * RADTOASEC, valid


def gnomonic_inverse(xi_asec, eta_asec, ra0_deg, dec0_deg):
    """Gnomonic逆投影: (xi, eta) 角秒 -> (RA, Dec) 度
    参考 siril 的 apply_match
    """
    DEGTORAD = np.pi / 180.0
    ASECTODEG = 1.0 / 3600.0
    RADTODEG = 180.0 / np.pi

    delta_ra = (xi_asec * ASECTODEG) * DEGTORAD
    delta_dec = (eta_asec * ASECTODEG) * DEGTORAD
    r_dec = dec0_deg * DEGTORAD

    z = np.cos(r_dec) - delta_dec * np.sin(r_dec)
    alpha = np.arctan2(delta_ra, z) * RADTODEG + ra0_deg
    sin_r = np.sin(r_dec)
    cos_r = np.cos(r_dec)
    denom = np.sqrt(1.0 + delta_ra ** 2 + delta_dec ** 2)
    delta = np.arcsin((sin_r + delta_dec * cos_r) / denom) * RADTODEG

    alpha = alpha % 360.0
    return alpha, delta


def project_gaia_to_pixel(cat_ra, cat_dec, center_ra, center_dec, scale_arcsec_px):
    """批量投影Gaia星到像素坐标
    返回: (cat_px, cat_py, valid) 相对于图像中心的像素坐标
    """
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, center_ra, center_dec)
    cat_px = xi / scale_arcsec_px
    cat_py = eta / scale_arcsec_px
    return cat_px, cat_py, valid


def apply_flip(cat_px, cat_py, flip_mode):
    """对Gaia像素坐标应用翻转"""
    px = cat_px.copy()
    py = cat_py.copy()
    if flip_mode & 1:
        px = -px
    if flip_mode & 2:
        py = -py
    return px, py


def select_match_stars(img_x, img_y, img_flux, img_saturated, cat_x, cat_y, cat_mag):
    """饱和星优先选择匹配星点
    返回: (sel_img_x, sel_img_y, sel_cat_x, sel_cat_y, sel_img_idx, sel_cat_idx)
    """
    n_saturated = int(np.sum(img_saturated))

    img_order = np.argsort(-img_flux)
    cat_order = np.argsort(cat_mag)

    if n_saturated >= MIN_SAT_FOR_PRIORITY:
        sat_mask = img_saturated.astype(bool)
        sel_img_idx = np.where(sat_mask)[0]
        n_cat_select = min(int(len(sel_img_idx) * 1.5), len(cat_x))
        sel_cat_idx = cat_order[:n_cat_select]
    else:
        sat_idx = np.where(img_saturated.astype(bool))[0]
        n_norm_needed = 100 - len(sat_idx)
        norm_mask = ~img_saturated[img_order].astype(bool)
        norm_idx = img_order[norm_mask][:max(0, n_norm_needed)]
        sel_img_idx = np.concatenate([sat_idx, norm_idx]) if len(sat_idx) > 0 else norm_idx
        n_cat_select = min(150, len(cat_x))
        sel_cat_idx = cat_order[:n_cat_select]

    logger.info("选择匹配星: 图像%d颗(饱和%d), 星表%d颗",
                len(sel_img_idx), n_saturated, len(sel_cat_idx))

    return (img_x[sel_img_idx], img_y[sel_img_idx],
            cat_x[sel_cat_idx], cat_y[sel_cat_idx],
            sel_img_idx, sel_cat_idx)


def set_triangle(x, y, s1, s2, s3):
    """构建单个三角形 (严格参考psm_star_alignment.cpp的set_triangle)
    返回: (a_index, b_index, c_index, a_length, ba_ratio, ca_ratio)
    """
    d12 = np.sqrt((x[s1] - x[s2]) ** 2 + (y[s1] - y[s2]) ** 2)
    d23 = np.sqrt((x[s2] - x[s3]) ** 2 + (y[s2] - y[s3]) ** 2)
    d13 = np.sqrt((x[s1] - x[s3]) ** 2 + (y[s1] - y[s3]) ** 2)

    if d12 >= d23 and d12 >= d13:
        ai = s3
        a = d12
        if d23 >= d13:
            bi = s1
            b = d23
            ci = s2
            c = d13
        else:
            bi = s2
            b = d13
            ci = s1
            c = d23
    elif d23 > d12 and d23 >= d13:
        ai = s1
        a = d23
        if d12 > d13:
            bi = s3
            b = d12
            ci = s2
            c = d13
        else:
            bi = s2
            b = d13
            ci = s3
            c = d12
    else:
        ai = s2
        a = d13
        if d12 > d23:
            bi = s3
            b = d12
            ci = s1
            c = d23
        else:
            bi = s1
            b = d23
            ci = s3
            c = d12

    ba = b / a if a > 0.0 else 1.0
    ca = c / a if a > 0.0 else 1.0
    return ai, bi, ci, a, ba, ca


def build_triangles(x, y, nbright):
    """构建三角形 (严格参考psm_star_alignment.cpp的stars_to_triangles)
    返回: list of (a_idx, b_idx, c_idx, a_length, ba_ratio, ca_ratio)
    """
    n = min(nbright, len(x))
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ai, bi, ci, a_len, ba, ca = set_triangle(x, y, i, j, k)
                if ba <= AT_MATCH_RATIO:
                    triangles.append((ai, bi, ci, a_len, ba, ca))
    return triangles


def triangle_match(img_x, img_y, cat_x, cat_y, nobj, radius, scale_min, scale_max):
    """三角形匹配 (严格参考psm_star_alignment.cpp的siril_triangle_match)
    返回: (成功标志, 仿射6参数元组, 匹配对列表)
    """
    na = len(img_x)
    nb = len(cat_x)
    nbright = min(nobj, max(na, nb))
    if nbright < AT_MATCH_STARTN_LINEAR:
        logger.warning("三角形匹配: 星数不足 nbright=%d < %d", nbright, AT_MATCH_STARTN_LINEAR)
        return False, None, []

    t0 = time.time()
    tris_a = build_triangles(img_x, img_y, nbright)
    tris_b = build_triangles(cat_x, cat_y, nbright)

    tris_a.sort(key=lambda t: (t[4], t[5]))
    tris_b.sort(key=lambda t: (t[4], t[5]))

    logger.info("三角形匹配: nbright=%d tris_a=%d tris_b=%d", nbright, len(tris_a), len(tris_b))

    vote = np.zeros((nbright, nbright), dtype=np.int32)
    rad2 = radius * radius
    match_count = 0

    for tb in tris_b:
        tb_a_idx, tb_b_idx, tb_c_idx, tb_a_len, tb_ba, tb_ca = tb
        if tb_a_idx >= nbright or tb_b_idx >= nbright or tb_c_idx >= nbright:
            continue
        ba_min = tb_ba - radius
        ba_max = tb_ba + radius

        lo = 0
        hi = len(tris_a) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if tris_a[mid][4] < ba_min:
                lo = mid + 1
            else:
                hi = mid
        start = lo

        for idx in range(start, len(tris_a)):
            ta = tris_a[idx]
            ta_a_idx, ta_b_idx, ta_c_idx, ta_a_len, ta_ba, ta_ca = ta
            if ta_a_idx >= nbright or ta_b_idx >= nbright or ta_c_idx >= nbright:
                continue
            if ta_ba > ba_max:
                break
            dba = ta_ba - tb_ba
            dca = ta_ca - tb_ca
            if dba * dba + dca * dca < rad2:
                ratio = ta_a_len / tb_a_len if tb_a_len > 0 else 0
                if ratio < scale_min or ratio > scale_max:
                    continue
                vote[ta_a_idx][tb_a_idx] += 1
                vote[ta_b_idx][tb_b_idx] += 1
                vote[ta_c_idx][tb_c_idx] += 1
                match_count += 1

    logger.info("三角形匹配: %d 匹配对", match_count)

    pairs_with_votes = []
    for i in range(nbright):
        for j in range(nbright):
            if vote[i][j] > 0:
                pairs_with_votes.append((vote[i][j], i, j))

    pairs_with_votes.sort(key=lambda p: (-p[0], p[1], p[2]))

    valid_pairs = []
    for v, i, j in pairs_with_votes:
        if v < AT_MATCH_MINVOTES:
            break
        valid_pairs.append((v, i, j))

    logger.info("有效匹配对(>=%d票): %d", AT_MATCH_MINVOTES, len(valid_pairs))

    if len(valid_pairs) < AT_MATCH_STARTN_LINEAR:
        logger.warning("三角形匹配: 有效对不足 %d < %d", len(valid_pairs), AT_MATCH_STARTN_LINEAR)
        return False, None, []

    wv = [p[0] for p in valid_pairs]
    wia = [p[1] for p in valid_pairs]
    wib = [p[2] for p in valid_pairs]

    result = iter_trans(img_x, img_y, cat_x, cat_y, wv, wia, wib,
                        len(valid_pairs),
                        recalc=False, max_iter=AT_MATCH_MAXITER,
                        halt_sigma=AT_MATCH_HALTSIGMA)
    if result is None:
        logger.warning("三角形匹配: iter_trans失败")
        return False, None, []

    a0, a1, a2, b0, b1, b2, nr, final_wia, final_wib = result
    if nr < AT_MATCH_REQUIRE:
        logger.warning("三角形匹配: 匹配数不足 %d < %d", nr, AT_MATCH_REQUIRE)
        return False, None, []

    elapsed = time.time() - t0
    logger.info("三角形匹配成功: %d对, 耗时%.2fs", nr, elapsed)
    return True, (a0, a1, a2, b0, b1, b2), list(zip(final_wia, final_wib))


def gauss_matrix_3x3(m, v):
    """3x3高斯消元法 (参考psm_star_alignment.cpp的gauss_matrix_3x3)
    m: 3x3矩阵, v: 3维向量
    原地修改，返回解向量或None
    """
    m = np.array(m, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    for col in range(3):
        max_row = col
        for row in range(col + 1, 3):
            if abs(m[row][col]) > abs(m[max_row][col]):
                max_row = row
        if abs(m[max_row][col]) < 1e-12:
            return None
        if max_row != col:
            m[[col, max_row]] = m[[max_row, col]]
            v[[col, max_row]] = v[[max_row, col]]
        for row in range(col + 1, 3):
            factor = m[row][col] / m[col][col]
            m[row, col:] -= factor * m[col, col:]
            v[row] -= factor * v[col]

    for i in range(2, -1, -1):
        v[i] /= m[i][i]
        for j in range(i):
            v[j] -= m[j][i] * v[i]

    return v


def calc_trans_linear(stars_a_x, stars_a_y, stars_b_x, stars_b_y, wia, wib, npairs):
    """线性仿射变换计算 (参考psm_star_alignment.cpp的calc_trans_linear)
    求解: x' = a0 + a1*x + a2*y, y' = b0 + b1*x + b2*y
    返回: (a0, a1, a2, b0, b1, b2) 或 None
    """
    if npairs < AT_MATCH_REQUIRE_LINEAR:
        return None

    s00 = s10 = s01 = s20 = s11 = s02 = 0.0
    sx0 = sx1 = sx2 = sy0 = sy1 = sy2 = 0.0

    for i in range(npairs):
        x1 = stars_a_x[wia[i]]
        y1 = stars_a_y[wia[i]]
        x2 = stars_b_x[wib[i]]
        y2 = stars_b_y[wib[i]]
        s00 += 1
        s10 += x1
        s01 += y1
        s20 += x1 * x1
        s11 += x1 * y1
        s02 += y1 * y1
        sx0 += x2
        sx1 += x2 * x1
        sx2 += x2 * y1
        sy0 += y2
        sy1 += y2 * x1
        sy2 += y2 * y1

    m = [[s00, s10, s01],
         [s10, s20, s11],
         [s01, s11, s02]]

    vx = [sx0, sx1, sx2]
    sol_x = gauss_matrix_3x3(m, vx)
    if sol_x is None:
        return None
    a0, a1, a2 = sol_x[0], sol_x[1], sol_x[2]

    vy = [sy0, sy1, sy2]
    sol_y = gauss_matrix_3x3(m, vy)
    if sol_y is None:
        return None
    b0, b1, b2 = sol_y[0], sol_y[1], sol_y[2]

    return a0, a1, a2, b0, b1, b2


def iter_trans(stars_a_x, stars_a_y, stars_b_x, stars_b_y,
               votes, wia, wib, nbright,
               recalc=False, max_iter=AT_MATCH_MAXITER,
               halt_sigma=AT_MATCH_HALTSIGMA):
    """迭代精化变换 (参考psm_star_alignment.cpp的iter_trans)
    返回: (a0,a1,a2,b0,b1,b2,nr,final_wia,final_wib) 或 None
    """
    if nbright < AT_MATCH_REQUIRE_LINEAR:
        return None

    wia = list(wia)
    wib = list(wib)
    votes = list(votes)
    nr = nbright

    initial_pairs = nr if recalc else min(AT_MATCH_STARTN_LINEAR, nr)
    result = calc_trans_linear(stars_a_x, stars_a_y, stars_b_x, stars_b_y,
                                wia, wib, initial_pairs)
    if result is None:
        logger.warning("iter_trans: calc_trans_linear失败 (npairs=%d)", initial_pairs)
        return None
    a0, a1, a2, b0, b1, b2 = result
    logger.debug("iter_trans: 初始变换 a0=%.2f a1=%.6f a2=%.6f b0=%.2f b1=%.6f b2=%.6f",
                 a0, a1, a2, b0, b1, b2)

    max_dist2 = AT_MATCH_MAXDIST ** 2
    iters = 0
    is_ok = True

    while iters < max_iter:
        dist2 = []
        for i in range(nr):
            sx = stars_a_x[wia[i]]
            sy = stars_a_y[wia[i]]
            nx = a0 + a1 * sx + a2 * sy
            ny = b0 + b1 * sx + b2 * sy
            bx = stars_b_x[wib[i]]
            by = stars_b_y[wib[i]]
            d2 = (nx - bx) ** 2 + (ny - by) ** 2
            dist2.append(d2)

        new_nr = nr
        i = 0
        while i < new_nr:
            if dist2[i] > max_dist2:
                del votes[i]
                del wia[i]
                del wib[i]
                del dist2[i]
                new_nr -= 1
            else:
                i += 1
        nr = new_nr

        if nr < AT_MATCH_REQUIRE_LINEAR:
            logger.warning("iter_trans: max_dist剪枝后nr=%d < %d", nr, AT_MATCH_REQUIRE_LINEAR)
            is_ok = False
            break

        dist2_sorted = sorted(dist2)
        idx_sigma = int(np.floor(nr * AT_MATCH_PERCENTILE + 0.5))
        if idx_sigma >= nr:
            idx_sigma = nr - 1
        sigma = dist2_sorted[idx_sigma]
        logger.debug("iter_trans: iter=%d nr=%d sigma(dist2)=%.4f halt_sigma=%.4f",
                     iters, nr, sigma, halt_sigma)

        if sigma <= halt_sigma:
            is_ok = True

        nb_sigma = 0
        i = 0
        while i < nr:
            if dist2[i] > AT_MATCH_NSIGMA * sigma:
                del votes[i]
                del wia[i]
                del wib[i]
                del dist2[i]
                nr -= 1
                nb_sigma += 1
            else:
                i += 1

        if nr < AT_MATCH_REQUIRE_LINEAR:
            is_ok = False
            break

        if nb_sigma == 0 and all(d <= max_dist2 for d in dist2[:nr]):
            is_ok = True
            break

        result = calc_trans_linear(stars_a_x, stars_a_y, stars_b_x, stars_b_y,
                                    wia, wib, nr)
        if result is None:
            is_ok = False
            break
        a0, a1, a2, b0, b1, b2 = result
        iters += 1
        if is_ok:
            break

    if is_ok and nr >= AT_MATCH_REQUIRE:
        return a0, a1, a2, b0, b1, b2, nr, wia[:nr], wib[:nr]
    return None


def nearest_neighbor_match(pred_x, pred_y, cat_x, cat_y, radius):
    """近邻1对1匹配 (使用scipy KDTree)
    返回: [(img_idx, cat_idx), ...]
    """
    if len(pred_x) == 0 or len(cat_x) == 0:
        return []

    pred_pts = np.column_stack([pred_x, pred_y])
    cat_pts = np.column_stack([cat_x, cat_y])

    tree = cKDTree(cat_pts)
    dists, idxs = tree.query(pred_pts, k=1)

    valid = dists < radius
    raw_pairs = list(zip(np.where(valid)[0], idxs[valid]))

    img_used = {}
    cat_used = {}
    pairs = []
    for img_i, cat_i in raw_pairs:
        if img_i in img_used or cat_i in cat_used:
            continue
        img_used[img_i] = True
        cat_used[cat_i] = True
        pairs.append((img_i, cat_i))

    return pairs


def match_lists_fast_kdtree(sa_x, sa_y, sb_x, sb_y, radius):
    """KDTree快速匹配 (参考psm_star_alignment.cpp的match_lists_fast)
    返回: (idx_a列表, idx_b列表)
    """
    if len(sa_x) == 0 or len(sb_x) == 0:
        return [], []

    sa_pts = np.column_stack([sa_x, sa_y])
    sb_pts = np.column_stack([sb_x, sb_y])

    tree_b = cKDTree(sb_pts)
    dists, idxs_b = tree_b.query(sa_pts, k=1)

    valid = dists < radius
    raw_ia = np.where(valid)[0]
    raw_ib = idxs_b[valid]

    img_best = {}
    cat_best = {}
    for k in range(len(raw_ia)):
        ia = int(raw_ia[k])
        ib = int(raw_ib[k])
        dd = float(dists[ia])

        if ia not in img_best or dd < img_best[ia][1]:
            img_best[ia] = (ib, dd)
        if ib not in cat_best or dd < cat_best[ib][1]:
            cat_best[ib] = (ia, dd)

    pairs = []
    for ia, (ib, d) in img_best.items():
        if ib in cat_best and cat_best[ib][0] == ia:
            pairs.append((ia, ib))

    idx_a = [p[0] for p in pairs]
    idx_b = [p[1] for p in pairs]
    return idx_a, idx_b


def apply_affine(x, y, a0, a1, a2, b0, b1, b2):
    """仿射变换"""
    return a0 + a1 * x + a2 * y, b0 + b1 * x + b2 * y


def recalc_trans_from_pairs(img_x, img_y, cat_x, cat_y, img_indices, cat_indices):
    """从匹配对重新计算仿射变换
    返回: (a0, a1, a2, b0, b1, b2) 或 None
    """
    nm = len(img_indices)
    if nm < AT_MATCH_REQUIRE:
        return None

    votes = [100] * nm
    result = iter_trans(img_x, img_y, cat_x, cat_y,
                        votes, list(img_indices), list(cat_indices), nm,
                        recalc=True, max_iter=AT_MATCH_MAXITER,
                        halt_sigma=AT_MATCH_HALTSIGMA)
    if result is None:
        return None
    a0, a1, a2, b0, b1, b2, nr, final_wia, final_wib = result
    if nr < AT_MATCH_REQUIRE:
        return None
    return a0, a1, a2, b0, b1, b2


def verify_match(img_x, img_y, cat_x, cat_y, affine, radii=None):
    """递减半径验证匹配
    每轮: 用仿射变换预测位置 → 近邻匹配 → iter_trans精化
    返回: (精化后仿射参数, 匹配对列表)
    """
    if radii is None:
        radii = [50.0, 30.0, 10.0, AT_MATCH_RADIUS]

    a0, a1, a2, b0, b1, b2 = affine
    pairs = []

    for radius in radii:
        pred_x, pred_y = apply_affine(img_x, img_y, a0, a1, a2, b0, b1, b2)
        idx_a, idx_b = match_lists_fast_kdtree(pred_x, pred_y, cat_x, cat_y, radius)
        if len(idx_a) < AT_MATCH_REQUIRE:
            break

        result = recalc_trans_from_pairs(img_x, img_y, cat_x, cat_y, idx_a, idx_b)
        if result is None:
            break
        a0, a1, a2, b0, b1, b2 = result

        pred_x, pred_y = apply_affine(img_x, img_y, a0, a1, a2, b0, b1, b2)
        idx_a, idx_b = match_lists_fast_kdtree(pred_x, pred_y, cat_x, cat_y, radius)
        pairs = list(zip(idx_a, idx_b))

        logger.info("  验证匹配(半径=%.0f): %d对", radius, len(pairs))

    return (a0, a1, a2, b0, b1, b2), pairs


def ransac_affine(img_x, img_y, cat_x, cat_y, img_indices, cat_indices,
                  thresh_px=RANSAC_THRESH_PX, max_iter=RANSAC_ITER):
    """RANSAC仿射拟合 (参考psm_star_alignment.cpp的ransac_affine)
    返回: (成功标志, 仿射6参数, 内点img索引, 内点cat索引)
    """
    nm = len(img_indices)
    if nm < 3:
        return False, None, [], []

    thresh2 = thresh_px ** 2
    best_count = 0
    best_affine = None
    rng = np.random.RandomState(42)

    for _ in range(max_iter):
        sel = rng.choice(nm, 3, replace=False)
        si = [img_indices[sel[0]], img_indices[sel[1]], img_indices[sel[2]]]
        ci = [cat_indices[sel[0]], cat_indices[sel[1]], cat_indices[sel[2]]]

        result = calc_trans_linear(img_x, img_y, cat_x, cat_y, si, ci, 3)
        if result is None:
            continue
        a0t, a1t, a2t, b0t, b1t, b2t = result

        count = 0
        for i in range(nm):
            sx = img_x[img_indices[i]]
            sy = img_y[img_indices[i]]
            nx = a0t + a1t * sx + a2t * sy
            ny = b0t + b1t * sx + b2t * sy
            dx = nx - cat_x[cat_indices[i]]
            dy = ny - cat_y[cat_indices[i]]
            if dx * dx + dy * dy < thresh2:
                count += 1

        if count > best_count:
            best_count = count
            best_affine = (a0t, a1t, a2t, b0t, b1t, b2t)

    if best_count < 3 or best_affine is None:
        return False, None, [], []

    a0, a1, a2, b0, b1, b2 = best_affine
    inlier_ia = []
    inlier_ib = []
    for i in range(nm):
        sx = img_x[img_indices[i]]
        sy = img_y[img_indices[i]]
        nx = a0 + a1 * sx + a2 * sy
        ny = b0 + b1 * sx + b2 * sy
        dx = nx - cat_x[cat_indices[i]]
        dy = ny - cat_y[cat_indices[i]]
        if dx * dx + dy * dy < thresh2:
            inlier_ia.append(img_indices[i])
            inlier_ib.append(cat_indices[i])

    logger.info("RANSAC: %d/%d 内点 (阈值=%.1f px)", len(inlier_ia), nm, thresh_px)

    result = calc_trans_linear(img_x, img_y, cat_x, cat_y, inlier_ia, inlier_ib, len(inlier_ia))
    if result is None:
        return False, None, [], []
    return True, result, inlier_ia, inlier_ib


def match_with_flip(img_x, img_y, img_flux, img_saturated,
                    cat_px, cat_py, cat_mag, flip_mode,
                    scale_arcsec_px, percent_scale_range=10.0):
    """对单种翻转模式执行完整匹配流程
    策略: 先尝试三角匹配，失败则用初始WCS信息做KDTree直接匹配
    返回: FlipMatchResult
    """
    t0 = time.time()
    mode_name = FLIP_MODES.get(flip_mode, f"模式{flip_mode}")
    logger.info("=== 翻转模式 %d (%s) ===", flip_mode, mode_name)

    flipped_cat_px, flipped_cat_py = apply_flip(cat_px, cat_py, flip_mode)

    sel_img_x, sel_img_y, sel_cat_x, sel_cat_y, sel_img_idx, sel_cat_idx = \
        select_match_stars(img_x, img_y, img_flux, img_saturated,
                           flipped_cat_px, flipped_cat_py, cat_mag)

    scale_range = 1.0 + percent_scale_range / 100.0
    scale_min = 1.0 / scale_range
    scale_max = scale_range

    best_affine = None
    best_matched = 0
    best_ia = []
    best_ib = []

    # 策略1: 三角匹配
    n_sat = int(np.sum(img_saturated))
    n_retries = 1 if n_sat >= MIN_SAT_FOR_PRIORITY else len(RETRY_COUNTS)

    for retry in range(n_retries):
        if n_sat >= MIN_SAT_FOR_PRIORITY:
            n_img_bright = min(int(np.sum(img_saturated)), 200)
            n_cat_bright = min(int(n_img_bright * 1.5), len(sel_cat_x))
        else:
            n_img_bright = min(RETRY_COUNTS[retry], len(sel_img_x))
            n_cat_bright = min(int(n_img_bright * 1.5), len(sel_cat_x))

        ri_x = sel_img_x[:n_img_bright]
        ri_y = sel_img_y[:n_img_bright]
        rc_x = sel_cat_x[:n_cat_bright]
        rc_y = sel_cat_y[:n_cat_bright]

        logger.info("  尝试%d: %d图像星 / %d星表星", retry + 1, n_img_bright, n_cat_bright)

        nobj = min(max(n_img_bright, n_cat_bright), 50)
        ok, affine, tri_pairs = triangle_match(
            ri_x, ri_y, rc_x, rc_y, nobj,
            AT_TRIANGLE_RADIUS, scale_min, scale_max
        )

        if not ok:
            logger.info("  三角匹配失败，尝试RANSAC回退")
            all_ia = list(range(len(ri_x)))
            all_ib = list(range(len(rc_x)))
            ok, affine, inlier_ia, inlier_ib = ransac_affine(
                ri_x, ri_y, rc_x, rc_y, all_ia, all_ib
            )
            if not ok:
                logger.info("  RANSAC也失败")
                continue

        a0, a1, a2, b0, b1, b2 = affine
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            logger.info("  变换退化 (det=%.8f)", det)
            continue

        img_trans_x, img_trans_y = apply_affine(ri_x, ri_y, a0, a1, a2, b0, b1, b2)

        match_radii = [50.0, 30.0, 10.0, AT_MATCH_RADIUS]
        ma, mb = [], []
        nm = 0

        for round_idx, match_r in enumerate(match_radii):
            ma, mb = match_lists_fast_kdtree(img_trans_x, img_trans_y, rc_x, rc_y, match_r)
            nm = len(ma)
            logger.info("  轮次%d (半径=%.0f): %d对", round_idx, match_r, nm)
            if nm < AT_MATCH_REQUIRE:
                break

            result = recalc_trans_from_pairs(ri_x, ri_y, rc_x, rc_y, ma, mb)
            if result is None:
                nm = 0
                break
            a0, a1, a2, b0, b1, b2 = result
            img_trans_x, img_trans_y = apply_affine(ri_x, ri_y, a0, a1, a2, b0, b1, b2)

        if nm < AT_MATCH_REQUIRE:
            logger.info("  匹配失败")
            continue

        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-10:
            continue

        scale_val = np.sqrt(a1 ** 2 + a2 ** 2)
        scale_ratio = scale_val / 1.0
        if scale_ratio < 0.5 or scale_ratio > 2.0:
            logger.info("  比例尺%.6f (预期~1.0, 比值=%.2f) 超出范围",
                         scale_val, scale_ratio)
            continue

        if nm > best_matched:
            best_matched = nm
            best_affine = (a0, a1, a2, b0, b1, b2)
            best_ia = ma
            best_ib = mb

        if best_matched > 10:
            break

    # 策略2: 如果三角匹配+RANSAC都失败，用初始WCS信息做KDTree直接匹配
    if best_matched < AT_MATCH_REQUIRE:
        logger.info("  三角匹配+RANSAC均失败，尝试KDTree直接匹配")
        a0_init, a1_init, a2_init = 0.0, 1.0, 0.0
        b0_init, b1_init, b2_init = 0.0, 0.0, 1.0

        match_radii = [100.0, 50.0, 30.0, 20.0, 10.0, 5.0, 2.0]
        for round_idx, match_r in enumerate(match_radii):
            pred_x, pred_y = apply_affine(img_x, img_y,
                                           a0_init, a1_init, a2_init,
                                           b0_init, b1_init, b2_init)
            ma, mb = match_lists_fast_kdtree(pred_x, pred_y, flipped_cat_px, flipped_cat_py, match_r)
            nm = len(ma)
            logger.info("  KDTree轮次%d (半径=%.0f): %d对", round_idx, match_r, nm)
            if nm < AT_MATCH_REQUIRE:
                break

            result = recalc_trans_from_pairs(img_x, img_y, flipped_cat_px, flipped_cat_py, ma, mb)
            if result is None:
                nm = 0
                break
            a0_init, a1_init, a2_init, b0_init, b1_init, b2_init = result
            det = a1_init * b2_init - a2_init * b1_init
            if abs(det) < 1e-10:
                nm = 0
                break
            scale_val = np.sqrt(a1_init ** 2 + a2_init ** 2)
            scale_ratio = scale_val / 1.0
            if scale_ratio < 0.3 or scale_ratio > 3.0:
                logger.info("  KDTree比例尺%.6f (比值=%.2f) 超出范围", scale_val, scale_ratio)
                nm = 0
                break

        if nm >= AT_MATCH_REQUIRE:
            best_matched = nm
            best_affine = (a0_init, a1_init, a2_init, b0_init, b1_init, b2_init)
            best_ia = ma
            best_ib = mb
            logger.info("  KDTree匹配成功: %d对", nm)

    if best_matched < AT_MATCH_REQUIRE:
        elapsed = time.time() - t0
        logger.info("翻转模式%d匹配失败, 耗时%.2fs", flip_mode, elapsed)
        return FlipMatchResult(flip_mode=flip_mode, affine=(0, 0, 0, 0, 0, 0),
                               matched_count=0, rms_arcsec=1e10)

    a0, a1, a2, b0, b1, b2 = best_affine

    pred_x, pred_y = apply_affine(img_x, img_y, a0, a1, a2, b0, b1, b2)
    all_ia, all_ib = match_lists_fast_kdtree(pred_x, pred_y, flipped_cat_px, flipped_cat_py, 5.0)
    nm_final = len(all_ia)

    if nm_final >= AT_MATCH_REQUIRE:
        result = recalc_trans_from_pairs(img_x, img_y, flipped_cat_px, flipped_cat_py, all_ia, all_ib)
        if result is not None:
            a0, a1, a2, b0, b1, b2 = result
            best_affine = (a0, a1, a2, b0, b1, b2)
            best_matched = nm_final

    rms_arcsec = 0.0
    if best_matched >= AT_MATCH_REQUIRE:
        a0, a1, a2, b0, b1, b2 = best_affine
        pred_x, pred_y = apply_affine(img_x, img_y, a0, a1, a2, b0, b1, b2)
        all_ia2, all_ib2 = match_lists_fast_kdtree(pred_x, pred_y, flipped_cat_px, flipped_cat_py, 5.0)
        if len(all_ia2) >= AT_MATCH_REQUIRE:
            rms_sum = 0.0
            for k in range(len(all_ia2)):
                px = pred_x[all_ia2[k]]
                py = pred_y[all_ia2[k]]
                cx = flipped_cat_px[all_ib2[k]]
                cy = flipped_cat_py[all_ib2[k]]
                d2 = (px - cx) ** 2 + (py - cy) ** 2
                rms_sum += d2
            rms_arcsec = np.sqrt(rms_sum / len(all_ia2)) * scale_arcsec_px
            best_matched = len(all_ia2)

    elapsed = time.time() - t0
    logger.info("翻转模式%d: %d对, RMS=%.4f角秒, 耗时%.2fs",
                flip_mode, best_matched, rms_arcsec, elapsed)

    return FlipMatchResult(flip_mode=flip_mode, affine=best_affine,
                           matched_count=best_matched, rms_arcsec=rms_arcsec)


def select_best_flip(results):
    """从4种模式结果中选择最佳（匹配数最多→RMS最小）"""
    valid = [r for r in results if r.matched_count >= AT_MATCH_REQUIRE]
    if not valid:
        return None
    valid.sort(key=lambda r: (-r.matched_count, r.rms_arcsec))
    best = valid[0]
    logger.info("最佳翻转模式: %d (%s), %d对, RMS=%.4f角秒",
                best.flip_mode, FLIP_MODES.get(best.flip_mode, "?"),
                best.matched_count, best.rms_arcsec)
    return best


def iterative_reprojection(img_x, img_y, img_flux, img_saturated,
                           cat_ra, cat_dec, cat_mag,
                           gaia_client, center_ra, center_dec,
                           affine, scale_arcsec_px, width, height,
                           flip_mode, max_trials=MAX_REPROJ_TRIALS,
                           conv_tol=CONV_TOLERANCE):
    """迭代重投影收敛 (siril风格)
    收敛条件: offset < conv_tol (arcsec)
    返回: (final_ra, final_dec, final_affine, matched_count, rms_arcsec)
    """
    a0, a1, a2, b0, b1, b2 = affine
    ra0, dec0 = center_ra, center_dec

    fov_w, fov_h, fov_diag = compute_fov(scale_arcsec_px, width, height)

    for trial in range(max_trials):
        conv = np.sqrt(a0 ** 2 + b0 ** 2) * scale_arcsec_px
        if conv < conv_tol:
            logger.info("重投影收敛: trial=%d, conv=%.4f角秒", trial, conv)
            break

        new_ra, new_dec = gnomonic_inverse(a0 * scale_arcsec_px, b0 * scale_arcsec_px, ra0, dec0)
        ra0, dec0 = float(new_ra), float(new_dec)
        a0, b0 = 0.0, 0.0
        logger.info("重投影trial=%d: RA=%.6f Dec=%.6f (偏移=%.4f角秒)", trial, ra0, dec0, conv)

        cat_px, cat_py, valid = project_gaia_to_pixel(cat_ra, cat_dec, ra0, dec0, scale_arcsec_px)
        valid_mask = valid
        if np.sum(valid_mask) < 10:
            logger.warning("重投影: 有效星数不足 %d", np.sum(valid_mask))
            break

        cat_px_f = cat_px[valid_mask]
        cat_py_f = cat_py[valid_mask]
        cat_mag_f = cat_mag[valid_mask]

        flipped_cat_px, flipped_cat_py = apply_flip(cat_px_f, cat_py_f, flip_mode)

        reproj_radii = [100.0, 50.0, 30.0, 10.0, 5.0]
        reproj_ok = False

        for r, radius in enumerate(reproj_radii):
            pred_x, pred_y = apply_affine(img_x, img_y,
                                           a0, a1, a2,
                                           b0, b1, b2)
            idx_a, idx_b = match_lists_fast_kdtree(pred_x, pred_y,
                                                     flipped_cat_px, flipped_cat_py, radius)
            nm = len(idx_a)
            logger.info("重投影trial=%d 轮次%d (半径=%.0f): %d对", trial, r, radius, nm)
            if nm < AT_MATCH_REQUIRE:
                break

            result = recalc_trans_from_pairs(img_x, img_y, flipped_cat_px, flipped_cat_py,
                                              idx_a, idx_b)
            if result is None:
                break
            a0, a1, a2, b0, b1, b2 = result
            if r == len(reproj_radii) - 1:
                reproj_ok = True

        if not reproj_ok:
            logger.warning("重投影匹配失败 trial=%d", trial)
            break

    conv = np.sqrt(a0 ** 2 + b0 ** 2) * scale_arcsec_px
    if conv <= conv_tol:
        logger.info("重投影收敛完成: %d次, conv=%.4f角秒", trial + 1, conv)
    else:
        logger.info("重投影未完全收敛: conv=%.4f角秒", conv)

    cat_px_final, cat_py_final, _ = project_gaia_to_pixel(cat_ra, cat_dec, ra0, dec0, scale_arcsec_px)
    pred_x, pred_y = apply_affine(img_x, img_y, a0, a1, a2, b0, b1, b2)
    flipped_cat_px, flipped_cat_py = apply_flip(cat_px_final, cat_py_final, flip_mode)
    idx_a, idx_b = match_lists_fast_kdtree(pred_x, pred_y, flipped_cat_px, flipped_cat_py, 5.0)
    matched_count = len(idx_a)

    rms_arcsec = 0.0
    if matched_count >= AT_MATCH_REQUIRE:
        rms_sum = 0.0
        for k in range(matched_count):
            px = pred_x[idx_a[k]]
            py = pred_y[idx_a[k]]
            cx = flipped_cat_px[idx_b[k]]
            cy = flipped_cat_py[idx_b[k]]
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            rms_sum += d2
        rms_arcsec = np.sqrt(rms_sum / matched_count) * scale_arcsec_px

    return ra0, dec0, (a0, a1, a2, b0, b1, b2), matched_count, rms_arcsec


class InitialWCS:
    """初始WCS生成器
    执行7步初始WCS生成:
    1. 计算像素尺度和FOV
    2. Gaia锥形查询 + 极限星等二分法
    3. Gnomonic投影
    4. 4种翻转模式饱和星优先三角匹配
    5. 选择最佳翻转模式
    6. 全星点验证匹配
    7. 迭代重投影收敛
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0):
        self._gaia_data_dir = gaia_data_dir
        self._db_type = db_type
        self._gaia_client = None

    def _ensure_gaia_client(self):
        if self._gaia_client is None:
            self._gaia_client = GaiaClientPy(self._gaia_data_dir, self._db_type)

    def solve(self, img_x, img_y, img_flux, img_saturated,
              center_ra, center_dec, focal_length_mm, pixel_size_um,
              width, height, scale_arcsec_px=0.0,
              percent_scale_range=10.0):
        """执行7步初始WCS生成

        参数:
            img_x: 图像星点X坐标 (相对于图像中心, Y向上为正)
            img_y: 图像星点Y坐标
            img_flux: 星点flux
            img_saturated: 饱和标记 (1=饱和, 0=正常)
            center_ra: 初始中心RA (度)
            center_dec: 初始中心Dec (度)
            focal_length_mm: 焦距 (mm)
            pixel_size_um: 像素尺寸 (um)
            width: 图像宽度 (像素)
            height: 图像高度 (像素)
            scale_arcsec_px: 已知像素尺度 (0=自动计算)
            percent_scale_range: 比例尺搜索范围 (百分比)

        返回:
            InitialWCSResult 或 None
        """
        t_total = time.time()
        img_x = np.asarray(img_x, dtype=np.float64)
        img_y = np.asarray(img_y, dtype=np.float64)
        img_flux = np.asarray(img_flux, dtype=np.float64)
        img_saturated = np.asarray(img_saturated, dtype=np.int32)

        n_img = len(img_x)
        n_sat = int(np.sum(img_saturated))
        logger.info("=" * 60)
        logger.info("初始WCS生成: %d图像星(饱和%d), 中心(%.4f, %.4f)",
                     n_img, n_sat, center_ra, center_dec)

        # Step 1: 计算像素尺度和FOV
        t0 = time.time()
        if scale_arcsec_px <= 0:
            scale_arcsec_px = compute_pixel_scale(focal_length_mm, pixel_size_um)
        fov_w, fov_h, fov_diag = compute_fov(scale_arcsec_px, width, height)
        logger.info("Step1: 尺度=%.3f角秒/px, FOV=%.2fx%.2f(对角%.2f度)",
                     scale_arcsec_px, fov_w, fov_h, fov_diag)
        logger.info("Step1耗时: %.3fs", time.time() - t0)

        # Step 2: Gaia锥形查询 + 极限星等二分法
        t0 = time.time()
        self._ensure_gaia_client()
        mag_limit, n_cat, cat_ra, cat_dec, cat_mag = gaia_cone_search_with_bisection(
            self._gaia_client, center_ra, center_dec, fov_diag, n_img
        )
        if n_cat < 10:
            logger.error("Gaia星数不足: %d", n_cat)
            return None
        sort_idx = np.argsort(cat_mag)
        cat_ra = cat_ra[sort_idx]
        cat_dec = cat_dec[sort_idx]
        cat_mag = cat_mag[sort_idx]
        logger.info("Step2耗时: %.3fs", time.time() - t0)

        # Step 3: Gnomonic投影
        t0 = time.time()
        cat_px, cat_py, valid = project_gaia_to_pixel(
            cat_ra, cat_dec, center_ra, center_dec, scale_arcsec_px
        )
        valid_mask = valid
        cat_px = cat_px[valid_mask]
        cat_py = cat_py[valid_mask]
        cat_ra_v = cat_ra[valid_mask]
        cat_dec_v = cat_dec[valid_mask]
        cat_mag_v = cat_mag[valid_mask]
        logger.info("Step3: 投影后星数=%d", len(cat_px))
        logger.info("Step3耗时: %.3fs", time.time() - t0)

        # Step 4: 4种翻转模式饱和星优先三角匹配
        t0 = time.time()
        flip_results = []
        for flip_mode in range(4):
            result = match_with_flip(
                img_x, img_y, img_flux, img_saturated,
                cat_px, cat_py, cat_mag_v, flip_mode,
                scale_arcsec_px, percent_scale_range
            )
            flip_results.append(result)
        logger.info("Step4耗时: %.3fs", time.time() - t0)

        # Step 5: 选择最佳翻转模式
        best = select_best_flip(flip_results)
        if best is None:
            logger.error("所有翻转模式匹配失败")
            return None

        # Step 6: 全星点验证匹配
        t0 = time.time()
        flipped_cat_px, flipped_cat_py = apply_flip(cat_px, cat_py, best.flip_mode)
        final_affine, final_pairs = verify_match(
            img_x, img_y, flipped_cat_px, flipped_cat_py, best.affine
        )
        logger.info("Step6: 验证后%d对", len(final_pairs))
        logger.info("Step6耗时: %.3fs", time.time() - t0)

        # Step 7: 迭代重投影收敛
        t0 = time.time()
        final_ra, final_dec, final_affine2, matched_count, rms_arcsec = iterative_reprojection(
            img_x, img_y, img_flux, img_saturated,
            cat_ra_v, cat_dec_v, cat_mag_v,
            self._gaia_client, center_ra, center_dec,
            final_affine, scale_arcsec_px, width, height, best.flip_mode
        )
        logger.info("Step7: 最终中心(%.6f, %.6f), %d对, RMS=%.4f角秒",
                     final_ra, final_dec, matched_count, rms_arcsec)
        logger.info("Step7耗时: %.3fs", time.time() - t0)

        # 计算旋转角和最终比例尺
        a0, a1, a2, b0, b1, b2 = final_affine2
        rotation_deg = np.degrees(np.arctan2(b1, a1))
        final_scale = np.sqrt(a1 ** 2 + a2 ** 2)
        rms_px = rms_arcsec / final_scale if final_scale > 0 else 0

        total_time = time.time() - t_total
        logger.info("初始WCS生成完成: 总耗时%.2fs", total_time)
        logger.info("  中心: RA=%.6f Dec=%.6f", final_ra, final_dec)
        logger.info("  旋转: %.4f度, 比例尺: %.4f角秒/px", rotation_deg, final_scale)
        logger.info("  翻转: %d (%s), 匹配: %d对, RMS: %.4f角秒 (%.3f px)",
                     best.flip_mode, FLIP_MODES.get(best.flip_mode, "?"),
                     matched_count, rms_arcsec, rms_px)

        return InitialWCSResult(
            center_ra=final_ra,
            center_dec=final_dec,
            rotation_deg=rotation_deg,
            scale_arcsec_px=final_scale,
            flip_mode=best.flip_mode,
            affine=final_affine2,
            matched_count=matched_count,
            rms_px=rms_px,
            rms_arcsec=rms_arcsec
        )

    def close(self):
        if self._gaia_client is not None:
            self._gaia_client.close()
            self._gaia_client = None

    def __del__(self):
        self.close()

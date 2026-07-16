"""Vector Match V4.0 Python ctypes wrapper

功能:
    V4.0 抽样投票向量法 Python 封装 — 在 V3.5 抽样投票核心架构基础上新增 5 大优化:
      Phase 0 : 密度匹配迭代星等查询（替代 V3.5 bisection_mag_limit）
      Phase A : PROSAC 优先采样（替代 V3.5 稀疏度加权抽样）
      Phase C : k-vector 快速角距索引（替代 V3.5 暴力 NN 搜索）
      Phase D': 贝叶斯假设验证 + 三角形双特征二级验证
      日志   : 详细的 V4.0 诊断日志（UTF-8 BOM，线程安全）

用途:
    Python 端调用 V4.0 C++ DLL 的接口，封装输入输出结构体，
    实现 density_match_query 密度匹配查询（Python 端完成实际 Gaia 查询），
    将结果通过 VM4_1SolveParams 传入 C++ vm4_1_solve() 完成匹配求解，
    输出格式与 V3.5 兼容（JSON 结构一致），确保下游可视化脚本可复用。

依赖:
    - vector_match_v2.py (GaiaClientPy, gnomonic_forward, VectorMatchResult, _apply_flip 等)
    - vector_match_v4_1.dll (C++ 编译产物)
    - numpy, ctypes, logging
"""
import ctypes, math, os, sys, json, time, logging, numpy as np
from typing import Optional
from vector_match_v2 import (
    GaiaClientPy, VectorMatchResult, _DEGTORAD,
    _build_catalog_vectors, _apply_flip, _apply_similarity,
    gnomonic_forward,
)

logger = logging.getLogger("vector_match_v4_1_cpp")
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


# ============================================================================
# ctypes 结构体定义（与 vm4_api.h 严格对应，_pack_=8 避免对齐问题）
# ============================================================================

class VM4_1SolveParamsC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        # === V3.5 继承字段 ===
        ("s0", ctypes.c_double), ("s_min", ctypes.c_double), ("s_max", ctypes.c_double),
        ("n_modes", ctypes.c_int), ("seed", ctypes.c_int),
        ("K_total", ctypes.c_int), ("batch_size", ctypes.c_int), ("min_samples", ctypes.c_int),
        ("K_top", ctypes.c_int), ("min_inliers", ctypes.c_int),
        ("fov_diag_asec", ctypes.c_double), ("img_width", ctypes.c_double), ("img_height", ctypes.c_double),
        ("center_ra", ctypes.c_double), ("center_dec", ctypes.c_double),
        ("wcs_out_path", ctypes.c_char_p),
        ("skip_sip", ctypes.c_int),
        ("expand_n_gaia", ctypes.c_int),
        ("expand_n_img", ctypes.c_int),
        ("radial_n_bins", ctypes.c_int),
        ("radial_fit_order", ctypes.c_int),
        ("radial_n_iters", ctypes.c_int),

        # === V4.0 新增字段 ===
        # Phase 0: 密度匹配查询参数
        ("k_match", ctypes.c_double),
        ("query_radius_factor", ctypes.c_double),
        ("m_lim_step", ctypes.c_double),
        ("m_lim_max_iter", ctypes.c_int),
        ("density_tolerance", ctypes.c_double),
        ("n_img_bright", ctypes.c_int),
        ("focal_length_mm", ctypes.c_double),
        ("pixel_size_um", ctypes.c_double),
        ("exposure_time_s", ctypes.c_double),

        # === V4.1 新增字段：不对称选星 (位置必须与 vm4_api.h 一致: Phase 0 之后, Phase C 之前) ===
        ("img_n_target", ctypes.c_int),               # 图像侧目标星数(默认50)
        ("gaia_density_ratio", ctypes.c_double),      # Gaia面密度/图像面密度(默认1.5)
        ("gaia_query_radius_factor", ctypes.c_double), # Gaia查询半径因子(默认0.55)

        # Phase C: k-vector 参数
        ("k_vector_eps", ctypes.c_double),
        ("use_kvector", ctypes.c_int),

        # Phase A: PROSAC 参数
        ("w_snr", ctypes.c_double),
        ("w_sparse", ctypes.c_double),
        ("w_sat", ctypes.c_double),
        ("prosac_T_max", ctypes.c_int),
        ("use_prosac", ctypes.c_int),

        # Phase D': 贝叶斯验证参数
        ("lnK_accept", ctypes.c_double),
        ("lnK_weak", ctypes.c_double),
        ("use_bayes", ctypes.c_int),

        # Phase D': 三角形验证参数
        ("eps_A", ctypes.c_double),
        ("eps_J", ctypes.c_double),
        ("triangle_pass_rate", ctypes.c_double),
        ("use_triangle", ctypes.c_int),

        # Task 7 集成新增可选输入字段
        ("snr_values", ctypes.POINTER(ctypes.c_double)),
        ("is_saturated_values", ctypes.POINTER(ctypes.c_int)),
        ("log_file_path", ctypes.c_char_p),
    ]


class VM4_1DebugInfoC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        # === V3.5 继承字段 ===
        ("theta_snr", ctypes.c_double), ("theta_peak_deg", ctypes.c_double),
        ("best_n_range", ctypes.c_int), ("median_noise", ctypes.c_double),
        ("n_phaseb_pairs", ctypes.c_int), ("n_phaseb_corr", ctypes.c_int),
        ("n_phasea_records", ctypes.c_int), ("n_phasec_expanded", ctypes.c_int),
        ("n_phased_clean", ctypes.c_int), ("n_phased_iterations", ctypes.c_int),
        ("mad_rms_arcsec", ctypes.c_double),
        ("n_expand_mutual", ctypes.c_int),
        ("n_expand_after_filter", ctypes.c_int),
        ("n_sip_total", ctypes.c_int),
        ("sip_order", ctypes.c_int),

        # === V4.0 新增字段 ===
        # Phase 0
        ("rho_img", ctypes.c_double),
        ("rho_target", ctypes.c_double),
        ("m_lim_final", ctypes.c_double),
        ("n_gaia_final", ctypes.c_int),
        ("m_lim_iterations", ctypes.c_int),

        # Phase C
        ("kvector_build_ms", ctypes.c_double),
        ("kvector_queries", ctypes.c_int),
        ("kvector_avg_candidates", ctypes.c_double),

        # Phase A
        ("prosac_quality_median", ctypes.c_double),
        ("prosac_pool_final", ctypes.c_int),

        # Phase D'
        ("bayes_lnK", ctypes.c_double),
        ("bayes_n_match", ctypes.c_int),
        ("bayes_decision", ctypes.c_int),
        ("triangle_total", ctypes.c_int),
        ("triangle_pass_ratio", ctypes.c_double),
    ]


class VM4_1SolveResultC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s", ctypes.c_double), ("theta", ctypes.c_double), ("tx", ctypes.c_double), ("ty", ctypes.c_double),
        ("n_inliers", ctypes.c_int), ("rms", ctypes.c_double),
        ("best_mode", ctypes.c_int), ("norm_score", ctypes.c_double),
        ("inlier_mask", ctypes.POINTER(ctypes.c_int)),
        ("success", ctypes.c_int), ("peak_snr", ctypes.c_double), ("n_samples", ctypes.c_int),
        ("debug", VM4_1DebugInfoC),
        ("sip_A", ctypes.c_double * 36), ("sip_B", ctypes.c_double * 36),
        ("cd", ctypes.c_double * 4), ("crval", ctypes.c_double * 2), ("crpix", ctypes.c_double * 2),
    ]


# ============================================================================
# 工具函数
# ============================================================================

def _find_dll() -> str:
    """查找 V4.0 DLL 路径"""
    p = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v4_1", "vector_match_v4_1.dll")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"V4.0 DLL not found: {p}")


def _find_log_dir() -> str:
    """查找日志目录（不存在则创建）"""
    log_dir = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_1")
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def _make_log_path() -> str:
    """生成带时间戳的日志文件路径: v4_YYYYMMDD_HHMMSS.log"""
    log_dir = _find_log_dir()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return os.path.join(log_dir, f"v4_1_{ts}.log")


def _read_wcs_json(path: str) -> dict:
    """读取 WCS JSON 文件（与 V3.5 格式兼容）"""
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return {
        'CD': np.array(d['CD'], dtype=np.float64),
        'CRVAL': np.array(d['CRVAL'], dtype=np.float64),
        'CRPIX': np.array(d['CRPIX'], dtype=np.float64),
        'SIP_A': np.array(d['SIP_A'], dtype=np.float64),
        'SIP_B': np.array(d['SIP_B'], dtype=np.float64),
        'RMS_PX': float(d['RMS_PX']),
    }


def density_match_query(gaia_client: GaiaClientPy,
                         center_ra: float, center_dec: float,
                         fov_diag_deg: float, n_img_bright: int,
                         k_match: float = 1.5, query_radius_factor: float = 1.0,
                         m_lim_step: float = 0.5, m_lim_max_iter: int = 8,
                         density_tolerance: float = 0.1,
                         focal_length_mm: float = 0.0, exposure_time_s: float = 1.0):
    """V4.0 密度匹配迭代星等查询（Phase 0）

    替代 V3.5 的 bisection_mag_limit，直接保证星表密度与图像亮星密度一致。

    算法:
        1. ρ_img = N_img / query_area
        2. N_target = k_match × N_img
        3. 从 m_cut 开始，0.5 等步长迭代极限星等
        4. 收敛条件: N_gaia ∈ [0.9, 1.1] × N_target

    Args:
        gaia_client: GaiaClientPy 实例
        center_ra, center_dec: 查询中心(度)
        fov_diag_deg: FOV 对角线(度)
        n_img_bright: 图像亮星数
        k_match: 星表密度匹配系数(默认1.5)
        query_radius_factor: 查询半径因子(默认1.0, 即1×FOV)
        m_lim_step: 极限星等迭代步长(默认0.5)
        m_lim_max_iter: 最大迭代次数(默认8)
        density_tolerance: 密度匹配容差(默认0.1, 即±10%)
        focal_length_mm: 焦距(mm)，用于初始 m_cut 估计
        exposure_time_s: 曝光时间(s)，用于初始 m_cut 估计

    Returns:
        dict: {
            'ra', 'dec', 'mag': Gaia 星表数组,
            'm_lim_final': 最终极限星等,
            'n_gaia_final': 最终 Gaia 星数,
            'm_lim_iterations': 迭代次数,
            'rho_img': 图像亮星密度,
            'rho_target': 目标星表密度,
            'n_target': 目标星数,
            'query_radius_deg': 查询半径(度),
            'query_area_deg2': 查询区域面积(平方度),
        }
    """
    # 1. 计算查询半径与区域
    query_radius_deg = fov_diag_deg * 0.5 * query_radius_factor
    query_area_deg2 = math.pi * query_radius_deg ** 2

    # 2. 计算图像亮星密度与目标星数
    if query_area_deg2 < 1e-10:
        raise ValueError(f"query_area_deg2={query_area_deg2} too small (fov_diag={fov_diag_deg})")
    rho_img = n_img_bright / query_area_deg2
    n_target = int(k_match * n_img_bright)
    rho_target = n_target / query_area_deg2

    # 3. 初始极限星等估计: m_cut ≈ 6 + 1.5·log10(f_mm) + 2·log10(t_s)
    if focal_length_mm > 0:
        m_cut = 6.0 + 1.5 * math.log10(max(focal_length_mm, 1.0)) + 2.0 * math.log10(max(exposure_time_s, 0.1))
    else:
        m_cut = 18.0  # 默认起始星等

    # 4. 迭代收敛
    n_target_low = int(n_target * (1.0 - density_tolerance))
    n_target_high = int(n_target * (1.0 + density_tolerance))

    sys.stderr.write(
        f"[vm4_1] Phase 0 密度匹配: n_img={n_img_bright} rho_img={rho_img:.3f}/deg² "
        f"n_target={n_target} (±{density_tolerance*100:.0f}%=[{n_target_low},{n_target_high}]) "
        f"query_r={query_radius_deg:.3f}° m_cut={m_cut:.2f}\n"
    )

    mag_query = m_cut
    ra_all = dec_all = mag_all = None
    iterations = 0
    for it in range(m_lim_max_iter):
        iterations = it + 1
        ra_t, dec_t, mag_t = gaia_client.cone_search(center_ra, center_dec, query_radius_deg, mag_query)
        n_stars = len(ra_t)
        sys.stderr.write(
            f"[vm4_1] Phase 0 iter#{it}: mag={mag_query:.2f} → {n_stars} stars "
            f"(target=[{n_target_low},{n_target_high}])\n"
        )

        if n_stars < n_target_low:
            # 星数不足，放宽星等
            mag_query += m_lim_step
        elif n_stars > n_target_high:
            # 星数过多，收紧星等
            mag_query -= m_lim_step
        else:
            # 收敛
            ra_all, dec_all, mag_all = ra_t, dec_t, mag_t
            sys.stderr.write(
                f"[vm4_1] Phase 0 收敛: mag={mag_query:.2f} n_gaia={n_stars} "
                f"(迭代{iterations}次, ρ_gaia={n_stars/query_area_deg2:.3f}/deg²)\n"
            )
            break

        ra_all, dec_all, mag_all = ra_t, dec_t, mag_t
    else:
        # 迭代次数用尽，使用最后一次查询结果
        sys.stderr.write(
            f"[vm4_1] Phase 0 未收敛(达到最大迭代{m_lim_max_iter}次), "
            f"使用 mag={mag_query:.2f} n_gaia={len(ra_all) if ra_all is not None else 0}\n"
        )

    if ra_all is None or len(ra_all) < 2:
        # 兜底: mag=22 全查询
        sys.stderr.write(f"[vm4_1] Phase 0 兜底 mag=22 查询\n")
        ra_all, dec_all, mag_all = gaia_client.cone_search(center_ra, center_dec, query_radius_deg, 22.0)

    return {
        'ra': ra_all, 'dec': dec_all, 'mag': mag_all,
        'm_lim_final': float(mag_query),
        'n_gaia_final': int(len(ra_all)),
        'm_lim_iterations': iterations,
        'rho_img': rho_img,
        'rho_target': rho_target,
        'n_target': n_target,
        'query_radius_deg': query_radius_deg,
        'query_area_deg2': query_area_deg2,
    }


def density_match_query_v4_1(gaia_client, center_ra, center_dec, fov_diag_deg,
                              n_img, gaia_density_ratio=1.5, gaia_query_radius_factor=0.55,
                              m_lim_step=0.5, m_lim_max_iter=15, density_tolerance=0.1,
                              focal_length_mm=1000.0, exposure_time_s=1.0):
    """V4.1 不对称密度匹配查询

    与V4.0区别:
      1. 查询半径 = fov_diag × gaia_query_radius_factor (V4.0是0.5×1.0)
      2. 目标星数 = gaia_density_ratio × n_img × (查询圆面积/图像面积) (V4.0是k_match×n_img)
      3. 自适应步长: 前4次m_lim_step, 后续减半

    Args:
        gaia_client: GaiaClientPy 实例
        center_ra, center_dec: 查询中心(度)
        fov_diag_deg: FOV 对角线(度)
        n_img: 图像侧选取星数
        gaia_density_ratio: Gaia目标面密度 / 图像面密度(默认1.5)
        gaia_query_radius_factor: Gaia查询半径因子(默认0.55, 即0.55×FOV对角线)
        m_lim_step: 极限星等迭代步长(默认0.5)
        m_lim_max_iter: 最大迭代次数(默认15)
        density_tolerance: 密度匹配容差(默认0.1, 即±10%)
        focal_length_mm: 焦距(mm)，用于初始 m_cut 估计
        exposure_time_s: 曝光时间(s)，用于初始 m_cut 估计

    Returns:
        dict: 包含 ra/dec/mag/m_lim/n_gaia/converged/iterations/query_radius_deg/rho_img/rho_target/n_target
    """
    # 查询半径
    query_radius_deg = fov_diag_deg * gaia_query_radius_factor
    query_area_deg2 = math.pi * query_radius_deg * query_radius_deg

    # 图像面积 (用FOV对角线估算: 假设图像近似正方形, 面积=(fov_diag/sqrt(2))²)
    img_area_deg2 = (fov_diag_deg / math.sqrt(2)) ** 2
    if img_area_deg2 <= 0:
        img_area_deg2 = query_area_deg2

    # 图像密度
    rho_img = n_img / img_area_deg2
    # Gaia目标密度
    rho_target = gaia_density_ratio * rho_img
    # 目标星数
    n_target = int(gaia_density_ratio * n_img * (query_area_deg2 / img_area_deg2))
    n_target = max(n_target, 50)  # 下限50

    # 初始星等
    m_cut = 6.0 + 1.5 * math.log10(focal_length_mm) + 2.0 * math.log10(max(exposure_time_s, 0.1))

    # 迭代
    n_lo = n_target * (1.0 - density_tolerance)
    n_hi = n_target * (1.0 + density_tolerance)

    m = m_cut
    n = 0
    converged = False
    iterations = 0

    logger.info(f"V4.1 Phase 0: ra={center_ra:.4f}, dec={center_dec:.4f}, "
                f"query_r={query_radius_deg:.4f}°, n_img={n_img}, "
                f"rho_img={rho_img:.2f}, rho_target={rho_target:.2f}, n_target={n_target}")
    sys.stderr.write(
        f"[vm4_1] V4.1 Phase 0 不对称密度匹配: n_img={n_img} rho_img={rho_img:.3f}/deg² "
        f"n_target={n_target} query_r={query_radius_deg:.3f}° m_cut={m_cut:.2f}\n"
    )

    ra_all = dec_all = mag_all = None
    for i in range(m_lim_max_iter):
        iterations = i + 1
        ra_t, dec_t, mag_t = gaia_client.cone_search(center_ra, center_dec, query_radius_deg, m)
        n = len(ra_t)
        ra_all, dec_all, mag_all = ra_t, dec_t, mag_t

        # V4.1自适应步长
        step = m_lim_step if i < 4 else m_lim_step * 0.5

        logger.info(f"V4.1 Phase 0 iter#{i}: mag={m:.2f} → {n} stars (target=[{n_lo:.0f},{n_hi:.0f}], step={step:.3f})")
        sys.stderr.write(
            f"[vm4_1] V4.1 Phase 0 iter#{i}: mag={m:.2f} → {n} stars "
            f"(target=[{n_lo:.0f},{n_hi:.0f}], step={step:.3f})\n"
        )

        if n < n_lo:
            m += step
        elif n > n_hi:
            m -= step
        else:
            converged = True
            sys.stderr.write(
                f"[vm4_1] V4.1 Phase 0 收敛: mag={m:.2f} n_gaia={n} (迭代{iterations}次)\n"
            )
            break

    if not converged:
        iterations = m_lim_max_iter
        logger.warning(f"V4.1 Phase 0 未收敛(达到max_iter={m_lim_max_iter}), 使用 mag={m:.2f} n_gaia={n}")
        sys.stderr.write(
            f"[vm4_1] V4.1 Phase 0 未收敛(达到max_iter={m_lim_max_iter}), "
            f"使用 mag={m:.2f} n_gaia={n}\n"
        )

    # 最终查询
    ra_all, dec_all, mag_all = gaia_client.cone_search(center_ra, center_dec, query_radius_deg, m)

    if len(ra_all) < 2:
        sys.stderr.write(f"[vm4_1] V4.1 Phase 0 兜底 mag=22 查询\n")
        ra_all, dec_all, mag_all = gaia_client.cone_search(center_ra, center_dec, query_radius_deg, 22.0)

    return {
        'ra': np.array(ra_all, dtype=np.float64),
        'dec': np.array(dec_all, dtype=np.float64),
        'mag': np.array(mag_all, dtype=np.float64),
        'm_lim_final': float(m),
        'n_gaia_final': int(len(ra_all)),
        'converged': converged,
        'iterations': iterations,
        'query_radius_deg': query_radius_deg,
        'query_area_deg2': query_area_deg2,
        'rho_img': rho_img,
        'rho_target': rho_target,
        'n_target': n_target,
    }


# ============================================================================
# V4.0 主封装类
# ============================================================================

class VectorMatchV4_1Cpp:
    """V4.0 抽样投票向量法 Python 封装

    用法:
        with VectorMatchV4_1Cpp(gaia_data_dir, db_type=1) as solver:
            result = solver.solve(img_x, img_y, img_flux, img_saturated,
                                   cra, cdec, fl, ps, w, h, exptime=1.0)
            if result:
                print(f"success: mode={result.flip_mode} n={result.matched_count} rms={result.rms_px:.3f}px")
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 1):
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._dll = ctypes.CDLL(_find_dll())
        self._dll.vm4_1_solve.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(VM4_1SolveParamsC), ctypes.POINTER(VM4_1SolveResultC),
        ]
        self._dll.vm4_1_solve.restype = ctypes.c_int
        self._closed = False

    def solve(self, img_x, img_y, img_flux, img_saturated,
              cra: float, cdec: float, fl: float, ps: float, w: int, h: int,
              wcs_out: Optional[str] = None, skip_sip: bool = False, exptime: float = 1.0,
              # V4.0 可选参数
              use_prosac: bool = True, use_kvector: bool = True,
              use_bayes: bool = True, use_triangle: bool = True,
              log_file_path: Optional[str] = None,
              # 密度匹配参数
              k_match: float = 1.5, m_lim_step: float = 0.5, m_lim_max_iter: int = 8,
              density_tolerance: float = 0.1,
              # PROSAC 参数
              w_snr: float = 0.4, w_sparse: float = 0.4, w_sat: float = 0.2,
              prosac_T_max: int = 10000,
              # k-vector 参数
              k_vector_eps: float = 2.0,
              # 贝叶斯参数
              lnK_accept: float = 20.7, lnK_weak: float = 6.9,
              # 三角形参数
              eps_A: float = 0.05, eps_J: float = 0.10, triangle_pass_rate: float = 0.8,
              # 星点选取参数
              n_img_total: int = 250,
              # V4.1 不对称选星参数
              img_n_target: int = 50,
              gaia_density_ratio: float = 1.5,
              gaia_query_radius_factor: float = 0.55,
              # 日志控制
              verbose: bool = False,
              ) -> Optional[VectorMatchResult]:
        """V4.1 完整求解流程（V4.0 不对称选星扩展）

        Phase 0 (Python): V4.1 不对称密度匹配 Gaia 查询
        Phase A→E (C++): PROSAC + SVD + k-vector + MAD + 贝叶斯/三角形 + 分层拟合

        Args:
            img_x, img_y, img_flux, img_saturated: 图像星点数组
            cra, cdec: 图像中心赤经/赤纬(度)
            fl: 焦距(mm)
            ps: 像元尺寸(um)
            w, h: 图像宽高
            wcs_out: WCS JSON 输出路径(None 则临时文件)
            skip_sip: 是否跳过 SIP 拟合
            exptime: 曝光时间(s)
            use_prosac/use_kvector/use_bayes/use_triangle: V4.0 模块开关
            log_file_path: 日志文件路径(None 则自动生成时间戳路径)
            k_match, m_lim_step, m_lim_max_iter, density_tolerance: V4.0 密度匹配参数(向后兼容, V4.1使用新参数)
            w_snr, w_sparse, w_sat, prosac_T_max: PROSAC 参数
            k_vector_eps: k-vector 角距查询容差(角秒)
            lnK_accept, lnK_weak: 贝叶斯阈值
            eps_A, eps_J, triangle_pass_rate: 三角形验证参数
            n_img_total: V4.0 图像星点总数(向后兼容, V4.1使用img_n_target)
            img_n_target: V4.1 图像侧目标星数(默认50)
                          饱和>img_n_target → 全选饱和星
                          否则 → 饱和全选 + 非饱和按flux降序补足到 img_n_target
            gaia_density_ratio: V4.1 Gaia面密度/图像面密度(默认1.5)
            gaia_query_radius_factor: V4.1 Gaia查询半径因子(默认0.55, 即0.55×FOV对角线)
            verbose: 是否输出C++ DLL的stderr日志(默认False抑制, True时输出)

        Returns:
            VectorMatchResult 或 None(失败)
        """
        s0 = 206.265 * ps / fl
        fov_diag = math.sqrt(w * w + h * h) * s0 / 3600.0

        # ====================================================================
        # 图像星点选取: V4.1 不对称选星策略
        #   饱和 > img_n_target → 全选饱和星
        #   否则 → 饱和全选 + 非饱和按flux降序补足到 img_n_target
        # ====================================================================
        img_x_arr = np.asarray(img_x, np.float64)
        img_y_arr = np.asarray(img_y, np.float64)
        img_flux_arr = np.asarray(img_flux, np.float64)
        img_sat_arr = np.asarray(img_saturated, np.bool_)

        sat_idx = np.where(img_sat_arr)[0]
        nsat = len(sat_idx)
        non_sat_idx = np.where(~img_sat_arr)[0]
        if len(non_sat_idx) > 0:
            non_sat_sorted = non_sat_idx[np.argsort(-img_flux_arr[non_sat_idx])]
        else:
            non_sat_sorted = np.array([], dtype=np.int64)

        # ====================================================================
        # V4.1 图像星点选取: 不对称选星策略
        #   饱和 > img_n_target → 全选饱和星
        #   否则 → 饱和全选 + 非饱和按flux降序补足到 img_n_target
        # ====================================================================
        if nsat > img_n_target:
            # 饱和星数量超过目标值, 全选饱和星
            sel_idx = sat_idx
            logger.info(f"V4.1选星: 饱和星{nsat}颗 > img_n_target={img_n_target}, 全选饱和星")
            sys.stderr.write(
                f"[vm4_1] V4.1选星: 饱和星{nsat}颗 > img_n_target={img_n_target}, 全选饱和星\n"
            )
        else:
            # 饱和 + 非饱和补足到 img_n_target
            n_needed = max(0, img_n_target - nsat)
            top_non_sat = non_sat_sorted[:n_needed]
            sel_idx = np.concatenate([sat_idx, top_non_sat])
            logger.info(f"V4.1选星: 饱和{nsat} + 非饱和{n_needed} = {len(sel_idx)}颗 (img_n_target={img_n_target})")
            sys.stderr.write(
                f"[vm4_1] V4.1选星: 饱和{nsat} + 非饱和{n_needed} = {len(sel_idx)}颗 (img_n_target={img_n_target})\n"
            )
        N = len(sel_idx)
        if N < 2:
            raise ValueError(f"N={N} too few image stars")

        cx, cy = w / 2.0, h / 2.0
        ux = (img_x_arr[sel_idx] - cx) * s0
        uy = -(img_y_arr[sel_idx] - cy) * s0
        U = np.column_stack([ux, uy])

        # PROSAC 质量分用 SNR: 近似为 flux 归一化
        img_flux_sel = img_flux_arr[sel_idx]
        if len(img_flux_sel) > 0:
            fmin, fmax = float(img_flux_sel.min()), float(img_flux_sel.max())
            snr_arr = (img_flux_sel - fmin) / max(fmax - fmin, 1e-10) * 100.0  # 归一化到 [0, 100]
        else:
            snr_arr = np.zeros(N, dtype=np.float64)
        sat_arr = img_sat_arr[sel_idx].astype(np.int32)

        # ====================================================================
        # Phase 0: V4.1 不对称密度匹配 Gaia 查询（Python 端完成）
        # 与V4.0区别:
        #   - 查询半径 = fov_diag × gaia_query_radius_factor (默认0.55)
        #   - 目标星数 = gaia_density_ratio × N × (查询圆面积/图像面积)
        #   - 自适应步长: 前4次m_lim_step, 后续减半
        # ====================================================================
        n_img_bright = N  # 用于密度计算的图像亮星数
        dm = density_match_query_v4_1(
            self._gaia, cra, cdec, fov_diag, n_img_bright,
            gaia_density_ratio=gaia_density_ratio,
            gaia_query_radius_factor=gaia_query_radius_factor,
            m_lim_step=m_lim_step, m_lim_max_iter=m_lim_max_iter,
            density_tolerance=density_tolerance,
            focal_length_mm=fl, exposure_time_s=exptime,
        )

        cat_ra_all = dm['ra']
        cat_dec_all = dm['dec']
        cat_mag_all = dm['mag']
        if len(cat_ra_all) < 2:
            raise ValueError(f"Gaia 查询返回 {len(cat_ra_all)} 颗星")

        sys.stderr.write(
            f"[vm4_1] V4.1 Phase 0 完成: m_lim={dm['m_lim_final']:.2f} n_gaia={dm['n_gaia_final']} "
            f"ρ_img={dm['rho_img']:.3f} ρ_target={dm['rho_target']:.3f} "
            f"iter={dm['iterations']} converged={dm['converged']}\n"
        )

        # gnomonic 投影筛选 FOV 内星
        xi_all, eta_all, valid_all = gnomonic_forward(cat_ra_all, cat_dec_all, cra, cdec)
        fov_half_w = w / 2 * s0
        fov_half_h = h / 2 * s0
        in_fov = valid_all & (np.abs(xi_all) < fov_half_w) & (np.abs(eta_all) < fov_half_h)
        fov_idx = np.where(in_fov)[0]

        if len(fov_idx) < 2:
            # FOV 内星太少，放宽到 1.5x FOV
            in_fov = valid_all & (np.abs(xi_all) < fov_half_w * 1.5) & (np.abs(eta_all) < fov_half_h * 1.5)
            fov_idx = np.where(in_fov)[0]

        if len(fov_idx) < 2:
            raise ValueError(f"FOV 内只有 {len(fov_idx)} 颗 Gaia 星")

        sys.stderr.write(f"[vm4_1] FOV 内 Gaia 星: {len(fov_idx)}/{len(cat_ra_all)}\n")

        # 按星等排序，取最亮的 N_target 颗（不超过 FOV 内总数）
        n_target = dm['n_target']
        fov_mag = cat_mag_all[fov_idx]
        sorted_order = np.argsort(fov_mag)
        sel_order = sorted_order[:min(n_target, len(sorted_order))]
        sel_idx_cat = fov_idx[sel_order]

        cat_ra = cat_ra_all[sel_idx_cat]
        cat_dec = cat_dec_all[sel_idx_cat]
        M = len(cat_ra)

        W = _build_catalog_vectors(cat_ra, cat_dec, cra, cdec)

        # ====================================================================
        # WCS JSON 输出路径
        # ====================================================================
        wcs_json_path = wcs_out
        if wcs_json_path is None:
            import tempfile
            fd, wcs_json_path = tempfile.mkstemp(suffix='.json', prefix='vm4_1_wcs_')
            os.close(fd)

        # ====================================================================
        # 日志文件路径（V4.0 新增）
        # ====================================================================
        if log_file_path is None:
            log_file_path = _make_log_path()
        sys.stderr.write(f"[vm4_1] 日志文件: {log_file_path}\n")

        # ====================================================================
        # 构造 VM4_1SolveParamsC 并调用 vm4_1_solve
        # ====================================================================
        params = VM4_1SolveParamsC()
        # V3.5 继承字段
        params.s0 = s0
        params.s_min = 0.9
        params.s_max = 1.1
        params.n_modes = 4
        params.seed = 42
        params.K_total = 10000
        params.batch_size = 1000
        params.min_samples = 2000
        params.K_top = 50
        params.min_inliers = max(5, int(N * 0.1))
        params.fov_diag_asec = fov_diag * 3600.0
        params.img_width = float(w)
        params.img_height = float(h)
        params.center_ra = float(cra)
        params.center_dec = float(cdec)
        params.wcs_out_path = wcs_json_path.encode('utf-8') if isinstance(wcs_json_path, str) else wcs_json_path
        params.skip_sip = 1 if skip_sip else 0
        params.expand_n_gaia = 1500
        params.expand_n_img = 1000
        params.radial_n_bins = 20
        params.radial_fit_order = 3
        params.radial_n_iters = 3

        # V4.0 新增字段 - Phase 0
        params.k_match = k_match
        params.query_radius_factor = 1.0
        params.m_lim_step = m_lim_step
        params.m_lim_max_iter = m_lim_max_iter
        params.density_tolerance = density_tolerance
        params.n_img_bright = n_img_bright
        params.focal_length_mm = fl
        params.pixel_size_um = ps
        params.exposure_time_s = exptime

        # V4.0 新增字段 - Phase C: k-vector
        params.k_vector_eps = k_vector_eps
        params.use_kvector = 1 if use_kvector else 0

        # V4.0 新增字段 - Phase A: PROSAC
        params.w_snr = w_snr
        params.w_sparse = w_sparse
        params.w_sat = w_sat
        params.prosac_T_max = prosac_T_max
        params.use_prosac = 1 if use_prosac else 0

        # V4.0 新增字段 - Phase D': 贝叶斯
        params.lnK_accept = lnK_accept
        params.lnK_weak = lnK_weak
        params.use_bayes = 1 if use_bayes else 0

        # V4.0 新增字段 - Phase D': 三角形
        params.eps_A = eps_A
        params.eps_J = eps_J
        params.triangle_pass_rate = triangle_pass_rate
        params.use_triangle = 1 if use_triangle else 0

        # V4.0 Task 7 可选输入字段
        snr_arr_c = np.ascontiguousarray(snr_arr, dtype=np.float64)
        sat_arr_c = np.ascontiguousarray(sat_arr, dtype=np.int32)
        params.snr_values = snr_arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        params.is_saturated_values = sat_arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        params.log_file_path = log_file_path.encode('utf-8') if isinstance(log_file_path, str) else log_file_path

        # V4.1 不对称选星参数
        params.img_n_target = img_n_target
        params.gaia_density_ratio = gaia_density_ratio
        params.gaia_query_radius_factor = gaia_query_radius_factor

        # 调用 DLL
        imask = np.zeros(N, dtype=np.int32)
        result = VM4_1SolveResultC()
        result.inlier_mask = imask.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # V4.1: stderr 抑制 — C++ DLL 通过 fprintf(stderr,...) 输出大量日志(Phase A 每次抽样都输出)
        # 默认 verbose=False 抑制 stderr(用 os.dup2 重定向 fd 2 到 /dev/null), True 时正常输出
        # 仅影响 stderr 输出, 不修改 C++ DLL 代码, 无需重编译
        if not verbose:
            devnull = open(os.devnull, 'w')
            old_stderr = os.dup(2)
            os.dup2(devnull.fileno(), 2)
            devnull.close()
        try:
            ret = self._dll.vm4_1_solve(
                np.ascontiguousarray(U, np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N,
                np.ascontiguousarray(W, np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
                ctypes.byref(params), ctypes.byref(result),
            )
        finally:
            if not verbose:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)

        # ====================================================================
        # 结果解析
        # ====================================================================
        if ret != 0 or result.success == 0:
            logger.warning("vm4_1_solve fail: ret=%d succ=%d", ret, result.success)
            # 即使失败也输出调试信息便于分析
            if result.debug.bayes_lnK != 0 or result.debug.triangle_total != 0:
                sys.stderr.write(
                    f"[vm4_1] 失败但已有部分结果: lnK={result.debug.bayes_lnK:.2f} "
                    f"tri_ratio={result.debug.triangle_pass_ratio:.3f} "
                    f"n_phased_clean={result.debug.n_phased_clean}\n"
                )
            return None
        if result.s < 0.9 or result.s > 1.1:
            logger.warning("s=%.4f out of range", result.s)
            return None

        s, theta, tx, ty = result.s, result.theta, result.tx, result.ty
        best_mode = result.best_mode
        cos_d = math.cos(cdec * _DEGTORAD)
        dra = -tx / (max(cos_d, 1e-10) * 3600.0)
        ddec = -ty / 3600.0
        cur_ra, cur_dec = cra + dra, cdec + ddec

        Wf = _apply_flip(W, best_mode)
        inl_mask = imask.astype(bool)
        s_final = s0 * s
        rot_deg = math.degrees(theta)

        # 计算 RMS（用变换后 NN 距离）
        rms_px, rms_asec = 0.0, 0.0
        if np.any(inl_mask):
            Wt_arr = _apply_similarity(Wf, s, theta, tx, ty)
            from scipy.spatial import cKDTree
            tree = cKDTree(Wt_arr)
            dists, idxs = tree.query(U, k=1)
            diffs = U[inl_mask] - Wt_arr[idxs[inl_mask]]
            rms_asec = float(np.sqrt(np.mean(np.sum(diffs ** 2, axis=1))))
            rms_px = rms_asec / s0

        ct, st = math.cos(theta), math.sin(theta)
        affine = (tx, s * ct, -s * st, ty, s * st, s * ct)
        r = VectorMatchResult(
            center_ra=cur_ra, center_dec=cur_dec,
            original_ra=cra, original_dec=cdec,
            rotation_deg=rot_deg,
            scale_arcsec_px=s_final, flip_mode=best_mode,
            matched_count=int(np.sum(inl_mask)),
            rms_px=rms_px, rms_arcsec=rms_asec, affine=affine,
        )
        r.solve_tx = tx
        r.solve_ty = ty
        r.solve_s = s
        r.s0 = s0

        # V3.5 调试信息
        r.theta_snr = float(result.debug.theta_snr)
        r.theta_peak_deg = float(result.debug.theta_peak_deg)
        r.best_n_range = int(result.debug.best_n_range)
        r.median_noise = float(result.debug.median_noise)
        r.n_phaseb_pairs = int(result.debug.n_phaseb_pairs)
        r.n_phaseb_corr = int(result.debug.n_phaseb_corr)
        r.n_phasea_records = int(result.debug.n_phasea_records)
        r.n_phasec_expanded = int(result.debug.n_phasec_expanded)
        r.n_phased_clean = int(result.debug.n_phased_clean)
        r.n_phased_iterations = int(result.debug.n_phased_iterations)
        r.mad_rms_arcsec = float(result.debug.mad_rms_arcsec)
        r.n_expand_mutual = int(result.debug.n_expand_mutual)
        r.n_expand_after_filter = int(result.debug.n_expand_after_filter)
        r.n_sip_total = int(result.debug.n_sip_total)
        r.sip_order = int(result.debug.sip_order)

        # V4.0 新增调试信息
        r.rho_img = float(result.debug.rho_img)
        r.rho_target = float(result.debug.rho_target)
        # Python 端填充实际密度查询值（C++ 端记录的是估算值）
        r.m_lim_final = dm['m_lim_final']
        r.n_gaia_final = dm['n_gaia_final']
        # V4.1 density_match_query_v4_1 返回 'iterations'，向后兼容 'm_lim_iterations'
        r.m_lim_iterations = dm.get('iterations', dm.get('m_lim_iterations', 0))
        r.kvector_build_ms = float(result.debug.kvector_build_ms)
        r.kvector_queries = int(result.debug.kvector_queries)
        r.kvector_avg_candidates = float(result.debug.kvector_avg_candidates)
        r.prosac_quality_median = float(result.debug.prosac_quality_median)
        r.prosac_pool_final = int(result.debug.prosac_pool_final)
        r.bayes_lnK = float(result.debug.bayes_lnK)
        r.bayes_n_match = int(result.debug.bayes_n_match)
        r.bayes_decision = int(result.debug.bayes_decision)
        r.triangle_total = int(result.debug.triangle_total)
        r.triangle_pass_ratio = float(result.debug.triangle_pass_ratio)

        sys.stderr.write(
            f"[vm4_1] 调试: lnK={r.bayes_lnK:.2f} tri_ratio={r.triangle_pass_ratio:.3f} "
            f"kv_build={r.kvector_build_ms:.1f}ms prosac_pool={r.prosac_pool_final} "
            f"m_lim={r.m_lim_final:.2f} n_gaia={r.n_gaia_final}\n"
        )

        # 从 JSON 文件读取 WCS 参数（与 V3.5 格式兼容）
        if os.path.exists(wcs_json_path):
            try:
                wcs_data = _read_wcs_json(wcs_json_path)
                r.cd = wcs_data['CD']
                r.crval = wcs_data['CRVAL']
                r.crpix = wcs_data['CRPIX']
                r.sip_A = wcs_data['SIP_A'].reshape(6, 6)
                r.sip_B = wcs_data['SIP_B'].reshape(6, 6)
                r.sip_rms_px = wcs_data['RMS_PX']
                logger.info("WCS JSON loaded: CD=%s CRVAL=%s SIP_RMS=%.3fpx",
                            r.cd.tolist(), r.crval.tolist(), r.sip_rms_px)
            except Exception as e:
                logger.warning("WCS JSON read failed: %s, fallback to ctypes", e)
                r.sip_rms_px = float(result.rms)
                r.cd = np.array([[float(result.cd[0]), float(result.cd[1])],
                                 [float(result.cd[2]), float(result.cd[3])]], dtype=np.float64)
                r.crval = np.array([float(result.crval[0]), float(result.crval[1])])
                r.crpix = np.array([float(result.crpix[0]), float(result.crpix[1])])
                r.sip_A = np.array([float(result.sip_A[i]) for i in range(36)], dtype=np.float64).reshape(6, 6)
                r.sip_B = np.array([float(result.sip_B[i]) for i in range(36)], dtype=np.float64).reshape(6, 6)
            if wcs_out is None:
                try:
                    os.remove(wcs_json_path)
                except OSError:
                    pass
        else:
            logger.warning("WCS JSON not found: %s", wcs_json_path)
            r.sip_rms_px = float(result.rms)
            r.cd = np.array([[float(result.cd[0]), float(result.cd[1])],
                             [float(result.cd[2]), float(result.cd[3])]], dtype=np.float64)
            r.crval = np.array([float(result.crval[0]), float(result.crval[1])])
            r.crpix = np.array([float(result.crpix[0]), float(result.crpix[1])])
            r.sip_A = np.zeros((6, 6), dtype=np.float64)
            r.sip_B = np.zeros((6, 6), dtype=np.float64)

        return r

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

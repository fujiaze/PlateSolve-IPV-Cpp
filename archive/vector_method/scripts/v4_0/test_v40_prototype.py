"""
V4.0 两阶段收敛 Platesolve — Numpy原型验证脚本

功能: 纯numpy/scipy实现V4.0核心算法, 验证两阶段+三进程同步架构
用途: 在投入C++开发前, 快速验证算法设计的可行性

核心架构 (详见 v4_0_design.md §2):
  第一阶段: 初始池探索 (三进程同步while循环)
    进程A: 指纹加权不重复抽样 → 约束跳过(进程B) → SCM评分 → 入控制点池
    进程B: 用模型约束引导抽样, 跳过非法配对
    进程C: RANSAC实时清洗 + 双向不匹配检查
    结束条件: RMS < 0.5px 且 无双向不匹配点 → LSQ拟合CD+SIP
  第二阶段: 扩增池匹配
    图像侧flux前2000 + 星表侧Gmag前5000
    CD+SIP模型投影 → 邻近双向匹配 → RANSAC → 全部控制点重拟合CD+SIP
"""

import sys, os, math, time, json
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import (
    GaiaClientPy, _DEGTORAD, _RADTOASEC, _ASECTORAD,
    gnomonic_forward, gnomonic_inverse,
    _apply_flip, _apply_similarity,
)


# ============================================================================
# 工具函数
# ============================================================================

def umeyama_svd(src, dst):
    """Umeyama SVD 相似变换: dst ≈ s·R·src + t

    Args:
        src: (K, 2) 源点 (星表侧Wf)
        dst: (K, 2) 目标点 (图像侧U)
    Returns:
        s, theta, tx, ty
    """
    K = src.shape[0]
    if K < 2:
        return 1.0, 0.0, 0.0, 0.0
    mu_src = np.mean(src, axis=0)
    mu_dst = np.mean(dst, axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.sum(src_c ** 2) / K
    cov = (dst_c.T @ src_c) / K
    U, S, Vt = np.linalg.svd(cov)

    det = np.linalg.det(U) * np.linalg.det(Vt)
    if det < 0:
        S[-1] *= -1.0
        U[:, -1] *= -1.0

    R = U @ Vt
    s = np.sum(S) / var_src if var_src > 1e-15 else 1.0
    t = mu_dst - s * R @ mu_src
    theta = math.atan2(R[1, 0], R[0, 0])

    return s, theta, t[0], t[1]


def fit_cd_lsq(pairs_px_sky, crpix1, crpix2, crval_ra, crval_dec):
    """LSQ直接拟合CD矩阵: 像素→天空度

    Args:
        pairs_px_sky: list of (x_px, y_px, ra_deg, dec_deg)
        crpix1, crpix2: 参考像素
        crval_ra, crval_dec: 参考天球坐标 (度)
    Returns:
        cd: 2x2 CD矩阵 [[cd00, cd01], [cd10, cd11]]
    """
    n = len(pairs_px_sky)
    if n < 3:
        return None

    cos_d = math.cos(crval_dec * math.pi / 180.0)
    A = np.zeros((2 * n, 4), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)

    for i, (x, y, ra, dec) in enumerate(pairs_px_sky):
        dx = x - crpix1
        dy = y - crpix2
        dra_cosd = (ra - crval_ra) * cos_d
        ddec = dec - crval_dec
        A[2*i,   0] = dx; A[2*i,   1] = dy
        A[2*i+1, 2] = dx; A[2*i+1, 3] = dy
        b[2*i]   = dra_cosd
        b[2*i+1] = ddec

    x_vec, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cd = np.array([[x_vec[0], x_vec[1]],
                    [x_vec[2], x_vec[3]]])
    return cd


def fit_sip_bic(pairs_px_sky, cd, crpix1, crpix2, crval_ra, crval_dec, max_order=2):
    """BIC逐阶SIP拟合: 在仿射残差上拟合高阶SIP系数

    Args:
        pairs_px_sky: list of (x_px, y_px, ra_deg, dec_deg)
        cd: 2x2 CD矩阵
        max_order: SIP最大阶数
    Returns:
        sip_order, sip_A, sip_B (6x6矩阵), rms_px
    """
    n = len(pairs_px_sky)
    if n < 10:
        return 0, np.zeros((6,6)), np.zeros((6,6)), 0.0

    cos_d = math.cos(crval_dec * math.pi / 180.0)
    cd_inv = np.linalg.inv(cd)

    # 计算线性残差: sky → CD逆 → 像素偏移
    residuals = []  # (dx, dy, x_norm, y_norm)
    for x, y, ra, dec in pairs_px_sky:
        dra_cosd = (ra - crval_ra) * cos_d
        ddec = dec - crval_dec
        sky_off = np.array([dra_cosd, ddec])
        px_off = cd_inv @ sky_off  # 线性预测的像素偏移
        actual_dx = x - crpix1
        actual_dy = y - crpix2
        res_x = actual_dx - px_off[0]
        res_y = actual_dy - px_off[1]
        # 归一化中心在CRPIX(0,0)
        residuals.append((res_x, res_y, actual_dx, actual_dy))

    res_arr = np.array(residuals)
    res_x = res_arr[:, 0]
    res_y = res_arr[:, 1]
    xn = res_arr[:, 2]  # 归一化像素坐标 (以CRPIX为原点)
    yn = res_arr[:, 3]

    best_order = 0
    best_bic = np.inf
    best_A = np.zeros((6,6))
    best_B = np.zeros((6,6))
    best_rms = float(np.sqrt(np.mean(res_x**2 + res_y**2)))

    for order in range(2, max_order + 1):
        # 构建多项式基: 对SIP, p+q >= 2 且 p+q <= order
        terms = []
        for p in range(order + 1):
            for q in range(order + 1):
                if p + q < 2 or p + q > order:
                    continue
                terms.append((p, q))

        k = len(terms)
        if 2 * n <= k:
            break

        # 构建设计矩阵
        M_A = np.zeros((n, k), dtype=np.float64)
        M_B = np.zeros((n, k), dtype=np.float64)
        for j, (p, q) in enumerate(terms):
            M_A[:, j] = xn**p * yn**q
            M_B[:, j] = xn**p * yn**q

        # LSQ拟合
        coeff_A, _, _, _ = np.linalg.lstsq(M_A, res_x, rcond=None)
        coeff_B, _, _, _ = np.linalg.lstsq(M_B, res_y, rcond=None)

        # 残差
        pred_x = M_A @ coeff_A
        pred_y = M_B @ coeff_B
        rms = float(np.sqrt(np.mean((res_x - pred_x)**2 + (res_y - pred_y)**2)))

        # BIC
        bic = n * math.log(max(rms**2, 1e-30)) + k * math.log(n)

        if bic < best_bic:
            best_bic = bic
            best_order = order
            best_rms = rms
            best_A = np.zeros((6,6))
            best_B = np.zeros((6,6))
            for j, (p, q) in enumerate(terms):
                best_A[p, q] = coeff_A[j]
                best_B[p, q] = coeff_B[j]

    return best_order, best_A, best_B, best_rms


# ============================================================================
# V4.0 SIGNATURE: 3-NN 构型签名
# ============================================================================

def compute_3nn_signatures(points):
    """预计算每颗星的3-NN构型签名

    Args:
        points: (N, 2) 点集, 单位角秒

    Returns:
        signatures: (N, 3) 3维签名向量 (f1, f2, f3)
            f1 = d1 / d_median  # 归一化最近邻距离 (密度特征)
            f2 = d2 / d1        # 第2/第1近邻距离比 (尺度无关形状)
            f3 = angle / π      # 两近邻夹角 (旋转不变量)
    """
    N = points.shape[0]
    if N < 4:
        return np.zeros((N, 3), dtype=np.float64)

    tree = cKDTree(points)
    dists, idxs = tree.query(points, k=4)  # k=4: 第0个是自身

    d1 = dists[:, 1]
    d2 = dists[:, 2]
    q1_idx = idxs[:, 1]
    q2_idx = idxs[:, 2]

    d_median = np.median(d1[d1 > 0]) if np.any(d1 > 0) else 1.0

    f1 = d1 / max(d_median, 1e-10)
    f2 = np.where(d1 > 1e-10, d2 / d1, 0.0)

    pq1 = points[q1_idx] - points
    pq2 = points[q2_idx] - points
    norm1 = np.linalg.norm(pq1, axis=1)
    norm2 = np.linalg.norm(pq2, axis=1)
    dots = np.sum(pq1 * pq2, axis=1)
    denom = norm1 * norm2
    cos_angle = np.clip(dots / np.maximum(denom, 1e-10), -1.0, 1.0)
    f3 = np.arccos(cos_angle) / np.pi

    return np.column_stack([f1, f2, f3])


# ============================================================================
# V4.0 COVERAGE: 多尺度空间覆盖度
# ============================================================================

def compute_coverage(U_pool, Wt, halfW, halfH, center_ratio=0.6, center_res=16, edge_res=8):
    """计算自适应多尺度网格的空间覆盖度

    Args:
        U_pool: (N, 2) 图像侧池向量
        Wt:     (M, 2) 变换后星表池向量
        halfW, halfH: 图像半宽/半高 (角秒)

    Returns:
        coverage: [0, 1] 直方图交集/并集
    """
    cx_lo, cx_hi = -center_ratio * halfW, center_ratio * halfW
    cy_lo, cy_hi = -center_ratio * halfH, center_ratio * halfH

    total_bins = center_res * center_res + 4 * edge_res * edge_res
    H_U = np.zeros(total_bins, dtype=np.float64)
    H_Wt = np.zeros(total_bins, dtype=np.float64)

    cx_bins = np.linspace(cx_lo, cx_hi, center_res + 1)
    cy_bins = np.linspace(cy_lo, cy_hi, center_res + 1)

    for pts, hist in [(U_pool, H_U), (Wt, H_Wt)]:
        # 中心区域
        mask_c = (pts[:, 0] >= cx_lo) & (pts[:, 0] < cx_hi) & \
                 (pts[:, 1] >= cy_lo) & (pts[:, 1] < cy_hi)
        if np.any(mask_c):
            xi = np.digitize(pts[mask_c, 0], cx_bins[:-1]) - 1
            yi = np.digitize(pts[mask_c, 1], cy_bins[:-1]) - 1
            xi = np.clip(xi, 0, center_res - 1)
            yi = np.clip(yi, 0, center_res - 1)
            gidx = yi * center_res + xi
            np.add.at(hist, gidx, 1.0)

        # 边缘上
        mask_t = pts[:, 1] >= cy_hi
        if np.any(mask_t):
            eb = np.linspace(-halfW, halfW, edge_res + 1)
            xi = np.digitize(pts[mask_t, 0], eb[:-1]) - 1
            xi = np.clip(xi, 0, edge_res - 1)
            gidx = center_res * center_res + xi
            np.add.at(hist, gidx, 1.0)

        # 边缘下
        mask_b = pts[:, 1] < cy_lo
        if np.any(mask_b):
            eb = np.linspace(-halfW, halfW, edge_res + 1)
            xi = np.digitize(pts[mask_b, 0], eb[:-1]) - 1
            xi = np.clip(xi, 0, edge_res - 1)
            gidx = center_res * center_res + edge_res + xi
            np.add.at(hist, gidx, 1.0)

        # 边缘左
        mask_l = (pts[:, 0] < cx_lo) & ~(mask_t | mask_b)
        if np.any(mask_l):
            eb = np.linspace(-halfH, halfH, edge_res + 1)
            yi = np.digitize(pts[mask_l, 1], eb[:-1]) - 1
            yi = np.clip(yi, 0, edge_res - 1)
            gidx = center_res * center_res + 2 * edge_res + yi
            np.add.at(hist, gidx, 1.0)

        # 边缘右
        mask_r = (pts[:, 0] >= cx_hi) & ~(mask_t | mask_b)
        if np.any(mask_r):
            eb = np.linspace(-halfH, halfH, edge_res + 1)
            yi = np.digitize(pts[mask_r, 1], eb[:-1]) - 1
            yi = np.clip(yi, 0, edge_res - 1)
            gidx = center_res * center_res + 3 * edge_res + yi
            np.add.at(hist, gidx, 1.0)

    sum_min = np.sum(np.minimum(H_U, H_Wt))
    sum_max = np.sum(np.maximum(H_U, H_Wt))

    return sum_min / max(sum_max, 1.0)


# ============================================================================
# V4.0 SCM: 结构一致性度量
# ============================================================================

def compute_scm(U_pool, Wt, halfW, halfH, U_sigs, eps_config=0.15, alpha=0.3, beta=0.7):
    """计算 SCM = Coverage^alpha × Configuration^beta

    Args:
        U_pool:   (N, 2) 图像侧池向量
        Wt:       (M, 2) 变换后星表池向量
        U_sigs:   (N, 3) U池预计算的构型签名
        eps_config: 签名归一化距离阈值

    Returns:
        scm, coverage, n_confirmed
    """
    N = U_pool.shape[0]

    coverage = compute_coverage(U_pool, Wt, halfW, halfH)

    Wt_sigs = compute_3nn_signatures(Wt)
    M = Wt.shape[0]

    n_confirmed = 0
    if M >= 4:
        tree_W = cKDTree(Wt)
        for i in range(N):
            u = U_pool[i]
            dists_W, nn_W = tree_W.query(u, k=min(5, M))
            if isinstance(dists_W, float):
                dists_W = [dists_W]; nn_W = [nn_W]
            matched = False
            for j_idx, j in enumerate(nn_W):
                sig_diff = np.linalg.norm(U_sigs[i] - Wt_sigs[j])
                if sig_diff < eps_config:
                    matched = True
                    break
            if matched:
                n_confirmed += 1

    config = n_confirmed / max(N, 1)
    scm = (coverage ** alpha) * (config ** beta)

    return scm, coverage, n_confirmed


# ============================================================================
# V4.0 指纹加权
# ============================================================================

def compute_fingerprint_weights(U_all, pool_idx, all_tree):
    """计算初始池中每颗星的指纹权重

    权重 = d_bright / max(d_global, 1)
    比值接近1 → 该星周围几乎全是亮星 → 构型独特

    Args:
        U_all: (N_all, 2) 全部图像星向量
        pool_idx: 初始池中的星索引
        all_tree: 全部检测星的KDTree
    Returns:
        weights: (len(pool_idx),) 归一化权重
    """
    n_pool = len(pool_idx)
    if n_pool == 0:
        return np.array([])

    U_pool = U_all[pool_idx]

    # 全局密度: 用全部检测星计算
    avg_nn_dist_all = np.median(all_tree.query(U_all, k=2)[0][:, 1]) if U_all.shape[0] > 1 else 1.0
    r_global = avg_nn_dist_all * 3.0

    d_global = np.zeros(n_pool)
    d_bright = np.zeros(n_pool)

    # 全局密度: 全部检测星在半径r_global内的数量
    global_counts = all_tree.query_ball_point(U_pool, r_global)
    for i in range(n_pool):
        d_global[i] = len(global_counts[i])

    # 亮星密度: 池内星在半径r_global内的数量
    if n_pool > 1:
        pool_tree = cKDTree(U_pool)
        bright_counts = pool_tree.query_ball_point(U_pool, r_global)
        for i in range(n_pool):
            d_bright[i] = len(bright_counts[i])
    else:
        d_bright[:] = 1.0

    weights = d_bright / np.maximum(d_global, 1.0)

    # 归一化到概率分布
    w_sum = np.sum(weights)
    if w_sum > 0:
        weights = weights / w_sum
    else:
        weights = np.ones(n_pool) / n_pool

    return weights


# ============================================================================
# V4.0 主类: 两阶段收敛求解器
# ============================================================================

class V40PrototypeSolver:
    """V4.0 两阶段收敛 Platesolve — Numpy原型

    第一阶段: 初始池探索 (三进程同步while循环)
      进程A: 指纹加权不重复抽样 → SCM评分 → 入控制点池
      进程B: 模型约束跳过非法配对
      进程C: RANSAC实时清洗 + 双向不匹配检查
      结束: RMS < 0.5px → LSQ拟合CD+SIP

    第二阶段: 扩增池匹配
      CD+SIP模型投影 → 邻近双向匹配 → RANSAC → 重拟合CD+SIP
    """

    def __init__(self, gaia_data_dir, db_type=1):
        self.gaia = GaiaClientPy(gaia_data_dir, db_type)

    def solve(self, img_x, img_y, img_flux, img_saturated,
              cra, cdec, fl, ps, w, h, exptime=1.0,
              K_max=5000, tau_scm=0.40, tau_nmin=3,
              verbose=True):
        """运行V4.0两阶段收敛

        Returns:
            dict with: success, best_mode, rms_px, wcs_data, ...
        """
        t0 = time.perf_counter()
        s0 = 206.265 * ps / fl
        halfW = w / 2.0 * s0
        halfH = h / 2.0 * s0
        fov_diag = math.sqrt(w * w + h * h) * s0

        # ========== 1. 构建全量U向量 ==========
        img_x_arr = np.asarray(img_x, np.float64)
        img_y_arr = np.asarray(img_y, np.float64)
        img_flux_arr = np.asarray(img_flux, np.float64)
        img_sat_arr = np.asarray(img_saturated, np.bool_)
        cx, cy = w / 2.0, h / 2.0

        sat_idx = np.where(img_sat_arr)[0]
        non_sat_idx = np.where(~img_sat_arr)[0]
        if len(non_sat_idx) > 0:
            non_sat_sorted = non_sat_idx[np.argsort(-img_flux_arr[non_sat_idx])]
        else:
            non_sat_sorted = np.array([], dtype=np.int64)
        all_idx = np.concatenate([sat_idx, non_sat_sorted])
        ux_all = (img_x_arr[all_idx] - cx) * s0
        uy_all = -(img_y_arr[all_idx] - cy) * s0  # Y轴翻转: 图像Y↓ 天球Dec↑
        U_all = np.column_stack([ux_all, uy_all])
        N_all = len(all_idx)
        nsat = len(sat_idx)

        if verbose:
            print(f"[V4] 图像星点: {N_all}颗 (饱和{nsat}), s0={s0:.4f}\"/px, FOV={fov_diag/3600:.2f}°")

        # ========== 2. Gaia查询 + 构建全量W向量 ==========
        m_cut = 6.0 + 1.5 * math.log10(max(fl, 1.0)) + 2.0 * math.log10(max(exptime, 0.1))
        query_radius = fov_diag * 0.5 / 3600.0
        mag_query = m_cut
        for attempt in range(10):
            ra_t, dec_t, mag_t = self.gaia.cone_search(cra, cdec, query_radius, mag_query)
            if len(ra_t) >= 200:
                break
            mag_query += 0.5
        else:
            ra_t, dec_t, mag_t = self.gaia.cone_search(cra, cdec, query_radius, 22.0)

        if len(ra_t) < 10:
            return {'success': False, 'error': f'Gaia仅{len(ra_t)}颗'}

        xi_all, eta_all, valid_all = gnomonic_forward(ra_t, dec_t, cra, cdec)
        in_fov = valid_all & (np.abs(xi_all) < halfW * 2.0) & (np.abs(eta_all) < halfH * 2.0)
        fov_idx = np.where(in_fov)[0]
        if len(fov_idx) < 10:
            return {'success': False, 'error': f'FOV内仅{len(fov_idx)}颗Gaia星'}

        fov_mag = mag_t[fov_idx]
        sorted_mag = np.argsort(fov_mag)
        W_all = np.column_stack([xi_all[fov_idx[sorted_mag]], eta_all[fov_idx[sorted_mag]]])
        M_all = W_all.shape[0]

        # Gaia原始RA/Dec (用于MATCH_PAIRS和CD拟合)
        gaia_ra_fov = ra_t[fov_idx[sorted_mag]]
        gaia_dec_fov = dec_t[fov_idx[sorted_mag]]

        if verbose:
            print(f"[V4] Gaia查询: mag={mag_query:.1f}, 返回{len(ra_t)}颗, FOV内{M_all}颗")

        # ========== 3. 4种翻转模式并行 (顺序执行模拟) ==========
        best_result = None
        best_rms = 1e30
        for mode in range(4):
            Wf_all = _apply_flip(W_all, mode)
            result = self._run_mode(
                U_all, Wf_all, s0, halfW, halfH,
                K_max, tau_scm, tau_nmin, mode, verbose,
                gaia_ra_fov, gaia_dec_fov, cra, cdec, w, h,
                img_x_arr, img_y_arr, all_idx, N_all, M_all
            )
            if result['success']:
                if result['rms_px'] < best_rms or \
                   (abs(result['rms_px'] - best_rms) < 0.01 and
                    result.get('n_pairs', 0) > (best_result.get('n_pairs', 0) if best_result else 0)):
                    best_result = result
                    best_result['best_mode'] = mode
                    best_rms = result['rms_px']

        elapsed = time.perf_counter() - t0
        if best_result is None:
            return {'success': False, 'error': '所有模式失败', 'time': elapsed}
        best_result['time'] = elapsed
        if verbose:
            print(f"\n[V4] 完成! mode={best_result['best_mode']}, "
                  f"RMS={best_result['rms_px']:.4f}px, "
                  f"耗时={elapsed:.3f}s")
        return best_result

    # ========================================================================
    # 单模式: 两阶段收敛
    # ========================================================================

    def _run_mode(self, U_all, Wf_all, s0, halfW, halfH,
                  K_max, tau_scm, tau_nmin, mode, verbose,
                  gaia_ra, gaia_dec, cra, cdec, w, h,
                  img_x_arr, img_y_arr, all_idx, N_all, M_all):
        """单模式两阶段收敛"""

        # ========== 初始池构建 ==========
        N_init = min(50, N_all)   # 图像侧: 饱和星+亮星 ~30-50
        M_init = min(100, M_all)  # 星表侧: 最亮Gaia ~100

        U_init = U_all[:N_init].copy()
        Wf_init = Wf_all[:M_init].copy()

        # SCM始终在亮星子集上计算
        U_bright = U_all[:min(50, N_all)].copy()
        U_bright_sigs = compute_3nn_signatures(U_bright)

        # 指纹权重
        all_tree = cKDTree(U_all)
        u_weights = compute_fingerprint_weights(U_all, np.arange(N_init), all_tree)

        # 不重复抽样位图: N_init × M_init
        visited = np.zeros(N_init * M_init, dtype=np.bool_)

        # 控制点池: list of (u_idx, w_idx) — 初始池内索引
        cp_pool = []           # 控制点对 (u_init_idx, w_init_idx)
        cp_outlier_count = {}  # pair_key → 离群计数

        # 模型状态
        model = None           # dict: s, theta, tx, ty
        theta_constraint = 15.0 * math.pi / 180.0  # 初始方向约束 15°

        # 统计
        k_total = 0
        k_since_last_cp = 0
        n_ransac_rounds = 0
        n_bidir_removed = 0

        rng = np.random.RandomState(42 + mode * 137)

        if verbose:
            print(f"  [V4 mode={mode}] 初始池: U={N_init} W={M_init}, 配对空间={N_init*M_init}")
            # 初始池空间分布
            init_px_x = img_x_arr[all_idx[:N_init]]
            init_px_y = img_y_arr[all_idx[:N_init]]
            print(f"  [V4 mode={mode}] 初始池U星分布: x=[{init_px_x.min():.0f},{init_px_x.max():.0f}] "
                  f"y=[{init_px_y.min():.0f},{init_px_y.max():.0f}] (图像{w}x{h})")

        # ========== 第一阶段: 三进程同步while循环 ==========
        phase1_done = False
        while k_total < K_max:
            # --- 进程A: 指纹加权不重复抽样 ---
            pair_found = False
            max_retries = 50  # 避免死循环
            for retry in range(max_retries):
                # 按指纹权重选U侧星
                u_idx = rng.choice(N_init, p=u_weights)
                w_idx = rng.randint(M_init)
                pair_key = u_idx * M_init + w_idx

                if visited[pair_key]:
                    continue

                # 标记为已访问
                visited[pair_key] = True
                pair_found = True
                break

            if not pair_found:
                # 初始池配对空间耗尽
                if verbose:
                    n_visited = int(np.sum(visited))
                    print(f"  [V4 mode={mode}] k={k_total} 初始池耗尽 (已访问{n_visited}/{N_init*M_init})")
                break

            # --- 计算单点变换 (s, θ, tx, ty) ---
            u_vec = U_init[u_idx]
            w_vec = Wf_init[w_idx]
            u_norm = math.hypot(u_vec[0], u_vec[1])
            w_norm = math.hypot(w_vec[0], w_vec[1])
            if w_norm < 1e-10:
                k_total += 1; k_since_last_cp += 1; continue

            s = u_norm / w_norm
            if s < 0.85 or s > 1.15:
                k_total += 1; k_since_last_cp += 1; continue

            theta = math.atan2(u_vec[1], u_vec[0]) - math.atan2(w_vec[1], w_vec[0])
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            tx = u_vec[0] - s * (cos_t * w_vec[0] - sin_t * w_vec[1])
            ty = u_vec[1] - s * (sin_t * w_vec[0] + cos_t * w_vec[1])

            if abs(tx) > 0.6 * halfW * 2 or abs(ty) > 0.6 * halfH * 2:
                k_total += 1; k_since_last_cp += 1; continue

            # --- 进程B: 模型约束跳过 ---
            if model is not None:
                # 方向约束: |θ - θ_model| < 阈值
                d_theta = abs(theta - model['theta'])
                d_theta = min(d_theta, 2 * math.pi - d_theta)
                if d_theta > theta_constraint:
                    k_total += 1; k_since_last_cp += 1; continue

                # s约束: |s - s_model| / s_model < 15%
                if abs(s - model['s']) / max(model['s'], 0.01) > 0.15:
                    k_total += 1; k_since_last_cp += 1; continue

                # 投影约束: 用模型将W投影到U空间, 检查投影位置是否接近u_idx
                w_proj = _apply_similarity(
                    Wf_init[w_idx:w_idx+1],
                    model['s'], model['theta'], model['tx'], model['ty'])
                proj_dist = math.hypot(w_proj[0, 0] - u_vec[0], w_proj[0, 1] - u_vec[1])
                # 阈值: 2像素 (角秒)
                if proj_dist > 2.0 * s0:
                    k_total += 1; k_since_last_cp += 1; continue

            # --- 验证: SCM(无模型) 或 投影约束(有模型) ---
            if model is None:
                # 进程A: 无模型时用SCM评估
                Wf_bright = Wf_init[:min(100, M_init)]
                Wt_bright = np.column_stack([
                    s * (cos_t * Wf_bright[:, 0] - sin_t * Wf_bright[:, 1]) + tx,
                    s * (sin_t * Wf_bright[:, 0] + cos_t * Wf_bright[:, 1]) + ty
                ])
                scm, coverage, n_confirmed = compute_scm(
                    U_bright, Wt_bright, halfW, halfH, U_bright_sigs)

                if scm < tau_scm or n_confirmed < tau_nmin:
                    k_total += 1; k_since_last_cp += 1; continue
            # else: 进程B已有模型, 投影约束已通过, 直接接受

            # --- 加入控制点池 ---
            cp_key = (u_idx, w_idx)
            if cp_key not in cp_outlier_count:
                cp_pool.append(cp_key)
                cp_outlier_count[cp_key] = 0
                k_since_last_cp = 0

                if verbose and len(cp_pool) <= 30:
                    tag = f'scm={scm:.3f}' if model is None else 'proj_ok'
                    print(f"  [V4 mode={mode}] k={k_total} +CP: #{len(cp_pool)} "
                          f"s={s:.4f} θ={math.degrees(theta):.2f}° {tag}")

            # 模型初始化: >=3对且无模型时用Umeyama初始化
            if model is None and len(cp_pool) >= 3:
                u_is = [p[0] for p in cp_pool]
                w_is = [p[1] for p in cp_pool]
                s_m, th_m, tx_m, ty_m = umeyama_svd(Wf_init[w_is], U_init[u_is])
                model = {'s': s_m, 'theta': th_m, 'tx': tx_m, 'ty': ty_m}
                if verbose:
                    print(f"  [V4 mode={mode}] k={k_total} 模型初始化: "
                          f"s={s_m:.4f} θ={math.degrees(th_m):.2f}°")

            # --- 进程C: RANSAC实时清洗 (每5个新控制点触发一次) ---
            if len(cp_pool) >= 5 and len(cp_pool) % 5 == 0:
                cp_pool, model = self._ransac_realtime(
                    cp_pool, cp_outlier_count, U_init, Wf_init, s0, rng)
                n_ransac_rounds += 1

                # 用全部控制点重新拟合模型 (比RANSAC子抽样更稳定)
                if len(cp_pool) >= 3:
                    u_is = [p[0] for p in cp_pool]
                    w_is = [p[1] for p in cp_pool]
                    s_m, th_m, tx_m, ty_m = umeyama_svd(Wf_init[w_is], U_init[u_is])
                    model = {'s': s_m, 'theta': th_m, 'tx': tx_m, 'ty': ty_m}

                # 模型更新后收紧约束
                if model is not None:
                    n_cp = len(cp_pool)
                    theta_constraint = max(2.0 * math.pi / 180.0,
                                           15.0 * math.pi / 180.0 * max(0.1, 1.0 - n_cp / 30.0))

            # --- 双向不匹配检查 (每15个控制点检查一次) ---
            if len(cp_pool) >= 15 and len(cp_pool) % 15 == 0 and model is not None:
                cp_pool, n_removed = self._bidirectional_check(
                    cp_pool, U_init, Wf_init, model, s0, halfW, halfH)
                n_bidir_removed += n_removed
                if n_removed > 0 and verbose:
                    print(f"  [V4 mode={mode}] k={k_total} 双向检查移除{n_removed}对, 剩余{len(cp_pool)}")

            # --- 收敛判定 ---
            if model is not None and len(cp_pool) >= 8:
                rms = self._compute_rms(cp_pool, U_init, Wf_init, model, s0)
                if verbose and len(cp_pool) % 5 == 0:
                    print(f"  [V4 mode={mode}] k={k_total} RMS={rms:.4f}px CP={len(cp_pool)}")
                # 第一阶段: RMS<2px即收敛 (Umeyama相似变换精度有限, 第二阶段CD+SIP精修)
                if rms < 2.0:
                    cp_pool, n_removed = self._bidirectional_check(
                        cp_pool, U_init, Wf_init, model, s0, halfW, halfH)
                    n_bidir_removed += n_removed
                    if len(cp_pool) >= 3:
                        cp_pool, model = self._ransac_realtime(
                            cp_pool, cp_outlier_count, U_init, Wf_init, s0, rng)
                        # 用全部控制点重新拟合
                        if len(cp_pool) >= 3:
                            u_is = [p[0] for p in cp_pool]
                            w_is = [p[1] for p in cp_pool]
                            s_m, th_m, tx_m, ty_m = umeyama_svd(Wf_init[w_is], U_init[u_is])
                            model = {'s': s_m, 'theta': th_m, 'tx': tx_m, 'ty': ty_m}
                        rms = self._compute_rms(cp_pool, U_init, Wf_init, model, s0)
                        if rms < 2.0:
                            phase1_done = True
                            if verbose:
                                print(f"  [V4 mode={mode}] k={k_total} ★ 第一阶段收敛! "
                                      f"CP={len(cp_pool)} RMS={rms:.4f}px")
                            break

            k_total += 1
            k_since_last_cp += 1

            # 长时间无新控制点 → 停止
            if k_since_last_cp >= 2000 and len(cp_pool) >= 3:
                if verbose:
                    print(f"  [V4 mode={mode}] k={k_total} 停滞{k_since_last_cp}次, CP={len(cp_pool)}")
                break

            if verbose and k_total % 1000 == 0:
                print(f"  [V4 mode={mode}] k={k_total:4d} CP={len(cp_pool)} "
                      f"visited={int(np.sum(visited))}/{N_init*M_init} "
                      f"θ_constr={math.degrees(theta_constraint):.1f}°")

        # ========== 第一阶段结束: 用Umeyama参数直接推导CD ==========
        if not phase1_done or len(cp_pool) < 3 or model is None:
            if verbose:
                print(f"  [V4 mode={mode}] 第一阶段未收敛: CP={len(cp_pool)}")
            return {'success': False, 'error': '第一阶段未收敛', 'mode': mode}

        # 构建匹配对 (像素坐标, 天球坐标)
        pairs_px_sky = self._build_pairs_px_sky(
            cp_pool, U_init, Wf_init, gaia_ra, gaia_dec,
            cra, cdec, w, h, s0, all_idx, img_x_arr, img_y_arr)

        if len(pairs_px_sky) < 3:
            return {'success': False, 'error': '匹配对不足', 'mode': mode}

        # CRVAL = gnomonic投影中心
        crval_ra = cra
        crval_dec = cdec
        crpix1, crpix2 = w / 2.0, h / 2.0

        # CD矩阵: 从Umeyama参数推导 (考虑t偏移)
        # Umeyama: U = s·R(θ)·Wf + t
        # U = [(x-cx)·s0, -(y-cy)·s0]
        # Wf = flip·W, W = [ξ, η] (角秒)
        # [ξ,η] = flip⁻¹·R(-θ)·(U - t) / s
        # [ΔRA·cosδ, ΔDec] = [ξ,η] / 3600
        # [ΔRA·cosδ, ΔDec] = flip⁻¹·R(-θ)·([(x-cx)·s0 - tx, -(y-cy)·s0 - ty]) / (s·3600)
        # = flip⁻¹·R(-θ)·[[s0,0],[0,-s0]]·[x-cx, y-cy] / (s·3600)
        #   - flip⁻¹·R(-θ)·[tx, ty] / (s·3600)
        # 第二项是常数偏移, 可以合并到CRVAL中
        # CD = flip⁻¹·R(-θ)·[[s0,0],[0,-s0]] / (s·3600)
        s_um = model['s']
        theta_um = model['theta']
        cos_t = math.cos(theta_um)
        sin_t = math.sin(theta_um)
        R_neg_theta = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        S_mat = np.array([[s0, 0], [0, -s0]])
        flip_inv = self._get_flip_matrix(mode)

        cd_from_umeyama = flip_inv @ R_neg_theta @ S_mat / (s_um * 3600.0)

        # t偏移 → CRVAL修正
        # CRVAL偏移 = flip⁻¹·R(-θ)·[tx, ty] / (s·3600)
        t_vec = np.array([model['tx'], model['ty']])
        crval_offset = flip_inv @ R_neg_theta @ t_vec / (s_um * 3600.0)
        # crval_offset是[ΔRA·cosδ, ΔDec] (度)
        crval_ra_um = cra - crval_offset[0] / math.cos(cdec * math.pi / 180.0)
        crval_dec_um = cdec - crval_offset[1]

        # 纯LSQ拟合CD (不用Umeyama正则化, 因为cosδ效应导致CD[0,0]≠CD[1,1])
        cd = fit_cd_lsq(pairs_px_sky, crpix1, crpix2, cra, cdec)
        crval_ra = cra
        crval_dec = cdec
        if cd is None:
            cd = cd_from_umeyama
            crval_ra = crval_ra_um
            crval_dec = crval_dec_um

        # SIP拟合
        sip_order, sip_A, sip_B, sip_rms = fit_sip_bic(
            pairs_px_sky, cd, crpix1, crpix2, crval_ra, crval_dec, max_order=2)

        # 第一阶段RMS
        rms_p1 = self._compute_rms_from_cd(pairs_px_sky, cd, crpix1, crpix2, crval_ra, crval_dec)

        if verbose:
            print(f"  [V4 mode={mode}] 第一阶段CD+SIP: RMS={rms_p1:.4f}px, "
                  f"SIP_order={sip_order}, CP={len(pairs_px_sky)}")

        # ========== 第二阶段: 迭代CD精修 ==========
        # 核心思路: 用当前CD做宽松匹配→重拟合CD→再匹配→再拟合
        # 逐步扩大覆盖范围, 直到CD逆投影质量收敛
        N_expand = min(2000, N_all)
        M_expand = min(5000, M_all)

        # 图像侧: 空间均匀采样
        U_expand_idx = self._spatial_uniform_index(img_x_arr, img_y_arr, all_idx, w, h,
                                                    N_expand, grid_n=16)
        U_expand = U_all[U_expand_idx]
        Wf_expand = Wf_all[:M_expand]

        cd_iter = cd.copy()
        sip_A_iter = sip_A.copy()
        sip_B_iter = sip_B.copy()
        sip_order_iter = sip_order
        crval_ra_iter = crval_ra
        crval_dec_iter = crval_dec

        for iter_round in range(4):
            # 逐步收紧匹配阈值: 15" → 10" → 5" → 3"
            threshold = [15.0, 10.0, 5.0, 3.0][iter_round]

            expand_pairs = self._bidirectional_match_expand(
                U_expand, Wf_expand, U_expand_idx,
                cd_iter, sip_A_iter, sip_B_iter, sip_order_iter,
                crpix1, crpix2, crval_ra_iter, crval_dec_iter, s0,
                gaia_ra, gaia_dec, all_idx, img_x_arr, img_y_arr,
                w, h, threshold_arcsec=threshold)

            if verbose:
                print(f"  [V4 mode={mode}] 迭代{iter_round}: 阈值={threshold}\" "
                      f"双向匹配={len(expand_pairs)}对")

            if len(expand_pairs) < 3:
                continue

            # MAD稳健过滤 (比RANSAC更适合大阈值匹配)
            expand_pairs = self._mad_filter_pairs(expand_pairs, cd_iter, crpix1, crpix2,
                                                   crval_ra_iter, crval_dec_iter, s0,
                                                   mad_threshold=3.0)

            # 空间均匀化 (增大max_per_cell让更多对参与CD拟合)
            expand_pairs = self._spatial_uniform_sample(expand_pairs, w, h, grid_n=8, max_per_cell=12)

            # 合并第一阶段+本轮匹配
            all_pairs_iter = pairs_px_sky + [p for p in expand_pairs if p not in pairs_px_sky]
            seen = set()
            unique_pairs = []
            for p in all_pairs_iter:
                key = (round(p[0], 1), round(p[1], 1), round(p[2], 6), round(p[3], 6))
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append(p)
            all_pairs_iter = unique_pairs

            # 重拟合CD
            crval_ra_iter = cra
            crval_dec_iter = cdec
            cd_new = fit_cd_lsq(all_pairs_iter, crpix1, crpix2, crval_ra_iter, crval_dec_iter)
            if cd_new is not None:
                cd_iter = cd_new

            sip_order_new, sip_A_new, sip_B_new, _ = fit_sip_bic(
                all_pairs_iter, cd_iter, crpix1, crpix2, crval_ra_iter, crval_dec_iter)
            sip_order_iter = sip_order_new
            sip_A_iter = sip_A_new
            sip_B_iter = sip_B_new

            if verbose:
                rms_iter = self._compute_rms_from_cd(all_pairs_iter, cd_iter, crpix1, crpix2,
                                                      crval_ra_iter, crval_dec_iter)
                # 匹配对空间覆盖
                mp_arr = np.array(all_pairs_iter)
                print(f"  [V4 mode={mode}] 迭代{iter_round}: CP={len(all_pairs_iter)} "
                      f"RMS={rms_iter:.4f}px "
                      f"x=[{mp_arr[:,0].min():.0f},{mp_arr[:,0].max():.0f}] "
                      f"y=[{mp_arr[:,1].min():.0f},{mp_arr[:,1].max():.0f}]")

        # 最终合并
        expand_pairs_final = self._bidirectional_match_expand(
            U_expand, Wf_expand, U_expand_idx,
            cd_iter, sip_A_iter, sip_B_iter, sip_order_iter,
            crpix1, crpix2, crval_ra_iter, crval_dec_iter, s0,
            gaia_ra, gaia_dec, all_idx, img_x_arr, img_y_arr,
            w, h, threshold_arcsec=3.0)

        if len(expand_pairs_final) >= 5:
            expand_pairs_final = self._ransac_filter_pairs(expand_pairs_final, s0, rng)
        expand_pairs_final = self._spatial_uniform_sample(expand_pairs_final, w, h, grid_n=8, max_per_cell=5)

        all_pairs = pairs_px_sky + [p for p in expand_pairs_final if p not in pairs_px_sky]
        seen = set()
        unique_pairs = []
        for p in all_pairs:
            key = (round(p[0], 1), round(p[1], 1), round(p[2], 6), round(p[3], 6))
            if key not in seen:
                seen.add(key)
                unique_pairs.append(p)
        all_pairs = unique_pairs

        if verbose:
            print(f"  [V4 mode={mode}] 合并控制点: {len(all_pairs)}对")

        # 最终CD+SIP拟合
        # CRVAL = gnomonic投影中心
        crval_ra = cra
        crval_dec = cdec

        cd_final = fit_cd_lsq(all_pairs, crpix1, crpix2, crval_ra, crval_dec)
        if cd_final is None:
            cd_final = cd  # 回退到第一阶段CD

        sip_order_final, sip_A_final, sip_B_final, sip_rms_final = fit_sip_bic(
            all_pairs, cd_final, crpix1, crpix2, crval_ra, crval_dec, max_order=2)

        rms_final = self._compute_rms_from_cd(all_pairs, cd_final, crpix1, crpix2, crval_ra, crval_dec)

        if verbose:
            print(f"  [V4 mode={mode}] 最终CD+SIP: RMS={rms_final:.4f}px, "
                  f"SIP_order={sip_order_final}, CP={len(all_pairs)}")

        # ========== 输出WCS ==========
        # 将SIP系数展平为36元素列表 (6x6)
        sip_A_flat = sip_A_final.flatten().tolist()
        sip_B_flat = sip_B_final.flatten().tolist()

        matched_pairs = [[float(p[0]), float(p[1]), float(p[2]), float(p[3])] for p in all_pairs]

        wcs_data = {
            'CD': cd_final.tolist(),
            'CRVAL': [crval_ra, crval_dec],
            'CRPIX': [crpix1, crpix2],
            'SIP_A': sip_A_flat,
            'SIP_B': sip_B_flat,
            'RMS_PX': float(rms_final),
            'SIP_ORDER': sip_order_final,
            'MATCH_PAIRS': matched_pairs,
        }

        # Umeyama参数 (用于诊断)
        if model is not None:
            s_final = model['s']
            theta_final = model['theta']
            tx_final = model['tx']
            ty_final = model['ty']
        else:
            s_final = 1.0
            theta_final = 0.0
            tx_final = 0.0
            ty_final = 0.0

        return {
            'success': True,
            'mode': mode,
            'n_pairs': len(all_pairs),
            'rms_px': float(rms_final),
            'rms_asec': float(rms_final * s0),
            's_final': s_final,
            'theta_final': theta_final,
            'tx': tx_final,
            'ty': ty_final,
            'k_total': k_total,
            'n_ransac_rounds': n_ransac_rounds,
            'n_bidir_removed': n_bidir_removed,
            'wcs_data': wcs_data,
        }

    def _get_flip_matrix(self, mode):
        """获取翻转矩阵 (自逆: flip⁻¹ = flip)

        mode=0: 无翻转 [[1,0],[0,1]]
        mode=1: flipX  [[-1,0],[0,1]]
        mode=2: flipY  [[1,0],[0,-1]]
        mode=3: flipXY [[-1,0],[0,-1]]
        """
        if mode == 0:
            return np.eye(2)
        elif mode == 1:
            return np.array([[-1, 0], [0, 1]], dtype=np.float64)
        elif mode == 2:
            return np.array([[1, 0], [0, -1]], dtype=np.float64)
        else:  # mode == 3
            return np.array([[-1, 0], [0, -1]], dtype=np.float64)

    # ========================================================================
    # RANSAC实时清洗
    # ========================================================================

    def _ransac_realtime(self, cp_pool, cp_outlier_count, U_init, Wf_init, s0, rng):
        """RANSAC实时清洗: 子抽样→Umeyama→全量验证→标记离群

        关键设计:
          1. 每次RANSAC重置离群计数器(不累积), 避免早期误判累积
          2. 只保留最佳内点集, 不做永久剔除(让后续RANSAC重新判断)
          3. 内点阈值2px, 宽松避免误杀

        Args:
            cp_pool: 控制点池 [(u_idx, w_idx), ...]
            cp_outlier_count: dict {pair_key: outlier_count} (此版本不使用累积计数)
            U_init, Wf_init: 初始池向量
            s0: 像素尺度
            rng: 随机数生成器

        Returns:
            (clean_pool, best_model)
        """
        n = len(cp_pool)
        if n < 5:
            if n >= 3:
                u_is = [p[0] for p in cp_pool]
                w_is = [p[1] for p in cp_pool]
                s, theta, tx, ty = umeyama_svd(Wf_init[w_is], U_init[u_is])
                model = {'s': s, 'theta': theta, 'tx': tx, 'ty': ty}
            else:
                model = None
            return cp_pool, model

        best_inlier_set = []
        best_model = None
        n_trials = min(50, n * 5)

        for trial in range(n_trials):
            sample_idx = rng.choice(n, min(3, n), replace=False)
            u_s = [cp_pool[i][0] for i in sample_idx]
            w_s = [cp_pool[i][1] for i in sample_idx]

            s_t, th_t, tx_t, ty_t = umeyama_svd(Wf_init[w_s], U_init[u_s])

            # 全量验证
            u_all = [p[0] for p in cp_pool]
            w_all = [p[1] for p in cp_pool]
            Wt = _apply_similarity(Wf_init[w_all], s_t, th_t, tx_t, ty_t)
            rdist = np.sqrt(np.sum((U_init[u_all] - Wt)**2, axis=1))

            # 内点阈值: 2像素
            inlier_mask = rdist < 2.0 * s0
            n_inliers = int(np.sum(inlier_mask))

            if n_inliers > len(best_inlier_set):
                best_inlier_set = [i for i in range(n) if inlier_mask[i]]
                best_model = {'s': s_t, 'theta': th_t, 'tx': tx_t, 'ty': ty_t}

        # 用最佳内点集重新拟合
        if len(best_inlier_set) >= 3:
            u_clean = [cp_pool[i][0] for i in best_inlier_set]
            w_clean = [cp_pool[i][1] for i in best_inlier_set]
            s, theta, tx, ty = umeyama_svd(Wf_init[w_clean], U_init[u_clean])
            best_model = {'s': s, 'theta': theta, 'tx': tx, 'ty': ty}
            # 只保留内点
            clean_pool = [cp_pool[i] for i in best_inlier_set]
        else:
            # RANSAC失败, 保留全部
            clean_pool = list(cp_pool)
            if n >= 3:
                u_all = [p[0] for p in cp_pool]
                w_all = [p[1] for p in cp_pool]
                s, theta, tx, ty = umeyama_svd(Wf_init[w_all], U_init[u_all])
                best_model = {'s': s, 'theta': theta, 'tx': tx, 'ty': ty}

        return clean_pool, best_model

    # ========================================================================
    # 双向不匹配检查
    # ========================================================================

    def _bidirectional_check(self, cp_pool, U_init, Wf_init, model, s0, halfW, halfH):
        """双向不匹配检查: 正反向投影最近邻一致性

        对每对控制点:
          1. 用模型将图像星投影到天球 → 在星表中找最近邻
          2. 用模型将星表星投影到图像 → 在检测星中找最近邻
          3. 若正反向最近邻不一致 → 标记为不匹配 → 剔除

        Args:
            cp_pool: 控制点池
            U_init, Wf_init: 初始池向量
            model: 当前模型
            s0: 像素尺度
            halfW, halfH: 图像半宽半高(角秒)

        Returns:
            (clean_pool, n_removed)
        """
        if model is None or len(cp_pool) < 3:
            return cp_pool, 0

        n = len(cp_pool)
        u_is = [p[0] for p in cp_pool]
        w_is = [p[1] for p in cp_pool]

        s, theta, tx, ty = model['s'], model['theta'], model['tx'], model['ty']

        # 正向: Wf → U空间 (用Umeyama变换)
        Wt = _apply_similarity(Wf_init[w_is], s, theta, tx, ty)

        # 反向: U → Wf空间 (Umeyama逆变换)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R_inv = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
        U_src = U_init[u_is]
        U_centered = U_src - np.array([tx, ty])
        Wf_pred = (R_inv @ U_centered.T).T / max(s, 1e-10)

        # 正向检查: Wf_pred在Wf_init中的最近邻是否是w_is
        tree_W = cKDTree(Wf_init)
        dists_fwd, nns_fwd = tree_W.query(Wf_pred, k=1)

        # 反向检查: Wt在U_init中的最近邻是否是u_is
        tree_U = cKDTree(U_init)
        dists_bwd, nns_bwd = tree_U.query(Wt, k=1)

        # 阈值: 3角秒
        threshold = 3.0

        clean_pool = []
        n_removed = 0
        for i in range(n):
            fwd_ok = (nns_fwd[i] == w_is[i]) or (dists_fwd[i] < threshold)
            bwd_ok = (nns_bwd[i] == u_is[i]) or (dists_bwd[i] < threshold)
            if fwd_ok and bwd_ok:
                clean_pool.append(cp_pool[i])
            else:
                n_removed += 1

        return clean_pool, n_removed

    # ========================================================================
    # 扩增池: 邻近双向匹配
    # ========================================================================

    def _bidirectional_match_expand(self, U_expand, Wf_expand, U_expand_idx,
                                    cd, sip_A, sip_B, sip_order,
                                    crpix1, crpix2, crval_ra, crval_dec, s0,
                                    gaia_ra, gaia_dec, all_idx, img_x_arr, img_y_arr,
                                    w, h, threshold_arcsec=3.0):
        """第二阶段: 用CD+SIP模型做邻近双向匹配

        Args:
            U_expand: (N, 2) 扩增池图像侧向量
            U_expand_idx: U_expand在U_all中的索引
            Wf_expand: (M, 2) 扩增池星表侧向量
            cd, sip_A, sip_B, sip_order: 第一阶段CD+SIP模型
            ... (其他参数用于构建匹配对)

        Returns:
            pairs_px_sky: list of (x_px, y_px, ra_deg, dec_deg)
        """
        N = U_expand.shape[0]
        M = Wf_expand.shape[0]
        if N < 2 or M < 2:
            return []

        cos_d = math.cos(crval_dec * math.pi / 180.0)
        cd_inv = np.linalg.inv(cd)

        # 正向: 图像星 → 天球坐标 → 在星表中找最近邻
        # 图像向量 → 像素偏移 → CD逆 → 天球偏移 → RA/Dec
        # U_expand是以图像中心为原点的角秒向量, 转回像素偏移
        px_offset = U_expand / s0  # 角秒 → 像素偏移 (注意Y翻转)
        # 实际上U_expand的构建: ux = (x - cx) * s0, uy = -(y - cy) * s0
        # 所以: x = ux/s0 + cx, y = -uy/s0 + cy
        x_px = U_expand[:, 0] / s0 + crpix1
        y_px = -U_expand[:, 1] / s0 + crpix2

        # 像素 → 天球 (用CD逆)
        dx = x_px - crpix1
        dy = y_px - crpix2

        # SIP修正 (如果有的话)
        if sip_order >= 2:
            sdx = np.zeros_like(dx)
            sdy = np.zeros_like(dy)
            for p in range(sip_order + 1):
                for q in range(sip_order + 1):
                    if p + q < 2 or p + q > sip_order:
                        continue
                    if abs(sip_A[p, q]) < 1e-30 and abs(sip_B[p, q]) < 1e-30:
                        continue
                    xc = np.clip(dx, -5e3, 5e3)
                    yc = np.clip(dy, -5e3, 5e3)
                    term = xc**p * yc**q
                    term = np.where(np.isfinite(term), term, 0.0)
                    sdx += sip_A[p, q] * term
                    sdy += sip_B[p, q] * term
            dx_corr = dx + sdx
            dy_corr = dy + sdy
        else:
            dx_corr = dx
            dy_corr = dy

        # 正向投影: 像素 → 天球 (用CD矩阵, 不是CD逆)
        # [dra*cosδ, ddec] = CD · [dx, dy]
        sky_off = np.column_stack([dx_corr, dy_corr]) @ cd.T  # (N, 2) [dra*cosd, ddec]
        pred_ra = sky_off[:, 0] / cos_d + crval_ra
        pred_dec = sky_off[:, 1] + crval_dec

        # 在星表中找最近邻
        # 用gnomonic投影将Gaia星转到切平面坐标
        xi_g, eta_g, valid_g = gnomonic_forward(gaia_ra, gaia_dec, crval_ra, crval_dec)
        W_sky = np.column_stack([xi_g, eta_g])

        # 将预测位置也转到切平面坐标
        xi_p, eta_p, _ = gnomonic_forward(pred_ra, pred_dec, crval_ra, crval_dec)
        pred_tan = np.column_stack([xi_p, eta_p])

        if W_sky.shape[0] < 2 or pred_tan.shape[0] < 2:
            return []

        tree_W = cKDTree(W_sky)
        dists_fwd, nns_fwd = tree_W.query(pred_tan, k=1)

        # 反向: 星表星 → 像素坐标 → 在图像星中找最近邻
        # 天球→像素: [dx, dy] = CD⁻¹ · [dra*cosδ, ddec]
        sky_off_W = np.column_stack([
            (gaia_ra - crval_ra) * cos_d,
            gaia_dec - crval_dec
        ])
        px_off_W = (sky_off_W @ np.linalg.inv(cd).T)  # (M, 2) 像素偏移

        # SIP修正
        if sip_order >= 2:
            dx_w = px_off_W[:, 0]
            dy_w = px_off_W[:, 1]
            sdx_w = np.zeros_like(dx_w)
            sdy_w = np.zeros_like(dy_w)
            for p in range(sip_order + 1):
                for q in range(sip_order + 1):
                    if p + q < 2 or p + q > sip_order:
                        continue
                    if abs(sip_A[p, q]) < 1e-30 and abs(sip_B[p, q]) < 1e-30:
                        continue
                    xc = np.clip(dx_w, -5e3, 5e3)
                    yc = np.clip(dy_w, -5e3, 5e3)
                    term = xc**p * yc**q
                    term = np.where(np.isfinite(term), term, 0.0)
                    sdx_w += sip_A[p, q] * term
                    sdy_w += sip_B[p, q] * term
            px_off_W[:, 0] -= sdx_w
            px_off_W[:, 1] -= sdy_w

        pred_px_W = px_off_W + np.array([crpix1, crpix2])

        # 在图像星中找最近邻 (用像素坐标)
        img_px = np.column_stack([x_px, y_px])
        tree_U = cKDTree(img_px)
        dists_bwd, nns_bwd = tree_U.query(pred_px_W, k=1)

        # 双向一致性检查
        threshold_px = threshold_arcsec / s0
        pairs = []

        for i in range(N):
            if dists_fwd[i] > threshold_arcsec:
                continue
            j = nns_fwd[i]  # 图像星i的最近星表星j
            # 反向: 星表星j的最近图像星是否是i
            if nns_bwd[j] == i and dists_bwd[j] < threshold_px:
                # 双向一致 — 直接用x_px/y_px (从U_expand向量计算, 已正确)
                pairs.append((float(x_px[i]), float(y_px[i]), float(gaia_ra[j]), float(gaia_dec[j])))

        return pairs

    # ========================================================================
    # RANSAC过滤匹配对
    # ========================================================================

    def _ransac_filter_pairs(self, pairs_px_sky, s0, rng, n_trials=300, threshold_px=2.0):
        """RANSAC过滤匹配对列表

        Args:
            pairs_px_sky: list of (x_px, y_px, ra_deg, dec_deg)
            s0: 像素尺度
            rng: 随机数生成器
            n_trials: RANSAC轮次
            threshold_px: 内点阈值(像素)

        Returns:
            filtered_pairs
        """
        n = len(pairs_px_sky)
        if n < 5:
            return pairs_px_sky

        # 转换为numpy数组
        arr = np.array(pairs_px_sky)
        x_px = arr[:, 0]
        y_px = arr[:, 1]
        ra = arr[:, 2]
        dec = arr[:, 3]

        crpix1 = np.median(x_px)
        crpix2 = np.median(y_px)
        crval_ra = np.median(ra)
        crval_dec = np.median(dec)
        cos_d = math.cos(crval_dec * math.pi / 180.0)

        best_inlier_set = []
        for trial in range(n_trials):
            sample_idx = rng.choice(n, min(3, n), replace=False)
            cd = fit_cd_lsq([pairs_px_sky[i] for i in sample_idx],
                           crpix1, crpix2, crval_ra, crval_dec)
            if cd is None:
                continue

            # 全量验证
            dx = x_px - crpix1
            dy = y_px - crpix2
            pred_ra_cosd = cd[0, 0] * dx + cd[0, 1] * dy
            pred_dec = cd[1, 0] * dx + cd[1, 1] * dy
            actual_ra_cosd = (ra - crval_ra) * cos_d
            actual_dec = dec - crval_dec

            # 残差 (角度 → 像素)
            res_ra = (actual_ra_cosd - pred_ra_cosd) / cos_d * 3600.0 / s0
            res_dec = (actual_dec - pred_dec) * 3600.0 / s0
            rdist = np.sqrt(res_ra**2 + res_dec**2)

            inlier_mask = rdist < threshold_px
            n_inliers = int(np.sum(inlier_mask))

            if n_inliers > len(best_inlier_set):
                best_inlier_set = [i for i in range(n) if inlier_mask[i]]

        if len(best_inlier_set) >= 3:
            return [pairs_px_sky[i] for i in best_inlier_set]
        return pairs_px_sky

    # ========================================================================
    # 空间均匀化采样
    # ========================================================================

    def _mad_filter_pairs(self, pairs_px_sky, cd, crpix1, crpix2,
                          crval_ra, crval_dec, s0, mad_threshold=3.0):
        """MAD稳健过滤: 用CD投影残差的中位数绝对偏差剔除离群点

        比RANSAC更适合大阈值匹配, 因为MAD对outlier比例不敏感

        Args:
            pairs_px_sky: list of (x_px, y_px, ra_deg, dec_deg)
            cd: 2x2 CD矩阵
            mad_threshold: MAD倍数阈值 (默认3.0)

        Returns:
            过滤后的匹配对列表
        """
        if len(pairs_px_sky) < 5:
            return pairs_px_sky

        arr = np.array(pairs_px_sky)
        x_px = arr[:, 0]; y_px = arr[:, 1]
        ra = arr[:, 2]; dec = arr[:, 3]

        cos_d = math.cos(crval_dec * math.pi / 180.0)
        dx = x_px - crpix1
        dy = y_px - crpix2

        # CD正向投影: 像素 → 天球
        pred_ra_cosd = cd[0, 0] * dx + cd[0, 1] * dy
        pred_dec = cd[1, 0] * dx + cd[1, 1] * dy

        actual_ra_cosd = (ra - crval_ra) * cos_d
        actual_dec = dec - crval_dec

        # 残差 (角秒)
        res_ra = (actual_ra_cosd - pred_ra_cosd) * 3600.0
        res_dec = (actual_dec - pred_dec) * 3600.0
        rdist = np.sqrt(res_ra**2 + res_dec**2)

        # MAD稳健阈值
        med = np.median(rdist)
        mad = np.median(np.abs(rdist - med))
        threshold = med + mad_threshold * 1.4826 * max(mad, 0.5)  # 最小MAD=0.5角秒

        inlier_mask = rdist < threshold
        result = [pairs_px_sky[i] for i in range(len(pairs_px_sky)) if inlier_mask[i]]
        return result

    def _spatial_uniform_sample(self, pairs_px_sky, w, h, grid_n=8, max_per_cell=5):
        """空间均匀化: 限制每个网格的最大匹配对数

        避免匹配对集中在星密集区域, 确保CD+SIP拟合不受局部密集簇影响

        Args:
            pairs_px_sky: list of (x_px, y_px, ra_deg, dec_deg)
            w, h: 图像尺寸
            grid_n: 网格数
            max_per_cell: 每个网格最大匹配对数

        Returns:
            均匀化后的匹配对列表
        """
        if len(pairs_px_sky) <= grid_n * grid_n * max_per_cell:
            return pairs_px_sky

        # 分配到网格
        cells = {}
        for p in pairs_px_sky:
            gi = min(int(p[0] / (w / grid_n)), grid_n - 1)
            gj = min(int(p[1] / (h / grid_n)), grid_n - 1)
            key = (gi, gj)
            if key not in cells:
                cells[key] = []
            cells[key].append(p)

        # 每个网格最多取max_per_cell个
        result = []
        for key, pairs in cells.items():
            result.extend(pairs[:max_per_cell])

        return result

    def _spatial_uniform_index(self, img_x, img_y, all_idx, w, h, N_target, grid_n=16):
        """空间均匀采样: 从全部检测星中按网格均匀选取索引

        每个网格按flux(暗→亮顺序, 因为all_idx已按亮→暗排序)取星,
        确保覆盖整个FOV

        Args:
            img_x, img_y: 全部检测星像素坐标
            all_idx: 排序后的索引 (亮→暗)
            w, h: 图像尺寸
            N_target: 目标采样数
            grid_n: 网格数

        Returns:
            U_all中的索引列表 (0 ~ N_all-1)
        """
        N_all = len(all_idx)
        if N_all <= N_target:
            return np.arange(N_all)

        # 分配到网格
        cell_size_x = w / grid_n
        cell_size_y = h / grid_n
        cells = {}  # (gi, gj) → [U_all_idx, ...]
        for i in range(N_all):
            gidx = all_idx[i]
            px_x = img_x[gidx]
            px_y = img_y[gidx]
            gi = min(int(px_x / cell_size_x), grid_n - 1)
            gj = min(int(px_y / cell_size_y), grid_n - 1)
            key = (gi, gj)
            if key not in cells:
                cells[key] = []
            cells[key].append(i)  # 存U_all中的索引

        # 每个网格取quota个
        n_cells = len(cells)
        quota = max(N_target // max(n_cells, 1), 1)
        result = []
        for key, indices in cells.items():
            result.extend(indices[:quota])

        # 如果不够, 从剩余中补充
        if len(result) < N_target:
            used = set(result)
            for i in range(N_all):
                if i not in used:
                    result.append(i)
                    if len(result) >= N_target:
                        break

        return np.array(result[:N_target])

    # ========================================================================
    # 辅助函数
    # ========================================================================

    def _build_pairs_px_sky(self, cp_pool, U_init, Wf_init,
                            gaia_ra, gaia_dec, cra, cdec, w, h, s0,
                            all_idx, img_x_arr, img_y_arr):
        """将控制点池转换为 (x_px, y_px, ra_deg, dec_deg) 列表

        Args:
            cp_pool: [(u_idx, w_idx), ...] 初始池内索引
            U_init, Wf_init: 初始池向量
            gaia_ra, gaia_dec: Gaia原始坐标
            all_idx: 全量图像星索引映射
            img_x_arr, img_y_arr: 原始像素坐标
        """
        pairs = []
        crpix1, crpix2 = w / 2.0, h / 2.0

        for u_idx, w_idx in cp_pool:
            # 图像侧: U向量 → 像素坐标
            # ux = (x - cx) * s0 → x = ux/s0 + cx
            # uy = -(y - cy) * s0 → y = -uy/s0 + cy
            u_vec = U_init[u_idx]
            x_px = u_vec[0] / s0 + crpix1
            y_px = -u_vec[1] / s0 + crpix2

            # 星表侧: 直接用Gaia原始坐标
            ra_deg = float(gaia_ra[w_idx])
            dec_deg = float(gaia_dec[w_idx])

            pairs.append((x_px, y_px, ra_deg, dec_deg))

        return pairs

    def _compute_rms(self, cp_pool, U_init, Wf_init, model, s0):
        """计算控制点池在当前模型下的RMS (像素)"""
        if len(cp_pool) < 2 or model is None:
            return 1e30

        u_is = [p[0] for p in cp_pool]
        w_is = [p[1] for p in cp_pool]

        Wt = _apply_similarity(Wf_init[w_is], model['s'], model['theta'],
                               model['tx'], model['ty'])
        rdist = np.sqrt(np.sum((U_init[u_is] - Wt)**2, axis=1))
        rms_asec = float(np.sqrt(np.mean(rdist**2)))
        return rms_asec / s0

    def _compute_rms_from_cd(self, pairs_px_sky, cd, crpix1, crpix2, crval_ra, crval_dec):
        """从CD矩阵计算匹配对的RMS (像素)"""
        if len(pairs_px_sky) < 2 or cd is None:
            return 1e30

        arr = np.array(pairs_px_sky)
        x_px = arr[:, 0]; y_px = arr[:, 1]
        ra = arr[:, 2]; dec = arr[:, 3]

        cos_d = math.cos(crval_dec * math.pi / 180.0)
        dx = x_px - crpix1
        dy = y_px - crpix2

        pred_ra_cosd = cd[0, 0] * dx + cd[0, 1] * dy
        pred_dec = cd[1, 0] * dx + cd[1, 1] * dy

        actual_ra_cosd = (ra - crval_ra) * cos_d
        actual_dec = dec - crval_dec

        # 残差: 角度差 → 像素
        # CD矩阵单位是度/像素, 所以残差单位也是度
        res_ra = (actual_ra_cosd - pred_ra_cosd) / cos_d * 3600.0
        res_dec = (actual_dec - pred_dec) * 3600.0

        # 需要s0来转像素, 但这里我们直接用角秒RMS
        # 用CD行列式估算s0
        cdet = abs(cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0])
        s_est = math.sqrt(cdet) * 3600.0  # 角秒/像素 (近似)

        rdist_asec = np.sqrt(res_ra**2 + res_dec**2)
        rms_asec = float(np.sqrt(np.mean(rdist_asec**2)))

        if s_est > 1e-10:
            return rms_asec / s_est
        return rms_asec

    def write_wcs_json(self, wcs_data, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(wcs_data, f, indent=2)
        return filepath

    def close(self):
        if self.gaia:
            self.gaia.close()
            self.gaia = None


# ============================================================================
# 测试入口
# ============================================================================

def test_quick():
    """快速单帧测试"""
    from astro_image_io import ImageReader

    reader = ImageReader()
    fits = os.path.join(PROJECT_ROOT, "testdata", "lights",
        "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
    img = reader.read(fits)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    cra0 = img.metadata.wcs.crval1
    cdec0 = img.metadata.wcs.crval2

    print(f"测试帧: M20_T2 Red, {w}x{h}, fl={fl}mm, ps={ps}µm")
    print(f"WCS: RA={cra0:.4f} Dec={cdec0:.4f}")

    from star_detector import StarDetector, SDetParamsPy
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    print(f"检测星点: {len(det.x)}颗 (饱和{np.sum(det.saturated)}颗)")

    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    solver = V40PrototypeSolver(gaia_dir, db_type=1)

    result = solver.solve(
        np.array(det.x, np.float64), np.array(det.y, np.float64),
        np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
        cra0, cdec0, fl, ps, w, h, exptime=300.0,
        K_max=5000, tau_scm=0.40, tau_nmin=3,
    )

    solver.close()

    if result['success']:
        print(f"\n========== 结果 ==========")
        print(f"成功: mode={result['best_mode']}, n_pairs={result['n_pairs']}")
        print(f"RMS: {result['rms_px']:.4f}px ({result['rms_asec']:.4f}\")")
        print(f"s={result['s_final']:.6f}, θ={math.degrees(result['theta_final']):.4f}°")
        print(f"总抽样: {result['k_total']}, RANSAC轮次: {result['n_ransac_rounds']}")
        print(f"双向检查移除: {result['n_bidir_removed']}对")

        wcs = result.get('wcs_data')
        if wcs:
            js_path = os.path.join(PROJECT_ROOT, "overlay_output", "_cp_V4_M20T2.json")
            solver.write_wcs_json(wcs, js_path)
            print(f"WCS JSON: {js_path}  (MATCH_PAIRS: {len(wcs['MATCH_PAIRS'])}对)")
            print(f"  CD: {wcs['CD']}")
            print(f"  CRVAL: {wcs['CRVAL']}")
            print(f"  SIP_ORDER: {wcs['SIP_ORDER']}")
    else:
        print(f"\n失败: {result.get('error', 'unknown')}")

    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  V4.0 两阶段收敛 Platesolve — Numpy原型测试")
    print("=" * 60)
    test_quick()

"""双对采样变换矩阵参数空间分布实验

数学推导:
  变换: W' = s·R(θ)·W + t  (2D相似变换, 4自由度)
  双对采样: (U[i1],Wf[j1]) 和 (U[i2],Wf[j2])
    ΔU = U[i1]-U[i2],  ΔW = Wf[j1]-Wf[j2]
    s = |ΔU|/|ΔW|        (线长比)
    θ = angle(ΔU)-angle(ΔW) (线角度差)
    t = U[i1] - s·R(θ)·Wf[j1]  (平移, 完整4参数!)

约束:
  - s ∈ [0.9, 1.1]  (刚性变换, 拒绝压缩/拉伸)
  - |ΔU| >= min_len (两点足够远, 避免误差放大)
  - 线长代表性选择 (控制计算量)

输出: (θ, tx, ty) 3D散点图, 颜色=n_in_range加权
"""
import os, sys, math, json, argparse
from collections import defaultdict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(_MINGW_BIN)
    except (OSError, FileNotFoundError): pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy, gnomonic_forward
from v4_3.vector_match_v4_3_cpp import V43Solver

TESTDATA = os.path.join(PROJECT_ROOT, "testdata")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "dual_param_space")
FULL_TEST_JSON = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "full_test", "full_test_all.json")

_RADTODEG = 180.0 / math.pi
_DEGTORAD = math.pi / 180.0


def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m/60.0 + sec/3600.0) * 15.0
    return float(s)

def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith("-"): sign = -1.0; s = s[1:]
    elif s.startswith("+"): s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m/60.0 + sec/3600.0)
    return float(s)

def find_fits_path(filename):
    for dirpath, _, filenames in os.walk(TESTDATA):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def build_lines(P, min_len, max_lines=100000):
    """构建线集: 所有点对 (i1<i2), 返回 (i1, i2, length, angle, dx, dy)
    超过 max_lines 时随机采样
    """
    import random
    rng = random.Random(42)
    N = len(P)
    total = N * (N - 1) // 2
    lines = []
    if total <= max_lines:
        for i1 in range(N):
            for i2 in range(i1+1, N):
                dx = P[i1, 0] - P[i2, 0]
                dy = P[i1, 1] - P[i2, 1]
                L = math.sqrt(dx*dx + dy*dy)
                if L >= min_len:
                    ang = math.atan2(dy, dx)
                    lines.append((i1, i2, L, ang, dx, dy))
    else:
        for i1 in range(N):
            for i2 in range(i1+1, N):
                dx = P[i1, 0] - P[i2, 0]
                dy = P[i1, 1] - P[i2, 1]
                L = math.sqrt(dx*dx + dy*dy)
                if L >= min_len:
                    ang = math.atan2(dy, dx)
                    lines.append((i1, i2, L, ang, dx, dy))
        rng.shuffle(lines)
        lines = lines[:max_lines]
    return lines


def select_representative_lengths(u_lines, w_lines, s_min=0.9, s_max=1.1, n_bins=30):
    """选择代表性线长: 统计U/W线长分布, 找重叠密集区, 选代表性长度
    
    策略: 把线长分bin, 统计每个bin内U线数和W线数,
          选 min(u_count, w_count) 最大的几个bin作为代表性长度
    """
    if not u_lines or not w_lines:
        return []
    
    u_lens = np.array([l[2] for l in u_lines])
    w_lens = np.array([l[2] for l in w_lines])
    
    # 用U线长范围分bin
    l_min = max(u_lens.min(), w_lens.min() * s_min)
    l_max = min(u_lens.max(), w_lens.max() * s_max)
    if l_max <= l_min:
        return []
    
    bins = np.linspace(l_min, l_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # 每个bin内U线数
    u_hist, _ = np.histogram(u_lens, bins=bins)
    # 每个bin内, W线长在 [bin*s_min, bin*s_max] 范围的数量
    w_counts = np.zeros(n_bins)
    for i in range(n_bins):
        bc = bin_centers[i]
        w_lo = bc * s_min
        w_hi = bc * s_max
        w_counts[i] = np.sum((w_lens >= w_lo) & (w_lens <= w_hi))
    
    # 得分 = min(u, w) (瓶颈), 选 top-K
    scores = np.minimum(u_hist, w_counts)
    # 选得分>0的bin, 按得分降序
    valid_idx = np.where(scores > 0)[0]
    if len(valid_idx) == 0:
        return []
    
    # 按得分排序, 取前 min(K, len) 个, 但要分散开(避免选相邻bin)
    sorted_idx = valid_idx[np.argsort(-scores[valid_idx])]
    selected = []
    for idx in sorted_idx:
        bc = bin_centers[idx]
        # 检查与已选的距离
        too_close = any(abs(bc - s) < (l_max - l_min) / n_bins * 2 for s in selected)
        if not too_close:
            selected.append(bc)
        if len(selected) >= 8:  # 最多8个代表性长度
            break
    
    return selected


def dual_pair_sample(U, Wf, u_lines, w_lines, rep_lengths, s_min, s_max,
                     s0, match_dist, tol_frac=0.08, max_pairs=50000):
    """双对采样: 在代表性长度附近配对U线和W线, 计算变换
    
    tol_frac: 线长容差比例 (±8%), 用于选U线
    """
    import random
    rng = random.Random(42)
    results = []

    # 按线长排序W线 (加速范围查询)
    w_lens = np.array([l[2] for l in w_lines])
    w_sorted_idx = np.argsort(w_lens)
    w_lens_sorted = w_lens[w_sorted_idx]

    total_pairs_tried = 0

    for rep_L in rep_lengths:
        tol = rep_L * tol_frac

        # U线: 长度在 [rep_L - tol, rep_L + tol]
        u_matches = [l for l in u_lines if abs(l[2] - rep_L) < tol]
        if not u_matches:
            continue

        # 对每条U线, 精确查找W线长度在 [Lu/s_max, Lu/s_min] 范围
        # 这样 s = Lu/Lw 必然在 [s_min, s_max] 内
        max_w_per_u = max(1, min(300, max_pairs // (len(u_matches) * len(rep_lengths) + 1)))

        for ul in u_matches:
            i1u, i2u, Lu, au, dxu, dyu = ul
            # W线长度范围: Lu/s_max <= Lw <= Lu/s_min
            w_lo = Lu / s_max
            w_hi = Lu / s_min
            w_lo_idx = int(np.searchsorted(w_lens_sorted, w_lo, side='left'))
            w_hi_idx = int(np.searchsorted(w_lens_sorted, w_hi, side='right'))
            n_w_avail = w_hi_idx - w_lo_idx
            if n_w_avail == 0:
                continue

            # 随机采样, 避免只取最短/最长的
            if n_w_avail <= max_w_per_u:
                w_pick_indices = range(w_lo_idx, w_hi_idx)
            else:
                w_pick_indices = rng.sample(range(w_lo_idx, w_hi_idx), max_w_per_u)

            for wk in w_pick_indices:
                wl = w_lines[w_sorted_idx[wk]]
                j1w, j2w, Lw, aw, dxw, dyw = wl
                total_pairs_tried += 1

                s = Lu / Lw  # 必然在 [s_min, s_max] 内
                theta = au - aw
                ct, st = math.cos(theta), math.sin(theta)
                tx = U[i1u, 0] - s * (ct * Wf[j1w, 0] - st * Wf[j1w, 1])
                ty = U[i1u, 1] - s * (st * Wf[j1w, 0] + ct * Wf[j1w, 1])

                results.append((s, theta, tx, ty, i1u, i2u, j1w, j2w))

                if len(results) >= max_pairs:
                    return results, total_pairs_tried

    return results, total_pairs_tried


def count_vector_match(U, Wf, s, theta, tol_len=0.1, tol_cos=0.95, min_len=0.0):
    """向量点积验证: 变换Wf→W' = s·R(θ)·Wf (纯线性, 无平移), 用向量方法检查匹配数

    数学基础:
      完整变换: U = s·R(θ)·W + t
      双对采样用差向量 ΔU = s·R(θ)·ΔW 消除 t, 准确估计 (s, θ)
      用 (s, θ) 做纯线性变换 W' = s·R(θ)·W (无平移)
      对于远离中心的星 (|U| >> t): U[i] ≈ W'[j], 可用向量方法验证

    向量方法 (替代cKDTree距离匹配):
      1. 长度比: |U[i]| / |W'[j]| ∈ [1-tol_len, 1+tol_len]  (10%容差)
      2. 方向:  U[i]·W'[j] / (|U[i]|·|W'[j]|) > tol_cos   (点积, 角度差<18°)

    点积加速:
      预计算 W'_norm = W' / |W'| (M×2)
      cos_angles = W'_norm[valid] @ U_norm  (矩阵乘法, 一次算所有候选)

    参数:
      tol_len: 长度比容差 (0.1 = ±10%)
      tol_cos: 方向余弦阈值 (0.95 ≈ 角度差18°)
      min_len: 最小向量长度过滤 (避免短向量受平移t影响)
    """
    ct, st = math.cos(theta), math.sin(theta)
    # 纯线性变换 (无平移)
    Wp = np.empty_like(Wf)
    Wp[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1])
    Wp[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1])

    # 计算长度
    U_len = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    Wp_len = np.sqrt(Wp[:, 0]**2 + Wp[:, 1]**2)

    # 防除零
    Wp_len_safe = np.maximum(Wp_len, 1e-10)

    # 预计算归一化向量 (M×2)
    Wp_norm = Wp / Wp_len_safe[:, np.newaxis]

    n_match = 0
    for i in range(len(U)):
        if U_len[i] < min_len:
            continue

        # 步骤1: 长度比筛选 (向量化)
        len_ratio = U_len[i] / Wp_len_safe
        valid = (len_ratio >= 1.0 - tol_len) & (len_ratio <= 1.0 + tol_len)
        if not np.any(valid):
            continue

        # 步骤2: 方向点积 (矩阵乘法, 一次算所有候选)
        U_norm = U[i] / max(U_len[i], 1e-10)
        cos_angles = Wp_norm[valid] @ U_norm  # (k,) 点积

        if np.max(cos_angles) > tol_cos:
            n_match += 1

    return n_match


def estimate_and_correct_center(U, Wf, s, theta, tol_len=0.1, tol_cos=0.95, min_len=0.0):
    """中心修正: 用匹配对估计平移t, 重新中心化向量组

    如果中心偏移大, t≠0导致向量方法失效.
    用已匹配的(U[i], W'[j])对估计 t = mean(U[i] - W'[j]),
    然后重新中心化: U'[i] = U[i] - t, 消除平移影响.

    返回: (t_vec, n_matched_used)  t_vec=(tx, ty)
    """
    ct, st = math.cos(theta), math.sin(theta)
    Wp = np.empty_like(Wf)
    Wp[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1])
    Wp[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1])

    U_len = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    Wp_len = np.sqrt(Wp[:, 0]**2 + Wp[:, 1]**2)
    Wp_len_safe = np.maximum(Wp_len, 1e-10)
    Wp_norm = Wp / Wp_len_safe[:, np.newaxis]

    t_accum = np.zeros(2)
    n_used = 0
    for i in range(len(U)):
        if U_len[i] < min_len:
            continue
        len_ratio = U_len[i] / Wp_len_safe
        valid = (len_ratio >= 1.0 - tol_len) & (len_ratio <= 1.0 + tol_len)
        if not np.any(valid):
            continue
        U_norm = U[i] / max(U_len[i], 1e-10)
        cos_angles = Wp_norm[valid] @ U_norm
        if np.max(cos_angles) > tol_cos:
            # 找最佳匹配的W'
            best_j_local = np.argmax(cos_angles)
            best_j = np.where(valid)[0][best_j_local]
            t_accum += U[i] - Wp[best_j]
            n_used += 1

    if n_used > 0:
        return (t_accum / n_used, n_used)
    return (np.zeros(2), 0)


def run_experiment(fits_path, gaia_client, star_detector, solver, output_dir):
    base = os.path.basename(fits_path)
    print(f"\n{'='*60}")
    print(f"=== {base} ===")
    print(f"{'='*60}")

    # === 生产版本参数 (vm43_entry.cpp 默认值) ===
    IMG_N_TARGET = 50
    GAIA_DENSITY_RATIO = 1.5
    GAIA_QUERY_RADIUS_FACTOR = 0.55
    M_LIM_STEP = 0.5
    M_LIM_MAX_ITER = 10
    DENSITY_TOLERANCE = 0.1

    # 1. 读取图像
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    pixels = img.to_numpy()

    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl
    print(f"图像: {w}x{h}, s0={s0:.4f}\"/px, fl={fl}mm, ps={ps}um")

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
    obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
    cra0 = _parse_ra_hms(obj_ra_str)
    cdec0 = _parse_dec_dms(obj_dec_str)

    # 2. 星点检测
    pixels_u16 = np.clip(pixels, 0, 65535).astype(np.uint16) if pixels.dtype != np.uint16 else pixels
    det = star_detector.detect_ex(pixels_u16)
    all_x = det.x
    all_y = det.y
    all_flux = det.flux
    all_sat = det.saturated
    n_total = len(all_x)

    # === 生产版本 U 选择 (vm43_select.cpp select_image_stars) ===
    # 饱和优先 + 非饱和按 flux 降序补足到 img_n_target
    sat_indices = [i for i in range(n_total) if all_sat[i]]
    non_sat_indices = [i for i in range(n_total) if not all_sat[i]]
    # 非饱和按 flux 降序
    non_sat_indices.sort(key=lambda i: -all_flux[i])

    if len(sat_indices) >= IMG_N_TARGET:
        u_indices = list(sat_indices)  # 饱和 > 50, 全选饱和
        u_type = "sat_only(>50全选)"
    else:
        n_need = IMG_N_TARGET - len(sat_indices)
        u_indices = list(sat_indices) + non_sat_indices[:n_need]
        u_type = f"sat{len(sat_indices)}+non{min(n_need, len(non_sat_indices))}"

    N = len(u_indices)
    print(f"检测星点: {n_total} (sat={len(sat_indices)}), U组: {N} ({u_type}) [img_n_target={IMG_N_TARGET}]")

    # 3. 构造 U 向量 (角秒, Y向上) - 与C++一致
    cx, cy = w / 2.0, h / 2.0
    U = np.zeros((N, 2), dtype=np.float64)
    for k, idx in enumerate(u_indices):
        U[k, 0] = (all_x[idx] - cx) * s0
        U[k, 1] = -(all_y[idx] - cy) * s0

    # === 生产版本 W 选择 (vm43_select.cpp compute_fov_density + density_match_iterate) ===
    fov_diag_asec = math.sqrt((w * s0)**2 + (h * s0)**2)
    fov_diag_deg = fov_diag_asec / 3600.0
    # 查询半径 = FOV对角线 × gaia_query_radius_factor
    query_radius_deg = fov_diag_deg * GAIA_QUERY_RADIUS_FACTOR
    query_area_sqdeg = math.pi * query_radius_deg**2
    img_area_sqdeg = (w * s0 / 3600.0) * (h * s0 / 3600.0)
    img_area_safe = max(img_area_sqdeg, 1e-10)
    # n_target = max(50, round(gaia_density_ratio × N × query_area/img_area))
    n_target_dbl = GAIA_DENSITY_RATIO * N * (query_area_sqdeg / img_area_safe)
    n_target = max(50, int(round(n_target_dbl)))

    # 初始极限星等: m_cut = 6 + 1.5×log10(f) + 2×log10(1.0)
    m_cut = 6.0 + 1.5 * math.log10(max(fl, 1.0)) + 2.0 * math.log10(1.0)

    print(f"  FOV_diag={fov_diag_deg:.4f}°, query_r={query_radius_deg:.4f}°, "
          f"n_target={n_target}, m_cut_init={m_cut:.3f}")

    # 自适应步长迭代极限星等
    n_lo = n_target * (1.0 - DENSITY_TOLERANCE)
    n_hi = n_target * (1.0 + DENSITY_TOLERANCE)
    m_lim = m_cut
    n_gaia_final = 0
    converged = False
    for i in range(M_LIM_MAX_ITER):
        tmp_ra, tmp_dec, tmp_mag = gaia_client.cone_search(cra0, cdec0, query_radius_deg, m_lim)
        n_gaia = len(tmp_ra)
        step = M_LIM_STEP if i < 4 else M_LIM_STEP * 0.5
        if n_lo <= n_gaia <= n_hi:
            converged = True
            n_gaia_final = n_gaia
            break
        if n_gaia < n_lo:
            m_lim += step  # 星太少, 放宽星等
        else:
            m_lim -= step  # 星太多, 收紧星等

    print(f"  迭代结果: m_lim={m_lim:.3f}, n_gaia={n_gaia_final}, converged={converged}")

    # 用最终 m_lim 查询 Gaia
    cat_ra, cat_dec, cat_mag = gaia_client.cone_search(cra0, cdec0, query_radius_deg, m_lim)
    if len(cat_ra) < 2:
        # 兜底 mag=22
        cat_ra, cat_dec, cat_mag = gaia_client.cone_search(cra0, cdec0, query_radius_deg, 22.0)
    print(f"  Gaia查询: radius={query_radius_deg:.3f}°, m_lim={m_lim:.2f}, 返回{len(cat_ra)}星")

    # Gnomonic投影 + FOV过滤 (与C++一致: |xi|<fov_half_w 且 |eta|<fov_half_h)
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, cra0, cdec0)
    fov_half_w = w / 2.0 * s0
    fov_half_h = h / 2.0 * s0
    fov_mask = valid & (np.abs(xi) < fov_half_w) & (np.abs(eta) < fov_half_h)
    if int(np.sum(fov_mask)) < 2:
        # 放宽 1.5×FOV
        fov_mask = valid & (np.abs(xi) < fov_half_w * 1.5) & (np.abs(eta) < fov_half_h * 1.5)
    xi_f = xi[fov_mask]
    eta_f = eta[fov_mask]
    mag_f = cat_mag[fov_mask]

    # 按星等升序 (最亮优先) 取前 n_target 颗
    sort_idx = np.argsort(mag_f)
    M_max = min(len(sort_idx), n_target)
    W = np.zeros((M_max, 2), dtype=np.float64)
    for k in range(M_max):
        W[k, 0] = xi_f[sort_idx[k]]
        W[k, 1] = eta_f[sort_idx[k]]
    print(f"  FOV内Gaia星: {int(np.sum(fov_mask))}, 取最亮{M_max}颗 [n_target={n_target}]")
    
    # 6. Solver结果 (ground truth)
    log_dir = os.path.join(output_dir, "logs", os.path.splitext(base)[0])
    result = solver.solve(
        image_path=fits_path, ra=cra0, dec=cdec0,
        focal_length_mm=fl, pixel_size_um=ps, log_dir=log_dir,
    )
    solver_mode = result.get("flip_mode", -1)
    solver_theta = result.get("rotation_deg", 0)
    solver_s = result.get("scale", 1.0)
    solver_tx = result.get("tx", 0.0)
    solver_ty = result.get("ty", 0.0)
    solver_rms = result.get("rms_px", 0)
    solver_lnK = result.get("bayes_lnK", 0)
    solver_matched = result.get("matched_count", 0)
    print(f"Solver: mode={solver_mode} θ={solver_theta:.2f}° s={solver_s:.4f} "
          f"tx={solver_tx:.1f} ty={solver_ty:.1f} RMS={solver_rms:.3f}px "
          f"lnK={solver_lnK:.1f} matched={solver_matched}")
    
    # 7. 构建线集
    # min_len: 至少视场对角线的 5%, 避免误差放大
    min_len = fov_diag_asec * 0.05
    print(f"\n构建线集 (min_len={min_len:.1f}\")...")
    u_lines = build_lines(U, min_len)
    print(f"  U线: {len(u_lines)} 条, 长度范围 [{min(l[2] for l in u_lines):.1f}, {max(l[2] for l in u_lines):.1f}]")
    
    # 8. 对4种flip模式分别实验
    mode_names = ["mode0 (无翻转)", "mode1 (X翻转)", "mode2 (Y翻转)", "mode3 (XY翻转)"]
    all_mode_results = []
    s_min, s_max = 0.9, 1.1
    # 向量方法参数: 长度比容差10%, 方向余弦阈值0.95(角度差<18°)
    tol_len = 0.1
    tol_cos = 0.95
    # min_len: 过滤短向量(受平移t影响大), 取FOV对角线10%
    vec_min_len = fov_diag_asec * 0.10

    for mode in range(4):
        fx = (mode == 1 or mode == 3)
        fy = (mode == 2 or mode == 3)
        Wf = np.empty_like(W)
        Wf[:, 0] = -W[:, 0] if fx else W[:, 0]
        Wf[:, 1] = -W[:, 1] if fy else W[:, 1]

        w_lines = build_lines(Wf, min_len)
        print(f"\n  {mode_names[mode]}: W线={len(w_lines)}条")

        if not u_lines or not w_lines:
            all_mode_results.append([])
            continue

        # 选代表性长度
        rep_lens = select_representative_lengths(u_lines, w_lines, s_min, s_max)
        print(f"    代表性长度: {[f'{r:.0f}' for r in rep_lens]}")

        if not rep_lens:
            all_mode_results.append([])
            continue

        # 双对采样 (s约束在线长比, 不需要match_dist)
        transforms, n_tried = dual_pair_sample(
            U, Wf, u_lines, w_lines, rep_lens, s_min, s_max, s0,
            match_dist=0.0,  # 不再使用距离匹配
            max_pairs=50000
        )
        print(f"    采样: 尝试{n_tried}对, 通过s约束{len(transforms)}个")

        # 计算n_vector_match (子集, 控制计算量)
        n_calc = min(len(transforms), 3000)
        if n_calc == 0:
            all_mode_results.append([])
            continue

        if len(transforms) > n_calc:
            rng = np.random.RandomState(42)
            subset_idx = rng.choice(len(transforms), n_calc, replace=False)
            subset = [transforms[k] for k in subset_idx]
        else:
            subset = transforms

        # 向量点积验证 (无平移变换 + 长度比 + 方向点积)
        scored = []
        for (s, theta, tx, ty, i1, i2, j1, j2) in subset:
            nr = count_vector_match(U, Wf, s, theta,
                                     tol_len=tol_len, tol_cos=tol_cos,
                                     min_len=vec_min_len)
            scored.append((s, theta, tx, ty, nr))

        all_mode_results.append(scored)

        if scored:
            nrs = np.array([x[4] for x in scored])
            print(f"    n_vector_match: max={nrs.max()} med={np.median(nrs):.0f} "
                  f"mean={nrs.mean():.1f} >=5: {np.sum(nrs>=5)}个")
    
    # 9. 可视化: 4mode × 1个3D图 = 4子图
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        f"双对采样 (θ, tx, ty) 3D参数空间分布\n"
        f"{base}\n"
        f"Solver: mode={solver_mode} θ={solver_theta:.2f}° s={solver_s:.4f} "
        f"tx={solver_tx:.1f}\" ty={solver_ty:.1f}\" RMS={solver_rms:.3f}px "
        f"lnK={solver_lnK:.1f} matched={solver_matched} | U={N}({u_type}) W={M_max} s0={s0:.3f}\"",
        fontsize=11, fontweight='bold'
    )
    
    for mode in range(4):
        ax = fig.add_subplot(2, 2, mode + 1, projection='3d')
        scored = all_mode_results[mode]
        
        if not scored:
            ax.text2D(0.5, 0.5, "无有效变换", transform=ax.transAxes,
                     ha='center', va='center', fontsize=14)
            ax.set_title(f"{mode_names[mode]} (无数据)", fontsize=10)
            continue
        
        θ_arr = np.array([(x[1] * _RADTODEG + 180) % 360 - 180 for x in scored])
        tx_arr = np.array([x[2] for x in scored])
        ty_arr = np.array([x[3] for x in scored])
        n_arr = np.array([x[4] for x in scored], dtype=float)
        
        # 颜色: n_in_range (log scale)
        colors = np.log10(n_arr + 1)
        # 大小: n_in_range
        sizes = 3 + n_arr * 1.5
        
        # 分离高/低分点
        high = n_arr >= 5
        low = ~high
        
        if np.any(low):
            ax.scatter(θ_arr[low], tx_arr[low], ty_arr[low],
                      c='lightgrey', s=2, alpha=0.2, zorder=1)
        if np.any(high):
            sc = ax.scatter(θ_arr[high], tx_arr[high], ty_arr[high],
                           c=colors[high], cmap='hot', s=sizes[high],
                           alpha=0.7, edgecolors='red', linewidths=0.3, zorder=2)
            plt.colorbar(sc, ax=ax, shrink=0.6, label='log10(n_in_range+1)')
        
        # 标记solver结果
        if mode == solver_mode:
            ax.scatter([solver_theta], [solver_tx], [solver_ty],
                      c='lime', s=200, marker='*', edgecolors='black', linewidths=1.5,
                      zorder=3, label=f'Solver: θ={solver_theta:.1f}° tx={solver_tx:.0f} ty={solver_ty:.0f}')
            ax.legend(fontsize=8, loc='upper left')
        
        ax.set_xlabel('θ (°)', fontsize=9)
        ax.set_ylabel('tx (")', fontsize=9)
        ax.set_zlabel('ty (")', fontsize=9)
        n_max = int(n_arr.max())
        n_high = int(np.sum(high))
        ax.set_title(f"{mode_names[mode]}  (n={len(scored)}, max_inliers={n_max}, ≥5: {n_high}个)",
                    fontsize=10)
        ax.view_init(elev=25, azim=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(output_dir, os.path.splitext(base)[0] + "_dual3d.png")
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  → 3D图: {out_path}")
    
    # 10. 保存CSV + 统计
    csv_path = os.path.join(output_dir, os.path.splitext(base)[0] + "_dual_transforms.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mode,s,theta_deg,tx,ty,n_in_range\n")
        for mode in range(4):
            for (s, theta, tx, ty, nr) in all_mode_results[mode]:
                td = (theta * _RADTODEG + 180) % 360 - 180
                f.write(f"{mode},{s:.6f},{td:.4f},{tx:.2f},{ty:.2f},{nr}\n")
    
    # 统计: 正确变换是否聚类?
    stats = {"filename": base, "solver": {
        "mode": solver_mode, "theta": solver_theta, "s": solver_s,
        "tx": solver_tx, "ty": solver_ty, "rms": solver_rms,
        "lnK": solver_lnK, "matched": solver_matched,
    }, "modes": {}}
    
    for mode in range(4):
        scored = all_mode_results[mode]
        if not scored:
            stats["modes"][mode] = {"n": 0}
            continue
        n_arr = np.array([x[4] for x in scored])
        θ_arr = np.array([(x[1] * _RADTODEG + 180) % 360 - 180 for x in scored])
        tx_arr = np.array([x[2] for x in scored])
        ty_arr = np.array([x[3] for x in scored])
        
        # 高分点(>=5)的聚类统计
        high = n_arr >= 5
        if np.sum(high) >= 3:
            θ_h = θ_arr[high]
            tx_h = tx_arr[high]
            ty_h = ty_arr[high]
            n_h = n_arr[high]
            stats["modes"][mode] = {
                "n_total": len(scored),
                "n_high": int(np.sum(high)),
                "max_inliers": int(n_arr.max()),
                "high_θ_mean": float(np.average(θ_h, weights=n_h)),
                "high_θ_std": float(np.sqrt(np.average((θ_h - np.average(θ_h, weights=n_h))**2, weights=n_h))),
                "high_tx_mean": float(np.average(tx_h, weights=n_h)),
                "high_tx_std": float(np.sqrt(np.average((tx_h - np.average(tx_h, weights=n_h))**2, weights=n_h))),
                "high_ty_mean": float(np.average(ty_h, weights=n_h)),
                "high_ty_std": float(np.sqrt(np.average((ty_h - np.average(ty_h, weights=n_h))**2, weights=n_h))),
            }
        else:
            stats["modes"][mode] = {
                "n_total": len(scored), "n_high": int(np.sum(high)),
                "max_inliers": int(n_arr.max()),
            }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="双对采样3D参数空间实验")
    parser.add_argument("--frames", nargs="*", default=None)
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("双对采样 (θ, tx, ty) 3D参数空间分布实验")
    print("=" * 60)
    
    if args.frames:
        frames = args.frames
    else:
        # 默认: 1成功 + 2失败(Type3 + Type1)
        frames = [
            "Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",  # Type3
            "NGC4945_FD_T3_flying_dutchman-20250206@043838-600S-Lum.fts",             # Type1
        ]
        # 找一个成功帧
        if os.path.exists(FULL_TEST_JSON):
            with open(FULL_TEST_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            for r in all_results:
                if (r.get("status") == "success" and r.get("bayes_lnK", 0) > 100
                        and r.get("rms_px", 99) < 1.0):
                    fn = r.get("filename", "")
                    if fn and find_fits_path(fn):
                        frames.insert(0, fn)
                        break
        print(f"实验帧: {frames}")
    
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(gaia_client=gaia_client, star_detector=star_detector)
    
    all_stats = []
    for fn in frames:
        fits_path = find_fits_path(fn)
        if not fits_path:
            print(f"跳过(未找到): {fn}")
            continue
        try:
            stats = run_experiment(fits_path, gaia_client, star_detector, solver, OUTPUT_DIR)
            all_stats.append(stats)
        except Exception as e:
            import traceback
            print(f"错误: {e}")
            traceback.print_exc()
    
    solver.close()
    gaia_client.close()
    star_detector.close()
    
    # 实验报告
    report_path = os.path.join(OUTPUT_DIR, "实验报告.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# 双对采样 3D 参数空间分布实验报告\n\n")
        f.write("## 实验目的\n\n")
        f.write("验证双对采样(两个控制点)是否能产生完整4参数变换(s,θ,tx,ty),\n")
        f.write("并在(θ,tx,ty)三维空间中形成聚类, 区分正确变换与噪声.\n\n")
        f.write("## 数学推导\n\n")
        f.write("```\n")
        f.write("变换: W' = s·R(θ)·W + t\n")
        f.write("双对: (U[i1],Wf[j1]), (U[i2],Wf[j2])\n")
        f.write("  ΔU = U[i1]-U[i2], ΔW = Wf[j1]-Wf[j2]\n")
        f.write("  s = |ΔU|/|ΔW|           (线长比)\n")
        f.write("  θ = angle(ΔU)-angle(ΔW)  (线角度差)\n")
        f.write("  t = U[i1]-s·R(θ)·Wf[j1]  (平移, 完整4参数)\n")
        f.write("```\n\n")
        f.write("## 约束条件\n\n")
        f.write("- s ∈ [0.9, 1.1] (刚性变换)\n")
        f.write("- 线长 ≥ 视场对角线5% (避免误差放大)\n")
        f.write("- 代表性长度选择 (控制计算量)\n")
        f.write("- n_in_range加权 (变换后近邻U的W星数)\n\n")
        f.write("## 实验结果\n\n")
        
        for stats in all_stats:
            f.write(f"### {stats['filename']}\n\n")
            sv = stats["solver"]
            f.write(f"**Solver结果**: mode={sv['mode']} θ={sv['theta']:.2f}° "
                    f"s={sv['s']:.4f} tx={sv['tx']:.1f}\" ty={sv['ty']:.1f}\" "
                    f"RMS={sv['rms']:.3f}px lnK={sv['lnK']:.1f} matched={sv['matched']}\n\n")
            
            f.write("| Mode | 总变换数 | 高分(≥5)数 | max_inliers | θ均值 | θ标准差 | tx均值 | tx标准差 | ty均值 | ty标准差 |\n")
            f.write("|------|---------|-----------|------------|------|--------|------|--------|------|--------|\n")
            for mode in range(4):
                m = stats["modes"].get(mode, {})
                if m.get("n_total", 0) == 0:
                    f.write(f"| {mode} | 0 | - | - | - | - | - | - | - | - |\n")
                elif "high_θ_mean" in m:
                    f.write(f"| {mode} | {m['n_total']} | {m['n_high']} | {m['max_inliers']} | "
                            f"{m['high_θ_mean']:.2f}° | {m['high_θ_std']:.2f}° | "
                            f"{m['high_tx_mean']:.1f}\" | {m['high_tx_std']:.1f}\" | "
                            f"{m['high_ty_mean']:.1f}\" | {m['high_ty_std']:.1f}\" |\n")
                else:
                    f.write(f"| {mode} | {m['n_total']} | {m['n_high']} | {m['max_inliers']} | "
                            f"- | - | - | - | - | - |\n")
            f.write("\n")
            
            # 聚类判断
            sv_mode = sv["mode"]
            m = stats["modes"].get(sv_mode, {})
            if "high_θ_mean" in m and m["n_high"] >= 3:
                θ_diff = abs(m["high_θ_mean"] - sv["theta"])
                tx_diff = abs(m["high_tx_mean"] - sv["tx"])
                ty_diff = abs(m["high_ty_mean"] - sv["ty"])
                f.write(f"**聚类分析(solver mode={sv_mode})**:\n")
                f.write(f"- 高分点θ均值与solver差: {θ_diff:.2f}° (std={m['high_θ_std']:.2f}°)\n")
                f.write(f"- 高分点tx均值与solver差: {tx_diff:.1f}\" (std={m['high_tx_std']:.1f}\")\n")
                f.write(f"- 高分点ty均值与solver差: {ty_diff:.1f}\" (std={m['high_ty_std']:.1f}\")\n")
                if θ_diff < 5 and m["high_θ_std"] < 5:
                    f.write(f"- **θ聚类良好** ✓\n")
                else:
                    f.write(f"- **θ聚类分散** ✗\n")
                f.write("\n")
        
        f.write("## 结论\n\n")
        f.write("(根据实验数据填写)\n")
    
    print(f"\n实验报告: {report_path}")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

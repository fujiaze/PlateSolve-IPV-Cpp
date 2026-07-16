"""相对向量法实际数据测试 (优化版)

优化点:
  1. 预计算 W 的距离矩阵 D[i,j] (N_w², 一次性), 第三星验证查表 O(1)
  2. k-vector 查询用 np.searchsorted 向量化
  3. 第三星验证用 numpy mask 批量判断
  4. Gaia 星对预构建 (距离,角度,a,b) 数组

星点选取 (与生产版本 V4.3 一致):
  - U: img_n_target=50 (饱和优先 + 非饱和按 flux 补足)
  - W: gaia_density_ratio=1.5, 自适应 n_target, FOV 内取最亮 n_target 颗

测试帧: Type3 失败帧 + 成功帧对比
"""
import os, sys, math, time, json, random
import numpy as np
import functools
print = functools.partial(print, flush=True)  # 强制无缓冲
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体支持
for _font in ["Microsoft YaHei", "SimHei", "SimSun", "KaiTi"]:
    try:
        matplotlib.font_manager.findfont(_font, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "relvec_real")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try: os.add_dll_directory(_MINGW_BIN)
    except: pass

sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy, gnomonic_forward
from v4_3.vector_match_v4_3_cpp import V43Solver

TESTDATA = os.path.join(PROJECT_ROOT, "testdata")
_RADTODEG = 180.0 / math.pi
_DEGTORAD = math.pi / 180.0


# ============================================================================
# 选星逻辑 (与 V4.3 生产版本一致)
# ============================================================================
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


def select_stars_production(fits_path, gaia_client, star_detector, solver):
    """与生产版本 V4.3 完全一致的选星逻辑
    
    返回:
      U: (N_u, 2) 图像侧星点 (角秒, Y向上)
      W: (N_w, 2) Gaia侧星点 (角秒, Y向上, 未翻转)
      s0: 像素尺度 (角秒/像素)
      cra0, cdec0: 中心指向 (度)
      img_w, img_h: 图像尺寸
      solver_result: solver求解结果 (ground truth)
    """
    base = os.path.basename(fits_path)
    print(f"\n{'='*60}")
    print(f"=== {base} ===")
    print(f"{'='*60}")
    
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
    all_sat = det.saturated
    n_total = len(all_x)
    
    # 3. U组选择 (img_n_target=50, 饱和优先)
    IMG_N_TARGET = 50
    sat_indices = [i for i in range(n_total) if all_sat[i]]
    normal_indices = [i for i in range(n_total) if not all_sat[i]]
    if len(sat_indices) >= IMG_N_TARGET:
        u_indices = list(sat_indices[:])
        u_type = "sat_only"
    else:
        n_need = IMG_N_TARGET - len(sat_indices)
        u_indices = list(sat_indices)
        u_indices.extend(normal_indices[:n_need])
        u_type = "sat+normal"
    
    N_u = len(u_indices)
    print(f"检测星点: {n_total} (sat={len(sat_indices)}), U组: {N_u} ({u_type})")
    
    # 4. 构造 U (角秒, Y向上, 与 V4.3 一致)
    cx, cy = w / 2.0, h / 2.0
    U = np.zeros((N_u, 2), dtype=np.float64)
    for k, idx in enumerate(u_indices):
        U[k, 0] = (all_x[idx] - cx) * s0
        U[k, 1] = -(all_y[idx] - cy) * s0  # Y翻转
    
    # 5. Gaia 查询 (与 V4.3 一致: 1.5x密度, 0.55x查询半径)
    # 先用 solver 求解, 获取 m_lim_final 和 n_gaia_final
    log_dir = os.path.join(OUTPUT_DIR, "solver_logs", os.path.splitext(base)[0])
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
    solver_snr = result.get("theta_snr", 0)
    solver_theta_peak = result.get("theta_peak_deg", 0)
    n_gaia_final = result.get("n_gaia_final", 0)
    m_lim_final = result.get("m_lim_final", 18.0)
    print(f"Solver: mode={solver_mode} θ={solver_theta:.2f}° s={solver_s:.4f} "
          f"tx={solver_tx:.1f} ty={solver_ty:.1f} RMS={solver_rms:.3f}px "
          f"lnK={solver_lnK:.1f} matched={solver_matched} SNR={solver_snr:.1f}")
    print(f"  m_lim={m_lim_final:.2f}, n_gaia={n_gaia_final}")
    
    # 6. 构造 W (与 V4.3 一致: gaia_query_radius_factor=0.55)
    fov_diag_asec = math.sqrt((w * s0)**2 + (h * s0)**2)
    fov_diag_deg = fov_diag_asec / 3600.0
    query_radius_deg = fov_diag_deg * 0.55  # 与生产版本一致
    cat_ra, cat_dec, cat_mag = gaia_client.cone_search(cra0, cdec0, query_radius_deg, m_lim_final)
    print(f"Gaia查询: radius={query_radius_deg:.3f}°, mag_lim={m_lim_final:.2f}, 返回{len(cat_ra)}星")
    
    # Gnomonic 投影 + FOV 过滤
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, cra0, cdec0)
    fov_half_w = w / 2.0 * s0
    fov_half_h = h / 2.0 * s0
    fov_mask = valid & (np.abs(xi) < fov_half_w) & (np.abs(eta) < fov_half_h)
    xi_f = xi[fov_mask]
    eta_f = eta[fov_mask]
    mag_f = cat_mag[fov_mask]
    
    # 取最亮 n_gaia_final 颗 (与 solver 一致)
    sort_idx = np.argsort(mag_f)
    M = min(len(sort_idx), n_gaia_final) if n_gaia_final > 0 else min(len(sort_idx), 100)
    W = np.zeros((M, 2), dtype=np.float64)
    for k in range(M):
        W[k, 0] = xi_f[sort_idx[k]]
        W[k, 1] = eta_f[sort_idx[k]]
    print(f"FOV内Gaia星: {int(np.sum(fov_mask))}, 取最亮{M}颗 (W)")
    
    return U, W, s0, cra0, cdec0, w, h, result


# ============================================================================
# 优化版相对向量法
# ============================================================================
class FastRelativeVectorMatcher:
    """优化版相对向量法匹配器
    
    优化:
      1. 预计算 W 距离矩阵 (N_w², 一次性)
      2. 预构建 Gaia 星对数组 (距离/角度/索引)
      3. k-vector 查询用 np.searchsorted
      4. 第三星验证用距离矩阵查表 + numpy mask
    """
    def __init__(self, W, s_min=0.9, s_max=1.1, min_len_frac=0.05, max_len_frac=0.8):
        N_w = len(W)
        self.W = W
        self.N_w = N_w
        self.s_min = s_min
        self.s_max = s_max
        
        # 预计算 W 距离矩阵 (N_w × N_w)
        # D[i,j] = |W[i] - W[j]|
        diff = W[:, None, :] - W[None, :, :]
        self.D_W = np.sqrt(np.sum(diff * diff, axis=2))
        
        # 预构建 Gaia 星对 (i<j)
        d_max = np.max(self.D_W) * max_len_frac
        d_min = np.max(self.D_W) * min_len_frac
        iu, ju = np.triu_indices(N_w, k=1)
        d_ij = self.D_W[iu, ju]
        mask = (d_ij >= d_min) & (d_ij <= d_max)
        self.pair_a = iu[mask]
        self.pair_b = ju[mask]
        self.pair_dist = d_ij[mask]
        # 角度: angle(W[a]-W[b])
        dx = W[self.pair_a, 0] - W[self.pair_b, 0]
        dy = W[self.pair_a, 1] - W[self.pair_b, 1]
        self.pair_angle = np.arctan2(dy, dx)
        
        # 按距离排序 (k-vector)
        sort_idx = np.argsort(self.pair_dist)
        self.pair_dist = self.pair_dist[sort_idx]
        self.pair_angle = self.pair_angle[sort_idx]
        self.pair_a = self.pair_a[sort_idx]
        self.pair_b = self.pair_b[sort_idx]
        
        print(f"    FastMatcher: N_w={N_w}, Gaia星对={len(self.pair_dist)}, "
              f"距离范围[{self.pair_dist.min():.0f}, {self.pair_dist.max():.0f}]\"")
    
    def match(self, U, n_samples=3000, dist_tol=3.0,
              use_third_star=True, third_star_tol=3.0, seed=42, max_cand=500):
        """运行相对向量法匹配

        返回: (votes_array, n_total_cand, n_passed, elapsed_sec)
        max_cand: 单次采样候选上限 (超过则随机采样), 控制计算量
        """
        t0 = time.time()
        rng = np.random.RandomState(seed)
        N_u = len(U)

        # 预计算 U 距离矩阵
        diff_u = U[:, None, :] - U[None, :, :]
        D_U = np.sqrt(np.sum(diff_u * diff_u, axis=2))

        votes = []
        n_total_cand = 0
        n_passed = 0
        progress_t0 = time.time()

        for s_idx in range(n_samples):
            # Step 1: 图像星对采样
            i = rng.randint(N_u)
            j = rng.randint(N_u)
            if i == j: continue
            d_img = D_U[i, j]
            if d_img < 1.0: continue
            θ_img = math.atan2(U[i, 1]-U[j, 1], U[i, 0]-U[j, 0])

            # Step 2: k-vector 距离查询 (向量化)
            d_lo = d_img / self.s_max
            d_hi = d_img / self.s_min
            i_lo = int(np.searchsorted(self.pair_dist, d_lo, side='left'))
            i_hi = int(np.searchsorted(self.pair_dist, d_hi, side='right'))
            n_cand = i_hi - i_lo
            n_total_cand += n_cand
            if n_cand == 0: continue

            # 候选过多时随机采样 (控制 (n_cand, N_w) 矩阵大小)
            if n_cand > max_cand:
                pick = rng.choice(n_cand, max_cand, replace=False)
                cand_a = self.pair_a[i_lo:i_hi][pick]
                cand_b = self.pair_b[i_lo:i_hi][pick]
                cand_dist = self.pair_dist[i_lo:i_hi][pick]
                cand_angle = self.pair_angle[i_lo:i_hi][pick]
                n_cand = max_cand
            else:
                cand_a = self.pair_a[i_lo:i_hi]
                cand_b = self.pair_b[i_lo:i_hi]
                cand_dist = self.pair_dist[i_lo:i_hi]
                cand_angle = self.pair_angle[i_lo:i_hi]

            if not use_third_star:
                # 无第三星验证, 全部投票
                θ_rot = (cand_angle - θ_img) * _RADTODEG
                θ_rot = np.mod(θ_rot + 180, 360) - 180
                votes.extend(θ_rot.tolist())
                n_passed += n_cand
            else:
                # Step 3: 第三星交叉验证 (完全向量化)
                k_idx = rng.randint(N_u)
                if k_idx == i or k_idx == j: continue
                d_ik_img = D_U[i, k_idx]
                d_jk_img = D_U[j, k_idx]

                s_est = d_img / cand_dist  # (n_cand,)
                d_ik_exp = d_ik_img / s_est
                d_jk_exp = d_jk_img / s_est
                s_rel_err = third_star_tol / np.maximum(cand_dist, 1.0)
                d_ik_tol = d_ik_exp * s_rel_err + third_star_tol
                d_jk_tol = d_jk_exp * s_rel_err + third_star_tol

                # 批量查距离矩阵: (n_cand, N_w)
                d_ac_all = self.D_W[cand_a, :]
                d_bc_all = self.D_W[cand_b, :]

                # 批量 mask: (n_cand, N_w)
                mask_ik = np.abs(d_ac_all - d_ik_exp[:, None]) < d_ik_tol[:, None]
                mask_jk = np.abs(d_bc_all - d_jk_exp[:, None]) < d_jk_tol[:, None]
                mask_both = mask_ik & mask_jk
                # 排除 a, b 自己
                cand_arange = np.arange(n_cand)
                mask_both[cand_arange, cand_a] = False
                mask_both[cand_arange, cand_b] = False
                has_c = np.any(mask_both, axis=1)

                n_pass = int(np.sum(has_c))
                if n_pass > 0:
                    θ_rot_pass = (cand_angle[has_c] - θ_img) * _RADTODEG
                    θ_rot_pass = np.mod(θ_rot_pass + 180, 360) - 180
                    votes.extend(θ_rot_pass.tolist())
                    n_passed += n_pass

            # 进度打印 (每 1000 samples)
            if (s_idx + 1) % 1000 == 0:
                elapsed_so_far = time.time() - progress_t0
                print(f"    进度 {s_idx+1}/{n_samples}: 候选累计={n_total_cand}, "
                      f"通过={n_passed}, 用时={elapsed_so_far:.1f}s")

        elapsed = time.time() - t0
        return np.array(votes), n_total_cand, n_passed, elapsed


# ============================================================================
# 单θ采样 (对比基准)
# ============================================================================
def single_theta_sampling(U, Wf, s_min=0.9, s_max=1.1, n_samples=5000, seed=42):
    t0 = time.time()
    rng = np.random.RandomState(seed)
    N_u, N_w = len(U), len(Wf)
    u_norm = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    u_ang = np.arctan2(U[:, 1], U[:, 0])
    w_norm = np.sqrt(Wf[:, 0]**2 + Wf[:, 1]**2)
    w_ang = np.arctan2(Wf[:, 1], Wf[:, 0])
    
    votes = []
    for _ in range(n_samples):
        i = rng.randint(N_u)
        j = rng.randint(N_w)
        if w_norm[j] < 1e-6: continue
        s = u_norm[i] / w_norm[j]
        if s_min <= s <= s_max:
            theta = (w_ang[j] - u_ang[i]) * _RADTODEG
            theta = (theta + 180) % 360 - 180
            votes.append(theta)
    return np.array(votes), time.time() - t0


# ============================================================================
# SNR 计算
# ============================================================================
def compute_snr(votes, theta_true_deg, bin_width=2.0):
    if len(votes) == 0:
        return 0.0, 0.0, 0, 0.0
    bins = np.arange(-180, 180 + bin_width, bin_width)
    hist, _ = np.histogram(votes, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    peak_idx = np.argmax(hist)
    peak_height = hist[peak_idx]
    theta_peak = bin_centers[peak_idx]
    mask = np.abs(bin_centers - theta_peak) > 10.0
    background = np.median(hist[mask]) if np.sum(mask) > 0 else np.mean(hist)
    background = max(background, 1.0)
    return peak_height / background, theta_peak, peak_height, background


# ============================================================================
# 可视化
# ============================================================================
def plot_result(votes_single, votes_relvec, theta_true_deg, title, out_path,
                snr_single=None, snr_relvec=None, time_single=None, time_relvec=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bin_width = 2.0
    bins = np.arange(-180, 180 + bin_width, bin_width)
    
    for ax, votes, label, snr_info, t_info in [
        (axes[0], votes_single, "单θ采样 (Phase A, t=0假设)", snr_single, time_single),
        (axes[1], votes_relvec, "相对向量法 (无t假设)", snr_relvec, time_relvec)
    ]:
        if len(votes) > 0:
            ax.hist(votes, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(theta_true_deg, color='red', linestyle='--', linewidth=2, label=f'θ_true={theta_true_deg:.1f}°')
        ax.set_xlabel('θ (°)', fontsize=11)
        ax.set_ylabel('投票数', fontsize=11)
        info = ""
        if snr_info: info += snr_info
        if t_info: info += f"\n{t_info}"
        ax.set_title(f"{label}\n{info}", fontsize=10)
        ax.legend(fontsize=10)
        ax.set_xlim(-180, 180)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → 图: {out_path}")


# ============================================================================
# 主测试函数
# ============================================================================
def test_frame(fits_path, gaia_client, star_detector, solver, output_dir):
    # 选星 (生产版本逻辑)
    U, W, s0, cra0, cdec0, img_w, img_h, solver_result = select_stars_production(
        fits_path, gaia_client, star_detector, solver
    )
    
    base = os.path.basename(fits_path)
    solver_mode = solver_result.get("flip_mode", -1)
    solver_theta = solver_result.get("rotation_deg", 0)
    solver_snr = solver_result.get("theta_snr", 0)
    solver_matched = solver_result.get("matched_count", 0)
    solver_lnK = solver_result.get("bayes_lnK", 0)
    solver_rms = solver_result.get("rms_px", 0)
    
    print(f"\n--- 相对向量法测试 ---")
    print(f"U={len(U)}星, W={len(W)}星, s0={s0:.4f}\"/px")

    # 对4种flip模式测试
    mode_names = ["mode0", "mode1(X翻转)", "mode2(Y翻转)", "mode3(XY翻转)"]
    all_results = {}
    solver_mode_votes = None  # 保存solver模式的votes用于画图

    for mode in range(4):
        fx = (mode == 1 or mode == 3)
        fy = (mode == 2 or mode == 3)
        Wf = np.empty_like(W)
        Wf[:, 0] = -W[:, 0] if fx else W[:, 0]
        Wf[:, 1] = -W[:, 1] if fy else W[:, 1]

        # 单θ采样
        votes_single, t_single = single_theta_sampling(U, Wf, n_samples=3000)
        snr_single, θ_peak_single, _, _ = compute_snr(votes_single, solver_theta if mode == solver_mode else 0)

        # 相对向量法 (优化版)
        matcher = FastRelativeVectorMatcher(Wf, s_min=0.9, s_max=1.1)
        votes_relvec, n_cand, n_pass, t_relvec = matcher.match(
            U, n_samples=3000, dist_tol=3.0, use_third_star=True, third_star_tol=3.0
        )
        snr_relvec, θ_peak_relvec, _, _ = compute_snr(votes_relvec, solver_theta if mode == solver_mode else 0)

        is_solver_mode = (mode == solver_mode)
        all_results[mode] = {
            'votes_single': len(votes_single), 'snr_single': snr_single,
            'θ_peak_single': θ_peak_single,
            'votes_relvec': len(votes_relvec), 'snr_relvec': snr_relvec,
            'θ_peak_relvec': θ_peak_relvec,
            'n_cand': n_cand, 'n_pass': n_pass,
            't_single': t_single, 't_relvec': t_relvec,
            'is_solver_mode': is_solver_mode,
        }

        marker = " ← solver模式" if is_solver_mode else ""
        print(f"  {mode_names[mode]}: 单θ SNR={snr_single:.2f} (θ={θ_peak_single:.1f}°, {t_single:.2f}s) | "
              f"相对向量 SNR={snr_relvec:.2f} (θ={θ_peak_relvec:.1f}°, {t_relvec:.2f}s, "
              f"候选{n_cand}→通过{n_pass}){marker}")

        # solver模式保存votes用于画图
        if is_solver_mode:
            solver_mode_votes = (votes_single, votes_relvec, snr_single, snr_relvec, t_single, t_relvec)

    # 对solver模式画图 (复用已计算的votes)
    if solver_mode >= 0 and solver_mode_votes is not None:
        votes_single, votes_relvec, snr_single, snr_relvec, t_single, t_relvec = solver_mode_votes
        plot_path = os.path.join(output_dir, f"{os.path.splitext(base)[0]}_mode{solver_mode}.png")
        plot_result(votes_single, votes_relvec, solver_theta,
                    f"{base}\nmode={solver_mode} (solver), U={len(U)}, W={len(W)}, matched={solver_matched}",
                    plot_path,
                    snr_single=f"SNR={snr_single:.2f}",
                    snr_relvec=f"SNR={snr_relvec:.2f}",
                    time_single=f"{t_single:.2f}s",
                    time_relvec=f"{t_relvec:.2f}s")

    return all_results, solver_result


def main():
    print("=" * 70)
    print("相对向量法实际数据测试 (优化版)")
    print("=" * 70)
    
    # 初始化客户端
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(gaia_client=gaia_client, star_detector=star_detector)
    
    # 测试帧: Type3失败帧 + 成功帧
    test_frames = [
        # Type3 失败帧 (t≠0, Phase A+B 不稳)
        "Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",
        "NGC7293_T2_HO_flying_dutchman-20250706@081055-1200S-H-alpha.fts",  # Bug4修复的帧
        "LDN43_LRGBH_flying_dutchman-20250520@022502-600S-Lum.fts",  # Bug4修复的帧
        # 成功帧 (对比)
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
    ]
    
    all_results = {}
    for filename in test_frames:
        fits_path = find_fits_path(filename)
        if not fits_path:
            print(f"\n!! 未找到: {filename}")
            continue
        try:
            mode_results, solver_result = test_frame(fits_path, gaia_client, star_detector, solver, OUTPUT_DIR)
            all_results[filename] = {
                'mode_results': mode_results,
                'solver': {
                    'mode': solver_result.get('flip_mode', -1),
                    'theta': solver_result.get('rotation_deg', 0),
                    'snr': solver_result.get('theta_snr', 0),
                    'matched': solver_result.get('matched_count', 0),
                    'lnK': solver_result.get('bayes_lnK', 0),
                    'rms': solver_result.get('rms_px', 0),
                }
            }
        except Exception as e:
            import traceback
            print(f"\n!! 测试失败 {filename}: {e}")
            print(traceback.format_exc())
    
    # 总结
    print("\n\n" + "=" * 90)
    print("总结")
    print("=" * 90)
    print(f"{'帧':<55} {'solver模式SNR':<12} {'相对向量SNR':<12} {'单θSNR':<10} {'替代?'}")
    print("-" * 90)
    for fn, r in all_results.items():
        solver_mode = r['solver']['mode']
        if solver_mode < 0: continue
        mr = r['mode_results'][solver_mode]
        solver_snr = r['solver']['snr']
        relvec_snr = mr['snr_relvec']
        single_snr = mr['snr_single']
        replace = "✓" if relvec_snr > 5 and abs(mr['θ_peak_relvec'] - r['solver']['theta']) < 5 else "✗"
        print(f"{os.path.basename(fn)[:54]:<55} {solver_snr:<12.2f} {relvec_snr:<12.2f} {single_snr:<10.2f} {replace}")
    print("=" * 90)
    
    # 保存结果
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n脚本异常: {e}")
        print(traceback.format_exc())

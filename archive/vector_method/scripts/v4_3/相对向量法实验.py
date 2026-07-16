"""相对向量法实验 (DMPDV核心机制验证)

数学推导:
  绝对向量: u_i = U[i] - center_image, w_a = W[a] - center_sky
            受平移 t 影响, t≠0 时 θ_true 分散

  相对向量: Δu_ij = U[i] - U[j],  Δw_ab = W[a] - W[b]
            与原点无关, t 自动消去!

  |Δu_ij| = s · |Δw_ab|     (距离比 = s, 已知 s≈s0/gnomonic_scale)
  angle(Δu_ij) - angle(Δw_ab) = θ_true  (纯旋转, 无平移)

算法 (Phase A2):
  Step 1: 图像星对采样 (i,j), 计算 d_img, θ_img
  Step 2: Gaia 星对 k-vector 距离查询 (模拟: 暴力/排序二分)
  Step 3: 第三星 k 交叉验证 (大幅降低背景)
  Step 4: θ直方图投票, SNR = peak/background

实验:
  - 合成数据: t≠0 (大平移), 验证相对向量法能聚类
  - SNR对比: 无交叉验证 vs 有交叉验证
  - 不同星密度下 SNR 表现
"""
import os, sys, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体支持 (Windows)
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
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "relvec_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")
class TeeLogger:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            try: s.write(data); s.flush()
            except: pass
    def flush(self):
        for s in self.streams:
            try: s.flush()
            except: pass

_log_fp = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = TeeLogger(sys.__stdout__, _log_fp)
sys.stderr = TeeLogger(sys.__stderr__, _log_fp)

_RADTODEG = 180.0 / math.pi
_DEGTORAD = math.pi / 180.0


def apply_transform(P, s, theta, tx, ty):
    ct, st = math.cos(theta), math.sin(theta)
    Q = np.empty_like(P)
    Q[:, 0] = s * (ct * P[:, 0] - st * P[:, 1]) + tx
    Q[:, 1] = s * (st * P[:, 0] + ct * P[:, 1]) + ty
    return Q


# ============================================================================
# k-vector 距离索引 (模拟, 用排序+二分)
# ============================================================================
class KVectorIndex:
    """Gaia星对距离索引 (模拟k-vector)
    预计算所有Gaia星对的距离, 按距离排序, 支持范围查询
    """
    def __init__(self, W, min_len, max_len, max_pairs=200000):
        self.W = W
        N = len(W)
        pairs = []
        rng = random.Random(123)
        # 构建所有星对 (i<j)
        for i in range(N):
            for j in range(i+1, N):
                dx = W[i, 0] - W[j, 0]
                dy = W[i, 1] - W[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                if min_len <= d <= max_len:
                    ang = math.atan2(dy, dx)
                    pairs.append((d, ang, i, j))
                    if len(pairs) >= max_pairs:
                        break
            if len(pairs) >= max_pairs:
                break
        # 按距离排序
        pairs.sort(key=lambda x: x[0])
        self.distances = np.array([p[0] for p in pairs])
        self.angles = np.array([p[1] for p in pairs])
        self.idx_a = np.array([p[2] for p in pairs], dtype=np.int32)
        self.idx_b = np.array([p[3] for p in pairs], dtype=np.int32)
        print(f"    KVectorIndex: {len(pairs)} 对 Gaia 星对, 距离范围 [{min_len:.0f}, {max_len:.0f}]\"")

    def query(self, d_target, tol):
        """查询距离在 [d_target-tol, d_target+tol] 的所有星对
        返回: (angles, idx_a, idx_b) 三个数组
        """
        lo = d_target - tol
        hi = d_target + tol
        i_lo = int(np.searchsorted(self.distances, lo, side='left'))
        i_hi = int(np.searchsorted(self.distances, hi, side='right'))
        return (self.angles[i_lo:i_hi], self.idx_a[i_lo:i_hi], self.idx_b[i_lo:i_hi])


# ============================================================================
# 单θ采样 (Phase A 路径1, 用于对比)
# ============================================================================
def single_theta_sampling(U, Wf, s0, s_min=0.9, s_max=1.1, n_samples=5000, seed=42):
    """单θ采样 (绝对向量, 假设t=0)
    返回: θ投票数组 (度, [-180, 180))

    注意: θ = angle(U[i]) - angle(Wf[j]) (与C++ solver一致, 是逆变换角度)
    但为与相对向量法(正向θ)统一, 这里返回 -θ = angle(Wf[j]) - angle(U[i])
    """
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
            # 正向θ = angle(Wf) - angle(U) (U→Wf的旋转)
            theta = (w_ang[j] - u_ang[i]) * _RADTODEG
            theta = (theta + 180) % 360 - 180
            votes.append(theta)
    return np.array(votes)


# ============================================================================
# 相对向量法 (Phase A2 路径2)
# ============================================================================
def relative_vector_sampling(U, Wf, kv_index, s0,
                              n_samples=5000, dist_tol=3.0,
                              s_min=0.9, s_max=1.1,
                              use_third_star=True, third_star_tol=3.0,
                              seed=42):
    """相对向量法: 图像星对采样 + k-vector距离查询 + (可选)第三星交叉验证 + θ投票

    参数:
      U: 图像侧星点 (角秒, Y向上)
      Wf: Gaia侧星点 (翻转后, 角秒)
      kv_index: KVectorIndex 索引
      s0: 像素尺度 (角秒/像素) - 这里U已是角秒, 不用
      n_samples: 图像星对采样次数
      dist_tol: 距离容差 (角秒, 单侧)
      s_min, s_max: 尺度比约束 (Gaia距离 = 图像距离 / s)
      use_third_star: 是否启用第三星交叉验证
      third_star_tol: 第三星距离验证容差 (角秒)

    返回: (θ投票数组, n_total_candidates, n_passed_third_star)

    注意:
      θ_rot = angle(Δw) - angle(Δu) = θ_true (正向: U→Wf的旋转)
      距离查询: Gaia距离 ∈ [d_img/s_max, d_img/s_min] (因为 s = d_img/d_gaia)
    """
    rng = np.random.RandomState(seed)
    N_u = len(U)

    votes = []
    n_total_cand = 0
    n_passed = 0

    for _ in range(n_samples):
        # Step 1: 图像星对采样
        i = rng.randint(N_u)
        j = rng.randint(N_u)
        if i == j: continue
        dx = U[i, 0] - U[j, 0]
        dy = U[i, 1] - U[j, 1]
        d_img = math.sqrt(dx*dx + dy*dy)
        if d_img < 1.0: continue  # 太近跳过
        θ_img = math.atan2(dy, dx)

        # Step 2: k-vector 距离查询
        # s = d_img / d_gaia ∈ [s_min, s_max]
        # → d_gaia ∈ [d_img/s_max, d_img/s_min]
        d_query = d_img / math.sqrt(s_min * s_max)  # 中心距离
        tol = d_query * (s_max/s_min - 1) / 2 + dist_tol  # 范围半宽 + 噪声容差
        gaia_angles, gaia_a, gaia_b = kv_index.query(d_query, tol)
        n_cand = len(gaia_angles)
        n_total_cand += n_cand

        if n_cand == 0: continue

        if not use_third_star:
            # 不用第三星验证, 所有候选都投票
            # θ_rot = angle(Δw) - angle(Δu) = θ_true (正向)
            for k in range(n_cand):
                θ_rot = (gaia_angles[k] - θ_img) * _RADTODEG
                θ_rot = (θ_rot + 180) % 360 - 180
                votes.append(θ_rot)
                n_passed += 1
        else:
            # Step 3: 第三星交叉验证 (关键: 用第一对确定的精确s, 容差只用噪声)
            k_idx = rng.randint(N_u)
            if k_idx == i or k_idx == j: continue

            d_ik_img = math.sqrt((U[i, 0]-U[k_idx, 0])**2 + (U[i, 1]-U[k_idx, 1])**2)
            d_jk_img = math.sqrt((U[j, 0]-U[k_idx, 0])**2 + (U[j, 1]-U[k_idx, 1])**2)

            for k in range(n_cand):
                a, b = gaia_a[k], gaia_b[k]
                # 第一对确定的精确 s = d_img / d_gaia_ab
                d_gaia_ab = math.sqrt((Wf[a, 0]-Wf[b, 0])**2 + (Wf[a, 1]-Wf[b, 1])**2)
                if d_gaia_ab < 1e-6: continue
                s_est = d_img / d_gaia_ab  # 精确s (从第一对)

                # 第三星期望距离: d_ik_gaia = d_ik_img / s_est
                d_ik_gaia_exp = d_ik_img / s_est
                d_jk_gaia_exp = d_jk_img / s_est

                # 容差: 只用噪声 (s的误差通过第一对距离比传递, 很小)
                # s的相对误差 ≈ noise/dist_ab, 对dist_ab=5000", noise=3" → 0.06%
                # 第三星距离误差 ≈ d_ik * 0.06% + noise ≈ 3" + 3" = 6"
                s_rel_err = third_star_tol / max(d_gaia_ab, 1.0)
                d_ik_tol = d_ik_gaia_exp * s_rel_err + third_star_tol
                d_jk_tol = d_jk_gaia_exp * s_rel_err + third_star_tol

                d_ac = np.sqrt((Wf[a, 0]-Wf[:, 0])**2 + (Wf[a, 1]-Wf[:, 1])**2)
                d_bc = np.sqrt((Wf[b, 0]-Wf[:, 0])**2 + (Wf[b, 1]-Wf[:, 1])**2)
                mask = (np.abs(d_ac - d_ik_gaia_exp) < d_ik_tol) & \
                       (np.abs(d_bc - d_jk_gaia_exp) < d_jk_tol)
                mask[a] = False
                mask[b] = False
                if np.any(mask):
                    θ_rot = (gaia_angles[k] - θ_img) * _RADTODEG
                    θ_rot = (θ_rot + 180) % 360 - 180
                    votes.append(θ_rot)
                    n_passed += 1

    return np.array(votes), n_total_cand, n_passed


# ============================================================================
# SNR 计算
# ============================================================================
def compute_snr(votes, theta_true_deg, bin_width=2.0):
    """计算θ直方图的SNR
    SNR = peak_height / background_median
    返回: (snr, theta_peak, peak_height, background)
    """
    if len(votes) == 0:
        return 0.0, 0.0, 0, 0.0
    bins = np.arange(-180, 180 + bin_width, bin_width)
    hist, _ = np.histogram(votes, bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    peak_idx = np.argmax(hist)
    peak_height = hist[peak_idx]
    theta_peak = bin_centers[peak_idx]

    # 背景 = 排除峰值附近±10°的中位数
    mask = np.abs(bin_centers - theta_peak) > 10.0
    if np.sum(mask) > 0:
        background = np.median(hist[mask])
    else:
        background = np.mean(hist)
    background = max(background, 1.0)

    snr = peak_height / background
    return snr, theta_peak, peak_height, background


# ============================================================================
# 可视化
# ============================================================================
def plot_theta_histogram(votes_single, votes_relvec, theta_true_deg, title, out_path, snr_single=None, snr_relvec=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bin_width = 2.0
    bins = np.arange(-180, 180 + bin_width, bin_width)

    for ax, votes, label, snr_info in [
        (axes[0], votes_single, "单θ采样 (t=0假设)", snr_single),
        (axes[1], votes_relvec, "相对向量法 (无假设)", snr_relvec)
    ]:
        if len(votes) > 0:
            ax.hist(votes, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(theta_true_deg, color='red', linestyle='--', linewidth=2, label=f'θ_true={theta_true_deg}°')
        ax.set_xlabel('θ (°)', fontsize=11)
        ax.set_ylabel('投票数', fontsize=11)
        ax.set_title(f"{label}\n{snr_info if snr_info else ''}", fontsize=11)
        ax.legend(fontsize=10)
        ax.set_xlim(-180, 180)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → 图: {out_path}")


# ============================================================================
# 实验: 合成数据验证
# ============================================================================
def run_experiment(N_u=80, N_w_extra=120, s_true=1.05, theta_true_deg=60.0,
                    tx_true=800.0, ty_true=400.0, noise_sigma=2.0,
                    fov_size=10000.0, dist_tol=3.0, n_samples=5000,
                    use_third_star=True, seed=42, label=""):
    """单次实验: 对比单θ采样 vs 相对向量法

    参数:
      N_u: 图像星数
      N_w_extra: Gaia额外噪声星数 (W = N_u真实 + N_w_extra噪声)
      s_true, theta_true_deg, tx_true, ty_true: 真实变换 (P→Q, 即U→Wf正向)
      noise_sigma: 加到Q上的高斯噪声
      fov_size: FOV大小 (角秒)
      dist_tol: k-vector距离查询容差
      use_third_star: 是否启用第三星交叉验证
    """
    rng = np.random.RandomState(seed)

    # 1. 生成U (图像侧)
    U = rng.uniform(-fov_size/2, fov_size/2, (N_u, 2))

    # 2. 应用真实变换得到Q_matched, 加噪声
    theta_true = theta_true_deg * _DEGTORAD
    Q_matched_clean = apply_transform(U, s_true, theta_true, tx_true, ty_true)
    Q_matched = Q_matched_clean + rng.normal(0, noise_sigma, Q_matched_clean.shape)

    # 3. 生成额外噪声星
    Q_extra = rng.uniform(-fov_size/2, fov_size/2, (N_w_extra, 2))

    # 4. 合并成Wf (打乱)
    Wf = np.vstack([Q_matched, Q_extra])
    rng.shuffle(Wf)

    print(f"\n{'='*70}")
    print(f"实验 {label}: N_u={N_u}, N_w={len(Wf)} (真实{N_u}+噪声{N_w_extra})")
    print(f"  真实变换: s={s_true}, θ={theta_true_deg}°, t=({tx_true},{ty_true})\"")
    print(f"  t/FOV = ({tx_true/fov_size*100:.1f}%, {ty_true/fov_size*100:.1f}%)")
    print(f"  noise={noise_sigma}\", dist_tol={dist_tol}\", 第三星验证={use_third_star}")
    print(f"{'='*70}")

    # === 方法1: 单θ采样 (t=0假设) ===
    print("\n[方法1] 单θ采样 (t=0假设):")
    votes_single = single_theta_sampling(U, Wf, s0=1.0, n_samples=n_samples, seed=seed)
    snr_single, θ_peak_single, peak_h_single, bg_single = compute_snr(votes_single, theta_true_deg)
    # θ_true 在单θ采样中的期望值 (因为t≠0, 真实θ分散)
    # 单θ采样: θ = angle(U[i]) - angle(Wf[j]), 当t≠0时, 这个角度差依赖于i,j
    # 理论上无单一峰值
    print(f"  投票数: {len(votes_single)}, SNR={snr_single:.2f}, θ_peak={θ_peak_single:.1f}°, peak={peak_h_single}, bg={bg_single:.1f}")
    print(f"  (真实θ={theta_true_deg}°, 但t≠0会导致θ分散)")

    # === 方法2: 相对向量法 ===
    print("\n[方法2] 相对向量法 (无t假设):")
    # 构建k-vector索引
    min_len = fov_size * 0.05
    max_len = fov_size * 0.8
    kv = KVectorIndex(Wf, min_len, max_len, max_pairs=500000)

    votes_relvec, n_cand, n_passed = relative_vector_sampling(
        U, Wf, kv, s0=1.0, n_samples=n_samples,
        dist_tol=dist_tol, s_min=0.9, s_max=1.1,
        use_third_star=use_third_star,
        third_star_tol=dist_tol, seed=seed
    )
    snr_relvec, θ_peak_relvec, peak_h_relvec, bg_relvec = compute_snr(votes_relvec, theta_true_deg)
    print(f"  投票数: {len(votes_relvec)}, 总候选: {n_cand}, 通过第三星: {n_passed}")
    print(f"  SNR={snr_relvec:.2f}, θ_peak={θ_peak_relvec:.1f}°, peak={peak_h_relvec}, bg={bg_relvec:.1f}")
    print(f"  (真实θ={theta_true_deg}°, 误差={abs(θ_peak_relvec-theta_true_deg):.2f}°)")

    # 判断成功
    single_ok = snr_single > 5.0 and abs(θ_peak_single - theta_true_deg) < 5.0
    relvec_ok = snr_relvec > 5.0 and abs(θ_peak_relvec - theta_true_deg) < 5.0
    print(f"\n  单θ采样: SNR={snr_single:.2f} {'✓' if single_ok else '✗'}")
    print(f"  相对向量: SNR={snr_relvec:.2f} {'✓' if relvec_ok else '✗'}")

    return {
        'votes_single': votes_single, 'votes_relvec': votes_relvec,
        'snr_single': snr_single, 'snr_relvec': snr_relvec,
        'θ_peak_single': θ_peak_single, 'θ_peak_relvec': θ_peak_relvec,
        'single_ok': single_ok, 'relvec_ok': relvec_ok,
    }


def main():
    print("=" * 70)
    print("相对向量法实验 (DMPDV核心机制验证)")
    print("生产版本星点参数: U≈50 (img_n_target), W≈70-150 (自适应n_target)")
    print("=" * 70)

    results = []

    # === 生产版本星点参数 ===
    # U: img_n_target=50 (饱和优先+非饱和按flux补足到50)
    # W: n_target = max(50, round(1.5×N_u×query_area/img_area)) ≈ 70-100
    #    实际Gaia查询返回更多, FOV过滤后取最亮n_target颗
    #    W包含U的全部真实匹配 + 额外噪声星 (星等限制导致U漏检)
    # 典型比例: W/U ≈ 1.5-3, 这里用 N_u=50, N_w=100 (含50真实+50噪声)

    # === 实验1: t≈0 (小平移<1%FOV), 两种方法都应成功 ===
    r = run_experiment(N_u=50, N_w_extra=50, s_true=1.0, theta_true_deg=45.0,
                       tx_true=50.0, ty_true=-30.0,
                       noise_sigma=2.0, fov_size=10000.0,
                       dist_tol=3.0, n_samples=5000,
                       use_third_star=True, seed=42, label="1_t0")
    results.append(("exp1_t0", r, 45.0))
    plot_theta_histogram(r['votes_single'], r['votes_relvec'], 45.0,
                          f"Exp1: t~0 (small), N_u=50, N_w=100",
                          os.path.join(OUTPUT_DIR, "exp1_t0.png"),
                          snr_single=f"SNR={r['snr_single']:.2f}",
                          snr_relvec=f"SNR={r['snr_relvec']:.2f}")

    # === 实验2: t=10%FOV (大平移), 单θ应失败, 相对向量应成功 ===
    r = run_experiment(N_u=50, N_w_extra=50, s_true=1.0, theta_true_deg=45.0,
                       tx_true=1000.0, ty_true=-800.0,
                       noise_sigma=2.0, fov_size=10000.0,
                       dist_tol=3.0, n_samples=5000,
                       use_third_star=True, seed=42, label="2_larget")
    results.append(("exp2_larget", r, 45.0))
    plot_theta_histogram(r['votes_single'], r['votes_relvec'], 45.0,
                          f"Exp2: t=10%FOV (large), N_u=50, N_w=100",
                          os.path.join(OUTPUT_DIR, "exp2_larget.png"),
                          snr_single=f"SNR={r['snr_single']:.2f}",
                          snr_relvec=f"SNR={r['snr_relvec']:.2f}")

    # === 实验3: t大 + s≠1 (s=1.1) ===
    r = run_experiment(N_u=50, N_w_extra=50, s_true=1.1, theta_true_deg=120.0,
                       tx_true=1500.0, ty_true=600.0,
                       noise_sigma=2.0, fov_size=10000.0,
                       dist_tol=3.0, n_samples=5000,
                       use_third_star=True, seed=42, label="3_s11_larget")
    results.append(("exp3_s11_larget", r, 120.0))
    plot_theta_histogram(r['votes_single'], r['votes_relvec'], 120.0,
                          f"Exp3: t large + s=1.1, N_u=50, N_w=100",
                          os.path.join(OUTPUT_DIR, "exp3_s11_larget.png"),
                          snr_single=f"SNR={r['snr_single']:.2f}",
                          snr_relvec=f"SNR={r['snr_relvec']:.2f}")

    # === 实验4: 无第三星验证 (对比背景) ===
    print("\n\n###### Exp4: 无第三星验证 (对比背景) ######")
    r_no3 = run_experiment(N_u=50, N_w_extra=50, s_true=1.0, theta_true_deg=45.0,
                            tx_true=1000.0, ty_true=-800.0,
                            noise_sigma=2.0, fov_size=10000.0,
                            dist_tol=3.0, n_samples=5000,
                            use_third_star=False, seed=42, label="4_no_third_star")
    plot_theta_histogram(r_no3['votes_single'], r_no3['votes_relvec'], 45.0,
                          f"Exp4: no 3rd-star verify (bg compare), t=10%FOV",
                          os.path.join(OUTPUT_DIR, "exp4_no_third_star.png"),
                          snr_single=f"SNR={r['snr_single']:.2f} (with 3rd)",
                          snr_relvec=f"SNR={r_no3['snr_relvec']:.2f} (no 3rd)")

    # === 实验5: 密集星场 (N_u=50, N_w=150, W/U=3) ===
    r = run_experiment(N_u=50, N_w_extra=100, s_true=1.05, theta_true_deg=60.0,
                       tx_true=800.0, ty_true=400.0,
                       noise_sigma=2.0, fov_size=10000.0,
                       dist_tol=3.0, n_samples=5000,
                       use_third_star=True, seed=42, label="5_dense")
    results.append(("exp5_dense", r, 60.0))
    plot_theta_histogram(r['votes_single'], r['votes_relvec'], 60.0,
                          f"Exp5: dense N_u=50, N_w=150 (W/U=3), t=8%FOV",
                          os.path.join(OUTPUT_DIR, "exp5_dense.png"),
                          snr_single=f"SNR={r['snr_single']:.2f}",
                          snr_relvec=f"SNR={r['snr_relvec']:.2f}")

    # === 实验6: 极端比例 (N_u=50, N_w=200, W/U=4, 模拟短焦密集场) ===
    r = run_experiment(N_u=50, N_w_extra=150, s_true=1.05, theta_true_deg=60.0,
                       tx_true=800.0, ty_true=400.0,
                       noise_sigma=2.0, fov_size=10000.0,
                       dist_tol=3.0, n_samples=5000,
                       use_third_star=True, seed=42, label="6_extreme")
    results.append(("exp6_extreme", r, 60.0))
    plot_theta_histogram(r['votes_single'], r['votes_relvec'], 60.0,
                          f"Exp6: extreme N_u=50, N_w=200 (W/U=4), t=8%FOV",
                          os.path.join(OUTPUT_DIR, "exp6_extreme.png"),
                          snr_single=f"SNR={r['snr_single']:.2f}",
                          snr_relvec=f"SNR={r['snr_relvec']:.2f}")

    # === 总结 ===
    print("\n\n" + "=" * 70)
    print("实验总结 (生产版本星点参数: U=50, W=100-200)")
    print("=" * 70)
    print(f"{'实验':<25} {'单θ SNR':<12} {'相对向量 SNR':<15} {'单θ':<8} {'相对向量':<10}")
    print("-" * 70)
    for name, r, θ_true in results:
        print(f"{name:<25} {r['snr_single']:<12.2f} {r['snr_relvec']:<15.2f} "
              f"{'OK' if r['single_ok'] else 'X':<8} {'OK' if r['relvec_ok'] else 'X':<10}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"\n脚本异常: {e}")
        print(err)
        _log_fp.write(f"\n脚本异常: {e}\n{err}\n")
        _log_fp.flush()
    finally:
        _log_fp.close()

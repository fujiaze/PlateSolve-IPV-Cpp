"""合成数据实验: 验证双对采样 + 完整4参数变换(含平移) 的正确性

实验思路:
  1. 生成一组随机点 P (模拟U, 图像侧星点向量组, 原点在图像中心)
  2. 对P应用已知变换 (s_true, θ_true, tx_true, ty_true) 得到 Q (模拟Wf, Gaia侧向量组)
     Q = s_true·R(θ_true)·P + t_true
  3. 用双对采样估计 (s, θ, tx, ty)
  4. 验证: 用完整变换 Wp = s·R(θ)·Wf + t 变换Q→Wp
  5. 检查 U(P) 和 Wp 的位置一致性 (欧氏距离)

关键点:
  - 双对采样用差向量 ΔU = s·R(θ)·ΔW 消除t, 准确估计(s, θ)
  - tx, ty 由 t = U[i1] - s·R(θ)·Wf[j1] 计算
  - 验证时必须用完整变换 (含t), 让两个向量组原点对齐
"""
import os, sys, math, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "synth_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 同时输出到stdout和日志文件
LOG_FILE = os.path.join(OUTPUT_DIR, "run.log")
class TeeLogger:
    def __init__(self, *streams):
        self.streams = streams
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
    """应用相似变换: Q = s·R(θ)·P + t"""
    ct, st = math.cos(theta), math.sin(theta)
    Q = np.empty_like(P)
    Q[:, 0] = s * (ct * P[:, 0] - st * P[:, 1]) + tx
    Q[:, 1] = s * (st * P[:, 0] + ct * P[:, 1]) + ty
    return Q


def build_lines(P, min_len, max_lines=100000):
    """构建线集: 所有点对 (i1<i2), 返回 (i1, i2, length, angle)"""
    rng = random.Random(42)
    N = len(P)
    lines = []
    for i1 in range(N):
        for i2 in range(i1+1, N):
            dx = P[i1, 0] - P[i2, 0]
            dy = P[i1, 1] - P[i2, 1]
            L = math.sqrt(dx*dx + dy*dy)
            if L >= min_len:
                ang = math.atan2(dy, dx)
                lines.append((i1, i2, L, ang))
    if len(lines) > max_lines:
        rng.shuffle(lines)
        lines = lines[:max_lines]
    return lines


def select_representative_lengths(u_lines, w_lines, s_min=0.9, s_max=1.1, n_bins=30, top_k=8):
    """选择代表性线长"""
    if not u_lines or not w_lines:
        return []
    u_lens = np.array([l[2] for l in u_lines])
    w_lens = np.array([l[2] for l in w_lines])
    l_min = max(u_lens.min(), w_lens.min() * s_min)
    l_max = min(u_lens.max(), w_lens.max() * s_max)
    if l_max <= l_min:
        return []
    bins = np.linspace(l_min, l_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    u_hist, _ = np.histogram(u_lens, bins=bins)
    w_counts = np.zeros(n_bins)
    for i in range(n_bins):
        bc = bin_centers[i]
        w_counts[i] = np.sum((w_lens >= bc * s_min) & (w_lens <= bc * s_max))
    scores = np.minimum(u_hist, w_counts)
    valid_idx = np.where(scores > 0)[0]
    if len(valid_idx) == 0:
        return []
    sorted_idx = valid_idx[np.argsort(-scores[valid_idx])]
    selected = []
    for idx in sorted_idx:
        bc = bin_centers[idx]
        too_close = any(abs(bc - s) < (l_max - l_min) / n_bins * 2 for s in selected)
        if not too_close:
            selected.append(bc)
        if len(selected) >= top_k:
            break
    return selected


def dual_pair_sample(U, Wf, u_lines, w_lines, rep_lengths, s_min, s_max,
                     tol_frac=0.08, max_pairs=20000):
    """双对采样: 配对U线和W线, 计算完整4参数变换(s, θ, tx, ty)"""
    rng = random.Random(42)
    results = []
    w_lens = np.array([l[2] for l in w_lines])
    w_sorted_idx = np.argsort(w_lens)
    w_lens_sorted = w_lens[w_sorted_idx]

    for rep_L in rep_lengths:
        tol = rep_L * tol_frac
        u_matches = [l for l in u_lines if abs(l[2] - rep_L) < tol]
        if not u_matches:
            continue
        max_w_per_u = max(1, min(300, max_pairs // (len(u_matches) * len(rep_lengths) + 1)))

        for ul in u_matches:
            i1u, i2u, Lu, au = ul
            w_lo = Lu / s_max
            w_hi = Lu / s_min
            w_lo_idx = int(np.searchsorted(w_lens_sorted, w_lo, side='left'))
            w_hi_idx = int(np.searchsorted(w_lens_sorted, w_hi, side='right'))
            n_w_avail = w_hi_idx - w_lo_idx
            if n_w_avail == 0:
                continue
            if n_w_avail <= max_w_per_u:
                w_pick_indices = range(w_lo_idx, w_hi_idx)
            else:
                w_pick_indices = rng.sample(range(w_lo_idx, w_hi_idx), max_w_per_u)

            for wk in w_pick_indices:
                wl = w_lines[w_sorted_idx[wk]]
                j1w, j2w, Lw, aw = wl
                s = Lu / Lw
                theta = au - aw
                ct, st = math.cos(theta), math.sin(theta)
                # 完整4参数: t = U[i1] - s·R(θ)·Wf[j1]
                tx = U[i1u, 0] - s * (ct * Wf[j1w, 0] - st * Wf[j1w, 1])
                ty = U[i1u, 1] - s * (st * Wf[j1w, 0] + ct * Wf[j1w, 1])
                results.append((s, theta, tx, ty, i1u, i2u, j1w, j2w))
                if len(results) >= max_pairs:
                    return results
    return results


def count_position_match(U, Wf, s, theta, tx, ty, match_dist):
    """完整变换验证: Wp = s·R(θ)·Wf + t, 统计U中多少星有近邻Wp (位置距离)"""
    ct, st = math.cos(theta), math.sin(theta)
    Wp = np.empty_like(Wf)
    Wp[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
    Wp[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
    # 对每个U[i], 找Wp中最近邻, 距离<match_dist则计为匹配
    n_match = 0
    for i in range(len(U)):
        dx = Wp[:, 0] - U[i, 0]
        dy = Wp[:, 1] - U[i, 1]
        dist2 = dx*dx + dy*dy
        if dist2.min() < match_dist * match_dist:
            n_match += 1
    return n_match


def run_synth_experiment(N_stars=80, s_true=1.05, theta_true_deg=37.5,
                          tx_true=500.0, ty_true=-300.0,
                          noise_sigma=2.0, fov_size=10000.0,
                          match_dist_frac=0.01, seed=42):
    """单次合成实验

    参数:
      N_stars: 星点数
      s_true, theta_true_deg, tx_true, ty_true: 真实变换参数
      noise_sigma: 加到Q上的高斯噪声(角秒), 模拟质心误差
      fov_size: FOV大小(角秒), 用于生成点
      match_dist_frac: 匹配距离 = fov_size × match_dist_frac
    """
    rng = np.random.RandomState(seed)
    match_dist = fov_size * match_dist_frac

    # 1. 生成随机点 P (模拟U, 图像侧)
    P = rng.uniform(-fov_size/2, fov_size/2, (N_stars, 2))

    # 2. 应用真实变换得到 Q (模拟Wf, Gaia侧), 加噪声
    theta_true = theta_true_deg * _DEGTORAD
    Q_clean = apply_transform(P, s_true, theta_true, tx_true, ty_true)
    Q = Q_clean + rng.normal(0, noise_sigma, Q_clean.shape)

    print(f"\n{'='*60}")
    print(f"合成实验: N={N_stars}, s_true={s_true}, θ_true={theta_true_deg}°, "
          f"t_true=({tx_true},{ty_true}), noise={noise_sigma}\"")
    print(f"  FOV={fov_size}\", match_dist={match_dist:.1f}\"")
    print(f"{'='*60}")

    # 3. 构建线集
    min_len = fov_size * 0.05
    u_lines = build_lines(P, min_len)
    w_lines = build_lines(Q, min_len)
    print(f"  U线: {len(u_lines)}条, W线: {len(w_lines)}条")

    # 4. 选代表性长度
    rep_lens = select_representative_lengths(u_lines, w_lines, s_min=0.9, s_max=1.1)
    print(f"  代表性长度: {[f'{r:.0f}' for r in rep_lens]}")

    # 5. 双对采样
    transforms = dual_pair_sample(P, Q, u_lines, w_lines, rep_lens,
                                   s_min=0.9, s_max=1.1, max_pairs=20000)
    print(f"  双对采样: {len(transforms)}个变换")

    # 6. 验证每个变换 (完整4参数, 含平移)
    n_calc = min(len(transforms), 3000)
    if len(transforms) > n_calc:
        rng2 = np.random.RandomState(42)
        subset_idx = rng2.choice(len(transforms), n_calc, replace=False)
        subset = [transforms[k] for k in subset_idx]
    else:
        subset = transforms

    scored = []
    for (s, theta, tx, ty, i1, i2, j1, j2) in subset:
        nr = count_position_match(P, Q, s, theta, tx, ty, match_dist)
        scored.append((s, theta, tx, ty, nr))

    n_arr = np.array([x[4] for x in scored])
    θ_arr = np.array([(x[1] * _RADTODEG + 180) % 360 - 180 for x in scored])
    tx_arr = np.array([x[2] for x in scored])
    ty_arr = np.array([x[3] for x in scored])

    # 7. 聚类分析: 高分点(n>=N_stars*0.5)是否聚集在真实变换附近?
    high_thresh = int(N_stars * 0.5)
    high = n_arr >= high_thresh
    n_high = int(np.sum(high))
    print(f"\n  n_position_match: max={n_arr.max()} med={np.median(n_arr):.0f} "
          f"mean={n_arr.mean():.1f} >={high_thresh}: {n_high}个")

    if n_high >= 3:
        # 加权统计 (用n作权重)
        w = n_arr[high]
        θ_h = θ_arr[high]
        tx_h = tx_arr[high]
        ty_h = ty_arr[high]
        θ_mean = np.average(θ_h, weights=w)
        θ_std = np.sqrt(np.average((θ_h - θ_mean)**2, weights=w))
        tx_mean = np.average(tx_h, weights=w)
        tx_std = np.sqrt(np.average((tx_h - tx_mean)**2, weights=w))
        ty_mean = np.average(ty_h, weights=w)
        ty_std = np.sqrt(np.average((ty_h - ty_mean)**2, weights=w))

        # 计算逆变换的理论值 (双对采样解的是Wf→U的逆变换)
        # 真实: Q = s_true·R(θ_true)·P + t_true (P→Q)
        # 逆: U = (1/s_true)·R(-θ_true)·Wf - (1/s_true)·R(-θ_true)·t_true
        s_inv = 1.0 / s_true
        θ_inv = -theta_true_deg
        ct_i = math.cos(-theta_true_deg * _DEGTORAD)
        st_i = math.sin(-theta_true_deg * _DEGTORAD)
        tx_inv = -(ct_i * tx_true - st_i * ty_true) / s_true
        ty_inv = -(st_i * tx_true + ct_i * ty_true) / s_true

        print(f"\n  高分点(≥{high_thresh})聚类分析:")
        print(f"    θ: 加权均值={θ_mean:.2f}°, 标准差={θ_std:.2f}° (逆变换理论={θ_inv:.2f}°, 差={abs(θ_mean-θ_inv):.2f}°)")
        print(f"    tx: 加权均值={tx_mean:.1f}\", 标准差={tx_std:.1f}\" (逆变换理论={tx_inv:.1f}\", 差={abs(tx_mean-tx_inv):.1f}\")")
        print(f"    ty: 加权均值={ty_mean:.1f}\", 标准差={ty_std:.1f}\" (逆变换理论={ty_inv:.1f}\", 差={abs(ty_mean-ty_inv):.1f}\")")

        # 判断是否聚类成功
        θ_ok = θ_std < 5.0  # θ标准差<5°算聚类
        tx_ok = tx_std < fov_size * 0.05  # tx标准差<5%FOV算聚类
        ty_ok = ty_std < fov_size * 0.05
        # 判断是否收敛到正确值
        θ_correct = abs(θ_mean - θ_inv) < 2.0
        tx_correct = abs(tx_mean - tx_inv) < fov_size * 0.02
        ty_correct = abs(ty_mean - ty_inv) < fov_size * 0.02
        all_ok = θ_ok and tx_ok and ty_ok and θ_correct and tx_correct and ty_correct
        print(f"\n  聚类判断: θ_std<5°={θ_ok}, tx_std<5%FOV={tx_ok}, ty_std<5%FOV={ty_ok}")
        print(f"  收敛判断: θ误差<2°={θ_correct}, tx误差<2%FOV={tx_correct}, ty误差<2%FOV={ty_correct}")
        print(f"  *** {'聚类成功且收敛正确 ✓' if all_ok else '存在问题 ✗'} ***")
    else:
        print(f"\n  高分点不足({n_high}个), 无法分析聚类")

    return scored


def visualize(scored, s_true, theta_true_deg, tx_true, ty_true, N_stars, fov_size, out_path):
    """3D可视化: (θ, tx, ty) 散点图, 颜色=n_match"""
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"合成数据双对采样 (θ, tx, ty) 3D参数空间\n"
        f"真实: s={s_true} θ={theta_true_deg}° tx={tx_true}\" ty={ty_true}\"\n"
        f"N={N_stars}, FOV={fov_size}\"",
        fontsize=12, fontweight='bold'
    )

    ax = fig.add_subplot(111, projection='3d')
    n_arr = np.array([x[4] for x in scored], dtype=float)
    θ_arr = np.array([(x[1] * _RADTODEG + 180) % 360 - 180 for x in scored])
    tx_arr = np.array([x[2] for x in scored])
    ty_arr = np.array([x[3] for x in scored])

    colors = np.log10(n_arr + 1)
    sizes = 5 + n_arr * 0.5

    high = n_arr >= N_stars * 0.5
    low = ~high

    if np.any(low):
        ax.scatter(θ_arr[low], tx_arr[low], ty_arr[low],
                  c='lightgrey', s=3, alpha=0.2, zorder=1)
    if np.any(high):
        sc = ax.scatter(θ_arr[high], tx_arr[high], ty_arr[high],
                       c=colors[high], cmap='hot', s=sizes[high],
                       alpha=0.7, edgecolors='red', linewidths=0.3, zorder=2)
        plt.colorbar(sc, ax=ax, shrink=0.6, label='log10(n_match+1)')

    # 标记真实变换
    ax.scatter([theta_true_deg], [tx_true], [ty_true],
              c='lime', s=300, marker='*', edgecolors='black', linewidths=2,
              zorder=3, label=f'真实: θ={theta_true_deg}° tx={tx_true} ty={ty_true}')
    ax.legend(fontsize=10, loc='upper left')

    ax.set_xlabel('θ (°)', fontsize=11)
    ax.set_ylabel('tx (\")', fontsize=11)
    ax.set_zlabel('ty (\")', fontsize=11)
    ax.view_init(elev=25, azim=45)
    n_max = int(n_arr.max())
    n_high = int(np.sum(high))
    ax.set_title(f"n={len(scored)}, max_match={n_max}, 高分(≥{int(N_stars*0.5)}): {n_high}个", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → 3D图: {out_path}")


def main():
    print("=" * 60)
    print("合成数据验证: 双对采样 + 完整4参数变换(含平移)")
    print("=" * 60)

    # 实验1: 理想情况, 无噪声, 小平移
    print("\n\n###### 实验1: 理想情况 (无噪声, 小平移) ######")
    scored1 = run_synth_experiment(
        N_stars=80, s_true=1.0, theta_true_deg=45.0,
        tx_true=100.0, ty_true=-80.0,
        noise_sigma=0.0, fov_size=10000.0,
        match_dist_frac=0.005,  # 50"
        seed=42
    )
    visualize(scored1, 1.0, 45.0, 100.0, -80.0, 80, 10000.0,
              os.path.join(OUTPUT_DIR, "synth1_ideal.png"))

    # 实验2: 有噪声, 小平移
    print("\n\n###### 实验2: 有噪声 (σ=2\", 小平移) ######")
    scored2 = run_synth_experiment(
        N_stars=80, s_true=1.0, theta_true_deg=45.0,
        tx_true=100.0, ty_true=-80.0,
        noise_sigma=2.0, fov_size=10000.0,
        match_dist_frac=0.005,
        seed=42
    )
    visualize(scored2, 1.0, 45.0, 100.0, -80.0, 80, 10000.0,
              os.path.join(OUTPUT_DIR, "synth2_noise_smallt.png"))

    # 实验3: 有噪声, 大平移 (模拟中心偏移大的情况)
    print("\n\n###### 实验3: 有噪声 (σ=2\", 大平移=10%FOV) ######")
    scored3 = run_synth_experiment(
        N_stars=80, s_true=1.0, theta_true_deg=45.0,
        tx_true=1000.0, ty_true=-800.0,  # 10%FOV
        noise_sigma=2.0, fov_size=10000.0,
        match_dist_frac=0.005,
        seed=42
    )
    visualize(scored3, 1.0, 45.0, 1000.0, -800.0, 80, 10000.0,
              os.path.join(OUTPUT_DIR, "synth3_noise_larget.png"))

    # 实验4: s≠1, 有噪声, 大平移
    print("\n\n###### 实验4: s=1.1, 有噪声, 大平移 ######")
    scored4 = run_synth_experiment(
        N_stars=80, s_true=1.1, theta_true_deg=120.0,
        tx_true=1500.0, ty_true=600.0,
        noise_sigma=2.0, fov_size=10000.0,
        match_dist_frac=0.005,
        seed=42
    )
    visualize(scored4, 1.1, 120.0, 1500.0, 600.0, 80, 10000.0,
              os.path.join(OUTPUT_DIR, "synth4_s11_larget.png"))

    # 实验5: 密集星场 (N=200), 有噪声, 大平移
    print("\n\n###### 实验5: 密集星场 (N=200), 有噪声, 大平移 ######")
    scored5 = run_synth_experiment(
        N_stars=200, s_true=1.05, theta_true_deg=60.0,
        tx_true=800.0, ty_true=400.0,
        noise_sigma=2.0, fov_size=10000.0,
        match_dist_frac=0.005,
        seed=42
    )
    visualize(scored5, 1.05, 60.0, 800.0, 400.0, 200, 10000.0,
              os.path.join(OUTPUT_DIR, "synth5_dense.png"))

    # 实验6: 模拟真实情况 - U和W非一一对应 (W比U多, 模拟星等限制)
    print("\n\n###### 实验6: 真实模拟 (U=80, W=200含U的全部+120噪声星, 大平移) ######")
    scored6 = run_synth_experiment_partial(
        N_u=80, N_w_extra=120,  # W = U变换后的80颗 + 120颗额外噪声星 = 200颗
        s_true=1.05, theta_true_deg=60.0,
        tx_true=800.0, ty_true=400.0,
        noise_sigma=2.0, fov_size=10000.0,
        match_dist_frac=0.005,
        seed=42
    )
    visualize(scored6, 1.05, 60.0, 800.0, 400.0, 80, 10000.0,
              os.path.join(OUTPUT_DIR, "synth6_partial_match.png"))

    # 实验7: 更极端 - U=50, W=300 (W是U的6倍, 模拟生产版本比例)
    print("\n\n###### 实验7: 极端比例 (U=50, W=300, 大平移) ######")
    scored7 = run_synth_experiment_partial(
        N_u=50, N_w_extra=250,
        s_true=1.05, theta_true_deg=60.0,
        tx_true=800.0, ty_true=400.0,
        noise_sigma=2.0, fov_size=10000.0,
        match_dist_frac=0.005,
        seed=42
    )
    visualize(scored7, 1.05, 60.0, 800.0, 400.0, 50, 10000.0,
              os.path.join(OUTPUT_DIR, "synth7_extreme_ratio.png"))

    print("\n\n" + "=" * 60)
    print("合成数据实验完成")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 60)


def run_synth_experiment_partial(N_u=80, N_w_extra=120, s_true=1.05, theta_true_deg=37.5,
                                   tx_true=500.0, ty_true=-300.0,
                                   noise_sigma=2.0, fov_size=10000.0,
                                   match_dist_frac=0.005, seed=42):
    """模拟真实情况: U和W非一一对应
    - U: N_u颗星 (图像侧)
    - W: N_u颗星(U变换后) + N_w_extra颗额外噪声星 (Gaia侧, 模拟星等限制导致U漏检)
    """
    rng = np.random.RandomState(seed)
    match_dist = fov_size * match_dist_frac

    # 1. 生成U (图像侧)
    U = rng.uniform(-fov_size/2, fov_size/2, (N_u, 2))

    # 2. 对U应用真实变换得到Q_matched (这些星在W中有对应)
    theta_true = theta_true_deg * _DEGTORAD
    Q_matched_clean = apply_transform(U, s_true, theta_true, tx_true, ty_true)
    Q_matched = Q_matched_clean + rng.normal(0, noise_sigma, Q_matched_clean.shape)

    # 3. 生成额外噪声星 (模拟Gaia中有的星但图像侧没检测到)
    Q_extra = rng.uniform(-fov_size/2, fov_size/2, (N_w_extra, 2))

    # 4. 合并成W (打乱顺序)
    Wf = np.vstack([Q_matched, Q_extra])
    rng.shuffle(Wf)

    print(f"\n{'='*60}")
    print(f"合成实验(非一一对应): U={N_u}, W={N_u}+{N_w_extra}={len(Wf)}, "
          f"s_true={s_true}, θ_true={theta_true_deg}°, t_true=({tx_true},{ty_true})")
    print(f"  FOV={fov_size}\", match_dist={match_dist:.1f}\", noise={noise_sigma}\"")
    print(f"{'='*60}")

    # 5. 构建线集 (注意: U线用U, W线用Wf)
    min_len = fov_size * 0.05
    u_lines = build_lines(U, min_len)
    w_lines = build_lines(Wf, min_len)
    print(f"  U线: {len(u_lines)}条, W线: {len(w_lines)}条")

    # 6. 选代表性长度
    rep_lens = select_representative_lengths(u_lines, w_lines, s_min=0.9, s_max=1.1)
    print(f"  代表性长度: {[f'{r:.0f}' for r in rep_lens]}")

    # 7. 双对采样
    transforms = dual_pair_sample(U, Wf, u_lines, w_lines, rep_lens,
                                   s_min=0.9, s_max=1.1, max_pairs=20000)
    print(f"  双对采样: {len(transforms)}个变换")

    # 8. 验证
    n_calc = min(len(transforms), 3000)
    if len(transforms) > n_calc:
        rng2 = np.random.RandomState(42)
        subset_idx = rng2.choice(len(transforms), n_calc, replace=False)
        subset = [transforms[k] for k in subset_idx]
    else:
        subset = transforms

    scored = []
    for (s, theta, tx, ty, i1, i2, j1, j2) in subset:
        nr = count_position_match(U, Wf, s, theta, tx, ty, match_dist)
        scored.append((s, theta, tx, ty, nr))

    n_arr = np.array([x[4] for x in scored])
    θ_arr = np.array([(x[1] * _RADTODEG + 180) % 360 - 180 for x in scored])
    tx_arr = np.array([x[2] for x in scored])
    ty_arr = np.array([x[3] for x in scored])

    # 高分阈值: U的50% (因为W有噪声星, 正确变换也只能匹配U中的星)
    high_thresh = int(N_u * 0.5)
    high = n_arr >= high_thresh
    n_high = int(np.sum(high))
    print(f"\n  n_position_match: max={n_arr.max()} med={np.median(n_arr):.0f} "
          f"mean={n_arr.mean():.1f} >={high_thresh}: {n_high}个")
    print(f"  (理论上正确变换应匹配 {N_u} 颗, 即U的全部)")

    if n_high >= 3:
        w = n_arr[high]
        θ_h = θ_arr[high]
        tx_h = tx_arr[high]
        ty_h = ty_arr[high]
        θ_mean = np.average(θ_h, weights=w)
        θ_std = np.sqrt(np.average((θ_h - θ_mean)**2, weights=w))
        tx_mean = np.average(tx_h, weights=w)
        tx_std = np.sqrt(np.average((tx_h - tx_mean)**2, weights=w))
        ty_mean = np.average(ty_h, weights=w)
        ty_std = np.sqrt(np.average((ty_h - ty_mean)**2, weights=w))

        # 逆变换理论值
        s_inv = 1.0 / s_true
        θ_inv = -theta_true_deg
        ct_i = math.cos(-theta_true_deg * _DEGTORAD)
        st_i = math.sin(-theta_true_deg * _DEGTORAD)
        tx_inv = -(ct_i * tx_true - st_i * ty_true) / s_true
        ty_inv = -(st_i * tx_true + ct_i * ty_true) / s_true

        print(f"\n  高分点(≥{high_thresh})聚类分析:")
        print(f"    θ: 加权均值={θ_mean:.2f}°, 标准差={θ_std:.2f}° (逆变换理论={θ_inv:.2f}°, 差={abs(θ_mean-θ_inv):.2f}°)")
        print(f"    tx: 加权均值={tx_mean:.1f}\", 标准差={tx_std:.1f}\" (逆变换理论={tx_inv:.1f}\", 差={abs(tx_mean-tx_inv):.1f}\")")
        print(f"    ty: 加权均值={ty_mean:.1f}\", 标准差={ty_std:.1f}\" (逆变换理论={ty_inv:.1f}\", 差={abs(ty_mean-ty_inv):.1f}\")")

        θ_ok = θ_std < 5.0
        tx_ok = tx_std < fov_size * 0.05
        ty_ok = ty_std < fov_size * 0.05
        θ_correct = abs(θ_mean - θ_inv) < 2.0
        tx_correct = abs(tx_mean - tx_inv) < fov_size * 0.02
        ty_correct = abs(ty_mean - ty_inv) < fov_size * 0.02
        all_ok = θ_ok and tx_ok and ty_ok and θ_correct and tx_correct and ty_correct
        print(f"\n  聚类判断: θ_std<5°={θ_ok}, tx_std<5%FOV={tx_ok}, ty_std<5%FOV={ty_ok}")
        print(f"  收敛判断: θ误差<2°={θ_correct}, tx误差<2%FOV={tx_correct}, ty误差<2%FOV={ty_correct}")
        print(f"  *** {'聚类成功且收敛正确 ✓' if all_ok else '存在问题 ✗'} ***")
    else:
        print(f"\n  高分点不足({n_high}个), 无法分析聚类")

    return scored


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

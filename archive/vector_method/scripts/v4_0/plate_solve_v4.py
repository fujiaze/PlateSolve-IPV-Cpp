"""V4.0 Plate Solve — 余弦相似度匹配 + 迭代精修

用途:
  天文图像plate solve, 仅依赖FITS header中的RA/DEC/焦距/相元,
  不需要预置WCS解. 通过点积(余弦)匹配+模长比迭代, 从随机抽样中
  自动发现正确的翻转模式和相似变换参数.

算法流程:
  第1步: 4模式抽样统计
    对4种翻转模式各5000次随机配对抽样, 计算候选相似变换,
    用余弦相似度匹配向量对, 记录n_matched.
    对θ分箱统计PNR(峰值/噪声比)/Z-score/excess,
    自动选择PNR最高的模式作为正确模式.

  第2步: 迭代精修 (仅最佳模式)
    从最高峰取n_matched最高的抽样作为种子:
    a. 余弦匹配 → 找方向一致的向量对
    b. MAD剔除离群匹配 (模长比异常)
    c. 4参数最小二乘拟合相似变换 (s, θ, tx, ty)
    d. 用新变换重新匹配 (之前因畸变排除的向量可能重新入选)
    e. 重复直到参数收敛

核心洞察:
  - 畸变影响模长, 不影响方向 → 余弦相似度匹配天然抗畸变
  - 正确变换下: 大量匹配对(n≈48), 模长比极一致(std≈0.002)
  - 错误变换下: 匹配对少(n≈15), 模长比分散(std≈0.15)
  - PNR>3即为强信号, 一眼区分正确/错误模式

输入:
  FITS图像文件 (通过命令行参数或文件选择框指定)
  需要: RA, DEC, FOCALLEN, XPIXSZ 在FITS header中

输出:
  在脚本所在目录下, 以图像文件名创建输出文件夹:
    <image_name>/
      mode_comparison.png   — 4模式PNR/Z/n_matched对比图
      refine_iter00.png     — 迭代0可视化 (向量叠加+匹配质量+残差)
      refine_iter01.png     — 迭代1可视化
      ...

使用:
  python plate_solve_v4.py                          # 弹出文件选择框
  python plate_solve_v4.py path/to/image.fts        # 命令行指定文件

依赖:
  numpy, matplotlib, astropy
  项目lib: astro_image_io, star_detector, vector_match_v2
"""
import sys
import os
import math
import re
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import (
    GaiaClientPy,
    _DEGTORAD,
    gnomonic_forward,
    _apply_flip,
)

logging.basicConfig(level=logging.WARNING)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_ra(raw):
    if raw is None:
        return None
    raw = str(raw).strip()
    m = re.match(r'(\d+)[\s:h](\d+)[\s:m](\d+\.?\d*)', raw)
    if m:
        return (float(m.group(1)) + float(m.group(2)) / 60.0 + float(m.group(3)) / 3600.0) * 15.0
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_dec(raw):
    if raw is None:
        return None
    raw = str(raw).strip()
    m = re.match(r'([+-]?)(\d+)[\s:d](\d+)[\s:m](\d+\.?\d*)', raw)
    if m:
        sign = -1.0 if m.group(1) == '-' else 1.0
        return sign * (float(m.group(2)) + float(m.group(3)) / 60.0 + float(m.group(4)) / 3600.0)
    try:
        return float(raw)
    except ValueError:
        return None


def load_frame(fits_path, gaia_dir=None):
    from astro_image_io import ImageReader
    from star_detector import StarDetector, SDetParamsPy

    if gaia_dir is None:
        gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")

    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz

    from astropy.io import fits as pf
    hdr = pf.getheader(fits_path)
    ra_raw = hdr.get("RA", hdr.get("OBJRA", None))
    dec_raw = hdr.get("DEC", hdr.get("OBJDEC", None))
    cra0 = _parse_ra(ra_raw)
    cdec0 = _parse_dec(dec_raw)
    if cra0 is None or cdec0 is None:
        raise RuntimeError(f"FITS header缺少RA/DEC: RA={ra_raw} DEC={dec_raw}")

    s0 = 206.265 * ps / fl
    halfW = w / 2.0 * s0
    halfH = h / 2.0 * s0
    fov_diag = math.sqrt(w * w + h * h) * s0

    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)

    img_x = np.array(det.x, np.float64)
    img_y = np.array(det.y, np.float64)
    img_flux = np.array(det.flux, np.float64)
    img_sat = np.array(det.saturated, dtype=np.bool_)

    sat_idx = np.where(img_sat)[0]
    non_sat_idx = np.where(~img_sat)[0]
    if len(non_sat_idx) > 0:
        non_sat_sorted = non_sat_idx[np.argsort(-img_flux[non_sat_idx])]
    else:
        non_sat_sorted = np.array([], dtype=np.int64)
    all_idx = np.concatenate([sat_idx, non_sat_sorted])

    cx, cy = w / 2.0, h / 2.0
    ux_all = (img_x[all_idx] - cx) * s0
    uy_all = -(img_y[all_idx] - cy) * s0
    U_all = np.column_stack([ux_all, uy_all])

    m_cut = 6.0 + 1.5 * math.log10(max(fl, 1.0)) + 2.0 * math.log10(300.0)
    query_radius = fov_diag * 0.5 / 3600.0
    mag_query = m_cut

    gaia = GaiaClientPy(gaia_dir, 1)
    for attempt in range(10):
        ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, query_radius, mag_query)
        if len(ra_t) >= 200:
            break
        mag_query += 0.5
    else:
        ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, query_radius, 22.0)
    gaia.close()

    xi_all, eta_all, valid_all = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
    in_fov = valid_all & (np.abs(xi_all) < halfW * 2.0) & (np.abs(eta_all) < halfH * 2.0)
    fov_idx = np.where(in_fov)[0]
    fov_mag = mag_t[fov_idx]
    sorted_mag = np.argsort(fov_mag)

    W_all = np.column_stack(
        [xi_all[fov_idx[sorted_mag]], eta_all[fov_idx[sorted_mag]]]
    )

    print(f"[数据] 图像: {w}x{h}, fl={fl}mm, s0={s0:.4f}\"/px")
    print(f"[数据] 星点: {len(all_idx)}颗 (饱和{len(sat_idx)}颗)")
    print(f"[数据] Gaia: {len(ra_t)}颗查询, {W_all.shape[0]}颗FOV内")
    print(f"[数据] 指向: RA={cra0:.4f}° Dec={cdec0:.4f}° (header)")

    return {
        "U_all": U_all, "W_all": W_all,
        "s0": s0, "halfW": halfW, "halfH": halfH,
        "fov_diag": fov_diag, "cra0": cra0, "cdec0": cdec0,
        "w": w, "h": h,
    }


def cosine_similarity_matrix(U, Wt):
    U_norm = np.linalg.norm(U, axis=1, keepdims=True)
    Wt_norm = np.linalg.norm(Wt, axis=1, keepdims=True)
    U_hat = U / np.maximum(U_norm, 1e-10)
    Wt_hat = Wt / np.maximum(Wt_norm, 1e-10)
    return U_hat @ Wt_hat.T


def find_cos_matches(U, Wt, cos_thresh=0.995, ratio_range=(0.7, 1.3)):
    cos_mat = cosine_similarity_matrix(U, Wt)
    U_norms = np.linalg.norm(U, axis=1)
    Wt_norms = np.linalg.norm(Wt, axis=1)

    matches = []
    used_w = set()
    best_cos_per_u = np.max(cos_mat, axis=1)
    u_order = np.argsort(-best_cos_per_u)

    for i in u_order:
        j = np.argmax(cos_mat[i])
        cos_val = cos_mat[i, j]
        if cos_val < cos_thresh or j in used_w:
            continue
        if Wt_norms[j] < 1e-10:
            continue
        ratio = U_norms[i] / Wt_norms[j]
        if ratio < ratio_range[0] or ratio > ratio_range[1]:
            continue
        matches.append((i, j, float(cos_mat[i, j]), ratio))
        used_w.add(j)

    return matches


def mad_filter(matches, key="ratio", threshold=3.5):
    if len(matches) <= 2:
        return matches
    vals = np.array([m[3] if key == "ratio" else m[2] for m in matches])
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    if mad < 1e-10:
        return matches
    z = np.abs(vals - med) / (mad * 1.4826)
    return [m for m, zi in zip(matches, z) if zi < threshold]


def fit_similarity_transform(U_matched, W_matched):
    n = len(U_matched)
    if n < 2:
        return None
    A = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    for k in range(n):
        wx, wy = W_matched[k]
        ux, uy = U_matched[k]
        A[2 * k] = [wx, -wy, 1, 0]
        A[2 * k + 1] = [wy, wx, 0, 1]
        b[2 * k] = ux
        b[2 * k + 1] = uy
    params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coef, tx, ty = params
    s = math.hypot(a, b_coef)
    theta = math.atan2(b_coef, a)
    return s, theta, tx, ty


def apply_transform(W, s, theta, tx, ty):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return s * (W @ R.T) + np.array([tx, ty])


def iterate_refine(U_pool, Wf_pool, s_init, theta_init, tx_init, ty_init,
                   output_dir, max_iter=20, cos_thresh=0.995, ratio_range=(0.7, 1.3),
                   tag="refine", do_vis=True):
    s, theta, tx, ty = s_init, theta_init, tx_init, ty_init
    history = []

    for it in range(max_iter):
        Wt = apply_transform(Wf_pool, s, theta, tx, ty)

        matches = find_cos_matches(U_pool, Wt, cos_thresh, ratio_range)
        if len(matches) < 2:
            history.append({"iter": it, "n_matched": 0, "converged": False})
            break

        n_before = len(matches)
        matches = mad_filter(matches, "ratio", 3.5)
        n_after = len(matches)
        if n_after < 2:
            history.append({"iter": it, "n_matched": 0, "converged": False})
            break

        U_matched = np.array([U_pool[m[0]] for m in matches])
        W_matched = np.array([Wf_pool[m[1]] for m in matches])
        ratios = np.array([m[3] for m in matches])
        cos_vals = np.array([m[2] for m in matches])

        new_params = fit_similarity_transform(U_matched.tolist(), W_matched.tolist())
        if new_params is None:
            history.append({"iter": it, "n_matched": n_after, "converged": False})
            break

        s_new, theta_new, tx_new, ty_new = new_params

        delta_s = abs(s_new - s)
        delta_theta = abs(theta_new - theta)
        delta_t = math.hypot(tx_new - tx, ty_new - ty)

        converged = delta_s < 1e-6 and delta_theta < math.radians(0.001) and delta_t < 0.01

        Wt_norms = np.linalg.norm(Wt[[m[1] for m in matches]], axis=1)
        U_norms = np.linalg.norm(U_pool[[m[0] for m in matches]], axis=1)
        residuals = U_norms - Wt_norms
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        record = {
            "iter": it, "n_matched": n_after,
            "n_matched_before": n_before, "n_matched_after": n_after,
            "s": s_new, "theta_deg": math.degrees(theta_new),
            "tx": tx_new, "ty": ty_new,
            "ratio_median": float(np.median(ratios)),
            "ratio_std": float(np.std(ratios)),
            "cos_median": float(np.median(cos_vals)),
            "rmse": rmse,
            "delta_s": delta_s, "delta_theta_deg": math.degrees(delta_theta),
            "delta_t": delta_t,
            "converged": converged,
            "matches": matches,
        }
        history.append(record)

        if do_vis:
            plot_vectors(U_pool, Wf_pool, Wt, matches, s_new,
                         math.degrees(theta_new), tx_new, ty_new, it, tag, output_dir)

        if converged:
            break

        s, theta, tx, ty = s_new, theta_new, tx_new, ty_new

    return history


def plot_vectors(U_pool, Wf_pool, Wt, matches, s, theta_deg, tx, ty, iteration, tag, output_dir):
    plt.rcParams.update({
        "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
        "figure.dpi": 120, "savefig.dpi": 150,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    fig = plt.figure(figsize=(20, 18))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.28)
    ax_vec = fig.add_subplot(gs[0, :])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_resid = fig.add_subplot(gs[1, 1])

    matched_u_idx = set(m[0] for m in matches) if matches else set()
    matched_w_idx = set(m[1] for m in matches) if matches else set()

    for i in range(len(U_pool)):
        is_m = i in matched_u_idx
        a = 0.85 if is_m else 0.15
        lw = 1.8 if is_m else 0.5
        ax_vec.plot([0, U_pool[i, 0]], [0, U_pool[i, 1]], "-", color="#CC0000", alpha=a, lw=lw)
        ax_vec.plot(U_pool[i, 0], U_pool[i, 1], "o", color="#CC0000", ms=4 if is_m else 2, alpha=a)

    for i in range(len(Wt)):
        is_m = i in matched_w_idx
        a = 0.7 if is_m else 0.12
        lw = 1.5 if is_m else 0.4
        ax_vec.plot([0, Wt[i, 0]], [0, Wt[i, 1]], "--", color="#0044CC", alpha=a, lw=lw)
        ax_vec.plot(Wt[i, 0], Wt[i, 1], "s", color="#0044CC", ms=3 if is_m else 1.5, alpha=a)

    if matches:
        lines = [[U_pool[m[0]], Wt[m[1]]] for m in matches]
        ax_vec.add_collection(LineCollection(lines, colors="#FFD700", linewidths=1.5, alpha=0.8, zorder=4))

    ax_vec.set_aspect("equal")
    ax_vec.legend(["U (matched)", "U (unmatched)", "W' (matched)", "W' (unmatched)", "Match link"],
                  loc="upper left", fontsize=9, framealpha=0.9)
    ax_vec.set_title(f"Vector overlay  —  iter {iteration}:  s = {s:.6f},  θ = {theta_deg:.4f}°,  "
                     f"tx = {tx:.1f},  ty = {ty:.1f},  n = {len(matches)}")

    if matches:
        ratios = np.array([m[3] for m in matches])
        cos_vals = np.array([m[2] for m in matches])
        r_med = float(np.median(ratios))
        r_std = float(np.std(ratios))
        c_med = float(np.median(cos_vals))

        ax_scatter.scatter(ratios, cos_vals, s=30, c="#2CA02C", alpha=0.8, edgecolors="darkgreen", linewidths=0.5)
        ax_scatter.axvline(x=r_med, color="red", ls="--", lw=1.5, label=f"ratio_med = {r_med:.4f}")
        ax_scatter.axhline(y=c_med, color="blue", ls="--", lw=1.5, label=f"cos_med = {c_med:.6f}")

        margin_r = max(0.02, r_std * 5)
        margin_c = max(0.002, np.std(cos_vals) * 5)
        ax_scatter.set_xlim(r_med - margin_r, r_med + margin_r)
        ax_scatter.set_ylim(c_med - margin_c, c_med + margin_c)
        ax_scatter.set_xlabel(r"Ratio $|\mathbf{U}| / |\mathbf{W}'|$")
        ax_scatter.set_ylabel("Cosine similarity")
        ax_scatter.set_title(f"Match quality:  ratio = {r_med:.4f} ± {r_std:.4f}")
        ax_scatter.legend(fontsize=9)

        U_norms = np.linalg.norm(U_pool[[m[0] for m in matches]], axis=1)
        Wt_norms = np.linalg.norm(Wt[[m[1] for m in matches]], axis=1)
        residuals = U_norms - Wt_norms
        rmse = float(np.sqrt(np.mean(residuals ** 2)))

        ax_resid.hist(residuals, bins=max(10, len(residuals) // 4), color="#FF7F0E", alpha=0.8, edgecolor="black")
        ax_resid.axvline(x=0, color="black", ls="--", lw=1)
        ax_resid.set_xlabel(r"$|\mathbf{U}| - |\mathbf{W}'|$  (arcsec)")
        ax_resid.set_ylabel("Count")
        ax_resid.set_title(f"Magnitude residual:  RMSE = {rmse:.2f}\"")

    fig.suptitle(
        f"Iteration {iteration}:   s = {s:.6f},   θ = {theta_deg:.4f}°,   "
        f"tx = {tx:.1f}\",   ty = {ty:.1f}\",   n = {len(matches)}",
        fontsize=15, fontweight="bold", y=0.98)

    plt.savefig(os.path.join(output_dir, f"{tag}_iter{iteration:02d}.png"))
    plt.close(fig)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})


def sample_mode(U_pool, Wf_pool, N_U, N_W, halfW, halfH, N_SAMPLES=5000):
    rng = np.random.RandomState(42)
    visited = set()
    sample_records = []

    for k in range(N_SAMPLES):
        for retry in range(20):
            i = rng.randint(N_U)
            j = rng.randint(N_W)
            pk = i * N_W + j
            if pk not in visited:
                visited.add(pk)
                break
        else:
            continue

        u_vec, w_vec = U_pool[i], Wf_pool[j]
        w_norm = math.hypot(w_vec[0], w_vec[1])
        if w_norm < 1e-10:
            continue
        u_norm = math.hypot(u_vec[0], u_vec[1])
        s = u_norm / w_norm
        if s < 0.85 or s > 1.15:
            continue
        theta = math.atan2(u_vec[1], u_vec[0]) - math.atan2(w_vec[1], w_vec[0])
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        tx = u_vec[0] - s * (cos_t * w_vec[0] - sin_t * w_vec[1])
        ty = u_vec[1] - s * (sin_t * w_vec[0] + cos_t * w_vec[1])

        if abs(tx) > 0.6 * halfW * 2 or abs(ty) > 0.6 * halfH * 2:
            continue

        Wt = apply_transform(Wf_pool, s, theta, tx, ty)
        n_m = len(find_cos_matches(U_pool, Wt, 0.995, (0.7, 1.3)))

        sample_records.append({
            "theta": theta, "theta_deg": math.degrees(theta),
            "s": s, "tx": tx, "ty": ty, "n_matched": n_m,
        })

    return sample_records


def compute_mode_stats(sample_records):
    BIN_DEG = 5
    n_bins = int(360 / BIN_DEG)
    bin_centers = np.linspace(-180 + BIN_DEG / 2, 180 - BIN_DEG / 2, n_bins)

    if not sample_records:
        return None

    n_matched_arr = np.array([r["n_matched"] for r in sample_records])
    noise_median = float(np.median(n_matched_arr))
    noise_mean = float(np.mean(n_matched_arr))
    noise_std = float(np.std(n_matched_arr))

    bin_n_max = np.zeros(n_bins)
    bin_n_sum = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    bin_n_vals = [[] for _ in range(n_bins)]

    for r in sample_records:
        b = int((r["theta_deg"] + 180) / BIN_DEG)
        b = max(0, min(n_bins - 1, b))
        bin_counts[b] += 1
        bin_n_sum[b] += r["n_matched"]
        bin_n_max[b] = max(bin_n_max[b], r["n_matched"])
        bin_n_vals[b].append(r["n_matched"])

    bin_n_mean = np.zeros(n_bins)
    for b in range(n_bins):
        if bin_n_vals[b]:
            bin_n_mean[b] = np.mean(bin_n_vals[b])

    pnr_arr = np.zeros(n_bins)
    zscore_arr = np.zeros(n_bins)
    excess_arr = np.zeros(n_bins)
    for b in range(n_bins):
        if bin_counts[b] > 0:
            pnr_arr[b] = bin_n_max[b] / max(noise_median, 1)
            zscore_arr[b] = (bin_n_mean[b] - noise_mean) / max(noise_std, 1)
            excess_arr[b] = bin_n_sum[b] - noise_median * bin_counts[b]

    peak_bin = np.argmax(excess_arr)
    peak_theta = bin_centers[peak_bin]

    return {
        "pnr": float(pnr_arr[peak_bin]),
        "zscore": float(zscore_arr[peak_bin]),
        "excess": float(excess_arr[peak_bin]),
        "n_max": int(bin_n_max[peak_bin]),
        "peak_theta": float(peak_theta),
        "noise_median": noise_median,
        "bin_centers": bin_centers,
        "bin_n_max": bin_n_max,
        "pnr_arr": pnr_arr,
        "zscore_arr": zscore_arr,
        "BIN_DEG": BIN_DEG,
    }


def main():
    print("=" * 70)
    print("  V4.0 Plate Solve — 余弦匹配 + 迭代精修")
    print("=" * 70)

    fits_path = None
    if len(sys.argv) > 1:
        fits_path = sys.argv[1]
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        fits_path = filedialog.askopenfilename(
            title="选择FITS图像文件",
            filetypes=[("FITS", "*.fits *.fit *.fts"), ("All", "*.*")],
        )
        root.destroy()
        if not fits_path:
            print("  未选择文件，退出")
            return

    img_name = os.path.splitext(os.path.basename(fits_path))[0]
    output_dir = os.path.join(SCRIPT_DIR, img_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n  文件: {fits_path}")
    print(f"  输出: {output_dir}")
    data = load_frame(fits_path)
    U_all = data["U_all"]
    W_all = data["W_all"]
    halfW, halfH = data["halfW"], data["halfH"]

    N_U = min(50, U_all.shape[0])
    N_W = min(100, W_all.shape[0])
    U_pool = U_all[:N_U]

    print(f"\n{'='*70}")
    print(f"  第1步: 4模式5000次抽样 → PNR对比 → 选最佳模式")
    print(f"{'='*70}")

    mode_results = {}
    for mode in range(4):
        Wf_pool = _apply_flip(W_all, mode)[:N_W]
        records = sample_mode(U_pool, Wf_pool, N_U, N_W, halfW, halfH, 5000)
        stats = compute_mode_stats(records)
        mode_results[mode] = {"records": records, "stats": stats}
        if stats:
            print(f"  mode={mode}: PNR={stats['pnr']:.2f} Z={stats['zscore']:.2f} "
                  f"excess={stats['excess']:.0f} n_max={stats['n_max']} "
                  f"peak_θ={stats['peak_theta']:.1f}°")
        else:
            print(f"  mode={mode}: 无有效抽样")

    best_mode = max(mode_results, key=lambda m: mode_results[m]["stats"]["pnr"] if mode_results[m]["stats"] else 0)
    best_stats = mode_results[best_mode]["stats"]
    print(f"\n  ★ 最佳模式: mode={best_mode} PNR={best_stats['pnr']:.2f} "
          f"peak_θ={best_stats['peak_theta']:.1f}°")

    BIN_DEG = best_stats["BIN_DEG"]
    bin_centers = best_stats["bin_centers"]

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    colors_4 = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD"]

    ax = axes[0, 0]
    for m in range(4):
        st = mode_results[m]["stats"]
        if st:
            ax.bar(bin_centers + (m - 1.5) * 1.2, st["bin_n_max"], width=1.0,
                   color=colors_4[m], alpha=0.7, label=f"mode={m}")
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("Max n_matched")
    ax.set_title("(a) Peak n_matched per θ bin")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    pnr_vals = [mode_results[m]["stats"]["pnr"] if mode_results[m]["stats"] else 0 for m in range(4)]
    ax.bar(range(4), pnr_vals, color=colors_4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"mode={m}" for m in range(4)])
    ax.set_ylabel("PNR")
    ax.set_title("(b) Peak-to-Noise Ratio")
    ax.axhline(y=3.0, color="black", ls="--", lw=1, label="PNR=3")
    ax.legend()

    ax = axes[1, 0]
    z_vals = [mode_results[m]["stats"]["zscore"] if mode_results[m]["stats"] else 0 for m in range(4)]
    ax.bar(range(4), z_vals, color=colors_4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"mode={m}" for m in range(4)])
    ax.set_ylabel("Z-score")
    ax.set_title("(c) Z-score")
    ax.axhline(y=3.0, color="black", ls="--", lw=1, label="3σ")
    ax.legend()

    ax = axes[1, 1]
    for m in range(4):
        recs = mode_results[m]["records"]
        if recs:
            ths = [r["theta_deg"] for r in recs]
            ns = [r["n_matched"] for r in recs]
            ax.scatter(ths, ns, s=2, alpha=0.15, c=colors_4[m], label=f"mode={m}")
    ax.set_xlabel("θ (deg)")
    ax.set_ylabel("n_matched")
    ax.set_title("(d) n_matched vs θ")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mode_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"\n{'='*70}")
    print(f"  第2步: mode={best_mode} 迭代精修")
    print(f"{'='*70}")

    Wf_pool = _apply_flip(W_all, best_mode)[:N_W]
    peak_theta = best_stats["peak_theta"]
    records = mode_results[best_mode]["records"]

    peak_samples = [r for r in records if abs(r["theta_deg"] - peak_theta) < BIN_DEG / 2]
    peak_samples.sort(key=lambda r: r["n_matched"], reverse=True)

    best_sample = peak_samples[0]
    print(f"\n  初始抽样: θ={best_sample['theta_deg']:.2f}° "
          f"s={best_sample['s']:.6f} tx={best_sample['tx']:.1f} ty={best_sample['ty']:.1f} "
          f"n={best_sample['n_matched']}")
    print(f"\n  迭代过程 (MAD剔除 → 4参数拟合 → 重新扩增):\n")

    hist = iterate_refine(
        U_pool, Wf_pool,
        best_sample["s"], best_sample["theta"], best_sample["tx"], best_sample["ty"],
        output_dir=output_dir, max_iter=20, tag="refine", do_vis=True)

    for rec in hist:
        it = rec["iter"]
        n_bef = rec.get("n_matched_before", rec["n_matched"])
        n_aft = rec.get("n_matched_after", rec["n_matched"])
        s_val = rec["s"]
        th = rec["theta_deg"]
        tx_v = rec.get("tx", 0)
        ty_v = rec.get("ty", 0)
        r_med = rec["ratio_median"]
        r_std = rec["ratio_std"]
        rmse = rec["rmse"]
        ds = rec.get("delta_s", 0)
        dth = rec.get("delta_theta_deg", 0)
        dt = rec.get("delta_t", 0)
        conv = " ★收敛" if rec.get("converged") else ""
        filtered = f"(-{n_bef - n_aft})" if n_bef != n_aft else ""
        print(f"  iter={it}: s={s_val:.6f} θ={th:.4f}° tx={tx_v:.1f} ty={ty_v:.1f} "
              f"n={n_aft}{filtered} ratio={r_med:.4f}±{r_std:.4f} rmse={rmse:.1f} "
              f"Δs={ds:.2e} Δθ={dth:.4f}° Δt={dt:.2f}{conv}")

    if hist and hist[-1]["n_matched"] > 0:
        last = hist[-1]
        print(f"\n  最终结果:")
        print(f"    s     = {last['s']:.6f}")
        print(f"    θ     = {last['theta_deg']:.4f}°")
        print(f"    tx    = {last.get('tx', 0):.2f}\"")
        print(f"    ty    = {last.get('ty', 0):.2f}\"")
        print(f"    n     = {last['n_matched']}")
        print(f"    ratio = {last['ratio_median']:.4f}±{last['ratio_std']:.4f}")
        print(f"    rmse  = {last['rmse']:.1f}\"")

    print(f"\n  输出目录: {output_dir}")


if __name__ == "__main__":
    main()

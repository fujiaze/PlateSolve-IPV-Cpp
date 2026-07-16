"""变换矩阵参数空间分布实验

对 Phase A 采样中所有 (i,j) 对计算变换参数 (s, θ, tx, ty),
在 4 维参数空间中可视化分布, 验证正确匹配是否形成紧密聚类。

变换格式: W' = s·R(θ)·Wf + t  (2D 相似变换, 4 自由度)
  - s = |U[i]| / |Wf[j]|           (缩放, 正确匹配 ≈ 1.0)
  - θ = atan2(U.y,U.x) - atan2(Wf.y,Wf.x)  (旋转)
  - tx = U.x - s·(cos·Wf.x - sin·Wf.y)     (X 平移)
  - ty = U.y - s·(sin·Wf.x + cos·Wf.y)     (Y 平移)

输出: 4 mode × 3 投影 = 12 子图的 PNG
"""
import os, sys, math, json, argparse

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
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy, gnomonic_forward
from v4_3.vector_match_v4_3_cpp import V43Solver

TESTDATA = os.path.join(PROJECT_ROOT, "testdata")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "param_space")
FULL_TEST_JSON = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "full_test", "full_test_all.json")

_RADTODEG = 180.0 / math.pi
_DEGTORAD = math.pi / 180.0
_ASEC_PER_RAD = 206264.80624709636


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


def compute_all_transforms(U, Wf, s0, s_min=0.9, s_max=1.1, max_subset=8000):
    """枚举所有 (i,j) 对, 计算变换参数 (s, θ, tx, ty) 和 n_in_range。

    U:  (N, 2)  图像星点向量 (角秒, Y向上)
    Wf: (M, 2)  Gaia 星点向量 (角秒, flip 后, Y向上)
    返回: list of (i, j, s, theta_deg, tx, ty, n_in_range)
    """
    N = len(U)
    M = len(Wf)
    if N == 0 or M == 0:
        return []

    # 计算向量和
    norm_U = np.sqrt(U[:, 0]**2 + U[:, 1]**2)
    angle_U = np.arctan2(U[:, 1], U[:, 0])
    norm_Wf = np.sqrt(Wf[:, 0]**2 + Wf[:, 1]**2)
    angle_Wf = np.arctan2(Wf[:, 1], Wf[:, 0])

    # 有效星点 (norm > 0)
    valid_U = norm_U > 1e-10
    valid_Wf = norm_Wf > 1e-10

    # FOV 对角线 (角秒)
    fov_diag = float(np.max(norm_U)) * 2.0
    max_t = fov_diag * 0.6

    s0_val = s0  # 角秒/像素
    match_dist = 5.0 * s0_val  # 匹配半径 (角秒)

    results = []
    valid_indices = []

    # 第1步: 枚举所有 (i,j), 计算 (s, θ, tx, ty), 过滤
    for i in range(N):
        if not valid_U[i]:
            continue
        for j in range(M):
            if not valid_Wf[j]:
                continue
            s = norm_U[i] / norm_Wf[j]
            if s < s_min or s > s_max:
                continue
            theta = angle_U[i] - angle_Wf[j]
            ct, st = math.cos(theta), math.sin(theta)
            tx = U[i, 0] - s * (ct * Wf[j, 0] - st * Wf[j, 1])
            ty = U[i, 1] - s * (st * Wf[j, 0] + ct * Wf[j, 1])
            if abs(tx) > max_t or abs(ty) > max_t:
                continue
            theta_deg = (theta * _RADTODEG + 180.0) % 360.0 - 180.0  # wrap [-180, 180)
            valid_indices.append((i, j, s, theta_deg, tx, ty))

    if not valid_indices:
        return []

    # 第2步: 对子集计算 n_in_range (用 KDTree 加速)
    n_total = len(valid_indices)
    if n_total <= max_subset:
        subset = valid_indices
    else:
        idx = np.random.choice(n_total, max_subset, replace=False)
        subset = [valid_indices[k] for k in idx]

    print(f"    有效对数: {n_total}, 计算n_in_range子集: {len(subset)}")

    for (i, j, s, theta_deg, tx, ty) in subset:
        theta = theta_deg * _DEGTORAD
        ct, st = math.cos(theta), math.sin(theta)
        # 变换所有 Wf → Wt
        Wt = np.empty_like(Wf)
        Wt[:, 0] = s * (ct * Wf[:, 0] - st * Wf[:, 1]) + tx
        Wt[:, 1] = s * (st * Wf[:, 0] + ct * Wf[:, 1]) + ty
        # KDTree 查最近邻
        tree = cKDTree(Wt)
        dist, nn_idx = tree.query(U, k=1)
        # 计数: 距离 < match_dist 且 scale ratio 在范围内
        d_ok = dist < match_dist
        sr = np.where(d_ok, norm_U / norm_Wf[nn_idx], 0.0)
        sr_ok = (sr >= s_min) & (sr <= s_max)
        n_in_range = int(np.sum(d_ok & sr_ok))
        results.append((i, j, s, theta_deg, tx, ty, n_in_range))

    return results


def run_experiment(fits_path, gaia_client, star_detector, solver, output_dir):
    """对单帧运行实验"""
    base = os.path.basename(fits_path)
    print(f"\n=== {base} ===")

    # 1. 读取图像
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    pixels = img.to_numpy()

    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl
    print(f"  图像: {w}×{h}, s0={s0:.4f}\"/px, fl={fl}mm, ps={ps}μm")

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

    sat_indices = [i for i in range(n_total) if all_sat[i]]
    normal_indices = [i for i in range(n_total) if not all_sat[i]]
    img_n_target = 50
    if len(sat_indices) >= img_n_target:
        u_indices = list(sat_indices[:])
        u_type = "sat_only"
    else:
        n_need = img_n_target - len(sat_indices)
        u_indices = list(sat_indices)
        u_indices.extend(normal_indices[:n_need])
        u_type = "sat+normal"

    N = len(u_indices)
    print(f"  检测星点: {n_total} (sat={len(sat_indices)}), U组: {N} ({u_type})")

    # 3. 构造 U 向量 (角秒, Y向上)
    cx, cy = w / 2.0, h / 2.0
    U = np.zeros((N, 2), dtype=np.float64)
    for k, idx in enumerate(u_indices):
        U[k, 0] = (all_x[idx] - cx) * s0
        U[k, 1] = -(all_y[idx] - cy) * s0

    # 4. 查询 Gaia 星表
    fov_diag_asec = math.sqrt((w * s0)**2 + (h * s0)**2)
    query_radius_deg = (fov_diag_asec * 0.5) / 3600.0 * 1.2  # 留余量
    mag_limit = 18.0
    cat_ra, cat_dec, cat_mag = gaia_client.cone_search(cra0, cdec0, query_radius_deg, mag_limit)
    if len(cat_ra) < 10:
        cat_ra, cat_dec, cat_mag = gaia_client.cone_search(cra0, cdec0, query_radius_deg, 22.0)
    print(f"  Gaia 查询: radius={query_radius_deg:.3f}°, mag_limit={mag_limit}, 返回 {len(cat_ra)} 星")

    # 5. Gnomonic 投影 + FOV 过滤
    xi, eta, valid = gnomonic_forward(cat_ra, cat_dec, cra0, cdec0)
    fov_half_w = w / 2.0 * s0
    fov_half_h = h / 2.0 * s0
    fov_mask = valid & (np.abs(xi) < fov_half_w) & (np.abs(eta) < fov_half_h)
    xi_f = xi[fov_mask]
    eta_f = eta[fov_mask]
    mag_f = cat_mag[fov_mask]

    # 按星等排序, 取最亮的 M 颗 (与 C++ 一致)
    sort_idx = np.argsort(mag_f)
    M_max = min(len(sort_idx), 500)
    W = np.zeros((M_max, 2), dtype=np.float64)
    for k in range(M_max):
        W[k, 0] = xi_f[sort_idx[k]]
        W[k, 1] = eta_f[sort_idx[k]]
    print(f"  FOV 内 Gaia 星: {int(np.sum(fov_mask))}, 取最亮 {M_max} 颗")

    # 6. 运行 solver 获取最终结果
    log_dir = os.path.join(output_dir, "logs", os.path.splitext(base)[0])
    result = solver.solve(
        image_path=fits_path, ra=cra0, dec=cdec0,
        focal_length_mm=fl, pixel_size_um=ps, log_dir=log_dir,
    )
    solver_mode = result.get("flip_mode", -1)
    solver_theta = result.get("rotation_deg", 0)
    solver_rms = result.get("rms_px", 0)
    solver_lnK = result.get("bayes_lnK", 0)
    solver_matched = result.get("matched_count", 0)
    print(f"  Solver: mode={solver_mode} θ={solver_theta:.2f}° RMS={solver_rms:.3f}px lnK={solver_lnK:.1f} matched={solver_matched}")

    # 7. 对 4 种 flip mode 计算所有变换
    mode_names = ["mode0 (无翻转)", "mode1 (X翻转)", "mode2 (Y翻转)", "mode3 (XY翻转)"]
    all_mode_results = []

    for mode in range(4):
        fx = (mode == 1 or mode == 3)
        fy = (mode == 2 or mode == 3)
        Wf = np.empty_like(W)
        Wf[:, 0] = -W[:, 0] if fx else W[:, 0]
        Wf[:, 1] = -W[:, 1] if fy else W[:, 1]

        print(f"  {mode_names[mode]}: 枚举 {N}×{M_max} = {N*M_max} 对...")
        transforms = compute_all_transforms(U, Wf, s0)
        all_mode_results.append(transforms)
        if transforms:
            n_arr = np.array([t[6] for t in transforms])
            print(f"    有效变换: {len(transforms)}, n_in_range: max={n_arr.max()} med={np.median(n_arr):.0f} mean={n_arr.mean():.1f}")

    # 8. 可视化
    fig, axes = plt.subplots(4, 3, figsize=(20, 22))
    fig.suptitle(f"{base}\n"
                 f"Solver: mode={solver_mode} θ={solver_theta:.2f}° RMS={solver_rms:.3f}px "
                 f"lnK={solver_lnK:.1f} matched={solver_matched} | U={N}({u_type}) W={M_max} s0={s0:.3f}\"",
                 fontsize=11, fontweight='bold')

    for mode in range(4):
        transforms = all_mode_results[mode]
        ax_θs, ax_θtx, ax_txty = axes[mode]

        if not transforms:
            for ax in [ax_θs, ax_θtx, ax_txty]:
                ax.text(0.5, 0.5, "无有效变换", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{mode_names[mode]} (无数据)", fontsize=9)
            continue

        s_arr = np.array([t[2] for t in transforms])
        θ_arr = np.array([t[3] for t in transforms])
        tx_arr = np.array([t[4] for t in transforms])
        ty_arr = np.array([t[5] for t in transforms])
        n_arr = np.array([t[6] for t in transforms], dtype=float)

        # 颜色: n_in_range (log scale, +1 避免 log(0))
        colors = np.log10(n_arr + 1)

        # 标记: n_in_range > 5 的点用大圆, 其他用小点
        high = n_arr > 5
        low = ~high

        # Col 1: (θ, s)
        ax = ax_θs
        if np.any(low):
            ax.scatter(θ_arr[low], s_arr[low], c='lightgrey', s=2, alpha=0.3, zorder=1)
        if np.any(high):
            sc = ax.scatter(θ_arr[high], s_arr[high], c=colors[high], cmap='hot', s=20+n_arr[high]*2,
                           alpha=0.7, edgecolors='red', linewidths=0.3, zorder=2)
        ax.set_xlabel('θ (°)')
        ax.set_ylabel('s')
        ax.set_title(f"{mode_names[mode]}  (n={len(transforms)}, max_inliers={int(n_arr.max())})", fontsize=9)
        if mode == solver_mode:
            ax.axvline(x=solver_theta, color='lime', linewidth=1.5, linestyle='--', label=f'solver θ={solver_theta:.1f}°')
            ax.legend(fontsize=7)

        # Col 2: (θ, tx)
        ax = ax_θtx
        if np.any(low):
            ax.scatter(θ_arr[low], tx_arr[low], c='lightgrey', s=2, alpha=0.3, zorder=1)
        if np.any(high):
            ax.scatter(θ_arr[high], tx_arr[high], c=colors[high], cmap='hot', s=20+n_arr[high]*2,
                      alpha=0.7, edgecolors='red', linewidths=0.3, zorder=2)
        ax.set_xlabel('θ (°)')
        ax.set_ylabel('tx (")')
        if mode == solver_mode:
            ax.axvline(x=solver_theta, color='lime', linewidth=1.5, linestyle='--')

        # Col 3: (tx, ty)
        ax = ax_txty
        if np.any(low):
            ax.scatter(tx_arr[low], ty_arr[low], c='lightgrey', s=2, alpha=0.3, zorder=1)
        if np.any(high):
            ax.scatter(tx_arr[high], ty_arr[high], c=colors[high], cmap='hot', s=20+n_arr[high]*2,
                      alpha=0.7, edgecolors='red', linewidths=0.3, zorder=2)
        ax.set_xlabel('tx (")')
        ax.set_ylabel('ty (")')
        if mode == solver_mode:
            ax.axvline(x=0, color='grey', linewidth=0.5, alpha=0.5)
            ax.axhline(y=0, color='grey', linewidth=0.5, alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(output_dir, os.path.splitext(base)[0] + "_paramspace.png")
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  → 保存: {output_path}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, os.path.splitext(base)[0] + "_transforms.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("mode,i,j,s,theta_deg,tx,ty,n_in_range\n")
        for mode in range(4):
            for (i, j, s, θ, tx, ty, nr) in all_mode_results[mode]:
                f.write(f"{mode},{i},{j},{s:.6f},{θ:.4f},{tx:.2f},{ty:.2f},{nr}\n")
    print(f"  → CSV: {csv_path}")

    return {
        "filename": base,
        "solver_mode": solver_mode,
        "solver_theta": solver_theta,
        "solver_rms": solver_rms,
        "solver_lnK": solver_lnK,
        "solver_matched": solver_matched,
        "n_u": N,
        "n_w": M_max,
        "u_type": u_type,
    }


def main():
    parser = argparse.ArgumentParser(description="变换矩阵参数空间分布实验")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="指定帧文件名 (不含路径). 默认选 1成功+1失败")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== 变换矩阵参数空间分布实验 ===\n")

    # 选择帧
    if args.frames:
        frames = args.frames
    else:
        # 默认: 1 成功帧 + 1 失败帧
        frames = [
            "Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",  # Type3 失败
            "NGC4945_FD_T3_flying_dutchman-20250206@043838-600S-Lum.fts",             # Type1 失败
        ]
        # 尝试找一个成功帧
        if os.path.exists(FULL_TEST_JSON):
            with open(FULL_TEST_JSON, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            for r in all_results:
                if r.get("status") == "success" and r.get("bayes_lnK", 0) > 100 and r.get("rms_px", 99) < 1.0:
                    fn = r.get("filename", "")
                    if fn and find_fits_path(fn):
                        frames.insert(0, fn)  # 成功帧放第一个
                        break
        print(f"默认选择 {len(frames)} 帧: {frames}")

    # 初始化
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(gaia_client=gaia_client, star_detector=star_detector)

    summaries = []
    for fn in frames:
        fits_path = find_fits_path(fn)
        if not fits_path:
            print(f"跳过 (文件未找到): {fn}")
            continue
        try:
            info = run_experiment(fits_path, gaia_client, star_detector, solver, OUTPUT_DIR)
            summaries.append(info)
        except Exception as e:
            import traceback
            print(f"错误: {e}")
            traceback.print_exc()

    solver.close()
    gaia_client.close()
    star_detector.close()

    # 保存汇总
    meta_path = os.path.join(OUTPUT_DIR, "experiment_summary.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n汇总: {meta_path}")


if __name__ == "__main__":
    main()

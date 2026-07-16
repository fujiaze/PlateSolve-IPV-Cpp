"""
功能: 黄金池点对连线直方图分析 — 通过成对距离分布估计尺度因子
用途: 构建高信噪比黄金池(饱和星+亮星)，对池内所有点两两连线，绘制不同长度的连线数量直方图，
      利用双侧直方图(图像侧 vs 星表侧)在尺度轴上的缩放关系，通过扫描s值估计尺度因子。
      注: 成对距离分布对旋转和翻转不变，无需4模式扫描。
"""
import argparse
import csv
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "plate_solve" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "astro_image_io" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "star_detector" / "python"))

from vector_match_v2 import GaiaClientPy, gnomonic_forward
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

CSV_COLUMNS = [
    "frame_name", "optical_system", "target", "filter",
    "nsat", "n_total", "n_pairs", "M_catalog",
    "best_s", "best_corr", "s0", "s_deviation_pct",
]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_frame_metadata(path):
    path = Path(path)
    name = path.name
    stem = path.stem
    parts = re.split(r"[_\-]", stem)
    optical_system = "unknown"
    for part in list(path.parts) + parts:
        if re.fullmatch(r"T[1-4]", str(part), flags=re.IGNORECASE):
            optical_system = str(part).upper()
            break
    target = parts[0] if parts else stem
    filt = "unknown"
    if "-" in stem:
        filt = stem.rsplit("-", 1)[-1]
    elif parts:
        filt = parts[-1]
    return {
        "optical_system": optical_system,
        "target": target,
        "filter": filt,
        "frame_path": str(path),
        "frame_name": name,
    }


def discover_frames(lights_dir):
    lights_dir = Path(lights_dir)
    suffixes = {".fit", ".fits", ".fts"}
    frames = []
    for path in sorted(lights_dir.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        meta = parse_frame_metadata(path)
        if meta["optical_system"] in {"T1", "T2", "T3", "T4"}:
            frames.append(path)
    return frames


def limit_frames_by_group(frames, per_group_limit):
    if per_group_limit is None or per_group_limit <= 0:
        return list(frames)
    counts = defaultdict(int)
    selected = []
    for path in frames:
        meta = parse_frame_metadata(path)
        key = (meta["optical_system"], meta["filter"])
        if counts[key] >= per_group_limit:
            continue
        counts[key] += 1
        selected.append(path)
    return selected


def build_golden_pool(det, cx, cy, s0):
    sat = np.array(det.saturated, dtype=bool)
    x = np.array(det.x, dtype=np.float64)
    y = np.array(det.y, dtype=np.float64)
    flux = np.array(det.flux, dtype=np.float64)
    nsat = int(np.sum(sat))
    if nsat >= 50:
        sel_x = x[sat]
        sel_y = y[sat]
        sat_mask = np.ones(len(sel_x), dtype=bool)
    else:
        sel_x = x[sat]
        sel_y = y[sat]
        sat_mask_list = [True] * nsat
        non_sat = ~sat
        need = 100 - nsat if nsat > 0 else 100
        if np.any(non_sat):
            non_idx = np.where(non_sat)[0]
            top_k = min(need, len(non_idx))
            top_idx = non_idx[np.argsort(flux[non_idx])[-top_k:]]
            sel_x = np.concatenate([sel_x, x[top_idx]])
            sel_y = np.concatenate([sel_y, y[top_idx]])
            sat_mask_list.extend([False] * top_k)
        if nsat == 0:
            log("警告: 黄金池中无饱和星，使用前100高flux非饱和星")
        sat_mask = np.array(sat_mask_list, dtype=bool)
    n_total = len(sel_x)
    U = np.column_stack([(sel_x - cx) * s0, -(sel_y - cy) * s0])
    return U, nsat, n_total, sat_mask


def build_catalog_pool(ra_arr, dec_arr, mag_arr, cra, cdec, half_w, half_h, target_count):
    xi, eta, valid = gnomonic_forward(ra_arr, dec_arr, cra, cdec)
    mask = valid & (np.abs(xi) <= half_w) & (np.abs(eta) <= half_h)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.empty((0, 2), dtype=np.float64)
    target_count = min(target_count, len(idx))
    sorted_idx = idx[np.argsort(mag_arr[idx])[:target_count]]
    W = np.column_stack([xi[sorted_idx], eta[sorted_idx]])
    return W


def compute_pairwise_distances(points):
    return pdist(points, metric="euclidean")


def build_distance_histogram(distances, n_bins=None):
    if len(distances) < 2:
        return np.array([1]), np.array([np.min(distances), np.max(distances)])
    if n_bins is None:
        iqr = np.percentile(distances, 75) - np.percentile(distances, 25)
        if iqr > 0:
            bin_width = 2.0 * iqr / (len(distances) ** (1.0 / 3.0))
        else:
            bin_width = (np.max(distances) - np.min(distances)) / 50.0
        if bin_width > 0:
            n_bins = max(int((np.max(distances) - np.min(distances)) / bin_width), 10)
        else:
            n_bins = 10
    counts, bin_edges = np.histogram(distances, bins=n_bins)
    return counts, bin_edges


def scan_scale_factor(dist_U, dist_W, s_range=(0.5, 2.0), s_step=0.001):
    s_values = np.arange(s_range[0], s_range[1] + s_step * 0.5, s_step)
    correlations = np.zeros(len(s_values))
    all_min = min(np.min(dist_U), np.min(dist_W) * s_range[0])
    all_max = max(np.max(dist_U), np.max(dist_W) * s_range[1])
    global_bins = np.linspace(all_min, all_max, 100)
    hist_U, _ = np.histogram(dist_U, bins=global_bins, density=True)
    for i, s in enumerate(s_values):
        dist_W_scaled = dist_W * s
        hist_W, _ = np.histogram(dist_W_scaled, bins=global_bins, density=True)
        if np.std(hist_U) < 1e-12 or np.std(hist_W) < 1e-12:
            correlations[i] = 0.0
            continue
        corr = np.corrcoef(hist_U, hist_W)[0, 1]
        correlations[i] = max(corr, 0.0) if np.isfinite(corr) else 0.0
    best_idx = np.argmax(correlations)
    best_s = float(s_values[best_idx])
    best_corr = float(correlations[best_idx])
    return s_values, correlations, best_s, best_corr


def parse_angle_value(raw, is_ra):
    text = str(raw).strip().replace(" ", ":")
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    value = abs(float(parts[0])) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0
    if is_ra:
        value *= 15.0
    if str(parts[0]).startswith("-"):
        value = -value
    return value


def read_center_from_header(fits_path):
    from astropy.io import fits as pf
    hdr = pf.getheader(fits_path)
    ra_raw = hdr.get("RA", hdr.get("OBJRA", None))
    dec_raw = hdr.get("DEC", hdr.get("OBJDEC", None))
    if ra_raw is None or dec_raw is None:
        return None
    return parse_angle_value(ra_raw, True), parse_angle_value(dec_raw, False)


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_single_frame(fits_path, gaia_dir, output_dir):
    meta = parse_frame_metadata(fits_path)
    reader = ImageReader()
    img = reader.read(str(fits_path))
    width = int(img.width)
    height = int(img.height)
    fl = float(img.metadata.observation.focallen)
    ps = float(img.metadata.observation.xpixsz)
    wcs = img.metadata.wcs
    if wcs and wcs.crval1 is not None and wcs.crval2 is not None:
        cra = float(wcs.crval1)
        cdec = float(wcs.crval2)
    else:
        center = read_center_from_header(fits_path)
        if center is None:
            raise RuntimeError("缺少WCS或RA/DEC头信息")
        cra, cdec = center
    s0 = 206.265 * ps / fl
    cx = width / 2.0
    cy = height / 2.0
    half_w = width * 0.5 * s0
    half_h = height * 0.5 * s0
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    U, nsat, n_total, sat_mask = build_golden_pool(det, cx, cy, s0)
    if len(U) < 3:
        raise RuntimeError(f"黄金池点数不足: N={len(U)}")
    n_pairs = n_total * (n_total - 1) // 2
    query_radius_deg = math.hypot(width, height) * s0 * 0.5 / 3600.0
    m_cut = 6.0 + 1.5 * math.log10(fl) + 2.0 * math.log10(300.0)
    gaia = GaiaClientPy(str(gaia_dir), 1)
    try:
        ra_t, dec_t, mag_t = gaia.cone_search(cra, cdec, query_radius_deg, min(m_cut, 22.0))
    finally:
        gaia.close()
    W = build_catalog_pool(ra_t, dec_t, mag_t, cra, cdec, half_w, half_h, n_total)
    if len(W) < 3:
        raise RuntimeError(f"星表池点数不足: M={len(W)}")
    dist_U = compute_pairwise_distances(U)
    dist_W = compute_pairwise_distances(W)
    s_array, corr_array, best_s, best_corr = scan_scale_factor(dist_U, dist_W)
    log(f"  best_s={best_s:.4f}, corr={best_corr:.4f}, nsat={nsat}, N={n_total}, M={len(W)}")
    s_deviation_pct = (best_s - 1.0) * 100.0
    output_dir = Path(output_dir)
    hist_dir = output_dir / "histograms"
    scan_dir = output_dir / "scans"
    pool_dir = output_dir / "pools"
    for d in [hist_dir, scan_dir, pool_dir]:
        d.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(
        r"[^A-Za-z0-9_.-]+", "_",
        f"{meta['optical_system']}_{meta['target']}_{meta['filter']}_{Path(meta['frame_name']).stem}",
    )
    plot_histogram_overlay(
        hist_dir / f"{safe_name}_hist.png",
        dist_U, dist_W, best_s, meta["frame_name"],
    )
    plot_scale_scan(
        scan_dir / f"{safe_name}_scan.png",
        s_array, corr_array, best_s, best_corr, meta["frame_name"],
    )
    plot_golden_pool(
        pool_dir / f"{safe_name}_pool.png",
        U, sat_mask, meta["frame_name"],
    )
    return {
        **meta,
        "nsat": nsat,
        "n_total": n_total,
        "n_pairs": n_pairs,
        "M_catalog": len(W),
        "best_s": best_s,
        "best_corr": best_corr,
        "s0": s0,
        "s_deviation_pct": s_deviation_pct,
    }


def plot_histogram_overlay(output_path, dist_U, dist_W, best_s, frame_name):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1])
    all_min = min(np.min(dist_U), np.min(dist_W) * best_s)
    all_max = max(np.max(dist_U), np.max(dist_W) * best_s)
    bins = np.linspace(all_min, all_max, 80)
    ax = axes[0]
    counts_U, _, _ = ax.hist(dist_U, bins=bins, density=True, alpha=0.6, color="#268bd2",
                              label=f"图像侧距离 (N={len(dist_U)})")
    counts_W, _, _ = ax.hist(dist_W * best_s, bins=bins, density=True, alpha=0.4, color="#dc322f",
                              label=f"星表侧距离 x s={best_s:.4f} (M={len(dist_W)})")
    ax.set_xlabel("弧秒")
    ax.set_ylabel("密度")
    ax.set_title(f"{frame_name}\n双侧距离分布叠加 (s={best_s:.4f})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax = axes[1]
    counts_U_raw, bin_edges_U = np.histogram(dist_U, bins=bins)
    counts_W_raw, _ = np.histogram(dist_W * best_s, bins=bins)
    bin_centers = 0.5 * (bin_edges_U[:-1] + bin_edges_U[1:])
    ax.bar(bin_centers, counts_U_raw, width=np.diff(bin_edges_U), alpha=0.6, color="#268bd2",
           label=f"图像侧 (N={len(dist_U)})")
    ax.bar(bin_centers, counts_W_raw, width=np.diff(bin_edges_U), alpha=0.4, color="#dc322f",
           label=f"星表侧 x s={best_s:.4f} (M={len(dist_W)})")
    ax.set_xlabel("弧秒")
    ax.set_ylabel("数量")
    ax.set_title("双侧距离分布 — 计数")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scale_scan(output_path, s_array, corr_array, best_s, best_corr, frame_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s_array, corr_array, color="#268bd2", linewidth=1.0)
    ax.axvline(best_s, color="#dc322f", linestyle="--", linewidth=1.5,
               label=f"峰值 s={best_s:.4f} corr={best_corr:.4f}")
    ax.axvline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7, label="s=1.0 (理论)")
    ax.set_xlabel("尺度因子 s")
    ax.set_ylabel("Pearson相关系数")
    ax.set_title(f"{frame_name}\n尺度扫描")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_golden_pool(output_path, U, sat_mask, frame_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    ax = axes[0]
    sat_pts = U[sat_mask]
    non_pts = U[~sat_mask]
    if len(U) <= 200:
        segments = []
        for i in range(len(U)):
            for j in range(i + 1, len(U)):
                segments.append([(U[i, 0], U[i, 1]), (U[j, 0], U[j, 1])])
        lc = LineCollection(segments, colors="gray", alpha=0.05, linewidths=0.2)
        ax.add_collection(lc)
    if len(non_pts) > 0:
        ax.scatter(non_pts[:, 0], non_pts[:, 1], c="#268bd2", s=15, alpha=0.8,
                   zorder=3, label=f"非饱和星 ({len(non_pts)})")
    if len(sat_pts) > 0:
        ax.scatter(sat_pts[:, 0], sat_pts[:, 1], c="#dc322f", s=20, alpha=0.8,
                   zorder=3, label=f"饱和星 ({len(sat_pts)})")
    ax.set_xlabel("弧秒")
    ax.set_ylabel("弧秒")
    ax.set_title(f"{frame_name}\n黄金池星点分布")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.15)
    ax = axes[1]
    dist_U = compute_pairwise_distances(U)
    counts, bin_edges = build_distance_histogram(dist_U)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_widths = np.diff(bin_edges)
    ax.bar(bin_centers, counts, width=bin_widths, color="#268bd2", alpha=0.7)
    ax.set_xlabel("弧秒")
    ax.set_ylabel("连线数量")
    ax.set_title(f"图像侧黄金池距离直方图\nN={len(U)}, C(N,2)={len(dist_U)}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_batch_experiment(lights_dir, gaia_dir, output_dir, limit_per_group):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = discover_frames(lights_dir)
    frames = limit_frames_by_group(frames, limit_per_group)
    log(f"发现待处理帧: {len(frames)}")
    rows = []
    error_rows = []
    for idx, path in enumerate(frames, 1):
        meta = parse_frame_metadata(path)
        t0 = time.perf_counter()
        try:
            log(f"[{idx}/{len(frames)}] 处理 {meta['optical_system']} {meta['target']} {meta['filter']} {meta['frame_name']}")
            result = run_single_frame(path, gaia_dir, output_dir)
            rows.append(result)
            elapsed = time.perf_counter() - t0
            log(f"[{idx}/{len(frames)}] 完成 s={result['best_s']:.4f} corr={result['best_corr']:.4f} dev={result['s_deviation_pct']:.2f}% elapsed={elapsed:.1f}s")
        except Exception as exc:
            error_rows.append({**meta, "error": str(exc)})
            log(f"[{idx}/{len(frames)}] 跳过/错误: {exc}")
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    write_csv(csv_dir / "golden_pool_results.csv", CSV_COLUMNS, rows)
    write_csv(csv_dir / "errors.csv",
              ["optical_system", "target", "filter", "frame_path", "frame_name", "error"],
              error_rows)
    if rows:
        summary_path = output_dir / "summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("黄金池点对连线直方图分析摘要\n")
            f.write(f"处理帧数: {len(rows)} 错误帧数: {len(error_rows)}\n\n")
            by_sys = defaultdict(list)
            for row in rows:
                by_sys[row["optical_system"]].append(row)
            for sys_name in sorted(by_sys):
                sys_rows = by_sys[sys_name]
                s_devs = [abs(r["s_deviation_pct"]) for r in sys_rows]
                corrs = [r["best_corr"] for r in sys_rows]
                f.write(f"{sys_name}: {len(sys_rows)}帧, "
                        f"|dev|中位={np.median(s_devs):.2f}%, "
                        f"|dev|P95={np.percentile(s_devs, 95):.2f}%, "
                        f"corr中位={np.median(corrs):.4f}\n")
            f.write("\n逐帧结果:\n")
            for row in rows:
                f.write(f"  {row['frame_name']}: s={row['best_s']:.4f} "
                        f"corr={row['best_corr']:.4f} s0={row['s0']:.4f} "
                        f"dev={row['s_deviation_pct']:.2f}% "
                        f"nsat={row['nsat']} N={row['n_total']} M={row['M_catalog']}\n")
    log(f"输出完成: {output_dir}")
    log(f"错误帧: {len(error_rows)}")


def main():
    parser = argparse.ArgumentParser(description="黄金池点对连线直方图分析")
    parser.add_argument("--lights-dir", type=str,
                        default=str(PROJECT_ROOT / "testdata" / "lights"))
    parser.add_argument("--gaia-dir", type=str,
                        default=str(PROJECT_ROOT / "GaiaDR3"))
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "output" / "golden_pool"))
    parser.add_argument("--single", type=str, default=None, help="单帧FITS路径")
    parser.add_argument("--limit-per-group", type=int, default=3)
    args = parser.parse_args()
    if args.single:
        result = run_single_frame(args.single, args.gaia_dir, args.output_dir)
        log(f"单帧结果: s={result['best_s']:.4f}, corr={result['best_corr']:.4f}, dev={result['s_deviation_pct']:.2f}%")
    else:
        run_batch_experiment(args.lights_dir, args.gaia_dir, args.output_dir, args.limit_per_group)


if __name__ == "__main__":
    main()

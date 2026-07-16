"""
功能: 黄金池最近邻连线分析 — 每颗星只与最近邻连线，对比图像侧与星表侧的结构
用途: 构建黄金池后，对每颗星找到其最近邻并连线，分别输出图像侧和星表侧的
      连线投影图与统计数据，用于直观对比两侧空间结构是否一致
"""
import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


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


def build_nearest_neighbor_graph(points):
    tree = cKDTree(points)
    dists, indices = tree.query(points, k=2)
    nn_dist = dists[:, 1]
    nn_idx = indices[:, 1]
    edges = set()
    for i in range(len(points)):
        j = int(nn_idx[i])
        edge = (min(i, j), max(i, j))
        edges.add(edge)
    edge_list = sorted(edges)
    edge_lengths = np.array([nn_dist[i] for i in range(len(points))])
    return edge_list, edge_lengths, nn_idx, nn_dist


def segments_intersect(p1, p2, p3, p4):
    def cross2d(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross2d(p3, p4, p1)
    d2 = cross2d(p3, p4, p2)
    d3 = cross2d(p1, p2, p3)
    d4 = cross2d(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def count_crossings(points, edge_list):
    n_cross = 0
    n_edges = len(edge_list)
    seg_pts = []
    for i, j in edge_list:
        seg_pts.append((points[i], points[j]))
    for a in range(n_edges):
        for b in range(a + 1, n_edges):
            i1, j1 = edge_list[a]
            i2, j2 = edge_list[b]
            if i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2:
                continue
            if segments_intersect(seg_pts[a][0], seg_pts[a][1],
                                  seg_pts[b][0], seg_pts[b][1]):
                n_cross += 1
    return n_cross


def print_stats(label, points, edge_list, edge_lengths, nn_dist):
    n_stars = len(points)
    n_edges = len(edge_list)
    n_mutual = sum(1 for i, j in edge_list
                   if nn_dist[i] < 1e-12 or nn_dist[j] < 1e-12 or
                   (nn_dist[i] > 0 and
                    abs(np.linalg.norm(points[i] - points[j]) - nn_dist[i]) < 1e-9 and
                    abs(np.linalg.norm(points[i] - points[j]) - nn_dist[j]) < 1e-9))
    mutual_pairs = 0
    for i, j in edge_list:
        pi_norm = np.linalg.norm(points[i] - points[j])
        if abs(pi_norm - nn_dist[i]) < 1e-9 and abs(pi_norm - nn_dist[j]) < 1e-9:
            mutual_pairs += 1
    total_length = sum(np.linalg.norm(points[i] - points[j]) for i, j in edge_list)
    edge_len_arr = np.array([np.linalg.norm(points[i] - points[j]) for i, j in edge_list])
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  星点数:              {n_stars}")
    print(f"  连线数:              {n_edges}")
    print(f"  互为最近邻对数:      {mutual_pairs}")
    print(f"  单向最近邻数:        {n_edges - mutual_pairs}")
    print(f"  连线总长度(弧秒):    {total_length:.2f}")
    print(f"  连线平均长度(弧秒):  {np.mean(edge_len_arr):.4f}")
    print(f"  连线中位长度(弧秒):  {np.median(edge_len_arr):.4f}")
    print(f"  连线最短(弧秒):      {np.min(edge_len_arr):.4f}")
    print(f"  连线最长(弧秒):      {np.max(edge_len_arr):.4f}")
    print(f"  连线长度标准差:      {np.std(edge_len_arr):.4f}")
    print(f"  最近邻距离均值:      {np.mean(nn_dist):.4f}")
    print(f"  最近邻距离中位:      {np.median(nn_dist):.4f}")
    print(f"  最近邻距离P95:       {np.percentile(nn_dist, 95):.4f}")
    print(f"{'='*60}")


def plot_nn_graph(output_path, points, edge_list, sat_mask, label, color_edge, color_sat, color_non):
    fig, ax = plt.subplots(figsize=(10, 10))
    segments = [[(points[i, 0], points[i, 1]), (points[j, 0], points[j, 1])]
                for i, j in edge_list]
    lc = LineCollection(segments, colors=color_edge, alpha=0.5, linewidths=0.8)
    ax.add_collection(lc)
    sat_pts = points[sat_mask] if sat_mask is not None and np.any(sat_mask) else np.empty((0, 2))
    non_pts = points[~sat_mask] if sat_mask is not None else points
    if len(non_pts) > 0:
        ax.scatter(non_pts[:, 0], non_pts[:, 1], c=color_non, s=18, alpha=0.85,
                   zorder=3, label=f"非饱和星 ({len(non_pts)})")
    if len(sat_pts) > 0:
        ax.scatter(sat_pts[:, 0], sat_pts[:, 1], c=color_sat, s=25, alpha=0.9,
                   zorder=4, label=f"饱和星 ({len(sat_pts)})")
    ax.set_xlabel("弧秒")
    ax.set_ylabel("弧秒")
    ax.set_title(f"{label}\n最近邻连线图 (N={len(points)}, 边={len(edge_list)})")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_nn_histogram(output_path, edge_len_U, edge_len_W, frame_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    ax.hist(edge_len_U, bins=40, color="#268bd2", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("弧秒")
    ax.set_ylabel("数量")
    ax.set_title(f"图像侧 最近邻连线长度分布\n均值={np.mean(edge_len_U):.2f}\" 中位={np.median(edge_len_U):.2f}\"")
    ax.grid(alpha=0.2)
    ax = axes[1]
    ax.hist(edge_len_W, bins=40, color="#dc322f", alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("弧秒")
    ax.set_ylabel("数量")
    ax.set_title(f"星表侧 最近邻连线长度分布\n均值={np.mean(edge_len_W):.2f}\" 中位={np.median(edge_len_W):.2f}\"")
    ax.grid(alpha=0.2)
    fig.suptitle(frame_name, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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


def run(fits_path, gaia_dir, output_dir):
    fits_path = Path(fits_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    log(f"图像: {width}x{height}, fl={fl:.0f}mm, ps={ps:.1f}µm, s0={s0:.4f}\"/px")
    log(f"中心: RA={cra:.4f}°, Dec={cdec:.4f}°, FOV={2*half_w:.0f}\"x{2*half_h:.0f}\"")
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    log(f"检测到星点: {len(det.x)} (饱和={sum(det.saturated)})")
    U, nsat, n_total, sat_mask = build_golden_pool(det, cx, cy, s0)
    log(f"黄金池: nsat={nsat}, N={n_total}")
    query_radius_deg = math.hypot(width, height) * s0 * 0.5 / 3600.0
    m_cut = 6.0 + 1.5 * math.log10(fl) + 2.0 * math.log10(300.0)
    gaia = GaiaClientPy(str(gaia_dir), 1)
    try:
        ra_t, dec_t, mag_t = gaia.cone_search(cra, cdec, query_radius_deg, min(m_cut, 22.0))
    finally:
        gaia.close()
    W = build_catalog_pool(ra_t, dec_t, mag_t, cra, cdec, half_w, half_h, n_total)
    log(f"星表池: M={len(W)}")
    log("构建最近邻连线...")
    edge_list_U, edge_len_U, nn_idx_U, nn_dist_U = build_nearest_neighbor_graph(U)
    edge_list_W, edge_len_W, nn_idx_W, nn_dist_W = build_nearest_neighbor_graph(W)
    edge_len_U_arr = np.array([np.linalg.norm(U[i] - U[j]) for i, j in edge_list_U])
    edge_len_W_arr = np.array([np.linalg.norm(W[i] - W[j]) for i, j in edge_list_W])
    print_stats("图像侧 (Image)", U, edge_list_U, edge_len_U_arr, nn_dist_U)
    print_stats("星表侧 (Catalog)", W, edge_list_W, edge_len_W_arr, nn_dist_W)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", fits_path.stem)
    log("生成图像侧连线投影图...")
    plot_nn_graph(
        output_dir / f"{safe_name}_nn_image.png",
        U, edge_list_U, sat_mask,
        f"图像侧 — {fits_path.name}", "#268bd2", "#dc322f", "#268bd2",
    )
    log("生成星表侧连线投影图...")
    cat_sat_mask = np.zeros(len(W), dtype=bool)
    if nsat < len(W):
        cat_sat_mask[:nsat] = True
    else:
        cat_sat_mask[:] = True
    plot_nn_graph(
        output_dir / f"{safe_name}_nn_catalog.png",
        W, edge_list_W, cat_sat_mask,
        f"星表侧 — {fits_path.name}", "#dc322f", "#859900", "#dc322f",
    )
    log("生成最近邻连线长度直方图...")
    plot_nn_histogram(
        output_dir / f"{safe_name}_nn_hist.png",
        edge_len_U_arr, edge_len_W_arr, fits_path.name,
    )
    if n_total <= 200:
        log("检查交叉线 (N<=200)...")
        n_cross_U = count_crossings(U, edge_list_U)
        n_cross_W = count_crossings(W, edge_list_W)
        print(f"  图像侧交叉线数: {n_cross_U} / {len(edge_list_U)} ({n_cross_U/len(edge_list_U)*100:.1f}%)")
        print(f"  星表侧交叉线数: {n_cross_W} / {len(edge_list_W)} ({n_cross_W/len(edge_list_W)*100:.1f}%)")
    else:
        log(f"跳过交叉检查 (N={n_total}>200)")
    ratio = np.median(edge_len_U_arr) / np.median(edge_len_W_arr) if np.median(edge_len_W_arr) > 0 else 0
    print(f"\n  中位长度比(图像/星表): {ratio:.4f}  (理论s=1.0时比值应≈1.0)")
    log(f"完成! 输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="黄金池最近邻连线分析")
    parser.add_argument("--fits", type=str, required=True, help="FITS文件路径")
    parser.add_argument("--gaia-dir", type=str, default=str(PROJECT_ROOT / "GaiaDR3"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "output" / "nn_graph"))
    args = parser.parse_args()
    run(args.fits, args.gaia_dir, args.output_dir)


if __name__ == "__main__":
    main()

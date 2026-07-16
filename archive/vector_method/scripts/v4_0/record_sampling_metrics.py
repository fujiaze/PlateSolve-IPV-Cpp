"""
功能: 批量记录V4.0四模式抽样指标、统计表和最佳覆盖图
用途: 验证density_corr、coverage_iou、score_dc_ci在T1/T2/T3/T4全通道帧上的峰噪分离能力
"""
import argparse
import csv
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "plate_solve" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "astro_image_io" / "python"))
sys.path.insert(0, str(PROJECT_ROOT / "lib" / "star_detector" / "python"))

from vector_match_v2 import GaiaClientPy, gnomonic_forward, _apply_flip
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

METRICS = ["density_corr", "coverage_iou", "score_dc_ci"]

SAMPLE_DETAIL_COLUMNS = [
    "optical_system", "target", "filter", "frame_path", "frame_name",
    "nsat", "N_image", "M_catalog", "halfW", "halfH",
    "mode", "sample_index", "ui", "wi", "u_norm", "w_norm",
    "s", "theta_rad", "theta_deg",
    "sample_status", "reject_reason", "is_finite_transform", "is_in_scale_window",
    "density_corr", "coverage_iou", "score_dc_ci", "wt_clip_count",
    "peak_theta_deg", "peak_s", "peak_count", "is_near_peak",
]

MODE_STATS_COLUMNS = [
    "optical_system", "target", "filter", "frame_path", "frame_name", "mode",
    "attempted_samples", "measured_samples", "in_scale_samples",
    "peak_theta_deg", "peak_s", "peak_count",
    "density_corr_mean", "density_corr_median", "density_corr_std", "density_corr_max",
    "density_corr_p50", "density_corr_p90", "density_corr_p95", "density_corr_p99", "density_corr_p999",
    "coverage_iou_mean", "coverage_iou_median", "coverage_iou_std", "coverage_iou_max",
    "coverage_iou_p50", "coverage_iou_p90", "coverage_iou_p95", "coverage_iou_p99", "coverage_iou_p999",
    "score_dc_ci_mean", "score_dc_ci_median", "score_dc_ci_std", "score_dc_ci_max",
    "score_dc_ci_p50", "score_dc_ci_p90", "score_dc_ci_p95", "score_dc_ci_p99", "score_dc_ci_p999",
    "density_corr_peak_noise_ratio", "density_corr_peak_minus_noise_median", "density_corr_peak_zscore",
    "coverage_iou_peak_noise_ratio", "coverage_iou_peak_minus_noise_median", "coverage_iou_peak_zscore",
    "score_dc_ci_peak_noise_ratio", "score_dc_ci_peak_minus_noise_median", "score_dc_ci_peak_zscore",
]

BEST_COLUMNS = [
    "optical_system", "target", "filter", "frame_path", "frame_name", "metric",
    "mode", "sample_index", "ui", "wi", "s", "theta_rad", "theta_deg",
    "density_corr", "coverage_iou", "score_dc_ci", "wt_clip_count",
    "peak_theta_deg", "peak_s", "peak_count", "is_near_peak",
    "attempted_samples", "measured_samples",
]

GLOBAL_COLUMNS = [
    "optical_system", "target", "filter", "metric", "frames", "best_max", "best_median",
    "best_p90", "best_p95", "weak_separation_frames", "mode_distribution",
]

ANALYSIS_COLUMNS = [
    "metric", "scope", "key", "count", "max", "median", "p90", "p95", "p99",
]

FRAME_ISSUE_COLUMNS = [
    "optical_system", "target", "filter", "frame_name", "issue_type", "detail",
]


class SequenceRng:
    def __init__(self, pairs):
        self.pairs = list(pairs)
        self.index = 0
        self.current = None

    def randrange(self, stop):
        if self.current is None:
            self.current = self.pairs[self.index % len(self.pairs)]
            self.index += 1
            value = self.current[0]
        else:
            value = self.current[1]
            self.current = None
        return int(value) % int(stop)


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


def gaia_query_radius_deg(width_px, height_px, arcsec_per_px):
    return math.hypot(width_px, height_px) * arcsec_per_px * 0.5 / 3600.0


def circular_catalog_mask(xi, eta, valid, fov_diag):
    radius = fov_diag * 0.5
    return valid & ((xi * xi + eta * eta) <= radius * radius)


def finite_value(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def transform_points(Wf, s, theta):
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.column_stack([
        s * (cos_t * Wf[:, 0] - sin_t * Wf[:, 1]),
        s * (sin_t * Wf[:, 0] + cos_t * Wf[:, 1]),
    ])


def clip_to_fov(points, halfW, halfH):
    mask = (np.abs(points[:, 0]) < halfW) & (np.abs(points[:, 1]) < halfH)
    return points[mask]


def density_corr(U, Wt_clip, halfW, halfH, grid_n=16):
    if len(U) < 3 or len(Wt_clip) < 3:
        return 0.0
    bx = np.linspace(-halfW, halfW, grid_n + 1)
    by = np.linspace(-halfH, halfH, grid_n + 1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bx, by])
    h_w, _, _ = np.histogram2d(Wt_clip[:, 0], Wt_clip[:, 1], bins=[bx, by])
    corr = np.corrcoef(h_u.ravel(), h_w.ravel())[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def coverage_iou(U, Wt_clip, halfW, halfH, grid_n=8):
    if len(U) < 3 or len(Wt_clip) < 3:
        return 0.0
    bx = np.linspace(-halfW, halfW, grid_n + 1)
    by = np.linspace(-halfH, halfH, grid_n + 1)
    h_u, _, _ = np.histogram2d(U[:, 0], U[:, 1], bins=[bx, by])
    h_w, _, _ = np.histogram2d(Wt_clip[:, 0], Wt_clip[:, 1], bins=[bx, by])
    h_u = (h_u > 0).astype(float)
    h_w = (h_w > 0).astype(float)
    inter = float(np.sum(h_u * h_w))
    union = float(np.sum(np.maximum(h_u, h_w)))
    return inter / max(union, 1.0)


def angle_dist_rad(a, b):
    d = (a - b) % (2.0 * np.pi)
    return np.minimum(d, 2.0 * np.pi - d)


def empty_record(mode, sample_index, ui, wi, u_norm, w_norm):
    return {
        "mode": mode,
        "sample_index": sample_index,
        "ui": ui,
        "wi": wi,
        "u_norm": u_norm,
        "w_norm": w_norm,
        "s": "",
        "theta_rad": "",
        "theta_deg": "",
        "sample_status": "",
        "reject_reason": "",
        "is_finite_transform": False,
        "is_in_scale_window": False,
        "density_corr": "",
        "coverage_iou": "",
        "score_dc_ci": "",
        "wt_clip_count": "",
        "peak_theta_deg": "",
        "peak_s": "",
        "peak_count": "",
        "is_near_peak": False,
    }


def sample_mode_records(U, Wf, mode, n_samples, halfW, halfH, rng=None, scale_window=(0.85, 1.15)):
    rng = rng if rng is not None else random.Random(12345 + int(mode))
    records = []
    N = len(U)
    M = len(Wf)
    for sample_index in range(int(n_samples)):
        ui = rng.randrange(N)
        wi = rng.randrange(M)
        u_vec = U[ui]
        w_vec = Wf[wi]
        u_norm = float(math.hypot(float(u_vec[0]), float(u_vec[1])))
        w_norm = float(math.hypot(float(w_vec[0]), float(w_vec[1])))
        record = empty_record(mode, sample_index, ui, wi, u_norm, w_norm)
        if w_norm < 1e-12:
            record["sample_status"] = "zero_w_norm"
            record["reject_reason"] = "zero_w_norm"
            records.append(record)
            continue
        s = u_norm / w_norm
        theta = math.atan2(float(u_vec[1]), float(u_vec[0])) - math.atan2(float(w_vec[1]), float(w_vec[0]))
        is_finite = math.isfinite(s) and math.isfinite(theta)
        record["s"] = float(s) if math.isfinite(s) else ""
        record["theta_rad"] = float(theta) if math.isfinite(theta) else ""
        record["theta_deg"] = float(math.degrees(theta)) if math.isfinite(theta) else ""
        record["is_finite_transform"] = bool(is_finite)
        record["is_in_scale_window"] = bool(scale_window[0] <= s <= scale_window[1]) if is_finite else False
        if not is_finite:
            record["sample_status"] = "non_finite"
            record["reject_reason"] = "non_finite"
            records.append(record)
            continue
        Wt = transform_points(Wf, s, theta)
        Wt_clip = clip_to_fov(Wt, halfW, halfH)
        dc = density_corr(U, Wt_clip, halfW, halfH)
        ci = coverage_iou(U, Wt_clip, halfW, halfH)
        score = dc * max(ci, 0.05)
        if not all(finite_value(v) for v in [dc, ci, score]):
            record["sample_status"] = "non_finite"
            record["reject_reason"] = "non_finite"
            records.append(record)
            continue
        record["sample_status"] = "measured"
        record["reject_reason"] = ""
        record["density_corr"] = float(dc)
        record["coverage_iou"] = float(ci)
        record["score_dc_ci"] = float(score)
        record["wt_clip_count"] = int(len(Wt_clip))
        records.append(record)
    summary = annotate_peak(records)
    return records, summary


def annotate_peak(records):
    measured = [r for r in records if r["sample_status"] == "measured"]
    attempted = len(records)
    in_scale = sum(1 for r in measured if r["is_in_scale_window"])
    summary = {
        "attempted_samples": attempted,
        "measured_samples": len(measured),
        "in_scale_samples": in_scale,
        "peak_theta_deg": "",
        "peak_s": "",
        "peak_count": 0,
    }
    if not measured:
        return summary
    thetas = np.array([float(r["theta_rad"]) for r in measured], dtype=np.float64)
    scales = np.array([float(r["s"]) for r in measured], dtype=np.float64)
    bins = np.linspace(-math.pi, math.pi, 73)
    hist, edges = np.histogram(((thetas + math.pi) % (2.0 * math.pi)) - math.pi, bins=bins)
    peak_bin = int(np.argmax(hist))
    peak_theta = float(0.5 * (edges[peak_bin] + edges[peak_bin + 1]))
    distances = angle_dist_rad(thetas, peak_theta)
    near_theta = distances < math.radians(5.0)
    peak_s = float(np.median(scales[near_theta])) if np.any(near_theta) else float(np.median(scales))
    near_peak = near_theta & (np.abs(scales - peak_s) / max(abs(peak_s), 1e-12) < 0.03)
    peak_count = int(np.sum(near_peak))
    for i, record in enumerate(measured):
        record["peak_theta_deg"] = float(math.degrees(peak_theta))
        record["peak_s"] = peak_s
        record["peak_count"] = peak_count
        record["is_near_peak"] = bool(near_peak[i])
    summary.update({
        "peak_theta_deg": float(math.degrees(peak_theta)),
        "peak_s": peak_s,
        "peak_count": peak_count,
    })
    return summary


def metric_stats(records, metric):
    values = [float(r[metric]) for r in records if r["sample_status"] == "measured" and finite_value(r[metric])]
    if not values:
        return {
            f"{metric}_mean": "", f"{metric}_median": "", f"{metric}_std": "", f"{metric}_max": "",
            f"{metric}_p50": "", f"{metric}_p90": "", f"{metric}_p95": "", f"{metric}_p99": "", f"{metric}_p999": "",
        }
    arr = np.array(values, dtype=np.float64)
    return {
        f"{metric}_mean": float(np.mean(arr)),
        f"{metric}_median": float(np.median(arr)),
        f"{metric}_std": float(np.std(arr)),
        f"{metric}_max": float(np.max(arr)),
        f"{metric}_p50": float(np.percentile(arr, 50)),
        f"{metric}_p90": float(np.percentile(arr, 90)),
        f"{metric}_p95": float(np.percentile(arr, 95)),
        f"{metric}_p99": float(np.percentile(arr, 99)),
        f"{metric}_p999": float(np.percentile(arr, 99.9)),
    }


def separation_stats(records, metric):
    peak = [float(r[metric]) for r in records if r["sample_status"] == "measured" and r.get("is_near_peak") and finite_value(r[metric])]
    noise = [float(r[metric]) for r in records if r["sample_status"] == "measured" and not r.get("is_near_peak") and finite_value(r[metric])]
    if not peak or not noise:
        return {
            f"{metric}_peak_noise_ratio": "",
            f"{metric}_peak_minus_noise_median": "",
            f"{metric}_peak_zscore": "",
        }
    peak_med = float(np.median(peak))
    noise_med = float(np.median(noise))
    noise_std = float(np.std(noise))
    return {
        f"{metric}_peak_noise_ratio": peak_med / max(abs(noise_med), 1e-9),
        f"{metric}_peak_minus_noise_median": peak_med - noise_med,
        f"{metric}_peak_zscore": (peak_med - noise_med) / max(noise_std, 1e-9),
    }


def mode_stats_row(meta, mode, records, summary):
    row = {key: "" for key in MODE_STATS_COLUMNS}
    row.update(meta)
    row.update({"mode": mode})
    row.update(summary)
    for metric in METRICS:
        row.update(metric_stats(records, metric))
        row.update(separation_stats(records, metric))
    return row


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_image_points(det, cx, cy, s0):
    sat = np.array(det.saturated, dtype=bool)
    x = np.array(det.x, dtype=np.float64)
    y = np.array(det.y, dtype=np.float64)
    flux = np.array(det.flux, dtype=np.float64)
    nsat = int(np.sum(sat))
    if nsat >= 50:
        sel_x = x[sat]
        sel_y = y[sat]
    else:
        sel_x = x[sat]
        sel_y = y[sat]
        non_sat = ~sat
        if np.any(non_sat):
            non_idx = np.where(non_sat)[0]
            top_k = min(50 - nsat, len(non_idx))
            top_idx = non_idx[np.argsort(flux[non_idx])[-top_k:]]
            sel_x = np.concatenate([sel_x, x[top_idx]])
            sel_y = np.concatenate([sel_y, y[top_idx]])
    U = np.column_stack([(sel_x - cx) * s0, -(sel_y - cy) * s0])
    return U, nsat


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


def build_frame_points(fits_path, gaia_dir):
    reader = ImageReader()
    img = reader.read(str(fits_path))
    width = int(img.width)
    height = int(img.height)
    fl = float(img.metadata.observation.focallen)
    ps = float(img.metadata.observation.xpixsz)
    wcs = img.metadata.wcs
    if wcs and wcs.crval1 is not None and wcs.crval2 is not None:
        cra0 = float(wcs.crval1)
        cdec0 = float(wcs.crval2)
    else:
        center = read_center_from_header(fits_path)
        if center is None:
            raise RuntimeError("缺少WCS或RA/DEC头信息")
        cra0, cdec0 = center
    s0 = 206.265 * ps / fl
    cx = width / 2.0
    cy = height / 2.0
    halfW = width * 0.5 * s0
    halfH = height * 0.5 * s0
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    U, nsat = select_image_points(det, cx, cy, s0)
    if len(U) < 5:
        raise RuntimeError(f"图像侧点数不足: N={len(U)}")
    fov_diag = math.hypot(width, height) * s0
    fov_area = (width * s0) * (height * s0)
    star_density = len(U) / max(fov_area, 1e-9)
    query_radius = gaia_query_radius_deg(width, height, s0)
    m_cut = 6.0 + 1.5 * math.log10(fl) + 2.0 * math.log10(300.0)
    gaia = GaiaClientPy(str(gaia_dir), 1)
    try:
        ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, query_radius, min(m_cut, 22.0))
    finally:
        gaia.close()
    xi_all, eta_all, valid = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
    mask = circular_catalog_mask(xi_all, eta_all, valid, fov_diag)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise RuntimeError("星表圆形搜索范围内无候选星")
    search_area = math.pi * (fov_diag * 0.5) ** 2
    target_count = max(int(star_density * search_area * 1.5), len(U))
    target_count = min(target_count, len(idx))
    sorted_idx = idx[np.argsort(mag_t[idx])[:target_count]]
    W = np.column_stack([xi_all[sorted_idx], eta_all[sorted_idx]])
    return {
        "U": U,
        "W": W,
        "nsat": nsat,
        "N_image": len(U),
        "M_catalog": len(W),
        "halfW": halfW,
        "halfH": halfH,
        "width": width,
        "height": height,
        "s0": s0,
    }


def enrich_records(records, meta, frame_data):
    common = {
        **meta,
        "nsat": frame_data["nsat"],
        "N_image": frame_data["N_image"],
        "M_catalog": frame_data["M_catalog"],
        "halfW": frame_data["halfW"],
        "halfH": frame_data["halfH"],
    }
    for record in records:
        record.update(common)
    return records


def choose_best_records(records_by_mode, summaries_by_mode):
    all_measured = []
    summary_lookup = {}
    for mode, records in records_by_mode.items():
        summary_lookup[mode] = summaries_by_mode[mode]
        all_measured.extend([r for r in records if r["sample_status"] == "measured"])
    best_rows = []
    for metric in METRICS:
        candidates = [r for r in all_measured if finite_value(r.get(metric, ""))]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: float(r[metric]))
        mode = int(best["mode"])
        summary = summary_lookup[mode]
        row = {key: "" for key in BEST_COLUMNS}
        row.update(best)
        row.update({
            "metric": metric,
            "attempted_samples": summary["attempted_samples"],
            "measured_samples": summary["measured_samples"],
        })
        best_rows.append(row)
    return best_rows


def plot_best_overlay(output_dir, meta, frame_data, best_rows):
    out_dir = Path(output_dir) / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    U = frame_data["U"]
    W = frame_data["W"]
    halfW = frame_data["halfW"]
    halfH = frame_data["halfH"]
    origin = np.array([0.0, 0.0])
    for row in best_rows:
        mode = int(row["mode"])
        Wf = _apply_flip(W, mode)
        s = float(row["s"])
        theta = float(row["theta_rad"])
        Wt = transform_points(Wf, s, theta)
        Wt_clip = clip_to_fov(Wt, halfW, halfH)
        ui = int(row["ui"])
        wi = int(row["wi"])
        metric = row["metric"]
        fig, ax = plt.subplots(figsize=(10, 9))
        for i in range(len(U)):
            ax.plot([0, U[i, 0]], [0, U[i, 1]], color="#2aa198", alpha=0.55, linewidth=0.8)
        ax.scatter(U[:, 0], U[:, 1], c="#2aa198", s=10, alpha=0.7, zorder=3)
        for j in range(len(Wt_clip)):
            ax.plot([0, Wt_clip[j, 0]], [0, Wt_clip[j, 1]], color="#dc322f", alpha=0.35, linewidth=0.6)
        ax.scatter(Wt_clip[:, 0], Wt_clip[:, 1], c="#dc322f", s=6, alpha=0.45, zorder=3)
        ax.plot([0, U[ui, 0]], [0, U[ui, 1]], color="#268bd2", linewidth=2.5, alpha=0.95, zorder=5, label=f"U[{ui}] best image ctrl")
        ax.scatter([U[ui, 0]], [U[ui, 1]], c="#268bd2", s=70, marker="o", zorder=6)
        ax.plot([0, Wt[wi, 0]], [0, Wt[wi, 1]], color="#b58900", linewidth=2.5, alpha=0.95, zorder=5, label=f"Wt[{wi}] best catalog ctrl")
        ax.scatter([Wt[wi, 0]], [Wt[wi, 1]], c="#b58900", s=80, marker="x", linewidths=2, zorder=6)
        ax.scatter([0], [0], c="black", s=30, marker="+", linewidths=1.5, zorder=7, label="center")
        rect_x = [-halfW, halfW, halfW, -halfW, -halfW]
        rect_y = [-halfH, -halfH, halfH, halfH, -halfH]
        ax.plot(rect_x, rect_y, "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlim(-halfW * 1.12, halfW * 1.12)
        ax.set_ylim(-halfH * 1.12, halfH * 1.12)
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(fontsize=8, loc="upper right")
        title = f"{meta['optical_system']} {meta['target']} {meta['filter']} {metric}"
        info = (
            f"{meta['frame_name']}\n"
            f"mode={mode} ui={ui} wi={wi} s={s:.6g} theta={float(row['theta_deg']):.3f} deg\n"
            f"dc={float(row['density_corr']):.4f} ci={float(row['coverage_iou']):.4f} score={float(row['score_dc_ci']):.4f} wt_clip={len(Wt_clip)}\n"
            f"nsat={frame_data['nsat']} N={frame_data['N_image']} M={frame_data['M_catalog']}\n"
            f"samples={row['attempted_samples']} measured={row['measured_samples']}\n"
            f"peak_theta={row['peak_theta_deg']} peak_s={row['peak_s']} peak_count={row['peak_count']}"
        )
        ax.set_title(title, fontsize=11)
        ax.text(0.01, 0.01, info, transform=ax.transAxes, fontsize=7.5, va="bottom", ha="left",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2})
        ax.set_xlabel("arcsec")
        ax.set_ylabel("arcsec")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{meta['optical_system']}_{meta['target']}_{meta['filter']}_{Path(meta['frame_name']).stem}_{metric}")
        fig.savefig(out_dir / f"{safe}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def process_frame(fits_path, gaia_dir, n_samples, output_dir, make_plots=True):
    meta = parse_frame_metadata(fits_path)
    frame_data = build_frame_points(fits_path, gaia_dir)
    all_detail = []
    mode_rows = []
    records_by_mode = {}
    summaries_by_mode = {}
    for mode in range(4):
        Wf = _apply_flip(frame_data["W"], mode)
        records, summary = sample_mode_records(frame_data["U"], Wf, mode, n_samples, frame_data["halfW"], frame_data["halfH"])
        enrich_records(records, meta, frame_data)
        all_detail.extend(records)
        records_by_mode[mode] = records
        summaries_by_mode[mode] = summary
        mode_rows.append(mode_stats_row(meta, mode, records, summary))
    best_rows = choose_best_records(records_by_mode, summaries_by_mode)
    if make_plots and best_rows:
        plot_best_overlay(output_dir, meta, frame_data, best_rows)
    return all_detail, mode_rows, best_rows, []


def aggregate_global(best_rows, mode_rows):
    rows = []
    by_group_metric = defaultdict(list)
    weak = defaultdict(int)
    modes = defaultdict(Counter)
    frame_keys = defaultdict(set)
    for row in best_rows:
        key = (row["optical_system"], row["target"], row["filter"], row["metric"])
        by_group_metric[key].append(float(row[row["metric"]]))
        modes[key][str(row["mode"])] += 1
        frame_keys[key].add(row["frame_name"])
    for row in mode_rows:
        for metric in METRICS:
            sep_key = f"{metric}_peak_zscore"
            if finite_value(row.get(sep_key, "")) and float(row[sep_key]) < 1.0:
                key = (row["optical_system"], row["target"], row["filter"], metric)
                weak[key] += 1
    for key, values in sorted(by_group_metric.items()):
        arr = np.array(values, dtype=np.float64)
        rows.append({
            "optical_system": key[0],
            "target": key[1],
            "filter": key[2],
            "metric": key[3],
            "frames": len(frame_keys[key]),
            "best_max": float(np.max(arr)),
            "best_median": float(np.median(arr)),
            "best_p90": float(np.percentile(arr, 90)),
            "best_p95": float(np.percentile(arr, 95)),
            "weak_separation_frames": weak[key],
            "mode_distribution": ";".join(f"{m}:{c}" for m, c in sorted(modes[key].items())),
        })
    return rows


def analyze_outputs(output_dir):
    output_dir = Path(output_dir)
    best_path = output_dir / "frame_best_metrics.csv"
    mode_path = output_dir / "frame_mode_stats.csv"
    best_rows = read_csv(best_path) if best_path.exists() else []
    mode_rows = read_csv(mode_path) if mode_path.exists() else []
    analysis_rows = []
    issue_rows = []
    for metric in METRICS:
        metric_rows = [r for r in best_rows if r.get("metric") == metric and finite_value(r.get(metric, ""))]
        scopes = [("global", [metric_rows])]
        by_opt = defaultdict(list)
        by_filter = defaultdict(list)
        by_target = defaultdict(list)
        for row in metric_rows:
            by_opt[row["optical_system"]].append(row)
            by_filter[row["filter"]].append(row)
            by_target[row["target"]].append(row)
        scopes.append(("optical_system", [[*v] for v in by_opt.values()]))
        scopes.append(("filter", [[*v] for v in by_filter.values()]))
        scopes.append(("target", [[*v] for v in by_target.values()]))
        for scope, groups in scopes:
            for group in groups:
                if not group:
                    continue
                values = np.array([float(r[metric]) for r in group], dtype=np.float64)
                key = "all"
                if scope == "optical_system":
                    key = group[0]["optical_system"]
                elif scope == "filter":
                    key = group[0]["filter"]
                elif scope == "target":
                    key = group[0]["target"]
                analysis_rows.append({
                    "metric": metric,
                    "scope": scope,
                    "key": key,
                    "count": len(values),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.percentile(values, 90)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                })
    best_by_frame = defaultdict(list)
    for row in best_rows:
        best_by_frame[(row["optical_system"], row["target"], row["filter"], row["frame_name"])].append(row)
    for key, rows in best_by_frame.items():
        modes = {row["mode"] for row in rows}
        if len(modes) > 1:
            issue_rows.append({"optical_system": key[0], "target": key[1], "filter": key[2], "frame_name": key[3], "issue_type": "mode_conflict", "detail": ",".join(sorted(modes))})
        for row in rows:
            value = float(row[row["metric"]]) if finite_value(row.get(row["metric"], "")) else 0.0
            if value < 0.05:
                issue_rows.append({"optical_system": key[0], "target": key[1], "filter": key[2], "frame_name": key[3], "issue_type": "low_score", "detail": f"{row['metric']}={value:.6g}"})
    for row in mode_rows:
        for metric in METRICS:
            sep = row.get(f"{metric}_peak_zscore", "")
            if sep != "" and finite_value(sep) and float(sep) < 1.0:
                issue_rows.append({"optical_system": row["optical_system"], "target": row["target"], "filter": row["filter"], "frame_name": row["frame_name"], "issue_type": "weak_separation", "detail": f"mode={row['mode']} {metric}_peak_zscore={float(sep):.6g}"})
    write_csv(output_dir / "analysis_summary.csv", ANALYSIS_COLUMNS, analysis_rows)
    write_csv(output_dir / "frame_issues.csv", FRAME_ISSUE_COLUMNS, issue_rows)
    summary_path = output_dir / "analysis_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("record_sampling_metrics 分析摘要\n")
        f.write(f"best_rows={len(best_rows)} mode_rows={len(mode_rows)} issues={len(issue_rows)}\n")
        for row in analysis_rows:
            if row["scope"] == "global":
                f.write(f"{row['metric']}: count={row['count']} max={row['max']:.6g} median={row['median']:.6g} p95={row['p95']:.6g}\n")
    return analysis_rows, issue_rows


def run_batch(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        analyze_outputs(output_dir)
        log(f"分析完成: {output_dir}")
        return 0
    frames = discover_frames(args.lights_dir)
    frames = limit_frames_by_group(frames, args.limit_per_group)
    if args.max_frames and args.max_frames > 0:
        frames = frames[:args.max_frames]
    log(f"发现待处理帧: {len(frames)}")
    detail_rows = []
    mode_rows = []
    best_rows = []
    error_rows = []
    for idx, path in enumerate(frames, 1):
        meta = parse_frame_metadata(path)
        t0 = time.perf_counter()
        try:
            log(f"[{idx}/{len(frames)}] 处理 {meta['optical_system']} {meta['target']} {meta['filter']} {meta['frame_name']}")
            details, modes, best, _ = process_frame(path, args.gaia_dir, args.samples, output_dir, not args.no_plots)
            detail_rows.extend(details)
            mode_rows.extend(modes)
            best_rows.extend(best)
            log(f"[{idx}/{len(frames)}] 完成 details={len(details)} modes={len(modes)} best={len(best)} elapsed={time.perf_counter() - t0:.1f}s")
        except Exception as exc:
            error = {**meta, "error": str(exc)}
            error_rows.append(error)
            log(f"[{idx}/{len(frames)}] 跳过/错误: {exc}")
    write_csv(output_dir / "sampling_details.csv", SAMPLE_DETAIL_COLUMNS, detail_rows)
    write_csv(output_dir / "frame_mode_stats.csv", MODE_STATS_COLUMNS, mode_rows)
    write_csv(output_dir / "frame_best_metrics.csv", BEST_COLUMNS, best_rows)
    write_csv(output_dir / "global_summary.csv", GLOBAL_COLUMNS, aggregate_global(best_rows, mode_rows))
    write_csv(output_dir / "errors.csv", ["optical_system", "target", "filter", "frame_path", "frame_name", "error"], error_rows)
    analyze_outputs(output_dir)
    log(f"输出完成: {output_dir}")
    log(f"错误帧: {len(error_rows)}")
    return 0


def build_arg_parser():
    parser = argparse.ArgumentParser(description="记录V4.0四模式全抽样指标并生成统计表与覆盖图")
    parser.add_argument("--lights-dir", default=str(PROJECT_ROOT / "testdata" / "lights"))
    parser.add_argument("--gaia-dir", default=str(PROJECT_ROOT / "GaiaDR3"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "record_sampling_metrics"))
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--limit-per-group", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())

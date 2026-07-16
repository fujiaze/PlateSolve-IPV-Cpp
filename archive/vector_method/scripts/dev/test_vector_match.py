"""
Vector Match 批量测试脚本

功能: 对testdata/lights/下所有FITS图像执行向量匹配plate solving
     16线程并发, 输出测试报告和每帧调试图像

用法: python test_vector_match.py
输出: output/vector_match_report.txt  报告
      output/debug/                    每帧调试图像(Gaia星标注)
"""

from __future__ import annotations

import os
import sys
import time
import glob
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 星点提取锁，确保同时只有一帧在执行星点检测
detect_lock = threading.Lock()

from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

import numpy as np
from vector_match_v2 import VectorMatch, gnomonic_forward, _apply_flip
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")
GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")

os.makedirs(DEBUG_DIR, exist_ok=True)


@dataclass
class TestResult:
    file_name: str
    panel: str
    filter_name: str
    exposure: str
    success: bool
    center_ra: float = 0.0
    center_dec: float = 0.0
    rotation_deg: float = 0.0
    scale_arcsec_px: float = 0.0
    flip_mode: int = -1
    matched_count: int = 0
    rms_px: float = 0.0
    rms_arcsec: float = 0.0
    n_detected: int = 0
    n_saturated: int = 0
    n_gaia: int = 0
    gaia_in_img: int = 0
    gaia_median_err_px: float = 0.0
    gaia_lt2px_pct: float = 0.0
    solve_time_s: float = 0.0
    detect_time_s: float = 0.0
    total_time_s: float = 0.0
    error_msg: str = ""


def parse_file_info(filename: str) -> tuple:
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("-")
    panel = ""
    filter_name = ""
    exposure = ""
    for p in parts:
        if p.startswith("mosaic"):
            panel = p
        if p.endswith("S"):
            exposure = p
        if p in ("Red", "Green", "Blue", "H-alpha", "Oiii"):
            filter_name = p
    return panel, filter_name, exposure


def process_single_file(file_path: str) -> TestResult:
    panel, filter_name, exposure = parse_file_info(file_path)
    file_name = os.path.basename(file_path)
    result = TestResult(
        file_name=file_name,
        panel=panel,
        filter_name=filter_name,
        exposure=exposure,
        success=False,
    )

    t_total_start = time.time()

    try:
        reader = ImageReader()
        img = reader.read(file_path)

        # 星点提取串行执行（加锁）
        with detect_lock:
            t_det_start = time.time()
            # 使用自适应fitRadius=0，C++内部自动估算FWHM
            params = SDetParamsPy(fitRadius=0)
            detector = StarDetector(params=params)
            det_result = detector.detect_ex(img.data)
            t_det_end = time.time()
        result.detect_time_s = t_det_end - t_det_start
        result.n_detected = det_result.count
        result.n_saturated = det_result.saturated_count

        if det_result.count < 2:
            result.error_msg = "星点不足"
            result.total_time_s = time.time() - t_total_start
            return result

        img_x = np.array(det_result.x, dtype=np.float64)
        img_y = np.array(det_result.y, dtype=np.float64)
        img_flux = np.array(det_result.flux, dtype=np.float64)
        img_sat = np.array(det_result.saturated, dtype=np.int32)

        center_ra = 0.0
        center_dec = 0.0
        focal_length = 200.0
        pixel_size = 6.0
        scale = 0.0

        if img.metadata.wcs and img.metadata.wcs.has_wcs:
            center_ra = img.metadata.wcs.crval1
            center_dec = img.metadata.wcs.crval2
            scale = img.metadata.wcs.pixel_scale

        if img.metadata.observation:
            if img.metadata.observation.focallen is not None:
                focal_length = img.metadata.observation.focallen
            if img.metadata.observation.xpixsz is not None:
                pixel_size = img.metadata.observation.xpixsz

        if center_ra == 0.0 and center_dec == 0.0:
            for kw in img.keywords:
                name = kw.name.upper()
                if name in ("OBJCTRA", "RA"):
                    val = kw.value
                    if isinstance(val, str):
                        parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                        if len(parts) >= 3:
                            center_ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
                        else:
                            parts2 = val.split()
                            if len(parts2) >= 3:
                                center_ra = (float(parts2[0]) + float(parts2[1]) / 60 + float(parts2[2]) / 3600) * 15
                    else:
                        center_ra = float(val)
                elif name in ("OBJCTDEC", "DEC"):
                    val = kw.value
                    if isinstance(val, str):
                        parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                        if len(parts) >= 3:
                            sign = -1 if parts[0].startswith("-") else 1
                            center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
                        else:
                            parts2 = val.split()
                            if len(parts2) >= 3:
                                sign = -1 if parts2[0].startswith("-") else 1
                                center_dec = sign * (abs(float(parts2[0])) + float(parts2[1]) / 60 + float(parts2[2]) / 3600)
                    else:
                        center_dec = float(val)

        if center_ra == 0.0 and center_dec == 0.0:
            result.error_msg = "无初始坐标"
            result.total_time_s = time.time() - t_total_start
            return result

        if scale > 0:
            s0 = scale
        elif focal_length > 0 and pixel_size > 0:
            s0 = 206.265 * pixel_size / focal_length
        else:
            result.error_msg = "无像素尺度"
            result.total_time_s = time.time() - t_total_start
            return result

        vm = VectorMatch(GAIA_DIR, db_type=2)

        t_solve_start = time.time()
        vm_result = vm.solve(
            img_x=img_x, img_y=img_y, img_flux=img_flux, img_saturated=img_sat,
            center_ra=center_ra, center_dec=center_dec,
            focal_length_mm=focal_length, pixel_size_um=pixel_size,
            width=img.width, height=img.height,
            scale_arcsec_px=s0,
        )
        t_solve_end = time.time()
        result.solve_time_s = t_solve_end - t_solve_start

        if vm_result is None:
            result.error_msg = "匹配失败"
            result.total_time_s = time.time() - t_total_start
            vm.close()
            return result

        result.success = True
        result.center_ra = vm_result.center_ra
        result.center_dec = vm_result.center_dec
        result.rotation_deg = vm_result.rotation_deg
        result.scale_arcsec_px = vm_result.scale_arcsec_px
        result.flip_mode = vm_result.flip_mode
        result.matched_count = vm_result.matched_count
        result.rms_px = vm_result.rms_px
        result.rms_arcsec = vm_result.rms_arcsec

        # WCS投影验证 + 调试图像
        a0, a1, a2, b0, b1, b2 = vm_result.affine
        import math
        fov_diag = math.sqrt(img.width ** 2 + img.height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        ra_arr, dec_arr, mag_arr = vm._gaia.cone_search(
            vm_result.center_ra, vm_result.center_dec, radius_deg, 22.0
        )
        result.n_gaia = len(ra_arr)

        sort_idx = np.argsort(mag_arr)
        n_top = min(1000, len(mag_arr))
        top_idx = sort_idx[:n_top]
        top_ra, top_dec = ra_arr[top_idx], dec_arr[top_idx]

        xi, eta, valid = gnomonic_forward(top_ra, top_dec, vm_result.center_ra, vm_result.center_dec)
        W = np.column_stack([xi[valid], eta[valid]])
        Wf = _apply_flip(W, vm_result.flip_mode)

        u = a0 + a1 * Wf[:, 0] + a2 * Wf[:, 1]
        v = b0 + b1 * Wf[:, 0] + b2 * Wf[:, 1]
        px = u / s0 + img.width / 2.0
        py = -v / s0 + img.height / 2.0

        in_img = (px >= 0) & (px < img.width) & (py >= 0) & (py < img.height)
        result.gaia_in_img = int(np.sum(in_img))

        if np.any(in_img):
            tree = cKDTree(np.column_stack([img_x, img_y]))
            dists, _ = tree.query(np.column_stack([px[in_img], py[in_img]]))
            result.gaia_median_err_px = float(np.median(dists))
            result.gaia_lt2px_pct = float(np.sum(dists < 2) / len(dists) * 100)

        # 生成调试图像
        try:
            from PIL import Image as PILImage
            data = img.data.astype(np.float64)
            p2, p98 = np.percentile(data, 2), np.percentile(data, 98)
            data_norm = np.clip((data - p2) / (p98 - p2), 0, 1)
            rgb = np.stack([data_norm, data_norm, data_norm], axis=-1)
            rgb = (rgb * 255).astype(np.uint8)

            cross_half = 8
            red = [255, 0, 0]
            px_in = px[in_img]
            py_in = py[in_img]
            for i in range(len(px_in)):
                ix = int(round(px_in[i]))
                iy = int(round(py_in[i]))
                for d in range(-cross_half, cross_half + 1):
                    if 0 <= ix + d < img.width:
                        rgb[iy, ix + d] = red
                    if 0 <= iy + d < img.height:
                        rgb[iy + d, ix] = red

            pil_img = PILImage.fromarray(rgb, "RGB")
            debug_name = os.path.splitext(file_name)[0] + "_debug.png"
            debug_path = os.path.join(DEBUG_DIR, debug_name)
            pil_img.save(debug_path)
        except Exception as e:
            pass

        vm.close()

    except Exception as e:
        result.error_msg = str(e)[:80]

    result.total_time_s = time.time() - t_total_start
    return result


def write_report(results: list, report_path: str):
    results.sort(key=lambda r: (r.panel, r.filter_name, r.exposure, r.file_name))

    success = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    lines = []
    lines.append("=" * 120)
    lines.append("Vector Match 批量测试报告")
    lines.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试图像: {len(results)} 帧")
    lines.append(f"成功: {len(success)}  失败: {len(fail)}")
    if success:
        lines.append(f"RMS(px)  中位数: {np.median([r.rms_px for r in success]):.3f}  "
                      f"均值: {np.mean([r.rms_px for r in success]):.3f}  "
                      f"最大: {np.max([r.rms_px for r in success]):.3f}  "
                      f"最小: {np.min([r.rms_px for r in success]):.3f}")
        lines.append(f"匹配数   中位数: {int(np.median([r.matched_count for r in success]))}  "
                      f"均值: {np.mean([r.matched_count for r in success]):.1f}")
        lines.append(f"投影误差  中位数: {np.median([r.gaia_median_err_px for r in success]):.2f}px  "
                      f"<2px占比: {np.mean([r.gaia_lt2px_pct for r in success]):.1f}%")
        lines.append(f"求解耗时 中位数: {np.median([r.solve_time_s for r in success]):.2f}s  "
                      f"检测耗时 中位数: {np.median([r.detect_time_s for r in success]):.2f}s")
    lines.append("=" * 120)

    header = (
        f"{'文件名':<62} {'面板':<8} {'滤镜':<8} {'曝光':<8} "
        f"{'结果':<4} {'RA':>11} {'Dec':>11} "
        f"{'旋转°':>8} {'尺度\"/px':>9} {'翻转':>4} "
        f"{'匹配':>4} {'RMSpx':>7} {'检测':>5} {'饱和':>4} {'Gaia':>5} "
        f"{'投影':>4} {'中位err':>8} {'<2px%':>6} "
        f"{'求解s':>6} {'检测s':>6} {'总计s':>6}"
    )
    lines.append(header)
    lines.append("-" * 120)

    for r in results:
        if r.success:
            status = "OK"
            ra_str = f"{r.center_ra:.6f}"
            dec_str = f"{r.center_dec:.6f}"
            rot_str = f"{r.rotation_deg:.4f}"
            scale_str = f"{r.scale_arcsec_px:.4f}"
            flip_str = str(r.flip_mode)
            match_str = str(r.matched_count)
            rms_str = f"{r.rms_px:.3f}"
            gaia_in = str(r.gaia_in_img)
            med_err = f"{r.gaia_median_err_px:.2f}"
            lt2 = f"{r.gaia_lt2px_pct:.1f}"
        else:
            status = "FAIL"
            ra_str = dec_str = rot_str = scale_str = flip_str = match_str = rms_str = "-"
            gaia_in = med_err = lt2 = "-"

        line = (
            f"{r.file_name:<62} {r.panel:<8} {r.filter_name:<8} {r.exposure:<8} "
            f"{status:<4} {ra_str:>11} {dec_str:>11} "
            f"{rot_str:>8} {scale_str:>9} {flip_str:>4} "
            f"{match_str:>4} {rms_str:>7} {r.n_detected:>5} {r.n_saturated:>4} {r.n_gaia:>5} "
            f"{gaia_in:>4} {med_err:>8} {lt2:>6} "
            f"{r.solve_time_s:>6.1f} {r.detect_time_s:>6.1f} {r.total_time_s:>6.1f}"
        )
        if not r.success:
            line += f"  [{r.error_msg}]"
        lines.append(line)

    lines.append("-" * 120)

    if success:
        by_filter = {}
        for r in success:
            key = r.filter_name
            if key not in by_filter:
                by_filter[key] = []
            by_filter[key].append(r)

        lines.append("")
        lines.append("按滤镜统计:")
        lines.append(f"  {'滤镜':<10} {'帧数':>4} {'RMS中位':>10} {'RMS均值':>10} {'匹配中位':>10} {'投影中位err':>12} {'<2px%':>8}")
        lines.append("  " + "-" * 70)
        for filt in sorted(by_filter.keys()):
            rs = by_filter[filt]
            lines.append(
                f"  {filt:<10} {len(rs):>4} {np.median([r.rms_px for r in rs]):>10.3f} "
                f"{np.mean([r.rms_px for r in rs]):>10.3f} {int(np.median([r.matched_count for r in rs])):>10} "
                f"{np.median([r.gaia_median_err_px for r in rs]):>12.2f} {np.mean([r.gaia_lt2px_pct for r in rs]):>8.1f}"
            )

    if fail:
        lines.append("")
        lines.append("失败列表:")
        for r in fail:
            lines.append(f"  {r.file_name}  原因: {r.error_msg}")

    lines.append("")
    lines.append("=" * 120)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return "\n".join(lines)


def main():
    print("Vector Match 批量测试")
    print(f"输出目录: {OUTPUT_DIR}")

    # 强制清空输出目录
    import shutil
    if os.path.exists(DEBUG_DIR):
        shutil.rmtree(DEBUG_DIR)
        os.makedirs(DEBUG_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, "vector_match_report.txt")
    if os.path.exists(report_path):
        os.remove(report_path)
    print("已清空旧输出")

    fits_files = []
    for panel_dir in glob.glob(os.path.join(PROJECT_ROOT, "testdata", "lights", "panel*")):
        for fts in glob.glob(os.path.join(panel_dir, "*.fts")):
            fits_files.append(fts)
    fits_files.sort()

    print(f"找到 {len(fits_files)} 个FITS文件")

    results = []
    completed = 0

    t_start = time.time()

    # 4线程并行，星点提取串行（通过锁控制）
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {executor.submit(process_single_file, fp): fp for fp in fits_files}
        for future in as_completed(future_to_file):
            fp = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(TestResult(
                    file_name=os.path.basename(fp),
                    panel="", filter_name="", exposure="",
                    success=False, error_msg=str(e)[:80],
                ))
            completed += 1
            r = results[-1]
            status = f"RMS={r.rms_px:.3f}px" if r.success else f"FAIL({r.error_msg})"
            print(f"  [{completed}/{len(fits_files)}] {r.file_name[:55]:<55} {status}")

    t_total = time.time() - t_start

    report_path = os.path.join(OUTPUT_DIR, "vector_match_report.txt")
    report_text = write_report(results, report_path)
    print(f"")
    print(report_text)
    print(f"")
    print(f"总耗时: {t_total:.1f}s, 报告: {report_path}")
    print(f"调试图像: {DEBUG_DIR}/")


if __name__ == "__main__":
    main()

"""
V2 vs V3 全量对比测试

对157帧分别运行V2和V3，对比成功率、精度、性能
"""

import os
import sys
import time
import math
import numpy as np
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import VectorMatch as VectorMatchV2, gnomonic_forward, _apply_flip
from vector_match_v3 import VectorMatch as VectorMatchV3
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 星点提取锁，确保同时只有一帧在执行星点检测
detect_lock = threading.Lock()


def get_all_test_files():
    """获取所有测试文件"""
    test_dir = os.path.join(PROJECT_ROOT, "testdata", "lights")
    files = []
    for panel in ["panel1", "panel2", "panel3"]:
        panel_dir = os.path.join(test_dir, panel)
        if os.path.isdir(panel_dir):
            for f in os.listdir(panel_dir):
                if f.endswith(".fts") or f.endswith(".fit"):
                    files.append(os.path.join("testdata", "lights", panel, f))
    return sorted(files)


def test_single(file_path, version_label, vm_class):
    """测试单帧"""
    file_name = os.path.basename(file_path)
    full_path = os.path.join(PROJECT_ROOT, file_path)
    
    result = {
        "file": file_name,
        "version": version_label,
        "success": False,
        "center_ra": 0.0,
        "center_dec": 0.0,
        "scale": 0.0,
        "rms_px": 0.0,
        "matched": 0,
        "proj_err": 0.0,
        "lt2px": 0.0,
        "detect_time": 0.0,
        "solve_time": 0.0,
        "error": "",
    }
    
    try:
        reader = ImageReader()
        img = reader.read(full_path)
        
        # 星点提取串行执行（加锁）
        with detect_lock:
            t_det_start = time.time()
            params = SDetParamsPy(fitRadius=0)
            detector = StarDetector(params=params)
            det_result = detector.detect_ex(img.data)
            t_det_end = time.time()
        
        result["detect_time"] = t_det_end - t_det_start
        
        if det_result.count < 2:
            result["error"] = "星点不足"
            return result
        
        img_x = np.array(det_result.x, dtype=np.float64)
        img_y = np.array(det_result.y, dtype=np.float64)
        img_flux = np.array(det_result.flux, dtype=np.float64)
        img_sat = np.array(det_result.saturated, dtype=np.int32)
        
        center_ra = 0.0
        center_dec = 0.0
        focal_length = 200.0
        pixel_size = 6.0
        
        if img.metadata.wcs and img.metadata.wcs.has_wcs:
            center_ra = img.metadata.wcs.crval1
            center_dec = img.metadata.wcs.crval2
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
                elif name in ("OBJCTDEC", "DEC"):
                    val = kw.value
                    if isinstance(val, str):
                        parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                        if len(parts) >= 3:
                            sign = -1 if parts[0].startswith("-") else 1
                            center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
        
        if center_ra == 0.0 and center_dec == 0.0:
            result["error"] = "无初始坐标"
            return result
        
        s0 = 206.265 * pixel_size / focal_length
        
        vm = vm_class(GAIA_DIR, db_type=2)
        t_solve_start = time.time()
        vm_result = vm.solve(
            img_x=img_x, img_y=img_y, img_flux=img_flux, img_saturated=img_sat,
            center_ra=center_ra, center_dec=center_dec,
            focal_length_mm=focal_length, pixel_size_um=pixel_size,
            width=img.width, height=img.height,
        )
        t_solve_end = time.time()
        result["solve_time"] = t_solve_end - t_solve_start
        
        if vm_result is None:
            result["error"] = "匹配失败"
            vm.close()
            return result
        
        result["success"] = True
        result["center_ra"] = vm_result.center_ra
        result["center_dec"] = vm_result.center_dec
        result["scale"] = vm_result.scale_arcsec_px
        result["rms_px"] = vm_result.rms_px
        result["matched"] = vm_result.matched_count
        
        # 投影验证（只用最亮的1000颗Gaia星）
        a0, a1, a2, b0, b1, b2 = vm_result.affine
        fov_diag = math.sqrt(img.width ** 2 + img.height ** 2) * s0 / 3600.0
        radius_deg = fov_diag * 1.2 / 2.0
        
        ra_arr, dec_arr, mag_arr = vm._gaia.cone_search(
            vm_result.center_ra, vm_result.center_dec, radius_deg, 22.0
        )
        
        if len(ra_arr) > 0:
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
            
            if np.any(in_img):
                tree = cKDTree(np.column_stack([img_x, img_y]))
                dists, _ = tree.query(np.column_stack([px[in_img], py[in_img]]))
                result["proj_err"] = float(np.median(dists))
                result["lt2px"] = float(np.sum(dists < 2) / len(dists) * 100)
        
        vm.close()
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def run_full_test(version_label, vm_class):
    """运行全量测试"""
    files = get_all_test_files()
    print(f"\n{'='*80}")
    print(f"{version_label} 全量测试: {len(files)}帧")
    print(f"{'='*80}")
    
    results = []
    n_threads = min(16, multiprocessing.cpu_count())
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(test_single, f, version_label, vm_class): f for f in files}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["success"]:
                print(f"  OK {result['file']}: RMS={result['rms_px']:.3f}px proj={result['proj_err']:.2f}px")
            else:
                print(f"  FAIL {result['file']}: {result['error']}")
    
    return results


def analyze_results(results, label):
    """分析结果"""
    success = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    
    n_success = len(success)
    n_fail = len(fail)
    success_rate = n_success / len(results) * 100
    
    if n_success > 0:
        rms_median = np.median([r["rms_px"] for r in success])
        rms_mean = np.mean([r["rms_px"] for r in success])
        proj_median = np.median([r["proj_err"] for r in success])
        lt2px_median = np.median([r["lt2px"] for r in success])
        matched_median = np.median([r["matched"] for r in success])
        detect_median = np.median([r["detect_time"] for r in success])
        solve_median = np.median([r["solve_time"] for r in success])
    else:
        rms_median = rms_mean = proj_median = lt2px_median = matched_median = detect_median = solve_median = 0
    
    print(f"\n{label} 统计:")
    print(f"  成功: {n_success}/{len(results)} ({success_rate:.1f}%)")
    print(f"  RMS(px): 中位={rms_median:.3f} 均值={rms_mean:.3f}")
    print(f"  投影误差(px): 中位={proj_median:.2f}")
    print(f"  <2px占比(%): 中位={lt2px_median:.1f}")
    print(f"  匹配数: 中位={matched_median:.0f}")
    print(f"  检测耗时(s): 中位={detect_median:.2f}")
    print(f"  求解耗时(s): 中位={solve_median:.2f}")
    
    if n_fail > 0:
        print(f"\n  失败帧:")
        for r in fail:
            print(f"    {r['file']}: {r['error']}")
    
    return {
        "label": label,
        "n_success": n_success,
        "n_fail": n_fail,
        "success_rate": success_rate,
        "rms_median": rms_median,
        "rms_mean": rms_mean,
        "proj_median": proj_median,
        "lt2px_median": lt2px_median,
        "matched_median": matched_median,
        "detect_median": detect_median,
        "solve_median": solve_median,
    }


def main():
    # V2测试
    v2_results = run_full_test("V2", VectorMatchV2)
    v2_stats = analyze_results(v2_results, "V2")
    
    # V3测试
    v3_results = run_full_test("V3", VectorMatchV3)
    v3_stats = analyze_results(v3_results, "V3")
    
    # 对比
    print(f"\n{'='*80}")
    print(f"V2 vs V3 对比")
    print(f"{'='*80}")
    print(f"  成功率: V2={v2_stats['success_rate']:.1f}% V3={v3_stats['success_rate']:.1f}% Δ={v3_stats['success_rate']-v2_stats['success_rate']:.1f}%")
    print(f"  RMS: V2={v2_stats['rms_median']:.3f}px V3={v3_stats['rms_median']:.3f}px Δ={v3_stats['rms_median']-v2_stats['rms_median']:.3f}px")
    print(f"  投影误差: V2={v2_stats['proj_median']:.2f}px V3={v3_stats['proj_median']:.2f}px Δ={v3_stats['proj_median']-v2_stats['proj_median']:.2f}px")
    print(f"  <2px占比: V2={v2_stats['lt2px_median']:.1f}% V3={v3_stats['lt2px_median']:.1f}% Δ={v3_stats['lt2px_median']-v2_stats['lt2px_median']:.1f}%")
    print(f"  匹配数: V2={v2_stats['matched_median']:.0f} V3={v3_stats['matched_median']:.0f} Δ={v3_stats['matched_median']-v2_stats['matched_median']:.0f}")
    print(f"  求解耗时: V2={v2_stats['solve_median']:.2f}s V3={v3_stats['solve_median']:.2f}s Δ={v3_stats['solve_median']-v2_stats['solve_median']:.2f}s")
    
    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "v2v3_full_comparison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("V2 vs V3 全量对比测试报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"V2 成功: {v2_stats['n_success']}/{len(v2_results)} ({v2_stats['success_rate']:.1f}%)\n")
        f.write(f"V3 成功: {v3_stats['n_success']}/{len(v3_results)} ({v3_stats['success_rate']:.1f}%)\n\n")
        f.write(f"V2 RMS中位: {v2_stats['rms_median']:.3f}px\n")
        f.write(f"V3 RMS中位: {v3_stats['rms_median']:.3f}px\n\n")
        f.write(f"V2 投影误差中位: {v2_stats['proj_median']:.2f}px\n")
        f.write(f"V3 投影误差中位: {v3_stats['proj_median']:.2f}px\n\n")
        f.write(f"V2 <2px占比中位: {v2_stats['lt2px_median']:.1f}%\n")
        f.write(f"V3 <2px占比中位: {v3_stats['lt2px_median']:.1f}%\n\n")
        f.write(f"V2 匹配数中位: {v2_stats['matched_median']:.0f}\n")
        f.write(f"V3 匹配数中位: {v3_stats['matched_median']:.0f}\n\n")
        f.write(f"V2 求解耗时中位: {v2_stats['solve_median']:.2f}s\n")
        f.write(f"V3 求解耗时中位: {v3_stats['solve_median']:.2f}s\n\n")
        
        f.write("V2 失败帧:\n")
        for r in [x for x in v2_results if not x["success"]]:
            f.write(f"  {r['file']}: {r['error']}\n")
        
        f.write("\nV3 失败帧:\n")
        for r in [x for x in v3_results if not x["success"]]:
            f.write(f"  {r['file']}: {r['error']}\n")
        
        f.write("\n详细结果:\n")
        f.write("-"*80 + "\n")
        for v2_r, v3_r in zip(v2_results, v3_results):
            f.write(f"{v2_r['file']}\n")
            f.write(f"  V2: success={v2_r['success']} RMS={v2_r['rms_px']:.3f} proj={v2_r['proj_err']:.2f} matched={v2_r['matched']} solve={v2_r['solve_time']:.2f}s\n")
            f.write(f"  V3: success={v3_r['success']} RMS={v3_r['rms_px']:.3f} proj={v3_r['proj_err']:.2f} matched={v3_r['matched']} solve={v3_r['solve_time']:.2f}s\n")
    
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Panel1四线程并发测试
Python只传入文件路径，C端完成全部处理
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(project_root, "lib", "plate_solve", "python"))

from plate_solve import PlateSolve, PlateSolveConfig

def solve_single_file(file_path: str, gaia_dir: str, config: PlateSolveConfig) -> dict:
    """单文件解析"""
    try:
        with PlateSolve(gaia_data_dir=gaia_dir) as solver:
            result = solver.solve_with_file(file_path, config)
        
        success = result.matched_count > 10 and result.rms_px < 5.0
        return {
            'file': os.path.basename(file_path),
            'success': success,
            'matched': result.matched_count,
            'rms': result.rms_px,
            'time': result.step1_time_sec + result.step2_time_sec
        }
    except Exception as e:
        return {
            'file': os.path.basename(file_path),
            'success': False,
            'error': str(e)
        }

def batch_test_parallel():
    print("=" * 80)
    print("Panel1四线程并发测试 - C端文件读取")
    print("=" * 80)
    
    gaia_dir = os.path.join(project_root, "GaiaDR3SP")
    test_dir = os.path.join(project_root, "testdata", "lights", "panel1")
    
    fits_files = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".fts")])
    print(f"\n共 {len(fits_files)} 个FITS文件")
    
    config = PlateSolveConfig()
    config.sip_order = 0
    config.max_iterations = 1
    
    results = []
    n_threads = 4
    
    t0 = time.time()
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(solve_single_file, f, gaia_dir, config): f
            for f in fits_files
        }
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            status = "OK" if result.get('success') else "FAIL"
            matched = result.get('matched', 0)
            rms = result.get('rms', 0)
            print(f"[{i+1}/{len(fits_files)}] {result['file']}: matched={matched} RMS={rms:.3f}px [{status}]")
    
    t_total = time.time() - t0
    
    print("\n" + "=" * 80)
    print("汇总")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r.get('success', False))
    fail_count = len(results) - success_count
    print(f"\n总计: {len(results)}帧, 成功: {success_count}, 失败: {fail_count}")
    print(f"总耗时: {t_total:.1f}s, 平均: {t_total/len(results):.1f}s/帧")
    print(f"并发效率: {n_threads}线程, 理论加速{n_threads}x")
    
    if success_count > 0:
        rms_vals = [r['rms'] for r in results if r.get('success')]
        match_vals = [r['matched'] for r in results if r.get('success')]
        print(f"RMS: min={min(rms_vals):.3f} max={max(rms_vals):.3f} avg={sum(rms_vals)/len(rms_vals):.3f}")
        print(f"匹配: min={min(match_vals)} max={max(match_vals)} avg={sum(match_vals)/len(match_vals):.1f}")
    
    if fail_count > 0:
        print("\n失败文件:")
        for r in results:
            if not r.get('success'):
                print(f"  {r['file']}: {r.get('error', 'matched<=10')}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    batch_test_parallel()
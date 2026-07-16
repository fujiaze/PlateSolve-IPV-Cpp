"""
测试Gaia客户端缓存效果

对比:
1. 冷启动查询 (首次)
2. 热缓存查询 (同参数重复)
3. 同天区不同mag_limit (解压块缓存命中)
4. bisection_mag_limit (7次查询, 解压块缓存逐步命中)
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
from vector_match_v2 import GaiaClientPy, bisection_mag_limit

GAIA_DATA_DIR = r"GaiaDR3SP"

def main():
    # panel1天区
    ra, dec, radius = 272.8083, -13.1839, 5.94

    print("=== 测试1: 冷启动 vs 热缓存 (同参数) ===")
    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    # 冷启动
    times_cold = []
    for i in range(3):
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, 8.5)
        t1 = time.perf_counter() - t0
        times_cold.append(t1)
        print(f"  第{i+1}次: {t1:.3f}s, 星数={len(ra_arr)}")

    # 热缓存 (同参数)
    times_hot = []
    for i in range(5):
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, 8.5)
        t1 = time.perf_counter() - t0
        times_hot.append(t1)
        print(f"  缓存第{i+1}次: {t1:.3f}s, 星数={len(ra_arr)}")

    print(f"\n  冷启动平均: {np.mean(times_cold):.3f}s")
    print(f"  热缓存平均: {np.mean(times_hot):.3f}s")
    print(f"  加速: {np.mean(times_cold)/np.mean(times_hot):.1f}x")

    print("\n=== 测试2: 解压块缓存 (同天区不同mag) ===")
    for mag in [8.0, 8.5, 9.0, 10.0, 12.0, 14.0]:
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, mag)
        t1 = time.perf_counter() - t0
        print(f"  mag={mag:5.1f}: {t1:.3f}s, 星数={len(ra_arr):>6d}")

    print("\n=== 测试3: bisection_mag_limit (解压块缓存) ===")
    t0 = time.perf_counter()
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, ra, dec, radius, 250)
    t_bis = time.perf_counter() - t0
    print(f"  bisection: {t_bis:.3f}s (M={M}, mag={mag_limit:.2f})")

    # 再做一次bisection (查询缓存应该命中)
    t0 = time.perf_counter()
    mag_limit2, M2, cat_ra2, cat_dec2, cat_mag2 = bisection_mag_limit(gaia, ra, dec, radius, 250)
    t_bis2 = time.perf_counter() - t0
    print(f"  bisection(缓存): {t_bis2:.3f}s (M={M2}, mag={mag_limit2:.2f})")
    print(f"  加速: {t_bis/t_bis2:.1f}x")

    print("\n=== 测试4: 不同天区 (panel2, panel3) ===")
    for name, test_ra, test_dec in [("panel1", 272.81, -13.18), ("panel2", 272.81, -18.20), ("panel3", 272.81, -23.22)]:
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(test_ra, test_dec, radius, 8.5)
        t1 = time.perf_counter() - t0
        print(f"  {name} (Dec={test_dec}): {t1:.3f}s, 星数={len(ra_arr)}")

    # 再查一次panel2 (解压块缓存应该命中)
    t0 = time.perf_counter()
    ra_arr, dec_arr, mag_arr = gaia.cone_search(272.81, -18.20, radius, 8.5)
    t1 = time.perf_counter() - t0
    print(f"  panel2(缓存): {t1:.3f}s")

    gaia.close()

    print("\n=== 测试5: 等待60s后缓存过期 ===")
    print("  (跳过, 太慢)")

if __name__ == '__main__':
    main()

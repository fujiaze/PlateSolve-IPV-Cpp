"""
验证: 正确radius下的Gaia查询耗时

panel1实际radius=5.94度, 之前测试用了3.0度
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
from vector_match_v2 import GaiaClientPy, bisection_mag_limit

GAIA_DATA_DIR = r"GaiaDR3SP"

def main():
    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    # panel1实际参数
    ra, dec = 272.8083, -13.1839
    radius = 5.94  # 实际值

    print(f"panel1天区: RA={ra}, Dec={dec}, radius={radius}°")
    print(f"搜索面积: π×{radius}²={3.14159*radius*radius:.1f}平方度\n")

    # 测试1: 不同mag_limit
    print("测试1: 不同mag_limit (radius=5.94°)")
    for mag in [8.0, 8.5, 9.0, 10.0, 12.0, 14.0, 18.0, 22.0]:
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, mag)
        t1 = time.perf_counter() - t0
        print(f"  mag={mag:5.1f}: {t1:.3f}s, 星数={len(ra_arr):>8d}")

    # 测试2: bisection模拟
    print(f"\n测试2: bisection_mag_limit (radius={radius}°)")
    t0 = time.perf_counter()
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, ra, dec, radius, 250)
    t_bis = time.perf_counter() - t0
    print(f"  bisection: {t_bis:.3f}s (M={M}, mag={mag_limit:.2f})")

    # 测试3: 一次查询mag=22 + 内存过滤
    print(f"\n测试3: 一次查询mag=22 + 内存过滤")
    t0 = time.perf_counter()
    ra_all, dec_all, mag_all = gaia.cone_search(ra, dec, radius, 22.0)
    t_query = time.perf_counter() - t0

    t0 = time.perf_counter()
    sort_idx = np.argsort(mag_all)
    n_top = min(250, len(mag_all))
    ra_top = ra_all[sort_idx[:n_top]]
    dec_top = dec_all[sort_idx[:n_top]]
    mag_top = mag_all[sort_idx[:n_top]]
    t_filter = time.perf_counter() - t0

    print(f"  查询: {t_query:.3f}s ({len(ra_all)}星), 过滤: {t_filter*1000:.1f}ms")
    print(f"  总计: {t_query+t_filter:.3f}s, 加速: {t_bis/(t_query+t_filter):.1f}x vs bisection")

    # 测试4: 同天区连续查询 (缓存效应)
    print(f"\n测试4: 同天区连续查询 (radius={radius}°, mag=8.5)")
    for i in range(5):
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, 8.5)
        t1 = time.perf_counter() - t0
        print(f"  第{i+1}次: {t1:.3f}s, 星数={len(ra_arr)}")

    # 测试5: 不同天区查询 (panel2 Dec=-18, panel3 Dec=-23)
    print(f"\n测试5: 不同天区 (radius={radius}°, mag=8.5)")
    for name, test_ra, test_dec in [("panel1", 272.81, -13.18), ("panel2", 272.81, -18.20), ("panel3", 272.81, -23.22)]:
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(test_ra, test_dec, radius, 8.5)
        t1 = time.perf_counter() - t0
        print(f"  {name} (Dec={test_dec}): {t1:.3f}s, 星数={len(ra_arr)}")

    gaia.close()


if __name__ == '__main__':
    main()

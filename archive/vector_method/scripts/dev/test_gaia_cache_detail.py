"""
验证: 同天区连续查询是否有缓存效应

测试:
1. 同参数连续查询2次
2. 不同mag_limit查询同天区
3. 测量C++内部耗时 vs Python数据拷贝耗时
"""

import os, sys, time, ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
from vector_match_v2 import GaiaClientPy

GAIA_DATA_DIR = r"GaiaDR3SP"

def main():
    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    # panel1天区参数
    ra, dec, radius = 272.8083, -13.1839, 3.0

    # 测试1: 同参数连续查询5次
    print("测试1: 同参数连续查询5次 (ra=272.81, dec=-13.18, radius=3.0, mag=8.5)")
    for i in range(5):
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, 8.5)
        t1 = time.perf_counter() - t0
        print(f"  第{i+1}次: {t1:.3f}s, 星数={len(ra_arr)}")

    # 测试2: 不同mag_limit查询同天区
    print("\n测试2: 不同mag_limit查询同天区")
    for mag in [8.0, 8.5, 9.0, 10.0, 12.0, 14.0]:
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, mag)
        t1 = time.perf_counter() - t0
        print(f"  mag={mag:5.1f}: {t1:.3f}s, 星数={len(ra_arr):>6d}")

    # 测试3: 直接调用C++ API, 分离C++查询 vs Python拷贝
    print("\n测试3: C++查询 vs Python拷贝耗时分离")
    ra_ptr = ctypes.POINTER(ctypes.c_double)()
    dec_ptr = ctypes.POINTER(ctypes.c_double)()
    mag_ptr = ctypes.POINTER(ctypes.c_float)()
    n_stars = ctypes.c_int()

    # C++查询
    t0 = time.perf_counter()
    ret = gaia._dll.gaia_client_cone_search_for_solver(
        gaia._handle, ra, dec, radius, 8.5,
        ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars))
    t_cpp = time.perf_counter() - t0
    count = n_stars.value
    print(f"  C++查询: {t_cpp:.3f}s, 星数={count}")

    # Python列表推导拷贝
    t0 = time.perf_counter()
    ra_arr1 = np.array([ra_ptr[i] for i in range(count)], dtype=np.float64)
    dec_arr1 = np.array([dec_ptr[i] for i in range(count)], dtype=np.float64)
    mag_arr1 = np.array([float(mag_ptr[i]) for i in range(count)], dtype=np.float64)
    t_py_copy = time.perf_counter() - t0
    print(f"  Python列表推导拷贝: {t_py_copy:.3f}s")

    # Python frombuffer拷贝
    t0 = time.perf_counter()
    ra_arr2 = np.frombuffer(ctypes.string_at(ra_ptr, count * 8), dtype=np.float64).copy()
    dec_arr2 = np.frombuffer(ctypes.string_at(dec_ptr, count * 8), dtype=np.float64).copy()
    mag_arr2 = np.frombuffer(ctypes.string_at(mag_ptr, count * 4), dtype=np.float32).astype(np.float64)
    t_np_copy = time.perf_counter() - t0
    print(f"  np.frombuffer拷贝: {t_np_copy*1000:.1f}ms")

    # 释放C++内存
    gaia._msvcrt.free(ra_ptr)
    gaia._msvcrt.free(dec_ptr)
    gaia._msvcrt.free(mag_ptr)

    # 测试4: 同天区mag=22 (大量星) 连续查询
    print("\n测试4: mag=22 (大量星) 连续查询3次")
    for i in range(3):
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, 22.0)
        t1 = time.perf_counter() - t0
        print(f"  第{i+1}次: {t1:.3f}s, 星数={len(ra_arr)}")

    # 测试5: bisection_mag_limit 模拟
    print("\n测试5: bisection_mag_limit 模拟 (7次cone_search)")
    mag_low, mag_high = 6.0, 22.0
    target = 200
    t_total = 0
    for iteration in range(7):
        mid = (mag_low + mag_high) / 2.0
        t0 = time.perf_counter()
        ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, mid)
        t1 = time.perf_counter() - t0
        t_total += t1
        count = len(ra_arr)
        if count < target:
            mag_low = mid
        else:
            mag_high = mid
        print(f"  迭代{iteration+1}: mag={mid:.2f}, 星数={count}, 耗时={t1:.3f}s (累计{t_total:.3f}s)")
    print(f"  bisection总耗时: {t_total:.3f}s")

    # 测试6: 一次查询mag=22 + 内存过滤
    print("\n测试6: 一次查询mag=22 + 内存过滤")
    t0 = time.perf_counter()
    ra_all, dec_all, mag_all = gaia.cone_search(ra, dec, radius, 22.0)
    t_query = time.perf_counter() - t0

    t0 = time.perf_counter()
    sort_idx = np.argsort(mag_all)
    n_top = min(target, len(mag_all))
    ra_top = ra_all[sort_idx[:n_top]]
    dec_top = dec_all[sort_idx[:n_top]]
    mag_top = mag_all[sort_idx[:n_top]]
    t_filter = time.perf_counter() - t0

    print(f"  查询: {t_query:.3f}s, 过滤: {t_filter*1000:.1f}ms, 总计: {t_query+t_filter:.3f}s")
    print(f"  加速: {t_total/(t_query+t_filter):.1f}x vs bisection")

    gaia.close()


if __name__ == '__main__':
    main()

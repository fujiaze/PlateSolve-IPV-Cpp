"""
验证: cone_search中Python数据拷贝是否是瓶颈

对比:
1. 完整cone_search (C++查询 + Python拷贝)
2. 只测Python数据拷贝 (模拟395万元素)
3. 用np.frombuffer替代列表推导
"""

import os
import sys
import time
import ctypes
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))

from vector_match_v2 import GaiaClientPy

GAIA_DATA_DIR = r"GaiaDR3SP"
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"


def main():
    gaia = GaiaClientPy(GAIA_DATA_DIR, db_type=0)

    # panel3 Dec=-23, mag=22 返回395万星
    ra, dec, radius, mag = 272.8083, -23.2239, 5.94, 22.0

    # 测试1: 完整cone_search (当前实现, 列表推导)
    print(f"测试1: 完整cone_search (列表推导)")
    t0 = time.perf_counter()
    ra_arr, dec_arr, mag_arr = gaia.cone_search(ra, dec, radius, mag)
    t1 = time.perf_counter() - t0
    print(f"  耗时: {t1:.2f}s, 星数: {len(ra_arr)}")

    # 测试2: 用ctypes直接调用, 分开测C++查询和Python拷贝
    print(f"\n测试2: 分开测C++查询 vs Python拷贝")
    ra_ptr = ctypes.POINTER(ctypes.c_double)()
    dec_ptr = ctypes.POINTER(ctypes.c_double)()
    mag_ptr = ctypes.POINTER(ctypes.c_float)()
    n_stars = ctypes.c_int()

    # 只测C++查询
    t0 = time.perf_counter()
    ret = gaia._dll.gaia_client_cone_search_for_solver(
        gaia._handle, ra, dec, radius, mag,
        ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars),
    )
    t_cpp = time.perf_counter() - t0
    count = n_stars.value
    print(f"  C++查询: {t_cpp:.2f}s, 星数: {count}")

    # 测Python列表推导拷贝
    t0 = time.perf_counter()
    ra_arr1 = np.array([ra_ptr[i] for i in range(count)], dtype=np.float64)
    t_copy_ra = time.perf_counter() - t0
    print(f"  Python拷贝RA (列表推导): {t_copy_ra:.2f}s")

    t0 = time.perf_counter()
    dec_arr1 = np.array([dec_ptr[i] for i in range(count)], dtype=np.float64)
    t_copy_dec = time.perf_counter() - t0
    print(f"  Python拷贝Dec (列表推导): {t_copy_dec:.2f}s")

    t0 = time.perf_counter()
    mag_arr1 = np.array([float(mag_ptr[i]) for i in range(count)], dtype=np.float64)
    t_copy_mag = time.perf_counter() - t0
    print(f"  Python拷贝Mag (列表推导+float转换): {t_copy_mag:.2f}s")

    t_copy_total = t_copy_ra + t_copy_dec + t_copy_mag
    print(f"  Python拷贝总计: {t_copy_total:.2f}s ({t_copy_total/t1*100:.1f}% of 总耗时)")

    # 测np.frombuffer拷贝 (零拷贝)
    t0 = time.perf_counter()
    ra_arr2 = np.frombuffer(ctypes.string_at(ra_ptr, count * 8), dtype=np.float64).copy()
    t_copy_ra2 = time.perf_counter() - t0
    print(f"\n  np.frombuffer拷贝RA: {t_copy_ra2*1000:.1f}ms")

    t0 = time.perf_counter()
    dec_arr2 = np.frombuffer(ctypes.string_at(dec_ptr, count * 8), dtype=np.float64).copy()
    t_copy_dec2 = time.perf_counter() - t0
    print(f"  np.frombuffer拷贝Dec: {t_copy_dec2*1000:.1f}ms")

    t0 = time.perf_counter()
    mag_arr2 = np.frombuffer(ctypes.string_at(mag_ptr, count * 4), dtype=np.float32).astype(np.float64)
    t_copy_mag2 = time.perf_counter() - t0
    print(f"  np.frombuffer拷贝Mag (float32→float64): {t_copy_mag2*1000:.1f}ms")

    t_copy_total2 = t_copy_ra2 + t_copy_dec2 + t_copy_mag2
    print(f"  np.frombuffer拷贝总计: {t_copy_total2*1000:.1f}ms")
    print(f"  加速: {t_copy_total/t_copy_total2:.0f}x")

    # 验证数据一致性
    print(f"\n  数据一致性检查:")
    print(f"    RA max diff: {np.max(np.abs(ra_arr1 - ra_arr2)):.2e}")
    print(f"    Dec max diff: {np.max(np.abs(dec_arr1 - dec_arr2)):.2e}")
    print(f"    Mag max diff: {np.max(np.abs(mag_arr1 - mag_arr2)):.2e}")

    # 释放C++内存
    gaia._msvcrt.free(ra_ptr)
    gaia._msvcrt.free(dec_ptr)
    gaia._msvcrt.free(mag_ptr)

    # 测试3: mag=8.5 (414星) 时的拷贝开销
    print(f"\n测试3: mag=8.5 (414星, 实际需要的星数)")
    t0 = time.perf_counter()
    ra_arr3, dec_arr3, mag_arr3 = gaia.cone_search(ra, dec, radius, 8.5)
    t_small = time.perf_counter() - t0
    print(f"  完整cone_search: {t_small:.2f}s, 星数: {len(ra_arr3)}")

    # 测试4: 一次查询mag=22 + frombuffer + 内存过滤
    print(f"\n测试4: 一次查询mag=22 + frombuffer + 内存过滤 (目标350星)")
    t0 = time.perf_counter()
    ra_ptr = ctypes.POINTER(ctypes.c_double)()
    dec_ptr = ctypes.POINTER(ctypes.c_double)()
    mag_ptr = ctypes.POINTER(ctypes.c_float)()
    n_stars = ctypes.c_int()
    ret = gaia._dll.gaia_client_cone_search_for_solver(
        gaia._handle, ra, dec, radius, 22.0,
        ctypes.byref(ra_ptr), ctypes.byref(dec_ptr), ctypes.byref(mag_ptr), ctypes.byref(n_stars),
    )
    count = n_stars.value
    # frombuffer拷贝
    ra_all = np.frombuffer(ctypes.string_at(ra_ptr, count * 8), dtype=np.float64).copy()
    dec_all = np.frombuffer(ctypes.string_at(dec_ptr, count * 8), dtype=np.float64).copy()
    mag_all = np.frombuffer(ctypes.string_at(mag_ptr, count * 4), dtype=np.float32).astype(np.float64)
    gaia._msvcrt.free(ra_ptr)
    gaia._msvcrt.free(dec_ptr)
    gaia._msvcrt.free(mag_ptr)
    # 内存过滤
    sort_idx = np.argsort(mag_all)
    n_top = min(350, len(mag_all))
    ra_top = ra_all[sort_idx[:n_top]]
    dec_top = dec_all[sort_idx[:n_top]]
    mag_top = mag_all[sort_idx[:n_top]]
    t_opt = time.perf_counter() - t0
    print(f"  一次查询+frombuffer+过滤: {t_opt:.2f}s, 星数: {len(ra_top)}")
    print(f"  vs bisection (7次查询): 加速{12.4/t_opt:.1f}x")

    gaia.close()


if __name__ == '__main__':
    main()

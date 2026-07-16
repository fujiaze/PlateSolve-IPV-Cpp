"""
smoke_test_v43.py - V4.3 Python 封装烟雾测试

测试目标:
    1. import V43Solver 成功
    2. 初始化 V43Solver (内部创建 GaiaClient + StarDetector)
    3. 调用 vm43_get_default_params 验证字段
    4. 资源释放正常

不进行端到端求解测试 (需要真实 FITS 数据, 在 Task 11 进行)
"""

import os
import sys
import logging

# MinGW runtime 必须在 PATH 中 (libgcc_s_seh, libstdc++, libgomp)
# Python 3.8+ 需要显式 add_dll_directory
_MINGW_BIN = r"C:\msys64\mingw64\bin"
if _MINGW_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except (OSError, FileNotFoundError):
        pass

# 配置日志
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# 项目路径
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
_PLATE_SOLVE_PY = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "python")
if _PLATE_SOLVE_PY not in sys.path:
    sys.path.insert(0, _PLATE_SOLVE_PY)

from v4_3.vector_match_v4_3_cpp import V43Solver, VM43SolveParams


def test_import_and_init():
    """测试 1: import + 初始化 + 关闭"""
    print("\n[测试 1] V43Solver 初始化测试")
    try:
        with V43Solver() as solver:
            print(f"  ✓ V43Solver 初始化成功")
            print(f"  ✓ DLL 路径: {os.path.basename(solver._dll_path)}")
            print(f"  ✓ GaiaClient: {type(solver._gaia_client).__name__}")
            print(f"  ✓ StarDetector: {type(solver._star_detector).__name__}")
            print(f"  ✓ GaiaClient._handle: {solver._gaia_client._handle}")
            print(f"  ✓ StarDetector._handle: {solver._star_detector._handle}")
        print(f"  ✓ V43Solver 已关闭")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_params():
    """测试 2: 默认参数获取"""
    print("\n[测试 2] 默认参数获取测试")
    try:
        # 通过 DLL 直接调用 vm43_get_default_params
        import ctypes
        dll_path = os.path.join(
            _PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_3",
            "vector_match_v4_3.dll")
        dll = ctypes.CDLL(dll_path)
        dll.vm43_get_default_params.argtypes = [ctypes.POINTER(VM43SolveParams)]
        dll.vm43_get_default_params.restype = None

        params = VM43SolveParams()
        dll.vm43_get_default_params(ctypes.byref(params))

        # 验证关键字段
        checks = [
            ("n_modes", params.n_modes, 4),
            ("img_n_target", params.img_n_target, 50),
            ("gaia_density_ratio", params.gaia_density_ratio, 1.5),
            ("gaia_query_radius_factor", params.gaia_query_radius_factor, 0.55),
            ("s_min", params.s_min, 0.9),
            ("s_max", params.s_max, 1.1),
            ("N_max", params.N_max, 1500),
            ("irm_max_iter", params.irm_max_iter, 10),
            ("irm_converge_eps", params.irm_converge_eps, 0.05),
            ("irm_lowe_ratio", params.irm_lowe_ratio, 0.7),
            ("irm_k_geometry", params.irm_k_geometry, 8),
            ("irm_huber_delta_factor", params.irm_huber_delta_factor, 1.345),
            ("sip_max_order", params.sip_max_order, 4),
        ]
        all_ok = True
        for name, actual, expected in checks:
            ok = abs(actual - expected) < 1e-9 if isinstance(expected, float) else actual == expected
            mark = "✓" if ok else "✗"
            print(f"  {mark} {name} = {actual} (期望 {expected})")
            if not ok:
                all_ok = False

        print(f"  {'✓' if all_ok else '✗'} 默认参数验证{'通过' if all_ok else '失败'}")
        return all_ok
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_struct_layout():
    """测试 3: 结构体 sizeof 检查 (粗略对齐验证)"""
    print("\n[测试 3] ctypes 结构体 sizeof 检查")
    try:
        import ctypes
        params_size = ctypes.sizeof(VM43SolveParams)
        result_size = ctypes.sizeof(ctypes.POINTER(ctypes.c_int))  # 占位
        print(f"  ✓ VM43SolveParams sizeof = {params_size} 字节")
        # 验证字段数
        n_fields = len(VM43SolveParams._fields_)
        print(f"  ✓ VM43SolveParams 字段数 = {n_fields}")
        # 应该是 49 个字段 (数一下)
        expected_fields = 49
        if n_fields == expected_fields:
            print(f"  ✓ 字段数匹配 ({expected_fields})")
            return True
        else:
            print(f"  ✗ 字段数不匹配: 实际 {n_fields}, 期望 {expected_fields}")
            return False
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def test_handle_injection():
    """测试 4: 句柄注入 (使用真实 GaiaClient + StarDetector)"""
    print("\n[测试 4] 句柄注入测试")
    try:
        with V43Solver() as solver:
            # 调用 solve 但传入不存在的图像 → 应在 vm43_select 阶段失败
            # 但不会在句柄检查阶段失败 (句柄已成功注入)
            result = solver.solve(
                image_path="non_existent.fits",
                ra=10.0, dec=20.0,
                focal_length_mm=1000.0, pixel_size_um=5.0,
            )
            # 应返回 success=False, 错误信息提及图像读取失败或类似
            if not result["success"]:
                print(f"  ✓ 不存在图像正确返回 success=False")
                print(f"  ✓ 错误信息: {result.get('error', '(空)')[:80]}")
                return True
            else:
                print(f"  ✗ 不存在图像却返回 success=True")
                return False
    except FileNotFoundError as e:
        # 这种情况也合理 (Python 端先检查文件存在)
        print(f"  ✓ Python 端文件检查拦截: {e}")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("V4.3 Python 封装烟雾测试")
    print("=" * 70)

    results = []
    results.append(("import_and_init", test_import_and_init()))
    results.append(("default_params", test_default_params()))
    results.append(("struct_layout", test_struct_layout()))
    results.append(("handle_injection", test_handle_injection()))

    print("\n" + "=" * 70)
    print("测试汇总:")
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}")
    print(f"\n{passed}/{total} 通过")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

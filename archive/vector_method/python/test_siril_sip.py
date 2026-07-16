# -*- coding: utf-8 -*-
"""
测试Siril风格5阶SIP拟合
"""
import numpy as np
import ctypes
import os
import sys

dll_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'plate_solve.dll'))
print(f"DLL路径: {dll_path}")
if not os.path.exists(dll_path):
    print(f"DLL not found: {dll_path}")
    sys.exit(1)

os.add_dll_directory(os.path.dirname(dll_path))
os.add_dll_directory(r'C:\msys64\mingw64\bin')

try:
    dll = ctypes.CDLL(dll_path)
except Exception as e:
    print(f"加载DLL失败: {e}")
    sys.exit(1)

print("=== 测试Siril风格5阶SIP拟合 ===")

np.random.seed(42)
n_stars = 500

img_x = np.random.uniform(-2000, 2000, n_stars).astype(np.float64)
img_y = np.random.uniform(-1500, 1500, n_stars).astype(np.float64)

tx, ty = 0.5, -0.3
scale = 1.0001
rot = 0.0002
cos_r, sin_r = np.cos(rot), np.sin(rot)

cat_x = tx + scale * (cos_r * img_x - sin_r * img_y)
cat_y = ty + scale * (sin_r * img_x + cos_r * img_y)

A20, A11, A02 = 1e-7, -5e-8, 2e-7
B20, B11, B02 = -2e-7, 3e-8, 1e-7
A30, A21, A12, A03 = 1e-11, -5e-12, 2e-11, -1e-11
B30, B21, B12, B03 = -1e-11, 2e-12, -3e-11, 1e-11

sip_dx = (A20 * img_x**2 + A11 * img_x * img_y + A02 * img_y**2 +
          A30 * img_x**3 + A21 * img_x**2 * img_y + A12 * img_x * img_y**2 + A03 * img_y**3)
sip_dy = (B20 * img_x**2 + B11 * img_x * img_y + B02 * img_y**2 +
          B30 * img_x**3 + B21 * img_x**2 * img_y + B12 * img_x * img_y**2 + B03 * img_y**3)

cat_x += sip_dx
cat_y += sip_dy

noise = 0.1
cat_x += np.random.normal(0, noise, n_stars)
cat_y += np.random.normal(0, noise, n_stars)

print(f"生成 {n_stars} 颗测试星点")
print(f"真实参数: tx={tx:.4f}, ty={ty:.4f}, scale={scale:.6f}, rot={rot:.6f}")
print(f"SIP系数 (2阶): A20={A20:.2e}, A11={A11:.2e}, A02={A02:.2e}")

class TRANS(ctypes.Structure):
    _fields_ = [
        ('order', ctypes.c_int),
        ('nm', ctypes.c_int),
        ('sx', ctypes.c_double),
        ('sy', ctypes.c_double),
        ('x00', ctypes.c_double), ('x10', ctypes.c_double), ('x01', ctypes.c_double),
        ('x20', ctypes.c_double), ('x11', ctypes.c_double), ('x02', ctypes.c_double),
        ('x30', ctypes.c_double), ('x21', ctypes.c_double), ('x12', ctypes.c_double), ('x03', ctypes.c_double),
        ('x40', ctypes.c_double), ('x31', ctypes.c_double), ('x22', ctypes.c_double), ('x13', ctypes.c_double), ('x04', ctypes.c_double),
        ('x50', ctypes.c_double), ('x41', ctypes.c_double), ('x32', ctypes.c_double), ('x23', ctypes.c_double), ('x14', ctypes.c_double), ('x05', ctypes.c_double),
        ('y00', ctypes.c_double), ('y10', ctypes.c_double), ('y01', ctypes.c_double),
        ('y20', ctypes.c_double), ('y11', ctypes.c_double), ('y02', ctypes.c_double),
        ('y30', ctypes.c_double), ('y21', ctypes.c_double), ('y12', ctypes.c_double), ('y03', ctypes.c_double),
        ('y40', ctypes.c_double), ('y31', ctypes.c_double), ('y22', ctypes.c_double), ('y13', ctypes.c_double), ('y04', ctypes.c_double),
        ('y50', ctypes.c_double), ('y41', ctypes.c_double), ('y32', ctypes.c_double), ('y23', ctypes.c_double), ('y14', ctypes.c_double), ('y05', ctypes.c_double),
    ]

class SIP_Coefficients(ctypes.Structure):
    _fields_ = [
        ('A', (ctypes.c_double * 6) * 6),
        ('B', (ctypes.c_double * 6) * 6),
        ('AP', (ctypes.c_double * 6) * 6),
        ('BP', (ctypes.c_double * 6) * 6),
        ('order', ctypes.c_int),
        ('valid', ctypes.c_int),
    ]

dll.psm_siril_refine.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
    ctypes.c_double, ctypes.c_double,
    ctypes.c_int, ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(TRANS),
    ctypes.POINTER(SIP_Coefficients),
    ctypes.POINTER(ctypes.c_double)
]
dll.psm_siril_refine.restype = ctypes.c_int

img_x_c = img_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
img_y_c = img_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
cat_x_c = cat_x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
cat_y_c = cat_y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

trans = TRANS()
sip = SIP_Coefficients()
rms = ctypes.c_double()

print("\n调用Siril风格SIP拟合...")
status = dll.psm_siril_refine(
    img_x_c, img_y_c, n_stars,
    cat_x_c, cat_y_c, n_stars,
    0.0, 0.0,
    4000, 3000,
    5,
    ctypes.byref(trans),
    ctypes.byref(sip),
    ctypes.byref(rms)
)

print(f"\n拟合状态: {status}")
print(f"RMS: {rms.value:.4f} px")

print(f"\nTRANS参数:")
print(f"  x00={trans.x00:.6f}, y00={trans.y00:.6f}")
print(f"  x10={trans.x10:.6f}, x01={trans.x01:.6f}")
print(f"  y10={trans.y10:.6f}, y01={trans.y01:.6f}")

print(f"\n拟合的仿射参数:")
fit_scale = np.sqrt(trans.x10**2 + trans.y10**2)
fit_rot = np.arctan2(trans.y10, trans.x10)
print(f"  scale={fit_scale:.6f} (真实={scale:.6f})")
print(f"  rot={fit_rot:.6f} rad (真实={rot:.6f})")
print(f"  tx={trans.x00:.6f} (真实={tx:.6f})")
print(f"  ty={trans.y00:.6f} (真实={ty:.6f})")

if sip.valid:
    print(f"\nSIP系数 (正向变换):")
    print(f"  A_20={sip.A[2][0]:.2e} (真实={A20:.2e})")
    print(f"  A_11={sip.A[1][1]:.2e} (真实={A11:.2e})")
    print(f"  A_02={sip.A[0][2]:.2e} (真实={A02:.2e})")
    print(f"  B_20={sip.B[2][0]:.2e} (真实={B20:.2e})")
    print(f"  B_11={sip.B[1][1]:.2e} (真实={B11:.2e})")
    print(f"  B_02={sip.B[0][2]:.2e} (真实={B02:.2e})")
    
    print(f"\nSIP系数 (逆向变换):")
    print(f"  AP_20={sip.AP[2][0]:.2e}")
    print(f"  AP_11={sip.AP[1][1]:.2e}")
    print(f"  AP_02={sip.AP[0][2]:.2e}")
    print(f"  BP_20={sip.BP[2][0]:.2e}")
    print(f"  BP_11={sip.BP[1][1]:.2e}")
    print(f"  BP_02={sip.BP[0][2]:.2e}")
    
    print(f"\n高阶系数:")
    print(f"  A_30={sip.A[3][0]:.2e}, A_21={sip.A[2][1]:.2e}, A_12={sip.A[1][2]:.2e}, A_03={sip.A[0][3]:.2e}")
    print(f"  A_40={sip.A[4][0]:.2e}, A_31={sip.A[3][1]:.2e}, A_22={sip.A[2][2]:.2e}, A_13={sip.A[1][3]:.2e}, A_04={sip.A[0][4]:.2e}")
    print(f"  A_50={sip.A[5][0]:.2e}, A_41={sip.A[4][1]:.2e}, A_32={sip.A[3][2]:.2e}, A_23={sip.A[2][3]:.2e}, A_14={sip.A[1][4]:.2e}, A_05={sip.A[0][5]:.2e}")
else:
    print("\nSIP系数计算失败")

print("\n=== 测试完成 ===")

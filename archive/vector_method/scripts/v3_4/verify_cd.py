"""验证CD矩阵逆投影 vs 直接相似变换投影的一致性"""
import os, sys, math, json, numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
from vector_match_v2 import _build_catalog_vectors, _apply_flip, _apply_similarity
from astro_image_io import ImageReader

# 读取WCS JSON
with open(os.path.join(PROJECT_ROOT, "vm34_wcs_output.json"), 'r') as f:
    wcs = json.load(f)
cd = np.array(wcs['CD'])
crval = np.array(wcs['CRVAL'])
crpix = np.array(wcs['CRPIX'])

# 从图像头获取正确参数
reader = ImageReader()
fname = "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
img = reader.read(os.path.join(PROJECT_ROOT, "testdata", "lights", fname))
fl = img.metadata.observation.focallen; ps = img.metadata.observation.xpixsz
s0 = 206.265*ps/fl; w, h = img.width, img.height
cra0 = img.metadata.wcs.crval1; cdec0 = img.metadata.wcs.crval2

# Phase D参数
s = 1.004448; theta = -89.11 * math.pi/180; tx = -1.142; ty = 0.900
flip_mode = 1
cra_s, cdec_s = crval  # 精修后中心

print(f"s0={s0:.6f} arcsec/px  fl={fl}  ps={ps}")
print(f"CD={cd.tolist()}")
print(f"CRVAL={crval}  CRPIX={crpix}")

# 测试几颗星
test_ras = np.array([270.5, 270.7, 270.9, 270.6, 270.8])
test_decs = np.array([-22.7, -22.8, -22.9, -23.0, -22.85])

# 方法1: 直接相似变换
W = _build_catalog_vectors(test_ras, test_decs, cra_s, cdec_s)
Wf = _apply_flip(W, flip_mode)
U = _apply_similarity(Wf, s, theta, tx, ty)
x1 = U[:, 0] / s0 + w/2
y1 = -U[:, 1] / s0 + h/2

# 方法2: CD矩阵逆投影
cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
cd_inv = np.array([[cd[1,1], -cd[0,1]], [-cd[1,0], cd[0,0]]]) / cdet
dra = test_ras - crval[0]
ddec = test_decs - crval[1]
xi = cd_inv[0,0]*dra + cd_inv[0,1]*ddec
eta = cd_inv[1,0]*dra + cd_inv[1,1]*ddec
x2 = xi + crpix[0]
y2 = eta + crpix[1]

print(f"\n直接相似变换 vs CD矩阵逆投影:")
print(f"{'星':>3} {'x_sim':>10} {'y_sim':>10} {'x_cd':>10} {'y_cd':>10} {'dx':>8} {'dy':>8}")
for i in range(len(test_ras)):
    dx = x1[i] - x2[i]; dy = y1[i] - y2[i]
    print(f"{i:3d} {x1[i]:10.2f} {y1[i]:10.2f} {x2[i]:10.2f} {y2[i]:10.2f} {dx:8.2f} {dy:8.2f}")

# 计算CD矩阵的理论值并对比
ct = math.cos(theta); st = math.sin(theta)
cos_dec = math.cos(cdec_s * math.pi/180)
s0_s_3600 = s0 / (s * 3600.0)
fx = (flip_mode == 1 or flip_mode == 3)
fy = (flip_mode == 2 or flip_mode == 3)
sign_x = -1.0 if fx else 1.0
sign_y = -1.0 if fy else 1.0
cd_theory = np.array([
    [sign_x * s0_s_3600 * ct / cos_dec, -sign_x * s0_s_3600 * st / cos_dec],
    [-sign_y * s0_s_3600 * st, -sign_y * s0_s_3600 * ct]
])
print(f"\nCD理论值: {cd_theory.tolist()}")
print(f"CD实际值: {cd.tolist()}")
print(f"CD差异: {(cd - cd_theory).tolist()}")

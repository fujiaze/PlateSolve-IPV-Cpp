import numpy as np

cp_csv = r"F:\Astro dev\Astro CS Normalization Database\debug_control_points.csv"
cp_data = np.loadtxt(cp_csv, delimiter=',', skiprows=1)

valid_mask = cp_data[:, 6] == 1
cp_valid = cp_data[valid_mask]

img_x_abs = cp_valid[:, 0]
img_y_abs = cp_valid[:, 1]
cat_x_abs = cp_valid[:, 2]
cat_y_abs = cp_valid[:, 3]
res_x = cp_valid[:, 4]
res_y = cp_valid[:, 5]

width = 4500
height = 3600
half_w = width / 2.0
half_h = height / 2.0

img_rel_x = img_x_abs - half_w
img_rel_y = half_h - img_y_abs
cat_rel_x = cat_x_abs - half_w
cat_rel_y = half_h - cat_y_abs

print("=== 验证CSV坐标转换 ===")
print(f"\n代码中的转换公式:")
print(f"  img_x_abs = cp->img_x + width/2")
print(f"  img_y_abs = -(cp->img_y - height/2) = height/2 - cp->img_y")
print(f"  cat_x_abs = cp->cat_x + width/2")
print(f"  cat_y_abs = -(cp->cat_y - height/2) = height/2 - cp->cat_y")

print(f"\n反向推导原始坐标:")
img_x_orig = img_x_abs - half_w
img_y_orig = half_h - img_y_abs
cat_x_orig = cat_x_abs - half_w
cat_y_orig = half_h - cat_y_abs

print(f"\n原始img坐标 (前5个):")
for i in range(5):
    print(f"  [{i}] img_x={img_x_orig[i]:.1f}, img_y={img_y_orig[i]:.1f}")

print(f"\n原始cat坐标 (前5个):")
for i in range(5):
    print(f"  [{i}] cat_x={cat_x_orig[i]:.1f}, cat_y={cat_y_orig[i]:.1f}")

print(f"\n残差验证:")
calc_res_x = cat_x_orig - img_x_orig
calc_res_y = cat_y_orig - img_y_orig
print(f"  计算残差: X={calc_res_x[:5]}, Y={calc_res_y[:5]}")
print(f"  CSV残差: X={res_x[:5]}, Y={res_y[:5]}")
print(f"  差异: X={calc_res_x[:5] - res_x[:5]}, Y={calc_res_y[:5] - res_y[:5]}")

print(f"\n=== 检查残差计算公式 ===")
print(f"代码中:")
print(f"  res_x_abs = cat_x_abs - img_x_abs")
print(f"  res_y_abs = cat_y_abs - img_y_abs")
print(f"\n展开:")
print(f"  res_x_abs = (cat_x + half_w) - (img_x + half_w) = cat_x - img_x")
print(f"  res_y_abs = (half_h - cat_y) - (half_h - img_y) = img_y - cat_y")
print(f"\n问题:")
print(f"  res_y_abs = img_y - cat_y, 但实际应该是 cat_y - img_y")
print(f"  这导致Y残差符号相反!")

print(f"\n验证:")
expected_res_y_abs = cat_y_abs - img_y_abs
print(f"  正确Y残差: {expected_res_y_abs[:5]}")
print(f"  CSV Y残差: {res_y[:5]}")
print(f"  符号相反: {expected_res_y_abs[:5] + res_y[:5]}")
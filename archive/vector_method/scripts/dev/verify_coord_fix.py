import numpy as np

gaia_csv = r"F:\Astro dev\Astro CS Normalization Database\debug_gaia_predictions.csv"
cp_csv = r"F:\Astro dev\Astro CS Normalization Database\debug_control_points.csv"

gaia_data = np.loadtxt(gaia_csv, delimiter=',', skiprows=1)
cp_data = np.loadtxt(cp_csv, delimiter=',', skiprows=1)

width = 4500
height = 3600
half_w = width / 2.0
half_h = height / 2.0

gaia_rel_x = gaia_data[:, 0]
gaia_rel_y = gaia_data[:, 1]

gaia_abs_x_old = gaia_rel_x + half_w
gaia_abs_y_old = half_h - gaia_rel_y

gaia_abs_x_new = gaia_rel_x + half_w
gaia_abs_y_new = gaia_rel_y + half_h

valid_mask = cp_data[:, 6] == 1
cp_valid = cp_data[valid_mask]
cp_img_x = cp_valid[:, 0]
cp_img_y = cp_valid[:, 1]
cp_cat_x = cp_valid[:, 2]
cp_cat_y = cp_valid[:, 3]

print("=== 坐标转换验证 ===")
print(f"\nGaia预测 (前5个):")
print(f"原始转换 (错误):")
for i in range(5):
    print(f"  [{i}] rel=({gaia_rel_x[i]:.1f}, {gaia_rel_y[i]:.1f}) -> abs_old=({gaia_abs_x_old[i]:.1f}, {gaia_abs_y_old[i]:.1f})")

print(f"\n正确转换:")
for i in range(5):
    print(f"  [{i}] rel=({gaia_rel_x[i]:.1f}, {gaia_rel_y[i]:.1f}) -> abs_new=({gaia_abs_x_new[i]:.1f}, {gaia_abs_y_new[i]:.1f})")

print(f"\n控制点 (前5个):")
for i in range(5):
    print(f"  [{i}] img=({cp_img_x[i]:.1f}, {cp_img_y[i]:.1f}), cat=({cp_cat_x[i]:.1f}, {cp_cat_y[i]:.1f})")

print(f"\n=== 检查Gaia预测和控制点cat坐标的一致性 ===")
matches_old = []
matches_new = []
for i in range(len(gaia_data)):
    for j in range(len(cp_valid)):
        if abs(gaia_abs_x_old[i] - cp_cat_x[j]) < 1 and abs(gaia_abs_y_old[i] - cp_cat_y[j]) < 1:
            matches_old.append((i, j))
        if abs(gaia_abs_x_new[i] - cp_cat_x[j]) < 1 and abs(gaia_abs_y_new[i] - cp_cat_y[j]) < 1:
            matches_new.append((i, j))

print(f"旧转换匹配数: {len(matches_old)}")
print(f"新转换匹配数: {len(matches_new)}")

if len(matches_new) > 0:
    print(f"\n使用正确转换后的匹配示例:")
    for m in matches_new[:5]:
        i, j = m
        print(f"  Gaia[{i}] abs=({gaia_abs_x_new[i]:.1f}, {gaia_abs_y_new[i]:.1f})")
        print(f"  CP[{j}] cat=({cp_cat_x[j]:.1f}, {cp_cat_y[j]:.1f}), img=({cp_img_x[j]:.1f}, {cp_img_y[j]:.1f})")
        print(f"  CP残差: ({cp_valid[j, 4]:.3f}, {cp_valid[j, 5]:.3f})")

print(f"\n=== 结论 ===")
print(f"正确的坐标转换公式:")
print(f"  gaia_abs_x = pred_x + half_w")
print(f"  gaia_abs_y = pred_y + half_h  (不需要取负号!)")
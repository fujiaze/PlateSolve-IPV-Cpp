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
gaia_abs_x = gaia_rel_x + half_w
gaia_abs_y = half_h - gaia_rel_y

valid_mask = cp_data[:, 6] == 1
cp_valid = cp_data[valid_mask]
cp_img_x = cp_valid[:, 0]
cp_img_y = cp_valid[:, 1]
cp_cat_x = cp_valid[:, 2]
cp_cat_y = cp_valid[:, 3]

print("=== 坐标转换验证 ===")
print(f"\nGaia预测 (前5个):")
for i in range(5):
    print(f"  [{i}] rel_x={gaia_rel_x[i]:.1f}, rel_y={gaia_rel_y[i]:.1f}")
    print(f"       -> abs_x={gaia_abs_x[i]:.1f}, abs_y={gaia_abs_y[i]:.1f}")

print(f"\n控制点 (前5个):")
for i in range(5):
    print(f"  [{i}] img_x={cp_img_x[i]:.1f}, img_y={cp_img_y[i]:.1f}")
    print(f"       cat_x={cp_cat_x[i]:.1f}, cat_y={cp_cat_y[i]:.1f}")

print(f"\n=== 检查Gaia预测和控制点cat坐标的一致性 ===")
matches = []
for i in range(len(gaia_data)):
    for j in range(len(cp_valid)):
        gaia_rel_x_i = gaia_rel_x[i]
        gaia_rel_y_i = gaia_rel_y[i]
        cp_cat_rel_x = cp_cat_x[j] - half_w
        cp_cat_rel_y = half_h - cp_cat_y[j]
        
        if abs(gaia_rel_x_i - cp_cat_rel_x) < 1 and abs(gaia_rel_y_i - cp_cat_rel_y) < 1:
            matches.append((i, j, gaia_rel_x_i - cp_cat_rel_x, gaia_rel_y_i - cp_cat_rel_y))

print(f"找到 {len(matches)} 个匹配")
if len(matches) > 0:
    for m in matches[:10]:
        print(f"  Gaia[{m[0]}] ~= CP[{m[1]}]: diff_x={m[2]:.3f}, diff_y={m[3]:.3f}")
        print(f"    Gaia: rel_x={gaia_rel_x[m[0]]:.1f}, rel_y={gaia_rel_y[m[0]]:.1f}")
        print(f"    CP cat_abs: x={cp_cat_x[m[1]]:.1f}, y={cp_cat_y[m[1]]:.1f}")
        print(f"    CP img_abs: x={cp_img_x[m[1]]:.1f}, y={cp_img_y[m[1]]:.1f}")
        print(f"    CP残差: res_x={cp_valid[m[1], 4]:.3f}, res_y={cp_valid[m[1], 5]:.3f}")

print(f"\n=== 检查坐标转换公式 ===")
print(f"控制点CSV存储的是绝对坐标:")
print(f"  img_x_abs = cp->img_x + width/2")
print(f"  img_y_abs = -(cp->img_y - height/2) = height/2 - cp->img_y")
print(f"\nGaia预测CSV存储的是相对坐标:")
print(f"  pred_x = 相对于中心的X")
print(f"  pred_y = 相对于中心的Y（取负号）")
print(f"\n可视化脚本转换:")
print(f"  gaia_abs_x = pred_x + half_w")
print(f"  gaia_abs_y = -(pred_y - half_h) = half_h - pred_y")
print(f"\n验证: 如果pred_y = -(像素y - half_h)，则像素y = half_h - pred_y")
print(f"  gaia_abs_y = half_h - pred_y = 像素y ✓")
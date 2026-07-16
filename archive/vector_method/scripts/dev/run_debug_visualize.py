import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

fits_path = r"F:\Astro dev\Astro CS Normalization Database\testdata\lights\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@051551-180S-Red.fts"
gaia_csv = r"F:\Astro dev\Astro CS Normalization Database\debug_gaia_predictions.csv"
step2_csv = r"F:\Astro dev\Astro CS Normalization Database\debug_control_points.csv"

with fits.open(fits_path) as hdul:
    img_data = hdul[0].data

height, width = img_data.shape
half_w = width / 2.0
half_h = height / 2.0

gaia_data = np.loadtxt(gaia_csv, delimiter=',', skiprows=1)
gaia_pred_x = gaia_data[:, 0] + half_w
gaia_pred_y = half_h - gaia_data[:, 1]

step2_data = np.loadtxt(step2_csv, delimiter=',', skiprows=1)
valid_mask = step2_data[:, 6] == 1
step2_valid = step2_data[valid_mask]
step2_img_x = step2_valid[:, 0]
step2_img_y = step2_valid[:, 1]
step2_cat_x = step2_valid[:, 2]
step2_cat_y = step2_valid[:, 3]

img_min, img_max = np.percentile(img_data, [0.5, 99.5])
stretched = np.clip((img_data - img_min) / (img_max - img_min), 0, 1)
rgb = np.stack([stretched, stretched, stretched], axis=-1).astype(np.float32)

fig1 = plt.figure(figsize=(width/100, height/100), dpi=100)
ax1 = fig1.add_axes([0, 0, 1, 1])
ax1.imshow(rgb, origin='lower', extent=[0, width, 0, height])

ax1.scatter(gaia_pred_x, gaia_pred_y, c='red', s=8, marker='+', linewidths=0.5)

ax1.set_xlim(0, width)
ax1.set_ylim(0, height)
ax1.axis('off')

output1 = r"F:\Astro dev\Astro CS Normalization Database\debug_step1_alignment.png"
plt.savefig(output1, dpi=100, bbox_inches='tight', pad_inches=0)
plt.close(fig1)

print(f"Step1 saved: {output1}")
print(f"  Gaia predictions (top 1000): {len(gaia_data)}")

fig2 = plt.figure(figsize=(width/100, height/100), dpi=100)
ax2 = fig2.add_axes([0, 0, 1, 1])
ax2.imshow(rgb, origin='lower', extent=[0, width, 0, height])

ax2.scatter(gaia_pred_x, gaia_pred_y, c='red', s=8, marker='+', linewidths=0.5)

for i in range(len(step2_valid)):
    ax2.plot([step2_img_x[i], step2_cat_x[i]], [step2_img_y[i], step2_cat_y[i]], 
            'yellow', linewidth=0.5, alpha=0.8)
ax2.scatter(step2_img_x, step2_img_y, c='yellow', s=8, marker='+', linewidths=0.5)
ax2.scatter(step2_cat_x, step2_cat_y, c='yellow', s=8, marker='+', linewidths=0.5)

ax2.set_xlim(0, width)
ax2.set_ylim(0, height)
ax2.axis('off')

output2 = r"F:\Astro dev\Astro CS Normalization Database\debug_step2_alignment.png"
plt.savefig(output2, dpi=100, bbox_inches='tight', pad_inches=0)
plt.close(fig2)

print(f"Step2 saved: {output2}")
print(f"  Gaia predictions (top 1000): {len(gaia_data)}")
print(f"  Step2 matches: {len(step2_valid)}")

step2_res_x = step2_cat_x - step2_img_x
step2_res_y = step2_cat_y - step2_img_y
step2_total = np.sqrt(step2_res_x**2 + step2_res_y**2)

print(f"\n=== Step2残差统计 ===")
print(f"  X: mean={np.mean(step2_res_x):.3f}, std={np.std(step2_res_x):.3f}")
print(f"  Y: mean={np.mean(step2_res_y):.3f}, std={np.std(step2_res_y):.3f}")
print(f"  Total: mean={np.mean(step2_total):.3f}, std={np.std(step2_total):.3f}")

print(f"\n标注说明:")
print(f"  Step1: 红色十字 = 前1000颗Gaia亮星预测位置")
print(f"  Step2: 红色十字 = 前1000颗Gaia亮星预测位置")
print(f"         黄色十字 = 匹配对的img位置和cat位置")
print(f"         黄色细线 = 连接匹配对")
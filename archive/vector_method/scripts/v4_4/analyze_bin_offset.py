"""分析所有帧的 bin 中心 vs 真值偏离，以及候选提取过滤范围是否足够"""
import csv

BASE = r"f:/Astro dev/Astro CS Normalization Database/lib/plate_solve/logs/v4_4/exp_relvec_test"

# (frame_dir, ransac_theta_true, tx_true, ty_true, status)
FRAMES = [
    ("Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red", -43.4, -646, -1375, "FAIL"),
    ("Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red", 152.6, -688, -3368, "OK"),
    ("LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum", -76.8, 490, 229, "OK"),
    ("LDN43_LRGBH_flying_dutchman-20250520@022502-600S-Lum", -94.4, 386, 175, "OK"),
    ("M20_T2_flying_dutchman-20250701@073331-300S-Red", 17.0, 299, -256, "FAIL"),
    ("M20_T2_flying_dutchman-20250719@004357-300S-Red", 178.6, -211, 241, "FAIL"),
    ("M20_T2_flying_dutchman-20250813@021541-300S-Red", 0.8, -48, 168, "FAIL"),
    ("NGC247_T2_flying_dutchman-20250816@033428-600S-Lum", -65.7, 1862, 422, "FAIL"),
    ("NGC55_T3_flying_dutchman-20250701@074114-600S-Red", 174.9, -1, -2, "FAIL"),
    ("LDN43_LRGBH_flying_dutchman-20250503@042947-1200S-H-alpha", 15.2, -555, -738, "OK"),
]

print(f"{'frame':<55} {'st':<5} {'th_rot':>7} {'pk_th':>7} {'tx_t':>8} {'pk_tx':>8} {'ty_t':>8} {'pk_ty':>8} {'dth':>5} {'dtx':>6} {'dty':>6}")
for frame, th_true, tx_t, ty_t, status in FRAMES:
    path = f"{BASE}/{frame}/focus_history.csv"
    try:
        rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    except FileNotFoundError:
        print(f"{frame}: NOT FOUND")
        continue
    last = rows[-1]
    pk_th = float(last['peak_theta'])
    pk_tx = float(last['peak_tx'])
    pk_ty = float(last['peak_ty'])
    th_rot = -th_true
    dth = abs(pk_th - th_rot)
    dtx = abs(pk_tx - tx_t)
    dty = abs(pk_ty - ty_t)
    name = frame[:55]
    print(f"{name:<55} {status:<5} {th_rot:>7.1f} {pk_th:>7.1f} {tx_t:>8.1f} {pk_tx:>8.1f} {ty_t:>8.1f} {pk_ty:>8.1f} {dth:>5.2f} {dtx:>6.1f} {dty:>6.1f}")

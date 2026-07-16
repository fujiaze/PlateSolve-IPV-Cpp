"""分析失败帧的 passed_pairs 聚集情况"""
import csv
import statistics
import collections

BASE = r"f:/Astro dev/Astro CS Normalization Database/lib/plate_solve/logs/v4_4/exp_relvec_test"

# (frame_dir, ransac_theta_true_deg, tx_true, ty_true)
# 注意: passed_pairs 的 theta_deg 是 θ_rot (采样域), θ_rot = -θ_true
FRAMES = [
    ("LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum", -76.8, 490, 229),
    ("M20_T2_flying_dutchman-20250701@073331-300S-Red", 17.0, 299, -256),
    ("Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red", -43.4, -646, -1375),
    ("NGC247_T2_flying_dutchman-20250816@033428-600S-Lum", -65.7, 1862, 422),
    ("NGC55_T3_flying_dutchman-20250701@074114-600S-Red", 174.9, -1, -2),
]

for frame, th_true, tx_true, ty_true in FRAMES:
    path = f"{BASE}/{frame}/passed_pairs.csv"
    try:
        rows = list(csv.DictReader(open(path, encoding='utf-8')))
    except FileNotFoundError:
        print(f"{frame}: NOT FOUND")
        continue
    d = [[float(r['theta_deg']), float(r['tx']), float(r['ty']), int(r['n_k_passed'])] for r in rows]
    th_rot = -th_true  # θ_rot = -θ_true
    print(f"\n=== {frame} ===")
    print(f"  total passed: {len(d)}, RANSAC: th_true={th_true} -> th_rot={th_rot} tx={tx_true} ty={ty_true}")

    # θ_rot 区内
    th_near = [x for x in d if abs(x[0] - th_rot) < 3]
    print(f"  th±3 near true: {len(th_near)}")
    if not th_near:
        continue
    kp = collections.Counter(x[3] for x in th_near)
    print(f"  k_passed dist: {dict(sorted(kp.items()))}")

    # tx/ty 分布
    txs = [x[1] for x in th_near]
    tys = [x[2] for x in th_near]
    print(f"  all k: tx med={statistics.median(txs):.1f} [{min(txs):.1f},{max(txs):.1f}]  "
          f"ty med={statistics.median(tys):.1f} [{min(tys):.1f},{max(tys):.1f}]")

    # k>=3
    hi3 = [x for x in th_near if x[3] >= 3]
    if hi3:
        txs2 = [x[1] for x in hi3]
        tys2 = [x[2] for x in hi3]
        print(f"  k>=3 ({len(hi3)}): tx med={statistics.median(txs2):.1f} [{min(txs2):.1f},{max(txs2):.1f}]  "
              f"ty med={statistics.median(tys2):.1f} [{min(tys2):.1f},{max(tys2):.1f}]")

    # 真值附近 (用 θ_rot)
    near = [x for x in d if abs(x[0] - th_rot) < 3 and abs(x[1] - tx_true) < 30 and abs(x[2] - ty_true) < 30]
    print(f"  near true (th_rot±3,tx±30,ty±30): {len(near)}")
    if near:
        for n in near[:8]:
            print(f"    th={n[0]:.2f} tx={n[1]:.1f} ty={n[2]:.1f} k={n[3]}")

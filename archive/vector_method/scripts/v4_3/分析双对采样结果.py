"""分析双对采样CSV结果"""
import pandas as pd
import numpy as np
import os

OUT = r"f:\Astro dev\Astro CS Normalization Database\lib\plate_solve\logs\v4_3\dual_param_space"

cases = [
    ("Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red", "成功", 1, 179.98, -63.7, 162.3),
    ("Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red", "Type3失败", 2, -179.80, 98.7, 76.5),
    ("NGC4945_FD_T3_flying_dutchman-20250206@043838-600S-Lum", "Type1失败", 0, -23.73, 32.8, -27.5),
]

for name, label, svm, svt, svtx, svty in cases:
    path = os.path.join(OUT, name + "_dual_transforms.csv")
    df = pd.read_csv(path)
    print(f"\n{'='*60}")
    print(f"=== {label}: {name[:50]} ===")
    print(f"Solver: mode={svm} theta={svt} deg tx={svtx} ty={svty}")
    print(f"总变换数: {len(df)}")

    for mode in range(4):
        dm = df[df["mode"] == mode]
        if len(dm) == 0:
            continue
        high = dm[dm["n_in_range"] >= 5]
        tag = "  <-- SOLVER MODE" if mode == svm else ""
        print(f"\n  mode{mode}: n={len(dm)}, n>=5: {len(high)}, max={dm.n_in_range.max()}{tag}")

        if len(high) >= 3:
            w = high["n_in_range"].values.astype(float)
            th = high["theta_deg"].values
            tx = high["tx"].values
            ty = high["ty"].values
            th_m = np.average(th, weights=w)
            th_s = np.sqrt(np.average((th - th_m) ** 2, weights=w))
            tx_m = np.average(tx, weights=w)
            tx_s = np.sqrt(np.average((tx - tx_m) ** 2, weights=w))
            ty_m = np.average(ty, weights=w)
            ty_s = np.sqrt(np.average((ty - ty_m) ** 2, weights=w))
            print(f"    加权均值: th={th_m:.2f}deg (std={th_s:.2f})  tx={tx_m:.1f} (std={tx_s:.1f})  ty={ty_m:.1f} (std={ty_s:.1f})")
            if mode == svm:
                print(f"    vs Solver: dth={abs(th_m - svt):.2f}deg  dtx={abs(tx_m - svtx):.1f}  dty={abs(ty_m - svty):.1f}")

    # Top10
    top = df.nlargest(10, "n_in_range")
    print(f"\n  Top10:")
    for _, r in top.iterrows():
        star = " *" if r["mode"] == svm else "  "
        print(f"   {star} mode={r['mode']:.0f} th={r['theta_deg']:.2f} tx={r['tx']:.1f} ty={r['ty']:.1f} n={r['n_in_range']:.0f}")

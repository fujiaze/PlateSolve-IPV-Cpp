# -*- coding: utf-8 -*-
"""功能: M20_T2 单帧冒烟测试 + CRVAL 偏差验证
用途: fix-adv-pa-phase1-bugs 修复后验证 CRVAL < 30"
"""
import os, sys
_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v2.tests.test_phase1 import run_one_frame, TEST_FRAMES

frame = TEST_FRAMES[0]  # M20_T2
rec = run_one_frame(frame)

print("\n" + "=" * 70)
print("冒烟测试结论:")
print("=" * 70)
if rec["success"]:
    print(f"  success=True, RMS={rec['rms']:.3f}\", vote_peak={rec['vote_peak']}")
    if rec["crval_dev"] is not None:
        status = "通过" if rec["crval_dev"] < 30.0 else "未通过"
        print(f"  CRVAL偏差={rec['crval_dev']:.2f}\" (期望<30\") → {status}")
    else:
        print(f"  CRVAL偏差=N/A (无header WCS)")
else:
    print(f"  success=False: {rec['message']}")
print(f"  耗时={rec['total_time']:.3f}s")

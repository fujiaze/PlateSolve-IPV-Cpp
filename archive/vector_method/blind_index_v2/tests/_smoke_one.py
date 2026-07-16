# -*- coding: utf-8 -*-
"""功能: 单帧冒烟测试 — 验证 solve_blind 在第一帧(M20_T2)上能正常工作
用途: 调试Phase 1测试, 确保导入/DR3/检测/投票/验证链路畅通后再跑全量
"""
import os, sys, time
_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index_v2.python.pipeline import solve_blind
from lib.plate_solve.blind_index_v2.python import logging_setup

logging_setup.setup_logging()

path = os.path.join(_PROJECT_ROOT, r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts")
print(f"冒烟测试: {os.path.basename(path)}")
t0 = time.time()
result = solve_blind(
    image_path=path,
    s0_arcsec_per_pixel=None,
    query_center_ra=None,
    query_center_dec=None,
    mag_limit=12.0,
)
elapsed = time.time() - t0
print(f"\n===== 结果 =====")
print(f"success={result.success}, 耗时={elapsed:.2f}s (wall), stage总={sum(result.stage_timings.values()):.2f}s")
print(f"s0={result.s0_arcsec_per_pixel:.4f}, n_detected={result.n_detected}, n_ref={result.n_reference}")
print(f"n_pairs={result.n_pairs}, n_image_pairs={result.n_image_pairs}")
print(f"vote_peak={result.vote_peak}, n_candidates={result.n_candidates}, candidates_tried={result.candidates_tried}")
print(f"RMS={result.best_rms_arcsec:.3f}")
if result.wcs:
    print(f"CRVAL=({result.wcs.crval1:.5f}, {result.wcs.crval2:.5f})")
    print(f"CD=[{result.wcs.cd[0,0]:.6e}, {result.wcs.cd[0,1]:.6e}; {result.wcs.cd[1,0]:.6e}, {result.wcs.cd[1,1]:.6e}]")
    print(f"n_inliers={result.wcs.n_inliers}")
print(f"message={result.message}")
print(f"timings={result.stage_timings}")

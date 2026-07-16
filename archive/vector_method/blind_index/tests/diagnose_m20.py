# -*- coding: utf-8 -*-
"""
M20 帧调试诊断脚本
功能: 诊断 M20_T2_Green 帧 k-vector 返回 0 候选的根因
用途: 对比图像/参考 d_AB 分布, 验证图像亮星是否对应 Gaia 参考星

假设:
    - 图像 top-N 亮星聚集在 M20 星云核心 → 小最近邻距离 → 小 d_AB
    - Gaia 参考星在 FOV 内均匀分布 → 大最近邻距离 → 大 d_AB
    - 两者 d_AB 分布不重叠 → k-vector 无候选

诊断输出:
    1. 参考 d_AB 分布(min/p10/p25/p50/p75/p90/max + 直方图)
    2. 图像 d_AB 分布(同上, 比较不同 pool_size)
    3. 图像 top-N 亮星 vs Gaia FOV 星位置匹配率
       (用 s0+CRVAL+WCS 反推图像星的天球位置, 与 Gaia 比对)
"""
from __future__ import annotations

import os
import sys

import numpy as np

# 项目根目录
_PROJECT_ROOT = os.path.normpath(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index.python.io_wrappers import (
    read_image, detect_stars, query_dr3, get_pixel_scale_from_header,
    get_pointing_from_header,
)
from lib.plate_solve.blind_index.python.quad_selector import (
    generate_image_quads, generate_reference_quads,
)
from lib.plate_solve.blind_index.python.kvector import build_kvector, kvector_query
from lib.plate_solve.blind_index.python.logging_setup import setup_logging
from lib.plate_solve.python.vector_match_v2 import gnomonic_forward, gnomonic_inverse


# 测试帧配置 (可通过命令行参数选择)
import sys as _sys
FRAME_NAME = _sys.argv[1] if len(_sys.argv) > 1 else "M20"

FRAMES = {
    "M20": {
        "path": r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts",
        "s0": 0.966883202701463,
        "query_ra": 270.700029168,
        "query_dec": -22.849924116,
    },
    "LDN43": {
        "path": r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum.fts",
        "s0": 0.9668267272256788,
        "query_ra": 248.60953524,
        "query_dec": -15.75893659,
    },
    "NGC55": {
        "path": r"testdata\lights\NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts",
        "s0": 0.9893,
        "query_ra": 3.7,
        "query_dec": -39.2,
    },
}

if FRAME_NAME not in FRAMES:
    print(f"未知帧: {FRAME_NAME}, 可选: {list(FRAMES.keys())}")
    _sys.exit(1)

_cfg = FRAMES[FRAME_NAME]
IMAGE_PATH = os.path.join(_PROJECT_ROOT, _cfg["path"])
S0 = _cfg["s0"]
QUERY_RA = _cfg["query_ra"]
QUERY_DEC = _cfg["query_dec"]
MAG_LIMIT = 12.0


def print_distribution(name: str, arr: np.ndarray, bins: int = 20):
    """打印数组的统计分布"""
    if len(arr) == 0:
        print(f"  [{name}] 空")
        return
    pcts = [0, 10, 25, 50, 75, 90, 100]
    vals = np.percentile(arr, pcts)
    print(f"  [{name}] N={len(arr)}, min={vals[0]:.2f}, p10={vals[1]:.2f}, "
          f"p25={vals[2]:.2f}, p50={vals[3]:.2f}, p75={vals[4]:.2f}, "
          f"p90={vals[5]:.2f}, max={vals[6]:.2f}")
    # 直方图
    hist, edges = np.histogram(arr, bins=bins)
    print(f"    直方图 ({bins} bins, range=[{edges[0]:.1f}, {edges[-1]:.1f}]):")
    for i, c in enumerate(hist):
        bar = "█" * min(60, c)
        print(f"      [{edges[i]:7.1f}, {edges[i+1]:7.1f}): {c:5d} {bar}")


def main():
    setup_logging(log_to_console=False)

    print("=" * 80)
    print(f"{FRAME_NAME} 帧 d_AB 分布诊断")
    print("=" * 80)
    print(f"图像: {IMAGE_PATH}")
    print(f"s0={S0:.4f} arcsec/pix, query=({QUERY_RA:.5f}, {QUERY_DEC:.5f})")
    print(f"mag_limit={MAG_LIMIT}")

    # ═══ 1. 读取图像 + 星点检测 ═══
    print("\n[1] 读取图像 + 星点检测")
    uint16_img, metadata = read_image(IMAGE_PATH)
    h, w = uint16_img.shape
    print(f"  图像尺寸: {w}x{h}")

    star_result = detect_stars(uint16_img)
    print(f"  检测星点: {star_result.count} 颗")
    print(f"  其中饱和: {int(np.sum(star_result.saturated))} 颗")

    # ═══ 2. 查询 DR3 ═══
    print("\n[2] 查询 Gaia DR3 (mag<12)")
    fov_diag_arcsec = float(np.sqrt(w**2 + h**2) * S0)
    radius_deg = (fov_diag_arcsec / 3600.0) * 1.5
    print(f"  FOV 对角线={fov_diag_arcsec/3600:.4f}°, 查询半径={radius_deg:.4f}°")

    ra_arr, dec_arr, mag_arr = query_dr3(QUERY_RA, QUERY_DEC, radius_deg, MAG_LIMIT)
    print(f"  DR3 返回: {len(ra_arr)} 颗 (mag<{MAG_LIMIT})")

    # ═══ 3. gnomonic 投影 ═══
    print("\n[3] gnomonic 投影到切平面")
    xi_arr, eta_arr, valid = gnomonic_forward(ra_arr, dec_arr, QUERY_RA, QUERY_DEC)
    xi_arr = xi_arr[valid]
    eta_arr = eta_arr[valid]
    ra_arr = ra_arr[valid]
    dec_arr = dec_arr[valid]
    mag_arr = mag_arr[valid]
    print(f"  有效参考星: {len(ra_arr)} 颗")

    # FOV 过滤
    half_w_arcsec = (w / 2.0) * S0
    half_h_arcsec = (h / 2.0) * S0
    fov_mask = (np.abs(xi_arr) <= half_w_arcsec) & (np.abs(eta_arr) <= half_h_arcsec)
    n_ref_in_fov = int(np.sum(fov_mask))
    xi_fov = xi_arr[fov_mask]
    eta_fov = eta_arr[fov_mask]
    ra_fov = ra_arr[fov_mask]
    dec_fov = dec_arr[fov_mask]
    mag_fov = mag_arr[fov_mask]
    print(f"  FOV 内参考星: {n_ref_in_fov}/{len(ra_arr)} 颗")
    print(f"  FOV 半宽={half_w_arcsec:.1f}\", 半高={half_h_arcsec:.1f}\"")
    if len(mag_fov) > 0:
        print(f"  FOV 内参考星 星等: min={mag_fov.min():.2f}, max={mag_fov.max():.2f}, "
              f"median={np.median(mag_fov):.2f}")

    # ═══ 4. 参考四边形 + d_AB 分布 ═══
    print("\n[4] 参考四边形 d_AB 分布 (FOV 内 338 颗)")
    ref_quads = generate_reference_quads(xi_fov, eta_fov, ra_fov, dec_fov)
    ref_d_ab = np.array([q.distances[0] for q in ref_quads])
    print(f"  参考四边形: {len(ref_quads)} 个")
    print_distribution("参考 d_AB", ref_d_ab, bins=20)

    # ═══ 5. 图像 d_AB 分布 - 不同 pool_size 对比 ═══
    print("\n[5] 图像四边形 d_AB 分布 - 不同 pool_size 对比")
    for pool_size in [20, 50, 100, 200, 338, 500, 1000]:
        if pool_size > star_result.count:
            continue
        image_quads = generate_image_quads(
            star_result.x, star_result.y, S0,
            ref_kvector_index=None,  # 不做 uniqueness 评分
            pool_size=pool_size,
        )
        if len(image_quads) == 0:
            print(f"\n  pool_size={pool_size}: 0 四边形")
            continue
        img_d_ab = np.array([q.d_AB for q in image_quads])
        print(f"\n  pool_size={pool_size}: {len(image_quads)} 四边形 (top-5 保留)")
        print_distribution(f"图像 d_AB (pool={pool_size})", img_d_ab, bins=10)
        # 检查与参考 d_AB 重叠
        if len(ref_d_ab) > 0:
            ref_min, ref_max = ref_d_ab.min(), ref_d_ab.max()
            img_in_ref_range = np.sum((img_d_ab >= ref_min) & (img_d_ab <= ref_max))
            print(f"    落在参考 d_AB 范围 [{ref_min:.1f}, {ref_max:.1f}] 内: "
                  f"{img_in_ref_range}/{len(img_d_ab)}")

    # ═══ 6. 图像亮星 vs Gaia FOV 星位置匹配 (测试两种 Y 方向) ═══
    print("\n[6] 图像 top-N 亮星 vs Gaia FOV 星位置匹配 (测试 Y-flip)")
    print("  使用 FITS 头 WCS 将图像星转到天球坐标, 与 Gaia 比对")
    if metadata.wcs is None or not metadata.wcs.has_wcs:
        print("  [跳过] FITS 头无 WCS")
    else:
        wcs = metadata.wcs
        # WCSKeywordsPy 用 cd1_1/cd1_2/cd2_1/cd2_2 (不是 cd 矩阵)
        print(f"  WCS: CRVAL=({wcs.crval1:.5f}, {wcs.crval2:.5f}), "
              f"CRPIX=({wcs.crpix1:.2f}, {wcs.crpix2:.2f})")
        print(f"  CD=[{wcs.cd1_1:.6e}, {wcs.cd1_2:.6e}; "
              f"{wcs.cd2_1:.6e}, {wcs.cd2_2:.6e}]")
        print(f"  pixel_scale={wcs.pixel_scale:.6f}, rotation={wcs.rotation_deg:.4f}°")

        from scipy.spatial import cKDTree
        gaia_pts = np.column_stack([ra_fov, dec_fov])
        gaia_tree = cKDTree(gaia_pts)
        cos_dec = np.cos(np.radians(QUERY_DEC))

        x = star_result.x
        y = star_result.y
        # 测试两种 Y 方向: 原始 vs Y-flipped (y_wcs = NAXIS2+1 - y)
        for y_flip in [False, True]:
            print(f"\n  --- Y方向: {'FLIPPED (y_wcs=NAXIS2+1-y)' if y_flip else '原始 (y直接用)'} ---")
            if y_flip:
                y_wcs = (h + 1) - y  # FITS 1-indexed: NAXIS2+1-y
            else:
                y_wcs = y

            dx = x - wcs.crpix1
            dy = y_wcs - wcs.crpix2
            delta_ra = wcs.cd1_1 * dx + wcs.cd1_2 * dy
            delta_dec = wcs.cd2_1 * dx + wcs.cd2_2 * dy
            img_ra = wcs.crval1 + delta_ra
            img_dec = wcs.crval2 + delta_dec

            # 限制图像星在 FOV 内 (WCS 边界 ±1°)
            in_fov = (img_ra > QUERY_RA - 1.0) & (img_ra < QUERY_RA + 1.0) & \
                     (img_dec > QUERY_DEC - 1.0) & (img_dec < QUERY_DEC + 1.0)
            print(f"  图像星 WCS 投影在 FOV ±1° 内: {int(np.sum(in_fov))}/{len(x)}")

            for n_top in [20, 50, 100, 338, 1000]:
                if n_top > star_result.count:
                    continue
                top_idx = np.arange(n_top)
                top_ra = img_ra[top_idx]
                top_dec = img_dec[top_idx]
                img_pts = np.column_stack([top_ra, top_dec])
                dists, _ = gaia_tree.query(img_pts, k=1)
                dists_arcsec = dists * 3600.0 * cos_dec
                n_matched = int(np.sum(dists_arcsec < 3.0))
                n_matched_5 = int(np.sum(dists_arcsec < 5.0))
                print(f"  top-{n_top:4d} 亮星: 3\"内={n_matched:4d} ({n_matched/n_top*100:5.1f}%), "
                      f"5\"内={n_matched_5:4d} ({n_matched_5/n_top*100:5.1f}%)")
                if len(dists_arcsec) > 0 and n_top in [20, 338]:
                    print(f"    最近邻距离: min={dists_arcsec.min():.2f}\", "
                          f"p50={np.percentile(dists_arcsec, 50):.2f}\"")

    # ═══ 7. 综合结论 ═══
    print("\n" + "=" * 80)
    print("[综合结论]")
    print("=" * 80)
    if len(ref_d_ab) > 0:
        ref_min, ref_max = ref_d_ab.min(), ref_d_ab.max()
        print(f"参考 d_AB 范围: [{ref_min:.1f}, {ref_max:.1f}]\"")
    print("\n请检查:")
    print("  1. 图像 d_AB 是否与参考 d_AB 范围重叠?")
    print("  2. 图像 top-N 亮星是否高比例匹配 Gaia FOV 星?")
    print("  3. 若匹配率低 → 图像检测可能包含非星点 artifact")
    print("  4. 若匹配率高但 d_AB 不重叠 → 图像亮星空间分布与 Gaia 不同")


if __name__ == "__main__":
    main()

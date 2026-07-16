# -*- coding: utf-8 -*-
"""
诊断脚本: 深入分析投票机制 — 为什么投票到错误的天区格
检查:
  1. header WCS识别真匹配 → k-vector是否找到这些真匹配
  2. 真匹配投票到哪个(天区,rot_bin)
  3. 假阳性峰值(cell #35498)的44票来自哪里
  4. PA_cat和θ_img的方向性分析
"""
from __future__ import annotations

import os
import sys
import math
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, _PROJECT_ROOT)

from lib.astro_image_io.python.astro_image_io import ImageReader
from lib.plate_solve.blind_index_v2.python.io_wrappers import (
    read_image, detect_stars, query_dr3, StarDetectionResult,
)
from lib.plate_solve.blind_index_v2.python.pair_index import (
    build_pair_library_with_kvector, kvector_query, PairLibrary,
    DEFAULT_K_NEIGHBORS, DEFAULT_DELTA, DEFAULT_D_MIN, DEFAULT_D_MAX,
)
from lib.plate_solve.blind_index_v2.python.image_features import extract_image_pairs
from lib.plate_solve.blind_index_v2.python.voting import (
    vote, detect_peaks, pix2ang, ang2pix, rot_bin_to_angle,
    DEFAULT_SIGMA_POS, DEFAULT_N_SIGMA, DEFAULT_ROT_BIN_DEG,
    _RA_BINS, _DEC_BINS,
)
from lib.plate_solve.blind_index_v2.python.pipeline import _select_brightest_stars
from lib.plate_solve.python.vector_match_v2 import (
    gnomonic_forward, gnomonic_inverse,
)
from lib.plate_solve.blind_index_v2.python.spherical_geom import (
    angular_distance_arcsec, position_angle_deg,
)

_DEGTORAD = math.pi / 180.0
_RADTOASEC = (180.0 / math.pi) * 3600.0


def haversine_arcsec(ra1, dec1, ra2, dec2):
    ra1r, dec1r = ra1 * _DEGTORAD, dec1 * _DEGTORAD
    ra2r, dec2r = ra2 * _DEGTORAD, dec2 * _DEGTORAD
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = math.sin(ddec/2.0)**2 + math.cos(dec1r)*math.cos(dec2r)*math.sin(dra/2.0)**2
    a = max(0.0, min(1.0, a))
    return 2.0*math.asin(math.sqrt(a))*_RADTOASEC


def main():
    image_path = os.path.join(_PROJECT_ROOT,
        r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts")

    print("=" * 70)
    print("投票机制深入诊断")
    print("=" * 70)

    # 1. 读取header WCS + 图像 + 检测星点
    reader = ImageReader()
    img_data = reader.read(image_path)
    meta = img_data.metadata
    w, h = meta.geometry.width, meta.geometry.height
    hdr_cd = np.array([[meta.wcs.cd1_1, meta.wcs.cd1_2],
                        [meta.wcs.cd2_1, meta.wcs.cd2_2]])
    hdr_crpix = (meta.wcs.crpix1, meta.wcs.crpix2)
    hdr_crval = (meta.wcs.crval1, meta.wcs.crval2)
    s0 = float(np.sqrt(abs(np.linalg.det(hdr_cd))) * 3600.0)
    print(f"图像: {w}x{h}, s0={s0:.4f}\"/px")
    print(f"Header CRVAL=({hdr_crval[0]:.5f}, {hdr_crval[1]:.5f})")

    uint16_img, _ = read_image(image_path)
    star_result = detect_stars(uint16_img)
    print(f"检测星点: {star_result.count}")

    # 2. 查询DR3 + 构建k-vector (与管线一致)
    fov_diag_deg = math.sqrt(w**2 + h**2) * s0 / 3600.0
    radius_deg = fov_diag_deg * 1.5
    ra_arr, dec_arr, mag_arr = query_dr3(hdr_crval[0], hdr_crval[1], radius_deg, 12.0)
    print(f"DR3查询: {len(ra_arr)}星 (半径={radius_deg:.3f}°, mag_limit=12)")

    kv = build_pair_library_with_kvector(
        ra_arr, dec_arr, mag_arr,
        k_neighbors=DEFAULT_K_NEIGHBORS, delta=DEFAULT_DELTA,
        d_min=DEFAULT_D_MIN, d_max=DEFAULT_D_MAX,
    )
    print(f"星对库: {kv.n_pairs}对")

    # 3. 用header WCS将图像星投影到RA/Dec, 找真匹配
    x_arr, y_arr = star_result.x, star_result.y
    dx = x_arr - (hdr_crpix[0] - 1.0)
    dy = y_arr - (hdr_crpix[1] - 1.0)
    xi_deg = hdr_cd[0,0]*dx + hdr_cd[0,1]*dy
    eta_deg = hdr_cd[1,0]*dx + hdr_cd[1,1]*dy
    img_ra, img_dec = gnomonic_inverse(xi_deg*3600.0, eta_deg*3600.0, hdr_crval[0], hdr_crval[1])

    # catalog在切平面
    xi_cat, eta_cat, valid_cat = gnomonic_forward(ra_arr, dec_arr, hdr_crval[0], hdr_crval[1])
    cat_xy = np.column_stack([xi_cat[valid_cat], eta_cat[valid_cat]])
    img_xy_tan = np.column_stack([xi_deg*3600.0, eta_deg*3600.0])
    tree = cKDTree(cat_xy)
    match_thresh = 3.0 * s0  # 3" 匹配阈值
    dists, idxs = tree.query(img_xy_tan, k=1)
    matched_mask = dists < match_thresh
    valid_idx = np.where(valid_cat)[0]
    img_matched_idx = np.where(matched_mask)[0]
    cat_matched_idx = valid_idx[idxs[matched_mask]]
    print(f"\n真匹配 (header WCS, 阈值={match_thresh:.1f}\"): {len(img_matched_idx)}/{star_result.count}")

    # 4. 选取Top-100最亮星 (与管线一致)
    x_sel, y_sel = _select_brightest_stars(star_result, 100)
    # 建立选中星在原始数组中的索引
    # _select_brightest_stars返回的是坐标, 需要找到原始索引
    # 简化: 直接用全部星的子集
    sel_mask = np.zeros(star_result.count, dtype=bool)
    for i in range(len(x_sel)):
        # 找到匹配的原始索引
        diffs = (star_result.x - x_sel[i])**2 + (star_result.y - y_sel[i])**2
        sel_mask[np.argmin(diffs)] = True
    sel_orig_idx = np.where(sel_mask)[0]
    print(f"Top-100最亮星: {len(sel_orig_idx)}颗")

    # 5. 提取图像星对特征
    d_img_arr, theta_img_arr, idx_i_arr, idx_j_arr = extract_image_pairs(x_sel, y_sel, s0)
    print(f"图像星对: {len(d_img_arr)}")

    # 6. 检查真匹配在Top-100中的比例
    top100_set = set(sel_orig_idx.tolist())
    true_match_in_top100 = [i for i in img_matched_idx if i in top100_set]
    print(f"真匹配在Top-100中: {len(true_match_in_top100)}/{len(img_matched_idx)}")

    # 7. 投票 (与管线一致)
    sigma_d = DEFAULT_SIGMA_POS * s0
    delta_d = DEFAULT_N_SIGMA * sigma_d
    votes = vote(d_img_arr, theta_img_arr, kv, s0, DEFAULT_SIGMA_POS, DEFAULT_N_SIGMA)
    total_votes = sum(votes.values())
    print(f"\n投票完成: 总票数={total_votes}, 投票格={len(votes)}")

    # 8. 峰值检测
    peaks = detect_peaks(votes, len(d_img_arr), top_k=10)
    print(f"\nTop-10峰值:")
    for i, (pix, rot_bin, count) in enumerate(peaks):
        ra_c, dec_c = pix2ang(pix)
        rot_ang = rot_bin_to_angle(rot_bin)
        print(f"  #{i+1}: cell={pix} ({ra_c:.4f}, {dec_c:.4f}), rot_bin={rot_bin} ({rot_ang:.1f}°), 票数={count}")

    # 9. 真天区格的票数
    true_cell = ang2pix(hdr_crval[0], hdr_crval[1])
    ra_tc, dec_tc = pix2ang(true_cell)
    print(f"\n真天区格: cell={true_cell} ({ra_tc:.4f}, {dec_tc:.4f})")
    print(f"  header指向: ({hdr_crval[0]:.5f}, {hdr_crval[1]:.5f})")

    true_cell_votes = {rb: c for (p, rb), c in votes.items() if p == true_cell}
    if true_cell_votes:
        print(f"  真天区格的投票 (按rot_bin):")
        for rb in sorted(true_cell_votes.keys()):
            print(f"    rot_bin={rb} ({rot_bin_to_angle(rb):.1f}°): {true_cell_votes[rb]}票")
    else:
        print(f"  真天区格无投票!")

    # 10. 分析真匹配的投票
    print(f"\n{'=' * 70}")
    print("真匹配投票分析 (Top-100中的真匹配对):")
    print(f"{'=' * 70}")

    # 对Top-100中的真匹配, 检查k-vector查询和投票
    # img_matched_idx在原始数组中, 需要映射到Top-100的索引
    orig_to_top100 = {orig: top for top, orig in enumerate(sel_orig_idx)}
    true_match_top100 = [orig_to_top100[i] for i in true_match_in_top100 if i in orig_to_top100]
    true_match_top100_set = set(true_match_top100)

    n_true_pairs = 0
    n_found_in_kv = 0
    n_voted_true_cell = 0
    true_rot_bins = []

    for ii in range(len(idx_i_arr)):
        i_top = int(idx_i_arr[ii])
        j_top = int(idx_j_arr[ii])
        if i_top not in true_match_top100_set or j_top not in true_match_top100_set:
            continue
        # 这两个图像星都有真匹配
        # 找到对应的catalog星
        i_orig = sel_orig_idx[i_top]
        j_orig = sel_orig_idx[j_top]
        # 在img_matched_idx中找
        i_match_pos = np.where(img_matched_idx == i_orig)[0]
        j_match_pos = np.where(img_matched_idx == j_orig)[0]
        if len(i_match_pos) == 0 or len(j_match_pos) == 0:
            continue
        I_cat = int(cat_matched_idx[i_match_pos[0]])
        J_cat = int(cat_matched_idx[j_match_pos[0]])

        d_img = float(d_img_arr[ii])
        theta_img = float(theta_img_arr[ii])
        d_cat = float(angular_distance_arcsec(ra_arr[I_cat], dec_arr[I_cat], ra_arr[J_cat], dec_arr[J_cat]))

        n_true_pairs += 1

        # k-vector查询
        idx_lo, idx_hi = kvector_query(kv, d_img, delta_d)
        if idx_lo > idx_hi:
            print(f"  真匹配对{n_true_pairs}: d_img={d_img:.2f}\", d_cat={d_cat:.2f}\" → k-vector无候选!")
            continue

        # 检查catalog pair (I,J)或(J,I)是否在k-vector结果中
        S = kv.S[idx_lo:idx_hi+1]
        tol = 1e-6
        match_IJ = np.where(
            (np.abs(S['ra_i']-ra_arr[I_cat])<tol) & (np.abs(S['dec_i']-dec_arr[I_cat])<tol) &
            (np.abs(S['ra_j']-ra_arr[J_cat])<tol) & (np.abs(S['dec_j']-dec_arr[J_cat])<tol)
        )[0]
        match_JI = np.where(
            (np.abs(S['ra_i']-ra_arr[J_cat])<tol) & (np.abs(S['dec_i']-dec_arr[J_cat])<tol) &
            (np.abs(S['ra_j']-ra_arr[I_cat])<tol) & (np.abs(S['dec_j']-dec_arr[I_cat])<tol)
        )[0]

        found_ij = len(match_IJ) > 0
        found_ji = len(match_JI) > 0
        found = found_ij or found_ji
        if found:
            n_found_in_kv += 1

        # 计算真匹配应该投票到的位置
        if found_ij:
            pa_cat = float(S['ra_i'][match_IJ[0]]), float(S['dec_i'][match_IJ[0]])
            pa_val = position_angle_deg(ra_arr[I_cat], dec_arr[I_cat], ra_arr[J_cat], dec_arr[J_cat])
            rot = (theta_img - pa_val) % 360.0
            rot_bin = int(rot / DEFAULT_ROT_BIN_DEG)
            vote_cell = ang2pix(ra_arr[I_cat], dec_arr[I_cat])
            true_rot_bins.append(rot)
            if vote_cell == true_cell:
                n_voted_true_cell += 1
            print(f"  真匹配对{n_true_pairs} (I→J): d_img={d_img:.2f}\", d_cat={d_cat:.2f}\", "
                  f"θ_img={theta_img:.1f}°, PA_cat={pa_val:.1f}°, rot={rot:.1f}° (bin={rot_bin}), "
                  f"投票cell={vote_cell}({'真' if vote_cell==true_cell else '非真'})")
        elif found_ji:
            pa_val = position_angle_deg(ra_arr[J_cat], dec_arr[J_cat], ra_arr[I_cat], dec_arr[I_cat])
            rot = (theta_img - pa_val) % 360.0
            rot_bin = int(rot / DEFAULT_ROT_BIN_DEG)
            vote_cell = ang2pix(ra_arr[J_cat], dec_arr[J_cat])
            true_rot_bins.append(rot)
            if vote_cell == true_cell:
                n_voted_true_cell += 1
            print(f"  真匹配对{n_true_pairs} (J→I): d_img={d_img:.2f}\", d_cat={d_cat:.2f}\", "
                  f"θ_img={theta_img:.1f}°, PA_cat={pa_val:.1f}°, rot={rot:.1f}° (bin={rot_bin}), "
                  f"投票cell={vote_cell}({'真' if vote_cell==true_cell else '非真'})")
        else:
            print(f"  真匹配对{n_true_pairs}: d_img={d_img:.2f}\", d_cat={d_cat:.2f}\" → 在k-vector中未找到!")

    print(f"\n真匹配汇总: {n_true_pairs}对, k-vector找到{n_found_in_kv}对, 投票到真天区{n_voted_true_cell}对")
    if true_rot_bins:
        print(f"  rot值分布: min={min(true_rot_bins):.1f}°, max={max(true_rot_bins):.1f}°, "
              f"mean={np.mean(true_rot_bins):.1f}°, std={np.std(true_rot_bins):.1f}°")

    # 11. 分析假阳性峰值
    if peaks:
        false_peak = peaks[0]
        fp_cell, fp_rot_bin, fp_count = false_peak
        fp_ra, fp_dec = pix2ang(fp_cell)
        print(f"\n{'=' * 70}")
        print(f"假阳性峰值分析: cell={fp_cell} ({fp_ra:.4f}, {fp_dec:.4f}), "
              f"rot_bin={fp_rot_bin} ({rot_bin_to_angle(fp_rot_bin):.1f}°), 票数={fp_count}")
        print(f"{'=' * 70}")

        # 统计假阳性峰值的catalog星分布
        fp_votes_detail = []
        for ii in range(len(d_img_arr)):
            d_img = float(d_img_arr[ii])
            theta_img = float(theta_img_arr[ii])
            idx_lo, idx_hi = kvector_query(kv, d_img, delta_d)
            if idx_lo > idx_hi:
                continue
            candidates = kv.S[idx_lo:idx_hi+1]
            pa_cat = candidates['PA_cat']
            rot = np.mod(theta_img - pa_cat, 360.0)
            rot_bin = (rot / DEFAULT_ROT_BIN_DEG).astype(np.int32)
            ra_i = candidates['ra_i']
            dec_i = candidates['dec_i']
            for k in range(len(candidates)):
                if int(rot_bin[k]) == fp_rot_bin:
                    pix = ang2pix(float(ra_i[k]), float(dec_i[k]))
                    if pix == fp_cell:
                        fp_votes_detail.append({
                            'img_pair': ii,
                            'd_img': d_img,
                            'theta_img': theta_img,
                            'pa_cat': float(pa_cat[k]),
                            'rot': float(rot[k]),
                            'ra_i': float(ra_i[k]),
                            'dec_i': float(dec_i[k]),
                        })

        print(f"  假阳性峰值的{len(fp_votes_detail)}票来自:")
        # 按catalog星位置分组
        fp_ra_arr = np.array([v['ra_i'] for v in fp_votes_detail])
        fp_dec_arr = np.array([v['dec_i'] for v in fp_votes_detail])
        print(f"  catalog星RA范围: [{fp_ra_arr.min():.4f}, {fp_ra_arr.max():.4f}], "
              f"Dec范围: [{fp_dec_arr.min():.4f}, {fp_dec_arr.max():.4f}]")
        print(f"  catalog星中心: ({np.mean(fp_ra_arr):.4f}, {np.mean(fp_dec_arr):.4f})")
        print(f"  图像星对θ_img范围: [{min(v['theta_img'] for v in fp_votes_detail):.1f}°, "
              f"{max(v['theta_img'] for v in fp_votes_detail):.1f}°]")
        print(f"  PA_cat范围: [{min(v['pa_cat'] for v in fp_votes_detail):.1f}°, "
              f"{max(v['pa_cat'] for v in fp_votes_detail):.1f}°]")
        print(f"  rot范围: [{min(v['rot'] for v in fp_votes_detail):.1f}°, "
              f"{max(v['rot'] for v in fp_votes_detail):.1f}°]")


if __name__ == "__main__":
    main()

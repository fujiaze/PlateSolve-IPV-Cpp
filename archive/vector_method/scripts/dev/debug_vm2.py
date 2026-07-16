"""
Vector Match V2 详细调试脚本

针对问题帧（RMS>1px或匹配失败），逐步记录算法中间数据，
找出匹配失败或精度差的根因。

用法: python debug_vm2.py
"""

import os
import sys
import math
import json
import logging
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

from vector_match_v2 import (
    VectorMatch, gnomonic_forward, gnomonic_inverse, _apply_flip,
    _build_image_vectors, _build_catalog_vectors,
    _ransac_rigid_v2, _find_fine_correspondences,
    _iterative_svd_refine, _compute_normalized_score,
    _apply_similarity, _count_inliers_1to1, _adaptive_tau,
    bisection_mag_limit,
    _DEGTORAD, _RADTODEG, _RADTOASEC,
)
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "output", "debug_detail")
os.makedirs(DEBUG_DIR, exist_ok=True)

logging.basicConfig(level=logging.DEBUG, format="[%(name)s] %(message)s")
logger = logging.getLogger("debug_vm2")

# 问题帧列表
PROBLEM_FILES = [
    # RMS > 1px (4帧Blue)
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@055805-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@061041-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@012543-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@062557-180S-Blue.fts",
    # 匹配失败 (2帧Oiii)
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250719@034325-600S-Oiii.fts",
]


def debug_single_file(file_path: str):
    """对单帧进行详细调试，记录每步中间数据"""
    file_name = os.path.basename(file_path)
    full_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.exists(full_path):
        logger.error("文件不存在: %s", full_path)
        return

    print(f"\n{'='*80}")
    print(f"调试帧: {file_name}")
    print(f"{'='*80}")

    # 1. 读取图像
    reader = ImageReader()
    img = reader.read(full_path)

    # 2. 星点检测
    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_sat = np.array(det_result.saturated, dtype=np.int32)

    n_sat = int(np.sum(img_sat))
    n_normal = len(img_x) - n_sat
    print(f"\n--- 星点检测 ---")
    print(f"  总星数: {len(img_x)}, 饱和星: {n_sat}, 正常星: {n_normal}")

    # 3. 提取初始参数
    center_ra = 0.0
    center_dec = 0.0
    focal_length = 200.0
    pixel_size = 6.0
    scale = 0.0

    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        center_ra = img.metadata.wcs.crval1
        center_dec = img.metadata.wcs.crval2
        scale = img.metadata.wcs.pixel_scale
    if img.metadata.observation:
        if img.metadata.observation.focallen is not None:
            focal_length = img.metadata.observation.focallen
        if img.metadata.observation.xpixsz is not None:
            pixel_size = img.metadata.observation.xpixsz

    if center_ra == 0.0 and center_dec == 0.0:
        for kw in img.keywords:
            name = kw.name.upper()
            if name in ("OBJCTRA", "RA"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                    if len(parts) >= 3:
                        center_ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
            elif name in ("OBJCTDEC", "DEC"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                    if len(parts) >= 3:
                        sign = -1 if parts[0].startswith("-") else 1
                        center_dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)

    s0 = scale if scale > 0 else 206.265 * pixel_size / focal_length
    s0_from_header = 206.265 * pixel_size / focal_length
    fov_diag = math.sqrt(img.width ** 2 + img.height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0

    print(f"\n--- 初始参数 ---")
    print(f"  RA={center_ra:.6f}° Dec={center_dec:.6f}°")
    print(f"  FOCALLEN={focal_length}mm XPIXSZ={pixel_size}μm")
    print(f"  s0(WCS)={scale:.4f}\"/px  s0(FOCALLEN/XPIXSZ)={s0_from_header:.4f}\"/px  s0(used)={s0:.4f}\"/px")
    print(f"  s0差异: {abs(s0 - s0_from_header)/s0_from_header*100:.2f}%")
    print(f"  FOV={fov_diag:.2f}°  查询半径={radius_deg:.2f}°")

    # 4. 构建U向量组
    U, N_img, n_sat_sel, sparsity = _build_image_vectors(
        img_x, img_y, img_flux, img_sat, s0, img.width, img.height
    )
    print(f"\n--- 图像向量组U ---")
    print(f"  选取: N_img={N_img} (饱和星={n_sat_sel})")
    print(f"  U范围: x=[{U[:,0].min():.1f}, {U[:,0].max():.1f}]\" y=[{U[:,1].min():.1f}, {U[:,1].max():.1f}]\"")
    print(f"  稀疏度: med={np.median(sparsity):.1f}\" min={sparsity.min():.1f}\" max={sparsity.max():.1f}\"")

    # 5. Gaia查询
    if n_sat_sel >= 50:
        N_gaia = math.ceil(1.5 * n_sat_sel)
    else:
        N_gaia = 150

    vm = VectorMatch(GAIA_DIR, db_type=2)
    mag_limit, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        vm._gaia, center_ra, center_dec, radius_deg, N_gaia
    )
    print(f"\n--- Gaia查询 ---")
    print(f"  目标N_gaia={N_gaia}, 极限星等={mag_limit:.2f}, 实际星数={M}")
    if M < 2:
        print(f"  ⚠️ Gaia星数不足!")
        vm.close()
        return

    # 6. 逐模式详细调试
    tau_coarse = max(1.0, 2.5 * s0)
    K = 3000
    min_inliers = max(5, int(N_img * 0.2))
    candidate_radius_coarse = fov_diag * 3600.0 * 0.1

    print(f"\n--- RANSAC参数 ---")
    print(f"  tau_coarse={tau_coarse:.2f}\" K={K} min_inliers={min_inliers} candidate_radius={candidate_radius_coarse:.1f}\"")

    mode_results = {}

    for mode in range(4):
        print(f"\n{'─'*60}")
        print(f"  翻转模式 {mode}")
        print(f"{'─'*60}")

        W = _build_catalog_vectors(cat_ra, cat_dec, center_ra, center_dec)
        Wf = _apply_flip(W, mode)
        print(f"  Wf范围: x=[{Wf[:,0].min():.1f}, {Wf[:,0].max():.1f}]\" y=[{Wf[:,1].min():.1f}, {Wf[:,1].max():.1f}]\"")

        # 粗候选对构建
        tree_W = cKDTree(Wf)
        pairs_coarse = []
        for i in range(N_img):
            idxs = tree_W.query_ball_point(U[i], candidate_radius_coarse)
            for j in idxs:
                pairs_coarse.append((i, j))
        candidates_per_u = [0] * N_img
        for i, j in pairs_coarse:
            candidates_per_u[i] += 1
        n_with_cand = sum(1 for c in candidates_per_u if c > 0)
        print(f"  粗候选: {len(pairs_coarse)}对, 有候选的U点={n_with_cand}/{N_img}, 平均{np.mean(candidates_per_u):.1f}个/U点")

        # RANSAC粗匹配
        s, theta, tx, ty, n_inliers, rms, inlier_mask = _ransac_rigid_v2(
            U, Wf, tau_coarse, K, min_inliers, vm._rng,
            candidate_radius_coarse, sparsity
        )
        print(f"  粗匹配: s={s:.4f} θ={math.degrees(theta):.2f}° tx={tx:.2f}\" ty={ty:.2f}\" n={n_inliers} rms={rms:.3f}\"")

        if n_inliers < min_inliers:
            print(f"  ❌ 粗匹配内点不足 n={n_inliers} < min={min_inliers}")
            mode_results[mode] = {"status": "coarse_fail", "n_inliers": n_inliers}
            continue

        # 分析粗匹配质量
        Wt = _apply_similarity(Wf, s, theta, tx, ty)
        tree_Wt = cKDTree(Wt)
        dists_all, _ = tree_Wt.query(U, k=1)
        n_close = int(np.sum(dists_all < tau_coarse))
        n_very_close = int(np.sum(dists_all < s0))
        print(f"  粗匹配后: 全部U→W'最近邻<tau={n_close}/{N_img}, <s0={n_very_close}/{N_img}")
        print(f"  最近邻距离: med={np.median(dists_all):.2f}\" p90={np.percentile(dists_all,90):.2f}\" max={dists_all.max():.2f}\"")

        # 精细候选对
        pairs_fine = _find_fine_correspondences(U, Wf, s, theta, tx, ty, tau_coarse)
        print(f"  精细候选: {len(pairs_fine)}对")

        # SVD精修
        s_ref, theta_ref, tx_ref, ty_ref, n_ref, rms_ref, mask_ref = \
            _iterative_svd_refine(U, Wf, inlier_mask, s0, s, theta, tx, ty, max_iter=10)

        if n_ref >= min_inliers:
            s, theta, tx, ty = s_ref, theta_ref, tx_ref, ty_ref
            n_inliers, rms, inlier_mask = n_ref, rms_ref, mask_ref
            print(f"  SVD精修: s={s:.4f} θ={math.degrees(theta):.2f}° n={n_inliers} rms={rms:.3f}\"")
        else:
            print(f"  ⚠️ SVD精修内点不足 n={n_ref} < min={min_inliers}, 使用粗匹配结果")

        # 归一化得分
        norm_score = _compute_normalized_score(n_inliers, rms, N_img, M, tau_coarse)
        print(f"  归一化得分: {norm_score:.4f}")

        # 检查s范围
        s_valid = 0.9 <= s <= 1.1
        print(f"  s范围检查: s={s:.4f} {'✅' if s_valid else '❌ 超出[0.9,1.1]'}")

        mode_results[mode] = {
            "status": "ok" if s_valid else "s_out_of_range",
            "s": s, "theta_deg": math.degrees(theta),
            "tx": tx, "ty": ty,
            "n_inliers": n_inliers, "rms": rms,
            "norm_score": norm_score, "s_valid": s_valid,
            "inlier_mask": inlier_mask, "Wf": Wf,
        }

    # 7. 选择最佳模式
    print(f"\n{'='*60}")
    print(f"  模式选择汇总")
    print(f"{'='*60}")
    for mode, res in mode_results.items():
        print(f"  模式{mode}: {res['status']} s={res.get('s',0):.4f} n={res.get('n_inliers',0)} rms={res.get('rms',0):.3f}\" score={res.get('norm_score',0):.4f}")

    best_mode = -1
    best_score = -1.0
    for mode, res in mode_results.items():
        if res["status"] == "ok" and res["norm_score"] > best_score:
            best_score = res["norm_score"]
            best_mode = mode

    if best_mode < 0 or best_score < 0.10:
        print(f"\n  ❌ 所有模式匹配失败! best_score={best_score:.4f}")
        print(f"\n  失败原因分析:")
        for mode, res in mode_results.items():
            if res["status"] == "coarse_fail":
                print(f"    模式{mode}: 粗匹配内点不足(n={res['n_inliers']})")
            elif res["status"] == "s_out_of_range":
                print(f"    模式{mode}: s={res['s']:.4f}超出范围")
            else:
                print(f"    模式{mode}: 得分过低 score={res['norm_score']:.4f}")
        vm.close()
        return

    print(f"\n  ✅ 最佳模式={best_mode} score={best_score:.4f}")

    # 8. 中心修正 + 渐进放宽tau RANSAC精修 (匹配当前算法)
    res = mode_results[best_mode]
    s, theta = res["s"], math.radians(res["theta_deg"])
    tx, ty = res["tx"], res["ty"]
    inlier_mask = res["inlier_mask"]
    Wf = res["Wf"]

    cos_d0 = math.cos(center_dec * _DEGTORAD)
    if abs(cos_d0) < 1e-10:
        cos_d0 = 1e-10
    delta_ra = tx / (cos_d0 * 3600.0)
    delta_dec = ty / 3600.0
    new_ra = center_ra + delta_ra
    new_dec = center_dec + delta_dec
    print(f"\n--- 中心修正 ---")
    print(f"  ΔRA={delta_ra:.6f}° ΔDec={delta_dec:.6f}°")
    print(f"  修正前: RA={center_ra:.6f}° Dec={center_dec:.6f}°")
    print(f"  修正后: RA={new_ra:.6f}° Dec={new_dec:.6f}°")

    # 重新投影
    W_new = _build_catalog_vectors(cat_ra, cat_dec, new_ra, new_dec)
    Wf_new = _apply_flip(W_new, best_mode)

    refine_radius = fov_diag * 3600.0 * 0.1
    min_inliers_refine = max(5, int(N_img * 0.2))

    print(f"\n--- 中心修正后RANSAC精修 (渐进放宽tau) ---")
    print(f"  refine_radius={refine_radius:.1f}\" min_inliers_refine={min_inliers_refine}")

    # 渐进放宽tau: 从1.0*s0逐步放宽到5.0*s0
    refine_success = False
    for tau_mult in [1.0, 2.0, 3.0, 5.0]:
        tau_try = max(0.5, tau_mult * s0)
        print(f"\n  尝试 tau={tau_mult}×s0={tau_try:.2f}\"")

        s2, theta2, tx2, ty2, n2, rms2, mask2 = _ransac_rigid_v2(
            U, Wf_new, tau_try, 3000, min_inliers_refine, vm._rng,
            refine_radius, sparsity
        )
        print(f"    RANSAC: s={s2:.4f} θ={math.degrees(theta2):.2f}° n={n2} rms={rms2:.3f}\"")

        if n2 >= min_inliers_refine:
            # RANSAC成功，再做SVD精修
            s3, theta3, tx3, ty3, n3, rms3, mask3 = _iterative_svd_refine(
                U, Wf_new, mask2, s0, s2, theta2, tx2, ty2, max_iter=10
            )
            print(f"    SVD精修: s={s3:.4f} θ={math.degrees(theta3):.2f}° n={n3} rms={rms3:.3f}\"")

            if n3 >= min_inliers_refine:
                s_final, theta_final = s3, theta3
                tx_final, ty_final = tx3, ty3
                inlier_mask = mask3
                Wf_use = Wf_new
                refine_success = True
                print(f"    ✅ RANSAC+SVD成功 (tau={tau_mult}×s0)")
            else:
                s_final, theta_final = s2, theta2
                tx_final, ty_final = tx2, ty2
                inlier_mask = mask2
                Wf_use = Wf_new
                refine_success = True
                print(f"    ✅ RANSAC成功, SVD内点不足 (tau={tau_mult}×s0)")
            break
        else:
            print(f"    ❌ 内点不足 n={n2} < min={min_inliers_refine}")

    if not refine_success:
        s_final, theta_final = s, theta
        tx_final, ty_final = tx, ty
        Wf_use = Wf
        print(f"\n  ⚠️ 中心修正后RANSAC全部失败, 使用粗匹配结果")

    # 最终RMS计算
    Wt_final = _apply_similarity(Wf_use, s_final, theta_final, tx_final, ty_final)
    tree_final = cKDTree(Wt_final)
    dists_final, idxs_final = tree_final.query(U, k=1)

    # 用内点计算RMS
    if np.any(inlier_mask):
        inlier_dists = dists_final[inlier_mask]
        rms_arcsec = float(np.sqrt(np.mean(inlier_dists ** 2)))
    else:
        rms_arcsec = float(np.sqrt(np.mean(dists_final ** 2)))
    rms_px = rms_arcsec / s0

    print(f"\n--- 最终结果 ---")
    print(f"  s={s_final:.4f} (s0×s={s0*s_final:.4f}\"/px)")
    print(f"  θ={math.degrees(theta_final):.4f}°")
    print(f"  tx={tx_final:.3f}\" ty={ty_final:.3f}\"")
    print(f"  内点数={int(np.sum(inlier_mask))}")
    print(f"  RMS={rms_arcsec:.3f}\" ({rms_px:.3f}px)")

    # 9. 投影验证 - 用全部Gaia星
    ra_all, dec_all, mag_all = vm._gaia.cone_search(
        new_ra, new_dec, radius_deg, 22.0
    )
    xi, eta, valid = gnomonic_forward(ra_all, dec_all, new_ra, new_dec)
    W_val = np.column_stack([xi[valid], eta[valid]])
    Wf_val = _apply_flip(W_val, best_mode)

    a0 = tx_final
    a1 = s_final * math.cos(theta_final)
    a2 = -s_final * math.sin(theta_final)
    b0 = ty_final
    b1 = s_final * math.sin(theta_final)
    b2 = s_final * math.cos(theta_final)

    u_pred = a0 + a1 * Wf_val[:, 0] + a2 * Wf_val[:, 1]
    v_pred = b0 + b1 * Wf_val[:, 0] + b2 * Wf_val[:, 1]
    px_pred = u_pred / s0 + img.width / 2.0
    py_pred = -v_pred / s0 + img.height / 2.0

    in_img = (px_pred >= 0) & (px_pred < img.width) & (py_pred >= 0) & (py_pred < img.height)
    print(f"\n--- 投影验证 ---")
    print(f"  Gaia星在图内: {np.sum(in_img)}/{len(Wf_val)}")

    if np.any(in_img):
        tree_img = cKDTree(np.column_stack([img_x, img_y]))
        val_dists, val_idxs = tree_img.query(np.column_stack([px_pred[in_img], py_pred[in_img]]))
        med_err = float(np.median(val_dists))
        lt2px = float(np.sum(val_dists < 2) / len(val_dists) * 100)
        print(f"  中位误差: {med_err:.2f}px, <2px: {lt2px:.1f}%")
        print(f"  误差分布: p25={np.percentile(val_dists,25):.2f} p50={np.percentile(val_dists,50):.2f} "
              f"p75={np.percentile(val_dists,75):.2f} p90={np.percentile(val_dists,90):.2f} max={val_dists.max():.2f}px")

        # 系统性偏移分析
        matched_img_x = img_x[val_idxs]
        matched_img_y = img_y[val_idxs]
        dx = px_pred[in_img] - matched_img_x
        dy = py_pred[in_img] - matched_img_y
        print(f"  系统偏移: dx_med={np.median(dx):.2f}px dy_med={np.median(dy):.2f}px")
        print(f"  系统偏移: dx_mean={np.mean(dx):.2f}px dy_mean={np.mean(dy):.2f}px")

        # 检查偏移是否有空间相关性（边缘vs中心）
        cx_img, cy_img = img.width / 2.0, img.height / 2.0
        r_from_center = np.sqrt((px_pred[in_img] - cx_img)**2 + (py_pred[in_img] - cy_img)**2)
        r_max = math.sqrt(cx_img**2 + cy_img**2)
        r_norm = r_from_center / r_max

        # 按距离分桶统计误差
        n_bins = 4
        bin_edges = np.linspace(0, 1, n_bins + 1)
        print(f"  误差vs距中心距离:")
        for b in range(n_bins):
            mask_bin = (r_norm >= bin_edges[b]) & (r_norm < bin_edges[b+1])
            if np.any(mask_bin):
                bin_err = val_dists[mask_bin]
                bin_dx = dx[mask_bin]
                bin_dy = dy[mask_bin]
                print(f"    r=[{bin_edges[b]:.2f},{bin_edges[b+1]:.2f}]: "
                      f"n={np.sum(mask_bin)} err_med={np.median(bin_err):.2f}px "
                      f"dx_med={np.median(bin_dx):.2f}px dy_med={np.median(bin_dy):.2f}px")

    # 10. 分析s0对结果的影响
    print(f"\n--- s0影响分析 ---")
    print(f"  当前s0={s0:.4f}\"/px, 匹配s={s_final:.4f}")
    print(f"  最终尺度= s0×s = {s0*s_final:.4f}\"/px")
    if abs(s0 - s0_from_header) > 0.01:
        print(f"  ⚠️ s0(WCS)={scale:.4f} ≠ s0(FOCALLEN/XPIXSZ)={s0_from_header:.4f}")
        print(f"  如果用s0={s0_from_header:.4f}, s需要={s0*s_final/s0_from_header:.4f}才能得到相同最终尺度")

    vm.close()
    print()


if __name__ == "__main__":
    for f in PROBLEM_FILES:
        debug_single_file(f)

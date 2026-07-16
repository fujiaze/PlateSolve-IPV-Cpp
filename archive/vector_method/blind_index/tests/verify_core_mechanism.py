# -*- coding: utf-8 -*-
"""
验证4SADQ-KV核心机制: 用WCS识别Gaia星对应的图像星, 构建四边形, 测试k-vector匹配
用途: 隔离star selection bug, 验证 k-vector + 5距离验证 + Umeyama SVD 核心机制本身是否正确

逻辑:
    1. 读取图像 + 检测星点
    2. 查询DR3 + gnomonic投影 + FOV过滤
    3. 用WCS把Gaia星投影到图像像素, 与检测星匹配(<5px) → "干净"图像星
    4. 用干净图像星构建图像四边形
    5. 用Gaia FOV星构建参考四边形 + k-vector索引
    6. k-vector查询 + 5距离验证
    7. 若有候选 → 求WCS, 检查RMS和CRVAL偏差

判定:
    成功 → 核心机制正确, bug在star selection (top-N亮星=饱和星)
    失败 → 核心机制本身有问题
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

# 项目根目录
_PROJECT_ROOT = os.path.normpath(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index.python.io_wrappers import (
    read_image, detect_stars, query_dr3, get_pixel_scale_from_header,
)
from lib.plate_solve.blind_index.python.quad_selector import (
    generate_image_quads, generate_reference_quads,
)
from lib.plate_solve.blind_index.python.kvector import build_kvector
from lib.plate_solve.blind_index.python.matcher import match_all_quads
from lib.plate_solve.blind_index.python.wcs_solver import (
    solve_wcs_from_candidate, angular_separation_arcsec,
)
from lib.plate_solve.blind_index.python.logging_setup import setup_logging

# 复用V3.5 gnomonic投影
from lib.plate_solve.python.vector_match_v2 import gnomonic_forward

setup_logging()


# ════════════════════════════════════════════════════════════════════
# 测试帧
# ════════════════════════════════════════════════════════════════════
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
}


def _wcs_to_image_pixels(ra_deg, dec_deg, wcs, naxis2):
    """
    用FITS头WCS把(RA,Dec)投影到图像像素坐标。
    WCS线性近似: delta_ra = CD1_1·dx + CD1_2·dy; delta_dec = CD2_1·dx + CD2_2·dy
    解(dx,dy): [dx;dy] = CD^-1 · [delta_ra; delta_dec]
    图像Y翻转: y_img = NAXIS2 + 1 - y_wcs
    """
    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    crval1 = wcs.crval1
    crval2 = wcs.crval2
    crpix1 = wcs.crpix1
    crpix2 = wcs.crpix2

    # CD矩阵 (deg/pixel)
    CD = np.array([
        [wcs.cd1_1, wcs.cd1_2],
        [wcs.cd2_1, wcs.cd2_2],
    ])
    det = CD[0, 0] * CD[1, 1] - CD[0, 1] * CD[1, 0]
    if abs(det) < 1e-30:
        return None, None
    CD_inv = np.linalg.inv(CD)

    # RA含cos(Dec)因子: 标准WCS中CD1_*已是deg/pixel (含1/cos(dec)因子),
    # 但CRVAL附近线性近似用delta_ra直接即可
    delta_ra = ra - crval1
    delta_dec = dec - crval2
    # 包装delta_ra到[-180,180]
    delta_ra = (delta_ra + 180.0) % 360.0 - 180.0

    # [dx; dy] = CD^-1 · [delta_ra; delta_dec]
    delta = np.column_stack([delta_ra, delta_dec])  # (N,2)
    dxdy = delta @ CD_inv.T  # (N,2)

    x_wcs = crpix1 + dxdy[:, 0] - 1.0  # FITS 1-indexed → 0-indexed
    y_wcs = crpix2 + dxdy[:, 1] - 1.0
    # Y翻转: 图像y向下, WCS y向上
    y_img = naxis2 - y_wcs
    return x_wcs, y_img


def verify_frame(frame_name: str, frame_cfg: dict, mag_limit: float = 12.0):
    """验证单帧核心机制。"""
    print(f"\n{'=' * 70}")
    print(f"[验证帧] {frame_name}")
    print(f"{'=' * 70}")

    abs_path = os.path.join(_PROJECT_ROOT, frame_cfg["path"])
    if not os.path.exists(abs_path):
        print(f"  [失败] 文件不存在: {abs_path}")
        return False

    s0 = frame_cfg["s0"]
    ra0 = frame_cfg["query_ra"]
    dec0 = frame_cfg["query_dec"]

    # ═══ 阶段1: 读取图像 + 检测星点 ═══
    t0 = time.time()
    uint16_img, metadata = read_image(abs_path)
    h, w = uint16_img.shape
    star_result = detect_stars(uint16_img)
    print(f"  图像: {w}x{h}, 检测星: {star_result.count}颗 "
          f"({np.sum(star_result.saturated==1)}饱和, {np.sum(star_result.saturated==0)}正常)")

    if metadata.wcs is None or not metadata.wcs.has_wcs:
        print(f"  [失败] FITS头无WCS, 无法验证")
        return False
    wcs_hdr = metadata.wcs
    print(f"  WCS: CRVAL=({wcs_hdr.crval1:.5f}, {wcs_hdr.crval2:.5f}), "
          f"CRPIX=({wcs_hdr.crpix1:.2f}, {wcs_hdr.crpix2:.2f})")
    print(f"  阶段1耗时: {time.time()-t0:.3f}s")

    # ═══ 阶段2: 查询DR3 + gnomonic投影 + FOV过滤 ═══
    t0 = time.time()
    fov_diag_arcsec = float(np.sqrt(w**2 + h**2) * s0)
    radius_deg = (fov_diag_arcsec / 3600.0) * 1.5
    ra_arr, dec_arr, mag_arr = query_dr3(ra0, dec0, radius_deg, mag_limit)
    print(f"  DR3查询: {len(ra_arr)}颗参考星 (mag_limit={mag_limit}, 半径={radius_deg:.3f}°)")

    xi_arr, eta_arr, valid = gnomonic_forward(ra_arr, dec_arr, ra0, dec0)
    xi_arr = xi_arr[valid]; eta_arr = eta_arr[valid]
    ra_arr = ra_arr[valid]; dec_arr = dec_arr[valid]

    # FOV过滤
    half_w = (w / 2.0) * s0
    half_h = (h / 2.0) * s0
    fov_mask = (np.abs(xi_arr) <= half_w) & (np.abs(eta_arr) <= half_h)
    xi_fov = xi_arr[fov_mask]; eta_fov = eta_arr[fov_mask]
    ra_fov = ra_arr[fov_mask]; dec_fov = dec_arr[fov_mask]
    print(f"  FOV过滤后: {len(ra_fov)}颗参考星")
    print(f"  阶段2耗时: {time.time()-t0:.3f}s")

    if len(ra_fov) < 4:
        print(f"  [失败] FOV内参考星不足4颗")
        return False

    # ═══ 阶段3: 用WCS把Gaia星投影到图像像素, 匹配检测星 ═══
    t0 = time.time()
    gaia_x, gaia_y = _wcs_to_image_pixels(ra_fov, dec_fov, wcs_hdr, h)
    if gaia_x is None:
        print(f"  [失败] WCS投影失败")
        return False

    # 在图像范围内的Gaia星
    in_img = (gaia_x >= 0) & (gaia_x < w) & (gaia_y >= 0) & (gaia_y < h)
    gaia_x_in = gaia_x[in_img]; gaia_y_in = gaia_y[in_img]
    ra_in = ra_fov[in_img]; dec_in = dec_fov[in_img]
    print(f"  Gaia星投影到图像内: {len(gaia_x_in)}/{len(gaia_x)}")

    # 与检测星匹配 (<5px)
    det_x = star_result.x; det_y = star_result.y
    matched_pairs = []  # (det_idx, gaia_idx, dist)
    for gi in range(len(gaia_x_in)):
        gx, gy = gaia_x_in[gi], gaia_y_in[gi]
        dist2 = (det_x - gx)**2 + (det_y - gy)**2
        best_idx = int(np.argmin(dist2))
        best_dist = float(np.sqrt(dist2[best_idx]))
        if best_dist < 5.0:
            matched_pairs.append((best_idx, gi, best_dist))

    print(f"  Gaia→图像星匹配 (<5px): {len(matched_pairs)}/{len(gaia_x_in)}")
    if len(matched_pairs) < 4:
        print(f"  [失败] 匹配数<4, 无法构建四边形")
        return False

    # 提取匹配的图像星坐标 (干净图像星, 已知对应Gaia星)
    matched_det_idx = np.array([p[0] for p in matched_pairs])
    matched_gaia_idx = np.array([p[1] for p in matched_pairs])
    matched_dist = np.array([p[2] for p in matched_pairs])
    print(f"  匹配距离: mean={matched_dist.mean():.3f}px, max={matched_dist.max():.3f}px")
    print(f"  阶段3耗时: {time.time()-t0:.3f}s")

    # ═══ 阶段4: 用干净图像星构建图像四边形 ═══
    t0 = time.time()
    clean_x = det_x[matched_det_idx]
    clean_y = det_y[matched_det_idx]
    print(f"  干净图像星: {len(clean_x)}颗 (已知对应Gaia星)")

    # 参考四边形 + k-vector索引
    ref_quads = generate_reference_quads(xi_fov, eta_fov, ra_fov, dec_fov)
    if len(ref_quads) == 0:
        print(f"  [失败] 参考四边形生成失败")
        return False
    kvector_index = build_kvector(ref_quads)
    if kvector_index is None:
        print(f"  [失败] k-vector索引构建失败")
        return False
    print(f"  参考四边形: {len(ref_quads)}个, k-vector: N={kvector_index.n_quads}, "
          f"d_range=[{kvector_index.d_min:.2f}, {kvector_index.d_max:.2f}]\"")

    # 图像四边形 (用干净图像星)
    # pool_size设为干净星数, pivot_count=15, top_n=5
    image_quads = generate_image_quads(
        clean_x, clean_y, s0,
        ref_kvector_index=kvector_index,
        pool_size=min(len(clean_x), 50),  # 用最多50颗干净星作亮星池
        pivot_count=15,
        n_neighbors=8,
        top_n=5,
    )
    print(f"  图像四边形: {len(image_quads)}个 (从干净星构建)")
    if len(image_quads) == 0:
        print(f"  [失败] 图像四边形生成失败")
        return False
    print(f"  阶段4耗时: {time.time()-t0:.3f}s")

    # 打印图像四边形 d_AB 范围
    img_d_abs = [q.d_AB for q in image_quads]
    ref_d_abs = [float(q.distances[0]) for q in ref_quads]
    print(f"  图像四边形 d_AB: {[f'{d:.2f}' for d in img_d_abs]}")
    print(f"  参考四边形 d_AB 范围: [{min(ref_d_abs):.2f}, {max(ref_d_abs):.2f}]\"")

    # ═══ 阶段5: k-vector查询 + 5距离验证 ═══
    t0 = time.time()
    candidates_per_quad = match_all_quads(image_quads, kvector_index, s0)
    total_cands = sum(len(c) for c in candidates_per_quad)
    print(f"  匹配候选总数: {total_cands}")
    for i, cands in enumerate(candidates_per_quad):
        if cands:
            print(f"    四边形#{i} d_AB={image_quads[i].d_AB:.2f}\" → {len(cands)}候选")
    print(f"  阶段5耗时: {time.time()-t0:.3f}s")

    if total_cands == 0:
        print(f"  [失败] 无匹配候选, 核心机制可能有问题")
        return False

    # ═══ 阶段6: 求WCS, 检查RMS和CRVAL偏差 ═══
    t0 = time.time()
    best_wcs = None
    best_rms = float("inf")
    best_cand = None
    for cands in candidates_per_quad:
        for cand in cands:
            wcs = solve_wcs_from_candidate(cand, ra0, dec0)
            if wcs is not None and wcs.rms_arcsec < best_rms:
                best_rms = wcs.rms_arcsec
                best_wcs = wcs
                best_cand = cand

    if best_wcs is None:
        print(f"  [失败] 所有候选WCS求解失败")
        return False

    # CRVAL偏差
    ra_solved = np.array([best_wcs.crval1])
    dec_solved = np.array([best_wcs.crval2])
    ra_expected = np.array([wcs_hdr.crval1])
    dec_expected = np.array([wcs_hdr.crval2])
    sep = float(angular_separation_arcsec(ra_solved, dec_solved, ra_expected, dec_expected)[0])

    print(f"\n  ─── 核心机制验证结果 ───")
    print(f"  最佳候选 RMS: {best_rms:.3f}\" (阈值<3\")")
    print(f"  solved CRVAL: ({best_wcs.crval1:.5f}, {best_wcs.crval2:.5f})")
    print(f"  expected CRVAL: ({wcs_hdr.crval1:.5f}, {wcs_hdr.crval2:.5f})")
    print(f"  CRVAL偏差: {sep:.2f}\" (阈值<30\")")
    print(f"  s(Umeyama)={best_wcs.s:.5f}, n_points={best_wcs.n_points}")
    print(f"  阶段6耗时: {time.time()-t0:.3f}s")

    pass_rms = best_rms < 3.0
    pass_crval = sep < 30.0
    print(f"\n  RMS通过: {pass_rms}, CRVAL通过: {pass_crval}")
    if pass_rms and pass_crval:
        print(f"  ✓ 核心机制验证通过! bug在star selection (top-N亮星=饱和星)")
    else:
        print(f"  ✗ 核心机制可能仍有问题 (RMS或CRVAL超阈值)")
    return pass_rms and pass_crval


def main():
    frame_name = sys.argv[1] if len(sys.argv) > 1 else "M20"
    if frame_name not in FRAMES:
        print(f"未知帧: {frame_name}, 可选: {list(FRAMES.keys())}")
        sys.exit(1)

    # 重定向stdout到文件 (UTF-8)
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "logs", "verify_core_out.txt"
    )
    log_path = os.path.normpath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    old_stdout = sys.stdout
    log_fp = open(log_path, "w", encoding="utf-8")

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    sys.stdout = _Tee(old_stdout, log_fp)
    try:
        ok = verify_frame(frame_name, FRAMES[frame_name])
        print(f"\n{'='*70}")
        print(f"最终结论: {'通过' if ok else '失败'}")
        print(f"{'='*70}")
    finally:
        sys.stdout = old_stdout
        log_fp.close()
    print(f"\n结果已写入: {log_path}")


if __name__ == "__main__":
    main()

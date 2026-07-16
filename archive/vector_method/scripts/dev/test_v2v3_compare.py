"""
V2 vs V3 对比测试 - 问题帧

对比V2(纯距离内点)和V3(距离+方向内点)在6个问题帧上的表现
"""

import os
import sys
import time
import math
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib", "star_detector", "python"))

import numpy as np
from scipy.spatial import cKDTree

from vector_match_v2 import VectorMatch as VectorMatchV2
from vector_match_v2 import gnomonic_forward, _apply_flip
from vector_match_v3 import VectorMatch as VectorMatchV3
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GAIA_DIR = os.path.join(PROJECT_ROOT, "GaiaDR3SP")

PROBLEM_FILES = [
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@055805-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@012543-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@061041-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@062557-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250719@034325-600S-Oiii.fts",
]


def get_initial_params(img):
    """从FITS头提取初始参数 (只用物理参数, 不用WCS pixel_scale)"""
    center_ra = 0.0
    center_dec = 0.0
    focal_length = 200.0
    pixel_size = 6.0

    if img.metadata.wcs and img.metadata.wcs.has_wcs:
        center_ra = img.metadata.wcs.crval1
        center_dec = img.metadata.wcs.crval2
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

    # s0只用物理参数计算, 不用WCS pixel_scale
    s0 = 206.265 * pixel_size / focal_length if focal_length > 0 and pixel_size > 0 else 6.0

    return center_ra, center_dec, focal_length, pixel_size, s0


def validate_result(vm_result, img, img_x, img_y, s0, gaia_client):
    """投影验证 (用原始投影中心)

    推导:
        U向量 = (像素偏移) × s0 (单位: 角秒)
        变换: U = s × R × Wf + t (s是相对于s0的缩放因子)
        affine参数: a1 = s × cos(theta) (用s, 不是s×s0)
        验证: u = a0 + a1×Wf_x + a2×Wf_y (单位: 角秒)
        转换: px = u / s0 + cx (用s0, 不是s×s0)
        
    注意: affine参数是基于原始投影中心的变换, 所以验证时也要用原始中心投影
    """
    # 用输入s0计算FOV和查询半径
    fov_diag = math.sqrt(img.width ** 2 + img.height ** 2) * s0 / 3600.0
    radius_deg = fov_diag * 1.2 / 2.0
    ra_arr, dec_arr, mag_arr = gaia_client.cone_search(
        vm_result.center_ra, vm_result.center_dec, radius_deg, 22.0
    )
    if len(ra_arr) == 0:
        return 0, 0.0, 0.0

    # 只用最亮的1000颗Gaia星验证（和之前报告一致）
    sort_idx = np.argsort(mag_arr)
    n_top = min(1000, len(mag_arr))
    top_idx = sort_idx[:n_top]
    top_ra, top_dec = ra_arr[top_idx], dec_arr[top_idx]

    a0, a1, a2, b0, b1, b2 = vm_result.affine
    # 用修正后的中心投影, affine参数中的tx/ty会自动补偿中心偏移
    xi, eta, valid = gnomonic_forward(top_ra, top_dec, vm_result.center_ra, vm_result.center_dec)
    W = np.column_stack([xi[valid], eta[valid]])
    Wf = _apply_flip(W, vm_result.flip_mode)

    u = a0 + a1 * Wf[:, 0] + a2 * Wf[:, 1]
    v = b0 + b1 * Wf[:, 0] + b2 * Wf[:, 1]
    # 用输入s0转换, 不是s_final
    px = u / s0 + img.width / 2.0
    py = -v / s0 + img.height / 2.0

    in_img = (px >= 0) & (px < img.width) & (py >= 0) & (py < img.height)
    if not np.any(in_img):
        return 0, 0.0, 0.0

    tree = cKDTree(np.column_stack([img_x, img_y]))
    dists, _ = tree.query(np.column_stack([px[in_img], py[in_img]]))
    return int(np.sum(in_img)), float(np.median(dists)), float(np.sum(dists < 2) / len(dists) * 100)


def test_single(file_path, version_label, vm_class):
    """测试单帧"""
    file_name = os.path.basename(file_path)
    full_path = os.path.join(PROJECT_ROOT, file_path)
    if not os.path.exists(full_path):
        return f"  {version_label}: 文件不存在"

    reader = ImageReader()
    img = reader.read(full_path)

    params = SDetParamsPy(fitRadius=0)
    detector = StarDetector(params=params)
    det_result = detector.detect_ex(img.data)

    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_sat = np.array(det_result.saturated, dtype=np.int32)

    center_ra, center_dec, focal_length, pixel_size, s0 = get_initial_params(img)

    vm = vm_class(GAIA_DIR, db_type=2)
    t0 = time.time()
    vm_result = vm.solve(
        img_x=img_x, img_y=img_y, img_flux=img_flux, img_saturated=img_sat,
        center_ra=center_ra, center_dec=center_dec,
        focal_length_mm=focal_length, pixel_size_um=pixel_size,
        width=img.width, height=img.height,
    )
    t1 = time.time()

    if vm_result is None:
        vm.close()
        return f"  {version_label}: 匹配失败 ({t1-t0:.1f}s)"

    n_gaia, med_err, lt2 = validate_result(vm_result, img, img_x, img_y, s0, vm._gaia)
    vm.close()

    return (f"  {version_label}: OK  s={vm_result.scale_arcsec_px:.4f}\"/px  "
            f"RMS={vm_result.rms_px:.3f}px({vm_result.rms_arcsec:.3f}\")  "
            f"n={vm_result.matched_count}  flip={vm_result.flip_mode}  "
            f"投影err={med_err:.2f}px  <2px={lt2:.1f}%  "
            f"耗时={t1-t0:.1f}s")


def main():
    print("=" * 80)
    print("V2 vs V3 对比测试 - 问题帧")
    print("=" * 80)

    for f in PROBLEM_FILES:
        file_name = os.path.basename(f)
        print(f"\n{file_name}")

        v2_result = test_single(f, "V2", VectorMatchV2)
        print(v2_result)

        v3_result = test_single(f, "V3", VectorMatchV3)
        print(v3_result)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

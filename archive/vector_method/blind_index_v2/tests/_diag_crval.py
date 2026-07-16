# -*- coding: utf-8 -*-
"""
诊断脚本: 验证solved WCS是否正确 — 比较图像中心的天空坐标
原因: WCSResult.crval1/2 = 候选切点(ra_center, dec_center), 不是图像中心
      两个WCS可能用不同切点但映射到相同天空位置 — 正确的比较方式是:
      在同一像素(如图像中心)用两个WCS分别计算RA/Dec, 比较差异
"""
from __future__ import annotations

import os
import sys
import math

import numpy as np

_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, _PROJECT_ROOT)

from lib.astro_image_io.python.astro_image_io import ImageReader
from lib.plate_solve.blind_index_v2.python.pipeline import solve_blind
from lib.plate_solve.python.vector_match_v2 import gnomonic_inverse, gnomonic_forward

_DEGTORAD = math.pi / 180.0
_RADTOASEC = (180.0 / math.pi) * 3600.0


def haversine_arcsec(ra1, dec1, ra2, dec2):
    ra1r, dec1r = ra1 * _DEGTORAD, dec1 * _DEGTORAD
    ra2r, dec2r = ra2 * _DEGTORAD, dec2 * _DEGTORAD
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = math.sin(ddec / 2.0) ** 2 + math.cos(dec1r) * math.cos(dec2r) * math.sin(dra / 2.0) ** 2
    a = max(0.0, min(1.0, a))
    return 2.0 * math.asin(math.sqrt(a)) * _RADTOASEC


def wcs_pixel_to_sky(x, y, cd, crpix1, crpix2, crval1, crval2):
    """
    用WCS (CD, CRPIX, CRVAL, TAN投影) 将像素(x,y)转为RA/Dec.
    CD单位: deg/pixel. CRPIX: 1-indexed.
    """
    dx = x - (crpix1 - 1.0)
    dy = y - (crpix2 - 1.0)
    xi_deg = cd[0, 0] * dx + cd[0, 1] * dy
    eta_deg = cd[1, 0] * dx + cd[1, 1] * dy
    ra, dec = gnomonic_inverse(xi_deg * 3600.0, eta_deg * 3600.0, crval1, crval2)
    return ra, dec


def main():
    image_path = os.path.join(_PROJECT_ROOT,
        r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts")

    print("=" * 70)
    print("诊断: solved WCS vs header WCS 图像中心天空坐标比较")
    print("=" * 70)

    # 1. 读取header WCS
    reader = ImageReader()
    img_data = reader.read(image_path)
    meta = img_data.metadata
    w = meta.geometry.width
    h = meta.geometry.height
    print(f"图像尺寸: {w}x{h}")

    if meta.wcs is None or not meta.wcs.has_wcs:
        print("错误: header无WCS")
        return

    hdr_cd = np.array([[meta.wcs.cd1_1, meta.wcs.cd1_2],
                        [meta.wcs.cd2_1, meta.wcs.cd2_2]])
    hdr_crpix = (meta.wcs.crpix1, meta.wcs.crpix2)
    hdr_crval = (meta.wcs.crval1, meta.wcs.crval2)
    print(f"\nHeader WCS:")
    print(f"  CRVAL = ({hdr_crval[0]:.5f}, {hdr_crval[1]:.5f})")
    print(f"  CRPIX = ({hdr_crpix[0]:.2f}, {hdr_crpix[1]:.2f})")
    print(f"  CD = {hdr_cd.tolist()}")

    # 2. 运行盲解析
    print(f"\n运行盲解析...")
    result = solve_blind(
        image_path=image_path,
        s0_arcsec_per_pixel=None,
        query_center_ra=None,
        query_center_dec=None,
        mag_limit=12.0,
    )

    print(f"\n盲解析结果: success={result.success}")
    if not result.success:
        print(f"  失败: {result.message}")
        return

    wcs = result.wcs
    print(f"  RMS = {wcs.rms_arcsec:.3f}\"")
    print(f"  n_inliers = {wcs.n_inliers}")
    print(f"  s = {wcs.s:.5f}")
    print(f"  Solved CRVAL (tangent point) = ({wcs.crval1:.5f}, {wcs.crval2:.5f})")
    print(f"  Solved CRPIX = ({wcs.crpix1:.2f}, {wcs.crpix2:.2f})")
    print(f"  Solved CD = {wcs.cd.tolist()}")

    # 3. 比较图像中心的天空坐标
    cx, cy = w / 2.0, h / 2.0
    hdr_ra, hdr_dec = wcs_pixel_to_sky(cx, cy, hdr_cd, hdr_crpix[0], hdr_crpix[1],
                                        hdr_crval[0], hdr_crval[1])
    sol_ra, sol_dec = wcs_pixel_to_sky(cx, cy, wcs.cd, wcs.crpix1, wcs.crpix2,
                                        wcs.crval1, wcs.crval2)

    print(f"\n{'=' * 70}")
    print(f"图像中心 ({cx:.1f}, {cy:.1f}) 天空坐标比较:")
    print(f"  Header WCS → RA={hdr_ra:.5f}°, Dec={hdr_dec:.5f}°")
    print(f"  Solved WCS → RA={sol_ra:.5f}°, Dec={sol_dec:.5f}°")
    dev = haversine_arcsec(hdr_ra, hdr_dec, sol_ra, sol_dec)
    print(f"  偏差 = {dev:.2f}\" ({dev/3600:.4f}°)")
    print(f"  通过 (<30\"): {'是' if dev < 30.0 else '否'}")

    # 4. 比较多个像素 (四角+中心)
    print(f"\n{'=' * 70}")
    print("多点验证 (5个像素点):")
    test_pixels = [
        (w / 2.0, h / 2.0, "中心"),
        (w * 0.25, h * 0.25, "左上1/4"),
        (w * 0.75, h * 0.25, "右上1/4"),
        (w * 0.25, h * 0.75, "左下1/4"),
        (w * 0.75, h * 0.75, "右下1/4"),
    ]
    for px, py, label in test_pixels:
        hra, hdec = wcs_pixel_to_sky(px, py, hdr_cd, hdr_crpix[0], hdr_crpix[1],
                                      hdr_crval[0], hdr_crval[1])
        sra, sdec = wcs_pixel_to_sky(px, py, wcs.cd, wcs.crpix1, wcs.crpix2,
                                      wcs.crval1, wcs.crval2)
        d = haversine_arcsec(hra, hdec, sra, sdec)
        print(f"  {label:<10} ({px:.0f},{py:.0f}): "
              f"hdr=({hra:.5f},{hdec:.5f}) sol=({sra:.5f},{sdec:.5f}) "
              f"dev={d:.2f}\"")

    # 5. 比较原始CRVAL差异
    crval_dev_raw = haversine_arcsec(hdr_crval[0], hdr_crval[1], wcs.crval1, wcs.crval2)
    print(f"\n{'=' * 70}")
    print(f"原始CRVAL差异 (切点不同, 非WCS正确性指标):")
    print(f"  Header CRVAL = ({hdr_crval[0]:.5f}, {hdr_crval[1]:.5f})")
    print(f"  Solved CRVAL = ({wcs.crval1:.5f}, {wcs.crval2:.5f})")
    print(f"  原始CRVAL偏差 = {crval_dev_raw:.2f}\" ({crval_dev_raw/3600:.4f}°)")
    print(f"  → 切点不同是正常的, 关键看图像中心天空坐标偏差")


if __name__ == "__main__":
    main()

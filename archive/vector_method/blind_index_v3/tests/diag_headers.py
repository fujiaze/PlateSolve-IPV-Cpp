# -*- coding: utf-8 -*-
"""
诊断脚本: 检查 4 帧测试数据的 FITS 头, 确认 CRPIX 偏移与 OBJCTRA 可用性。
功能: 读取 CRVAL/CRPIX/CD/OBJCTRA/OBJCTDEC/IMAGE_SIZE, 计算图像几何中心的天球坐标
用途: 排查 M20_T2 dx/dy 过大 (~half grid) 的根本原因
"""
from __future__ import annotations

import os
import sys
import math

_PROJECT_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.astro_image_io.python.astro_image_io import ImageReader


def haversine_deg(ra1, dec1, ra2, dec2):
    ra1r, dec1r = math.radians(ra1), math.radians(dec1)
    ra2r, dec2r = math.radians(ra2), math.radians(dec2)
    dra = ra2r - ra1r
    ddec = dec2r - dec1r
    a = math.sin(ddec/2)**2 + math.cos(dec1r)*math.cos(dec2r)*math.sin(dra/2)**2
    return math.degrees(2*math.asin(min(1.0, math.sqrt(max(0.0, a)))))


FRAMES = [
    ("M20_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights", "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts")),
    ("LDN43", os.path.join(_PROJECT_ROOT, "testdata", "lights", "LDN43_LRGBH_flying_dutchman-20250503@032713-1200S-Red.fts")),
    ("NGC247_T2", os.path.join(_PROJECT_ROOT, "testdata", "lights", "NGC247_T2_flying_dutchman-20250816@034607-600S-Red.fts")),
    ("NGC55_T3", os.path.join(_PROJECT_ROOT, "testdata", "lights", "NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts")),
]


def main():
    print("=" * 100)
    print("FITS 头诊断: CRPIX 偏移 vs OBJCTRA 可用性")
    print("=" * 100)
    for name, path in FRAMES:
        print(f"\n>>> {name}")
        print(f"  路径: {path}")
        if not os.path.exists(path):
            print("  [缺失]")
            continue
        reader = ImageReader()
        try:
            img_data = reader.read_header_only(path)
        except Exception as e:
            print(f"  读取失败: {e}")
            continue

        try:
            # 图像尺寸
            w = img_data.width
            h = img_data.height
            print(f"  图像尺寸: {w} x {h}")

            # WCS 关键字
            crval1 = img_data.get_keyword_float("CRVAL1", default=float("nan"))
            crval2 = img_data.get_keyword_float("CRVAL2", default=float("nan"))
            crpix1 = img_data.get_keyword_float("CRPIX1", default=float("nan"))
            crpix2 = img_data.get_keyword_float("CRPIX2", default=float("nan"))
            cd11 = img_data.get_keyword_float("CD1_1", default=float("nan"))
            cd12 = img_data.get_keyword_float("CD1_2", default=float("nan"))
            cd21 = img_data.get_keyword_float("CD2_1", default=float("nan"))
            cd22 = img_data.get_keyword_float("CD2_2", default=float("nan"))

            has_wcs = not (math.isnan(crval1) or math.isnan(crpix1) or math.isnan(cd11))
            print(f"  WCS: {'有' if has_wcs else '无'}")
            if has_wcs:
                print(f"    CRVAL = ({crval1:.6f}, {crval2:.6f}) 度")
                print(f"    CRPIX = ({crpix1:.3f}, {crpix2:.3f}) (1-indexed FITS 约定)")
                print(f"    CD    = [[{cd11:.3e}, {cd12:.3e}], [{cd21:.3e}, {cd22:.3e}]]")

                # 图像几何中心 (FITS 1-indexed, 0-indexed = w/2, h/2)
                # FITS 像素中心约定: 像素 (1,1) 中心在 (1.0, 1.0), 图像几何中心 = (w/2 + 0.5, h/2 + 0.5) 1-indexed
                # 但通常 CRPIX 用 1-indexed, 图像几何中心 1-indexed = ((w+1)/2, (h+1)/2)
                geo_cx_1 = (w + 1) / 2.0
                geo_cy_1 = (h + 1) / 2.0
                print(f"    图像几何中心 (1-indexed) = ({geo_cx_1:.3f}, {geo_cy_1:.3f})")
                print(f"    CRPIX 偏移 = ({crpix1 - geo_cx_1:.3f}, {crpix2 - geo_cy_1:.3f}) 像素")

                # 计算图像几何中心的天球坐标 (用 CD 线性近似)
                dpx_x = geo_cx_1 - crpix1  # 原始像素位移 (1-indexed, 但差值与 0-indexed 相同)
                dpx_y = geo_cy_1 - crpix2
                ra_geo = crval1 + dpx_x * cd11 + dpx_y * cd12
                dec_geo = crval2 + dpx_x * cd21 + dpx_y * cd22
                sep = haversine_deg(crval1, crval2, ra_geo, dec_geo) * 3600.0
                print(f"    图像几何中心天球坐标 = ({ra_geo:.6f}, {dec_geo:.6f}) 度")
                print(f"    与 CRVAL 距离 = {sep:.1f}\"")

            # OBJCTRA / OBJCTDEC
            objctra = img_data.get_keyword("OBJCTRA")
            objctdec = img_data.get_keyword("OBJCTDEC")
            ra_kw = img_data.get_keyword("RA")
            dec_kw = img_data.get_keyword("DEC")
            print(f"  OBJCTRA = {objctra!r}, OBJCTDEC = {objctdec!r}")
            print(f"  RA       = {ra_kw!r}, DEC       = {dec_kw!r}")

            # FOCALLEN / XPIXSZ
            focallen = img_data.get_keyword_float("FOCALLEN", default=float("nan"))
            xpixsz = img_data.get_keyword_float("XPIXSZ", default=float("nan"))
            print(f"  FOCALLEN = {focallen}, XPIXSZ = {xpixsz}")
            if not math.isnan(focallen) and not math.isnan(xpixsz):
                s0 = 206.265 * xpixsz / focallen
                print(f"  s0 (计算) = {s0:.4f} arcsec/pixel")
        finally:
            img_data.close()


if __name__ == "__main__":
    main()

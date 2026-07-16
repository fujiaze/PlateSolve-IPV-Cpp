# -*- coding: utf-8 -*-
"""功能: 临时检查候选测试帧的FITS头信息
用途: 为Phase 1测试选帧提供依据 (has_wcs/crval/pixel_scale/focallen/xpixsz/OBJECT/宽高)
"""
import os
import sys

# 加入项目根目录到 sys.path
_PROJECT_ROOT = r"f:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, _PROJECT_ROOT)

from lib.astro_image_io.python.astro_image_io import ImageReader


CANDIDATES = [
    r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts",
    r"testdata\lights1\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
    r"testdata\lights\NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts",
    r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum.fts",
    r"testdata\lights\NGC247_T2_flying_dutchman-20250816@033428-600S-Lum.fts",
]


def main():
    reader = ImageReader()
    print("=" * 90)
    print("FITS头信息检查")
    print("=" * 90)
    for rel in CANDIDATES:
        full = os.path.join(_PROJECT_ROOT, rel)
        if not os.path.exists(full):
            print(f"[缺失] {rel}")
            continue
        try:
            meta = reader.read_metadata(full)
        except Exception as e:
            print(f"[错误] {rel}: {e}")
            continue

        geo = meta.geometry
        wcs = meta.wcs
        obs = meta.observation

        has_wcs = wcs is not None and wcs.has_wcs
        crval1 = wcs.crval1 if has_wcs else 0.0
        crval2 = wcs.crval2 if has_wcs else 0.0
        pixel_scale = wcs.pixel_scale if has_wcs else 0.0
        focallen = obs.focallen if obs else None
        xpixsz = obs.xpixsz if obs else None
        object_name = obs.object_name if obs else ""

        # 计算s0备选
        s0_focal = None
        if focallen and xpixsz and focallen > 0 and xpixsz > 0:
            s0_focal = 206.265 * xpixsz / focallen

        print(f"\n文件: {os.path.basename(rel)}")
        print(f"  OBJECT    = {object_name}")
        print(f"  尺寸      = {geo.width} x {geo.height}")
        print(f"  has_wcs   = {has_wcs}")
        print(f"  CRVAL     = ({crval1:.5f}, {crval2:.5f}) deg")
        print(f"  pixel_scale(WCS) = {pixel_scale:.4f} arcsec/px")
        print(f"  FOCALLEN  = {focallen} mm")
        print(f"  XPIXSZ    = {xpixsz} um")
        print(f"  s0(focal) = {s0_focal:.4f} arcsec/px" if s0_focal else "  s0(focal) = N/A")
        if has_wcs and pixel_scale > 0:
            fov_w = geo.width * pixel_scale / 3600.0
            fov_h = geo.height * pixel_scale / 3600.0
            print(f"  FOV       = {fov_w:.3f} x {fov_h:.3f} deg")
    print("=" * 90)


if __name__ == "__main__":
    main()

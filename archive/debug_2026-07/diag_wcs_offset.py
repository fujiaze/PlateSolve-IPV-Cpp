# -*- coding: utf-8 -*-
"""验证: WCS 投影在边缘的残差 — Gaia 投影位置 vs 实际亮峰位置偏移"""
import os
import sys
import json
import numpy as np

_PROJECT_ROOT = os.path.normpath(os.path.dirname(os.path.abspath(__file__)))
_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except OSError:
        pass

_ASTRO_IO_DIR = os.path.join(_PROJECT_ROOT, "lib", "astro_image_io")
os.environ["PATH"] = _ASTRO_IO_DIR + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_ASTRO_IO_DIR)
    except OSError:
        pass

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "photometric_calib",
                                "gradient_estimator", "python"))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "lib", "photometric_calib",
                                "spectrum_integrator", "python"))
from astro_image_io import ImageReader
from wcs_transform import WCSTransform
from gaia_spectrum_client import GaiaSpectrumClient
from astropy.io import fits as astropy_fits
from astropy.coordinates import angular_separation


def find_peak_in_window(image, cx, cy, fitRadius=8):
    """在以 (cx,cy) 为中心, fitRadius 为半径的窗口内找实际亮峰位置"""
    h, w = image.shape[:2]
    x0 = max(0, int(cx) - fitRadius)
    y0 = max(0, int(cy) - fitRadius)
    x1 = min(w, int(cx) + fitRadius + 1)
    y1 = min(h, int(cy) + fitRadius + 1)
    patch = image[y0:y1, x0:x1].astype(np.float64)
    # 找最大值位置
    py, px = np.unravel_index(np.argmax(patch), patch.shape)
    # 转换为全局坐标
    peak_x = x0 + px
    peak_y = y0 + py
    peak_val = patch[py, px]
    # 质心 (仅对亮于阈值的像素)
    bkg = np.median(patch)
    threshold = bkg + 0.1 * (peak_val - bkg)
    mask = patch > threshold
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        weights = patch[mask] - bkg
        centroid_x = x0 + np.sum(xs * weights) / np.sum(weights)
        centroid_y = y0 + np.sum(ys * weights) / np.sum(weights)
    else:
        centroid_x = peak_x
        centroid_y = peak_y
    return peak_x, peak_y, peak_val, centroid_x, centroid_y, bkg


def main():
    fits_path = os.path.join(
        _PROJECT_ROOT, "testdata", "results", "Galaxy_Center_T4", "panel1", "Red",
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red",
        "01_calibrated.fits")

    print("=" * 70)
    print("WCS 投影残差验证: Gaia 投影位置 vs 实际亮峰位置")
    print("=" * 70)

    # 读取 WCS 和图像
    with astropy_fits.open(fits_path, mode='readonly') as hdul:
        header = hdul[0].header
    sip_order = int(header.get('A_ORDER', 0))
    sip_a = None
    sip_b = None
    if sip_order > 0:
        sip_a = [0.0] * 36
        sip_b = [0.0] * 36
        for i in range(sip_order + 1):
            for j in range(sip_order + 1 - i):
                key_a = 'A_%d_%d' % (i, j)
                key_b = 'B_%d_%d' % (i, j)
                if key_a in header:
                    sip_a[i * 6 + j] = float(header[key_a])
                if key_b in header:
                    sip_b[i * 6 + j] = float(header[key_b])

    wcs_transform = WCSTransform(
        crpix1=float(header.get('CRPIX1', 0.0)),
        crpix2=float(header.get('CRPIX2', 0.0)),
        crval1=float(header.get('CRVAL1', 0.0)),
        crval2=float(header.get('CRVAL2', 0.0)),
        cd11=float(header.get('CD1_1', 0.0)),
        cd12=float(header.get('CD1_2', 0.0)),
        cd21=float(header.get('CD2_1', 0.0)),
        cd22=float(header.get('CD2_2', 0.0)),
        sip_order=sip_order,
        sip_a=sip_a if sip_order > 0 else None,
        sip_b=sip_b if sip_order > 0 else None,
        ctype1=str(header.get('CTYPE1', 'RA---TAN')),
        ctype2=str(header.get('CTYPE2', 'DEC--TAN')),
    )
    print("WCS: CTYPE=(%s, %s), SIP_ORDER=%d" % (
        header.get('CTYPE1', ''), header.get('CTYPE2', ''), sip_order))
    print("CD: [[%.6e, %.6e], [%.6e, %.6e]]" % (
        float(header.get('CD1_1', 0)), float(header.get('CD1_2', 0)),
        float(header.get('CD2_1', 0)), float(header.get('CD2_2', 0))))

    reader = ImageReader()
    image_data = reader.read(fits_path)
    image = image_data.data
    img_w, img_h = image_data.width, image_data.height

    # Gaia 查询
    ra_center_arr, dec_center_arr = wcs_transform.pixel_to_sky_batch(
        np.array([img_w / 2.0]), np.array([img_h / 2.0]))
    ra_center = float(ra_center_arr[0])
    dec_center = float(dec_center_arr[0])

    corner_xs = np.array([0.0, float(img_w), 0.0, float(img_w)])
    corner_ys = np.array([0.0, 0.0, float(img_h), float(img_h)])
    corner_ra, corner_dec = wcs_transform.pixel_to_sky_batch(corner_xs, corner_ys)
    max_sep_rad = 0.0
    for i in range(4):
        sep = angular_separation(
            ra_center * np.pi / 180.0, dec_center * np.pi / 180.0,
            float(corner_ra[i]) * np.pi / 180.0, float(corner_dec[i]) * np.pi / 180.0)
        if sep > max_sep_rad:
            max_sep_rad = sep
    fov_radius_deg = float(max_sep_rad * 180.0 / np.pi)
    cone_radius_deg = fov_radius_deg * 1.05

    gaia_data_dir = os.path.join(_PROJECT_ROOT, "GaiaDR3SP")
    with GaiaSpectrumClient(gaia_data_dir, db_type=2) as client:
        gaia_stars = client.cone_search_with_spectrum(
            ra_center, dec_center, cone_radius_deg, 8.0, 16.0)

    gaia_ra = np.array([s.ra for s in gaia_stars], dtype=np.float64)
    gaia_dec = np.array([s.dec for s in gaia_stars], dtype=np.float64)
    gaia_px, gaia_py = wcs_transform.sky_to_pixel_batch(gaia_ra, gaia_dec)
    in_img = (gaia_px >= 10) & (gaia_px < img_w - 10) & (gaia_py >= 10) & (gaia_py < img_h - 10)
    gaia_px_in = gaia_px[in_img]
    gaia_py_in = gaia_py[in_img]
    gaia_mag_in = np.array([gaia_stars[i].mag_g for i in np.where(in_img)[0]])
    print("Gaia 星: 图像内 (留 10px 边距) %d" % len(gaia_px_in))

    # 只选亮星 (mag < 11), 避免暗星找不到峰
    bright_mask = gaia_mag_in < 11.0
    bright_px = gaia_px_in[bright_mask]
    bright_py = gaia_py_in[bright_mask]
    bright_mag = gaia_mag_in[bright_mask]
    print("亮星 (mag<11): %d" % len(bright_px))

    # 对每颗亮星, 找窗口内实际亮峰, 计算偏移
    cx_c, cy_c = img_w / 2.0, img_h / 2.0
    r_max = float(np.sqrt(img_w**2 + img_h**2)) / 2.0

    offsets = []  # (r, offset_px, mag)
    for i in range(len(bright_px)):
        cx, cy = bright_px[i], bright_py[i]
        mag = bright_mag[i]
        r = np.sqrt((cx - cx_c)**2 + (cy - cy_c)**2) / r_max
        peak_x, peak_y, peak_val, cen_x, cen_y, bkg = find_peak_in_window(image, cx, cy)
        # 用质心偏移 (更准确)
        offset = np.sqrt((cen_x - cx)**2 + (cen_y - cy)**2)
        # 只保留确实有亮峰的 (peak_val > bkg + 1000)
        if peak_val > bkg + 1000:
            offsets.append((r, offset, mag, peak_val, bkg))

    offsets_arr = np.array([(o[0], o[1], o[2]) for o in offsets])
    print("\n找到亮峰的亮星: %d" % len(offsets))

    print("\nWCS 投影残差 (质心偏移) 按径向:")
    print("%-12s %-8s %-12s %-12s %-12s" % (
        "r区间", "n", "偏移median", "偏移mean", "偏移max"))
    for r_lo, r_hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (offsets_arr[:, 0] >= r_lo) & (offsets_arr[:, 0] < r_hi)
        n = mask.sum()
        if n == 0:
            continue
        off = offsets_arr[mask, 1]
        print("  [%.1f,%.1f)  %-8d %-12.2f %-12.2f %-12.2f" % (
            r_lo, r_hi, n, np.median(off), np.mean(off), off.max()))

    # 偏移 > 2px 的比例
    print("\n偏移 > 2px (fit window 无法拟合):")
    for r_lo, r_hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (offsets_arr[:, 0] >= r_lo) & (offsets_arr[:, 0] < r_hi)
        n = mask.sum()
        if n == 0:
            continue
        n_bad = np.sum(offsets_arr[mask, 1] > 2.0)
        print("  r=[%.1f,%.1f): n=%d, 偏移>2px=%d (%.1f%%)" % (
            r_lo, r_hi, n, n_bad, 100.0 * n_bad / n))

    # 显示前 10 个偏移最大的星
    print("\n偏移最大的 10 颗星:")
    sorted_idx = np.argsort(offsets_arr[:, 1])[::-1][:10]
    print("%-6s %-10s %-10s %-10s %-10s %-10s" % (
        "i", "r", "mag", "偏移px", "peak_val", "bkg"))
    for i in sorted_idx:
        r, off, mag = offsets_arr[i]
        print("  %-6d %-10.2f %-10.2f %-10.2f" % (i, r, mag, off))

    # 对比: 重新投影验证 WCS 一致性
    # 取 Gaia 星投影到像素, 再反投影回天球, 检查是否回到原位
    print("\n" + "=" * 70)
    print("WCS 往返一致性: sky->pix->sky 应该回到原位")
    print("=" * 70)
    test_ra = gaia_ra[:100]
    test_dec = gaia_dec[:100]
    px, py = wcs_transform.sky_to_pixel_batch(test_ra, test_dec)
    back_ra, back_dec = wcs_transform.pixel_to_sky_batch(px, py)
    d_ra = np.abs(back_ra - test_ra) * 3600  # arcsec
    d_dec = np.abs(back_dec - test_dec) * 3600
    print("sky->pix->sky 往返残差:")
    print("  RA: median=%.4f\", max=%.4f\"" % (np.median(d_ra), d_ra.max()))
    print("  Dec: median=%.4f\", max=%.4f\"" % (np.median(d_dec), d_dec.max()))


if __name__ == "__main__":
    main()
